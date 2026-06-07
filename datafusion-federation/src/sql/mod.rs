mod analyzer;
pub mod ast_analyzer;
mod executor;
mod schema;
mod table;
mod table_reference;

use std::{any::Any, fmt, sync::Arc, vec};

use analyzer::RewriteTableScanAnalyzer;
use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::{Schema, SchemaRef},
    common::{
        tree_node::{Transformed, TreeNode},
        Statistics,
    },
    config::ConfigOptions,
    error::{DataFusionError, Result},
    execution::{context::SessionState, TaskContext},
    logical_expr::{Extension, LogicalPlan},
    optimizer::{optimizer::Optimizer, OptimizerConfig, OptimizerRule},
    physical_expr::EquivalenceProperties,
    physical_plan::{
        execution_plan::{Boundedness, EmissionType},
        filter_pushdown::{
            ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
        },
        metrics::MetricsSet,
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PhysicalExpr, PlanProperties,
        SendableRecordBatchStream,
    },
    sql::{sqlparser::ast::Statement, unparser::Unparser},
};

pub use executor::{AstAnalyzer, LogicalOptimizer, SQLExecutor, SQLExecutorRef, SqlQueryRewriter};
pub use schema::{MultiSchemaProvider, SQLSchemaProvider};
pub use table::{RemoteTable, SQLTable, SQLTableSource};
pub use table_reference::RemoteTableRef;

use crate::{
    get_table_source, schema_cast, FederatedPlanNode, FederationPlanner, FederationProvider,
};

// SQLFederationProvider provides federation to SQL DMBSs.
#[derive(Debug)]
pub struct SQLFederationProvider {
    pub optimizer: Arc<Optimizer>,
    pub executor: Arc<dyn SQLExecutor>,
}

impl SQLFederationProvider {
    pub fn new(executor: Arc<dyn SQLExecutor>) -> Self {
        Self {
            optimizer: Arc::new(Optimizer::with_rules(vec![Arc::new(
                SQLFederationOptimizerRule::new(executor.clone()),
            )])),
            executor,
        }
    }
}

impl FederationProvider for SQLFederationProvider {
    fn name(&self) -> &str {
        "sql_federation_provider"
    }

    fn compute_context(&self) -> Option<String> {
        self.executor.compute_context()
    }

    fn optimizer(&self) -> Option<Arc<Optimizer>> {
        Some(self.optimizer.clone())
    }
}

#[derive(Debug)]
struct SQLFederationOptimizerRule {
    planner: Arc<SQLFederationPlanner>,
}

impl SQLFederationOptimizerRule {
    pub fn new(executor: Arc<dyn SQLExecutor>) -> Self {
        Self {
            planner: Arc::new(SQLFederationPlanner::new(Arc::clone(&executor))),
        }
    }
}

impl OptimizerRule for SQLFederationOptimizerRule {
    /// Try to rewrite `plan` to an optimized form, returning `Transformed::yes`
    /// if the plan was rewritten and `Transformed::no` if it was not.
    ///
    /// Note: this function is only called if [`Self::supports_rewrite`] returns
    /// true. Otherwise the Optimizer calls  [`Self::try_optimize`]
    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if let LogicalPlan::Extension(Extension { ref node }) = plan {
            if node.name() == "Federated" {
                // Avoid attempting double federation
                return Ok(Transformed::no(plan));
            }
        }

        let fed_plan = FederatedPlanNode::new(plan.clone(), self.planner.clone());
        let ext_node = Extension {
            node: Arc::new(fed_plan),
        };

        let mut plan = LogicalPlan::Extension(ext_node);
        if let Some(mut rewriter) = self.planner.executor.logical_optimizer() {
            plan = rewriter(plan)?;
        }

        Ok(Transformed::yes(plan))
    }

    /// A human readable name for this analyzer rule
    fn name(&self) -> &str {
        "federate_sql"
    }

    /// Does this rule support rewriting owned plans (rather than by reference)?
    fn supports_rewrite(&self) -> bool {
        true
    }
}

#[derive(Debug)]
pub struct SQLFederationPlanner {
    pub executor: Arc<dyn SQLExecutor>,
}

impl SQLFederationPlanner {
    pub fn new(executor: Arc<dyn SQLExecutor>) -> Self {
        Self { executor }
    }
}

#[async_trait]
impl FederationPlanner for SQLFederationPlanner {
    async fn plan_federation(
        &self,
        node: &FederatedPlanNode,
        _session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Arc::new(node.plan().schema().as_arrow().clone());
        let plan = node.plan().clone();
        let statistics = self.executor.statistics(&plan).await?;
        let input = Arc::new(VirtualExecutionPlan::new(
            plan,
            Arc::clone(&self.executor),
            statistics,
        ));
        let schema_cast_exec = schema_cast::SchemaCastScanExec::new(input, schema);
        Ok(Arc::new(schema_cast_exec))
    }
}

#[derive(Debug, Clone)]
pub struct VirtualExecutionPlan {
    plan: LogicalPlan,
    executor: Arc<dyn SQLExecutor>,
    props: Arc<PlanProperties>,
    statistics: Statistics,
    filters: Vec<Arc<dyn PhysicalExpr>>,
}

impl VirtualExecutionPlan {
    pub fn new(plan: LogicalPlan, executor: Arc<dyn SQLExecutor>, statistics: Statistics) -> Self {
        let schema: Schema = plan.schema().as_arrow().clone();
        let props = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::new(schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            plan,
            executor,
            props,
            statistics,
            filters: Vec::new(),
        }
    }

    pub fn plan(&self) -> &LogicalPlan {
        &self.plan
    }

    pub fn executor(&self) -> &Arc<dyn SQLExecutor> {
        &self.executor
    }

    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }

    fn schema(&self) -> SchemaRef {
        let df_schema = self.plan.schema().as_arrow().clone();
        Arc::new(df_schema)
    }

    fn final_sql(&self) -> Result<String> {
        let plan = self.plan.clone();
        let plan = RewriteTableScanAnalyzer::rewrite(plan)?;
        let (logical_optimizers, ast_analyzers, sql_query_rewriters) = gather_analyzers(&plan)?;
        let plan = apply_logical_optimizers(plan, logical_optimizers)?;
        let ast = self.plan_to_statement(&plan)?;
        let ast = self.rewrite_with_executor_ast_analyzer(ast)?;
        let ast = apply_ast_analyzers(ast, ast_analyzers)?;
        apply_sql_query_rewriters(ast.to_string(), sql_query_rewriters)
    }

    fn rewrite_with_executor_ast_analyzer(
        &self,
        ast: Statement,
    ) -> Result<Statement, datafusion::error::DataFusionError> {
        if let Some(mut analyzer) = self.executor.ast_analyzer() {
            Ok(analyzer(ast)?)
        } else {
            Ok(ast)
        }
    }

    fn plan_to_statement(&self, plan: &LogicalPlan) -> Result<Statement> {
        Unparser::new(self.executor.dialect().as_ref()).plan_to_sql(plan)
    }
}

fn gather_analyzers(
    plan: &LogicalPlan,
) -> Result<(
    Vec<LogicalOptimizer>,
    Vec<AstAnalyzer>,
    Vec<SqlQueryRewriter>,
)> {
    let mut logical_optimizers = vec![];
    let mut ast_analyzers = vec![];
    let mut sql_query_rewriters = vec![];

    plan.apply(|node| {
        if let LogicalPlan::TableScan(table) = node {
            let provider = get_table_source(&table.source)
                .expect("caller is virtual exec so this is valid")
                .expect("caller is virtual exec so this is valid");
            if let Some(source) = provider.as_any().downcast_ref::<SQLTableSource>() {
                if let Some(analyzer) = source.table.logical_optimizer() {
                    logical_optimizers.push(analyzer);
                }
                if let Some(analyzer) = source.table.ast_analyzer() {
                    ast_analyzers.push(analyzer);
                }
                if let Some(rewriter) = source.table.sql_query_rewriter() {
                    sql_query_rewriters.push(rewriter);
                }
            }
        }
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    })?;

    Ok((logical_optimizers, ast_analyzers, sql_query_rewriters))
}

fn apply_logical_optimizers(
    mut plan: LogicalPlan,
    analyzers: Vec<LogicalOptimizer>,
) -> Result<LogicalPlan> {
    for mut analyzer in analyzers {
        let old_schema = plan.schema().clone();
        plan = analyzer(plan)?;
        let new_schema = plan.schema();
        if &old_schema != new_schema {
            return Err(DataFusionError::Execution(format!(
                "Schema altered during logical analysis, expected: {}, found: {}",
                old_schema, new_schema
            )));
        }
    }
    Ok(plan)
}

fn apply_ast_analyzers(mut statement: Statement, analyzers: Vec<AstAnalyzer>) -> Result<Statement> {
    for mut analyzer in analyzers {
        statement = analyzer(statement)?;
    }
    Ok(statement)
}

fn apply_sql_query_rewriters(
    mut query: String,
    rewriters: Vec<SqlQueryRewriter>,
) -> Result<String> {
    for mut rewriter in rewriters {
        query = rewriter(query)?;
    }
    Ok(query)
}

impl DisplayAs for VirtualExecutionPlan {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        write!(f, "VirtualExecutionPlan")?;
        write!(f, " name={}", self.executor.name())?;
        if let Some(ctx) = self.executor.compute_context() {
            write!(f, " compute_context={ctx}")?;
        };
        let mut plan = match RewriteTableScanAnalyzer::rewrite(self.plan.clone()) {
            Ok(plan) => plan,
            Err(_) => self.plan.clone(),
        };
        if let Ok(statement) = self.plan_to_statement(&plan) {
            write!(f, " base_sql={statement}")?;
        }

        let (logical_optimizers, ast_analyzers, sql_query_rewriters) = match gather_analyzers(&plan)
        {
            Ok(analyzers) => analyzers,
            Err(_) => return Ok(()),
        };

        let old_plan = plan.clone();

        plan = match apply_logical_optimizers(plan, logical_optimizers) {
            Ok(plan) => plan,
            _ => return Ok(()),
        };

        let statement = match self.plan_to_statement(&plan) {
            Ok(statement) => statement,
            _ => return Ok(()),
        };

        if plan != old_plan {
            write!(f, " rewritten_logical_sql={statement}")?;
        }

        let old_statement = statement.clone();
        let statement = match self.rewrite_with_executor_ast_analyzer(statement) {
            Ok(statement) => statement,
            _ => return Ok(()),
        };
        if old_statement != statement {
            write!(f, " rewritten_executor_sql={statement}")?;
        }

        let old_statement = statement.clone();
        let statement = match apply_ast_analyzers(statement, ast_analyzers) {
            Ok(statement) => statement,
            _ => return Ok(()),
        };
        if old_statement != statement {
            write!(f, " rewritten_ast_analyzer={statement}")?;
        }

        let sql = statement.to_string();
        let rewritten_sql = match apply_sql_query_rewriters(sql.clone(), sql_query_rewriters) {
            Ok(sql) => sql,
            _ => return Ok(()),
        };
        if sql != rewritten_sql {
            write!(f, " rewritten_sql_query={rewritten_sql}")?;
        }

        Ok(())
    }
}

impl ExecutionPlan for VirtualExecutionPlan {
    fn name(&self) -> &str {
        "sql_federation_exec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.executor
            .execute(&self.final_sql()?, self.schema(), &self.filters)
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.props
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Statistics> {
        Ok(self.statistics.clone())
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.executor.metrics()
    }

    fn handle_child_pushdown_result(
        &self,
        _phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let parent_filters: Vec<_> = child_pushdown_result
            .clone()
            .parent_filters
            .into_iter()
            .map(|f| f.filter)
            .collect();

        if parent_filters.is_empty() {
            return Ok(FilterPushdownPropagation {
                filters: vec![],
                updated_node: None,
            });
        }

        let filters_pushed_down = vec![PushedDown::Yes; parent_filters.len()];
        let mut node = self.clone();
        node.filters = parent_filters;

        Ok(FilterPushdownPropagation {
            filters: filters_pushed_down,
            updated_node: Some(Arc::new(node)),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::sql::{
        RemoteTableRef, SQLExecutor, SQLFederationProvider, SQLTable, SQLTableSource,
    };
    use crate::FederatedTableProviderAdaptor;
    use async_trait::async_trait;
    use datafusion::arrow::datatypes::{Schema, SchemaRef};
    use datafusion::common::tree_node::TreeNodeRecursion;
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::sql::unparser::dialect::Dialect;
    use datafusion::sql::unparser::{self};
    use datafusion::sql::TableReference;
    use datafusion::{
        arrow::datatypes::{DataType, Field},
        datasource::TableProvider,
        execution::context::SessionContext,
    };

    use super::table::RemoteTable;
    use super::*;

    #[derive(Debug, Clone)]
    struct TestExecutor {
        compute_context: String,
    }

    #[async_trait]
    impl SQLExecutor for TestExecutor {
        fn name(&self) -> &str {
            "TestExecutor"
        }

        fn compute_context(&self) -> Option<String> {
            Some(self.compute_context.clone())
        }

        fn dialect(&self) -> Arc<dyn Dialect> {
            Arc::new(unparser::dialect::DefaultDialect {})
        }

        fn execute(
            &self,
            _query: &str,
            _schema: SchemaRef,
            _filters: &[Arc<dyn PhysicalExpr>],
        ) -> Result<SendableRecordBatchStream> {
            unimplemented!()
        }

        async fn table_names(&self) -> Result<Vec<String>> {
            unimplemented!()
        }

        async fn get_table_schema(&self, _table_name: &str) -> Result<SchemaRef> {
            unimplemented!()
        }
    }

    fn get_test_table_provider(name: String, executor: TestExecutor) -> Arc<dyn TableProvider> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("c", DataType::Date32, false),
        ]));
        let table_ref = RemoteTableRef::try_from(name).unwrap();
        let table = Arc::new(RemoteTable::new(table_ref, schema));
        let provider = Arc::new(SQLFederationProvider::new(Arc::new(executor)));
        let table_source = Arc::new(SQLTableSource { provider, table });
        Arc::new(FederatedTableProviderAdaptor::new(table_source))
    }

    fn get_test_table_provider_with_table(
        table: Arc<dyn SQLTable>,
        executor: TestExecutor,
    ) -> Arc<dyn TableProvider> {
        let provider = Arc::new(SQLFederationProvider::new(Arc::new(executor)));
        let table_source = Arc::new(SQLTableSource::new_with_table(provider, table));
        Arc::new(FederatedTableProviderAdaptor::new(table_source))
    }

    #[derive(Debug)]
    struct SqlRewriteTable {
        table: RemoteTable,
        rewrite_calls: Arc<AtomicUsize>,
        suffix: String,
    }

    impl SqlRewriteTable {
        fn new(
            table_ref: RemoteTableRef,
            schema: SchemaRef,
            rewrite_calls: Arc<AtomicUsize>,
            suffix: impl Into<String>,
        ) -> Self {
            Self {
                table: RemoteTable::new(table_ref, schema),
                rewrite_calls,
                suffix: suffix.into(),
            }
        }
    }

    impl SQLTable for SqlRewriteTable {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn table_reference(&self) -> TableReference {
            self.table.table_reference().clone()
        }

        fn schema(&self) -> SchemaRef {
            Arc::clone(self.table.schema())
        }

        fn sql_query_rewriter(&self) -> Option<SqlQueryRewriter> {
            let rewrite_calls = Arc::clone(&self.rewrite_calls);
            let suffix = self.suffix.clone();
            Some(Box::new(move |sql| {
                rewrite_calls.fetch_add(1, Ordering::SeqCst);
                Ok(format!("{sql} {suffix}"))
            }))
        }
    }

    #[tokio::test]
    async fn basic_sql_federation_test() -> Result<(), DataFusionError> {
        let test_executor_a = TestExecutor {
            compute_context: "a".into(),
        };

        let test_executor_b = TestExecutor {
            compute_context: "b".into(),
        };

        let table_a1_ref = "table_a1".to_string();
        let table_a1 = get_test_table_provider(table_a1_ref.clone(), test_executor_a.clone());

        let table_a2_ref = "table_a2".to_string();
        let table_a2 = get_test_table_provider(table_a2_ref.clone(), test_executor_a);

        let table_b1_ref = "table_b1(1)".to_string();
        let table_b1_df_ref = "table_local_b1".to_string();

        let table_b1 = get_test_table_provider(table_b1_ref.clone(), test_executor_b);

        // Create a new SessionState with the optimizer rule we created above
        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_table(table_a1_ref.clone(), table_a1).unwrap();
        ctx.register_table(table_a2_ref.clone(), table_a2).unwrap();
        ctx.register_table(table_b1_df_ref.clone(), table_b1)
            .unwrap();

        let query = r#"
            SELECT * FROM table_a1
            UNION ALL
            SELECT * FROM table_a2
            UNION ALL
            SELECT * FROM table_local_b1;
        "#;

        let df = ctx.sql(query).await?;

        let logical_plan = df.into_optimized_plan()?;

        let mut table_a1_federated = false;
        let mut table_a2_federated = false;
        let mut table_b1_federated = false;

        let _ = logical_plan.apply(|node| {
            if let LogicalPlan::Extension(node) = node {
                if let Some(node) = node.node.as_any().downcast_ref::<FederatedPlanNode>() {
                    let _ = node.plan().apply(|node| {
                        if let LogicalPlan::TableScan(table) = node {
                            if table.table_name.table() == table_a1_ref {
                                table_a1_federated = true;
                            }
                            if table.table_name.table() == table_a2_ref {
                                table_a2_federated = true;
                            }
                            // assuming table name is rewritten via analyzer
                            if table.table_name.table() == table_b1_df_ref {
                                table_b1_federated = true;
                            }
                        }
                        Ok(TreeNodeRecursion::Continue)
                    });
                }
            }
            Ok(TreeNodeRecursion::Continue)
        });

        assert!(table_a1_federated);
        assert!(table_a2_federated);
        assert!(table_b1_federated);

        let physical_plan = ctx.state().create_physical_plan(&logical_plan).await?;

        let mut final_queries = vec![];

        let _ = physical_plan.apply(|node| {
            if node.name() == "sql_federation_exec" {
                let node = node
                    .as_any()
                    .downcast_ref::<VirtualExecutionPlan>()
                    .unwrap();

                final_queries.push(node.final_sql()?);
            }
            Ok(TreeNodeRecursion::Continue)
        });

        let expected = vec![
            "SELECT table_a1.a, table_a1.b, table_a1.c FROM table_a1",
            "SELECT table_a2.a, table_a2.b, table_a2.c FROM table_a2",
            "SELECT table_b1.a, table_b1.b, table_b1.c FROM table_b1(1) AS table_b1",
        ];

        assert_eq!(
            HashSet::<&str>::from_iter(final_queries.iter().map(|x| x.as_str())),
            HashSet::from_iter(expected)
        );

        Ok(())
    }

    #[tokio::test]
    async fn multi_reference_sql_federation_test() -> Result<(), DataFusionError> {
        let test_executor_a = TestExecutor {
            compute_context: "test".into(),
        };

        let lowercase_table_ref = "default.table".to_string();
        let lowercase_local_table_ref = "dftable".to_string();
        let lowercase_table =
            get_test_table_provider(lowercase_table_ref.clone(), test_executor_a.clone());

        let capitalized_table_ref = "default.Table(1)".to_string();
        let capitalized_local_table_ref = "dfview".to_string();
        let capitalized_table =
            get_test_table_provider(capitalized_table_ref.clone(), test_executor_a);

        // Create a new SessionState with the optimizer rule we created above
        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_table(lowercase_local_table_ref.clone(), lowercase_table)
            .unwrap();
        ctx.register_table(capitalized_local_table_ref.clone(), capitalized_table)
            .unwrap();

        let query = r#"
                SELECT * FROM dftable
                UNION ALL
                SELECT * FROM dfview;
            "#;

        let df = ctx.sql(query).await?;

        let logical_plan = df.into_optimized_plan()?;

        let mut lowercase_table = false;
        let mut capitalized_table = false;

        let _ = logical_plan.apply(|node| {
            if let LogicalPlan::Extension(node) = node {
                if let Some(node) = node.node.as_any().downcast_ref::<FederatedPlanNode>() {
                    let _ = node.plan().apply(|node| {
                        if let LogicalPlan::TableScan(table) = node {
                            if table.table_name.table() == lowercase_local_table_ref {
                                lowercase_table = true;
                            }
                            if table.table_name.table() == capitalized_local_table_ref {
                                capitalized_table = true;
                            }
                        }
                        Ok(TreeNodeRecursion::Continue)
                    });
                }
            }
            Ok(TreeNodeRecursion::Continue)
        });

        assert!(lowercase_table);
        assert!(capitalized_table);

        let physical_plan = ctx.state().create_physical_plan(&logical_plan).await?;

        let mut final_queries = vec![];

        let _ = physical_plan.apply(|node| {
            if node.name() == "sql_federation_exec" {
                let node = node
                    .as_any()
                    .downcast_ref::<VirtualExecutionPlan>()
                    .unwrap();

                final_queries.push(node.final_sql()?);
            }
            Ok(TreeNodeRecursion::Continue)
        });

        let expected = vec![
            r#"SELECT "table".a, "table".b, "table".c FROM "default"."table" UNION ALL SELECT "Table".a, "Table".b, "Table".c FROM "default"."Table"(1) AS Table"#,
        ];

        assert_eq!(
            HashSet::<&str>::from_iter(final_queries.iter().map(|x| x.as_str())),
            HashSet::from_iter(expected)
        );

        Ok(())
    }

    /// EXPLAIN ANALYZE must not federate the Analyze wrapper — only the inner
    /// query should be federated. Otherwise the SQL Unparser fails because it
    /// cannot convert Analyze to SQL.
    #[tokio::test]
    async fn explain_analyze_not_federated() -> Result<(), DataFusionError> {
        let executor = TestExecutor {
            compute_context: "a".into(),
        };

        let table_ref = "test_table".to_string();
        let table = get_test_table_provider(table_ref.clone(), executor);

        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_table(table_ref, table).unwrap();

        let plan = ctx
            .sql("EXPLAIN ANALYZE SELECT * FROM test_table")
            .await?
            .into_optimized_plan()?;

        // The top-level node must be Analyze, not Federated.
        assert!(
            matches!(plan, LogicalPlan::Analyze(_)),
            "Expected Analyze at root, got: {}",
            plan.display_indent()
        );

        // The inner plan should contain a Federated extension node.
        let mut found_federated = false;
        plan.apply(|node| {
            if let LogicalPlan::Extension(ext) = node {
                if ext.node.name() == "Federated" {
                    found_federated = true;
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        assert!(
            found_federated,
            "Expected a Federated node inside the Analyze plan"
        );

        // Physical planning should succeed (this is where it used to fail).
        let physical_plan = ctx.state().create_physical_plan(&plan).await?;
        assert_eq!(physical_plan.name(), "AnalyzeExec");

        Ok(())
    }

    #[tokio::test]
    async fn sql_query_rewriter_hook_invoked_and_rewrites_sql() -> Result<(), DataFusionError> {
        let executor = TestExecutor {
            compute_context: "rewrite".into(),
        };
        let rewrite_calls = Arc::new(AtomicUsize::new(0));
        let table_ref = "table_with_rewriter".to_string();
        let table = Arc::new(SqlRewriteTable::new(
            table_ref.clone().try_into().unwrap(),
            Arc::new(Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Utf8, false),
                Field::new("c", DataType::Date32, false),
            ])),
            Arc::clone(&rewrite_calls),
            "/* rewritten by sql_query_rewriter */",
        ));
        let table_provider = get_test_table_provider_with_table(table, executor);

        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_table(table_ref.clone(), table_provider)
            .unwrap();

        let query = format!("SELECT * FROM {table_ref}");
        let df = ctx.sql(&query).await?;
        let logical_plan = df.into_optimized_plan()?;
        let physical_plan = ctx.state().create_physical_plan(&logical_plan).await?;

        let mut final_queries = vec![];
        let _ = physical_plan.apply(|node| {
            if node.name() == "sql_federation_exec" {
                let node = node
                    .as_any()
                    .downcast_ref::<VirtualExecutionPlan>()
                    .unwrap();
                final_queries.push(node.final_sql()?);
            }
            Ok(TreeNodeRecursion::Continue)
        });

        let [final_query] = final_queries.as_slice() else {
            panic!("expected a single federated SQL query");
        };

        assert!(final_query.ends_with("/* rewritten by sql_query_rewriter */"));
        assert_eq!(rewrite_calls.load(Ordering::SeqCst), 1);

        Ok(())
    }

    // -------------------------------------------------------------------------
    // EXISTS / NOT EXISTS federation (port of spice patch #74 onto v0.5.3).
    //
    // In v0.5.3, `FederationOptimizerRule` (optimizer/mod.rs) gained `Expr::Exists`
    // handling so correlated EXISTS / NOT EXISTS subqueries are recognised and
    // federated together with their outer query when both sides share a provider.
    //
    // These tests assert the *exact federated SQL* emitted to the remote engine
    // (via `VirtualExecutionPlan::final_sql`), which is the surface where the
    // DataFusion-52.5 analyzer-rule architecture produced an alias bug
    // (dropping the inner subquery alias and emitting tautological
    // `lineitem.x = lineitem.x` predicates). The v0.5.3 unparser redesign emits
    // the EXISTS subquery as a properly-aliased derived table
    // (`(SELECT ... FROM lineitem AS l2) AS __correlated_sq_1`), so the bug is
    // fixed by construction; these tests guard against regressing it.
    // -------------------------------------------------------------------------

    fn exists_table(
        name: &str,
        fields: Vec<Field>,
        executor: &TestExecutor,
    ) -> Arc<dyn TableProvider> {
        let schema = Arc::new(Schema::new(fields));
        let table_ref = RemoteTableRef::try_from(name.to_string()).unwrap();
        let table = Arc::new(RemoteTable::new(table_ref, schema));
        let provider = Arc::new(SQLFederationProvider::new(Arc::new(executor.clone())));
        let table_source = Arc::new(SQLTableSource { provider, table });
        Arc::new(FederatedTableProviderAdaptor::new(table_source))
    }

    /// Collects the SQL sent to each federated (`VirtualExecutionPlan`) node.
    async fn federated_sqls(ctx: &SessionContext, query: &str) -> Vec<String> {
        let df = ctx.sql(query).await.unwrap();
        let logical_plan = df.into_optimized_plan().unwrap();
        let physical_plan = ctx
            .state()
            .create_physical_plan(&logical_plan)
            .await
            .unwrap();
        let mut sqls = vec![];
        let _ = physical_plan.apply(|node| {
            if node.name() == "sql_federation_exec" {
                let node = node
                    .as_any()
                    .downcast_ref::<VirtualExecutionPlan>()
                    .unwrap();
                sqls.push(node.final_sql()?);
            }
            Ok(TreeNodeRecursion::Continue)
        });
        sqls
    }

    fn orders_pk(ex: &TestExecutor) -> Arc<dyn TableProvider> {
        exists_table(
            "orders",
            vec![Field::new("o_orderkey", DataType::Int64, false)],
            ex,
        )
    }
    fn lineitem_ol(ex: &TestExecutor) -> Arc<dyn TableProvider> {
        exists_table(
            "lineitem",
            vec![
                Field::new("l_orderkey", DataType::Int64, false),
                Field::new("l_suppkey", DataType::Int64, false),
            ],
            ex,
        )
    }
    fn supplier(ex: &TestExecutor) -> Arc<dyn TableProvider> {
        exists_table(
            "supplier",
            vec![
                Field::new("s_suppkey", DataType::Int64, false),
                Field::new("s_name", DataType::Utf8, false),
            ],
            ex,
        )
    }

    /// Same-provider EXISTS: the whole query (outer + correlated subquery) must
    /// be federated as one SQL statement so the backend can decorrelate it.
    #[tokio::test]
    async fn same_provider_exists_federates_as_single_unit() {
        let ex = TestExecutor {
            compute_context: "ctx_a".into(),
        };
        let ctx = SessionContext::new_with_state(crate::default_session_state());
        ctx.register_table("orders", orders_pk(&ex)).unwrap();
        ctx.register_table("lineitem", lineitem_ol(&ex)).unwrap();

        let sqls = federated_sqls(
            &ctx,
            "SELECT o_orderkey FROM orders WHERE EXISTS \
             (SELECT 1 FROM lineitem WHERE l_orderkey = o_orderkey)",
        )
        .await;

        assert_eq!(sqls.len(), 1, "expected one federated query, got {sqls:?}");
        assert_eq!(
            sqls[0],
            "SELECT orders.o_orderkey FROM orders WHERE EXISTS \
             (SELECT 1 FROM (SELECT 1, lineitem.l_orderkey FROM lineitem) AS __correlated_sq_1 \
             WHERE (__correlated_sq_1.l_orderkey = orders.o_orderkey))"
        );
    }

    /// Same-provider JOIN + correlated NOT EXISTS using an inner alias `l2`
    /// against an unaliased outer `lineitem` (TPC-H Q21 shape).
    ///
    /// Regression guard for Copilot thread (2): the inner alias `l2` must NOT be
    /// dropped. The emitted SQL must keep the inner subquery columns distinct
    /// from the outer `lineitem` columns (via the `__correlated_sq_1` derived
    /// table) so the predicate is not the tautological `lineitem.x = lineitem.x`.
    #[tokio::test]
    async fn same_provider_join_not_exists_keeps_inner_alias() {
        let ex = TestExecutor {
            compute_context: "ctx_a".into(),
        };
        let ctx = SessionContext::new_with_state(crate::default_session_state());
        ctx.register_table("supplier", supplier(&ex)).unwrap();
        ctx.register_table("lineitem", lineitem_ol(&ex)).unwrap();
        ctx.register_table("orders", orders_pk(&ex)).unwrap();

        let sqls = federated_sqls(
            &ctx,
            "SELECT s_name FROM supplier \
             JOIN lineitem ON s_suppkey = l_suppkey \
             JOIN orders ON o_orderkey = l_orderkey \
             WHERE NOT EXISTS ( \
                 SELECT 1 FROM lineitem AS l2 \
                 WHERE l2.l_orderkey = lineitem.l_orderkey \
                 AND l2.l_suppkey <> lineitem.l_suppkey \
             )",
        )
        .await;

        assert_eq!(sqls.len(), 1, "expected one federated query, got {sqls:?}");
        let sql = &sqls[0];
        // The inner subquery is unparsed as a derived table over `lineitem AS l2`,
        // preserving the inner alias.
        assert!(
            sql.contains("FROM lineitem AS l2"),
            "inner alias l2 dropped: {sql}"
        );
        // The correlated predicate must compare the inner derived-table columns to
        // the OUTER lineitem columns -- never `lineitem.x = lineitem.x`.
        assert!(
            sql.contains("__correlated_sq_1.l_orderkey = lineitem.l_orderkey")
                && sql.contains("__correlated_sq_1.l_suppkey <> lineitem.l_suppkey"),
            "correlated predicate lost inner/outer distinction: {sql}"
        );
        assert!(
            !sql.contains("lineitem.l_orderkey = lineitem.l_orderkey"),
            "tautological self-comparison present (alias bug regressed): {sql}"
        );
    }

    /// Same-provider correlated NOT EXISTS with BOTH outer (`l1`) and inner
    /// (`l2`) aliases (TPC-H Q21 full shape, including ORDER BY).
    ///
    /// Regression guard for Copilot thread (3): the emitted SQL must use the
    /// `l1` / `l2` aliases consistently and must not fall back to bare
    /// `lineitem.*` references (which are invalid once the table is aliased).
    #[tokio::test]
    async fn same_provider_aliased_not_exists_uses_aliases() {
        let ex = TestExecutor {
            compute_context: "ctx_a".into(),
        };
        let ctx = SessionContext::new_with_state(crate::default_session_state());
        ctx.register_table("supplier", supplier(&ex)).unwrap();
        ctx.register_table(
            "lineitem",
            exists_table(
                "lineitem",
                vec![
                    Field::new("l_orderkey", DataType::Int64, false),
                    Field::new("l_suppkey", DataType::Int64, false),
                    Field::new("l_commitdate", DataType::Date32, true),
                    Field::new("l_receiptdate", DataType::Date32, true),
                ],
                &ex,
            ),
        )
        .unwrap();
        ctx.register_table("orders", orders_pk(&ex)).unwrap();

        let sqls = federated_sqls(
            &ctx,
            "SELECT s_name, count(*) AS numwait \
             FROM supplier, lineitem l1, orders \
             WHERE s_suppkey = l1.l_suppkey \
               AND l1.l_orderkey = o_orderkey \
               AND l1.l_receiptdate > l1.l_commitdate \
               AND NOT EXISTS ( \
                   SELECT 1 FROM lineitem l2 \
                   WHERE l2.l_orderkey = l1.l_orderkey \
                     AND l2.l_suppkey <> l1.l_suppkey \
               ) \
             GROUP BY s_name \
             ORDER BY numwait DESC, s_name",
        )
        .await;

        assert_eq!(sqls.len(), 1, "expected one federated query, got {sqls:?}");
        let sql = &sqls[0];
        assert!(sql.contains("lineitem AS l1"), "outer alias l1 lost: {sql}");
        assert!(sql.contains("lineitem AS l2"), "inner alias l2 lost: {sql}");
        // Correlation references the outer alias `l1`, not bare `lineitem`.
        assert!(
            sql.contains("__correlated_sq_1.l_orderkey = l1.l_orderkey")
                && sql.contains("__correlated_sq_1.l_suppkey <> l1.l_suppkey"),
            "correlation does not use outer alias l1: {sql}"
        );
        assert!(
            !sql.contains("lineitem.l_orderkey = lineitem.l_orderkey"),
            "tautological self-comparison present (alias bug regressed): {sql}"
        );
    }

    /// Cross-provider NOT EXISTS: outer and subquery are on different providers,
    /// so they must be federated as TWO separate SQL statements (DataFusion
    /// performs the anti-join locally).
    #[tokio::test]
    async fn cross_provider_not_exists_splits() {
        let ex_a = TestExecutor {
            compute_context: "ctx_a".into(),
        };
        let ex_b = TestExecutor {
            compute_context: "ctx_b".into(),
        };
        let ctx = SessionContext::new_with_state(crate::default_session_state());
        ctx.register_table("orders", orders_pk(&ex_a)).unwrap();
        ctx.register_table("lineitem", lineitem_ol(&ex_b)).unwrap();

        let sqls = federated_sqls(
            &ctx,
            "SELECT o_orderkey FROM orders WHERE NOT EXISTS \
             (SELECT 1 FROM lineitem WHERE l_orderkey = o_orderkey)",
        )
        .await;

        assert_eq!(
            sqls.len(),
            2,
            "cross-provider EXISTS must split into two federated queries, got {sqls:?}"
        );
    }

    /// Top-level ORDER BY on a single federated provider is pushed down into the
    /// remote SQL and the whole plan is federated.
    ///
    /// Documents the behavior referenced by Copilot thread (4): v0.5.3 (matching
    /// upstream) federates the ORDER BY into the remote query rather than keeping
    /// a local top-level `SortExec`. This is upstream-by-design (the spice patch
    /// that added local sort preservation, #71, was reverted by #72 for causing
    /// correctness issues, and is not part of the v0.5.3 redesign).
    #[tokio::test]
    async fn top_level_order_by_is_pushed_down() {
        let ex = TestExecutor {
            compute_context: "ctx_a".into(),
        };
        let ctx = SessionContext::new_with_state(crate::default_session_state());
        ctx.register_table("orders", orders_pk(&ex)).unwrap();

        let q = "SELECT o_orderkey FROM orders ORDER BY o_orderkey DESC";
        let sqls = federated_sqls(&ctx, q).await;
        assert_eq!(sqls.len(), 1, "expected one federated query, got {sqls:?}");
        assert_eq!(
            sqls[0],
            "SELECT orders.o_orderkey FROM orders ORDER BY orders.o_orderkey DESC NULLS FIRST"
        );
    }
}
