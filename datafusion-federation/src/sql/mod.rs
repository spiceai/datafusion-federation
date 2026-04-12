mod analyzer;
pub mod ast_analyzer;
mod executor;
pub mod optimizer;
mod schema;
mod table;
mod table_reference;

use std::{any::Any, fmt, sync::Arc, vec};

use analyzer::{collect_known_rewrites, RewriteTableScanAnalyzer};
use ast_analyzer::RewriteMultiTableReference;
use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::{Schema, SchemaRef},
    common::DFSchema,
    common::{tree_node::TreeNode, Statistics},
    config::ConfigOptions,
    error::{DataFusionError, Result},
    execution::{context::SessionState, TaskContext},
    logical_expr::{Extension, LogicalPlan, Projection, Sort, SubqueryAlias},
    optimizer::{optimize_unions::OptimizeUnions, Analyzer, AnalyzerRule, Optimizer},
    physical_expr::{EquivalenceProperties, LexOrdering, create_physical_sort_expr},
    physical_plan::{
        execution_plan::{Boundedness, EmissionType},
        filter_pushdown::{
            ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
        },
        metrics::MetricsSet,
        sorts::sort::SortExec,
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PhysicalExpr, PlanProperties,
        SendableRecordBatchStream,
    },
    sql::{sqlparser::ast::Statement, unparser::Unparser},
};
use optimizer::{OptimizeProjectionsFederation, PushDownFilterFederation};

pub use ast_analyzer::{AstAnalyzer, AstAnalyzerRule};
pub use executor::{LogicalOptimizer, SQLExecutor, SQLExecutorRef, SqlQueryRewriter};
pub use schema::{MultiSchemaProvider, SQLSchemaProvider};
pub use table::{RemoteTable, SQLTable, SQLTableSource};
pub use table_reference::{MultiPartTableReference, RemoteTableRef};

use crate::{
    get_table_source, schema_cast, FederatedPlanNode, FederationAnalyzerForLogicalPlan,
    FederationAnalyzerRule, FederationPlanner, FederationProvider,
};

/// Returns a federation analyzer rule that is optimized for SQL federation.
pub fn federation_analyzer_rule() -> FederationAnalyzerRule {
    FederationAnalyzerRule::new().with_optimizer(Optimizer::with_rules(vec![
        Arc::new(OptimizeUnions::new()),
        Arc::new(PushDownFilterFederation::new()),
        Arc::new(OptimizeProjectionsFederation::new()),
    ]))
}

// SQLFederationProvider provides federation to SQL DMBSs.
#[derive(Debug)]
pub struct SQLFederationProvider {
    analyzer: Arc<Analyzer>,
    pub(crate) executor: Arc<dyn SQLExecutor>,
}

impl SQLFederationProvider {
    pub fn new(executor: Arc<dyn SQLExecutor>) -> Self {
        Self {
            analyzer: Arc::new(Analyzer::with_rules(vec![Arc::new(
                SQLFederationAnalyzerRule::new(Arc::clone(&executor)),
            )])),
            executor,
        }
    }

    pub fn executor(&self) -> &Arc<dyn SQLExecutor> {
        &self.executor
    }
}

impl FederationProvider for SQLFederationProvider {
    fn name(&self) -> &str {
        "sql_federation_provider"
    }

    fn compute_context(&self) -> Option<String> {
        self.executor.compute_context()
    }

    fn analyzer(&self, plan: &LogicalPlan) -> Option<FederationAnalyzerForLogicalPlan> {
        if self.executor.can_execute_plan(plan) {
            Some(Arc::clone(&self.analyzer).into())
        } else {
            Some(FederationAnalyzerForLogicalPlan::Unable)
        }
    }
}

#[derive(Debug)]
struct SQLFederationAnalyzerRule {
    planner: Arc<SQLFederationPlanner>,
}

impl SQLFederationAnalyzerRule {
    pub fn new(executor: Arc<dyn SQLExecutor>) -> Self {
        Self {
            planner: Arc::new(SQLFederationPlanner::new(Arc::clone(&executor))),
        }
    }
}

impl AnalyzerRule for SQLFederationAnalyzerRule {
    /// Try to rewrite `plan` to an optimized form.
    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        if let LogicalPlan::Extension(Extension { ref node }) = plan {
            if node.name() == "Federated" {
                // Avoid attempting double federation
                return Ok(plan);
            }
        }

        let mut plan = LogicalPlan::Extension(Extension {
            node: Arc::new(FederatedPlanNode::new(plan.clone(), self.planner.clone())),
        });
        if let Some(mut rewriter) = self.planner.executor.logical_optimizer() {
            plan = rewriter(plan)?;
        }

        Ok(plan)
    }

    /// A human readable name for this analyzer rule
    fn name(&self) -> &str {
        "federate_sql"
    }
}

#[derive(Debug)]
pub struct SQLFederationPlanner {
    pub(crate) executor: Arc<dyn SQLExecutor>,
}

impl SQLFederationPlanner {
    pub fn new(executor: Arc<dyn SQLExecutor>) -> Self {
        Self { executor }
    }

    pub fn executor(&self) -> &Arc<dyn SQLExecutor> {
        &self.executor
    }
}

#[async_trait]
impl FederationPlanner for SQLFederationPlanner {
    async fn plan_federation(
        &self,
        node: &FederatedPlanNode,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Arc::new(node.plan().schema().as_arrow().clone());
        let plan = node.plan().clone();
        let statistics = self.executor.statistics(&plan).await?;
        let input = Arc::new(VirtualExecutionPlan::new(
            plan.clone(),
            Arc::clone(&self.executor),
            statistics,
        ));
        let schema_cast_exec: Arc<dyn ExecutionPlan> =
            Arc::new(schema_cast::SchemaCastScanExec::new(input, schema));

        // When a `Sort` node is present at the top of the federated logical plan
        // (possibly behind a `Projection` or `SubqueryAlias`), the SQL Unparser
        // may push the `ORDER BY` inside a subquery:
        //
        //   SELECT ... FROM (SELECT ... ORDER BY col)   -- outer has no ORDER BY!
        //
        // SQL does not guarantee that ordering from a subquery is preserved by
        // the outer query.  When the remote engine returns data across multiple
        // batches the rows can arrive in arbitrary order, silently violating the
        // sort contract.
        //
        // Fix: detect the sort and add a local `SortExec` so DataFusion enforces
        // the required ordering regardless of what the remote engine returns.
        if let Some(sort) = find_top_sort(&plan) {
            // Resolve sort expressions against the federation output schema so
            // that fully-qualified column references (e.g. `t.schema.tbl.id`)
            // are correctly mapped to the output columns (e.g. `id`).
            let output_schema = plan.schema();
            let execution_props = session_state.execution_props();
            match sort
                .expr
                .iter()
                .map(|e| create_physical_sort_expr(e, output_schema.as_ref(), execution_props))
                .collect::<Result<Vec<_>>>()
            {
                Ok(physical_sort_exprs) if !physical_sort_exprs.is_empty() => {
                    if let Some(lex_ordering) = LexOrdering::new(physical_sort_exprs) {
                        return Ok(Arc::new(SortExec::new(lex_ordering, schema_cast_exec)));
                    }
                }
                _ => {} // fall through if resolution fails
            }
        }

        Ok(schema_cast_exec)
    }
}

/// Walk the top of a logical plan through transparent wrapper nodes
/// (`Projection`, `SubqueryAlias`) to find the first `Sort` node, if any.
///
/// Returns `None` if no `Sort` is encountered before a non-transparent node.
fn find_top_sort(plan: &LogicalPlan) -> Option<&Sort> {
    match plan {
        LogicalPlan::Sort(sort) => Some(sort),
        LogicalPlan::Projection(Projection { input, .. }) => find_top_sort(input),
        LogicalPlan::SubqueryAlias(SubqueryAlias { input, .. }) => find_top_sort(input),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct VirtualExecutionPlan {
    plan: LogicalPlan,
    executor: Arc<dyn SQLExecutor>,
    props: PlanProperties,
    statistics: Statistics,
    filters: Vec<Arc<dyn PhysicalExpr>>,
}

impl VirtualExecutionPlan {
    pub fn new(plan: LogicalPlan, executor: Arc<dyn SQLExecutor>, statistics: Statistics) -> Self {
        let schema: Schema = <DFSchema as AsRef<Schema>>::as_ref(plan.schema().as_ref()).clone();
        let props = PlanProperties::new(
            EquivalenceProperties::new(Arc::new(schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
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
        let df_schema = self.plan.schema().as_ref();
        Arc::new(<DFSchema as AsRef<Schema>>::as_ref(df_schema).clone())
    }

    fn final_sql(&self) -> Result<String> {
        let plan = self.plan.clone();
        let known_rewrites = collect_known_rewrites(&plan)?;
        let plan = RewriteTableScanAnalyzer::rewrite(plan, &known_rewrites)?;
        let (logical_optimizers, ast_analyzers, sql_query_rewriters) = gather_analyzers(&plan)?;
        let plan = apply_logical_optimizers(plan, logical_optimizers)?;
        let ast = self.plan_to_statement(&plan)?;
        let ast = self.rewrite_with_executor_ast_analyzer(ast)?;
        let mut ast = apply_ast_analyzers(ast, ast_analyzers)?;
        RewriteMultiTableReference::rewrite(&mut ast, known_rewrites);
        apply_sql_query_rewriters(ast.to_string(), sql_query_rewriters)
    }

    fn rewrite_with_executor_ast_analyzer(
        &self,
        ast: Statement,
    ) -> Result<Statement, datafusion::error::DataFusionError> {
        if let Some(mut analyzer) = self.executor.ast_analyzer() {
            Ok(analyzer.analyze(ast)?)
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
                "Schema altered during logical analysis, expected: {old_schema}, found: {new_schema}",
            )));
        }
    }
    Ok(plan)
}

fn apply_ast_analyzers(mut statement: Statement, analyzers: Vec<AstAnalyzer>) -> Result<Statement> {
    for mut analyzer in analyzers {
        statement = analyzer.analyze(statement)?;
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
        let known_rewrites = match collect_known_rewrites(&self.plan) {
            Ok(rewrites) => rewrites,
            Err(_) => return Ok(()),
        };
        let mut plan = match RewriteTableScanAnalyzer::rewrite(self.plan.clone(), &known_rewrites) {
            Ok(plan) => plan,
            Err(_) => self.plan.clone(),
        };
        if let Ok(statement) = self.plan_to_statement(&plan) {
            write!(f, " base_sql={statement}")?;
        }

        let (logical_optimizers, ast_analyzers, _sql_query_rewriters) =
            match gather_analyzers(&plan) {
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

        let final_sql = match self.final_sql() {
            Ok(sql) => sql,
            _ => return Ok(()),
        };
        if old_statement.to_string() != final_sql {
            write!(f, " rewritten_sql={final_sql}")?;
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

    fn properties(&self) -> &PlanProperties {
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

#[allow(clippy::type_complexity)]
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
    use datafusion::logical_expr::expr::Alias;
    use datafusion::logical_expr::Projection;
    use datafusion::prelude::Expr;
    use datafusion::sql::unparser::dialect::Dialect;
    use datafusion::sql::unparser::{self};
    use datafusion::{
        arrow::datatypes::{DataType, Field},
        datasource::TableProvider,
        execution::context::SessionContext,
    };

    use super::table::RemoteTable;
    use super::*;

    #[derive(Clone)]
    struct TestExecutor {
        compute_context: String,

        // Return true if this subtree of a logicalplan cannot be federated
        cannot_federate: Option<Arc<dyn Fn(&LogicalPlan) -> bool + Send + Sync>>,
    }

    impl std::fmt::Debug for TestExecutor {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("TestExecutor")
                .field("compute_context", &self.compute_context)
                .field("cannot_federate_fn", &self.cannot_federate.is_some())
                .finish_non_exhaustive()
        }
    }

    #[async_trait]
    impl SQLExecutor for TestExecutor {
        fn name(&self) -> &str {
            "TestExecutor"
        }

        fn compute_context(&self) -> Option<String> {
            Some(self.compute_context.clone())
        }

        fn can_execute_plan(&self, logical_plan: &LogicalPlan) -> bool {
            let Some(ref fnc) = self.cannot_federate else {
                return true;
            };
            !logical_plan.exists(|p| Ok(fnc(p))).unwrap_or(false)
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

        fn table_reference(&self) -> MultiPartTableReference {
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
            cannot_federate: None,
        };

        let test_executor_b = TestExecutor {
            compute_context: "b".into(),
            cannot_federate: None,
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
    async fn basic_sql_federation_analyzer_rule_test() -> Result<(), DataFusionError> {
        let alias_non_federate: Arc<dyn Fn(&LogicalPlan) -> bool + Send + Sync> =
            Arc::new(|plan| match plan {
                LogicalPlan::Projection(Projection { expr, .. }) => expr.iter().any(|e| match e {
                    Expr::Alias(Alias { name, .. }) => name == "non_federate",
                    _ => false,
                }),
                _ => false,
            });

        let test_executor_a = TestExecutor {
            compute_context: "a".into(),
            cannot_federate: Some(Arc::clone(&alias_non_federate)),
        };

        let test_executor_b = TestExecutor {
            compute_context: "b".into(),
            cannot_federate: None,
        };

        let table_a1_ref = "table_a1".to_string();
        let table_a1 = get_test_table_provider(table_a1_ref.clone(), test_executor_a.clone());

        let table_b1_ref = "table_b1".to_string();
        let table_b1 = get_test_table_provider(table_b1_ref.clone(), test_executor_b.clone());

        let table_b2_ref = "table_b2".to_string();
        let table_b2 = get_test_table_provider(table_b2_ref.clone(), test_executor_b);

        // Create a new SessionState with the optimizer rule we created above
        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);
        ctx.add_analyzer_rule(Arc::new(FederationAnalyzerRule::default()));

        ctx.register_table(table_a1_ref.clone(), table_a1).unwrap();
        ctx.register_table(table_b1_ref.clone(), table_b1).unwrap();
        ctx.register_table(table_b2_ref.clone(), table_b2).unwrap();

        // Basic unsupported federation of `AS 'non_federate'`. Note filter non_federate > 0 can be
        // pushed down since it will be optimised into `Filter: table_a1.a > Int64(0)`.
        insta::assert_snapshot!(ctx
            .sql(
                r#"SELECT a as non_federate, b, c FROM (SELECT a, b, c FROM table_a1) WHERE a > 0"#,
            )
            .await?
            .into_optimized_plan()?
            .display_indent(), @r"
        Projection: table_a1.a AS non_federate, table_a1.b, table_a1.c
          Federated
         Projection: table_a1.a, table_a1.b, table_a1.c
          Filter: table_a1.a > Int64(0)
            TableScan: table_a1
        ");

        // Basic join of two different context tables.
        insta::assert_snapshot!(ctx
            .sql(
                r#"SELECT b.a, b.b, a.b, a.c FROM table_a1 a JOIN table_b1 b ON a.a=b.a"#,
            )
            .await?
            .into_optimized_plan()?
            .display_indent(), @r"
        Projection: b.a, b.b, a.b, a.c
          Inner Join: a.a = b.a
            Federated
         Projection: a.a, a.b, a.c
          SubqueryAlias: a
            TableScan: table_a1
            Projection: b.a, b.b
              Federated
         Projection: b.a, b.b, b.c
          SubqueryAlias: b
            TableScan: table_b1
        "
        );

        // Basic join of two same-context tables.
        insta::assert_snapshot!(ctx
            .sql(
                r#"SELECT b.a, b.b, a.b, a.c FROM table_b1 a JOIN table_b2 b ON a.a=b.a"#,
            )
            .await?
            .into_optimized_plan()?
            .display_indent(), @r"
        Federated
         Projection: b.a, b.b, a.b, a.c
          Inner Join:  Filter: a.a = b.a
            SubqueryAlias: a
              TableScan: table_b1
            SubqueryAlias: b
              TableScan: table_b2
        "
        );

        // JOIN ON different contexts, one child has non-federateable [`LogicalPlan`].
        insta::assert_snapshot!(ctx
            .sql(
                r#"SELECT a.*, j.non_federate FROM (SELECT b.a AS a, b.b as 'non_federate', a.b as b, a.c as c FROM table_b1 a JOIN table_b2 b ON a.a=b.a) j JOIN table_a1 a ON j.a = a.a"#,
            )
            .await?
            .into_optimized_plan()?
            .display_indent(), @r"
        Projection: a.a, a.b, a.c, j.non_federate
          Inner Join: j.a = a.a
            Projection: j.a, j.non_federate
              Federated
         Projection: j.a, j.non_federate, j.b, j.c
          SubqueryAlias: j
            Projection: b.a, b.b AS non_federate, a.b, a.c
              Inner Join:  Filter: a.a = b.a
                SubqueryAlias: a
                  TableScan: table_b1
                SubqueryAlias: b
                  TableScan: table_b2
            Federated
         Projection: a.a, a.b, a.c
          SubqueryAlias: a
            TableScan: table_a1
        "
        );

        Ok(())
    }

    #[tokio::test]
    async fn multi_reference_sql_federation_test() -> Result<(), DataFusionError> {
        let test_executor_a = TestExecutor {
            compute_context: "test".into(),
            cannot_federate: None,
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
            cannot_federate: None,
        };

        let table_ref = "test_table".to_string();
        let table = get_test_table_provider(table_ref.clone(), executor);

        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_table(table_ref, table).unwrap();

        // EXPLAIN ANALYZE wraps the query in LogicalPlan::Analyze.
        // The federation analyzer must NOT wrap the Analyze node itself.
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
            cannot_federate: None,
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
        physical_plan.apply(|node| {
            if node.name() == "sql_federation_exec" {
                let node = node
                    .as_any()
                    .downcast_ref::<VirtualExecutionPlan>()
                    .unwrap();
                final_queries.push(node.final_sql()?);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        let [final_query] = final_queries.as_slice() else {
            panic!("expected a single federated SQL query");
        };

        assert!(final_query.ends_with("/* rewritten by sql_query_rewriter */"));
        assert_eq!(rewrite_calls.load(Ordering::SeqCst), 1);

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Tests for sort-ordering correctness (issue: federation does not preserve
    // sort ordering across multiple output batches).
    // -------------------------------------------------------------------------

    /// Verify that `find_top_sort` traverses through `Projection` and
    /// `SubqueryAlias` wrappers to locate a `Sort` node.
    #[test]
    fn find_top_sort_walks_projections() -> Result<(), DataFusionError> {
        use datafusion::common::DFSchema;
        use datafusion::logical_expr::{
            LogicalPlan, Projection, Sort, SubqueryAlias,
            SortExpr,
        };
        use datafusion::prelude::col;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let df_schema = Arc::new(DFSchema::try_from(schema.as_ref().clone())?);

        // Build a minimal Leaf plan (empty values) as the sort input.
        let empty = datafusion::logical_expr::LogicalPlan::EmptyRelation(
            datafusion::logical_expr::EmptyRelation {
                produce_one_row: false,
                schema: df_schema.clone(),
            },
        );

        let sort_expr = SortExpr {
            expr: col("id"),
            asc: true,
            nulls_first: false,
        };
        let sort_node = LogicalPlan::Sort(Sort {
            expr: vec![sort_expr],
            input: Arc::new(empty),
            fetch: None,
        });

        // find_top_sort on a bare Sort returns Some.
        assert!(find_top_sort(&sort_node).is_some(), "bare Sort");

        // Wrap in a Projection — find_top_sort should still find the Sort.
        let wrapped_in_proj = LogicalPlan::Projection(Projection::try_new(
            df_schema
                .columns()
                .iter()
                .map(|c| datafusion::prelude::Expr::Column(c.clone()))
                .collect(),
            Arc::new(sort_node.clone()),
        )?);
        assert!(
            find_top_sort(&wrapped_in_proj).is_some(),
            "Sort under Projection"
        );

        // Wrap in SubqueryAlias — find_top_sort should still find the Sort.
        let wrapped_in_alias = LogicalPlan::SubqueryAlias(SubqueryAlias::try_new(
            Arc::new(sort_node.clone()),
            "alias",
        )?);
        assert!(
            find_top_sort(&wrapped_in_alias).is_some(),
            "Sort under SubqueryAlias"
        );

        // A plan without any Sort at the top returns None.
        let empty2 = datafusion::logical_expr::LogicalPlan::EmptyRelation(
            datafusion::logical_expr::EmptyRelation {
                produce_one_row: false,
                schema: df_schema.clone(),
            },
        );
        assert!(find_top_sort(&empty2).is_none(), "no Sort");

        Ok(())
    }

    /// When a federated plan contains a top-level `Sort`, `plan_federation`
    /// must wrap the result in a `SortExec` to guarantee correct row ordering
    /// even when the remote engine returns data in multiple batches.
    ///
    /// Without this fix the SQL Unparser emits `ORDER BY` only inside a
    /// subquery, which SQL engines are not required to propagate to the outer
    /// query.
    #[tokio::test]
    async fn sort_exec_wraps_virtual_plan_for_ordered_query() -> Result<(), DataFusionError> {
        let executor = TestExecutor {
            compute_context: "sort_exec_test".into(),
            cannot_federate: None,
        };
        let table_ref = "t".to_string();
        let table = get_test_table_provider(table_ref.clone(), executor);

        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_table(table_ref.clone(), table).unwrap();

        // `ORDER BY a` at the top level: the whole plan (Sort → TableScan) is
        // federated.  Our fix should add a SortExec on top so that multi-batch
        // results arrive in the correct order.
        let plan = ctx
            .sql("SELECT a, b FROM t ORDER BY a ASC")
            .await?
            .into_optimized_plan()?;

        let physical_plan = ctx.state().create_physical_plan(&plan).await?;

        // Walk the physical plan: there must be a SortExec that wraps the
        // VirtualExecutionPlan (possibly through SchemaCastScanExec).
        let mut found_sort_over_virtual = false;
        physical_plan.apply(|node| {
            if node.name() == "SortExec" {
                node.apply(|child| {
                    if child.name() == "sql_federation_exec" {
                        found_sort_over_virtual = true;
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
                if found_sort_over_virtual {
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        assert!(
            found_sort_over_virtual,
            "Expected a SortExec wrapping VirtualExecutionPlan to enforce sort \
             ordering across multiple batches."
        );

        Ok(())
    }

    /// A query without `ORDER BY` must NOT get a spurious `SortExec` wrapping
    /// the `VirtualExecutionPlan`.
    #[tokio::test]
    async fn no_sort_exec_for_unordered_query() -> Result<(), DataFusionError> {
        let executor = TestExecutor {
            compute_context: "no_sort_exec_test".into(),
            cannot_federate: None,
        };
        let table_ref = "t".to_string();
        let table = get_test_table_provider(table_ref.clone(), executor);

        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_table(table_ref.clone(), table).unwrap();

        let plan = ctx
            .sql("SELECT a, b FROM t")
            .await?
            .into_optimized_plan()?;

        let physical_plan = ctx.state().create_physical_plan(&plan).await?;

        let mut sort_over_virtual = false;
        physical_plan.apply(|node| {
            if node.name() == "SortExec" {
                node.apply(|child| {
                    if child.name() == "sql_federation_exec" {
                        sort_over_virtual = true;
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        assert!(
            !sort_over_virtual,
            "Did not expect a SortExec wrapping VirtualExecutionPlan for an \
             unordered query."
        );

        Ok(())
    }
}
