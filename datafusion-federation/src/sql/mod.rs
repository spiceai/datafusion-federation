mod analyzer;
pub mod ast_analyzer;
mod executor;
pub mod optimizer;
mod remote_plan;
mod schema;
mod table;
mod table_reference;

use std::{fmt, sync::Arc, vec};

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
    logical_expr::{Extension, LogicalPlan},
    optimizer::{
        optimize_unions::OptimizeUnions, Analyzer, AnalyzerRule, Optimizer, OptimizerRule,
    },
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
use optimizer::{OptimizeProjectionsFederation, PushDownFilterFederation};

pub use ast_analyzer::{AstAnalyzer, AstAnalyzerRule};
pub use executor::{
    LogicalOptimizer, SQLExecutor, SQLExecutorRef, SQLFilterPushDown, SqlQueryRewriter,
};
pub use remote_plan::{AttachRemotePlans, RemotePlanExec, RemotePlanNode};
pub use schema::{MultiSchemaProvider, SQLSchemaProvider};
pub use table::{RemoteTable, SQLTable, SQLTableSource};
pub use table_reference::{MultiPartTableReference, RemoteTableRef};

use crate::{
    get_table_source, schema_cast, FederatedPlanNode, FederatedQueryType,
    FederationAnalyzerForLogicalPlan, FederationAnalyzerRule, FederationPlanner,
    FederationProvider,
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

    fn pre_federation_optimizer_rules(&self) -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
        self.executor.pre_federation_optimizer_rules()
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
        let (plan, query_type) = match plan {
            LogicalPlan::Explain(explain) => (
                explain.plan.as_ref().clone(),
                Some(FederatedQueryType::Explain),
            ),
            LogicalPlan::Analyze(analyze) => (
                analyze.input.as_ref().clone(),
                Some(FederatedQueryType::Analyze),
            ),
            plan => (plan, None),
        };

        if let LogicalPlan::Extension(Extension { ref node }) = plan {
            if node.name() == "Federated" {
                // Avoid attempting double federation
                return Ok(plan);
            }
        }

        let mut plan = LogicalPlan::Extension(Extension {
            node: Arc::new(FederatedPlanNode::new_with_query_type(
                plan.clone(),
                self.planner.clone(),
                query_type,
            )),
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
        _session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Arc::new(node.plan().schema().as_arrow().clone());
        let plan = node.plan().clone();
        let statistics = self.executor.statistics(&plan).await?;
        let virtual_plan = VirtualExecutionPlan::new(
            plan,
            Arc::clone(&self.executor),
            statistics,
            node.query_type(),
        );

        // When the query is being explained, fetch the remote engine's own plan now —
        // this is the only place an async call can be made. It is held pending, not
        // attached, until [`AttachRemotePlans`] runs after physical optimization.
        let virtual_plan = match node.query_type() {
            Some(query_type) => {
                let sql = virtual_plan.final_sql()?;
                match self.executor.explain_plan(&sql, query_type).await? {
                    Some(remote_plan) => virtual_plan.with_remote_plan(remote_plan),
                    None => virtual_plan,
                }
            }
            None => virtual_plan,
        };

        let schema_cast_exec = schema_cast::SchemaCastScanExec::new(Arc::new(virtual_plan), schema);
        Ok(Arc::new(schema_cast_exec))
    }
}

#[derive(Debug, Clone)]
pub struct VirtualExecutionPlan {
    plan: LogicalPlan,
    executor: Arc<dyn SQLExecutor>,
    props: Arc<PlanProperties>,
    statistics: Statistics,
    query_type: Option<FederatedQueryType>,
    filters: Vec<Arc<dyn PhysicalExpr>>,
    /// The remote engine's plan, once [`AttachRemotePlans`] has materialized it.
    remote_plan: Option<Arc<dyn ExecutionPlan>>,
    /// The remote engine's plan as fetched during planning. Held aside rather than
    /// exposed through `children()` so this node stays a leaf while DataFusion
    /// optimizes: rules such as `EnsureCooperative` and `EnforceDistribution` treat
    /// leaves differently, and grafting early changes the plan that runs.
    pending_remote_plan: Option<RemotePlanNode>,
}

impl VirtualExecutionPlan {
    pub fn new(
        plan: LogicalPlan,
        executor: Arc<dyn SQLExecutor>,
        statistics: Statistics,
        query_type: Option<FederatedQueryType>,
    ) -> Self {
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
            props: Arc::new(props),
            statistics,
            query_type,
            filters: Vec::new(),
            remote_plan: None,
            pending_remote_plan: None,
        }
    }

    /// Records the remote engine's plan, to be attached as this node's child once
    /// physical optimization is done. See [`AttachRemotePlans`].
    #[must_use]
    pub fn with_remote_plan(mut self, remote_plan: RemotePlanNode) -> Self {
        self.pending_remote_plan = Some(remote_plan);
        self
    }

    /// Turns a pending remote plan into this node's child. `None` when there is
    /// nothing to attach, so callers can leave the plan untouched.
    fn materialize_remote_plan(&self) -> Option<Self> {
        let pending = self.pending_remote_plan.clone()?;
        Some(Self {
            remote_plan: Some(RemotePlanExec::new(pending)),
            pending_remote_plan: None,
            ..self.clone()
        })
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

    /// The directive this query was federated under, if it was explained.
    pub fn query_type(&self) -> Option<FederatedQueryType> {
        self.query_type
    }

    /// The remote engine's plan, present only when the query is being explained and
    /// the executor implements [`SQLExecutor::explain_plan`].
    pub fn remote_plan(&self) -> Option<&Arc<dyn ExecutionPlan>> {
        self.remote_plan.as_ref()
    }

    fn schema(&self) -> SchemaRef {
        let df_schema = self.plan.schema().as_ref();
        Arc::new(<DFSchema as AsRef<Schema>>::as_ref(df_schema).clone())
    }

    /// The SQL sent to the remote engine.
    ///
    /// `EXPLAIN`/`EXPLAIN ANALYZE` never change this. The directive would make the
    /// remote return its own plan text — a different schema than
    /// [`schema_cast::SchemaCastScanExec`] expects, and for SQLite not even valid
    /// syntax. The remote plan is obtained separately, via
    /// [`SQLExecutor::explain_plan`], and attached as this node's child.
    fn final_sql(&self) -> Result<String> {
        self.rewrite_plan_to_sql(self.plan.clone())
    }

    fn rewrite_plan_to_sql(&self, plan: LogicalPlan) -> Result<String> {
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
            if let Some(source) =
                (provider.as_ref() as &dyn std::any::Any).downcast_ref::<SQLTableSource>()
            {
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
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        // `EXPLAIN FORMAT TREE` lays each node out in a narrow box and splits the
        // text on newlines, so the single-line form below word-wraps into unreadable
        // fragments. Emit one `key=value` per line instead, and only the SQL that is
        // actually sent — the intermediate rewrites belong in the indent format.
        if matches!(t, DisplayFormatType::TreeRender) {
            writeln!(f, "name={}", self.executor.name())?;
            if let Some(ctx) = self.executor.compute_context() {
                writeln!(f, "compute_context={ctx}")?;
            }
            if let Ok(sql) = self.final_sql() {
                writeln!(f, "sql={sql}")?;
            }
            return Ok(());
        }

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

    fn schema(&self) -> SchemaRef {
        self.schema()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        // Only ever the remote engine's plan, and only while explaining. It is
        // display-only, and `with_new_children` below keeps it out of reach of
        // physical optimizer rules.
        self.remote_plan.iter().collect()
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

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        Ok(Arc::new(self.statistics.clone()))
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
        // Ask the executor whether it will apply each filter inside execute().
        // Filters the executor claims to handle are accepted (PushedDown::Yes), allowing
        // the parent FilterExec to be removed — the executor is then responsible for
        // applying them (e.g. by injecting them into the SQL at execution time).
        // Filters the executor does not handle are declined (PushedDown::No), keeping
        // the FilterExec in place for local evaluation.
        //
        // Note: filters that were part of the federation plan are already baked into
        // final_sql() and never appear here — this path only sees filters that were
        // NOT absorbed during logical federation planning.
        let filter_refs: Vec<&dyn PhysicalExpr> = child_pushdown_result
            .parent_filters
            .iter()
            .map(|f| f.filter.as_ref())
            .collect();
        let executor_support = self.executor.supports_filters_pushdown(&filter_refs);

        // Exact   → pass to execute(), remove FilterExec (PushedDown::Yes)
        // Inexact → pass to execute() as a hint, keep FilterExec (PushedDown::No)
        // Unsupported → do not pass, keep FilterExec (PushedDown::No)
        let pushdown_results: Vec<PushedDown> = executor_support
            .iter()
            .map(|s| match s {
                SQLFilterPushDown::Exact => PushedDown::Yes,
                SQLFilterPushDown::Inexact | SQLFilterPushDown::Unsupported => PushedDown::No,
            })
            .collect();

        let accepted: Vec<Arc<dyn PhysicalExpr>> = child_pushdown_result
            .parent_filters
            .iter()
            .zip(&executor_support)
            .filter(|(_, s)| matches!(s, SQLFilterPushDown::Exact | SQLFilterPushDown::Inexact))
            .map(|(f, _)| Arc::clone(&f.filter))
            .collect();

        if accepted.is_empty() {
            return Ok(FilterPushdownPropagation {
                filters: pushdown_results,
                updated_node: None,
            });
        }

        let mut node = self.clone();
        node.filters = accepted;
        Ok(FilterPushdownPropagation {
            filters: pushdown_results,
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
    use datafusion::arrow::array::Array;
    use datafusion::arrow::datatypes::{Schema, SchemaRef};
    use datafusion::common::tree_node::TreeNodeRecursion;
    use datafusion::execution::SendableRecordBatchStream;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::logical_expr::expr::Alias;
    use datafusion::logical_expr::Projection;
    use datafusion::optimizer::eliminate_filter::EliminateFilter;
    use datafusion::optimizer::OptimizerRule;
    use datafusion::prelude::Expr;
    use datafusion::sql::unparser::dialect::Dialect;
    use datafusion::sql::unparser::{self};
    use datafusion::{
        arrow::datatypes::{DataType, Field},
        datasource::TableProvider,
        execution::config::SessionConfig,
        execution::context::SessionContext,
    };

    use super::table::RemoteTable;
    use super::*;
    use crate::FederatedQueryPlanner;

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

        fn pre_federation_optimizer_rules(&self) -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
            if self.compute_context == "pre_federation_optimizer" {
                vec![Arc::new(EliminateFilter::new())]
            } else {
                vec![]
            }
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
                let node = node.downcast_ref::<VirtualExecutionPlan>().unwrap();

                final_queries.push(node.final_sql()?);
            }
            Ok(TreeNodeRecursion::Continue)
        });

        let expected = vec![
            "SELECT table_a1.a, table_a1.b, table_a1.c FROM table_a1",
            "SELECT table_a2.a, table_a2.b, table_a2.c FROM table_a2",
            "SELECT table_b1.a, table_b1.b, table_b1.c FROM table_b1(1) table_b1",
        ];

        assert_eq!(
            HashSet::<&str>::from_iter(final_queries.iter().map(|x| x.as_str())),
            HashSet::from_iter(expected)
        );

        Ok(())
    }

    #[tokio::test]
    async fn provider_optimizer_runs_before_capability_check() -> Result<(), DataFusionError> {
        let executor = TestExecutor {
            compute_context: "pre_federation_optimizer".into(),
            cannot_federate: Some(Arc::new(|plan| matches!(plan, LogicalPlan::Filter(_)))),
        };
        let table_name = "provider_optimizer_table";
        let table = get_test_table_provider(table_name.to_string(), executor);
        let ctx = SessionContext::new();
        ctx.register_table(table_name, table)?;

        let plan = ctx
            .state()
            .create_logical_plan(&format!("SELECT * FROM {table_name} WHERE true"))
            .await?;
        let analyzed = federation_analyzer_rule().analyze(plan, &ConfigOptions::default())?;

        let LogicalPlan::Extension(Extension { node }) = analyzed else {
            panic!("expected the provider optimizer to make the full plan federatable");
        };
        let federated = node
            .as_any()
            .downcast_ref::<FederatedPlanNode>()
            .expect("expected a FederatedPlanNode");
        assert!(
            !federated
                .plan()
                .exists(|plan| Ok(matches!(plan, LogicalPlan::Filter(_))))?,
            "the plan checked for capability and wrapped for federation must be optimized"
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
                let node = node.downcast_ref::<VirtualExecutionPlan>().unwrap();

                final_queries.push(node.final_sql()?);
            }
            Ok(TreeNodeRecursion::Continue)
        });

        let expected = vec![
            r#"SELECT "table".a, "table".b, "table".c FROM "default"."table" UNION ALL SELECT "Table".a, "Table".b, "Table".c FROM "default"."Table"(1) Table"#,
        ];

        assert_eq!(
            HashSet::<&str>::from_iter(final_queries.iter().map(|x| x.as_str())),
            HashSet::from_iter(expected)
        );

        Ok(())
    }

    #[tokio::test]
    async fn explain_federation_test() -> Result<(), DataFusionError> {
        let executor = TestExecutor {
            compute_context: "test".into(),
            cannot_federate: None,
        };

        let local_table_ref = "table_local_explain".to_string();
        let remote_table_ref = "table_b1(1)".to_string();
        let table = get_test_table_provider(remote_table_ref, executor);

        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);
        ctx.register_table(local_table_ref, table).unwrap();

        let plan = ctx
            .sql("EXPLAIN SELECT * FROM table_local_explain")
            .await?
            .into_optimized_plan()?;

        let LogicalPlan::Explain(ref explain) = plan else {
            panic!("Expected Explain at root, got: {}", plan.display_indent());
        };

        let mut found_federated = false;
        explain.plan.apply(|node| {
            if let LogicalPlan::Extension(extension) = node {
                if extension
                    .node
                    .as_any()
                    .downcast_ref::<FederatedPlanNode>()
                    .is_some()
                {
                    found_federated = true;
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        assert!(
            found_federated,
            "Expected a Federated node inside the Explain plan"
        );

        let annotated_plan = crate::FederatedQueryPlanner::annotate_query_directives(&plan)?
            .expect("Explain root should be annotated");
        let LogicalPlan::Explain(ref annotated_explain) = annotated_plan else {
            panic!(
                "Expected annotated Explain at root, got: {}",
                annotated_plan.display_indent()
            );
        };

        let physical_plan = ctx
            .state()
            .create_physical_plan(annotated_explain.plan.as_ref())
            .await?;
        let mut final_queries = vec![];
        physical_plan.apply(|node| {
            if node.name() == "sql_federation_exec" {
                let node = node.downcast_ref::<VirtualExecutionPlan>().unwrap();
                final_queries.push(node.final_sql()?);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        // The query itself is sent unchanged; the directive is not injected into it.
        // The remote plan comes from `SQLExecutor::explain_plan` instead, which this
        // mock executor does not implement.
        assert_eq!(
            final_queries,
            vec!["SELECT table_b1.a, table_b1.b, table_b1.c FROM table_b1(1) table_b1".to_string()]
        );

        let batches = ctx
            .sql("EXPLAIN SELECT * FROM table_local_explain")
            .await?
            .collect()
            .await?;

        let explain_output: Vec<String> = batches
            .iter()
            .flat_map(|batch| {
                let col = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<datafusion::arrow::array::StringArray>()
                    .unwrap();
                (0..col.len()).map(move |index| col.value(index).to_string())
            })
            .collect();

        let combined = explain_output.join("\n");
        assert!(
            combined.contains(
                "rewritten_sql=SELECT table_b1.a, table_b1.b, table_b1.c FROM table_b1(1) table_b1"
            ),
            "Expected the un-prefixed remote SQL in output, got:\n{combined}",
        );
        assert!(
            !combined.contains("EXPLAIN SELECT"),
            "The directive must not be injected into the federated query, got:\n{combined}",
        );

        Ok(())
    }

    /// EXPLAIN ANALYZE keeps AnalyzeExec at the DataFusion level, and the federated
    /// query is sent unchanged so it returns rows for AnalyzeExec to measure. The
    /// remote engine's own plan is fetched separately, via
    /// [`SQLExecutor::explain_plan`], which this mock executor does not implement.
    #[tokio::test]
    async fn explain_analyze_federation_test() -> Result<(), DataFusionError> {
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

        let mut final_queries = vec![];
        physical_plan.apply(|node| {
            if node.name() == "sql_federation_exec" {
                let node = node.downcast_ref::<VirtualExecutionPlan>().unwrap();
                final_queries.push(node.final_sql()?);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        assert_eq!(
            final_queries,
            vec!["SELECT test_table.a, test_table.b, test_table.c FROM test_table".to_string()]
        );

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
                let node = node.downcast_ref::<VirtualExecutionPlan>().unwrap();
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

    // --- EXISTS / NOT EXISTS federation tests ---

    fn make_table_with_schema(
        name: &str,
        schema: SchemaRef,
        executor: &TestExecutor,
    ) -> Arc<dyn TableProvider> {
        let table_ref = RemoteTableRef::try_from(name.to_string()).unwrap();
        let table = Arc::new(RemoteTable::new(table_ref, schema));
        let provider = Arc::new(SQLFederationProvider::new(Arc::new(executor.clone())));
        let table_source = Arc::new(SQLTableSource { provider, table });
        Arc::new(FederatedTableProviderAdaptor::new(table_source))
    }

    fn orders_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("o_orderkey", DataType::Int64, false),
            Field::new("o_custkey", DataType::Int64, false),
            Field::new("o_orderstatus", DataType::Utf8, false),
            Field::new("o_orderdate", DataType::Date32, true),
        ]))
    }

    fn lineitem_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("l_orderkey", DataType::Int64, false),
            Field::new("l_suppkey", DataType::Int64, false),
            Field::new("l_commitdate", DataType::Date32, true),
            Field::new("l_receiptdate", DataType::Date32, true),
        ]))
    }

    fn supplier_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("s_suppkey", DataType::Int64, false),
            Field::new("s_name", DataType::Utf8, false),
            Field::new("s_nationkey", DataType::Int64, false),
        ]))
    }

    /// Creates a session state with a fixed `target_partitions` count so that
    /// snapshot tests produce deterministic physical plans regardless of the
    /// number of CPU cores on the host.
    fn deterministic_session_state() -> SessionState {
        let rules = crate::default_analyzer_rules();
        SessionStateBuilder::new()
            .with_config(SessionConfig::default().with_target_partitions(4))
            .with_analyzer_rules(rules)
            .with_query_planner(Arc::new(FederatedQueryPlanner::new()))
            .with_default_features()
            .build()
    }

    /// Runs `EXPLAIN <query>`, collects the output, and returns a formatted
    /// string containing both logical and physical plans.
    async fn explain_query(ctx: &SessionContext, query: &str) -> String {
        let explain_sql = format!("EXPLAIN {query}");
        let batches = ctx
            .sql(&explain_sql)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        let formatted = datafusion::arrow::util::pretty::pretty_format_batches(&batches).unwrap();
        formatted.to_string()
    }

    /// Same-provider EXISTS: both tables from the same compute context.
    /// The entire plan should be federated as a single unit so the backend
    /// can decorrelate EXISTS into a semi-join.
    #[tokio::test]
    async fn same_provider_exists_federated_as_single_unit() {
        let executor_a = TestExecutor {
            compute_context: "ctx_a".into(),
            cannot_federate: None,
        };

        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_table(
            "orders",
            make_table_with_schema("orders", orders_schema(), &executor_a),
        )
        .unwrap();
        ctx.register_table(
            "lineitem",
            make_table_with_schema("lineitem", lineitem_schema(), &executor_a),
        )
        .unwrap();

        insta::assert_snapshot!(
            "same_provider_exists",
            explain_query(
                &ctx,
                "SELECT o_orderkey FROM orders WHERE EXISTS \
                 (SELECT 1 FROM lineitem WHERE l_orderkey = o_orderkey)",
            )
            .await
        );
    }

    /// Same-provider NOT EXISTS: mirrors TPC-H Q21 structure.
    /// Must be federated as one unit so the backend can decorrelate to anti-join.
    #[tokio::test]
    async fn same_provider_not_exists_federated_as_single_unit() {
        let executor_a = TestExecutor {
            compute_context: "ctx_a".into(),
            cannot_federate: None,
        };

        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_table(
            "supplier",
            make_table_with_schema("supplier", supplier_schema(), &executor_a),
        )
        .unwrap();
        ctx.register_table(
            "lineitem",
            make_table_with_schema("lineitem", lineitem_schema(), &executor_a),
        )
        .unwrap();

        insta::assert_snapshot!(
            "same_provider_not_exists",
            explain_query(
                &ctx,
                "SELECT s_name FROM supplier WHERE NOT EXISTS \
                 (SELECT 1 FROM lineitem WHERE l_suppkey = s_suppkey \
                  AND l_receiptdate > l_commitdate)",
            )
            .await
        );
    }

    /// Cross-provider EXISTS: outer table on provider A, subquery table on provider B.
    /// Each side must be independently federated (multiple Federated nodes).
    #[tokio::test]
    async fn cross_provider_exists_separately_federated() {
        let executor_a = TestExecutor {
            compute_context: "ctx_a".into(),
            cannot_federate: None,
        };
        let executor_b = TestExecutor {
            compute_context: "ctx_b".into(),
            cannot_federate: None,
        };

        let state = deterministic_session_state();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_table(
            "orders",
            make_table_with_schema("orders", orders_schema(), &executor_a),
        )
        .unwrap();
        ctx.register_table(
            "lineitem",
            make_table_with_schema("lineitem", lineitem_schema(), &executor_b),
        )
        .unwrap();

        insta::assert_snapshot!(
            "cross_provider_exists",
            explain_query(
                &ctx,
                "SELECT o_orderkey FROM orders WHERE EXISTS \
                 (SELECT 1 FROM lineitem WHERE l_orderkey = o_orderkey)",
            )
            .await
        );
    }

    /// Cross-provider NOT EXISTS: outer table on provider A, subquery table on provider B.
    #[tokio::test]
    async fn cross_provider_not_exists_separately_federated() {
        let executor_a = TestExecutor {
            compute_context: "ctx_a".into(),
            cannot_federate: None,
        };
        let executor_b = TestExecutor {
            compute_context: "ctx_b".into(),
            cannot_federate: None,
        };

        let state = deterministic_session_state();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_table(
            "orders",
            make_table_with_schema("orders", orders_schema(), &executor_a),
        )
        .unwrap();
        ctx.register_table(
            "lineitem",
            make_table_with_schema("lineitem", lineitem_schema(), &executor_b),
        )
        .unwrap();

        insta::assert_snapshot!(
            "cross_provider_not_exists",
            explain_query(
                &ctx,
                "SELECT o_orderkey FROM orders WHERE NOT EXISTS \
                 (SELECT 1 FROM lineitem WHERE l_orderkey = o_orderkey)",
            )
            .await
        );
    }

    /// TPC-H Q21 pattern: same-provider JOIN + NOT EXISTS + EXISTS.
    /// All tables on the same provider. The entire plan must be federated
    /// as a single unit so the backend handles decorrelation.
    #[tokio::test]
    async fn same_provider_join_with_not_exists_federated_as_single_unit() {
        let executor_a = TestExecutor {
            compute_context: "ctx_a".into(),
            cannot_federate: None,
        };

        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_table(
            "supplier",
            make_table_with_schema("supplier", supplier_schema(), &executor_a),
        )
        .unwrap();
        ctx.register_table(
            "lineitem",
            make_table_with_schema("lineitem", lineitem_schema(), &executor_a),
        )
        .unwrap();
        ctx.register_table(
            "orders",
            make_table_with_schema("orders", orders_schema(), &executor_a),
        )
        .unwrap();

        insta::assert_snapshot!(
            "same_provider_join_not_exists",
            explain_query(
                &ctx,
                "SELECT s_name \
                 FROM supplier \
                 JOIN lineitem ON s_suppkey = l_suppkey \
                 JOIN orders ON o_orderkey = l_orderkey \
                 WHERE NOT EXISTS ( \
                     SELECT 1 FROM lineitem AS l2 \
                     WHERE l2.l_orderkey = lineitem.l_orderkey \
                     AND l2.l_suppkey <> lineitem.l_suppkey \
                 )",
            )
            .await
        );
    }

    /// Mixed providers: JOIN across providers + EXISTS subquery on a different provider.
    /// supplier(ctx_a) JOIN lineitem(ctx_b) WHERE EXISTS on orders(ctx_a).
    #[tokio::test]
    async fn mixed_provider_join_with_exists() {
        let executor_a = TestExecutor {
            compute_context: "ctx_a".into(),
            cannot_federate: None,
        };
        let executor_b = TestExecutor {
            compute_context: "ctx_b".into(),
            cannot_federate: None,
        };

        let state = deterministic_session_state();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_table(
            "supplier",
            make_table_with_schema("supplier", supplier_schema(), &executor_a),
        )
        .unwrap();
        ctx.register_table(
            "lineitem",
            make_table_with_schema("lineitem", lineitem_schema(), &executor_b),
        )
        .unwrap();
        ctx.register_table(
            "orders",
            make_table_with_schema("orders", orders_schema(), &executor_a),
        )
        .unwrap();

        insta::assert_snapshot!(
            "mixed_provider_join_exists",
            explain_query(
                &ctx,
                "SELECT s_name \
                 FROM supplier \
                 JOIN lineitem ON l_suppkey = s_suppkey \
                 WHERE EXISTS ( \
                     SELECT 1 FROM orders \
                     WHERE o_custkey = s_suppkey \
                 )",
            )
            .await
        );
    }

    /// Same-provider NOT EXISTS with aliased outer table.
    /// The outer query uses `lineitem l1` (alias). The NOT EXISTS subquery
    /// references the alias: `l1.l_orderkey`. All tables are same provider.
    #[tokio::test]
    async fn same_provider_aliased_table_not_exists() {
        let executor_a = TestExecutor {
            compute_context: "ctx_a".into(),
            cannot_federate: None,
        };

        let state = crate::default_session_state();
        let ctx = SessionContext::new_with_state(state);

        ctx.register_table(
            "supplier",
            make_table_with_schema("supplier", supplier_schema(), &executor_a),
        )
        .unwrap();
        ctx.register_table(
            "lineitem",
            make_table_with_schema("lineitem", lineitem_schema(), &executor_a),
        )
        .unwrap();
        ctx.register_table(
            "orders",
            make_table_with_schema("orders", orders_schema(), &executor_a),
        )
        .unwrap();

        // TPC-H Q21 pattern: aliased lineitem l1, NOT EXISTS references l1.*
        insta::assert_snapshot!(
            "same_provider_aliased_not_exists",
            explain_query(
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
            .await
        );
    }

    /// When `can_execute_plan` returns `false` for a `Filter → TableScan` plan at a
    /// non-root level, the federation analyzer must still federate the `TableScan`
    /// child and leave the `Filter` above it for local execution.
    ///
    /// Before the fix the `(false, Some(_))` match arm caught `Some(Unable)` and
    /// returned early without trying children, so nothing got federated at all.
    #[tokio::test]
    async fn can_execute_plan_non_root_unable_federates_child_table_scan(
    ) -> Result<(), DataFusionError> {
        // Block federation of any plan node that is a Filter — simulates a
        // denied UDF in the WHERE clause.
        let executor = TestExecutor {
            compute_context: "ctx".into(),
            cannot_federate: Some(Arc::new(|plan| matches!(plan, LogicalPlan::Filter(_)))),
        };

        let table = get_test_table_provider("t".into(), executor);
        let ctx = SessionContext::new_with_state(crate::default_session_state());
        ctx.register_table("t", table).unwrap();

        let df = ctx.sql("SELECT * FROM t WHERE a > 5").await?;
        let physical_plan = df.create_physical_plan().await?;

        let mut has_filter_exec = false;
        let mut federation_sqls: Vec<String> = Vec::new();

        physical_plan.apply(|node| {
            if node.name() == "FilterExec" {
                has_filter_exec = true;
            }
            if node.name() == "sql_federation_exec" {
                let vp = node.downcast_ref::<VirtualExecutionPlan>().unwrap();
                federation_sqls.push(vp.final_sql()?);
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        assert!(
            has_filter_exec,
            "FilterExec must be present for the denied filter"
        );
        assert!(
            !federation_sqls.is_empty(),
            "VirtualExecutionPlan must be present — TableScan should still be federated"
        );
        for sql in &federation_sqls {
            assert!(
                !sql.to_lowercase().contains("where"),
                "Federated SQL must not contain the denied filter predicate; got: {sql}"
            );
        }

        Ok(())
    }

    // ── helpers shared by the filter-pushdown tests ──────────────────────────

    fn make_vp_with_executor(executor: TestExecutor) -> VirtualExecutionPlan {
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let plan = datafusion::logical_expr::LogicalPlan::EmptyRelation(
            datafusion::logical_expr::EmptyRelation {
                produce_one_row: false,
                schema: Arc::new(
                    datafusion::common::DFSchema::try_from((*schema).clone()).unwrap(),
                ),
            },
        );
        VirtualExecutionPlan::new(
            plan,
            Arc::new(executor),
            Statistics::new_unknown(&schema),
            None,
        )
    }

    fn make_child_result(n: usize) -> ChildPushdownResult {
        use datafusion::common::ScalarValue;
        use datafusion::physical_expr::expressions::Literal;
        use datafusion::physical_plan::filter_pushdown::ChildFilterPushdownResult;

        let dummy: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(ScalarValue::Boolean(Some(true))));
        let parent_filters = (0..n)
            .map(|_| ChildFilterPushdownResult {
                filter: Arc::clone(&dummy),
                child_results: vec![PushedDown::Yes],
            })
            .collect();
        ChildPushdownResult {
            parent_filters,
            self_filters: vec![],
        }
    }

    fn run_pushdown(
        vp: &VirtualExecutionPlan,
        n: usize,
    ) -> FilterPushdownPropagation<Arc<dyn ExecutionPlan>> {
        use datafusion::config::ConfigOptions;
        use datafusion::physical_plan::filter_pushdown::FilterPushdownPhase;
        vp.handle_child_pushdown_result(
            FilterPushdownPhase::Post,
            make_child_result(n),
            &ConfigOptions::default(),
        )
        .unwrap()
    }

    // ── per-variant tests ─────────────────────────────────────────────────────

    /// Default executor: all filters unsupported → PushedDown::No, no node update.
    #[test]
    fn virtual_execution_plan_declines_filter_pushdown() {
        let vp = make_vp_with_executor(TestExecutor {
            compute_context: "ctx".into(),
            cannot_federate: None,
        });
        let result = run_pushdown(&vp, 2);

        assert!(
            result.updated_node.is_none(),
            "Unsupported: node must not be updated"
        );
        assert!(
            result.filters.iter().all(|f| matches!(f, PushedDown::No)),
            "Unsupported: every filter must be PushedDown::No; got: {:?}",
            result.filters
        );
    }

    /// Executor that accepts all filters as Exact: FilterExec is removed (PushedDown::Yes)
    /// and filters are stored on the updated node for injection into execute().
    #[test]
    fn virtual_execution_plan_exact_filter_pushdown() {
        #[derive(Clone, Debug)]
        struct ExactExecutor(TestExecutor);

        #[async_trait]
        impl SQLExecutor for ExactExecutor {
            fn name(&self) -> &str {
                self.0.name()
            }
            fn compute_context(&self) -> Option<String> {
                self.0.compute_context()
            }
            fn dialect(&self) -> Arc<dyn Dialect> {
                self.0.dialect()
            }
            fn supports_filters_pushdown(
                &self,
                filters: &[&dyn PhysicalExpr],
            ) -> Vec<SQLFilterPushDown> {
                vec![SQLFilterPushDown::Exact; filters.len()]
            }
            fn execute(
                &self,
                q: &str,
                s: SchemaRef,
                f: &[Arc<dyn PhysicalExpr>],
            ) -> Result<SendableRecordBatchStream> {
                self.0.execute(q, s, f)
            }
            async fn table_names(&self) -> Result<Vec<String>> {
                self.0.table_names().await
            }
            async fn get_table_schema(&self, t: &str) -> Result<SchemaRef> {
                self.0.get_table_schema(t).await
            }
        }

        let vp = make_vp_with_executor(TestExecutor {
            compute_context: "ctx".into(),
            cannot_federate: None,
        });
        // Re-wrap with ExactExecutor
        let vp2 = VirtualExecutionPlan {
            executor: Arc::new(ExactExecutor(TestExecutor {
                compute_context: "ctx".into(),
                cannot_federate: None,
            })),
            ..vp
        };
        let result = run_pushdown(&vp2, 2);

        assert!(
            result.updated_node.is_some(),
            "Exact: node must be updated with accepted filters"
        );
        assert!(
            result.filters.iter().all(|f| matches!(f, PushedDown::Yes)),
            "Exact: every filter must be PushedDown::Yes (FilterExec removed); got: {:?}",
            result.filters
        );
        // Filters must be stored on the updated node for execute().
        let updated = result.updated_node.unwrap();
        let updated_vp = updated.downcast_ref::<VirtualExecutionPlan>().unwrap();
        assert_eq!(
            updated_vp.filters.len(),
            2,
            "Exact: both filters must be stored on the node"
        );
    }

    /// Executor that accepts all filters as Inexact: FilterExec is kept (PushedDown::No)
    /// but filters are stored on the updated node so execute() can use them as hints.
    #[test]
    fn virtual_execution_plan_inexact_filter_pushdown() {
        #[derive(Clone, Debug)]
        struct InexactExecutor(TestExecutor);

        #[async_trait]
        impl SQLExecutor for InexactExecutor {
            fn name(&self) -> &str {
                self.0.name()
            }
            fn compute_context(&self) -> Option<String> {
                self.0.compute_context()
            }
            fn dialect(&self) -> Arc<dyn Dialect> {
                self.0.dialect()
            }
            fn supports_filters_pushdown(
                &self,
                filters: &[&dyn PhysicalExpr],
            ) -> Vec<SQLFilterPushDown> {
                vec![SQLFilterPushDown::Inexact; filters.len()]
            }
            fn execute(
                &self,
                q: &str,
                s: SchemaRef,
                f: &[Arc<dyn PhysicalExpr>],
            ) -> Result<SendableRecordBatchStream> {
                self.0.execute(q, s, f)
            }
            async fn table_names(&self) -> Result<Vec<String>> {
                self.0.table_names().await
            }
            async fn get_table_schema(&self, t: &str) -> Result<SchemaRef> {
                self.0.get_table_schema(t).await
            }
        }

        let vp = make_vp_with_executor(TestExecutor {
            compute_context: "ctx".into(),
            cannot_federate: None,
        });
        let vp2 = VirtualExecutionPlan {
            executor: Arc::new(InexactExecutor(TestExecutor {
                compute_context: "ctx".into(),
                cannot_federate: None,
            })),
            ..vp
        };
        let result = run_pushdown(&vp2, 2);

        assert!(
            result.updated_node.is_some(),
            "Inexact: node must be updated so filters reach execute()"
        );
        assert!(
            result.filters.iter().all(|f| matches!(f, PushedDown::No)),
            "Inexact: FilterExec must stay (PushedDown::No); got: {:?}",
            result.filters
        );
        let updated = result.updated_node.unwrap();
        let updated_vp = updated.downcast_ref::<VirtualExecutionPlan>().unwrap();
        assert_eq!(
            updated_vp.filters.len(),
            2,
            "Inexact: both filters must be stored on the node for execute()"
        );
    }
}
