mod executor;
mod rewrite;
mod schema;

use std::{any::Any, collections::HashMap, fmt, sync::Arc, vec};

use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::{Schema, SchemaRef},
    common::{tree_node::Transformed, Column},
    error::Result,
    execution::{context::SessionState, TaskContext},
    logical_expr::{
        expr::{
            AggregateFunction, AggregateFunctionParams, Alias, Exists, InList, InSubquery,
            PlannedReplaceSelectItem, ScalarFunction, Sort, Unnest, WildcardOptions,
            WindowFunction, WindowFunctionParams,
        },
        Between, BinaryExpr, Case, Cast, Expr, Extension, GroupingSet, Like, Limit, LogicalPlan,
        Subquery, TryCast,
    },
    optimizer::{optimizer::Optimizer, OptimizerConfig, OptimizerRule},
    physical_expr::EquivalenceProperties,
    physical_plan::{
        execution_plan::{Boundedness, EmissionType},
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
    },
    sql::{
        sqlparser::ast::Statement,
        unparser::{plan_to_sql, Unparser},
        TableReference,
    },
};

pub use executor::{AstAnalyzer, SQLExecutor, SQLExecutorRef};
pub use schema::{MultiSchemaProvider, SQLSchemaProvider, SQLTableSource};

use crate::{
    get_table_source, schema_cast, FederatedPlanNode, FederationPlanner, FederationProvider,
};

// SQLFederationProvider provides federation to SQL DMBSs.
#[derive(Debug)]
pub struct SQLFederationProvider {
    optimizer: Arc<Optimizer>,
    executor: Arc<dyn SQLExecutor>,
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
    planner: Arc<dyn FederationPlanner>,
}

impl SQLFederationOptimizerRule {
    pub fn new(executor: Arc<dyn SQLExecutor>) -> Self {
        Self {
            planner: Arc::new(SQLFederationPlanner::new(Arc::clone(&executor))),
        }
    }
}

impl AnalyzerRule for SQLFederationOptimizerRule {
    /// Try to rewrite `plan` to an optimized form.
    fn analyze(&self, plan: LogicalPlan, _config: &dyn OptimizerConfig) -> Result<LogicalPlan> {
        if let LogicalPlan::Extension(Extension { ref node }) = plan {
            if node.name() == "Federated" {
                // Avoid attempting double federation
                return Ok(plan);
            }
        }
        // Simply accept the entire plan for now
        let fed_plan = FederatedPlanNode::new(plan.clone(), self.planner.clone());
        let ext_node = Extension {
            node: Arc::new(fed_plan),
        };
        Ok(LogicalPlan::Extension(ext_node))
    }

    /// A human readable name for this analyzer rule
    fn name(&self) -> &str {
        "federate_sql"
    }
}

struct SQLFederationPlanner {
    executor: Arc<dyn SQLExecutor>,
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
        let input = Arc::new(VirtualExecutionPlan::new(
            node.plan().clone(),
            Arc::clone(&self.executor),
        ));
        let schema_cast_exec = schema_cast::SchemaCastScanExec::new(input, schema);
        Ok(Arc::new(schema_cast_exec))
    }
}

#[derive(Debug, Clone)]
struct VirtualExecutionPlan {
    plan: LogicalPlan,
    executor: Arc<dyn SQLExecutor>,
    props: PlanProperties,
}

impl VirtualExecutionPlan {
    pub fn new(plan: LogicalPlan, executor: Arc<dyn SQLExecutor>) -> Self {
        let schema: Schema = plan.schema().as_ref().into();
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
        }
    }

    fn schema(&self) -> SchemaRef {
        let df_schema = self.plan.schema().as_ref();
        Arc::new(Schema::from(df_schema))
    }

    fn sql(&self) -> Result<String> {
        // Find all table scans, recover the SQLTableSource, find the remote table name and replace the name of the TableScan table.
        let mut known_rewrites = HashMap::new();
        let subquery_uses_partial_path = self.executor.subquery_use_partial_path();
        let rewritten_plan = rewrite::plan::rewrite_table_scans(
            &self.plan,
            &mut known_rewrites,
            subquery_uses_partial_path,
            &mut None,
        )?;
        let mut ast = self.plan_to_sql(&rewritten_plan)?;

        // If there are any MultiPartTableReference, rewrite the AST to use the original table names.
        let multi_table_reference_rewrites = known_rewrites
            .into_iter()
            .filter_map(|(table_ref, rewrite)| match rewrite {
                MultiPartTableReference::Multi(rewrite) => Some((table_ref, rewrite)),
                _ => None,
            })
            .collect::<HashMap<TableReference, MultiTableReference>>();
        tracing::trace!("multi_table_reference_rewrites: {multi_table_reference_rewrites:?}");
        if !multi_table_reference_rewrites.is_empty() {
            rewrite::ast::rewrite_multi_part_statement(&mut ast, &multi_table_reference_rewrites);
        }

        if let Some(analyzer) = self.executor.ast_analyzer() {
            ast = analyzer(ast)?;
        }

        Ok(format!("{ast}"))
    }

    fn plan_to_sql(&self, plan: &LogicalPlan) -> Result<Statement> {
        Unparser::new(self.executor.dialect().as_ref()).plan_to_sql(plan)
    }
}

impl DisplayAs for VirtualExecutionPlan {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> std::fmt::Result {
        write!(f, "VirtualExecutionPlan")?;
        let Ok(ast) = plan_to_sql(&self.plan) else {
            return Ok(());
        };
        write!(f, " name={}", self.executor.name())?;
        if let Some(ctx) = self.executor.compute_context() {
            write!(f, " compute_context={ctx}")?;
        };

        write!(f, " sql={ast}")?;
        if let Ok(query) = self.sql() {
            write!(f, " rewritten_sql={query}")?;
        };

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
        self.executor.execute(self.sql()?.as_str(), self.schema())
    }

    fn properties(&self) -> &PlanProperties {
        &self.props
    }
}
