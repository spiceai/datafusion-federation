use async_trait::async_trait;
use core::fmt;
use datafusion::{
    arrow::datatypes::SchemaRef,
    common::Statistics,
    error::Result,
    logical_expr::LogicalPlan,
    physical_plan::{metrics::MetricsSet, PhysicalExpr, SendableRecordBatchStream},
    sql::unparser::dialect::Dialect,
};
use std::sync::Arc;

use super::ast_analyzer::AstAnalyzer;

pub type SQLExecutorRef = Arc<dyn SQLExecutor>;

pub type LogicalOptimizer = Box<dyn FnMut(LogicalPlan) -> Result<LogicalPlan>>;
pub type SqlQueryRewriter = Box<dyn FnMut(String) -> Result<String>>;

#[async_trait]
pub trait SQLExecutor: Sync + Send {
    /// Executor name
    fn name(&self) -> &str;

    /// Executor compute context allows differentiating the remote compute context
    /// such as authorization or active database.
    ///
    /// Note: returning None here may cause incorrect federation with other providers of the
    /// same name that also have a compute_context of None.
    /// Instead try to return a unique string that will never match any other
    /// provider's context.
    fn compute_context(&self) -> Option<String>;

    /// The specific SQL dialect (currently supports 'sqlite', 'postgres', 'flight')
    fn dialect(&self) -> Arc<dyn Dialect>;

    /// Returns if this executor can execute the query that would be produced from this logical plan.
    ///
    /// This is used to indicate to the federation logic that part of this plan cannot be federated,
    /// i.e. if there are UDFs that only DataFusion can execute.
    fn can_execute_plan(&self, _logical_plan: &LogicalPlan) -> bool {
        true
    }

    /// Returns the analyzer rule specific for this engine to modify the logical plan before execution
    fn logical_optimizer(&self) -> Option<LogicalOptimizer> {
        None
    }

    /// Returns an AST analyzer specific for this engine to modify the AST before execution
    fn ast_analyzer(&self) -> Option<AstAnalyzer> {
        None
    }

    /// Returns whether this executor will apply `filter` when it is passed to [`Self::execute`].
    ///
    /// [`VirtualExecutionPlan`](super::VirtualExecutionPlan) calls this for each physical filter
    /// expression that a parent [`FilterExec`](datafusion::physical_plan::filter::FilterExec) wants
    /// to push down. Returning `true` allows the `FilterExec` to be removed from the plan
    /// (the executor is then responsible for applying the filter inside [`Self::execute`]).
    /// Returning `false` keeps the `FilterExec` in place for local evaluation.
    ///
    /// The default is `false` — filters are not applied. Override to opt in, e.g. for
    /// runtime-only expressions like `DynamicFilterPhysicalExpr` that can be injected into
    /// the SQL at execution time.
    fn can_handle_filter(&self, _filter: &dyn PhysicalExpr) -> bool {
        false
    }

    /// Execute a SQL query.
    ///
    /// `filters` contain physical expressions for which [`Self::can_handle_filter`] returned
    /// `true`. Their concrete values may only be available at execution time (e.g.
    /// `DynamicFilterPhysicalExpr`), so they must be incorporated into the SQL query when the
    /// stream is polled.
    fn execute(
        &self,
        query: &str,
        schema: SchemaRef,
        filters: &[Arc<dyn PhysicalExpr>],
    ) -> Result<SendableRecordBatchStream>;

    /// Returns statistics for this `SQLExecutor` node. If statistics are not available, it should
    /// return [`Statistics::new_unknown`] (the default), not an error. See the `ExecutionPlan`
    /// trait.
    async fn statistics(&self, plan: &LogicalPlan) -> Result<Statistics> {
        Ok(Statistics::new_unknown(plan.schema().as_arrow()))
    }

    /// Returns the tables provided by the remote
    async fn table_names(&self) -> Result<Vec<String>>;

    /// Returns the schema of table_name within this [`SQLExecutor`]
    async fn get_table_schema(&self, table_name: &str) -> Result<SchemaRef>;

    /// Returns the execution metrics, if available.
    fn metrics(&self) -> Option<MetricsSet> {
        None
    }
}

impl fmt::Debug for dyn SQLExecutor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {:?}", self.name(), self.compute_context())
    }
}

impl fmt::Display for dyn SQLExecutor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {:?}", self.name(), self.compute_context())
    }
}
