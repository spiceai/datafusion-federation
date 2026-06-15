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

/// Indicates how a physical filter expression is handled when pushed down to a
/// [`SQLExecutor`] via [`SQLExecutor::supports_filters_pushdown`].
///
/// Mirrors the semantics of
/// [`TableProviderFilterPushDown`](datafusion::logical_expr::TableProviderFilterPushDown).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SQLFilterPushDown {
    /// The executor cannot apply this filter. The parent `FilterExec` is kept in the
    /// plan and evaluates the filter locally.
    Unsupported,
    /// The executor will apply this filter exactly inside [`SQLExecutor::execute`].
    /// The parent `FilterExec` is removed from the plan.
    Exact,
    /// The executor will apply this filter as a hint (e.g. for early pruning in the
    /// generated SQL), but may not apply it precisely. The parent `FilterExec` is kept
    /// in the plan to guarantee correctness.
    Inexact,
}

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

    /// Returns how each of the provided physical filter expressions is handled when pushed
    /// down to this executor. Called once with all candidate filters; returns one
    /// [`SQLFilterPushDown`] per filter.
    ///
    /// - [`Unsupported`](SQLFilterPushDown::Unsupported): filter is not passed to
    ///   [`Self::execute`]; the parent `FilterExec` stays in the plan for local evaluation.
    /// - [`Exact`](SQLFilterPushDown::Exact): filter is passed to [`Self::execute`] and
    ///   applied precisely; the parent `FilterExec` is removed.
    /// - [`Inexact`](SQLFilterPushDown::Inexact): filter is passed to [`Self::execute`] as
    ///   a hint (e.g. for early pruning in the generated SQL) but may not be applied
    ///   exactly; the parent `FilterExec` is kept for correctness.
    ///
    /// The default returns [`Unsupported`](SQLFilterPushDown::Unsupported) for every filter.
    /// Override to opt in, e.g. for runtime-only expressions like `DynamicFilterPhysicalExpr`
    /// that can be injected into the SQL at execution time.
    fn supports_filters_pushdown(&self, filters: &[&dyn PhysicalExpr]) -> Vec<SQLFilterPushDown> {
        vec![SQLFilterPushDown::Unsupported; filters.len()]
    }

    /// Execute a SQL query.
    ///
    /// `filters` contain physical expressions for which [`Self::supports_filters_pushdown`]
    /// returned [`Exact`](SQLFilterPushDown::Exact) or [`Inexact`](SQLFilterPushDown::Inexact).
    /// Their concrete values may only be available at execution time (e.g.
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
