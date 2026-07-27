//! The remote engine's own query plan, grafted into the DataFusion plan.
//!
//! When a federated query is explained, the federation layer asks the executor for
//! the remote engine's plan (see [`SQLExecutor::explain_plan`]) and attaches it
//! below the federated node as real [`ExecutionPlan`] nodes. Every `EXPLAIN` format
//! then renders it natively — its own lines under `FORMAT INDENT`, its own boxes
//! under `FORMAT TREE` — instead of the plan stopping at the federation boundary.
//!
//! These nodes exist only to be displayed. `EXPLAIN` never executes its input, and
//! for `EXPLAIN ANALYZE` the rows come from the federated node itself, so
//! [`RemotePlanExec::execute`] is unreachable and returns an error.
//!
//! [`SQLExecutor::explain_plan`]: crate::sql::SQLExecutor::explain_plan

use std::{fmt, sync::Arc};

use datafusion::{
    arrow::datatypes::{Schema, SchemaRef},
    common::tree_node::{Transformed, TreeNode},
    common::Statistics,
    config::ConfigOptions,
    error::{DataFusionError, Result},
    execution::TaskContext,
    physical_expr::EquivalenceProperties,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        execution_plan::SchedulingType,
        execution_plan::{Boundedness, EmissionType},
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
    },
};

use super::VirtualExecutionPlan;

/// One operator of a remote engine's query plan, as the engine reported it.
///
/// The shape is deliberately loose: engines disagree on what an operator is called
/// and what is worth saying about it, so `name` and `details` are passed through
/// verbatim rather than mapped onto DataFusion's operators.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RemotePlanNode {
    /// Operator name as the remote engine calls it, e.g. `SEQ_SCAN` or `SCAN`.
    pub name: String,
    /// Engine-reported details, rendered in the order given.
    pub details: Vec<(String, String)>,
    /// Inputs of this operator, in the engine's own order.
    pub children: Vec<RemotePlanNode>,
}

impl RemotePlanNode {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            details: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Adds a detail line. Values are engine-formatted strings; empty ones are
    /// dropped so an absent metric does not render as a blank field.
    #[must_use]
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let value = value.into();
        if !value.is_empty() {
            self.details.push((key.into(), value));
        }
        self
    }

    #[must_use]
    pub fn with_child(mut self, child: RemotePlanNode) -> Self {
        self.children.push(child);
        self
    }

    #[must_use]
    pub fn with_children(mut self, children: impl IntoIterator<Item = RemotePlanNode>) -> Self {
        self.children.extend(children);
        self
    }

    /// Total operator count, including this node.
    pub fn len(&self) -> usize {
        1 + self.children.iter().map(Self::len).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.name.is_empty() && self.children.is_empty()
    }
}

/// A [`RemotePlanNode`] as a display-only [`ExecutionPlan`].
#[derive(Debug)]
pub struct RemotePlanExec {
    name: String,
    details: Vec<(String, String)>,
    children: Vec<Arc<dyn ExecutionPlan>>,
    props: Arc<PlanProperties>,
}

impl RemotePlanExec {
    /// Converts `node` and everything below it into execution-plan nodes.
    pub fn new(node: RemotePlanNode) -> Arc<Self> {
        let children = node
            .children
            .into_iter()
            .map(|child| Self::new(child) as Arc<dyn ExecutionPlan>)
            .collect();

        // The remote plan carries no data through DataFusion, so an empty schema is
        // the honest description of this node's output.
        let schema = Arc::new(Schema::empty());
        // Declared cooperative because this node is never executed: `EnsureCooperative`
        // would otherwise wrap the deepest remote operator in a yield point that can
        // never be reached, which would show up in the plan as noise.
        let props = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        )
        .with_scheduling_type(SchedulingType::Cooperative);

        Arc::new(Self {
            name: node.name,
            details: node.details,
            children,
            props: Arc::new(props),
        })
    }

    pub fn details(&self) -> &[(String, String)] {
        &self.details
    }
}

impl DisplayAs for RemotePlanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        if matches!(t, DisplayFormatType::TreeRender) {
            // The tree renderer labels the box with `name()` and turns each
            // `key=value` line into a field, so the name is not repeated here.
            for (key, value) in &self.details {
                writeln!(f, "{key}={value}")?;
            }
            return Ok(());
        }

        write!(f, "RemotePlan: {}", self.name)?;
        for (key, value) in &self.details {
            write!(f, " {key}={value}")?;
        }
        Ok(())
    }
}

impl ExecutionPlan for RemotePlanExec {
    fn name(&self) -> &str {
        &self.name
    }

    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::empty())
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.children.iter().collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // The remote plan is a fixed record of what the remote engine reported;
        // rewriting it would misrepresent that engine's plan.
        Ok(self)
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        Err(DataFusionError::Plan(format!(
            "{} describes a plan executed by the remote engine and cannot be executed by DataFusion",
            self.name
        )))
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.props
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        Ok(Arc::new(Statistics::new_unknown(&Schema::empty())))
    }
}

/// Attaches each federated node's remote plan, once DataFusion has finished
/// optimizing.
///
/// Must run **last**. A [`VirtualExecutionPlan`] is a leaf until its remote plan is
/// attached, and several built-in rules treat leaves specially — `EnsureCooperative`
/// wraps them in a yield point, `EnforceDistribution` does not repartition them.
/// Attaching earlier makes the node a non-leaf and changes the plan that runs, so
/// `EXPLAIN` would describe something other than the real query.
///
/// [`crate::default_session_state`] installs this after the default rules. A session
/// built by hand needs to do the same, or federated plans simply stop at the
/// federation node as they did before.
#[derive(Debug, Default)]
pub struct AttachRemotePlans {}

impl AttachRemotePlans {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PhysicalOptimizerRule for AttachRemotePlans {
    fn name(&self) -> &str {
        "attach_remote_plans"
    }

    fn schema_check(&self) -> bool {
        // The remote nodes carry an empty schema and sit below the federated node, so
        // they do not change any operator's output schema.
        true
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            let Some(federated) = node.downcast_ref::<VirtualExecutionPlan>() else {
                return Ok(Transformed::no(node));
            };
            match federated.materialize_remote_plan() {
                Some(attached) => Ok(Transformed::yes(Arc::new(attached))),
                None => Ok(Transformed::no(node)),
            }
        })
        .map(|transformed| transformed.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::physical_plan::displayable;

    fn sample() -> RemotePlanNode {
        RemotePlanNode::new("ORDER_BY")
            .with_detail("order_by", "id ASC")
            .with_child(
                RemotePlanNode::new("SEQ_SCAN")
                    .with_detail("table", "main.measurements")
                    .with_detail("filters", "id>1")
                    // Empty values are dropped rather than rendered blank.
                    .with_detail("dropped", ""),
            )
    }

    #[test]
    fn counts_every_operator() {
        assert_eq!(sample().len(), 2);
        assert!(RemotePlanNode::default().is_empty());
    }

    #[test]
    fn renders_the_remote_tree_under_indent_format() {
        let plan = RemotePlanExec::new(sample());
        let rendered = displayable(plan.as_ref()).indent(false).to_string();

        assert_eq!(
            rendered,
            "RemotePlan: ORDER_BY order_by=id ASC\
             \n  RemotePlan: SEQ_SCAN table=main.measurements filters=id>1\n"
        );
    }

    #[test]
    fn names_the_node_after_the_remote_operator() {
        // FORMAT TREE labels each box with `name()`.
        assert_eq!(RemotePlanExec::new(sample()).name(), "ORDER_BY");
    }

    #[test]
    fn cannot_be_executed() {
        let plan = RemotePlanExec::new(sample());
        // `SendableRecordBatchStream` is not `Debug`, so `expect_err` is unavailable.
        let Err(err) = plan.execute(0, Arc::new(TaskContext::default())) else {
            panic!("remote plan nodes are display-only and must not execute");
        };
        assert!(
            err.to_string().contains("remote engine"),
            "unexpected error: {err}"
        );
    }
}
