use datafusion::{
    common::{error::Result, tree_node::Transformed},
    logical_expr::LogicalPlan,
    optimizer::{push_down_filter::PushDownFilter, ApplyOrder, OptimizerConfig, OptimizerRule},
};

/// A wrapper around DataFusion's [`PushDownFilter`](https://github.com/apache/datafusion/blob/main/datafusion/optimizer/src/push_down_filter.rs) rule.
///
/// This wrapper prevents running the rule in a way that would break the SQL unparser, i.e. pushing a filter beyond a SubqueryAlias.
#[derive(Default, Debug)]
pub struct PushDownFilterFederation {
    inner: PushDownFilter,
}

impl PushDownFilterFederation {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl OptimizerRule for PushDownFilterFederation {
    fn name(&self) -> &str {
        "federation_sql_push_down_filter"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        self.inner.apply_order()
    }

    fn supports_rewrite(&self) -> bool {
        self.inner.supports_rewrite()
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if !should_run_rule_for_node(&plan) {
            return Ok(Transformed::no(plan));
        }

        self.inner.rewrite(plan, config)
    }
}

fn should_run_rule_for_node(node: &LogicalPlan) -> bool {
    if let LogicalPlan::Filter(x) = node {
        // Applying the `push_down_filter_rule` to certain nodes like `SubqueryAlias`, `Aggregate`, and `CrossJoin`
        // can cause issues during unparsing, thus the optimization is only applied to nodes that are currently supported.
        matches!(
            x.input.as_ref(),
            LogicalPlan::Join(_)
                | LogicalPlan::TableScan(_)
                | LogicalPlan::Projection(_)
                | LogicalPlan::Filter(_)
                | LogicalPlan::Distinct(_)
                | LogicalPlan::Sort(_)
        )
    } else {
        true
    }
}
