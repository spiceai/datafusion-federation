use datafusion::{
    common::{
        error::Result,
        tree_node::{Transformed, TreeNodeRecursion},
    },
    logical_expr::{Distinct, LogicalPlan},
    optimizer::{push_down_filter::PushDownFilter, ApplyOrder, OptimizerConfig, OptimizerRule},
};

/// A wrapper around DataFusion's [`PushDownFilter`](https://github.com/apache/datafusion/blob/main/datafusion/optimizer/src/push_down_filter.rs) rule.
///
/// This wrapper prevents running the rule in a way that would break the SQL unparser, i.e. pushing a filter beyond a SubqueryAlias.
#[derive(Debug)]
pub struct PushDownFilterFederation {
    inner: PushDownFilter,
    filter_sql_operators: bool,
}

impl Default for PushDownFilterFederation {
    fn default() -> Self {
        Self {
            inner: PushDownFilter::new(),
            filter_sql_operators: true,
        }
    }
}

impl PushDownFilterFederation {
    pub(crate) fn for_analyzer() -> Self {
        Self {
            inner: PushDownFilter::new(),
            filter_sql_operators: false,
        }
    }

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

    #[expect(deprecated)]
    fn supports_rewrite(&self) -> bool {
        self.inner.supports_rewrite()
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if !should_run_rule_for_node(&plan, self.filter_sql_operators)? {
            return Ok(Transformed::no(plan));
        }

        self.inner.rewrite(plan, config)
    }
}

fn should_run_rule_for_node(node: &LogicalPlan, filter_sql_operators: bool) -> Result<bool> {
    if let LogicalPlan::Filter(x) = node {
        // The inner rule can merge filters and recurse past this wrapper.
        // Keep every DISTINCT ON boundary intact before delegating to it.
        let mut distinct_on = false;
        x.input.apply_with_subqueries(|input| {
            distinct_on |= matches!(input, LogicalPlan::Distinct(Distinct::On(_)));
            Ok(if distinct_on {
                TreeNodeRecursion::Stop
            } else {
                TreeNodeRecursion::Continue
            })
        })?;
        if distinct_on {
            return Ok(false);
        }
        if !filter_sql_operators {
            return Ok(true);
        }
        // Applying the `push_down_filter_rule` to certain nodes like `SubqueryAlias`, `Aggregate`, and `CrossJoin`
        // can cause issues during unparsing, thus the optimization is only applied to nodes that are currently supported.
        Ok(matches!(
            x.input.as_ref(),
            LogicalPlan::Join(_)
                | LogicalPlan::TableScan(_)
                | LogicalPlan::Projection(_)
                | LogicalPlan::Filter(_)
                | LogicalPlan::Distinct(Distinct::All(_))
                | LogicalPlan::Sort(_)
        ))
    } else {
        Ok(true)
    }
}
