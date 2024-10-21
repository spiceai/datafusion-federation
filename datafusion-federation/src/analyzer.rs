use std::sync::Arc;

use datafusion::{
    common::{tree_node::{Transformed, TreeNode, TreeNodeRewriter}, Column},
    config::{ConfigOptions, OptimizerOptions},
    datasource::source_as_provider,
    error::Result,
    logical_expr::{Expr, LogicalPlan, Projection, TableScan, TableSource},
    optimizer::{analyzer::AnalyzerRule, optimize_projections::OptimizeProjections, optimizer::ApplyOrder, push_down_filter::{self, PushDownFilter}, simplify_expressions::SimplifyExpressions, OptimizerConfig, OptimizerContext},
    sql::TableReference,
};

use crate::{optimize::optimize_plan, FederatedTableProviderAdaptor, FederatedTableSource, FederationProviderRef};


#[derive(Default)]
pub struct FederationAnalyzerRule {}

impl AnalyzerRule for FederationAnalyzerRule {
    // Walk over the plan, look for the largest subtrees that only have
    // TableScans from the same FederationProvider.
    // There 'largest sub-trees' are passed to their respective FederationProvider.optimizer.
    fn analyze(&self, plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan> {
    
        if !contains_federated_table(&plan)? {
            return Ok(plan);
        }

        let plan = optimize_plan(plan)?;

        let (optimized, _) = self.optimize_recursively(&plan, None, config)?;
        if let Some(result) = optimized {
            return Ok(result);
        }
        Ok(plan.clone())
    }

    /// A human readable name for this optimizer rule
    fn name(&self) -> &str {
        "federation_optimizer_rule"
    }
}

fn contains_federated_table(plan: &LogicalPlan) -> Result<bool> {
    let federated_table_exists = plan.exists(|x| {
        if let LogicalPlan::TableScan(TableScan { ref source, .. }) = x {
            return Ok(get_table_source(source)?.is_some());
        }
        Ok(false)
    })?;

    Ok(federated_table_exists)
}

impl FederationAnalyzerRule {
    pub fn new() -> Self {
        Self::default()
    }

    // optimize_recursively recursively finds the largest sub-plans that can be federated
    // to a single FederationProvider.
    // Returns a plan if a sub-tree was federated, otherwise None.
    // Returns a FederationProvider if it covers the entire sub-tree, otherwise None.
    fn optimize_recursively(
        &self,
        plan: &LogicalPlan,
        parent: Option<&LogicalPlan>,
        _config: &ConfigOptions,
    ) -> Result<(Option<LogicalPlan>, Option<FederationProviderRef>)> {
        // Check if this node determines the FederationProvider
        let sole_provider = self.get_federation_provider(plan)?;
        if sole_provider.is_some() {
            return Ok((None, sole_provider));
        }

        // optimize_inputs
        let inputs = plan.inputs();
        if inputs.is_empty() {
            return Ok((None, None));
        }

        let (new_inputs, providers): (Vec<_>, Vec<_>) = inputs
            .iter()
            .map(|i| self.optimize_recursively(i, Some(plan), _config))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();

        // Note: assumes provider is None if ambiguous
        let first_provider = providers.first().unwrap();
        let is_singular = providers.iter().all(|p| p.is_some() && p == first_provider);

        if is_singular {
            if parent.is_none() {
                // federate the entire plan
                if let Some(provider) = first_provider {
                    if let Some(optimizer) = provider.analyzer() {
                        let optimized =
                            optimizer.execute_and_check(plan.clone(), _config, |_, _| {})?;
                        return Ok((Some(optimized), None));
                    }
                    return Ok((None, None));
                }
                return Ok((None, None));
            }
            // The largest sub-plan is higher up.
            return Ok((None, first_provider.clone()));
        }

        // The plan is ambiguous, any inputs that are not federated and
        // have a sole provider, should be federated.
        let new_inputs = new_inputs
            .into_iter()
            .enumerate()
            .map(|(i, new_sub_plan)| {
                if let Some(sub_plan) = new_sub_plan {
                    // Already federated
                    return Ok(sub_plan);
                }
                let sub_plan = inputs.get(i).unwrap();
                // Check if the input has a sole provider and can be federated.
                if let Some(provider) = providers.get(i).unwrap() {
                    if let Some(optimizer) = provider.analyzer() {
                        let wrapped = wrap_projection((*sub_plan).clone())?;

                        let optimized = optimizer.execute_and_check(wrapped, _config, |_, _| {})?;
                        return Ok(optimized);
                    }
                    // No federation for this sub-plan (no analyzer)
                    return Ok((*sub_plan).clone());
                }
                // No federation for this sub-plan (no provider)
                Ok((*sub_plan).clone())
            })
            .collect::<Result<Vec<_>>>()?;

        let new_plan = match plan {
            // Unnest returns columns to unnest as `expressions` but does not support passing them back to `with_new_exprs`.
            // Instead, it uses data from its internal representation to create a new plan.
            LogicalPlan::Unnest(_) => plan.with_new_exprs(vec![], new_inputs)?,
            _ => plan.with_new_exprs(plan.expressions(), new_inputs)?,
        };

        Ok((Some(new_plan), None))
    }

    fn get_federation_provider(&self, plan: &LogicalPlan) -> Result<Option<FederationProviderRef>> {
        match plan {
            LogicalPlan::TableScan(TableScan { ref source, .. }) => {
                let Some(federated_source) = get_table_source(source)? else {
                    return Ok(None);
                };
                let provider = federated_source.federation_provider();
                Ok(Some(provider))
            }
            _ => Ok(None),
        }
    }
}

fn wrap_projection(plan: LogicalPlan) -> Result<LogicalPlan> {
    // TODO: minimize requested columns
    match plan {
        //LogicalPlan::Projection(_) | LogicalPlan::TableScan(_)  => Ok(plan),
        LogicalPlan::Projection(_) => Ok(plan),
        _ => {
            let expr = plan
                .schema()
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    Expr::Column(Column::from_qualified_name(format!(
                        "{}.{}",
                        plan.schema()
                            .qualified_field(i)
                            .0
                            .map(TableReference::table)
                            .unwrap_or_default(),
                        f.name()
                    )))
                })
                .collect::<Vec<Expr>>();
            Ok(LogicalPlan::Projection(Projection::try_new(
                expr,
                Arc::new(plan),
            )?))
        }
    }
}

pub fn get_table_source(
    source: &Arc<dyn TableSource>,
) -> Result<Option<Arc<dyn FederatedTableSource>>> {
    // Unwrap TableSource
    let source = source_as_provider(source)?;

    // Get FederatedTableProviderAdaptor
    let Some(wrapper) = source
        .as_any()
        .downcast_ref::<FederatedTableProviderAdaptor>()
    else {
        return Ok(None);
    };

    // Return original FederatedTableSource
    Ok(Some(Arc::clone(&wrapper.source)))
}