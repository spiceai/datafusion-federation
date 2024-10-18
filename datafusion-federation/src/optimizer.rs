use std::sync::Arc;

use datafusion::{
    common::Column,
    datasource::source_as_provider,
    error::Result,
    logical_expr::{Expr, LogicalPlan, Projection, TableScan, TableSource},
    optimizer::{OptimizerConfig, OptimizerRule},
    sql::TableReference,
};

use crate::{FederatedTableProviderAdaptor, FederatedTableSource, FederationProviderRef};

#[derive(Default)]
pub struct FederationOptimizerRule {}

impl OptimizerRule for FederationOptimizerRule {
    fn try_optimize(
        &self,
        plan: &LogicalPlan,
        config: &dyn datafusion::optimizer::OptimizerConfig,
    ) -> Result<Option<LogicalPlan>> {
        println!("BEFORE\n{}\n<<< BEFORE", plan.display_graphviz());
        
        let (optimized, _) = self.optimize_plan_recursively(plan, None, config, 0)?;
        if let Some(p) = optimized.as_ref() {
            println!("AFTER\n{}\n<<< AFTER", p.display_graphviz());
            println!("AFTER2\n{:?}", p);
        }
        Ok(optimized)
    }

    fn supports_rewrite(&self) -> bool {
        false
    }
    
    fn name(&self) -> &str {
        "federation_optimizer_rule"
    }
}

impl FederationOptimizerRule {
    pub fn new() -> Self {
        Self::default()
    }

    fn optimize_plan_recursively(
        &self,
        plan: &LogicalPlan,
        parent: Option<&LogicalPlan>,
        _config: &dyn OptimizerConfig,
        depth: usize
    ) -> Result<(Option<LogicalPlan>, Option<FederationProviderRef>)> {
        if let Some(parent) = parent {
            println!("{}: optimize_plan_recursively plan:`\n{}\n` parent:`\n{}\n`", depth, plan, parent);
        } else {
            println!("{}: optimize_plan_recursively plan:`\n{}\n` parent:`\n{:?}\n`", depth, plan, parent);
        }
        // Check if this node determines the FederationProvider
        let sole_provider = self.get_federation_provider(plan)?;
        println!("{}: sole_provider `{:?}`", depth, sole_provider.as_ref().map(|p| p.name()));
        if sole_provider.is_some() {
            return Ok((None, sole_provider));
        }

        // optimize_inputs
        let inputs = plan.inputs();
        for (i, input) in inputs.iter().enumerate() {
            match input {
                LogicalPlan::TableScan(TableScan { ref source, .. }) => {
                    println!("{}: input {} `TableScan` schema {:?}", depth, i, source.schema());
                }
                _ =>{}
            };
            println!("{}: input {} {:?}", depth, i, input);
        }
        if inputs.is_empty() {
            return Ok((None, None));
        }

        println!("{}: Generating new inputs", depth);
        let (new_inputs, providers): (Vec<_>, Vec<_>) = inputs
            .iter()
            .map(|i| self.optimize_plan_recursively(i, Some(plan), _config, depth+1))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();

        for (i, input) in new_inputs.iter().enumerate() {
            if let Some(input) = input {
                println!("{}: new_input {} `{}`", depth, i, input);
            } else {
                println!("{}: new_input {} `{:?}`", depth, i, input);
            }
        }

        for (i, provider) in providers.iter().enumerate() {
            println!("{}: provider {} {:?}", depth, i, provider.as_ref().map(|p| p.name()));
        }

        // Note: assumes provider is None if ambiguous
        let first_provider = providers.first().unwrap();
        let is_singular = providers.iter().all(|p| p.is_some() && p == first_provider);
        println!("{}: is_singular {}", depth, is_singular);
        if is_singular {
            if parent.is_none() {
                // federate the entire plan
                println!("{}: federate the entire plan", depth);
                if let Some(provider) = first_provider {
                    if let Some(optimizer) = provider.optimizer() {
                        let optimized =
                            optimizer.optimize(plan.clone(), _config, |_, _| {})?;
                        return Ok((Some(optimized), None));
                    }
                    return Ok((None, None));
                }
                return Ok((None, None));
            }
            // The largest sub-plan is higher up.
            println!("{}: The largest sub-plan is higher up", depth);
            return Ok((None, first_provider.clone()));
        }

        println!("{}: Generating new inputs", depth);
        // The plan is ambiguous, any inputs that are not federated and
        // have a sole provider, should be federated.
        let new_inputs = new_inputs
            .into_iter()
            .enumerate()
            .map(|(i, new_sub_plan)| {
                
                if let Some(sub_plan) = new_sub_plan {
                    println!("{}: new_sub_plan is federated {} `{:?}`", depth, i, sub_plan);
                    // Already federated
                    return Ok(sub_plan);
                } else {
                    println!("{}: new_sub_plan is not federated {} `{:?}`", depth, i, new_sub_plan);
                }
                let sub_plan = inputs.get(i).unwrap();
                println!("{}: new_sub_plan2 {} `{}`", depth, i, sub_plan);
                // Check if the input has a sole provider and can be federated.
                if let Some(provider) = providers.get(i).unwrap() {
                    println!("{}: provider2 {} {:?}", depth, i, provider.name());
                    if let Some(optimizer) = provider.optimizer() {
                        let wrapped = wrap_projection((*sub_plan).clone())?;
                        //let wrapped = (*sub_plan).clone();
                        println!("{}: wrapped2 {} `{}`", depth, i, wrapped);

                        println!("{} Optimizing with:", depth);
                        for rule in &optimizer.rules {
                            println!("{}: rule {:?}", depth, rule.name());
                        }
                        let optimized = optimizer.optimize(wrapped, _config, |_, _| {})?;
                        println!("{}: optimized2 {} `{}`", depth, i, optimized);
                        return Ok(optimized);
                    }
                    // No federation for this sub-plan (no analyzer)
                    return Ok((*sub_plan).clone());
                }
                // No federation for this sub-plan (no provider)
                Ok((*sub_plan).clone())
            })
            .collect::<Result<Vec<_>>>()?;

        for (i, input) in new_inputs.iter().enumerate() {
            println!("{}: new_input2 {} {}", depth, i, input);
        }

        let new_plan = plan.with_new_exprs(plan.expressions(), new_inputs)?;

        println!("{}: new_plan {}", depth, new_plan);

        Ok((Some(new_plan), None))
    }

    fn get_federation_provider(&self, plan: &LogicalPlan) -> Result<Option<FederationProviderRef>> {
        match plan {
            LogicalPlan::TableScan(TableScan { ref source, .. }) => {
                let Some(federated_source) = get_table_source(source)? else {
                    return Ok(None);
                };
                println!("!!OPT!! source plan: {:?}", plan);
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
            println!("!!OPT!!: expr: {:?}", expr);
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
