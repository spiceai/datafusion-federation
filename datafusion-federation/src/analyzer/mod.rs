mod scan_result;

use crate::{FederatedTableProviderAdaptor, FederatedTableSource, FederationProviderRef};
use crate::{FederationAnalyzerForLogicalPlan, FederationProvider};
use datafusion::logical_expr::{col, expr::Exists, expr::InSubquery, LogicalPlanBuilder};
use datafusion::optimizer::optimize_unions::OptimizeUnions;
use datafusion::optimizer::push_down_filter::PushDownFilter;
use datafusion::optimizer::{Optimizer, OptimizerContext, OptimizerRule};
use datafusion::{
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    config::ConfigOptions,
    datasource::source_as_provider,
    error::Result,
    logical_expr::{Expr, Extension, LogicalPlan, Projection, TableScan, TableSource},
    optimizer::analyzer::AnalyzerRule,
    sql::TableReference,
};
use scan_result::ScanResult;
use std::collections::HashMap;
use std::sync::Arc;

/// An analyzer rule to identifying sub-plans to federate
///
/// The analyzer logic walks over the plan, look for the largest subtrees that only have
/// TableScans from the same [`FederationProvider`]. The 'largest sub-trees' are passed to their
/// respective [`FederationProvider::analyzer`].
#[derive(Debug)]
pub struct FederationAnalyzerRule {
    // Optimization rules to run before the federated plan is created
    optimizer: Optimizer,
}

impl AnalyzerRule for FederationAnalyzerRule {
    // Walk over the plan, look for the largest subtrees that only have
    // TableScans from the same FederationProvider.
    // There 'largest sub-trees' are passed to their respective FederationProvider.optimizer.
    fn analyze(&self, plan: LogicalPlan, config: &ConfigOptions) -> Result<LogicalPlan> {
        // DML plans must not be federated: the SQL unparser's dml_to_sql is
        // unimplemented, and wrapping DML in a FederatedPlanNode hides the Dml
        // node from write-permission validators (security bypass). Leave DML
        // plans unwrapped so validators see them and DataFusion's physical
        // planner can dispatch delete_from/update to the table provider.
        if matches!(plan, LogicalPlan::Dml(_)) {
            return Ok(plan);
        }

        if !contains_federated_table(&plan)? {
            return Ok(plan);
        }

        // Run optimizer rules before federation
        let plan = self
            .optimizer
            .optimize(plan, &OptimizerContext::new(), |_, _| {})?;

        // Find all federation providers for TableReferences that appear in the plan, to resolve OuterRefColumns
        let providers = get_plan_provider_recursively(&plan)?;
        let explain_context = Self::explain_context_template(&plan);

        match self.analyze_plan_recursively(&plan, true, config, &providers, explain_context)? {
            (Some(optimized_plan), _) => Ok(optimized_plan),
            (None, _) => Ok(plan),
        }
    }

    /// A human readable name for this optimizer rule
    fn name(&self) -> &str {
        "federation_optimizer_rule"
    }
}

impl Default for FederationAnalyzerRule {
    fn default() -> Self {
        Self {
            optimizer: Optimizer::with_rules(Self::default_optimizer_rules()),
        }
    }
}

impl FederationAnalyzerRule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn default_optimizer_rules() -> Vec<Arc<dyn OptimizerRule + Send + Sync>> {
        vec![
            Arc::new(OptimizeUnions::new()),
            Arc::new(PushDownFilter::new()),
        ]
    }

    /// Override the default optimizer with custom rules
    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }

    fn optimize_for_provider(
        provider: &FederationProviderRef,
        plan: LogicalPlan,
        config: &ConfigOptions,
    ) -> Result<LogicalPlan> {
        let rules = provider.pre_federation_optimizer_rules();
        if rules.is_empty() {
            return Ok(plan);
        }

        Optimizer::with_rules(rules).optimize(
            plan,
            &OptimizerContext::new_with_config_options(Arc::new(config.clone())),
            |_, _| {},
        )
    }

    /// The `EXPLAIN`/`EXPLAIN ANALYZE` wrapper to re-apply around federated sub-plans.
    ///
    /// Held behind an [`Arc`] because it is threaded through the whole plan recursion:
    /// cloning the wrapper itself for every node and expression would deep-copy the
    /// plan repeatedly.
    fn explain_context_template(plan: &LogicalPlan) -> Option<Arc<LogicalPlan>> {
        match plan {
            LogicalPlan::Explain(_) | LogicalPlan::Analyze(_) => Some(Arc::new(plan.clone())),
            _ => None,
        }
    }

    fn wrap_federated_plan(
        plan: LogicalPlan,
        explain_context: Option<&LogicalPlan>,
    ) -> Result<LogicalPlan> {
        if matches!(plan, LogicalPlan::Explain(_) | LogicalPlan::Analyze(_)) {
            return Ok(plan);
        }

        match explain_context {
            Some(wrapper) => wrapper.with_new_exprs(wrapper.expressions(), vec![plan]),
            None => Ok(plan),
        }
    }

    /// Scans a plan to see if it belongs to a single [`FederationProvider`].
    ///
    /// `scope` is the candidate that would become one remote statement, which is
    /// `plan` itself except when scanning a subquery, where it stays the enclosing
    /// candidate. Only [`Self::scan_expr_recursively`] reads it, to decide whether a
    /// correlated reference stays bound inside the statement it would be emitted in.
    fn scan_plan_recursively(
        &self,
        plan: &LogicalPlan,
        providers: &HashMap<TableReference, Arc<dyn FederationProvider>>,
        scope: &LogicalPlan,
    ) -> Result<ScanResult> {
        let mut sole_provider: ScanResult = ScanResult::None;

        plan.apply(&mut |p: &LogicalPlan| -> Result<TreeNodeRecursion> {
            let exprs_provider = self.scan_plan_exprs(p, providers, scope)?;
            sole_provider.merge(exprs_provider);

            if sole_provider.is_ambiguous() {
                return Ok(TreeNodeRecursion::Stop);
            }

            let (sub_provider, _) = get_leaf_provider(p)?;
            sole_provider.merge(sub_provider);

            Ok(sole_provider.check_recursion())
        })?;

        Ok(sole_provider)
    }

    /// Scans a plan's expressions to see if it belongs to a single [`FederationProvider`].
    fn scan_plan_exprs(
        &self,
        plan: &LogicalPlan,
        providers: &HashMap<TableReference, Arc<dyn FederationProvider>>,
        scope: &LogicalPlan,
    ) -> Result<ScanResult> {
        let mut sole_provider: ScanResult = ScanResult::None;

        let exprs = plan.expressions();
        for expr in &exprs {
            let expr_result = self.scan_expr_recursively(expr, providers, scope)?;
            sole_provider.merge(expr_result);

            if sole_provider.is_ambiguous() {
                return Ok(sole_provider);
            }
        }

        Ok(sole_provider)
    }

    /// scans an expression to see if it belongs to a single [`FederationProvider`]
    fn scan_expr_recursively(
        &self,
        expr: &Expr,
        providers: &HashMap<TableReference, Arc<dyn FederationProvider>>,
        scope: &LogicalPlan,
    ) -> Result<ScanResult> {
        let mut sole_provider: ScanResult = ScanResult::None;

        expr.apply(&mut |e: &Expr| -> Result<TreeNodeRecursion> {
            match e {
                Expr::ScalarSubquery(ref subquery) => {
                    let plan_result =
                        self.scan_plan_recursively(&subquery.subquery, providers, scope)?;

                    sole_provider.merge(plan_result);
                    Ok(sole_provider.check_recursion())
                }
                Expr::InSubquery(ref insubquery) => {
                    let plan_result = self.scan_plan_recursively(
                        &insubquery.subquery.subquery,
                        providers,
                        scope,
                    )?;

                    sole_provider.merge(plan_result);
                    Ok(sole_provider.check_recursion())
                }
                Expr::Exists(ref exists) => {
                    let plan_result =
                        self.scan_plan_recursively(&exists.subquery.subquery, providers, scope)?;

                    sole_provider.merge(plan_result);
                    Ok(sole_provider.check_recursion())
                }
                Expr::OuterReferenceColumn(_, ref col) => {
                    if let Some(table) = &col.relation {
                        if let Some(plan_result) = providers.get(table) {
                            sole_provider.merge(ScanResult::Distinct(Arc::clone(plan_result)));
                            return Ok(sole_provider.check_recursion());
                        }
                        // A relation that scans nothing has no provider to report, but
                        // the remote engine renders it inline and its alias is defined
                        // inside the same statement, so the correlation stays bound and
                        // constrains nothing about which engine runs the statement.
                        if scope_names_a_scanless_relation(scope, table)? {
                            return Ok(sole_provider.check_recursion());
                        }
                    }
                    // The reference names a relation this candidate does not define, or
                    // one it cannot be shown to define uniquely. Federating would emit
                    // an identifier the remote engine binds to something else or not at
                    // all, so leave the correlation to DataFusion.
                    sole_provider = ScanResult::Ambiguous;
                    Ok(TreeNodeRecursion::Stop)
                }
                _ => Ok(TreeNodeRecursion::Continue),
            }
        })?;

        Ok(sole_provider)
    }

    /// Recursively finds the largest sub-plans that can be federated
    /// to a single FederationProvider.
    ///
    /// Returns a plan if a sub-tree was federated, otherwise None.
    ///
    /// Returns a ScanResult of all FederationProviders in the subtree.
    fn analyze_plan_recursively(
        &self,
        plan: &LogicalPlan,
        is_root: bool,
        config: &ConfigOptions,
        providers: &HashMap<TableReference, Arc<dyn FederationProvider>>,
        explain_context: Option<Arc<LogicalPlan>>,
    ) -> Result<(Option<LogicalPlan>, ScanResult)> {
        let explain_context = explain_context.or_else(|| Self::explain_context_template(plan));
        let mut sole_provider: ScanResult = ScanResult::None;

        if let LogicalPlan::Extension(Extension { ref node }) = plan {
            if node.name() == "Federated" {
                // Avoid attempting double federation
                return Ok((None, ScanResult::Ambiguous));
            }
        }

        // Check if this plan node is a leaf that determines the FederationProvider
        let (leaf_provider, _) = get_leaf_provider(plan)?;

        // Check if the expressions contain, a potentially different, FederationProvider
        let exprs_result = self.scan_plan_exprs(plan, providers, plan)?;

        // Return early if this is a leaf and there is no ambiguity with the expressions.
        if leaf_provider.is_some() && (exprs_result.is_none() || exprs_result == leaf_provider) {
            return Ok((None, leaf_provider.into()));
        }
        // Aggregate leaf & expression providers
        sole_provider.merge(leaf_provider);
        sole_provider.merge(exprs_result.clone());

        let inputs = plan.inputs();
        // Return early if there are no sources.
        if inputs.is_empty() && sole_provider.is_none() {
            return Ok((None, ScanResult::None));
        }

        // Recursively analyze inputs
        let input_results = inputs
            .iter()
            .map(|i| {
                self.analyze_plan_recursively(i, false, config, providers, explain_context.clone())
            })
            .collect::<Result<Vec<_>>>()?;

        // Aggregate the input providers
        input_results.iter().for_each(|(_, scan_result)| {
            sole_provider.merge(scan_result.clone());
        });

        if sole_provider.is_none() {
            // No providers found
            // TODO: Is/should this be reachable?
            return Ok((None, ScanResult::None));
        }

        // Federate Exprs when Exprs provider is ambiguous or Exprs provider differs from the sole_provider of current plan
        // When Exprs provider is the same as sole_provider and non-ambiguous, the larger sub-plan is higher up
        let optimize_expressions = exprs_result.is_some()
            && (!(sole_provider == exprs_result) || exprs_result.is_ambiguous());

        // If all sources are federated to the same provider
        if let ScanResult::Distinct(provider) = sole_provider {
            let prepared_plan = if matches!(plan, LogicalPlan::Analyze(_) | LogicalPlan::Explain(_))
            {
                plan.clone()
            } else {
                Self::optimize_for_provider(&provider, plan.clone(), config)?
            };

            // Explain and Analyze wrappers stay in the DataFusion plan so their
            // physical operators can still run. The corresponding directive is
            // injected into the federated subquery instead.
            let federated_plan =
                Self::wrap_federated_plan(prepared_plan.clone(), explain_context.as_deref())?;
            let provider_analyzer =
                if matches!(plan, LogicalPlan::Analyze(_) | LogicalPlan::Explain(_)) {
                    None
                } else {
                    // Ask about the query itself, not the EXPLAIN wrapper: an executor
                    // whose can_execute_plan rejects Explain/Analyze would otherwise
                    // refuse to federate a query it can perfectly well run.
                    provider.analyzer(&prepared_plan)
                };
            match (is_root, provider_analyzer) {
                (false, Some(FederationAnalyzerForLogicalPlan::With(_))) => {
                    // The largest sub-plan is higher up.
                    return Ok((None, ScanResult::Distinct(provider)));
                }
                (true, Some(FederationAnalyzerForLogicalPlan::With(analyzer))) => {
                    // If this is the root plan node; federate the entire plan
                    let optimized =
                        analyzer.execute_and_check(federated_plan, config, |_, _| {})?;
                    return Ok((Some(optimized), ScanResult::None));
                }
                (_, None | Some(FederationAnalyzerForLogicalPlan::Unable)) => {
                    // Provider CAN'T federate this specific plan shape
                    // Fall through to try federating children instead
                }
            }
        }

        // The plan is ambiguous; any input that is not yet optimized and has a
        // sole provider represents a largest sub-plan and should be federated.
        //
        // We loop over the input optimization results, federate where needed and
        // return a complete list of new inputs for the optimized plan.
        let new_inputs = input_results
            .into_iter()
            .enumerate()
            .map(|(i, (input_plan, input_result))| {
                if let Some(federated_plan) = input_plan {
                    // Already federated deeper in the plan tree
                    return Ok(federated_plan);
                }

                let original_input = (*inputs.get(i).unwrap()).clone();
                if input_result.is_ambiguous() {
                    // Can happen if the input is already federated, so use
                    // the original input.
                    return Ok(original_input);
                }

                let provider = input_result.unwrap()?;
                let Some(provider) = provider else {
                    // No provider for this input; use the original input.
                    return Ok(original_input);
                };

                let projected_input = wrap_projection(original_input.clone())?;
                let prepared_input =
                    Self::optimize_for_provider(&provider, projected_input, config)?;

                // Ask about the query itself rather than the EXPLAIN wrapper applied
                // below, so an executor that rejects Explain/Analyze in
                // can_execute_plan still federates the query it wraps.
                let Some(FederationAnalyzerForLogicalPlan::With(analyzer)) =
                    provider.analyzer(&prepared_input)
                else {
                    // Either provider has no analyzer, or cannot federate [`LogicalPlan`].
                    return Ok(original_input);
                };

                let federated_input =
                    Self::wrap_federated_plan(prepared_input, explain_context.as_deref())?;

                // Replace the input with the federated counterpart
                analyzer.execute_and_check(federated_input, config, |_, _| {})
            })
            .collect::<Result<Vec<_>>>()?;

        // Optimize expressions if needed
        let new_expressions = if optimize_expressions {
            self.analyze_plan_exprs(plan, config, providers, explain_context)?
        } else {
            plan.expressions()
        };

        // Construct the optimized plan
        let new_plan = plan.with_new_exprs(new_expressions, new_inputs)?;

        // Return the federated plan
        Ok((Some(new_plan), ScanResult::Ambiguous))
    }

    /// Analyzes all exprs of a plan
    fn analyze_plan_exprs(
        &self,
        plan: &LogicalPlan,
        config: &ConfigOptions,
        providers: &HashMap<TableReference, Arc<dyn FederationProvider>>,
        explain_context: Option<Arc<LogicalPlan>>,
    ) -> Result<Vec<Expr>> {
        plan.expressions()
            .iter()
            .map(|expr| {
                let transformed = expr.clone().transform(&|e| {
                    self.analyze_expr_recursively(e, config, providers, explain_context.clone())
                })?;
                Ok(transformed.data)
            })
            .collect::<Result<Vec<_>>>()
    }

    /// recursively analyze expressions
    /// Current logic: individually federate every sub-query.
    fn analyze_expr_recursively(
        &self,
        expr: Expr,
        _config: &ConfigOptions,
        providers: &HashMap<TableReference, Arc<dyn FederationProvider>>,
        explain_context: Option<Arc<LogicalPlan>>,
    ) -> Result<Transformed<Expr>> {
        match expr {
            Expr::ScalarSubquery(ref subquery) => {
                // Analyze as root to force federating the sub-query
                let (new_subquery, _) = self.analyze_plan_recursively(
                    &subquery.subquery,
                    true,
                    _config,
                    providers,
                    explain_context.clone(),
                )?;
                let Some(new_subquery) = new_subquery else {
                    return Ok(Transformed::no(expr));
                };

                // ScalarSubqueryToJoin optimizer rule doesn't support federated node (LogicalPlan::Extension(_)) as subquery
                // Wrap a `non-op` Projection LogicalPlan outside the federated node to facilitate ScalarSubqueryToJoin optimization
                if matches!(new_subquery, LogicalPlan::Extension(_)) {
                    let all_columns = new_subquery
                        .schema()
                        .fields()
                        .iter()
                        .map(|field| col(field.name()))
                        .collect::<Vec<_>>();

                    let projection_plan = LogicalPlanBuilder::from(new_subquery)
                        .project(all_columns)?
                        .build()?;

                    return Ok(Transformed::yes(Expr::ScalarSubquery(
                        subquery.with_plan(projection_plan.into()),
                    )));
                }

                Ok(Transformed::yes(Expr::ScalarSubquery(
                    subquery.with_plan(new_subquery.into()),
                )))
            }
            Expr::InSubquery(ref in_subquery) => {
                let (new_subquery, _) = self.analyze_plan_recursively(
                    &in_subquery.subquery.subquery,
                    true,
                    _config,
                    providers,
                    explain_context,
                )?;
                let Some(new_subquery) = new_subquery else {
                    return Ok(Transformed::no(expr));
                };

                // DecorrelatePredicateSubquery optimizer rule doesn't support federated node (LogicalPlan::Extension(_)) as subquery
                // Wrap a `non-op` Projection LogicalPlan outside the federated node to facilitate DecorrelatePredicateSubquery optimization
                if matches!(new_subquery, LogicalPlan::Extension(_)) {
                    let all_columns = new_subquery
                        .schema()
                        .fields()
                        .iter()
                        .map(|field| col(field.name()))
                        .collect::<Vec<_>>();

                    let projection_plan = LogicalPlanBuilder::from(new_subquery)
                        .project(all_columns)?
                        .build()?;

                    return Ok(Transformed::yes(Expr::InSubquery(InSubquery::new(
                        in_subquery.expr.clone(),
                        in_subquery.subquery.with_plan(projection_plan.into()),
                        in_subquery.negated,
                    ))));
                }

                Ok(Transformed::yes(Expr::InSubquery(InSubquery::new(
                    in_subquery.expr.clone(),
                    in_subquery.subquery.with_plan(new_subquery.into()),
                    in_subquery.negated,
                ))))
            }
            Expr::Exists(ref exists) => {
                let (new_subquery, _) = self.analyze_plan_recursively(
                    &exists.subquery.subquery,
                    true,
                    _config,
                    providers,
                    explain_context,
                )?;
                let Some(new_subquery) = new_subquery else {
                    return Ok(Transformed::no(expr));
                };

                // DecorrelatePredicateSubquery optimizer rule doesn't support federated node
                // (LogicalPlan::Extension(_)) as subquery.
                // Wrap a no-op Projection outside the federated node to facilitate optimization.
                if matches!(new_subquery, LogicalPlan::Extension(_)) {
                    let all_columns = new_subquery
                        .schema()
                        .fields()
                        .iter()
                        .map(|field| col(field.name()))
                        .collect::<Vec<_>>();

                    let projection_plan = LogicalPlanBuilder::from(new_subquery)
                        .project(all_columns)?
                        .build()?;

                    return Ok(Transformed::yes(Expr::Exists(Exists {
                        subquery: exists.subquery.with_plan(projection_plan.into()),
                        negated: exists.negated,
                    })));
                }

                Ok(Transformed::yes(Expr::Exists(Exists {
                    subquery: exists.subquery.with_plan(new_subquery.into()),
                    negated: exists.negated,
                })))
            }
            _ => Ok(Transformed::no(expr)),
        }
    }
}

/// NopFederationProvider is used to represent tables that are not federated, but
/// are resolved by DataFusion. This simplifies the logic of the optimizer rule.
#[derive(Debug)]
pub(crate) struct NopFederationProvider {}

impl FederationProvider for NopFederationProvider {
    fn name(&self) -> &str {
        "nop"
    }

    fn compute_context(&self) -> Option<String> {
        None
    }

    fn analyzer(&self, _plan: &LogicalPlan) -> Option<FederationAnalyzerForLogicalPlan> {
        None
    }
}

/// Recursively find the [`FederationProvider`] for all [`TableReference`] instances in the plan.
/// This is used to resolve the federation providers for [`Expr::OuterReferenceColumn`].
fn get_plan_provider_recursively(
    plan: &LogicalPlan,
) -> Result<HashMap<TableReference, Arc<dyn FederationProvider>>> {
    let mut providers: HashMap<TableReference, Arc<dyn FederationProvider>> = HashMap::new();

    plan.apply_with_subqueries(&mut |p: &LogicalPlan| -> Result<TreeNodeRecursion> {
        // Register SubqueryAlias names (e.g. `lineitem l1`) so that OuterReferenceColumn resolved
        // against the alias (e.g. `l1.l_orderkey`) can find the correct provider. Without this,
        // correlated subqueries that reference an aliased outer table mark the scan as Ambiguous,
        // breaking same-provider federation.
        if let LogicalPlan::SubqueryAlias(subquery_alias) = p {
            let alias_ref = TableReference::bare(subquery_alias.alias.table().to_string());
            subquery_alias
                .input
                .apply(&mut |child| -> Result<TreeNodeRecursion> {
                    if let (Some(provider), Some(table_reference)) = get_leaf_provider(child)? {
                        providers.insert(alias_ref.clone(), Arc::clone(&provider));
                        providers.insert(table_reference, provider);
                        return Ok(TreeNodeRecursion::Stop);
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
            return Ok(TreeNodeRecursion::Continue);
        }

        if let (Some(federation_provider), Some(table_reference)) = get_leaf_provider(p)? {
            providers.insert(table_reference, federation_provider);
        }

        Ok(TreeNodeRecursion::Continue)
    })?;

    Ok(providers)
}

fn wrap_projection(plan: LogicalPlan) -> Result<LogicalPlan> {
    // TODO: minimize requested columns
    match plan {
        LogicalPlan::Projection(_) => Ok(plan),
        _ => {
            let expr = plan
                .schema()
                .columns()
                .iter()
                .map(|c| Expr::Column(c.clone()))
                .collect::<Vec<Expr>>();
            Ok(LogicalPlan::Projection(Projection::try_new(
                expr,
                Arc::new(plan),
            )?))
        }
    }
}

/// Whether `scope` defines exactly one relation named `relation`, and that relation
/// scans nothing.
///
/// This is the question a correlated reference the provider map cannot resolve
/// actually poses: not "which engine owns this relation" but "if this whole
/// candidate becomes one remote statement, does the identifier still bind, and to
/// this same relation". A relation built only from constants — a `VALUES` list, or
/// the `UNION ALL` of literal selects a hand-written `hours`-style CTE plans to —
/// is emitted inline by the unparser and carries its own alias into that
/// statement, so the answer is yes and the reference says nothing about which
/// engine should run it.
///
/// "Scans nothing" is the whole test because every `TableScan` reports a provider:
/// a federated one reports its own, and a non-federated one reports
/// [`NopFederationProvider`]. So a relation that scans anything is already in the
/// provider map, or belongs to an engine this candidate cannot be handed to, and
/// either way it is not this branch's business.
///
/// A unique match is required, and a duplicate name is refused rather than
/// resolved. Names are compared on the last segment, so a qualified scan and a
/// bare reference to it collide deliberately: over-matching costs a refusal, where
/// binding a correlation to the wrong relation of the same name would return wrong
/// rows with no error.
///
/// The walk is `apply`, not `apply_with_subqueries`: the relation a correlated
/// reference binds to is in the candidate's own `FROM`, reached through its inputs.
/// A relation defined inside a *nested* subquery is not visible at the reference's
/// position, so counting one would be a false match on nothing more than a reused
/// alias — and the plan would then be federated alone, with that identifier unbound
/// in the emitted SQL.
fn scope_names_a_scanless_relation(scope: &LogicalPlan, relation: &TableReference) -> Result<bool> {
    let wanted = relation.table();
    let mut matches = 0usize;
    let mut scanless = false;

    scope.apply(&mut |node: &LogicalPlan| -> Result<TreeNodeRecursion> {
        let named = match node {
            LogicalPlan::SubqueryAlias(alias) => alias.alias.table() == wanted,
            LogicalPlan::TableScan(scan) => scan.table_name.table() == wanted,
            _ => false,
        };
        if named {
            matches += 1;
            scanless = matches == 1 && !plan_scans_anything(node)?;
        }
        // Stop at each query block. A `SubqueryAlias` is itself a relation of this
        // scope and the relations inside it are not, and a `Union`'s branches are
        // separate `SELECT`s whose relations are not visible to each other or to
        // anything above — a node above a union sees its output columns, not its
        // inputs. Either way the block's own name has just been counted, so
        // descending would only find relations nothing outside it can refer to.
        Ok(
            if matches!(node, LogicalPlan::SubqueryAlias(_) | LogicalPlan::Union(_)) {
                TreeNodeRecursion::Jump
            } else {
                TreeNodeRecursion::Continue
            },
        )
    })?;

    Ok(matches == 1 && scanless)
}

/// Whether any `TableScan` appears in `plan`, including inside its subquery
/// expressions — a scan reached only through a correlated subquery is still a scan
/// the relation depends on.
fn plan_scans_anything(plan: &LogicalPlan) -> Result<bool> {
    let mut found = false;
    plan.apply_with_subqueries(&mut |node: &LogicalPlan| -> Result<TreeNodeRecursion> {
        if matches!(node, LogicalPlan::TableScan(_)) {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

/// Whether any federated table appears in `plan`, **including inside its subquery
/// expressions**.
///
/// A plan whose only federated tables are reached through a scalar, `IN` or
/// `EXISTS` subquery is the case this has to descend for: the outer query's own
/// `FROM` may be a constant relation, so a walk that stops at the plan's inputs
/// reports no federated table and [`FederationAnalyzerRule::analyze`] returns
/// before doing anything. Nothing is then federated at any level, and a statement
/// whose only tables belong to one engine reaches it as one scan per table
/// reference instead of one query.
fn contains_federated_table(plan: &LogicalPlan) -> Result<bool> {
    let mut found = false;
    plan.apply_with_subqueries(&mut |node: &LogicalPlan| -> Result<TreeNodeRecursion> {
        let LogicalPlan::TableScan(TableScan { source, .. }) = node else {
            return Ok(TreeNodeRecursion::Continue);
        };
        if get_table_source(source)?.is_some() {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

fn get_leaf_provider(
    plan: &LogicalPlan,
) -> Result<(Option<FederationProviderRef>, Option<TableReference>)> {
    match plan {
        LogicalPlan::TableScan(TableScan {
            ref table_name,
            ref source,
            ..
        }) => {
            let table_reference = table_name.clone();
            let Some(federated_source) = get_table_source(source)? else {
                if is_cte_work_table(source)? {
                    // Not a table at all — see [`is_cte_work_table`].
                    return Ok((None, None));
                }
                // Table is not federated but provided by a standard table provider.
                // We use a placeholder federation provider to simplify the logic.
                return Ok((
                    Some(Arc::new(NopFederationProvider {})),
                    Some(table_reference),
                ));
            };
            let provider = federated_source.federation_provider();
            Ok((Some(provider), Some(table_reference)))
        }
        _ => Ok((None, None)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::config::ConfigOptions;
    use datafusion::logical_expr::{lit, DmlStatement, EmptyRelation, WriteOp};
    use datafusion::optimizer::analyzer::AnalyzerRule;
    use datafusion::sql::TableReference;
    use std::sync::Arc;

    // Minimal TableSource needed to construct a DmlStatement.
    #[derive(Debug)]
    struct MockTableSource {
        schema: SchemaRef,
    }

    impl MockTableSource {
        fn new() -> Self {
            Self {
                schema: Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)])),
            }
        }
    }

    impl datafusion::logical_expr::TableSource for MockTableSource {
        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }
    }

    #[test]
    fn dml_plan_is_returned_unchanged() {
        let rule = FederationAnalyzerRule::new();
        let config = ConfigOptions::default();

        let empty = Arc::new(LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: false,
            schema: Arc::new(datafusion::common::DFSchema::empty()),
        }));
        let source = Arc::new(MockTableSource::new());
        let dml = LogicalPlan::Dml(DmlStatement::new(
            TableReference::bare("t"),
            source,
            WriteOp::Delete,
            empty,
        ));

        let result = rule.analyze(dml.clone(), &config).unwrap();

        // The plan must come back as-is — Dml, not wrapped in a FederatedPlanNode.
        assert!(
            matches!(result, LogicalPlan::Dml(_)),
            "expected Dml plan, got {result:?}"
        );
    }

    /// `SELECT 0 AS hr UNION ALL SELECT 1 AS hr`, aliased — the shape a hand-written
    /// hour-bucket CTE plans to, and a relation that scans nothing.
    fn scanless_relation(alias: &str) -> LogicalPlan {
        let one_row = || {
            LogicalPlanBuilder::new(LogicalPlan::EmptyRelation(EmptyRelation {
                produce_one_row: true,
                schema: Arc::new(datafusion::common::DFSchema::empty()),
            }))
        };
        one_row()
            .project(vec![lit(0i64).alias("hr")])
            .expect("first branch")
            .union(
                one_row()
                    .project(vec![lit(1i64).alias("hr")])
                    .expect("second branch")
                    .build()
                    .expect("build second branch"),
            )
            .expect("union")
            .alias(alias)
            .expect("alias")
            .build()
            .expect("build")
    }

    fn scan(table: &str) -> LogicalPlan {
        LogicalPlanBuilder::scan(table, Arc::new(MockTableSource::new()), None)
            .expect("scan")
            .build()
            .expect("build scan")
    }

    /// A correlated reference whose relation scans nothing must not be read as a
    /// second engine.
    ///
    /// The provider map is keyed off the relations that scan something, so a constant
    /// relation is absent from it by construction. Reading that absence as
    /// "ambiguous" refuses the whole candidate: the verdict propagates through every
    /// enclosing node, nothing is federated at any level, and a statement whose only
    /// tables are one engine's degrades to one scan per table reference.
    ///
    /// The relation is emitted inline and carries its own alias into the statement,
    /// so the correlation stays bound and says nothing about which engine runs it.
    #[test]
    fn a_correlated_reference_to_a_scanless_relation_is_not_a_second_engine() {
        assert!(
            scope_names_a_scanless_relation(&scanless_relation("h"), &TableReference::bare("h"))
                .expect("resolve h"),
            "a constant relation the candidate defines has to keep the correlation bound"
        );
    }

    /// A relation the candidate does not define at all stays ambiguous: federating
    /// would emit an identifier the remote engine cannot bind.
    #[test]
    fn a_correlated_reference_to_an_undefined_relation_stays_ambiguous() {
        assert!(
            !scope_names_a_scanless_relation(&scanless_relation("h"), &TableReference::bare("g"))
                .expect("resolve g"),
            "a name the candidate never defines must not be treated as bound"
        );
    }

    /// A relation that scans something is not this branch's business — it either has
    /// a provider already or belongs to another engine.
    #[test]
    fn a_correlated_reference_to_a_scanning_relation_stays_ambiguous() {
        let scope = LogicalPlanBuilder::from(scan("orders"))
            .alias("h")
            .expect("alias")
            .build()
            .expect("build");
        assert!(
            !scope_names_a_scanless_relation(&scope, &TableReference::bare("h"))
                .expect("resolve h"),
            "a relation that scans a table must not be rendered inline into another \
             engine's statement"
        );
    }

    /// A relation of that name reached only through a *nested* query block is not the
    /// binding: it is not visible at the reference's position, so counting it would
    /// federate the plan alone and leave that identifier unbound in the emitted SQL.
    /// Two ways to nest one, an expression subquery and a derived table.
    #[test]
    fn a_relation_named_only_inside_a_nested_subquery_is_not_the_binding() {
        let scope = LogicalPlanBuilder::from(scan("orders"))
            .filter(datafusion::logical_expr::in_subquery(
                col("orders.id"),
                Arc::new(scanless_relation("h")),
            ))
            .expect("filter")
            .build()
            .expect("build");
        assert!(
            !scope_names_a_scanless_relation(&scope, &TableReference::bare("h"))
                .expect("resolve h"),
            "an alias reused inside a nested subquery is not what the correlation binds \
             to, and treating it as one emits an unbound identifier"
        );

        // And in a sibling `UNION` branch, which is its own `SELECT`: a reference in
        // one branch binds to an enclosing scope, never to the other branch.
        let sibling = LogicalPlanBuilder::from(scanless_relation("h"))
            .union(scanless_relation("g"))
            .expect("union")
            .build()
            .expect("build");
        assert!(
            !scope_names_a_scanless_relation(&sibling, &TableReference::bare("h"))
                .expect("resolve h"),
            "a relation aliased in one union branch is not visible to the other"
        );

        // The same, nested in a derived table rather than an expression subquery.
        let derived = LogicalPlanBuilder::from(scanless_relation("h"))
            .alias("d")
            .expect("derived alias")
            .build()
            .expect("build");
        assert!(
            !scope_names_a_scanless_relation(&derived, &TableReference::bare("h"))
                .expect("resolve h"),
            "an alias inside a derived table is not visible outside it"
        );
    }

    /// Two relations of one name are refused rather than resolved. Binding the
    /// correlation to the wrong one of them would return wrong rows with no error,
    /// where refusing costs only the pushdown.
    #[test]
    fn a_duplicated_relation_name_is_refused_rather_than_resolved() {
        let scope = LogicalPlanBuilder::from(scanless_relation("h"))
            .cross_join(
                LogicalPlanBuilder::from(scan("orders"))
                    .alias("h")
                    .expect("alias")
                    .build()
                    .expect("build"),
            )
            .expect("cross join")
            .build()
            .expect("build");
        assert!(
            !scope_names_a_scanless_relation(&scope, &TableReference::bare("h"))
                .expect("resolve h"),
            "an ambiguous name must be refused, not resolved to the constant relation"
        );
    }
}

/// Whether this scan is a recursive CTE reading *itself*.
///
/// The recursive term of a `RecursiveQuery` refers to the CTE by name, and that
/// reference is planned as a `TableScan` over a `CteWorkTable`. Treated like any
/// other unfederated table it would take the placeholder provider above, and
/// that placeholder is not neutral: `ScanResult::merge` leaves `None` alone but
/// turns two *different* `Distinct` providers into `Ambiguous`. So a query
/// joining a recursive CTE to a federated table merges
/// `Distinct(remote).merge(Distinct(Nop))` into `Ambiguous`, the boundary falls
/// below the join, and only the bare scan federates — even when the CTE reads no
/// table at all and the remote could run the whole statement.
///
/// A work table is not a local table. The name resolves *inside* the enclosing
/// `RecursiveQuery`, in whichever engine evaluates it, so it constrains the
/// choice of engine no more than a `VALUES` list does — and a `VALUES` list is
/// already neutral here, because it is not a `TableScan`. Answering `None` says
/// exactly that, and leaves the enclosing `RecursiveQuery` free to federate on
/// the strength of the tables it really reads.
///
/// Federating one still requires the dialect to render `WITH RECURSIVE`; a
/// dialect that cannot declines, as it does for any other plan it cannot unparse.
fn is_cte_work_table(source: &Arc<dyn TableSource>) -> Result<bool> {
    Ok(source_as_provider(source)?
        .downcast_ref::<datafusion::datasource::cte_worktable::CteWorkTable>()
        .is_some())
}

#[allow(clippy::missing_errors_doc)]
pub fn get_table_source(
    source: &Arc<dyn TableSource>,
) -> Result<Option<Arc<dyn FederatedTableSource>>> {
    // Unwrap TableSource
    let source = source_as_provider(source)?;

    // Get FederatedTableProviderAdaptor
    let Some(wrapper) = source.downcast_ref::<FederatedTableProviderAdaptor>() else {
        return Ok(None);
    };

    // Return original FederatedTableSource
    Ok(Some(Arc::clone(&wrapper.source)))
}
