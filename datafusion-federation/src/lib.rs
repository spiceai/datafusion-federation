mod analyzer;
mod optimize;
mod plan_node;
pub mod schema_cast;
#[cfg(feature = "sql")]
pub mod sql;
mod table_provider;
pub mod table_reference;

use std::{
    fmt,
    hash::{Hash, Hasher},
    sync::Arc,
};

use datafusion::{
    execution::session_state::{SessionState, SessionStateBuilder},
    optimizer::{
        analyzer::{
            expand_wildcard_rule::ExpandWildcardRule, inline_table_scan::InlineTableScan,
            resolve_grouping_function::ResolveGroupingFunction, type_coercion::TypeCoercion,
        },
        Analyzer, AnalyzerRule,
    },
};

pub use analyzer::{get_table_source, FederationAnalyzerRule};
pub use plan_node::{
    FederatedPlanNode, FederatedPlanner, FederatedQueryPlanner, FederationPlanner,
};
pub use table_provider::{FederatedTableProviderAdaptor, FederatedTableSource};

pub fn default_session_state() -> SessionState {
    let rules = default_analyzer_rules();
    SessionStateBuilder::new()
        .with_analyzer_rules(rules)
        .with_query_planner(Arc::new(FederatedQueryPlanner::new()))
        .with_default_features()
        .build()
}

/// datafusion-federation customizes the order of the analyzer rules, since some of them are only relevant when `DataFusion` is executing the query,
/// as opposed to when underlying federated query engines will execute the query.
///
/// This list should be kept in sync with the default rules in `Analyzer::new()`, but with the federation analyzer rule added.
pub fn default_analyzer_rules() -> Vec<Arc<dyn AnalyzerRule + Send + Sync>> {
    vec![
        Arc::new(ExpandWildcardRule::new()),
        Arc::new(InlineTableScan::new()),
        Arc::new(FederationAnalyzerRule::new()),
        // The rest of these rules are run after the federation analyzer since they only affect internal DataFusion execution.
        Arc::new(ResolveGroupingFunction::new()),
        Arc::new(TypeCoercion::new()),
    ]
}

pub type FederationProviderRef = Arc<dyn FederationProvider>;

pub trait FederationProvider: Send + Sync {
    // Returns the name of the provider, used for comparison.
    fn name(&self) -> &str;

    // Returns the compute context in which this federation provider
    // will execute a query. For example: database instance & catalog.
    fn compute_context(&self) -> Option<String>;

    // Returns an analyzer that can cut out part of the plan
    // to federate it.
    fn analyzer(&self) -> Option<Arc<Analyzer>>;
}

impl fmt::Display for dyn FederationProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {:?}", self.name(), self.compute_context())
    }
}

impl PartialEq<dyn FederationProvider> for dyn FederationProvider {
    /// Comparing name, args and return_type
    fn eq(&self, other: &dyn FederationProvider) -> bool {
        self.name() == other.name() && self.compute_context() == other.compute_context()
    }
}

impl Hash for dyn FederationProvider {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name().hash(state);
        self.compute_context().hash(state);
    }
}

impl Eq for dyn FederationProvider {}

#[cfg(test)]
mod tests {
    use datafusion::optimizer::Analyzer;

    /// Verifies that the default analyzer rules are in the expected order.
    ///
    /// If this test fails, `DataFusion` has modified the default analyzer rules and `get_analyzer_rules()` should be updated.
    #[test]
    fn test_verify_default_analyzer_rules() {
        let default_rules = Analyzer::new().rules;
        assert_eq!(
            default_rules.len(),
            4,
            "Default analyzer rules have changed"
        );
        let expected_rule_names = vec![
            "inline_table_scan",
            "expand_wildcard_rule",
            "resolve_grouping_function",
            "type_coercion",
        ];
        for (rule, expected_name) in default_rules.iter().zip(expected_rule_names.into_iter()) {
            assert_eq!(
                expected_name,
                rule.name(),
                "Default analyzer rule order has changed"
            );
        }
    }
}
