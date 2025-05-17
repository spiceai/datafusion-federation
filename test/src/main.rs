mod component;

use component::{overwrite_default_schema, MockExecutor};
use datafusion::{
    execution::{
        context::SessionContext, options::CsvReadOptions, session_state::SessionStateBuilder,
    },
    optimizer::{
        analyzer::{
            expand_wildcard_rule::ExpandWildcardRule, inline_table_scan::InlineTableScan,
            resolve_grouping_function::ResolveGroupingFunction, type_coercion::TypeCoercion,
        },
        AnalyzerRule,
    },
};

use datafusion_federation::{FederatedQueryPlanner, FederationAnalyzerRule};
use datafusion_federation_sql::{MultiSchemaProvider, SQLFederationProvider, SQLSchemaProvider};
use std::sync::Arc;

pub struct Query {
    pub name: Arc<str>,
    pub sql: Arc<str>,
}

macro_rules! generate_tpch_queries {
    ( $( $i:tt ),* ) => {
        vec![
            $(
                Query {
                    name: concat!("tpch_", stringify!($i)).into(),
                    sql: include_str!(concat!("../queries/tpch/", stringify!($i), ".sql")).into(),
                }
            ),*
        ]
    }
}

pub fn get_tpch_test_queries() -> Vec<Query> {
    generate_tpch_queries!(
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q16, q17, q18, q19, q20, q21,
        q22
    )
}

#[must_use]
pub fn get_analyzer_rules() -> Vec<Arc<dyn AnalyzerRule + Send + Sync>> {
    vec![
        Arc::new(InlineTableScan::new()),
        Arc::new(ExpandWildcardRule::new()),
        Arc::new(FederationAnalyzerRule::new()),
        Arc::new(ResolveGroupingFunction::new()),
        Arc::new(TypeCoercion::new()),
    ]
}

const TPCH_TABLES: [(&str, &str); 8] = [
    ("customer", "./data/customer.csv"),
    ("lineitem", "./data/lineitem.csv"),
    ("nation", "./data/nation.csv"),
    ("orders", "./data/orders.csv"),
    ("part", "./data/part.csv"),
    ("partsupp", "./data/partsupp.csv"),
    ("region", "./data/region.csv"),
    ("supplier", "./data/supplier.csv"),
];

async fn get_federation_provider(tables: Vec<(String, String)>) -> SQLSchemaProvider {
    let remote_ctx = Arc::new(SessionContext::new());
    for (table_name, csv_path) in tables {
        remote_ctx
            .register_csv(table_name, &csv_path, CsvReadOptions::new())
            .await
            .expect("Register csv file");
    }

    let known_tables: Vec<String> = TPCH_TABLES.iter().map(|&x| x.0.into()).collect();
    let executor = Arc::new(MockExecutor::new(remote_ctx, "engine1".to_string()));
    let federation_provider = Arc::new(SQLFederationProvider::new(executor));
    SQLSchemaProvider::new_with_tables(federation_provider, known_tables)
        .await
        .expect("Create new schema provider with tables")
}

#[tokio::main]
async fn main() {
    let tpch_tables: Vec<(String, String)> = TPCH_TABLES
        .iter()
        .map(|&(name, path)| (name.into(), path.into()))
        .collect();

    let federation_provider = Arc::new(get_federation_provider(tpch_tables).await);

    let state = SessionStateBuilder::new()
        .with_query_planner(Arc::new(FederatedQueryPlanner::new()))
        .with_analyzer_rules(get_analyzer_rules())
        .with_default_features()
        .build();

    let schema_provider = MultiSchemaProvider::new(vec![federation_provider]);
    overwrite_default_schema(&state, Arc::new(schema_provider))
        .expect("Overwrite the default schema form the main context");
    let ctx = SessionContext::new_with_state(state);

    let test_queries = get_tpch_test_queries();

    for query in test_queries {
        let sql = query.sql.to_string();
        let name = query.name.to_string();

        let plan = ctx
            .state()
            .create_logical_plan(&sql)
            .await
            .expect("Create a logical plan");

        let optimized_plan = ctx
            .state()
            .optimize(&plan)
            .expect("Optimize the logical plan");

        let display_string = format!("{}", optimized_plan.display_indent());

        insta::with_settings!({
            description => format!("Federated Query Explain"),
            snapshot_path => "../snapshots"
        }, {
            insta::assert_snapshot!(format!("tpch_{name}_explain"), display_string);
        });
    }
}
