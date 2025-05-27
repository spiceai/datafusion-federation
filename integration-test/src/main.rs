mod bench;
mod validation;

use anyhow::{anyhow, Result};
use bench::{Benchmark, Query};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::sql::TableReference;
use datafusion_federation::{FederatedQueryPlanner, FederationAnalyzerRule};
use datafusion_table_providers::{
    duckdb::DuckDBTableFactory, sql::db_connection_pool::duckdbpool::DuckDbConnectionPool,
};
use duckdb::{AccessMode, Connection};

use datafusion::{
    execution::{context::SessionContext, session_state::SessionStateBuilder},
    optimizer::{
        analyzer::{
            expand_wildcard_rule::ExpandWildcardRule, inline_table_scan::InlineTableScan,
            resolve_grouping_function::ResolveGroupingFunction, type_coercion::TypeCoercion,
        },
        AnalyzerRule,
    },
};

pub fn get_analyzer_rules() -> Vec<Arc<dyn AnalyzerRule + Send + Sync>> {
    vec![
        Arc::new(InlineTableScan::new()),
        Arc::new(ExpandWildcardRule::new()),
        Arc::new(FederationAnalyzerRule::new()),
        Arc::new(ResolveGroupingFunction::new()),
        Arc::new(TypeCoercion::new()),
    ]
}

fn generate_tpch_database(db_path: &Path) -> Result<()> {
    println!("Generating TPC-H database at: {}", db_path.display());

    let db_path_str = db_path
        .to_str()
        .ok_or_else(|| anyhow!("Failed to convert db_path to string"))?;

    let conn = Connection::open(db_path_str)
        .map_err(|e| anyhow!("Failed to open DuckDB connection: {}", e))?;

    // Install TPC-H extension
    conn.execute("INSTALL tpch;", [])
        .map_err(|e| anyhow!("Failed to install tpch extension: {}", e))?;

    // Load TPC-H extension
    conn.execute("LOAD tpch;", [])
        .map_err(|e| anyhow!("Failed to load tpch extension: {}", e))?;

    // Generate TPC-H data with scale factor 1
    conn.execute("CALL dbgen(sf=1);", [])
        .map_err(|e| anyhow!("Failed to generate tpch data: {}", e))?;

    println!("TPC-H database generated successfully");
    Ok(())
}

fn get_duckdb_table_factory(db_name: String) -> Result<DuckDBTableFactory> {
    let db_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or_else(|| anyhow!("Failed to get parent directory"))?
        .join(db_name);

    // Check if database exists, if not generate it
    if !db_path.exists() {
        generate_tpch_database(&db_path)?;
    }

    let duckdb_pool = Arc::new(
        DuckDbConnectionPool::new_file(
            db_path
                .to_str()
                .ok_or_else(|| anyhow!("Failed to convert db_path to string"))?,
            &AccessMode::ReadOnly,
        )
        .map_err(|e| anyhow!("Failed to create DuckDB connection pool: {}", e))?,
    );

    Ok(DuckDBTableFactory::new(duckdb_pool))
}

#[tokio::main]
async fn main() {
    let benchmark: Arc<dyn Benchmark> = Arc::new(bench::TpchBenchmark);

    let duckdb_table_factory = get_duckdb_table_factory(benchmark.db_file_name())
        .expect("unable to create DuckDB connection pool");

    let state = SessionStateBuilder::new()
        .with_query_planner(Arc::new(FederatedQueryPlanner::new()))
        .with_analyzer_rules(get_analyzer_rules())
        .with_default_features()
        .build();
    let ctx = SessionContext::new_with_state(state);

    let _ = register_federated_duckdb_tables(&ctx, benchmark.table_names(), &duckdb_table_factory)
        .await;

    let test_queries = benchmark.queries();

    for query in test_queries {
        run_test_query(&ctx, Arc::clone(&benchmark), &query)
            .await
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to run {} query {}: {}",
                    benchmark.name(),
                    query.name,
                    err
                )
            });
    }
}

async fn run_test_query(
    ctx: &SessionContext,
    benchmark: Arc<dyn Benchmark>,
    query: &Query,
) -> Result<()> {
    let df = ctx.sql(&query.sql).await?;

    let plan = df.clone().explain(false, false)?.collect().await?;
    let plan_display = pretty_format_batches(&plan)?.to_string();

    insta::with_settings!({
        description => format!("Federated Query Explain"),
        snapshot_path => "../snapshots/explain",
        filters => vec![
            (r"compute_context=.*/([^/]+\.db)", "compute_context=$1")
        ],
    }, {
        insta::assert_snapshot!(format!("{}_{}_explain", benchmark.name(), query.name), plan_display.clone());
    });

    let result = df.collect().await.inspect_err(|_| {
        println!("{plan_display}");
    })?;
    benchmark.validate(query, &result)?;

    Ok(())
}

async fn register_federated_duckdb_tables(
    ctx: &SessionContext,
    table_names: Vec<String>,
    duckdb_table_factory: &DuckDBTableFactory,
) -> Result<()> {
    for table_name in table_names {
        ctx.register_table(
            &table_name,
            duckdb_table_factory
                .table_provider(TableReference::bare(table_name.as_str()))
                .await
                .map_err(|e| anyhow!("Failed to register duckdb table: {}", e))?,
        )?;
    }

    Ok(())
}
