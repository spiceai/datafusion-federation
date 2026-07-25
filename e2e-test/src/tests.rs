//! `EXPLAIN` / `EXPLAIN ANALYZE` federation, executed against real engines.

use std::sync::Arc;

use datafusion::{
    arrow::util::pretty::pretty_format_batches, error::Result, execution::context::SessionContext,
};

#[cfg(feature = "duckdb")]
use crate::engines::DuckDbExecutor;
use crate::{engines::SqliteExecutor, federated_context, TABLE};

/// Filtered and ordered so the federated SQL is more than a bare table scan.
const QUERY: &str = "SELECT id, name FROM measurements WHERE id > 1 ORDER BY id";

#[cfg(feature = "duckdb")]
async fn duckdb_ctx() -> Result<SessionContext> {
    federated_context(Arc::new(DuckDbExecutor::new()?)).await
}

async fn sqlite_ctx() -> Result<SessionContext> {
    federated_context(Arc::new(SqliteExecutor::new()?)).await
}

/// Runs `sql` to completion and renders the result as text.
async fn run(ctx: &SessionContext, sql: &str) -> Result<String> {
    let batches = ctx.sql(sql).await?.collect().await?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

/// Every operator of the remote sub-plan, innermost first. A federated `EXPLAIN`
/// must show all of these, not just an opaque `Federated` leaf.
const REMOTE_LOGICAL_PLAN: &[&str] = &[
    "TableScan: measurements",
    "Filter: measurements.id > Int64(1)",
    "Projection: measurements.id, measurements.name",
    "Sort: measurements.id ASC NULLS LAST",
];

fn assert_contains(output: &str, engine: &str, format: &str, needle: &str) {
    assert!(
        output.contains(needle),
        "[{engine}/{format}] expected {needle:?} in the plan:\n{output}"
    );
}

/// Asserts the rendered plan reconstructs the whole federated tree: the federation
/// node plus every operator of the remote sub-plan it wraps.
fn assert_reconstructs_logical_tree(output: &str, engine: &str, format: &str) {
    assert_contains(output, engine, format, "Federated");
    for operator in REMOTE_LOGICAL_PLAN {
        assert_contains(output, engine, format, operator);
    }
}

/// The physical side: the federation node, and the SQL that carries the remote
/// sub-plan across the boundary. `node` differs by format — the indent format
/// prints the `DisplayAs` string, the tree format labels boxes by `name()`.
fn assert_reconstructs_physical_tree(output: &str, engine: &str, format: &str, node: &str) {
    for needle in ["SchemaCastScanExec", node, TABLE] {
        assert_contains(output, engine, format, needle);
    }
}

/// Baseline: the query federates and returns the right rows, so a failure below
/// can't be blamed on the fixture.
async fn assert_query_federates(ctx: &SessionContext, engine: &str) -> Result<()> {
    let output = run(ctx, QUERY).await?;
    for expected in ["beta", "gamma"] {
        assert!(
            output.contains(expected),
            "[{engine}] expected {expected:?} in:\n{output}"
        );
    }
    assert!(
        !output.contains("alpha"),
        "[{engine}] the id > 1 filter was not applied:\n{output}"
    );
    Ok(())
}

/// `EXPLAIN` is never executed remotely — `ExplainExec` only prints the plan — so
/// the remote SQL carries the directive for the user to run themselves.
async fn assert_explain(ctx: &SessionContext, engine: &str) -> Result<()> {
    let output = run(ctx, &format!("EXPLAIN {QUERY}")).await?;

    assert_reconstructs_logical_tree(&output, engine, "indent");
    assert_reconstructs_physical_tree(&output, engine, "indent", "VirtualExecutionPlan");
    assert!(
        output.contains("EXPLAIN SELECT"),
        "[{engine}/indent] expected EXPLAIN-prefixed remote SQL:\n{output}"
    );

    Ok(())
}

/// `EXPLAIN ANALYZE` *is* executed: `AnalyzeExec` drains the federated child to
/// measure it. The remote SQL must stay un-prefixed, or the remote returns its own
/// plan text — a different schema than `SchemaCastScanExec` expects.
async fn assert_explain_analyze(ctx: &SessionContext, engine: &str) -> Result<()> {
    let output = run(ctx, &format!("EXPLAIN ANALYZE {QUERY}")).await?;

    assert_reconstructs_physical_tree(&output, engine, "analyze", "VirtualExecutionPlan");
    assert!(
        !output.contains("EXPLAIN ANALYZE SELECT"),
        "[{engine}/analyze] remote SQL must not be prefixed with EXPLAIN ANALYZE:\n{output}"
    );
    assert!(
        output.contains("metrics=") || output.contains("output_rows"),
        "[{engine}/analyze] expected execution metrics:\n{output}"
    );

    Ok(())
}

/// `FORMAT TREE` renders the physical plan into fixed-width boxes, splitting on
/// newlines. `fmt_as` emits one `key=value` per line for this format, so the
/// federation node stays readable instead of word-wrapping into fragments.
async fn assert_explain_format_tree(ctx: &SessionContext, engine: &str) -> Result<()> {
    let output = run(ctx, &format!("EXPLAIN FORMAT TREE {QUERY}")).await?;

    assert_reconstructs_physical_tree(&output, engine, "tree", "sql_federation_exec");
    for label in ["name:", "compute_context:", "sql:"] {
        assert_contains(&output, engine, "tree", label);
    }
    assert!(
        !output.contains("base_sql"),
        "[{engine}/tree] the tree format should show only the SQL that is sent:\n{output}"
    );

    Ok(())
}

/// `FORMAT PGJSON` renders the logical plan as JSON. `FederatedPlanNode` has no
/// logical inputs — that is what keeps DataFusion from re-planning the federated
/// sub-plan — so the remote tree arrives in the node's `Detail` string rather than
/// as nested `Plans` entries.
async fn assert_explain_format_pgjson(ctx: &SessionContext, engine: &str) -> Result<()> {
    let output = run(ctx, &format!("EXPLAIN FORMAT PGJSON {QUERY}")).await?;

    assert_contains(&output, engine, "pgjson", "\"Node Type\": \"Federated\"");
    assert_reconstructs_logical_tree(&output, engine, "pgjson");

    Ok(())
}

/// `FORMAT GRAPHVIZ` renders the logical plan as a DOT graph; the remote tree is
/// the federation node's label.
async fn assert_explain_format_graphviz(ctx: &SessionContext, engine: &str) -> Result<()> {
    let output = run(ctx, &format!("EXPLAIN FORMAT GRAPHVIZ {QUERY}")).await?;

    assert_contains(&output, engine, "graphviz", "digraph");
    assert_reconstructs_logical_tree(&output, engine, "graphviz");

    Ok(())
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_query_federates() -> Result<()> {
    assert_query_federates(&duckdb_ctx().await?, "duckdb").await
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_explain() -> Result<()> {
    assert_explain(&duckdb_ctx().await?, "duckdb").await
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_explain_analyze() -> Result<()> {
    assert_explain_analyze(&duckdb_ctx().await?, "duckdb").await
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_explain_format_tree() -> Result<()> {
    assert_explain_format_tree(&duckdb_ctx().await?, "duckdb").await
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_explain_format_pgjson() -> Result<()> {
    assert_explain_format_pgjson(&duckdb_ctx().await?, "duckdb").await
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_explain_format_graphviz() -> Result<()> {
    assert_explain_format_graphviz(&duckdb_ctx().await?, "duckdb").await
}

#[tokio::test]
async fn sqlite_query_federates() -> Result<()> {
    assert_query_federates(&sqlite_ctx().await?, "sqlite").await
}

#[tokio::test]
async fn sqlite_explain() -> Result<()> {
    assert_explain(&sqlite_ctx().await?, "sqlite").await
}

#[tokio::test]
async fn sqlite_explain_analyze() -> Result<()> {
    assert_explain_analyze(&sqlite_ctx().await?, "sqlite").await
}

#[tokio::test]
async fn sqlite_explain_format_tree() -> Result<()> {
    assert_explain_format_tree(&sqlite_ctx().await?, "sqlite").await
}

#[tokio::test]
async fn sqlite_explain_format_pgjson() -> Result<()> {
    assert_explain_format_pgjson(&sqlite_ctx().await?, "sqlite").await
}

#[tokio::test]
async fn sqlite_explain_format_graphviz() -> Result<()> {
    assert_explain_format_graphviz(&sqlite_ctx().await?, "sqlite").await
}
