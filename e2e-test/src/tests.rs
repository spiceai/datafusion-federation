//! `EXPLAIN` / `EXPLAIN ANALYZE` federation, executed against real engines.
//!
//! These assert the behaviour that must hold; [`crate::snapshots`] captures the
//! full rendered plans so unintended changes to them are visible in review.

#[cfg(feature = "duckdb")]
use crate::duckdb_ctx;
use crate::{run, sqlite_ctx, QUERY, TABLE};
use datafusion::{error::Result, execution::context::SessionContext};

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

/// Operators the remote engine reports for [`QUERY`], which the federation layer
/// runs its EXPLAIN to obtain and grafts below the federated node. DuckDB names its
/// scan `SEQ_SCAN`; SQLite's `EXPLAIN QUERY PLAN` says `SCAN`.
fn remote_operators(engine: &str) -> &'static [&'static str] {
    match engine {
        "duckdb" => &["SEQ_SCAN", "ORDER_BY"],
        "sqlite" => &["SCAN"],
        other => panic!("no expected remote operators for {other}"),
    }
}

/// The physical side: the federation node, the SQL that crosses the boundary, and
/// the remote engine's own operators grafted underneath. `node` differs by format —
/// the indent format prints the `DisplayAs` string, the tree format labels boxes by
/// `name()`.
fn assert_reconstructs_physical_tree(output: &str, engine: &str, format: &str, node: &str) {
    for needle in ["SchemaCastScanExec", node, TABLE] {
        assert_contains(output, engine, format, needle);
    }
    for operator in remote_operators(engine) {
        assert_contains(output, engine, format, operator);
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

/// A federated `EXPLAIN` runs the remote engine's own EXPLAIN and grafts the
/// operators it reports below the federated node, so the plan spans both sides of
/// the boundary. The SQL sent for the query itself is never rewritten.
async fn assert_explain(ctx: &SessionContext, engine: &str) -> Result<()> {
    let output = run(ctx, &format!("EXPLAIN {QUERY}")).await?;

    assert_reconstructs_logical_tree(&output, engine, "indent");
    assert_reconstructs_physical_tree(&output, engine, "indent", "VirtualExecutionPlan");
    assert!(
        !output.contains("EXPLAIN SELECT"),
        "[{engine}/indent] the query SQL must not carry the directive:\n{output}"
    );

    Ok(())
}

/// `EXPLAIN ANALYZE` executes the federated child so `AnalyzeExec` can measure it,
/// and separately asks the remote for its measured plan. The query SQL stays
/// un-prefixed either way — a remote `EXPLAIN ANALYZE` returns plan text, whose
/// schema is not what `SchemaCastScanExec` expects.
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
    assert!(
        !output.contains("CooperativeExec") || output.contains("sql_federation_exec"),
        "[{engine}/tree] federation node missing:\n{output}"
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
