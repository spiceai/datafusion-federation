//! Snapshot coverage for every `EXPLAIN` output format, per engine.
//!
//! The assertions in [`crate::tests`] pin the properties that must hold. These
//! capture the whole rendered plan, so any change to how a federated plan is
//! displayed — the `Federated` wrapper, the remote sub-plan under it, the SQL sent,
//! the box layout of `FORMAT TREE` — shows up as a reviewable diff instead of
//! passing unnoticed.
//!
//! `EXPLAIN ANALYZE` reports wall-clock timings and row/byte counters that differ
//! run to run, so those are redacted by [`METRIC_FILTERS`] before comparison.

#[cfg(feature = "duckdb")]
use crate::duckdb_ctx;
use crate::{run, sqlite_ctx, QUERY};
use datafusion::{error::Result, execution::context::SessionContext};

/// Redactions for the parts of `EXPLAIN ANALYZE` output that are timing- and
/// machine-dependent. Without these the snapshot would fail on every run.
const METRIC_FILTERS: &[(&str, &str)] = &[
    // elapsed_compute=77.17µs, elapsed_compute=1.2ms, …
    (
        r"elapsed_compute=[0-9.]+(ns|µs|ms|s)",
        "elapsed_compute=[TIME]",
    ),
    // output_bytes=0.0 B, output_bytes=1.2 KB, …
    (r"output_bytes=[0-9.]+ [A-Za-z]+", "output_bytes=[BYTES]"),
    // Any remaining bare duration metric, e.g. fetch_time=…, repartition_time=…
    (r"_time=[0-9.]+(ns|µs|ms|s)", "_time=[TIME]"),
];

/// Renders `sql` against `ctx` and snapshots it under `name`.
async fn assert_plan_snapshot(ctx: &SessionContext, name: &str, sql: &str) -> Result<()> {
    let plan = run(ctx, sql).await?;

    insta::with_settings!({
        description => sql.to_string(),
        filters => METRIC_FILTERS.to_vec(),
        omit_expression => true,
    }, {
        insta::assert_snapshot!(name, plan);
    });

    Ok(())
}

/// The formats to capture, as (snapshot suffix, `EXPLAIN` variant).
///
/// `indent` and `tree` render the physical plan; `pgjson` and `graphviz` render the
/// logical plan. `analyze` is the only one that executes the federated child.
fn formats() -> Vec<(&'static str, String)> {
    vec![
        ("indent", format!("EXPLAIN {QUERY}")),
        ("analyze", format!("EXPLAIN ANALYZE {QUERY}")),
        ("tree", format!("EXPLAIN FORMAT TREE {QUERY}")),
        ("pgjson", format!("EXPLAIN FORMAT PGJSON {QUERY}")),
        ("graphviz", format!("EXPLAIN FORMAT GRAPHVIZ {QUERY}")),
    ]
}

async fn snapshot_all_formats(ctx: &SessionContext, engine: &str) -> Result<()> {
    for (suffix, sql) in formats() {
        assert_plan_snapshot(ctx, &format!("{engine}_explain_{suffix}"), &sql).await?;
    }
    Ok(())
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_explain_formats() -> Result<()> {
    snapshot_all_formats(&duckdb_ctx().await?, "duckdb").await
}

#[tokio::test]
async fn sqlite_explain_formats() -> Result<()> {
    snapshot_all_formats(&sqlite_ctx().await?, "sqlite").await
}
