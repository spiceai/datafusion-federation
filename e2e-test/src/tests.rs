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

/// Execute an unoptimized fragment so scalar-subquery naming crosses the actual
/// table-rewrite boundary before the outer projection consumes its column.
async fn assert_scalar_subquery_output_name(
    executor: std::sync::Arc<dyn datafusion_federation::sql::SQLExecutor>,
) -> Result<()> {
    use datafusion::{
        arrow::array::Array,
        common::Statistics,
        physical_plan::{collect, displayable},
    };
    use datafusion_federation::sql::VirtualExecutionPlan;
    use std::sync::Arc;

    let ctx = crate::federated_context(Arc::clone(&executor)).await?;
    let table = ctx.table_provider(TABLE).await?;
    crate::overwrite_default_schema(
        &ctx.state(),
        Arc::new(datafusion::catalog::MemorySchemaProvider::new()),
    )?;
    ctx.register_table("app_table", Arc::clone(&table))?;
    ctx.register_table(TABLE, table)?;
    let sql = r#"SELECT q."max(app_table.id)" FROM (SELECT (SELECT max(id) FROM app_table)) q"#;
    let plan = ctx.sql(sql).await?.logical_plan().clone();
    let statistics = Statistics::new_unknown(plan.schema().as_arrow());
    let execution = Arc::new(VirtualExecutionPlan::new(plan, executor, statistics, None));
    println!("{sql} => {}", displayable(execution.as_ref()).one_line());
    let batches = collect(execution, ctx.task_ctx()).await?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].schema().field(0).name(), "max(app_table.id)");
    let values = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .expect("max of integer IDs");
    assert_eq!(values.len(), 1);
    assert!(!values.is_null(0));
    assert_eq!(values.value(0), 3);
    Ok(())
}

#[tokio::test]
async fn sqlite_scalar_subquery_output_name() -> Result<()> {
    assert_scalar_subquery_output_name(std::sync::Arc::new(crate::engines::SqliteExecutor::new()?))
        .await
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_scalar_subquery_output_name() -> Result<()> {
    assert_scalar_subquery_output_name(std::sync::Arc::new(crate::engines::DuckDbExecutor::new()?))
        .await
}

#[cfg(feature = "duckdb")]
async fn assert_distinct_on_rewrite(
    select: &str,
    on_key: &str,
    filter: bool,
    local_fallback: bool,
    expected: &[i64],
) -> Result<()> {
    use datafusion::{
        arrow::array::{Array, Int64Array},
        common::{Column, Statistics},
        logical_expr::{lit, Expr, LogicalPlanBuilder},
        physical_plan::{collect, displayable},
    };
    use datafusion_federation::sql::VirtualExecutionPlan;
    use std::sync::Arc;

    let executor = Arc::new(crate::engines::DuckDbExecutor::new()?);
    executor.duplicate_rows()?;
    let ctx = crate::federated_context(executor.clone()).await?;
    let table = ctx.table_provider(TABLE).await?;
    crate::overwrite_default_schema(
        &ctx.state(),
        Arc::new(datafusion::catalog::MemorySchemaProvider::new()),
    )?;
    ctx.register_table("app_table", table)?;
    let sql =
        format!("SELECT DISTINCT ON ({on_key}) {select} FROM app_table ORDER BY {on_key}, id");
    let plan = ctx.sql(&sql).await?.logical_plan().clone();
    let original_name = plan.schema().field(0).name().clone();
    let column = Expr::Column(Column::from(plan.schema().qualified_field(0)));
    let builder = LogicalPlanBuilder::from(plan);
    // Keep the consumer directly above DistinctOn; optimizing first can hide
    // stale schemas by lowering DISTINCT ON into other operators.
    let plan = if filter {
        builder.filter(column.gt(lit(1_i64)))?.build()?
    } else {
        builder.project(vec![column])?.build()?
    };
    let batches = if local_fallback {
        let column = Expr::Column(Column::from(plan.schema().qualified_field(0)));
        let stacked = LogicalPlanBuilder::from(plan.clone())
            .filter(column.lt(lit(3_i64)))?
            .build()?;
        let mut batches = Vec::new();
        let candidates = if expected.is_empty() {
            vec![plan, stacked]
        } else {
            vec![plan]
        };
        for candidate in candidates {
            let physical = ctx
                .execute_logical_plan(candidate)
                .await?
                .create_physical_plan()
                .await?;
            let rendered = displayable(physical.as_ref()).indent(true).to_string();
            println!("{rendered}");
            assert!(rendered.contains("FilterExec"), "{rendered}");
            assert!(rendered.contains("VirtualExecutionPlan"), "{rendered}");
            batches.extend(collect(physical, ctx.task_ctx()).await?);
        }
        batches
    } else {
        let statistics = Statistics::new_unknown(plan.schema().as_arrow());
        let execution = Arc::new(VirtualExecutionPlan::new(plan, executor, statistics, None));
        println!("{sql} => {}", displayable(execution.as_ref()).one_line());
        collect(execution, ctx.task_ctx()).await?
    };
    let mut actual = Vec::new();
    for batch in batches {
        assert_eq!(batch.schema().field(0).name(), &original_name);
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("integer DISTINCT ON result");
        assert_eq!(values.null_count(), 0);
        actual.extend(values.values().iter().copied());
    }
    assert_eq!(actual, expected);
    Ok(())
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_distinct_on_filter_rewrites_qualifiers() -> Result<()> {
    assert_distinct_on_rewrite("id", "id", true, true, &[2, 3]).await
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_distinct_on_computed_output_keeps_name() -> Result<()> {
    assert_distinct_on_rewrite("id + 1", "id", false, false, &[2, 3, 4]).await
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_distinct_on_scalar_output_keeps_name() -> Result<()> {
    assert_distinct_on_rewrite(
        "(SELECT max(id) FROM app_table)",
        "id",
        false,
        false,
        &[3, 3, 3],
    )
    .await
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_distinct_on_explicit_alias_is_preserved() -> Result<()> {
    assert_distinct_on_rewrite("id AS v", "id", false, false, &[1, 2, 3]).await
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_distinct_on_non_key_filter_stays_after_distinct() -> Result<()> {
    assert_distinct_on_rewrite("id", "id % 1", true, true, &[]).await
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_distinct_on_unsafe_direct_filter_is_refused() {
    let error = assert_distinct_on_rewrite("id", "id % 1", true, false, &[])
        .await
        .expect_err("direct unsafe SQL must be refused");
    assert!(
        matches!(error, datafusion::error::DataFusionError::NotImplemented(_)),
        "{error}"
    );
}

#[cfg(feature = "duckdb")]
#[tokio::test]
async fn duckdb_distinct_on_anonymous_subquery_filter_stays_local() -> Result<()> {
    use datafusion::{
        arrow::array::Int64Array,
        physical_plan::{collect, displayable},
    };
    use std::sync::Arc;
    let executor = Arc::new(crate::engines::DuckDbExecutor::new()?);
    executor.duplicate_rows()?;
    let ctx = crate::federated_context(executor).await?;
    let table = ctx.table_provider(TABLE).await?;
    crate::overwrite_default_schema(
        &ctx.state(),
        Arc::new(datafusion::catalog::MemorySchemaProvider::new()),
    )?;
    ctx.register_table("app_table", table)?;
    let physical = ctx.sql("SELECT id FROM (SELECT DISTINCT ON (id) id FROM app_table ORDER BY id) WHERE id > 1 ORDER BY id")
        .await?.create_physical_plan().await?;
    let rendered = displayable(physical.as_ref()).indent(true).to_string();
    println!("{rendered}");
    assert!(rendered.contains("FilterExec"), "{rendered}");
    assert!(rendered.contains("VirtualExecutionPlan"), "{rendered}");
    let mut actual = Vec::new();
    for batch in collect(physical, ctx.task_ctx()).await? {
        actual.extend(
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("integer result")
                .values()
                .iter()
                .copied(),
        );
    }
    assert_eq!(actual, vec![2, 3]);
    Ok(())
}
