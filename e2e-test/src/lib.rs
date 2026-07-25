//! End-to-end federation tests against real remote SQL engines.
//!
//! The unit tests in `datafusion-federation` use a mock executor whose `execute` is
//! `unimplemented!()`, so they can only inspect the SQL that *would* be sent. These
//! tests run it: a DuckDB and a SQLite database are populated in memory and
//! federated through [`SQLExecutor`], which is the only way to cover the paths where
//! the remote engine has to accept the generated SQL and return rows.
//!
//! That distinction matters most for `EXPLAIN` vs `EXPLAIN ANALYZE`: `ExplainExec`
//! only prints the plan, while `AnalyzeExec` drains the federated child to measure
//! it — so only the latter actually executes remotely.
//!
//! The crate is test-only; it exists so the heavy native engine dependencies stay
//! out of the published `datafusion-federation` crate.

#![cfg(test)]

mod engines;
mod snapshots;
mod tests;

use std::sync::Arc;

use datafusion::{
    arrow::{
        array::{Array, StringArray},
        util::pretty::pretty_format_batches,
    },
    catalog::SchemaProvider,
    error::{DataFusionError, Result},
    execution::context::{SessionContext, SessionState},
};
use datafusion_federation::sql::{SQLExecutor, SQLFederationProvider, SQLSchemaProvider};

#[cfg(feature = "duckdb")]
use crate::engines::DuckDbExecutor;
use crate::engines::SqliteExecutor;

/// The table every test federates.
pub const TABLE: &str = "measurements";

/// Rows loaded into each engine. `id` is used to exercise filter pushdown.
pub const FIXTURE_ROWS: &[(i64, &str, f64)] =
    &[(1, "alpha", 1.5), (2, "beta", 2.5), (3, "gamma", 3.5)];

/// Filtered and ordered so the federated SQL is more than a bare table scan.
pub const QUERY: &str = "SELECT id, name FROM measurements WHERE id > 1 ORDER BY id";

/// Points the default catalog's default schema at `schema`, so unqualified table
/// names resolve to the federated provider.
fn overwrite_default_schema(state: &SessionState, schema: Arc<dyn SchemaProvider>) -> Result<()> {
    let options = &state.config().options().catalog;
    let catalog = state
        .catalog_list()
        .catalog(options.default_catalog.as_str())
        .expect("default catalog should exist");

    catalog.register_schema(options.default_schema.as_str(), schema)?;
    Ok(())
}

/// Builds a session whose only table is `TABLE`, federated to `executor`.
async fn federated_context(executor: Arc<dyn SQLExecutor>) -> Result<SessionContext> {
    let provider = Arc::new(SQLFederationProvider::new(executor));
    let schema_provider =
        Arc::new(SQLSchemaProvider::new_with_tables(provider, vec![TABLE.to_string()]).await?);

    let state = datafusion_federation::default_session_state();
    overwrite_default_schema(&state, schema_provider)?;

    Ok(SessionContext::new_with_state(state))
}

#[cfg(feature = "duckdb")]
pub async fn duckdb_ctx() -> Result<SessionContext> {
    federated_context(Arc::new(DuckDbExecutor::new()?)).await
}

pub async fn sqlite_ctx() -> Result<SessionContext> {
    federated_context(Arc::new(SqliteExecutor::new()?)).await
}

/// Runs `sql` to completion and renders the result as a text table.
pub async fn run(ctx: &SessionContext, sql: &str) -> Result<String> {
    let batches = ctx.sql(sql).await?.collect().await?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

/// Runs an `EXPLAIN` and returns just the plan column.
///
/// Deliberately not the `pretty_format_batches` table: that pads every row to the
/// widest cell, so a value of a different length — a remote operator timing, say —
/// shifts the whole table even when the plan is unchanged. The plan text on its own
/// is both stable and far easier to read in a snapshot.
pub async fn plan_text(ctx: &SessionContext, sql: &str) -> Result<String> {
    let batches = ctx.sql(sql).await?.collect().await?;

    let mut lines = Vec::new();
    for batch in &batches {
        let column = batch.column(batch.num_columns() - 1);
        let plans = column
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "expected the plan column of `{sql}` to be Utf8, got {}",
                    column.data_type()
                ))
            })?;
        for index in 0..plans.len() {
            lines.push(plans.value(index).to_string());
        }
    }

    Ok(lines.join("\n"))
}
