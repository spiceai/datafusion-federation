//! [`SQLExecutor`] implementations backed by real, in-process engines.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
#[cfg(feature = "duckdb")]
use datafusion::sql::unparser::dialect::DuckDBDialect;
use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, Float64Builder, Int64Builder, RecordBatch, StringArray, StringBuilder,
        },
        datatypes::{DataType, Field, Schema, SchemaRef},
    },
    error::{DataFusionError, Result},
    physical_plan::{stream::RecordBatchStreamAdapter, PhysicalExpr, SendableRecordBatchStream},
    sql::unparser::dialect::{Dialect, SqliteDialect},
};
use datafusion_federation::sql::{RemotePlanNode, SQLExecutor};
use datafusion_federation::FederatedQueryType;

use crate::{FIXTURE_ROWS, TABLE};

fn remote_err(engine: &str, e: impl std::fmt::Display) -> DataFusionError {
    DataFusionError::External(format!("{engine}: {e}").into())
}

/// Streams already-materialized batches, declaring `schema` as the stream schema.
fn stream(schema: SchemaRef, batches: Vec<RecordBatch>) -> SendableRecordBatchStream {
    Box::pin(RecordBatchStreamAdapter::new(
        schema,
        futures::stream::iter(batches.into_iter().map(Ok)),
    ))
}

/// Federates to an in-memory DuckDB database.
#[cfg(feature = "duckdb")]
pub struct DuckDbExecutor {
    conn: Mutex<duckdb::Connection>,
}

#[cfg(feature = "duckdb")]
impl DuckDbExecutor {
    /// Creates the database and loads [`FIXTURE_ROWS`].
    pub fn new() -> Result<Self> {
        let conn = duckdb::Connection::open_in_memory().map_err(|e| remote_err("duckdb", e))?;
        conn.execute_batch(&format!(
            "CREATE TABLE {TABLE} (id BIGINT, name VARCHAR, value DOUBLE);"
        ))
        .map_err(|e| remote_err("duckdb", e))?;
        for (id, name, value) in FIXTURE_ROWS {
            conn.execute(
                &format!("INSERT INTO {TABLE} VALUES ({id}, '{name}', {value})"),
                [],
            )
            .map_err(|e| remote_err("duckdb", e))?;
        }
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn query(&self, sql: &str) -> Result<(SchemaRef, Vec<RecordBatch>)> {
        let conn = self.conn.lock().expect("duckdb mutex poisoned");
        let mut stmt = conn.prepare(sql).map_err(|e| remote_err("duckdb", e))?;
        let batches: Vec<RecordBatch> = stmt
            .query_arrow([])
            .map_err(|e| remote_err("duckdb", e))?
            .collect();
        Ok((stmt.schema(), batches))
    }
}

#[cfg(feature = "duckdb")]
#[async_trait]
impl SQLExecutor for DuckDbExecutor {
    fn name(&self) -> &str {
        "duckdb_e2e"
    }

    fn compute_context(&self) -> Option<String> {
        Some("duckdb_e2e".to_string())
    }

    fn dialect(&self) -> Arc<dyn Dialect> {
        Arc::new(DuckDBDialect::default())
    }

    fn execute(
        &self,
        query: &str,
        schema: SchemaRef,
        _filters: &[Arc<dyn PhysicalExpr>],
    ) -> Result<SendableRecordBatchStream> {
        let (_, batches) = self.query(query)?;
        Ok(stream(schema, batches))
    }

    async fn table_names(&self) -> Result<Vec<String>> {
        Ok(vec![TABLE.to_string()])
    }

    async fn get_table_schema(&self, table_name: &str) -> Result<SchemaRef> {
        let (schema, _) = self.query(&format!("SELECT * FROM {table_name} LIMIT 0"))?;
        Ok(schema)
    }

    /// DuckDB renders its default `EXPLAIN` as box-drawing art, which is not worth
    /// parsing. `FORMAT JSON` reports the same plan as a tree of
    /// `{name, extra_info, children}`, and under `ANALYZE` a query profile whose
    /// operators additionally carry measured timings and row counts.
    async fn explain_plan(
        &self,
        query: &str,
        query_type: FederatedQueryType,
    ) -> Result<Option<RemotePlanNode>> {
        let statement = match query_type {
            FederatedQueryType::Explain => format!("EXPLAIN (FORMAT JSON) {query}"),
            FederatedQueryType::Analyze => format!("EXPLAIN (ANALYZE, FORMAT JSON) {query}"),
        };

        let (_, batches) = self.query(&statement)?;
        let Some(json) = last_column_text(&batches) else {
            return Ok(None);
        };
        let value: serde_json::Value =
            serde_json::from_str(&json).map_err(|e| remote_err("duckdb", e))?;

        Ok(match query_type {
            // `EXPLAIN` yields an array holding the root operator.
            FederatedQueryType::Explain => value
                .as_array()
                .and_then(|nodes| nodes.first())
                .map(duckdb_node),
            // `ANALYZE` yields the query profile; its single child is the
            // `EXPLAIN_ANALYZE` operator wrapping the plan we asked about, which is an
            // artifact of the statement rather than part of the query.
            FederatedQueryType::Analyze => value
                .get("children")
                .and_then(|c| c.as_array())
                .and_then(|nodes| nodes.first())
                .and_then(|explain_analyze| {
                    explain_analyze
                        .get("children")
                        .and_then(|c| c.as_array())
                        .and_then(|nodes| nodes.first())
                })
                .map(duckdb_node),
        })
    }
}

/// The text of the last column of `batches`, concatenated. Both DuckDB explain
/// shapes put the payload there.
fn last_column_text(batches: &[RecordBatch]) -> Option<String> {
    let mut out = String::new();
    for batch in batches {
        let column = batch.column(batch.num_columns().checked_sub(1)?);
        let values = column.as_any().downcast_ref::<StringArray>()?;
        for index in 0..values.len() {
            out.push_str(values.value(index));
        }
    }
    (!out.is_empty()).then_some(out)
}

/// Converts one DuckDB plan node, and everything below it, to a [`RemotePlanNode`].
fn duckdb_node(value: &serde_json::Value) -> RemotePlanNode {
    // `EXPLAIN` names the operator `name`; the `ANALYZE` profile uses
    // `operator_name` and adds measured columns beside it.
    let name = value
        .get("operator_name")
        .or_else(|| value.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("UNKNOWN");

    let mut node = RemotePlanNode::new(name)
        .with_detail("rows", json_scalar(value.get("operator_cardinality")))
        .with_detail("timing", json_scalar(value.get("operator_timing")));

    if let Some(extra) = value.get("extra_info").and_then(|e| e.as_object()) {
        for (key, detail) in extra {
            node = node.with_detail(key.to_lowercase(), json_scalar(Some(detail)));
        }
    }

    node.with_children(
        value
            .get("children")
            .and_then(|c| c.as_array())
            .map(|children| children.iter().map(duckdb_node).collect::<Vec<_>>())
            .unwrap_or_default(),
    )
}

/// Renders a JSON value as a single line. Arrays become comma-separated so a
/// multi-column projection stays on one detail line.
fn json_scalar(value: Option<&serde_json::Value>) -> String {
    match value {
        None | Some(serde_json::Value::Null) => String::new(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .map(|item| json_scalar(Some(item)))
            .collect::<Vec<_>>()
            .join(", "),
        Some(other) => other.to_string(),
    }
}

/// Federates to an in-memory SQLite database.
pub struct SqliteExecutor {
    conn: Mutex<rusqlite::Connection>,
}

impl SqliteExecutor {
    /// Creates the database and loads [`FIXTURE_ROWS`].
    pub fn new() -> Result<Self> {
        let conn = rusqlite::Connection::open_in_memory().map_err(|e| remote_err("sqlite", e))?;
        conn.execute_batch(&format!(
            "CREATE TABLE {TABLE} (id INTEGER, name TEXT, value REAL);"
        ))
        .map_err(|e| remote_err("sqlite", e))?;
        for (id, name, value) in FIXTURE_ROWS {
            conn.execute(
                &format!("INSERT INTO {TABLE} VALUES ({id}, '{name}', {value})"),
                [],
            )
            .map_err(|e| remote_err("sqlite", e))?;
        }
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// SQLite is untyped at the row level, so rows are read positionally and
    /// converted according to `schema` — which the unparser derives from the same
    /// projection order it emitted.
    fn query(&self, sql: &str, schema: &SchemaRef) -> Result<Vec<RecordBatch>> {
        let conn = self.conn.lock().expect("sqlite mutex poisoned");
        let mut stmt = conn.prepare(sql).map_err(|e| remote_err("sqlite", e))?;
        let mut rows = stmt.query([]).map_err(|e| remote_err("sqlite", e))?;

        let mut builders: Vec<ColumnBuilder> = schema
            .fields()
            .iter()
            .map(|field| ColumnBuilder::new(field.data_type()))
            .collect::<Result<_>>()?;

        while let Some(row) = rows.next().map_err(|e| remote_err("sqlite", e))? {
            for (index, builder) in builders.iter_mut().enumerate() {
                builder.append(row, index)?;
            }
        }

        let columns = builders
            .into_iter()
            .map(ColumnBuilder::finish)
            .collect::<Vec<_>>();

        let batch = RecordBatch::try_new(Arc::clone(schema), columns)?;
        Ok(vec![batch])
    }
}

#[async_trait]
impl SQLExecutor for SqliteExecutor {
    fn name(&self) -> &str {
        "sqlite_e2e"
    }

    fn compute_context(&self) -> Option<String> {
        Some("sqlite_e2e".to_string())
    }

    fn dialect(&self) -> Arc<dyn Dialect> {
        Arc::new(SqliteDialect {})
    }

    fn execute(
        &self,
        query: &str,
        schema: SchemaRef,
        _filters: &[Arc<dyn PhysicalExpr>],
    ) -> Result<SendableRecordBatchStream> {
        let batches = self.query(query, &schema)?;
        Ok(stream(schema, batches))
    }

    async fn table_names(&self) -> Result<Vec<String>> {
        Ok(vec![TABLE.to_string()])
    }

    async fn get_table_schema(&self, table_name: &str) -> Result<SchemaRef> {
        let conn = self.conn.lock().expect("sqlite mutex poisoned");
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({table_name})"))
            .map_err(|e| remote_err("sqlite", e))?;
        let columns = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?))
            })
            .map_err(|e| remote_err("sqlite", e))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| remote_err("sqlite", e))?;

        let fields = columns
            .into_iter()
            .map(|(name, declared_type)| {
                let data_type = match declared_type.to_uppercase().as_str() {
                    "INTEGER" => DataType::Int64,
                    "REAL" => DataType::Float64,
                    "TEXT" => DataType::Utf8,
                    other => {
                        return Err(DataFusionError::NotImplemented(format!(
                            "sqlite column type {other} is not used by these tests"
                        )))
                    }
                };
                Ok(Field::new(name, data_type, true))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Arc::new(Schema::new(fields)))
    }

    /// SQLite has no `EXPLAIN ANALYZE`; `EXPLAIN` alone returns VDBE opcodes, which
    /// describe the bytecode rather than the plan. `EXPLAIN QUERY PLAN` is the useful
    /// one: `(id, parent, notused, detail)` rows forming a tree by parent pointer, so
    /// both directives report the same estimated plan with no timings.
    async fn explain_plan(
        &self,
        query: &str,
        _query_type: FederatedQueryType,
    ) -> Result<Option<RemotePlanNode>> {
        let conn = self.conn.lock().expect("sqlite mutex poisoned");
        let mut stmt = conn
            .prepare(&format!("EXPLAIN QUERY PLAN {query}"))
            .map_err(|e| remote_err("sqlite", e))?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .map_err(|e| remote_err("sqlite", e))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(|e| remote_err("sqlite", e))?;

        Ok(sqlite_tree(&rows, 0).into_iter().next())
    }
}

/// Builds the children of `parent` from SQLite's `(id, parent, detail)` rows.
///
/// The first word of `detail` is the operator (`SCAN`, `SEARCH`, `USE TEMP B-TREE`),
/// which reads well as a node name; the remainder is what it applies to.
fn sqlite_tree(rows: &[(i64, i64, String)], parent: i64) -> Vec<RemotePlanNode> {
    rows.iter()
        .filter(|(_, row_parent, _)| *row_parent == parent)
        .map(|(id, _, detail)| {
            let (operator, rest) = split_sqlite_detail(detail);
            RemotePlanNode::new(operator)
                .with_detail("detail", rest)
                .with_children(sqlite_tree(rows, *id))
        })
        .collect()
}

/// Splits `SCAN measurements` into `("SCAN", "measurements")`. SQLite's multi-word
/// operators are upper-case, so the split takes every leading upper-case word.
fn split_sqlite_detail(detail: &str) -> (String, String) {
    let mut operator = Vec::new();
    let mut rest = Vec::new();

    for word in detail.split_whitespace() {
        let still_operator = rest.is_empty()
            && word
                .chars()
                .all(|c| c.is_ascii_uppercase() || c == '-' || c.is_ascii_digit());
        if still_operator {
            operator.push(word);
        } else {
            rest.push(word);
        }
    }

    if operator.is_empty() {
        return (detail.to_string(), String::new());
    }
    (operator.join(" "), rest.join(" "))
}

/// The Arrow types these tests need from SQLite's dynamically typed rows.
enum ColumnBuilder {
    Int64(Int64Builder),
    Float64(Float64Builder),
    Utf8(StringBuilder),
}

impl ColumnBuilder {
    fn new(data_type: &DataType) -> Result<Self> {
        match data_type {
            DataType::Int64 => Ok(Self::Int64(Int64Builder::new())),
            DataType::Float64 => Ok(Self::Float64(Float64Builder::new())),
            DataType::Utf8 => Ok(Self::Utf8(StringBuilder::new())),
            other => Err(DataFusionError::NotImplemented(format!(
                "sqlite results of type {other} are not used by these tests"
            ))),
        }
    }

    fn append(&mut self, row: &rusqlite::Row, index: usize) -> Result<()> {
        match self {
            Self::Int64(builder) => {
                builder.append_option(row.get(index).map_err(|e| remote_err("sqlite", e))?)
            }
            Self::Float64(builder) => {
                builder.append_option(row.get(index).map_err(|e| remote_err("sqlite", e))?)
            }
            Self::Utf8(builder) => builder.append_option(
                row.get::<_, Option<String>>(index)
                    .map_err(|e| remote_err("sqlite", e))?,
            ),
        }
        Ok(())
    }

    fn finish(mut self) -> ArrayRef {
        match &mut self {
            Self::Int64(builder) => Arc::new(builder.finish()),
            Self::Float64(builder) => Arc::new(builder.finish()),
            Self::Utf8(builder) => Arc::new(builder.finish()),
        }
    }
}
