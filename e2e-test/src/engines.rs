//! [`SQLExecutor`] implementations backed by real, in-process engines.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::{
    arrow::{
        array::{ArrayRef, Float64Builder, Int64Builder, RecordBatch, StringBuilder},
        datatypes::{DataType, Field, Schema, SchemaRef},
    },
    error::{DataFusionError, Result},
    physical_plan::{stream::RecordBatchStreamAdapter, PhysicalExpr, SendableRecordBatchStream},
    sql::unparser::dialect::{Dialect, DuckDBDialect, SqliteDialect},
};
use datafusion_federation::sql::SQLExecutor;

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
pub struct DuckDbExecutor {
    conn: Mutex<duckdb::Connection>,
}

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
