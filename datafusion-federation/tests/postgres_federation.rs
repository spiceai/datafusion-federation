/// Integration test: sort alias in CASE ORDER BY against a real Postgres database.
///
/// # Bug being tested
///
/// DataFusion's SQL federation generates queries of the form:
///
/// ```sql
/// ORDER BY CASE WHEN ("lochierarchy" = 0) THEN "category" END ASC NULLS LAST
/// ```
///
/// where `lochierarchy` is a SELECT-list alias for `grouping(category) + grouping(class)`.
/// PostgreSQL rejects this with SQLSTATE 42703 ("column lochierarchy does not exist") because
/// it does not resolve SELECT-list aliases inside compound ORDER BY expressions.
///
/// # Fix
///
/// `datafusion_federation::sql`'s `inline_sort_projection_aliases` rewrites the ORDER BY
/// expression before unparsing, substituting the alias definition in place of the alias
/// reference. Without that rewrite this test would fail with a Postgres 42703 error.
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::{
    arrow::{
        array::{
            ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
            StringArray,
        },
        datatypes::{DataType, Field, Schema, SchemaRef},
        error::ArrowError,
        record_batch::RecordBatch,
    },
    error::{DataFusionError, Result as DfResult},
    execution::{context::SessionContext, session_state::SessionStateBuilder},
    optimizer::{
        analyzer::{
            resolve_grouping_function::ResolveGroupingFunction, type_coercion::TypeCoercion,
        },
        AnalyzerRule,
    },
    physical_plan::{stream::RecordBatchStreamAdapter, PhysicalExpr, SendableRecordBatchStream},
    sql::unparser::dialect::Dialect,
};
use datafusion_federation::{
    sql::{
        federation_analyzer_rule, RemoteTable, RemoteTableRef, SQLExecutor, SQLFederationProvider,
        SQLTableSource,
    },
    FederatedQueryPlanner, FederatedTableProviderAdaptor,
};
use futures::stream;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;
use tokio::sync::Mutex;
use tokio_postgres::{Client, NoTls, Row};

// ---------------------------------------------------------------------------
// PostgresExecutor
// ---------------------------------------------------------------------------

struct PostgresExecutor {
    client: Arc<Mutex<Client>>,
    connection_string: String,
}

impl std::fmt::Debug for PostgresExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresExecutor")
            .field("connection_string", &self.connection_string)
            .finish()
    }
}

impl PostgresExecutor {
    async fn connect(conn_str: &str) -> anyhow::Result<Self> {
        let (client, connection) = tokio_postgres::connect(conn_str, NoTls).await?;

        // Spawn the connection task; it runs until the client is dropped.
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("postgres connection error: {e}");
            }
        });

        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            connection_string: conn_str.to_string(),
        })
    }

    async fn execute_ddl(&self, sql: &str) -> anyhow::Result<()> {
        self.client.lock().await.execute(sql, &[]).await?;
        Ok(())
    }
}

#[async_trait]
impl SQLExecutor for PostgresExecutor {
    fn name(&self) -> &str {
        "postgres"
    }

    fn compute_context(&self) -> Option<String> {
        Some(self.connection_string.clone())
    }

    fn dialect(&self) -> Arc<dyn Dialect> {
        Arc::new(datafusion::sql::unparser::dialect::PostgreSqlDialect {})
    }

    fn execute(
        &self,
        query: &str,
        schema: SchemaRef,
        _filters: &[Arc<dyn PhysicalExpr>],
    ) -> DfResult<SendableRecordBatchStream> {
        let client = Arc::clone(&self.client);
        let query = query.to_string();
        let schema_clone = Arc::clone(&schema);

        let fut = async move {
            let locked = client.lock().await;
            let rows = locked
                .query(&query, &[])
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            pg_rows_to_batch(&rows, &schema_clone)
        };

        let stream = stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    async fn table_names(&self) -> DfResult<Vec<String>> {
        let locked = self.client.lock().await;
        let rows = locked
            .query(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
                &[],
            )
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok(rows
            .iter()
            .map(|r| r.get::<_, String>(0))
            .collect())
    }

    async fn get_table_schema(&self, table_name: &str) -> DfResult<SchemaRef> {
        let locked = self.client.lock().await;
        let rows = locked
            .query(
                "SELECT column_name, data_type \
                 FROM information_schema.columns \
                 WHERE table_schema = 'public' AND table_name = $1 \
                 ORDER BY ordinal_position",
                &[&table_name],
            )
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let fields: Vec<Field> = rows
            .iter()
            .map(|row| {
                let col_name: String = row.get(0);
                let pg_type: String = row.get(1);
                Field::new(col_name, pg_data_type(&pg_type), true)
            })
            .collect();

        Ok(Arc::new(Schema::new(fields)))
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn pg_data_type(pg_type: &str) -> DataType {
    match pg_type {
        "text" | "character varying" | "character" | "varchar" => DataType::Utf8,
        "integer" | "int" | "int4" => DataType::Int32,
        "bigint" | "int8" => DataType::Int64,
        "double precision" | "float8" => DataType::Float64,
        "real" | "float4" => DataType::Float32,
        "boolean" | "bool" => DataType::Boolean,
        "numeric" | "decimal" => DataType::Float64,
        _ => DataType::Utf8,
    }
}

fn pg_rows_to_batch(rows: &[Row], schema: &SchemaRef) -> DfResult<RecordBatch> {
    let columns: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(col_idx, field)| -> DfResult<ArrayRef> {
            match field.data_type() {
                DataType::Utf8 => {
                    let vals: Vec<Option<String>> = rows
                        .iter()
                        .map(|r| r.try_get::<_, Option<String>>(col_idx).unwrap_or(None))
                        .collect();
                    Ok(Arc::new(StringArray::from(vals)) as ArrayRef)
                }
                DataType::Int32 => {
                    let vals: Vec<Option<i32>> = rows
                        .iter()
                        .map(|r| r.try_get::<_, Option<i32>>(col_idx).unwrap_or(None))
                        .collect();
                    Ok(Arc::new(Int32Array::from(vals)) as ArrayRef)
                }
                DataType::Int64 => {
                    let vals: Vec<Option<i64>> = rows
                        .iter()
                        .map(|r| r.try_get::<_, Option<i64>>(col_idx).unwrap_or(None))
                        .collect();
                    Ok(Arc::new(Int64Array::from(vals)) as ArrayRef)
                }
                DataType::Float64 => {
                    let vals: Vec<Option<f64>> = rows
                        .iter()
                        .map(|r| r.try_get::<_, Option<f64>>(col_idx).unwrap_or(None))
                        .collect();
                    Ok(Arc::new(Float64Array::from(vals)) as ArrayRef)
                }
                DataType::Float32 => {
                    let vals: Vec<Option<f32>> = rows
                        .iter()
                        .map(|r| r.try_get::<_, Option<f32>>(col_idx).unwrap_or(None))
                        .collect();
                    Ok(Arc::new(Float32Array::from(vals)) as ArrayRef)
                }
                DataType::Boolean => {
                    let vals: Vec<Option<bool>> = rows
                        .iter()
                        .map(|r| r.try_get::<_, Option<bool>>(col_idx).unwrap_or(None))
                        .collect();
                    Ok(Arc::new(BooleanArray::from(vals)) as ArrayRef)
                }
                _ => {
                    // Fallback: treat any unknown type as Utf8
                    let vals: Vec<Option<String>> = rows
                        .iter()
                        .map(|r| r.try_get::<_, Option<String>>(col_idx).unwrap_or(None))
                        .collect();
                    Ok(Arc::new(StringArray::from(vals)) as ArrayRef)
                }
            }
        })
        .collect::<DfResult<_>>()?;

    RecordBatch::try_new(Arc::clone(schema), columns)
        .map_err(|e: ArrowError| DataFusionError::from(e))
}

fn make_session_context() -> SessionContext {
    let analyzer_rules: Vec<Arc<dyn AnalyzerRule + Send + Sync>> = vec![
        Arc::new(federation_analyzer_rule()),
        Arc::new(ResolveGroupingFunction::new()),
        Arc::new(TypeCoercion::new()),
    ];

    let state = SessionStateBuilder::new()
        .with_query_planner(Arc::new(FederatedQueryPlanner::new()))
        .with_analyzer_rules(analyzer_rules)
        .with_default_features()
        .build();

    SessionContext::new_with_state(state)
}

fn register_table(
    ctx: &SessionContext,
    name: &str,
    executor: Arc<PostgresExecutor>,
    schema: SchemaRef,
) -> DfResult<()> {
    let table_ref = RemoteTableRef::try_from(name.to_string())
        .map_err(|e| DataFusionError::Plan(format!("invalid table ref '{name}': {e}")))?;
    let table = Arc::new(RemoteTable::new(table_ref, schema));
    let provider = Arc::new(SQLFederationProvider::new(executor));
    let table_source = Arc::new(SQLTableSource::new_with_table(provider, table));
    let federated_provider = Arc::new(FederatedTableProviderAdaptor::new(table_source));
    ctx.register_table(name, federated_provider)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sort_alias_in_case_order_by_succeeds_against_postgres() {
    // Start a Postgres container via testcontainers.
    let pg = Postgres::default()
        .start()
        .await
        .expect("failed to start Postgres container");

    let port = pg.get_host_port_ipv4(5432).await.unwrap();
    let conn_str = format!(
        "host=localhost port={port} user=postgres password=postgres dbname=postgres"
    );

    let executor = Arc::new(
        PostgresExecutor::connect(&conn_str)
            .await
            .expect("failed to connect to Postgres"),
    );

    // Create a minimal table shaped like the TPC-DS items table used in q86.
    executor
        .execute_ddl(
            "CREATE TABLE items (
                category TEXT,
                class    TEXT,
                amount   DOUBLE PRECISION
            )",
        )
        .await
        .expect("CREATE TABLE failed");

    // Insert rows that cover multiple categories / classes so ROLLUP produces
    // meaningful grouping levels (including the grand total).
    for (cat, cls, amt) in [
        ("Electronics", "Phones", 100.0_f64),
        ("Electronics", "Laptops", 200.0_f64),
        ("Clothing", "Shirts", 50.0_f64),
        ("Clothing", "Pants", 75.0_f64),
    ] {
        executor
            .execute_ddl(&format!(
                "INSERT INTO items (category, class, amount) VALUES ('{cat}', '{cls}', {amt})"
            ))
            .await
            .expect("INSERT failed");
    }

    // Register the table in a federated DataFusion session context.
    let ctx = make_session_context();
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, true),
        Field::new("class", DataType::Utf8, true),
        Field::new("amount", DataType::Float64, true),
    ]));
    register_table(&ctx, "items", Arc::clone(&executor), schema)
        .expect("failed to register table");

    // TPC-DS q86-shaped query: ORDER BY uses a CASE expression whose condition
    // references the SELECT-list alias `lochierarchy`.  Without
    // `inline_sort_projection_aliases` the federation layer would emit:
    //
    //   ORDER BY CASE WHEN ("lochierarchy" = 0) THEN "category" END
    //
    // and Postgres would reject it with SQLSTATE 42703 because the alias is not
    // visible inside the CASE.  With the fix the alias definition is inlined:
    //
    //   ORDER BY CASE WHEN ((grouping("category") + grouping("class")) = 0)
    //                 THEN "category" END
    let sql = "
        SELECT
            sum(amount)                                AS total_sum,
            category,
            class,
            (grouping(category) + grouping(class))    AS lochierarchy
        FROM items
        GROUP BY ROLLUP(category, class)
        ORDER BY
            lochierarchy DESC NULLS FIRST,
            CASE WHEN lochierarchy = 0 THEN category END ASC NULLS LAST
        LIMIT 10
    ";

    let result = ctx
        .sql(sql)
        .await
        .expect("sql() failed")
        .collect()
        .await;

    assert!(
        result.is_ok(),
        "Query failed — Postgres SQLSTATE 42703 regression (sort alias not inlined): {:?}",
        result.err()
    );
}
