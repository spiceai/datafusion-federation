use std::sync::Arc;

use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::SchemaRef,
    catalog::SchemaProvider,
    error::{DataFusionError, Result},
    execution::context::{SessionContext, SessionState},
    physical_plan::{stream::RecordBatchStreamAdapter, SendableRecordBatchStream},
    sql::unparser::dialect::{DefaultDialect, Dialect},
};
use futures::TryStreamExt;

use datafusion_federation_sql::SQLExecutor;

pub fn overwrite_default_schema(
    state: &SessionState,
    schema: Arc<dyn SchemaProvider>,
) -> Result<()> {
    let options = &state.config().options().catalog;
    let catalog = state
        .catalog_list()
        .catalog(options.default_catalog.as_str())
        .unwrap();

    catalog.register_schema(options.default_schema.as_str(), schema)?;
    Ok(())
}

pub struct MockExecutor {
    session: Arc<SessionContext>,
    name: String,
}

impl MockExecutor {
    pub fn new(session: Arc<SessionContext>, name: String) -> Self {
        Self { session, name }
    }
}

#[async_trait]
impl SQLExecutor for MockExecutor {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_context(&self) -> Option<String> {
        Some(format!("{}-exec", self.name))
    }

    fn execute(&self, sql: &str, schema: SchemaRef) -> Result<SendableRecordBatchStream> {
        // Execute it using the remote datafusion session context
        let future_stream = _execute(self.session.clone(), sql.to_string());
        let stream = futures::stream::once(future_stream).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream,
        )))
    }

    async fn table_names(&self) -> Result<Vec<String>> {
        Err(DataFusionError::NotImplemented(
            "table inference not implemented".to_string(),
        ))
    }

    async fn get_table_schema(&self, table_name: &str) -> Result<SchemaRef> {
        let sql = format!("select * from {table_name} limit 1");
        let df = self.session.sql(&sql).await?;
        let schema = df.schema().as_arrow().clone();
        Ok(Arc::new(schema))
    }

    fn dialect(&self) -> Arc<dyn Dialect> {
        Arc::new(DefaultDialect {})
    }
}

async fn _execute(ctx: Arc<SessionContext>, sql: String) -> Result<SendableRecordBatchStream> {
    ctx.sql(&sql).await?.execute_stream().await
}
