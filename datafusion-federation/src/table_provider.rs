use std::{any::Any, borrow::Cow, sync::Arc};

use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::SchemaRef,
    catalog::Session,
    common::Constraints,
    datasource::TableProvider,
    error::{DataFusionError, Result},
    logical_expr::{
        dml::InsertOp, Expr, LogicalPlan, TableProviderFilterPushDown, TableSource, TableType,
    },
    physical_plan::ExecutionPlan,
};

use crate::FederationProvider;

// FederatedTableSourceWrapper helps to recover the FederatedTableSource
// from a TableScan. This wrapper may be avoidable.
#[derive(Debug)]
pub struct FederatedTableProviderAdaptor {
    pub source: Arc<dyn FederatedTableSource>,
    pub table_provider: Option<Arc<dyn TableProvider>>,
}

impl FederatedTableProviderAdaptor {
    pub fn new(source: Arc<dyn FederatedTableSource>) -> Self {
        Self {
            source,
            table_provider: None,
        }
    }

    /// Creates a new FederatedTableProviderAdaptor that falls back to the
    /// provided TableProvider. This is useful if used within a DataFusion
    /// context without the federation optimizer.
    pub fn new_with_provider(
        source: Arc<dyn FederatedTableSource>,
        table_provider: Arc<dyn TableProvider>,
    ) -> Self {
        Self {
            source,
            table_provider: Some(table_provider),
        }
    }
}

#[async_trait]
impl TableProvider for FederatedTableProviderAdaptor {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn schema(&self) -> SchemaRef {
        if let Some(table_provider) = &self.table_provider {
            return table_provider.schema();
        }

        self.source.schema()
    }
    fn constraints(&self) -> Option<&Constraints> {
        if let Some(table_provider) = &self.table_provider {
            return table_provider
                .constraints()
                .or_else(|| self.source.constraints());
        }

        self.source.constraints()
    }
    fn table_type(&self) -> TableType {
        if let Some(table_provider) = &self.table_provider {
            return table_provider.table_type();
        }

        self.source.table_type()
    }
    fn get_logical_plan(&self) -> Option<Cow<'_, LogicalPlan>> {
        if let Some(table_provider) = &self.table_provider {
            return table_provider
                .get_logical_plan()
                .or_else(|| self.source.get_logical_plan());
        }

        self.source.get_logical_plan()
    }
    fn get_column_default(&self, column: &str) -> Option<&Expr> {
        if let Some(table_provider) = &self.table_provider {
            return table_provider
                .get_column_default(column)
                .or_else(|| self.source.get_column_default(column));
        }

        self.source.get_column_default(column)
    }
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        if let Some(table_provider) = &self.table_provider {
            return table_provider.supports_filters_pushdown(filters);
        }

        Ok(vec![
            TableProviderFilterPushDown::Unsupported;
            filters.len()
        ])
    }

    // Scan is not supported; the adaptor should be replaced
    // with a virtual TableProvider that provides federation for a sub-plan.
    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(table_provider) = &self.table_provider {
            return table_provider.scan(state, projection, filters, limit).await;
        }

        Err(DataFusionError::NotImplemented(
            "FederatedTableProviderAdaptor cannot scan".to_string(),
        ))
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(table_provider) = &self.table_provider {
            return table_provider.insert_into(_state, input, insert_op).await;
        }

        Err(DataFusionError::NotImplemented(
            "FederatedTableProviderAdaptor cannot insert_into".to_string(),
        ))
    }

    // DML (DELETE / UPDATE) is handled by leaving the `LogicalPlan::Dml` node
    // intact (see `FederationOptimizerRule`) and dispatching it through the
    // adaptor to the underlying provider. Delegate to the inner provider when
    // one is present; otherwise the adaptor cannot perform DML.
    async fn delete_from(
        &self,
        state: &dyn Session,
        filters: Vec<Expr>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(table_provider) = &self.table_provider {
            return table_provider.delete_from(state, filters).await;
        }

        Err(DataFusionError::NotImplemented(
            "FederatedTableProviderAdaptor cannot delete_from".to_string(),
        ))
    }

    async fn update(
        &self,
        state: &dyn Session,
        assignments: Vec<(String, Expr)>,
        filters: Vec<Expr>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(table_provider) = &self.table_provider {
            return table_provider.update(state, assignments, filters).await;
        }

        Err(DataFusionError::NotImplemented(
            "FederatedTableProviderAdaptor cannot update".to_string(),
        ))
    }
}

// FederatedTableProvider extends DataFusion's TableProvider trait
// to allow grouping of TableScans of the same FederationProvider.
#[async_trait]
pub trait FederatedTableSource: TableSource {
    // Return the FederationProvider associated with this Table
    fn federation_provider(&self) -> Arc<dyn FederationProvider>;
}

impl std::fmt::Debug for dyn FederatedTableSource {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "FederatedTableSource: {:?}",
            self.federation_provider().name()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::logical_expr::{col, lit};
    use datafusion::optimizer::optimizer::Optimizer;

    fn test_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]))
    }

    // Minimal FederatedTableSource for tests (no real federation).
    #[derive(Debug)]
    struct TestSource {
        schema: SchemaRef,
    }

    impl TableSource for TestSource {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }
    }

    #[derive(Debug)]
    struct TestProvider;

    impl FederationProvider for TestProvider {
        fn name(&self) -> &str {
            "test"
        }
        fn compute_context(&self) -> Option<String> {
            None
        }
        fn optimizer(&self) -> Option<Arc<Optimizer>> {
            None
        }
    }

    impl FederatedTableSource for TestSource {
        fn federation_provider(&self) -> Arc<dyn FederationProvider> {
            Arc::new(TestProvider)
        }
    }

    // A TableProvider whose DML methods return a recognizable sentinel error so
    // we can prove the adaptor delegated to it (rather than returning its own
    // "cannot ..." error).
    #[derive(Debug)]
    struct RecordingProvider {
        schema: SchemaRef,
    }

    #[async_trait]
    impl TableProvider for RecordingProvider {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }
        fn table_type(&self) -> TableType {
            TableType::Base
        }
        async fn scan(
            &self,
            _state: &dyn Session,
            _projection: Option<&Vec<usize>>,
            _filters: &[Expr],
            _limit: Option<usize>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Err(DataFusionError::NotImplemented(
                "recording_scan".to_string(),
            ))
        }
        async fn delete_from(
            &self,
            _state: &dyn Session,
            _filters: Vec<Expr>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Err(DataFusionError::NotImplemented(
                "recording_delete_from".to_string(),
            ))
        }
        async fn update(
            &self,
            _state: &dyn Session,
            _assignments: Vec<(String, Expr)>,
            _filters: Vec<Expr>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Err(DataFusionError::NotImplemented(
                "recording_update".to_string(),
            ))
        }
    }

    fn adaptor_with_provider() -> FederatedTableProviderAdaptor {
        let source = Arc::new(TestSource {
            schema: test_schema(),
        });
        let provider = Arc::new(RecordingProvider {
            schema: test_schema(),
        });
        FederatedTableProviderAdaptor::new_with_provider(source, provider)
    }

    fn session() -> datafusion::execution::session_state::SessionState {
        crate::default_session_state()
    }

    #[tokio::test]
    async fn delete_from_delegates_to_inner_provider() {
        let adaptor = adaptor_with_provider();
        let state = session();
        let err = adaptor
            .delete_from(&state, vec![col("id").eq(lit(1))])
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("recording_delete_from"),
            "expected delegation to inner provider, got: {err}"
        );
    }

    #[tokio::test]
    async fn update_delegates_to_inner_provider() {
        let adaptor = adaptor_with_provider();
        let state = session();
        let err = adaptor
            .update(
                &state,
                vec![("id".to_string(), lit(2))],
                vec![col("id").eq(lit(1))],
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("recording_update"),
            "expected delegation to inner provider, got: {err}"
        );
    }

    #[tokio::test]
    async fn delete_from_without_inner_provider_is_not_implemented() {
        let source = Arc::new(TestSource {
            schema: test_schema(),
        });
        let adaptor = FederatedTableProviderAdaptor::new(source);
        let state = session();
        let err = adaptor.delete_from(&state, vec![]).await.unwrap_err();
        assert!(
            err.to_string()
                .contains("FederatedTableProviderAdaptor cannot delete_from"),
            "got: {err}"
        );
    }
}
