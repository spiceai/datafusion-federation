use async_trait::async_trait;

use super::{table::SQLTable, RemoteTableRef, SQLTableSource};
use crate::{sql::SQLFederationProvider, FederatedTableProviderAdaptor};
use crate::{
    table_reference::MultiPartTableReference, FederatedTableProviderAdaptor, FederatedTableSource,
    FederationProvider,
};
use datafusion::logical_expr::{TableSource, TableType};
use datafusion::{
    arrow::datatypes::SchemaRef, catalog::SchemaProvider, datasource::TableProvider, error::Result,
};
use futures::future::join_all;
use std::{any::Any, sync::Arc};

/// An in-memory schema provider for SQL tables.
#[derive(Debug)]
pub struct SQLSchemaProvider {
    tables: Vec<Arc<SQLTableSource>>,
}

impl SQLSchemaProvider {
    /// Creates a new SQLSchemaProvider from a [`SQLFederationProvider`].
    /// Initializes the schema provider by fetching table names and schema from the federation provider's executor,
    pub async fn new(provider: Arc<SQLFederationProvider>) -> Result<Self> {
        let tables = Arc::clone(&provider.executor)
            .table_names()
            .await?
            .iter()
            .map(RemoteTableRef::try_from)
            .collect::<Result<Vec<_>>>()?;

        Self::new_with_table_references(provider, tables).await
    }

    /// Creates a new SQLSchemaProvider from a SQLFederationProvider and a list of table references.
    /// Fetches the schema for each table using the executor's implementation.
    pub async fn new_with_tables<T: AsRef<str>>(
        provider: Arc<SQLFederationProvider>,
        tables: impl IntoIterator<Item = T>,
    ) -> Result<Self> {
        let tables = tables
            .into_iter()
            .map(|x| RemoteTableRef::try_from(x.as_ref()))
            .collect::<Result<Vec<_>>>()?;

        let futures: Vec<_> = tables
            .into_iter()
            .map(|t| SQLTableSource::new(Arc::clone(&provider), t))
            .collect();
        let results: Result<Vec<_>> = join_all(futures).await.into_iter().collect();
        let tables = results?.into_iter().map(Arc::new).collect();
        Ok(Self { tables })
    }

    /// Creates a new SQLSchemaProvider from a SQLFederationProvider and a list of custom table instances.
    pub fn new_with_custom_tables(
        provider: Arc<SQLFederationProvider>,
        tables: Vec<Arc<dyn SQLTable>>,
    ) -> Self {
        Self {
            tables: tables
                .into_iter()
                .map(|table| SQLTableSource::new_with_table(provider.clone(), table))
                .map(Arc::new)
                .collect(),
        }
    }

    pub async fn new_with_table_references(
        provider: Arc<SQLFederationProvider>,
        tables: Vec<RemoteTableRef>,
    ) -> Result<Self> {
        let futures: Vec<_> = tables
            .into_iter()
            .map(|t| SQLTableSource::new(Arc::clone(&provider), t))
            .collect();
        let results: Result<Vec<_>> = join_all(futures).await.into_iter().collect();
        let tables = results?.into_iter().map(Arc::new).collect();
        Ok(Self { tables })
    }
}

#[async_trait]
impl SchemaProvider for SQLSchemaProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_names(&self) -> Vec<String> {
        self.tables
            .iter()
            .map(|source| source.table_reference().to_string())
            .collect()
    }

    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        if let Some(source) = self
            .tables
            .iter()
            .find(|s| s.table_reference().to_string().eq(name))
        {
            let adaptor = FederatedTableProviderAdaptor::new(source.clone());
            return Ok(Some(Arc::new(adaptor)));
        }
        Ok(None)
    }

    fn table_exist(&self, name: &str) -> bool {
        self.tables
            .iter()
            .any(|source| source.table_reference().to_string().eq(name))
    }
}

#[derive(Debug)]
pub struct MultiSchemaProvider {
    children: Vec<Arc<dyn SchemaProvider>>,
}

impl MultiSchemaProvider {
    pub fn new(children: Vec<Arc<dyn SchemaProvider>>) -> Self {
        Self { children }
    }
}

#[async_trait]
impl SchemaProvider for MultiSchemaProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_names(&self) -> Vec<String> {
        self.children.iter().flat_map(|p| p.table_names()).collect()
    }

    async fn table(&self, name: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        for child in &self.children {
            if let Ok(Some(table)) = child.table(name).await {
                return Ok(Some(table));
            }
        }
        Ok(None)
    }

    fn table_exist(&self, name: &str) -> bool {
        self.children.iter().any(|p| p.table_exist(name))
    }
}

// TODO merge/rework
#[derive(Debug)]
pub struct SQLTableSource {
    provider: Arc<SQLFederationProvider>,
    table_name: MultiPartTableReference,
    schema: SchemaRef,
}

impl SQLTableSource {
    // creates a SQLTableSource and infers the table schema
    pub async fn new(
        provider: Arc<SQLFederationProvider>,
        table_name: impl Into<MultiPartTableReference>,
    ) -> Result<Self> {
        let table_name = table_name.into();
        let schema = Arc::clone(&provider)
            .executor
            .get_table_schema(table_name.to_string().as_str())
            .await?;
        Self::new_with_schema(provider, table_name, schema)
    }

    pub fn new_with_schema(
        provider: Arc<SQLFederationProvider>,
        table_name: impl Into<MultiPartTableReference>,
        schema: SchemaRef,
    ) -> Result<Self> {
        Ok(Self {
            provider,
            table_name: table_name.into(),
            schema,
        })
    }

    pub fn table_name(&self) -> &MultiPartTableReference {
        &self.table_name
    }
}

impl FederatedTableSource for SQLTableSource {
    fn remote_table_name(&self) -> Option<MultiPartTableReference> {
        Some(self.table_name.clone())
    }

    fn federation_provider(&self) -> Arc<dyn FederationProvider> {
        Arc::clone(&self.provider) as Arc<dyn FederationProvider>
    }
}

impl TableSource for SQLTableSource {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
    fn table_type(&self) -> TableType {
        TableType::Temporary
    }
}
