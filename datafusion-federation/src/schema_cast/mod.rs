use async_stream::stream;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::Statistics;
use datafusion::config::ConfigOptions;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{expressions::Column, PhysicalSortExpr};
use datafusion::physical_plan::filter_pushdown::{FilterDescription, FilterPushdownPhase};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::sort_pushdown::SortOrderPushdownResult;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    ChildStats, DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PhysicalExpr,
    PlanProperties, StatisticsArgs,
};
use futures::StreamExt;
use std::fmt;
use std::sync::Arc;

mod intervals_cast;
mod lists_cast;
pub mod record_convert;
mod struct_cast;

#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct SchemaCastScanExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    metrics_set: ExecutionPlanMetricsSet,
}

impl SchemaCastScanExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, schema: SchemaRef) -> Self {
        let eq_properties = input.equivalence_properties().clone();
        let emission_type = input.pipeline_behavior();
        let boundedness = input.boundedness();
        let properties = PlanProperties::new(
            eq_properties,
            input.output_partitioning().clone(),
            emission_type,
            boundedness,
        );
        Self {
            input,
            schema,
            properties: Arc::new(properties),
            metrics_set: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for SchemaCastScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SchemaCastScanExec")
    }
}

impl ExecutionPlan for SchemaCastScanExec {
    fn name(&self) -> &str {
        "SchemaCastScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    /// Casts whole batches to `schema`; holds no expressions of its own.
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    /// Prevents the introduction of additional `RepartitionExec` and processing input in parallel.
    /// This guarantees that the input is processed as a single stream, preserving the order of the data.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() == 1 {
            Ok(Arc::new(Self::new(
                Arc::clone(&children[0]),
                Arc::clone(&self.schema),
            )))
        } else {
            Err(DataFusionError::Execution(
                "SchemaCastScanExec expects exactly one input".to_string(),
            ))
        }
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let mut stream = self.input.execute(partition, context)?;
        let schema = Arc::clone(&self.schema);
        let baseline_metrics = BaselineMetrics::new(&self.metrics_set, partition);

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&schema),
            {
                stream! {
                    while let Some(batch) = stream.next().await {
                        let _timer = baseline_metrics.elapsed_compute().timer();
                        let batch = record_convert::try_cast_to(batch?, Arc::clone(&schema));
                        let batch = batch.map_err(|e| { DataFusionError::External(Box::new(e)) });
                        if let Ok(ref b) = batch {
                            baseline_metrics.output_rows().add(b.num_rows());
                        }
                        yield batch;
                    }
                }
            },
        )))
    }

    /// Casting keeps every column at its position, so a sort on columns the cast
    /// leaves at their type is the same sort on the input. A column whose type
    /// changes is not passed through: the cast can change how its values order.
    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> Result<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        let input_schema = self.input.schema();
        for sort in order {
            let Some(column) = sort.expr.downcast_ref::<Column>() else {
                return Ok(SortOrderPushdownResult::Unsupported);
            };
            let index = column.index();
            if index >= input_schema.fields().len()
                || index >= self.schema.fields().len()
                || input_schema.field(index).name() != column.name()
                || input_schema.field(index).data_type() != self.schema.field(index).data_type()
            {
                return Ok(SortOrderPushdownResult::Unsupported);
            }
        }
        let schema = Arc::clone(&self.schema);
        Ok(self
            .input
            .try_pushdown_sort(order)?
            .map(|input| Arc::new(Self::new(input, Arc::clone(&schema))) as Arc<dyn ExecutionPlan>))
    }

    /// Casting keeps every row, so a limit applies to the input unchanged.
    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        let input = self.input.with_fetch(limit)?;
        Some(Arc::new(Self::new(input, Arc::clone(&self.schema))))
    }

    fn fetch(&self) -> Option<usize> {
        self.input.fetch()
    }

    fn child_stats_requests(&self, partition: Option<usize>) -> Vec<ChildStats> {
        vec![ChildStats::At(partition)]
    }

    /// Casting changes column types but not row counts, so the input's statistics
    /// describe this node's output.
    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        _args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        input_stats.first().map(Arc::clone).ok_or_else(|| {
            DataFusionError::Internal("SchemaCastScanExec expects exactly one input".to_string())
        })
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics_set.clone_inner())
    }

    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterDescription> {
        FilterDescription::from_children(parent_filters, &self.children())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Int32Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::stats::Precision;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::StatisticsContext;

    /// The cast node forwards its input's statistics, so a planner sizing a join
    /// or answering `COUNT(*)` sees the federated source's row count.
    #[test]
    fn schema_cast_forwards_input_statistics() {
        let input_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&input_schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .expect("record batch");
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], input_schema, None)
            .expect("memory exec");
        let output_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let exec = SchemaCastScanExec::new(input, output_schema);

        let stats = StatisticsContext::new()
            .compute(&exec, &StatisticsArgs::new())
            .expect("statistics");
        assert_eq!(stats.num_rows, Precision::Exact(3));
    }
}
