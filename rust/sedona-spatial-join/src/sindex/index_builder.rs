use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion_execution::{memory_pool::{MemoryConsumer, MemoryPool}, SendableRecordBatchStream, TaskContext};
use datafusion_expr::JoinType;
use datafusion_physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use sedona_common::{SedonaOptions, SpatialJoinOptions};
use sedona_expr::statistics::GeoStatistics;
use datafusion_common::Result;

use crate::{operand_evaluator::create_operand_evaluator, sindex::{build_side_batch::{BuildSideBatch, SendableBuildSideBatchStream}, collect::{BuildPartition, BuildSideBatchesCollector, CollectBuildSideMetrics}, index::SpatialIndex, inmem::index_builder::InMemorySpatialIndexBuilder}, spatial_predicate::SpatialPredicate};

pub(crate) trait SpatialIndexBuilder {
    async fn add_partitions(&mut self, partitions: Vec<BuildPartition>) -> Result<()>;

    fn with_stats(&mut self, stats: GeoStatistics) -> Result<()>;

    fn build(self) -> Result<Arc<dyn SpatialIndex>>;
}

/// Metrics for the build phase of the spatial join.
#[derive(Clone, Debug)]
pub(crate) struct SpatialJoinBuildMetrics {
    /// Total time for collecting build-side of join
    pub(crate) build_time: metrics::Time,
    /// Memory used by the spatial-index in bytes
    pub(crate) build_mem_used: metrics::Gauge,
}

impl SpatialJoinBuildMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            build_time: MetricBuilder::new(metrics).subset_time("build_time", partition),
            build_mem_used: MetricBuilder::new(metrics).gauge("build_mem_used", partition),
        }
    }
}

pub(crate) async fn build_spatial_index(
    schema: SchemaRef,
    spatial_predicate: SpatialPredicate,
    options: SpatialJoinOptions,
    join_type: JoinType,
    probe_threads_count: usize,
    memory_pool: Arc<dyn MemoryPool>,
    metrics: SpatialJoinBuildMetrics,
    build_partitions: Vec<BuildPartition>,
) -> Result<Arc<dyn SpatialIndex>> {
    let contains_external_stream = build_partitions.iter().any(|partition| partition.build_side_batch_stream.is_external());
    if !contains_external_stream {
        let mut index_builder = InMemorySpatialIndexBuilder::new(
            schema,
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            memory_pool,
            metrics,
        )?;
        index_builder.add_partitions(build_partitions).await?;
        index_builder.build()
    } else {
        // Box::new(ExternalSpatialIndexBuilder::new())
        todo!()
    }
}

/// The prealloc size for the refiner reservation. This is used to reduce the frequency of growing
/// the reservation when updating the refiner memory reservation.
const REFINER_RESERVATION_PREALLOC_SIZE: usize = 10 * 1024 * 1024; // 10MB

pub(crate) async fn build_index(
    context: Arc<TaskContext>,
    build_schema: SchemaRef,
    build_streams: Vec<SendableRecordBatchStream>,
    spatial_predicate: SpatialPredicate,
    join_type: JoinType,
    probe_threads_count: usize,
    metrics: &ExecutionPlanMetricsSet,
) -> Result<Arc<dyn SpatialIndex>> {

    let session_config = context.session_config();
    let sedona_options = session_config
        .options()
        .extensions
        .get::<SedonaOptions>()
        .cloned()
        .unwrap_or_default();
    let memory_pool = context.memory_pool();
    let runtime_env = context.runtime_env();
    let spill_compression = session_config.spill_compression();
    let evaluator = create_operand_evaluator(&spatial_predicate, sedona_options.spatial_join.clone());
    let collector = BuildSideBatchesCollector::new(evaluator, runtime_env, spill_compression);
    let num_partitions = build_streams.len();
    let mut build_metrics = Vec::with_capacity(num_partitions);
    let mut reservations = Vec::with_capacity(num_partitions);
    for k in 0..num_partitions {
        let consumer = MemoryConsumer::new(format!("SpatialJoinCollectBuildSide[{}]", k)).with_can_spill(true);
        let reservation = consumer.register(memory_pool);
        reservations.push(reservation);
        build_metrics.push(CollectBuildSideMetrics::new(k, metrics));
    }

    let build_partitions = collector.collect_all(build_streams, reservations, build_metrics).await?;

    build_spatial_index(
        build_schema,
        spatial_predicate,
        sedona_options.spatial_join,
        join_type,
        probe_threads_count,
        Arc::clone(memory_pool),
        SpatialJoinBuildMetrics::new(0, metrics),
        build_partitions,
    ).await
}
