use std::sync::Arc;

use datafusion::config::SpillCompression;
use datafusion_common::Result;
use datafusion_common_runtime::JoinSet;
use datafusion_execution::{
    memory_pool::MemoryReservation, runtime_env::RuntimeEnv, SendableRecordBatchStream,
};
use datafusion_physical_plan::{
    metrics::{self, ExecutionPlanMetricsSet, MetricBuilder, SpillMetrics},
    SpillManager,
};
use futures::StreamExt;
use sedona_common::sedona_internal_err;
use sedona_functions::st_analyze_aggr::AnalyzeAccumulator;
use sedona_schema::datatypes::WKB_GEOMETRY;

use crate::{
    operand_evaluator::OperandEvaluator,
    sindex::collect::{
        build_side_batch::BuildSideBatch,
        build_side_batch_stream::{
            external::ExternalBuildSideBatchStream, in_mem::InMemoryBuildSideBatchStream,
            SendableBuildSideBatchStream,
        },
        spill::build_side_batch_to_spilled_batch,
        BuildPartition,
    },
};

/// A collector for evaluating the spatial expression on build side batches and collect
/// them as asynchronous streams with additional statistics. The asynchronous streams
/// could then be fed into the spatial index builder to build an in-memory or external
/// spatial index, depending on the statistics collected by the collector.
#[derive(Clone)]
pub(crate) struct BuildSideBatchesCollector {
    evaluator: Arc<dyn OperandEvaluator>,
    runtime_env: Arc<RuntimeEnv>,
    spill_compression: SpillCompression,
}

pub(crate) struct CollectBuildSideMetrics {
    /// Number of batches collected
    num_batches: metrics::Count,
    /// Number of rows collected
    num_rows: metrics::Count,
    /// Total in-memory size of batches collected. If the batches were spilled, this size is the
    /// in-memory size if we load all batches into memory. This does not represent the in-memory size
    /// of the resulting BuildPartition.
    total_size_bytes: metrics::Gauge,
    /// Total time taken to collect and process the build side batches. This does not include the time awaiting
    /// for batches from the input stream.
    time_taken: metrics::Time,
    /// Spill metrics of build partitions collecting phase
    spill_metrics: SpillMetrics,
}

impl CollectBuildSideMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            num_batches: MetricBuilder::new(metrics).counter("build_input_batches", partition),
            num_rows: MetricBuilder::new(metrics).counter("build_input_rows", partition),
            total_size_bytes: MetricBuilder::new(metrics)
                .gauge("build_input_total_size_bytes", partition),
            time_taken: MetricBuilder::new(metrics)
                .subset_time("build_input_collection_time", partition),
            spill_metrics: SpillMetrics::new(metrics, partition),
        }
    }
}

impl BuildSideBatchesCollector {
    pub fn new(
        evaluator: Arc<dyn OperandEvaluator>,
        runtime_env: Arc<RuntimeEnv>,
        spill_compression: SpillCompression,
    ) -> Self {
        BuildSideBatchesCollector {
            evaluator,
            runtime_env,
            spill_compression,
        }
    }

    pub async fn collect(
        &self,
        mut stream: SendableRecordBatchStream,
        mut reservation: MemoryReservation,
        metrics: &CollectBuildSideMetrics,
    ) -> Result<BuildPartition> {
        let evaluator = self.evaluator.as_ref();
        let mut spill_file_opt = None;
        let mut spill_manager_opt: Option<SpillManager> = None;
        let mut in_mem_batches: Vec<BuildSideBatch> = Vec::new();
        let mut analyzer = AnalyzeAccumulator::new(WKB_GEOMETRY, WKB_GEOMETRY);

        while let Some(record_batch) = stream.next().await {
            let record_batch = record_batch?;
            let _timer = metrics.time_taken.timer();

            // Process the record batch and create a BuildSideBatch
            let geom_array = evaluator.evaluate_build(&record_batch)?;

            for wkb in geom_array.wkbs().iter().flatten() {
                analyzer.update_statistics(wkb, wkb.buf().len())?;
            }

            let build_side_batch = BuildSideBatch {
                batch: record_batch,
                geom_array,
            };

            let in_mem_size = build_side_batch.in_mem_size();
            metrics.num_batches.add(1);
            metrics.num_rows.add(build_side_batch.num_rows());
            metrics.total_size_bytes.add(in_mem_size);

            if spill_file_opt.is_none() {
                // Collected batches are in memory, no spilling happened for this patition before. We'll try
                // storing this batch in memory first, and switch to writing everything to disk if we fail
                // to grow the reservation.
                if reservation.try_grow(in_mem_size).is_err() {
                    // Spill all in memory batches, and write future batches to spill file
                    let schema = build_side_batch.batch.schema();
                    let spill_manager = SpillManager::new(
                        Arc::clone(&self.runtime_env),
                        metrics.spill_metrics.clone(),
                        schema,
                    ).with_compression_type(self.spill_compression);
                    let mut in_progress_file =
                        spill_manager.create_in_progress_file("collect_build_partition")?;
                    for in_mem_batch in &in_mem_batches {
                        let spilled_batch = build_side_batch_to_spilled_batch(&in_mem_batch)?;
                        in_progress_file.append_batch(&spilled_batch)?;
                    }
                    in_mem_batches.clear();
                    reservation.free();
                    spill_manager_opt = Some(spill_manager);
                    spill_file_opt = Some(in_progress_file);
                }
            }

            match &mut spill_file_opt {
                None => {
                    in_mem_batches.push(build_side_batch);
                }
                Some(spill_file) => {
                    let spilled_batch = build_side_batch_to_spilled_batch(&build_side_batch)?;
                    spill_file.append_batch(&spilled_batch)?;
                }
            }
        }

        let build_side_batch_stream: SendableBuildSideBatchStream = match spill_file_opt {
            Some(mut spill_file) => {
                let finished = spill_file.finish()?;
                match finished {
                    Some(temp_file) => {
                        if !in_mem_batches.is_empty() {
                            return sedona_internal_err!(
                                "In-memory batches should have been spilled when spill file exists"
                            );
                        }
                        Box::pin(ExternalBuildSideBatchStream::try_new(
                            spill_manager_opt.unwrap(),
                            temp_file,
                        )?)
                    }
                    None => Box::pin(InMemoryBuildSideBatchStream::new(vec![])),
                }
            }
            None => Box::pin(InMemoryBuildSideBatchStream::new(
                in_mem_batches,
            )),
        };

        Ok(BuildPartition {
            build_side_batch_stream,
            geo_statistics: analyzer.finish(),
            reservation,
        })
    }

    pub async fn collect_all(
        &self,
        streams: Vec<SendableRecordBatchStream>,
        reservations: Vec<MemoryReservation>,
        metrics_vec: Vec<CollectBuildSideMetrics>,
    ) -> Result<Vec<BuildPartition>> {
        if streams.is_empty() {
            return Ok(vec![]);
        }

        // Spawn all tasks to scan all build streams concurrently
        let mut join_set = JoinSet::new();
        for (partition_id, ((stream, metrics), reservation)) in streams
            .into_iter()
            .zip(metrics_vec)
            .zip(reservations)
            .enumerate()
        {
            let collector = self.clone();
            join_set.spawn(async move {
                let result = collector.collect(stream, reservation, &metrics).await;
                (partition_id, result)
            });
        }

        // Wait for all async tasks to finish. Results may be returned in arbitrary order,
        // so we need to reorder them by partition_id later.
        let results = join_set.join_all().await;

        // Reorder results according to partition ids
        let mut partitions: Vec<Option<BuildPartition>> = Vec::with_capacity(results.len());
        partitions.resize_with(results.len(), || None);
        for result in results {
            let (partition_id, partition_result) = result;
            let partition = partition_result?;
            partitions[partition_id] = Some(partition);
        }

        Ok(partitions.into_iter().map(|v| v.unwrap()).collect())
    }
}
