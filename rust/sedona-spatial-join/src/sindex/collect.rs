use std::{collections::VecDeque, pin::Pin, sync::Arc, task::{Context, Poll}, time::{Duration, Instant}};

use arrow::ipc::writer::StreamWriter;
use arrow_array::RecordBatch;
use datafusion::{config::SpillCompression, parquet::record};
use datafusion_common::Result;
use datafusion_common_runtime::JoinSet;
use datafusion_execution::{disk_manager::RefCountedTempFile, runtime_env::RuntimeEnv, SendableRecordBatchStream};
use datafusion_expr::ColumnarValue;
use datafusion_physical_plan::{metrics::{self, ExecutionPlanMetricsSet, MetricBuilder, ScopedTimerGuard, SpillMetrics, Time}, spill, SpillManager};
use futures::{future, stream::Collect, Stream, StreamExt};
use geo_types::Rect;
use sedona_expr::statistics::GeoStatistics;
use sedona_functions::st_analyze_aggr::AnalyzeAccumulator;
use sedona_schema::datatypes::WKB_GEOMETRY;
use wkb::reader::Wkb;

use crate::{concurrent_reservation::ConcurrentReservation, index::SpatialJoinBuildMetrics, operand_evaluator::{EvaluatedGeometryArray, OperandEvaluator}, sindex::{build_side_batch::{BuildSideBatch, BuildSideBatchStream, SendableBuildSideBatchStream}, collect}};

struct InMemoryBuildSideBatchStream {
    batches: VecDeque<BuildSideBatch>,
}

impl InMemoryBuildSideBatchStream {
    fn new(batches: Vec<BuildSideBatch>) -> Self {
        InMemoryBuildSideBatchStream { batches: VecDeque::from(batches) }
    }
}

impl BuildSideBatchStream for InMemoryBuildSideBatchStream {
    fn is_external(&self) -> bool {
        false
    }
}

impl futures::Stream for InMemoryBuildSideBatchStream {
    type Item = Result<BuildSideBatch>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let front = self.get_mut().batches.pop_front();
        match front {
            Some(batch) => Poll::Ready(Some(Ok(batch))),
            None => Poll::Ready(None),
        }
    }
}

struct ExternalBuildSideBatchStream {
    // TODO: implement spilled batch stream
}

impl ExternalBuildSideBatchStream {
    fn new(spill_file: RefCountedTempFile) -> Self {
        todo!()
    }
}

impl BuildSideBatchStream for ExternalBuildSideBatchStream {
    fn is_external(&self) -> bool {
        true
    }
}

impl futures::Stream for ExternalBuildSideBatchStream {
    type Item = Result<BuildSideBatch>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        todo!()
    }
}

pub(crate) struct BuildPartition {
    build_side_batch_stream: SendableBuildSideBatchStream,
    geo_statistics: GeoStatistics,
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
            total_size_bytes: MetricBuilder::new(metrics).gauge("build_input_total_size_bytes", partition),
            time_taken: MetricBuilder::new(metrics).subset_time("build_input_collection_time", partition),
            spill_metrics: SpillMetrics::new(metrics, partition),
        }
    }
}

/// A collector for evaluating the spatial expression on build side batches and collect
/// them as asynchronous streams with additional statistics. The asynchronous streams
/// could then be fed into the spatial index builder to build an in-memory or external
/// spatial index, depending on the statistics collected by the collector.
#[derive(Clone)]
pub(crate) struct BuildSideBatchesCollector {
    evaluator: Arc<dyn OperandEvaluator>,
    reservation: Arc<ConcurrentReservation>,
    runtime_env: Arc<RuntimeEnv>,
    spill_compression: SpillCompression,
}

impl BuildSideBatchesCollector {
    pub fn new(evaluator: Arc<dyn OperandEvaluator>, reservation: Arc<ConcurrentReservation>, runtime_env: Arc<RuntimeEnv>, spill_compression: SpillCompression) -> Self {
        BuildSideBatchesCollector { evaluator, reservation, runtime_env, spill_compression }
    }

    pub async fn collect(&self, mut stream: SendableRecordBatchStream, metrics: &CollectBuildSideMetrics) -> Result<BuildPartition> {
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
                if self.reservation.reserve(in_mem_size).is_err() {
                    // Spill all in memory batches, and write future batches to spill file
                    let schema = build_side_batch.batch.schema();
                    let spill_manager = SpillManager::new(Arc::clone(&self.runtime_env), metrics.spill_metrics.clone(), schema);
                    let mut in_progress_file = spill_manager.create_in_progress_file("collect_build_partition")?;
                    for in_mem_batch in &in_mem_batches {
                        // TODO: create a temporary batch with extended schema to include evaluated geometry columns
                        in_progress_file.append_batch(&in_mem_batch.batch)?;
                    }
                    in_mem_batches.clear();
                    spill_manager_opt = Some(spill_manager);
                    spill_file_opt = Some(in_progress_file);
                }
            }

            match &mut spill_file_opt {
                None => {
                    in_mem_batches.push(build_side_batch);
                }
                Some(spill_file) => {
                    // TODO: create a temporary batch with extended schema to include evaluated geometry columns
                    spill_file.append_batch(&build_side_batch.batch)?;
                }
            }
        }

        let build_side_batch_stream: SendableBuildSideBatchStream = match spill_file_opt {
            Some(mut spill_file) => {
                let finished = spill_file.finish()?;
                match finished {
                    Some(temp_file) => {
                        // let stream = spill_manager_opt.unwrap().read_spill_as_stream(temp_file);
                        Box::pin(ExternalBuildSideBatchStream::new(temp_file))
                    },
                    None => Box::pin(InMemoryBuildSideBatchStream::new(vec![])),
                }
            }
            None => {
                Box::pin(InMemoryBuildSideBatchStream::new(in_mem_batches))
            }
        };
    
        Ok(BuildPartition {
            build_side_batch_stream,
            geo_statistics: analyzer.finish(),
        })
    }

    pub async fn collect_all(&self, streams: Vec<SendableRecordBatchStream>, metrics_vec: Vec<CollectBuildSideMetrics>) -> Result<Vec<BuildPartition>> {
        if streams.is_empty() {
            return Ok(vec![]);
        }

        // Spawn all tasks to scan all build streams concurrently
        let mut join_set = JoinSet::new();
        for (partition_id, (stream, metrics)) in streams.into_iter().zip(metrics_vec).enumerate() {
            let collector = self.clone();
            join_set.spawn(async move {
                let result = collector.collect(stream, &metrics).await;
                (partition_id, result)
            });
        }

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
