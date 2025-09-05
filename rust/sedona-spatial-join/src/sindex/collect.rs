use std::{collections::VecDeque, pin::Pin, sync::Arc, task::{Context, Poll}, time::{Duration, Instant}};

use arrow::ipc::writer::StreamWriter;
use arrow_array::RecordBatch;
use datafusion::parquet::record;
use datafusion_common::Result;
use datafusion_execution::{disk_manager::RefCountedTempFile, runtime_env::RuntimeEnv, SendableRecordBatchStream};
use datafusion_expr::ColumnarValue;
use datafusion_physical_plan::{metrics::ScopedTimerGuard, spill};
use futures::{future, Stream, StreamExt};
use geo_types::Rect;
use sedona_functions::st_analyze_aggr::AnalyzeAccumulator;
use sedona_schema::datatypes::WKB_GEOMETRY;
use wkb::reader::Wkb;

use crate::{concurrent_reservation::ConcurrentReservation, operand_evaluator::{EvaluatedGeometryArray, OperandEvaluator}};

/// BuildSide batch containing the original record batch from the build side and the evaluated
/// geometry array.
pub(crate) struct BuildSideBatch {
    /// Original record batch polled from the build side stream
    batch: RecordBatch,
    /// Evaluated geometry array, containing the geometry array containing geometries to be joined,
    /// rects of joined geometries, evaluated distance columnar values if we are running a distance
    /// join and the distance expression is bound to the build side, etc.
    geom_array: EvaluatedGeometryArray,
}

impl BuildSideBatch {
    pub fn in_mem_size(&self) -> usize {
        // NOTE: sometimes `geom_array` will reuse the memory of `batch`, especially when
        // the expression for evaluating the geometry is a simple column reference. In this case,
        // the in_mem_size will be overestimated.
        self.batch.get_array_memory_size() + self.geom_array.in_mem_size()
    }

    pub fn wkb(&self, idx: usize) -> Option<&Wkb<'_>> {
        let wkbs = self.geom_array.wkbs();
        wkbs[idx].as_ref()
    }

    pub fn rects(&self) -> &Vec<(usize, Rect<f32>)> {
        &self.geom_array.rects
    }

    pub fn distance(&self) -> &Option<ColumnarValue> {
        &self.geom_array.distance
    }
}

pub(crate) trait BuildSideBatchStream: Stream<Item = Result<BuildSideBatch>> {
    fn is_external(&self) -> bool;
}

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

struct SpilledBuildSideBatchStream {
    // TODO: implement spilled batch stream
}

impl SpilledBuildSideBatchStream {
    fn new() -> Self {
        todo!()
    }
}

impl BuildSideBatchStream for SpilledBuildSideBatchStream {
    fn is_external(&self) -> bool {
        true
    }
}

impl futures::Stream for SpilledBuildSideBatchStream {
    type Item = Result<BuildSideBatch>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        todo!()
    }
}

type SendableBuildSideBatchStream = Pin<Box<dyn BuildSideBatchStream + Send>>;

pub(crate) struct BuildPartition {
    build_side_batch_stream: SendableBuildSideBatchStream,
    metrics: Metrics,
}

pub(crate) struct Metrics {
    time_taken: Duration,
    num_batches: usize,
    num_rows: usize,
    total_size_bytes: usize,
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            time_taken: Duration::ZERO,
            num_batches: 0,
            num_rows: 0,
            total_size_bytes: 0,
        }
    }
}

/// A collector for evaluating the spatial expression on build side batches and collect
/// them as asynchronous streams with additional statistics. The asynchronous streams
/// could then be fed into the spatial index builder to build an in-memory or external
/// spatial index, depending on the statistics collected by the collector.
pub(crate) struct BuildSideBatchesCollector {
    evaluator: Arc<dyn OperandEvaluator>,
    reservation: ConcurrentReservation,
    runtime_env: Arc<RuntimeEnv>,
}

impl BuildSideBatchesCollector {
    pub fn new(evaluator: Arc<dyn OperandEvaluator>, reservation: ConcurrentReservation, runtime_env: Arc<RuntimeEnv>) -> Self {
        BuildSideBatchesCollector { evaluator, reservation, runtime_env }
    }

    pub async fn collect(&self, mut stream: SendableRecordBatchStream) -> Result<BuildPartition> {
        let evaluator = self.evaluator.as_ref();
        let mut spill_file_opt: Option<RefCountedTempFile> = None;
        let mut in_mem_batches: Vec<BuildSideBatch> = Vec::new();
        let mut analyzer = AnalyzeAccumulator::new(WKB_GEOMETRY, WKB_GEOMETRY);

        let mut metrics = Metrics::new();
        while let Some(record_batch) = stream.next().await {
            let record_batch = record_batch?;
            // start high-resolution timer for this iteration
            let start = Instant::now();

            metrics.num_batches += 1;
            metrics.num_rows += record_batch.num_rows();

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
            metrics.total_size_bytes += in_mem_size;

            if spill_file_opt.is_none() {
                if self.reservation.reserve(in_mem_size).is_err() {
                    // Spill all in memory batches, and write future batches to spill file
                    let spill_file = self.runtime_env.disk_manager.create_tmp_file("spill when collecting build side batches")?;
                    let inner_file = spill_file.inner();
                    let schema = build_side_batch.batch.schema();
                    for in_mem_batch in &in_mem_batches {
                        let mut writer = StreamWriter::try_new_buffered(inner_file, &schema)?;
                        writer.write(&in_mem_batch.batch)?;
                        writer.finish()?;
                    }
                    in_mem_batches.clear();
                    spill_file_opt = Some(spill_file);
                }
            }

            match &spill_file_opt {
                None => {
                    in_mem_batches.push(build_side_batch);
                }
                Some(spill_file) => {
                    let inner_file = spill_file.inner();
                    let schema = build_side_batch.batch.schema();
                    let mut writer = StreamWriter::try_new_buffered(inner_file, &schema)?;
                    writer.write(&build_side_batch.batch)?;
                    writer.finish()?;
                }
            }

            // accumulate elapsed time for this iteration
            let elapsed = start.elapsed();
            metrics.time_taken += elapsed;
        }
    
        Ok(BuildPartition {
            build_side_batch_stream: Box::pin(InMemoryBuildSideBatchStream::new(in_mem_batches)),
            metrics,
        })
    }

    pub async fn collect_all(&self, streams: Vec<SendableRecordBatchStream>) -> Vec<BuildPartition> {
        todo!()
    }
}
