use std::{
    collections::VecDeque,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::{Duration, Instant},
};

use arrow::{
    array::{Float32Array, Float32Builder, Float64Array, NullBufferBuilder, StructBuilder},
    buffer::NullBuffer,
    ipc::{writer::StreamWriter, RecordBatchBuilder},
};
use arrow_array::{Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema};
use datafusion::config::SpillCompression;
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_common_runtime::JoinSet;
use datafusion_execution::{
    disk_manager::RefCountedTempFile,
    memory_pool::{MemoryConsumer, MemoryReservation},
    runtime_env::{self, RuntimeEnv},
    SendableRecordBatchStream,
};
use datafusion_expr::ColumnarValue;
use datafusion_physical_plan::{
    metrics::{self, ExecutionPlanMetricsSet, MetricBuilder, ScopedTimerGuard, SpillMetrics, Time},
    spill, SpillManager,
};
use futures::{future, stream::Collect, Stream, StreamExt};
use geo::coord;
use geo_types::Rect;
use sedona_expr::statistics::GeoStatistics;
use sedona_functions::st_analyze_aggr::AnalyzeAccumulator;
use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY};
use wkb::reader::Wkb;

use crate::{
    index::SpatialJoinBuildMetrics,
    operand_evaluator::{EvaluatedGeometryArray, OperandEvaluator},
    sindex::{
        build_side_batch::{BuildSideBatch, BuildSideBatchStream, SendableBuildSideBatchStream},
        collect,
    },
};

struct InMemoryBuildSideBatchStream {
    batches: VecDeque<BuildSideBatch>,
}

impl InMemoryBuildSideBatchStream {
    fn new(batches: Vec<BuildSideBatch>) -> Self {
        InMemoryBuildSideBatchStream {
            batches: VecDeque::from(batches),
        }
    }
}

impl BuildSideBatchStream for InMemoryBuildSideBatchStream {
    fn is_external(&self) -> bool {
        false
    }

    fn reservation(&self) -> &MemoryReservation {
        todo!()
    }

    fn take_reservation(self) -> MemoryReservation {
        todo!()
    }
}

impl futures::Stream for InMemoryBuildSideBatchStream {
    type Item = Result<BuildSideBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
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
    fn new(spill_manager: SpillManager, spill_file: RefCountedTempFile) -> Self {
        todo!()
    }
}

impl BuildSideBatchStream for ExternalBuildSideBatchStream {
    fn is_external(&self) -> bool {
        true
    }

    fn reservation(&self) -> &MemoryReservation {
        todo!()
    }

    fn take_reservation(self) -> MemoryReservation {
        todo!()
    }
}

impl futures::Stream for ExternalBuildSideBatchStream {
    type Item = Result<BuildSideBatch>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}

pub(crate) struct BuildPartition {
    pub build_side_batch_stream: SendableBuildSideBatchStream,
    pub geo_statistics: GeoStatistics,
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
                    );
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
                        // let stream = spill_manager_opt.unwrap().read_spill_as_stream(temp_file);
                        Box::pin(ExternalBuildSideBatchStream::new(
                            spill_manager_opt.unwrap(),
                            temp_file,
                        ))
                    }
                    None => Box::pin(InMemoryBuildSideBatchStream::new(vec![])),
                }
            }
            None => Box::pin(InMemoryBuildSideBatchStream::new(in_mem_batches)),
        };

        Ok(BuildPartition {
            build_side_batch_stream,
            geo_statistics: analyzer.finish(),
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

fn schema_of_spilled_build_side_batch(
    orig_schema: &Schema,
    sedona_type: &SedonaType,
) -> Result<Schema> {
    let data_field = Field::new(
        "data",
        DataType::Struct(orig_schema.fields().clone()),
        false,
    );
    let geom_field = sedona_type.to_storage_field("geom", true)?;
    let rect_field = Field::new(
        "rect",
        DataType::Struct(Fields::from(vec![
            Field::new("min_x", DataType::Float32, false),
            Field::new("min_y", DataType::Float32, false),
            Field::new("max_x", DataType::Float32, false),
            Field::new("max_y", DataType::Float32, false),
        ])),
        true,
    );
    let dist_field = Field::new("dist", DataType::Float64, true);
    let schema = Schema::new(vec![data_field, geom_field, rect_field, dist_field]);
    Ok(schema)
}

fn build_side_batch_to_spilled_batch(build_side_batch: &BuildSideBatch) -> Result<RecordBatch> {
    let orig_schema = build_side_batch.batch.schema();
    let geom_array = &build_side_batch.geom_array;
    let sedona_type = &geom_array.sedona_type;
    let data_inner_fields = Fields::from(orig_schema.fields().clone());
    let data_struct_field = Field::new("data", DataType::Struct(data_inner_fields.clone()), false);
    let geom_field = sedona_type.to_storage_field("geom", true)?;

    let rect_inner_fields = Fields::from(vec![
        Field::new("min_x", DataType::Float32, false),
        Field::new("min_y", DataType::Float32, false),
        Field::new("max_x", DataType::Float32, false),
        Field::new("max_y", DataType::Float32, false),
    ]);
    let rect_field = Field::new("rect", DataType::Struct(rect_inner_fields.clone()), true);
    let dist_field = Field::new("dist", DataType::Float64, true);
    let schema = Schema::new(vec![data_struct_field, geom_field, rect_field, dist_field]);

    let data_batch = &build_side_batch.batch;
    let data_arrays = data_batch.columns().to_vec();

    let data_struct_array = StructArray::try_new(data_inner_fields, data_arrays, None)?;

    let num_rows = data_batch.num_rows();
    let mut min_x_builder = Float32Builder::with_capacity(num_rows);
    let mut min_y_builder = Float32Builder::with_capacity(num_rows);
    let mut max_x_builder = Float32Builder::with_capacity(num_rows);
    let mut max_y_builder = Float32Builder::with_capacity(num_rows);
    let mut null_buffer_builder = NullBufferBuilder::new_with_len(num_rows);

    let rects = build_side_batch.rects();
    for i in 0..num_rows {
        let rect_opt = &rects[i];
        if let Some(rect) = rect_opt {
            min_x_builder.append_value(rect.min().x);
            min_y_builder.append_value(rect.min().y);
            max_x_builder.append_value(rect.max().x);
            max_y_builder.append_value(rect.max().y);
            null_buffer_builder.append_non_null();
        } else {
            min_x_builder.append_value(0.0);
            min_y_builder.append_value(0.0);
            max_x_builder.append_value(0.0);
            max_y_builder.append_value(0.0);
            null_buffer_builder.append_null();
        }
    }

    let min_x_array = min_x_builder.finish();
    let min_y_array = min_y_builder.finish();
    let max_x_array = max_x_builder.finish();
    let max_y_array = max_y_builder.finish();
    let null_buffer = null_buffer_builder.finish();

    let rect_array = StructArray::try_new(
        rect_inner_fields,
        vec![
            Arc::new(min_x_array),
            Arc::new(min_y_array),
            Arc::new(max_x_array),
            Arc::new(max_y_array),
        ],
        null_buffer,
    )?;

    let mut dist_builder = arrow::array::Float64Builder::with_capacity(num_rows);
    match &geom_array.distance {
        Some(ColumnarValue::Scalar(scalar)) => match scalar {
            ScalarValue::Float64(dist_value) => {
                for _ in 0..num_rows {
                    dist_builder.append_option(*dist_value);
                }
            }
            _ => {
                return Err(DataFusionError::Internal(
                    "Distance columnar value is not a Float64Array".to_string(),
                ));
            }
        },
        Some(ColumnarValue::Array(array)) => {
            let float_array = array
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap();
            dist_builder.append_array(float_array);
        }
        None => {
            for _ in 0..num_rows {
                dist_builder.append_null();
            }
        }
    }

    let dist_array = dist_builder.finish();

    let columns = vec![
        Arc::new(data_struct_array) as Arc<dyn arrow::array::Array>,
        Arc::clone(&geom_array.geometry_array),
        Arc::new(rect_array) as Arc<dyn arrow::array::Array>,
        Arc::new(dist_array) as Arc<dyn arrow::array::Array>,
    ];

    let record_batch = RecordBatch::try_new(Arc::new(schema), columns)?;
    Ok(record_batch)
}

fn spilled_batch_to_build_side_batch(record_batch: RecordBatch) -> Result<BuildSideBatch> {
    // Extract the data struct array (column 0) and convert back to the original RecordBatch
    let data_array = record_batch
        .column(0)
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            DataFusionError::Internal("Expected data column to be a StructArray".to_string())
        })?;

    let data_schema = Arc::new(Schema::new(match data_array.data_type() {
        DataType::Struct(fields) => fields.clone(),
        _ => {
            return Err(DataFusionError::Internal(
                "Expected data column to have Struct data type".to_string(),
            ))
        }
    }));

    let data_columns = (0..data_array.num_columns())
        .map(|i| Arc::clone(data_array.column(i)))
        .collect::<Vec<_>>();

    let batch = RecordBatch::try_new(data_schema, data_columns)?;

    // Extract the geometry array (column 1)
    let geom_array = Arc::clone(record_batch.column(1));

    // Determine the SedonaType from the geometry field in the record batch schema
    let schema = record_batch.schema();
    let geom_field = schema.field(1);
    let sedona_type = SedonaType::from_storage_field(geom_field)?;

    // Extract the rect array (column 2) and convert back to Vec<Option<Rect<f32>>>
    let rect_array = record_batch
        .column(2)
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            DataFusionError::Internal("Expected rect column to be a StructArray".to_string())
        })?;

    let min_x_array = rect_array
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            DataFusionError::Internal("Expected min_x to be Float32Array".to_string())
        })?;
    let min_y_array = rect_array
        .column(1)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            DataFusionError::Internal("Expected min_y to be Float32Array".to_string())
        })?;
    let max_x_array = rect_array
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            DataFusionError::Internal("Expected max_x to be Float32Array".to_string())
        })?;
    let max_y_array = rect_array
        .column(3)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            DataFusionError::Internal("Expected max_y to be Float32Array".to_string())
        })?;

    let mut rects = Vec::with_capacity(rect_array.len());
    for i in 0..rect_array.len() {
        if rect_array.is_null(i) {
            rects.push(None);
        } else {
            let min_x = min_x_array.value(i);
            let min_y = min_y_array.value(i);
            let max_x = max_x_array.value(i);
            let max_y = max_y_array.value(i);
            let rect = Rect::new(coord! { x: min_x, y: min_y }, coord! { x: max_x, y: max_y });
            rects.push(Some(rect));
        }
    }

    // Extract the distance array (column 3) and convert back to ColumnarValue
    let dist_array = record_batch
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| {
            DataFusionError::Internal("Expected dist column to be Float64Array".to_string())
        })?;

    let distance = if dist_array.len() > 0 {
        // Check if all values are the same (scalar case)
        let first_value = if dist_array.is_null(0) {
            None
        } else {
            Some(dist_array.value(0))
        };

        let all_same = (1..dist_array.len()).all(|i| {
            let current_value = if dist_array.is_null(i) {
                None
            } else {
                Some(dist_array.value(i))
            };
            current_value == first_value
        });

        if all_same {
            Some(ColumnarValue::Scalar(ScalarValue::Float64(first_value)))
        } else {
            Some(ColumnarValue::Array(Arc::clone(record_batch.column(3))))
        }
    } else {
        None
    };

    // Create EvaluatedGeometryArray
    let mut geom_array = EvaluatedGeometryArray::try_new(geom_array, &sedona_type)?;
    geom_array.distance = distance;
    // Note: rects are already computed in try_new, but we need to replace them with the ones from the spilled batch
    // because the spilled batch may have been modified (e.g., filtered)
    geom_array.rects = rects;

    Ok(BuildSideBatch { batch, geom_array })
}
