// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::{fs::File, io::BufReader, sync::Arc};

use arrow::{
    array::{Float32Array, Float32Builder, Float64Array, NullBufferBuilder},
    ipc::{
        reader::StreamReader,
        writer::{IpcWriteOptions, StreamWriter},
    },
};
use arrow_array::{Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema, SchemaRef};
use datafusion::config::SpillCompression;
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_execution::{disk_manager::RefCountedTempFile, runtime_env::RuntimeEnv};
use datafusion_expr::ColumnarValue;
use datafusion_physical_plan::metrics::SpillMetrics;
use geo::coord;
use geo_types::Rect;
use sedona_schema::datatypes::SedonaType;

use crate::{evaluated_batch::EvaluatedBatch, operand_evaluator::EvaluatedGeometryArray};

/// Writer for spilling evaluated batches to disk
pub(crate) struct SpillWriter {
    /// The temporary spill file being written to
    in_progress_file: RefCountedTempFile,
    /// Stream writer writes data to spill file in IPC format
    writer: StreamWriter<File>,
    /// The spill metrics to update
    metrics: SpillMetrics,

    /// Schema of the spilled record batches. It is augmented from the schema of original record batches
    /// The spill_schema has 4 fields:
    /// * `data`: StructArray containing the original record batch columns
    /// * `geom`: geometry array in storage format
    /// * `rect`: StructArray containing min_x, min_y, max_x, max_y fields
    /// * `dist`: distance field
    spill_schema: Schema,
    /// Inner fields of the "data" StructArray in the spilled record batches
    data_inner_fields: Fields,
    /// Inner fields of the "rect" StructArray in the spilled record batches
    rect_inner_fields: Fields,
}

impl SpillWriter {
    pub fn try_new(
        env: Arc<RuntimeEnv>,
        schema: SchemaRef,
        sedona_type: &SedonaType,
        request_description: &str,
        compression: SpillCompression,
        metrics: SpillMetrics,
    ) -> Result<Self> {
        // Construct schema of record batches to be written. The written batches is augmented from the original record batches.
        let data_inner_fields = schema.fields().clone();
        let data_struct_field =
            Field::new("data", DataType::Struct(data_inner_fields.clone()), false);
        let geom_field = sedona_type.to_storage_field("geom", true)?;
        let rect_inner_fields = Fields::from(vec![
            Field::new("min_x", DataType::Float32, false),
            Field::new("min_y", DataType::Float32, false),
            Field::new("max_x", DataType::Float32, false),
            Field::new("max_y", DataType::Float32, false),
        ]);
        let rect_field = Field::new("rect", DataType::Struct(rect_inner_fields.clone()), true);
        let dist_field = Field::new("dist", DataType::Float64, true);
        let spill_schema = Schema::new(vec![data_struct_field, geom_field, rect_field, dist_field]);

        // Create spill file
        let in_progress_file = env.disk_manager.create_tmp_file(request_description)?;
        let spill_file_path = in_progress_file.path();
        let file = File::create(spill_file_path)?;

        let mut write_options = IpcWriteOptions::default();
        write_options = write_options.try_with_compression(compression.into())?;
        let writer = StreamWriter::try_new_with_options(file, &spill_schema, write_options)?;
        metrics.spill_file_count.add(1);

        Ok(Self {
            in_progress_file,
            writer,
            metrics,

            spill_schema,
            data_inner_fields,
            rect_inner_fields,
        })
    }

    pub fn append(&mut self, evaluated_batch: &EvaluatedBatch) -> Result<()> {
        let num_rows = evaluated_batch.num_rows();
        let num_bytes = evaluated_batch.in_mem_size();
        let record_batch = self.spilled_record_batch(evaluated_batch)?;
        self.writer.write(&record_batch).map_err(|e| {
            DataFusionError::Execution(format!(
                "Failed to write RecordBatch to spill file {:?}: {}",
                self.in_progress_file.path(),
                e
            ))
        })?;
        self.metrics.spilled_rows.add(num_rows);
        self.metrics.spilled_bytes.add(num_bytes);
        Ok(())
    }

    pub fn finish(self) -> Result<RefCountedTempFile> {
        let mut in_progress_file = self.in_progress_file;
        in_progress_file.update_disk_usage()?;
        let size = in_progress_file.current_disk_usage();
        self.metrics.spilled_bytes.add(size as usize);
        Ok(in_progress_file)
    }

    fn spilled_record_batch(&self, evaluated_batch: &EvaluatedBatch) -> Result<RecordBatch> {
        let num_rows = evaluated_batch.num_rows();

        // Store the original data batch into a StructArray
        let data_batch = &evaluated_batch.batch;
        let data_arrays = data_batch.columns().to_vec();
        let data_struct_array =
            StructArray::try_new(self.data_inner_fields.clone(), data_arrays, None)?;

        // Store bbox into a StructArray
        let mut min_x_builder = Float32Builder::with_capacity(num_rows);
        let mut min_y_builder = Float32Builder::with_capacity(num_rows);
        let mut max_x_builder = Float32Builder::with_capacity(num_rows);
        let mut max_y_builder = Float32Builder::with_capacity(num_rows);
        let mut null_buffer_builder = NullBufferBuilder::new_with_len(num_rows);
        for rect_opt in evaluated_batch.rects() {
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
            self.rect_inner_fields.clone(),
            vec![
                Arc::new(min_x_array),
                Arc::new(min_y_array),
                Arc::new(max_x_array),
                Arc::new(max_y_array),
            ],
            null_buffer,
        )?;

        // Store dist into a Float64Array
        let mut dist_builder = arrow::array::Float64Builder::with_capacity(num_rows);
        let geom_array = &evaluated_batch.geom_array;
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

        // Assemble the final spilled RecordBatch
        let columns = vec![
            Arc::new(data_struct_array) as Arc<dyn arrow::array::Array>,
            Arc::clone(&geom_array.geometry_array),
            Arc::new(rect_array) as Arc<dyn arrow::array::Array>,
            Arc::new(dist_array) as Arc<dyn arrow::array::Array>,
        ];
        let spilled_record_batch =
            RecordBatch::try_new(Arc::new(self.spill_schema.clone()), columns)?;
        Ok(spilled_record_batch)
    }
}

pub(crate) struct SpillReader {
    stream_reader: StreamReader<BufReader<File>>,
}

impl SpillReader {
    pub fn try_new(temp_file: &RefCountedTempFile) -> Result<Self> {
        let file = File::open(temp_file.path())?;
        let mut stream_reader = StreamReader::try_new_buffered(file, None)?;
        unsafe {
            stream_reader = stream_reader.with_skip_validation(true);
        }
        Ok(Self { stream_reader })
    }

    pub fn next_batch(&mut self) -> Option<Result<EvaluatedBatch>> {
        let result_batch_opt = self.stream_reader.next();
        result_batch_opt.map(|result_batch| {
            result_batch
                .map_err(|e| e.into())
                .and_then(spilled_batch_to_build_side_batch)
        })
    }
}

fn spilled_batch_to_build_side_batch(record_batch: RecordBatch) -> Result<EvaluatedBatch> {
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

    let distance = if !dist_array.is_empty() {
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

    Ok(EvaluatedBatch { batch, geom_array })
}
