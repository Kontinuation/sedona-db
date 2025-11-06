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

use std::sync::Arc;

use arrow::array::{Float32Array, Float32Builder, Float64Array, NullBufferBuilder};
use arrow_array::{Array, RecordBatch, StructArray};
use arrow_schema::{DataType, Field, Fields, Schema};
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::ColumnarValue;
use geo::coord;
use geo_types::Rect;
use sedona_schema::datatypes::SedonaType;

use crate::{collect::build_side_batch::BuildSideBatch, operand_evaluator::EvaluatedGeometryArray};

pub(crate) fn build_side_batch_to_spilled_batch(
    build_side_batch: &BuildSideBatch,
) -> Result<RecordBatch> {
    let orig_schema = build_side_batch.batch.schema();
    let geom_array = &build_side_batch.geom_array;
    let sedona_type = &geom_array.sedona_type;
    let data_inner_fields = orig_schema.fields().clone();
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

    for rect_opt in build_side_batch.rects() {
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

#[allow(dead_code)]
pub(crate) fn spilled_batch_to_build_side_batch(
    record_batch: RecordBatch,
) -> Result<BuildSideBatch> {
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

    Ok(BuildSideBatch { batch, geom_array })
}
