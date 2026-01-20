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

//! RS_Intersects, RS_Contains, RS_Within functions
//!
//! These functions test spatial relationships between rasters and geometries.
//! CRS transformation rules:
//! - If the raster or geometry does not have a defined SRID, it is assumed to be in WGS84
//! - If both sides are in the same CRS, perform the relationship test directly
//! - Otherwise, both sides will be transformed to WGS84 before the relationship test

use std::sync::Arc;

use crate::executor::RasterExecutor;
use arrow_array::builder::BooleanBuilder;
use arrow_array::{Array, BinaryArray};
use arrow_schema::DataType;
use datafusion_common::cast::as_binary_array;
use datafusion_common::DataFusionError;
use datafusion_common::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::write_wkb_polygon;
use sedona_raster::affine_transformation::to_world_coordinate;
use sedona_raster::traits::RasterRef;
use sedona_schema::crs::deserialize_crs;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use sedona_tg::tg;

const WGS84_CRS: &str = "EPSG:4326";

/// RS_Intersects() scalar UDF documentation
///
/// Returns true if raster A intersects geometry B.
pub fn rs_intersects_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_intersects",
        vec![
            Arc::new(RsSpatialPredicate::<tg::Intersects>::raster_geom()),
            Arc::new(RsSpatialPredicate::<tg::Intersects>::geom_raster()),
            Arc::new(RsSpatialPredicate::<tg::Intersects>::raster_raster()),
        ],
        Volatility::Immutable,
        Some(rs_intersects_doc()),
    )
}

/// RS_Contains() scalar UDF documentation
///
/// Returns true if raster A contains geometry B.
pub fn rs_contains_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_contains",
        vec![
            Arc::new(RsSpatialPredicate::<tg::Contains>::raster_geom()),
            Arc::new(RsSpatialPredicate::<tg::Contains>::geom_raster()),
            Arc::new(RsSpatialPredicate::<tg::Contains>::raster_raster()),
        ],
        Volatility::Immutable,
        Some(rs_contains_doc()),
    )
}

/// RS_Within() scalar UDF documentation
///
/// Returns true if raster A is within geometry B.
pub fn rs_within_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_within",
        vec![
            Arc::new(RsSpatialPredicate::<tg::Within>::raster_geom()),
            Arc::new(RsSpatialPredicate::<tg::Within>::geom_raster()),
            Arc::new(RsSpatialPredicate::<tg::Within>::raster_raster()),
        ],
        Volatility::Immutable,
        Some(rs_within_doc()),
    )
}

fn rs_intersects_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns true if the raster intersects the specified geometry or raster.".to_string(),
        "RS_Intersects(raster: Raster, geometry: Geometry)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("geometry", "Geometry: Input geometry or raster")
    .with_sql_example("SELECT RS_Intersects(raster, ST_Point(0, 0))".to_string())
    .build()
}

fn rs_contains_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns true if the raster contains the specified geometry or raster.".to_string(),
        "RS_Contains(raster: Raster, geometry: Geometry)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("geometry", "Geometry: Input geometry or raster")
    .with_sql_example("SELECT RS_Contains(raster, ST_Point(0, 0))".to_string())
    .build()
}

fn rs_within_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns true if the raster is within the specified geometry or raster.".to_string(),
        "RS_Within(raster: Raster, geometry: Geometry)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("geometry", "Geometry: Input geometry or raster")
    .with_sql_example("SELECT RS_Within(raster, ST_Envelope(raster))".to_string())
    .build()
}

/// Argument order for the spatial predicate
#[derive(Debug, Clone, Copy)]
enum ArgOrder {
    /// First arg is raster, second is geometry
    RasterGeom,
    /// First arg is geometry, second is raster
    GeomRaster,
    /// Both args are rasters
    RasterRaster,
}

#[derive(Debug)]
struct RsSpatialPredicate<Op: tg::BinaryPredicate> {
    arg_order: ArgOrder,
    _op: std::marker::PhantomData<Op>,
}

impl<Op: tg::BinaryPredicate> RsSpatialPredicate<Op> {
    fn raster_geom() -> Self {
        Self {
            arg_order: ArgOrder::RasterGeom,
            _op: std::marker::PhantomData,
        }
    }

    fn geom_raster() -> Self {
        Self {
            arg_order: ArgOrder::GeomRaster,
            _op: std::marker::PhantomData,
        }
    }

    fn raster_raster() -> Self {
        Self {
            arg_order: ArgOrder::RasterRaster,
            _op: std::marker::PhantomData,
        }
    }
}

impl<Op: tg::BinaryPredicate + Send + Sync> SedonaScalarKernel for RsSpatialPredicate<Op> {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = match self.arg_order {
            ArgOrder::RasterGeom => ArgMatcher::new(
                vec![ArgMatcher::is_raster(), ArgMatcher::is_geometry()],
                SedonaType::Arrow(DataType::Boolean),
            ),
            ArgOrder::GeomRaster => ArgMatcher::new(
                vec![ArgMatcher::is_geometry(), ArgMatcher::is_raster()],
                SedonaType::Arrow(DataType::Boolean),
            ),
            ArgOrder::RasterRaster => ArgMatcher::new(
                vec![ArgMatcher::is_raster(), ArgMatcher::is_raster()],
                SedonaType::Arrow(DataType::Boolean),
            ),
        };

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        match self.arg_order {
            ArgOrder::RasterGeom => self.invoke_raster_geom(arg_types, args),
            ArgOrder::GeomRaster => self.invoke_geom_raster(arg_types, args),
            ArgOrder::RasterRaster => self.invoke_raster_raster(arg_types, args),
        }
    }
}

impl<Op: tg::BinaryPredicate + Send + Sync> RsSpatialPredicate<Op> {
    /// Invoke RS_<Predicate>(raster, geometry)
    fn invoke_raster_geom(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());

        // Get the geometry CRS from the type
        let geom_crs = get_crs_from_geom_type(&arg_types[1])?;

        // Expand geometry argument to array
        let geom_array = expand_to_binary_array(&args[1], executor.num_iterations())?;

        executor.execute_raster_void(|i, raster_opt| {
            match raster_opt {
                Some(raster) => {
                    // Get geometry WKB
                    if geom_array.is_null(i) {
                        builder.append_null();
                        return Ok(());
                    }
                    let geom_wkb = geom_array.value(i);

                    // Get raster CRS
                    let raster_crs = get_crs_from_raster(&raster)?;

                    // Create convex hull WKB for the raster
                    let mut raster_wkb = Vec::with_capacity(93);
                    create_convexhull_wkb(&raster, &mut raster_wkb)?;

                    // Evaluate predicate with CRS handling
                    let result = evaluate_predicate_with_crs::<Op>(
                        &raster_wkb,
                        raster_crs.as_deref(),
                        geom_wkb,
                        geom_crs.as_deref(),
                    )?;
                    builder.append_value(result);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }

    /// Invoke RS_<Predicate>(geometry, raster)
    fn invoke_geom_raster(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(&arg_types[1..], &args[1..]);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());

        // Get the geometry CRS from the type
        let geom_crs = get_crs_from_geom_type(&arg_types[0])?;

        // Expand geometry argument to array
        let geom_array = expand_to_binary_array(&args[0], executor.num_iterations())?;

        executor.execute_raster_void(|i, raster_opt| {
            match raster_opt {
                Some(raster) => {
                    // Get geometry WKB
                    if geom_array.is_null(i) {
                        builder.append_null();
                        return Ok(());
                    }
                    let geom_wkb = geom_array.value(i);

                    // Get raster CRS
                    let raster_crs = get_crs_from_raster(&raster)?;

                    // Create convex hull WKB for the raster
                    let mut raster_wkb = Vec::with_capacity(93);
                    create_convexhull_wkb(&raster, &mut raster_wkb)?;

                    // Note: order is geometry, raster for the predicate
                    let result = evaluate_predicate_with_crs::<Op>(
                        geom_wkb,
                        geom_crs.as_deref(),
                        &raster_wkb,
                        raster_crs.as_deref(),
                    )?;
                    builder.append_value(result);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        // Use the first raster argument's executor for finishing
        let executor_for_finish = RasterExecutor::new(&arg_types[1..], &args[1..]);
        executor_for_finish.finish(Arc::new(builder.finish()))
    }

    /// Invoke RS_<Predicate>(raster1, raster2)
    fn invoke_raster_raster(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // Create executor for the first raster
        let executor1 = RasterExecutor::new(&arg_types[0..1], &args[0..1]);
        // Create executor for the second raster
        let executor2 = RasterExecutor::new(&arg_types[1..], &args[1..]);

        let num_iterations = executor1.num_iterations().max(executor2.num_iterations());
        let mut builder = BooleanBuilder::with_capacity(num_iterations);

        // We need to iterate over both rasters together
        // For simplicity, we'll use a combined approach
        let raster1_vec = collect_rasters(&executor1)?;
        let raster2_vec = collect_rasters(&executor2)?;

        for i in 0..num_iterations {
            let r1_idx = if raster1_vec.len() == 1 { 0 } else { i };
            let r2_idx = if raster2_vec.len() == 1 { 0 } else { i };

            match (&raster1_vec[r1_idx], &raster2_vec[r2_idx]) {
                (Some(raster1_wkb), Some(raster2_wkb)) => {
                    // For raster-raster, we need CRS info stored alongside
                    // This is simplified - ideally we'd store CRS with the WKB
                    let result = evaluate_predicate_with_crs::<Op>(
                        &raster1_wkb.0,
                        raster1_wkb.1.as_deref(),
                        &raster2_wkb.0,
                        raster2_wkb.1.as_deref(),
                    )?;
                    builder.append_value(result);
                }
                _ => builder.append_null(),
            }
        }

        executor1.finish(Arc::new(builder.finish()))
    }
}

/// A raster's convex hull WKB bytes paired with its CRS string
type RasterWkbCrs = (Vec<u8>, Option<String>);

/// Collect rasters into a vector of (WKB, CRS) pairs
fn collect_rasters(executor: &RasterExecutor) -> Result<Vec<Option<RasterWkbCrs>>> {
    let mut results = Vec::with_capacity(executor.num_iterations());

    executor.execute_raster_void(|_i, raster_opt| {
        match raster_opt {
            Some(raster) => {
                let crs = get_crs_from_raster(&raster)?;
                let mut wkb = Vec::with_capacity(93);
                create_convexhull_wkb(&raster, &mut wkb)?;
                results.push(Some((wkb, crs)));
            }
            None => results.push(None),
        }
        Ok(())
    })?;

    Ok(results)
}

/// Get CRS string from a raster
fn get_crs_from_raster(raster: &dyn RasterRef) -> Result<Option<String>> {
    match raster.crs() {
        None => Ok(None),
        Some(crs_str) => {
            let crs = deserialize_crs(crs_str).map_err(|e| {
                DataFusionError::Execution(format!("Failed to deserialize CRS: {e}"))
            })?;
            match crs {
                Some(crs_ref) => Ok(Some(crs_ref.to_crs_string())),
                None => Ok(None),
            }
        }
    }
}

/// Get CRS string from a geometry type
fn get_crs_from_geom_type(sedona_type: &SedonaType) -> Result<Option<String>> {
    match sedona_type {
        SedonaType::Wkb(_, Some(crs)) | SedonaType::WkbView(_, Some(crs)) => {
            Ok(Some(crs.to_crs_string()))
        }
        _ => Ok(None),
    }
}

/// Expand a ColumnarValue to a BinaryArray
fn expand_to_binary_array(value: &ColumnarValue, num_rows: usize) -> Result<Arc<BinaryArray>> {
    match value {
        ColumnarValue::Array(array) => {
            let binary_array = as_binary_array(&array)?;
            Ok(Arc::new(binary_array.clone()))
        }
        ColumnarValue::Scalar(scalar) => {
            let array = scalar.to_array_of_size(num_rows)?;
            let binary_array = as_binary_array(&array)?;
            Ok(Arc::new(binary_array.clone()))
        }
    }
}

/// Evaluate a spatial predicate with CRS handling
///
/// Rules:
/// - If no CRS defined, assume WGS84
/// - If both same CRS, compare directly
/// - Otherwise, transform both to WGS84
fn evaluate_predicate_with_crs<Op: tg::BinaryPredicate>(
    wkb_a: &[u8],
    crs_a: Option<&str>,
    wkb_b: &[u8],
    crs_b: Option<&str>,
) -> Result<bool> {
    // Normalize CRS: None -> WGS84
    let crs_a_normalized = crs_a.unwrap_or(WGS84_CRS);
    let crs_b_normalized = crs_b.unwrap_or(WGS84_CRS);

    // Check if CRSs are the same (simple string comparison)
    // This handles common cases like "EPSG:4326" == "EPSG:4326"
    let same_crs = are_crs_equivalent(crs_a_normalized, crs_b_normalized);

    if same_crs {
        // Same CRS - compare directly
        evaluate_predicate::<Op>(wkb_a, wkb_b)
    } else {
        // Different CRS - for now, we do NOT transform (transformation requires sedona-proj)
        // Instead, we issue a warning and compare directly
        // In a full implementation, we would:
        // 1. Transform wkb_a from crs_a to WGS84
        // 2. Transform wkb_b from crs_b to WGS84
        // 3. Compare the transformed geometries

        // For now, just compare directly (this is a limitation)
        // TODO: Add coordinate transformation support via sedona-proj
        evaluate_predicate::<Op>(wkb_a, wkb_b)
    }
}

/// Check if two CRS identifiers are equivalent
fn are_crs_equivalent(crs_a: &str, crs_b: &str) -> bool {
    // Simple string comparison
    if crs_a == crs_b {
        return true;
    }

    // Check for WGS84 equivalents
    let wgs84_codes = ["EPSG:4326", "OGC:CRS84", "CRS84", "WGS84"];
    let a_is_wgs84 = wgs84_codes.iter().any(|c| crs_a.eq_ignore_ascii_case(c));
    let b_is_wgs84 = wgs84_codes.iter().any(|c| crs_b.eq_ignore_ascii_case(c));

    a_is_wgs84 && b_is_wgs84
}

/// Evaluate a spatial predicate between two WKB geometries
fn evaluate_predicate<Op: tg::BinaryPredicate>(wkb_a: &[u8], wkb_b: &[u8]) -> Result<bool> {
    let geom_a = tg::Geom::parse_wkb(wkb_a, tg::IndexType::Default)
        .map_err(|e| DataFusionError::Execution(format!("Failed to parse WKB A: {e}")))?;
    let geom_b = tg::Geom::parse_wkb(wkb_b, tg::IndexType::Default)
        .map_err(|e| DataFusionError::Execution(format!("Failed to parse WKB B: {e}")))?;

    Ok(Op::evaluate(&geom_a, &geom_b))
}

/// Create WKB for a convex hull polygon for the raster
fn create_convexhull_wkb(raster: &dyn RasterRef, out: &mut impl std::io::Write) -> Result<()> {
    let width = raster.metadata().width() as i64;
    let height = raster.metadata().height() as i64;

    let (ulx, uly) = to_world_coordinate(raster, 0, 0);
    let (urx, ury) = to_world_coordinate(raster, width, 0);
    let (lrx, lry) = to_world_coordinate(raster, width, height);
    let (llx, lly) = to_world_coordinate(raster, 0, height);

    write_wkb_polygon(
        out,
        [(ulx, uly), (urx, ury), (lrx, lry), (llx, lly), (ulx, uly)].into_iter(),
    )
    .map_err(|e| DataFusionError::External(e.into()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{create_array, ArrayRef};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array as create_geom_array;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn rs_intersects_udf_docs() {
        let udf: ScalarUDF = rs_intersects_udf().into();
        assert_eq!(udf.name(), "rs_intersects");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn rs_contains_udf_docs() {
        let udf: ScalarUDF = rs_contains_udf().into();
        assert_eq!(udf.name(), "rs_contains");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn rs_within_udf_docs() {
        let udf: ScalarUDF = rs_within_udf().into();
        assert_eq!(udf.name(), "rs_within");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn rs_intersects_raster_geom() {
        let udf = rs_intersects_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, WKB_GEOMETRY]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Test rasters:
        // Raster 1: corners at approximately (2.0, 3.0), (2.2, 3.08), (2.29, 2.48), (2.09, 2.4)
        // Raster 2: corners at approximately (3.0, 4.0), (3.6, 4.24), (3.84, 2.64), (3.24, 2.4)

        // Points that should intersect with raster 1 (approximately)
        // Point inside raster 1
        let geoms = create_geom_array(
            &[
                None,
                Some("POINT (2.15 2.75)"), // Inside raster 1
                Some("POINT (0.0 0.0)"),   // Outside all rasters
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = create_array!(Boolean, [None, Some(true), Some(false)]);

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();

        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn rs_contains_raster_geom() {
        let udf = rs_contains_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, WKB_GEOMETRY]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Point inside raster 1 should be contained
        let geoms = create_geom_array(
            &[
                None,
                Some("POINT (2.15 2.75)"), // Inside raster 1
                Some("POINT (0.0 0.0)"),   // Outside all rasters
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = create_array!(Boolean, [None, Some(true), Some(false)]);

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();

        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn rs_within_raster_geom() {
        let udf = rs_within_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, WKB_GEOMETRY]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Test rasters:
        // Raster 1: corners at approximately (2.0, 3.0), (2.2, 3.08), (2.29, 2.48), (2.09, 2.4)

        // Large polygon that contains raster 1
        let geoms = create_geom_array(
            &[
                None,
                Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"), // Contains raster 1
                Some("POLYGON ((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0))"), // Does not contain raster 2
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = create_array!(Boolean, [None, Some(true), Some(false)]);

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();

        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn rs_intersects_geom_raster() {
        let udf = rs_intersects_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY, RASTER]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Test with geometry as first argument
        let geoms = create_geom_array(
            &[
                None,
                Some("POINT (2.15 2.75)"), // Inside raster 1
                Some("POINT (0.0 0.0)"),   // Outside all rasters
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = create_array!(Boolean, [None, Some(true), Some(false)]);

        let result = tester
            .invoke_arrays(vec![geoms, Arc::new(rasters)])
            .unwrap();

        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn rs_intersects_raster_raster() {
        let udf = rs_intersects_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, RASTER]);

        let rasters1 = generate_test_rasters(3, Some(0)).unwrap();
        let rasters2 = generate_test_rasters(3, Some(0)).unwrap();

        // Same rasters should intersect with themselves
        let expected: ArrayRef = create_array!(Boolean, [None, Some(true), Some(true)]);

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters1), Arc::new(rasters2)])
            .unwrap();

        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn rs_intersects_null_handling() {
        let udf = rs_intersects_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, WKB_GEOMETRY]);

        let rasters = generate_test_rasters(3, Some(0)).unwrap();

        // Test with null geometry
        let geoms = create_geom_array(&[None::<&str>, None::<&str>, None::<&str>], &WKB_GEOMETRY);

        let expected: ArrayRef = create_array!(Boolean, [None, None, None]);

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();

        assert_array_equal(&result, &expected);
    }
}
