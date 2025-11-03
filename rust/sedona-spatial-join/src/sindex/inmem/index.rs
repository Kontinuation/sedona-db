use once_cell::sync::OnceCell;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion_common::{utils::proxy::VecAllocExt, DataFusionError, Result};
use datafusion_common_runtime::JoinSet;
use datafusion_execution::{
    memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation},
    SendableRecordBatchStream,
};
use datafusion_expr::{ColumnarValue, JoinType};
use datafusion_physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use futures::StreamExt;
use geo_index::rtree::distance::{
    DistanceMetric, EuclideanDistance, GeometryAccessor, HaversineDistance,
};
use geo_index::rtree::{sort::HilbertSort, RTree, RTreeBuilder, RTreeIndex};
use geo_index::IndexableNum;
use geo_types::{Geometry, Point, Rect};
use parking_lot::Mutex;
use sedona_expr::statistics::GeoStatistics;
use sedona_functions::st_analyze_aggr::AnalyzeAccumulator;
use sedona_geo::to_geo::item_to_geometry;
use sedona_geo_generic_alg::algorithm::Centroid;
use sedona_schema::datatypes::WKB_GEOMETRY;
use wkb::reader::Wkb;

use crate::{
    concurrent_reservation::ConcurrentReservation,
    index::IndexQueryResult,
    operand_evaluator::{create_operand_evaluator, EvaluatedGeometryArray, OperandEvaluator},
    refine::{create_refiner, IndexQueryResultRefiner},
    sindex::{
        build_side_batch::BuildSideBatch,
        index::{JoinResultMetrics, SpatialIndex},
        utils::{KnnComponents, SedonaKnnAdapter},
    },
    spatial_predicate::SpatialPredicate,
    utils::need_produce_result_in_final,
};
use arrow::array::BooleanBufferBuilder;
use sedona_common::{option::SpatialJoinOptions, ExecutionMode};

pub(crate) struct InMemorySpatialIndex {
    schema: SchemaRef,

    /// The spatial predicate evaluator for the spatial predicate.
    evaluator: Arc<dyn OperandEvaluator>,

    /// The refiner for refining the index query results.
    refiner: Arc<dyn IndexQueryResultRefiner>,

    /// R-tree index for the geometry batches. It takes MBRs as query windows and returns
    /// data indexes. These data indexes should be translated using `data_id_to_batch_pos` to get
    /// the original geometry batch index and row index, or translated using `prepared_geom_idx_vec`
    /// to get the prepared geometries array index.
    rtree: RTree<f32>,

    /// Indexed batches containing evaluated geometry arrays. It contains the original record
    /// batches and geometry arrays obtained by evaluating the geometry expression on the build side.
    indexed_batches: Vec<BuildSideBatch>,
    /// An array for translating rtree data index to geometry batch index and row index
    data_id_to_batch_pos: Vec<(i32, i32)>,

    /// An array for translating rtree data index to consecutive index. Each geometry may be indexed by
    /// multiple boxes, so there could be multiple data indexes for the same geometry. A mapping for
    /// squashing the index makes it easier for persisting per-geometry auxiliary data for evaluating
    /// the spatial predicate. This is extensively used by the spatial predicate evaluators for storing
    /// prepared geometries.
    geom_idx_vec: Vec<usize>,

    /// Shared bitmap builders for visited left indices, one per batch
    visited_left_side: Option<Mutex<Vec<BooleanBufferBuilder>>>,

    /// Counter of running probe-threads, potentially able to update `bitmap`.
    /// Each time a probe thread finished probing the index, it will decrement the counter.
    /// The last finished probe thread will produce the extra output batches for unmatched
    /// build side when running left-outer joins. See also [`report_probe_completed`].
    probe_threads_counter: AtomicUsize,

    /// Shared KNN components (distance metrics and geometry cache) for efficient KNN queries
    knn_components: KnnComponents,

    /// Memory reservation for tracking the memory usage of the spatial index
    /// Cleared on `InMemorySpatialIndex` drop
    #[expect(dead_code)]
    reservation: MemoryReservation,
}

impl InMemorySpatialIndex {
    pub fn empty(
        spatial_predicate: SpatialPredicate,
        schema: SchemaRef,
        options: SpatialJoinOptions,
        probe_threads_counter: AtomicUsize,
        reservation: MemoryReservation,
        memory_pool: Arc<dyn MemoryPool>,
    ) -> Self {
        let evaluator = create_operand_evaluator(&spatial_predicate, options.clone());
        let refiner = create_refiner(
            options.spatial_library,
            &spatial_predicate,
            options.clone(),
            0,
            GeoStatistics::empty(),
        );
        let rtree = RTreeBuilder::<f32>::new(0).finish::<HilbertSort>();
        Self {
            schema,
            evaluator,
            refiner,
            rtree,
            data_id_to_batch_pos: Vec::new(),
            indexed_batches: Vec::new(),
            geom_idx_vec: Vec::new(),
            visited_left_side: None,
            probe_threads_counter,
            knn_components: KnnComponents::new(0, &[], memory_pool.clone()).unwrap(), // Empty index has no cache
            reservation,
        }
    }

    pub fn new(
        schema: SchemaRef,
        evaluator: Arc<dyn OperandEvaluator>,
        refiner: Arc<dyn IndexQueryResultRefiner>,
        rtree: RTree<f32>,
        data_id_to_batch_pos: Vec<(i32, i32)>,
        indexed_batches: Vec<BuildSideBatch>,
        geom_idx_vec: Vec<usize>,
        visited_left_side: Option<Mutex<Vec<BooleanBufferBuilder>>>,
        probe_threads_counter: AtomicUsize,
        knn_components: KnnComponents,
        reservation: MemoryReservation,
    ) -> Self {
        Self {
            schema,
            evaluator,
            refiner,
            rtree,
            data_id_to_batch_pos,
            indexed_batches,
            geom_idx_vec,
            visited_left_side,
            probe_threads_counter,
            knn_components,
            reservation,
        }
    }

    pub(crate) fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Create a KNN geometry accessor for accessing geometries with caching
    fn create_knn_accessor(&self) -> SedonaKnnAdapter<'_> {
        SedonaKnnAdapter::new(
            &self.indexed_batches,
            &self.data_id_to_batch_pos,
            &self.knn_components,
        )
    }

    /// Get the batch at the given index.
    pub(crate) fn get_indexed_batch(&self, batch_idx: usize) -> &RecordBatch {
        &self.indexed_batches[batch_idx].batch
    }

    /// Query the spatial index with a probe geometry to find matching build-side geometries.
    ///
    /// This method implements a two-phase spatial join query:
    /// 1. **Filter phase**: Uses the R-tree index with the probe geometry's bounding rectangle
    ///    to quickly identify candidate geometries that might satisfy the spatial predicate
    /// 2. **Refinement phase**: Evaluates the exact spatial predicate on candidates to determine
    ///    actual matches
    ///
    /// # Arguments
    /// * `probe_wkb` - The probe geometry in WKB format
    /// * `probe_rect` - The minimum bounding rectangle of the probe geometry
    /// * `distance` - Optional distance parameter for distance-based spatial predicates
    /// * `build_batch_positions` - Output vector that will be populated with (batch_idx, row_idx)
    ///   pairs for each matching build-side geometry
    ///
    /// # Returns
    /// * `JoinResultMetrics` containing the number of actual matches (`count`) and the number
    ///   of candidates from the filter phase (`candidate_count`)
    pub(crate) fn query(
        &self,
        probe_wkb: &Wkb,
        probe_rect: &Rect<f32>,
        distance: &Option<f64>,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<JoinResultMetrics> {
        let min = probe_rect.min();
        let max = probe_rect.max();
        let mut candidates = self.rtree.search(min.x, min.y, max.x, max.y);
        if candidates.is_empty() {
            return Ok(JoinResultMetrics {
                count: 0,
                candidate_count: 0,
            });
        }

        // Sort and dedup candidates to avoid duplicate results when we index one geometry
        // using several boxes.
        candidates.sort_unstable();
        candidates.dedup();

        // Refine the candidates retrieved from the r-tree index by evaluating the actual spatial predicate
        self.refine(probe_wkb, &candidates, distance, build_batch_positions)
    }

    /// Query the spatial index for k nearest neighbors of a given geometry.
    ///
    /// This method finds the k nearest neighbors to the probe geometry using:
    /// 1. R-tree's built-in neighbors() method for efficient KNN search
    /// 2. Distance refinement using actual geometry calculations
    /// 3. Tie-breaker handling when enabled
    ///
    /// # Arguments
    ///
    /// * `probe_wkb` - WKB representation of the probe geometry
    /// * `k` - Number of nearest neighbors to find
    /// * `use_spheroid` - Whether to use spheroid distance calculation
    /// * `include_tie_breakers` - Whether to include additional results with same distance as kth neighbor
    /// * `build_batch_positions` - Output vector for matched positions
    ///
    /// # Returns
    ///
    /// * `JoinResultMetrics` containing the number of actual matches and candidates processed
    pub(crate) fn query_knn(
        &self,
        probe_wkb: &Wkb,
        k: u32,
        use_spheroid: bool,
        include_tie_breakers: bool,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<JoinResultMetrics> {
        if k == 0 {
            return Ok(JoinResultMetrics {
                count: 0,
                candidate_count: 0,
            });
        }

        // Check if index is empty
        if self.indexed_batches.is_empty() || self.data_id_to_batch_pos.is_empty() {
            return Ok(JoinResultMetrics {
                count: 0,
                candidate_count: 0,
            });
        }

        // Convert probe WKB to geo::Geometry
        let probe_geom = match item_to_geometry(probe_wkb) {
            Ok(geom) => geom,
            Err(_) => {
                // Empty or unsupported geometries (e.g., POINT EMPTY) return empty results
                return Ok(JoinResultMetrics {
                    count: 0,
                    candidate_count: 0,
                });
            }
        };

        // Select the appropriate distance metric
        let distance_metric: &dyn DistanceMetric<f32> = if use_spheroid {
            &self.knn_components.haversine_metric
        } else {
            &self.knn_components.euclidean_metric
        };

        // Create geometry accessor for on-demand WKB decoding and caching
        let geometry_accessor = self.create_knn_accessor();

        // Use neighbors_geometry to find k nearest neighbors
        let initial_results = self.rtree.neighbors_geometry(
            &probe_geom,
            Some(k as usize),
            None, // no max_distance filter
            distance_metric,
            &geometry_accessor,
        );

        if initial_results.is_empty() {
            return Ok(JoinResultMetrics {
                count: 0,
                candidate_count: 0,
            });
        }

        let mut final_results = initial_results;
        let mut candidate_count = final_results.len();

        // Handle tie-breakers if enabled
        if include_tie_breakers && !final_results.is_empty() && k > 0 {
            // Calculate distances for the initial k results to find the k-th distance
            let mut distances_with_indices: Vec<(f64, u32)> = Vec::new();

            for &result_idx in &final_results {
                if (result_idx as usize) < self.data_id_to_batch_pos.len() {
                    if let Some(item_geom) = geometry_accessor.get_geometry(result_idx as usize) {
                        let distance = distance_metric.distance_to_geometry(&probe_geom, item_geom);
                        if let Some(distance_f64) = distance.to_f64() {
                            distances_with_indices.push((distance_f64, result_idx));
                        }
                    }
                }
            }

            // Sort by distance
            distances_with_indices
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Find the k-th distance (if we have at least k results)
            if distances_with_indices.len() >= k as usize {
                let k_idx = (k as usize)
                    .min(distances_with_indices.len())
                    .saturating_sub(1);
                let max_distance = distances_with_indices[k_idx].0;

                // For tie-breakers, create spatial envelope around probe centroid and use rtree.search()

                let probe_centroid = probe_geom.centroid().unwrap_or(Point::new(0.0, 0.0));
                let probe_x = probe_centroid.x() as f32;
                let probe_y = probe_centroid.y() as f32;
                let max_distance_f32 = match f32::from_f64(max_distance) {
                    Some(val) => val,
                    None => {
                        // If conversion fails, return empty results for this probe
                        return Ok(JoinResultMetrics {
                            count: 0,
                            candidate_count: 0,
                        });
                    }
                };

                // Create envelope bounds around probe centroid
                let min_x = probe_x - max_distance_f32;
                let min_y = probe_y - max_distance_f32;
                let max_x = probe_x + max_distance_f32;
                let max_y = probe_y + max_distance_f32;

                // Use rtree.search() with envelope bounds (like the old code)
                let expanded_results = self.rtree.search(min_x, min_y, max_x, max_y);

                candidate_count = expanded_results.len();

                // Calculate distances for all results and find ties
                let mut all_distances_with_indices: Vec<(f64, u32)> = Vec::new();

                for &result_idx in &expanded_results {
                    if (result_idx as usize) < self.data_id_to_batch_pos.len() {
                        if let Some(item_geom) = geometry_accessor.get_geometry(result_idx as usize)
                        {
                            let distance =
                                distance_metric.distance_to_geometry(&probe_geom, item_geom);
                            if let Some(distance_f64) = distance.to_f64() {
                                all_distances_with_indices.push((distance_f64, result_idx));
                            }
                        }
                    }
                }

                // Sort by distance
                all_distances_with_indices
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // Include all results up to and including those with the same distance as the k-th result
                const DISTANCE_TOLERANCE: f64 = 1e-9;
                let mut tie_breaker_results: Vec<u32> = Vec::new();

                for (i, &(distance, result_idx)) in all_distances_with_indices.iter().enumerate() {
                    if i < k as usize {
                        // Include the first k results
                        tie_breaker_results.push(result_idx);
                    } else if (distance - max_distance).abs() <= DISTANCE_TOLERANCE {
                        // Include tie-breakers (same distance as k-th result)
                        tie_breaker_results.push(result_idx);
                    } else {
                        // No more ties, stop
                        break;
                    }
                }

                final_results = tie_breaker_results;
            }
        } else {
            // When tie-breakers are disabled, limit results to exactly k
            if final_results.len() > k as usize {
                final_results.truncate(k as usize);
            }
        }

        // Convert results to build_batch_positions using existing data_id_to_batch_pos mapping
        for &result_idx in &final_results {
            if (result_idx as usize) < self.data_id_to_batch_pos.len() {
                build_batch_positions.push(self.data_id_to_batch_pos[result_idx as usize]);
            }
        }

        Ok(JoinResultMetrics {
            count: final_results.len(),
            candidate_count,
        })
    }

    fn refine(
        &self,
        probe_wkb: &Wkb,
        candidates: &[u32],
        distance: &Option<f64>,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<JoinResultMetrics> {
        let candidate_count = candidates.len();

        let mut index_query_results = Vec::with_capacity(candidate_count);
        for data_idx in candidates {
            let pos = self.data_id_to_batch_pos[*data_idx as usize];
            let (batch_idx, row_idx) = pos;
            let indexed_batch = &self.indexed_batches[batch_idx as usize];
            let build_wkb = indexed_batch.wkb(row_idx as usize);
            let Some(build_wkb) = build_wkb else {
                continue;
            };
            let distance = self.evaluator.resolve_distance(
                indexed_batch.distance(),
                row_idx as usize,
                distance,
            )?;
            let geom_idx = self.geom_idx_vec[*data_idx as usize];
            index_query_results.push(IndexQueryResult {
                wkb: build_wkb,
                distance,
                geom_idx,
                position: pos,
            });
        }

        if index_query_results.is_empty() {
            return Ok(JoinResultMetrics {
                count: 0,
                candidate_count,
            });
        }

        let results = self.refiner.refine(probe_wkb, &index_query_results)?;
        let num_results = results.len();
        build_batch_positions.extend(results);

        Ok(JoinResultMetrics {
            count: num_results,
            candidate_count,
        })
    }

    /// Check if the index needs more probe statistics to determine the optimal execution mode.
    ///
    /// # Returns
    /// * `bool` - `true` if the index needs more probe statistics, `false` otherwise.
    pub(crate) fn need_more_probe_stats(&self) -> bool {
        self.refiner.need_more_probe_stats()
    }

    /// Merge the probe statistics into the index.
    ///
    /// # Arguments
    /// * `stats` - The probe statistics to merge.
    pub(crate) fn merge_probe_stats(&self, stats: GeoStatistics) {
        self.refiner.merge_probe_stats(stats);
    }

    /// Get the bitmaps for tracking visited left-side indices. The bitmaps will be updated
    /// by the spatial join stream when producing output batches during index probing phase.
    pub(crate) fn visited_left_side(&self) -> Option<&Mutex<Vec<BooleanBufferBuilder>>> {
        self.visited_left_side.as_ref()
    }

    /// Decrements counter of running threads, and returns `true`
    /// if caller is the last running thread
    pub(crate) fn report_probe_completed(&self) -> bool {
        self.probe_threads_counter.fetch_sub(1, Ordering::Relaxed) == 1
    }

    /// Get the memory usage of the refiner in bytes.
    pub(crate) fn get_refiner_mem_usage(&self) -> usize {
        self.refiner.mem_usage()
    }

    /// Get the actual execution mode used by the refiner
    pub(crate) fn get_actual_execution_mode(&self) -> ExecutionMode {
        self.refiner.actual_execution_mode()
    }
}

impl SpatialIndex for InMemorySpatialIndex {
    fn get_indexed_batch(&self, batch_idx: usize) -> &RecordBatch {
        self.get_indexed_batch(batch_idx)
    }

    fn query(
        &self,
        probe_wkb: &Wkb,
        probe_rect: &Rect<f32>,
        distance: &Option<f64>,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<JoinResultMetrics> {
        self.query(probe_wkb, probe_rect, distance, build_batch_positions)
    }

    fn query_knn(
        &self,
        probe_wkb: &Wkb,
        k: u32,
        use_spheroid: bool,
        include_tie_breakers: bool,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<JoinResultMetrics> {
        self.query_knn(
            probe_wkb,
            k,
            use_spheroid,
            include_tie_breakers,
            build_batch_positions,
        )
    }
}
