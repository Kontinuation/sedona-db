use arrow_array::RecordBatch;
use geo::Rect;
use wkb::reader::Wkb;

use datafusion_common::Result;

pub trait SpatialIndex {
    /// Get the batch at the given index.
    fn get_indexed_batch(&self, batch_idx: usize) -> &RecordBatch;
    
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
    fn query(
        &self,
        probe_wkb: &Wkb,
        probe_rect: &Rect<f32>,
        distance: &Option<f64>,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<JoinResultMetrics>;

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
    fn query_knn(
        &self,
        probe_wkb: &Wkb,
        k: u32,
        use_spheroid: bool,
        include_tie_breakers: bool,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<JoinResultMetrics>;
}


#[derive(Debug)]
pub struct JoinResultMetrics {
    pub count: usize,
    pub candidate_count: usize,
}
