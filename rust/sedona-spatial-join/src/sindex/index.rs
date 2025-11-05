use geo_index::rtree::RTree;

pub(crate) mod spatial_index;
pub(crate) mod spatial_index_builder;
mod knn_adapter;

pub(crate) use spatial_index::SpatialIndex;
pub(crate) use spatial_index_builder::SpatialIndexBuilder;
use wkb::reader::Wkb;

// Type aliases for better readability
type SpatialRTree = RTree<f32>;
type DataIdToBatchPos = Vec<(i32, i32)>;
type RTreeBuildResult = (SpatialRTree, DataIdToBatchPos);

/// Rough estimate for in-memory size of the rtree per rect in bytes
const RTREE_MEMORY_ESTIMATE_PER_RECT: usize = 60;

/// The result of a spatial index query
pub(crate) struct IndexQueryResult<'a, 'b> {
    pub wkb: &'b Wkb<'a>,
    pub distance: Option<f64>,
    pub geom_idx: usize,
    pub position: (i32, i32),
}

/// The metrics for a spatial index query
#[derive(Debug)]
pub(crate) struct QueryResultMetrics {
    pub count: usize,
    pub candidate_count: usize,
}
