use arrow_array::RecordBatch;
use geo::Rect;
use geo_index::rtree::RTree;
use wkb::reader::Wkb;

use datafusion_common::Result;

// Type aliases for better readability
type SpatialRTree = RTree<f32>;
type DataIdToBatchPos = Vec<(i32, i32)>;
type RTreeBuildResult = (SpatialRTree, DataIdToBatchPos);

/// Rough estimate for in-memory size of the rtree per rect in bytes
const RTREE_MEMORY_ESTIMATE_PER_RECT: usize = 60;

#[derive(Debug)]
pub struct JoinResultMetrics {
    pub count: usize,
    pub candidate_count: usize,
}

pub mod spatial_index;
pub mod spatial_index_builder;
