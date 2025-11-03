use geo_index::rtree::RTree;

// Type aliases for better readability
type SpatialRTree = RTree<f32>;
type DataIdToBatchPos = Vec<(i32, i32)>;
type RTreeBuildResult = (SpatialRTree, DataIdToBatchPos);

/// Rough estimate for in-memory size of the rtree per rect in bytes
const RTREE_MEMORY_ESTIMATE_PER_RECT: usize = 60;

pub(crate) mod index;
pub(crate) mod index_builder;
