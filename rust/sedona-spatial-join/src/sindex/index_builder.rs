use std::sync::Arc;

use arrow_schema::SchemaRef;
use sedona_expr::statistics::GeoStatistics;
use datafusion_common::Result;

use crate::sindex::{build_side_batch::{BuildSideBatch, SendableBuildSideBatchStream}, index::SpatialIndex};

pub(crate) trait SpatialIndexBuilder {
    fn add_batch(&mut self, indexed_batch: BuildSideBatch);

    fn with_stats(&mut self, stats: GeoStatistics);

    fn build(self, schema: SchemaRef) -> Result<Arc<dyn SpatialIndex>>;
}

pub(crate) fn create_spatial_index_builder(streams: Vec<SendableBuildSideBatchStream>) -> Box<dyn SpatialIndexBuilder> {
    // Placeholder implementation: choose the appropriate builder based on streams or other criteria
    todo!()
}
