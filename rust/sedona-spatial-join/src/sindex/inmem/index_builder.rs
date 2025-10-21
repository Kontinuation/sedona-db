use std::sync::Arc;
use arrow_schema::SchemaRef;
use sedona_expr::statistics::GeoStatistics;

use datafusion_common::Result;
use crate::sindex::{build_side_batch::BuildSideBatch, index::SpatialIndex, index_builder::SpatialIndexBuilder};

pub(crate) struct InMemorySpatialIndexBuilder {

}

impl InMemorySpatialIndexBuilder {
    fn new() -> Self {
        InMemorySpatialIndexBuilder {

        }
    }

    fn build(self) -> Result<Arc<dyn SpatialIndex>> {
        todo!()
    }
}

impl SpatialIndexBuilder for InMemorySpatialIndexBuilder {
    fn add_batch(&mut self, indexed_batch: BuildSideBatch) {
        todo!()
    }

    fn with_stats(&mut self, stats: GeoStatistics) {
        todo!()
    }
    
    fn build(self, schema: SchemaRef) -> Result<Arc<dyn SpatialIndex>> {
        todo!()
    }
}
