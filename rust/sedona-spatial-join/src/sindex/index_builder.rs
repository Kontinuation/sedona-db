use std::sync::Arc;

use crate::sindex::SpatialIndex;

pub(crate) struct SpatialIndexBuilder {

}

impl SpatialIndexBuilder {
    fn new() -> Self {
        SpatialIndexBuilder {

        }
    }

    fn build(self) -> Arc<dyn SpatialIndex> {
        todo!()
    }
}
