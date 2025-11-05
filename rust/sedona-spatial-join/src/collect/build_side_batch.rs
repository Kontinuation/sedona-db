use arrow_array::RecordBatch;
use datafusion_expr::ColumnarValue;
use geo::Rect;
use wkb::reader::Wkb;

use crate::operand_evaluator::EvaluatedGeometryArray;

/// BuildSide batch containing the original record batch from the build side and the evaluated
/// geometry array.
pub(crate) struct BuildSideBatch {
    /// Original record batch polled from the build side stream
    pub batch: RecordBatch,
    /// Evaluated geometry array, containing the geometry array containing geometries to be joined,
    /// rects of joined geometries, evaluated distance columnar values if we are running a distance
    /// join and the distance expression is bound to the build side, etc.
    pub geom_array: EvaluatedGeometryArray,
}

impl BuildSideBatch {
    pub fn in_mem_size(&self) -> usize {
        // NOTE: sometimes `geom_array` will reuse the memory of `batch`, especially when
        // the expression for evaluating the geometry is a simple column reference. In this case,
        // the in_mem_size will be overestimated. It is a conservative estimation so there's no risk
        // of running out of memory because of underestimation.
        self.batch.get_array_memory_size() + self.geom_array.in_mem_size()
    }

    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    pub fn wkb(&self, idx: usize) -> Option<&Wkb<'_>> {
        let wkbs = self.geom_array.wkbs();
        wkbs[idx].as_ref()
    }

    pub fn rects(&self) -> &Vec<Option<Rect<f32>>> {
        &self.geom_array.rects
    }

    pub fn distance(&self) -> &Option<ColumnarValue> {
        &self.geom_array.distance
    }
}
