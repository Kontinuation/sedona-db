// /// Indexed batch containing the original record batch and the evaluated geometry array.
// pub(crate) struct IndexedBatch {
//     batch: RecordBatch,
//     geom_array: EvaluatedGeometryArray,
// }

// impl IndexedBatch {
//     pub fn in_mem_size(&self) -> usize {
//         // NOTE: sometimes `geom_array` will reuse the memory of `batch`, especially when
//         // the expression for evaluating the geometry is a simple column reference. In this case,
//         // the in_mem_size will be overestimated.
//         self.batch.get_array_memory_size() + self.geom_array.in_mem_size()
//     }

//     pub fn wkb(&self, idx: usize) -> Option<&Wkb<'_>> {
//         let wkbs = self.geom_array.wkbs();
//         wkbs[idx].as_ref()
//     }

//     pub fn rects(&self) -> &Vec<(usize, Rect<f32>)> {
//         &self.geom_array.rects
//     }

//     pub fn distance(&self) -> &Option<ColumnarValue> {
//         &self.geom_array.distance
//     }
// }
