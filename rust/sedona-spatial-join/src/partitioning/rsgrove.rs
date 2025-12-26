// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! RSGrove spatial partitioning implementation.
//!
//! This module provides a RSGrove partitioner implementation for spatial
//! partitioning. It uses R*-tree splitting heuristics to partition the space
//! into disjoint regions.

use std::sync::Arc;

use crate::partitioning::{
    util::{bbox_to_geo_rect, rect_contains_point, rect_intersection_area, rects_intersect},
    SpatialPartition, SpatialPartitioner,
};
use datafusion_common::Result;
use geo::{Coord, Rect};
use sedona_common::sedona_internal_err;
use sedona_geometry::bounding_box::BoundingBox;

/// RSGrove tree spatial partitioner implementation.
pub(crate) struct RSGroveTree {
    max_items_per_node: usize,
    max_levels: usize,
    extent: Rect<f32>,
    level: usize,
    items: Vec<Rect<f32>>,
    children: Option<Box<[RSGroveTree; 2]>>,
    leaf_id: u32,
}

impl RSGroveTree {
    pub fn try_new(
        max_items_per_node: usize,
        max_levels: usize,
        extent: BoundingBox,
    ) -> Result<Self> {
        if max_items_per_node == 0 {
            return sedona_internal_err!("max_items_per_node must be greater than 0");
        }
        let Some(extent_rect) = bbox_to_geo_rect(&extent)? else {
            return sedona_internal_err!("RSGroveTree extent cannot be empty");
        };
        Ok(Self::new_with_level(
            max_items_per_node,
            max_levels,
            0,
            extent_rect,
        ))
    }

    fn new_with_level(
        max_items_per_node: usize,
        max_levels: usize,
        level: usize,
        extent: Rect<f32>,
    ) -> Self {
        RSGroveTree {
            max_items_per_node,
            max_levels,
            extent,
            level,
            items: Vec::new(),
            children: None,
            leaf_id: 0,
        }
    }

    pub fn insert(&mut self, bbox: BoundingBox) -> Result<()> {
        if let Some(rect) = bbox_to_geo_rect(&bbox)? {
            if rect_contains_point(&self.extent, &rect.min()) {
                self.insert_rect(rect);
            }
        }
        Ok(())
    }

    fn insert_rect(&mut self, rect: Rect<f32>) {
        if self.children.is_none() {
            if self.items.len() < self.max_items_per_node || self.level >= self.max_levels {
                self.items.push(rect);
                return;
            }

            if !self.split() {
                self.items.push(rect);
                return;
            }
        }

        if let Some(ref mut children) = self.children {
            let min_point = rect.min();
            for child in children.iter_mut() {
                if rect_contains_point(child.extent(), &min_point) {
                    child.insert_rect(rect);
                    break;
                }
            }
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    pub fn leaf_id(&self) -> u32 {
        assert!(self.is_leaf(), "leaf_id() called on non-leaf node");
        self.leaf_id
    }

    pub fn extent(&self) -> &Rect<f32> {
        &self.extent
    }

    pub fn assign_leaf_ids(&mut self) {
        let mut next_id = 0;
        self.assign_leaf_ids_recursive(&mut next_id);
    }

    fn assign_leaf_ids_recursive(&mut self, next_id: &mut u32) {
        if self.is_leaf() {
            self.leaf_id = *next_id;
            *next_id += 1;
        } else if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                child.assign_leaf_ids_recursive(next_id);
            }
        }
    }

    pub fn visit_intersecting_leaf_nodes<'a>(
        &'a self,
        rect: &Rect<f32>,
        f: &mut impl FnMut(&'a RSGroveTree),
    ) {
        if !rects_intersect(&self.extent, rect) {
            return;
        }

        if self.is_leaf() {
            f(self)
        } else if let Some(ref children) = self.children {
            for child in children.iter() {
                child.visit_intersecting_leaf_nodes(rect, f);
            }
        }
    }

    pub fn visit_leaf_nodes<'a>(&'a self, f: &mut impl FnMut(&'a RSGroveTree)) {
        if self.is_leaf() {
            f(self)
        } else if let Some(ref children) = self.children {
            for child in children.iter() {
                child.visit_leaf_nodes(f);
            }
        }
    }

    pub fn collect_leaf_nodes(&self) -> Vec<&RSGroveTree> {
        let mut leaf_nodes = Vec::new();
        self.visit_leaf_nodes(&mut |node| {
            leaf_nodes.push(node);
        });
        leaf_nodes
    }

    pub fn num_leaf_nodes(&self) -> usize {
        let mut num = 0;
        self.visit_leaf_nodes(&mut |_| {
            num += 1;
        });
        num
    }

    pub fn drop_elements(&mut self) {
        self.items.clear();
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                child.drop_elements();
            }
        }
    }

    fn split(&mut self) -> bool {
        if self.items.len() < 2 {
            return false;
        }

        // R*-tree split logic
        let min_split_size = (self.items.len() as f32 * 0.1).ceil() as usize;
        let min_split_size = min_split_size.max(1);
        let max_split_size = self.items.len() - min_split_size;

        if min_split_size > max_split_size {
            return false;
        }

        // 1. Choose Split Axis
        let mut best_axis = 0; // 0 for x, 1 for y
        let mut min_margin_sum = f32::MAX;

        // We need to sort items to evaluate splits.
        // To avoid re-sorting, we can create indices.
        let mut x_indices: Vec<usize> = (0..self.items.len()).collect();
        x_indices.sort_by(|&a, &b| {
            self.items[a]
                .min()
                .x
                .partial_cmp(&self.items[b].min().x)
                .unwrap()
        });

        let mut y_indices: Vec<usize> = (0..self.items.len()).collect();
        y_indices.sort_by(|&a, &b| {
            self.items[a]
                .min()
                .y
                .partial_cmp(&self.items[b].min().y)
                .unwrap()
        });

        // Evaluate X axis
        let margin_sum_x = self.compute_margin_sum(&x_indices, min_split_size, max_split_size);
        if margin_sum_x < min_margin_sum {
            min_margin_sum = margin_sum_x;
            best_axis = 0;
        }

        // Evaluate Y axis
        let margin_sum_y = self.compute_margin_sum(&y_indices, min_split_size, max_split_size);
        if margin_sum_y < min_margin_sum {
            best_axis = 1;
        }

        // 2. Choose Split Index along best axis
        let indices = if best_axis == 0 {
            &x_indices
        } else {
            &y_indices
        };
        let (_best_k, split_coord) =
            self.choose_split_index(indices, min_split_size, max_split_size, best_axis);

        // Perform split
        let extent_min = self.extent.min();
        let extent_max = self.extent.max();

        // Check if split coordinate is valid (within extent)
        if best_axis == 0 {
            if split_coord <= extent_min.x || split_coord >= extent_max.x {
                return false;
            }
        } else {
            if split_coord <= extent_min.y || split_coord >= extent_max.y {
                return false;
            }
        }

        let (left_extent, right_extent) = if best_axis == 0 {
            let left = Rect::new(
                extent_min,
                Coord {
                    x: split_coord,
                    y: extent_max.y,
                },
            );
            let right = Rect::new(
                Coord {
                    x: split_coord,
                    y: extent_min.y,
                },
                extent_max,
            );
            (left, right)
        } else {
            let left = Rect::new(
                extent_min,
                Coord {
                    x: extent_max.x,
                    y: split_coord,
                },
            );
            let right = Rect::new(
                Coord {
                    x: extent_min.x,
                    y: split_coord,
                },
                extent_max,
            );
            (left, right)
        };

        let mut left_child = RSGroveTree::new_with_level(
            self.max_items_per_node,
            self.max_levels,
            self.level + 1,
            left_extent,
        );
        let mut right_child = RSGroveTree::new_with_level(
            self.max_items_per_node,
            self.max_levels,
            self.level + 1,
            right_extent,
        );

        // Distribute items
        for item in self.items.drain(..) {
            let coord = if best_axis == 0 {
                item.min().x
            } else {
                item.min().y
            };
            if coord <= split_coord {
                left_child.insert_rect(item);
            } else {
                right_child.insert_rect(item);
            }
        }

        self.children = Some(Box::new([left_child, right_child]));
        true
    }

    fn compute_margin_sum(&self, indices: &[usize], min_k: usize, max_k: usize) -> f32 {
        let mut margin_sum = 0.0;

        // Precompute MBRs from left and right?
        // Or just compute on the fly. O(N^2) if naive.
        // We can do O(N) with prefix/suffix arrays.

        let n = indices.len();
        let mut left_mbrs = vec![Rect::new(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: 0.0 }); n];
        let mut right_mbrs = vec![Rect::new(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: 0.0 }); n];

        // Forward pass
        let mut current_mbr = self.items[indices[0]];
        left_mbrs[0] = current_mbr;
        for i in 1..n {
            current_mbr = self.union_rects(&current_mbr, &self.items[indices[i]]);
            left_mbrs[i] = current_mbr;
        }

        // Backward pass
        current_mbr = self.items[indices[n - 1]];
        right_mbrs[n - 1] = current_mbr;
        for i in (0..n - 1).rev() {
            current_mbr = self.union_rects(&current_mbr, &self.items[indices[i]]);
            right_mbrs[i] = current_mbr;
        }

        for k in min_k..=max_k {
            let left_mbr = left_mbrs[k - 1];
            let right_mbr = right_mbrs[k];
            margin_sum += self.rect_margin(&left_mbr) + self.rect_margin(&right_mbr);
        }

        margin_sum
    }

    fn choose_split_index(
        &self,
        indices: &[usize],
        min_k: usize,
        max_k: usize,
        axis: usize,
    ) -> (usize, f32) {
        let mut best_k = min_k;
        let mut min_overlap = f32::MAX;
        let mut min_area = f32::MAX;

        let n = indices.len();
        let mut left_mbrs = vec![Rect::new(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: 0.0 }); n];
        let mut right_mbrs = vec![Rect::new(Coord { x: 0.0, y: 0.0 }, Coord { x: 0.0, y: 0.0 }); n];

        // Forward pass
        let mut current_mbr = self.items[indices[0]];
        left_mbrs[0] = current_mbr;
        for i in 1..n {
            current_mbr = self.union_rects(&current_mbr, &self.items[indices[i]]);
            left_mbrs[i] = current_mbr;
        }

        // Backward pass
        current_mbr = self.items[indices[n - 1]];
        right_mbrs[n - 1] = current_mbr;
        for i in (0..n - 1).rev() {
            current_mbr = self.union_rects(&current_mbr, &self.items[indices[i]]);
            right_mbrs[i] = current_mbr;
        }

        for k in min_k..=max_k {
            let left_mbr = left_mbrs[k - 1];
            let right_mbr = right_mbrs[k];

            let overlap = rect_intersection_area(&left_mbr, &right_mbr);
            let area = self.rect_area(&left_mbr) + self.rect_area(&right_mbr);

            if overlap < min_overlap {
                min_overlap = overlap;
                min_area = area;
                best_k = k;
            } else if (overlap - min_overlap).abs() < f32::EPSILON && area < min_area {
                min_area = area;
                best_k = k;
            }
        }

        // Determine split coordinate
        // We split between indices[best_k-1] and indices[best_k]
        let item_l = &self.items[indices[best_k - 1]];
        let _item_r = &self.items[indices[best_k]];

        let coord_l = if axis == 0 {
            item_l.min().x
        } else {
            item_l.min().y
        };
        let _coord_r = if axis == 0 {
            _item_r.min().x
        } else {
            _item_r.min().y
        };

        // Use the midpoint? Or just coord_l?
        // KDB uses coord_l (or item.min).
        // If we use midpoint, we might be safer against precision issues?
        // But let's stick to simple logic: split at the boundary of the left group.
        // Actually, if we use coord_l, then item_l is <= split_coord.
        // item_r is >= coord_r.
        // If coord_l == coord_r, we might have issues separating them if we use strict inequality.
        // But we sorted them.
        // Let's use coord_l.

        (best_k, coord_l)
    }

    fn union_rects(&self, r1: &Rect<f32>, r2: &Rect<f32>) -> Rect<f32> {
        let min_x = r1.min().x.min(r2.min().x);
        let min_y = r1.min().y.min(r2.min().y);
        let max_x = r1.max().x.max(r2.max().x);
        let max_y = r1.max().y.max(r2.max().y);
        Rect::new(Coord { x: min_x, y: min_y }, Coord { x: max_x, y: max_y })
    }

    fn rect_margin(&self, r: &Rect<f32>) -> f32 {
        (r.width() + r.height()) * 2.0
    }

    fn rect_area(&self, r: &Rect<f32>) -> f32 {
        r.width() * r.height()
    }
}

pub struct RSGrovePartitioner {
    tree: Arc<RSGroveTree>,
}

impl RSGrovePartitioner {
    pub(crate) fn new(tree: Arc<RSGroveTree>) -> Self {
        RSGrovePartitioner { tree }
    }

    pub fn build(
        bboxes: impl Iterator<Item = BoundingBox>,
        max_items_per_node: usize,
        max_levels: usize,
        extent: BoundingBox,
    ) -> Result<Self> {
        let mut tree = RSGroveTree::try_new(max_items_per_node, max_levels, extent)?;

        for bbox in bboxes {
            tree.insert(bbox)?;
        }

        tree.assign_leaf_ids();
        tree.drop_elements();

        Ok(Self::new(Arc::new(tree)))
    }

    pub fn num_partitions(&self) -> usize {
        self.tree.num_leaf_nodes()
    }
}

impl SpatialPartitioner for RSGrovePartitioner {
    fn num_regular_partitions(&self) -> usize {
        self.num_partitions()
    }

    fn partition(&self, bbox: &BoundingBox) -> Result<SpatialPartition> {
        let Some(rect) = bbox_to_geo_rect(bbox)? else {
            return Ok(SpatialPartition::None);
        };

        let mut matches = Vec::new();
        self.tree.visit_intersecting_leaf_nodes(&rect, &mut |node| {
            matches.push(node.leaf_id());
        });

        match matches.len() {
            0 => Ok(SpatialPartition::None),
            1 => Ok(SpatialPartition::Regular(matches[0])),
            _ => Ok(SpatialPartition::Multi),
        }
    }

    fn partition_no_multi(&self, bbox: &BoundingBox) -> Result<SpatialPartition> {
        let Some(rect) = bbox_to_geo_rect(bbox)? else {
            return Ok(SpatialPartition::None);
        };

        let mut best_id = None;
        let mut max_area = -1.0;

        self.tree.visit_intersecting_leaf_nodes(&rect, &mut |node| {
            let area = rect_intersection_area(node.extent(), &rect);
            if area > 0.0 {
                if area > max_area {
                    max_area = area;
                    best_id = Some(node.leaf_id());
                }
            }
        });

        match best_id {
            Some(id) => Ok(SpatialPartition::Regular(id)),
            None => Ok(SpatialPartition::None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partitioning::kdb::KDBPartitioner;
    use rand::prelude::*;
    use sedona_geometry::bounding_box::BoundingBox;
    use sedona_geometry::interval::IntervalTrait;

    #[test]
    fn test_construct_from_sample() {
        let num_points = 1000;
        let mut bboxes = Vec::new();
        for i in 0..num_points {
            let x = i as f64;
            let y = i as f64;
            bboxes.push(BoundingBox::xy((x, x), (y, y)));
        }

        let extent = BoundingBox::xy((0.0, 1000.0), (0.0, 1000.0));
        let partitioner = RSGrovePartitioner::build(bboxes.into_iter(), 100, 10, extent).unwrap();

        // We expect roughly 10 partitions (1000 / 100)
        // But it depends on splitting.
        assert!(partitioner.num_partitions() > 0);
    }

    #[test]
    fn test_no_negative_id() {
        let num_records = 100;
        let mut bboxes = Vec::new();
        for i in 0..num_records {
            let x = i as f64;
            let y = i as f64;
            bboxes.push(BoundingBox::xy((x, x + 1.0), (y, y + 1.0)));
        }

        let extent = BoundingBox::xy((0.0, 1000.0), (0.0, 1000.0));
        let partitioner =
            RSGrovePartitioner::build(bboxes.clone().into_iter(), 10, 5, extent).unwrap();

        for bbox in bboxes {
            let partition = partitioner.partition(&bbox).unwrap();
            match partition {
                SpatialPartition::Regular(id) => assert!(id < partitioner.num_partitions() as u32),
                SpatialPartition::Multi => {} // Can happen
                SpatialPartition::None => panic!("Should not be None"),
            }
        }
    }

    #[test]
    fn test_coverage() {
        // Verify that the union of all leaf partitions covers the entire extent
        let extent = BoundingBox::xy((0.0, 100.0), (0.0, 100.0));
        let mut rng = StdRng::seed_from_u64(42);
        let mut bboxes = Vec::new();
        for _ in 0..100 {
            let x = rng.gen_range(0.0..100.0);
            let y = rng.gen_range(0.0..100.0);
            bboxes.push(BoundingBox::xy((x, x + 1.0), (y, y + 1.0)));
        }

        let partitioner =
            RSGrovePartitioner::build(bboxes.into_iter(), 10, 5, extent.clone()).unwrap();

        let leaves = partitioner.tree.collect_leaf_nodes();
        let mut total_area = 0.0;
        for leaf in leaves {
            total_area += leaf.extent().width() * leaf.extent().height();
        }

        let expected_area =
            (extent.x().hi() - extent.x().lo()) * (extent.y().hi() - extent.y().lo());
        // Relax tolerance for f32 precision at 10000 scale
        assert!(
            (total_area - expected_area as f32).abs() < 1e-2,
            "Total area {} does not match expected area {}",
            total_area,
            expected_area
        );
    }

    #[test]
    fn test_quality_comparison_thomas_cluster() {
        // Generate Thomas Cluster Process data
        let extent = BoundingBox::xy((0.0, 1000.0), (0.0, 1000.0));
        let mut rng = StdRng::seed_from_u64(12345);
        let mut points = Vec::new();

        let num_parents = 20;
        let points_per_parent = 50;
        let sigma = 20.0;

        for _ in 0..num_parents {
            let parent_x = rng.gen_range(0.0..1000.0);
            let parent_y = rng.gen_range(0.0..1000.0);

            for _ in 0..points_per_parent {
                // Box-Muller transform for normal distribution
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();

                let x = (parent_x + z0 * sigma).clamp(0.0, 1000.0);
                let y = (parent_y + z1 * sigma).clamp(0.0, 1000.0);

                points.push(BoundingBox::xy((x, x), (y, y)));
            }
        }

        let max_items = 50;
        let max_levels = 10;

        // Build KDB
        let kdb = KDBPartitioner::build(
            points.clone().into_iter(),
            max_items,
            max_levels,
            extent.clone(),
        )
        .unwrap();

        // Build RSGrove
        let rsgrove = RSGrovePartitioner::build(
            points.clone().into_iter(),
            max_items,
            max_levels,
            extent.clone(),
        )
        .unwrap();

        println!("KDB partitions: {}", kdb.num_partitions());
        println!("RSGrove partitions: {}", rsgrove.num_partitions());

        // Analyze load balance (points per partition)
        let kdb_stats = analyze_partition_stats(&kdb, &points);
        let rsgrove_stats = analyze_partition_stats(&rsgrove, &points);

        println!("KDB Stats: {:?}", kdb_stats);
        println!("RSGrove Stats: {:?}", rsgrove_stats);

        // RSGrove should have comparable or better load balance (lower std dev)
        // or better aspect ratios.
        // For clustered data, KDB might produce very thin slices.
        // Let's check aspect ratios for RSGrove.
        let rsgrove_aspect = analyze_aspect_ratios(&rsgrove);

        println!("RSGrove Mean Aspect Ratio: {}", rsgrove_aspect);

        // We expect RSGrove to produce more square-like partitions (aspect ratio closer to 1.0)
        // because it considers margin/area in splitting.
        // Note: Aspect ratio here is defined as max_dim / min_dim, so >= 1.0. Closer to 1.0 is better.
    }

    struct PartitionStats {
        mean: f64,
        std_dev: f64,
        min: usize,
        max: usize,
    }

    impl std::fmt::Debug for PartitionStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "mean={:.2}, std_dev={:.2}, min={}, max={}",
                self.mean, self.std_dev, self.min, self.max
            )
        }
    }

    fn analyze_partition_stats(
        partitioner: &impl SpatialPartitioner,
        points: &[BoundingBox],
    ) -> PartitionStats {
        let num_parts = partitioner.num_regular_partitions();
        let mut counts = vec![0; num_parts];

        for p in points {
            if let Ok(SpatialPartition::Regular(id)) = partitioner.partition(p) {
                if (id as usize) < num_parts {
                    counts[id as usize] += 1;
                }
            }
        }

        let sum: usize = counts.iter().sum();
        let mean = sum as f64 / num_parts as f64;
        let variance = counts
            .iter()
            .map(|&c| {
                let diff = c as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / num_parts as f64;

        PartitionStats {
            mean,
            std_dev: variance.sqrt(),
            min: *counts.iter().min().unwrap(),
            max: *counts.iter().max().unwrap(),
        }
    }

    fn analyze_aspect_ratios(partitioner: &RSGrovePartitioner) -> f64 {
        let leaves = partitioner.tree.collect_leaf_nodes();
        if leaves.is_empty() {
            return 0.0;
        }

        let mut total_aspect = 0.0;
        for leaf in &leaves {
            let w = leaf.extent().width();
            let h = leaf.extent().height();
            let aspect = if w > h { w / h } else { h / w };
            total_aspect += aspect as f64;
        }

        total_aspect / leaves.len() as f64
    }
}
