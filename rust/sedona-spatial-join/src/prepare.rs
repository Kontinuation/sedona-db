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

use std::{mem, sync::Arc};

use arrow_schema::SchemaRef;
use datafusion_common::Result;
use datafusion_common_runtime::JoinSet;
use datafusion_execution::{
    disk_manager::RefCountedTempFile, memory_pool::MemoryConsumer, SendableRecordBatchStream,
    TaskContext,
};
use datafusion_expr::JoinType;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use fastrand::Rng;
use sedona_common::{sedona_internal_err, NumSpatialPartitionsConfig, SedonaOptions};
use sedona_expr::statistics::GeoStatistics;
use sedona_geometry::bounding_box::BoundingBox;

use crate::{
    index::{
        memory_plan::{compute_memory_plan, PartitionMemorySummary},
        partitioned_index_provider::PartitionedIndexProvider,
        BuildSideBatchesCollector, CollectBuildSideMetrics, SpatialJoinBuildMetrics,
    },
    partitioning::{
        flat::FlatPartitioner,
        kdb::KDBPartitioner,
        stream_repartitioner::{
            repartition_evaluated_batches, SpilledPartition, SpilledPartitions,
        },
        PartitionedSide, SpatialPartition, SpatialPartitioner,
    },
    probe::partitioned_stream_provider::ProbeStreamOptions,
    spatial_predicate::SpatialPredicate,
    utils::bbox_sampler::BoundingBoxSamples,
};

pub(crate) struct SpatialJoinComponents {
    pub partitioned_index_provider: Arc<PartitionedIndexProvider>,
    pub probe_stream_options: ProbeStreamOptions,
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn prepare_spatial_join_components(
    context: Arc<TaskContext>,
    build_schema: SchemaRef,
    build_streams: Vec<SendableRecordBatchStream>,
    spatial_predicate: SpatialPredicate,
    join_type: JoinType,
    probe_threads_count: usize,
    metrics: ExecutionPlanMetricsSet,
    seed: u64,
) -> Result<SpatialJoinComponents> {
    let session_config = context.session_config();
    let target_batch_size = session_config.batch_size();
    let sedona_options = session_config
        .options()
        .extensions
        .get::<SedonaOptions>()
        .cloned()
        .unwrap_or_default();
    let concurrent = sedona_options.spatial_join.concurrent_build_side_collection;
    let spill_compression = session_config.spill_compression();
    let memory_pool = context.memory_pool();
    let num_partitions = build_streams.len();
    if num_partitions == 0 {
        log::info!("Build side has no data. Creating empty spatial index.");
        let partitioned_index_provider = PartitionedIndexProvider::new_empty(
            build_schema,
            spatial_predicate,
            sedona_options.spatial_join,
            join_type,
            probe_threads_count,
            Arc::clone(memory_pool),
            SpatialJoinBuildMetrics::new(0, &metrics),
        );
        let probe_stream_options = ProbeStreamOptions {
            partitioner: None,
            target_batch_rows: target_batch_size,
            spill_compression: session_config.spill_compression(),
            buffer_bytes_threshold: 0,
        };
        return Ok(SpatialJoinComponents {
            partitioned_index_provider: Arc::new(partitioned_index_provider),
            probe_stream_options,
        });
    }

    let runtime_env = context.runtime_env();
    let collector = BuildSideBatchesCollector::new(
        spatial_predicate.clone(),
        sedona_options.spatial_join.clone(),
        Arc::clone(&runtime_env),
        spill_compression,
    );
    let mut collect_metrics_vec = Vec::with_capacity(num_partitions);
    let mut reservations = Vec::with_capacity(num_partitions);
    for k in 0..num_partitions {
        let consumer =
            MemoryConsumer::new(format!("SpatialJoinCollectBuildSide[{k}]")).with_can_spill(true);
        let reservation = consumer.register(memory_pool);
        reservations.push(reservation);
        collect_metrics_vec.push(CollectBuildSideMetrics::new(k, &metrics));
    }

    let mut rng = Rng::with_seed(seed);
    let mut build_partitions = collector
        .collect_all(
            build_streams,
            reservations,
            collect_metrics_vec.clone(),
            concurrent,
            rng.u64(0..0xFFFF),
        )
        .await?;
    let memory_plan =
        compute_memory_plan(build_partitions.iter().map(PartitionMemorySummary::from))?;
    let mut memory_plan_str = String::new();
    if memory_plan.debug_print(&mut memory_plan_str).is_ok() {
        log::info!(
            "Computed memory plan for spatial join:\n{}",
            memory_plan_str
        );
    }
    let memory_for_intermittent_usage = memory_plan.memory_for_intermittent_usage;
    let num_partitions = match sedona_options.spatial_join.debug.num_spatial_partitions {
        NumSpatialPartitionsConfig::Auto => memory_plan.num_partitions,
        NumSpatialPartitionsConfig::Fixed(n) => {
            log::info!("Override number of spatial partitions to {}", n);
            n
        }
    };

    if num_partitions == 1 {
        log::info!("Running single-partitioned in-memory spatial join");
        let partitioned_index_provider = PartitionedIndexProvider::new_single_partition(
            build_schema,
            spatial_predicate,
            sedona_options.spatial_join,
            join_type,
            probe_threads_count,
            build_partitions,
            Arc::clone(memory_pool),
            SpatialJoinBuildMetrics::new(0, &metrics),
        );
        let probe_stream_options = ProbeStreamOptions {
            partitioner: None,
            target_batch_rows: target_batch_size,
            spill_compression: session_config.spill_compression(),
            buffer_bytes_threshold: 0,
        };
        Ok(SpatialJoinComponents {
            partitioned_index_provider: Arc::new(partitioned_index_provider),
            probe_stream_options,
        })
    } else {
        // TODO: implement spatial partitioned spatial join for KNN joins
        if matches!(spatial_predicate, SpatialPredicate::KNearestNeighbors(_)) {
            return sedona_internal_err!(
                "KNN spatial predicate is not supported for multi-partition spatial join"
            );
        }

        let mut bbox_samples = BoundingBoxSamples::empty();
        let mut geo_stats = GeoStatistics::empty();
        for partition in &mut build_partitions {
            let samples = mem::take(&mut partition.bbox_samples);
            bbox_samples = bbox_samples.combine(samples, &mut rng);
            geo_stats.merge(&partition.geo_statistics);
        }

        let extent = geo_stats
            .bbox()
            .cloned()
            .unwrap_or(BoundingBox::xy((0, 0), (0, 0)));
        let extent = if extent.is_empty() {
            BoundingBox::xy((0, 0), (0, 0))
        } else {
            extent
        };
        let mut samples = bbox_samples.take_samples();
        let max_items_per_node = 1.max(samples.len() / num_partitions);
        let max_levels = num_partitions;

        log::info!(
            "Number of samples: {}, max_items_per_node: {}, max_levels: {}",
            samples.len(),
            max_items_per_node,
            max_levels
        );
        rng.shuffle(&mut samples);
        let kdb_partitioner =
            KDBPartitioner::build(samples.into_iter(), max_items_per_node, max_levels, extent)?;
        log::info!(
            "Built KDB spatial partitioner with {} partitions",
            num_partitions
        );
        let mut kdb_dbg_str = String::new();
        if kdb_partitioner.debug_print(&mut kdb_dbg_str).is_ok() {
            log::info!("KDB partitioner debug info:\n{}", kdb_dbg_str);
        }

        let build_partitioner: Arc<dyn SpatialPartitioner> = Arc::new(kdb_partitioner);
        let num_partitions = build_partitioner.num_regular_partitions();
        log::info!("Actual number of spatial partitions: {}", num_partitions);

        // Spawn each task for each build partition to repartition the data using the spatial partitioner for
        // the build/indexed side
        let mut join_set = JoinSet::new();
        let buffer_bytes_threshold = memory_for_intermittent_usage / build_partitions.len();
        let mut reservations = Vec::with_capacity(build_partitions.len());
        for (k, partition) in build_partitions.into_iter().enumerate() {
            let stream = partition.build_side_batch_stream;
            let reservation = partition.reservation;
            let metrics = &collect_metrics_vec[k];
            let spill_metrics = metrics.spill_metrics();
            let runtime_env = Arc::clone(&runtime_env);
            let partitioner = Arc::clone(&build_partitioner);
            join_set.spawn(async move {
                let partitioned_spill_files = repartition_evaluated_batches(
                    runtime_env,
                    stream,
                    partitioner,
                    PartitionedSide::BuildSide,
                    spill_compression,
                    spill_metrics,
                    buffer_bytes_threshold,
                    target_batch_size,
                )
                .await;
                partitioned_spill_files
            });
            reservations.push(reservation);
        }

        let results = join_set.join_all().await;
        let mut partitioned_spill_files_vec = Vec::with_capacity(results.len());
        for result in results {
            partitioned_spill_files_vec.push(result?);
        }

        log::info!("Per-index-partition spilled partitions collected.");
        for (k, spilled_partitions) in partitioned_spill_files_vec.iter().enumerate() {
            let mut s = String::new();
            if spilled_partitions.debug_print(&mut s).is_ok() {
                log::info!("Build partition {} spilled partitions:\n{}", k, s);
            }
        }

        let merged_spilled_partitions = merge_spilled_partitions(partitioned_spill_files_vec)?;
        let mut s = String::new();
        if merged_spilled_partitions.debug_print(&mut s).is_ok() {
            log::info!("Build side spatial partitions:\n{}", s);
        }

        // Sanity check: Multi and None partitions must be empty. All the geometries in the build side
        // should fall into regular partitions
        for partition in [SpatialPartition::None, SpatialPartition::Multi] {
            let spilled_partition = merged_spilled_partitions.spilled_partition(partition)?;
            if !spilled_partition.spill_files().is_empty() {
                return sedona_internal_err!(
                    "Build side spatial partitions {:?} should be empty",
                    partition
                );
            }
        }

        // Build a flat partitioner using these partitions
        let mut partition_bounds = Vec::with_capacity(num_partitions);
        for k in 0..num_partitions {
            let partition = SpatialPartition::Regular(k as u32);
            let partition_bound = merged_spilled_partitions
                .spilled_partition(partition)?
                .bounding_box()
                .cloned()
                .unwrap_or(BoundingBox::empty());
            partition_bounds.push(partition_bound);
        }
        let probe_partitioner = Arc::new(FlatPartitioner::try_new(partition_bounds)?);

        let partitioned_index_provider = PartitionedIndexProvider::new_multi_partition(
            build_schema,
            spatial_predicate,
            sedona_options.spatial_join,
            join_type,
            probe_threads_count,
            merged_spilled_partitions,
            Arc::clone(memory_pool),
            SpatialJoinBuildMetrics::new(0, &metrics),
            reservations,
        );

        let buffer_bytes_threshold = memory_for_intermittent_usage / probe_threads_count;
        let probe_stream_options = ProbeStreamOptions {
            partitioner: Some(probe_partitioner),
            target_batch_rows: target_batch_size,
            spill_compression,
            buffer_bytes_threshold,
        };

        Ok(SpatialJoinComponents {
            partitioned_index_provider: Arc::new(partitioned_index_provider),
            probe_stream_options,
        })
    }
}

/// Aggregate the spill files and bounds of each spatial partition collected from all build partitions
fn merge_spilled_partitions(
    spilled_partitions_vec: Vec<SpilledPartitions>,
) -> Result<SpilledPartitions> {
    let Some(first) = spilled_partitions_vec.first() else {
        return sedona_internal_err!("spilled_partitions_vec cannot be empty");
    };

    let slots = first.slots();
    let total_slots = slots.total_slots();
    let mut merged_spill_files: Vec<Vec<Arc<RefCountedTempFile>>> =
        (0..total_slots).map(|_| Vec::new()).collect();
    let mut partition_geo_stats: Vec<GeoStatistics> =
        (0..total_slots).map(|_| GeoStatistics::empty()).collect();
    let mut partition_num_rows: Vec<usize> = (0..total_slots).map(|_| 0).collect();

    for spilled_partitions in spilled_partitions_vec {
        let partitions = spilled_partitions.into_spilled_partitions()?;
        for (slot_idx, partition) in partitions.into_iter().enumerate() {
            let (spill_files, geo_stats, num_rows) = partition.into_inner();
            partition_geo_stats[slot_idx].merge(&geo_stats);
            merged_spill_files[slot_idx].extend(spill_files);
            partition_num_rows[slot_idx] += num_rows;
        }
    }

    let merged_partitions = merged_spill_files
        .into_iter()
        .zip(partition_geo_stats)
        .zip(partition_num_rows)
        .map(|((spill_files, geo_stats), num_rows)| {
            SpilledPartition::new(spill_files, geo_stats, num_rows)
        })
        .collect();

    Ok(SpilledPartitions::new(slots, merged_partitions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partitioning::partition_slots::PartitionSlots;
    use datafusion_execution::runtime_env::RuntimeEnv;
    use sedona_geometry::interval::IntervalTrait;

    fn sample_geo_stats(bbox: (f64, f64, f64, f64), total_geometries: i64) -> GeoStatistics {
        GeoStatistics::empty()
            .with_bbox(Some(BoundingBox::xy((bbox.0, bbox.1), (bbox.2, bbox.3))))
            .with_total_geometries(total_geometries)
    }

    fn sample_partition(
        env: &Arc<RuntimeEnv>,
        labels: &[&str],
        bbox: (f64, f64, f64, f64),
        total_geometries: i64,
    ) -> Result<SpilledPartition> {
        let mut files = Vec::with_capacity(labels.len());
        for label in labels {
            files.push(Arc::new(env.disk_manager.create_tmp_file(label)?));
        }
        Ok(SpilledPartition::new(
            files,
            sample_geo_stats(bbox, total_geometries),
            total_geometries as usize,
        ))
    }

    #[test]
    fn merge_spilled_partitions_combines_files_and_stats() -> Result<()> {
        let runtime_env = Arc::new(RuntimeEnv::default());
        let slots = PartitionSlots::new(2);

        let partitions_a = vec![
            sample_partition(&runtime_env, &["r0_a"], (0.0, 1.0, 0.0, 1.0), 10)?,
            sample_partition(&runtime_env, &["r1_a"], (10.0, 11.0, -1.0, 1.0), 5)?,
            sample_partition(&runtime_env, &["none_a"], (-5.0, -4.0, -5.0, -4.0), 2)?,
            SpilledPartition::empty(),
        ];
        let first = SpilledPartitions::new(slots, partitions_a);

        let partitions_b = vec![
            sample_partition(&runtime_env, &["r0_b1", "r0_b2"], (5.0, 6.0, 5.0, 6.0), 20)?,
            sample_partition(&runtime_env, &[], (12.0, 13.0, 2.0, 3.0), 8)?,
            SpilledPartition::empty(),
            sample_partition(&runtime_env, &["multi_b"], (50.0, 51.0, 50.0, 51.0), 1)?,
        ];
        let second = SpilledPartitions::new(slots, partitions_b);

        let merged = merge_spilled_partitions(vec![first, second])?;

        assert_eq!(merged.spill_file_count(), 6);

        let regular0 = merged.spilled_partition(SpatialPartition::Regular(0))?;
        assert_eq!(regular0.spill_files().len(), 3);
        assert_eq!(regular0.geo_statistics().total_geometries(), Some(30));
        let bbox0 = regular0.geo_statistics().bbox().unwrap();
        assert_eq!(bbox0.x().lo(), 0.0);
        assert_eq!(bbox0.x().hi(), 6.0);
        assert_eq!(bbox0.y().lo(), 0.0);
        assert_eq!(bbox0.y().hi(), 6.0);

        let regular1 = merged.spilled_partition(SpatialPartition::Regular(1))?;
        assert_eq!(regular1.spill_files().len(), 1);
        assert_eq!(regular1.geo_statistics().total_geometries(), Some(13));
        let bbox1 = regular1.geo_statistics().bbox().unwrap();
        assert_eq!(bbox1.x().lo(), 10.0);
        assert_eq!(bbox1.x().hi(), 13.0);
        assert_eq!(bbox1.y().lo(), -1.0);
        assert_eq!(bbox1.y().hi(), 3.0);

        let none_partition = merged.spilled_partition(SpatialPartition::None)?;
        assert_eq!(none_partition.spill_files().len(), 1);
        assert_eq!(none_partition.geo_statistics().total_geometries(), Some(2));

        let multi_partition = merged.spilled_partition(SpatialPartition::Multi)?;
        assert_eq!(multi_partition.spill_files().len(), 1);
        assert_eq!(multi_partition.geo_statistics().total_geometries(), Some(1));

        Ok(())
    }
}
