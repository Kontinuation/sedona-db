# GDAL Stacked Branches

This note records the current GDAL-related branches, their PRs, dependencies, status, and purpose.

## Current Stack Overview

```text
upstream/main
  -> feat/sedona-gdal-safe-ops
    -> feat/sedona-gdal-safe-facade
```

`feat/sedona-gdal-safe-foundation`, `feat/sedona-gdal-safe-primitives`, and `feat/sedona-gdal-safe-core` are now merged into `upstream/main`.

`raster-all` is a separate WIP aggregation branch that combines the broader GDAL-backed raster work.

## Merged History

```text
feat/sedona-gdal-safe-foundation
  -> feat/sedona-gdal-safe-primitives
    -> feat/sedona-gdal-safe-core
      -> upstream/main
```

## Branches

### `feat/sedona-gdal-safe-foundation`

- PR: https://github.com/apache/sedona-db/pull/696
- Title: `feat(sedona-gdal): add foundational wrapper utilities`
- Status: merged into `upstream/main` on 2026-03-13
- Depends on: `upstream/main` at the time it was developed
- Used by:
  - historical base for `feat/sedona-gdal-safe-primitives`
- Purpose:
  - adds foundational safe-wrapper utilities on top of the GDAL FFI crate
  - introduces GDAL option list helpers, raster type abstractions, and expanded shared error handling

### `feat/sedona-gdal-safe-primitives`

- PR: https://github.com/apache/sedona-db/pull/695
- Title: `feat(sedona-gdal): add geometry and spatial ref primitives`
- Status: merged into `upstream/main` on 2026-03-17
- Depends on:
  - `feat/sedona-gdal-safe-foundation` during review
  - now included in `upstream/main`
- Used by:
  - `feat/sedona-gdal-safe-core`
- Purpose:
  - adds standalone geometry, geotransform, spatial reference, and VSI wrappers
  - exposes the first reusable vector primitive module for later dataset and raster operation wrappers

### `feat/sedona-gdal-safe-core`

- PR: https://github.com/apache/sedona-db/pull/699
- Title: `feat(sedona-gdal): add dataset and vector/raster wrappers`
- Status: merged into `upstream/main`
- Latest local rebase status:
  - rebased onto current `upstream/main` after `feat/sedona-gdal-safe-primitives` merged
  - updated remaining null-pointer sites to use `GdalApi::last_null_pointer_err`
  - `cargo test -p sedona-gdal` passes
  - doc comments tightened for dataset, layer, feature, and rasterband APIs
  - current merged tip on `main`: `2822b306`
- Depends on:
  - `upstream/main` at the time it was developed
- Used by:
  - `feat/sedona-gdal-safe-ops`
  - `feat/sedona-gdal-safe-facade` indirectly through `feat/sedona-gdal-safe-ops`
- Purpose:
  - adds safe wrappers for GDAL datasets, drivers, raster bands, features, and layers
  - wires the core raster/vector modules that higher-level operations build on

### `feat/sedona-gdal-safe-ops`

- PR: https://github.com/apache/sedona-db/pull/698
- Title: `feat(sedona-gdal): add raster operations and vrt support`
- Status: open draft PR
- Latest local rebase status:
  - rebased onto current `main` after `feat/sedona-gdal-safe-core` merged
  - keeps the dataset ownership cleanup and vector dataset tests from `main`
  - `cargo test -p sedona-gdal --lib` passes
  - fixed affine rasterize error reporting to preserve the original GDAL error code
  - current local and remote tip: `87357456`
- Depends on:
  - `upstream/main`
- Used by:
  - `feat/sedona-gdal-safe-facade`
  - higher-level raster/GDAL work
- Purpose:
  - adds VRT dataset support plus rasterize, rasterize-affine, and polygonize wrappers
  - layers higher-level raster operations on top of the dataset/raster/vector wrapper stack

### `feat/sedona-gdal-safe-facade`

- PR: https://github.com/apache/sedona-db/pull/697
- Title: `feat(sedona-gdal): add convenience facade and mem builder`
- Status: open draft PR
- Latest local rebase status:
  - rebased onto the updated `feat/sedona-gdal-safe-ops`
  - updated MEM and VRT constructors to use the always-owned `Dataset::new(...)`
  - `cargo test -p sedona-gdal --lib` passes
  - current local and remote tip: `006695f1`
- Depends on:
  - `feat/sedona-gdal-safe-ops`
- Used by:
  - higher-level GDAL consumers
- Purpose:
  - adds the high-level `Gdal` facade and `with_global_gdal` convenience entry point
  - adds the MEM dataset builder on top of the lower-level dataset and raster wrappers
  - keeps the top-level API explicit by importing concrete raster/vector modules instead of relying on wrapper re-export aliases

### `raster-all`

- PR: https://github.com/apache/sedona-db/pull/704
- Title: `[WIP] feat(raster): add GDAL-based raster functions`
- Status: open draft PR
- Depends on:
  - broad GDAL stack; intended as an aggregation branch rather than a single reviewable dependency step
- Purpose:
  - contains the full GDAL-backed raster implementation stack, including `c/sedona-gdal`, `rust/sedona-raster-gdal`, and related runtime/workspace wiring
  - serves as a WIP umbrella branch to test the whole surface area before splitting work into smaller PRs

## Local Fetch Status

The following local references are relevant right now:

- merged history kept locally:
  - `feat/sedona-gdal-safe-foundation`
  - `feat/sedona-gdal-safe-primitives`
- merged into `main`:
  - `feat/sedona-gdal-safe-core`
- remaining stacked branches aligned on `kontinuation`:
  - `feat/sedona-gdal-safe-ops`
  - `feat/sedona-gdal-safe-facade`
- separate branch to keep in sync from `kontinuation`:
  - `raster-all`
