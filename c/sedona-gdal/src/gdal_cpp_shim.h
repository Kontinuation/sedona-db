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

#ifndef SEDONA_GDAL_CPP_SHIM_H_INCLUDED
#define SEDONA_GDAL_CPP_SHIM_H_INCLUDED

#if defined(SEDONA_GDAL_BUILD)
#include <gdal.h>
#else
typedef void* GDALDatasetH;
typedef int GDALDataType;
typedef long long GSpacing;
#endif

#if defined(_WIN32)
#if defined(SEDONA_GDAL_BUILD)
#define SEDONA_GDAL_API __declspec(dllexport)
#else
#define SEDONA_GDAL_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define SEDONA_GDAL_API __attribute__((visibility("default")))
#else
#define SEDONA_GDAL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

SEDONA_GDAL_API GDALDatasetH sedona_gdal_mem_create_internal(
    int x_size, int y_size, int band_count, const GDALDataType* band_types,
    const void* const* band_data, const GSpacing* pixel_offsets,
    const GSpacing* line_offsets);

SEDONA_GDAL_API void sedona_gdal_dataset_close(GDALDatasetH dataset);

#ifdef __cplusplus
}
#endif

#endif
