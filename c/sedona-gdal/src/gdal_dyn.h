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

#ifndef SEDONA_GDAL_DYN_H_INCLUDED
#define SEDONA_GDAL_DYN_H_INCLUDED

#include <stddef.h>

typedef void* GDALDatasetH;
typedef int GDALDataType;
typedef long long GSpacing;

#ifdef __cplusplus
extern "C" {
#endif

struct SedonaGdalApi {
  GDALDatasetH (*sedona_gdal_mem_create_internal)(int x_size, int y_size, int band_count,
                                                  const GDALDataType* band_types,
                                                  const void* const* band_data,
                                                  const GSpacing* pixel_offsets,
                                                  const GSpacing* line_offsets);
  void (*release)(struct SedonaGdalApi*);
  void* private_data;
  void* shim;
};

int sedona_gdal_dyn_api_init(struct SedonaGdalApi* api, const char* shared_object_path,
                             char* err_msg, int len);

int sedona_gdal_dyn_api_init_from_current_process(struct SedonaGdalApi* api,
                                                  char* err_msg, int len);

#ifdef __cplusplus
}
#endif

#endif
