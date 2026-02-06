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

#include "gdal_cpp_shim.h"

#include <string>

#include "cpl_conv.h"
#include "cpl_string.h"
#include "memdataset.h"

GDALDatasetH sedona_gdal_mem_create_internal(int x_size, int y_size, int band_count,
                                             const GDALDataType* band_types,
                                             const void* const* band_data,
                                             const GSpacing* pixel_offsets,
                                             const GSpacing* line_offsets) {
  if (x_size <= 0 || y_size <= 0 || band_count < 0 || band_data == nullptr ||
      band_types == nullptr) {
    return nullptr;
  }

  MEMDataset* dataset = MEMDataset::Create("", x_size, y_size, 0, GDT_Byte, nullptr);
  if (dataset == nullptr) {
    return nullptr;
  }

  for (int i = 0; i < band_count; ++i) {
    const void* band_ptr = band_data[i];
    const GDALDataType band_type = band_types[i];
    if (band_ptr == nullptr) {
      GDALClose(dataset);
      return nullptr;
    }

    const GSpacing pixel = pixel_offsets ? pixel_offsets[i] : 0;
    const GSpacing line = line_offsets ? line_offsets[i] : 0;
    const GSpacing resolved_pixel =
        pixel > 0 ? pixel : GDALGetDataTypeSizeBytes(band_type);
    const GSpacing resolved_line =
        line > 0 ? line : resolved_pixel * static_cast<GSpacing>(x_size);

    char data_ptr_str[64];
    CPLPrintPointer(data_ptr_str, const_cast<void*>(band_ptr),
                    static_cast<int>(sizeof(data_ptr_str)));

    const std::string pixel_str = std::to_string(static_cast<GIntBig>(resolved_pixel));
    const std::string line_str = std::to_string(static_cast<GIntBig>(resolved_line));

    char** options = nullptr;
    options = CSLSetNameValue(options, "DATAPOINTER", data_ptr_str);
    options = CSLSetNameValue(options, "PIXELOFFSET", pixel_str.c_str());
    options = CSLSetNameValue(options, "LINEOFFSET", line_str.c_str());

    const CPLErr err = dataset->AddBand(band_type, options);
    CSLDestroy(options);
    if (err != CE_None) {
      GDALClose(dataset);
      return nullptr;
    }
  }

  return GDALDataset::ToHandle(dataset);
}

void sedona_gdal_dataset_close(GDALDatasetH dataset) {
  if (dataset != nullptr) {
    GDALClose(dataset);
  }
}
