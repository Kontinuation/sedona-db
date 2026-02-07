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

#include "gdal_dyn.h"

#if defined(_WIN32)
#define TARGETING_WINDOWS
#include <tchar.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <errno.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int CPLErr;
typedef int OGRErr;

#define CE_None 0
#define GDT_Byte 1

#ifdef TARGETING_WINDOWS
static void win32_get_last_error(char* err_msg, int len) {
  wchar_t info[256];
  unsigned int error_code = GetLastError();
  int info_length = FormatMessageW(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, /* flags */
      NULL,                                                       /* message source*/
      error_code,                                /* the message (error) ID */
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), /* default language */
      info,                                      /* the buffer */
      sizeof(info) / sizeof(wchar_t),            /* size in wchars */
      NULL);
  int num_bytes =
      WideCharToMultiByte(CP_UTF8, 0, info, info_length, err_msg, len, NULL, NULL);
  num_bytes = (num_bytes < (len - 1)) ? num_bytes : (len - 1);
  err_msg[num_bytes] = '\0';
}
#endif

static void* try_load_gdal_symbol(void* handle, const char* func_name) {
#ifndef TARGETING_WINDOWS
  return dlsym(handle, func_name);
#else
  return GetProcAddress((HMODULE)handle, func_name);
#endif
}

struct SedonaGdalShim;

struct SedonaGdalShim {
  GDALDatasetH (*mem_dataset_create)(const char* name, int x_size, int y_size,
                                     int band_count, GDALDataType data_type,
                                     char** options);
  CPLErr (*gdal_add_band)(GDALDatasetH dataset, GDALDataType data_type, char** options);
  int (*gdal_get_data_type_size_bytes)(GDALDataType data_type);
  void (*gdal_close)(GDALDatasetH dataset);
  void (*csl_destroy)(char** options);
  char** (*csl_set_name_value)(char** list, const char* name, const char* value);
  void (*cpl_print_pointer)(char* buf, const void* ptr, int len);
};

static struct SedonaGdalShim* g_sedona_gdal_shim = NULL;

static void sedona_gdal_dyn_release_api(struct SedonaGdalApi* api) {
  if (api->shim != NULL) {
    if (api->shim == g_sedona_gdal_shim) {
      g_sedona_gdal_shim = NULL;
    }
    free(api->shim);
  }
  if (api->private_data != NULL) {
#ifdef TARGETING_WINDOWS
    FreeLibrary((HMODULE)api->private_data);
#else
    dlclose(api->private_data);
#endif
  }
  memset(api, 0, sizeof(struct SedonaGdalApi));
}

static int load_symbol_any(void* handle, const char** names, void** func, char* err_msg,
                           int len) {
  for (int i = 0; names[i] != NULL; ++i) {
    void* candidate = try_load_gdal_symbol(handle, names[i]);
    if (candidate != NULL) {
      *func = candidate;
      return 0;
    }
  }
#ifndef TARGETING_WINDOWS
  snprintf(err_msg, len, "%s", dlerror());
#else
  win32_get_last_error(err_msg, len);
#endif
  return -1;
}

static int load_gdal_shim(struct SedonaGdalShim* shim, void* handle, char* err_msg,
                          int len) {
  const char* mem_create_symbols[] = {
      "_ZN10MEMDataset6CreateEPKciii12GDALDataTypePPc",
      "_ZN10MEMDataset6CreateEPKciii12GDALDataTypePPKc",
      "?Create@MEMDataset@@SAPEAV1@PEBDHHH4GDALDataType@@PEAPEAD@Z", NULL};
  const char* csl_set_name_value_symbols[] = {"CSLSetNameValue", NULL};
  const char* csl_destroy_symbols[] = {"CSLDestroy", NULL};
  const char* cpl_print_pointer_symbols[] = {"CPLPrintPointer", NULL};
  const char* gdal_add_band_symbols[] = {"GDALAddBand", NULL};
  const char* gdal_get_data_type_size_bytes_symbols[] = {"GDALGetDataTypeSizeBytes",
                                                         NULL};
  const char* gdal_close_symbols[] = {"GDALClose", NULL};

  char mem_err[256];
  if (load_symbol_any(handle, mem_create_symbols, (void**)&shim->mem_dataset_create,
                      mem_err, (int)sizeof(mem_err)) != 0) {
    snprintf(err_msg, len, "Failed to resolve MEMDataset::Create: %s", mem_err);
    return -1;
  }
  if (load_symbol_any(handle, gdal_add_band_symbols, (void**)&shim->gdal_add_band,
                      err_msg, len) != 0) {
    return -1;
  }
  if (load_symbol_any(handle, gdal_get_data_type_size_bytes_symbols,
                      (void**)&shim->gdal_get_data_type_size_bytes, err_msg, len) != 0) {
    return -1;
  }
  if (load_symbol_any(handle, gdal_close_symbols, (void**)&shim->gdal_close, err_msg,
                      len) != 0) {
    return -1;
  }
  if (load_symbol_any(handle, csl_set_name_value_symbols,
                      (void**)&shim->csl_set_name_value, err_msg, len) != 0) {
    return -1;
  }
  if (load_symbol_any(handle, csl_destroy_symbols, (void**)&shim->csl_destroy, err_msg,
                      len) != 0) {
    return -1;
  }
  if (load_symbol_any(handle, cpl_print_pointer_symbols, (void**)&shim->cpl_print_pointer,
                      err_msg, len) != 0) {
    return -1;
  }

  return 0;
}

static int load_gdal_from_handle(struct SedonaGdalApi* api, void* handle, char* err_msg,
                                 int len);

static int load_gdal_from_current_process(struct SedonaGdalApi* api, char* err_msg,
                                          int len) {
#ifndef TARGETING_WINDOWS
  void* handle = dlopen(NULL, RTLD_LOCAL | RTLD_NOW);
  if (handle == NULL) {
    snprintf(err_msg, len, "%s", dlerror());
    return -1;
  }
  int result = load_gdal_from_handle(api, handle, err_msg, len);
  if (result != 0) {
    dlclose(handle);
  }
  return result;
#else
  HMODULE module = GetModuleHandleW(NULL);
  if (module == NULL) {
    win32_get_last_error(err_msg, len);
    return -1;
  }
  return load_gdal_from_handle(api, module, err_msg, len);
#endif
}

static GDALDatasetH sedona_gdal_mem_create_internal_impl(int x_size, int y_size,
                                                         int band_count,
                                                         const GDALDataType* band_types,
                                                         const void* const* band_data,
                                                         const GSpacing* pixel_offsets,
                                                         const GSpacing* line_offsets) {
  struct SedonaGdalShim* shim = g_sedona_gdal_shim;
  if (shim == NULL) {
    return NULL;
  }
  if (x_size <= 0 || y_size <= 0 || band_count < 0 || band_data == NULL ||
      band_types == NULL) {
    return NULL;
  }

  GDALDatasetH dataset = shim->mem_dataset_create("", x_size, y_size, 0, GDT_Byte, NULL);
  if (dataset == NULL) {
    return NULL;
  }

  for (int i = 0; i < band_count; ++i) {
    const void* band_ptr = band_data[i];
    const GDALDataType band_type = band_types[i];
    if (band_ptr == NULL) {
      shim->gdal_close(dataset);
      return NULL;
    }

    const GSpacing pixel = pixel_offsets ? pixel_offsets[i] : 0;
    const GSpacing line = line_offsets ? line_offsets[i] : 0;
    const GSpacing resolved_pixel =
        pixel > 0 ? pixel : (GSpacing)shim->gdal_get_data_type_size_bytes(band_type);
    const GSpacing resolved_line = line > 0 ? line : resolved_pixel * (GSpacing)x_size;

    char data_ptr_str[64];
    shim->cpl_print_pointer(data_ptr_str, band_ptr, (int)sizeof(data_ptr_str));

    char pixel_str[32];
    char line_str[32];
    snprintf(pixel_str, sizeof(pixel_str), "%lld", (long long)resolved_pixel);
    snprintf(line_str, sizeof(line_str), "%lld", (long long)resolved_line);

    char** options = NULL;
    options = shim->csl_set_name_value(options, "DATAPOINTER", data_ptr_str);
    options = shim->csl_set_name_value(options, "PIXELOFFSET", pixel_str);
    options = shim->csl_set_name_value(options, "LINEOFFSET", line_str);

    const CPLErr err = shim->gdal_add_band(dataset, band_type, options);
    shim->csl_destroy(options);
    if (err != CE_None) {
      shim->gdal_close(dataset);
      return NULL;
    }
  }

  return dataset;
}

static int load_gdal_from_handle(struct SedonaGdalApi* api, void* handle, char* err_msg,
                                 int len) {
  struct SedonaGdalShim* shim = calloc(1, sizeof(struct SedonaGdalShim));
  if (shim == NULL) {
    snprintf(err_msg, len, "%s", "Cannot allocate sedona gdal shim");
    return -1;
  }

  if (load_gdal_shim(shim, handle, err_msg, len) != 0) {
    free(shim);
    return -1;
  }

  g_sedona_gdal_shim = shim;
  api->sedona_gdal_mem_create_internal = &sedona_gdal_mem_create_internal_impl;
  api->release = &sedona_gdal_dyn_release_api;
  api->private_data = handle;
  api->shim = shim;

  return 0;
}

int sedona_gdal_dyn_api_init(struct SedonaGdalApi* api, const char* shared_object_path,
                             char* err_msg, int len) {
#ifndef TARGETING_WINDOWS
  void* handle = dlopen(shared_object_path, RTLD_LOCAL | RTLD_NOW);
  if (handle == NULL) {
    snprintf(err_msg, len, "%s", dlerror());
    return -1;
  }
#else
  int num_chars = MultiByteToWideChar(CP_UTF8, 0, shared_object_path, -1, NULL, 0);
  wchar_t* wpath = calloc(num_chars, sizeof(wchar_t));
  if (wpath == NULL) {
    snprintf(err_msg, len, "%s", "Cannot allocate memory for wpath");
    return -1;
  }
  MultiByteToWideChar(CP_UTF8, 0, shared_object_path, -1, wpath, num_chars);
  HMODULE module = LoadLibraryW(wpath);
  free(wpath);
  if (module == NULL) {
    win32_get_last_error(err_msg, len);
    return -1;
  }
  void* handle = module;
#endif
  int result = load_gdal_from_handle(api, handle, err_msg, len);
#ifndef TARGETING_WINDOWS
  if (result != 0) {
    dlclose(handle);
  }
#endif
  return result;
}

int sedona_gdal_dyn_api_init_from_current_process(struct SedonaGdalApi* api,
                                                  char* err_msg, int len) {
  if (load_gdal_from_current_process(api, err_msg, len) == 0) {
    return 0;
  }

#ifndef TARGETING_WINDOWS
  const char* candidates[] = {"libgdal.dylib",
                              "libgdal.so",
                              "libgdal.so.3",
                              "libgdal.so.2",
                              "libgdal.32.3.6.4.dylib",
                              "libgdal.32.dylib",
                              "libgdal.31.dylib",
                              "libgdal.30.dylib",
                              "gdal.dll",
                              NULL};
  for (int i = 0; candidates[i] != NULL; ++i) {
    void* handle = dlopen(candidates[i], RTLD_LOCAL | RTLD_NOW);
    if (handle == NULL) {
      continue;
    }
    int result = load_gdal_from_handle(api, handle, err_msg, len);
    if (result == 0) {
      return 0;
    }
    dlclose(handle);
  }
#else
  const wchar_t* candidates[] = {L"gdal.dll", NULL};
  for (int i = 0; candidates[i] != NULL; ++i) {
    HMODULE module = LoadLibraryW(candidates[i]);
    if (module == NULL) {
      continue;
    }
    if (load_gdal_from_handle(api, module, err_msg, len) == 0) {
      return 0;
    }
    FreeLibrary(module);
  }
#endif

  return -1;
}
