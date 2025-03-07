/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>
#include "kernels/types.h"

extern void rotary_embedding_kernel(
  ascend_type type,
  bool is_neox,
  void* stream,
  int64_t* positions,
  void* query_dst,
  void* key_dst,
  void* query,
  void* key,
  void* cos_sin_cache,
  const int rot_dim,
  const int64_t query_stride,
  const int64_t key_stride,
  const int64_t dst_query_stride,
  const int64_t dst_key_stride,
  const int num_heads,
  const int num_kv_heads,
  const int head_size,
  const int64_t num_tokens,
  const int64_t loop_cnt,
  int aivNum);