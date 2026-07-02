/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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

#include <cstdint>
#include <string>

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>

namespace vllm_ascend {

int64_t zb_init(int64_t rank, int64_t world_size, int64_t local_mem_size, const std::string &server_ip_port);
int64_t zb_alloc(int64_t element_count, int64_t element_size);
at::Tensor zb_alloc_tensor(c10::ArrayRef<int64_t> shape, at::ScalarType dtype, const std::string &device);
at::Tensor zb_alias_tensor(const at::Tensor &base, c10::ArrayRef<int64_t> shape, at::ScalarType dtype);
void zb_free(int64_t ptr);
void zb_finalize();
int64_t zb_get_ext_info();
bool zb_is_initialized();

} // namespace vllm_ascend
