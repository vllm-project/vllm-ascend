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

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>
#include <pybind11/pybind11.h>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclnn/opdev/platform.h"
#include "ops.h"
#include "utils.h"

namespace vllm_ascend {

void rotary_embedding(at::Tensor &positions, at::Tensor &query, at::Tensor &key,
    int64_t head_size, at::Tensor &cos_sin_cache,  bool is_neox)
{
    int32_t deviceId = 0;
    int64_t num_tokens = positions.numel();
    int positions_ndim = positions.dim();
    TORCH_CHECK(
        positions_ndim == 1 || positions_ndim == 2,
        "positions must have shape [num_tokens] or [batch_size, seq_len]");
    if (positions_ndim == 1) {
      TORCH_CHECK(
          query.size(0) == positions.size(0) && key.size(0) == positions.size(0),
          "query, key and positions must have the same number of tokens");
    }
    if (positions_ndim == 2) {
      TORCH_CHECK(
          query.size(0) == positions.size(0) &&
              key.size(0) == positions.size(0) &&
              query.size(1) == positions.size(1) &&
              key.size(1) == positions.size(1),
          "query, key and positions must have the same batch_size and seq_len");
    }

    int query_hidden_size = query.numel() / num_tokens;
    int key_hidden_size = key.numel() / num_tokens;
    TORCH_CHECK(query_hidden_size % head_size == 0);
    TORCH_CHECK(key_hidden_size % head_size == 0);

    // Make sure query and key have consistent number of heads
    int num_heads = query_hidden_size / head_size;
    int num_kv_heads = key_hidden_size / head_size;
    TORCH_CHECK(num_heads % num_kv_heads == 0);

    int rot_dim = cos_sin_cache.size(1);
    int64_t *position_ids_ptr = positions.data_ptr<int64_t>();
    void *query_ptr = query.data_ptr();
    void *key_ptr = key.data_ptr();
    void *cos_sin_cache_ptr = cos_sin_cache.data_ptr();
    int64_t query_stride = query.stride(-2);
    int64_t key_stride = key.stride(-2);
    at::ScalarType scalar_type = query.scalar_type();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("rotary_embedding");
    cmd.SetCustomHandler([scalar_type, is_neox, num_tokens, stream, position_ids_ptr, 
                          query_ptr, key_ptr, cos_sin_cache_ptr, rot_dim, query_stride, key_stride,
                          num_heads, num_kv_heads, head_size]() -> int {
        auto dtype_num = get_dtype_from_torch(scalar_type);
        fe::PlatFormInfos platform_infos;
        int device_id = 0;
        fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos);
        uint32_t aivNum = platform_infos.GetCoreNumByType("aiv");
        uint32_t loop_cnt = (num_tokens + aivNum - 1) / aivNum;
        rotary_embedding_impl(dtype_num, is_neox, stream, position_ids_ptr, query_ptr, key_ptr, query_ptr,
                                key_ptr, cos_sin_cache_ptr, rot_dim, query_stride, key_stride, query_stride,
                                key_stride, num_heads, num_kv_heads, head_size, num_tokens, loop_cnt, aivNum);
        return 0;
    });
    cmd.Run();
    return ;
}
} // namespace vllm_ascend

TORCH_LIBRARY_EXPAND(_C, ops)
{
    // vLLM-Ascend custom ops

    // Rotary embedding
    // Apply GPT-NeoX style rotary embedding to query and key.
    ops.def(
        "rotary_embedding(Tensor positions, Tensor! query,"
        "                 Tensor! key, int head_size,"
        "                 Tensor cos_sin_cache, bool is_neox) -> ()");
    ops.impl("rotary_embedding", torch::kPrivateUse1, &vllm_ascend::rotary_embedding);
}

REGISTER_EXTENSION(_C)
