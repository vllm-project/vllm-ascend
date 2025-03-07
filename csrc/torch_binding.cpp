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

#include "cache.h"
#include "ops.h"
#include "core/registration.h"

#include <torch/library.h>

std::tuple<at::Tensor, at::Tensor> rotary_embedding(
  at::Tensor& positions,
  at::Tensor& query,
  at::Tensor& key,
  at::Tensor& cos_sin_cache,
  int64_t head_size,
  bool is_neox
) {
  // CHECK_ACL(aclInit(nullptr));
  int32_t deviceId = 0;
  int64_t num_tokens = query.numel() / query.size(-1);;
  // std::cout << " num tokens equals to: " << num_tokens << std::endl;
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int64_t* position_ids_ptr = positions.data_ptr<int64_t>();
  void* query_ptr = query.data_ptr();
  void* key_ptr = key.data_ptr();
  void* cos_sin_cache_ptr = cos_sin_cache.data_ptr();
  // int64_t parallel_cnt = num_tokens * num_kv_heads;
  at::Tensor query_dst = at::empty({num_tokens, num_heads, head_size}, query.options().dtype(q_offset.scalar_type()));
  at::Tensor key_dst = at::empty({num_tokens, num_kv_heads, head_size}, key.options().dtype(k_offset.scalar_type()));
  void* query_dst_ptr = query_dst.data_ptr();
  void* key_dst_ptr = key_dst.data_ptr();
  int64_t query_stride = query.stride(-2);
  int64_t key_stride = key.stride(-2);
  int64_t dst_query_stride = query_dst.stride(0);
  int64_t dst_key_stride = key_dst.stride(0);
  at::ScalarType scalar_type = query.scalar_type();
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at_npu::native::OpCommand cmd;
  cmd.Name("turbo_rope_quant");
  cmd.SetCustomHandler([
    scalar_type,
    is_neox,
    num_tokens,
    stream,
    position_ids_ptr,
    query_dst_ptr,
    key_dst_ptr,
    query_ptr,
    key_ptr,
    cos_sin_cache_ptr,
    rot_dim,
    query_stride,
    key_stride,
    dst_query_stride,
    dst_key_stride,
    num_heads,
    num_kv_heads,
    head_size
  ]() -> int {
      auto dtype_num = get_dtype_from_torch(scalar_type);
      fe::PlatFormInfos platform_infos;
      int device_id = 0;
      fe::PlatformInfoManager::GeInstance().GetRuntimePlatformInfosByDevice(device_id, platform_infos);
      uint32_t aivNum = platform_infos.GetCoreNumByType("aiv");
      uint32_t loop_cnt = (num_tokens + aivNum - 1) / aivNum;
      ascendc::rotary_embedding_quant(
        dtype_num,
        is_neox,
        stream,
        position_ids_ptr,
        query_dst_ptr,
        key_dst_ptr,
        query_ptr,
        key_ptr,
        cos_sin_cache_ptr,
        rot_dim,
        query_stride,
        key_stride,
        dst_query_stride,
        dst_key_stride,
        num_heads,
        num_kv_heads,
        head_size,
        num_tokens,
        loop_cnt,
        aivNum
      );
      return 0;
  });
  cmd.Run();
  return {query_dst, key_dst};
}


TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // vLLM-Ascend custom ops

  // Rotary embedding
  // Apply GPT-NeoX style rotary embedding to query and key.
  ops.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor! key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  ops.impl("rotary_embedding", torch::kPrivateUse1, &rotary_embedding);
}


REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
