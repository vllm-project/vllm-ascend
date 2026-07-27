/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
#ifndef STORE_KV_BLOCK_METADATA_TORCH_ADPT_H
#define STORE_KV_BLOCK_METADATA_TORCH_ADPT_H
// #include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

// Compute grouping metadata (group_len / group_key_idx / group_key_cache_idx)
// for slot_mapping on AICPU. The AICPU kernel reads slot_mapping directly from
// device memory, so the host-side slot_mapping_list is no longer needed.
//
// Outputs are pre-allocated with the same length as slot_mapping and zero-filled
// by the kernel for unused entries. The caller can detect the actual group count
// by scanning for the first zero group_len entry.
std::tuple<at::Tensor, at::Tensor, at::Tensor> store_kv_block_metadata(
    const at::Tensor &slot_mapping_npu,
    int64_t block_size)
{
    TORCH_CHECK(slot_mapping_npu.numel() > 0, "Tensor slot_mapping_npu is empty.");
    TORCH_CHECK(block_size > 0, "block_size must be positive, but got ", block_size);

    int64_t slot_mapping_len = slot_mapping_npu.numel();
    auto opts = at::TensorOptions(slot_mapping_npu.options().device()).dtype(torch::kInt32);

    // Worst-case group count equals slot_mapping_len (each slot is its own group).
    at::Tensor group_len = at::empty({slot_mapping_len}, opts);
    at::Tensor group_key_idx = at::empty({slot_mapping_len}, opts);
    at::Tensor group_key_cache_idx = at::empty({slot_mapping_len}, opts);

    EXEC_NPU_CMD(aclnnStoreKvBlockMetadata,
                 slot_mapping_npu,
                 block_size,
                 group_len,
                 group_key_idx,
                 group_key_cache_idx);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(
        group_len, group_key_idx, group_key_cache_idx);
}

}  // namespace vllm_ascend

#endif  // STORE_KV_BLOCK_METADATA_TORCH_ADPT_H
