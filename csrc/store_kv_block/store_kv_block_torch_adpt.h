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

#ifndef STORE_KV_BLOCK_TORCH_ADPT_H
#define STORE_KV_BLOCK_TORCH_ADPT_H
#include <climits>  
namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor, at::Tensor> store_kv_block_pre(
    const at::Tensor &slot_mapping_npu,
    at::IntArrayRef slot_mapping_list,
    int64_t block_size)
{

    int64_t slot_mapping_len =  slot_mapping_list.size();

    std::vector<int32_t> length(16, 0);
    std::vector<int32_t> key_idx(16, 0);
    std::vector<int32_t> key_cache_idx(16, 0);
    int32_t idx_slotmap = 0;
    int32_t idx_groups = 0;

    while (idx_slotmap < slot_mapping_len) {


        int32_t current_idx = slot_mapping_list[idx_slotmap];
        if(current_idx <0){
            idx_slotmap++;
            continue;
        }

        int32_t block_id = current_idx / block_size;       
        int32_t y= current_idx % block_size;

        key_idx[idx_groups] = idx_slotmap;                      
        key_cache_idx[idx_groups] = current_idx;    

        int32_t j = idx_slotmap;

        if(j+1 < slot_mapping_len &&slot_mapping_list[j+1]!=slot_mapping_list[j]+1 ) {
            j++;

        }else{
            int32_t idx_stride = std::min(block_size-y,slot_mapping_len-idx_slotmap)-1;
            int32_t expected_last =  current_idx + idx_stride;
            int32_t expected_last_idx = idx_slotmap + (expected_last-current_idx);

            if (expected_last == slot_mapping_list[expected_last_idx]){
                j = expected_last_idx+1;
            }else{
                while(j+1 < slot_mapping_len && slot_mapping_list[j] / block_size == block_id && slot_mapping_list[j+1] ==slot_mapping_list[j]+1) {
                    j++;
                }
            }
        }

        length[idx_groups] = (j - idx_slotmap);
        idx_slotmap = j;
        idx_groups++;


        if(idx_groups>=length.capacity()){
            int32_t new_capacity = length.capacity() * 2;
            length.reserve(new_capacity);
            key_idx.reserve(new_capacity);
            key_cache_idx.reserve(new_capacity);

            for (int32_t k = idx_groups; k < new_capacity; ++k){
                length.emplace_back(0);
                key_idx.emplace_back(0);
                key_cache_idx.emplace_back(0);
            } 
        }
    }

    at::Tensor group_len = at::empty({idx_groups},
        at::TensorOptions(slot_mapping_npu.options().device()).dtype(torch::kInt32)
        );
    void* group_len_addr = group_len.data_ptr();

    at::Tensor group_key_idx = at::empty({idx_groups},
        at::TensorOptions(slot_mapping_npu.options().device()).dtype(torch::kInt32)
        );
    void* group_key_idx_addr = group_key_idx.data_ptr();
    
    at::Tensor group_key_cache_idx = at::empty({idx_groups},
        at::TensorOptions(slot_mapping_npu.options().device()).dtype(torch::kInt32)
        );
    void* group_key_cache_idx_addr = group_key_cache_idx.data_ptr();

    uint32_t device_size=idx_groups*sizeof(length[0]);
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    aclrtMemcpyKind memcpy_type=ACL_MEMCPY_HOST_TO_DEVICE;
    aclrtMemcpyAsync(group_len_addr, device_size, &length[0], device_size, ACL_MEMCPY_HOST_TO_DEVICE, stream); 
    aclrtMemcpyAsync(group_key_idx_addr, device_size, &key_idx[0], device_size, ACL_MEMCPY_HOST_TO_DEVICE, stream);  
    aclrtMemcpyAsync(group_key_cache_idx_addr, device_size, &key_cache_idx[0], device_size, ACL_MEMCPY_HOST_TO_DEVICE, stream);   

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(group_len, group_key_idx, group_key_cache_idx);

}

void store_kv_block(
    const at::Tensor &key_in,
    const at::Tensor &key_cache_in,
    const at::Tensor &group_len,
    const at::Tensor &group_key_idx,
    const at::Tensor &group_key_cache_idx,
    int64_t block_size)
{

    EXEC_NPU_CMD(aclnnStoreKVBlock, key_in, key_cache_in,group_len, group_key_idx, group_key_cache_idx, block_size);
    
} 

}
#endif