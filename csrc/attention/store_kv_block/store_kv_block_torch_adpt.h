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
//  #include "../aclnn_torch_adapter/op_api_common.h"

#ifndef STORE_KV_BLOCK_TORCH_ADPT_H
#define STORE_KV_BLOCK_TORCH_ADPT_H
#include <climits>  
namespace vllm_ascend {

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