/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_common.h"

namespace vllm_ascend::fia_v2_sink_opapi {

thread_local char g_hash_buf[g_hash_buf_size];
thread_local int g_hash_offset = 0;
thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

const std::vector<std::string> g_custom_lib_path = get_custom_lib_path();
const std::vector<std::string> g_default_custom_lib_path = get_default_custom_lib_path();

void *GetOpApiFuncAddrFromFeatureLib(const char *api_name)
{
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_infer_handler, "libaclnn_ops_infer.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(ops_train_handler, "libaclnn_ops_train.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(math_handler, "libaclnn_math.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(sparse_handler, "libaclnn_sparse.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(fft_handler, "libaclnn_fft.so", api_name);
    GET_OP_API_FUNC_FROM_FEATURE_LIB(rand_handler, "libaclnn_rand.so", api_name);
    return nullptr;
}

c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape)
{
    c10::SmallVector<int64_t, SIZE> shape_small_vec;
    for (uint64_t i = 0; i < shape.size(); i++) {
        shape_small_vec.emplace_back(shape[i]);
    }

    return shape_small_vec;
}

}  // namespace vllm_ascend::fia_v2_sink_opapi
