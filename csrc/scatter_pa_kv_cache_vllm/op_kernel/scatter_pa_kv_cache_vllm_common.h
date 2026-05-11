/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_pa_kv_cache_vllm_common.h
 * \brief
 */
#ifndef SCATTER_PA_KV_CACHE_VLLM_COMMON_H
#define SCATTER_PA_KV_CACHE_VLLM_COMMON_H


namespace ScatterPaKvCacheVllm {
/**
 * Get the block size of unified buffer in bytes
 */
__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

constexpr int32_t BLOCK_SIZE = GetUbBlockSize();
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t SCALAR_INPUT_PARAMETERS = 4; // 算子输入input3~7是四个int32类型的参数
constexpr uint32_t COPY_NUM = 512;              // 用于设置计算rope压缩均值前的搬运大小
constexpr uint32_t MAX_UB_SIZE = 192 * 1024;    // UB size
constexpr uint32_t TASK_MULTIPLE = 2;           // Compress_rope模式下KV分核，分核任务量翻倍
constexpr int16_t MAX_MASK_NUM = 64;            // count模式下ADD计算最多处理256B，对应float元素为64个
constexpr int16_t MIN_MASK_NUM = 32; // 当前约束headsize为32的整数倍，设置32为了约束范围内尽可能提高计算带宽
constexpr int32_t MAX_FLOAT_NUM = 30 * 1024; // 考虑充分利用UB空间，localtensor可申请的最大浮点数个数
constexpr int32_t PER_FLOAT_NUM = 15 * 1024;       // 开启doublebuffer后，一个buffer内的浮点数个数
constexpr uint32_t TOTAL_BUFFER_SIZE = 180 * 1024; // 留出一部分空间给标量
constexpr int32_t MAX_UB_USED = MAX_UB_SIZE / (BUFFER_NUM * 2);
} // namespace ScatterPaKvCacheVllm
#endif
