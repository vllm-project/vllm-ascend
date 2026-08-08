/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prompt_flash_attention.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#if (__NPU_ARCH__ == 5102)
#include "arch38/prompt_flash_attention_entry_regbase.h"
#else
#include "prompt_flash_attention_arch22.h"
#endif

extern "C" __global__ __aicore__ void prompt_flash_attention_FIAS(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                             __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                                             __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                             __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                             __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                             __gm__ uint8_t* quant_offset2, __gm__ uint8_t* antiquant_scale, __gm__ uint8_t* antiquant_offset,
                                                             __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                                             __gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset,
                                                             __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset,
                                                             __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                                             __gm__ uint8_t * queryRope, __gm__ uint8_t * keyRope, __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* learnableSink,
                                                             __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
                                                             __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    {
    #if (__NPU_ARCH__ == 5102)
        prompt_flash_attention_FIAS_regbase(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV,
            deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset,
            blocktable, queryPaddingSize, kvPaddingSize, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, 
            value_antiquant_offset, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, dequantScaleQuery, attentionOut,
            softmaxLse, workspace, tiling);    
    #else
        prompt_flash_attention_FIAS_arch22(query, key, value, pseShift, attenMask, actualSeqLengths,
            actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquant_scale, antiquant_offset,
            blocktable, queryPaddingSize, kvPaddingSize, key_antiquant_scale, key_antiquant_offset, value_antiquant_scale, 
            value_antiquant_offset, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, dequantScaleQuery,
            learnableSink, attentionOut, softmaxLse, workspace, tiling);
    #endif
    }    
}

extern "C" __global__ __aicore__ void prompt_flash_attention(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                             __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                                             __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                             __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                             __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                             __gm__ uint8_t* quant_offset2, __gm__ uint8_t* attentionOut,
                                                             __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    prompt_flash_attention_FIAS(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                                quant_offset2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr, nullptr, attentionOut, nullptr, workspace, tiling);
}