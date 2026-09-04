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
 * \file incre_flash_attention.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#if (__NPU_ARCH__ == 5102)
#include "../prompt_flash_attention/arch38/prompt_flash_attention_entry_regbase.h"
#else
#include "incre_flash_attention_arch22.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void
incre_flash_attention_FIAS(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask, 
                        __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, 
                        __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale,
                        __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *blocktable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
                        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
                        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
                        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *attentionOut,
                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if (__NPU_ARCH__ != 5102)
    incre_flash_attention_FIAS_arch22(query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, deqScale1, quantScale1,
                                    deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blocktable, queryPaddingSize,
                                    kvPaddingSize, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, keySharedPrefix,
                                    valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, keyRopeAntiquantScale, dequantScaleQuery,
                                    attentionOut, softmaxLse, workspace, tiling);
#endif
}
extern "C" __global__ __aicore__ void
incre_flash_attention(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
                      __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *deqScale1,
                      __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
                      __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                      __gm__ uint8_t *blocktable, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut,
                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
#if (__NPU_ARCH__ == 5102)
    prompt_flash_attention_FIAS_regbase(query, key, value, pseShift, attenMask, nullptr, actualSeqLengths, deqScale1,
                                        quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blocktable, nullptr,
                                        kvPaddingSize, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                        attentionOut, nullptr, workspace, tiling);
#else
    incre_flash_attention_FIAS(query, key, value, pseShift, attenMask, nullptr, actualSeqLengths, deqScale1, quantScale1,
                            deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blocktable, nullptr,
                            kvPaddingSize, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                            attentionOut, nullptr, workspace, tiling);
#endif
}