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
 * \file incre_flash_attention_apt.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "arch35/incre_flash_attention_template_tiling_key.h"
#include "arch35/incre_flash_attention_entry_regbase.h"
#ifdef NOT_DYNAMIC_COMPILE
#include "../../prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_entry_regbase.h"
#else
#include "../prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_entry_regbase.h"
#endif
using namespace AscendC;

template<uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode, bool hasAttenMask, bool hasRope, 
  bool isPa, bool isFd, bool emptyTensor, bool enableKVPrefix, bool enableS1OutSplit>
 __global__ __aicore__ void
incre_flash_attention_FIAS(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
                        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                        __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
                        __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                        __gm__ uint8_t *blocktable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
                        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
                        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
                        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
                        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *attentionOut,
                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    incre_flash_attention_FIAS_regbase<inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, enableKVPrefix, enableS1OutSplit>(
                                    query, key, value, pseShift, attenMask, actualSeqLengthsQ, actualSeqLengths, deqScale1, quantScale1,
                                    deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blocktable, queryPaddingSize,
                                    kvPaddingSize, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, keySharedPrefix,
                                    valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, workspace, tiling);
}
template<uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode, bool hasAttenMask, bool hasRope, 
  bool isPa, bool isFd, bool emptyTensor, bool enableKVPrefix, bool enableS1OutSplit>
 __global__ __aicore__ void
incre_flash_attention(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
                      __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *deqScale1,
                      __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
                      __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
                      __gm__ uint8_t *blocktable, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *attentionOut,
                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    if (quantMode >= 15) {
        prompt_flash_attention_FIAS_regbase<inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, enableKVPrefix, enableS1OutSplit>(
                                            query, key, value, pseShift, attenMask, nullptr, actualSeqLengths, deqScale1,
                                            quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blocktable, nullptr,
                                            kvPaddingSize, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, attentionOut, nullptr, workspace, tiling);
    } else {
        incre_flash_attention_FIAS<inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, isPa, isFd, emptyTensor, enableKVPrefix, enableS1OutSplit>(
                                    query, key, value, pseShift, attenMask, nullptr, actualSeqLengths, deqScale1, quantScale1,
                                    deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset, blocktable, nullptr,
                                    kvPaddingSize, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                    attentionOut, nullptr, workspace, tiling);
    }
}