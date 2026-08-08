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
 * \file fused_infer_attention_score_apt.cpp 
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
// ifa must include before pfa
#define FIA_ENABLE_MLA
#include "fused_infer_attention_score_template_tiling_key.h"
#ifdef NOT_DYNAMIC_COMPILE
#if __has_include("../../incre_flash_attention/op_kernel/arch35/incre_flash_attention_entry_regbase.h")
#include "../../incre_flash_attention/op_kernel/arch35/incre_flash_attention_entry_regbase.h"
#include "../../prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_entry_regbase.h"
#else
#include "../incre_flash_attention/arch35/incre_flash_attention_entry_regbase.h"
#include "../prompt_flash_attention/arch35/prompt_flash_attention_entry_regbase.h"
#endif
#else
#include "../incre_flash_attention/incre_flash_attention.cpp"
#include "../prompt_flash_attention/prompt_flash_attention.cpp"
#endif
#include "fused_infer_attention_score_tilingkey.h"

#define FullQuantTiling 15
template <uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode, bool hasAttenMask, bool hasRope, 
  uint8_t KvLayoutType, bool isFd, bool emptyTensor, bool enableKVPrefix, bool enableS1OutSplit, bool isReconstructTemp>
__global__ __aicore__ void fused_infer_attention_score(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                    __gm__ uint8_t* pse_shift, __gm__ uint8_t* attenMask,
                                                    __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
                                                    __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1,
                                                    __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
                                                    __gm__ uint8_t* quant_offset2, __gm__ uint8_t* antiquantScale,
                                                    __gm__ uint8_t* antiquantOffset, __gm__ uint8_t* blocktable,
                                                    __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                                    __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset,
                                                    __gm__ uint8_t* valueAntiquantScale, __gm__ uint8_t* valueAntiquantOffset,
                                                    __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix,
                                                    __gm__ uint8_t* actualSharedPrefixLen, __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope,
                                                    __gm__ uint8_t* keyRopeAntiquantScale, __gm__ uint8_t* dequantScaleQuery,
                                                    __gm__ uint8_t* learnableSink, __gm__ uint8_t* qStartIdx, __gm__ uint8_t* kvStartIdx,
                                                    __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    if (quantMode >= FullQuantTiling) {
        //pfa 模板
        prompt_flash_attention_FIAS_regbase<inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, KvLayoutType, isFd, emptyTensor, enableKVPrefix, enableS1OutSplit,isReconstructTemp>(
                                    query, key, value, pse_shift, attenMask, actualSeqLengths, actualSeqLengthsKV, deq_scale1, quant_scale1,                                    
                                    deq_scale2, quant_scale2, quant_offset2, antiquantScale, antiquantOffset, blocktable, queryPaddingSize,
                                    kvPaddingSize, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,                                    
                                    keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope,
                                    dequantScaleQuery, learnableSink, attentionOut, softmaxLse, workspace, tiling);
    } else {
        //ifa 模板
        incre_flash_attention_FIAS_regbase<inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope, KvLayoutType, isFd, emptyTensor, enableKVPrefix, enableS1OutSplit>(
                                    query, key, value, pse_shift, attenMask, actualSeqLengths, actualSeqLengthsKV, deq_scale1, quant_scale1,
                                    deq_scale2, quant_scale2, quant_offset2, antiquantScale, antiquantOffset, blocktable, queryPaddingSize,
                                    kvPaddingSize, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, keySharedPrefix,
                                    valueSharedPrefix, actualSharedPrefixLen, attentionOut, softmaxLse, workspace, tiling);
    }
}