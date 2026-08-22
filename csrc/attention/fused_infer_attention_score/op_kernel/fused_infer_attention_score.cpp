/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file fused_infer_attention_score.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/quantization/ascend_quant.h"
#include "kernel_operator_list_tensor_intf.h"
// ifa must include before pfa
#define FIA_ENABLE_MLA

#include "fused_infer_attention_score_tilingkey.h"

#if (__NPU_ARCH__ == 5102)
#ifdef NOT_DYNAMIC_COMPILE
#if __has_include("../../prompt_flash_attention/op_kernel/arch38/prompt_flash_attention_entry_regbase.h")
#include "../../prompt_flash_attention/op_kernel/arch38/prompt_flash_attention_entry_regbase.h"
#include "../../incre_flash_attention/op_kernel/incre_flash_attention.cpp"
#else
#include "../prompt_flash_attention/arch38/prompt_flash_attention_entry_regbase.h"
#include "../incre_flash_attention/incre_flash_attention.cpp"
#endif
#else
#include "../prompt_flash_attention/arch38/prompt_flash_attention_entry_regbase.h"
#include "../incre_flash_attention/incre_flash_attention.cpp"
#endif

#else
#ifdef NOT_DYNAMIC_COMPILE
#include "../../incre_flash_attention/op_kernel/incre_flash_attention_arch22.h"
#include "../../prompt_flash_attention/op_kernel/prompt_flash_attention_arch22.h"
#else
#include "../incre_flash_attention/incre_flash_attention.cpp"
#include "../prompt_flash_attention/prompt_flash_attention.cpp"
#endif
#include "fused_infer_attention_score_v3.cpp"
#include "flash_attention_interface.cpp"
#endif

#define FullQuantTiling 15
extern "C" __global__ __aicore__ void fused_infer_attention_score(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
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
#if (__NPU_ARCH__ == 5102)
    prompt_flash_attention_FIAS_regbase(query, key, value, pse_shift, attenMask, actualSeqLengths, actualSeqLengthsKV,
        deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, antiquantScale, antiquantOffset, blocktable,
        queryPaddingSize, kvPaddingSize, keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale,
        valueAntiquantOffset, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope,
        dequantScaleQuery, attentionOut, softmaxLse, workspace, tiling);
#else
    if (TILING_KEY_VAR >= FAI_FLAG_TILING) {
        __gm__ uint8_t *user = GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        FIA_COPY_TILING_DATA(FAInferTilingData, tiling);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING);

        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_SWAMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_SWAMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_SWAMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_SWAMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_SWAMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_SWAMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_SWAMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_SWAMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SPLITFUSE_TILING);

        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_FULLMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_FULLMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_FULLMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_FULLMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_FULLMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_FULLMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_FULLMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_FULLMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_FULLMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_FULLMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_FULLMASK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_FULLMASK_SPLITFUSE_TILING);

        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_SWAMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_SWAMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_SWAMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_SWAMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_SWAMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_SWAMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_SWAMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_SWAMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SINK_SPLITFUSE_TILING);

        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING);

        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING_FD);

        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING_FD);

        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING_FD);

        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING_FD);

        // Decoding 场景 (pagedCacheFlag == true && qSeqlen == 1 && NO_MASK && !lseFlag)
        TILING_KEY_IS(QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_DECODING_TILING);
        TILING_KEY_IS(QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_DECODING_TILING);

        #if TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, half, false, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                half, half, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_SWAMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_FULLMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::FULL_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                half, half, float, true, false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_SWAMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                half, half, float, true, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_FULLMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::FULL_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_FULLMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                half, half, float, true, false, FaiKernel::MaskType::FULL_MASK, FaiKernel::inputLayout::TND>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_SWAMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_FULLMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::FULL_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN, bfloat16_t, bfloat16_t, float, true,
                false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_SWAMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_FULLMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::FULL_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_FULLMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN, bfloat16_t, bfloat16_t, float, true,
                false, FaiKernel::MaskType::FULL_MASK, FaiKernel::inputLayout::TND>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_LOW_PREC_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, half, false, false, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                half, half, float, true, false, FaiKernel::MaskType::NO_MASK,
                FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_SWAMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::MASK_SWA,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_FULLMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::FULL_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                half, half, float, true, false, FaiKernel::MaskType::MASK_CAUSAL,
                FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_SWAMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::MASK_SWA,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                half, half, float, true, false, FaiKernel::MaskType::MASK_SWA,
                FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_FULLMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::FULL_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_FULLMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                half, half, float, true, false, FaiKernel::MaskType::FULL_MASK,
                FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::NO_MASK,
                FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_SWAMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::MASK_SWA,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
         #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_FULLMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::FULL_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_CAUSAL,
                FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_SWAMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_SWA,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_SWA,
                FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_FULLMASK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::FULL_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_FULLMASK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::FULL_MASK,
                FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                half, half, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
                NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_SWAMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                half, half, float, true, false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
                NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_SWAMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SINK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                half, half, float, true, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND,
                NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING
            SplitFuse::FAInfer<layout::nZ, layout::zN,
                bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
                NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
                query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
                actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_SWAMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_SWAMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_SWA, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            half, half, float, true, false, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_SWAMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, false, FaiKernel::MaskType::MASK_SWA,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            half, half, float, true, false, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_SWAMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, false, FaiKernel::MaskType::MASK_SWA,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            half, half, float, true, false, FaiKernel::MaskType::MASK_SWA,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_SWAMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, false, FaiKernel::MaskType::MASK_SWA,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_SWAMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_SWA,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_SWAMASK_SINK_SPLITFUSE_TILING
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, false, FaiKernel::MaskType::MASK_SWA,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        //FD(IS_FD = true)
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            half, half, float, true, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            half, half, float, true, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            half, half, float, true, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            half, half, float, true, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            half, half, float, true, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            half, half, float, true, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::MASK_CAUSAL, FaiKernel::inputLayout::TND,
            NpuArch::Epilogue::LseMode::NONE, NpuArch::Epilogue::SinkMode::ENABLE>( 
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            half, half, float, true, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, false, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            half, half, float, true, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            half, half, float, true, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_NOMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::NO_MASK,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_NOCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, false, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::ColumnMajor, layout::RowMajor,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_LSEOUT_TND_PAGEDCACHE_KVNZ_CAUSALMASK_SINK_SPLITFUSE_TILING_FD
        SplitFuse::FAInfer<layout::nZ, layout::zN,
            bfloat16_t, bfloat16_t, float, true, true, FaiKernel::MaskType::MASK_CAUSAL,
            FaiKernel::inputLayout::TND, NpuArch::Epilogue::LseMode::OUT_ONLY, NpuArch::Epilogue::SinkMode::ENABLE>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);

        // Decoding 场景 (pagedCacheFlag == true && qSeqlen == 1 && NO_MASK && !lseFlag)
        #elif TILING_KEY_VAR == QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_DECODING_TILING
        SplitFuse::FAInferDecoding<half, half, float, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #elif TILING_KEY_VAR == QBF16_KVBF16_OUTBF16_NOLSEOUT_TND_PAGEDCACHE_NOMASK_DECODING_TILING
        SplitFuse::FAInferDecoding<bfloat16_t, bfloat16_t, float, true, FaiKernel::MaskType::NO_MASK, FaiKernel::inputLayout::TND>(
            query, key, value, pse_shift, attenMask, blocktable, attentionOut, softmaxLse,
            actualSeqLengths, actualSeqLengthsKV, user, tiling, learnableSink);
        #endif
    } else if (TILING_KEY_VAR >= PFA_FlAG_TILING) {
        prompt_flash_attention_FIAS_arch22(query, key, value, pse_shift, attenMask, actualSeqLengths,
                                    actualSeqLengthsKV, deq_scale1, quant_scale1,
                                    deq_scale2, quant_scale2, quant_offset2, antiquantScale, 
                                    antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize, 
                                    keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, 
                                    valueAntiquantOffset, keySharedPrefix, valueSharedPrefix, 
                                    actualSharedPrefixLen, queryRope, keyRope, dequantScaleQuery, learnableSink,
                                    attentionOut, softmaxLse, workspace, tiling);
    } else if (TILING_KEY_VAR >= FIA_FLAG_TILING) {
        fused_infer_attention(query, key, value, pse_shift, attenMask, actualSeqLengths,
                            actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                            quant_offset2, antiquantScale, antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize,
                            keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                            keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, keyRopeAntiquantScale,
                            learnableSink, attentionOut, softmaxLse, workspace, tiling);
    } else {
        incre_flash_attention_FIAS_arch22(query, key, value, pse_shift, attenMask, actualSeqLengths,
                                actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2,
                                quant_offset2, antiquantScale, antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize,
                                keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                                keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, keyRopeAntiquantScale, dequantScaleQuery,
                                attentionOut, softmaxLse, workspace, tiling);
    }
#endif
}