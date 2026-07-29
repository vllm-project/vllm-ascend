/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

/*!
 * \file custom_fused_infer_attention_v310.cpp
 * \brief
 */

#include "kernel_operator.h"

#if (__CCE_AICORE__ == 200)
#include "unpad_paged_attention_decoder.h"
#endif
using namespace AscendC;

#define INVOKE_IFA_NEW_GQA_OP_IMPL(templateClass, ...)                                                                 \
    do {                                                                                                               \
        templateClass<IFAType<__VA_ARGS__>> op;                                                                        \
        GET_TILING_DATA_WITH_STRUCT(IncreFlashAttentionTilingAtbDataV2, tiling_data_in, tiling);                       \
        const IncreFlashAttentionTilingAtbDataV2 *__restrict tiling_data = &tiling_data_in;                            \
        op.Init(query, key, value, attnMask, actualSeqLengthsQ, actualSeqLengths, blocktable, attentionOut, user,      \
                tiling_data);                                                                                          \
        op.Process();                                                                                                  \
    } while (0)


extern "C" __global__ __aicore__ void custom_fused_infer_attention_v310_FIAS(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attnMask,
    __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;

    __gm__ uint8_t *user = GetUserWorkspace(workspace);
#if (__CCE_AICORE__ > 200)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT16) && (ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) && (ORIG_DTYPE_KEY == DT_FLOAT16)
    TILING_KEY_IS(30000000000200000);
    TILING_KEY_IS(30000000000200001);
    #if (__CCE_AICORE__ <= 200)
        #if TILING_KEY_VAR == 30000000000200000
            INVOKE_IFA_NEW_GQA_OP_IMPL(
                PagedAttentionDecoderMask, half, half, half, half, true, false, LAYOUT::BSND,
                false, false, LAYOUT::BSND, AMLAMODE::NORMAL, false, IncreFlashAttentionTilingAtbDataV2);
        #elif TILING_KEY_VAR == 30000000000200001
            INVOKE_IFA_NEW_GQA_OP_IMPL(
                PagedAttentionDecoderMask, half, half, half, half, true, false, LAYOUT::TND,
                false, false, LAYOUT::TND, AMLAMODE::NORMAL, false, IncreFlashAttentionTilingAtbDataV2);
        #endif
    #endif
#endif
}

extern "C" __global__ __aicore__ void
custom_fused_infer_attention_v310(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *attnMask,
    __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsKv,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    custom_fused_infer_attention_v310_FIAS(
        query, key, value, attnMask, actualSeqLengthsQ, actualSeqLengthsKv, blocktable,
        attentionOut, workspace, tiling);
}
