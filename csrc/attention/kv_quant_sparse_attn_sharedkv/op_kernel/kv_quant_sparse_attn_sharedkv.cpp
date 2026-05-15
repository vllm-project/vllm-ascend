/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file kv_quant_sparse_attn_sharedkv.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kv_quant_sparse_attn_sharedkv_template_tiling_key.h"
#include "arch35/kv_quant_sparse_attn_sharedkv_scfa_kernel.h"
#include "kv_quant_sparse_attn_sharedkv_common.h"

using namespace AscendC;

#if defined(__DAV_C310_CUBE__)
#define SAS_OP_IMPL(templateClass, tilingdataClass, ...)                                          \
    do {                                                                                          \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,               \
            BaseApi::SCFABlockCube<__VA_ARGS__>, BaseApi::SCFABlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                \
            BaseApi::SCFABlockVecDummy<__VA_ARGS__>, BaseApi::SCFABlockVec<__VA_ARGS__>>::type;   \
        templateClass<CubeBlockType, VecBlockType> op;                                            \
        op.Init(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ,  \
                seqUsedQ, seqUsedKV, sinks, metadata, attentionOut, user, nullptr, &tPipe);    \
        op.Process();                                                                             \
    } while (0)
#else
#define SAS_OP_IMPL(templateClass, tilingdataClass, ...)                                          \
    do {                                                                                          \
        using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,               \
            BaseApi::SCFABlockCube<__VA_ARGS__>, BaseApi::SCFABlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,                \
            BaseApi::SCFABlockVecDummy<__VA_ARGS__>, BaseApi::SCFABlockVec<__VA_ARGS__>>::type;   \
        templateClass<CubeBlockType, VecBlockType> op;                                            \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tilingDataIn, tiling);                       \
        const tilingdataClass *__restrict tilingData = &tilingDataIn;                             \
        op.Init(query, oriKV, cmpKV, cmpSparseIndices, oriBlockTable, cmpBlockTable, cuSeqlensQ,  \
                seqUsedQ, seqUsedKV, sinks, metadata, attentionOut, user, tilingData, &tPipe); \
        op.Process();                                                                             \
    } while (0)
#endif

template<int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>
 __global__ __aicore__ void
kv_quant_sparse_attn_sharedkv(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                       __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t* oriBlockTable,
                       __gm__ uint8_t* cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                       __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKV,
                       __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmax_lse, 
                       __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    SAS_OP_IMPL(BaseApi::KvQuantSparseAttnSharedkvScfa, KvQuantSparseAttnSharedkvTilingData, bfloat16_t,
        fp8_e4m3fn_t, float, bfloat16_t, FLASH_DECODE, true, static_cast<SAS_LAYOUT>(LAYOUT_T),
        static_cast<SAS_KV_LAYOUT>(KV_LAYOUT_T), static_cast<SASTemplateMode>(TEMPLATE_MODE));
}