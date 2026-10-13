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
 * \file fused_quant_scatter_copy_sparse_flash_attention.cpp
 * \brief
 */

#ifndef FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_PREFETCH
#define FUSED_QUANT_SCATTER_COPY_SPARSE_FLASH_ATTENTION_PREFETCH 1
#endif

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "fused_quant_scatter_copy_sparse_flash_attention_template_tiling_key.h"
#include "fused_quant_scatter_copy_sparse_flash_attention_copy.h"

using FusedQuantScatterCopySparseFlashAttentionTilingDataMla = FusedQuantScatterCopySparseFlashAttentionTilingData;
#include "arch35/fused_quant_scatter_copy_sparse_flash_attention_kernel_mla.h"

template<bool SPLIT_HEADS>
__global__ __aicore__ void fused_quant_scatter_copy_sparse_flash_attention(
    GM_ADDR query, GM_ADDR hbmKvBytes, GM_ADDR dramKvBytes,
    GM_ADDR hbmBlockTable, GM_ADDR dramBlockTable, GM_ADDR topkDstSlots,
    GM_ADDR topkSrcIds, GM_ADDR topkMissCounts,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKv,
    GM_ADDR numCacheTokens, GM_ADDR missSrcIds, GM_ADDR missDstSlots,
    GM_ADDR missCounts,
    GM_ADDR hbmKvAlias, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    (void)hbmKvAlias;
    GET_TILING_DATA_WITH_STRUCT(FusedQuantScatterCopySparseFlashAttentionTilingData, tilingData, tiling);
    AscendC::TPipe pipe;
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    const bool fallback = fused_quant_scatter_copy_sfa::RequiresUnionCopy(missCounts, topkMissCounts,
        actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, tilingData);
    if (fallback) {
        fused_quant_scatter_copy_sfa::UnionCopy(pipe, hbmKvBytes, dramKvBytes, hbmBlockTable,
            dramBlockTable, missSrcIds, missDstSlots, missCounts, tilingData);
    }
    using QueryType = typename std::conditional<ORIG_DTYPE_QUERY == DT_BF16, bfloat16_t, half>::type;
    using Cube = typename std::conditional<g_coreType == AscendC::AIC,
        BaseApi::QSFAMatmulService<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>,
        BaseApi::QSFAMatmulServiceDummy<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>>::type;
    using Vector = typename std::conditional<g_coreType == AscendC::AIC,
        BaseApi::QSFAVectorServiceDummy<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>,
        BaseApi::QSFAVectorService<QueryType, fp8_e4m3fn_t, float, QueryType, false, true,
            QSFA_LAYOUT::TND, QSFA_LAYOUT::PA_BSND, QSFATemplateMode::SCFA_TEMPLATE_MODE, SPLIT_HEADS>>::type;
    BaseApi::FusedQuantScatterCopySparseFlashAttentionMla<Cube, Vector> op;
    op.Init(query, hbmKvBytes, hbmKvBytes, topkDstSlots, nullptr, nullptr, hbmBlockTable,
        actualSeqLengthsQuery, actualSeqLengthsKv, numCacheTokens, attentionOut,
        userWorkspace, &tilingData, &pipe);
    op.InitSourceAwareGather(dramKvBytes, dramBlockTable, topkDstSlots, topkSrcIds, topkMissCounts, &tilingData,
        !fallback, userWorkspace + fused_quant_scatter_copy_sfa::AttentionWorkspaceBytes(
            tilingData.singleCoreParams.usedCoreNum));
    op.Process();
}
