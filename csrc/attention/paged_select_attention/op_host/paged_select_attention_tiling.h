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
 * \file paged_select_attention_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_PAGEDSELECTATTENTION_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_PAGEDSELECTATTENTION_H_
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "paged_select_attention_tiling_compile_info.h"
#include "paged_select_attention_tiling_index.h"

#ifdef ASCENDC_OP_TEST
#define FIA_EXTERN_C extern "C"
#else
#define FIA_EXTERN_C
#endif

namespace optiling {
// This header is retained as a compatibility surface for shared arch32 attention
// tiling utilities that still include the old fused tiling-data types from the
// renamed operator tree. The specialized paged-select operator itself uses the
// narrower host tiling data in paged_select_attention_infer_tiling.h.
const uint32_t FIA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐
// 基础参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, headDimRope)
TILING_DATA_FIELD_DEF(uint32_t, actualSeqS1Dims)
TILING_DATA_FIELD_DEF(uint32_t, actualSeqS2Dims)
TILING_DATA_FIELD_DEF(uint32_t, accumQSeqFlag)
TILING_DATA_FIELD_DEF(uint32_t, accumKVSeqFlag)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint32_t, batchContinuous)
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
TILING_DATA_FIELD_DEF(uint32_t, needInit)
TILING_DATA_FIELD_DEF(uint32_t, slidingFlag)
TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
TILING_DATA_FIELD_DEF(uint32_t, isLegacyIfa)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionBaseParamsOp, FusedInferAttentionBaseParams)

// PageAttention 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionPageAttentionParams)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionPageAttentionParamsOp, FusedInferAttentionPageAttentionParams)

// AttenMask 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionMaskParams)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskBatchStride)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskStride)
TILING_DATA_FIELD_DEF(int32_t, preToken)
TILING_DATA_FIELD_DEF(int32_t, nextToken)
TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid)
TILING_DATA_FIELD_DEF(uint32_t, isExistRowInvalid)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionMaskParamsOp, FusedInferAttentionMaskParams)

// 内切基本块参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionInnerSplitParams)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionInnerSplitParamsOp, FusedInferAttentionInnerSplitParams)

// workspace参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionWorkspaceParams)
TILING_DATA_FIELD_DEF(uint32_t, mm1ResSize)
TILING_DATA_FIELD_DEF(uint32_t, mm2ResSize)
TILING_DATA_FIELD_DEF(uint32_t, fdAccumOutSize)
TILING_DATA_FIELD_DEF(uint32_t, fdLogSumExpSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionWorkspaceParamsOp, FusedInferAttentionWorkspaceParams)

// 外切分核参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionOuterSplitParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, bN2End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2End)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionOuterSplitParamsOp, FusedInferAttentionOuterSplitParams)

// FlashDecode规约参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionFlashDecodeParams)
TILING_DATA_FIELD_DEF(uint32_t, numOfFdHead)
TILING_DATA_FIELD_DEF(uint32_t, reserved)
TILING_DATA_FIELD_DEF(uint32_t, gS1BaseSizeOfFd)                                    // FD负载均衡中，每个FD任务按gS1切分的基本size
TILING_DATA_FIELD_DEF(uint32_t, usedVecNumOfFd)                                     // FD负载均衡中，用到的vector数
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, bN2IdxOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1IdxOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2SplitNumOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2SplitStartIdxOfCore)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1SplitNumOfFdHead)          // FD负载均衡中，每个FD任务按gS1基本size切分后的份数
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1LastPartSizeOfFdHead)      // FD负载均衡中，每个FD任务按gS1基本size切分后，最后一份的gS1大小，即尾块大小
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHead)        // FD负载均衡中，每个vector核处理的最后一个FD任务的序号
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHeadSplit)   // FD负载均衡中，每个vector核处理的最后一个FD任务的子划分的序号
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionFlashDecodeParamsOp, FusedInferAttentionFlashDecodeParams)

// 公共前缀
BEGIN_TILING_DATA_DEF(FusedInferAttentionPrefixParams)
TILING_DATA_FIELD_DEF(uint64_t, prefixMaxLen)
TILING_DATA_FIELD_DEF(uint64_t, prefixLen)
TILING_DATA_FIELD_DEF(bool, prefixFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionPrefixParamsOp, FusedInferAttentionPrefixParams)
// Pse 注册参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionPseParams)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftFlag)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftByBatch)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS1)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS2)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionPseParamsOp, FusedInferAttentionPseParams)

// Left Padding 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionLeftPaddingParams)
TILING_DATA_FIELD_DEF(uint32_t, qPaddingFlag)
TILING_DATA_FIELD_DEF(uint32_t, kvPaddingFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionLeftPaddingParamsOp, FusedInferAttentionLeftPaddingParams)

//MLA非量化模板TilingData
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPageAttentionParams, pageAttenParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionMaskParams, maskParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionWorkspaceParams, workspaceParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionOuterSplitParams, outerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionFlashDecodeParams, fdParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPrefixParams, prefixParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionPseParams, pseParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionLeftPaddingParams, leftPaddingParams);
END_TILING_DATA_DEF

// empty tenmsor 模板TilingData
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreEmptyTensorTilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalOutputSize)
TILING_DATA_FIELD_DEF(uint64_t, singleCoreSize)
TILING_DATA_FIELD_DEF(uint64_t, totalLseSize)
TILING_DATA_FIELD_DEF(uint64_t, singleCoreLseSize)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
END_TILING_DATA_DEF

// 后量化 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionPostQuantParams)
TILING_DATA_FIELD_DEF(uint32_t, isPerChnOut)
TILING_DATA_FIELD_DEF(uint32_t, isOutQuantTypeBf16)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionPostQuantParamsOp, FusedInferAttentionPostQuantParams)

// 全量化 参数 当前无
BEGIN_TILING_DATA_DEF(FusedInferAttentionFullQuantParams)
TILING_DATA_FIELD_DEF(uint32_t, placeHolder)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionFullQuantParamsOp, FusedInferAttentionFullQuantParams)

// L2 Cache 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionL2CacheParams)
TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionL2CacheParamsOp, FusedInferAttentionL2CacheParams)

// MSD 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionMsdParams)
TILING_DATA_FIELD_DEF(uint32_t, msdIterNum)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionMsdParamsOp, FusedInferAttentionMsdParams)

// 伪量化 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionAntiqParams)
TILING_DATA_FIELD_DEF(uint32_t, antiqSeqSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionAntiqParamsOp, FusedInferAttentionAntiqParams)

extern "C" {
ge::graphStatus DeviceDoOpTilingPagedSelectAttention(gert::TilingContext *context);
}
ge::graphStatus TilingPagedSelectAttention(gert::TilingContext *context);
FIA_EXTERN_C ge::graphStatus DoOpTilingPagedSelectAttention(gert::TilingContext *context);
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_PAGEDSELECTATTENTION_H_
