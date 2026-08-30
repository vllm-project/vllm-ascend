/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_v2_sink_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSINK_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSINK_H_

#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "fused_infer_attention_score_v2_sink_tiling_compile_info.h"
#include "fused_infer_attention_score_v2_sink_tiling_index.h"

#ifdef ASCENDC_OP_TEST
#define FIA_EXTERN_C extern "C"
#else
#define FIA_EXTERN_C
#endif

namespace optiling {
const uint32_t FIA_MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐
const uint32_t FIA_MAX_PA_STRIDE_NUM = 5;
// 基础参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreV2SinkBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, qTSize)
TILING_DATA_FIELD_DEF(uint32_t, kTSize)
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
TILING_DATA_FIELD_DEF(uint32_t, batchInvariant)
TILING_DATA_FIELD_DEF(uint32_t, softmaxMaxSumFlag)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScoreV2SinkBaseParamsOp, FusedInferAttentionScoreV2SinkBaseParams)

// PageAttention 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreV2SinkPageAttentionParams)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF_ARR(uint64_t, FIA_MAX_PA_STRIDE_NUM, keyStride)
TILING_DATA_FIELD_DEF_ARR(uint64_t, FIA_MAX_PA_STRIDE_NUM, valueStride)
TILING_DATA_FIELD_DEF_ARR(uint64_t, FIA_MAX_PA_STRIDE_NUM, keyRopeStride)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScoreV2SinkPageAttentionParamsOp,
                           FusedInferAttentionScoreV2SinkPageAttentionParams)

// AttenMask 参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreV2SinkMaskParams)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskSize)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskStride)
TILING_DATA_FIELD_DEF(int32_t, preToken)
TILING_DATA_FIELD_DEF(int32_t, nextToken)
TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid)
TILING_DATA_FIELD_DEF(uint32_t, isExistRowInvalid)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(int64_t, sinkNumber)
TILING_DATA_FIELD_DEF(int64_t, keySinkNumber)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScoreV2SinkMaskParamsOp, FusedInferAttentionScoreV2SinkMaskParams)

// 内切基本块参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreV2SinkInnerSplitParams)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScoreV2SinkInnerSplitParamsOp,
                           FusedInferAttentionScoreV2SinkInnerSplitParams)

// workspace参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreV2SinkWorkspaceParams)
TILING_DATA_FIELD_DEF(uint32_t, mm1ResSize)
TILING_DATA_FIELD_DEF(uint32_t, mm2ResSize)
TILING_DATA_FIELD_DEF(uint32_t, fdAccumOutSize)
TILING_DATA_FIELD_DEF(uint32_t, fdLogSumExpSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScoreV2SinkWorkspaceParamsOp,
                           FusedInferAttentionScoreV2SinkWorkspaceParams)

// 外切分核参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreV2SinkOuterSplitParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, bN2End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2End)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScoreV2SinkOuterSplitParamsOp,
                           FusedInferAttentionScoreV2SinkOuterSplitParams)

// FlashDecode规约参数
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreV2SinkFlashDecodeParams)
TILING_DATA_FIELD_DEF(uint32_t, numOfFdHead)
TILING_DATA_FIELD_DEF(uint32_t, reserved)
TILING_DATA_FIELD_DEF(uint32_t, gS1BaseSizeOfFd) // FD负载均衡中，每个FD任务按gS1切分的基本size
TILING_DATA_FIELD_DEF(uint32_t, usedVecNumOfFd) // FD负载均衡中，用到的vector数
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, bN2IdxOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, gS1IdxOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2SplitNumOfFdHead)
TILING_DATA_FIELD_DEF_ARR(uint32_t, FIA_MAX_AIC_CORE_NUM, s2SplitStartIdxOfCore)
TILING_DATA_FIELD_DEF_ARR(uint32_t,
                          FIA_MAX_AIC_CORE_NUM,
                          gS1SplitNumOfFdHead) // FD负载均衡中，每个FD任务按gS1基本size切分后的份数
TILING_DATA_FIELD_DEF_ARR(
    uint32_t,
    FIA_MAX_AIC_CORE_NUM,
    gS1LastPartSizeOfFdHead) // FD负载均衡中，每个FD任务按gS1基本size切分后，最后一份的gS1大小，即尾块大小
TILING_DATA_FIELD_DEF_ARR(uint32_t,
                          FIA_MAX_AIC_CORE_NUM * 2,
                          gS1IdxEndOfFdHead) // FD负载均衡中，每个vector核处理的最后一个FD任务的序号
TILING_DATA_FIELD_DEF_ARR(uint32_t,
                          FIA_MAX_AIC_CORE_NUM * 2,
                          gS1IdxEndOfFdHeadSplit) // FD负载均衡中，每个vector核处理的最后一个FD任务的子划分的序号
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScoreV2SinkFlashDecodeParamsOp,
                           FusedInferAttentionScoreV2SinkFlashDecodeParams)

// 非量化模板TilingData
BEGIN_TILING_DATA_DEF(FusedInferAttentionScoreV2SinkTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionScoreV2SinkBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionScoreV2SinkPageAttentionParams, pageAttenParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionScoreV2SinkMaskParams, maskParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionScoreV2SinkWorkspaceParams, workspaceParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionScoreV2SinkInnerSplitParams, innerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionScoreV2SinkOuterSplitParams, outerSplitParams);
TILING_DATA_FIELD_DEF_STRUCT(FusedInferAttentionScoreV2SinkFlashDecodeParams, fdParams);
END_TILING_DATA_DEF

extern "C" {
ge::graphStatus DeviceDoOpTilingFusedInferAttentionScoreV2Sink(gert::TilingContext *context);
}
ge::graphStatus TilingFusedInferAttentionScoreV2Sink(gert::TilingContext *context);
FIA_EXTERN_C ge::graphStatus DoOpTilingFusedInferAttentionScoreV2Sink(gert::TilingContext *context);
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSINKSCORE_H_