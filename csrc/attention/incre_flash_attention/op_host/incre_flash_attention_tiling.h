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
 * \file incre_flash_attention_tiling.h
 * \brief

 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_NEW_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling.h"
#include "../../prompt_flash_attention/op_host/prompt_flash_attention_tiling_context.h"

#ifdef ASCENDC_OP_TEST
#define IFA_EXTERN_C extern "C"
#else
#define IFA_EXTERN_C
#endif
namespace optiling {
BEGIN_TILING_DATA_DEF(IncreFlashAttentionInitOutputParams)
TILING_DATA_FIELD_DEF(uint32_t, isPerChnOut)
TILING_DATA_FIELD_DEF(uint32_t, isOutQuantTypeBf16)
TILING_DATA_FIELD_DEF(uint32_t, singleCoreSize)
TILING_DATA_FIELD_DEF(uint32_t, singleCoreLseSize)
TILING_DATA_FIELD_DEF(int64_t, totalOutputSize)
TILING_DATA_FIELD_DEF(int64_t, totalLseOutputSize)
TILING_DATA_FIELD_DEF(uint32_t, needInit)
TILING_DATA_FIELD_DEF(uint32_t, isBSNDOut)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionInitOutputParamsOp, IncreFlashAttentionInitOutputParams)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(uint32_t, headSize)
TILING_DATA_FIELD_DEF(uint32_t, headSizeV)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerSeq)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, kvHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, headNumRatio)
TILING_DATA_FIELD_DEF(uint32_t, qHeadNum)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, batchContinuousFlag)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftFlag)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftB)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS)
TILING_DATA_FIELD_DEF(uint32_t, pseShiftS0)
TILING_DATA_FIELD_DEF(uint32_t, selectWithByteMaskTmpMinSize)
TILING_DATA_FIELD_DEF(uint32_t, actualLenQDims)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDims)
TILING_DATA_FIELD_DEF(uint32_t, qPaddingFlag)
TILING_DATA_FIELD_DEF(uint32_t, kvPaddingFlag)
TILING_DATA_FIELD_DEF(uint32_t, msdIterNum)
TILING_DATA_FIELD_DEF(uint32_t, l2CacheOffFlag)
TILING_DATA_FIELD_DEF(uint32_t, antiquantPerTensorFlag)
TILING_DATA_FIELD_DEF(uint32_t, antiquantPerHeadFlag)
TILING_DATA_FIELD_DEF(uint32_t, antiquantParamsInPagedAttentionFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskBatch)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskQSize)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskSize)
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
TILING_DATA_FIELD_DEF(uint32_t, totalBlockNum)
TILING_DATA_FIELD_DEF(uint32_t, paKvShapeType)
TILING_DATA_FIELD_DEF(uint32_t, antiqSeqSize)
TILING_DATA_FIELD_DEF(int32_t, preToken)
TILING_DATA_FIELD_DEF(int32_t, nextToken)
TILING_DATA_FIELD_DEF(uint32_t, isRowInvalid)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint32_t, slidingFlag)
TILING_DATA_FIELD_DEF(int64_t, windowSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionBaseParamsOp, IncreFlashAttentionBaseParams)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionCoreParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, coreSidxEnd); // 50:MAX_CORE_NUM of 910b coreSidxEnd数组首地址要保证8字节对齐
TILING_DATA_FIELD_DEF_ARR(uint32_t, 66, coreSidxEndRegbase); // 66:MAX_CORE_NUM of Ascend 950
TILING_DATA_FIELD_DEF_ARR(uint32_t, 66, coreSposStartRegbase); // 66:MAX_CORE_NUM of Ascend 950
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionCoreParamsOp, IncreFlashAttentionCoreParams);

BEGIN_TILING_DATA_DEF(IncreFlashAttentionSplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, headSplit)
TILING_DATA_FIELD_DEF(uint32_t, maskHeadStride)
TILING_DATA_FIELD_DEF(uint32_t, maskBatchStride)
TILING_DATA_FIELD_DEF(uint32_t, qTokens)
TILING_DATA_FIELD_DEF(uint32_t, isTriu)
TILING_DATA_FIELD_DEF(uint32_t, maxSeqlen)
TILING_DATA_FIELD_DEF(uint32_t, totalQBlockNum)
TILING_DATA_FIELD_DEF(uint32_t, seqStepQ)
TILING_DATA_FIELD_DEF(uint32_t, seqStepKv)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, startBlk); 
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, endBlk); 
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, startBatch); 
TILING_DATA_FIELD_DEF_ARR(uint32_t, 50, endBatch);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSplitCoreParamsOp, IncreFlashAttentionSplitCoreParams);

BEGIN_TILING_DATA_DEF(IncreFlashAttentionSingleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopTimes);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSizeTail);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, blockSplitBn2Range);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitedBatchRange);
TILING_DATA_FIELD_DEF(uint32_t, groupSplitSize);
TILING_DATA_FIELD_DEF(uint32_t, s1SplitSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSingleCoreParamsOp, IncreFlashAttentionSingleCoreParams)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionSingleCoreTensorSize)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSingleCoreTensorSizeOp, IncreFlashAttentionSingleCoreTensorSize)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionSplitKVParams)
TILING_DATA_FIELD_DEF(uint32_t, s2)
TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopSize)
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSplitKVParamsOp, IncreFlashAttentionSplitKVParams)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSplitKVParams, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionCoreParams, increFlashAttentionCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSingleCoreParams, increFlashAttentionSingleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSingleCoreTensorSize, increFlashAttentionSingleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionInitOutputParams, outputParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionTilingDataOp, IncreFlashAttentionTilingData)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingDataPrefix)
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionTilingData, base);
TILING_DATA_FIELD_DEF(uint64_t, prefixAttenOutOffset); // 临时输出偏移
TILING_DATA_FIELD_DEF(uint64_t, userPromptAttenOutOffset);
TILING_DATA_FIELD_DEF(uint64_t, tmpLseOffset);
TILING_DATA_FIELD_DEF(uint64_t, prefixLen); // prefix 长度
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum); // combine 分核参数，参考普通bn分核流程，总数不超过numBlocks
TILING_DATA_FIELD_DEF(uint32_t, blockSplitBn2Range);
TILING_DATA_FIELD_DEF(uint32_t, tailSplitedBatchRange);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, batchSizeQ);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionTilingDataPrefixOp, IncreFlashAttentionTilingDataPrefix)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingDataV2)
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionTilingData, tilingBase);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionTilingDataPrefix, tilingPrefix);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttention, IncreFlashAttentionTilingDataV2)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingAtbDataV2)
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionBaseParams, tilingBase);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSplitCoreParams, tilingPerCore);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_30000000000200000, IncreFlashAttentionTilingAtbDataV2)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_30000000000200001, IncreFlashAttentionTilingAtbDataV2)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_30000000000200302, IncreFlashAttentionTilingAtbDataV2)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_30000000000222322, IncreFlashAttentionTilingAtbDataV2)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionEmptyInputTilingData)
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionInitOutputParams, outputParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_13, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_14, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_27, IncreFlashAttentionEmptyInputTilingData)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_30, IncreFlashAttentionEmptyInputTilingData)

REGISTER_TILING_DATA_CLASS(IncreFlashAttention_1000000000000000090, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_10000000000000090, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_1000000000000000020, PromptFlashAttentionTilingData)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_1002122000000021012, FlashAttentionScoreSimplifiedTilingData)

REGISTER_TILING_DATA_CLASS(IncreFlashAttention_1001121000000021012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_1002312000000021012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(IncreFlashAttention_1001311000000021012, FlashAttentionScoreSimplifiedTilingData)
} // namespace optiling
#endif