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
 * \file incre_flash_attention_tiling_mla.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_MLA_NEW_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INCREFLASHATTENTIONSCORE_MLA_NEW_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t MAX_AIC_CORE_NUM = 26; // 25 + 1 保证数组8字节对齐

BEGIN_TILING_DATA_DEF(IncreFlashAttentionBaseParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup) // G
TILING_DATA_FIELD_DEF(uint32_t, actualLenQDims)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDims)
TILING_DATA_FIELD_DEF(uint32_t, antiquantMode)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskFlag)
TILING_DATA_FIELD_DEF(uint32_t, attenMaskSize)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint32_t, outputLayout) // 输出Layout
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseFlag)
TILING_DATA_FIELD_DEF(uint32_t, tSeqSize)   // TND T轴总长度（含padding）
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionBaseParamsMlaOp, IncreFlashAttentionBaseParamsMla)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionCoreParamsMla)
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM, coreBEnd);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM, coreSidxEnd);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM, coreS1OuterEnd);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM, coreS2End);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionCoreParamsMlaOp, IncreFlashAttentionCoreParamsMla)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionSingleCoreParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, singleProcessSInnerSize);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, groupSplitSize); // G 切分
TILING_DATA_FIELD_DEF(uint32_t, s1SplitSize);    // S 切分
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSingleCoreParamsMlaOp, IncreFlashAttentionSingleCoreParamsMla)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionSingleCoreTensorSizeMla)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSingleCoreTensorSizeMlaOp, IncreFlashAttentionSingleCoreTensorSizeMla)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionSplitKVParamsMla)
TILING_DATA_FIELD_DEF(uint32_t, s2)             // S2切分份数
TILING_DATA_FIELD_DEF(uint32_t, sInnerLoopSize) 
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)   // FD workspace
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize)  // FD workspace
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionSplitKVParamsMlaOp, IncreFlashAttentionSplitKVParamsMla)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionTNDSplitCoreParamsMla) // TND flashDecode相关
TILING_DATA_FIELD_DEF(uint32_t, tndFDCoreArrLen);
TILING_DATA_FIELD_DEF(uint32_t, reserve);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM, balanceFDCoreBArr);  // Bidx
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM, balanceFDCoreS1Arr); // s1Idx
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM, balanceFDCoreKVSplitArr); // 规约任务对应的切分份数
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM, balanceFDCoreStartKVSplitNum); // 当前核的第一个块对应的S2外切索引
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHead);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM * 2, gS1IdxEndOfFdHeadSplit);
TILING_DATA_FIELD_DEF(uint32_t, usedVecNumOfFd);
TILING_DATA_FIELD_DEF(uint32_t, FdBalanceFlag);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM * 2, gS1SplitNumOfFdHeadMla);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_AIC_CORE_NUM * 2, gS1LastPartSizeOfFdHead);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(IncreFlashAttentionTNDSplitCoreParamsMlaOp, IncreFlashAttentionTNDSplitCoreParamsMla)

BEGIN_TILING_DATA_DEF(IncreFlashAttentionTilingDataMla)
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionBaseParamsMla, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSplitKVParamsMla, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionTNDSplitCoreParamsMla, tndSplitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionCoreParamsMla, increFlashAttentionCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSingleCoreParamsMla, increFlashAttentionSingleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(IncreFlashAttentionSingleCoreTensorSizeMla, increFlashAttentionSingleCoreTensorSize);
END_TILING_DATA_DEF

} // namespace optiling

#endif
