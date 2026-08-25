/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TURBOQUANT_SFA_TILING_DATA_H
#define TURBOQUANT_SFA_TILING_DATA_H
#include <cstdint>
struct alignas(8) TurboQuantSparseFlashAttentionBaseParamsMla {
uint32_t batchSize;
uint32_t seqSize;
uint32_t qSeqSize;
int64_t blockSize;
uint32_t maxBlockNumPerBatch;
uint32_t actualLenDimsQ;
uint32_t actualLenDimsKV;
float scaleValue;
uint32_t nNumOfQInOneGroup;
uint32_t outputLayout;
uint32_t sparseMode;
int64_t sparseBlockSize;
uint32_t sparseBlockCount;
int64_t dSizeVInput;
uint32_t headDim;
uint32_t ropeHeadDim;
int64_t keyQuantMode;
int64_t valueQuantMode;
int64_t tileSize;
uint32_t isActualLenDimsNull;
uint32_t isActualLenDimsKVNull;
};
struct alignas(8) TurboQuantSparseFlashAttentionSingleCoreParamsMla {
uint32_t usedCoreNum;
};
struct alignas(8) TurboQuantSparseFlashAttentionSingleCoreTensorSizeMla {
uint32_t mmResUbSize;
uint32_t bmm2ResUbSize;
};
struct alignas(8) TurboQuantSparseFlashAttentionSplitKVParamsMla {
uint32_t s2;             // S2切分份数
uint32_t accumOutSize;   // FD workspace
uint32_t logSumExpSize;  // FD workspace
};
struct alignas(8) TurboQuantSparseFlashAttentionInnerSplitParams {
uint32_t mBaseSize;
uint32_t s2BaseSize;
};
struct alignas(8) TurboQuantSparseFlashAttentionTilingDataMla {
TurboQuantSparseFlashAttentionBaseParamsMla baseParams;
TurboQuantSparseFlashAttentionSplitKVParamsMla splitKVParams;
TurboQuantSparseFlashAttentionSingleCoreParamsMla singleCoreParams;
TurboQuantSparseFlashAttentionSingleCoreTensorSizeMla singleCoreTensorSize;
TurboQuantSparseFlashAttentionInnerSplitParams innerSplitParams;
};
#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    for (uint32_t _ti = 0; _ti < sizeof(tiling_struct); ++_ti) \
        reinterpret_cast<uint8_t*>(&tiling_data)[_ti] = reinterpret_cast<__gm__ uint8_t*>(tiling_arg)[_ti];
#endif
#endif
