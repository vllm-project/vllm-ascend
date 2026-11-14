/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef FUSED_INDEXER_PROLOGUE_TILING_DATA_H_
#define FUSED_INDEXER_PROLOGUE_TILING_DATA_H_

#include <cstdint>

#if __has_include("../../mla_preprocess/op_kernel/mla_preprocess_tiling_data.h")
#include "../../mla_preprocess/op_kernel/mla_preprocess_tiling_data.h"
#else
#include "../../../mla_preprocess/op_kernel/mla_preprocess_tiling_data.h"
#endif

struct FusedIndexerPrologueTilingData {
    PpMatmulTilingData matmul;
    uint32_t numTokens{0};
    uint32_t inputDim{0};
    uint32_t headDim{128};
    uint32_t ropeDim{64};
    uint32_t vectorCoreNum{0};
    uint32_t cacheBlocks{0};
    uint32_t cacheBlockSize{0};
    float epsilon{1e-6F};
    uint64_t kCacheStride0{0};
    uint64_t kCacheStride1{0};
    uint64_t scaleCacheStride0{0};
    uint64_t scaleCacheStride1{0};
};

#endif  // FUSED_INDEXER_PROLOGUE_TILING_DATA_H_
