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
 * \file gather_pa_kv_cache_vllm_tiling.h
 * \brief
 */

#ifndef GATHER_PA_KV_CACHE_VLLM_TILING_H
#define GATHER_PA_KV_CACHE_VLLM_TILING_H

#include <string>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "../../causal_conv1d/tiling_base/tiling_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GatherPaKvCacheVllmTilingData)
TILING_DATA_FIELD_DEF(int32_t, blockSize);
TILING_DATA_FIELD_DEF(int32_t, numTokens);
TILING_DATA_FIELD_DEF(int32_t, numblkTabCol);
TILING_DATA_FIELD_DEF(int32_t, tokenSizeK);
TILING_DATA_FIELD_DEF(int32_t, tokenSizeV);
TILING_DATA_FIELD_DEF(int32_t, typeByte);
TILING_DATA_FIELD_DEF(int32_t, hasSeqStarts);
TILING_DATA_FIELD_DEF(int32_t, isSeqLensCumsum);
TILING_DATA_FIELD_DEF(int64_t, kCacheBlockStride);
TILING_DATA_FIELD_DEF(int64_t, vCacheBlockStride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GatherPaKvCacheVllm, GatherPaKvCacheVllmTilingData);


struct Tiling4GatherPaKvCacheVllmCompileInfo {
    uint32_t coreNum;
    uint64_t ubSize;
    uint32_t sysWorkspaceSize;
};

ge::graphStatus Tiling4GatherPaKvCacheVllm(gert::TilingContext *context);
ge::graphStatus TilingPrepare4GatherPaKvCacheVllm(gert::TilingParseContext *context);

BEGIN_TILING_DATA_DEF(GatherPaKvCacheVllmTilingDataV35)
TILING_DATA_FIELD_DEF(uint32_t, seqLenAccumSize);
TILING_DATA_FIELD_DEF(int64_t, batchCount);
TILING_DATA_FIELD_DEF(int64_t, batchPerCore);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, kvCacheBlockSize);
TILING_DATA_FIELD_DEF(int64_t, blockTableWidth);
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
TILING_DATA_FIELD_DEF(uint64_t, hiddenSizeK);
TILING_DATA_FIELD_DEF(uint64_t, hiddenSizeV);
TILING_DATA_FIELD_DEF(int64_t, numTokens);
TILING_DATA_FIELD_DEF(uint64_t, maxUbHiddenSizeK);
TILING_DATA_FIELD_DEF(uint64_t, maxUbHiddenSizeV);
TILING_DATA_FIELD_DEF(uint64_t, maxUbHiddenSize);
END_TILING_DATA_DEF;

constexpr uint64_t TILING_KEY_1111 = 1111;
constexpr uint64_t TILING_KEY_1110 = 1110;
constexpr uint64_t TILING_KEY_1101 = 1101;
constexpr uint64_t TILING_KEY_1100 = 1100;
constexpr uint64_t TILING_KEY_1011 = 1011;
constexpr uint64_t TILING_KEY_1010 = 1010;
constexpr uint64_t TILING_KEY_1001 = 1001;
constexpr uint64_t TILING_KEY_1000 = 1000;

struct GatherPaKvCacheVllmTilingKeyMap {
    uint64_t tilingKey;
    bool isCacheModeNorm;
    bool isSeqLenCumSum;
    bool hasSeqOffset;
};

constexpr std::array<GatherPaKvCacheVllmTilingKeyMap, 8> tilingKeyTable = {{
    // tilingKey, isCacheModeNorm, isSeqLenCumSum, hasSeqOffset
    // ND
    {TILING_KEY_1111, true, true, true},
    {TILING_KEY_1110, true, true, false},
    {TILING_KEY_1101, true, false, true},
    {TILING_KEY_1100, true, false, false},
    // NZ
    {TILING_KEY_1011, false, true, true},
    {TILING_KEY_1010, false, true, false},
    {TILING_KEY_1001, false, false, true},
    {TILING_KEY_1000, false, false, false},
}};

REGISTER_TILING_DATA_CLASS(GatherPaKvCacheVllm_1111, GatherPaKvCacheVllmTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCacheVllm_1110, GatherPaKvCacheVllmTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCacheVllm_1101, GatherPaKvCacheVllmTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCacheVllm_1100, GatherPaKvCacheVllmTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCacheVllm_1011, GatherPaKvCacheVllmTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCacheVllm_1010, GatherPaKvCacheVllmTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCacheVllm_1001, GatherPaKvCacheVllmTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCacheVllm_1000, GatherPaKvCacheVllmTilingDataV35);

struct GatherPaKvCacheVllmCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class GatherPaKvCacheVllmTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GatherPaKvCacheVllmTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }
    ~GatherPaKvCacheVllmTiling() override
    {
    }
    int64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t needCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    int64_t batchCount_ = 0;
    uint32_t cacheDTypeByteSize_ = 0;
    uint32_t indexByteSize_ = 0;
    int64_t blockTableWidth_ = 0;
    int64_t blockSize_ = 0;
    uint64_t hiddenSizeK_ = 0;
    uint64_t hiddenSizeV_ = 0;
    int64_t numBlocks_ = 0;
    int64_t numTokens_ = 0;
    size_t kCacheDimNum_ = 0;
    size_t vCacheDimNum_ = 0;
    gert::Shape kCacheShape_;
    gert::Shape vCacheShape_;
    gert::Shape blockTableShape_;
    gert::Shape seqLenShape_;
    gert::Shape keyShape_;
    gert::Shape valueShape_;
    gert::Shape seqOffsetShape_;
    std::string cacheMode_;
    bool isCacheModeNorm_;
    bool isSeqLenCumSum_;
    bool hasSeqOffset_;

private:
    ge::graphStatus GetAttrs();
    ge::graphStatus GetInputKeyCache();
    ge::graphStatus GetInputValueCache();
    ge::graphStatus GetInputBlockTables();
    ge::graphStatus GetInputSeqLens();
    ge::graphStatus GetInputOutputKey();
    ge::graphStatus GetInputOutputValue();
    ge::graphStatus GetInputSeqOffset();

    uint64_t tilingKey_ = 0;
    GatherPaKvCacheVllmTilingDataV35 tilingData_;
};
} // namespace optiling
#endif // GATHER_PA_KV_CACHE_TILING_H