/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_ATTENTION_SCORE_TILING_H
#define SPARSE_ATTENTION_SCORE_TILING_H

#include <cstdint>
#include <array>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

constexpr uint64_t SASA_BASE_TILING = 10000;
constexpr uint64_t SASA_FP16_D128_TILING = SASA_BASE_TILING + 1;
constexpr uint64_t SASA_BF16_D128_TILING = SASA_BASE_TILING + 2;
constexpr uint64_t SASA_FP8_D128_TILING = SASA_BASE_TILING + 3;
constexpr uint64_t SASA_FP8_D128_BF16_TILING = SASA_BASE_TILING + 4;
constexpr uint64_t SASA_FP16_D128_ARCH35_FD_TILING = SASA_BASE_TILING + 5;
constexpr uint64_t SASA_BF16_D128_ARCH35_FD_TILING = SASA_BASE_TILING + 6;
constexpr uint64_t SASA_FP8_D128_ARCH35_FD_TILING = SASA_BASE_TILING + 7;
constexpr uint64_t SASA_FP8_D128_BF16_ARCH35_FD_TILING = SASA_BASE_TILING + 8;
constexpr uint64_t SASA_FP16_D128_ARCH35_NTD_TILING = SASA_BASE_TILING + 17;
constexpr uint64_t SASA_BF16_D128_ARCH35_NTD_TILING = SASA_BASE_TILING + 18;
constexpr uint64_t SASA_FP16_D128_ARCH35_NTD_FD_TILING = SASA_BASE_TILING + 19;
constexpr uint64_t SASA_BF16_D128_ARCH35_NTD_FD_TILING = SASA_BASE_TILING + 20;

constexpr uint32_t SASA_FD_MAX_AIC = 32;
constexpr uint32_t SASA_FD_MAX_BASE_TASK = 32;

constexpr uint64_t SASA_BASE_ARCH22_TILING = 20000;
constexpr uint64_t SASA_FP16_D128_ARCH22_TILING = SASA_BASE_ARCH22_TILING + 1;
constexpr uint64_t SASA_BF16_D128_ARCH22_TILING = SASA_BASE_ARCH22_TILING + 2;
constexpr uint64_t SASA_FP16_D128_ARCH22_FD_TILING = SASA_BASE_ARCH22_TILING + 3;
constexpr uint64_t SASA_BF16_D128_ARCH22_FD_TILING = SASA_BASE_ARCH22_TILING + 4;
constexpr uint64_t SASA_FP16_D128_ARCH22_NTD_TILING = SASA_BASE_ARCH22_TILING + 5;
constexpr uint64_t SASA_BF16_D128_ARCH22_NTD_TILING = SASA_BASE_ARCH22_TILING + 6;
constexpr uint64_t SASA_FP16_D128_ARCH22_NTD_FD_TILING = SASA_BASE_ARCH22_TILING + 7;
constexpr uint64_t SASA_BF16_D128_ARCH22_NTD_FD_TILING = SASA_BASE_ARCH22_TILING + 8;

BEGIN_TILING_DATA_DEF(SparseAttentionScoreTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, topK);
TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, totalQTokens);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, innerPrecise);
TILING_DATA_FIELD_DEF(uint32_t, maxQSeqlen);
// Workspace大小
TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize);
TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize);
TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize);
TILING_DATA_FIELD_DEF(uint64_t, updateSize);
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);
TILING_DATA_FIELD_DEF(uint64_t, tilingKey);
TILING_DATA_FIELD_DEF(uint32_t, groupSize);
TILING_DATA_FIELD_DEF(uint32_t, qBaseTile);
TILING_DATA_FIELD_DEF(uint32_t, kvBaseTile);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileM);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileN);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileKLeft);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileKRight);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileM);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileN);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileKLeft);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileKRight);
TILING_DATA_FIELD_DEF(uint32_t, qL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, kL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, vL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, pL1BufNum);
// FlashDecoding metadata. All workspace offsets are bytes
// relative to AscendC::GetUserWorkspace(workspace).
TILING_DATA_FIELD_DEF(uint32_t, fdLseSubStride);
TILING_DATA_FIELD_DEF(uint32_t, layoutMode);
TILING_DATA_FIELD_DEF(uint32_t, qTokenStride);
TILING_DATA_FIELD_DEF(uint32_t, qHeadStride);
TILING_DATA_FIELD_DEF(uint32_t, kvHeadStride);
TILING_DATA_FIELD_DEF(uint32_t, kvTokenStride);
TILING_DATA_FIELD_DEF(uint32_t, fdUsedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, fdIdentityCount);
// FD core-range metadata.
TILING_DATA_FIELD_DEF(uint32_t, fdCorePerCoreTaskNum);
TILING_DATA_FIELD_DEF_ARR(uint32_t, SASA_FD_MAX_AIC, fdCoreTaskStart);
TILING_DATA_FIELD_DEF_ARR(uint32_t, SASA_FD_MAX_AIC, fdCoreTaskEnd);
TILING_DATA_FIELD_DEF_ARR(uint32_t, SASA_FD_MAX_AIC, fdCoreStartBaseTask);
TILING_DATA_FIELD_DEF_ARR(uint32_t, SASA_FD_MAX_AIC, fdCoreStartBlockIdx);
TILING_DATA_FIELD_DEF_ARR(uint32_t, SASA_FD_MAX_AIC, fdCoreEndBaseTask);
TILING_DATA_FIELD_DEF_ARR(uint32_t, SASA_FD_MAX_AIC, fdCoreEndBlockIdx);
// FD combine-range metadata.
TILING_DATA_FIELD_DEF(uint32_t, fdCombineTaskNum);
TILING_DATA_FIELD_DEF_ARR(uint32_t, SASA_FD_MAX_BASE_TASK, fdCombineBaseTask);
TILING_DATA_FIELD_DEF_ARR(uint32_t, SASA_FD_MAX_BASE_TASK, fdPartialStartByBase);
TILING_DATA_FIELD_DEF_ARR(uint32_t, SASA_FD_MAX_BASE_TASK, fdPartialCountByBase);
TILING_DATA_FIELD_DEF(uint64_t, fdIdentityOffset);
TILING_DATA_FIELD_DEF(uint64_t, fdPartialLseOffset);
TILING_DATA_FIELD_DEF(uint64_t, fdPartialOOffset);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SparseAttentionScore_950, SparseAttentionScoreTilingData)

struct SparseAttentionScoreCompileInfo {
    uint32_t inputDataByte = 2;
    ge::DataType inputDataType;
    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t sysWorkspaceSize = 0;
    platform_ascendc::SocVersion socVersion;
};

struct SASAFdCoreRange {
    uint32_t perCoreTaskNum = 0;
    uint32_t usedCoreNum = 0;
    std::array<uint32_t, SASA_FD_MAX_AIC> taskStart{};
    std::array<uint32_t, SASA_FD_MAX_AIC> taskEnd{};
    std::array<uint32_t, SASA_FD_MAX_AIC> startBaseTask{};
    std::array<uint32_t, SASA_FD_MAX_AIC> startBlockIdx{};
    std::array<uint32_t, SASA_FD_MAX_AIC> endBaseTask{};
    std::array<uint32_t, SASA_FD_MAX_AIC> endBlockIdx{};
};

struct SASAFdCombineRange {
    uint32_t combineTaskNum = 0;
    uint32_t partialTaskNum = 0;
    std::array<uint32_t, SASA_FD_MAX_BASE_TASK> baseTask{};
    std::array<uint32_t, SASA_FD_MAX_BASE_TASK> partialStartByBase{};
    std::array<uint32_t, SASA_FD_MAX_BASE_TASK> partialCountByBase{};
};

class SASATiling {
public:
    SASATiling() = default;
    ~SASATiling() = default;

    ge::graphStatus GetTiling(gert::TilingContext *context,
                              SparseAttentionScoreTilingData &tilingData);
    ge::graphStatus SetTilingData(gert::TilingContext *context,
                                  SparseAttentionScoreTilingData &tilingData);

private:
    ge::graphStatus GetNpuInfo(gert::TilingContext *context);
    ge::graphStatus ParseAttrs(gert::TilingContext *context);
    ge::graphStatus ParseInputTensors(gert::TilingContext *context);
    ge::graphStatus ParseSeqlens(gert::TilingContext *context);
    ge::graphStatus ParseSelectNumIdx(gert::TilingContext *context);
    ge::graphStatus CalculateTaskSplit(gert::TilingContext *context);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *context);
    ge::graphStatus FillTilingData(gert::TilingContext *context);
    ge::graphStatus CheckAttentionOutDtype(gert::TilingContext *sasContext);
    uint64_t GenerateTilingKey();

private:
    uint32_t batch_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t embeddingSize_ = 0;
    uint32_t blockSize_ = 128;
    uint32_t topK_ = 16;
    uint32_t maxBlocksPerBatch_ = 0;
    uint32_t totalQTokens_ = 0;
    uint32_t maxQSeqlen_ = 0;
    uint32_t totalTaskNum_ = 0;
    float scaleValue_ = 0.0f;
    uint32_t innerPrecise_ = 0;

    const int32_t *qSeqLenList_ = nullptr;
    const int32_t *kvSeqLenList_ = nullptr;
    const int32_t *selectNumIdxList_ = nullptr;
    uint64_t qSeqLenCount_ = 0;
    uint64_t kvSeqLenCount_ = 0;

    uint64_t mm1OutSize_ = 0;
    uint64_t smOnlineOutSize_ = 0;
    uint64_t mm2OutSize_ = 0;
    uint64_t updateSize_ = 0;
    uint64_t workSpaceSize_ = 0;

    bool enableFd_ = false;
    // "TND": Q=TND, KV=TND-in-block(BSND); "NTD": Q=NTD, KV=NTD-in-block(BNSD);
    // "TND_BNSD": Q=TND, KV=BNSD (mixed).
    bool isQNtd_ = false;
    bool isKvNtd_ = false;
    uint32_t qTokenStride_ = 0;
    uint32_t qHeadStride_ = 0;
    uint32_t kvHeadStride_ = 0;
    uint32_t kvTokenStride_ = 0;
    uint32_t fdIdentityCount_ = 0;
    uint32_t fdLseSubStride_ = 0;
    uint64_t fdIdentityOffset_ = 0;
    uint64_t fdPartialLseOffset_ = 0;
    uint64_t fdPartialLseSize_ = 0;
    uint64_t fdPartialOOffset_ = 0;
    uint64_t fdPartialOSize_ = 0;
    SASAFdCoreRange fdCoreRange_{};
    SASAFdCombineRange fdCombineRange_{};

    uint32_t blockDim_ = 20;
    uint32_t aivNum_ = 0;
    uint32_t aicNum_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t l1Size_ = 0;
    uint64_t libapiSize_ = 0;
    uint32_t socVer_ = 0;

    ge::DataType dataType_ = ge::DT_FLOAT16;
    ge::DataType attentionOutDtype_ = ge::DT_FLOAT16;

    SparseAttentionScoreTilingData *tilingData_ = nullptr;
};

}  // namespace optiling

#endif  // SPARSE_ATTENTION_SCORE_TILING_H
