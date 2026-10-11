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
 * \file hc_pre_tiling_arch35.cpp
 * \brief
 */

#include <sstream>
#include "hc_pre_tiling.h"

using namespace ge;
namespace optiling {
namespace HcPreTilingRegbase {
namespace {
constexpr uint64_t WORKSPACE_SIZE = 32;
int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y) {
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y) {
    return CeilDiv(x, y) * y;
}

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t REPEAT_SIZE = 256;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t BF16_BYTES = 2;
constexpr int64_t HALF_SPLIT = 2;
constexpr int64_t PRE_POST_TENSOR_COUNT = 2;
constexpr int64_t BASE_HEAD_COUNT = 2;
constexpr int64_t K_L1_ALIGN = 128;
constexpr int64_t VECTOR_D_ALIGN = 32;
constexpr int64_t PREMIX_INPUT_INDEX = 4;
constexpr int64_t PRE_OUTPUT_INDEX = 3;
constexpr uint64_t NO_PREMIX_MK_KEY = 1000;
constexpr uint64_t PREMIX_MK_KEY = 1002;
constexpr int64_t C0_SIZE_HOST = BLOCK_SIZE / sizeof(float);
constexpr uint64_t M_L1_MAX_SIZE = 256;
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
constexpr uint64_t A_L1_SIZE = K_L1_ALIGN * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
constexpr uint64_t K_UB_MAX_SIZE = 256;
constexpr uint64_t M_L1_UBAWARE_MAX_SIZE = 208;

struct CoreRowTiling {
    int64_t rowOfFormerBlock = 0;
    int64_t rowOfTailBlock = 0;
    int64_t usedCoreNum = 0;
    int64_t rowLoopOfFormerBlock = 0;
    int64_t rowLoopOfTailBlock = 0;
    int64_t tailRowFactorOfFormerBlock = 0;
    int64_t tailRowFactorOfTailBlock = 0;
};

struct RowAndDTiling {
    int64_t rowFactor = 0;
    int64_t dLoop = 0;
    int64_t dFactor = 0;
    int64_t tailDFactor = 0;
};

struct UbBufferConfig {
    int64_t hcMix = 0;
    int64_t hcMult = 0;
    int64_t hcMultAlign = 0;
    int64_t kBlockNum = 0;
    bool isKSplit = false;
};

bool CalcCoreRowTiling(int64_t bs, int64_t coreNum, CoreRowTiling &rowTiling)
{
    if (bs <= 0 || coreNum <= 0) {
        return false;
    }
    rowTiling.rowOfFormerBlock = CeilDiv(bs, coreNum);
    rowTiling.usedCoreNum = std::min(CeilDiv(bs, rowTiling.rowOfFormerBlock), coreNum);
    rowTiling.rowOfTailBlock = bs - (rowTiling.usedCoreNum - 1) * rowTiling.rowOfFormerBlock;
    return true;
}

bool CompleteCoreRowTiling(CoreRowTiling &rowTiling, int64_t rowFactor)
{
    if (rowFactor <= 0) {
        return false;
    }
    rowTiling.rowLoopOfFormerBlock = CeilDiv(rowTiling.rowOfFormerBlock, rowFactor);
    rowTiling.rowLoopOfTailBlock = CeilDiv(rowTiling.rowOfTailBlock, rowFactor);
    rowTiling.tailRowFactorOfFormerBlock = rowTiling.rowOfFormerBlock -
                                           (rowTiling.rowLoopOfFormerBlock - 1) * rowFactor;
    rowTiling.tailRowFactorOfTailBlock = rowTiling.rowOfTailBlock -
                                         (rowTiling.rowLoopOfTailBlock - 1) * rowFactor;
    return true;
}

uint64_t SelectPremixML1Size(int64_t bs, uint64_t mDimNum, uint64_t kSize)
{
    uint64_t bestM = M_L1_MAX_SIZE;
    uint64_t bestCost = ~static_cast<uint64_t>(0);
    for (uint64_t mL1Size = M_L1_MAX_SIZE; mL1Size >= AscendC::BLOCK_CUBE;
         mL1Size -= AscendC::BLOCK_CUBE) {
        const uint64_t kL1Size =
            std::min(A_L1_SIZE / mL1Size, K_L1_MAX_SIZE) / K_L1_ALIGN * K_L1_ALIGN;
        if (kL1Size / HALF_SPLIT >= K_UB_MAX_SIZE || kSize % kL1Size != 0) {
            continue;
        }
        const uint64_t cost =
            static_cast<uint64_t>(CeilDiv(CeilDiv(bs, static_cast<int64_t>(mL1Size)),
                                          static_cast<int64_t>(mDimNum))) * mL1Size;
        if (cost < bestCost) {
            bestCost = cost;
            bestM = mL1Size;
        }
    }
    return std::min(bestM, M_L1_UBAWARE_MAX_SIZE);
}

struct PremixUbConfig {
    int64_t hcMix = 0;
    int64_t hcMult = 0;
    int64_t hcMultAlign = 0;
    int64_t kBlockNum = 0;
    int64_t maxMRows = 0;
    int64_t maxGroupK = 0;
    int64_t d = 0;
    int64_t kL1Size = 0;
    int64_t multCoreSplitKSize = 0;
    bool isKSplit = false;
    bool hasPreMix = false;
    bool hasPreOut = false;
};

struct PremixUbTiling {
    int64_t mUbSize = 0;
    int64_t kUbSize = 0;
    int64_t bufferPool0Size = 0;
    int64_t bufferPool1Size = 0;
    int64_t phase1Size = 0;
};

int64_t CalcNoPremixUbBufferSize(const UbBufferConfig &config, int64_t rowFactor, int64_t dFactor)
{
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);
    const int64_t hcMixAlign = RoundUp(config.hcMix, floatAlign);
    const int64_t dAlign = RoundUp(dFactor, 16);
    int64_t totalSize = rowFactor * hcMixAlign * sizeof(float);                        // mixes
    totalSize += rowFactor * config.hcMult * dAlign * BF16_BYTES * DOUBLE_BUFFER;      // x
    totalSize += rowFactor * dAlign * BF16_BYTES * DOUBLE_BUFFER;                      // y
    totalSize += rowFactor * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER * PRE_POST_TENSOR_COUNT; // pre, post
    totalSize += rowFactor * config.hcMult * config.hcMult * sizeof(float) * DOUBLE_BUFFER; // combFrag
    if (config.isKSplit) {
        totalSize += config.kBlockNum * rowFactor * hcMixAlign * sizeof(float) * DOUBLE_BUFFER; // mm
        totalSize += config.kBlockNum * RoundUp(rowFactor, floatAlign) * sizeof(float) *
                     DOUBLE_BUFFER; // rms
        totalSize += (BASE_HEAD_COUNT + config.hcMult) * config.hcMultAlign * sizeof(float); // bases
    }
    return totalSize;
}

bool IsPremixMFused(const PremixUbConfig &config, int64_t mUbSize, int64_t kUbSize)
{
    return !config.isKSplit && config.hasPreMix && kUbSize > 0 &&
           mUbSize >= config.maxMRows &&
           config.multCoreSplitKSize == config.hcMult * config.d &&
           config.d % kUbSize == 0;
}

int64_t CalcPremixPhase1Size(const PremixUbConfig &config, int64_t mUbSize, int64_t kUbSize)
{
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);
    const int64_t bf16Align = BLOCK_SIZE / BF16_BYTES;
    const int64_t hcMixAlign = RoundUp(config.hcMix, floatAlign);
    const int64_t kUbFloatAlign = RoundUp(kUbSize, floatAlign);
    const int64_t kUbBf16Align = RoundUp(kUbSize, bf16Align);
    const bool fusedY = IsPremixMFused(config, mUbSize, kUbSize);

    int64_t totalSize = 0;
    if (fusedY) {
        totalSize = mUbSize * kUbBf16Align * BF16_BYTES * DOUBLE_BUFFER; // x
        totalSize += config.hcMult * mUbSize *
                     (kUbFloatAlign * sizeof(float) + BLOCK_SIZE);       // cast segments
        totalSize += mUbSize * hcMixAlign * sizeof(float);               // preMix
        totalSize += mUbSize * kUbBf16Align * BF16_BYTES * DOUBLE_BUFFER; // y
    } else if (config.isKSplit) {
        totalSize = mUbSize * kUbBf16Align * BF16_BYTES * DOUBLE_BUFFER; // x
        totalSize += config.maxMRows * hcMixAlign * sizeof(float);             // preMix
        totalSize += mUbSize * kUbBf16Align * BF16_BYTES * DOUBLE_BUFFER;     // y
        totalSize += mUbSize * (kUbFloatAlign * sizeof(float) + BLOCK_SIZE);   // cast
        const int64_t yAccSliceNum = CeilDiv(config.maxGroupK, kUbSize);
        totalSize += config.maxMRows * kUbFloatAlign * sizeof(float) *
                     yAccSliceNum;                                             // y accumulator
    } else {
        totalSize = mUbSize * kUbBf16Align * BF16_BYTES * DOUBLE_BUFFER; // x
        totalSize += mUbSize * (kUbFloatAlign * sizeof(float) + BLOCK_SIZE); // cast
    }
    totalSize += RoundUp(mUbSize, C0_SIZE_HOST) * kUbFloatAlign * sizeof(float) *
                 DOUBLE_BUFFER;                                                // ND2NZ
    if (config.isKSplit) {
        totalSize += RoundUp(config.maxMRows, floatAlign) * sizeof(float) *
                     DOUBLE_BUFFER;                                            // rms
    }
    return totalSize;
}

int64_t CalcMBufferPool1Size(uint64_t ubSize, int64_t maxMRows, int64_t hcMix,
                             int64_t hcMult, int64_t hcMultAlign)
{
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);
    const int64_t mmXSize = maxMRows * RoundUp(hcMix, floatAlign) * sizeof(float);
    const int64_t rmsSize = RoundUp(maxMRows, floatAlign) * sizeof(float);
    const int64_t baseSize = (BASE_HEAD_COUNT + hcMult) * hcMultAlign * sizeof(float);
    return DownAlign(static_cast<int64_t>(ubSize) - mmXSize - rmsSize - baseSize, BLOCK_SIZE);
}

int64_t CalcPremixMPhase2Size(
    const PremixUbConfig &config, int64_t mUbSize, int64_t kUbSize, int64_t rowFactor)
{
    const int64_t hcMixAlign = RoundUp(config.hcMix, BLOCK_SIZE / sizeof(float));
    int64_t totalSize = rowFactor * hcMixAlign * sizeof(float); // mixes
    totalSize += rowFactor * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER; // post
    if (config.hasPreOut) {
        totalSize += rowFactor * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER; // pre
    }
    if (!IsPremixMFused(config, mUbSize, kUbSize)) {
        totalSize += rowFactor * config.hcMult * RoundUp(config.d, BF16_BYTES * C0_SIZE_HOST) *
                     BF16_BYTES * DOUBLE_BUFFER; // x
        totalSize += rowFactor * RoundUp(config.d, BF16_BYTES * C0_SIZE_HOST) *
                     BF16_BYTES * DOUBLE_BUFFER; // y
        totalSize += rowFactor * hcMixAlign * sizeof(float) * DOUBLE_BUFFER; // preMix
    }
    return totalSize;
}

int64_t CalcPremixMKPhase2Size(const PremixUbConfig &config, int64_t rowFactor)
{
    const int64_t floatAlign = BLOCK_SIZE / sizeof(float);
    const int64_t hcMixAlign = RoundUp(config.hcMix, floatAlign);
    int64_t totalSize = config.kBlockNum * rowFactor * hcMixAlign * sizeof(float) * DOUBLE_BUFFER; // mm
    totalSize += config.kBlockNum * RoundUp(rowFactor, floatAlign) * sizeof(float) * DOUBLE_BUFFER; // rms
    totalSize += rowFactor * hcMixAlign * sizeof(float); // mixes
    totalSize += rowFactor * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER; // post
    totalSize += config.hcMultAlign * sizeof(float) * BASE_HEAD_COUNT +
                 config.hcMult * config.hcMultAlign * sizeof(float); // bases
    if (config.hasPreOut) {
        totalSize += rowFactor * config.hcMultAlign * sizeof(float) * DOUBLE_BUFFER; // pre
    }
    return totalSize;
}

template <typename FitsInUb>
int64_t FindMaxFactor(int64_t maxFactor, const FitsInUb &fitsInUb)
{
    int64_t left = 1;
    int64_t right = maxFactor;
    int64_t result = 0;
    while (left <= right) {
        const int64_t middle = left + (right - left) / 2;
        if (fitsInUb(middle)) {
            result = middle;
            left = middle + 1;
        } else {
            right = middle - 1;
        }
    }
    return result;
}

void TryPremixUbCandidate(const PremixUbConfig &config, int64_t availableSize,
                          int64_t kUbSize, PremixUbTiling &ubTiling, int64_t &bestChunkCount)
{
    if (kUbSize <= 0 || kUbSize > config.kL1Size) {
        return;
    }
    int64_t mUbSize = config.maxMRows;
    if (config.isKSplit) {
        mUbSize = FindMaxFactor(config.maxMRows, [&config, kUbSize, availableSize](int64_t candidateMUb) {
            return CalcPremixPhase1Size(config, candidateMUb, kUbSize) <= availableSize;
        });
    } else if (!IsPremixMFused(config, mUbSize, kUbSize) ||
               CalcPremixPhase1Size(config, mUbSize, kUbSize) > availableSize) {
        return;
    }
    if (mUbSize == 0) {
        return;
    }
    const bool fusedY = IsPremixMFused(config, mUbSize, kUbSize);
    const int64_t vectorBlocks = fusedY ? CeilDiv(kUbSize, REPEAT_SIZE / sizeof(float)) : 1;
    const int64_t chunkCount = CeilDiv(config.maxMRows, mUbSize) *
                               CeilDiv(config.maxGroupK, kUbSize) * vectorBlocks;
    if (ubTiling.mUbSize == 0 || chunkCount < bestChunkCount ||
        (chunkCount == bestChunkCount && mUbSize > ubTiling.mUbSize) ||
        (chunkCount == bestChunkCount && mUbSize == ubTiling.mUbSize && kUbSize > ubTiling.kUbSize)) {
        bestChunkCount = chunkCount;
        ubTiling.mUbSize = mUbSize;
        ubTiling.kUbSize = kUbSize;
        ubTiling.phase1Size = CalcPremixPhase1Size(config, mUbSize, kUbSize);
    }
}

template <typename FitsInUb>
int64_t FindFirstDFactorDivisor(int64_t d, const FitsInUb &fitsInUb)
{
    int64_t left = HALF_SPLIT;
    int64_t right = d;
    while (left < right) {
        const int64_t middle = left + (right - left) / 2;
        if (fitsInUb(1, CeilDiv(d, middle))) {
            right = middle;
        } else {
            left = middle + 1;
        }
    }
    return left;
}

template <typename FitsInUb>
bool CalcRowAndDTiling(int64_t d, int64_t maxRowFactor, const FitsInUb &fitsInUb, RowAndDTiling &result)
{
    if (d <= 0 || maxRowFactor <= 0 || !fitsInUb(1, 1)) {
        return false;
    }

    result.rowFactor = 1;
    result.dFactor = d;
    if (fitsInUb(1, d)) {
        result.rowFactor = FindMaxFactor(maxRowFactor, [&fitsInUb, d](int64_t rowFactor) {
            return fitsInUb(rowFactor, d);
        });
    } else {
        result.dFactor = CeilDiv(d, FindFirstDFactorDivisor(d, fitsInUb));
        if (result.dFactor > VECTOR_D_ALIGN) {
            result.dFactor = DownAlign(result.dFactor, VECTOR_D_ALIGN);
        }
    }
    result.dLoop = CeilDiv(d, result.dFactor);
    result.tailDFactor = d % result.dFactor == 0 ? result.dFactor : d % result.dFactor;
    return result.rowFactor > 0 && result.dFactor > 0;
}
}

class HcPreTilingRegbase {
public:
    explicit HcPreTilingRegbase(gert::TilingContext* tilingContext) : context_(tilingContext)
        {
        }
    ~HcPreTilingRegbase() = default;

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    ge::graphStatus GetAttr();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CalcOpTiling();
    ge::graphStatus CalcRegbaseCommonTiling(bool hasPreMix, bool isKSplit);
    ge::graphStatus CalcRegbaseOpTiling();
    ge::graphStatus CalcMKSplitCorePart2Tiling();
    ge::graphStatus CalcPremixMSplitTiling();
    ge::graphStatus CalcPremixMKSplitTiling();
    ge::graphStatus InitPremixUbConfig(bool isKSplit, PremixUbConfig &config);
    ge::graphStatus SelectPremixUbTiling(bool isKSplit, PremixUbConfig &config, PremixUbTiling &ubTiling);
    ge::graphStatus CalcPremixRowTiling(bool isKSplit, CoreRowTiling &coreRowTiling,
                                        RowAndDTiling &rowAndDTiling, PremixUbTiling &ubTiling);
    ge::graphStatus CalcNoPremixRowTiling(bool isKSplit, const CoreRowTiling &coreRowTiling,
                                          RowAndDTiling &rowAndDTiling, int64_t mUbSize,
                                          int64_t &bufferPool1Size);
    void SaveCommonTiling(bool hasPreMix, bool isKSplit, const CoreRowTiling &coreRowTiling,
                          const RowAndDTiling &rowAndDTiling, const PremixUbTiling &ubTiling);
private:
    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_ = 0;
    HcPreTilingData tilingData_;
    uint64_t aivCoreNum_ = 0;
    uint64_t aicCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t ubSize_ = 0;
    int64_t bs_ = 0;
    int64_t hcMix_ = 0;
    int64_t hcMult_ = 0;
    int64_t d_ = 0;
    int64_t iterTimes_ = 0;
    double hcEps_ = 0.0;
    double normEps_ = 0.0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

ge::graphStatus HcPreTilingRegbase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<HcPreCompileInfo>();
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_LOG_E(context_, "compile info is null"),
                      return ge::GRAPH_FAILED);
        aivCoreNum_ = compileInfoPtr->coreNum;
        aicCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aivCoreNum_ = ascendcPlatform.GetCoreNumAiv();
        aicCoreNum_ = ascendcPlatform.GetCoreNumAic();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    auto hcMultAttr = attrs->GetAttrPointer<int64_t>(0);
    hcMult_ = hcMultAttr == nullptr ? 4 : *hcMultAttr;

    auto iterTimesAttr = attrs->GetAttrPointer<int64_t>(1);
    iterTimes_ = iterTimesAttr == nullptr ? 20 : *iterTimesAttr;

    auto epsAttr = attrs->GetAttrPointer<float>(2);
    hcEps_ = epsAttr == nullptr ? 1e-6 : *epsAttr;

    auto normEpsAttr = attrs->GetAttrPointer<float>(3);
    normEps_ = normEpsAttr == nullptr ? 1e-6 : *normEpsAttr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::GetShapeAttrsInfoInner()
{
    // (b, s, hc_mult, d) or (bs, hc_mult, d)
    auto xShape = context_->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context_, xShape, return ge::GRAPH_FAILED);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    if (xDimNum == 3) {
        bs_ = xShape->GetStorageShape().GetDim(0);
        hcMult_ = xShape->GetStorageShape().GetDim(1);
        d_ = xShape->GetStorageShape().GetDim(2);
    } else if (xDimNum == 4) {
        int64_t b = xShape->GetStorageShape().GetDim(0);
        int64_t s = xShape->GetStorageShape().GetDim(1);
        bs_ = b * s;
        hcMult_ = xShape->GetStorageShape().GetDim(2);
        d_ = xShape->GetStorageShape().GetDim(3);
    } else {
        OPS_LOG_E(context_->GetNodeName(), "x dim num should be 3 or 4, but is %zu", xDimNum);
        return ge::GRAPH_FAILED;
    }

    auto shapeHcFn = context_->GetInputShape(1);
    OPS_LOG_E_IF_NULL(context_, shapeHcFn, return ge::GRAPH_FAILED);
    hcMix_ = shapeHcFn->GetStorageShape().GetDim(0);
    OPS_ERR_IF(shapeHcFn->GetStorageShape().GetDim(1) != d_ * hcMult_,
                    OPS_LOG_E(context_->GetNodeName(),
                             "HcFn dim 1 should be equal with d_ * hcMult_  %ld, but is %ld", d_ * hcMult_, shapeHcFn->GetStorageShape().GetDim(1)),
                    return ge::GRAPH_FAILED);

    auto shapeHcScale = context_->GetInputShape(2);
    OPS_LOG_E_IF_NULL(context_, shapeHcScale, return ge::GRAPH_FAILED);
    int64_t scaleFirstDim = shapeHcScale->GetStorageShape().GetDim(0);
    OPS_ERR_IF(scaleFirstDim != 3,
                    OPS_LOG_E(context_->GetNodeName(),
                             "hc_scale size should be equal with 3, but is %ld", scaleFirstDim),
                    return ge::GRAPH_FAILED);

    auto shapeHcBase = context_->GetInputShape(3);
    OPS_LOG_E_IF_NULL(context_, shapeHcBase, return ge::GRAPH_FAILED);
    int64_t baseFirstDim = shapeHcBase->GetStorageShape().GetDim(0);
    OPS_ERR_IF(baseFirstDim != hcMix_,
                    OPS_LOG_E(context_->GetNodeName(),
                             "hc_base size should be equal with mixhc, but is %ld", baseFirstDim),
                    return ge::GRAPH_FAILED);

    tilingData_.set_hasPreMix(context_->GetInputShape(PREMIX_INPUT_INDEX) != nullptr ? 1 : 0);
    tilingData_.set_hasPreOut(context_->GetOutputShape(PRE_OUTPUT_INDEX) != nullptr ? 1 : 0);

    OPS_ERR_IF(GetAttr() != ge::GRAPH_SUCCESS,
                  OPS_LOG_E(context_->GetNodeName(), "get attr failed."),
                  return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus HcPreTilingRegbase::CalcPremixRowTiling(
    bool isKSplit, CoreRowTiling &coreRowTiling, RowAndDTiling &rowAndDTiling, PremixUbTiling &ubTiling)
{
    PremixUbConfig config;
    OPS_ERR_IF(SelectPremixUbTiling(isKSplit, config, ubTiling) != ge::GRAPH_SUCCESS,
               OPS_LOG_E(context_->GetNodeName(), "Failed to select premix UB tiling"),
               return ge::GRAPH_FAILED);
    const int64_t availableSize = isKSplit ? ubTiling.bufferPool0Size : ubTiling.bufferPool1Size;
    const int64_t maxRowFactor = isKSplit ? coreRowTiling.rowOfFormerBlock : config.maxMRows;
    const auto fitsInUb = [&config, &ubTiling, isKSplit, availableSize](int64_t rowFactor) {
        return (isKSplit ? CalcPremixMKPhase2Size(config, rowFactor) :
                           CalcPremixMPhase2Size(config, ubTiling.mUbSize, ubTiling.kUbSize,
                                                 rowFactor)) <= availableSize;
    };
    rowAndDTiling.rowFactor = FindMaxFactor(maxRowFactor, fitsInUb);
    OPS_ERR_IF(rowAndDTiling.rowFactor == 0,
               OPS_LOG_E(context_->GetNodeName(),
                         "UB is insufficient for premix phase2: availableSize=%ld, isKSplit=%d",
                         availableSize, isKSplit ? 1 : 0),
               return ge::GRAPH_FAILED);
    rowAndDTiling.dLoop = 1;
    rowAndDTiling.dFactor = d_;
    rowAndDTiling.tailDFactor = d_;
    OPS_LOG_I(context_->GetNodeName(),
              "Premix %s-split UB tiling: mL1=%ld, kL1=%ld, mUb=%ld, kUb=%ld, "
              "phase1=%ld/%ld, phase2=%ld/%ld, rowFactor=%ld",
              isKSplit ? "MK" : "M", tilingData_.get_mL1Size(), tilingData_.get_kL1Size(),
              ubTiling.mUbSize, ubTiling.kUbSize, ubTiling.phase1Size, availableSize,
              isKSplit ? CalcPremixMKPhase2Size(config, rowAndDTiling.rowFactor) :
                         CalcPremixMPhase2Size(config, ubTiling.mUbSize, ubTiling.kUbSize, rowAndDTiling.rowFactor),
              availableSize, rowAndDTiling.rowFactor);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::CalcNoPremixRowTiling(
    bool isKSplit, const CoreRowTiling &coreRowTiling, RowAndDTiling &rowAndDTiling,
    int64_t mUbSize, int64_t &bufferPool1Size)
{
    const int64_t hcMultAlign = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    bufferPool1Size = isKSplit ? 0 :
        CalcMBufferPool1Size(ubSize_, mUbSize, hcMix_, hcMult_, hcMultAlign);
    const int64_t availableSize = isKSplit ? static_cast<int64_t>(ubSize_) : bufferPool1Size;
    UbBufferConfig config;
    config.hcMix = hcMix_;
    config.hcMult = hcMult_;
    config.hcMultAlign = hcMultAlign;
    config.kBlockNum = tilingData_.get_cubeBlockDimK();
    config.isKSplit = isKSplit;
    const auto fitsInUb = [&config, availableSize](int64_t rowFactor, int64_t dFactor) {
        return CalcNoPremixUbBufferSize(config, rowFactor, dFactor) <= availableSize;
    };
    const int64_t maxRowFactor = isKSplit ? coreRowTiling.rowOfFormerBlock : mUbSize;
    OPS_ERR_IF(availableSize <= 0 || !CalcRowAndDTiling(d_, maxRowFactor, fitsInUb, rowAndDTiling),
               OPS_LOG_E(context_->GetNodeName(),
                         "UB is insufficient for no-premix tiling: availableSize=%ld, isKSplit=%d",
                         availableSize, isKSplit ? 1 : 0),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void HcPreTilingRegbase::SaveCommonTiling(
    bool hasPreMix, bool isKSplit, const CoreRowTiling &coreRowTiling,
    const RowAndDTiling &rowAndDTiling, const PremixUbTiling &ubTiling)
{
    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(RoundUp(hcMult_, BLOCK_SIZE / sizeof(float)));
    tilingData_.set_rowOfFormerBlock(coreRowTiling.rowOfFormerBlock);
    tilingData_.set_rowOfTailBlock(coreRowTiling.rowOfTailBlock);
    tilingData_.set_rowLoopOfFormerBlock(coreRowTiling.rowLoopOfFormerBlock);
    tilingData_.set_rowLoopOfTailBlock(coreRowTiling.rowLoopOfTailBlock);
    tilingData_.set_tailRowFactorOfFormerBlock(coreRowTiling.tailRowFactorOfFormerBlock);
    tilingData_.set_tailRowFactorOfTailBlock(coreRowTiling.tailRowFactorOfTailBlock);
    tilingData_.set_dLoop(rowAndDTiling.dLoop);
    tilingData_.set_dFactor(rowAndDTiling.dFactor);
    tilingData_.set_tailDFactor(rowAndDTiling.tailDFactor);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_kUbSize(ubTiling.kUbSize);
    tilingData_.set_mUbSize(ubTiling.mUbSize);
    tilingData_.set_kBlockFactor(tilingData_.get_cubeBlockDimK());
    if (isKSplit) {
        tilingData_.set_stage2RowFactor(rowAndDTiling.rowFactor);
        tilingData_.set_secondUsedCoreNum(coreRowTiling.usedCoreNum);
    } else {
        tilingData_.set_rowFactor(rowAndDTiling.rowFactor);
        tilingData_.set_rowInnerFactor(rowAndDTiling.rowFactor);
    }
    if (hasPreMix || !isKSplit) {
        tilingData_.set_bufferPool0Size(ubTiling.bufferPool0Size);
        tilingData_.set_bufferPool1Size(ubTiling.bufferPool1Size);
    }
    OPS_LOG_I(context_->GetNodeName(),
              "HcPre arch35 UB tiling: key=%lu, rows=%ld/%ld, usedCores=%ld, "
              "rowFactor=%ld, dFactor=%ld, dLoop=%ld, mUb=%ld, kUb=%ld",
              tilingKey_, coreRowTiling.rowOfFormerBlock, coreRowTiling.rowOfTailBlock,
              coreRowTiling.usedCoreNum, rowAndDTiling.rowFactor, rowAndDTiling.dFactor,
              rowAndDTiling.dLoop, ubTiling.mUbSize, ubTiling.kUbSize);
}

ge::graphStatus HcPreTilingRegbase::CalcRegbaseCommonTiling(bool hasPreMix, bool isKSplit)
{
    CoreRowTiling coreRowTiling;
    OPS_ERR_IF(!CalcCoreRowTiling(bs_, static_cast<int64_t>(aivCoreNum_), coreRowTiling),
               OPS_LOG_E(context_->GetNodeName(), "Invalid row tiling: bs=%ld, aivCoreNum=%lu",
                         bs_, aivCoreNum_),
               return ge::GRAPH_FAILED);
    RowAndDTiling rowAndDTiling;
    PremixUbTiling ubTiling;
    ubTiling.mUbSize = CeilDiv(tilingData_.get_mL1Size(), HALF_SPLIT);
    ubTiling.kUbSize = tilingData_.get_kL1Size() / HALF_SPLIT;
    ubTiling.bufferPool0Size = static_cast<int64_t>(ubSize_);
    if (hasPreMix) {
        if (CalcPremixRowTiling(isKSplit, coreRowTiling, rowAndDTiling, ubTiling) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (CalcNoPremixRowTiling(isKSplit, coreRowTiling, rowAndDTiling,
        ubTiling.mUbSize, ubTiling.bufferPool1Size) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OPS_ERR_IF(!CompleteCoreRowTiling(coreRowTiling, rowAndDTiling.rowFactor),
               OPS_LOG_E(context_->GetNodeName(), "Invalid rowFactor=%ld", rowAndDTiling.rowFactor),
               return ge::GRAPH_FAILED);
    SaveCommonTiling(hasPreMix, isKSplit, coreRowTiling, rowAndDTiling, ubTiling);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::CalcRegbaseOpTiling()
{
    return CalcRegbaseCommonTiling(false, false);
}

ge::graphStatus HcPreTilingRegbase::CalcMKSplitCorePart2Tiling()
{
    return CalcRegbaseCommonTiling(false, true);
}

ge::graphStatus HcPreTilingRegbase::InitPremixUbConfig(bool isKSplit, PremixUbConfig &config)
{
    config.hcMix = hcMix_;
    config.hcMult = hcMult_;
    config.hcMultAlign = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    config.kBlockNum = tilingData_.get_cubeBlockDimK();
    config.maxMRows = CeilDiv(tilingData_.get_mL1Size(), HALF_SPLIT);
    config.isKSplit = isKSplit;
    config.d = d_;
    config.kL1Size = tilingData_.get_kL1Size();
    config.multCoreSplitKSize = tilingData_.get_multCoreSplitKSize();
    config.hasPreMix = context_->GetInputShape(PREMIX_INPUT_INDEX) != nullptr;
    config.hasPreOut = context_->GetOutputShape(PRE_OUTPUT_INDEX) != nullptr;

    OPS_ERR_IF(hcMult_ <= 0 || tilingData_.get_kL1Size() <= 0 || config.maxMRows <= 0,
               OPS_LOG_E(context_->GetNodeName(),
                         "Invalid premix UB tiling input: hcMult=%ld, mL1Size=%ld, kL1Size=%ld",
                         hcMult_, tilingData_.get_mL1Size(), tilingData_.get_kL1Size()),
               return ge::GRAPH_FAILED);

    int64_t dPerCore = d_;
    if (isKSplit) {
        OPS_ERR_IF(tilingData_.get_multCoreSplitKSize() % hcMult_ != 0,
                   OPS_LOG_E(context_->GetNodeName(),
                             "Premix K split size %ld is not divisible by hcMult %ld",
                             tilingData_.get_multCoreSplitKSize(), hcMult_),
                   return ge::GRAPH_FAILED);
        dPerCore = std::min(d_, tilingData_.get_multCoreSplitKSize() / hcMult_);
    }
    config.maxGroupK = std::min(tilingData_.get_kL1Size(), dPerCore);
    OPS_ERR_IF(config.maxGroupK <= 0,
               OPS_LOG_E(context_->GetNodeName(), "Invalid premix maxGroupK=%ld", config.maxGroupK),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::SelectPremixUbTiling(
    bool isKSplit, PremixUbConfig &config, PremixUbTiling &ubTiling)
{
    if (InitPremixUbConfig(isKSplit, config) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ubTiling.bufferPool0Size = static_cast<int64_t>(ubSize_);
    ubTiling.bufferPool1Size = isKSplit ? 0 : CalcMBufferPool1Size(
        ubSize_, config.maxMRows, config.hcMix, config.hcMult, config.hcMultAlign);
    const int64_t availableSize = isKSplit ? ubTiling.bufferPool0Size : ubTiling.bufferPool1Size;
    OPS_ERR_IF(availableSize <= 0,
               OPS_LOG_E(context_->GetNodeName(),
                         "Premix fixed buffers exceed UB: ubSize=%lu, availableSize=%ld, maxMRows=%ld",
                         ubSize_, availableSize, config.maxMRows),
               return ge::GRAPH_FAILED);

    const int64_t fullKUb = RoundUp(config.maxGroupK, C0_SIZE_HOST);
    const int64_t halfKUb = RoundUp(CeilDiv(config.maxGroupK, HALF_SPLIT), C0_SIZE_HOST);
    const int64_t kUbCandidates[2] = {halfKUb, fullKUb};
    int64_t bestChunkCount = 0;
    for (int64_t kUbSize : kUbCandidates) {
        if (ubTiling.kUbSize != 0 && kUbSize == kUbCandidates[0] && kUbCandidates[0] == kUbCandidates[1]) {
            continue;
        }
        TryPremixUbCandidate(config, availableSize, kUbSize, ubTiling, bestChunkCount);
    }

    OPS_ERR_IF(ubTiling.mUbSize == 0,
               OPS_LOG_E(context_->GetNodeName(),
                         "UB is insufficient for premix phase1: ubSize=%lu, availableSize=%ld, "
                         "maxMRows=%ld, maxGroupK=%ld",
                         ubSize_, availableSize, config.maxMRows, config.maxGroupK),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::CalcPremixMSplitTiling()
{
    return CalcRegbaseCommonTiling(true, false);
}

ge::graphStatus HcPreTilingRegbase::CalcPremixMKSplitTiling()
{
    return CalcRegbaseCommonTiling(true, true);
}

ge::graphStatus HcPreTilingRegbase::CalcOpTiling()
{
    OPS_ERR_IF(bs_ <= 0 || hcMult_ <= 0 || d_ <= 0 || aicCoreNum_ == 0 || aivCoreNum_ == 0,
               OPS_LOG_E(context_->GetNodeName(),
                         "Invalid arch35 tiling input: bs=%ld, hcMult=%ld, d=%ld, aic=%lu, aiv=%lu",
                         bs_, hcMult_, d_, aicCoreNum_, aivCoreNum_),
               return ge::GRAPH_FAILED);
    const uint64_t kSize = hcMult_ * d_;
    const bool hasPreMix = context_->GetInputShape(PREMIX_INPUT_INDEX) != nullptr;
    tilingData_.set_k(kSize);
    uint64_t mDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);
    uint64_t kDimNum = aicCoreNum_ / mDimNum;
    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);
    uint64_t actualKBlockNum = CeilDiv(kSize, splitKSize);

    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_cubeBlockDimK(actualKBlockNum);
    tilingData_.set_multCoreSplitMSize(singleCoreM);
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
    tilingData_.set_kL1Size(
        std::min(A_L1_SIZE / tilingData_.get_mL1Size(), K_L1_MAX_SIZE) / K_L1_ALIGN * K_L1_ALIGN);

    tilingData_.set_cvLoopKSize(1024);
    const bool isKSplit = kDimNum != 1;
    if (!isKSplit && hasPreMix) {
        mDimNum = aicCoreNum_;
        tilingData_.set_cubeBlockDimM(mDimNum);
        tilingData_.set_mL1Size(SelectPremixML1Size(bs_, mDimNum, kSize));
        tilingData_.set_kL1Size(
            std::min(A_L1_SIZE / tilingData_.get_mL1Size(), K_L1_MAX_SIZE) / K_L1_ALIGN * K_L1_ALIGN);
    }
    tilingKey_ = (hasPreMix ? PREMIX_MK_KEY : NO_PREMIX_MK_KEY) + (isKSplit ? 0 : 1);
    OPS_LOG_I(context_->GetNodeName(),
              "HcPre arch35 tiling decision: key=%lu, premix=%d, kSplit=%d, "
              "mDim=%lu, kDim=%lu, kBlocks=%lu, splitK=%lu, mL1=%ld, kL1=%ld",
              tilingKey_, hasPreMix ? 1 : 0, isKSplit ? 1 : 0, mDimNum, kDimNum,
              actualKBlockNum, splitKSize, tilingData_.get_mL1Size(), tilingData_.get_kL1Size());
    if (isKSplit) {
        return hasPreMix ? CalcPremixMKSplitTiling() : CalcMKSplitCorePart2Tiling();
    }
    return hasPreMix ? CalcPremixMSplitTiling() : CalcRegbaseOpTiling();
}


ge::graphStatus HcPreTilingRegbase::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::GetWorkspaceSize()
{
    if (tilingKey_ == NO_PREMIX_MK_KEY || tilingKey_ == PREMIX_MK_KEY) {
        // K分核模板需要预留Workspace大小
        workspaceSize_ = tilingData_.get_kBlockFactor() * tilingData_.get_bs() * tilingData_.get_hcMix() * 4 + tilingData_.get_kBlockFactor() * tilingData_.get_bs() * 4 + 16 * 1024 * 1024;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::PostTiling()
{
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(aicCoreNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}
}  // namespace optiling
