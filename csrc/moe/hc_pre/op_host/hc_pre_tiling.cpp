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
 * \file hc_pre_tiling.cpp
 * \brief
 */

#include <sstream>
#include "hc_pre_tiling.h"
#include "hc_pre_tiling_arch35.h"

using namespace ge;
namespace optiling {
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
constexpr int64_t UB_RESEVED_SIZE = 8192;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr uint64_t M_L1_MAX_SIZE = 256;
constexpr uint64_t CV_LOOP_K_SIZE = 1024;
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
constexpr uint64_t A_L1_SIZE = 128 * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
constexpr int64_t K_L1_ALIGN_SIZE = 128;
constexpr int64_t HC_MULT_ATTR_IDX = 0;
constexpr int64_t ITER_TIMES_ATTR_IDX = 1;
constexpr int64_t HC_EPS_ATTR_IDX = 2;
constexpr int64_t NORM_EPS_ATTR_IDX = 3;
constexpr int64_t DEFAULT_ITER_TIMES = 20;
constexpr int64_t SUPPORTED_HC_MULT = 4;
constexpr int64_t SQUARE_SUM_SIZE = 16;
constexpr int64_t MASK_PATTERN_DIM_SIZE = 16;
constexpr int64_t PRE_POST_ROW_WORK_SCALE = 3;
constexpr int64_t SINKHORN_ROW_WORK_SCALE = 64;
constexpr int64_t SINKHORN_SETUP_ITER_EQUIVALENT = 4;
constexpr int64_t SMALL_BS_THRESHOLD = 32;
constexpr uint64_t MPMD_TILING_KEY = 0;
constexpr uint64_t ORIGINAL_TILING_KEY = 1;
constexpr uint64_t PREMIX_TILING_KEY = 2;
}

ge::graphStatus HcPreTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<HcPreCompileInfo>();
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_LOG_E(context_, "compile info is null"),
                      return ge::GRAPH_FAILED);
        aivCoreNum_ = compileInfoPtr->coreNum;
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

ge::graphStatus HcPreTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    auto hcMultAttr = attrs->GetAttrPointer<int64_t>(HC_MULT_ATTR_IDX);
    hcMult_ = hcMultAttr == nullptr ? 4 : *hcMultAttr;

    auto iterTimesAttr = attrs->GetAttrPointer<int64_t>(ITER_TIMES_ATTR_IDX);
    iterTimes_ = iterTimesAttr == nullptr ? DEFAULT_ITER_TIMES : *iterTimesAttr;

    auto hcEpsAttr = attrs->GetAttrPointer<float>(HC_EPS_ATTR_IDX);
    hcEps_ = hcEpsAttr == nullptr ? 1e-6 : *hcEpsAttr;

    auto normEpsAttr = attrs->GetAttrPointer<float>(NORM_EPS_ATTR_IDX);
    normEps_ = normEpsAttr == nullptr ? 1e-6 : *normEpsAttr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTiling::GetShapeAttrsInfoInner()
{
    OPS_ERR_IF(GetAttr() != ge::GRAPH_SUCCESS,
                  OPS_LOG_E(context_->GetNodeName(), "get attr failed."),
                  return ge::GRAPH_FAILED);

    // (b, s, hc_mult, d) or (bs, hc_mult, d)
    auto xShape = context_->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context_, xShape, return ge::GRAPH_FAILED);
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    int64_t shapeHcMult = 0;
    if (xDimNum == 3) {
        bs_ = xShape->GetStorageShape().GetDim(0);
        shapeHcMult = xShape->GetStorageShape().GetDim(1);
        d_ = xShape->GetStorageShape().GetDim(2);
    } else if (xDimNum == 4) {
        int64_t b = xShape->GetStorageShape().GetDim(0);
        int64_t s = xShape->GetStorageShape().GetDim(1);
        bs_ = b * s;
        shapeHcMult = xShape->GetStorageShape().GetDim(2);
        d_ = xShape->GetStorageShape().GetDim(3);
    }
    OPS_ERR_IF(shapeHcMult != hcMult_,
                OPS_LOG_E(context_->GetNodeName(),
                          "hc_mult attr %ld should match x hc_mult dimension %ld", hcMult_, shapeHcMult),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(hcMult_ != SUPPORTED_HC_MULT,
                OPS_LOG_E(context_->GetNodeName(),
                          "HcPre only supports hc_mult=%ld, but got %ld", SUPPORTED_HC_MULT, hcMult_),
                return ge::GRAPH_FAILED);

    auto shapeHcFn = context_->GetInputShape(1);
    hcMix_ = shapeHcFn->GetStorageShape().GetDim(0);
    OPS_ERR_IF(shapeHcFn->GetStorageShape().GetDim(1) != d_ * hcMult_,
                OPS_LOG_E(context_->GetNodeName(),
                "HcFn dim 1 should be equal with d_ * hcMult_  %ld, but is %ld",
                d_ * hcMult_, shapeHcFn->GetStorageShape().GetDim(1)),
                return ge::GRAPH_FAILED);

    auto shapeHcScale = context_->GetInputShape(2);
    int64_t scaleFirstDim = shapeHcScale->GetStorageShape().GetDim(0);
    OPS_ERR_IF(scaleFirstDim != 3,
                    OPS_LOG_E(context_->GetNodeName(),
                             "hc_scale size should be equal with 3, but is %ld", scaleFirstDim),
                    return ge::GRAPH_FAILED);

    auto shapeHcBase = context_->GetInputShape(3);
    int64_t baseFirstDim = shapeHcBase->GetStorageShape().GetDim(0);
    OPS_ERR_IF(baseFirstDim != hcMix_,
                    OPS_LOG_E(context_->GetNodeName(),
                             "hc_base size should be equal with mixhc, but is %ld", baseFirstDim),
                    return ge::GRAPH_FAILED);

    // 可选输入 pre_mix: [bs, hc_mult] fp32; gen_placeholder模式下GetInputShape恒非空, 必须用GetOptionalInputTensor判空
    tilingData_.set_hasPreMix(context_->GetOptionalInputTensor(4) != nullptr ? 1 : 0);
    // 可选输出 pre: 无GetOptionalOutputTensor对应API, 用shape内容判别(占位shape无有效首维)
    auto preOutShape = context_->GetOutputShape(3);
    bool hasPreOut = preOutShape != nullptr && preOutShape->GetStorageShape().GetDimNum() > 0 &&
                     preOutShape->GetStorageShape().GetDim(0) > 0;
    tilingData_.set_hasPreOut(hasPreOut ? 1 : 0);

    return ge::GRAPH_SUCCESS;
}


void HcPreTiling::CalcVectorCoreNum()
{
    int64_t totalCoreNum = static_cast<int64_t>(aivCoreNum_);
    int64_t effectiveIterTimes = std::max(iterTimes_, static_cast<int64_t>(1));

    // Estimate the work per row and test each core split with rounded row counts.
    uint64_t prePostRowWork = static_cast<uint64_t>(PRE_POST_ROW_WORK_SCALE) * d_;
    uint64_t sinkhornRowWork = static_cast<uint64_t>(SINKHORN_ROW_WORK_SCALE) * hcMult_ *
        (effectiveIterTimes + SINKHORN_SETUP_ITER_EQUIVALENT);
    uint64_t bestLatency = 0;
    uint64_t bestImbalance = 0;
    bool hasCandidate = false;

    for (int64_t candidatePrePostCoreNum = 1; candidatePrePostCoreNum < totalCoreNum;
        candidatePrePostCoreNum++) {
        int64_t candidateSinkhornCoreNum = totalCoreNum - candidatePrePostCoreNum;
        uint64_t prePostLatency = static_cast<uint64_t>(CeilDiv(bs_, candidatePrePostCoreNum)) *
            prePostRowWork;
        uint64_t sinkhornLatency = static_cast<uint64_t>(CeilDiv(bs_, candidateSinkhornCoreNum)) *
            sinkhornRowWork;
        uint64_t latency = std::max(prePostLatency, sinkhornLatency);
        uint64_t imbalance = prePostLatency > sinkhornLatency ?
            prePostLatency - sinkhornLatency : sinkhornLatency - prePostLatency;
        if (!hasCandidate || latency < bestLatency ||
            (latency == bestLatency && imbalance < bestImbalance)) {
            hasCandidate = true;
            bestLatency = latency;
            bestImbalance = imbalance;
            prePostCoreNum_ = candidatePrePostCoreNum;
            sinkhornCoreNum_ = candidateSinkhornCoreNum;
        }
    }
}

ge::graphStatus HcPreTiling::CalcMKSplitCoreMembasePart2Tiling()
{
    // preMix 场景使用独立的 tilingKey=2 和按行分配策略。
    if (tilingData_.get_hasPreMix() != 0) {
        return CalcPremixPart2Tiling();
    }
    bool useOriginal = bs_ <= SMALL_BS_THRESHOLD;
    if (useOriginal) {
        tilingKey_ = ORIGINAL_TILING_KEY;
        rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    } else {
        tilingKey_ = MPMD_TILING_KEY;
        OPS_ERR_IF(aivCoreNum_ < 2,
            OPS_LOG_E(context_->GetNodeName(),
                "HcPre MPMD requires at least two vector cores, but got %lu", aivCoreNum_),
            return ge::GRAPH_FAILED);
        CalcVectorCoreNum();
        rowOfFormerBlock_ = CeilDiv(bs_, prePostCoreNum_);
    }

    int64_t stage2CoreNum = useOriginal ? static_cast<int64_t>(aivCoreNum_) : prePostCoreNum_;
    usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), stage2CoreNum);
    rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    int64_t mix0OriginSize = kBlockNum * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t mix1OriginSize = kBlockNum * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t mix0Size = rowOnceLoop * hcMultAlign_ * sizeof(float);
    int64_t mix1Size = rowOnceLoop * hcMultAlign_ * sizeof(float);
    int64_t squareSumSize = kBlockNum * RoundUp(rowOnceLoop * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
    int64_t squareReduceSize = rowOnceLoop * SQUARE_SUM_SIZE * sizeof(float);
    int64_t rsqrtSize = RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t xCastSize = rowOnceLoop * hcMult_ * RoundUp(d_, 8) * sizeof(float);
    int64_t yCastSize = rowOnceLoop * RoundUp(d_, 8) * sizeof(float);
    int64_t rowBrcb0Size = RoundUp(rowOnceLoop, 8) * BLOCK_SIZE;
    int64_t hcBrcb1Size = RoundUp(rowOnceLoop * hcMultAlign_, 8) * BLOCK_SIZE;
    int64_t maskPatternSize = (rowOnceLoop + MASK_PATTERN_DIM_SIZE - 1) * BLOCK_SIZE;
    auto calcOriginalExtraSize = [&](int64_t rowFactor) {
        if (!useOriginal) {
            return static_cast<int64_t>(0);
        }
        int64_t mix2OriginSize = kBlockNum * rowFactor * hcMult_ * hcMultAlign_ *
            sizeof(float) * DOUBLE_BUFFER;
        int64_t combFragSize = rowFactor * hcMult_ * hcMultAlign_ *
            sizeof(float) * DOUBLE_BUFFER;
        int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);
        int64_t reduceBufSize = rowFactor * hcMultAlign_ * sizeof(float);
        int64_t mix2Size = rowFactor * hcMult_ * hcMultAlign_ * sizeof(float);
        return mix2OriginSize + combFragSize + base2Size + reduceBufSize + mix2Size;
    };
    int64_t originalExtraSize = calcOriginalExtraSize(rowOnceLoop);

    int64_t totalSize = mix0OriginSize + mix1OriginSize + mix0Size + mix1Size +
                     squareSumSize + squareReduceSize + rsqrtSize + xSize + ySize + postSize + base0Size + base1Size +
                     xCastSize + yCastSize +
                     rowBrcb0Size + hcBrcb1Size + maskPatternSize + originalExtraSize;
    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mix0OriginSize + mix1OriginSize + mix0Size + mix1Size +
                     squareSumSize + squareReduceSize + rsqrtSize +
                     postSize + base0Size + base1Size +
                     rowBrcb0Size + hcBrcb1Size + maskPatternSize + originalExtraSize;
        int64_t ubRemain = ubSize_ - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            xCastSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 8) * sizeof(float);
            yCastSize = rowOnceLoop * RoundUp(dFactor_, 8) * sizeof(float);
            int64_t targetSize = xSize + ySize + xCastSize + yCastSize;
            if (targetSize <= ubRemain) {
                break;
            }
            base++;
        }
        if (dFactor_ > 32) {
            dFactor_ = DownAlign(dFactor_, 32);
        }
        dLoop_ = CeilDiv(d_, dFactor_);
        tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
    }

    // d全载,尝试搬入更多的bs
    if (dFactor_ == d_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            mix0OriginSize = kBlockNum * rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            mix1OriginSize = kBlockNum * rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            mix0Size = rowFactor_ * hcMultAlign_ * sizeof(float);
            mix1Size = rowFactor_ * hcMultAlign_ * sizeof(float);
            squareSumSize = kBlockNum * RoundUp(rowFactor_ * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
            squareReduceSize = rowFactor_ * SQUARE_SUM_SIZE * sizeof(float);
            rsqrtSize = RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
            xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            xCastSize = rowFactor_ * hcMult_ * RoundUp(d_, 8) * sizeof(float);
            yCastSize = rowFactor_ * RoundUp(d_, 8) * sizeof(float);
            rowBrcb0Size = RoundUp(rowFactor_, 8) * BLOCK_SIZE;
            hcBrcb1Size = RoundUp(rowFactor_ * hcMultAlign_, 8) * BLOCK_SIZE;
            maskPatternSize = (rowFactor_ + MASK_PATTERN_DIM_SIZE - 1) * BLOCK_SIZE;
            originalExtraSize = calcOriginalExtraSize(rowFactor_);
            totalSize = mix0OriginSize + mix1OriginSize + mix0Size + mix1Size +
            squareSumSize + squareReduceSize + rsqrtSize +
            xSize + ySize + postSize + base0Size + base1Size + xCastSize + yCastSize +
            rowBrcb0Size + hcBrcb1Size + maskPatternSize + originalExtraSize;
            if (totalSize > ubSize_) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
    }

    // The pre/post transforms only need row-sized state; y still uses the
    // full-D buffers above. Give pre/post its own row tile so a full-D shape
    // does not force sigmoid and RMSNorm setup to run once per row.
    int64_t prePostRowFactor = rowFactor_;
    if (!useOriginal) {
        auto calcPrePostSize = [&](int64_t candidateRow) {
            int64_t originSize = kBlockNum * candidateRow * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            int64_t reduceSize = candidateRow * hcMultAlign_ * sizeof(float);
            int64_t squareSize = kBlockNum * RoundUp(candidateRow * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
            int64_t squareReduce = candidateRow * SQUARE_SUM_SIZE * sizeof(float);
            int64_t rsqrt = RoundUp(candidateRow, BLOCK_SIZE / sizeof(float)) * sizeof(float);
            int64_t post = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            int64_t rowBrcb = RoundUp(candidateRow, BLOCK_SIZE / sizeof(float)) * BLOCK_SIZE;
            int64_t hcBrcb = RoundUp(candidateRow * hcMultAlign_, BLOCK_SIZE / sizeof(float)) * BLOCK_SIZE;
            int64_t mask = (candidateRow + MASK_PATTERN_DIM_SIZE - 1) * BLOCK_SIZE;
            int64_t yX = rowFactor_ * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t yY = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t yXCast = rowFactor_ * hcMult_ * RoundUp(dFactor_, 8) * sizeof(float);
            int64_t yYCast = rowFactor_ * RoundUp(dFactor_, 8) * sizeof(float);
            return originSize * 2 + reduceSize * 2 + squareSize + squareReduce + rsqrt +
                post + base0Size + base1Size + yX + yY + yXCast + yYCast + rowBrcb + hcBrcb + mask;
        };
        prePostRowFactor = 1;
        for (int64_t candidateRow = 1; candidateRow <= rowOfFormerBlock_; candidateRow++) {
            if (calcPrePostSize(candidateRow) > static_cast<int64_t>(ubSize_)) {
                break;
            }
            prePostRowFactor = candidateRow;
        }
        // Cap the pre/post row batch at four after checking UB capacity.
        if (prePostRowFactor > 4) {
            prePostRowFactor = 4;
        }
    }
    int64_t prePostRowLoopOfFormerBlock = CeilDiv(rowOfFormerBlock_, prePostRowFactor);
    int64_t prePostRowLoopOfTailBlock = CeilDiv(rowOfTailBlock_, prePostRowFactor);
    int64_t prePostTailRowFactorOfFormerBlock = rowOfFormerBlock_ % prePostRowFactor == 0 ?
        prePostRowFactor : rowOfFormerBlock_ % prePostRowFactor;
    int64_t prePostTailRowFactorOfTailBlock = rowOfTailBlock_ % prePostRowFactor == 0 ?
        prePostRowFactor : rowOfTailBlock_ % prePostRowFactor;
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;

    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(hcMultAlign_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_stage2RowFactor(rowFactor_);
    tilingData_.set_secondUsedCoreNum(usedAivCoreNums_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    tilingData_.set_prePostCoreNum(prePostCoreNum_);
    tilingData_.set_sinkhornCoreNum(sinkhornCoreNum_);
    tilingData_.set_prePostRowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_prePostRowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_prePostRowLoopOfFormerBlock(prePostRowLoopOfFormerBlock);
    tilingData_.set_prePostRowLoopOfTailBlock(prePostRowLoopOfTailBlock);
    tilingData_.set_prePostRowFactor(prePostRowFactor);
    tilingData_.set_prePostTailRowFactorOfFormerBlock(prePostTailRowFactorOfFormerBlock);
    tilingData_.set_prePostTailRowFactorOfTailBlock(prePostTailRowFactorOfTailBlock);
    if (useOriginal) {
        return ge::GRAPH_SUCCESS;
    }
    return CalcSinkhornTiling();
}

// preMix 场景的 Part2 tiling：按行分配，UB 预算包含 mix2、combFrag、base2 和 reduceBuf。
ge::graphStatus HcPreTiling::CalcPremixPart2Tiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    int64_t mix0OriginSize = kBlockNum * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t mix1OriginSize = kBlockNum * rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t mix2OriginSize = kBlockNum * rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t mix0Size = rowOnceLoop * hcMultAlign_ * sizeof(float);
    int64_t mix1Size = rowOnceLoop * hcMultAlign_ * sizeof(float);
    int64_t mix2Size = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float);
    int64_t squareSumSize = kBlockNum * RoundUp(rowOnceLoop * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
    int64_t rsqrtSize = RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);
    int64_t xCastSize = rowOnceLoop * hcMult_ * RoundUp(d_, 8) * sizeof(float);
    int64_t yCastSize = rowOnceLoop * RoundUp(d_, 8) * sizeof(float);
    int64_t rowBrcb0Size = RoundUp(rowOnceLoop, 8) * BLOCK_SIZE;
    int64_t hcBrcb1Size = RoundUp(rowOnceLoop * hcMultAlign_, 8) * BLOCK_SIZE;
    int64_t reduceBufSize = rowOnceLoop * hcMultAlign_ * sizeof(float);
    int64_t maskPatternSize = BLOCK_SIZE * 16;

    int64_t totalSize = mix0OriginSize + mix1OriginSize + mix2OriginSize +
                     mix0Size + mix1Size + mix2Size + squareSumSize + rsqrtSize +
                     xSize + ySize + postSize + combFragSize + base0Size + base1Size +
                     base2Size + xCastSize + yCastSize +
                     rowBrcb0Size + hcBrcb1Size + reduceBufSize + maskPatternSize;
    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mix0OriginSize + mix1OriginSize + mix2OriginSize +
                     mix0Size + mix1Size + mix2Size + squareSumSize + rsqrtSize +
                     postSize + combFragSize + base0Size + base1Size + base2Size +
                     rowBrcb0Size + hcBrcb1Size + reduceBufSize + maskPatternSize;
        int64_t ubRemain = ubSize_ - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            xCastSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 8) * sizeof(float);
            yCastSize = rowOnceLoop * RoundUp(dFactor_, 8) * sizeof(float);
            int64_t targetSize = xSize + ySize + xCastSize + yCastSize;
            if (targetSize <= ubRemain) {
                break;
            }
            base++;
        }
        if (dFactor_ > 32) {
            dFactor_ = DownAlign(dFactor_, 32);
        }
        dLoop_ = CeilDiv(d_, dFactor_);
        tailDFactor_ = d_ % dFactor_ == 0 ? dFactor_ : d_ % dFactor_;
    }

    // d全载,尝试搬入更多的bs
    if (dFactor_ == d_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            mix0OriginSize = kBlockNum * rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            mix1OriginSize = kBlockNum * rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            mix2OriginSize = kBlockNum * rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            mix0Size = rowFactor_ * hcMultAlign_ * sizeof(float);
            mix1Size = rowFactor_ * hcMultAlign_ * sizeof(float);
            mix2Size = rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float);
            squareSumSize = kBlockNum * RoundUp(rowFactor_ * 16, 16) * sizeof(float) * DOUBLE_BUFFER;
            rsqrtSize = RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            xCastSize = rowFactor_ * hcMult_ * RoundUp(d_, 8) * sizeof(float);
            yCastSize = rowFactor_ * RoundUp(d_, 8) * sizeof(float);
            rowBrcb0Size = RoundUp(rowFactor_, 8) * BLOCK_SIZE;
            hcBrcb1Size = RoundUp(rowFactor_ * hcMultAlign_, 8) * BLOCK_SIZE;
            reduceBufSize = rowFactor_ * hcMultAlign_ * sizeof(float);
            maskPatternSize = BLOCK_SIZE;
            totalSize = mix0OriginSize + mix1OriginSize + mix2OriginSize +
            mix0Size + mix1Size + mix2Size + squareSumSize + rsqrtSize +
            xSize + ySize + postSize + combFragSize + base0Size + base1Size +
            base2Size + xCastSize + yCastSize +
            rowBrcb0Size + hcBrcb1Size + reduceBufSize + maskPatternSize;
            if (totalSize > ubSize_) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > rowOfFormerBlock_ ? rowFactor_ - 1 : rowFactor_;
    }
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;

    tilingKey_ = PREMIX_TILING_KEY;
    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(hcMultAlign_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_stage2RowFactor(rowFactor_);
    tilingData_.set_secondUsedCoreNum(usedAivCoreNums_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTiling::CalcSinkhornTiling()
{
    sinkhornRowOfFormerBlock_ = CeilDiv(bs_, sinkhornCoreNum_);
    sinkhornUsedCoreNum_ = std::min(CeilDiv(bs_, sinkhornRowOfFormerBlock_), sinkhornCoreNum_);
    sinkhornRowOfTailBlock_ = bs_ - (sinkhornUsedCoreNum_ - 1) * sinkhornRowOfFormerBlock_;

    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();
    auto calcTotalSize = [&](int64_t rowFactor) {
        int64_t rowFactorAlign = RoundUp(rowFactor, BLOCK_SIZE / sizeof(float));
        int64_t matrixSize = hcMult_ * hcMult_;
        int64_t mix2OriginSize = kBlockNum * rowFactor * matrixSize * sizeof(float) * DOUBLE_BUFFER;
        int64_t combFragSize = rowFactor * matrixSize * sizeof(float) * DOUBLE_BUFFER;
        int64_t squareSumSize = kBlockNum * RoundUp(rowFactor * SQUARE_SUM_SIZE, SQUARE_SUM_SIZE) *
                                sizeof(float) * DOUBLE_BUFFER;
        int64_t rsqrtSize = RoundUp(rowFactor, BLOCK_SIZE / sizeof(float)) * sizeof(float);
        int64_t squareReduceSize = rowFactor * SQUARE_SUM_SIZE * sizeof(float);
        int64_t mixes02ReduceSize = matrixSize * rowFactorAlign * sizeof(float);
        int64_t reduceBufSize = hcMult_ * rowFactorAlign * sizeof(float);
        int64_t rowBrcb0Size = RoundUp(rowFactor, BLOCK_SIZE / sizeof(float)) * BLOCK_SIZE;
        int64_t hcBrcb1Size = matrixSize * BLOCK_SIZE;
        int64_t maskPatternSize = (rowFactor + MASK_PATTERN_DIM_SIZE - 1) * BLOCK_SIZE;
        return mix2OriginSize + combFragSize + squareSumSize + rsqrtSize + squareReduceSize +
               mixes02ReduceSize + reduceBufSize + rowBrcb0Size + hcBrcb1Size + maskPatternSize;
    };

    int64_t base2Size = hcMult_ * hcMult_ * sizeof(float);

    int64_t maxSinkhornRowFactor = std::min(sinkhornRowOfFormerBlock_, static_cast<int64_t>(64));
    // Keep double buffering for generalized batch sizes. The row-factor search
    // accounts for both queue slots so the allocation stays in UB.
    sinkhornRowFactor_ = 1;
    while (sinkhornRowFactor_ <= maxSinkhornRowFactor) {
        if (calcTotalSize(sinkhornRowFactor_) + base2Size > static_cast<int64_t>(ubSize_)) {
            sinkhornRowFactor_--;
            break;
        }
        sinkhornRowFactor_++;
    }
    sinkhornRowFactor_ = std::min(sinkhornRowFactor_, maxSinkhornRowFactor);
    OPS_ERR_IF(sinkhornRowFactor_ < 1,
        OPS_LOG_E(context_->GetNodeName(), "HcPre sinkhorn buffers exceed UB size %lu", ubSize_),
        return ge::GRAPH_FAILED);

    sinkhornRowLoopOfFormerBlock_ = CeilDiv(sinkhornRowOfFormerBlock_, sinkhornRowFactor_);
    sinkhornRowLoopOfTailBlock_ = CeilDiv(sinkhornRowOfTailBlock_, sinkhornRowFactor_);
    sinkhornTailRowFactorOfFormerBlock_ = sinkhornRowOfFormerBlock_ % sinkhornRowFactor_ == 0 ?
        sinkhornRowFactor_ : sinkhornRowOfFormerBlock_ % sinkhornRowFactor_;
    sinkhornTailRowFactorOfTailBlock_ = sinkhornRowOfTailBlock_ % sinkhornRowFactor_ == 0 ?
        sinkhornRowFactor_ : sinkhornRowOfTailBlock_ % sinkhornRowFactor_;

    tilingData_.set_sinkhornUsedCoreNum(sinkhornUsedCoreNum_);
    tilingData_.set_sinkhornRowOfFormerBlock(sinkhornRowOfFormerBlock_);
    tilingData_.set_sinkhornRowOfTailBlock(sinkhornRowOfTailBlock_);
    tilingData_.set_sinkhornRowLoopOfFormerBlock(sinkhornRowLoopOfFormerBlock_);
    tilingData_.set_sinkhornRowLoopOfTailBlock(sinkhornRowLoopOfTailBlock_);
    tilingData_.set_sinkhornRowFactor(sinkhornRowFactor_);
    tilingData_.set_sinkhornTailRowFactorOfFormerBlock(sinkhornTailRowFactorOfFormerBlock_);
    tilingData_.set_sinkhornTailRowFactorOfTailBlock(sinkhornTailRowFactorOfTailBlock_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTiling::CalcOpTiling() {
    uint64_t kSize = hcMult_ * d_;
    tilingData_.set_k(kSize);
    // 计算bs_轴切核
    uint64_t mDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t kDimNum = aicCoreNum_ / mDimNum;
    if (kDimNum == 1) {
        // K cannot use the remaining AICs, so distribute M across all cores.
        mDimNum = aicCoreNum_;
    } else if (aicCoreNum_ % mDimNum != 0) {
        // Align mDimNum up to the nearest factor of aicCoreNum_ so that
        // mDimNum * kDimNum == aicCoreNum_ holds exactly. Without this the
        // integer division truncates kDimNum and the trailing AIC cores idle
        // (e.g. bs=2024: mDimNum=8, kDimNum=2, only 16 of 20 AICs run).
        for (uint64_t factor = mDimNum; factor <= aicCoreNum_; ++factor) {
            if (aicCoreNum_ % factor == 0) {
                mDimNum = factor;
                break;
            }
        }
    }
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, static_cast<int64_t>(mDimNum)), AscendC::BLOCK_CUBE);
    kDimNum = aicCoreNum_ / mDimNum;
    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);

    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_cubeBlockDimK(CeilDiv(kSize, splitKSize));
    tilingData_.set_multCoreSplitMSize(singleCoreM);
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
    tilingData_.set_kL1Size(std::min(A_L1_SIZE / tilingData_.get_mL1Size(),
    static_cast<uint64_t>(K_L1_MAX_SIZE)) / K_L1_ALIGN_SIZE * K_L1_ALIGN_SIZE);

    tilingData_.set_cvLoopKSize(CV_LOOP_K_SIZE);

    // vector stage1 tiling
    tilingData_.set_cubeCoreNum(static_cast<int64_t>(aicCoreNum_));
    // x type bfloat16, y type float32 and double
    // exit node 1 b16 input Queue and 1 b32 output Queue
    int64_t lineByteSize = (sizeof(int16_t) + sizeof(int32_t)) * DOUBLE_BUFFER * tilingData_.get_cvLoopKSize();
    int64_t stage1MFactorValue = ubSize_ / lineByteSize;
    tilingData_.set_stage1MFactor(stage1MFactorValue);
    return CalcMKSplitCoreMembasePart2Tiling();
}


ge::graphStatus HcPreTiling::DoOpTiling()
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

ge::graphStatus HcPreTiling::GetWorkspaceSize()
{
    uint64_t xCastFp32BufSize = tilingData_.get_mL1Size() * RoundUp(tilingData_.get_cvLoopKSize(), 128);
    uint64_t workspaceSize1 = tilingData_.get_cubeCoreNum() * DOUBLE_BUFFER * xCastFp32BufSize * sizeof(float);

    uint64_t mmLastAxisSize = RoundUp(tilingData_.get_hcMix(), 128);
    uint64_t workspaceSize2 = RoundUp(tilingData_.get_cubeBlockDimK() *
    tilingData_.get_bs() * mmLastAxisSize * sizeof(float), 512);

    uint64_t squareSumSize = RoundUp(tilingData_.get_cubeBlockDimK() *
    RoundUp(tilingData_.get_bs(), 16) * 16 * sizeof(float), 512);

    uint64_t requiredSize = workspaceSize1 + workspaceSize2 + squareSumSize + 16 * 1024 * 1024; // 16MB 预留缓冲

    uint64_t defaultSize = 16 * 1024 * 1024 + 192 * 1024 * 1024; // 208MB
    workspaceSize_ = requiredSize > defaultSize ? requiredSize : defaultSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTiling::PostTiling()
{
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(aicCoreNum_);
    if (tilingKey_ == MPMD_TILING_KEY) {
        auto ret = context_->SetScheduleMode(1);
        OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
            OPS_LOG_E(context_, "Failed to set schedule mode to 1 for HcPre."),
            return ge::GRAPH_FAILED);
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForHcPre(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForHcPre(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("HcPre", "Tiling context is null"),
               return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR("TilingForMoeGatingTopKHash",
    "Tiling platformInfo is null"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND950) {
        OPS_LOG_I(context, "Using arch35 tiling for ASCEND950");
        HcPreTilingRegbase::HcPreTilingRegbase hcPreTiling(context);
        return hcPreTiling.DoOpTiling();
    }
    HcPreTiling hcPreTiling(context);
    return hcPreTiling.DoOpTiling();
}

IMPL_OP_OPTILING(HcPre)
    .Tiling(TilingForHcPre)
    .TilingParse<HcPreCompileInfo>(TilingPrepareForHcPre);

}  // namespace optiling
