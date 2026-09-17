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
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
constexpr uint64_t A_L1_SIZE = 128 * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
constexpr int64_t K_L1_ALIGN_SIZE = 128;
constexpr int64_t HC_MULT_ATTR_IDX = 0;
constexpr int64_t ITER_TIMES_ATTR_IDX = 1;
constexpr int64_t HC_EPS_ATTR_IDX = 2;
constexpr int64_t NORM_EPS_ATTR_IDX = 3;
constexpr int64_t DEFAULT_ITER_TIMES = 20;
constexpr int64_t X_DTYPE_SIZE = 2;              // x and y are bfloat16
constexpr int64_t PRE_POST_MIX_NUM = 2;          // the mixes01 buffers carry pre and post together
constexpr int64_t SQUARE_SUM_SIZE = 16;          // must match SQUARE_SUM_SIZE in the kernel
constexpr int64_t MASK_PATTERN_ELEM_NUM = 128;   // MASK_PATTERN_BASE_SIZE * MASK_PATTERN_REPEAT_SIZE
// The batched sinkhorn column stage addresses one row of the hc_mult x hc_mult
// matrix per UB block, and needs at least two rows to accumulate.
constexpr int64_t MIN_HC_MULT = 2;
constexpr int64_t MAX_HC_MULT = BLOCK_SIZE / static_cast<int64_t>(sizeof(float));
// Upper bound on the comb fragment rows staged for one batched sinkhorn burst. The
// column stage of an iteration costs the same few instructions for one row or for
// many, so the gain flattens out well before this point and the remaining UB is
// worth more to the x and y buffers. Keep COMB_ROW_FACTOR_MAX * MAX_HC_MULT within
// the 255 repeat limit of the WholeReduceSum in the row stage.
constexpr int64_t COMB_ROW_FACTOR_MAX = 32;
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
    }

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

    tilingData_.set_hasPreMix(context_->GetInputShape(4) != nullptr ? 1 : 0);
    tilingData_.set_hasPreOut(context_->GetOutputShape(3) != nullptr ? 1 : 0);

    OPS_ERR_IF(GetAttr() != ge::GRAPH_SUCCESS,
                  OPS_LOG_E(context_->GetNodeName(), "get attr failed."),
                  return ge::GRAPH_FAILED);

    // GetAttr has the final word on hcMult_. The batched sinkhorn column stage in
    // stage 2 addresses one row of the hc_mult x hc_mult matrix per UB block, and
    // needs at least two rows to accumulate.
    OPS_ERR_IF(hcMult_ < MIN_HC_MULT || hcMult_ > MAX_HC_MULT,
                    OPS_LOG_E(context_->GetNodeName(),
                             "hc_mult should be in [%ld, %ld], but is %ld", MIN_HC_MULT, MAX_HC_MULT, hcMult_),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


// UB bytes HcPreMembaseKSplitCorePart2::Init allocates for a given split. Mirrors
// its InitBuffer calls one for one, so it is the place to keep in step when a buffer
// is added there. combRowFactor only feeds the three sinkhorn staging buffers.
int64_t HcPreTiling::CalcStage2UbSize(int64_t rowFactor, int64_t dFactor, int64_t combRowFactor) const
{
    const int64_t kBlockNum = tilingData_.get_cubeBlockDimK();
    const int64_t floatSize = static_cast<int64_t>(sizeof(float));
    const int64_t floatPerBlock = BLOCK_SIZE / floatSize;
    // the kernel aligns the d axis on x's dtype for both the b16 and the fp32 copies
    const int64_t dFactorAlign = RoundUp(dFactor, BLOCK_SIZE / X_DTYPE_SIZE);
    const int64_t xElemNum = rowFactor * hcMult_ * dFactorAlign;

    int64_t queSize = DOUBLE_BUFFER *
        (kBlockNum * rowFactor * hcMultAlign_ * PRE_POST_MIX_NUM * floatSize +  // mixesQue01
         kBlockNum * rowFactor * hcMult_ * hcMultAlign_ * floatSize +           // mixesQue2
         kBlockNum * rowFactor * SQUARE_SUM_SIZE * floatSize +                  // squareSumQue
         xElemNum * X_DTYPE_SIZE +                                             // xQue
         rowFactor * dFactorAlign * X_DTYPE_SIZE +                             // yQue
         rowFactor * hcMultAlign_ * floatSize +                                // postQue
         combRowFactor * hcMult_ * hcMultAlign_ * floatSize);                  // combFragQue

    int64_t bufSize =
        (PRE_POST_MIX_NUM + hcMult_) * hcMultAlign_ * floatSize +              // hcBaseBuf0/1/2
        RoundUp(rowFactor, floatPerBlock) * BLOCK_SIZE +                       // rowBrcbBuf0
        RoundUp(combRowFactor * hcMultAlign_, floatPerBlock) * BLOCK_SIZE +    // hcBrcbBuf1
        combRowFactor * hcMultAlign_ * floatSize +                             // reduceBuf
        rowFactor * hcMultAlign_ * PRE_POST_MIX_NUM * floatSize +              // mixes01ReduceBuf
        rowFactor * hcMultAlign_ * hcMult_ * floatSize +                       // mixes02ReduceBuf
        rowFactor * hcMultAlign_ * floatSize +                                 // squareReduceBuf
        xElemNum * floatSize +                                                 // xCastBuf
        rowFactor * dFactorAlign * floatSize +                                 // yCastBuf
        RoundUp(rowFactor, floatPerBlock) * floatSize +                        // rsqrtBuf
        RoundUp(MASK_PATTERN_ELEM_NUM, floatPerBlock) * floatSize +            // maskPatternBuf
        rowFactor * hcMultAlign_ * floatSize;                                  // preMixBuf

    return queSize + bufSize;
}

// Pick how many comb fragment rows stage 2 stages before it runs the sinkhorn
// iterations over them. Growing this only costs the three staging buffers, so it is
// decided here against the real UB budget rather than fixed in the kernel: at
// d = 4096 the x buffers leave little slack, at d = 7168 dFactor is halved and the
// slack pays for a bigger burst.
// The chunk grows in whole stage2RowFactor groups because the stage-2 loop stages
// one group per iteration and flushes once the chunk is full; a bound that is not a
// multiple of the group size could be stepped over. Starting from stage2RowFactor
// means a UB budget with no slack at all reproduces the unbatched footprint exactly.
void HcPreTiling::CalcCombRowFactor()
{
    int64_t maxCombRows = std::min(COMB_ROW_FACTOR_MAX, rowOfFormerBlock_);
    combRowFactor_ = rowFactor_;
    while (combRowFactor_ + rowFactor_ <= maxCombRows &&
           CalcStage2UbSize(rowFactor_, dFactor_, combRowFactor_ + rowFactor_) <= static_cast<int64_t>(ubSize_)) {
        combRowFactor_ += rowFactor_;
    }
}

ge::graphStatus HcPreTiling::CalcMKSplitCoreMembasePart2Tiling()
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
    CalcCombRowFactor();

    tilingData_.set_bs(bs_);
    tilingData_.set_hcMix(hcMix_);
    tilingData_.set_hcMult(hcMult_);
    tilingData_.set_d(d_);
    tilingData_.set_hcMultAlign(hcMultAlign_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_stage2RowFactor(rowFactor_);
    tilingData_.set_secondUsedCoreNum(usedAivCoreNums_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_combRowFactor(combRowFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTiling::CalcOpTiling() {
    uint64_t kSize = hcMult_ * d_;
    tilingData_.set_k(kSize);
    // 计算bs_轴切核
    uint64_t mDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);
    uint64_t kDimNum = aicCoreNum_ / mDimNum;
    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);

    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_cubeBlockDimK(CeilDiv(kSize, splitKSize));
    tilingData_.set_multCoreSplitMSize(singleCoreM);
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
    tilingData_.set_kL1Size(std::min(A_L1_SIZE / tilingData_.get_mL1Size(),
    static_cast<uint64_t>(K_L1_MAX_SIZE)) / K_L1_ALIGN_SIZE * K_L1_ALIGN_SIZE);

    tilingData_.set_cvLoopKSize(1024);

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
    context_->SetTilingKey(0);
    context_->SetBlockDim(aicCoreNum_);
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
