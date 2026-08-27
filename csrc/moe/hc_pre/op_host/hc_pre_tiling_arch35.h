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
constexpr uint64_t M_L1_MAX_SIZE = 256;
constexpr uint64_t K_MULIT_CORE_SPLIT_BASE_SIZE = 256;
constexpr uint64_t A_L1_SIZE = 128 * 256;
constexpr uint64_t K_L1_MAX_SIZE = 1024;
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
    ge::graphStatus CalcRegbaseOpTiling();
    ge::graphStatus CalcMKSplitCorePart2Tiling();
private:
    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_ = 0;
    HcPreTilingData tilingData_;
    uint64_t aivCoreNum_ = 0;
    uint64_t aicCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;
    uint64_t usedCoreNums_ = 0;
    uint64_t usedAivCoreNums_ = 0;
    uint64_t ubSize_ = 0;
    int64_t bs_ = 0;
    int64_t hcMix_ = 0;
    int64_t hcMult_ = 0;
    int64_t d_ = 0;
    int64_t hcMultAlign_ = 0;
    int64_t rowOfFormerBlock_ = 0;
    int64_t rowOfTailBlock_ = 0;
    int64_t rowLoopOfFormerBlock_ = 0;
    int64_t rowLoopOfTailBlock_ = 0;
    int64_t rowFactor_ = 0;
    int64_t tailRowFactorOfFormerBlock_ = 0;
    int64_t tailRowFactorOfTailBlock_= 0;
    int64_t dLoop_ = 0;
    int64_t dFactor_ = 0;
    int64_t tailDFactor_ = 0;
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
    }

    auto shapeHcFn = context_->GetInputShape(1);
    hcMix_ = shapeHcFn->GetStorageShape().GetDim(0);
    OPS_ERR_IF(shapeHcFn->GetStorageShape().GetDim(1) != d_ * hcMult_,
                    OPS_LOG_E(context_->GetNodeName(),
                             "HcFn dim 1 should be equal with d_ * hcMult_  %ld, but is %ld", d_ * hcMult_, shapeHcFn->GetStorageShape().GetDim(1)),
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

    OPS_ERR_IF(GetAttr() != ge::GRAPH_SUCCESS,
                  OPS_LOG_E(context_->GetNodeName(), "get attr failed."),
                  return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus HcPreTilingRegbase::CalcRegbaseOpTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    int64_t mixSize = rowOnceLoop * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    // comb is packed in UB as hcMult * hcMult, so size it by the actual footprint.
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMult_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);

    uint64_t kUbSize = tilingData_.get_kL1Size() / 2; // 先按2倍系数计算，m最大256，需保证kub小于256
    uint64_t mUbSize = CeilDiv(tilingData_.get_mL1Size(), 2);

    int64_t mmXBufSize = mUbSize * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t rmsNormBufSize =  RoundUp(mUbSize, BLOCK_SIZE / sizeof(float)) * sizeof(float);
    int64_t bufferPool0Size = ubSize_;
    int64_t bufferPool1Size = DownAlign(bufferPool0Size - mmXBufSize - rmsNormBufSize - base0Size - base1Size - base2Size, BLOCK_SIZE);

    int64_t totalSize = mixSize + xSize + ySize + postSize + combFragSize;
    rowFactor_ = rowOnceLoop;
    if (totalSize <= bufferPool1Size) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mixSize + postSize + combFragSize;
        int64_t ubRemain = bufferPool1Size - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t targetSize = xSize + ySize;
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
        while (rowFactor_ <= mUbSize) {
            mixSize = rowFactor_ * RoundUp(hcMix_, BLOCK_SIZE / sizeof(float)) * sizeof(float);
            xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMult_ * sizeof(float) * DOUBLE_BUFFER;
            totalSize = mixSize + xSize + ySize + postSize + combFragSize;
            if (totalSize > bufferPool1Size) {
                rowFactor_ = rowFactor_ - 1;
                break;
            }
            rowFactor_ = rowFactor_ + 1;
        }
        rowFactor_ = rowFactor_ > mUbSize ? rowFactor_ - 1 : rowFactor_;
    }

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
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_iterTimes(iterTimes_);
    tilingData_.set_hcEps(hcEps_);
    tilingData_.set_normEps(normEps_);

    tilingData_.set_bufferPool0Size(bufferPool0Size);
    tilingData_.set_bufferPool1Size(bufferPool1Size);

    tilingData_.set_kUbSize(kUbSize);
    tilingData_.set_mUbSize(mUbSize);

    tilingData_.set_kBlockFactor(tilingData_.get_cubeBlockDimK());

    tilingData_.set_rowInnerFactor(rowFactor_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::CalcMKSplitCorePart2Tiling()
{
    uint64_t kUbSize = tilingData_.get_kL1Size() / 2; // 先按2倍系数计算，m最大256，需保证kub小于256
    uint64_t mUbSize = CeilDiv(tilingData_.get_mL1Size(), 2);

    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(aivCoreNum_));
    usedAivCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(aivCoreNum_));
    rowOfTailBlock_ = bs_ - (usedAivCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);
    int64_t kBlockNum = tilingData_.get_cubeBlockDimK();

    hcMultAlign_ = RoundUp(hcMult_, BLOCK_SIZE / sizeof(float));
    uint64_t hcMixAlign = RoundUp(hcMix_, BLOCK_SIZE / sizeof(float));
    int64_t mmSize = kBlockNum * rowOnceLoop * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
    int64_t mixSize = rowOnceLoop * hcMixAlign * sizeof(float);
    int64_t rmsSize = kBlockNum * RoundUp(rowOnceLoop, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
    int64_t xSize = rowOnceLoop * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
    int64_t ySize = rowOnceLoop * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
    int64_t postSize = rowOnceLoop * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
    // comb is packed in UB as hcMult * hcMult, so size it by the actual footprint.
    int64_t combFragSize = rowOnceLoop * hcMult_ * hcMult_ * sizeof(float) * DOUBLE_BUFFER;
    int64_t base0Size = hcMultAlign_ * sizeof(float);
    int64_t base1Size = hcMultAlign_ * sizeof(float);
    int64_t base2Size = hcMult_ * hcMultAlign_ * sizeof(float);

    int64_t totalSize = mmSize + mixSize + rmsSize + xSize + ySize + postSize + combFragSize + base0Size + base1Size + base2Size;
    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = d_;
        tailDFactor_ = dFactor_;
    } else {
        int64_t usedUbSize = mmSize + mixSize + rmsSize + postSize + combFragSize + base0Size + base1Size + base2Size;
        int64_t ubRemain = ubSize_ - usedUbSize;
        dFactor_ = d_;
        int64_t base = 2;
        while (1) {
            dFactor_ = CeilDiv(d_, base);
            xSize = rowOnceLoop * hcMult_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            int64_t targetSize = xSize + ySize;
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
            mmSize = kBlockNum * rowFactor_ * hcMixAlign * sizeof(float) * DOUBLE_BUFFER;
            mixSize =  rowFactor_ * hcMixAlign * sizeof(float);
            rmsSize = kBlockNum * RoundUp(rowFactor_, BLOCK_SIZE / sizeof(float)) * sizeof(float) * DOUBLE_BUFFER;
            xSize = rowFactor_ * hcMult_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER; // x是bfloat16_t 类型
            ySize = rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER;
            postSize = rowFactor_ * hcMultAlign_ * sizeof(float) * DOUBLE_BUFFER;
            combFragSize = rowFactor_ * hcMult_ * hcMult_ * sizeof(float) * DOUBLE_BUFFER;
            base0Size = hcMultAlign_ * sizeof(float);
            base1Size = hcMultAlign_ * sizeof(float);
            base2Size = hcMult_ * hcMultAlign_ * sizeof(float);

            totalSize = mmSize + mixSize + rmsSize + xSize + ySize + postSize + combFragSize + base0Size + base1Size + base2Size;
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
    tilingData_.set_kUbSize(kUbSize);
    tilingData_.set_mUbSize(mUbSize);
    tilingData_.set_kBlockFactor(tilingData_.get_cubeBlockDimK());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcPreTilingRegbase::CalcOpTiling() {
    uint64_t kSize = hcMult_ * d_;
    tilingData_.set_k(kSize);
    // 计算bs_轴切核
    uint64_t mDimNum = std::min(aicCoreNum_, static_cast<uint64_t>(CeilDiv(bs_, M_L1_MAX_SIZE)));
    uint64_t singleCoreM = RoundUp(CeilDiv(bs_, mDimNum), AscendC::BLOCK_CUBE);
    uint64_t kDimNum = aicCoreNum_ / mDimNum;

    uint64_t splitKSize = RoundUp(CeilDiv(kSize, kDimNum), K_MULIT_CORE_SPLIT_BASE_SIZE);
    uint64_t actualKBlockNum = CeilDiv(kSize, splitKSize);

    tilingData_.set_cubeBlockDimM(mDimNum);
    tilingData_.set_cubeBlockDimK(actualKBlockNum);
    tilingData_.set_multCoreSplitMSize(singleCoreM); // todo: 这个 tiling 根本没有使用，是否需要删掉
    tilingData_.set_mL1Size(std::min(M_L1_MAX_SIZE, singleCoreM));
    tilingData_.set_multCoreSplitKSize(splitKSize);
    tilingData_.set_kL1Size(std::min(A_L1_SIZE / tilingData_.get_mL1Size(), static_cast<uint64_t>(K_L1_MAX_SIZE)) / 128 * 128);

    tilingData_.set_cvLoopKSize(1024);
    if (kDimNum != 1) {
        tilingKey_ = 1000;
        return CalcMKSplitCorePart2Tiling();
    }
    tilingKey_ = 1001;
    return CalcRegbaseOpTiling();
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
    if (tilingKey_ == 1000) {
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
