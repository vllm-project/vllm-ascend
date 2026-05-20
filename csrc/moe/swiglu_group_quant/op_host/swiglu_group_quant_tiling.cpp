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
 * \file swiglu_group_quant_tiling.cpp
 * \brief
 */

#include <sstream>
#include "swiglu_group_quant_tiling.h"

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
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t PER_BLOCK_FP16 = 128;
constexpr int64_t PER_MX_FP16 = 32;
constexpr int64_t STATIC_QUANT = 1;
constexpr int64_t MX_QUANT = 2;
constexpr int64_t FP8_QUANT = 3;
constexpr size_t ATTR_INDEX_DST_TYPE = 0;
constexpr size_t ATTR_INDEX_QUANT_MODE = 1;
constexpr size_t ATTR_INDEX_GROUP_SIZE = 2;
constexpr size_t ATTR_INDEX_ROUND_SCALE = 3;
constexpr size_t ATTR_INDEX_UE8M0_SCALE = 4;
constexpr size_t ATTR_INDEX_OUTPUT_ORIGIN = 5;
constexpr size_t ATTR_INDEX_GROUP_LIST_TYPE = 6;
constexpr size_t ATTR_INDEX_CLAMP_VALUE = 7;
constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INPUT_INDEX_TOPK_WEIGHT = 1;
constexpr size_t INPUT_INDEX_GROUP_INDEX = 2;
constexpr size_t CACHE_LINE_SIZE = 128;
constexpr int64_t GROUP_LIST_TYPE_COUNT = 0;
constexpr int64_t GROUP_LIST_TYPE_CUMSUM = 1;
constexpr int64_t GROUP_QUANT_TILING_KEY = 1;
constexpr int64_t MX_QUANT_TILING_KEY = 2;
constexpr int64_t FP8_QUANT_TILING_KEY = 31;
constexpr int64_t FP8_QUANT_YORIGIN_TILING_KEY = 32;

}

ge::graphStatus SwigluGroupQuantTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<SwigluGroupQuantCompileInfo>();
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_LOG_E(context_, "compile info is null"),
                      return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    auto quantModeAttr = attrs->GetAttrPointer<int>(ATTR_INDEX_QUANT_MODE);
    quantMode_ = quantModeAttr == nullptr ? STATIC_QUANT : *quantModeAttr;
    if (quantMode_ != STATIC_QUANT && quantMode_ != MX_QUANT && quantMode_ != FP8_QUANT) {
        return ge::GRAPH_FAILED;
    }
    splitFactor_ = quantMode_ == MX_QUANT ? PER_MX_FP16 : PER_BLOCK_FP16;

    auto roundScaleAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_ROUND_SCALE);
    if (roundScaleAttr != nullptr) {
        roundScale_ = (*roundScaleAttr) ? 1 : 0;
    }

    auto ue8m0ScaleAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_UE8M0_SCALE);
    if (ue8m0ScaleAttr != nullptr) {
        ue8m0Scale_ = (*ue8m0ScaleAttr) ? 1 : 0;
    }

    auto outputOriginAttr = attrs->GetAttrPointer<bool>(ATTR_INDEX_OUTPUT_ORIGIN);
    if (outputOriginAttr != nullptr) {
        outputOrigin_ = (*outputOriginAttr) ? 1 : 0;
    }

    auto groupListTypeAttr = attrs->GetAttrPointer<int>(ATTR_INDEX_GROUP_LIST_TYPE);
    if (groupListTypeAttr != nullptr && *groupListTypeAttr != GROUP_LIST_TYPE_COUNT) {
        OPS_LOG_E(context_, "group_list_type only support 0(count mode) now.");
        return ge::GRAPH_FAILED;
    }
    groupListType_ = groupListTypeAttr == nullptr ? GROUP_LIST_TYPE_COUNT : *groupListTypeAttr;

    auto clampValueAttr = attrs->GetAttrPointer<float>(ATTR_INDEX_CLAMP_VALUE);
    if (clampValueAttr != nullptr) {
        if (*clampValueAttr != 0.0) {
            clampValue_ = *clampValueAttr;
            hasClampValue_ = 1;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetShapeAttrsInfoInner()
{
    // (b, s, hc_mix)
    auto shapeX = context_->GetInputShape(INPUT_INDEX_X);
    OPS_LOG_E_IF_NULL(context_, shapeX, return ge::GRAPH_FAILED);

    auto xStorageShape = shapeX->GetStorageShape();
    bs_ = 1;
    for (size_t i = 0; i < xStorageShape.GetDimNum() - 1; i++) {
        bs_ = bs_ * xStorageShape.GetDim(i);
    }
    d_ = xStorageShape.GetDim(xStorageShape.GetDimNum() - 1);
    if (d_ % 2 != 0) {
        OPS_LOG_E(context_->GetNodeName(), "x last Dim[%ld] is not divisible by 2.", d_);
        return ge::GRAPH_FAILED;
    }

    auto topkWeightDesc = context_->GetOptionalInputDesc(INPUT_INDEX_TOPK_WEIGHT);
    if (topkWeightDesc != nullptr) {
        auto topkWeightShape = context_->GetOptionalInputShape(INPUT_INDEX_TOPK_WEIGHT);
        if (topkWeightShape != nullptr) {
            auto topkWeightStorageShape = topkWeightShape->GetStorageShape();
            if (topkWeightStorageShape.GetDimNum() != 0) {
                hasTopkWeight_ = true;
            }
        }
    }

    auto groupIndexDesc = context_->GetOptionalInputDesc(INPUT_INDEX_GROUP_INDEX);
    if (groupIndexDesc != nullptr) {
        auto groupIndexShape = context_->GetOptionalInputShape(INPUT_INDEX_GROUP_INDEX);
        if (groupIndexShape != nullptr) {
            auto groupIndexStorageShape = groupIndexShape->GetStorageShape();
            g_ = 1;
            for (size_t i = 0; i < groupIndexStorageShape.GetDimNum(); i++) {
                g_ = g_ * groupIndexStorageShape.GetDim(i);
            }
            hasGroupIndex_ = true;
        }
    }

    // Get Attrs
    if (GetAttr() == ge::GRAPH_FAILED) {
        OPS_LOG_E(context_->GetNodeName(), "Get attr failed.");
        return ge::GRAPH_FAILED;
    }

    OPS_ERR_IF((quantMode_ == FP8_QUANT && d_ % 256 != 0),
                  OPS_LOG_E(context_->GetNodeName(), "x last Dim must be divisible by 256 when quant_mode == %d.", FP8_QUANT),
                  return ge::GRAPH_FAILED);

    splitD_ = d_ / 2;
    scaleCol_ = CeilDiv(splitD_, splitFactor_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcGroupIndexTiling()
{
    if (hasGroupIndex_ && groupListType_ == GROUP_LIST_TYPE_COUNT) {
        gFactor_ = g_;
        int64_t groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / sizeof(int64_t)) * DOUBLE_BUFFER * sizeof(int64_t);
        int64_t groupIndexSumSize = BLOCK_SIZE;
        if (groupIndexSize + groupIndexSumSize <= ubSize_) {
            gLoop_ = 1;
            tailGFactor_ = gFactor_;
        } else {
            int64_t base = 2;
            while(1) {
                gFactor_ = CeilDiv(g_, base);
                groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / sizeof(int64_t)) * DOUBLE_BUFFER * sizeof(int64_t);
                if (groupIndexSize + groupIndexSumSize < ubSize_) {
                    break;
                }
                base++;
            }
            if (gFactor_ > CACHE_LINE_SIZE / sizeof(int64_t)) {
                gFactor_ = DownAlign(gFactor_, CACHE_LINE_SIZE / sizeof(int64_t));
            }
            gLoop_ = CeilDiv(g_, gFactor_);
            tailGFactor_ = g_ % gFactor_ == 0 ? gFactor_ : g_ % gFactor_;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcMxQuantOpTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    int64_t x0Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
    int64_t x1Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
    int64_t swigluSize = rowOnceLoop * RoundUp(splitD_, 16) * 2;
    int64_t maxExpSize = rowOnceLoop * RoundUp(scaleCol_, 16) * 2;
    int64_t invScaleSize = rowOnceLoop * RoundUp(scaleCol_, 16) * 2;
    int64_t ySize = rowOnceLoop * RoundUp(splitD_, 32) * 1 * DOUBLE_BUFFER;
    int64_t scaleSize = rowOnceLoop * RoundUp(scaleCol_, 32) * 1 * DOUBLE_BUFFER;

    int64_t totalSize = x0Size + x1Size + swigluSize + maxExpSize + invScaleSize + ySize + scaleSize;

    int64_t topkWeightSize = RoundUp(rowOnceLoop, 8) * 4 * DOUBLE_BUFFER;
    totalSize = hasTopkWeight_ ? totalSize + topkWeightSize : totalSize;

    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = splitD_;
        tailDFactor_ = dFactor_;
    } else {
        dFactor_ = splitD_;
        int64_t base = 1;
        while (totalSize < ubSize_) {
            dFactor_ = base * splitFactor_;
            scaleCol_ = CeilDiv(dFactor_, splitFactor_);
            x0Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            x1Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            swigluSize = rowOnceLoop * RoundUp(dFactor_, 16) * 2;
            maxExpSize = rowOnceLoop * RoundUp(scaleCol_, 16) * 2;
            invScaleSize = rowOnceLoop * RoundUp(scaleCol_, 16) * 2;
            ySize = rowOnceLoop * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
            scaleSize = rowOnceLoop * RoundUp(scaleCol_, 32) * 1 * DOUBLE_BUFFER;
            totalSize = x0Size + x1Size + swigluSize + maxExpSize + invScaleSize + ySize + scaleSize;
            if (hasTopkWeight_) {
                totalSize += topkWeightSize;
            }
            base++;
        }
        dFactor_ = (base - 1) * splitFactor_;
        scaleCol_ = CeilDiv(dFactor_, splitFactor_);
        dLoop_ = CeilDiv(splitD_, dFactor_);
        tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
    }

    // d全载,尝试搬入更多的bs
    if (dFactor_ == splitD_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            x0Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            x1Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            swigluSize = rowFactor_ * RoundUp(dFactor_, 16) * 2;
            maxExpSize = rowFactor_ * RoundUp(scaleCol_, 16) * 2;
            invScaleSize = rowFactor_ * RoundUp(scaleCol_, 16) * 2;
            ySize = rowFactor_ * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
            scaleSize = rowFactor_ * RoundUp(scaleCol_, 32) * 1 * DOUBLE_BUFFER;
            totalSize = x0Size + x1Size + swigluSize + maxExpSize + invScaleSize + ySize + scaleSize;
            if (hasTopkWeight_) {
                topkWeightSize = RoundUp(rowFactor_, 8) * 4 * DOUBLE_BUFFER;
                totalSize += topkWeightSize;
            }
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
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcGroupQuantOpTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    int64_t x0Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
    int64_t x1Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
    int64_t ySize = rowOnceLoop * RoundUp(splitD_, 32) * 1 * DOUBLE_BUFFER;
    int64_t scaleSize = RoundUp(rowOnceLoop * scaleCol_, 8) * 4 * DOUBLE_BUFFER;

    int64_t totalSize = x0Size + x1Size + ySize + scaleSize;

    int64_t topkWeightSize = RoundUp(rowOnceLoop, 8) * 4 * DOUBLE_BUFFER;
    totalSize = hasTopkWeight_ ? totalSize + topkWeightSize : totalSize;

    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = splitD_;
        tailDFactor_ = dFactor_;
    } else {
        dFactor_ = splitD_;
        int64_t base = 1;
        while (totalSize < ubSize_) {
            dFactor_ = base * splitFactor_;
            scaleCol_ = CeilDiv(dFactor_, splitFactor_);
            x0Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            x1Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            ySize = rowOnceLoop * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
            scaleSize = RoundUp(rowOnceLoop * scaleCol_, 8) * 4 * DOUBLE_BUFFER;
            totalSize = x0Size + x1Size + ySize + scaleSize;
            if (hasTopkWeight_) {
                totalSize += topkWeightSize;
            }
            base++;
        }
        dFactor_ = (base - 1) * splitFactor_;
        scaleCol_ = CeilDiv(dFactor_, splitFactor_);
        dLoop_ = CeilDiv(splitD_, dFactor_);
        tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
    }

    // d全载,尝试搬入更多的bs
    if (dFactor_ == splitD_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            x0Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            x1Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            ySize = rowFactor_ * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
            scaleSize = RoundUp(rowFactor_ * scaleCol_, 8) * 4 * DOUBLE_BUFFER;
            totalSize = x0Size + x1Size + ySize + scaleSize;
            if (hasTopkWeight_) {
                topkWeightSize = RoundUp(rowFactor_, 8) * 4 * DOUBLE_BUFFER;
                totalSize += topkWeightSize;
            }
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
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::CalcFp8QuantOpTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;

    int64_t minRowPerCore = 1;
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, minRowPerCore);

    int64_t x0Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
    int64_t x1Size = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
    int64_t ySize = rowOnceLoop * RoundUp(splitD_, 32) * 1 * DOUBLE_BUFFER;
    int64_t scaleSize = ue8m0Scale_ ? RoundUp(rowOnceLoop * scaleCol_, 32) * 1 * DOUBLE_BUFFER : RoundUp(rowOnceLoop * scaleCol_, 8) * 4 * DOUBLE_BUFFER;

    int64_t totalSize = x0Size + x1Size + ySize + scaleSize;

    int64_t topkWeightSize = RoundUp(rowOnceLoop, 8) * 4 * DOUBLE_BUFFER;
    totalSize = hasTopkWeight_ ? totalSize + topkWeightSize : totalSize;

    int64_t yOriginSize = rowOnceLoop * RoundUp(splitD_, 16) * 2 * DOUBLE_BUFFER;
    totalSize = outputOrigin_ ? totalSize + yOriginSize : totalSize;


    rowFactor_ = rowOnceLoop;
    if (totalSize <= ubSize_) {
        // row和d均可以在ub内全载
        dLoop_ = 1;
        dFactor_ = splitD_;
        tailDFactor_ = dFactor_;
    } else {
        dFactor_ = splitD_;
        int64_t base = 1;
        while (totalSize < ubSize_) {
            dFactor_ = base * splitFactor_;
            scaleCol_ = CeilDiv(dFactor_, splitFactor_);
            x0Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            x1Size = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            ySize = rowOnceLoop * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
            scaleSize = RoundUp(rowOnceLoop * scaleCol_, 8) * 4 * DOUBLE_BUFFER;
            totalSize = x0Size + x1Size + ySize + scaleSize;
            if (hasTopkWeight_) {
                totalSize += topkWeightSize;
            }
            if (outputOrigin_) {
                yOriginSize = rowOnceLoop * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                totalSize += yOriginSize;
            }
            base++;
        }
        dFactor_ = (base - 1) * splitFactor_;
        scaleCol_ = CeilDiv(dFactor_, splitFactor_);
        dLoop_ = CeilDiv(splitD_, dFactor_);
        tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
    }

    // d全载,尝试搬入更多的bs
    if (dFactor_ == splitD_) {
        while (rowFactor_ <= rowOfFormerBlock_) {
            x0Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            x1Size = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
            ySize = rowFactor_ * RoundUp(dFactor_, 32) * 1 * DOUBLE_BUFFER;
            scaleSize = RoundUp(rowFactor_ * scaleCol_, 8) * 4 * DOUBLE_BUFFER;
            totalSize = x0Size + x1Size + ySize + scaleSize;
            if (hasTopkWeight_) {
                topkWeightSize = RoundUp(rowFactor_, 8) * 4 * DOUBLE_BUFFER;
                totalSize += topkWeightSize;
            }
            if (outputOrigin_) {
                yOriginSize = rowFactor_ * RoundUp(dFactor_, 16) * 2 * DOUBLE_BUFFER;
                totalSize += yOriginSize;
            }
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
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void SwigluGroupQuantTiling::SetTilingData()
{
    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_splitD(splitD_);
    tilingData_.set_scaleCol(scaleCol_);
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
    tilingData_.set_roundScale(roundScale_);
    tilingData_.set_ue8m0Scale(ue8m0Scale_);
    tilingData_.set_outputOrigin(outputOrigin_);
    tilingData_.set_clampValue(clampValue_);
    tilingData_.set_g(g_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_gLoop(gLoop_);
    tilingData_.set_gFactor(gFactor_);
    tilingData_.set_tailGFactor(tailGFactor_);
    tilingData_.set_groupListType(groupListType_);
    tilingData_.set_coreNum(coreNum_);
    tilingData_.set_hasClampValue(hasClampValue_);
}

ge::graphStatus SwigluGroupQuantTiling::CalcOpTiling() {
    ge::graphStatus status;
    status = CalcGroupIndexTiling();
    if (status == ge::GRAPH_FAILED) {
        return status;
    }
    if (quantMode_ == STATIC_QUANT) {
        status = CalcGroupQuantOpTiling();
    } else if (quantMode_ == MX_QUANT) {
        status = CalcMxQuantOpTiling();
    } else {
        status = CalcFp8QuantOpTiling();
    }
    return status;
}

ge::graphStatus SwigluGroupQuantTiling::DoOpTiling()
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
    int64_t tilingKey = quantMode_;
    if (quantMode_ == STATIC_QUANT) {
        tilingKey = GROUP_QUANT_TILING_KEY;
    } else if (quantMode_ == MX_QUANT) {
        tilingKey = MX_QUANT_TILING_KEY;
    } else if (quantMode_ == FP8_QUANT) {
        if (outputOrigin_) {
            tilingKey = FP8_QUANT_YORIGIN_TILING_KEY;
        } else {
            tilingKey = FP8_QUANT_TILING_KEY;
        }
    }
    context_->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupQuantTiling::PostTiling()
{
    if (hasGroupIndex_) {
        context_->SetBlockDim(coreNum_);
    } else {
        context_->SetBlockDim(usedCoreNums_);
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForSwigluGroupQuant(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSwigluGroupQuant(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SwigluGroupQuant", "Tiling context is null"),
               return ge::GRAPH_FAILED);
    SwigluGroupQuantTiling SwigluGroupQuantTiling(context);
    return SwigluGroupQuantTiling.DoOpTiling();
}

IMPL_OP_OPTILING(SwigluGroupQuant)
    .Tiling(TilingForSwigluGroupQuant)
    .TilingParse<SwigluGroupQuantCompileInfo>(TilingPrepareForSwigluGroupQuant);

}  // namespace optiling
