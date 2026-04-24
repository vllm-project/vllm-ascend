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
 * \file rope_regbase_tiling_base.cc
 * \brief
 */

// #include "tiling_base/tiling_templates_registry.h"
#include "tiling/tiling_api.h"
// #include "tiling_base/tiling_base.h"
#include "platform/platform_info.h"
#include "inplace_partial_rotary_mul_tiling.h"
#include <graph/utils/type_utils.h>
// #include "log/log.h"

namespace {
constexpr int64_t X_INDEX = 0;
constexpr int64_t COS_INDEX = 1;
constexpr int64_t SIN_INDEX = 2;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t DIM_NUM = 4;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t HALF_INTERLEAVE_MODE_COEF = 2;
constexpr int64_t QUARTER_MODE_COEF = 4;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t D_LIMIT = 1024;
const std::vector<ge::DataType> SUPPORT_DTYPE = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16};
} // namespace

namespace optiling {
ge::graphStatus RopeRegBaseTilingClass::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
        socVersion_ = ascendcPlatform.GetSocVersion();
    } else {
        auto compileInfoPtr = reinterpret_cast<const RotaryPositionEmbeddingCompileInfo *>(context_->GetCompileInfo());
        OPS_ERR_IF(compileInfoPtr == nullptr, OPS_LOG_E(context_, "compile info is null"), return ge::GRAPH_FAILED);
        aicoreParams_.blockDim = compileInfoPtr->blockDim;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        socVersion_ = compileInfoPtr->socVersion;
    }
    blockSize_ = BLOCK_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::CheckNullptr()
{
    for (int64_t i = 0; i <= SIN_INDEX; i++) {
        auto desc = context_->GetInputDesc(i);
        OPS_ERR_IF(desc == nullptr, OPS_LOG_E(context_, "input %ld desc is nullptr.", i), return ge::GRAPH_FAILED);
        auto shape = context_->GetInputShape(i);
        OPS_ERR_IF(shape == nullptr, OPS_LOG_E(context_, "input %ld shape is nullptr.", i), return ge::GRAPH_FAILED);
    }
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    OPS_ERR_IF(yDesc == nullptr, OPS_LOG_E(context_, "output desc is nullptr."), return ge::GRAPH_FAILED);
    auto yShape = context_->GetOutputShape(Y_INDEX);
    OPS_ERR_IF(yShape == nullptr, OPS_LOG_E(context_, "output shape is nullptr."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::CheckShapeAllPositive(const int64_t idx) const
{
    auto shape = context_->GetInputShape(idx)->GetStorageShape();
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OPS_ERR_IF(
            shape.GetDim(i) <= 0,
            OPS_LOG_E(context_, "input %ld has non positive shape, dim %lu actual %ld .", idx, i, shape.GetDim(i)),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
bool RopeRegBaseTilingClass::IsRotaryPosEmbeddingMode(const int32_t mode) const
{
    switch (mode) {
        case static_cast<int32_t>(RotaryPosEmbeddingMode::HALF):
        case static_cast<int32_t>(RotaryPosEmbeddingMode::INTERLEAVE):
        case static_cast<int32_t>(RotaryPosEmbeddingMode::QUARTER):
        case static_cast<int32_t>(RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE):
            return true;
        default:
            return false;
    }
}

ge::graphStatus RopeRegBaseTilingClass::CheckShapeAllPositive() const
{
    OPS_ERR_IF(CheckShapeAllPositive(X_INDEX) != ge::GRAPH_SUCCESS, OPS_LOG_E(context_, "x has non positive shape."),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(CheckShapeAllPositive(COS_INDEX) != ge::GRAPH_SUCCESS, OPS_LOG_E(context_, "cos has non positive shape."),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(CheckShapeAllPositive(SIN_INDEX) != ge::GRAPH_SUCCESS, OPS_LOG_E(context_, "sin has non positive shape."),
                return ge::GRAPH_FAILED);
    auto yShape = context_->GetOutputShape(Y_INDEX)->GetStorageShape();
    for (size_t i = 0; i < yShape.GetDimNum(); i++) {
        OPS_ERR_IF(yShape.GetDim(i) <= 0,
                    OPS_LOG_E(context_, "output has non positive shape, dim %lu actual %ld .", i, yShape.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::JudgeLayoutByShape(const gert::Shape &xShape, const gert::Shape &cosShape)
{
    uint64_t xShape0 = xShape.GetDim(DIM_0);
    uint64_t xShape1 = xShape.GetDim(DIM_1);
    uint64_t xShape2 = xShape.GetDim(DIM_2);
    uint64_t cosShape0 = cosShape.GetDim(DIM_0);
    uint64_t cosShape1 = cosShape.GetDim(DIM_1);
    uint64_t cosShape2 = cosShape.GetDim(DIM_2);
    if (xShape0 == cosShape0 && xShape1 == cosShape1 && xShape2 == cosShape2) { // BSND
        layout_ = RopeLayout::NO_BROADCAST;
    } else if (cosShape0 == 1 && cosShape1 == 1 && cosShape2 == 1) { // (111D)
        layout_ = RopeLayout::BROADCAST_BSN;
    } else if (cosShape2 == 1 && cosShape0 == 1 && xShape1 == cosShape1) { // BSND (1S1D)
        layout_ = RopeLayout::BSND;
    } else if (cosShape2 == 1 && xShape0 == cosShape0 && (cosShape1 == 1 || cosShape1 == xShape1)) { // SBND (S11D,
                                                                                                     // SB1D), BSND
                                                                                                     // (BS1D)
        layout_ = RopeLayout::SBND;
    } else if (cosShape1 == 1 && xShape2 == cosShape2 && (cosShape0 == 1 || cosShape0 == xShape0)) { // BNSD (11SD,
                                                                                                     // B1SD)
        layout_ = RopeLayout::BNSD;
    } else if (cosShape0 == 1 && xShape1 == cosShape1 && xShape2 == cosShape2) { // 1SND
        layout_ = RopeLayout::BNSD;
        is1snd_ = true;
    } else {
        OPS_LOG_E(context_->GetNodeName(), "the shape of x and sin not satisfy the broadcast.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::CheckShape()
{
    auto &xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    auto &yShape = context_->GetOutputShape(Y_INDEX)->GetStorageShape();
    OPS_ERR_IF(xShape.GetDimNum() != DIM_NUM, OPS_LOG_E(context_, "dim of x expect 4, actual %lu.", xShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(cosShape.GetDimNum() != DIM_NUM,
                OPS_LOG_E(context_, "dim of cos expect 4, actual %lu.", cosShape.GetDimNum()), return ge::GRAPH_FAILED);
    OPS_ERR_IF(sinShape.GetDimNum() != DIM_NUM,
                OPS_LOG_E(context_, "dim of sin expect 4, actual %lu.", sinShape.GetDimNum()), return ge::GRAPH_FAILED);
    OPS_ERR_IF(yShape.GetDimNum() != DIM_NUM,
                OPS_LOG_E(context_, "dim of output expect 4, actual %lu.", yShape.GetDimNum()), return ge::GRAPH_FAILED);
    OPS_ERR_IF(cosShape != sinShape,
                OPS_LOG_E(context_,
                        "shape of cos and sin should be same, actual cos shape is (%ld, %ld, %ld, %ld), sin shape is "
                        "(%ld, %ld, %ld, %ld). ",
                        cosShape.GetDim(DIM_0), cosShape.GetDim(DIM_1), cosShape.GetDim(DIM_2), cosShape.GetDim(DIM_3),
                        sinShape.GetDim(DIM_0), sinShape.GetDim(DIM_1), sinShape.GetDim(DIM_2), sinShape.GetDim(DIM_3)),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(xShape != yShape,
                OPS_LOG_E(context_,
                        "shape of x and output should be same, actual x shape is (%ld, "
                        "%ld, %ld, %ld), output shape is (%ld, %ld, %ld, %ld). ",
                        xShape.GetDim(DIM_0), xShape.GetDim(DIM_1), xShape.GetDim(DIM_2), xShape.GetDim(DIM_3),
                        yShape.GetDim(DIM_0), yShape.GetDim(DIM_1), yShape.GetDim(DIM_2), yShape.GetDim(DIM_3)),
                return ge::GRAPH_FAILED);
    // OPS_ERR_IF(
    //     (cosShape.GetDim(DIM_3) != xShape.GetDim(DIM_3)),
    //     OPS_LOG_E(context_,
    //             "D of x, cos, sin and output should be same, actual x is %ld, cos is %ld, sin is %ld, output is %ld. ",
    //             xShape.GetDim(DIM_3), cosShape.GetDim(DIM_3), sinShape.GetDim(DIM_3), yShape.GetDim(DIM_3)),
    //     return ge::GRAPH_FAILED);
    OPS_ERR_IF(CheckRotaryModeShapeRelation(xShape.GetDim(DIM_3)) != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_, "D is invalid for rotary mode."), return ge::GRAPH_FAILED);
    return CheckShapeAllPositive();
}

ge::graphStatus RopeRegBaseTilingClass::CheckDtypeAndAttr()
{
    dtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    OPS_ERR_IF(std::find(SUPPORT_DTYPE.begin(), SUPPORT_DTYPE.end(), dtype_) == SUPPORT_DTYPE.end(),
                OPS_LOG_E(context_->GetNodeName(), "Only support F32, BF16, F16 datetype, actual %s.",
                        ge::TypeUtils::DataTypeToSerialString(dtype_).c_str()),
                return ge::GRAPH_FAILED);
    for (int64_t i = X_INDEX; i <= SIN_INDEX; i++) {
        auto type = context_->GetInputDesc(i)->GetDataType();
        OPS_ERR_IF(type != dtype_,
                    OPS_LOG_E(context_, "input %ld datatype expect %s, actual %s.", i,
                            ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(type).c_str()),
                    return ge::GRAPH_FAILED);
    }
    auto outputType = context_->GetOutputDesc(Y_INDEX)->GetDataType();
    OPS_ERR_IF(outputType != dtype_,
                OPS_LOG_E(context_, "output datatype expect %s, actual %s.",
                        ge::TypeUtils::DataTypeToSerialString(dtype_).c_str(),
                        ge::TypeUtils::DataTypeToSerialString(outputType).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::CheckParam()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_, "platform info is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    if (!IsRegbaseSocVersion()) {
        return ge::GRAPH_SUCCESS;
    }
    OPS_ERR_IF(CheckNullptr() != ge::GRAPH_SUCCESS, OPS_LOG_E(context_, "check nullptr fail."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(CheckDtypeAndAttr() != ge::GRAPH_SUCCESS, OPS_LOG_E(context_, "check dtype and attr fail."),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(CheckShape() != ge::GRAPH_SUCCESS, OPS_LOG_E(context_, "check shape fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::CheckRotaryModeShapeRelation(const int64_t d)
{
    OPS_ERR_IF(d > D_LIMIT, OPS_LOG_E(context_, "D must be small than %ld, actual %ld.", D_LIMIT, d),
                return ge::GRAPH_FAILED);
    if (rotaryMode_ == RotaryPosEmbeddingMode::HALF || rotaryMode_ == RotaryPosEmbeddingMode::INTERLEAVE ||
        rotaryMode_ == RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE) {
        OPS_ERR_IF(
            d % HALF_INTERLEAVE_MODE_COEF != 0,
            OPS_LOG_E(context_, "D must be multiples of 2 in half, interleave and interleave-half mode, actual %ld.", d),
            return ge::GRAPH_FAILED);
    } else if (rotaryMode_ == RotaryPosEmbeddingMode::QUARTER) {
        OPS_ERR_IF(d % QUARTER_MODE_COEF != 0,
                    OPS_LOG_E(context_, "D must be multiples of 4 in quarter mode, actual %ld.", d),
                    return ge::GRAPH_FAILED);
    }
    if (rotaryMode_ == RotaryPosEmbeddingMode::HALF || rotaryMode_ == RotaryPosEmbeddingMode::DEEPSEEK_INTERLEAVE) {
        dSplitCoef_ = HALF_INTERLEAVE_MODE_COEF;
    } else if (rotaryMode_ == RotaryPosEmbeddingMode::QUARTER) {
        dSplitCoef_ = QUARTER_MODE_COEF;
    } else {
        dSplitCoef_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::JudgeSliceInfo() {
    OPS_LOG_E(context_, "TEST LOG, sliceStart_ =  %ld. sliceEnd_ =  %ld", sliceStart_, sliceEnd_); // TODO
    if (sliceStart_ < 0 || sliceEnd_ < 0 || sliceLength_ <= 0 || sliceEnd_ > d_) {
        OPS_LOG_E(context_, "slice info fail, sliceStart_ =  %ld. sliceEnd_ =  %ld", sliceStart_, sliceEnd_);
        return ge::GRAPH_FAILED;
    }
    if (cosd_ != sind_ || cosd_ != sliceLength_) {
        OPS_LOG_E(context_, "slice info fail, sliceLength_ =  %ld. cosd_ =  %ld", sliceLength_, cosd_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClass::GetShapeAttrsInfo()
{
    const gert::RuntimeAttrs *attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);
    const int32_t *mode = attrs->GetAttrPointer<int32_t>(0);
    int32_t modeValue = (mode == nullptr) ? 0 : static_cast<int32_t>(*mode);
    OPS_ERR_IF(IsRotaryPosEmbeddingMode(modeValue) != true,
                OPS_LOG_E(context_->GetNodeName(), "mode only support 0, 1, 2 3, actual %d.", modeValue),
                return ge::GRAPH_FAILED);
    rotaryMode_ = static_cast<RotaryPosEmbeddingMode>(modeValue);

    OPS_ERR_IF(CheckParam() != ge::GRAPH_SUCCESS, OPS_LOG_E(context_, "check param fail."), return ge::GRAPH_FAILED);

    dtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto &xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    auto &cosShape = context_->GetInputShape(COS_INDEX)->GetStorageShape();
    auto &sinShape = context_->GetInputShape(SIN_INDEX)->GetStorageShape();
    OPS_ERR_IF(JudgeLayoutByShape(xShape, cosShape) != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_, "JudgeLayoutByShape fail."), return ge::GRAPH_FAILED);

    d_ = xShape.GetDim(DIM_3);
    cosd_ = cosShape.GetDim(DIM_3);
    sind_ = sinShape.GetDim(DIM_3);
    if (layout_ == RopeLayout::BSND) {
        b_ = xShape.GetDim(DIM_0);
        cosb_ = cosShape.GetDim(DIM_0);
        s_ = xShape.GetDim(DIM_1);
        n_ = xShape.GetDim(DIM_2);
    } else if (layout_ == RopeLayout::BNSD || layout_ == RopeLayout::NO_BROADCAST ||
               layout_ == RopeLayout::BROADCAST_BSN) {
        b_ = xShape.GetDim(DIM_0);
        cosb_ = cosShape.GetDim(DIM_0);
        n_ = xShape.GetDim(DIM_1);
        s_ = xShape.GetDim(DIM_2);
        // 1XXX情况下，reshape成11XX
        if (is1snd_ == true) {
            s_ = s_ * n_;
            n_ = 1;
        }
    } else if (layout_ == RopeLayout::SBND) {
        s_ = xShape.GetDim(DIM_0);
        b_ = xShape.GetDim(DIM_1);
        cosb_ = cosShape.GetDim(DIM_1);
        n_ = xShape.GetDim(DIM_2);
    }

    // 获取slice
    const gert::ContinuousVector *sliceRangeListPtr = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    if (sliceRangeListPtr->GetSize() == 0) {
        sliceStart_ = 0;
        sliceEnd_ = d_;
    }
    else {
        const int64_t *expertRangeList = reinterpret_cast<const int64_t *>(sliceRangeListPtr->GetData());
        sliceStart_ = expertRangeList[0];
        sliceEnd_ = expertRangeList[1];
    }
    sliceLength_ = sliceEnd_ - sliceStart_;
    OPS_ERR_IF(JudgeSliceInfo() != ge::GRAPH_SUCCESS,
               OPS_LOG_E(context_, "JudgeSliceInfo fail."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling