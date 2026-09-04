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
 * \file fia_tiling_shape.cpp
 * \brief
 */

#include <vector>
#include <algorithm>
#include "fia_tiling_shape.h"


namespace optiling {
static const std::map<FiaLayout, std::vector<FiaAxis>> FIA_LAYOUT_AXIS_MAP = {
    {FiaLayout::BSH, {FiaAxis::B, FiaAxis::S, FiaAxis::H}},
    {FiaLayout::BSND, {FiaAxis::B, FiaAxis::S, FiaAxis::N, FiaAxis::D}},
    {FiaLayout::BNSD, {FiaAxis::B, FiaAxis::N, FiaAxis::S, FiaAxis::D}},
    {FiaLayout::NZ, {FiaAxis::Bn, FiaAxis::N, FiaAxis::D1, FiaAxis::Bs, FiaAxis::D0}},
    {FiaLayout::TND, {FiaAxis::T, FiaAxis::N, FiaAxis::D}},
    {FiaLayout::NBSD, {FiaAxis::N, FiaAxis::B, FiaAxis::S, FiaAxis::D}},
    {FiaLayout::NTD, {FiaAxis::N, FiaAxis::T, FiaAxis::D}},
    {FiaLayout::BS2, {FiaAxis::B, FiaAxis::S2}},
    {FiaLayout::S1S2, {FiaAxis::S1, FiaAxis::S2}},
    {FiaLayout::BnBsH, {FiaAxis::Bn, FiaAxis::Bs, FiaAxis::H}},
    {FiaLayout::BnNBsD, {FiaAxis::Bn, FiaAxis::N, FiaAxis::Bs, FiaAxis::D}},
    {FiaLayout::BNS1S2, {FiaAxis::B, FiaAxis::N, FiaAxis::S1, FiaAxis::S2}},
    {FiaLayout::INS1S2, {FiaAxis::CONST, FiaAxis::N, FiaAxis::S1, FiaAxis::S2}},
    {FiaLayout::BNS11, {FiaAxis::B, FiaAxis::N, FiaAxis::S1, FiaAxis::CONST}},
    {FiaLayout::TN1, {FiaAxis::T, FiaAxis::N, FiaAxis::CONST}},
    {FiaLayout::BS1S2, {FiaAxis::B, FiaAxis::S1, FiaAxis::S2}},
    {FiaLayout::B1S1S2, {FiaAxis::B, FiaAxis::CONST, FiaAxis::S1, FiaAxis::S2}},
    {FiaLayout::IS1S2, {FiaAxis::CONST, FiaAxis::S1, FiaAxis::S2}},
    {FiaLayout::I1S1S2, {FiaAxis::CONST, FiaAxis::CONST, FiaAxis::S1, FiaAxis::S2}},
    {FiaLayout::S1S1, {FiaAxis::S1}},
};

static bool equal_to(const int64_t& a, const int64_t& b)
{
    return (a == b);
}

static bool greater(const int64_t& a, const int64_t& b)
{
    return (a > b);
}

static bool greater_equal(const int64_t& a, const int64_t& b)
{
    return (a >= b);
}

static bool less(const int64_t& a, const int64_t& b)
{
    return (a < b);
}

static bool less_equal(const int64_t& a, const int64_t& b)
{
    return (a <= b);
}

static bool not_equal_to(const int64_t& a, const int64_t& b)
{
    return (a != b);
}

static bool ignore_input(const int64_t& a, const int64_t& b)
{
    (void)a;
    (void)b;
    return true;
}

static ge::graphStatus GetLayoutAxes(std::vector<FiaAxis> &layoutAxes, const FiaLayout &layout,
    const std::string &opName, const std::string &funcName)
{
    auto it = FIA_LAYOUT_AXIS_MAP.find(layout);
    if (it == FIA_LAYOUT_AXIS_MAP.end()) {
        OP_LOGE(opName, "[%s] compare layout %s is unsupported.",
            funcName.c_str(), LayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    layoutAxes = it->second;
    return ge::GRAPH_SUCCESS;
}

const std::map<FiaCompareType, CompareFunc<int64_t>> FiaTilingShapeCompare::compareFuncMap_ = {
    {FiaCompareType::EQUAL, equal_to},
    {FiaCompareType::GREATER, greater},
    {FiaCompareType::GREATER_EQUAL, greater_equal},
    {FiaCompareType::LESS, less},
    {FiaCompareType::LESS_EQUAL, less_equal},
    {FiaCompareType::NOT_EQUAL, not_equal_to},
    {FiaCompareType::IGNORE_INPUT, ignore_input}
    
};

std::string LayoutToSerialString(FiaLayout layout)
{
    const std::map<FiaLayout, std::string> layout2Str = {
        { FiaLayout::BSH, "BSH" },
        { FiaLayout::BSND, "BSND" },
        { FiaLayout::BNSD, "BNSD" },
        { FiaLayout::NZ, "NZ" },
        { FiaLayout::TND, "TND" },
        { FiaLayout::NBSD, "NBSD" },
        { FiaLayout::NTD, "NTD" },
        { FiaLayout::S1S2, "S1S2" },
        { FiaLayout::BS2, "BS2" },
        { FiaLayout::BnBsH, "BnBsH" },
        { FiaLayout::BnNBsD, "BnNBsD" },
        { FiaLayout::BNS1S2, "BNS1S2" },
        { FiaLayout::INS1S2, "1NS1S2" },
        { FiaLayout::BNS11, "BNS11" },
        { FiaLayout::TN1, "TN1" },
        { FiaLayout::BS1S2, "BS1S2" },
        { FiaLayout::B1S1S2, "B1S1S2" },
        { FiaLayout::IS1S2, "1S1S2" },
        { FiaLayout::I1S1S2, "11S1S2" }
    };

    if (layout2Str.find(layout) != layout2Str.end()) {
        return layout2Str.at(layout);
    }
    return "UNKNOWN";
}

static const std::string AXIS_SERIAL_STRINGS[] = {
    "B", "S", "N", "D", "H", "T", "D1", "D0", "S1", "S2", "Bn", "Bs", "CONST"
};

std::string AxisToSerialString(FiaAxis axis)
{
    uint32_t idx = static_cast<uint32_t>(axis);
    return (idx < sizeof(AXIS_SERIAL_STRINGS) / sizeof(AXIS_SERIAL_STRINGS[0]))
        ? AXIS_SERIAL_STRINGS[idx] : "UNKNOWN";
}

bool FiaTilingShape::HasAxis(const FiaAxis &axis) const
{   
    const auto& layoutIt = FIA_LAYOUT_AXIS_MAP.find(layout_);
    if (layoutIt == FIA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<FiaAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }

    return true;
}

size_t FiaTilingShape::GetAxisIdx(const FiaAxis &axis) const
{
    if (HasAxis(axis)) {
        const std::vector<FiaAxis>& axes = FIA_LAYOUT_AXIS_MAP.find(layout_)->second;
        const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
        return std::distance(axes.begin(), axisIt);
    }
    return 0;
}

int64_t FiaTilingShape::GetAxisNum(const FiaAxis &axis) const
{
    return HasAxis(axis) ? shape_.GetDim(GetAxisIdx(axis)) : invalidDimValue_;
}

ge::graphStatus FiaTilingShape::CheckHasAxis(const FiaAxis &axis, const std::string &funcName) const
{
    if (shape_.GetDimNum() == 0) {
        std::string reason = "[" + funcName + "] The shape dim of " + name_ + " cannot be 0";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, name_.c_str(),
            std::to_string(0).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    std::vector<FiaAxis> layoutAxes;
    if (GetLayoutAxes(layoutAxes, layout_, opName_, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shape_.GetDimNum() != layoutAxes.size()) {
        std::string dimStr = std::to_string(shape_.GetDimNum());
        std::string reason = "[" + funcName + "] The shape dim of " + name_ +
            " must be equal to the axes size of layout(" + LayoutToSerialString(layout_) + "): " +
            std::to_string(layoutAxes.size());
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, name_.c_str(), dimStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    if ((axis == FiaAxis::D)) {
        if (HasShapeD()) {
            return ge::GRAPH_SUCCESS;
        } else if (!HasShapeH()) {
            OP_LOGE(opName_, "[%s] %s's layout is %s, do not have D and H.",
                funcName.c_str(), name_.c_str(), LayoutToSerialString(layout_).c_str());
            return ge::GRAPH_FAILED;
        } else if (!hasSetN_) {
            OP_LOGE(opName_, "[%s] %s's N is not specified, cannot caculate D by H.", funcName.c_str(), name_.c_str());
            return ge::GRAPH_FAILED;
        } else if (N_ == 0) {
            OP_LOGE(opName_, "[%s] %s's N is 0.", funcName.c_str(), name_.c_str());
            return ge::GRAPH_FAILED;
        } else if (GetShapeH() % N_ != 0) {
            OP_LOGE(opName_, "[%s] %s's H(%ld) should be an integer multiple of N(%ld).",
            funcName.c_str(), name_.c_str(), GetShapeH(), N_);
            return ge::GRAPH_FAILED;
        }
    } else if (HasAxis(axis)) {
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(opName_, "[%s] %s's layout is %s, %s is not exists.",
        funcName.c_str(), name_.c_str(), LayoutToSerialString(layout_).c_str(),
        AxisToSerialString(axis).c_str());
    return ge::GRAPH_FAILED;
}

std::string FiaTilingShapeCompare::CompareTypeToSerialString(const FiaCompareType compareType) const
{
    switch (compareType) {
        case FiaCompareType::EQUAL: 
            return "EQUAL";
        case FiaCompareType::GREATER: 
            return "GREATER";
        case FiaCompareType::GREATER_EQUAL: 
            return "GREATER_EQUAL";
        case FiaCompareType::LESS: 
            return "LESS";
        case FiaCompareType::LESS_EQUAL: 
            return "LESS_EQUAL";
        case FiaCompareType::NOT_EQUAL: 
            return "NOT_EQUAL";
        default: 
            return "UNKNOWN";
    }
}

std::string FiaTilingShapeCompare::CompareTypeToSerialSymbolString(const FiaCompareType &compareType) const
{
    switch (compareType) {
        case FiaCompareType::EQUAL: 
            return "==";
        case FiaCompareType::GREATER: 
            return ">";
        case FiaCompareType::GREATER_EQUAL: 
            return ">=";
        case FiaCompareType::LESS: 
            return "<";
        case FiaCompareType::LESS_EQUAL: 
            return "<=";
        case FiaCompareType::NOT_EQUAL: 
            return "!=";
        default: 
            return "UNKNOWN";
    }
}

ge::graphStatus FiaTilingShapeCompare::GetExpectedShapeSpecial(gert::Shape &shapeExpected,
    const FiaTilingShapeCompareParam &param, const std::string &funcName) const
{
    if (layout_ == FiaLayout::BNS1S2) {
        shapeExpected = gert::Shape({param.B, param.N, param.S1, param.S2});
    } else if (layout_ == FiaLayout::INS1S2) {
        shapeExpected = gert::Shape({param.CONST, param.N, param.S1, param.S2});
    } else if (layout_ == FiaLayout::BNS11) {
        shapeExpected = gert::Shape({param.B, param.N, param.S1, param.CONST});
    } else if (layout_ == FiaLayout::TN1) {
        shapeExpected = gert::Shape({param.T, param.N, param.CONST});
    } else if (layout_ == FiaLayout::BS2) {
        shapeExpected = gert::Shape({param.B, param.S2});
    } else if (layout_ == FiaLayout::S1S2) {
        shapeExpected = gert::Shape({param.S1, param.S2});
    } else if (layout_ == FiaLayout::BS1S2) {
        shapeExpected = gert::Shape({param.B, param.S1, param.S2});
    } else if (layout_ == FiaLayout::B1S1S2) {
        shapeExpected = gert::Shape({param.B, param.CONST, param.S1, param.S2});
    } else if (layout_ == FiaLayout::IS1S2) {
        shapeExpected = gert::Shape({param.CONST, param.S1, param.S2});
    } else if (layout_ == FiaLayout::I1S1S2) {
        shapeExpected = gert::Shape({param.CONST, param.CONST, param.S1, param.S2});
    } else if (layout_ == FiaLayout::S1S1) {
        shapeExpected = gert::Shape({param.S1});
    } else {
        OP_LOGE(opName_, "[%s] layout %s is unsupported", funcName.c_str(), LayoutToSerialString(layout_).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingShapeCompare::GetExpectedShape(gert::Shape &shapeExpected,
    const FiaTilingShapeCompareParam &param, const std::string &funcName) const
{
    if (layout_ == FiaLayout::BSH) {
        shapeExpected = gert::Shape({param.B, param.S, param.H});
    } else if (layout_ == FiaLayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout_ == FiaLayout::BNSD) {
        shapeExpected = gert::Shape({param.B, param.N, param.S, param.D});
    } else if (layout_ == FiaLayout::BnBsH) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.H});
    } else if (layout_ == FiaLayout::BnNBsD) {
        shapeExpected = gert::Shape({param.Bn, param.N, param.Bs, param.D});
    } else if (layout_ == FiaLayout::NZ) {
        shapeExpected = gert::Shape({param.Bn, param.N, param.D / param.D0, param.Bs, param.D0});
    } else if (layout_ == FiaLayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout_ == FiaLayout::NBSD) {
        shapeExpected = gert::Shape({param.N, param.B, param.S, param.D});
    } else if (layout_ == FiaLayout::NTD) {
        shapeExpected = gert::Shape({param.N, param.T, param.D});
    } else {
        return GetExpectedShapeSpecial(shapeExpected, param, funcName);
    }
    return ge::GRAPH_SUCCESS;
}

FiaCompareType FiaTilingShapeCompare::GetCompareType(const std::map<FiaAxis, FiaCompareType> &compareTypeMap,
    const FiaAxis &axis) const
{
    auto it = compareTypeMap.find(axis);
    auto compareType = FiaCompareType::EQUAL;
    if (it != compareTypeMap.end()) {
        compareType = it->second;
    }
    return compareType;
}

ge::graphStatus FiaTilingShapeCompare::GetCompareFunc(const FiaCompareType &compareType, 
    CompareFunc<int64_t> &compareFunc, const std::string &funcName) const
{
    auto it = compareFuncMap_.find(compareType);
    if (it == compareFuncMap_.end()) {
        OP_LOGE(opName_, "[%s] compare type %s is unsupported.", funcName.c_str(),
            CompareTypeToSerialString(compareType).c_str());
        return ge::GRAPH_FAILED;
    }
    compareFunc = it->second;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingShapeCompare::CompareShape(FiaTilingShapeCompareParam &param, const std::string &funcName) const
{
    param.H = param.N * param.D;
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    std::vector<FiaAxis> layoutAxes;
    if (GetLayoutAxes(layoutAxes, layout_, opName_, funcName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if ((shape_.GetDimNum() != shapeExpected.GetDimNum()) || (shape_.GetDimNum() != layoutAxes.size())) {
        OP_LOGE(opName_,
            "[%s] %s shape dimension is %zu, expected shape dimension is %zu, layout(%s) axes size is %zu, they should be equal.",
            funcName.c_str(), name_.c_str(), shape_.GetDimNum(), shapeExpected.GetDimNum(),
            LayoutToSerialString(layout_).c_str(), layoutAxes.size());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape_.GetDimNum(); i++) {
        auto axis = layoutAxes[i];
        auto compareType = GetCompareType(param.compareTypeMap, axis);
        CompareFunc<int64_t> compareFunc;
        if (GetCompareFunc(compareType, compareFunc, funcName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        if (!compareFunc(shape_.GetDim(i), shapeExpected.GetDim(i))) {
            if (param.compareTypeMap.empty()) {
                OP_LOGE(opName_, "[%s] %s layout is %s, shape %s should be equal to %s.",
                    funcName.c_str(), name_.c_str(), LayoutToSerialString(layout_).c_str(),
                    GetShapeStr(shape_).c_str(), GetShapeStr(shapeExpected).c_str());
            } else {
                OP_LOGE(opName_, "[%s] %s layout is %s, shape is %s, expected shape is %s, axis %s(%ld) should be %s expected %ld.",
                    funcName.c_str(), name_.c_str(), LayoutToSerialString(layout_).c_str(),
                    GetShapeStr(shape_).c_str(), GetShapeStr(shapeExpected).c_str(), AxisToSerialString(axis).c_str(),
                    shape_.GetDim(i), CompareTypeToSerialSymbolString(compareType).c_str(), shapeExpected.GetDim(i));
            }
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling