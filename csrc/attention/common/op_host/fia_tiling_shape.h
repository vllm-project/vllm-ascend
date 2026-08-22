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
 * \file fia_tiling_shape.h
 * \brief
 */
#ifndef FIA_TILING_SHAPE_H
#define FIA_TILING_SHAPE_H

#include <sstream>
#include "register/tilingdata_base.h"
#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include <tiling/platform/platform_ascendc.h>
#include "err/ops_err.h"
#include <vector>

namespace optiling {
template <typename T> using CompareFunc = bool (*)(const T&, const T&);

enum class FiaLayout : uint32_t {
    // stardard
    BSH = 0,
    BSND = 1,
    BNSD = 2,
    NZ = 3,
    TND = 4,
    NBSD = 5,
    NTD = 6,
    // for attention mask
    // Qs != 1
    S1S2 = 7,
    // Qs = 1
    BS2 = 8,
    // PA
    BnBsH = 11,
    BnNBsD = 12,
    BNS1S2 = 13,
    INS1S2 = 14,
    BNS11 = 15,
    TN1 = 16,
    BS1S2 = 18,
    B1S1S2 = 19,
    IS1S2 = 20,
    I1S1S2 = 21,
    S1S1 = 22,
};

enum class FiaAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    H = 4,
    T = 5,
    D1 = 6,
    D0 = 7,
    S1 = 8,
    S2 = 9,
    Bn = 10,
    Bs = 11,
    CONST = 12
};

enum class FiaCompareType : uint32_t {
    EQUAL = 0,
    GREATER = 1,
    GREATER_EQUAL = 2,
    LESS = 3,
    LESS_EQUAL = 4,
    NOT_EQUAL = 5,
    IGNORE_INPUT = 6
};

struct FiaTilingShapeCompareParam {
    int64_t B = 1;
    int64_t S = 1;
    int64_t N = 1;
    int64_t D = 1;
    int64_t H = 1;
    int64_t T = 1;
    int64_t Bn = 1;
    int64_t Bs = 1;
    int64_t D0 = 16;
    int64_t S1 = 1;
    int64_t S2 = 1;
    int64_t CONST = 1;
    std::map<FiaAxis, FiaCompareType> compareTypeMap = {};
};

static std::string GetShapeStr(gert::Shape shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

std::string LayoutToSerialString(FiaLayout layout);
std::string AxisToSerialString(FiaAxis axis);

class FiaTilingShape {
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

public:
    FiaTilingShape(const gert::Shape &shape, FiaLayout layout, std::string name, std::string opName,
        int64_t N = std::numeric_limits<int64_t>::min()) :
        shape_(shape), layout_(layout), name_(name), opName_(opName)
    {
        if (HasShapeH() && N != std::numeric_limits<int64_t>::min()) {
            N_ = N;
            hasSetN_ = true;
        }
    };

public:
    const gert::Shape &shape_;
    FiaLayout layout_;
    std::string name_ ;
    std::string opName_;
    bool hasSetN_ = false;
    int64_t N_ = 1;

    size_t GetDimNum() const { return shape_.GetDimNum(); }

    bool HasShapeB() const 
    { 
        return HasAxis(FiaAxis::B); 
    }
    bool HasShapeS() const 
    { 
        return HasAxis(FiaAxis::S); 
    }
    bool HasShapeH() const { 
        return HasAxis(FiaAxis::H); 
    }
    bool HasShapeN() const 
    { 
        return HasAxis(FiaAxis::N); 
    }
    bool HasShapeT() const 
    { 
        return HasAxis(FiaAxis::T); 
    }
    bool HasShapeD1() const 
    { 
        return HasAxis(FiaAxis::D1); 
    }
    bool HasShapeD0() const 
    { 
        return HasAxis(FiaAxis::D0); 
    }
    bool HasShapeD() const
    {
        if (HasAxis(FiaAxis::D)) { return true; }
        if (HasShapeH() && hasSetN_ && N_ != 0 && GetShapeH() % N_ == 0) { return true; }
        if (HasShapeD1() && HasShapeD0()) { return true; }
        return false;
    }

    int64_t GetShapeB() const { return GetAxisNum(FiaAxis::B); }
    int64_t GetShapeS() const { return GetAxisNum(FiaAxis::S); }
    int64_t GetShapeN() const { return GetAxisNum(FiaAxis::N); }
    int64_t GetShapeH() const { return GetAxisNum(FiaAxis::H); }
    int64_t GetShapeT() const { return GetAxisNum(FiaAxis::T); }
    int64_t GetShapeD1() const { return GetAxisNum(FiaAxis::D1); }
    int64_t GetShapeD0() const { return GetAxisNum(FiaAxis::D0); }
    int64_t GetShapeD() const
    {
        if (HasAxis(FiaAxis::D)) { return shape_.GetDim(GetAxisIdx(FiaAxis::D)); }
        if (HasShapeH() && hasSetN_ && N_ != 0 && GetShapeH() % N_ == 0) { return GetShapeH() / N_; }
        if (HasShapeD1() && HasShapeD0()) { return GetShapeD1() * GetShapeD0(); }
        return invalidDimValue_;
    }
    int64_t GetShapeBlockSize() const { return GetAxisNum(FiaAxis::Bs); }

    ge::graphStatus CheckHasShapeB(const std::string &funcName) const { return CheckHasAxis(FiaAxis::B, funcName); }
    ge::graphStatus CheckHasShapeS(const std::string &funcName) const { return CheckHasAxis(FiaAxis::S, funcName); }
    ge::graphStatus CheckHasShapeD(const std::string &funcName) const { return CheckHasAxis(FiaAxis::D, funcName); }
    ge::graphStatus CheckHasShapeN(const std::string &funcName) const { return CheckHasAxis(FiaAxis::N, funcName); }
    ge::graphStatus CheckHasShapeH(const std::string &funcName) const { return CheckHasAxis(FiaAxis::H, funcName); }
    ge::graphStatus CheckHasShapeT(const std::string &funcName) const { return CheckHasAxis(FiaAxis::T, funcName); }
    ge::graphStatus CheckHasShapeBlockSize(const std::string &funcName) const
    {
        return CheckHasAxis(FiaAxis::Bs, funcName);
    }

private:
    bool HasAxis(const FiaAxis &axis) const;
    size_t GetAxisIdx(const FiaAxis &axis) const;
    int64_t GetAxisNum(const FiaAxis &axis) const;
    ge::graphStatus CheckHasAxis(const FiaAxis &axis, const std::string &funcName) const;
};

class FiaTilingShapeCompare {
    static const std::map<FiaCompareType, CompareFunc<int64_t>> compareFuncMap_;

public:
    FiaTilingShapeCompare(const gert::Shape &shape, FiaLayout layout, std::string name, std::string opName) :
        shape_(shape), layout_(layout), name_(name), opName_(opName) {};

public:
    const gert::Shape &shape_;
    FiaLayout layout_;
    std::string name_ ;
    std::string opName_;

    std::string CompareTypeToSerialString(const FiaCompareType compareType) const;
    std::string CompareTypeToSerialSymbolString(const FiaCompareType &compareType) const;
    ge::graphStatus GetExpectedShapeSpecial(gert::Shape &shapeExpected,
        const FiaTilingShapeCompareParam &param, const std::string &funcName) const;
    ge::graphStatus GetExpectedShape(gert::Shape &shapeExpected,
        const FiaTilingShapeCompareParam &param, const std::string &funcName) const;
    FiaCompareType GetCompareType(const std::map<FiaAxis, FiaCompareType> &compareTypeMap,
        const FiaAxis &axis) const;
    ge::graphStatus GetCompareFunc(const FiaCompareType &compareType, 
        CompareFunc<int64_t> &compareFunc, const std::string &funcName) const;
    ge::graphStatus CompareShape(FiaTilingShapeCompareParam &param, const std::string &funcName) const;
};
} // namespace optiling
#endif // FIA_TILING_SHAPE_H