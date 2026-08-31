/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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

#include "fia_tiling_info.h"

namespace optiling {
template <typename T>
using CompareFunc = bool (*)(const T&, const T&);

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

class FiaTilingShape {
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

public:
    FiaTilingShape(const gert::Shape &shape, FiaLayout layout, std::string name, std::string opName,
        int64_t N = std::numeric_limits<int64_t>::min()) :
        shape_(shape), layout_(layout), name_(name), opName_(opName)
    {
        if (HasH() && N != std::numeric_limits<int64_t>::min()) {
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

    bool HasB() const
    {
        return HasAxis(FiaAxis::B);
    }
    bool HasS() const
    {
        return HasAxis(FiaAxis::S);
    }
    bool HasH() const
    {
        return HasAxis(FiaAxis::H);
    }
    bool HasN() const
    {
        return HasAxis(FiaAxis::N);
    }
    bool HasT() const
    {
        return HasAxis(FiaAxis::T);
    }
    bool HasD1() const
    {
        return HasAxis(FiaAxis::D1);
    }
    bool HasD0() const
    {
        return HasAxis(FiaAxis::D0);
    }
    bool HasD() const
    {
        if (HasAxis(FiaAxis::D)) {
            return true;
        }
        if (HasH() && hasSetN_ && N_ != 0 && GetH() % N_ == 0) {
            return true;
        }
        if (HasD1() && HasD0()) {
            return true;
        }
        return false;
    }

    int64_t GetB() const { return GetAxisNum(FiaAxis::B); }
    int64_t GetS() const { return GetAxisNum(FiaAxis::S); }
    int64_t GetN() const { return GetAxisNum(FiaAxis::N); }
    int64_t GetH() const { return GetAxisNum(FiaAxis::H); }
    int64_t GetT() const { return GetAxisNum(FiaAxis::T); }
    int64_t GetD1() const { return GetAxisNum(FiaAxis::D1); }
    int64_t GetD0() const { return GetAxisNum(FiaAxis::D0); }
    int64_t GetD() const
    {
        if (HasAxis(FiaAxis::D)) { return shape_.GetDim(GetAxisIdx(FiaAxis::D)); }
        if (HasH() && hasSetN_ && N_ != 0 && GetH() % N_ == 0) { return GetH() / N_; }
        if (HasD1() && HasD0()) { return GetD1() * GetD0(); }
        return invalidDimValue_;
    }

    ge::graphStatus CheckHasB(const std::string &funcName) const { return CheckHasAxis(FiaAxis::B, funcName); }
    ge::graphStatus CheckHasS(const std::string &funcName) const { return CheckHasAxis(FiaAxis::S, funcName); }
    ge::graphStatus CheckHasD(const std::string &funcName) const { return CheckHasAxis(FiaAxis::D, funcName); }
    ge::graphStatus CheckHasN(const std::string &funcName) const { return CheckHasAxis(FiaAxis::N, funcName); }
    ge::graphStatus CheckHasH(const std::string &funcName) const { return CheckHasAxis(FiaAxis::H, funcName); }
    ge::graphStatus CheckHasT(const std::string &funcName) const { return CheckHasAxis(FiaAxis::T, funcName); }

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
