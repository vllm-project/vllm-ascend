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
 * \file matrix_coord.hpp
 * \brief
 */

#ifndef MATRIX_COORD_HPP
#define MATRIX_COORD_HPP

#include "../attn_infra/coord.hpp"

namespace NpuArch {

template <
    uint32_t ROW_ = 1,
    uint32_t COLUMN_ = 1
>
struct MatrixShape {
    static constexpr uint32_t ROW = ROW_;
    static constexpr uint32_t COLUMN = COLUMN_;

    static constexpr int64_t COUNT = ROW * COLUMN;

    HOST_DEVICE
    static Coord<2> ToCoord()
    {
        return MakeCoord(ROW, COLUMN);
    }
};

/// MatrixCoord wraps Coord<2, uint32_t> to provide a helper for accessing named dimensions. Classes
/// expecting a coordinate in the rank=2 index space of a matrix should use MatrixCoord.
struct MatrixCoord : public Coord<2, uint32_t> {
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=2
    using Base = Coord<2, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// Rows dimension
    static constexpr uint32_t ROW_INDEX = 0;

    /// Columns dimension
    static constexpr uint32_t COLUMN_INDEX = 1;

    /// Default ctor
    HOST_DEVICE
    MatrixCoord() {}

    /// Constructs from Coord<2>
    HOST_DEVICE
    MatrixCoord(Coord<2, Index> const &coord) : Base(coord) {}

    /// Helper to construct from a row and column
    HOST_DEVICE
    MatrixCoord(Index row, Index column) : Base(MakeCoord(row, column)) {}

    /// Helper to construct from a row and column, which are LongIndex based
    HOST_DEVICE
    MatrixCoord(LongIndex row, LongIndex column) : Base(MakeCoord(Index(row), Index(column))) {}

    /// Returns the row of the coordinate
    HOST_DEVICE
    Index const &row() const { return this->At(ROW_INDEX); }

    /// Returns the row of the coordinate
    HOST_DEVICE
    Index &row() { return this->At(ROW_INDEX); }

    /// Returns the column of the coordinate
    HOST_DEVICE
    Index const &column() const { return this->At(COLUMN_INDEX); }

    /// Returns the column of the coordinate
    HOST_DEVICE
    Index &column() { return this->At(COLUMN_INDEX); }

    /// Element-wise addition
    HOST_DEVICE
    MatrixCoord operator+(Base const &b) const
    {
        return MatrixCoord(Base::operator+(b));
    }

    /// In-place addition
    HOST_DEVICE
    MatrixCoord &operator+=(Base const &b)
    {
        Base::operator+=(b);
        return *this;
    }
};

} // namespace NpuArch

#endif