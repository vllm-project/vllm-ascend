/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_MATRIX_ROW_MAJOR_HPP
#define FUSED_MATRIX_ROW_MAJOR_HPP

#include "../../../attn_infra/fused_base_defs.hpp"
#include "../../../attn_infra/fused_coord.hpp"
#include "../../../attn_infra/detail/fused_alignment.hpp"
#include "../../../attn_infra/fused_matrix_coord.hpp"

namespace NpuArch::layout {

/// Mapping function for row-major matrices
struct RowMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Constructor
    HOST_DEVICE
    RowMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(cols), LongIndex(1))) {}

    /// Constructor
    HOST_DEVICE
    RowMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(ldm, LongIndex(1))) {}

    /// Ctor
    HOST_DEVICE
    RowMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element>
    HOST_DEVICE
    static RowMajor MakeLayoutInUb(MatrixCoord const &shape)
    {
        return RowMajor(shape.row(), shape.column(),
            NpuArch::Detail::Alignment::RoundUp<BYTE_PER_C0 / sizeof(Element)>(shape.column()));
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) * stride_[0] + LongIndex(coord.column());
    }

    /// Returns the layout of a tile_common.
    HOST_DEVICE
    RowMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return RowMajor(tileShape, stride());
    }

    /// Returns the shape of the layout
    HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    HOST_DEVICE
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    HOST_DEVICE
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    HOST_DEVICE
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    HOST_DEVICE
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    //
    // Data members
    //

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

}  // namespace NpuArch::layout

#endif  // FUSED_MATRIX_ROW_MAJOR_HPP