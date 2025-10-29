/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_LAYOUT_VECTOR_HPP
#define CATLASS_LAYOUT_VECTOR_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"

namespace Catlass::layout {

struct VectorLayout {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 1;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Shape vector
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

    /// Logical coordinate
    using TensorCoord = Coord<RANK, Index>;

public:
    // Methods

    CATLASS_HOST_DEVICE
    VectorLayout(Index size = 0) : shape_(MakeCoord(size)), stride_(MakeCoord(LongIndex(1))) {}

    CATLASS_HOST_DEVICE
    VectorLayout(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element>
    CATLASS_HOST_DEVICE
    static VectorLayout MakeLayoutInUb(TensorCoord const &tileShape)
    {
        return VectorLayout{RoundUp<BYTE_PER_BLK / sizeof(Element)>(tileShape[0])};
    }

    CATLASS_HOST_DEVICE
    LongIndex GetOffset(TensorCoord const &coord) const
    {
        return stride_[0] * coord[0];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    VectorLayout GetTileLayout(TensorCoord const &tileShape) const
    {
        return VectorLayout(tileShape, stride());
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    /// Stride data member
    Shape shape_;
    Stride stride_;
};

} // namespace Catlass::layout

#endif  // CATLASS_LAYOUT_VECTOR_HPP