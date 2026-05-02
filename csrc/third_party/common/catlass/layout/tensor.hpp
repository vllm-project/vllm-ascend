/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_LAYOUT_TENSOR_HPP
#define CATLASS_LAYOUT_TENSOR_HPP

#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/conv_coord.hpp"

namespace Catlass::layout {

struct NC1HWC0 { // (Batch, C1, H, W, C0)
    /// Logical rank of tensor
    static constexpr int RANK = 5;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Ctor
    CATLASS_HOST_DEVICE
    NC1HWC0(Index batch = 0, Index c1 = 0, Index h = 0, Index w = 0, Index c0 = 0)
        : shape_(MakeCoord(batch, c1, h, w, c0))
    {
        LongIndex strideBatch = c1 * h * w * c0;
        LongIndex strideC1 = h * w * c0;
        LongIndex strideH = w * c0;
        LongIndex strideW = c0;
        LongIndex strideC0 = 1;
        stride_ = MakeCoord(strideBatch, strideC1, strideH, strideW, strideC0);
    }

    CATLASS_HOST_DEVICE
    NC1HWC0(Index batch, Index c1, Index h, Index w, Index c0,
        LongIndex strideBatch, LongIndex strideC1, LongIndex strideH, LongIndex strideW, LongIndex strideC0)
        : shape_(MakeCoord(batch, c1, h, w, c0)), 
        stride_(MakeCoord(strideBatch, strideC1, strideH, strideW, strideC0)) {}

    CATLASS_HOST_DEVICE
    NC1HWC0(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (batch, c1, h, w, c0)
    template <class Element>
    CATLASS_HOST_DEVICE
    static NC1HWC0 MakeLayout(Index batch, Index c1, Index h, Index w, Index c0) {
        return NC1HWC0(batch, c1, h, w, c0);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (batch, c1, h, w, c0)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(Conv2dFmapCoord const &coord) const {
        return LongIndex(coord.batch()) * stride_[0] +
            LongIndex(coord.c1()) * stride_[1] + 
            LongIndex(coord.h()) * stride_[2] +
            LongIndex(coord.w()) * stride_[3] +
            LongIndex(coord.c0()) * stride_[4]; 
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    NC1HWC0 GetTileLayout(Conv2dFmapCoord const &tileShape) const {
        return NC1HWC0(tileShape, stride());
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape &shape() {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index &shape(int idx) {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride &stride() {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index &stride(int idx) {
        return stride_[idx];
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    size_t Capacity() {
        return static_cast<size_t>(shape_[0]) * stride_[0];
    }

private:
    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

struct CI1KHKWCOCI0 { // (Cin1, Kh, Kw, Cout, C0)
    /// Logical rank of tensor
    static constexpr int RANK = 5;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Ctor
    CATLASS_HOST_DEVICE
    CI1KHKWCOCI0(Index cin1 = 0, Index kh = 0, Index kw = 0, Index cout = 0, Index c0 = 0)
        : shape_(MakeCoord(cin1, kh, kw, cout, c0))
    {
        LongIndex strideCin1 = kh * kw * cout * c0;
        LongIndex strideKh = kw * cout * c0;
        LongIndex strideKw = cout * c0;
        LongIndex strideCout = c0;
        LongIndex strideC0 = 1;
        stride_ = MakeCoord(strideCin1, strideKh, strideKw, strideCout, strideC0);
    }

    CATLASS_HOST_DEVICE
    CI1KHKWCOCI0(Index cin1, Index kh, Index kw, Index cout, Index c0,
        LongIndex strideCin1, LongIndex strideKh, LongIndex strideKw,
        LongIndex strideCout, LongIndex strideC0)
        : shape_(MakeCoord(cin1, kh, kw, cout, c0)), 
        stride_(MakeCoord(strideCin1, strideKh, strideKw, strideCout, strideC0)) {}

    CATLASS_HOST_DEVICE
    CI1KHKWCOCI0(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (cin1, h, w, c0)
    template <class Element>
    CATLASS_HOST_DEVICE
    static CI1KHKWCOCI0 MakeLayout(Index cin1, Index kh, Index kw, Index cout, Index c0) {
        return CI1KHKWCOCI0(cin1, kh, kw, cout, c0);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (cin1, kh, kw, cout, c0)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(Conv2dFilterCoord const &coord) const {
        return LongIndex(coord.cin1()) * stride_[0] + 
            LongIndex(coord.kh()) * stride_[1] +
            LongIndex(coord.kw()) * stride_[2] +
            LongIndex(coord.cout()) * stride_[3] +
            LongIndex(coord.c0()) * stride_[4]; 
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    CI1KHKWCOCI0 GetTileLayout(Conv2dFilterCoord const &tileShape) const {
        return CI1KHKWCOCI0(tileShape, stride());
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape &shape() {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index &shape(int idx) {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride &stride() {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index &stride(int idx) {
        return stride_[idx];
    }

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    size_t Capacity() {
        return static_cast<size_t>(shape_[0]) * stride_[0];
    }

private:
    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

}  // namespace Catlass::layout

#endif  // CATLASS_LAYOUT_TENSOR_HPP