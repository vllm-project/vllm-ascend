/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_LAYOUT_MATRIX_HPP
#define CATLASS_LAYOUT_MATRIX_HPP

#include "catlass/catlass.hpp"
#include "catlass/coord.hpp"
#include "catlass/numeric_size.hpp"
#include "catlass/detail/alignment.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/conv_coord.hpp"

namespace Catlass::layout {

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
    CATLASS_HOST_DEVICE
    RowMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(cols), LongIndex(1))) {}

    /// Constructor
    CATLASS_HOST_DEVICE
    RowMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(ldm, LongIndex(1))) {}

    /// Ctor
    CATLASS_HOST_DEVICE
    RowMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element>
    CATLASS_HOST_DEVICE
    static RowMajor MakeLayout(Index rows, Index cols)
    {
        return RowMajor(rows, cols);
    }

    template <class Element>
    CATLASS_HOST_DEVICE
    static RowMajor MakeLayoutInUb(MatrixCoord const &shape)
    {
        constexpr uint32_t ELE_NUM_PER_BLK = BytesToBits(BYTE_PER_BLK) / SizeOfBits<Element>::value;
        return RowMajor(shape.row(), shape.column(), RoundUp<ELE_NUM_PER_BLK>(shape.column()));
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) * stride_[0] + LongIndex(coord.column());
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    RowMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return RowMajor(tileShape, stride());
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

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(shape_[0]) * stride_[0];
    }

protected:
    //
    // Data members
    //

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for col-major matrices
struct ColumnMajor {
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
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE
    ColumnMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), LongIndex(rows))) {}

    /// Constructor
    CATLASS_HOST_DEVICE
    ColumnMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), ldm)) {}

    /// Ctor
    CATLASS_HOST_DEVICE
    ColumnMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element>
    CATLASS_HOST_DEVICE
    static ColumnMajor MakeLayout(Index rows, Index cols)
    {
        return ColumnMajor(rows, cols);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_[1];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    ColumnMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return ColumnMajor(tileShape, stride());
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

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(shape_[1]) * stride_[1];
    }

protected:
    //
    // Data members
    //

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for nZ matrices which is col-major inside fractal and row-major between fractal
struct nZ {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr
    nZ(Index orgRows = 0,                 /// Number of rows of origin matrices
       Index orgCols = 0,                 /// Number of cols of origin matrices
       Index rowsInFractal = 0,           /// Number of rows inside the fractal
       Index rowsByFractal = 0,           /// number of rows by the fractal
       Index colsInFractal = 0,           /// number of cols inside the fractal
       Index colsByFractal = 0,           /// number of cols by the fractal
       LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
       LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
       LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
       LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr
    nZ(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr
    static nZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
        Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return nZ(orgRows,
                  orgCols,
                  ELE_NUM_PER_C0,
                  rowsRound / ELE_NUM_PER_C0,
                  C0_NUM_PER_FRACTAL,
                  colsRound / C0_NUM_PER_FRACTAL,
                  1,
                  colsRound * ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  ELE_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
            (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    nZ GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)),
            shape(2), CeilDiv(tileOriShape.column(), shape(2))
        );
        return nZ(tileOriShape, tileShape, stride());
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
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

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(stride_[1]) * shape_[1];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for zN matrices which is row-major inside fractal and col-major between fractal
struct zN {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr
    zN(Index orgRows = 0,                 /// Number of rows of origin matrices
       Index orgCols = 0,                 /// Number of cols of origin matrices
       Index rowsInFractal = 0,           /// Number of rows inside the fractal
       Index rowsByFractal = 0,           /// number of rows by the fractal
       Index colsInFractal = 0,           /// number of cols inside the fractal
       Index colsByFractal = 0,           /// number of cols by the fractal
       LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
       LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
       LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
       LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr
    zN(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr
    static zN MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zN(orgRows,
                  orgCols,
                  C0_NUM_PER_FRACTAL,
                  rowsRound / C0_NUM_PER_FRACTAL,
                  ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  ELE_NUM_PER_FRACTAL,
                  1,
                  rowsRound * ELE_NUM_PER_C0);
    }

    CATLASS_HOST_DEVICE
    static zN MakeLayoutInL0C(MatrixCoord const &shape)
    {
        return zN(shape.row(),
                  shape.column(),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.row()),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.column()),
                  C0_NUM_PER_FRACTAL,
                  C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL,
                  1,
                  RoundUp<C0_NUM_PER_FRACTAL>(shape.row()) * C0_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
            (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    zN GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)),
            shape(2), CeilDiv(tileOriShape.column(), shape(2))
        );
        return zN(tileOriShape, tileShape, stride());
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
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

    /// Returns the length of the layout
    CATLASS_HOST_DEVICE
    LongIndex Capacity() const
    {
        return static_cast<LongIndex>(stride_[3]) * shape_[3];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for zN matrices which is row-major inside fractal and row-major between fractal
struct zZ {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr
    zZ(Index orgRows = 0,                 /// Number of rows of origin matrices
       Index orgCols = 0,                 /// Number of cols of origin matrices
       Index rowsInFractal = 0,           /// Number of rows inside the fractal
       Index rowsByFractal = 0,           /// number of rows by the fractal
       Index colsInFractal = 0,           /// number of cols inside the fractal
       Index colsByFractal = 0,           /// number of cols by the fractal
       LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
       LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
       LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
       LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr
    zZ(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE constexpr
    static zZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zZ(orgRows,
                  orgCols,
                  C0_NUM_PER_FRACTAL,
                  rowsRound / C0_NUM_PER_FRACTAL,
                  ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  colsRound * C0_NUM_PER_FRACTAL,
                  1,
                  ELE_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
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
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for padding rowmajor matrices
/// A special data layout designed to improve the efficiency of matrix operations in non-512B aligned scenarios.
/// This layout is row-major within blocks and also row-major between blocks.
struct PaddingRowMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Constructor
    CATLASS_HOST_DEVICE
    PaddingRowMajor(Index orgRows = 0, Index orgCols = 0, Index blockRows = 0, Index blockCols = 0) :
        orgShape_(MakeCoord(orgRows, orgCols)),
        shape_(MakeCoord(blockRows, CeilDiv(orgRows, blockRows), blockCols, CeilDiv(orgCols, blockCols))),
        stride_(MakeCoord((LongIndex)blockCols, (LongIndex)blockRows * (LongIndex)RoundUp(orgCols, blockCols),
        (LongIndex)1, (LongIndex)blockRows * (LongIndex)blockCols)) {}

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        LongIndex blockRows = (LongIndex)shape_[0];
        LongIndex blockCols = (LongIndex)shape_[2];
        return (LongIndex)coord.row() / blockRows * stride_[1]
            + (LongIndex)coord.column() / blockCols * stride_[3]
            + (LongIndex)coord.row() % blockRows * stride_[0]
            + (LongIndex)coord.column() % blockCols;
    }

    CATLASS_HOST_DEVICE
    PaddingRowMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return PaddingRowMajor(tileShape.row(), tileShape.column(), shape_[0], shape_[2]);
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
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
    //
    // Data members
    //

    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for padding columnmajor matrices
/// A special data layout designed to improve the efficiency of matrix operations in non-512B aligned scenarios.
/// This layout is column-major within blocks and also column-major between blocks.
struct PaddingColumnMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Constructor
    CATLASS_HOST_DEVICE
    PaddingColumnMajor(Index orgRows = 0, Index orgCols = 0, Index blockRows = 0, Index blockCols = 0) :
        orgShape_(MakeCoord(orgRows, orgCols)),
        shape_(MakeCoord(blockRows, CeilDiv(orgRows, blockRows), blockCols, CeilDiv(orgCols, blockCols))),
        stride_(MakeCoord((LongIndex)1, (LongIndex)blockRows * (LongIndex)blockCols, (LongIndex)blockRows,
        (LongIndex)RoundUp(orgRows, blockRows) * (LongIndex)blockCols)) {}

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        LongIndex blockRows = (LongIndex)shape_[0];
        LongIndex blockCols = (LongIndex)shape_[2];
        return (LongIndex)coord.row() / blockRows * stride_[1]
            + (LongIndex)coord.column() / blockCols * stride_[3]
            + (LongIndex)coord.row() % blockRows
            + (LongIndex)coord.column() % blockCols * stride_[2];
    }

    CATLASS_HOST_DEVICE
    PaddingColumnMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return PaddingColumnMajor(tileShape.row(), tileShape.column(), shape_[0], shape_[2]);
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
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
    //
    // Data members
    //

    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

///////////////////////
// new add layout nN
// nN layout
struct nN {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE
    nN(Index orgRows = 0,  /// Number of rows of origin matrices
    Index orgCols = 0,  /// Number of cols of origin matrices

    Index rowsInFractal = 0,  /// Number of rows inside the fractal
    Index rowsByFractal = 0,  /// number of rows by the fractal
    Index colsInFractal = 0,  /// number of cols inside the fractal
    Index colsByFractal = 0,  /// number of cols by the fractal

    LongIndex strideRowsInFractal = 0,  /// number of elements between adjacent rows inside the fractal
    LongIndex strideRowsByFractal = 0,  /// number of elements between adjacent fractal rows
    LongIndex strideColsInFractal = 0,  /// number of elements between adjacent cols inside the fractal
    LongIndex strideColsByFractal = 0)  /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
        shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
        stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {
    }

    /// Ctor
    CATLASS_HOST_DEVICE
    nN(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    CATLASS_HOST_DEVICE static nN MakeLayout(Index orgRows, Index orgCols) {
        static constexpr uint32_t ELE_NUM_PER_C0 = BytesToBits(BYTE_PER_C0) / SizeOfBits<Element>::value;
        static constexpr uint32_t ELE_NUM_PER_FRACTAL = BytesToBits(BYTE_PER_FRACTAL) / SizeOfBits<Element>::value;
        Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return nN(orgRows,
                orgCols,

                ELE_NUM_PER_C0,
                rowsRound / ELE_NUM_PER_C0,
                C0_NUM_PER_FRACTAL,
                colsRound / C0_NUM_PER_FRACTAL,

                1,
                ELE_NUM_PER_FRACTAL,
                ELE_NUM_PER_C0,
                rowsRound * C0_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(MatrixCoord const& coord) const {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index& orgShape(int idx) {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape shape() const {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    Shape& shape() {
        return shape_;
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index shape(int idx) const {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    CATLASS_HOST_DEVICE
    typename Shape::Index& shape(int idx) {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride stride() const {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    Stride& stride() {
        return stride_;
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index stride(int idx) const {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    CATLASS_HOST_DEVICE
    typename Stride::Index& stride(int idx) {
        return stride_[idx];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

struct NDC1HWC0 {
public:
    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    /// (N,D,C1,H,W,C0)
    static constexpr int ORG_SHAPE_RANK = 6;

    static constexpr int RANK = 5;
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;
    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr NDC1HWC0(
        Index batch = 0,
        Index D = 0,
        Index C1 = 0,
        Index H = 0,
        Index W = 0,
        Index C0 = 0,

        Index rowsInFractal = 0,  /// Number of rows inside the fractal
        Index rowsByFractal = 0,  /// number of rows by the fractal
        Index colsInFractal = 0,  /// number of cols inside the fractal
        Index colsByFractal = 0,  /// number of cols by the fractal

        LongIndex strideC0 = 0,                      /// number of elements between adjacent C0 cols
        LongIndex strideHW = 0,                       /// number of elements between adjacent W rows
        LongIndex StrideC1 = 0,                      /// number of elements between adjacent C1 cols
        LongIndex StrideD = 0,                       /// number of elements between adjacent D batchCols
        LongIndex StrideN = 0                        /// number of elements between adjacent batch
        )
        : orgShape_(MakeCoord(batch, D, C1, H, W, C0)),
          shape_(MakeCoord(batch, rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideC0, strideHW, StrideC1, StrideD, StrideN))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr
    NDC1HWC0(OrgShape orgshape, Shape shape, Stride stride) : orgShape_(orgshape), shape_(shape), stride_(stride) {}

    CATLASS_HOST_DEVICE constexpr
    static NDC1HWC0 MakeLayout(Index Batch, Index D, Index C1, Index H, Index W, Index C0)
    {
        return NDC1HWC0(Batch,
            D,
            C1,
            H,
            W,
            C0,

            W,
            H,
            C0,
            D * C1,

            1,                   /// StrideC0
            C0,                  /// StrideHW
            H * W * C0,          /// StrideC1
            H * W * C0 * C1,     /// StrideD
            H * W * C0 * C1 * D  /// StrideN
        );
    }

    // CATLASS_HOST_DEVICE
    /// Returns the offset of a coordinate in linear memory.
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(Conv3d6HdCoord const &coord) const
    {
        return LongIndex(coord.n()) * stride_[4] + LongIndex(coord.d()) * stride_[3] +
               LongIndex(coord.c1()) * stride_[2] + LongIndex(coord.hw()) * stride_[1];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    NDC1HWC0 GetTileLayout(OrgShape const &tileOriShape) const
    {
        Shape tileShape = MakeCoord(
            tileOriShape[0],
            tileOriShape[4],
            tileOriShape[3],
            shape(3),
            tileOriShape[1] * tileOriShape[2]
        );

        Stride tileStride = MakeCoord(stride(0),
            stride(1),
            (LongIndex)(tileOriShape[3] * tileOriShape[4] * shape(3)),
            (LongIndex)(tileOriShape[2] * tileOriShape[3] * tileOriShape[4] * shape(3)),
            (LongIndex)(tileOriShape[1] * tileOriShape[2] * tileOriShape[3] * tileOriShape[4] * shape(3))
        );
        return NDC1HWC0(tileOriShape, tileShape, tileStride);
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
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
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

struct KDC1KHKWN1N0C0 {
public:
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 4;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    CATLASS_HOST_DEVICE constexpr KDC1KHKWN1N0C0(Index KdC1KhKw = 0,  /// Merging Kd,Kh,Kw,C1 axes of KDC1KHKWN1N0C0
        Index N1 = 0,                                    /// Cout = N1*N0
        Index N0 = 0,
        Index C0 = 0,

        Index rowsInFractal = 0,           /// Number of rows inside the fractal
        Index rowsByFractal = 0,           /// number of rows by the fractal
        Index colsInFractal = 0,           /// number of cols inside the fractal
        Index colsByFractal = 0,           /// number of cols by the fractal

        LongIndex strideC0 = 0,     /// number of elements between adjacent rows inside the fractal
        LongIndex StrideDC1HW = 0,  /// number of elements between adjacent fractal rows
        LongIndex strideN0 = 0,     /// number of elements between adjacent cols inside the fractal
        LongIndex strideN1 = 0      /// number of elements between adjacent fractal cols
        )
        : orgShape_(MakeCoord(KdC1KhKw, N1, N0, C0)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideC0, strideN0, strideN1, StrideDC1HW))
    {}

    /// Ctor
    CATLASS_HOST_DEVICE constexpr
    KDC1KHKWN1N0C0(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (Kd*C1*Kh*Kw,N1,N0,C0)
    CATLASS_HOST_DEVICE constexpr
    static KDC1KHKWN1N0C0 MakeLayout(Index KdC1KhKw, Index N1, Index N0, Index C0)
    {
        return KDC1KHKWN1N0C0(KdC1KhKw,
            N1,
            N0,
            C0,

            C0,
            KdC1KhKw,
            N0,
            N1,

            1,             /// StrideC0
            C0 * N0 * N1,  /// StrideDC1HW
            C0,            /// StrideN0
            C0 * N0        /// StrideN1
        );
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (KdC1KhKw_idx, N1_idx)
    CATLASS_HOST_DEVICE
    LongIndex GetOffset(Conv3dFracZ3dCoord const &coord) const
    {
        return LongIndex(coord.kdc1khkw()) * stride_[3] + LongIndex(coord.n1()) * stride_[1];
    }

    /// Returns the layout of a tile.
    CATLASS_HOST_DEVICE
    KDC1KHKWN1N0C0 GetTileLayout(OrgShape const &tileOriShape) const
    {
        Shape tileShape = MakeCoord(shape(0),  /// C0
            tileOriShape[0],                   /// Kd*C1*Kh*Kw
            shape(2),                          /// N0
            tileOriShape[1]                    /// N1
        );
        Stride tileStride = MakeCoord(stride(0),            /// TileStrideC0
            stride(2) * tileOriShape[1] * tileOriShape[2],  /// TileStrideDC1HW
            (LongIndex)shape(0),                            /// TileStrideN0
            stride(2) * tileOriShape[2]                    /// TileStrideN1
        );
        return KDC1KHKWN1N0C0(tileOriShape, tileShape, tileStride);
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    CATLASS_HOST_DEVICE
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
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
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};
}  // namespace Catlass::layout

#endif  // CATLASS_LAYOUT_MATRIX_HPP