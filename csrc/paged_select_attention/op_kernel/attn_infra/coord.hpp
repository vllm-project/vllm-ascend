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
 * \file coord.hpp
 * \brief
 */

#ifndef COORD_HPP
#define COORD_HPP

#include "../attn_infra/base_defs.hpp"

namespace NpuArch {

/// Statically-sized array specifying Coords within a tensor
template <
    int RANK_,                         ///< Logical rank of coordinate
    class Index_ = uint32_t,        ///< Index type used for each dimension
    class LongIndex_ = int64_t      ///< Long index type used for linear offsets
>
struct Coord {
public:
    // Number of elements in Coord
    static const int RANK = RANK_;

    // Index typen used to store elements
    using Index = Index_;

    // Type used to represent linear offsets
    using LongIndex = LongIndex_;

    // Default ctor initializes uniformly
    HOST_DEVICE constexpr
    explicit Coord(Index value = Index(0))
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = value;
        }
    }

    // Constructs from an array of integers
    HOST_DEVICE constexpr
    Coord(Index const (&idx_)[RANK])
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = idx_[i];
        }
    }

    // Constructs frrom an array of integers
    HOST_DEVICE
    int Argmin() const
    {
        int i = 0;
        for (int j = 1; j < RANK; ++j) {
            if (idx[j] < idx[i]) {
                i = j;
            }
        }
        return i;
    }

    // Returns the index of the dimension with greatest value
    HOST_DEVICE
    int Argmax() const
    {
        int i = 0;
        for (int j = 1; j < RANK; ++j) {
            if (idx[j] > idx[i]) {
                i = j;
            }
        }
        return i;
    }

    // Returns true if Coord is non-zero
    HOST_DEVICE
    explicit operator bool() const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i]) {
                return true;
            }
        }
        return false;
    }

    // Return true if Coord is uniformly zero.
    HOST_DEVICE
    bool operator!() const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i]) {
                return false;
            }
        }
        return true;
    }

    // Element-wise addition
    HOST_DEVICE
    Coord operator+(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c.idx[i] = idx[i] + b.idx[i];
        }
        return c;
    }

    // Add a scalar to each element
    HOST_DEVICE
    Coord operator+(const Index val) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c.idx[i] = idx[i] + val;
        }
        return c;
    }

    // Element-wise subtraction
    HOST_DEVICE
    Coord operator-(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] - b.idx[i];
        }
        return c;
    }

    // Subtract a scalar from each element
    HOST_DEVICE
    Coord operator-(Index const val) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c.idx[i] = idx[i] - val;
        }
        return c;
    }

    // Element-wise multiply
    HOST_DEVICE
    Coord operator*(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] * b.idx[i];
        }
        return c;
    }

    // Element-wise division
    HOST_DEVICE
    Coord operator/(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] / b.idx[i];
        }
        return c;
    }

    // Element-wise mod
    HOST_DEVICE
    Coord operator%(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] % b.idx[i];
        }
        return c;
    }

    // In-place addition
    HOST_DEVICE
    Coord &operator+=(Coord const &b)
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] += b.idx[i];
        }
        return *this;
    }

    // In-place equal
    HOST_DEVICE
    bool operator==(Coord const &b) const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != b.idx[i]) {
                return false;
            }
        }
        return true;
    }

    // In-place equal
    HOST_DEVICE
    bool operator==(Index const val) const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != val) {
                return false;
            }
        }
        return true;
    }

    // Member acces operator
    HOST_DEVICE
    Index &operator[](int dim)
    {
        return idx[dim];
    }

    // Member access operator
    HOST_DEVICE
    Index const &operator[](int dim) const
    {
        return idx[dim];
    }

    // Gets the index of a given Coord element
    template <int DIM>
    HOST_DEVICE
    Index &At()
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    HOST_DEVICE
    Index &At(int dim)
    {
        return idx[dim];
    }

    // Gets the index of a given Coord element
    template <int DIM>
    HOST_DEVICE
    Index const &At() const
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    HOST_DEVICE
    Index const &At(int dim) const
    {
        return idx[dim];
    }

    template <int... Is>
    HOST_DEVICE
    auto GetCoordByAxis() const
    {
        Index idx_[sizeof...(Is)]{idx[Is]...};
        return Coord<sizeof...(Is), Index, LongIndex>{idx_};
    }

    HOST_DEVICE
    static Coord Min(Coord const &a, Coord const &b)
    {
        Coord res;
        for (int i = 0; i < RANK; ++i) {
            res[i] = a[i] < b[i] ? a[i] : b[i];
        }
        return res;
    }

private:
    // Indices
    Index idx[RANK];
};

// Helper to make a 1-element coordinate
template <class T>
HOST_DEVICE constexpr
Coord<1, T> MakeCoord(T dim0)
{
    T values[1] = {dim0};
    return Coord<1, T>(values);
}

/// Helper to make a 2-element coordinate
template <class T>
HOST_DEVICE constexpr
Coord<2, T> MakeCoord(T dim0, T dim1)
{
    T values[2] = {dim0, dim1};
    return Coord<2, T>(values);
}

/// Helper to make a 3-element coordinate
template <class T>
HOST_DEVICE constexpr
Coord<3, T> MakeCoord(T dim0, T dim1, T dim2)
{
    T values[3] = {dim0, dim1, dim2};
    return Coord<3, T>(values);
}

/// Helper to make a 4-element coordinate
template <class T>
HOST_DEVICE constexpr
Coord<4, T> MakeCoord(T dim0, T dim1, T dim2, T dim3)
{
    T values[4] = {dim0, dim1, dim2, dim3};
    return Coord<4, T>(values);
}

}  // namespace NpuArch

#endif  // COORD_HPP