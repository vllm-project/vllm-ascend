/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

    HOST_DEVICE
    int Argmin() const
    {
        return ArgminImpl<1>(0);
    }

    // Returns the index of the dimension with greatest value
    HOST_DEVICE
    int Argmax() const
    {
        return ArgmaxImpl<1>(0);
    }

    // Returns true if Coord is non-zero
    HOST_DEVICE
    explicit operator bool() const
    {
        return AnyImpl<0>();
    }

    // Return true if Coord is uniformly zero.
    HOST_DEVICE
    bool operator!() const
    {
        return !AnyImpl<0>();
    }

    // Element-wise addition
    HOST_DEVICE
    Coord operator+(Coord const &b) const
    {
        Coord c;
        AddCoordImpl<0>(c, b);
        return c;
    }

    // Add a scalar to each element
    HOST_DEVICE
    Coord operator+(const Index val) const
    {
        Coord c;
        AddScalarImpl<0>(c, val);
        return c;
    }

    // Element-wise subtraction
    HOST_DEVICE
    Coord operator-(Coord const &b) const
    {
        Coord c;
        SubCoordImpl<0>(c, b);
        return c;
    }

    // Subtract a scalar from each element
    HOST_DEVICE
    Coord operator-(Index const val) const
    {
        Coord c;
        SubScalarImpl<0>(c, val);
        return c;
    }

    // Element-wise multiply
    HOST_DEVICE
    Coord operator*(Coord const &b) const
    {
        Coord c;
        MulCoordImpl<0>(c, b);
        return c;
    }

    // Element-wise division
    HOST_DEVICE
    Coord operator/(Coord const &b) const
    {
        Coord c;
        DivCoordImpl<0>(c, b);
        return c;
    }

    // Element-wise mod
    HOST_DEVICE
    Coord operator%(Coord const &b) const
    {
        Coord c;
        ModCoordImpl<0>(c, b);
        return c;
    }

    // In-place addition
    HOST_DEVICE
    Coord &operator+=(Coord const &b)
    {
        PlusEqualImpl<0>(b);
        return *this;
    }

    // In-place equal
    HOST_DEVICE
    bool operator==(Coord const &b) const
    {
        return EqualCoordImpl<0>(b);
    }

    // In-place equal
    HOST_DEVICE
    bool operator==(Index const val) const
    {
        return EqualScalarImpl<0>(val);
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
    template <int N>
    HOST_DEVICE
    int ArgminImpl(int i) const
    {
        if constexpr (N == RANK) {
            return i;
        }
        else {
            return ArgminImpl<N + 1>(idx[N] < idx[i] ? N : i);
        }
    }

    template <int N>
    HOST_DEVICE
    int ArgmaxImpl(int i) const
    {
        if constexpr (N == RANK) {
            return i;
        }
        else {
            return ArgmaxImpl<N + 1>(idx[N] > idx[i] ? N : i);
        }
    }
    
    template <int N>
    HOST_DEVICE
    bool AnyImpl() const
    {
        if constexpr (N == RANK) {
            return false;
        }
        else {
            return idx[N] || AnyImpl<N + 1>();
        }
    }

    template <int N>
    HOST_DEVICE
    void AddCoordImpl(Coord &c, Coord const &b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] + b.idx[N];
            AddCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    HOST_DEVICE
    void AddScalarImpl(Coord &c, Index const val) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] + val;
            AddScalarImpl<N + 1>(c, val);
        }
    }

    template <int N>
    HOST_DEVICE
    void SubCoordImpl(Coord &c, Coord const &b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] - b.idx[N];
            SubCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    HOST_DEVICE
    void SubScalarImpl(Coord &c, Index const val) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] - val;
            SubScalarImpl<N + 1>(c, val);
        }
    }

    template <int N>
    HOST_DEVICE
    void MulCoordImpl(Coord &c, Coord const &b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] * b.idx[N];
            MulCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    HOST_DEVICE
    void DivCoordImpl(Coord &c, Coord const &b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] / b.idx[N];
            DivCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    HOST_DEVICE
    void ModCoordImpl(Coord &c, Coord const &b) const
    {
        if constexpr (N < RANK) {
            c.idx[N] = idx[N] % b.idx[N];
            ModCoordImpl<N + 1>(c, b);
        }
    }

    template <int N>
    HOST_DEVICE
    void PlusEqualImpl(Coord const &b)
    {
        if constexpr (N < RANK) {
            idx[N] += b.idx[N];
            PlusEqualImpl<N + 1>(b);
        }
    }

    template <int N>
    HOST_DEVICE
    bool EqualCoordImpl(Coord const &b) const
    {
        if constexpr (N == RANK) {
            return true;
        }
        else {
            return idx[N] == b.idx[N] && EqualCoordImpl<N + 1>(b);
        }
    }

    template <int N>
    HOST_DEVICE
    bool EqualScalarImpl(Index const val) const
    {
        if constexpr (N == RANK) {
            return true;
        }
        else {
            return idx[N] == val && EqualScalarImpl<N + 1>(val);
        }
    }

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

/// Helper to make a 5-element coordinate
template <class T>
HOST_DEVICE constexpr
Coord<5, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4)
{
    T values[5] = {dim0, dim1, dim2, dim3, dim4};
    return Coord<5, T>(values);
}

/// Helper to make a 6-element coordinate
template <class T>
HOST_DEVICE constexpr
Coord<6, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4, T dim5)
{
    T values[6] = {dim0, dim1, dim2, dim3, dim4, dim5};
    return Coord<6, T>(values);
}

/// Helper to make a 7-element coordinate
template <class T>
HOST_DEVICE constexpr
Coord<7, T> MakeCoord(T dim0, T dim1, T dim2, T dim3, T dim4, T dim5, T dim6)
{
    T values[7] = {dim0, dim1, dim2, dim3, dim4, dim5, dim6};
    return Coord<7, T>(values);
}

}  // namespace NpuArch

#endif  // COORD_HPP