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
 * \file gemm_coord.hpp
 * \brief
 */

#ifndef GEMM_COORD_HPP
#define GEMM_COORD_HPP

#include "../attn_infra/coord.hpp"

namespace NpuArch {

/// Shape of a matrix multiply-add operation
template <
    /// Rows of matrix product
    uint32_t M_ = 1,
    /// Columns of matrix product
    uint32_t N_ = 1,
    /// Inner dimension of matrix product
    uint32_t K_ = 1
>
struct GemmShape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;

    static constexpr int64_t MN = M * N;
    static constexpr int64_t MK = M * K;
    static constexpr int64_t KN = N * K;
    static constexpr int64_t MNK = M * N * K;

    static constexpr int64_t COUNT = MNK;

    /// Returns a Coord object
    HOST_DEVICE
    static Coord<3> ToCoord()
    {
        return MakeCoord(M, N, K);
    }

    HOST_DEVICE
    static Coord<2> ToCoordMN()
    {
        return MakeCoord(M, N);
    }

    HOST_DEVICE
    static Coord<2> ToCoordMK()
    {
        return MakeCoord(M, K);
    }

    HOST_DEVICE
    static Coord<2> ToCoordKN()
    {
        return MakeCoord(K, N);
    }
};

/// GemmCoord is a structure derived from Coord<3> that specifies a location within the
/// coordinate space of a Gemm problem.
struct GemmCoord : public Coord<3, uint32_t> {
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=3
    using Base = Coord<3, Index>;

    /// Gemm M dimension - rows of the output C matrix
    static constexpr int M_INDEX = 0;

    /// Gemm N dimension - columns of the output C matrix
    static constexpr int N_INDEX = 1;

    /// Gemm K dimension - inner dimension of the Gemm problem
    static constexpr int K_INDEX = 2;

    /// Default ctor
    HOST_DEVICE
    GemmCoord() {}

    /// Constructs from Coord<3> and a batch
    HOST_DEVICE
    GemmCoord(Coord<3, Index> const &coord) : Base(coord) {}

    /// Helper to construct from a K, N, M, batch variables
    HOST_DEVICE
    GemmCoord(Index m, Index n, Index k) : Base(MakeCoord(m, n, k)) {}

    /// Returns the Gemm M coordinate
    HOST_DEVICE
    Index const &m() const
    {
        return this->At(M_INDEX);
    }

    /// Returns reference to the Gemm M coordinate
    HOST_DEVICE
    Index &m()
    {
        return this->At(M_INDEX);
    }

    /// Returns the Gemm N coordinate
    HOST_DEVICE
    Index const &n() const
    {
        return this->At(N_INDEX);
    }

    /// Returns reference to the Gemm N coordinate
    HOST_DEVICE
    Index &n()
    {
        return this->At(N_INDEX);
    }

    /// Returns the Gemm K coordinate
    HOST_DEVICE
    Index const &k() const
    {
        return this->At(K_INDEX);
    }

    /// Returns reference to the Gemm K coordinate
    HOST_DEVICE
    Index &k()
    {
        return this->At(K_INDEX);
    }

    HOST_DEVICE
    auto GetCoordMN() const
    {
        return this->GetCoordByAxis<M_INDEX, N_INDEX>();
    }

    HOST_DEVICE
    auto GetCoordMK() const
    {
        return this->GetCoordByAxis<M_INDEX, K_INDEX>();
    }

    HOST_DEVICE
    auto GetCoordKN() const
    {
        return this->GetCoordByAxis<K_INDEX, N_INDEX>();
    }
};

} // namespace NpuArch

#endif  // GEMM_COORD_HPP
