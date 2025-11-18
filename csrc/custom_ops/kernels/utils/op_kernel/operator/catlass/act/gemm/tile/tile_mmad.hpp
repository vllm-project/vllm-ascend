/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef ACT_GEMM_TILE_TILE_MMAD_HPP
#define ACT_GEMM_TILE_TILE_MMAD_HPP

#include "../../../act/act.hpp"
#include "../../../act/gemm/helper.hpp"
namespace Act::Gemm::Tile {

///////////////////////////////////////////////////////////

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// GemmType for A matrix operand
    class AType_,
    /// GemmType type for B matrix operand
    class BType_,
    /// GemmType type for Bias operand
    class BiasType_>
struct TileMmad {
    using ElementA = typename AType_::Element;
    using ElementB = typename BType_::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // Methods

    ACT_DEVICE
    TileMmad() {}

    ACT_DEVICE
    void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
                    AscendC::LocalTensor<ElementA> const &l0ATensor, AscendC::LocalTensor<ElementB> const &l0BTensor,
                    uint32_t m, uint32_t n, uint32_t k, bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;

        AscendC::Mmad(l0CTensor, l0ATensor, l0BTensor, mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

///////////////////////////////////////////TileMmadTla/////////////////////////////////////////////////

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// Tensor type for A matrix operand
    class TensorA,
    /// Tensor type for B matrix operand
    class TensorB,
    /// Tensor type for C matrix operand
    class TensorC,
    /// Tensor type for Bias operand
    class TensorBias = void>
struct TileMmadTla {
    // Methods

    ACT_DEVICE
    TileMmadTla() {}

    ACT_DEVICE
    void operator()(TensorC const &l0CTensor, TensorA const &l0ATensor, TensorB const &l0BTensor, bool initC = true,
                    uint8_t unitFlag = 0)
    {
        const uint32_t m = get<0>(l0ATensor.orgShape());
        const uint32_t n = get<1>(l0BTensor.orgShape());
        const uint32_t k = get<1>(l0ATensor.orgShape());

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;

        AscendC::Mmad(l0CTensor.data(), l0ATensor.data(), l0BTensor.data(), mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace Act::Gemm::Tile

#endif  // ACT_GEMM_TILE_TILE_MMAD_HPP
