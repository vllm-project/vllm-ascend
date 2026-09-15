/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_regbase.h
 * \brief CausalConv1d RegBase kernel implementation for arch35 (Ascend 950).
 */

#ifndef CAUSAL_CONV1D_REGBASE_H
#define CAUSAL_CONV1D_REGBASE_H

namespace NsCausalConv1d {

using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint16_t REGBASE_VECTOR_LENGTH = VECTOR_REG_WIDTH / sizeof(float);

constexpr CastTrait CAST_TRAIT_B16_TO_B32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr CastTrait CAST_TRAIT_B32_TO_B16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T, int32_t kTemplateWidth>
__aicore__ inline void RestoreFnLocalPartialsRegbase(
    LocalTensor<T> history, LocalTensor<float> weightF, LocalTensor<float> state0F, LocalTensor<float> state1F,
    LocalTensor<float> state2F, uint32_t dataCount, uint32_t tileStep)
{
    __ubuf__ T *historyAddr = (__ubuf__ T *)history.GetPhyAddr();
    __ubuf__ float *weightAddr = (__ubuf__ float *)weightF.GetPhyAddr();
    __ubuf__ float *state0Addr = (__ubuf__ float *)state0F.GetPhyAddr();
    __ubuf__ float *state1Addr = (__ubuf__ float *)state1F.GetPhyAddr();
    __ubuf__ float *state2Addr = (__ubuf__ float *)state2F.GetPhyAddr();

    uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(dataCount, REGBASE_VECTOR_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<T> historyT;
        RegTensor<float> historyF;
        RegTensor<float> weight0F;
        RegTensor<float> weight1F;
        RegTensor<float> weight2F;
        RegTensor<float> state0F;
        RegTensor<float> state1F;
        RegTensor<float> state2F;
        RegTensor<float> tmpF;
        MaskReg pregLoop;
        for (uint16_t j = 0; j < colLoopTimes; ++j) {
            pregLoop = UpdateMask<float>(dataCount);
            LoadAlign<float>(weight0F, weightAddr + j * REGBASE_VECTOR_LENGTH);
            if constexpr (kTemplateWidth >= 3) {
                LoadAlign<float>(weight1F, weightAddr + tileStep + j * REGBASE_VECTOR_LENGTH);
            }
            if constexpr (kTemplateWidth >= 4) {
                LoadAlign<float>(weight2F, weightAddr + 2 * tileStep + j * REGBASE_VECTOR_LENGTH);
            }

            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(historyT, historyAddr + j * REGBASE_VECTOR_LENGTH);
            Cast<float, T, CAST_TRAIT_B16_TO_B32>(historyF, historyT, pregLoop);
            Mul(state0F, historyF, weight0F, pregLoop);

            if constexpr (kTemplateWidth >= 3) {
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(
                    historyT, historyAddr + tileStep + j * REGBASE_VECTOR_LENGTH);
                Cast<float, T, CAST_TRAIT_B16_TO_B32>(historyF, historyT, pregLoop);
                Mul(tmpF, historyF, weight1F, pregLoop);
                Add(state0F, state0F, tmpF, pregLoop);
                Mul(state1F, historyF, weight0F, pregLoop);
            }

            if constexpr (kTemplateWidth >= 4) {
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(
                    historyT, historyAddr + 2 * tileStep + j * REGBASE_VECTOR_LENGTH);
                Cast<float, T, CAST_TRAIT_B16_TO_B32>(historyF, historyT, pregLoop);
                Mul(tmpF, historyF, weight2F, pregLoop);
                Add(state0F, state0F, tmpF, pregLoop);
                Mul(tmpF, historyF, weight1F, pregLoop);
                Add(state1F, state1F, tmpF, pregLoop);
                Mul(state2F, historyF, weight0F, pregLoop);
            }

            StoreAlign<float>(state0Addr + j * REGBASE_VECTOR_LENGTH, state0F, pregLoop);
            if constexpr (kTemplateWidth >= 3) {
                StoreAlign<float>(state1Addr + j * REGBASE_VECTOR_LENGTH, state1F, pregLoop);
            }
            if constexpr (kTemplateWidth >= 4) {
                StoreAlign<float>(state2Addr + j * REGBASE_VECTOR_LENGTH, state2F, pregLoop);
            }
        }
    }
}

template <typename T, int32_t kTemplateWidth, bool hasActivation>
__aicore__ inline void ComputeFnRollingTokenRegbase(
    LocalTensor<T> ring, LocalTensor<float> weightF, LocalTensor<float> state0F, LocalTensor<float> state1F,
    LocalTensor<float> state2F, LocalTensor<T> out, uint32_t dataCount, uint32_t tileStep)
{
    __ubuf__ T *ringAddr = (__ubuf__ T *)ring.GetPhyAddr();
    __ubuf__ float *weightAddr = (__ubuf__ float *)weightF.GetPhyAddr();
    __ubuf__ float *state0Addr = (__ubuf__ float *)state0F.GetPhyAddr();
    __ubuf__ float *state1Addr = (__ubuf__ float *)state1F.GetPhyAddr();
    __ubuf__ float *state2Addr = (__ubuf__ float *)state2F.GetPhyAddr();
    __ubuf__ T *outAddr = (__ubuf__ T *)out.GetPhyAddr();

    uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(dataCount, REGBASE_VECTOR_LENGTH));
    __VEC_SCOPE__
    {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<T> tokenT;
        RegTensor<float> tokenF;
        RegTensor<float> weight0F;
        RegTensor<float> weight1F;
        RegTensor<float> weight2F;
        RegTensor<float> weight3F;
        RegTensor<float> state0F;
        RegTensor<float> state1F;
        RegTensor<float> state2F;
        RegTensor<float> outputF;
        RegTensor<float> tmpF;
        MaskReg pregLoop;
        for (uint16_t j = 0; j < colLoopTimes; ++j) {
            pregLoop = UpdateMask<float>(dataCount);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(tokenT, ringAddr + j * REGBASE_VECTOR_LENGTH);
            Cast<float, T, CAST_TRAIT_B16_TO_B32>(tokenF, tokenT, pregLoop);
            LoadAlign<float>(weight0F, weightAddr + j * REGBASE_VECTOR_LENGTH);
            LoadAlign<float>(weight1F, weightAddr + tileStep + j * REGBASE_VECTOR_LENGTH);
            if constexpr (kTemplateWidth >= 3) {
                LoadAlign<float>(weight2F, weightAddr + 2 * tileStep + j * REGBASE_VECTOR_LENGTH);
            }
            if constexpr (kTemplateWidth >= 4) {
                LoadAlign<float>(weight3F, weightAddr + 3 * tileStep + j * REGBASE_VECTOR_LENGTH);
            }

            LoadAlign<float>(state0F, state0Addr + j * REGBASE_VECTOR_LENGTH);
            if constexpr (kTemplateWidth == 2) {
                Mul(tmpF, tokenF, weight1F, pregLoop);
            } else if constexpr (kTemplateWidth == 3) {
                Mul(tmpF, tokenF, weight2F, pregLoop);
            } else if constexpr (kTemplateWidth == 4) {
                Mul(tmpF, tokenF, weight3F, pregLoop);
            }
            Add(outputF, state0F, tmpF, pregLoop);

            if constexpr (kTemplateWidth == 2) {
                Mul(state0F, tokenF, weight0F, pregLoop);
            } else if constexpr (kTemplateWidth == 3) {
                LoadAlign<float>(state1F, state1Addr + j * REGBASE_VECTOR_LENGTH);
                Mul(tmpF, tokenF, weight1F, pregLoop);
                Add(state0F, state1F, tmpF, pregLoop);
                Mul(state1F, tokenF, weight0F, pregLoop);
                StoreAlign<float>(state1Addr + j * REGBASE_VECTOR_LENGTH, state1F, pregLoop);
            } else if constexpr (kTemplateWidth == 4) {
                LoadAlign<float>(state1F, state1Addr + j * REGBASE_VECTOR_LENGTH);
                LoadAlign<float>(state2F, state2Addr + j * REGBASE_VECTOR_LENGTH);
                Mul(tmpF, tokenF, weight2F, pregLoop);
                Add(state0F, state1F, tmpF, pregLoop);
                Mul(tmpF, tokenF, weight1F, pregLoop);
                Add(state1F, state2F, tmpF, pregLoop);
                Mul(state2F, tokenF, weight0F, pregLoop);
                StoreAlign<float>(state1Addr + j * REGBASE_VECTOR_LENGTH, state1F, pregLoop);
                StoreAlign<float>(state2Addr + j * REGBASE_VECTOR_LENGTH, state2F, pregLoop);
            }
            StoreAlign<float>(state0Addr + j * REGBASE_VECTOR_LENGTH, state0F, pregLoop);

            if constexpr (hasActivation) {
                Muls(tmpF, outputF, -1.0f, pregLoop);
                Exp(tmpF, tmpF, pregLoop);
                Adds(tmpF, tmpF, 1.0f, pregLoop);
                Div(outputF, outputF, tmpF, pregLoop);
            }

            if constexpr (IsSameType<T, float>::value) {
                StoreAlign<float>(outAddr + j * REGBASE_VECTOR_LENGTH, outputF, pregLoop);
            } else {
                Cast<T, float, CAST_TRAIT_B32_TO_B16>(tokenT, outputF, pregLoop);
                StoreAlign<T, StoreDist::DIST_PACK_B32>(outAddr + j * REGBASE_VECTOR_LENGTH, tokenT, pregLoop);
            }
        }
    }
}

template <typename T, bool hasActivation>
__aicore__ inline void ComputeFnRollingOutputRegbase(LocalTensor<T> ring, LocalTensor<float> currF,
                                                     LocalTensor<float> state0F, LocalTensor<float> weightF,
                                                     uint32_t dataCount)
{
    __ubuf__ T *ringAddr = (__ubuf__ T *)ring.GetPhyAddr();
    __ubuf__ float *currFAddr = (__ubuf__ float *)currF.GetPhyAddr();
    __ubuf__ float *state0FAddr = (__ubuf__ float *)state0F.GetPhyAddr();
    __ubuf__ float *weightFAddr = (__ubuf__ float *)weightF.GetPhyAddr();

    uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(dataCount, REGBASE_VECTOR_LENGTH));
    __VEC_SCOPE__
    {
        RegTensor<T> ring;
        RegTensor<float> currF;
        RegTensor<float> state0F;
        RegTensor<float> weightF;
        RegTensor<float> tmp;
        MaskReg pregLoop;
        for (uint16_t j = 0; j < colLoopTimes; ++j) {
            pregLoop = UpdateMask<float>(dataCount);
            DataCopy<T, LoadDist::DIST_UNPACK_B16>(ring, ringAddr + j * REGBASE_VECTOR_LENGTH);
            DataCopy(state0F, state0FAddr + j * REGBASE_VECTOR_LENGTH);
            DataCopy(weightF, weightFAddr + j * REGBASE_VECTOR_LENGTH);
            Cast<float, T, CAST_TRAIT_B16_TO_B32>(currF, ring, pregLoop);
            Mul(currF, currF, weightF, pregLoop);
            Add(state0F, state0F, currF, pregLoop);
            if constexpr (hasActivation) {
                Muls(tmp, state0F, -1.0f, pregLoop);
                Exp(tmp, tmp, pregLoop);
                Adds(tmp, tmp, 1.0f, pregLoop);
                Div(currF, state0F, tmp, pregLoop);
                DataCopy(currFAddr + j * REGBASE_VECTOR_LENGTH, currF, pregLoop);
            } else {
                DataCopy(state0FAddr + j * REGBASE_VECTOR_LENGTH, state0F, pregLoop);
            }
        }
    }
}

template <typename T>
static __simd_vf__ inline void AdvanceFnLocalPartialsWidthTwo(__ubuf__ T *ringAddr,
                                                              __ubuf__ float *weight0FAddr,
                                                              __ubuf__ float *state0FAddr, uint32_t dataCount,
                                                              uint16_t colLoopTimes)
{
    RegTensor<T> ring;
    RegTensor<float> currF;
    RegTensor<float> weight0F;
    RegTensor<float> state0F;
    MaskReg pregLoop;
    for (uint16_t j = 0; j < colLoopTimes; ++j) {
        pregLoop = UpdateMask<float>(dataCount);
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(ring, ringAddr + j * REGBASE_VECTOR_LENGTH);
        DataCopy(weight0F, weight0FAddr + j * REGBASE_VECTOR_LENGTH);
        Cast<float, T, CAST_TRAIT_B16_TO_B32>(currF, ring, pregLoop);
        Mul(state0F, currF, weight0F, pregLoop);
        DataCopy(state0FAddr + j * REGBASE_VECTOR_LENGTH, state0F, pregLoop);
    }
}

template <typename T>
static __simd_vf__ inline void AdvanceFnLocalPartialsWidthThree(
    __ubuf__ T *ringAddr, __ubuf__ float *weight0FAddr, __ubuf__ float *weight1FAddr,
    __ubuf__ float *state0FAddr, __ubuf__ float *state1FAddr, uint32_t dataCount, uint16_t colLoopTimes)
{
    RegTensor<T> ring;
    RegTensor<float> currF;
    RegTensor<float> weight0F;
    RegTensor<float> weight1F;
    RegTensor<float> state0F;
    RegTensor<float> state1F;
    MaskReg pregLoop;
    for (uint16_t j = 0; j < colLoopTimes; ++j) {
        pregLoop = UpdateMask<float>(dataCount);
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(ring, ringAddr + j * REGBASE_VECTOR_LENGTH);
        DataCopy(state1F, state1FAddr + j * REGBASE_VECTOR_LENGTH);
        Cast<float, T, CAST_TRAIT_B16_TO_B32>(currF, ring, pregLoop);
        DataCopy(weight1F, weight1FAddr + j * REGBASE_VECTOR_LENGTH);
        Mul(state0F, currF, weight1F, pregLoop);
        DataCopy(weight0F, weight0FAddr + j * REGBASE_VECTOR_LENGTH);
        Add(state0F, state0F, state1F, pregLoop);
        Mul(state1F, currF, weight0F, pregLoop);
        DataCopy(state0FAddr + j * REGBASE_VECTOR_LENGTH, state0F, pregLoop);
        DataCopy(state1FAddr + j * REGBASE_VECTOR_LENGTH, state1F, pregLoop);
    }
}

template <typename T>
static __simd_vf__ inline void AdvanceFnLocalPartialsWidthFour(
    __ubuf__ T *ringAddr, __ubuf__ float *weight0FAddr, __ubuf__ float *weight1FAddr,
    __ubuf__ float *weight2FAddr, __ubuf__ float *state0FAddr, __ubuf__ float *state1FAddr,
    __ubuf__ float *state2FAddr, uint32_t dataCount, uint16_t colLoopTimes)
{
    RegTensor<T> ring;
    RegTensor<float> currF;
    RegTensor<float> weight0F;
    RegTensor<float> weight1F;
    RegTensor<float> weight2F;
    RegTensor<float> state0F;
    RegTensor<float> state1F;
    RegTensor<float> state2F;
    MaskReg pregLoop;
    for (uint16_t j = 0; j < colLoopTimes; ++j) {
        pregLoop = UpdateMask<float>(dataCount);
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(ring, ringAddr + j * REGBASE_VECTOR_LENGTH);
        DataCopy(state1F, state1FAddr + j * REGBASE_VECTOR_LENGTH);
        DataCopy(state2F, state2FAddr + j * REGBASE_VECTOR_LENGTH);
        Cast<float, T, CAST_TRAIT_B16_TO_B32>(currF, ring, pregLoop);
        DataCopy(weight2F, weight2FAddr + j * REGBASE_VECTOR_LENGTH);
        Mul(state0F, currF, weight2F, pregLoop);
        DataCopy(weight1F, weight1FAddr + j * REGBASE_VECTOR_LENGTH);
        Add(state0F, state0F, state1F, pregLoop);
        Mul(state1F, currF, weight1F, pregLoop);
        DataCopy(weight0F, weight0FAddr + j * REGBASE_VECTOR_LENGTH);
        Add(state1F, state1F, state2F, pregLoop);
        Mul(state2F, currF, weight0F, pregLoop);
        DataCopy(state0FAddr + j * REGBASE_VECTOR_LENGTH, state0F, pregLoop);
        DataCopy(state1FAddr + j * REGBASE_VECTOR_LENGTH, state1F, pregLoop);
        DataCopy(state2FAddr + j * REGBASE_VECTOR_LENGTH, state2F, pregLoop);
    }
}

template <typename T, int32_t kTemplateWidth>
__aicore__ inline void AdvanceFnLocalPartialsRegbase(LocalTensor<T> ring, LocalTensor<float> weightF,
                                                     LocalTensor<float> state0F, LocalTensor<float> state1F,
                                                     LocalTensor<float> state2F, uint32_t dataCount,
                                                     uint32_t weightStep)
{
    uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(dataCount, REGBASE_VECTOR_LENGTH));

    __ubuf__ T *ringAddr = (__ubuf__ T *)ring.GetPhyAddr();
    __ubuf__ float *weight0FAddr = (__ubuf__ float *)weightF.GetPhyAddr();
    __ubuf__ float *state0FAddr = (__ubuf__ float *)state0F.GetPhyAddr();
    if constexpr (kTemplateWidth == 2) {
        AscendC::VF_CALL<AdvanceFnLocalPartialsWidthTwo<T>>(ringAddr, weight0FAddr, state0FAddr, dataCount,
                                                            colLoopTimes);
    } else if constexpr (kTemplateWidth == 3) {
        __ubuf__ float *weight1FAddr = weight0FAddr + weightStep;
        __ubuf__ float *state1FAddr = (__ubuf__ float *)state1F.GetPhyAddr();
        AscendC::VF_CALL<AdvanceFnLocalPartialsWidthThree<T>>(
            ringAddr, weight0FAddr, weight1FAddr, state0FAddr, state1FAddr, dataCount, colLoopTimes);
    } else if constexpr (kTemplateWidth == 4) {
        __ubuf__ float *weight1FAddr = weight0FAddr + weightStep;
        __ubuf__ float *weight2FAddr = weight1FAddr + weightStep;
        __ubuf__ float *state1FAddr = (__ubuf__ float *)state1F.GetPhyAddr();
        __ubuf__ float *state2FAddr = (__ubuf__ float *)state2F.GetPhyAddr();
        AscendC::VF_CALL<AdvanceFnLocalPartialsWidthFour<T>>(
            ringAddr, weight0FAddr, weight1FAddr, weight2FAddr, state0FAddr, state1FAddr, state2FAddr, dataCount,
            colLoopTimes);
    }
}

}

#endif
