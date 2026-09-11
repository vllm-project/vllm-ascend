/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#pragma once

#ifndef CATLASS_ARCH
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define CATLASS_ARCH 3510
#else
#define CATLASS_ARCH 2201
#endif
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "../chunk_kda_fwd_varlen.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#ifndef FLA_NPU_REGBASE_HPP_INCLUDED
#define FLA_NPU_REGBASE_HPP_INCLUDED
#include "kernel_utils/vector/regbase.hpp"
#endif
#endif
#include "tla/layout.hpp"
#include "tla/tensor.hpp"
#include "chunk_kda_fwd_post_wu.h"

using namespace AscendC;

namespace KdaPrepare {
namespace {
using KdaInt64 = tla::Int<64>;
using KdaInt128 = tla::Int<128>;
constexpr float LN2 = 0.69314718055994530942f;
constexpr float RCP_LN2 = 1.44269504088896340736f;
constexpr float KDA_EXP2_CLAMP = 80.0f;
constexpr float KDA_EXP_INPUT_MAX = KDA_EXP2_CLAMP * LN2;
constexpr float KDA_EXP_INPUT_MIN = -KDA_EXP2_CLAMP * LN2;
constexpr float KDA_SCORE_EXP2_CLAMP = 120.0f;
constexpr float KDA_SCORE_EXP2_MIN_CLAMP = 126.0f;
constexpr float KDA_SCORE_EXP_INPUT_MAX = KDA_SCORE_EXP2_CLAMP * LN2;
constexpr float KDA_SCORE_EXP_INPUT_MIN = -KDA_SCORE_EXP2_MIN_CLAMP * LN2;
constexpr float KDA_FP16_MAX = 65504.0f;
constexpr uint32_t EXP2_UB_ELEMENTS = 256;
constexpr uint32_t EXP2_UB_BYTES = EXP2_UB_ELEMENTS * (sizeof(float) + sizeof(uint16_t));
constexpr uint32_t EXP2_EVENT_ID = 0;
constexpr uint32_t KDA_SOLVE_BT = 64;
constexpr uint32_t KDA_SOLVE_MATRIX_ELEMENTS = KDA_SOLVE_BT * KDA_SOLVE_BT;
constexpr uint32_t KDA_SOLVE_SCRATCH_X = 0;
constexpr uint32_t KDA_SOLVE_SCRATCH_Y0 = 1;
constexpr uint32_t KDA_SOLVE_SCRATCH_TMP = 2;
constexpr uint32_t KDA_SOLVE_SCRATCH_Y1 = 3;
constexpr uint32_t KDA_SOLVE_SCRATCH_IDENTITY = 4;
constexpr uint32_t KDA_SOLVE_SCRATCH_RAW_AKK = KDA_SOLVE_SCRATCH_Y1;
constexpr uint32_t KDA_SOLVE_SCRATCH_RAW_AQK = KDA_SOLVE_SCRATCH_IDENTITY;
constexpr uint32_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint32_t KDA_SOLVE_PIPELINE_DEPTH = 4;
constexpr uint32_t KDA_SOLVE_DIAG_BT = 16;
constexpr uint32_t KDA_SOLVE_DIAG_BLOCKS = KDA_SOLVE_BT / KDA_SOLVE_DIAG_BT;
constexpr uint32_t KDA_SOLVE_DIAG_MCH_ITERS = 3;
// Keep the local safe-gate exponent span within the BF16 score range while
// reducing repeated gate-factor work and AIV/AIC handshakes.
constexpr uint32_t KDA_SCORE_REF_BC = 32;
constexpr uint32_t KDA_SAFE_SCORE_REF_BC = 32;
constexpr uint32_t KDA_VEC_ARENA_ELEMENTS = 32768;
constexpr uint32_t KDA_BITS_PER_MASK_BYTE = 8;
constexpr uint32_t KDA_SELECT_COL_BLOCKS = 2;
constexpr uint32_t KDA_SELECT_COL_MASK_BYTES = KDA_SOLVE_MATRIX_ELEMENTS / KDA_BITS_PER_MASK_BYTE;
constexpr uint32_t KDA_SELECT_MASK_BYTES = KDA_SELECT_COL_BLOCKS * KDA_SELECT_COL_MASK_BYTES;
constexpr uint32_t KDA_SELECT_AQK_MASK_BYTE_OFFSET = 120 * 1024;
constexpr uint32_t KDA_SELECT_AKK_MASK_BYTE_OFFSET = KDA_SELECT_AQK_MASK_BYTE_OFFSET + KDA_SELECT_MASK_BYTES;
constexpr uint32_t KDA_SELECT_ZERO_BYTE_OFFSET = KDA_SELECT_AKK_MASK_BYTE_OFFSET + KDA_SELECT_MASK_BYTES;
constexpr uint32_t KDA_SELECT_ZERO_FLOAT_OFFSET = KDA_SELECT_ZERO_BYTE_OFFSET / sizeof(float);
constexpr uint8_t KDA_SCORE_DONE_FLAG0 = 2;
constexpr uint8_t KDA_SCORE_DONE_FLAG1 = 3;
constexpr uint8_t KDA_SCORE_READY_FLAG0 = 4;
constexpr uint8_t KDA_SCORE_READY_FLAG1 = 5;
constexpr uint8_t KDA_SOLVE_DONE_FLAG = 6;
constexpr uint8_t KDA_SOLVE_READY_FLAG = 7;
constexpr uint8_t KDA_POST_READY_FLAG = 0;
constexpr uint8_t KDA_POST_FREE_FLAG = 1;
constexpr uint32_t KDA_SCORE_QUEUE_DEPTH = 2;
constexpr uint32_t KDA_SCORE_LANES = 2;
constexpr uint32_t KDA_POST_QUEUE_DEPTH = 4;
constexpr uint32_t KDA_POST_QUEUE_STORAGE = KDA_POST_QUEUE_DEPTH + 1;
constexpr uint32_t KDA_DIRECT_SCORE_QUEUE_DEPTH = 2;
constexpr uint32_t KDA_DIRECT_SCORE_ROWS = 32;
constexpr uint32_t KDA_DIRECT_SCORE_MATRIX_ELEMENTS = KDA_DIRECT_SCORE_ROWS * 64;
constexpr uint32_t KDA_DIRECT_SCORE_SLOT_ELEMENTS = 3 * KDA_DIRECT_SCORE_MATRIX_ELEMENTS;
constexpr uint32_t KDA_DIRECT_SCORE_FLOAT_OFFSET = 20 * 1024;
constexpr uint32_t KDA_DIRECT_SCORE_UB_BYTE_OFFSET =
    EXP2_UB_BYTES + KDA_DIRECT_SCORE_FLOAT_OFFSET * sizeof(float);
constexpr uint32_t KDA_DIRECT_SCORE_L1_SLOT_ELEMENTS = 2 * KDA_DIRECT_SCORE_ROWS * 128;
constexpr uint32_t KDA_DIRECT_SCORE_L1_SLOT_BYTES =
    KDA_DIRECT_SCORE_L1_SLOT_ELEMENTS * sizeof(uint16_t);
constexpr uint32_t KDA_DIRECT_SCORE_L1_B_OFFSET =
    KDA_SCORE_QUEUE_DEPTH * KDA_SCORE_LANES * KDA_DIRECT_SCORE_L1_SLOT_BYTES;
constexpr uint64_t KDA_DIRECT_SCORE_FREE_FLAG = 8;
constexpr uint64_t KDA_DIRECT_SCORE_READY_FLAG = 10;
constexpr uint64_t KDA_DIRECT_SCORE_SUBBLOCK_FLAG_STRIDE = 16;
constexpr uint32_t KDA_SCORE_SCRATCH_SLOTS = KDA_SCORE_QUEUE_DEPTH * KDA_SCORE_LANES;
constexpr uint32_t KDA_SYNC_REVERSE_DEPTH = 1;
constexpr uint32_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint32_t KDA_SCORE_SCRATCH_QG = 0;
constexpr uint32_t KDA_SCORE_SCRATCH_W = 1;
constexpr uint32_t KDA_SCORE_SCRATCH_KG = 2;
constexpr uint64_t KDA_WORKSPACE_ALIGN = 512;
constexpr uint32_t KDA_GATE_TILE_ROWS = 16;
constexpr uint32_t KDA_GATE_PIPELINE_DEPTH = 3;
constexpr uint32_t KDA_AIV_UB_BUDGET_BYTES = 192 * 1024;
constexpr uint32_t KDA_LOCAL_GK_FLOAT_OFFSET = 10 * 1024;
constexpr uint32_t KDA_SCALED_QG_FLOAT_OFFSET = 18 * 1024;
constexpr bool KDA_ARCH35_ENABLE_HEAD_PAIR = true;
constexpr bool KDA_ARCH35_ENABLE_MANUAL_SCORE_PIPELINE = false;
constexpr bool KDA_ARCH35_ENABLE_DIRECT_SCORE_UB = true;
constexpr bool KDA_ARCH35_ENABLE_DIRECT_SCORE_L1 = false;
constexpr uint16_t KDA_ARCH35_SCORE_EVENT = 3;
constexpr uint16_t KDA_ARCH35_SCORE_W_EVENT = 4;
constexpr uint16_t KDA_ARCH35_SOLVE_FIX_EVENT = 7;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <bool HAS_BIAS>
static __simd_vf__ inline void AccumulateRawSafeGateChunk128Regbase(
    __ubuf__ float *input, __ubuf__ float *bias, __ubuf__ float *acc,
    uint16_t rows, float expA, float lowerBound)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FLOAT_ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint16_t ROW_ELEMENTS = 2 * FLOAT_ELEMENTS_PER_REG;

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> accZeroReg;
    RegTensor<float> accOneReg;
    RegTensor<float> oneZeroReg;
    RegTensor<float> oneOneReg;
    RegTensor<float> biasZeroReg;
    RegTensor<float> biasOneReg;
    LoadAlign<float, LoadDist::DIST_NORM>(accZeroReg, acc);
    LoadAlign<float, LoadDist::DIST_NORM>(accOneReg, acc + FLOAT_ELEMENTS_PER_REG);
    Duplicate(oneZeroReg, 1.0f, floatMask);
    Duplicate(oneOneReg, 1.0f, floatMask);
    if constexpr (HAS_BIAS) {
        LoadAlign<float, LoadDist::DIST_NORM>(biasZeroReg, bias);
        LoadAlign<float, LoadDist::DIST_NORM>(biasOneReg, bias + FLOAT_ELEMENTS_PER_REG);
    }

    const float gateScale = lowerBound * RCP_LN2;
    for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t rowOffset = static_cast<uint32_t>(row) * ROW_ELEMENTS;
        RegTensor<float> gateZeroReg;
        RegTensor<float> gateOneReg;
        RegTensor<float> sigmoidZeroReg;
        RegTensor<float> sigmoidOneReg;
        LoadAlign<float, LoadDist::DIST_NORM>(gateZeroReg, input + rowOffset);
        LoadAlign<float, LoadDist::DIST_NORM>(gateOneReg, input + rowOffset + FLOAT_ELEMENTS_PER_REG);
        if constexpr (HAS_BIAS) {
            Add(gateZeroReg, gateZeroReg, biasZeroReg, floatMask);
            Add(gateOneReg, gateOneReg, biasOneReg, floatMask);
        }
        Muls(gateZeroReg, gateZeroReg, -expA, floatMask);
        Muls(gateOneReg, gateOneReg, -expA, floatMask);
        Exp(gateZeroReg, gateZeroReg, floatMask);
        Exp(gateOneReg, gateOneReg, floatMask);
        Adds(gateZeroReg, gateZeroReg, 1.0f, floatMask);
        Adds(gateOneReg, gateOneReg, 1.0f, floatMask);
        Div(sigmoidZeroReg, oneZeroReg, gateZeroReg, floatMask);
        Div(sigmoidOneReg, oneOneReg, gateOneReg, floatMask);
        Muls(sigmoidZeroReg, sigmoidZeroReg, gateScale, floatMask);
        Muls(sigmoidOneReg, sigmoidOneReg, gateScale, floatMask);
        Add(accZeroReg, accZeroReg, sigmoidZeroReg, floatMask);
        Add(accOneReg, accOneReg, sigmoidOneReg, floatMask);
        StoreAlign(input + rowOffset, accZeroReg, floatMask);
        StoreAlign(input + rowOffset + FLOAT_ELEMENTS_PER_REG, accOneReg, floatMask);
    }
    StoreAlign(acc, accZeroReg, floatMask);
    StoreAlign(acc + FLOAT_ELEMENTS_PER_REG, accOneReg, floatMask);
}

template <typename InputT>
__simd_callee__ inline void LoadKdaGateRegbasePair(
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    __ubuf__ InputT *src,
    AscendC::MicroAPI::MaskReg &inputMask)
{
    using namespace AscendC::MicroAPI;
    if constexpr (std::is_same<InputT, float>()) {
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zeroReg, oneReg, src);
    } else {
        RegTensor<InputT> inputReg;
        LoadIn<InputT, false>(inputReg, src);
        CastHalf2Float<InputT>(zeroReg, oneReg, inputReg, inputMask);
    }
}

template <typename OutputT>
__simd_callee__ inline void ClampKdaGateRegbaseOutput(
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    if constexpr (std::is_same<OutputT, half>()) {
        Mins(zeroReg, zeroReg, KDA_FP16_MAX, floatMask);
        Mins(oneReg, oneReg, KDA_FP16_MAX, floatMask);
        Maxs(zeroReg, zeroReg, -KDA_FP16_MAX, floatMask);
        Maxs(oneReg, oneReg, -KDA_FP16_MAX, floatMask);
    }
}

template <typename OutputT, bool USE_REF, bool NEGATIVE>
__simd_callee__ inline void BuildKdaGateRegbaseExp(
    AscendC::MicroAPI::RegTensor<float> &expZeroReg,
    AscendC::MicroAPI::RegTensor<float> &expOneReg,
    AscendC::MicroAPI::RegTensor<float> &gateZeroReg,
    AscendC::MicroAPI::RegTensor<float> &gateOneReg,
    __ubuf__ float *ref,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    constexpr float expInputMax =
        std::is_same<OutputT, bfloat16_t>() ? KDA_SCORE_EXP_INPUT_MAX : KDA_EXP_INPUT_MAX;
    constexpr float expInputMin =
        std::is_same<OutputT, bfloat16_t>() ? KDA_SCORE_EXP_INPUT_MIN : KDA_EXP_INPUT_MIN;
    if constexpr (USE_REF) {
        RegTensor<float> refZeroReg;
        RegTensor<float> refOneReg;
        LoadAlign<float, LoadDist::DIST_DINTLV_B32>(refZeroReg, refOneReg, ref);
        if constexpr (NEGATIVE) {
            SubFloatTwoReg(expZeroReg, expOneReg, refZeroReg, refOneReg,
                           gateZeroReg, gateOneReg, floatMask);
        } else {
            SubFloatTwoReg(expZeroReg, expOneReg, gateZeroReg, gateOneReg,
                           refZeroReg, refOneReg, floatMask);
        }
    } else if constexpr (NEGATIVE) {
        Muls(expZeroReg, gateZeroReg, -1.0f, floatMask);
        Muls(expOneReg, gateOneReg, -1.0f, floatMask);
    } else {
        Adds(expZeroReg, gateZeroReg, 0.0f, floatMask);
        Adds(expOneReg, gateOneReg, 0.0f, floatMask);
    }
    Muls(expZeroReg, expZeroReg, LN2, floatMask);
    Muls(expOneReg, expOneReg, LN2, floatMask);
    MinsFloatTwoReg(expZeroReg, expOneReg, expZeroReg, expOneReg,
                    expInputMax, floatMask);
    Maxs(expZeroReg, expZeroReg, expInputMin, floatMask);
    Maxs(expOneReg, expOneReg, expInputMin, floatMask);
    ExpFloatTwoReg(expZeroReg, expOneReg, expZeroReg, expOneReg, floatMask);
}

template <typename OutputT>
__simd_callee__ inline void StoreKdaGateRegbasePair(
    __ubuf__ OutputT *dst,
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::MaskReg &inputMask,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    RegTensor<OutputT> outputReg;
    ClampKdaGateRegbaseOutput<OutputT>(zeroReg, oneReg, floatMask);
    CastFloat2Half<OutputT>(outputReg, zeroReg, oneReg, floatMask);
    StoreAlign(dst, outputReg, inputMask);
}

template <typename InputT, typename OutputT, typename GK_T, bool USE_REF>
static __simd_vf__ inline void PrepareKdaGateQwRegbase(
    __ubuf__ InputT *q, __ubuf__ InputT *k, __ubuf__ OutputT *qOut,
    __ubuf__ OutputT *kOut, __ubuf__ InputT *qDirect, __ubuf__ InputT *kDirect,
    __ubuf__ GK_T *gate, __ubuf__ float *ref, uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(InputT);

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<InputT>(activeCount);
            uint32_t offset = rowOffset + col;

            RegTensor<float> gateZeroReg;
            RegTensor<float> gateOneReg;
            RegTensor<float> expZeroReg;
            RegTensor<float> expOneReg;
            RegTensor<float> directZeroReg;
            RegTensor<float> directOneReg;
            RegTensor<float> inputZeroReg;
            RegTensor<float> inputOneReg;
            RegTensor<float> outputZeroReg;
            RegTensor<float> outputOneReg;

            LoadKdaGateRegbasePair<GK_T>(gateZeroReg, gateOneReg, gate + offset, inputMask);
            BuildKdaGateRegbaseExp<OutputT, USE_REF, false>(
                expZeroReg, expOneReg, gateZeroReg, gateOneReg, ref + col, floatMask);
            BuildKdaGateRegbaseExp<InputT, false, false>(
                directZeroReg, directOneReg, gateZeroReg, gateOneReg, ref + col, floatMask);

            LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, q + offset, inputMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           directZeroReg, directOneReg, floatMask);
            StoreKdaGateRegbasePair<InputT>(
                qDirect + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           expZeroReg, expOneReg, floatMask);
            StoreKdaGateRegbasePair<OutputT>(
                qOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);

            LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, k + offset, inputMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           directZeroReg, directOneReg, floatMask);
            StoreKdaGateRegbasePair<InputT>(
                kDirect + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           expZeroReg, expOneReg, floatMask);
            StoreKdaGateRegbasePair<OutputT>(
                kOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
        }
    }
}

template <typename InputT, typename OutputT, typename GK_T, bool USE_REF>
static __simd_vf__ inline void PrepareKdaGateKgRegbase(
    __ubuf__ OutputT *kg, __ubuf__ InputT *k, __ubuf__ GK_T *gate,
    __ubuf__ float *ref, uint16_t rows, uint16_t cols, uint16_t validRows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(InputT);

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<InputT>(activeCount);
            uint32_t offset = rowOffset + col;

            RegTensor<float> gateZeroReg;
            RegTensor<float> gateOneReg;
            RegTensor<float> expZeroReg;
            RegTensor<float> expOneReg;
            RegTensor<float> inputZeroReg;
            RegTensor<float> inputOneReg;
            RegTensor<float> outputZeroReg;
            RegTensor<float> outputOneReg;

            LoadKdaGateRegbasePair<GK_T>(gateZeroReg, gateOneReg, gate + offset, inputMask);
            BuildKdaGateRegbaseExp<OutputT, USE_REF, true>(
                expZeroReg, expOneReg, gateZeroReg, gateOneReg, ref + col, floatMask);
            LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, k + offset, inputMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           expZeroReg, expOneReg, floatMask);
            if constexpr (USE_REF) {
                if (row >= validRows) {
                    Duplicate(outputZeroReg, 0.0f, floatMask);
                    Duplicate(outputOneReg, 0.0f, floatMask);
                }
            }
            StoreKdaGateRegbasePair<OutputT>(
                kg + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
        }
    }
}

template <typename InputT, typename OutputT, typename GK_T, bool USE_REF, bool STORE_DIRECT,
          bool EXPORT_FINAL_KG, bool SCALE_SCORE_W = false>
static __simd_vf__ inline void PrepareKdaGateQwKgRegbase(
    __ubuf__ InputT *q, __ubuf__ InputT *k, __ubuf__ OutputT *qOut,
    __ubuf__ OutputT *wOut, __ubuf__ OutputT *kgOut, __ubuf__ InputT *qDirect,
    __ubuf__ InputT *wDirect, __ubuf__ InputT *v, __ubuf__ InputT *vDirect,
    __ubuf__ InputT *finalKgOut, __ubuf__ float *beta, __ubuf__ GK_T *gate,
    __ubuf__ float *ref, __ubuf__ float *finalRef,
    uint16_t rows, uint16_t cols, uint16_t validRows)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(InputT);
    constexpr float scoreExpInputMax =
        std::is_same<OutputT, bfloat16_t>() ? KDA_SCORE_EXP_INPUT_MAX : KDA_EXP_INPUT_MAX;
    constexpr float scoreExpInputMin =
        std::is_same<OutputT, bfloat16_t>() ? KDA_SCORE_EXP_INPUT_MIN : KDA_EXP_INPUT_MIN;

    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<InputT>(activeCount);
            uint32_t offset = rowOffset + col;

            RegTensor<float> gateZeroReg;
            RegTensor<float> gateOneReg;
            RegTensor<float> posZeroReg;
            RegTensor<float> posOneReg;
            RegTensor<float> negZeroReg;
            RegTensor<float> negOneReg;
            RegTensor<float> directZeroReg;
            RegTensor<float> directOneReg;
            RegTensor<float> finalNegZeroReg;
            RegTensor<float> finalNegOneReg;
            RegTensor<float> finalKZeroReg;
            RegTensor<float> finalKOneReg;
            RegTensor<float> inputZeroReg;
            RegTensor<float> inputOneReg;
            RegTensor<float> outputZeroReg;
            RegTensor<float> outputOneReg;
            RegTensor<float> betaReg;

            LoadKdaGateRegbasePair<GK_T>(gateZeroReg, gateOneReg, gate + offset, inputMask);
            if constexpr (USE_REF) {
                RegTensor<float> refZeroReg;
                RegTensor<float> refOneReg;
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(refZeroReg, refOneReg, ref + col);
                SubFloatTwoReg(posZeroReg, posOneReg, gateZeroReg, gateOneReg,
                               refZeroReg, refOneReg, floatMask);
                SubFloatTwoReg(negZeroReg, negOneReg, refZeroReg, refOneReg,
                               gateZeroReg, gateOneReg, floatMask);
            } else {
                Adds(posZeroReg, gateZeroReg, 0.0f, floatMask);
                Adds(posOneReg, gateOneReg, 0.0f, floatMask);
                Muls(negZeroReg, gateZeroReg, -1.0f, floatMask);
                Muls(negOneReg, gateOneReg, -1.0f, floatMask);
            }
            if constexpr (STORE_DIRECT) {
                Adds(directZeroReg, gateZeroReg, 0.0f, floatMask);
                Adds(directOneReg, gateOneReg, 0.0f, floatMask);
                Muls(directZeroReg, directZeroReg, LN2, floatMask);
                Muls(directOneReg, directOneReg, LN2, floatMask);
                MinsFloatTwoReg(directZeroReg, directOneReg, directZeroReg, directOneReg,
                                KDA_EXP_INPUT_MAX, floatMask);
                Maxs(directZeroReg, directZeroReg, KDA_EXP_INPUT_MIN, floatMask);
                Maxs(directOneReg, directOneReg, KDA_EXP_INPUT_MIN, floatMask);
                ExpFloatTwoReg(directZeroReg, directOneReg, directZeroReg, directOneReg, floatMask);
            }
            Muls(posZeroReg, posZeroReg, LN2, floatMask);
            Muls(posOneReg, posOneReg, LN2, floatMask);
            Muls(negZeroReg, negZeroReg, LN2, floatMask);
            Muls(negOneReg, negOneReg, LN2, floatMask);
            MinsFloatTwoReg(posZeroReg, posOneReg, posZeroReg, posOneReg,
                            scoreExpInputMax, floatMask);
            MinsFloatTwoReg(negZeroReg, negOneReg, negZeroReg, negOneReg,
                            scoreExpInputMax, floatMask);
            Maxs(posZeroReg, posZeroReg, scoreExpInputMin, floatMask);
            Maxs(posOneReg, posOneReg, scoreExpInputMin, floatMask);
            Maxs(negZeroReg, negZeroReg, scoreExpInputMin, floatMask);
            Maxs(negOneReg, negOneReg, scoreExpInputMin, floatMask);
            ExpFloatTwoReg(posZeroReg, posOneReg, posZeroReg, posOneReg, floatMask);
            ExpFloatTwoReg(negZeroReg, negOneReg, negZeroReg, negOneReg, floatMask);

            LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, q + offset, inputMask);
            if constexpr (STORE_DIRECT) {
                MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                               directZeroReg, directOneReg, floatMask);
                StoreKdaGateRegbasePair<InputT>(
                    qDirect + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            }
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           posZeroReg, posOneReg, floatMask);
            StoreKdaGateRegbasePair<OutputT>(
                qOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);

            LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, k + offset, inputMask);
            if constexpr (EXPORT_FINAL_KG) {
                // wOut may alias k, and STORE_DIRECT later reuses inputZeroReg/inputOneReg for V.
                Adds(finalKZeroReg, inputZeroReg, 0.0f, floatMask);
                Adds(finalKOneReg, inputOneReg, 0.0f, floatMask);
            }
            if constexpr (STORE_DIRECT || SCALE_SCORE_W) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(betaReg, beta + row);
            }
            if constexpr (STORE_DIRECT) {
                MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                               directZeroReg, directOneReg, floatMask);
                RegTensor<InputT> roundedReg;
                ClampKdaGateRegbaseOutput<InputT>(outputZeroReg, outputOneReg, floatMask);
                CastFloat2Half<InputT>(roundedReg, outputZeroReg, outputOneReg, floatMask);
                CastHalf2Float<InputT>(outputZeroReg, outputOneReg, roundedReg, inputMask);
                Mul(outputZeroReg, outputZeroReg, betaReg, floatMask);
                Mul(outputOneReg, outputOneReg, betaReg, floatMask);
                StoreKdaGateRegbasePair<InputT>(
                    wDirect + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            }
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           posZeroReg, posOneReg, floatMask);
            if constexpr (SCALE_SCORE_W) {
                Mul(outputZeroReg, outputZeroReg, betaReg, floatMask);
                Mul(outputOneReg, outputOneReg, betaReg, floatMask);
            }
            StoreKdaGateRegbasePair<OutputT>(
                wOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           negZeroReg, negOneReg, floatMask);
            if constexpr (USE_REF) {
                if (row >= validRows) {
                    Duplicate(outputZeroReg, 0.0f, floatMask);
                    Duplicate(outputOneReg, 0.0f, floatMask);
                }
            }
            StoreKdaGateRegbasePair<OutputT>(
                kgOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);

            if constexpr (STORE_DIRECT) {
                LoadKdaGateRegbasePair<InputT>(inputZeroReg, inputOneReg, v + offset, inputMask);
                Mul(outputZeroReg, inputZeroReg, betaReg, floatMask);
                Mul(outputOneReg, inputOneReg, betaReg, floatMask);
                StoreKdaGateRegbasePair<InputT>(
                    vDirect + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            }
            if constexpr (EXPORT_FINAL_KG) {
                RegTensor<float> finalRefZeroReg;
                RegTensor<float> finalRefOneReg;
                LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                    finalRefZeroReg, finalRefOneReg, finalRef + col);
                SubFloatTwoReg(finalNegZeroReg, finalNegOneReg,
                               finalRefZeroReg, finalRefOneReg,
                               gateZeroReg, gateOneReg, floatMask);
                Muls(finalNegZeroReg, finalNegZeroReg, LN2, floatMask);
                Muls(finalNegOneReg, finalNegOneReg, LN2, floatMask);
                MinsFloatTwoReg(finalNegZeroReg, finalNegOneReg,
                                finalNegZeroReg, finalNegOneReg,
                                KDA_EXP_INPUT_MAX, floatMask);
                Maxs(finalNegZeroReg, finalNegZeroReg, KDA_EXP_INPUT_MIN, floatMask);
                Maxs(finalNegOneReg, finalNegOneReg, KDA_EXP_INPUT_MIN, floatMask);
                ExpFloatTwoReg(finalNegZeroReg, finalNegOneReg,
                               finalNegZeroReg, finalNegOneReg, floatMask);
                MulFloatTwoReg(outputZeroReg, outputOneReg, finalKZeroReg, finalKOneReg,
                               finalNegZeroReg, finalNegOneReg, floatMask);
                StoreKdaGateRegbasePair<InputT>(
                    finalKgOut + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
            }
        }
    }
}

static __simd_vf__ inline void ForwardSubDiag16Regbase(__ubuf__ float *diag, uint16_t valid)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t DIAG_SIZE = KDA_SOLVE_DIAG_BT;
    uint32_t activeCount = DIAG_SIZE;
    MaskReg rowMask = UpdateMask<float>(activeCount);

    for (uint16_t row = 2; row < valid; ++row) {
        RegTensor<float> currentReg;
        RegTensor<float> scaleReg;
        RegTensor<float> matrixReg;
        RegTensor<float> productReg;
        RegTensor<float> sumReg;
        LoadAlign(currentReg, diag + static_cast<uint32_t>(row) * DIAG_SIZE);
        Duplicate(sumReg, 0.0f, rowMask);

        for (uint16_t sourceRow = 0; sourceRow < row; ++sourceRow) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(
                scaleReg, diag + static_cast<uint32_t>(row) * DIAG_SIZE + sourceRow);
            LoadAlign(matrixReg, diag + static_cast<uint32_t>(sourceRow) * DIAG_SIZE);
            Mul(productReg, matrixReg, scaleReg, rowMask);
            Add(sumReg, sumReg, productReg, rowMask);
        }
        Add(currentReg, currentReg, sumReg, rowMask);
        StoreAlign(diag + static_cast<uint32_t>(row) * DIAG_SIZE, currentReg, rowMask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

static __simd_vf__ inline void SelectCausalRows64Regbase(
    __ubuf__ float *aqk, __ubuf__ float *akk, uint16_t rowBegin, uint16_t rowCount)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ROW_ELEMENTS = 64;

    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t localRow = 0; localRow < rowCount; ++localRow) {
        const uint16_t row = rowBegin + localRow;
        const uint32_t rowOffset = static_cast<uint32_t>(localRow) * ROW_ELEMENTS;
        RegTensor<float> zeroReg;
        RegTensor<float> aqkInputReg;
        RegTensor<float> akkInputReg;
        RegTensor<float> aqkReg;
        RegTensor<float> akkReg;
        uint32_t aqkCount = static_cast<uint32_t>(row) + 1;
        uint32_t akkCount = static_cast<uint32_t>(row);
        MaskReg aqkMask = UpdateMask<float>(aqkCount);
        MaskReg akkMask = UpdateMask<float>(akkCount);
        Duplicate(zeroReg, 0.0f, fullMask);
        LoadAlign(aqkInputReg, aqk + rowOffset);
        LoadAlign(akkInputReg, akk + rowOffset);
        Select(aqkReg, aqkInputReg, zeroReg, aqkMask);
        Select(akkReg, akkInputReg, zeroReg, akkMask);
        StoreAlign(aqk + rowOffset, aqkReg, fullMask);
        StoreAlign(akk + rowOffset, akkReg, fullMask);
    }
}

static __simd_vf__ inline void ForwardSubDiag16StridedRegbase(
    __ubuf__ float *matrix, uint16_t rowStride, uint16_t rowBegin, uint16_t colBegin,
    uint16_t valid)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t DIAG_SIZE = KDA_SOLVE_DIAG_BT;
    uint32_t activeCount = DIAG_SIZE;
    MaskReg rowMask = UpdateMask<float>(activeCount);

    for (uint16_t row = 2; row < valid; ++row) {
        uint32_t currentOffset =
            static_cast<uint32_t>(rowBegin + row) * rowStride + colBegin;
        RegTensor<float> currentReg;
        RegTensor<float> scaleReg;
        RegTensor<float> matrixReg;
        RegTensor<float> productReg;
        RegTensor<float> sumReg;
        LoadAlign(currentReg, matrix + currentOffset);
        Duplicate(sumReg, 0.0f, rowMask);

        for (uint16_t sourceRow = 0; sourceRow < row; ++sourceRow) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(
                scaleReg, matrix + currentOffset + sourceRow);
            uint32_t sourceOffset =
                static_cast<uint32_t>(rowBegin + sourceRow) * rowStride + colBegin;
            LoadAlign(matrixReg, matrix + sourceOffset);
            Mul(productReg, matrixReg, scaleReg, rowMask);
            Add(sumReg, sumReg, productReg, rowMask);
        }
        Add(currentReg, currentReg, sumReg, rowMask);
        StoreAlign(matrix + currentOffset, currentReg, rowMask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}

static __simd_vf__ inline void ForwardSubDiag16PairStridedRegbase(
    __ubuf__ float *matrix, uint16_t rowStride, uint16_t colBegin,
    uint16_t firstValid, uint16_t secondValid)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t DIAG_SIZE = KDA_SOLVE_DIAG_BT;
    constexpr uint16_t SECOND_LOCAL_ROW = DIAG_SIZE;
    uint32_t activeCount = DIAG_SIZE;
    MaskReg rowMask = UpdateMask<float>(activeCount);

    for (uint16_t row = 2; row < DIAG_SIZE; ++row) {
        uint32_t firstCurrentOffset =
            static_cast<uint32_t>(row) * rowStride + colBegin;
        uint32_t secondCurrentOffset =
            static_cast<uint32_t>(SECOND_LOCAL_ROW + row) * rowStride +
            colBegin + DIAG_SIZE;
        RegTensor<float> firstCurrentReg;
        RegTensor<float> secondCurrentReg;
        RegTensor<float> firstSumReg;
        RegTensor<float> secondSumReg;
        LoadAlign(firstCurrentReg, matrix + firstCurrentOffset);
        LoadAlign(secondCurrentReg, matrix + secondCurrentOffset);
        Duplicate(firstSumReg, 0.0f, rowMask);
        Duplicate(secondSumReg, 0.0f, rowMask);

        if (row < firstValid || row < secondValid) {
            for (uint16_t sourceRow = 0; sourceRow < row; ++sourceRow) {
                RegTensor<float> firstScaleReg;
                RegTensor<float> secondScaleReg;
                RegTensor<float> firstMatrixReg;
                RegTensor<float> secondMatrixReg;
                RegTensor<float> firstProductReg;
                RegTensor<float> secondProductReg;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(
                    firstScaleReg, matrix + firstCurrentOffset + sourceRow);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(
                    secondScaleReg, matrix + secondCurrentOffset + sourceRow);
                uint32_t firstSourceOffset =
                    static_cast<uint32_t>(sourceRow) * rowStride + colBegin;
                uint32_t secondSourceOffset =
                    static_cast<uint32_t>(SECOND_LOCAL_ROW + sourceRow) * rowStride +
                    colBegin + DIAG_SIZE;
                LoadAlign(firstMatrixReg, matrix + firstSourceOffset);
                LoadAlign(secondMatrixReg, matrix + secondSourceOffset);
                Mul(firstProductReg, firstMatrixReg, firstScaleReg, rowMask);
                Mul(secondProductReg, secondMatrixReg, secondScaleReg, rowMask);
                Add(firstSumReg, firstSumReg, firstProductReg, rowMask);
                Add(secondSumReg, secondSumReg, secondProductReg, rowMask);
            }
        }
        if (row < firstValid) {
            Add(firstCurrentReg, firstCurrentReg, firstSumReg, rowMask);
            StoreAlign(matrix + firstCurrentOffset, firstCurrentReg, rowMask);
        }
        if (row < secondValid) {
            Add(secondCurrentReg, secondCurrentReg, secondSumReg, rowMask);
            StoreAlign(matrix + secondCurrentOffset, secondCurrentReg, rowMask);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    RegTensor<int32_t> indexReg;
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(indexReg, 0);
    for (uint16_t row = 0; row < DIAG_SIZE; ++row) {
        MaskReg diagMask;
        CompareScalar<int32_t, CMPMODE::EQ>(
            diagMask, indexReg, static_cast<int32_t>(row), rowMask);
        uint32_t firstOffset = static_cast<uint32_t>(row) * rowStride + colBegin;
        uint32_t secondOffset =
            static_cast<uint32_t>(SECOND_LOCAL_ROW + row) * rowStride +
            colBegin + DIAG_SIZE;
        RegTensor<float> firstReg;
        RegTensor<float> secondReg;
        LoadAlign(firstReg, matrix + firstOffset);
        LoadAlign(secondReg, matrix + secondOffset);
        Adds(firstReg, firstReg, 1.0f, diagMask);
        Adds(secondReg, secondReg, 1.0f, diagMask);
        StoreAlign(matrix + firstOffset, firstReg, rowMask);
        StoreAlign(matrix + secondOffset, secondReg, rowMask);
    }
}

static __simd_vf__ inline void ApplyKdaRowScaleRegbase(
    __ubuf__ float *matrix, __ubuf__ float *rowScale, uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t FP32_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<float> matrixReg0;
    RegTensor<float> matrixReg1;
    RegTensor<float> scaleReg0;
    RegTensor<float> scaleReg1;

    uint16_t row = 0;
    for (; row + 1 < rows; row += 2) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg1, rowScale + row + 1);
        for (uint16_t col = 0; col < cols; col += FP32_PER_REG) {
            uint32_t activeCount0 = static_cast<uint32_t>(cols - col);
            uint32_t activeCount1 = activeCount0;
            MaskReg mask0 = UpdateMask<float>(activeCount0);
            MaskReg mask1 = UpdateMask<float>(activeCount1);
            uint32_t offset0 = static_cast<uint32_t>(row) * cols + col;
            uint32_t offset1 = static_cast<uint32_t>(row + 1) * cols + col;
            LoadAlign(matrixReg0, matrix + offset0);
            LoadAlign(matrixReg1, matrix + offset1);
            Mul(matrixReg0, matrixReg0, scaleReg0, mask0);
            Mul(matrixReg1, matrixReg1, scaleReg1, mask1);
            StoreAlign(matrix + offset0, matrixReg0, mask0);
            StoreAlign(matrix + offset1, matrixReg1, mask1);
        }
    }
    if (row < rows) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg0, rowScale + row);
        for (uint16_t col = 0; col < cols; col += FP32_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg mask = UpdateMask<float>(activeCount);
            uint32_t offset = static_cast<uint32_t>(row) * cols + col;
            LoadAlign(matrixReg0, matrix + offset);
            Mul(matrixReg0, matrixReg0, scaleReg0, mask);
            StoreAlign(matrix + offset, matrixReg0, mask);
        }
    }
}
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
using KdaArchTag = Catlass::Arch::Ascend950;
#else
using KdaArchTag = Catlass::Arch::AtlasA2;
#endif
using KdaDispatchPolicy = Catlass::Gemm::MmadPingpong<KdaArchTag, true, false>;
using KdaScoreDispatchPolicy =
    Catlass::Gemm::MmadPingpongTlaMulti<KdaArchTag, true, false, 1, true, 2, 1, 2, 2>;
static_assert(KdaScoreDispatchPolicy::ENABLE_L1_RESIDENT,
              "KDA Aqk/Akk score MMAD must keep the shared right matrix resident in L1");
static_assert(KdaScoreDispatchPolicy::L1B_STAGES == 1,
              "KDA Aqk/Akk score MMAD needs one L1 B slot so the second MMAD reuses it");
using KdaSolveDispatchPolicy = Catlass::Gemm::MmadPingpong<KdaArchTag, true, false>;
static_assert(!KdaSolveDispatchPolicy::USE_HF32_MODE, "KDA triangular solve must use IEEE FP32 Cube mode");
using KdaL1TileShape = tla::Shape<KdaInt64, KdaInt128, KdaInt128>;
using KdaL0TileShape = KdaL1TileShape;
using KdaSolveL1TileShape = tla::Shape<KdaInt64, KdaInt64, KdaInt64>;
using KdaSolveL0TileShape = KdaSolveL1TileShape;

__aicore__ inline uint32_t FloatToBits(float value)
{
    union Bits {
        __aicore__ Bits() {}
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

__aicore__ inline float BitsToFloat(uint32_t value)
{
    union Bits {
        __aicore__ Bits() {}
        uint32_t u;
        float f;
    } bits;
    bits.u = value;
    return bits.f;
}

__aicore__ inline uint16_t Bf16ToBits(bfloat16_t value)
{
    union Bits {
        __aicore__ Bits() {}
        bfloat16_t f;
        uint16_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

__aicore__ inline bfloat16_t BitsToBf16(uint16_t value)
{
    union Bits {
        __aicore__ Bits() {}
        uint16_t u;
        bfloat16_t f;
    } bits;
    bits.u = value;
    return bits.f;
}

template <typename T>
__aicore__ inline T FloatToType(float value)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        uint32_t bits = FloatToBits(value);
        uint32_t bias = 0x7FFFu + ((bits >> 16) & 1u);
        return BitsToBf16(static_cast<uint16_t>((bits + bias) >> 16));
    }
    return static_cast<T>(value);
}

template <bool SAFE_GATE, typename T, typename GK_T = float, typename BETA_T = float,
          uint32_t COMPILE_BT = 0, uint32_t COMPILE_K = 0, uint32_t COMPILE_V = 0>
class ChunkKdaFwdPrepareKernel {
public:
    using OUT_T = T;
    using AKK_T = float;
    using SCORE_T =
        std::conditional_t<SAFE_GATE && IsSameType<T, half>::value, bfloat16_t, T>;
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR rawG,
                                GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR beta, GM_ADDR initialState,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR preparedQG, GM_ADDR preparedAqk,
                                GM_ADDR propagatedVNew, GM_ADDR propagatedH, GM_ADDR o, GM_ADDR finalState, GM_ADDR aqk,
                                GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
                                GM_ADDR finalKg, GM_ADDR workspace, const TilingData &tiling, TPipe *pipe,
                                bool initVecBuffers = true, bool storeQG = true)
    {
        pipe_ = pipe;
        q_.SetGlobalBuffer((__gm__ T *)q);
        k_.SetGlobalBuffer((__gm__ T *)k);
        v_.SetGlobalBuffer((__gm__ T *)v);
        gk_.SetGlobalBuffer((__gm__ GK_T *)gk);
        rawG_.SetGlobalBuffer((__gm__ float *)rawG);
        aLog_.SetGlobalBuffer((__gm__ float *)aLog);
        dtBias_.SetGlobalBuffer((__gm__ float *)dtBias);
        beta_.SetGlobalBuffer((__gm__ BETA_T *)beta);
        if (initialState != nullptr) {
            initialState_.SetGlobalBuffer((__gm__ float *)initialState);
        }
        cuSeqlensAddr_ = reinterpret_cast<__gm__ int64_t *>(cuSeqlens);
        if (preparedQG != nullptr) {
            preparedQG_.SetGlobalBuffer((__gm__ T *)preparedQG);
        }
        if (preparedAqk != nullptr) {
            preparedAqk_.SetGlobalBuffer((__gm__ T *)preparedAqk);
        }
        if (propagatedVNew != nullptr) {
            propagatedVNew_.SetGlobalBuffer((__gm__ T *)propagatedVNew);
        }
        if (propagatedH != nullptr) {
            propagatedH_.SetGlobalBuffer((__gm__ T *)propagatedH);
        }
        chunkIndicesAddr_ = reinterpret_cast<__gm__ int64_t *>(chunkIndices);
        o_.SetGlobalBuffer((__gm__ OUT_T *)o);
        finalState_.SetGlobalBuffer((__gm__ float *)finalState);
        aqk_.SetGlobalBuffer((__gm__ float *)aqk);
        akk_.SetGlobalBuffer((__gm__ AKK_T *)akk);
        w_.SetGlobalBuffer((__gm__ T *)w);
        u_.SetGlobalBuffer((__gm__ OUT_T *)u);
        qg_.SetGlobalBuffer((__gm__ T *)qg);
        kg_.SetGlobalBuffer((__gm__ T *)kg);
        vNew_.SetGlobalBuffer((__gm__ T *)vNew);
        h_.SetGlobalBuffer((__gm__ float *)h);
        finalKg_.SetGlobalBuffer((__gm__ T *)finalKg);
        solveWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);

        B_ = tiling.batch;
        N_ = tiling.seqNum;
        H_ = tiling.qHeadNum;
        HV_ = tiling.vHeadNum;
        T_ = tiling.seqlen;
        K_ = COMPILE_K == 0 ? tiling.kHeadDim : COMPILE_K;
        V_ = COMPILE_V == 0 ? tiling.vHeadDim : COMPILE_V;
        BT_ = COMPILE_BT == 0 ? tiling.chunkSize : COMPILE_BT;
        NT_ = tiling.totalChunks;
        scale_ = tiling.scale;
        hasInitial_ = tiling.hasInitialState;
        isVarLen_ = tiling.isVarLen;
        inputSequenceMajor_ = tiling.inputSequenceMajor;
        fusePostWu_ = tiling.fusePostWu;
        materializeFinalKg_ = tiling.fusePostWu || tiling.fusePostWuIntoFwdH;
        computeGateInPrepare_ = tiling.computeGateInPrepare;
        hasALog_ = tiling.hasALog;
        hasDtBias_ = tiling.hasDtBias;
        lowerBound_ = tiling.lowerBound;
        storeQG_ = storeQG;
        usedCoreNum_ = tiling.prepareUsedCoreNum;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && !IsSameType<T, float>::value) {
            headPairMode_ = KDA_ARCH35_ENABLE_HEAD_PAIR &&
                HV_ % KDA_SCORE_LANES == 0;
        }
#endif
        constexpr uint64_t solvePipelineDepth = SAFE_GATE ? KDA_SOLVE_PIPELINE_DEPTH : 1;
        const uint64_t solveBytes =
            usedCoreNum_ * solvePipelineDepth * KDA_SOLVE_SCRATCH_SLOTS * BT_ * BT_ * sizeof(float);
        const uint64_t alignedSolveBytes =
            (solveBytes + KDA_WORKSPACE_ALIGN - 1) / KDA_WORKSPACE_ALIGN * KDA_WORKSPACE_ALIGN;
        scoreWorkspace_.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + alignedSolveBytes));
        if ASCEND_IS_AIV {
            uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
            solveCoreIdx_ = subBlockNum == 0 ? 0 : static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        } else {
            solveCoreIdx_ = static_cast<uint64_t>(GetBlockIdx());
        }
        if (pipe_ != nullptr && initVecBuffers) {
            pipe_->InitBuffer(exp2Buf_, EXP2_UB_BYTES);
            pipe_->InitBuffer(vecBuf_, KDA_VEC_ARENA_ELEMENTS * sizeof(float));
            const uint64_t gateStageElems = GatePipelineRows() * K_;
            const uint64_t gateInputSlotBytes = GateInputSlotBytes();
            const uint64_t gatePipelineBytes =
                GateBufferDepth() * (gateInputSlotBytes + gateStageElems * sizeof(T));
            pipe_->InitBuffer(gateWritebackBuf_, static_cast<uint32_t>(gatePipelineBytes));
            AllocVectorEvents();
        }
    }
    __aicore__ inline void ProcessAivOnly()
    {
        isAivOnly_ = true;
        ProcessPreAiv();
        ReleaseVectorEvents();
    }

    __aicore__ inline void ProcessAiv()
    {
        ProcessPreAiv();
        ReleaseVectorEvents();
    }

    __aicore__ inline void ProcessAic()
    {
        ProcessPreAic();
    }

    template <typename PostWuOp>
    __aicore__ inline void ProcessAicFused(PostWuOp &postWu)
    {
        ProcessPreAicHeadPairFused(postWu);
    }

private:
    __aicore__ inline void AllocVectorEvents()
    {
        mte2ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte2Event_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        mte3ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
        mte2ToMte3Event_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        vToSEvent_ = pipe_->AllocEventID<HardEvent::V_S>();
        for (uint32_t slot = 0; slot < KDA_GATE_PIPELINE_DEPTH; ++slot) {
            mte3ToMte2Events_[slot] = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
        }
        vectorEventsAllocated_ = true;
    }

    __aicore__ inline void ReleaseVectorEvents()
    {
        if (!vectorEventsAllocated_) {
            return;
        }
        pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::V_MTE2>(vToMte2Event_);
        pipe_->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Event_);
        pipe_->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVEvent_);
        pipe_->ReleaseEventID<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        pipe_->ReleaseEventID<HardEvent::V_S>(vToSEvent_);
        for (uint32_t slot = 0; slot < KDA_GATE_PIPELINE_DEPTH; ++slot) {
            pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
        }
        vectorEventsAllocated_ = false;
    }

    __aicore__ inline uint64_t QOffset(uint64_t b, uint64_t h, uint64_t t, uint64_t d) const
    {
        if (inputSequenceMajor_) {
            return ((b * T_ + t) * H_ + h) * K_ + d;
        }
        return ((b * H_ + h) * T_ + t) * K_ + d;
    }

    __aicore__ inline uint64_t VInputOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d) const
    {
        if (inputSequenceMajor_) {
            return ((b * T_ + t) * HV_ + hv) * V_ + d;
        }
        return ((b * HV_ + hv) * T_ + t) * V_ + d;
    }

    __aicore__ inline uint64_t RawGateOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d) const
    {
        if (inputSequenceMajor_) {
            return ((b * T_ + t) * HV_ + hv) * K_ + d;
        }
        return ((b * HV_ + hv) * T_ + t) * K_ + d;
    }

    __aicore__ inline uint64_t KVOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d, uint64_t dim) const
    {
        return ((b * HV_ + hv) * T_ + t) * dim + d;
    }

    __aicore__ inline uint64_t BetaOffset(uint64_t b, uint64_t hv, uint64_t t) const
    {
        return (b * HV_ + hv) * T_ + t;
    }

    __aicore__ inline uint64_t AOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t j) const
    {
        return ((b * HV_ + hv) * T_ + t) * BT_ + j;
    }

    __aicore__ inline uint64_t HOffset(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t d, uint64_t r) const
    {
        return (((b * HV_ + hv) * NT_ + chunkIdx) * K_ + d) * V_ + r;
    }

    __aicore__ inline uint64_t WScratchOffset(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t t, uint64_t d) const
    {
        return (((b * HV_ + hv) * NT_ + chunkIdx) * BT_ + t) * K_ + d;
    }

    __aicore__ inline uint64_t SolveScratchOffset(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                  uint64_t slot) const
    {
        (void)b;
        (void)hv;
        (void)chunkIdx;
        constexpr uint64_t solvePipelineDepth = SAFE_GATE ? KDA_SOLVE_PIPELINE_DEPTH : 1;
        uint64_t matrixElements = BT_ * BT_;
        return ((solveCoreIdx_ * solvePipelineDepth + activeSolveSlot_) * KDA_SOLVE_SCRATCH_SLOTS + slot) *
               matrixElements;
    }

    __aicore__ inline uint64_t ScoreScratchOffset(uint64_t slot, uint64_t plane, uint64_t t = 0,
                                                  uint64_t d = 0) const
    {
        return (((solveCoreIdx_ * KDA_SCORE_SCRATCH_SLOTS + slot) * KDA_SCORE_SCRATCH_PLANES + plane) * BT_ + t) *
                   K_ +
               d;
    }

    __aicore__ inline uint64_t ScoreScratchSlot(uint64_t queueSlot, uint64_t lane, bool pairHeads) const
    {
        return pairHeads ? queueSlot * KDA_SCORE_LANES + lane : queueSlot;
    }



    __aicore__ inline uint64_t ScoreRefBlockSize() const
    {
        if constexpr (SAFE_GATE) {
            return KDA_SAFE_SCORE_REF_BC;
        }
        return KDA_SCORE_REF_BC;
    }

    __aicore__ inline uint64_t ScoreRowBlockCount(uint64_t curT, uint64_t rowBegin) const
    {
        uint64_t blockSize = ScoreRefBlockSize();
        uint64_t rowCount = curT - rowBegin;
        if (rowCount > blockSize) {
            rowCount = blockSize;
        }
        return rowCount;
    }

    __aicore__ inline uint64_t ScoreRefToken(uint64_t start, uint64_t curT, uint64_t rowBegin,
                                             uint64_t rowCount) const
    {
        uint64_t ref = rowBegin + rowCount / 2;
        if (ref >= curT) {
            ref = curT - 1;
        }
        return start + ref;
    }

    __aicore__ inline void RunExp2(LocalTensor<float> &tensor, uint32_t count)
    {
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        ClampExpInput(tensor, count);
        Exp(tensor, tensor, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
    }

    __aicore__ inline void ClampExpInput(LocalTensor<float> &tensor, uint32_t count)
    {
        Mins(tensor, tensor, KDA_EXP_INPUT_MAX, count);
        PipeBarrier<PIPE_V>();
        Maxs(tensor, tensor, KDA_EXP_INPUT_MIN, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ClampScoreExpInput(LocalTensor<float> &tensor, uint32_t count)
    {
        constexpr float expInputMax =
            IsSameType<SCORE_T, bfloat16_t>::value ? KDA_SCORE_EXP_INPUT_MAX : KDA_EXP_INPUT_MAX;
        constexpr float expInputMin =
            IsSameType<SCORE_T, bfloat16_t>::value ? KDA_SCORE_EXP_INPUT_MIN : KDA_EXP_INPUT_MIN;
        Mins(tensor, tensor, expInputMax, count);
        PipeBarrier<PIPE_V>();
        Maxs(tensor, tensor, expInputMin, count);
        PipeBarrier<PIPE_V>();
    }

    template <typename OutputT>
    __aicore__ inline void ClampFp32ForCast(LocalTensor<float> &tensor, uint32_t count)
    {
        if constexpr (IsSameType<OutputT, half>::value) {
            Mins(tensor, tensor, KDA_FP16_MAX, count);
            PipeBarrier<PIPE_V>();
            Maxs(tensor, tensor, -KDA_FP16_MAX, count);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ClampFp32ToOutputType(LocalTensor<float> &tensor, uint32_t count)
    {
        ClampFp32ForCast<T>(tensor, count);
    }

    template <typename CopyT>
    __aicore__ inline void CopyVectorIn(LocalTensor<CopyT> &dst, GlobalTensor<CopyT> &src, uint64_t offset,
                                        uint64_t count)
    {
        uint64_t rowBytes = count * static_cast<uint64_t>(sizeof(CopyT));
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst, src[offset], static_cast<uint32_t>(count));
            return;
        }
        DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[offset], params, padParams);
    }

    template <typename CopyT>
    __aicore__ inline void CopyRowsIn(LocalTensor<CopyT> &dst, GlobalTensor<CopyT> &src,
                                      uint64_t offset, uint64_t rows, uint64_t cols,
                                      uint64_t rowStride)
    {
        if (rows == 0 || cols == 0) {
            return;
        }
        if (rowStride == cols) {
            CopyVectorIn(dst, src, offset, rows * cols);
            return;
        }
        DataCopyExtParams params{
            static_cast<uint16_t>(rows),
            static_cast<uint32_t>(cols * sizeof(CopyT)),
            static_cast<uint32_t>((rowStride - cols) * sizeof(CopyT)),
            0,
            0};
        DataCopyPadExtParams<CopyT> padParams{false, 0, 0, 0};
        DataCopyPad(dst, src[offset], params, padParams);
    }

    template <typename CopyT>
    __aicore__ inline void CopyVectorOut(GlobalTensor<CopyT> &dst, uint64_t offset, LocalTensor<CopyT> &src,
                                         uint64_t count)
    {
        uint64_t rowBytes = count * static_cast<uint64_t>(sizeof(CopyT));
        if (rowBytes >= 32 && rowBytes % 32 == 0) {
            DataCopy(dst[offset], src, static_cast<uint32_t>(count));
            return;
        }
        DataCopyParams params{1, static_cast<uint16_t>(rowBytes), 0, 0};
        DataCopyPad(dst[offset], src, params);
    }

    template <typename CopyT>
    __aicore__ inline void CopyRowIn(LocalTensor<CopyT> &dst, GlobalTensor<CopyT> &src, uint64_t offset)
    {
        CopyVectorIn(dst, src, offset, K_);
    }

    template <typename CopyT>
    __aicore__ inline void CopyRowOut(GlobalTensor<CopyT> &dst, uint64_t offset, LocalTensor<CopyT> &src)
    {
        CopyVectorOut(dst, offset, src, K_);
    }

    __aicore__ inline LocalTensor<float> VecScratch(uint64_t slot)
    {
        return vecBuf_.Get<float>()[slot * EXP2_UB_ELEMENTS];
    }

    __aicore__ inline uint64_t GateStageElems() const
    {
        return GatePipelineRows() * K_;
    }

    __aicore__ inline uint64_t GatePipelineRows() const
    {
        constexpr uint64_t fixedBytes =
            static_cast<uint64_t>(KDA_VEC_ARENA_ELEMENTS) * sizeof(float) + EXP2_UB_BYTES;
        constexpr uint64_t availableBytes = KDA_AIV_UB_BUDGET_BYTES - fixedBytes;
        uint64_t bytesPerRow = K_ * KDA_GATE_PIPELINE_DEPTH * (3 * sizeof(T) + sizeof(GK_T));
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            bytesPerRow = GateBufferDepth() *
                          (K_ * (4 * sizeof(T) + sizeof(GK_T)) + sizeof(BETA_T));
        }
#endif
        uint64_t rows = bytesPerRow == 0 ? 0 : availableBytes / bytesPerRow;
        return rows < KDA_GATE_TILE_ROWS ? rows : KDA_GATE_TILE_ROWS;
    }

    __aicore__ inline constexpr uint64_t GateBufferDepth() const
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            return 2;
        }
#endif
        return KDA_GATE_PIPELINE_DEPTH;
    }

    __aicore__ inline uint64_t GateInputSlotBytes() const
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            return GateStageElems() * (3 * sizeof(T) + sizeof(GK_T)) +
                   GatePipelineRows() * sizeof(BETA_T);
        }
#endif
        return GateStageElems() * (2 * sizeof(T) + sizeof(GK_T));
    }

    __aicore__ inline LocalTensor<T> GateQTyped(uint64_t slot)
    {
        uint64_t byteOffset = slot * GateInputSlotBytes();
        return gateWritebackBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline LocalTensor<T> GateKTyped(uint64_t slot)
    {
        uint64_t byteOffset = slot * GateInputSlotBytes() + GateStageElems() * sizeof(T);
        return gateWritebackBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline LocalTensor<GK_T> GateGTyped(uint64_t slot)
    {
        uint64_t byteOffset = slot * GateInputSlotBytes() + 2 * GateStageElems() * sizeof(T);
        return gateWritebackBuf_.Get<GK_T>()[byteOffset / sizeof(GK_T)];
    }

    __aicore__ inline LocalTensor<T> GateVTyped(uint64_t slot)
    {
        uint64_t byteOffset = slot * GateInputSlotBytes() +
                              GateStageElems() * (2 * sizeof(T) + sizeof(GK_T));
        return gateWritebackBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline LocalTensor<BETA_T> GateBetaTyped(uint64_t slot)
    {
        uint64_t byteOffset = slot * GateInputSlotBytes() +
                              GateStageElems() * (3 * sizeof(T) + sizeof(GK_T));
        return gateWritebackBuf_.Get<BETA_T>()[byteOffset / sizeof(BETA_T)];
    }

    __aicore__ inline LocalTensor<T> GateKgTyped(uint64_t slot)
    {
        uint64_t byteOffset = GateBufferDepth() * GateInputSlotBytes() +
                              slot * GateStageElems() * sizeof(T);
        return gateWritebackBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline LocalTensor<GK_T> LocalGateChunk()
    {
        constexpr uint64_t byteOffset =
            static_cast<uint64_t>(KDA_LOCAL_GK_FLOAT_OFFSET) * sizeof(float);
        return vecBuf_.Get<GK_T>()[byteOffset / sizeof(GK_T)];
    }

    __aicore__ inline LocalTensor<GK_T> GateScoreTyped(uint64_t slot, uint64_t tileRow)
    {
        if constexpr (IsSameType<GK_T, float>::value) {
            if (computeGateInPrepare_) {
                return LocalGateChunk()[tileRow * K_];
            }
        }
        return GateGTyped(slot);
    }

    __aicore__ inline void LoadGateScoreRef(
        LocalTensor<float> dst, uint64_t b, uint64_t hv, uint64_t token)
    {
        if constexpr (IsSameType<GK_T, float>::value) {
            if (computeGateInPrepare_) {
                const uint64_t tileRow = token - activeGateChunkStart_;
                Adds(dst, LocalGateChunk()[tileRow * K_], 0.0f, static_cast<uint32_t>(K_));
                PipeBarrier<PIPE_V>();
                return;
            }
        }
        LoadAsFloatRow(gk_, KVOffset(b, hv, token, 0, K_), dst, K_);
    }

    __aicore__ inline void PrefetchQKGate(uint64_t slot, uint64_t b, uint64_t h, uint64_t hv,
                                          uint64_t token, uint64_t elems)
    {
        const uint64_t rows = elems / K_;
        LocalTensor<T> qTyped = GateQTyped(slot);
        LocalTensor<T> kTyped = GateKTyped(slot);
        LocalTensor<GK_T> gateTyped = GateGTyped(slot);
        CopyRowsIn(qTyped, q_, QOffset(b, h, token, 0), rows, K_, inputSequenceMajor_ ? H_ * K_ : K_);
        CopyRowsIn(kTyped, k_, QOffset(b, h, token, 0), rows, K_, inputSequenceMajor_ ? H_ * K_ : K_);
        if (!computeGateInPrepare_) {
            CopyVectorIn(gateTyped, gk_, KVOffset(b, hv, token, 0, K_), elems);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            LocalTensor<T> vTyped = GateVTyped(slot);
            LocalTensor<BETA_T> betaTyped = GateBetaTyped(slot);
            CopyRowsIn(vTyped, v_, VInputOffset(b, hv, token, 0), rows, V_,
                       inputSequenceMajor_ ? HV_ * V_ : V_);
            CopyVectorIn(betaTyped, beta_, BetaOffset(b, hv, token), rows);
        }
#endif
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline float LoadGateExpA(uint64_t hv)
    {
        if (!hasALog_) {
            return 1.0f;
        }
        LocalTensor<float> scalar = exp2Buf_.Get<float>();
        CopyVectorIn(scalar, aLog_, hv, 1);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        Exp(scalar, scalar, 1);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(vToSEvent_);
        WaitFlag<HardEvent::V_S>(vToSEvent_);
        __ubuf__ float *ptr = (__ubuf__ float *)scalar.GetPhyAddr();
        return ptr[0];
    }

    __aicore__ inline void PrefetchRawGateTile(
        uint64_t slot, uint64_t b, uint64_t hv, uint64_t token, uint64_t rows)
    {
        (void)slot;
        const uint64_t tileRow = token - activeGateChunkStart_;
        LocalTensor<float> gate =
            LocalGateChunk().template ReinterpretCast<float>()[tileRow * K_];
        CopyRowsIn(gate, rawG_, RawGateOffset(b, hv, token, 0), rows, K_,
                   inputSequenceMajor_ ? HV_ * K_ : K_);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline void MaterializeRawGateChunkArch35(
        uint64_t b, uint64_t hv, uint64_t start, uint64_t rows)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (!computeGateInPrepare_) {
            return;
        }
        if constexpr (!IsSameType<GK_T, float>::value) {
            return;
        } else {
            activeGateChunkStart_ = start;
            const float expA = LoadGateExpA(hv);
            LocalTensor<float> acc = exp2Buf_.Get<float>();
            LocalTensor<float> bias = exp2Buf_.Get<float>()[K_];
            if (hasDtBias_) {
                CopyVectorIn(bias, dtBias_, hv * K_, K_);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            }
            Duplicate(acc, 0.0f, static_cast<uint32_t>(K_));
            PipeBarrier<PIPE_V>();

            const uint64_t tileRows = GatePipelineRows();
            const uint64_t tileCount = (rows + tileRows - 1) / tileRows;
            uint64_t currentRows = rows < tileRows ? rows : tileRows;
            PrefetchRawGateTile(0, b, hv, start, currentRows);
            for (uint64_t tile = 0; tile < tileCount; ++tile) {
                const uint64_t slot = tile & 1;
                const uint64_t tileRow = tile * tileRows;
                currentRows = rows - tileRow;
                if (currentRows > tileRows) {
                    currentRows = tileRows;
                }
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

                const uint64_t nextTile = tile + 1;
                if (nextTile < tileCount) {
                    const uint64_t nextSlot = nextTile & 1;
                    uint64_t nextRows = rows - nextTile * tileRows;
                    if (nextRows > tileRows) {
                        nextRows = tileRows;
                    }
                    PrefetchRawGateTile(nextSlot, b, hv, start + nextTile * tileRows, nextRows);
                }

                LocalTensor<float> gate =
                    LocalGateChunk().template ReinterpretCast<float>()[tileRow * K_];
                if (hasDtBias_) {
                    AccumulateRawSafeGateChunk128Regbase<true>(
                        (__ubuf__ float *)gate.GetPhyAddr(), (__ubuf__ float *)bias.GetPhyAddr(),
                        (__ubuf__ float *)acc.GetPhyAddr(), static_cast<uint16_t>(currentRows),
                        expA, lowerBound_);
                } else {
                    AccumulateRawSafeGateChunk128Regbase<false>(
                        (__ubuf__ float *)gate.GetPhyAddr(), (__ubuf__ float *)bias.GetPhyAddr(),
                        (__ubuf__ float *)acc.GetPhyAddr(), static_cast<uint16_t>(currentRows),
                        expA, lowerBound_);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
                WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
                CopyVectorOut(gk_, KVOffset(b, hv, start + tileRow, 0, K_), gate,
                              currentRows * K_);
            }
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }
#else
        (void)b;
        (void)hv;
        (void)start;
        (void)rows;
#endif
    }

    __aicore__ inline LocalTensor<T> GateDirectQ(uint64_t slot)
    {
        return vecBuf_.Get<T>()[slot * 3 * GateStageElems()];
    }

    __aicore__ inline LocalTensor<T> GateDirectW(uint64_t slot)
    {
        return GateDirectQ(slot)[GateStageElems()];
    }

    __aicore__ inline LocalTensor<T> GateDirectV(uint64_t slot)
    {
        return GateDirectQ(slot)[2 * GateStageElems()];
    }

    __aicore__ inline LocalTensor<float> GateBetaFloat(uint64_t slot)
    {
        constexpr uint64_t directBytes =
            KDA_GATE_PIPELINE_DEPTH * 3 * KDA_GATE_TILE_ROWS * COMPILE_K * sizeof(T);
        return vecBuf_.Get<float>()[directBytes / sizeof(float) + slot * KDA_GATE_TILE_ROWS];
    }

    __aicore__ inline void StorePreparedQG(uint64_t b, uint64_t hv, uint64_t token,
                                           LocalTensor<T> directQ, uint64_t elems)
    {
        static_assert(KDA_SCALED_QG_FLOAT_OFFSET + KDA_GATE_TILE_ROWS * 128 <=
                      KDA_DIRECT_SCORE_FLOAT_OFFSET);
        const uint64_t offset = KVOffset(b, hv, token, 0, K_);
        if (storeQG_) {
            CopyVectorOut(qg_, offset, directQ, elems);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }
        LocalTensor<float> scaledQG = vecBuf_.Get<float>()[KDA_SCALED_QG_FLOAT_OFFSET];
        Cast(scaledQG, directQ, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        Muls(scaledQG, scaledQG, scale_, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        ClampFp32ToOutputType(scaledQG, static_cast<uint32_t>(elems));
        Cast(directQ, scaledQG, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        CopyVectorOut(kg_, offset, directQ, elems);
    }

    __aicore__ inline void PrefetchKGate(uint64_t slot, uint64_t b, uint64_t h, uint64_t hv,
                                         uint64_t token, uint64_t elems)
    {
        const uint64_t rows = elems / K_;
        LocalTensor<T> kTyped = GateQTyped(slot);
        LocalTensor<GK_T> gateTyped = GateGTyped(slot);
        CopyRowsIn(kTyped, k_, QOffset(b, h, token, 0), rows, K_, inputSequenceMajor_ ? H_ * K_ : K_);
        if (!computeGateInPrepare_) {
            CopyVectorIn(gateTyped, gk_, KVOffset(b, hv, token, 0, K_), elems);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline void WaitGateInputReady()
    {
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline void WaitGateOutputForMte2(uint64_t slot = 0)
    {
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
    }

    __aicore__ inline void WaitGateOutputForVector()
    {
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void SignalGateOutputDone()
    {
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void SignalGateOutputDoneForMte2(uint64_t slot)
    {
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[slot]);
    }

    template <typename CopyT>
    __aicore__ inline void LoadAsFloatRow(GlobalTensor<CopyT> &src, uint64_t srcOffset, LocalTensor<float> &dst,
                                          uint64_t count)
    {
        if constexpr (IsSameType<CopyT, float>::value) {
            CopyVectorIn(dst, src, srcOffset, count);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Adds(dst, dst, 0.0f, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
            WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        } else {
            constexpr uint32_t typedOffset = EXP2_UB_ELEMENTS * sizeof(float) / sizeof(CopyT);
            LocalTensor<CopyT> rowLocal = exp2Buf_.Get<CopyT>()[typedOffset];
            CopyVectorIn(rowLocal, src, srcOffset, count);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(dst, rowLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
            WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        }
        PipeBarrier<PIPE_V>();
    }

    template <typename CopyT>
    __aicore__ inline void LoadAsFloatVector(GlobalTensor<CopyT> &src, uint64_t srcOffset,
                                              LocalTensor<float> &dst, LocalTensor<CopyT> &typedScratch,
                                              uint64_t count)
    {
        if constexpr (IsSameType<CopyT, float>::value) {
            CopyVectorIn(dst, src, srcOffset, count);
        } else {
            CopyVectorIn(typedScratch, src, srcOffset, count);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        if constexpr (!IsSameType<CopyT, float>::value) {
            Cast(dst, typedScratch, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
        }
    }

    template <typename CopyT>
    __aicore__ inline void StoreFloatRow(GlobalTensor<CopyT> &dst, uint64_t dstOffset, LocalTensor<float> &src,
                                         uint64_t count)
    {
        if constexpr (IsSameType<CopyT, float>::value) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(dst, dstOffset, src, count);
        } else {
            constexpr uint32_t typedOffset = EXP2_UB_ELEMENTS * sizeof(float) / sizeof(CopyT);
            LocalTensor<CopyT> rowLocal = exp2Buf_.Get<CopyT>()[typedOffset];
            Cast(rowLocal, src, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(dst, dstOffset, rowLocal, count);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }





    __aicore__ inline LocalTensor<float> Exp2NegG(uint64_t b, uint64_t hv, uint64_t t)
    {
        LocalTensor<float> exp2Local = exp2Buf_.Get<float>();
        LoadAsFloatRow(gk_, KVOffset(b, hv, t, 0, K_), exp2Local, K_);
        Muls(exp2Local, exp2Local, -LN2, static_cast<uint32_t>(K_));
        PipeBarrier<PIPE_V>();
        RunExp2(exp2Local, static_cast<uint32_t>(K_));
        return exp2Local;
    }

    __aicore__ inline void PrepareScoreFactorsBulk(uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
                                                    uint64_t subBlockIdx, uint64_t subBlockNum,
                                                    uint64_t refToken, uint64_t scoreRowBegin,
                                                    uint64_t scoreRowCount, uint64_t validColEnd,
                                                    uint64_t finalRefToken, uint64_t scoreSlot)
    {
        const bool useDirectScoreL1 =
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KDA_ARCH35_ENABLE_DIRECT_SCORE_L1 && KDA_ARCH35_ENABLE_DIRECT_SCORE_UB &&
            SAFE_GATE && BT_ == 64 && K_ == 128 && V_ == 128 &&
            subBlockNum == 1 && scoreRowCount == KDA_DIRECT_SCORE_ROWS &&
            finalRefToken == start + BT_ - 1;
#else
            false;
#endif
        LocalTensor<float> refFp32 = exp2Buf_.Get<float>();
        LoadGateScoreRef(refFp32, b, hv, refToken);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        constexpr bool exportFinalKg =
            SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128;
        LocalTensor<float> finalRefFp32 = exp2Buf_.Get<float>()[K_];
        if constexpr (exportFinalKg) {
            LoadGateScoreRef(finalRefFp32, b, hv, finalRefToken);
        }
#endif

        uint64_t qwBegin = scoreRowBegin + (scoreRowCount * subBlockIdx) / subBlockNum;
        uint64_t qwEnd = scoreRowBegin + (scoreRowCount * (subBlockIdx + 1)) / subBlockNum;
        uint64_t qwMaxRows = GatePipelineRows();
        bool qwOutputPending = false;
        uint64_t qwSlot = 0;
        if (qwBegin < qwEnd && qwMaxRows > 0) {
            uint64_t firstRows = qwEnd - qwBegin;
            if (firstRows > qwMaxRows) {
                firstRows = qwMaxRows;
            }
            PrefetchQKGate(qwSlot, b, h, hv, start + qwBegin, firstRows * K_);
        }
        for (uint64_t tileRow = qwBegin; tileRow < qwEnd && qwMaxRows > 0; tileRow += qwMaxRows) {
            uint64_t tileRows = qwEnd - tileRow;
            if (tileRows > qwMaxRows) {
                tileRows = qwMaxRows;
            }
            uint64_t elems = tileRows * K_;
            LocalTensor<T> qTyped = GateQTyped(qwSlot);
            LocalTensor<T> kTyped = GateKTyped(qwSlot);
            LocalTensor<SCORE_T> qScore = qTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<SCORE_T> kScore = kTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<GK_T> gateTyped = GateScoreTyped(qwSlot, tileRow);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            LocalTensor<SCORE_T> kgScore =
                GateKgTyped(qwSlot).template ReinterpretCast<SCORE_T>();
            LocalTensor<T> vTyped = GateVTyped(qwSlot);
            LocalTensor<BETA_T> betaTyped = GateBetaTyped(qwSlot);
            LocalTensor<T> directQ = GateDirectQ(qwSlot);
            LocalTensor<T> directW = GateDirectW(qwSlot);
            LocalTensor<T> directV = GateDirectV(qwSlot);
            LocalTensor<float> betaFp32 = GateBetaFloat(qwSlot);
#endif
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> qFp32 = arena;
            LocalTensor<float> kFp32 = arena[elems];
            LocalTensor<float> gFp32 = arena[2 * elems];
            LocalTensor<float> expFp32 = arena[3 * elems];
            LocalTensor<float> outFp32 = arena[4 * elems];
#endif

            WaitGateInputReady();
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            Cast(qFp32, qTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            if constexpr (IsSameType<GK_T, float>::value) {
                gFp32 = gateTyped;
            } else {
                Cast(gFp32, gateTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            }
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (IsSameType<BETA_T, float>::value) {
                Adds(betaFp32, betaTyped, 0.0f, static_cast<uint32_t>(tileRows));
            } else {
                Cast(betaFp32, betaTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(tileRows));
            }
            PipeBarrier<PIPE_V>();
#endif
            if (qwOutputPending) {
                WaitGateOutputForMte2();
            }
            uint64_t nextTileRow = tileRow + qwMaxRows;
            if (nextTileRow < qwEnd) {
                uint64_t nextRows = qwEnd - nextTileRow;
                if (nextRows > qwMaxRows) {
                    nextRows = qwMaxRows;
                }
                PrefetchQKGate(qwSlot ^ 1, b, h, hv, start + nextTileRow, nextRows * K_);
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            bool fuseQwKg = SAFE_GATE && BT_ == 64 && K_ == 128 && V_ == 128 && subBlockNum == 1;
            if (fuseQwKg) {
                PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, true, exportFinalKg, true>(
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(qScore.GetPhyAddr()),
                    (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kScore.GetPhyAddr()),
                    (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kgScore.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(directQ.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(directW.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(vTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(directV.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(vTyped.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(betaFp32.GetPhyAddr()),
                    (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(finalRefFp32.GetPhyAddr()),
                    static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_),
                    static_cast<uint16_t>(tileRows));
            } else {
                PrepareKdaGateQwRegbase<T, SCORE_T, GK_T, true>(
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(qScore.GetPhyAddr()),
                    (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kScore.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(directQ.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(directW.GetPhyAddr()),
                    (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                    static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_));
            }
#else
            PipeBarrier<PIPE_V>();
            for (uint64_t row = 0; row < tileRows; ++row) {
                Sub(expFp32[row * K_], gFp32[row * K_], refFp32, static_cast<uint32_t>(K_));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampScoreExpInput(expFp32, static_cast<uint32_t>(elems));
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Mul(outFp32, qFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
            Cast(qScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
            Cast(kScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
#endif

            if (qwOutputPending) {
                WaitGateOutputForVector();
            }
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            if (useDirectScoreL1) {
                Catlass::Arch::Resource<KdaArchTag> resource;
                LocalTensor<SCORE_T> scoreL1 =
                    resource.l1Buf.template GetBufferByByte<SCORE_T>(
                        scoreSlot * KDA_DIRECT_SCORE_L1_SLOT_BYTES);
                const uint64_t localRow = tileRow - scoreRowBegin;
                constexpr uint32_t c0Elements = 16;
                constexpr uint32_t c0Blocks = 128 / c0Elements;
                constexpr uint32_t rowBlockElements = 16 * c0Elements;
                LocalTensor<SCORE_T> nzScratch =
                    resource.ubBuf.template GetBufferByByte<SCORE_T>(
                        KDA_DIRECT_SCORE_UB_BYTE_OFFSET +
                        (scoreSlot / KDA_SCORE_LANES) *
                            KDA_DIRECT_SCORE_SLOT_ELEMENTS * sizeof(float) +
                        2 * KDA_DIRECT_SCORE_MATRIX_ELEMENTS * sizeof(float));
                LocalTensor<SCORE_T> qNz = nzScratch;
                LocalTensor<SCORE_T> kNz = qNz[tileRows * K_];
                constexpr uint8_t srcRepeatStride = 128 * sizeof(SCORE_T) / 32;
                for (uint32_t colBlock = 0; colBlock < c0Blocks; ++colBlock) {
                    const uint64_t srcOffset = colBlock * c0Elements;
                    const uint64_t dstOffset = colBlock * tileRows * c0Elements;
                    Copy(qNz[dstOffset], qScore[srcOffset], c0Elements,
                         static_cast<uint8_t>(tileRows), {1, 1, 1, srcRepeatStride});
                    Copy(kNz[dstOffset], kScore[srcOffset], c0Elements,
                         static_cast<uint8_t>(tileRows), {1, 1, 1, srcRepeatStride});
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
                WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
                const uint64_t qL1Offset = (localRow / 16) * rowBlockElements;
                const uint64_t kL1Offset =
                    ((KDA_DIRECT_SCORE_ROWS + localRow) / 16) * rowBlockElements;
                DataCopyParams nzCopyParams{
                    c0Blocks, static_cast<uint16_t>(tileRows), 0,
                    static_cast<uint16_t>(64 - tileRows)};
                DataCopy(scoreL1[qL1Offset], qNz, nzCopyParams);
                DataCopy(scoreL1[kL1Offset], kNz, nzCopyParams);
            } else {
                CopyVectorOut(scoreWorkspace_,
                              ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG, tileRow),
                              qScore, elems);
                CopyVectorOut(scoreWorkspace_,
                              ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W, tileRow),
                              kScore, elems);
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if (fuseQwKg) {
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG, tileRow),
                              kgScore, elems);
            }
                StorePreparedQG(b, hv, start + tileRow, directQ, elems);
                CopyVectorOut(w_, KVOffset(b, hv, start + tileRow, 0, K_), directW, elems);
                CopyVectorOut(vNew_, KVOffset(b, hv, start + tileRow, 0, V_), directV,
                              tileRows * V_);
                if constexpr (exportFinalKg) {
                    CopyVectorOut(finalKg_, KVOffset(b, hv, start + tileRow, 0, K_), vTyped, elems);
                }
#endif
            SignalGateOutputDone();
            qwOutputPending = true;
            qwSlot ^= 1;
        }
        if (qwOutputPending) {
            WaitGateOutputForMte2();
            WaitGateOutputForVector();
        }

        bool fuseQwKg = false;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        fuseQwKg = SAFE_GATE && BT_ == 64 && K_ == 128 && V_ == 128 && subBlockNum == 1;
#endif
        uint64_t kgRows = fuseQwKg ? scoreRowBegin : validColEnd;
        uint64_t kgBegin = (kgRows * subBlockIdx) / subBlockNum;
        uint64_t kgEnd = (kgRows * (subBlockIdx + 1)) / subBlockNum;
        uint64_t kgMaxRows = GatePipelineRows();
        bool kgOutputPending = false;
        uint64_t kgSlot = 0;
        if (kgBegin < kgEnd && kgMaxRows > 0) {
            uint64_t firstRows = kgEnd - kgBegin;
            if (firstRows > kgMaxRows) {
                firstRows = kgMaxRows;
            }
            PrefetchKGate(kgSlot, b, h, hv, start + kgBegin, firstRows * K_);
        }
        for (uint64_t tileRow = kgBegin; tileRow < kgEnd && kgMaxRows > 0; tileRow += kgMaxRows) {
            uint64_t tileRows = kgEnd - tileRow;
            if (tileRows > kgMaxRows) {
                tileRows = kgMaxRows;
            }
            uint64_t elems = tileRows * K_;
            LocalTensor<T> kTyped = GateQTyped(kgSlot);
            LocalTensor<SCORE_T> kgScore = kTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<GK_T> gateTyped = GateScoreTyped(kgSlot, tileRow);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> kFp32 = arena;
            LocalTensor<float> gFp32 = arena[elems];
            LocalTensor<float> expFp32 = arena[2 * elems];
            LocalTensor<float> outFp32 = arena[3 * elems];
#endif

            WaitGateInputReady();
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            if constexpr (IsSameType<GK_T, float>::value) {
                gFp32 = gateTyped;
            } else {
                Cast(gFp32, gateTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            }
#endif
            if (kgOutputPending) {
                WaitGateOutputForMte2();
            }
            uint64_t nextTileRow = tileRow + kgMaxRows;
            if (nextTileRow < kgEnd) {
                uint64_t nextRows = kgEnd - nextTileRow;
                if (nextRows > kgMaxRows) {
                    nextRows = kgMaxRows;
                }
                PrefetchKGate(kgSlot ^ 1, b, h, hv, start + nextTileRow, nextRows * K_);
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            PrepareKdaGateKgRegbase<T, SCORE_T, GK_T, true>(
                (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kgScore.GetPhyAddr()),
                (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_),
                static_cast<uint16_t>(tileRows));
#else
            PipeBarrier<PIPE_V>();
            for (uint64_t row = 0; row < tileRows; ++row) {
                Sub(expFp32[row * K_], refFp32, gFp32[row * K_], static_cast<uint32_t>(K_));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampScoreExpInput(expFp32, static_cast<uint32_t>(elems));
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
            Cast(kgScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
#endif

            if (kgOutputPending) {
                WaitGateOutputForVector();
            }
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG, tileRow),
                          kgScore, elems);
            SignalGateOutputDone();
            kgOutputPending = true;
            kgSlot ^= 1;
        }
        if (kgOutputPending) {
            WaitGateOutputForMte2();
            WaitGateOutputForVector();
        }
    }

    __aicore__ inline void PrepareGateProductsBulk(uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
                                                   uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum,
                                                   bool useRef, uint64_t refToken, uint64_t validColEnd,
                                                   bool writeScoreScratch, uint64_t scoreSlot)
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum || K_ == 0) {
            return;
        }
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        if (rowBegin >= rowEnd) {
            return;
        }

        uint64_t maxRows = GatePipelineRows();
        if (maxRows == 0) {
            return;
        }
        LocalTensor<float> refFp32 = exp2Buf_.Get<float>();
        if (useRef) {
            LoadGateScoreRef(refFp32, b, hv, refToken);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        LocalTensor<float> finalRefFp32 = exp2Buf_.Get<float>()[K_];
        if (materializeFinalKg_) {
            LoadGateScoreRef(finalRefFp32, b, hv, start + curT - 1);
        }
        const bool fuseScoreWriteback =
            writeScoreScratch && useRef && K_ * 2 <= EXP2_UB_ELEMENTS;
#endif

        bool outputPending = false;
        uint64_t gateSlot = 0;
        uint64_t firstRows = rowEnd - rowBegin;
        if (firstRows > maxRows) {
            firstRows = maxRows;
        }
        PrefetchQKGate(gateSlot, b, h, hv, start + rowBegin, firstRows * K_);
        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += maxRows) {
            uint64_t tileRows = rowEnd - tileRow;
            if (tileRows > maxRows) {
                tileRows = maxRows;
            }
            uint64_t elems = tileRows * K_;
            LocalTensor<T> qTyped = GateQTyped(gateSlot);
            LocalTensor<T> kTyped = GateKTyped(gateSlot);
            LocalTensor<T> kgTyped = GateKgTyped(gateSlot);
            LocalTensor<SCORE_T> qScore = qTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<SCORE_T> wScore = kTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<SCORE_T> kgScore = kgTyped.template ReinterpretCast<SCORE_T>();
            LocalTensor<GK_T> gateTyped = GateScoreTyped(gateSlot, tileRow);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> qFp32 = arena;
            LocalTensor<float> kFp32 = arena[elems];
            LocalTensor<float> gFp32 = arena[2 * elems];
            LocalTensor<float> expFp32 = arena[3 * elems];
            LocalTensor<float> outFp32 = arena[4 * elems];
#else
            LocalTensor<T> vTyped = GateVTyped(gateSlot);
            LocalTensor<BETA_T> betaTyped = GateBetaTyped(gateSlot);
            LocalTensor<T> directQ = GateDirectQ(gateSlot);
            LocalTensor<T> directW = GateDirectW(gateSlot);
            LocalTensor<T> directV = GateDirectV(gateSlot);
            LocalTensor<float> betaFp32 = GateBetaFloat(gateSlot);
#endif

            uint64_t token = start + tileRow;
            WaitGateInputReady();
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            Cast(qFp32, qTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            if constexpr (IsSameType<GK_T, float>::value) {
                gFp32 = gateTyped;
            } else {
                Cast(gFp32, gateTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            }
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (IsSameType<BETA_T, float>::value) {
                Adds(betaFp32, betaTyped, 0.0f, static_cast<uint32_t>(tileRows));
            } else {
                Cast(betaFp32, betaTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(tileRows));
            }
            PipeBarrier<PIPE_V>();
#endif
            uint64_t nextTileRow = tileRow + maxRows;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            uint64_t nextGateSlot = (gateSlot + 1) % KDA_GATE_PIPELINE_DEPTH;
            if (nextTileRow < rowEnd) {
                uint64_t tileIndex = (tileRow - rowBegin) / maxRows;
                if (tileIndex + 1 >= KDA_GATE_PIPELINE_DEPTH) {
                    WaitGateOutputForMte2(nextGateSlot);
                }
                uint64_t nextRows = rowEnd - nextTileRow;
                if (nextRows > maxRows) {
                    nextRows = maxRows;
                }
                PrefetchQKGate(nextGateSlot, b, h, hv, start + nextTileRow, nextRows * K_);
            }
#else
            if (outputPending) {
                WaitGateOutputForMte2();
            }
            if (nextTileRow < rowEnd) {
                uint64_t nextRows = rowEnd - nextTileRow;
                if (nextRows > maxRows) {
                    nextRows = maxRows;
                }
                PrefetchQKGate(gateSlot ^ 1, b, h, hv, start + nextTileRow, nextRows * K_);
            }
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            uint16_t validRows = static_cast<uint16_t>(tileRows);
            if (useRef && tileRow >= validColEnd) {
                validRows = 0;
            } else if (useRef && tileRow + tileRows > validColEnd) {
                validRows = static_cast<uint16_t>(validColEnd - tileRow);
            }
            if (writeScoreScratch) {
                if (fuseScoreWriteback) {
                    if (materializeFinalKg_) {
                        PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, true, true, true>(
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                            (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(qScore.GetPhyAddr()),
                            (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(wScore.GetPhyAddr()),
                            (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kgScore.GetPhyAddr()),
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(directQ.GetPhyAddr()),
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(directW.GetPhyAddr()),
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(vTyped.GetPhyAddr()),
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(directV.GetPhyAddr()),
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(vTyped.GetPhyAddr()),
                            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaFp32.GetPhyAddr()),
                            (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                            (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                            (__ubuf__ float *)reinterpret_cast<uint64_t>(finalRefFp32.GetPhyAddr()),
                            static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_),
                            validRows);
                    } else {
                        PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, true, false, true>(
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                            (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(qScore.GetPhyAddr()),
                            (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(wScore.GetPhyAddr()),
                            (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kgScore.GetPhyAddr()),
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(directQ.GetPhyAddr()),
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(directW.GetPhyAddr()),
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(vTyped.GetPhyAddr()),
                            (__ubuf__ T *)reinterpret_cast<uint64_t>(directV.GetPhyAddr()),
                            nullptr,
                            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaFp32.GetPhyAddr()),
                            (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                            (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                            nullptr,
                            static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_), validRows);
                    }
                } else {
                    PrepareKdaGateQwKgRegbase<T, SCORE_T, GK_T, true, false, false>(
                        (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                        (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                        (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(qScore.GetPhyAddr()),
                        (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(wScore.GetPhyAddr()),
                        (__ubuf__ SCORE_T *)reinterpret_cast<uint64_t>(kgScore.GetPhyAddr()),
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                        (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                        nullptr,
                        static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_), validRows);
                }
            } else if (useRef) {
                PrepareKdaGateQwKgRegbase<T, T, GK_T, true, false, false>(
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kgTyped.GetPhyAddr()),
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                    nullptr,
                    static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_), validRows);
            } else {
                PrepareKdaGateQwKgRegbase<T, T, GK_T, false, false, false>(
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(qTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kTyped.GetPhyAddr()),
                    (__ubuf__ T *)reinterpret_cast<uint64_t>(kgTyped.GetPhyAddr()),
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateTyped.GetPhyAddr()),
                    (__ubuf__ float *)reinterpret_cast<uint64_t>(refFp32.GetPhyAddr()),
                    nullptr,
                    static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_), validRows);
            }
#else
            PipeBarrier<PIPE_V>();

            if (useRef) {
                for (uint64_t row = 0; row < tileRows; ++row) {
                    Sub(expFp32[row * K_], gFp32[row * K_], refFp32, static_cast<uint32_t>(K_));
                }
            } else {
                Adds(expFp32, gFp32, 0.0f, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            if (writeScoreScratch) {
                ClampScoreExpInput(expFp32, static_cast<uint32_t>(elems));
            } else {
                ClampExpInput(expFp32, static_cast<uint32_t>(elems));
            }
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            Mul(outFp32, qFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            if (writeScoreScratch) {
                ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
                Cast(qScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            } else {
                ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
                Cast(qTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();

            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            if (writeScoreScratch) {
                ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
                Cast(wScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            } else {
                ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
                Cast(kTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();

            if (useRef) {
                for (uint64_t row = 0; row < tileRows; ++row) {
                    Sub(expFp32[row * K_], refFp32, gFp32[row * K_], static_cast<uint32_t>(K_));
                }
            } else {
                Muls(expFp32, gFp32, -1.0f, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();
            Muls(expFp32, expFp32, LN2, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            if (writeScoreScratch) {
                ClampScoreExpInput(expFp32, static_cast<uint32_t>(elems));
            } else {
                ClampExpInput(expFp32, static_cast<uint32_t>(elems));
            }
            Exp(expFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            Mul(outFp32, kFp32, expFp32, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            if (useRef && tileRow + tileRows > validColEnd) {
                for (uint64_t row = 0; row < tileRows; ++row) {
                    if (tileRow + row >= validColEnd) {
                        Duplicate(outFp32[row * K_], 0.0f, static_cast<uint32_t>(K_));
                    }
                }
                PipeBarrier<PIPE_V>();
            }
            if (writeScoreScratch) {
                ClampFp32ForCast<SCORE_T>(outFp32, static_cast<uint32_t>(elems));
            } else {
                ClampFp32ToOutputType(outFp32, static_cast<uint32_t>(elems));
            }
            if (outputPending) {
                WaitGateOutputForVector();
            }
            if (writeScoreScratch) {
                Cast(kgScore, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            } else {
                Cast(kgTyped, outFp32, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            }
            PipeBarrier<PIPE_V>();
#endif

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            if (writeScoreScratch) {
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG, tileRow),
                              qScore, elems);
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W, tileRow),
                              wScore, elems);
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG, tileRow),
                              kgScore, elems);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                if (fuseScoreWriteback) {
                    StorePreparedQG(b, hv, token, directQ, elems);
                    CopyVectorOut(w_, KVOffset(b, hv, token, 0, K_), directW, elems);
                    CopyVectorOut(vNew_, KVOffset(b, hv, token, 0, V_), directV, tileRows * V_);
                    if (materializeFinalKg_) {
                        CopyVectorOut(finalKg_, KVOffset(b, hv, token, 0, K_), vTyped, elems);
                    }
                }
#endif
            } else {
                CopyVectorOut(qg_, KVOffset(b, hv, token, 0, K_), qTyped, elems);
                CopyVectorOut(w_, KVOffset(b, hv, token, 0, K_), kTyped, elems);
                CopyVectorOut(kg_, KVOffset(b, hv, token, 0, K_), kgTyped, elems);
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            SignalGateOutputDoneForMte2(gateSlot);
#else
            SignalGateOutputDone();
#endif
            outputPending = true;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            gateSlot = (gateSlot + 1) % KDA_GATE_PIPELINE_DEPTH;
#else
            gateSlot ^= 1;
#endif
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        uint64_t tileCount = (rowEnd - rowBegin + maxRows - 1) / maxRows;
        uint64_t firstPending =
            tileCount > KDA_GATE_PIPELINE_DEPTH ? tileCount - KDA_GATE_PIPELINE_DEPTH : 0;
        for (uint64_t tile = firstPending; tile < tileCount; ++tile) {
            WaitGateOutputForMte2(tile % KDA_GATE_PIPELINE_DEPTH);
        }
#else
        if (outputPending) {
            WaitGateOutputForMte2();
            WaitGateOutputForVector();
        }
#endif
        return;
    }

    __aicore__ inline void ZeroScoreScratchRange(uint64_t scoreSlot, uint64_t planeBegin,
                                                 uint64_t planeEnd, uint64_t firstRow,
                                                 uint64_t rowEnd)
    {
        if (firstRow >= rowEnd) {
            return;
        }
        const uint64_t maxRows = GatePipelineRows();
        LocalTensor<SCORE_T> zeroLocal = GateQTyped(0).template ReinterpretCast<SCORE_T>();
        for (uint64_t row = firstRow; row < rowEnd; row += maxRows) {
            uint64_t rows = rowEnd - row;
            if (rows > maxRows) {
                rows = maxRows;
            }
            const uint64_t elems = rows * K_;
            Duplicate(zeroLocal, static_cast<SCORE_T>(0), static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            for (uint64_t plane = planeBegin; plane < planeEnd; ++plane) {
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, plane, row),
                              zeroLocal, elems);
            }
            SignalGateOutputDone();
            WaitGateOutputForMte2();
            WaitGateOutputForVector();
        }
    }

    __aicore__ inline void ZeroScoreScratchPadding(uint64_t scoreSlot,
                                                   uint64_t scoreRowBegin,
                                                   uint64_t scoreRowCount,
                                                   uint64_t validColEnd,
                                                   uint64_t subBlockIdx,
                                                   uint64_t subBlockNum)
    {
        if (subBlockNum == 0 || subBlockIdx + 1 != subBlockNum) {
            return;
        }
        const uint64_t validRowEnd = scoreRowBegin + scoreRowCount;
        const uint64_t paddedRowEnd = (validRowEnd + 15) / 16 * 16;
        const uint64_t paddedColEnd = BT_;
        ZeroScoreScratchRange(scoreSlot, KDA_SCORE_SCRATCH_QG,
                              KDA_SCORE_SCRATCH_KG, validRowEnd, paddedRowEnd);
        ZeroScoreScratchRange(scoreSlot, KDA_SCORE_SCRATCH_KG,
                              KDA_SCORE_SCRATCH_PLANES, validColEnd, paddedColEnd);
    }

    __aicore__ inline void PrepareGateProducts(uint64_t b, uint64_t h, uint64_t hv, uint64_t start, uint64_t curT,
                                               uint64_t subBlockIdx, uint64_t subBlockNum, bool useRef = false,
                                               uint64_t refToken = 0, uint64_t validColEnd = 0,
                                               bool writeScoreScratch = false, uint64_t scoreSlot = 0,
                                               uint64_t scoreRowBegin = 0, uint64_t scoreRowCount = 0)
    {
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum) {
            return;
        }
        if (validColEnd == 0 || validColEnd > curT) {
            validColEnd = curT;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (writeScoreScratch && curT == BT_ && scoreRowBegin == 0 &&
            scoreRowCount == curT && validColEnd == curT) {
            PrepareGateProductsBulk(b, h, hv, start, curT, subBlockIdx, subBlockNum, useRef, refToken,
                                    validColEnd, writeScoreScratch, scoreSlot);
            ZeroScoreScratchPadding(scoreSlot, scoreRowBegin, scoreRowCount, validColEnd,
                                    subBlockIdx, subBlockNum);
            return;
        }
#endif
        if (writeScoreScratch) {
            PrepareScoreFactorsBulk(b, h, hv, start, subBlockIdx, subBlockNum, refToken, scoreRowBegin,
                                    scoreRowCount, validColEnd, start + curT - 1, scoreSlot);
            ZeroScoreScratchPadding(scoreSlot, scoreRowBegin, scoreRowCount, validColEnd,
                                    subBlockIdx, subBlockNum);
            return;
        }
        PrepareGateProductsBulk(b, h, hv, start, curT, subBlockIdx, subBlockNum, useRef, refToken,
                                validColEnd, writeScoreScratch, scoreSlot);
    }

    __aicore__ inline void ComputeRawAqkAkkCube(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                uint64_t start, uint64_t curT)
    {
        ComputeRawAqkAkkCubeBlock(b, hv, chunkIdx, start, curT, 0, curT);
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    template <uint32_t N>
    __aicore__ inline void ComputeRawAqkAkkCubeStableBlockDirectUbArch35(
        uint64_t b, uint64_t hv, uint64_t start, uint64_t rowBegin,
        uint64_t scoreSlot, uint8_t subBlockIdx, uint32_t directSlot)
    {
        using ElementA = SCORE_T;
        using ElementB = SCORE_T;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
        using DirectTileCopy = Common::Tile::PackedTileCopyTlaToUB<
            KdaArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, ElementA, LayoutTagL1A>;

        constexpr uint32_t scoreRows = KDA_SAFE_SCORE_REF_BC;
        constexpr uint32_t packedRows = scoreRows * 2;
        constexpr uint32_t k = 128;
        static_assert(scoreRows == 32 && packedRows == 64);
        static_assert(N == 32 || N == 64);

        Catlass::Arch::Resource<KdaArchTag> resource;
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(K_, BT_);
        auto tensorQPos = tla::MakeTensor(
            scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG)],
            layoutA, Catlass::Arch::PositionGM{});
        auto tensorKPos = tla::MakeTensor(
            scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W)],
            layoutA, Catlass::Arch::PositionGM{});
        auto tensorKNeg = tla::MakeTensor(
            scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG)],
            layoutB, Catlass::Arch::PositionGM{});

        auto blockQPos = GetTile(
            tensorQPos, tla::MakeCoord(rowBegin, 0), tla::MakeShape(scoreRows, k));
        auto blockKPos = GetTile(
            tensorKPos, tla::MakeCoord(rowBegin, 0), tla::MakeShape(scoreRows, k));
        auto blockKNeg = GetTile(
            tensorKNeg, tla::MakeCoord(0, 0), tla::MakeShape(k, N));

        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockQPos)>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockKNeg)>;

        static_assert(sizeof(ElementA) == sizeof(uint16_t));
        LocalTensor<ElementA> l1A = resource.l1Buf.template GetBufferByByte<ElementA>(
            scoreSlot * KDA_DIRECT_SCORE_L1_SLOT_BYTES);
        LocalTensor<ElementB> l1B = resource.l1Buf.template GetBufferByByte<ElementB>(
            KDA_DIRECT_SCORE_L1_B_OFFSET);
        LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        LocalTensor<ElementC> l0C = resource.l0CBuf.template GetBufferByByte<ElementC>(0);
        auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(packedRows, k);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(k, N);
        auto layoutL0A = tla::MakeLayout<ElementA, LayoutTagL0A>(packedRows, k);
        auto layoutL0B = tla::MakeLayout<ElementB, LayoutTagL0B>(k, N);
        auto layoutL0C = tla::MakeLayoutL0C(packedRows, N);
        auto tensorL1A = tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1A = GetTile(
            tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(packedRows, k));
        auto tileL1AQ = GetTile(
            tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(scoreRows, k));
        auto tileL1AK = GetTile(
            tensorL1A, tla::MakeCoord(scoreRows, 0), tla::MakeShape(scoreRows, k));
        auto tileL1B = GetTile(
            tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, N));
        auto tileL0A = GetTile(
            tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(packedRows, k));
        auto tileL0B = GetTile(
            tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, N));
        auto tileL0C = GetTile(
            tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(packedRows, N));
        auto tileL0CTop = GetTile(
            tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(scoreRows, N));
        auto tileL0CBottom = GetTile(
            tensorL0C, tla::MakeCoord(scoreRows, 0), tla::MakeShape(scoreRows, N));
        LocalTensor<ElementC> directBase = resource.ubBuf.template GetBufferByByte<ElementC>(
            KDA_DIRECT_SCORE_UB_BYTE_OFFSET +
            directSlot * KDA_DIRECT_SCORE_SLOT_ELEMENTS * sizeof(ElementC));
        LocalTensor<ElementC> directAqk = directBase;
        LocalTensor<ElementC> directAkk = directBase[KDA_DIRECT_SCORE_MATRIX_ELEMENTS];
        auto layoutDirect = tla::MakeLayout<ElementC, LayoutTagC>(scoreRows, BT_);
        auto tensorDirectAqk = tla::MakeTensor(
            directAqk, layoutDirect, Catlass::Arch::PositionUB{});
        auto tensorDirectAkk = tla::MakeTensor(
            directAkk, layoutDirect, Catlass::Arch::PositionUB{});
        auto blockDirectAqk = GetTile(
            tensorDirectAqk, tla::MakeCoord(0, 0), tla::MakeShape(scoreRows, N));
        auto blockDirectAkk = GetTile(
            tensorDirectAkk, tla::MakeCoord(0, 0), tla::MakeShape(scoreRows, N));
        using CopyL0CToDirectUb =
            typename DirectTileCopy::template CopyL0CToDst<decltype(blockDirectAqk)>;

        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToDirectUb copyL0CToDirectUb;
        TileMmad tileMmad;

        copyGmToL1A(tileL1AQ, blockQPos);
        copyGmToL1A(tileL1AK, blockKPos);
        copyGmToL1B(tensorL1B, blockKNeg);
        SetFlag<HardEvent::MTE2_MTE1>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(KDA_ARCH35_SCORE_EVENT);
        copyL1ToL0A(tileL0A, tileL1A);
        copyL1ToL0B(tileL0B, tileL1B);
        SetFlag<HardEvent::MTE1_M>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::MTE1_M>(KDA_ARCH35_SCORE_EVENT);
        tileMmad(tileL0C, tileL0A, tileL0B, packedRows, N, k, true, 0);
        SetFlag<HardEvent::M_FIX>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::M_FIX>(KDA_ARCH35_SCORE_EVENT);
        const uint64_t flagOffset =
            directSlot + subBlockIdx * KDA_DIRECT_SCORE_SUBBLOCK_FLAG_STRIDE;
        CrossCoreWaitFlag<0x4, PIPE_FIX>(KDA_DIRECT_SCORE_FREE_FLAG + flagOffset);
        copyL0CToDirectUb(
            blockDirectAqk, tileL0CTop, KDA_DIRECT_SCORE_ROWS, subBlockIdx, 1, 0);
        copyL0CToDirectUb(
            blockDirectAkk, tileL0CBottom, KDA_DIRECT_SCORE_ROWS, subBlockIdx, 1, 0);
        CrossCoreSetFlag<0x4, PIPE_FIX>(KDA_DIRECT_SCORE_READY_FLAG + flagOffset);
        SetFlag<HardEvent::FIX_M>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::FIX_M>(KDA_ARCH35_SCORE_EVENT);
    }

    template <uint32_t N>
    __aicore__ inline void PrefetchRawAqkAkkHeadPairArch35(
        uint64_t rowBegin, uint64_t scoreSlotBase, uint32_t l1BaseOffset,
        TEventID readyEvent)
    {
        using ElementA = SCORE_T;
        using ElementB = SCORE_T;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
            ElementC, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;

        constexpr uint32_t scoreRows = KDA_SAFE_SCORE_REF_BC;
        constexpr uint32_t packedRows = scoreRows * 2;
        constexpr uint32_t n = N;
        constexpr uint32_t k = 128;
        constexpr uint32_t l1ABytes = packedRows * k * sizeof(ElementA);
        constexpr uint32_t l1BBytes = k * n * sizeof(ElementB);
        constexpr uint32_t l1LaneBytes = l1ABytes + l1BBytes;
        static_assert(scoreRows == 32 && packedRows == 64);
        static_assert(N == 32 || N == 64);

        Catlass::Arch::Resource<KdaArchTag> resource;
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(K_, BT_);
        auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(packedRows, k);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(k, n);

        for (uint64_t lane = 0; lane < KDA_SCORE_LANES; ++lane) {
            const uint64_t scoreSlot = scoreSlotBase + lane;
            LocalTensor<ElementA> l1A = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1BaseOffset + static_cast<uint32_t>(lane) * l1LaneBytes);
            LocalTensor<ElementB> l1B = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BaseOffset + static_cast<uint32_t>(lane) * l1LaneBytes + l1ABytes);
            auto tensorL1A = tla::MakeTensor(
                l1A, layoutL1A, Catlass::Arch::PositionL1{});
            auto tensorL1B = tla::MakeTensor(
                l1B, layoutL1B, Catlass::Arch::PositionL1{});
            auto tileL1AQ = GetTile(
                tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(scoreRows, k));
            auto tileL1AK = GetTile(
                tensorL1A, tla::MakeCoord(scoreRows, 0), tla::MakeShape(scoreRows, k));

            auto tensorQPos = tla::MakeTensor(
                scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG)],
                layoutA, Catlass::Arch::PositionGM{});
            auto tensorKPos = tla::MakeTensor(
                scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W)],
                layoutA, Catlass::Arch::PositionGM{});
            auto tensorKNeg = tla::MakeTensor(
                scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG)],
                layoutB, Catlass::Arch::PositionGM{});
            auto blockQPos = GetTile(
                tensorQPos, tla::MakeCoord(rowBegin, 0), tla::MakeShape(scoreRows, k));
            auto blockKPos = GetTile(
                tensorKPos, tla::MakeCoord(rowBegin, 0), tla::MakeShape(scoreRows, k));
            auto blockKNeg = GetTile(
                tensorKNeg, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
            using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockQPos)>;
            using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockKNeg)>;
            CopyGmToL1A copyGmToL1A;
            CopyGmToL1B copyGmToL1B;
            copyGmToL1A(tileL1AQ, blockQPos);
            copyGmToL1A(tileL1AK, blockKPos);
            copyGmToL1B(tensorL1B, blockKNeg);
        }
        SetFlag<HardEvent::MTE2_MTE1>(readyEvent);
    }

    template <uint32_t N>
    __aicore__ inline void ComputeRawAqkAkkCubeStableHeadPairDirectUbArch35(
        uint64_t rowBegin, uint64_t scoreSlotBase, uint32_t directSlot)
    {
        using ElementA = SCORE_T;
        using ElementB = SCORE_T;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
            ElementC, LayoutTagC>;
        using DirectTileCopy = Common::Tile::PackedTileCopyTlaToUB<
            KdaArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB,
            ElementC, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
            KdaArchTag, ElementA, LayoutTagL1A>;

        constexpr uint32_t scoreRows = KDA_SAFE_SCORE_REF_BC;
        constexpr uint32_t packedRows = scoreRows * 2;
        constexpr uint32_t n = N;
        constexpr uint32_t k = 128;
        constexpr uint32_t l1ABytes = packedRows * k * sizeof(ElementA);
        constexpr uint32_t l1BBytes = k * n * sizeof(ElementB);
        constexpr uint32_t l1LaneBytes = l1ABytes + l1BBytes;
        static_assert(scoreRows == 32 && packedRows == 64);
        static_assert(N == 32 || N == 64);

        PrefetchRawAqkAkkHeadPairArch35<N>(
            rowBegin, scoreSlotBase, 0, KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(KDA_ARCH35_SCORE_EVENT);

        Catlass::Arch::Resource<KdaArchTag> resource;
        auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(packedRows, k);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(k, n);

        constexpr uint32_t l0ABytes = packedRows * k * sizeof(ElementA);
        constexpr uint32_t l0BBytes = k * n * sizeof(ElementB);
        constexpr uint32_t l0CBytes = packedRows * n * sizeof(ElementC);
        LocalTensor<ElementA> l0A0 =
            resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementA> l0A1 =
            resource.l0ABuf.template GetBufferByByte<ElementA>(l0ABytes);
        LocalTensor<ElementB> l0B0 =
            resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        LocalTensor<ElementB> l0B1 =
            resource.l0BBuf.template GetBufferByByte<ElementB>(l0BBytes);
        LocalTensor<ElementC> l0C0 =
            resource.l0CBuf.template GetBufferByByte<ElementC>(0);
        LocalTensor<ElementC> l0C1 =
            resource.l0CBuf.template GetBufferByByte<ElementC>(l0CBytes);
        auto layoutL0A = tla::MakeLayout<ElementA, LayoutTagL0A>(packedRows, k);
        auto layoutL0B = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
        auto layoutL0C = tla::MakeLayoutL0C(packedRows, n);
        auto tensorL0A0 = tla::MakeTensor(l0A0, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0A1 = tla::MakeTensor(l0A1, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B0 = tla::MakeTensor(l0B0, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0B1 = tla::MakeTensor(l0B1, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C0 = tla::MakeTensor(l0C0, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tensorL0C1 = tla::MakeTensor(l0C1, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL0A0 = GetTile(
            tensorL0A0, tla::MakeCoord(0, 0), tla::MakeShape(packedRows, k));
        auto tileL0A1 = GetTile(
            tensorL0A1, tla::MakeCoord(0, 0), tla::MakeShape(packedRows, k));
        auto tileL0B0 = GetTile(
            tensorL0B0, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0B1 = GetTile(
            tensorL0B1, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0C0 = GetTile(
            tensorL0C0, tla::MakeCoord(0, 0), tla::MakeShape(packedRows, n));
        auto tileL0C1 = GetTile(
            tensorL0C1, tla::MakeCoord(0, 0), tla::MakeShape(packedRows, n));
        auto tileL0C0Top = GetTile(
            tensorL0C0, tla::MakeCoord(0, 0), tla::MakeShape(scoreRows, n));
        auto tileL0C0Bottom = GetTile(
            tensorL0C0, tla::MakeCoord(scoreRows, 0), tla::MakeShape(scoreRows, n));
        auto tileL0C1Top = GetTile(
            tensorL0C1, tla::MakeCoord(0, 0), tla::MakeShape(scoreRows, n));
        auto tileL0C1Bottom = GetTile(
            tensorL0C1, tla::MakeCoord(scoreRows, 0), tla::MakeShape(scoreRows, n));
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        LocalTensor<ElementA> l1A0 =
            resource.l1Buf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementA> l1A1 =
            resource.l1Buf.template GetBufferByByte<ElementA>(l1LaneBytes);
        LocalTensor<ElementB> l1B0 =
            resource.l1Buf.template GetBufferByByte<ElementB>(l1ABytes);
        LocalTensor<ElementB> l1B1 =
            resource.l1Buf.template GetBufferByByte<ElementB>(l1LaneBytes + l1ABytes);
        auto tensorL1A0 = tla::MakeTensor(l1A0, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1A1 = tla::MakeTensor(l1A1, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1B0 = tla::MakeTensor(l1B0, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL1B1 = tla::MakeTensor(l1B1, layoutL1B, Catlass::Arch::PositionL1{});
        auto tileL1A0 = GetTile(
            tensorL1A0, tla::MakeCoord(0, 0), tla::MakeShape(packedRows, k));
        auto tileL1A1 = GetTile(
            tensorL1A1, tla::MakeCoord(0, 0), tla::MakeShape(packedRows, k));
        auto tileL1B0 = GetTile(
            tensorL1B0, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL1B1 = GetTile(
            tensorL1B1, tla::MakeCoord(0, 0), tla::MakeShape(k, n));

        LocalTensor<ElementC> directBase = resource.ubBuf.template GetBufferByByte<ElementC>(
            KDA_DIRECT_SCORE_UB_BYTE_OFFSET +
            directSlot * KDA_DIRECT_SCORE_SLOT_ELEMENTS * sizeof(ElementC));
        LocalTensor<ElementC> directAqk = directBase;
        LocalTensor<ElementC> directAkk = directBase[KDA_DIRECT_SCORE_MATRIX_ELEMENTS];
        auto layoutDirect = tla::MakeLayout<ElementC, LayoutTagC>(scoreRows, BT_);
        auto tensorDirectAqk = tla::MakeTensor(
            directAqk, layoutDirect, Catlass::Arch::PositionUB{});
        auto tensorDirectAkk = tla::MakeTensor(
            directAkk, layoutDirect, Catlass::Arch::PositionUB{});
        auto blockDirectAqk = GetTile(
            tensorDirectAqk, tla::MakeCoord(0, 0), tla::MakeShape(scoreRows, n));
        auto blockDirectAkk = GetTile(
            tensorDirectAkk, tla::MakeCoord(0, 0), tla::MakeShape(scoreRows, n));
        using CopyL0CToDirectUb =
            typename DirectTileCopy::template CopyL0CToDst<decltype(blockDirectAqk)>;
        CopyL0CToDirectUb copyL0CToDirectUb;

        copyL1ToL0A(tileL0A0, tileL1A0);
        copyL1ToL0B(tileL0B0, tileL1B0);
        SetFlag<HardEvent::MTE1_M>(KDA_ARCH35_SCORE_EVENT);
        copyL1ToL0A(tileL0A1, tileL1A1);
        copyL1ToL0B(tileL0B1, tileL1B1);
        SetFlag<HardEvent::MTE1_M>(KDA_ARCH35_SCORE_W_EVENT);

        WaitFlag<HardEvent::MTE1_M>(KDA_ARCH35_SCORE_EVENT);
        tileMmad(tileL0C0, tileL0A0, tileL0B0, packedRows, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::M_FIX>(KDA_ARCH35_SCORE_EVENT);
        const uint64_t flagOffset0 = directSlot;
        CrossCoreWaitFlag<0x4, PIPE_FIX>(KDA_DIRECT_SCORE_FREE_FLAG + flagOffset0);
        copyL0CToDirectUb(
            blockDirectAqk, tileL0C0Top, KDA_DIRECT_SCORE_ROWS, 0, 1, 0);
        copyL0CToDirectUb(
            blockDirectAkk, tileL0C0Bottom, KDA_DIRECT_SCORE_ROWS, 0, 1, 0);
        CrossCoreSetFlag<0x4, PIPE_FIX>(KDA_DIRECT_SCORE_READY_FLAG + flagOffset0);
        SetFlag<HardEvent::FIX_M>(KDA_ARCH35_SCORE_EVENT);

        WaitFlag<HardEvent::MTE1_M>(KDA_ARCH35_SCORE_W_EVENT);
        tileMmad(tileL0C1, tileL0A1, tileL0B1, packedRows, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(KDA_ARCH35_SCORE_W_EVENT);
        WaitFlag<HardEvent::M_FIX>(KDA_ARCH35_SCORE_W_EVENT);
        const uint64_t flagOffset1 =
            directSlot + KDA_DIRECT_SCORE_SUBBLOCK_FLAG_STRIDE;
        CrossCoreWaitFlag<0x4, PIPE_FIX>(KDA_DIRECT_SCORE_FREE_FLAG + flagOffset1);
        copyL0CToDirectUb(
            blockDirectAqk, tileL0C1Top, KDA_DIRECT_SCORE_ROWS, 1, 1, 0);
        copyL0CToDirectUb(
            blockDirectAkk, tileL0C1Bottom, KDA_DIRECT_SCORE_ROWS, 1, 1, 0);
        CrossCoreSetFlag<0x4, PIPE_FIX>(KDA_DIRECT_SCORE_READY_FLAG + flagOffset1);
        SetFlag<HardEvent::FIX_M>(KDA_ARCH35_SCORE_W_EVENT);

        WaitFlag<HardEvent::FIX_M>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::FIX_M>(KDA_ARCH35_SCORE_W_EVENT);
    }

    __aicore__ inline void ComputeRawAqkAkkCubeFullArch35(
        uint64_t b, uint64_t hv, uint64_t start, uint64_t scoreSlot)
    {
        using ElementA = SCORE_T;
        using ElementB = SCORE_T;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, ElementA, LayoutTagL1A>;

        constexpr uint32_t m = 64;
        constexpr uint32_t n = 64;
        constexpr uint32_t k = 128;
        Catlass::Arch::Resource<KdaArchTag> resource;
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(m, n);
        auto tensorQPos = tla::MakeTensor(
            scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG)],
            layoutA, Catlass::Arch::PositionGM{});
        auto tensorKPos = tla::MakeTensor(
            scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W)],
            layoutA, Catlass::Arch::PositionGM{});
        auto tensorKNeg = tla::MakeTensor(
            scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG)],
            layoutB, Catlass::Arch::PositionGM{});
        auto tensorAqk = tla::MakeTensor(
            aqk_[AOffset(b, hv, start, 0)], layoutC, Catlass::Arch::PositionGM{});
        auto tensorAkk = tla::MakeTensor(
            akk_[AOffset(b, hv, start, 0)], layoutC, Catlass::Arch::PositionGM{});
        auto blockQPos = GetTile(tensorQPos, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto blockKPos = GetTile(tensorKPos, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto blockKNeg = GetTile(tensorKNeg, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto blockAkk = GetTile(tensorAkk, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockQPos)>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockKNeg)>;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(blockAqk)>;

        constexpr uint32_t l1ABytes = m * k * sizeof(ElementA);
        LocalTensor<ElementA> l1A0 = resource.l1Buf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementA> l1A1 = resource.l1Buf.template GetBufferByByte<ElementA>(l1ABytes);
        LocalTensor<ElementB> l1B = resource.l1Buf.template GetBufferByByte<ElementB>(2 * l1ABytes);
        LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        LocalTensor<ElementC> l0C = resource.l0CBuf.template GetBufferByByte<ElementC>(0);
        auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(m, k);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(k, n);
        auto layoutL0A = tla::MakeLayout<ElementA, LayoutTagL0A>(m, k);
        auto layoutL0B = tla::MakeLayout<ElementB, LayoutTagL0B>(k, n);
        auto layoutL0C = tla::MakeLayoutL0C(m, n);
        auto tensorL1A0 = tla::MakeTensor(l1A0, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1A1 = tla::MakeTensor(l1A1, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1A0 = GetTile(tensorL1A0, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1A1 = GetTile(tensorL1A1, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToDst copyL0CToDst;
        TileMmad tileMmad;

        copyGmToL1B(tensorL1B, blockKNeg);
        copyGmToL1A(tensorL1A0, blockQPos);
        SetFlag<HardEvent::MTE2_MTE1>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(KDA_ARCH35_SCORE_EVENT);
        copyL1ToL0B(tileL0B, tileL1B);
        copyL1ToL0A(tileL0A, tileL1A0);
        SetFlag<HardEvent::MTE1_M>(KDA_ARCH35_SCORE_EVENT);
        copyGmToL1A(tensorL1A1, blockKPos);
        SetFlag<HardEvent::MTE2_MTE1>(KDA_ARCH35_SCORE_W_EVENT);
        WaitFlag<HardEvent::MTE1_M>(KDA_ARCH35_SCORE_EVENT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(KDA_ARCH35_SCORE_EVENT);
        SetFlag<HardEvent::M_MTE1>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(KDA_ARCH35_SCORE_W_EVENT);
        WaitFlag<HardEvent::M_FIX>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::M_MTE1>(KDA_ARCH35_SCORE_EVENT);
        copyL0CToDst(blockAqk, tensorL0C);
        SetFlag<HardEvent::FIX_MTE2>(KDA_ARCH35_SCORE_EVENT);
        copyL1ToL0B(tileL0B, tileL1B);
        copyL1ToL0A(tileL0A, tileL1A1);
        SetFlag<HardEvent::MTE1_M>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::FIX_MTE2>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::MTE1_M>(KDA_ARCH35_SCORE_EVENT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::M_FIX>(KDA_ARCH35_SCORE_EVENT);
        copyL0CToDst(blockAkk, tensorL0C);
        SetFlag<HardEvent::FIX_MTE2>(KDA_ARCH35_SCORE_EVENT);
        WaitFlag<HardEvent::FIX_MTE2>(KDA_ARCH35_SCORE_EVENT);
    }
#endif

    __aicore__ inline void ComputeRawAqkAkkCubeBlock(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                     uint64_t start, uint64_t curT,
                                                     uint64_t rowBegin, uint64_t rowCount,
                                                     bool readScoreScratch = false, uint64_t scoreSlot = 0,
                                                     uint64_t colCount = 0)
    {
        if (colCount == 0 || colCount > curT) {
            colCount = curT;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            if (KDA_ARCH35_ENABLE_MANUAL_SCORE_PIPELINE && fusePostWu_ &&
                rowBegin == 0 && rowCount == 64 && colCount == 64) {
                ComputeRawAqkAkkCubeFullArch35(b, hv, start, scoreSlot);
                return;
            }
        }
#endif
        using ElementA = SCORE_T;
        using ElementB = SCORE_T;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::ColumnMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaScoreDispatchPolicy, KdaL1TileShape, KdaL0TileShape,
                                                              ElementA, ElementB, ElementC, void, TileCopy>;

        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(K_, BT_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(BT_, BT_);
        const bool paddedTail = curT < BT_;
        const uint64_t mmRowCount = paddedTail ? (rowCount + 15) / 16 * 16 : rowCount;
        const uint64_t mmColCount = paddedTail ? BT_ : colCount;
        Catlass::GemmCoord shape{static_cast<uint32_t>(mmRowCount), static_cast<uint32_t>(mmColCount),
                                 static_cast<uint32_t>(K_)};

        (void)readScoreScratch;
        auto tensorQPos =
            tla::MakeTensor(scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG)],
                            layoutA, Catlass::Arch::PositionGM{});
        auto tensorKPos =
            tla::MakeTensor(scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W)],
                            layoutA, Catlass::Arch::PositionGM{});
        auto tensorKNeg =
            tla::MakeTensor(scoreWorkspace_[ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG)],
                            layoutB, Catlass::Arch::PositionGM{});
        auto aqkBase = paddedTail
            ? solveWorkspace_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_RAW_AQK)]
            : aqk_[AOffset(b, hv, start, 0)];
        auto akkBase = paddedTail
            ? solveWorkspace_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_RAW_AKK)]
            : akk_[AOffset(b, hv, start, 0)];
        auto tensorAqk = tla::MakeTensor(aqkBase, layoutC, Catlass::Arch::PositionGM{});
        auto tensorAkk = tla::MakeTensor(akkBase, layoutC, Catlass::Arch::PositionGM{});

        auto blockQPos = GetTile(tensorQPos, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockKPos = GetTile(tensorKPos, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.k()));
        auto blockKNeg = GetTile(tensorKNeg, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.n()));
        auto blockAkk = GetTile(tensorAkk, tla::MakeCoord(rowBegin, 0), tla::MakeShape(shape.m(), shape.n()));

        blockMmad.preSetFlags();
        blockMmad(blockQPos, blockKNeg, blockAqk, shape);
        blockMmad(blockKPos, blockKNeg, blockAkk, shape);
        blockMmad.finalWaitFlags();
    }

    __aicore__ inline bool UseAkkCubeSolve(uint64_t curT) const
    {
        return curT > 0 && curT <= BT_ && (BT_ == 64 || BT_ == 128) && K_ >= 16 && V_ >= 16 &&
               V_ <= 256 && K_ % 16 == 0 && V_ % 16 == 0;
    }

    __aicore__ inline bool UsePostWuCube(uint64_t curT) const
    {
        return curT > 0 && curT <= BT_ && (BT_ == 64 || BT_ == 128) && K_ >= 16 && V_ >= 16 &&
               V_ <= 256 && K_ % 16 == 0 && V_ % 16 == 0;
    }

    __aicore__ inline void CopyLocalFloat(LocalTensor<float> dst, LocalTensor<float> src, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        Adds(dst, src, 0.0f, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FillLocalFloat(LocalTensor<float> dst, float value, uint64_t count)
    {
        if (count == 0) {
            return;
        }
        Duplicate(dst, value, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ForwardSubDiag16(LocalTensor<float> diag, LocalTensor<float> row,
                                             LocalTensor<float> prod, LocalTensor<float> rowBrcb,
                                             LocalTensor<float> reduced, uint64_t valid)
    {
        constexpr uint32_t brcbStride = 8;
        constexpr uint32_t diagSize = KDA_SOLVE_DIAG_BT;
        constexpr uint8_t rowBlk = diagSize * sizeof(float) / 32;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ForwardSubDiag16Regbase(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(diag.GetPhyAddr()),
            static_cast<uint16_t>(valid));
#else
        for (uint64_t i = 2; i < valid; ++i) {
            uint32_t rowOffset = static_cast<uint32_t>(i * diagSize);
            DataCopy(row, diag[rowOffset], diagSize);
            PipeBarrier<PIPE_V>();

            Brcb(rowBrcb, row, diagSize / brcbStride, {1, 8});
            PipeBarrier<PIPE_V>();
            for (uint32_t col = 0; col < diagSize; col += brcbStride) {
                Mul(prod[col], diag[col], rowBrcb, brcbStride, static_cast<uint8_t>(diagSize),
                    {1, 1, 0, rowBlk, rowBlk, 1});
            }
            PipeBarrier<PIPE_V>();

            uint32_t remain = diagSize;
            while (remain > 1) {
                uint32_t calcCount = (remain / 2) * diagSize;
                remain = (remain + 1) / 2;
                Add(prod, prod, prod[remain * diagSize], calcCount);
                PipeBarrier<PIPE_V>();
            }
            DataCopy(reduced, prod, diagSize);
            PipeBarrier<PIPE_V>();
            Add(row, row, reduced, diagSize);
            PipeBarrier<PIPE_V>();
            DataCopy(diag[rowOffset], row, diagSize);
            PipeBarrier<PIPE_V>();
        }
#endif

        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        for (uint32_t i = 0; i < diagSize; ++i) {
            uint32_t diagOffset = i * diagSize + i;
            if (i < valid) {
                diag.SetValue(diagOffset, diag.GetValue(diagOffset) + 1.0f);
            } else {
                diag.SetValue(diagOffset, 1.0f);
            }
        }
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
    }

    __aicore__ inline void SolveDiagonalBlocksInRows(LocalTensor<float> akkMat, LocalTensor<float> xMat,
                                                      LocalTensor<float> arena, uint64_t scratchBase,
                                                      uint64_t curT, uint64_t rowBegin, uint64_t rowCount)
    {
        constexpr uint32_t diagSize = KDA_SOLVE_DIAG_BT;
        constexpr uint32_t diagElements = diagSize * diagSize;
        constexpr uint32_t brcbElements = diagSize * 8;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        (void)akkMat;
        (void)arena;
        (void)scratchBase;
        uint64_t rowEnd = rowBegin + rowCount;
        for (uint64_t blockBegin = 0; blockBegin < BT_; blockBegin += diagSize) {
            if (blockBegin < rowBegin || blockBegin + diagSize > rowEnd) {
                continue;
            }
            uint64_t localBlockRow = blockBegin - rowBegin;
            uint64_t valid = blockBegin < curT ? curT - blockBegin : 0;
            if (valid > diagSize) {
                valid = diagSize;
            }
            ForwardSubDiag16StridedRegbase(
                (__ubuf__ float *)reinterpret_cast<uint64_t>(xMat.GetPhyAddr()),
                static_cast<uint16_t>(BT_), static_cast<uint16_t>(localBlockRow),
                static_cast<uint16_t>(blockBegin), static_cast<uint16_t>(valid));
            SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
            WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
            for (uint32_t rowIdx = 0; rowIdx < diagSize; ++rowIdx) {
                uint32_t diagOffset =
                    static_cast<uint32_t>((localBlockRow + rowIdx) * BT_ + blockBegin + rowIdx);
                if (rowIdx < valid) {
                    xMat.SetValue(diagOffset, xMat.GetValue(diagOffset) + 1.0f);
                } else {
                    xMat.SetValue(diagOffset, 1.0f);
                }
            }
            SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
            WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        }
#else
        LocalTensor<float> diag = arena[scratchBase];
        LocalTensor<float> row = diag[diagElements];
        LocalTensor<float> prod = row[diagSize];
        LocalTensor<float> rowBrcb = prod[diagElements];
        LocalTensor<float> reduced = rowBrcb[brcbElements];

        uint64_t rowEnd = rowBegin + rowCount;
        for (uint64_t blockBegin = 0; blockBegin < BT_; blockBegin += diagSize) {
            if (blockBegin < rowBegin || blockBegin + diagSize > rowEnd) {
                continue;
            }
            Duplicate(diag, 0.0f, diagElements);
            PipeBarrier<PIPE_V>();

            uint64_t localBlockRow = blockBegin - rowBegin;
            uint64_t valid = blockBegin < curT ? curT - blockBegin : 0;
            if (valid > diagSize) {
                valid = diagSize;
            }
            for (uint32_t rowIdx = 0; rowIdx < diagSize; ++rowIdx) {
                uint64_t srcOffset = (localBlockRow + rowIdx) * BT_ + blockBegin;
                Muls(diag[rowIdx * diagSize], akkMat[srcOffset], -1.0f, diagSize);
            }
            PipeBarrier<PIPE_V>();

            ForwardSubDiag16(diag, row, prod, rowBrcb, reduced, valid);
            for (uint32_t rowIdx = 0; rowIdx < diagSize; ++rowIdx) {
                uint64_t dstOffset = (localBlockRow + rowIdx) * BT_ + blockBegin;
                Adds(xMat[dstOffset], diag[rowIdx * diagSize], 0.0f, diagSize);
            }
            PipeBarrier<PIPE_V>();
        }
#endif
    }

    __aicore__ inline void BuildPrefixMask(LocalTensor<float> dst, uint64_t prefix, uint64_t count)
    {
        if (prefix > count) {
            prefix = count;
        }
        Duplicate(dst, 0.0f, static_cast<uint32_t>(count));
        if (prefix > 0) {
            Duplicate(dst, 1.0f, static_cast<uint32_t>(prefix));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline uint64_t BuildCausalMask(uint64_t threshold, uint64_t colBegin) const
    {
        if (threshold <= colBegin) {
            return ~0ULL;
        }
        if (threshold >= colBegin + KDA_SOLVE_BT) {
            return 0ULL;
        }
        return ~0ULL << (threshold - colBegin);
    }

    __aicore__ inline void BuildCausalSelectMasks(LocalTensor<uint8_t> aqkMask, LocalTensor<uint8_t> akkMask,
                                                  uint64_t rowBegin, uint64_t rowCount, uint64_t colBegin)
    {
        __ubuf__ uint64_t *aqkMaskPtr = reinterpret_cast<__ubuf__ uint64_t *>(aqkMask.GetPhyAddr());
        __ubuf__ uint64_t *akkMaskPtr = reinterpret_cast<__ubuf__ uint64_t *>(akkMask.GetPhyAddr());
        for (uint32_t localRow = 0; localRow < rowCount; ++localRow) {
            uint32_t row = static_cast<uint32_t>(rowBegin + localRow);
            aqkMaskPtr[localRow] = BuildCausalMask(static_cast<uint64_t>(row) + 1, colBegin);
            akkMaskPtr[localRow] = BuildCausalMask(static_cast<uint64_t>(row), colBegin);
        }
    }

    __aicore__ inline void SelectCausalRows(LocalTensor<float> aqkMat, LocalTensor<float> akkMat,
                                            uint64_t rowBegin, uint64_t rowCount)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            SelectCausalRows64Regbase(
                (__ubuf__ float *)reinterpret_cast<uint64_t>(aqkMat.GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(akkMat.GetPhyAddr()),
                static_cast<uint16_t>(rowBegin), static_cast<uint16_t>(rowCount));
            PipeBarrier<PIPE_V>();
            return;
        }
#endif
        LocalTensor<uint8_t> aqkMask = vecBuf_.Get<uint8_t>()[KDA_SELECT_AQK_MASK_BYTE_OFFSET];
        LocalTensor<uint8_t> akkMask = vecBuf_.Get<uint8_t>()[KDA_SELECT_AKK_MASK_BYTE_OFFSET];
        LocalTensor<float> zeroLocal = vecBuf_.Get<float>()[KDA_SELECT_ZERO_FLOAT_OFFSET];
        Duplicate(zeroLocal, 0.0f, 8);
        PipeBarrier<PIPE_V>();

        uint64_t colBlockCount = (BT_ + KDA_SOLVE_BT - 1) / KDA_SOLVE_BT;
        for (uint64_t colBlock = 0; colBlock < colBlockCount; ++colBlock) {
            uint64_t maskOffset = colBlock * KDA_SELECT_COL_MASK_BYTES;
            uint64_t colBegin = colBlock * KDA_SOLVE_BT;
            BuildCausalSelectMasks(aqkMask[maskOffset], akkMask[maskOffset], rowBegin, rowCount, colBegin);
        }
        SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);

        uint8_t rowStride = static_cast<uint8_t>(BT_ * sizeof(float) / 32);
        BinaryRepeatParams repeatParams = {1, 0, 1, rowStride, 0, rowStride};
        for (uint64_t colBlock = 0; colBlock < colBlockCount; ++colBlock) {
            uint64_t maskOffset = colBlock * KDA_SELECT_COL_MASK_BYTES;
            uint64_t colBegin = colBlock * KDA_SOLVE_BT;
            Select(aqkMat[colBegin], aqkMask[maskOffset], zeroLocal, aqkMat[colBegin],
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, KDA_SOLVE_BT, static_cast<uint8_t>(rowCount), repeatParams);
            Select(akkMat[colBegin], akkMask[maskOffset], zeroLocal, akkMat[colBegin],
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, KDA_SOLVE_BT, static_cast<uint8_t>(rowCount), repeatParams);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
    }

    __aicore__ inline void PrepareAqkAkkSolveInput64(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> xMat = arena[2 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaBrcb = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT];
        LocalTensor<float> maskLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512];
        LocalTensor<float> oneHotLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512 + KDA_SOLVE_BT];

        LoadAsFloatRow(beta_, BetaOffset(b, hv, start), betaLocal, KDA_SOLVE_BT);
        Brcb(betaBrcb, betaLocal, 8, {1, 8});
        PipeBarrier<PIPE_V>();

        DataCopy(aqkMat, aqk_[AOffset(b, hv, start, 0)], KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(akkMat, akk_[AOffset(b, hv, start, 0)], KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        for (uint64_t col = 0; col < KDA_SOLVE_BT; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, KDA_SOLVE_BT, {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
        SelectCausalRows(aqkMat, akkMat, 0, KDA_SOLVE_BT);

        Muls(xMat, akkMat, -1.0f, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();
        for (uint64_t row = 0; row < KDA_SOLVE_BT; ++row) {
            BuildPrefixMask(maskLocal, row + 1, KDA_SOLVE_BT);
            BuildPrefixMask(oneHotLocal, row, KDA_SOLVE_BT);
            Sub(maskLocal, maskLocal, oneHotLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
            Add(xMat[row * KDA_SOLVE_BT], xMat[row * KDA_SOLVE_BT], maskLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(aqk_[AOffset(b, hv, start, 0)], aqkMat, KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(akk_[AOffset(b, hv, start, 0)], akkMat, KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X)], xMat,
                 KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void PrepareAqkAkkSolveInputTail(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                       uint64_t start, uint64_t curT)
    {
        uint64_t elemCount = curT * KDA_SOLVE_BT;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> xMat = arena[2 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS];
        LocalTensor<float> betaBrcb = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT];
        LocalTensor<float> maskLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512];
        LocalTensor<float> oneHotLocal = arena[3 * KDA_SOLVE_MATRIX_ELEMENTS + KDA_SOLVE_BT + 512 + KDA_SOLVE_BT];

        FillLocalFloat(betaLocal, 0.0f, KDA_SOLVE_BT);
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        LoadAsFloatRow(beta_, BetaOffset(b, hv, start), betaLocal, curT);
        Brcb(betaBrcb, betaLocal, 8, {1, 8});
        PipeBarrier<PIPE_V>();

        DataCopy(aqkMat, aqk_[AOffset(b, hv, start, 0)], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        if (elemCount < KDA_SOLVE_MATRIX_ELEMENTS) {
            FillLocalFloat(aqkMat[elemCount], 0.0f, KDA_SOLVE_MATRIX_ELEMENTS - elemCount);
        }
        DataCopy(akkMat, akk_[AOffset(b, hv, start, 0)], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        if (elemCount < KDA_SOLVE_MATRIX_ELEMENTS) {
            FillLocalFloat(akkMat[elemCount], 0.0f, KDA_SOLVE_MATRIX_ELEMENTS - elemCount);
        }

        for (uint64_t col = 0; col < KDA_SOLVE_BT; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, KDA_SOLVE_BT, {1, 1, 1, 8, 8, 1});
            PipeBarrier<PIPE_V>();
        }
        SelectCausalRows(aqkMat, akkMat, 0, KDA_SOLVE_BT);

        Muls(xMat, akkMat, -1.0f, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();
        for (uint64_t row = 0; row < KDA_SOLVE_BT; ++row) {
            BuildPrefixMask(maskLocal, row + 1, KDA_SOLVE_BT);
            BuildPrefixMask(oneHotLocal, row, KDA_SOLVE_BT);
            Sub(maskLocal, maskLocal, oneHotLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
            Add(xMat[row * KDA_SOLVE_BT], xMat[row * KDA_SOLVE_BT], maskLocal, KDA_SOLVE_BT);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(aqk_[AOffset(b, hv, start, 0)], aqkMat, static_cast<uint32_t>(elemCount));
        DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X)], xMat,
                 KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(h_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0)], akkMat,
                 KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void GetSolveRowRange(uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum,
                                            uint64_t &rowBegin, uint64_t &rowEnd) const
    {
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum) {
            rowBegin = 0;
            rowEnd = 0;
            return;
        }
        rowBegin = (curT * subBlockIdx) / subBlockNum;
        rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void InitializeDirectScoreUbArch35()
    {
        for (uint32_t slot = 0; slot < KDA_DIRECT_SCORE_QUEUE_DEPTH; ++slot) {
            CrossCoreSetFlag<0x4, PIPE_V>(KDA_DIRECT_SCORE_FREE_FLAG + slot);
        }
    }

    __aicore__ inline void ProcessDirectScoreSolveRowsArch35(
        uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
        uint64_t rowBegin, uint32_t directSlot)
    {
        CrossCoreWaitFlag<0x4, PIPE_V>(KDA_DIRECT_SCORE_READY_FLAG + directSlot);

        Catlass::Arch::Resource<KdaArchTag> resource;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> directBase = resource.ubBuf.template GetBufferByByte<float>(
            KDA_DIRECT_SCORE_UB_BYTE_OFFSET +
            directSlot * KDA_DIRECT_SCORE_SLOT_ELEMENTS * sizeof(float));
        LocalTensor<float> aqkMat = directBase;
        LocalTensor<float> akkMat = directBase[KDA_DIRECT_SCORE_MATRIX_ELEMENTS];
        LocalTensor<float> xMat = directBase[2 * KDA_DIRECT_SCORE_MATRIX_ELEMENTS];

        SelectCausalRows64Regbase(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(aqkMat.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(akkMat.GetPhyAddr()),
            static_cast<uint16_t>(rowBegin), KDA_DIRECT_SCORE_ROWS);
        PipeBarrier<PIPE_V>();
        Muls(xMat, akkMat, -1.0f, KDA_DIRECT_SCORE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();
        SolveDiagonalBlocksInRows(
            akkMat, xMat, arena, 0, BT_, rowBegin, KDA_DIRECT_SCORE_ROWS);

        LocalTensor<T> aqkTyped = GateQTyped(0);
        Muls(aqkMat, aqkMat, scale_, KDA_DIRECT_SCORE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();
        ClampFp32ToOutputType(aqkMat, KDA_DIRECT_SCORE_MATRIX_ELEMENTS);
        Cast(aqkTyped, aqkMat, RoundMode::CAST_RINT, KDA_DIRECT_SCORE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();

        const uint64_t token = start + rowBegin;
        const uint64_t xBase =
            SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        CopyVectorOut(o_, AOffset(b, hv, token, 0), aqkTyped,
                      KDA_DIRECT_SCORE_MATRIX_ELEMENTS);
        DataCopy(solveWorkspace_[xBase], xMat, KDA_DIRECT_SCORE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        CrossCoreSetFlag<0x4, PIPE_V>(KDA_DIRECT_SCORE_FREE_FLAG + directSlot);
    }
#endif

    __aicore__ inline void PrepareAqkAkkSolveInputRows(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                       uint64_t start, uint64_t curT, uint64_t rowBegin,
                                                       uint64_t rowEnd, bool storeLToAkk, bool storeLToScratch)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t validRowCount = rowBegin < curT ? curT - rowBegin : 0;
        if (validRowCount > rowCount) {
            validRowCount = rowCount;
        }
        uint64_t elemCount = rowCount * BT_;
        uint64_t validElemCount = validRowCount * BT_;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> aqkMat = arena;
        LocalTensor<float> akkMat = arena[elemCount];
        LocalTensor<float> xMat = arena[2 * elemCount];
        LocalTensor<float> betaLocal = arena[3 * elemCount];
        LocalTensor<float> betaBrcb = arena[3 * elemCount + BT_];
        LocalTensor<float> maskLocal = arena[3 * elemCount + BT_ + 512];
        LocalTensor<float> oneHotLocal = arena[3 * elemCount + BT_ + 512 + BT_];

        uint64_t token = start + rowBegin;

        constexpr bool scoreCanIncludeBeta =
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128;
#else
            false;
#endif
        const bool scoreIncludesBeta = scoreCanIncludeBeta && HV_ % KDA_SCORE_LANES == 0;
        if (validRowCount < rowCount) {
            FillLocalFloat(aqkMat, 0.0f, elemCount);
            FillLocalFloat(akkMat, 0.0f, elemCount);
            if (!scoreIncludesBeta) {
                FillLocalFloat(betaLocal, 0.0f, rowCount);
            }
        }
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        if (validRowCount > 0) {
            if (!scoreIncludesBeta) {
                LoadAsFloatRow(beta_, BetaOffset(b, hv, token), betaLocal, validRowCount);
            }
            if (curT < BT_) {
                DataCopy(aqkMat,
                         solveWorkspace_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_RAW_AQK) +
                                         rowBegin * BT_],
                         static_cast<uint32_t>(validElemCount));
                DataCopy(akkMat,
                         solveWorkspace_[SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_RAW_AKK) +
                                         rowBegin * BT_],
                         static_cast<uint32_t>(validElemCount));
            } else {
                DataCopy(aqkMat, aqk_[AOffset(b, hv, token, 0)], static_cast<uint32_t>(validElemCount));
                DataCopy(akkMat, akk_[AOffset(b, hv, token, 0)], static_cast<uint32_t>(validElemCount));
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (!scoreIncludesBeta) {
            ApplyKdaRowScaleRegbase(
                (__ubuf__ float *)reinterpret_cast<uint64_t>(akkMat.GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(betaLocal.GetPhyAddr()),
                static_cast<uint16_t>(rowCount), static_cast<uint16_t>(BT_));
            PipeBarrier<PIPE_V>();
        }
#else
        Brcb(betaBrcb, betaLocal, static_cast<uint8_t>((rowCount + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();
        uint8_t rowStride = static_cast<uint8_t>(BT_ * sizeof(float) / 32);
        for (uint64_t col = 0; col < BT_; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, static_cast<uint8_t>(rowCount),
                {1, 1, 0, rowStride, rowStride, 1});
        }
        PipeBarrier<PIPE_V>();
#endif
        if (validRowCount > 0) {
            SelectCausalRows(aqkMat, akkMat, rowBegin, validRowCount);
        }

        Muls(xMat, akkMat, -1.0f, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_V>();
        if constexpr (SAFE_GATE) {
            uint64_t scratchBase = 3 * elemCount + BT_ + 512 + 2 * BT_;
            SolveDiagonalBlocksInRows(akkMat, xMat, arena, scratchBase, curT, rowBegin, rowCount);
        } else if (curT < BT_) {
            uint64_t scratchBase = 3 * elemCount + BT_ + 512 + 2 * BT_;
            SolveDiagonalBlocksInRows(akkMat, xMat, arena, scratchBase, curT, rowBegin, rowCount);
        } else {
            for (uint64_t localRow = 0; localRow < rowCount; ++localRow) {
                uint64_t row = rowBegin + localRow;
                BuildPrefixMask(maskLocal, row + 1, BT_);
                BuildPrefixMask(oneHotLocal, row, BT_);
                Sub(maskLocal, maskLocal, oneHotLocal, static_cast<uint32_t>(BT_));
                PipeBarrier<PIPE_V>();
                Add(xMat[localRow * BT_], xMat[localRow * BT_], maskLocal, static_cast<uint32_t>(BT_));
                PipeBarrier<PIPE_V>();
            }
        }

        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;
        uint64_t lBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0) + rowBegin * BT_;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            LocalTensor<T> aqkTyped = GateQTyped(0);
            if (validElemCount > 0) {
                Muls(aqkMat, aqkMat, scale_, static_cast<uint32_t>(validElemCount));
                PipeBarrier<PIPE_V>();
                ClampFp32ToOutputType(aqkMat, static_cast<uint32_t>(validElemCount));
                Cast(aqkTyped, aqkMat, RoundMode::CAST_RINT, static_cast<uint32_t>(validElemCount));
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            if (validElemCount > 0) {
                CopyVectorOut(o_, AOffset(b, hv, token, 0), aqkTyped, validElemCount);
            }
            DataCopy(solveWorkspace_[xBase], xMat, static_cast<uint32_t>(elemCount));
            if (storeLToScratch) {
                DataCopy(solveWorkspace_[lBase], akkMat, static_cast<uint32_t>(elemCount));
            }
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            return;
        }
#endif
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        if (validRowCount > 0) {
            DataCopy(aqk_[AOffset(b, hv, token, 0)], aqkMat, static_cast<uint32_t>(validElemCount));
            if (storeLToAkk) {
                DataCopy(akk_[AOffset(b, hv, token, 0)], akkMat, static_cast<uint32_t>(validElemCount));
            }
        }
        DataCopy(solveWorkspace_[xBase], xMat, static_cast<uint32_t>(elemCount));
        if (storeLToScratch) {
            DataCopy(solveWorkspace_[lBase], akkMat, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void CubeGemmSolveSub(GlobalTensor<float> &tensorA, uint64_t baseA, uint64_t rowA, uint64_t colA,
                                            GlobalTensor<float> &tensorB, uint64_t baseB, uint64_t rowB, uint64_t colB,
                                            GlobalTensor<float> &tensorC, uint64_t baseC, uint64_t rowC, uint64_t colC,
                                            uint32_t m, uint32_t n, uint32_t k)
    {
        using ElementA = float;
        using ElementB = float;
        using ElementC = float;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaSolveDispatchPolicy, KdaSolveL1TileShape,
                                                              KdaSolveL0TileShape, ElementA, ElementB, ElementC,
                                                              void, TileCopy>;
        Catlass::Arch::Resource<KdaArchTag> resource;
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(BT_, BT_);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(BT_, BT_);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(BT_, BT_);
        auto tensorLayoutA = tla::MakeTensor(tensorA[baseA], layoutA, Catlass::Arch::PositionGM{});
        auto tensorLayoutB = tla::MakeTensor(tensorB[baseB], layoutB, Catlass::Arch::PositionGM{});
        auto tensorLayoutC = tla::MakeTensor(tensorC[baseC], layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, n, k};
        auto blockA = GetTile(tensorLayoutA, tla::MakeCoord(rowA, colA), tla::MakeShape(shape.m(), shape.k()));
        auto blockB = GetTile(tensorLayoutB, tla::MakeCoord(rowB, colB), tla::MakeShape(shape.k(), shape.n()));
        auto blockC = GetTile(tensorLayoutC, tla::MakeCoord(rowC, colC), tla::MakeShape(shape.m(), shape.n()));
        BlockMmad blockMmad(resource);
        blockMmad(blockA, blockB, blockC, shape);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        SetFlag<HardEvent::FIX_MTE2>(KDA_ARCH35_SOLVE_FIX_EVENT);
        WaitFlag<HardEvent::FIX_MTE2>(KDA_ARCH35_SOLVE_FIX_EVENT);
#else
        PipeBarrier<PIPE_ALL>();
#endif
    }

    __aicore__ inline void AddSolveTmpToX(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                          bool storeAkk)
    {
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        DataCopy(xLocal, h_[xBase], KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(tmpLocal, h_[tmpBase], KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        Add(xLocal, xLocal, tmpLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(h_[xBase], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        if (storeAkk) {
            DataCopy(akk_[AOffset(b, hv, start, 0)], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void AddSolveTmpToXTail(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t curT, bool storeAkk)
    {
        uint64_t elemCount = curT * KDA_SOLVE_BT;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[KDA_SOLVE_MATRIX_ELEMENTS];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        DataCopy(xLocal, h_[xBase], KDA_SOLVE_MATRIX_ELEMENTS);
        DataCopy(tmpLocal, h_[tmpBase], KDA_SOLVE_MATRIX_ELEMENTS);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        Add(xLocal, xLocal, tmpLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(h_[xBase], xLocal, KDA_SOLVE_MATRIX_ELEMENTS);
        if (storeAkk) {
            DataCopy(akk_[AOffset(b, hv, start, 0)], xLocal, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void AddSolveTmpToXRows(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t curT, uint64_t rowBegin, uint64_t rowEnd, bool storeAkk)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t validRowCount = rowBegin < curT ? curT - rowBegin : 0;
        if (validRowCount > rowCount) {
            validRowCount = rowCount;
        }
        uint64_t elemCount = rowCount * BT_;
        uint64_t validElemCount = validRowCount * BT_;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[elemCount];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP) + rowBegin * BT_;
        uint64_t token = start + rowBegin;

        DataCopy(xLocal, solveWorkspace_[xBase], static_cast<uint32_t>(elemCount));
        DataCopy(tmpLocal, solveWorkspace_[tmpBase], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        Add(xLocal, xLocal, tmpLocal, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(solveWorkspace_[xBase], xLocal, static_cast<uint32_t>(elemCount));
        if (storeAkk && validRowCount > 0) {
            DataCopy(akk_[AOffset(b, hv, token, 0)], xLocal, static_cast<uint32_t>(validElemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void AddSolveTmpToXDiagRows(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                  uint64_t rowBegin, uint64_t rowEnd, bool storeAkk)
    {
        uint64_t rowCount = rowEnd - rowBegin;
        if (rowCount == 0) {
            return;
        }
        uint64_t elemCount = rowCount * BT_;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> xLocal = arena;
        LocalTensor<float> tmpLocal = arena[elemCount];
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP) + rowBegin * BT_;
        uint64_t token = start + rowBegin;

        DataCopy(xLocal, solveWorkspace_[xBase], static_cast<uint32_t>(elemCount));
        DataCopy(tmpLocal, solveWorkspace_[tmpBase], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

        for (uint64_t localRow = 0; localRow < rowCount; ++localRow) {
            uint64_t row = rowBegin + localRow;
            uint64_t col = (row / KDA_SOLVE_DIAG_BT) * KDA_SOLVE_DIAG_BT;
            uint64_t offset = localRow * BT_ + col;
            Add(xLocal[offset], xLocal[offset], tmpLocal[offset], KDA_SOLVE_DIAG_BT);
            PipeBarrier<PIPE_V>();
        }

        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(solveWorkspace_[xBase], xLocal, static_cast<uint32_t>(elemCount));
        if (storeAkk) {
            DataCopy(akk_[AOffset(b, hv, token, 0)], xLocal, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void StoreSolveXRowsToAkk(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                uint64_t curT, uint64_t rowBegin, uint64_t rowEnd)
    {
        uint64_t validRowCount = rowBegin < curT ? curT - rowBegin : 0;
        uint64_t rowCount = rowEnd - rowBegin;
        if (validRowCount > rowCount) {
            validRowCount = rowCount;
        }
        if (validRowCount == 0) {
            return;
        }
        uint64_t elemCount = validRowCount * BT_;
        LocalTensor<float> xLocal = vecBuf_.Get<float>();
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + rowBegin * BT_;

        DataCopy(xLocal, solveWorkspace_[xBase], static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        DataCopy(akk_[AOffset(b, hv, start + rowBegin, 0)], xLocal, static_cast<uint32_t>(elemCount));
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
    }

    __aicore__ inline void ComputeAkkMergeCube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        uint64_t aiBase = AOffset(b, hv, start, 0);
        uint64_t negABase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        for (uint32_t mergeSize = 2 * KDA_SOLVE_DIAG_BT; mergeSize <= BT_; mergeSize *= 2) {
            uint32_t half = mergeSize / 2;
            for (uint32_t block = 0; block < BT_; block += mergeSize) {
                uint32_t lower = block + half;
                CubeGemmSolveSub(akk_, aiBase, lower, lower, solveWorkspace_, negABase, lower, block,
                                 solveWorkspace_, tmpBase, 0, 0, half, half, half);
                CubeGemmSolveSub(solveWorkspace_, tmpBase, 0, 0, akk_, aiBase, block, block,
                                 akk_, aiBase, lower, block, half, half, half);
            }
        }
    }

    __aicore__ inline void ComputeAkkMergeCubeWorkspace(uint64_t b, uint64_t hv, uint64_t chunkIdx)
    {
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        for (uint32_t mergeSize = 2 * KDA_SOLVE_DIAG_BT; mergeSize <= BT_; mergeSize *= 2) {
            uint32_t half = mergeSize / 2;
            for (uint32_t block = 0; block < BT_; block += mergeSize) {
                uint32_t lower = block + half;
                CubeGemmSolveSub(solveWorkspace_, xBase, lower, lower, solveWorkspace_, xBase, lower, block,
                                 solveWorkspace_, tmpBase, 0, 0, half, half, half);
                CubeGemmSolveSub(solveWorkspace_, tmpBase, 0, 0, solveWorkspace_, xBase, block, block,
                                 solveWorkspace_, xBase, lower, block, half, half, half);
            }
        }
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    template <uint32_t TILE_SIZE, bool TRANSPOSE>
    __aicore__ inline void LoadSolveTile(LocalTensor<float> dst, LocalTensor<float> src)
    {
        static_assert(TILE_SIZE == 16 || TILE_SIZE == 32, "arch35 solve tile must be 16 or 32 rows");
        constexpr uint32_t rowFractals = TILE_SIZE / 16;
        constexpr uint32_t columnFractals = TILE_SIZE / 8;
        LoadData2DParamsV2 loadParams;
        loadParams.mStartPosition = 0;
        loadParams.kStartPosition = 0;
        loadParams.mStep = rowFractals;
        loadParams.kStep = columnFractals;
        loadParams.srcStride = rowFractals;
        loadParams.dstStride = rowFractals;
        loadParams.ifTranspose = TRANSPOSE;
        LoadData(dst, src, loadParams);
    }

    __aicore__ inline void ComputeAkkMergeCubeWorkspaceArch35(uint64_t b, uint64_t hv, uint64_t chunkIdx)
    {
        (void)b;
        (void)hv;
        SetMMLayoutTransform(true);

        using Element = float;
        using LayoutTag = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, Element, LayoutTag, Element,
                                                                LayoutTag, Element, LayoutTag>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, Element, LayoutTagL1A>;

        constexpr uint32_t maxTile = 32;
        constexpr uint32_t tileSlotBytes = maxTile * maxTile * sizeof(Element);
        constexpr uint32_t a0Slot = 0;
        constexpr uint32_t b0Slot = 1;
        constexpr uint32_t a1Slot = 2;
        constexpr uint32_t b1Slot = 3;
        constexpr uint32_t diag0Slot = 4;
        constexpr uint32_t diag1Slot = 5;
        constexpr uint32_t tmp0Slot = 6;
        constexpr uint32_t tmp1Slot = 7;

        Catlass::Arch::Resource<KdaArchTag> resource;
        LocalTensor<Element> l1A0 = resource.l1Buf.template GetBufferByByte<Element>(a0Slot * tileSlotBytes);
        LocalTensor<Element> l1B0 = resource.l1Buf.template GetBufferByByte<Element>(b0Slot * tileSlotBytes);
        LocalTensor<Element> l1A1 = resource.l1Buf.template GetBufferByByte<Element>(a1Slot * tileSlotBytes);
        LocalTensor<Element> l1B1 = resource.l1Buf.template GetBufferByByte<Element>(b1Slot * tileSlotBytes);
        LocalTensor<Element> l1Diag0 =
            resource.l1Buf.template GetBufferByByte<Element>(diag0Slot * tileSlotBytes);
        LocalTensor<Element> l1Diag1 =
            resource.l1Buf.template GetBufferByByte<Element>(diag1Slot * tileSlotBytes);
        LocalTensor<Element> l1Tmp0 =
            resource.l1Buf.template GetBufferByByte<Element>(tmp0Slot * tileSlotBytes);
        LocalTensor<Element> l1Tmp1 =
            resource.l1Buf.template GetBufferByByte<Element>(tmp1Slot * tileSlotBytes);

        LocalTensor<Element> l0A0 = resource.l0ABuf.template GetBufferByByte<Element>(a0Slot * tileSlotBytes);
        LocalTensor<Element> l0B0 = resource.l0BBuf.template GetBufferByByte<Element>(b0Slot * tileSlotBytes);
        LocalTensor<Element> l0A1 = resource.l0ABuf.template GetBufferByByte<Element>(a1Slot * tileSlotBytes);
        LocalTensor<Element> l0B1 = resource.l0BBuf.template GetBufferByByte<Element>(b1Slot * tileSlotBytes);
        LocalTensor<Element> l0A2 =
            resource.l0ABuf.template GetBufferByByte<Element>(diag0Slot * tileSlotBytes);
        LocalTensor<Element> l0B2 =
            resource.l0BBuf.template GetBufferByByte<Element>(diag0Slot * tileSlotBytes);
        LocalTensor<Element> l0A3 =
            resource.l0ABuf.template GetBufferByByte<Element>(diag1Slot * tileSlotBytes);
        LocalTensor<Element> l0B3 =
            resource.l0BBuf.template GetBufferByByte<Element>(diag1Slot * tileSlotBytes);
        LocalTensor<Element> l0C0 = resource.l0CBuf.template GetBufferByByte<Element>(a0Slot * tileSlotBytes);
        LocalTensor<Element> l0C1 = resource.l0CBuf.template GetBufferByByte<Element>(a1Slot * tileSlotBytes);
        LocalTensor<Element> l0C2 =
            resource.l0CBuf.template GetBufferByByte<Element>(diag0Slot * tileSlotBytes);
        LocalTensor<Element> l0C3 =
            resource.l0CBuf.template GetBufferByByte<Element>(diag1Slot * tileSlotBytes);

        TileMmad tileMmad;
        const uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        const uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);
        auto gmLayout = tla::MakeLayout<Element, LayoutTag>(BT_, BT_);
        auto tensorX = tla::MakeTensor(solveWorkspace_[xBase], gmLayout, Catlass::Arch::PositionGM{});
        auto tensorTmp = tla::MakeTensor(solveWorkspace_[tmpBase], gmLayout, Catlass::Arch::PositionGM{});

        constexpr uint32_t tile16 = 16;
        auto shape16 = tla::MakeShape(tile16, tile16);
        auto blockA0 = GetTile(tensorX, tla::MakeCoord(16, 16), shape16);
        auto blockB0 = GetTile(tensorX, tla::MakeCoord(16, 0), shape16);
        auto blockA1 = GetTile(tensorX, tla::MakeCoord(48, 48), shape16);
        auto blockB1 = GetTile(tensorX, tla::MakeCoord(48, 32), shape16);
        auto blockDiag0 = GetTile(tensorX, tla::MakeCoord(0, 0), shape16);
        auto blockDiag1 = GetTile(tensorX, tla::MakeCoord(32, 32), shape16);
        auto blockTmp0 = GetTile(tensorTmp, tla::MakeCoord(0, 0), shape16);
        auto blockTmp1 = GetTile(tensorTmp, tla::MakeCoord(16, 0), shape16);
        auto blockOut0 = GetTile(tensorX, tla::MakeCoord(16, 0), shape16);
        auto blockOut1 = GetTile(tensorX, tla::MakeCoord(48, 32), shape16);

        using CopyGmToL1A16 = typename TileCopy::template CopyGmToL1A<decltype(blockA0)>;
        using CopyGmToL1B16 = typename TileCopy::template CopyGmToL1B<decltype(blockB0)>;
        using CopyL0CToDst16 = typename TileCopy::template CopyL0CToDst<decltype(blockTmp0)>;
        CopyGmToL1A16 copyGmToL1A16;
        CopyGmToL1B16 copyGmToL1B16;
        CopyL0CToDst16 copyL0CToDst16;

        auto tensorL1A0 = tla::MakeTensor(
            l1A0, tla::MakeLayout<Element, LayoutTagL1A>(tile16, tile16), Catlass::Arch::PositionL1{});
        auto tensorL1B0 = tla::MakeTensor(
            l1B0, tla::MakeLayout<Element, LayoutTagL1B>(tile16, tile16), Catlass::Arch::PositionL1{});
        auto tensorL1A1 = tla::MakeTensor(
            l1A1, tla::MakeLayout<Element, LayoutTagL1A>(tile16, tile16), Catlass::Arch::PositionL1{});
        auto tensorL1B1 = tla::MakeTensor(
            l1B1, tla::MakeLayout<Element, LayoutTagL1B>(tile16, tile16), Catlass::Arch::PositionL1{});
        auto tensorL1Diag0 = tla::MakeTensor(
            l1Diag0, tla::MakeLayout<Element, LayoutTagL1B>(tile16, tile16), Catlass::Arch::PositionL1{});
        auto tensorL1Diag1 = tla::MakeTensor(
            l1Diag1, tla::MakeLayout<Element, LayoutTagL1B>(tile16, tile16), Catlass::Arch::PositionL1{});
        auto tensorL1Tmp0 = tla::MakeTensor(
            l1Tmp0, tla::MakeLayout<Element, LayoutTagL1A>(tile16, tile16), Catlass::Arch::PositionL1{});
        auto tensorL1Tmp1 = tla::MakeTensor(
            l1Tmp1, tla::MakeLayout<Element, LayoutTagL1A>(tile16, tile16), Catlass::Arch::PositionL1{});

        auto tensorL0A0 = tla::MakeTensor(
            l0A0, tla::MakeLayout<Element, LayoutTagL0A>(tile16, tile16), Catlass::Arch::PositionL0A{});
        auto tensorL0B0 = tla::MakeTensor(
            l0B0, tla::MakeLayout<Element, LayoutTagL0B>(tile16, tile16), Catlass::Arch::PositionL0B{});
        auto tensorL0A1 = tla::MakeTensor(
            l0A1, tla::MakeLayout<Element, LayoutTagL0A>(tile16, tile16), Catlass::Arch::PositionL0A{});
        auto tensorL0B1 = tla::MakeTensor(
            l0B1, tla::MakeLayout<Element, LayoutTagL0B>(tile16, tile16), Catlass::Arch::PositionL0B{});
        auto tensorL0A2 = tla::MakeTensor(
            l0A2, tla::MakeLayout<Element, LayoutTagL0A>(tile16, tile16), Catlass::Arch::PositionL0A{});
        auto tensorL0B2 = tla::MakeTensor(
            l0B2, tla::MakeLayout<Element, LayoutTagL0B>(tile16, tile16), Catlass::Arch::PositionL0B{});
        auto tensorL0A3 = tla::MakeTensor(
            l0A3, tla::MakeLayout<Element, LayoutTagL0A>(tile16, tile16), Catlass::Arch::PositionL0A{});
        auto tensorL0B3 = tla::MakeTensor(
            l0B3, tla::MakeLayout<Element, LayoutTagL0B>(tile16, tile16), Catlass::Arch::PositionL0B{});
        auto tensorL0C0 = tla::MakeTensor(l0C0, tla::MakeLayoutL0C(tile16, tile16), Catlass::Arch::PositionL0C{});
        auto tensorL0C1 = tla::MakeTensor(l0C1, tla::MakeLayoutL0C(tile16, tile16), Catlass::Arch::PositionL0C{});
        auto tensorL0C2 = tla::MakeTensor(l0C2, tla::MakeLayoutL0C(tile16, tile16), Catlass::Arch::PositionL0C{});
        auto tensorL0C3 = tla::MakeTensor(l0C3, tla::MakeLayoutL0C(tile16, tile16), Catlass::Arch::PositionL0C{});
        uint32_t localRow16 = 0;
        uint32_t localColumn16 = 0;
        auto tileL0A0 = GetTile(tensorL0A0, tla::MakeCoord(localRow16, localColumn16), shape16);
        auto tileL0B0 = GetTile(tensorL0B0, tla::MakeCoord(localRow16, localColumn16), shape16);
        auto tileL0A1 = GetTile(tensorL0A1, tla::MakeCoord(localRow16, localColumn16), shape16);
        auto tileL0B1 = GetTile(tensorL0B1, tla::MakeCoord(localRow16, localColumn16), shape16);
        auto tileL0A2 = GetTile(tensorL0A2, tla::MakeCoord(localRow16, localColumn16), shape16);
        auto tileL0B2 = GetTile(tensorL0B2, tla::MakeCoord(localRow16, localColumn16), shape16);
        auto tileL0A3 = GetTile(tensorL0A3, tla::MakeCoord(localRow16, localColumn16), shape16);
        auto tileL0B3 = GetTile(tensorL0B3, tla::MakeCoord(localRow16, localColumn16), shape16);
        auto tileL0C0 = GetTile(tensorL0C0, tla::MakeCoord(localRow16, localColumn16), shape16);
        auto tileL0C1 = GetTile(tensorL0C1, tla::MakeCoord(localRow16, localColumn16), shape16);
        auto tileL0C2 = GetTile(tensorL0C2, tla::MakeCoord(localRow16, localColumn16), shape16);
        auto tileL0C3 = GetTile(tensorL0C3, tla::MakeCoord(localRow16, localColumn16), shape16);

        copyGmToL1A16(tensorL1A0, blockA0);
        copyGmToL1B16(tensorL1B0, blockB0);
        copyGmToL1A16(tensorL1A1, blockA1);
        copyGmToL1B16(tensorL1B1, blockB1);
        copyGmToL1B16(tensorL1Diag0, blockDiag0);
        copyGmToL1B16(tensorL1Diag1, blockDiag1);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);

        LoadSolveTile<tile16, false>(l0A0, l1A0);
        LoadSolveTile<tile16, true>(l0B0, l1B0);
        LoadSolveTile<tile16, false>(l0A1, l1A1);
        LoadSolveTile<tile16, true>(l0B1, l1B1);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        tileMmad(tileL0C0, tileL0A0, tileL0B0, tile16, tile16, tile16, true, 0b11);
        SetFlag<HardEvent::M_FIX>(0);
        tileMmad(tileL0C1, tileL0A1, tileL0B1, tile16, tile16, tile16, true, 0b11);
        SetFlag<HardEvent::M_FIX>(1);
        WaitFlag<HardEvent::M_FIX>(0);
        copyL0CToDst16(blockTmp0, tileL0C0, 0b11);
        SetFlag<HardEvent::FIX_MTE2>(0);
        WaitFlag<HardEvent::M_FIX>(1);
        copyL0CToDst16(blockTmp1, tileL0C1, 0b11);
        SetFlag<HardEvent::FIX_MTE2>(1);
        WaitFlag<HardEvent::FIX_MTE2>(0);
        copyGmToL1A16(tensorL1Tmp0, blockTmp0);
        WaitFlag<HardEvent::FIX_MTE2>(1);
        copyGmToL1A16(tensorL1Tmp1, blockTmp1);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);

        LoadSolveTile<tile16, false>(l0A2, l1Tmp0);
        LoadSolveTile<tile16, true>(l0B2, l1Diag0);
        LoadSolveTile<tile16, false>(l0A3, l1Tmp1);
        LoadSolveTile<tile16, true>(l0B3, l1Diag1);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        tileMmad(tileL0C2, tileL0A2, tileL0B2, tile16, tile16, tile16, true, 0b11);
        SetFlag<HardEvent::M_FIX>(2);
        tileMmad(tileL0C3, tileL0A3, tileL0B3, tile16, tile16, tile16, true, 0b11);
        SetFlag<HardEvent::M_FIX>(3);
        WaitFlag<HardEvent::M_FIX>(2);
        copyL0CToDst16(blockOut0, tileL0C2, 0b11);
        SetFlag<HardEvent::FIX_MTE2>(2);
        WaitFlag<HardEvent::M_FIX>(3);
        copyL0CToDst16(blockOut1, tileL0C3, 0b11);
        SetFlag<HardEvent::FIX_MTE2>(3);

        constexpr uint32_t tile32 = 32;
        auto shape32 = tla::MakeShape(tile32, tile32);
        auto blockA32 = GetTile(tensorX, tla::MakeCoord(32, 32), shape32);
        auto blockB32 = GetTile(tensorX, tla::MakeCoord(32, 0), shape32);
        auto blockDiag32 = GetTile(tensorX, tla::MakeCoord(0, 0), shape32);
        auto blockTmp32 = GetTile(tensorTmp, tla::MakeCoord(0, 0), shape32);
        auto blockOut32 = GetTile(tensorX, tla::MakeCoord(32, 0), shape32);
        using CopyGmToL1A32 = typename TileCopy::template CopyGmToL1A<decltype(blockA32)>;
        using CopyGmToL1B32 = typename TileCopy::template CopyGmToL1B<decltype(blockB32)>;
        using CopyL0CToDst32 = typename TileCopy::template CopyL0CToDst<decltype(blockTmp32)>;
        CopyGmToL1A32 copyGmToL1A32;
        CopyGmToL1B32 copyGmToL1B32;
        CopyL0CToDst32 copyL0CToDst32;

        auto tensorL1A32 = tla::MakeTensor(
            l1A0, tla::MakeLayout<Element, LayoutTagL1A>(tile32, tile32), Catlass::Arch::PositionL1{});
        auto tensorL1B32 = tla::MakeTensor(
            l1B0, tla::MakeLayout<Element, LayoutTagL1B>(tile32, tile32), Catlass::Arch::PositionL1{});
        auto tensorL1Diag32 = tla::MakeTensor(
            l1Diag0, tla::MakeLayout<Element, LayoutTagL1B>(tile32, tile32), Catlass::Arch::PositionL1{});
        auto tensorL1Tmp32 = tla::MakeTensor(
            l1Tmp0, tla::MakeLayout<Element, LayoutTagL1A>(tile32, tile32), Catlass::Arch::PositionL1{});
        auto tensorL0A32 = tla::MakeTensor(
            l0A0, tla::MakeLayout<Element, LayoutTagL0A>(tile32, tile32), Catlass::Arch::PositionL0A{});
        auto tensorL0B32 = tla::MakeTensor(
            l0B0, tla::MakeLayout<Element, LayoutTagL0B>(tile32, tile32), Catlass::Arch::PositionL0B{});
        auto tensorL0C32 = tla::MakeTensor(l0C0, tla::MakeLayoutL0C(tile32, tile32), Catlass::Arch::PositionL0C{});
        uint32_t localRow32 = 0;
        uint32_t localColumn32 = 0;
        auto tileL0A32 = GetTile(tensorL0A32, tla::MakeCoord(localRow32, localColumn32), shape32);
        auto tileL0B32 = GetTile(tensorL0B32, tla::MakeCoord(localRow32, localColumn32), shape32);
        auto tileL0C32 = GetTile(tensorL0C32, tla::MakeCoord(localRow32, localColumn32), shape32);

        WaitFlag<HardEvent::FIX_MTE2>(2);
        WaitFlag<HardEvent::FIX_MTE2>(3);
        copyGmToL1A32(tensorL1A32, blockA32);
        copyGmToL1B32(tensorL1B32, blockB32);
        copyGmToL1B32(tensorL1Diag32, blockDiag32);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadSolveTile<tile32, false>(l0A0, l1A0);
        LoadSolveTile<tile32, true>(l0B0, l1B0);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        tileMmad(tileL0C32, tileL0A32, tileL0B32, tile32, tile32, tile32, true, 0b11);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        copyL0CToDst32(blockTmp32, tileL0C32, 0b11);
        SetFlag<HardEvent::FIX_MTE2>(0);
        WaitFlag<HardEvent::FIX_MTE2>(0);
        copyGmToL1A32(tensorL1Tmp32, blockTmp32);
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
        LoadSolveTile<tile32, false>(l0A0, l1Tmp0);
        LoadSolveTile<tile32, true>(l0B0, l1Diag0);
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        tileMmad(tileL0C32, tileL0A32, tileL0B32, tile32, tile32, tile32, true, 0b11);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);
        copyL0CToDst32(blockOut32, tileL0C32, 0b11);
        SetFlag<HardEvent::FIX_MTE2>(0);
        WaitFlag<HardEvent::FIX_MTE2>(0);
        SetMMLayoutTransform(false);
    }
#endif

    __aicore__ inline void ComputeAkkMergeCubeWorkspaceDispatch(uint64_t b, uint64_t hv, uint64_t chunkIdx)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            ComputeAkkMergeCubeWorkspace(b, hv, chunkIdx);
            return;
        }
#endif
        ComputeAkkMergeCubeWorkspace(b, hv, chunkIdx);
    }

    __aicore__ inline void ComputeAkkInverseMchFull(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start)
    {
        uint64_t aBase = AOffset(b, hv, start, 0);
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t yBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0);
        uint64_t yNextBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y1);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);

        uint32_t diagBlocks = static_cast<uint32_t>(BT_ / KDA_SOLVE_DIAG_BT);
        for (uint32_t block = 0; block < diagBlocks; ++block) {
            uint32_t off = block * KDA_SOLVE_DIAG_BT;
            CubeGemmSolveSub(akk_, aBase, off, off, akk_, aBase, off, off, solveWorkspace_, yBase, off, off,
                             KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
        }
        for (uint32_t iter = 0; iter < KDA_SOLVE_DIAG_MCH_ITERS; ++iter) {
            for (uint32_t block = 0; block < diagBlocks; ++block) {
                uint32_t off = block * KDA_SOLVE_DIAG_BT;
                CubeGemmSolveSub(solveWorkspace_, xBase, off, off, solveWorkspace_, yBase, off, off,
                                 solveWorkspace_, tmpBase, off, off,
                                 KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(mchSyncDoneFlag_);
            if (iter + 1 < KDA_SOLVE_DIAG_MCH_ITERS) {
                for (uint32_t block = 0; block < diagBlocks; ++block) {
                    uint32_t off = block * KDA_SOLVE_DIAG_BT;
                    CubeGemmSolveSub(solveWorkspace_, yBase, off, off, solveWorkspace_, yBase, off, off,
                                     solveWorkspace_, yNextBase, off, off,
                                     KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT, KDA_SOLVE_DIAG_BT);
                }
            }
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(mchSyncReadyFlag_);
            if (iter + 1 < KDA_SOLVE_DIAG_MCH_ITERS) {
                uint64_t oldYBase = yBase;
                yBase = yNextBase;
                yNextBase = oldYBase;
            }
        }
        ComputeAkkMergeCube(b, hv, chunkIdx, start);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(mchSyncDoneFlag_);
    }

    __aicore__ inline void ScaleRowsByBeta(GlobalTensor<T> &src, GlobalTensor<T> &dst, uint64_t b, uint64_t hv,
                                           uint64_t start, uint64_t rowBegin, uint64_t rowCount, uint64_t dim,
                                           LocalTensor<float> &betaLocal, LocalTensor<float> &betaBrcb,
                                           LocalTensor<float> &matrixLocal, bool sourceSequenceMajor = false)
    {
        constexpr uint64_t vecElemsPerRepeat = 64;
        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        uint64_t elemCount = rowCount * dim;
        uint64_t baseOffset = KVOffset(b, hv, start + rowBegin, 0, dim);
        uint64_t sourceOffset = sourceSequenceMajor
                                    ? VInputOffset(b, hv, start + rowBegin, 0)
                                    : baseOffset;
        uint64_t sourceStride = sourceSequenceMajor ? HV_ * dim : dim;

        if constexpr (IsSameType<T, float>::value) {
            CopyRowsIn(matrixLocal, src, sourceOffset, rowCount, dim, sourceStride);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        } else {
            LocalTensor<T> matrixTyped = vecBuf_.Get<T>()[typedOffset];
            CopyRowsIn(matrixTyped, src, sourceOffset, rowCount, dim, sourceStride);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(matrixLocal, matrixTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
        }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        ApplyKdaRowScaleRegbase(
            (__ubuf__ float *)reinterpret_cast<uint64_t>(matrixLocal.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(betaLocal.GetPhyAddr()),
            static_cast<uint16_t>(rowCount), static_cast<uint16_t>(dim));
#else
        uint8_t repeatStride = static_cast<uint8_t>(dim * sizeof(float) / 32);
        for (uint64_t col = 0; col < dim; col += vecElemsPerRepeat) {
            uint64_t mask = dim - col;
            if (mask > vecElemsPerRepeat) {
                mask = vecElemsPerRepeat;
            }
            Mul(matrixLocal[col], matrixLocal[col], betaBrcb, mask, static_cast<uint8_t>(rowCount),
                {1, 1, 0, repeatStride, repeatStride, 1});
        }
        PipeBarrier<PIPE_V>();
#endif

        if constexpr (IsSameType<T, float>::value) {
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(dst[baseOffset], matrixLocal, static_cast<uint32_t>(elemCount));
        } else {
            LocalTensor<T> matrixTyped = vecBuf_.Get<T>()[typedOffset];
            Cast(matrixTyped, matrixLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(dst[baseOffset], matrixTyped, static_cast<uint32_t>(elemCount));
        }
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

    __aicore__ inline void PrepareWuCubeInputs(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT,
                                               uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t rowsPerSubBlock = (curT + subBlockNum - 1) / subBlockNum;
        uint64_t rowBegin = subBlockIdx * rowsPerSubBlock;
        if (rowBegin >= curT) {
            return;
        }
        uint64_t rowCount = curT - rowBegin;
        if (rowCount > rowsPerSubBlock) {
            rowCount = rowsPerSubBlock;
        }
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> betaLocal = arena;
        LocalTensor<float> betaBrcb = arena[KDA_SOLVE_BT];
        LocalTensor<float> matrixLocal = arena[KDA_SOLVE_BT + 512];
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        // Keep each arch35 Cast/regbase panel within an 8K-element UB instruction span.
        constexpr uint64_t maxScaleElements = 8192;
        if (rowCount * K_ > maxScaleElements || rowCount * V_ > maxScaleElements) {
            constexpr uint64_t tileRows = 16;
            for (uint64_t tileRow = 0; tileRow < rowCount; tileRow += tileRows) {
                uint64_t tileCount = rowCount - tileRow;
                if (tileCount > tileRows) {
                    tileCount = tileRows;
                }
                LoadAsFloatRow(beta_, BetaOffset(b, hv, start + rowBegin + tileRow), betaLocal, tileCount);
                ScaleRowsByBeta(w_, w_, b, hv, start, rowBegin + tileRow, tileCount, K_,
                                betaLocal, betaBrcb, matrixLocal);
                ScaleRowsByBeta(v_, vNew_, b, hv, start, rowBegin + tileRow, tileCount, V_,
                                betaLocal, betaBrcb, matrixLocal, inputSequenceMajor_);
            }
            return;
        }
#endif
        LoadAsFloatRow(beta_, BetaOffset(b, hv, start + rowBegin), betaLocal, rowCount);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
        Brcb(betaBrcb, betaLocal, static_cast<uint8_t>((rowCount + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();
#endif
        ScaleRowsByBeta(w_, w_, b, hv, start, rowBegin, rowCount, K_, betaLocal, betaBrcb, matrixLocal);
        ScaleRowsByBeta(v_, vNew_, b, hv, start, rowBegin, rowCount, V_, betaLocal, betaBrcb,
                        matrixLocal, inputSequenceMajor_);
    }

    __aicore__ inline void FinalizePrepareIntermediates(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                        uint64_t start, uint64_t curT,
                                                        uint64_t subBlockIdx, uint64_t subBlockNum)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        constexpr bool qgScaledAlreadyStored =
            SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128;
#else
        constexpr bool qgScaledAlreadyStored = false;
#endif
        constexpr uint64_t tileRows = 32;
        // Keep tail rows on the same AIV that owns their padded solve rows. Splitting by curT would
        // move short-tail export to AIV1 while AIV0 is still writing the solved matrix.
        const uint64_t rowBegin = (BT_ * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (BT_ * (subBlockIdx + 1)) / subBlockNum;
        if (rowEnd > curT) {
            rowEnd = curT;
        }
        if (rowBegin >= rowEnd) {
            return;
        }
        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += tileRows) {
            const uint64_t rows = (rowEnd - tileRow) > tileRows ? tileRows : (rowEnd - tileRow);
            const uint64_t matrixElems = rows * BT_;
            const uint64_t qgElems = rows * K_;
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> aqkLocal = arena;
            LocalTensor<float> akkLocal = arena[matrixElems];
            LocalTensor<float> qgLocal = arena[2 * matrixElems];
            const uint64_t typedOffset =
                (2 * matrixElems + qgElems) * sizeof(float) / sizeof(T);
            LocalTensor<T> typedBase = vecBuf_.Get<T>()[typedOffset];
            LocalTensor<T> aqkTyped = typedBase;
            LocalTensor<T> akkTyped = typedBase[matrixElems];
            LocalTensor<T> qgTyped = typedBase[2 * matrixElems];

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
                const uint64_t xBase =
                    SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X) + tileRow * BT_;
                CopyVectorIn(akkLocal, solveWorkspace_, xBase, matrixElems);
            } else {
                CopyVectorIn(aqkLocal, aqk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
                CopyVectorIn(akkLocal, akk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
            }
#else
            CopyVectorIn(aqkLocal, aqk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
            CopyVectorIn(akkLocal, akk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
#endif
            if constexpr (!qgScaledAlreadyStored) {
                CopyVectorIn(qgTyped, qg_, KVOffset(b, hv, start + tileRow, 0, K_), qgElems);
            }
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
                Muls(aqkLocal, aqkLocal, scale_, static_cast<uint32_t>(matrixElems));
            }
#else
            Muls(aqkLocal, aqkLocal, scale_, static_cast<uint32_t>(matrixElems));
#endif
            if constexpr (!qgScaledAlreadyStored) {
                Cast(qgLocal, qgTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(qgElems));
                PipeBarrier<PIPE_V>();
                Muls(qgLocal, qgLocal, scale_, static_cast<uint32_t>(qgElems));
                PipeBarrier<PIPE_V>();
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
                ClampFp32ToOutputType(aqkLocal, static_cast<uint32_t>(matrixElems));
            }
#else
            ClampFp32ToOutputType(aqkLocal, static_cast<uint32_t>(matrixElems));
#endif
            ClampFp32ToOutputType(akkLocal, static_cast<uint32_t>(matrixElems));
            if constexpr (!qgScaledAlreadyStored) {
                ClampFp32ToOutputType(qgLocal, static_cast<uint32_t>(qgElems));
            }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
                Cast(aqkTyped, aqkLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(matrixElems));
            }
#else
            Cast(aqkTyped, aqkLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(matrixElems));
#endif
            Cast(akkTyped, akkLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(matrixElems));
            if constexpr (!qgScaledAlreadyStored) {
                Cast(qgTyped, qgLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(qgElems));
            }
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
                CopyVectorOut(o_, AOffset(b, hv, start + tileRow, 0), aqkTyped, matrixElems);
            }
#else
            CopyVectorOut(o_, AOffset(b, hv, start + tileRow, 0), aqkTyped, matrixElems);
#endif
            CopyVectorOut(u_, AOffset(b, hv, start + tileRow, 0), akkTyped, matrixElems);
            if constexpr (!qgScaledAlreadyStored) {
                CopyVectorOut(kg_, KVOffset(b, hv, start + tileRow, 0, K_), qgTyped, qgElems);
            }
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Events_[0]);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        }
    }

    __aicore__ inline bool ResolveFlatChunk(uint64_t task, uint64_t &seq, uint64_t &b, uint64_t &h, uint64_t &hv,
                                            uint64_t &chunkIdx, uint64_t &start, uint64_t &end)
    {
        hv = task % HV_;
        uint64_t flatChunk = task / HV_;
        if (!isVarLen_) {
            seq = flatChunk / NT_;
            b = seq;
            chunkIdx = flatChunk % NT_;
            start = chunkIdx * BT_;
            end = start + BT_;
            if (end > T_) {
                end = T_;
            }
        } else {
            if (!KdaVarlen::ResolveChunkRange(
                    cuSeqlensAddr_, chunkIndicesAddr_, N_, T_, BT_, flatChunk,
                    seq, start, end)) {
                return false;
            }
            b = 0;
            chunkIdx = flatChunk;
        }
        h = hv / (HV_ / H_);
        return start < end;
    }

    __aicore__ inline void ProcessChunkPreAiv(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                              uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                              uint64_t subBlockNum)
    {
        if constexpr (IsSameType<AKK_T, float>::value) {
            ProcessChunkPreAivFp32(b, h, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void JoinAivMte3()
    {
        if constexpr (CORE_TYPE == AscendC::AIV) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            if (!isAivOnly_) {
                if (!headPairMode_) {
                    Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                }
                PipeBarrier<PIPE_MTE3>();
            }
#endif
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void RunAicAfterBothAivReady(uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        if constexpr (CORE_TYPE == AscendC::AIV) {
            (void)subBlockIdx;
            (void)subBlockNum;
            JoinAivMte3();
            if constexpr (SAFE_GATE) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncReadyFlag_);
                Catlass::Arch::CrossCoreWaitFlag(syncDoneFlag_);
            } else {
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(mchSyncReadyFlag_);
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(mchSyncDoneFlag_);
            }
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void SignalAicSolveReady()
    {
        if constexpr (CORE_TYPE == AscendC::AIV) {
            JoinAivMte3();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncReadyFlag_);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void WaitAicSolveDone()
    {
        if constexpr (CORE_TYPE == AscendC::AIV) {
            Catlass::Arch::CrossCoreWaitFlag(syncDoneFlag_);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void SignalPostWuReady()
    {
        if constexpr (CORE_TYPE == AscendC::AIV) {
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(postWuReadyFlag_);
        }
    }

    __aicore__ inline void ProcessChunkPreAivFp32(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                                  uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                                  uint64_t subBlockNum, bool deferSafeSolve = false,
                                                  bool waitPendingSafeSolve = false,
                                                  uint64_t scoreLane = 0, bool pairHeads = false)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        if constexpr (IsSameType<T, float>::value) {
            return;
        }

        if (K_ < 16) {
            return;
        }
        MaterializeRawGateChunkArch35(b, hv, start, curT);
        bool usePostWuCube = UsePostWuCube(curT);
        bool useAkkCubeSolve = UseAkkCubeSolve(curT);
        uint64_t solveRowBegin = 0;
        uint64_t solveRowEnd = 0;
        GetSolveRowRange(BT_, subBlockIdx, subBlockNum, solveRowBegin, solveRowEnd);
        // Safe-gate score factors need the bounded 16-row reference span.
        // A single 64-row reference loses BF16 dynamic range for valid gates.
        const bool useFullChunkScore = false;
        uint64_t scoreBlockSize = useFullChunkScore ? curT : ScoreRefBlockSize();
        uint64_t scoreBlockCount = (curT + scoreBlockSize - 1) / scoreBlockSize;
        uint64_t pipelineBlockCount = useFullChunkScore
            ? scoreBlockCount
            : (scoreBlockCount + KDA_SCORE_QUEUE_DEPTH - 1) / KDA_SCORE_QUEUE_DEPTH *
                  KDA_SCORE_QUEUE_DEPTH;
        const bool useDirectScoreUb =
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KDA_ARCH35_ENABLE_DIRECT_SCORE_UB && pairHeads && curT == 64 && scoreBlockCount == 2;
#else
            false;
#endif
        bool firstSolveRowsPrepared = false;
        for (uint64_t block = 0; block < pipelineBlockCount; ++block) {
            if (block < scoreBlockCount) {
                uint64_t rowBegin = block * scoreBlockSize;
                uint64_t rowCount = useFullChunkScore
                    ? curT - rowBegin
                    : ScoreRowBlockCount(curT, rowBegin);
                uint64_t refToken = ScoreRefToken(start, curT, rowBegin, rowCount);
                uint64_t queueSlot = useFullChunkScore
                    ? activeSolveSlot_ / KDA_SCORE_LANES
                    : block % KDA_SCORE_QUEUE_DEPTH;
                uint64_t scoreSlot = ScoreScratchSlot(queueSlot, scoreLane, pairHeads);
                PrepareGateProducts(b, h, hv, start, curT, subBlockIdx, subBlockNum, true, refToken,
                                    rowBegin + rowCount, true, scoreSlot,
                                    rowBegin, rowCount);
            }
            JoinAivMte3();
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_);
            if (block > 0) {
                if constexpr (SAFE_GATE) {
                    if (waitPendingSafeSolve) {
                        WaitAicSolveDone();
                        waitPendingSafeSolve = false;
                    }
                }
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
                    if (useDirectScoreUb && useAkkCubeSolve && block == 1) {
                        ProcessDirectScoreSolveRowsArch35(
                            b, hv, chunkIdx, start, 0, 0);
                        firstSolveRowsPrepared = true;
                    } else if (pairHeads && useAkkCubeSolve && block == 1) {
                        uint64_t firstRowBegin = 0;
                        uint64_t firstRowEnd = 0;
                        GetSolveRowRange(
                            BT_, 0, KDA_SCORE_LANES, firstRowBegin, firstRowEnd);
                        PrepareAqkAkkSolveInputRows(
                            b, hv, chunkIdx, start, curT,
                            firstRowBegin, firstRowEnd, false, false);
                        firstSolveRowsPrepared = true;
                    }
                }
#endif
            }
        }
        bool fusedScoreWriteback = false;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        fusedScoreWriteback = SAFE_GATE && BT_ == 64 && K_ == 128 && V_ == 128;
#endif
        if (!fusedScoreWriteback) {
            // The final score MMAD only consumes scoreWorkspace_. Run the
            // independent gate writeback while AIC drains its MMAD/Fixpipe path.
            PrepareGateProducts(b, h, hv, start, curT, subBlockIdx, subBlockNum);
        }
        if (pipelineBlockCount > 0) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (useDirectScoreUb && useAkkCubeSolve) {
            ProcessDirectScoreSolveRowsArch35(
                b, hv, chunkIdx, start, KDA_DIRECT_SCORE_ROWS, 1);
        }
#endif
        if constexpr (SAFE_GATE) {
            if (waitPendingSafeSolve) {
                WaitAicSolveDone();
                waitPendingSafeSolve = false;
            }
        }
        if (useAkkCubeSolve) {
            bool fullChunk = curT == BT_;
            if constexpr (SAFE_GATE) {
                if (pairHeads) {
                    if (!useDirectScoreUb) {
                        uint64_t firstRowPart = firstSolveRowsPrepared ? 1 : 0;
                        for (uint64_t rowPart = firstRowPart;
                             rowPart < KDA_SCORE_LANES; ++rowPart) {
                            uint64_t pairRowBegin = 0;
                            uint64_t pairRowEnd = 0;
                            GetSolveRowRange(
                                BT_, rowPart, KDA_SCORE_LANES, pairRowBegin, pairRowEnd);
                            PrepareAqkAkkSolveInputRows(
                                b, hv, chunkIdx, start, curT,
                                pairRowBegin, pairRowEnd, false, false);
                        }
                    }
                } else {
                    PrepareAqkAkkSolveInputRows(
                        b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd, false, false);
                }
                if (deferSafeSolve) {
                    SignalAicSolveReady();
                    return;
                }
                RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
                    StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
                }
#else
                StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
#endif
            } else {
                PrepareAqkAkkSolveInputRows(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd,
                                            fullChunk, false);
                if (!fullChunk) {
                    RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
                    StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
                } else {
                    uint32_t solveIters = KDA_SOLVE_DIAG_MCH_ITERS;
                    RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
                    for (uint32_t iter = 0; iter < solveIters; ++iter) {
                        AddSolveTmpToXDiagRows(b, hv, chunkIdx, start, solveRowBegin, solveRowEnd,
                                               iter + 1 == solveIters);
                        RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
                    }
                }
            }
        }
        // Host validation guarantees every accepted shape has enough workspace for this cube path.
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            if (HV_ % KDA_SCORE_LANES != 0) {
                PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
            }
        } else {
            PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
        }
#else
        PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
#endif
        FinalizePrepareIntermediates(b, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);
    }

    __aicore__ inline void FinishDeferredSafeChunk(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                   uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                                   uint64_t subBlockNum)
    {
        uint64_t curT = end - start;
        uint64_t solveRowBegin = 0;
        uint64_t solveRowEnd = 0;
        GetSolveRowRange(BT_, subBlockIdx, subBlockNum, solveRowBegin, solveRowEnd);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (!(SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128)) {
            StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
        }
#else
        StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            if (HV_ % KDA_SCORE_LANES != 0) {
                PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
            }
        } else {
            PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
        }
#else
        PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
#endif
        FinalizePrepareIntermediates(b, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);
    }

    __aicore__ inline void FinishDeferredSafeChunkPair(
        uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start, uint64_t end)
    {
        FinishDeferredSafeChunk(b, hv, chunkIdx, start, end, 0, 1);
    }

    __aicore__ inline void ProcessChunkPreAic(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t end)
    {
        if constexpr (IsSameType<AKK_T, float>::value) {
            ProcessChunkPreAicFp32(b, hv, chunkIdx, start, end);
        }
    }

    __aicore__ inline void ProcessChunkPreAicFp32(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                  uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0 || K_ < 16) {
            return;
        }
        uint64_t scoreBlockSize = ScoreRefBlockSize();
        uint64_t scoreBlockCount = (curT + scoreBlockSize - 1) / scoreBlockSize;
        uint64_t pipelineBlockCount =
            (scoreBlockCount + KDA_SCORE_QUEUE_DEPTH - 1) / KDA_SCORE_QUEUE_DEPTH * KDA_SCORE_QUEUE_DEPTH;
        for (uint64_t block = 0; block < pipelineBlockCount; ++block) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
            if (block < scoreBlockCount) {
                uint64_t rowBegin = block * scoreBlockSize;
                uint64_t rowCount = ScoreRowBlockCount(curT, rowBegin);
                ComputeRawAqkAkkCubeBlock(b, hv, chunkIdx, start, curT, rowBegin, rowCount, true,
                                          block % KDA_SCORE_QUEUE_DEPTH, rowBegin + rowCount);
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
        }
        bool usePostWuCube = UsePostWuCube(curT);
        bool useAkkCubeSolve = UseAkkCubeSolve(curT);
        if (useAkkCubeSolve) {
            if constexpr (SAFE_GATE) {
                Catlass::Arch::CrossCoreWaitFlag(syncReadyFlag_);
                ComputeAkkMergeCubeWorkspaceDispatch(b, hv, chunkIdx);
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(syncDoneFlag_);
            } else {
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(mchSyncReadyFlag_);
                if (curT == BT_) {
                    ComputeAkkInverseMchFull(b, hv, chunkIdx, start);
                } else {
                    ComputeAkkMergeCubeWorkspace(b, hv, chunkIdx);
                    Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(mchSyncDoneFlag_);
                }
            }
        }
        (void)usePostWuCube;
        (void)chunkIdx;
    }

    __aicore__ inline void ProcessChunkPreAicHeadPairFp32(
        uint64_t b, uint64_t hvBase, uint64_t chunkIdx, uint64_t start, uint64_t end,
        uint64_t localTaskIdx)
    {
        uint64_t curT = end - start;
        if (curT == 0 || K_ < 16) {
            return;
        }
        const bool useFullChunkScore = false;
        uint64_t scoreBlockSize = useFullChunkScore ? curT : ScoreRefBlockSize();
        uint64_t scoreBlockCount = (curT + scoreBlockSize - 1) / scoreBlockSize;
        uint64_t pipelineBlockCount = useFullChunkScore
            ? scoreBlockCount
            : (scoreBlockCount + KDA_SCORE_QUEUE_DEPTH - 1) / KDA_SCORE_QUEUE_DEPTH *
                  KDA_SCORE_QUEUE_DEPTH;
        for (uint64_t block = 0; block < pipelineBlockCount; ++block) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
            if (block < scoreBlockCount) {
                uint64_t rowBegin = block * scoreBlockSize;
                uint64_t rowCount = useFullChunkScore
                    ? curT - rowBegin
                    : ScoreRowBlockCount(curT, rowBegin);
                uint64_t queueSlot = useFullChunkScore
                    ? localTaskIdx % KDA_SCORE_QUEUE_DEPTH
                    : block % KDA_SCORE_QUEUE_DEPTH;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
                bool directScoreDispatched = false;
                if constexpr (SAFE_GATE) {
                    if (KDA_ARCH35_ENABLE_DIRECT_SCORE_UB && curT == 64 && rowCount == 32) {
                        uint64_t scoreSlotBase = ScoreScratchSlot(queueSlot, 0, true);
                        if (rowBegin == 0) {
                            ComputeRawAqkAkkCubeStableHeadPairDirectUbArch35<32>(
                                rowBegin, scoreSlotBase, static_cast<uint32_t>(block));
                        } else {
                            ComputeRawAqkAkkCubeStableHeadPairDirectUbArch35<64>(
                                rowBegin, scoreSlotBase, static_cast<uint32_t>(block));
                        }
                        directScoreDispatched = true;
                    }
                }
                if (!directScoreDispatched)
#endif
                {
                    for (uint64_t lane = 0; lane < KDA_SCORE_LANES; ++lane) {
                        uint64_t hv = hvBase + lane;
                        uint64_t scoreSlot = ScoreScratchSlot(queueSlot, lane, true);
                        activeSolveSlot_ =
                            (localTaskIdx % (KDA_SOLVE_PIPELINE_DEPTH / KDA_SCORE_LANES)) *
                                KDA_SCORE_LANES + lane;
                        ComputeRawAqkAkkCubeBlock(
                            b, hv, chunkIdx, start, curT, rowBegin, rowCount, true,
                            scoreSlot, rowBegin + rowCount);
                    }
                }
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
        }

        if (UseAkkCubeSolve(curT)) {
            Catlass::Arch::CrossCoreWaitFlag(syncReadyFlag_);
            for (uint64_t lane = 0; lane < KDA_SCORE_LANES; ++lane) {
                activeSolveSlot_ =
                    (localTaskIdx % (KDA_SOLVE_PIPELINE_DEPTH / KDA_SCORE_LANES)) * KDA_SCORE_LANES + lane;
                ComputeAkkMergeCubeWorkspaceDispatch(b, hvBase + lane, chunkIdx);
            }
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(syncDoneFlag_);
        }
    }

    __aicore__ inline bool ResolveFlatChunkForHv(
        uint64_t flatChunk, uint64_t hv, uint64_t &seq, uint64_t &b, uint64_t &h,
        uint64_t &chunkIdx, uint64_t &start, uint64_t &end)
    {
        if (!isVarLen_) {
            seq = flatChunk / NT_;
            b = seq;
            chunkIdx = flatChunk % NT_;
            start = chunkIdx * BT_;
            end = start + BT_;
            if (end > T_) {
                end = T_;
            }
        } else {
            if (!KdaVarlen::ResolveChunkRange(
                    cuSeqlensAddr_, chunkIndicesAddr_, N_, T_, BT_, flatChunk,
                    seq, start, end)) {
                return false;
            }
            b = 0;
            chunkIdx = flatChunk;
        }
        h = hv / (HV_ / H_);
        return start < end;
    }

    __aicore__ inline void ProcessPreAivHeadPair()
    {
        const uint64_t subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        const uint64_t coreNum = usedCoreNum_;
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        const uint64_t chunkCount = isVarLen_ ? NT_ : B_ * NT_;
        const uint64_t headWindows = HV_ / KDA_SCORE_LANES;
        const uint64_t taskNum = chunkCount * headWindows;
        bool pendingValid = false;
        uint64_t pendingB = 0;
        uint64_t pendingHv = 0;
        uint64_t pendingChunkIdx = 0;
        uint64_t pendingStart = 0;
        uint64_t pendingEnd = 0;
        uint64_t pendingSlot = 0;
        uint64_t localTaskIdx = 0;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128) {
            if (KDA_ARCH35_ENABLE_DIRECT_SCORE_UB) {
                InitializeDirectScoreUbArch35();
            }
        }
#endif

        for (uint64_t task = coreIdx; task < taskNum; task += coreNum, ++localTaskIdx) {
            uint64_t flatChunk = task / headWindows;
            uint64_t hv = (task % headWindows) * KDA_SCORE_LANES + subBlockIdx;
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (!ResolveFlatChunkForHv(flatChunk, hv, seq, b, h, chunkIdx, start, end)) {
                continue;
            }
            (void)seq;
            uint64_t currentSlot =
                (localTaskIdx % (KDA_SOLVE_PIPELINE_DEPTH / KDA_SCORE_LANES)) *
                    KDA_SCORE_LANES +
                subBlockIdx;
            activeSolveSlot_ = currentSlot;
            bool deferSolve = UseAkkCubeSolve(end - start);
            ProcessChunkPreAivFp32(
                b, h, hv, chunkIdx, start, end, 0, 1, deferSolve, pendingValid, subBlockIdx, true);
            if (pendingValid) {
                activeSolveSlot_ = pendingSlot;
                FinishDeferredSafeChunkPair(
                    pendingB, pendingHv, pendingChunkIdx, pendingStart, pendingEnd);
                if (fusePostWu_) {
                    SignalPostWuReady();
                }
            }
            pendingValid = deferSolve;
            if (pendingValid) {
                pendingB = b;
                pendingHv = hv;
                pendingChunkIdx = chunkIdx;
                pendingStart = start;
                pendingEnd = end;
                pendingSlot = currentSlot;
            }
        }
        if (pendingValid) {
            WaitAicSolveDone();
            activeSolveSlot_ = pendingSlot;
            FinishDeferredSafeChunkPair(
                pendingB, pendingHv, pendingChunkIdx, pendingStart, pendingEnd);
            if (fusePostWu_) {
                SignalPostWuReady();
            }
        }
    }

    __aicore__ inline void ProcessPreAicHeadPair()
    {
        const uint64_t chunkCount = isVarLen_ ? NT_ : B_ * NT_;
        const uint64_t headWindows = HV_ / KDA_SCORE_LANES;
        const uint64_t taskNum = chunkCount * headWindows;
        const uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t localTaskIdx = 0;
        for (uint64_t task = GetBlockIdx(); task < taskNum; task += coreNum, ++localTaskIdx) {
            uint64_t flatChunk = task / headWindows;
            uint64_t hvBase = (task % headWindows) * KDA_SCORE_LANES;
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (ResolveFlatChunkForHv(flatChunk, hvBase, seq, b, h, chunkIdx, start, end)) {
                (void)seq;
                (void)h;
                ProcessChunkPreAicHeadPairFp32(b, hvBase, chunkIdx, start, end, localTaskIdx);
            }
        }
    }

    template <typename PostWuOp>
    __aicore__ inline void ProcessPreAicHeadPairFused(PostWuOp &postWu)
    {
        const uint64_t chunkCount = isVarLen_ ? NT_ : B_ * NT_;
        const uint64_t headWindows = HV_ / KDA_SCORE_LANES;
        const uint64_t taskNum = chunkCount * headWindows;
        const uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t batchB[KDA_POST_QUEUE_STORAGE];
        uint64_t batchHvBase[KDA_POST_QUEUE_STORAGE];
        uint64_t batchStart[KDA_POST_QUEUE_STORAGE];
        uint64_t batchEnd[KDA_POST_QUEUE_STORAGE];
        uint16_t batchCount = 0;
        uint64_t localTaskIdx = 0;

        for (uint64_t task = GetBlockIdx(); task < taskNum; task += coreNum, ++localTaskIdx) {
            uint64_t flatChunk = task / headWindows;
            uint64_t hvBase = (task % headWindows) * KDA_SCORE_LANES;
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (!ResolveFlatChunkForHv(flatChunk, hvBase, seq, b, h, chunkIdx, start, end)) {
                continue;
            }
            (void)seq;
            (void)h;
            ProcessChunkPreAicHeadPairFp32(b, hvBase, chunkIdx, start, end, localTaskIdx);
            batchB[batchCount] = b;
            batchHvBase[batchCount] = hvBase;
            batchStart[batchCount] = start;
            batchEnd[batchCount] = end;
            ++batchCount;

            if (batchCount == KDA_POST_QUEUE_STORAGE) {
                for (uint16_t i = 0; i < KDA_POST_QUEUE_DEPTH; ++i) {
                    Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(postWuReadyFlag_);
                }
                postWu.ProcessPreparedHeadPairBatchArch35(
                    batchB, batchHvBase, batchStart, batchEnd,
                    KDA_POST_QUEUE_DEPTH);
                batchB[0] = batchB[KDA_POST_QUEUE_DEPTH];
                batchHvBase[0] = batchHvBase[KDA_POST_QUEUE_DEPTH];
                batchStart[0] = batchStart[KDA_POST_QUEUE_DEPTH];
                batchEnd[0] = batchEnd[KDA_POST_QUEUE_DEPTH];
                batchCount = 1;
            }
        }

        if (batchCount > 0) {
            for (uint16_t i = 0; i < batchCount; ++i) {
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(postWuReadyFlag_);
            }
            postWu.ProcessPreparedHeadPairBatchArch35(
                batchB, batchHvBase, batchStart, batchEnd,
                batchCount);
        }
    }

    __aicore__ inline void ProcessPreAiv()
    {
        if constexpr (IsSameType<T, float>::value) {
            isAivOnly_ = true;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE && !IsSameType<T, float>::value) {
            if (headPairMode_ && !isAivOnly_) {
                ProcessPreAivHeadPair();
                return;
            }
        }
#endif
        uint64_t subBlockNum = isAivOnly_ ? 1 : static_cast<uint64_t>(GetSubBlockNum());
        if (subBlockNum == 0) {
            return;
        }
        uint64_t subBlockIdx = isAivOnly_ ? 0 : static_cast<uint64_t>(GetSubBlockIdx());
        uint64_t coreNum = isAivOnly_ ? static_cast<uint64_t>(GetBlockNum()) : usedCoreNum_;
        uint64_t coreIdx = isAivOnly_ ? static_cast<uint64_t>(GetBlockIdx()) :
                                        static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
        if constexpr (SAFE_GATE && !IsSameType<T, float>::value) {
            bool pendingValid = false;
            uint64_t pendingB = 0;
            uint64_t pendingHv = 0;
            uint64_t pendingChunkIdx = 0;
            uint64_t pendingStart = 0;
            uint64_t pendingEnd = 0;
            uint64_t pendingSlot = 0;
            uint64_t localTaskIdx = 0;
            for (uint64_t task = coreIdx; task < taskNum; task += coreNum, ++localTaskIdx) {
                uint64_t seq = 0;
                uint64_t b = 0;
                uint64_t h = 0;
                uint64_t hv = 0;
                uint64_t chunkIdx = 0;
                uint64_t start = 0;
                uint64_t end = 0;
                if (!ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                    continue;
                }
                (void)seq;
                uint64_t currentSlot = localTaskIdx % KDA_SOLVE_PIPELINE_DEPTH;
                activeSolveSlot_ = currentSlot;
                bool deferSolve = UseAkkCubeSolve(end - start);
                if (!deferSolve && pendingValid) {
                    WaitAicSolveDone();
                    activeSolveSlot_ = pendingSlot;
                    FinishDeferredSafeChunk(pendingB, pendingHv, pendingChunkIdx, pendingStart, pendingEnd,
                                            subBlockIdx, subBlockNum);
                    pendingValid = false;
                    activeSolveSlot_ = currentSlot;
                }
                ProcessChunkPreAivFp32(b, h, hv, chunkIdx, start, end, subBlockIdx, subBlockNum,
                                      deferSolve, pendingValid);
                if (pendingValid) {
                    activeSolveSlot_ = pendingSlot;
                    FinishDeferredSafeChunk(pendingB, pendingHv, pendingChunkIdx, pendingStart, pendingEnd,
                                            subBlockIdx, subBlockNum);
                }
                pendingValid = deferSolve;
                if (pendingValid) {
                    pendingB = b;
                    pendingHv = hv;
                    pendingChunkIdx = chunkIdx;
                    pendingStart = start;
                    pendingEnd = end;
                    pendingSlot = currentSlot;
                }
            }
            if (pendingValid) {
                WaitAicSolveDone();
                activeSolveSlot_ = pendingSlot;
                FinishDeferredSafeChunk(pendingB, pendingHv, pendingChunkIdx, pendingStart, pendingEnd,
                                        subBlockIdx, subBlockNum);
            }
            return;
        }
        for (uint64_t task = coreIdx; task < taskNum; task += coreNum) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                (void)seq;
                ProcessChunkPreAiv(b, h, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
            }
        }
    }

    __aicore__ inline void ProcessPreAic()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (SAFE_GATE) {
            if (headPairMode_) {
                ProcessPreAicHeadPair();
                return;
            }
        }
#endif
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t localTaskIdx = 0;
        for (uint64_t task = GetBlockIdx(); task < taskNum; task += coreNum, ++localTaskIdx) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                if constexpr (SAFE_GATE) {
                    activeSolveSlot_ = localTaskIdx % KDA_SOLVE_PIPELINE_DEPTH;
                }
                (void)seq;
                (void)h;
                ProcessChunkPreAic(b, hv, chunkIdx, start, end);
            }
        }
    }


private:
    GlobalTensor<T> q_;
    GlobalTensor<T> k_;
    GlobalTensor<T> v_;
    GlobalTensor<GK_T> gk_;
    GlobalTensor<float> rawG_;
    GlobalTensor<float> aLog_;
    GlobalTensor<float> dtBias_;
    GlobalTensor<BETA_T> beta_;
    GlobalTensor<float> initialState_;
    GlobalTensor<OUT_T> o_;
    GlobalTensor<float> finalState_;
    GlobalTensor<float> aqk_;
    GlobalTensor<AKK_T> akk_;
    GlobalTensor<T> w_;
    GlobalTensor<OUT_T> u_;
    GlobalTensor<T> qg_;
    GlobalTensor<T> kg_;
    GlobalTensor<T> vNew_;
    GlobalTensor<float> h_;
    GlobalTensor<T> finalKg_;
    GlobalTensor<T> preparedQG_;
    GlobalTensor<T> preparedAqk_;
    GlobalTensor<T> propagatedVNew_;
    GlobalTensor<T> propagatedH_;
    GlobalTensor<float> solveWorkspace_;
    GlobalTensor<SCORE_T> scoreWorkspace_;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> exp2Buf_;
    TBuf<TPosition::VECCALC> vecBuf_;
    TBuf<TPosition::VECCALC> gateWritebackBuf_;
    TEventID mte2ToVEvent_ = 0;
    TEventID vToMte2Event_ = 0;
    TEventID vToMte3Event_ = 0;
    TEventID mte3ToVEvent_ = 0;
    TEventID mte2ToMte3Event_ = 0;
    TEventID vToSEvent_ = 0;
    TEventID mte3ToMte2Events_[KDA_GATE_PIPELINE_DEPTH] = {0, 0, 0};
    bool vectorEventsAllocated_ = false;
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SCORE_QUEUE_DEPTH> scoreReadyFlag_{KDA_SCORE_READY_FLAG0,
                                                                                  KDA_SCORE_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SCORE_QUEUE_DEPTH> scoreDoneFlag_{KDA_SCORE_DONE_FLAG0,
                                                                                 KDA_SCORE_DONE_FLAG1};
    // Solve has one outstanding task per core. Reuse the primary score IDs as an ordered token stream;
    // score credits remain on the reverse IDs, so no additional hardware flag IDs are consumed.
    Catlass::Arch::CrossCoreFlag syncReadyFlag_{KDA_SOLVE_READY_FLAG};
    Catlass::Arch::CrossCoreFlag syncDoneFlag_{KDA_SOLVE_DONE_FLAG};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_POST_QUEUE_DEPTH> postWuReadyFlag_{
        KDA_POST_READY_FLAG, KDA_POST_FREE_FLAG};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SYNC_REVERSE_DEPTH> mchSyncReadyFlag_{
        KDA_SCORE_READY_FLAG0, KDA_SCORE_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SYNC_REVERSE_DEPTH> mchSyncDoneFlag_{
        KDA_SCORE_DONE_FLAG0, KDA_SCORE_DONE_FLAG1};
    uint64_t B_ = 0;
    uint64_t N_ = 0;
    uint64_t H_ = 0;
    uint64_t HV_ = 0;
    uint64_t T_ = 0;
    uint64_t K_ = 0;
    uint64_t V_ = 0;
    uint64_t BT_ = 0;
    uint64_t NT_ = 0;
    float scale_ = 1.0f;
    bool hasInitial_ = false;
    bool isVarLen_ = false;
    bool isAivOnly_ = false;
    bool headPairMode_ = false;
    bool inputSequenceMajor_ = false;
    bool fusePostWu_ = false;
    bool materializeFinalKg_ = false;
    bool computeGateInPrepare_ = false;
    bool hasALog_ = false;
    bool hasDtBias_ = false;
    bool storeQG_ = true;
    float lowerBound_ = -5.0f;
    uint64_t usedCoreNum_ = 1;
    uint64_t solveCoreIdx_ = 0;
    uint64_t activeSolveSlot_ = 0;
    uint64_t activeGateChunkStart_ = 0;
    __gm__ int64_t *chunkIndicesAddr_ = nullptr;
    __gm__ int64_t *cuSeqlensAddr_ = nullptr;
};
} // namespace

template <bool SAFE_GATE, typename T, typename GK_T, typename BETA_T, typename TilingData,
          uint32_t COMPILE_BT = 0, uint32_t COMPILE_K = 0, uint32_t COMPILE_V = 0>
__aicore__ inline void RunChunkKdaPrepare(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR rawG, GM_ADDR aLog,
    GM_ADDR dtBias, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR akk, GM_ADDR qg,
    GM_ADDR qgScaled, GM_ADDR wSeed, GM_ADDR uSeed, GM_ADDR finalKg, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe, bool storeQG = true)
{
    GM_ADDR aqkFp32 = userWorkspace + tiling.prepareAqkFp32Offset;
    GM_ADDR akkFp32 = userWorkspace + tiling.prepareAkkFp32Offset;
    GM_ADDR prepareScratch = userWorkspace + tiling.prepareScratchOffset;

    if ASCEND_IS_AIC {
        ChunkKdaFwdPrepareKernel<SAFE_GATE, T, GK_T, BETA_T, COMPILE_BT, COMPILE_K, COMPILE_V> op;
        op.Init(q, k, v, gk, rawG, aLog, dtBias, beta, initialState, cuSeqlens, chunkIndices,
                nullptr, nullptr, nullptr, nullptr, aqk, userWorkspace, aqkFp32, akkFp32,
                wSeed, akk, qg, qgScaled, uSeed, userWorkspace, finalKg,
                prepareScratch, tiling, &pipe, false, storeQG);
        if (tiling.fusePostWu) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            KdaPostWu::ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> postWu;
            postWu.Init(nullptr, k, nullptr, gk, beta, initialState, cuSeqlens, chunkIndices,
                        wSeed, akk, uSeed, nullptr, userWorkspace, userWorkspace, userWorkspace,
                        akk, wSeed, uSeed, userWorkspace, finalKg, userWorkspace,
                        prepareScratch, prepareScratch, tiling, &pipe, false);
            op.ProcessAicFused(postWu);
#else
            op.ProcessAic();
#endif
        } else {
            op.ProcessAic();
        }
    }
    if ASCEND_IS_AIV {
        ChunkKdaFwdPrepareKernel<SAFE_GATE, T, GK_T, BETA_T, COMPILE_BT, COMPILE_K, COMPILE_V> op;
        op.Init(q, k, v, gk, rawG, aLog, dtBias, beta, initialState, cuSeqlens, chunkIndices,
                nullptr, nullptr, nullptr, nullptr, aqk, userWorkspace, aqkFp32, akkFp32,
                wSeed, akk, qg, qgScaled, uSeed, userWorkspace, finalKg,
                prepareScratch, tiling, &pipe, true, storeQG);
        op.ProcessAiv();
    }
}

} // namespace KdaPrepare
