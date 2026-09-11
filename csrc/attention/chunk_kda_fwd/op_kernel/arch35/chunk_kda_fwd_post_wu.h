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
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
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

using namespace AscendC;

namespace KdaPostWu {
namespace {
using KdaInt64 = tla::Int<64>;
using KdaInt128 = tla::Int<128>;
constexpr float LN2 = 0.69314718055994530942f;
constexpr float KDA_EXP2_CLAMP = 80.0f;
constexpr float KDA_EXP_INPUT_MAX = KDA_EXP2_CLAMP * LN2;
constexpr float KDA_EXP_INPUT_MIN = -KDA_EXP2_CLAMP * LN2;
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
constexpr uint32_t KDA_SOLVE_SCRATCH_SLOTS = 5;
constexpr uint32_t KDA_SOLVE_DIAG_BT = 16;
constexpr uint32_t KDA_SOLVE_DIAG_BLOCKS = KDA_SOLVE_BT / KDA_SOLVE_DIAG_BT;
constexpr uint32_t KDA_SOLVE_DIAG_MCH_ITERS = 3;
constexpr uint32_t KDA_SCORE_REF_BC = 16;
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
constexpr uint32_t KDA_SCORE_QUEUE_DEPTH = 2;
constexpr uint32_t KDA_SYNC_REVERSE_DEPTH = 1;
constexpr uint32_t KDA_SCORE_SCRATCH_PLANES = 3;
constexpr uint32_t KDA_SCORE_SCRATCH_QG = 0;
constexpr uint32_t KDA_SCORE_SCRATCH_W = 1;
constexpr uint32_t KDA_SCORE_SCRATCH_KG = 2;
constexpr uint64_t KDA_WORKSPACE_ALIGN = 512;
constexpr uint32_t KDA_GATE_TILE_ROWS = 32;
constexpr uint32_t KDA_TYPICAL_GATE_TILE_ROWS = 16;
constexpr uint32_t KDA_TYPICAL_GATE_PIPELINE_ROWS = 32;
constexpr uint16_t KDA_TYPICAL_GATE_PIPELINE_STAGES = 3;
constexpr uint32_t KDA_POST_EVENT = 3;
constexpr uint32_t KDA_POST_EVENT_NEXT = 4;
constexpr uint32_t KDA_POST_EVENT_FIX = 5;
constexpr uint32_t KDA_POST_PIPELINE_L1_SLOT_BYTES = 24 * 1024;
constexpr uint32_t KDA_POST_PIPELINE_L1_A_BYTES = 64 * 64 * sizeof(uint16_t);
constexpr uint32_t KDA_POST_PIPELINE_L1_B_BYTES = 64 * 128 * sizeof(uint16_t);
constexpr uint32_t KDA_POST_PIPELINE_L1_U_SLOT_BYTES = 64 * 128 * sizeof(uint16_t);
constexpr uint32_t KDA_POST_PIPELINE_L0_A_SLOT_BYTES = 64 * 64 * sizeof(uint16_t);
constexpr uint32_t KDA_POST_PIPELINE_L0_B_SLOT_BYTES = 64 * 256 * sizeof(uint16_t);
constexpr uint32_t KDA_POST_PIPELINE_L0_C_SLOT_BYTES = 64 * 256 * sizeof(float);
constexpr uint16_t KDA_POST_PIPELINE_STAGE_COUNT = 2;
constexpr uint16_t KDA_POST_FUSED_BATCH_TASKS = 4;
constexpr uint16_t KDA_POST_HEAD_PAIR_LANES = 2;
constexpr uint16_t KDA_POST_PIPELINE_U_EVENT = KDA_POST_EVENT_FIX;
constexpr bool KDA_ENABLE_POST_AIC_PIPELINE = true;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <typename InputT>
__simd_callee__ inline void LoadPostKdaGateRegbasePair(
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
__simd_callee__ inline void StorePostKdaGateRegbasePair(
    __ubuf__ OutputT *dst,
    AscendC::MicroAPI::RegTensor<float> &zeroReg,
    AscendC::MicroAPI::RegTensor<float> &oneReg,
    AscendC::MicroAPI::MaskReg &inputMask,
    AscendC::MicroAPI::MaskReg &floatMask)
{
    using namespace AscendC::MicroAPI;
    if constexpr (std::is_same<OutputT, half>()) {
        Mins(zeroReg, zeroReg, KDA_FP16_MAX, floatMask);
        Mins(oneReg, oneReg, KDA_FP16_MAX, floatMask);
        Maxs(zeroReg, zeroReg, -KDA_FP16_MAX, floatMask);
        Maxs(oneReg, oneReg, -KDA_FP16_MAX, floatMask);
    }
    RegTensor<OutputT> outputReg;
    CastFloat2Half<OutputT>(outputReg, zeroReg, oneReg, floatMask);
    StoreAlign(dst, outputReg, inputMask);
}

template <typename T, typename GK_T>
static __simd_vf__ inline void ComputePostKdaKgRegbase(
    __ubuf__ T *kAndKg, __ubuf__ GK_T *gate, __ubuf__ float *ref,
    uint16_t rows, uint16_t cols)
{
    using namespace AscendC::MicroAPI;
    constexpr uint16_t ELEMENTS_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(T);
    MaskReg floatMask = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < rows; ++row) {
        uint32_t rowOffset = static_cast<uint32_t>(row) * cols;
        for (uint16_t col = 0; col < cols; col += ELEMENTS_PER_REG) {
            uint32_t activeCount = static_cast<uint32_t>(cols - col);
            MaskReg inputMask = UpdateMask<T>(activeCount);
            uint32_t offset = rowOffset + col;

            RegTensor<float> gateZeroReg;
            RegTensor<float> gateOneReg;
            RegTensor<float> refZeroReg;
            RegTensor<float> refOneReg;
            RegTensor<float> expZeroReg;
            RegTensor<float> expOneReg;
            RegTensor<float> inputZeroReg;
            RegTensor<float> inputOneReg;
            RegTensor<float> outputZeroReg;
            RegTensor<float> outputOneReg;

            LoadPostKdaGateRegbasePair<GK_T>(
                gateZeroReg, gateOneReg, gate + offset, inputMask);
            LoadAlign<float, LoadDist::DIST_DINTLV_B32>(
                refZeroReg, refOneReg, ref + col);
            SubFloatTwoReg(expZeroReg, expOneReg, refZeroReg, refOneReg,
                           gateZeroReg, gateOneReg, floatMask);
            Muls(expZeroReg, expZeroReg, LN2, floatMask);
            Muls(expOneReg, expOneReg, LN2, floatMask);
            MinsFloatTwoReg(expZeroReg, expOneReg, expZeroReg, expOneReg,
                            KDA_EXP_INPUT_MAX, floatMask);
            Maxs(expZeroReg, expZeroReg, KDA_EXP_INPUT_MIN, floatMask);
            Maxs(expOneReg, expOneReg, KDA_EXP_INPUT_MIN, floatMask);
            ExpFloatTwoReg(expZeroReg, expOneReg, expZeroReg, expOneReg, floatMask);

            LoadPostKdaGateRegbasePair<T>(
                inputZeroReg, inputOneReg, kAndKg + offset, inputMask);
            MulFloatTwoReg(outputZeroReg, outputOneReg, inputZeroReg, inputOneReg,
                           expZeroReg, expOneReg, floatMask);
            StorePostKdaGateRegbasePair<T>(
                kAndKg + offset, outputZeroReg, outputOneReg, inputMask, floatMask);
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

template <typename T, typename GK_T = float, typename BETA_T = float>
class ChunkKdaFwdPostWuKernel {
public:
    using OUT_T = T;
    using AKK_T = T;
    template <typename TilingData>
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
                                GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR preparedQG, GM_ADDR preparedAqk,
                                GM_ADDR propagatedVNew, GM_ADDR propagatedH, GM_ADDR o, GM_ADDR finalState, GM_ADDR aqk,
                                GM_ADDR akk, GM_ADDR w, GM_ADDR u, GM_ADDR qg, GM_ADDR kg, GM_ADDR vNew, GM_ADDR h,
                                GM_ADDR workspace, const TilingData &tiling, TPipe *pipe,
                                bool initVecBuffers = true)
    {
        pipe_ = pipe;
        q_.SetGlobalBuffer((__gm__ T *)q);
        k_.SetGlobalBuffer((__gm__ T *)k);
        v_.SetGlobalBuffer((__gm__ T *)v);
        gk_.SetGlobalBuffer((__gm__ GK_T *)gk);
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
        solveWorkspace_.SetGlobalBuffer((__gm__ float *)workspace);

        B_ = tiling.batch;
        N_ = tiling.seqNum;
        H_ = tiling.qHeadNum;
        HV_ = tiling.vHeadNum;
        T_ = tiling.seqlen;
        K_ = tiling.kHeadDim;
        V_ = tiling.vHeadDim;
        BT_ = tiling.chunkSize;
        NT_ = tiling.totalChunks;
        scale_ = tiling.scale;
        hasInitial_ = tiling.hasInitialState;
        isVarLen_ = tiling.isVarLen;
        inputSequenceMajor_ = tiling.inputSequenceMajor;
        usedCoreNum_ = tiling.postWuUsedCoreNum;
        if ASCEND_IS_AIV {
            uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
            solveCoreIdx_ = subBlockNum == 0 ? 0 : static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        } else {
            solveCoreIdx_ = static_cast<uint64_t>(GetBlockIdx());
        }
        if (pipe_ != nullptr && initVecBuffers) {
            pipe_->InitBuffer(exp2Buf_, EXP2_UB_BYTES);
            pipe_->InitBuffer(vecBuf_, KDA_VEC_ARENA_ELEMENTS * sizeof(float));
            const uint64_t gateWritebackRows =
                ScoreVectorMaxRows(5 * sizeof(float) + 2 * sizeof(T) + sizeof(GK_T));
            pipe_->InitBuffer(gateWritebackBuf_,
                              static_cast<uint32_t>(gateWritebackRows * K_ *
                                                    (3 * sizeof(T) + sizeof(GK_T))));
            AllocVectorEvents();
        }
    }
    __aicore__ inline void ProcessAiv()
    {
        ProcessPostAiv();
        ReleaseVectorEvents();
    }

    __aicore__ inline void ProcessAic()
    {
        ProcessPostAic();
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void ProcessPreparedFullHeadPairBatchArch35(
        const uint64_t *batchB, const uint64_t *batchHvBase,
        const uint64_t *batchStart, uint16_t taskCount)
    {
        if (taskCount == 0) {
            return;
        }
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        Catlass::Arch::Resource<KdaArchTag> resource;
        const uint16_t itemCount = taskCount * KDA_POST_HEAD_PAIR_LANES;
        uint16_t slot = 0;
        uint16_t usedSlotCount = 1;
        uint16_t taskIdx = 0;
        uint16_t lane = 0;
        uint64_t b = batchB[taskIdx];
        uint64_t hv = batchHvBase[taskIdx] + lane;
        uint64_t start = batchStart[taskIdx];
        InitializePostWuPipelineEvents();
        PrefetchPostWuPipelineArch35(resource, slot, b, hv, start, BT_, false);
        PrefetchPostWuPipelineU(resource, slot, b, hv, start, BT_, false);

        for (uint16_t item = 0; item < itemCount; ++item) {
            const uint16_t nextItem = item + 1;
            if (nextItem < itemCount) {
                const uint16_t nextTaskIdx = nextItem / KDA_POST_HEAD_PAIR_LANES;
                const uint16_t nextLane = nextItem % KDA_POST_HEAD_PAIR_LANES;
                const uint16_t nextSlot = slot ^ 1;
                const bool reuseSlot = nextItem >= KDA_POST_PIPELINE_STAGE_COUNT;
                PrefetchPostWuPipelineArch35(
                    resource, nextSlot, batchB[nextTaskIdx],
                    batchHvBase[nextTaskIdx] + nextLane, batchStart[nextTaskIdx], BT_, reuseSlot);
                PrefetchPostWuPipelineU(
                    resource, nextSlot, batchB[nextTaskIdx],
                    batchHvBase[nextTaskIdx] + nextLane, batchStart[nextTaskIdx], BT_, reuseSlot);
                if (!reuseSlot) {
                    ++usedSlotCount;
                }
            }

            ComputePrefetchedPostWuPipelineArch35(resource, slot, b, hv, start, BT_);
            if (nextItem < itemCount) {
                taskIdx = nextItem / KDA_POST_HEAD_PAIR_LANES;
                lane = nextItem % KDA_POST_HEAD_PAIR_LANES;
                b = batchB[taskIdx];
                hv = batchHvBase[taskIdx] + lane;
                start = batchStart[taskIdx];
                slot ^= 1;
            }
        }
        FinalizePostWuPipelineEvents(usedSlotCount);
    }

    __aicore__ inline void ProcessPreparedTailHeadPairArch35(
        uint64_t b, uint64_t hvBase, uint64_t start, uint64_t curT)
    {
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        Catlass::Arch::Resource<KdaArchTag> resource;
        InitializePostWuPipelineEvents();
        for (uint16_t lane = 0; lane < KDA_POST_HEAD_PAIR_LANES; ++lane) {
            PrefetchPostWuPipelineArch35(
                resource, lane, b, hvBase + lane, start, curT, false);
            PrefetchPostWuPipelineU(
                resource, lane, b, hvBase + lane, start, curT, false);
        }
        for (uint16_t lane = 0; lane < KDA_POST_HEAD_PAIR_LANES; ++lane) {
            ComputePrefetchedPostWuPipelineArch35(
                resource, lane, b, hvBase + lane, start, curT);
        }
        FinalizePostWuPipelineEvents(KDA_POST_HEAD_PAIR_LANES);
    }

    __aicore__ inline void ProcessPreparedTailSingleArch35(
        uint64_t b, uint64_t hv, uint64_t start, uint64_t curT)
    {
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        Catlass::Arch::Resource<KdaArchTag> resource;
        InitializePostWuPipelineSlot(0);
        PrefetchPostWuPipelineArch35(resource, 0, b, hv, start, curT, false);
        PrefetchPostWuPipelineU(resource, 0, b, hv, start, curT, false);
        ComputePrefetchedPostWuPipelineArch35(resource, 0, b, hv, start, curT);
        FinalizePostWuPipelineEvents(1);
    }

    __aicore__ inline void ProcessPreparedHeadPairBatchArch35(
        const uint64_t *batchB, const uint64_t *batchHvBase,
        const uint64_t *batchStart, const uint64_t *batchEnd, uint16_t taskCount)
    {
        uint16_t fullRunBegin = 0;
        for (uint16_t task = 0; task < taskCount; ++task) {
            if (batchEnd[task] - batchStart[task] == BT_) {
                continue;
            }
            if (task > fullRunBegin) {
                ProcessPreparedFullHeadPairBatchArch35(
                    batchB + fullRunBegin, batchHvBase + fullRunBegin,
                    batchStart + fullRunBegin, task - fullRunBegin);
            }
            ProcessPreparedTailHeadPairArch35(
                batchB[task], batchHvBase[task], batchStart[task],
                batchEnd[task] - batchStart[task]);
            fullRunBegin = task + 1;
        }
        if (fullRunBegin < taskCount) {
            ProcessPreparedFullHeadPairBatchArch35(
                batchB + fullRunBegin, batchHvBase + fullRunBegin,
                batchStart + fullRunBegin, taskCount - fullRunBegin);
        }
    }
#endif

private:
    __aicore__ inline void AllocVectorEvents()
    {
        mte2ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte2Event_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        mte3ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
        mte2ToMte3Event_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        mte3ToMte2Event_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
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
        pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        vectorEventsAllocated_ = false;
    }

    __aicore__ inline uint64_t QOffset(uint64_t b, uint64_t h, uint64_t t, uint64_t d) const
    {
        if (inputSequenceMajor_) {
            return ((b * T_ + t) * H_ + h) * K_ + d;
        }
        return ((b * H_ + h) * T_ + t) * K_ + d;
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
        uint64_t matrixElements = BT_ * BT_;
        return solveCoreIdx_ * KDA_SOLVE_SCRATCH_SLOTS * matrixElements + slot * matrixElements;
    }

    __aicore__ inline uint64_t ScoreScratchOffset(uint64_t slot, uint64_t plane, uint64_t t = 0,
                                                  uint64_t d = 0) const
    {
        return (((solveCoreIdx_ * KDA_SCORE_QUEUE_DEPTH + slot) * KDA_SCORE_SCRATCH_PLANES + plane) * BT_ + t) *
                   K_ +
               d;
    }



    __aicore__ inline uint64_t ScoreRefBlockSize() const
    {
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

    __aicore__ inline void ClampFp32ToOutputType(LocalTensor<float> &tensor, uint32_t count)
    {
        if constexpr (IsSameType<T, half>::value) {
            Mins(tensor, tensor, KDA_FP16_MAX, count);
            PipeBarrier<PIPE_V>();
            Maxs(tensor, tensor, -KDA_FP16_MAX, count);
            PipeBarrier<PIPE_V>();
        }
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
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
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


    __aicore__ inline uint64_t ScoreVectorMaxRows(uint64_t bytesPerElem) const
    {
        constexpr uint64_t arenaBytes = static_cast<uint64_t>(KDA_VEC_ARENA_ELEMENTS) * sizeof(float);
        uint64_t maxRows = (arenaBytes / bytesPerElem) / K_;
        if (K_ >= 128 && maxRows > 32) {
            maxRows = 32;
        }
        return maxRows;
    }
    __aicore__ inline bool UsePostWuCube(uint64_t curT) const
    {
        return curT > 0 && curT <= BT_ && (BT_ == 64 || BT_ == 128) && K_ >= 16 && V_ >= 16 &&
               V_ <= 256 && K_ % 16 == 0 && V_ % 16 == 0;
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline bool UsePostWuCubeArch35(uint64_t curT) const
    {
        return curT == 64 && BT_ == 64 && K_ == 128 && V_ == 128;
    }

    __aicore__ inline bool UseFullPostWuPipelineArch35(uint64_t curT) const
    {
        return curT == 64 && BT_ == 64 && K_ == 128 && V_ == 128;
    }

    __aicore__ inline void ComputePostWuCubeFusedArch35(
        uint64_t b, uint64_t hv, uint64_t start, uint64_t curT)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, ElementA, LayoutTagA, ElementB, LayoutTagB, ElementC, LayoutTagC>;

        constexpr uint32_t capacityM = 64;
        constexpr uint32_t n = 128;
        constexpr uint32_t capacityK = 64;
        const uint32_t m = static_cast<uint32_t>(curT);
        const uint32_t k = static_cast<uint32_t>(curT);
        SetLoadDataPaddingValue<T>(static_cast<T>(0));

        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(capacityM, capacityK);
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(capacityK, n);
        auto layoutC = tla::MakeLayout<ElementC, LayoutTagC>(capacityM, n);
        auto tensorA = tla::MakeTensor(
            preparedAqk_[AOffset(b, hv, start, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorW = tla::MakeTensor(
            preparedQG_[KVOffset(b, hv, start, 0, K_)], layoutB, Catlass::Arch::PositionGM{});
        auto tensorU = tla::MakeTensor(
            propagatedVNew_[KVOffset(b, hv, start, 0, V_)], layoutB, Catlass::Arch::PositionGM{});
        auto tensorWOut = tla::MakeTensor(
            w_[KVOffset(b, hv, start, 0, K_)], layoutC, Catlass::Arch::PositionGM{});
        auto tensorUOut = tla::MakeTensor(
            u_[KVOffset(b, hv, start, 0, V_)], layoutC, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto blockU = GetTile(tensorU, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto blockWOut = GetTile(tensorWOut, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto blockUOut = GetTile(tensorUOut, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        Catlass::Arch::Resource<KdaArchTag> resource;
        constexpr uint32_t aBytes = capacityM * capacityK * sizeof(ElementA);
        constexpr uint32_t bBytes = capacityK * n * sizeof(ElementB);
        LocalTensor<ElementA> l1A = resource.l1Buf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l1B0 = resource.l1Buf.template GetBufferByByte<ElementB>(aBytes);
        LocalTensor<ElementB> l1B1 = resource.l1Buf.template GetBufferByByte<ElementB>(aBytes + bBytes);
        LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        LocalTensor<float> l0C = resource.l0CBuf.template GetBufferByByte<float>(0);

        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockA)>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockW)>;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(blockWOut)>;
        using TileMmad =
            Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, ElementA, LayoutTagL1A>;

        auto layoutL1A = tla::MakeLayout<ElementA, LayoutTagL1A>(capacityM, capacityK);
        auto layoutL1B = tla::MakeLayout<ElementB, LayoutTagL1B>(capacityK, n);
        auto layoutL0A = tla::MakeLayout<ElementA, LayoutTagL0A>(capacityM, capacityK);
        auto layoutL0B = tla::MakeLayout<ElementB, LayoutTagL0B>(capacityK, n);
        auto layoutL0C = tla::MakeLayoutL0C(capacityM, n);
        auto tensorL1A = tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1B0 = tla::MakeTensor(l1B0, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL1B1 = tla::MakeTensor(l1B1, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1B0 = GetTile(tensorL1B0, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL1B1 = GetTile(tensorL1B1, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToDst copyL0CToDst;
        TileMmad tileMmad;

        copyGmToL1A(tensorL1A, blockA);
        copyGmToL1B(tensorL1B0, blockW);
        SetFlag<HardEvent::MTE2_MTE1>(KDA_POST_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(KDA_POST_EVENT);
        copyL1ToL0A(tileL0A, tileL1A);
        copyL1ToL0B(tileL0B, tileL1B0);
        SetFlag<HardEvent::MTE1_M>(KDA_POST_EVENT);
        copyGmToL1B(tensorL1B1, blockU);
        SetFlag<HardEvent::MTE2_MTE1>(KDA_POST_EVENT_NEXT);
        WaitFlag<HardEvent::MTE1_M>(KDA_POST_EVENT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(KDA_POST_EVENT);
        SetFlag<HardEvent::M_MTE1>(KDA_POST_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(KDA_POST_EVENT_NEXT);
        WaitFlag<HardEvent::M_FIX>(KDA_POST_EVENT);
        WaitFlag<HardEvent::M_MTE1>(KDA_POST_EVENT);
        copyL0CToDst(blockWOut, tileL0C);
        SetFlag<HardEvent::FIX_M>(KDA_POST_EVENT_FIX);

        copyL1ToL0B(tileL0B, tileL1B1);
        SetFlag<HardEvent::MTE1_M>(KDA_POST_EVENT_NEXT);
        WaitFlag<HardEvent::FIX_M>(KDA_POST_EVENT_FIX);
        WaitFlag<HardEvent::MTE1_M>(KDA_POST_EVENT_NEXT);
        tileMmad(tileL0C, tileL0A, tileL0B, m, n, k, true, 0);
        SetFlag<HardEvent::M_FIX>(KDA_POST_EVENT);
        WaitFlag<HardEvent::M_FIX>(KDA_POST_EVENT);
        copyL0CToDst(blockUOut, tileL0C);
        SetFlag<HardEvent::FIX_M>(KDA_POST_EVENT_FIX);
        WaitFlag<HardEvent::FIX_M>(KDA_POST_EVENT_FIX);
    }

    __aicore__ inline void FinalizePostWuPipelineEvents(uint16_t usedSlotCount)
    {
        for (uint16_t slot = 0; slot < usedSlotCount; ++slot) {
            WaitFlag<HardEvent::MTE1_MTE2>(KDA_POST_EVENT + slot);
            WaitFlag<HardEvent::MTE1_MTE2>(KDA_POST_PIPELINE_U_EVENT + slot);
            WaitFlag<HardEvent::M_MTE1>(KDA_POST_EVENT + slot);
            WaitFlag<HardEvent::FIX_M>(KDA_POST_EVENT + slot);
        }
    }

    __aicore__ inline void InitializePostWuPipelineEvents()
    {
        for (uint16_t slot = 0; slot < KDA_POST_PIPELINE_STAGE_COUNT; ++slot) {
            InitializePostWuPipelineSlot(slot);
        }
    }

    __aicore__ inline void InitializePostWuPipelineSlot(uint16_t slot)
    {
        SetFlag<HardEvent::M_MTE1>(KDA_POST_EVENT + slot);
        SetFlag<HardEvent::FIX_M>(KDA_POST_EVENT + slot);
    }

    __aicore__ inline void PrefetchPostWuPipelineArch35(
        Catlass::Arch::Resource<KdaArchTag> &resource, uint16_t slot,
        uint64_t b, uint64_t hv, uint64_t start, uint64_t curT, bool reuseSlot)
    {
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, T, LayoutTagA, T, LayoutTagB, T, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;

        constexpr uint32_t capacityM = 64;
        constexpr uint32_t n = 128;
        constexpr uint32_t capacityK = 64;
        const uint32_t m = static_cast<uint32_t>(curT);
        const uint32_t k = static_cast<uint32_t>(curT);
        auto layoutA = tla::MakeLayout<T, LayoutTagA>(capacityM, capacityK);
        auto layoutB = tla::MakeLayout<T, LayoutTagB>(capacityK, n);
        auto tensorA = tla::MakeTensor(
            preparedAqk_[AOffset(b, hv, start, 0)], layoutA, Catlass::Arch::PositionGM{});
        auto tensorW = tla::MakeTensor(
            preparedQG_[KVOffset(b, hv, start, 0, K_)], layoutB, Catlass::Arch::PositionGM{});
        auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        using CopyGmToL1A = typename TileCopy::template CopyGmToL1A<decltype(blockA)>;
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockW)>;
        CopyGmToL1A copyGmToL1A;
        CopyGmToL1B copyGmToL1B;

        uint32_t slotBase = static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L1_SLOT_BYTES;
        LocalTensor<T> l1A = resource.l1Buf.template GetBufferByByte<T>(slotBase);
        LocalTensor<T> l1W = resource.l1Buf.template GetBufferByByte<T>(
            slotBase + KDA_POST_PIPELINE_L1_A_BYTES);
        auto layoutL1A = tla::MakeLayout<T, LayoutTagL1A>(capacityM, capacityK);
        auto layoutL1B = tla::MakeLayout<T, LayoutTagL1B>(capacityK, n);
        auto tensorL1A = tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1W = tla::MakeTensor(l1W, layoutL1B, Catlass::Arch::PositionL1{});

        uint16_t pipelineEvent = KDA_POST_EVENT + slot;
        if (reuseSlot) {
            WaitFlag<HardEvent::MTE1_MTE2>(pipelineEvent);
        }
        copyGmToL1A(tensorL1A, blockA);
        copyGmToL1B(tensorL1W, blockW);
        SetFlag<HardEvent::MTE2_MTE1>(pipelineEvent);
    }

    __aicore__ inline void PrefetchPostWuPipelineU(
        Catlass::Arch::Resource<KdaArchTag> &resource, uint16_t slot,
        uint64_t b, uint64_t hv, uint64_t start, uint64_t curT, bool reuseStage)
    {
        using LayoutTagB = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, T, Catlass::layout::RowMajor, T, LayoutTagB,
            T, Catlass::layout::RowMajor>;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;

        constexpr uint32_t capacityK = 64;
        constexpr uint32_t n = 128;
        const uint32_t k = static_cast<uint32_t>(curT);
        auto layoutB = tla::MakeLayout<T, LayoutTagB>(capacityK, n);
        auto tensorU = tla::MakeTensor(
            propagatedVNew_[KVOffset(b, hv, start, 0, V_)],
            layoutB, Catlass::Arch::PositionGM{});
        auto blockU = GetTile(tensorU, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        using CopyGmToL1B = typename TileCopy::template CopyGmToL1B<decltype(blockU)>;
        CopyGmToL1B copyGmToL1B;

        uint32_t uOffset = KDA_POST_PIPELINE_STAGE_COUNT * KDA_POST_PIPELINE_L1_SLOT_BYTES +
                           static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L1_U_SLOT_BYTES;
        LocalTensor<T> l1U = resource.l1Buf.template GetBufferByByte<T>(uOffset);
        auto layoutL1B = tla::MakeLayout<T, LayoutTagL1B>(capacityK, n);
        auto tensorL1U = tla::MakeTensor(l1U, layoutL1B, Catlass::Arch::PositionL1{});

        uint16_t pipelineEvent = KDA_POST_PIPELINE_U_EVENT + slot;
        if (reuseStage) {
            WaitFlag<HardEvent::MTE1_MTE2>(pipelineEvent);
        }
        copyGmToL1B(tensorL1U, blockU);
        SetFlag<HardEvent::MTE2_MTE1>(pipelineEvent);
    }

    __aicore__ inline void ComputePrefetchedPostWuPipelineArch35(
        Catlass::Arch::Resource<KdaArchTag> &resource, uint16_t slot,
        uint64_t b, uint64_t hv, uint64_t start, uint64_t curT)
    {
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<
            KdaArchTag, T, LayoutTagA, T, LayoutTagB, T, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, T, LayoutTagL1A>;

        constexpr uint32_t capacityM = 64;
        constexpr uint32_t n = 128;
        constexpr uint32_t packedN = 256;
        constexpr uint32_t capacityK = 64;
        const uint32_t m = static_cast<uint32_t>(curT);
        const uint32_t k = static_cast<uint32_t>(curT);
        auto layoutC = tla::MakeLayout<T, LayoutTagC>(capacityM, n);
        auto tensorWOut = tla::MakeTensor(
            w_[KVOffset(b, hv, start, 0, K_)], layoutC, Catlass::Arch::PositionGM{});
        auto tensorUOut = tla::MakeTensor(
            u_[KVOffset(b, hv, start, 0, V_)], layoutC, Catlass::Arch::PositionGM{});
        auto blockWOut = GetTile(tensorWOut, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto blockUOut = GetTile(tensorUOut, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(blockWOut)>;

        uint32_t l1Base = static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L1_SLOT_BYTES;
        LocalTensor<T> l1A = resource.l1Buf.template GetBufferByByte<T>(l1Base);
        LocalTensor<T> l1W = resource.l1Buf.template GetBufferByByte<T>(
            l1Base + KDA_POST_PIPELINE_L1_A_BYTES);
        uint32_t uOffset = KDA_POST_PIPELINE_STAGE_COUNT * KDA_POST_PIPELINE_L1_SLOT_BYTES +
                           static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L1_U_SLOT_BYTES;
        LocalTensor<T> l1U = resource.l1Buf.template GetBufferByByte<T>(uOffset);
        LocalTensor<T> l0A = resource.l0ABuf.template GetBufferByByte<T>(
            static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L0_A_SLOT_BYTES);
        LocalTensor<T> l0B = resource.l0BBuf.template GetBufferByByte<T>(
            static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L0_B_SLOT_BYTES);
        LocalTensor<float> l0C = resource.l0CBuf.template GetBufferByByte<float>(
            static_cast<uint32_t>(slot) * KDA_POST_PIPELINE_L0_C_SLOT_BYTES);

        auto layoutL1A = tla::MakeLayout<T, LayoutTagL1A>(capacityM, capacityK);
        auto layoutL1B = tla::MakeLayout<T, LayoutTagL1B>(capacityK, n);
        auto layoutL0A = tla::MakeLayout<T, LayoutTagL0A>(capacityM, capacityK);
        auto layoutL0B = tla::MakeLayout<T, LayoutTagL0B>(capacityK, packedN);
        auto layoutL0C = tla::MakeLayoutL0C(capacityM, packedN);
        auto tensorL1A = tla::MakeTensor(l1A, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1W = tla::MakeTensor(l1W, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL1U = tla::MakeTensor(l1U, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1A = GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL1W = GetTile(tensorL1W, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL1U = GetTile(tensorL1U, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileL0BW = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileL0BU = GetTile(tensorL0B, tla::MakeCoord(0, n), tla::MakeShape(k, n));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k, packedN));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, packedN));
        auto tileL0CW = GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
        auto tileL0CU = GetTile(tensorL0C, tla::MakeCoord(0, n), tla::MakeShape(m, n));

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToDst copyL0CToDst;
        TileMmad tileMmad;

        uint16_t pipelineEvent = KDA_POST_EVENT + slot;
        uint16_t uPipelineEvent = KDA_POST_PIPELINE_U_EVENT + slot;
        WaitFlag<HardEvent::MTE2_MTE1>(pipelineEvent);
        WaitFlag<HardEvent::MTE2_MTE1>(uPipelineEvent);
        WaitFlag<HardEvent::M_MTE1>(pipelineEvent);
        copyL1ToL0A(tileL0A, tileL1A);
        copyL1ToL0B(tileL0BW, tileL1W);
        copyL1ToL0B(tileL0BU, tileL1U);
        SetFlag<HardEvent::MTE1_M>(pipelineEvent);
        SetFlag<HardEvent::MTE1_MTE2>(pipelineEvent);
        SetFlag<HardEvent::MTE1_MTE2>(uPipelineEvent);
        WaitFlag<HardEvent::MTE1_M>(pipelineEvent);
        WaitFlag<HardEvent::FIX_M>(pipelineEvent);
        tileMmad(tileL0C, tileL0A, tileL0B, m, packedN, k, true, 0);
        SetFlag<HardEvent::M_MTE1>(pipelineEvent);
        SetFlag<HardEvent::M_FIX>(pipelineEvent);
        WaitFlag<HardEvent::M_FIX>(pipelineEvent);
        copyL0CToDst(blockWOut, tileL0CW);
        copyL0CToDst(blockUOut, tileL0CU);
        SetFlag<HardEvent::FIX_M>(pipelineEvent);
    }
#endif

    __aicore__ inline void ComputePostWuCube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                             uint64_t curT)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (UsePostWuCubeArch35(curT)) {
            ComputePostWuCubeFusedArch35(b, hv, start, curT);
            return;
        }
#endif
        using ElementA = AKK_T;
        using ElementB = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        using WTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, T, LayoutTagC>;
#else
        using WTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, float, LayoutTagC>;
#endif
        using UTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, OUT_T, LayoutTagC>;
        using PostL1TileShape128 = tla::Shape<KdaInt128, KdaInt128, tla::_256>;
        using PostL0TileShape128 = tla::Shape<KdaInt128, KdaInt128, KdaInt128>;
        using PostL1TileShape256 = tla::Shape<KdaInt128, tla::_256, tla::_256>;
        using PostL0TileShape256 = tla::Shape<KdaInt128, tla::_256, KdaInt64>;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        using WBlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape128,
                                                               PostL0TileShape128,
                                                               ElementA, ElementB, T, void, WTileCopy>;
#else
        using WBlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape128,
                                                               PostL0TileShape128,
                                                               ElementA, ElementB, float, void, WTileCopy>;
#endif
        using UBlockMmad128 = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape128,
                                                                  PostL0TileShape128,
                                                                  ElementA, ElementB, OUT_T, void, UTileCopy>;
        using UBlockMmad256 = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape256,
                                                                  PostL0TileShape256,
                                                                  ElementA, ElementB, OUT_T, void, UTileCopy>;
        LayoutTagA tagA = LayoutTagA::template MakeLayout<ElementA>(BT_, BT_);
        auto layoutA = tla::MakeLayoutFromTag(tagA);
        auto tensorA = tla::MakeTensor(preparedAqk_[AOffset(b, hv, start, 0)], layoutA,
                                       Catlass::Arch::PositionGM{});

        {
            LayoutTagB tagB = LayoutTagB::template MakeLayout<ElementB>(BT_, K_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            LayoutTagC tagC = LayoutTagC::template MakeLayout<T>(BT_, K_);
#else
            LayoutTagC tagC = LayoutTagC::template MakeLayout<float>(BT_, K_);
#endif
            auto layoutB = tla::MakeLayoutFromTag(tagB);
            auto layoutC = tla::MakeLayoutFromTag(tagC);
            Catlass::GemmCoord shape{static_cast<uint32_t>(curT), static_cast<uint32_t>(K_),
                                     static_cast<uint32_t>(curT)};
            auto tensorB = tla::MakeTensor(preparedQG_[KVOffset(b, hv, start, 0, K_)], layoutB,
                                           Catlass::Arch::PositionGM{});
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            auto tensorC = tla::MakeTensor(w_[KVOffset(b, hv, start, 0, K_)], layoutC,
                                           Catlass::Arch::PositionGM{});
#else
            auto tensorC = tla::MakeTensor(h_[WScratchOffset(b, hv, chunkIdx, 0, 0)], layoutC,
                                            Catlass::Arch::PositionGM{});
#endif
            auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
            auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
            auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
            Catlass::Arch::Resource<KdaArchTag> wResource;
            WBlockMmad wBlockMmad(wResource);
            wBlockMmad(blockA, blockB, blockC, shape);
            PipeBarrier<PIPE_ALL>();
        }

        {
            LayoutTagB tagB = LayoutTagB::template MakeLayout<ElementB>(BT_, V_);
            LayoutTagC tagC = LayoutTagC::template MakeLayout<OUT_T>(BT_, V_);
            auto layoutB = tla::MakeLayoutFromTag(tagB);
            auto layoutC = tla::MakeLayoutFromTag(tagC);
            Catlass::GemmCoord shape{static_cast<uint32_t>(curT), static_cast<uint32_t>(V_),
                                     static_cast<uint32_t>(curT)};
            auto tensorB = tla::MakeTensor(propagatedVNew_[KVOffset(b, hv, start, 0, V_)], layoutB,
                                           Catlass::Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(u_[KVOffset(b, hv, start, 0, V_)], layoutC,
                                           Catlass::Arch::PositionGM{});
            auto blockA = GetTile(tensorA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
            auto blockB = GetTile(tensorB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
            auto blockC = GetTile(tensorC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
            Catlass::Arch::Resource<KdaArchTag> uResource;
            if (V_ <= 128) {
                UBlockMmad128 uBlockMmad(uResource);
                uBlockMmad(blockA, blockB, blockC, shape);
            } else {
                UBlockMmad256 uBlockMmad(uResource);
                uBlockMmad(blockA, blockB, blockC, shape);
            }
            PipeBarrier<PIPE_ALL>();
        }

    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline bool UseTypicalPostWuGate(uint64_t curT) const
    {
        return curT == 64 && BT_ == 64 && K_ == 128 && V_ == 128;
    }

    __aicore__ inline uint64_t TypicalGateStageElems() const
    {
        return static_cast<uint64_t>(KDA_TYPICAL_GATE_TILE_ROWS) * 128;
    }

    __aicore__ inline uint64_t TypicalGateStageBytes() const
    {
        return TypicalGateStageElems() * (sizeof(T) + sizeof(GK_T));
    }

    __aicore__ inline LocalTensor<T> TypicalGateK(uint64_t slot)
    {
        return gateWritebackBuf_.Get<T>()[slot * TypicalGateStageBytes() / sizeof(T)];
    }

    __aicore__ inline LocalTensor<GK_T> TypicalGateG(uint64_t slot)
    {
        uint64_t byteOffset = slot * TypicalGateStageBytes() + TypicalGateStageElems() * sizeof(T);
        return gateWritebackBuf_.Get<GK_T>()[byteOffset / sizeof(GK_T)];
    }

    __aicore__ inline void PrefetchTypicalKg(uint64_t slot, uint64_t b, uint64_t h, uint64_t hv,
                                             uint64_t token, uint64_t rows)
    {
        uint64_t elems = rows * K_;
        LocalTensor<T> kStage = TypicalGateK(slot);
        LocalTensor<GK_T> gateStage = TypicalGateG(slot);
        CopyRowsIn(kStage, k_, QOffset(b, h, token, 0), rows, K_,
                   inputSequenceMajor_ ? H_ * K_ : K_);
        DataCopy(gateStage, gk_[KVOffset(b, hv, token, 0, K_)], static_cast<uint32_t>(elems));
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline void ComputeTypicalKg(uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
                                            uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        if (rowBegin >= rowEnd) {
            return;
        }

        LocalTensor<float> gateLast = exp2Buf_.Get<float>();
        LoadAsFloatRow(gk_, KVOffset(b, hv, start + curT - 1, 0, K_), gateLast, K_);

        uint64_t slot = 0;
        uint64_t firstRows = rowEnd - rowBegin;
        if (firstRows > KDA_TYPICAL_GATE_TILE_ROWS) {
            firstRows = KDA_TYPICAL_GATE_TILE_ROWS;
        }
        PrefetchTypicalKg(slot, b, h, hv, start + rowBegin, firstRows);

        bool outputPending = false;
        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += KDA_TYPICAL_GATE_TILE_ROWS) {
            uint64_t tileRows = rowEnd - tileRow;
            if (tileRows > KDA_TYPICAL_GATE_TILE_ROWS) {
                tileRows = KDA_TYPICAL_GATE_TILE_ROWS;
            }
            uint64_t elems = tileRows * K_;
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

            if (outputPending) {
                WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            }
            uint64_t nextTileRow = tileRow + KDA_TYPICAL_GATE_TILE_ROWS;
            if (nextTileRow < rowEnd) {
                uint64_t nextRows = rowEnd - nextTileRow;
                if (nextRows > KDA_TYPICAL_GATE_TILE_ROWS) {
                    nextRows = KDA_TYPICAL_GATE_TILE_ROWS;
                }
                PrefetchTypicalKg(slot ^ 1, b, h, hv, start + nextTileRow, nextRows);
            }

            LocalTensor<T> kAndKg = TypicalGateK(slot);
            LocalTensor<GK_T> gateStage = TypicalGateG(slot);
            ComputePostKdaKgRegbase<T, GK_T>(
                (__ubuf__ T *)reinterpret_cast<uint64_t>(kAndKg.GetPhyAddr()),
                (__ubuf__ GK_T *)reinterpret_cast<uint64_t>(gateStage.GetPhyAddr()),
                (__ubuf__ float *)reinterpret_cast<uint64_t>(gateLast.GetPhyAddr()),
                static_cast<uint16_t>(tileRows), static_cast<uint16_t>(K_));

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(kg_[KVOffset(b, hv, start + tileRow, 0, K_)], kAndKg,
                     static_cast<uint32_t>(elems));
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            outputPending = true;
            slot ^= 1;
        }
        if (outputPending) {
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        }
    }

    __aicore__ inline uint64_t TypicalGatePipelineStageElems() const
    {
        return static_cast<uint64_t>(KDA_TYPICAL_GATE_PIPELINE_ROWS) * 128;
    }

    __aicore__ inline uint64_t TypicalGatePipelineStageBytes() const
    {
        return TypicalGatePipelineStageElems() * (sizeof(T) + sizeof(float)) +
               128 * sizeof(float);
    }

    __aicore__ inline LocalTensor<T> TypicalGatePipelineK(uint64_t slot)
    {
        uint64_t byteOffset = slot * TypicalGatePipelineStageBytes();
        return vecBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline LocalTensor<float> TypicalGatePipelineG(uint64_t slot)
    {
        uint64_t byteOffset = slot * TypicalGatePipelineStageBytes() +
                              TypicalGatePipelineStageElems() * sizeof(T);
        return vecBuf_.Get<float>()[byteOffset / sizeof(float)];
    }

    __aicore__ inline LocalTensor<float> TypicalGatePipelineRef(uint64_t slot)
    {
        uint64_t byteOffset = slot * TypicalGatePipelineStageBytes() +
                              TypicalGatePipelineStageElems() * (sizeof(T) + sizeof(float));
        return vecBuf_.Get<float>()[byteOffset / sizeof(float)];
    }

    __aicore__ inline bool CanPipelineTypicalKg(
        uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum) const
    {
        if constexpr (!IsSameType<GK_T, float>::value) {
            return false;
        }
        if (!UseTypicalPostWuGate(curT) || subBlockNum == 0) {
            return false;
        }
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        return rowBegin < rowEnd && rowEnd - rowBegin <= KDA_TYPICAL_GATE_PIPELINE_ROWS;
    }

    __aicore__ inline void PrefetchTypicalKgPipeline(
        uint64_t slot, uint64_t b, uint64_t h, uint64_t hv, uint64_t start,
        uint64_t curT, uint64_t rowBegin, uint64_t rowEnd)
    {
        if constexpr (IsSameType<GK_T, float>::value) {
            uint64_t elems = (rowEnd - rowBegin) * K_;
            LocalTensor<T> kStage = TypicalGatePipelineK(slot);
            LocalTensor<float> gateStage = TypicalGatePipelineG(slot);
            LocalTensor<float> refStage = TypicalGatePipelineRef(slot);
            CopyRowsIn(kStage, k_, QOffset(b, h, start + rowBegin, 0), rowEnd - rowBegin, K_,
                       inputSequenceMajor_ ? H_ * K_ : K_);
            DataCopy(gateStage, gk_[KVOffset(b, hv, start + rowBegin, 0, K_)],
                     static_cast<uint32_t>(elems));
            DataCopy(refStage, gk_[KVOffset(b, hv, start + curT - 1, 0, K_)],
                     static_cast<uint32_t>(K_));
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        }
    }

    __aicore__ inline void ComputeTypicalKgPipelineRegs(
        uint64_t slot, uint64_t rowBegin, uint64_t rowEnd)
    {
        uint64_t rows = rowEnd - rowBegin;
        LocalTensor<T> kAndKg = TypicalGatePipelineK(slot);
        LocalTensor<float> gateStage = TypicalGatePipelineG(slot);
        LocalTensor<float> refStage = TypicalGatePipelineRef(slot);
        ComputePostKdaKgRegbase<T, float>(
            (__ubuf__ T *)reinterpret_cast<uint64_t>(kAndKg.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(gateStage.GetPhyAddr()),
            (__ubuf__ float *)reinterpret_cast<uint64_t>(refStage.GetPhyAddr()),
            static_cast<uint16_t>(rows), static_cast<uint16_t>(K_));
    }

    __aicore__ inline void StoreTypicalKgPipeline(
        uint64_t slot, uint64_t b, uint64_t hv, uint64_t start,
        uint64_t rowBegin, uint64_t rowEnd)
    {
        uint64_t elems = (rowEnd - rowBegin) * K_;
        LocalTensor<T> kAndKg = TypicalGatePipelineK(slot);
        SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
        DataCopy(kg_[KVOffset(b, hv, start + rowBegin, 0, K_)], kAndKg,
                 static_cast<uint32_t>(elems));
        SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
    }
#endif

    __aicore__ inline void CopyScratchWAndFinalizeKg(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                                     uint64_t start, uint64_t curT, uint64_t subBlockIdx,
                                                     uint64_t subBlockNum)
    {
        constexpr uint64_t typedOffsetFloats = 20480;
        constexpr uint64_t typedOffset = typedOffsetFloats * sizeof(float) / sizeof(T);
        constexpr uint64_t kgFp32Planes = 4;
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        if (rowBegin >= rowEnd) {
            return;
        }
        uint64_t maxRows = (typedOffsetFloats / kgFp32Planes) / K_;
        if (maxRows > 32) {
            maxRows = 32;
        }
        if (maxRows == 0) {
            return;
        }

        uint64_t last = start + curT - 1;
        LocalTensor<float> arena = vecBuf_.Get<float>();
        LocalTensor<float> gateLast = exp2Buf_.Get<float>();
        LocalTensor<T> typedLocal = vecBuf_.Get<T>()[typedOffset];
        LoadAsFloatRow(gk_, KVOffset(b, hv, last, 0, K_), gateLast, K_);

        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += maxRows) {
            uint64_t tileRows = rowEnd - tileRow;
            if (tileRows > maxRows) {
                tileRows = maxRows;
            }
            uint64_t elemCount = tileRows * K_;
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            uint64_t scratchBase = WScratchOffset(b, hv, chunkIdx, tileRow, 0);
#else
            (void)chunkIdx;
#endif
            uint64_t token = start + tileRow;

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
            DataCopy(arena, h_[scratchBase], static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(typedLocal, arena, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            DataCopy(w_[KVOffset(b, hv, token, 0, K_)], typedLocal, static_cast<uint32_t>(elemCount));
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
#endif

            LocalTensor<float> kLocal = arena;
            LocalTensor<float> gLocal = arena[elemCount];
            LocalTensor<float> expLocal = arena[2 * elemCount];
            LocalTensor<float> outLocal = arena[3 * elemCount];
            const uint64_t gateOffsetBytes = (typedOffset + elemCount) * sizeof(T);
            LocalTensor<GK_T> gateTyped = vecBuf_.Get<GK_T>()[
                (gateOffsetBytes + sizeof(GK_T) - 1) / sizeof(GK_T)];
            CopyRowsIn(typedLocal, k_, QOffset(b, h, token, 0), tileRows, K_,
                       inputSequenceMajor_ ? H_ * K_ : K_);
            LoadAsFloatVector(gk_, KVOffset(b, hv, token, 0, K_), gLocal, gateTyped, elemCount);
            Cast(kLocal, typedLocal, RoundMode::CAST_NONE, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();

            for (uint64_t row = 0; row < tileRows; ++row) {
                Sub(expLocal[row * K_], gateLast, gLocal[row * K_], static_cast<uint32_t>(K_));
            }
            PipeBarrier<PIPE_V>();
            Muls(expLocal, expLocal, LN2, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            ClampExpInput(expLocal, static_cast<uint32_t>(elemCount));
            Exp(expLocal, expLocal, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            Mul(outLocal, kLocal, expLocal, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(outLocal, static_cast<uint32_t>(elemCount));
            Cast(typedLocal, outLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elemCount));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(kg_, KVOffset(b, hv, token, 0, K_), typedLocal, elemCount);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        }
        if (rowEnd == curT) {
            CopyVectorIn(typedLocal, k_, QOffset(b, h, last, 0), K_);
            SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
            WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
            CopyVectorOut(kg_, KVOffset(b, hv, last, 0, K_), typedLocal, K_);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvent_);
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline void ComputeTailWuRow(GlobalTensor<SrcTensor> &src, GlobalTensor<DstTensor> &dst,
                                            uint64_t akkBase, uint64_t srcBase, uint64_t dstBase, uint64_t curT,
                                            uint64_t dim, uint64_t rowStride)
    {
        LocalTensor<float> acc = vecBuf_.Get<float>();
        LocalTensor<float> value = vecBuf_.Get<float>()[512];
        LocalTensor<SrcTensor> typed = vecBuf_.Get<SrcTensor>()[4096];
        LocalTensor<T> coefficientTyped = exp2Buf_.Get<T>();
        LocalTensor<float> coefficients = exp2Buf_.Get<float>()[128];
        CopyVectorIn(coefficientTyped, preparedAqk_, akkBase, curT);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        Cast(coefficients, coefficientTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(curT));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::V_S>(EXP2_EVENT_ID);
        for (uint64_t j = 0; j < curT; ++j) {
            LoadAsFloatVector(src, srcBase + j * rowStride, value, typed, dim);
            float coefficient = coefficients.GetValue(j);
            SetFlag<HardEvent::S_V>(EXP2_EVENT_ID);
            WaitFlag<HardEvent::S_V>(EXP2_EVENT_ID);
            Muls(value, value, coefficient, static_cast<uint32_t>(dim));
            PipeBarrier<PIPE_V>();
            if (j == 0) {
                Adds(acc, value, 0.0f, static_cast<uint32_t>(dim));
            } else {
                Add(acc, acc, value, static_cast<uint32_t>(dim));
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
            WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        }
        SetFlag<HardEvent::S_MTE2>(EXP2_EVENT_ID);
        WaitFlag<HardEvent::S_MTE2>(EXP2_EVENT_ID);
        ClampFp32ToOutputType(acc, static_cast<uint32_t>(dim));
        StoreFloatRow(dst, dstBase, acc, dim);
    }

    __aicore__ inline void ComputeTailWuVector(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                               uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        // preparedQG_ and w_ alias. Each subblock owns disjoint columns and
        // writes rows from last to first, so every lower-triangular source row
        // remains live through its final use without desynchronizing the AIVs.
        uint64_t colBegin = (K_ * subBlockIdx) / subBlockNum;
        uint64_t colEnd = (K_ * (subBlockIdx + 1)) / subBlockNum;
        for (uint64_t row = curT; row > 0; --row) {
            uint64_t rowIdx = row - 1;
            ComputeTailWuRow(
                preparedQG_, w_, AOffset(b, hv, start + rowIdx, 0), KVOffset(b, hv, start, colBegin, K_),
                KVOffset(b, hv, start + rowIdx, colBegin, K_), curT, colEnd - colBegin, K_);
        }

        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        for (uint64_t row = rowBegin; row < rowEnd; ++row) {
            ComputeTailWuRow(
                propagatedVNew_, u_, AOffset(b, hv, start + row, 0), KVOffset(b, hv, start, 0, V_),
                KVOffset(b, hv, start + row, 0, V_), curT, V_, V_);
        }
    }
#endif
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

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline bool ResolveHeadMajorChunk(
        uint64_t task, uint64_t &seq, uint64_t &b, uint64_t &h, uint64_t &hv,
        uint64_t &chunkIdx, uint64_t &start, uint64_t &end)
    {
        uint64_t flatTask = 0;
        if (isVarLen_) {
            hv = task / NT_;
            uint64_t flatChunk = task % NT_;
            flatTask = flatChunk * HV_ + hv;
        } else {
            uint64_t entity = task / NT_;
            uint64_t localChunk = task % NT_;
            b = entity / HV_;
            hv = entity % HV_;
            uint64_t flatChunk = b * NT_ + localChunk;
            flatTask = flatChunk * HV_ + hv;
        }
        return ResolveFlatChunk(flatTask, seq, b, h, hv, chunkIdx, start, end);
    }

    __aicore__ inline void GetHeadMajorTaskRange(
        uint64_t coreIdx, uint64_t coreNum, uint64_t taskNum,
        uint64_t &taskBegin, uint64_t &taskEnd) const
    {
        uint64_t tasksPerCore = (taskNum + coreNum - 1) / coreNum;
        taskBegin = coreIdx * tasksPerCore;
        taskEnd = taskBegin + tasksPerCore;
        if (taskBegin > taskNum) {
            taskBegin = taskNum;
        }
        if (taskEnd > taskNum) {
            taskEnd = taskNum;
        }
    }

    __aicore__ inline void ProcessPostAivPipelineArch35(
        uint64_t taskBegin, uint64_t taskEnd, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t task = taskBegin;
        while (task < taskEnd) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            bool resolved = ResolveHeadMajorChunk(task, seq, b, h, hv, chunkIdx, start, end);
            uint64_t curT = resolved ? end - start : 0;
            if (!resolved || !CanPipelineTypicalKg(curT, subBlockIdx, subBlockNum)) {
                if (resolved) {
                    (void)seq;
                    ProcessChunkPostAiv(
                        b, h, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
                }
                ++task;
                continue;
            }

            uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
            uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
            uint16_t slot = 0;
            bool outputPending = false;
            PrefetchTypicalKgPipeline(
                slot, b, h, hv, start, curT, rowBegin, rowEnd);

            while (true) {
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

                uint64_t nextTask = task + 1;
                uint64_t nextSeq = 0;
                uint64_t nextB = 0;
                uint64_t nextH = 0;
                uint64_t nextHv = 0;
                uint64_t nextChunkIdx = 0;
                uint64_t nextStart = 0;
                uint64_t nextEnd = 0;
                bool nextResolved = nextTask < taskEnd && ResolveHeadMajorChunk(
                    nextTask, nextSeq, nextB, nextH, nextHv, nextChunkIdx, nextStart, nextEnd);
                uint64_t nextCurT = nextResolved ? nextEnd - nextStart : 0;
                bool nextIsTypical = nextResolved &&
                    CanPipelineTypicalKg(nextCurT, subBlockIdx, subBlockNum);
                uint64_t nextRowBegin = 0;
                uint64_t nextRowEnd = 0;
                if (nextIsTypical) {
                    nextRowBegin = (nextCurT * subBlockIdx) / subBlockNum;
                    nextRowEnd = (nextCurT * (subBlockIdx + 1)) / subBlockNum;
                    uint16_t nextSlot = (slot + 1) % KDA_TYPICAL_GATE_PIPELINE_STAGES;
                    PrefetchTypicalKgPipeline(
                        nextSlot, nextB, nextH, nextHv, nextStart, nextCurT,
                        nextRowBegin, nextRowEnd);
                }

                ComputeTypicalKgPipelineRegs(slot, rowBegin, rowEnd);
                if (outputPending) {
                    WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
                    outputPending = false;
                }
                StoreTypicalKgPipeline(slot, b, hv, start, rowBegin, rowEnd);
                outputPending = true;

                if (!nextIsTypical) {
                    WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
                    task = nextTask;
                    break;
                }

                task = nextTask;
                seq = nextSeq;
                b = nextB;
                h = nextH;
                hv = nextHv;
                chunkIdx = nextChunkIdx;
                start = nextStart;
                end = nextEnd;
                curT = nextCurT;
                rowBegin = nextRowBegin;
                rowEnd = nextRowEnd;
                slot = (slot + 1) % KDA_TYPICAL_GATE_PIPELINE_STAGES;
                (void)seq;
                (void)chunkIdx;
                (void)end;
                (void)curT;
            }
        }
    }
#endif

    __aicore__ inline void ProcessChunkPostAiv(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                               uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                               uint64_t subBlockNum)
    {
        uint64_t curT = end - start;
        if (curT == 0 || !UsePostWuCube(curT)) {
            return;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (curT < BT_) {
            ComputeTailWuVector(b, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);
            CopyScratchWAndFinalizeKg(
                b, h, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);
            return;
        }
        if (UseTypicalPostWuGate(curT)) {
            ComputeTypicalKg(b, h, hv, start, curT, subBlockIdx, subBlockNum);
            return;
        } else {
            CopyScratchWAndFinalizeKg(b, h, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);
        }
#else
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(syncDoneFlag_);
        CopyScratchWAndFinalizeKg(b, h, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);
#endif
    }

    __aicore__ inline void ProcessChunkPostAic(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                               uint64_t end)
    {
        if constexpr (IsSameType<AKK_T, T>::value) {
            ProcessChunkPostAicTyped(b, hv, chunkIdx, start, end);
        }
    }

    __aicore__ inline void ProcessChunkPostAicTyped(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                    uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0 || !UsePostWuCube(curT)) {
            return;
        }
        ComputePostWuCube(b, hv, chunkIdx, start, curT);
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ != 310
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
#endif
    }

    __aicore__ inline void ProcessPostAiv()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        if (subBlockNum == 0) {
            return;
        }
        uint64_t subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (BT_ == 64 && K_ == 128 && V_ == 128) {
            uint64_t taskBegin = 0;
            uint64_t taskEnd = 0;
            GetHeadMajorTaskRange(coreIdx, coreNum, taskNum, taskBegin, taskEnd);
            ProcessPostAivPipelineArch35(taskBegin, taskEnd, subBlockIdx, subBlockNum);
            return;
        }
#endif
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
                ProcessChunkPostAiv(b, h, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
            }
        }
    }

    __aicore__ inline void ProcessPostAic()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (KDA_ENABLE_POST_AIC_PIPELINE && BT_ == 64 && K_ == 128 && V_ == 128) {
            ProcessPostAicPipelineArch35();
            return;
        }
#endif
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        for (uint64_t task = GetBlockIdx(); task < taskNum; task += coreNum) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            if (ResolveFlatChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                (void)seq;
                (void)h;
                ProcessChunkPostAic(b, hv, chunkIdx, start, end);
            }
        }
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void ProcessPostAicPipelineArch35()
    {
        static_assert(sizeof(T) == sizeof(uint16_t),
                      "arch35 PostWU pipeline is specialized for fp16/bf16 inputs");
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        uint64_t taskBegin = 0;
        uint64_t taskEnd = 0;
        GetHeadMajorTaskRange(coreIdx, coreNum, taskNum, taskBegin, taskEnd);
        if (taskEnd - taskBegin < KDA_POST_PIPELINE_STAGE_COUNT) {
            for (uint64_t task = taskBegin; task < taskEnd; ++task) {
                uint64_t seq = 0;
                uint64_t b = 0;
                uint64_t h = 0;
                uint64_t hv = 0;
                uint64_t chunkIdx = 0;
                uint64_t start = 0;
                uint64_t end = 0;
                if (ResolveHeadMajorChunk(task, seq, b, h, hv, chunkIdx, start, end)) {
                    (void)seq;
                    (void)h;
                    if (end - start == BT_) {
                        ProcessPreparedTailSingleArch35(b, hv, start, BT_);
                    }
                }
            }
            return;
        }
        uint64_t task = taskBegin;
        Catlass::Arch::Resource<KdaArchTag> resource;

        while (task < taskEnd) {
            uint64_t seq = 0;
            uint64_t b = 0;
            uint64_t h = 0;
            uint64_t hv = 0;
            uint64_t chunkIdx = 0;
            uint64_t start = 0;
            uint64_t end = 0;
            bool resolved = ResolveHeadMajorChunk(task, seq, b, h, hv, chunkIdx, start, end);
            uint64_t curT = resolved ? end - start : 0;
            if (!resolved || !UseFullPostWuPipelineArch35(curT)) {
                (void)seq;
                (void)h;
                ++task;
                continue;
            }

            uint16_t slot = 0;
            uint16_t usedSlotCount = 1;
            InitializePostWuPipelineSlot(slot);
            PrefetchPostWuPipelineArch35(resource, slot, b, hv, start, curT, false);
            PrefetchPostWuPipelineU(resource, slot, b, hv, start, curT, false);
            while (true) {
                uint64_t nextTask = task + 1;
                uint64_t nextSeq = 0;
                uint64_t nextB = 0;
                uint64_t nextH = 0;
                uint64_t nextHv = 0;
                uint64_t nextChunkIdx = 0;
                uint64_t nextStart = 0;
                uint64_t nextEnd = 0;
                bool nextResolved = nextTask < taskEnd && ResolveHeadMajorChunk(
                    nextTask, nextSeq, nextB, nextH, nextHv, nextChunkIdx, nextStart, nextEnd);
                uint64_t nextCurT = nextResolved ? nextEnd - nextStart : 0;
                bool nextIsTypical = nextResolved && UseFullPostWuPipelineArch35(nextCurT);
                if (nextIsTypical) {
                    uint16_t nextSlot = slot ^ 1;
                    bool reuseSlot = usedSlotCount == KDA_POST_PIPELINE_STAGE_COUNT;
                    if (!reuseSlot) {
                        InitializePostWuPipelineSlot(nextSlot);
                    }
                    PrefetchPostWuPipelineArch35(
                        resource, nextSlot, nextB, nextHv, nextStart, nextCurT, reuseSlot);
                    PrefetchPostWuPipelineU(
                        resource, nextSlot, nextB, nextHv, nextStart, nextCurT, reuseSlot);
                    if (!reuseSlot) {
                        ++usedSlotCount;
                    }
                }

                ComputePrefetchedPostWuPipelineArch35(resource, slot, b, hv, start, curT);
                if (!nextIsTypical) {
                    FinalizePostWuPipelineEvents(usedSlotCount);
                    task = nextTask;
                    break;
                }
                task = nextTask;
                seq = nextSeq;
                b = nextB;
                h = nextH;
                hv = nextHv;
                chunkIdx = nextChunkIdx;
                start = nextStart;
                end = nextEnd;
                curT = nextCurT;
                slot ^= 1;
                (void)seq;
                (void)h;
                (void)chunkIdx;
                (void)end;
                (void)curT;
            }
        }
    }
#endif


private:
    GlobalTensor<T> q_;
    GlobalTensor<T> k_;
    GlobalTensor<T> v_;
    GlobalTensor<GK_T> gk_;
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
    GlobalTensor<T> preparedQG_;
    GlobalTensor<T> preparedAqk_;
    GlobalTensor<T> propagatedVNew_;
    GlobalTensor<T> propagatedH_;
    GlobalTensor<float> solveWorkspace_;
    GlobalTensor<T> scoreWorkspace_;
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> exp2Buf_;
    TBuf<TPosition::VECCALC> vecBuf_;
    TBuf<TPosition::VECCALC> gateWritebackBuf_;
    TEventID mte2ToVEvent_ = 0;
    TEventID vToMte2Event_ = 0;
    TEventID vToMte3Event_ = 0;
    TEventID mte3ToVEvent_ = 0;
    TEventID mte2ToMte3Event_ = 0;
    TEventID mte3ToMte2Event_ = 0;
    bool vectorEventsAllocated_ = false;
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SCORE_QUEUE_DEPTH> scoreReadyFlag_{KDA_SCORE_READY_FLAG0,
                                                                                  KDA_SCORE_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SCORE_QUEUE_DEPTH> scoreDoneFlag_{KDA_SCORE_DONE_FLAG0,
                                                                                 KDA_SCORE_DONE_FLAG1};
    // Score production is fully drained before solve starts, so the solve handshake can safely reuse
    // the existing score flags without consuming additional hardware flag IDs.
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SYNC_REVERSE_DEPTH> syncReadyFlag_{KDA_SCORE_READY_FLAG0,
                                                                                  KDA_SCORE_READY_FLAG1};
    Catlass::Arch::CrossCoreFlagWithReverse<KDA_SYNC_REVERSE_DEPTH> syncDoneFlag_{KDA_SCORE_DONE_FLAG0,
                                                                                 KDA_SCORE_DONE_FLAG1};
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
    bool inputSequenceMajor_ = false;
    uint64_t usedCoreNum_ = 1;
    uint64_t solveCoreIdx_ = 0;
    __gm__ int64_t *chunkIndicesAddr_ = nullptr;
    __gm__ int64_t *cuSeqlensAddr_ = nullptr;
};
} // namespace

template <typename T, typename GK_T, typename BETA_T, typename TilingData>
__aicore__ inline void RunChunkKdaPostWu(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR wSeed, GM_ADDR akk, GM_ADDR uSeed,
    GM_ADDR w, GM_ADDR u, GM_ADDR kg, GM_ADDR vNew, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe)
{
    GM_ADDR postScratch = userWorkspace + tiling.postWuScratchOffset;
    if ASCEND_IS_AIC {
        ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                wSeed, akk, uSeed, nullptr, userWorkspace, userWorkspace, userWorkspace, akk, w, u,
                userWorkspace, kg, vNew, postScratch, postScratch, tiling, &pipe, false);
        op.ProcessAic();
    }
    if ASCEND_IS_AIV {
        ChunkKdaFwdPostWuKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                wSeed, akk, uSeed, nullptr, userWorkspace, userWorkspace, userWorkspace, akk, w, u,
                userWorkspace, kg, vNew, postScratch, postScratch, tiling, &pipe);
        op.ProcessAiv();
    }
}

} // namespace KdaPostWu
