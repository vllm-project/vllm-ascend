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
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "../chunk_kda_fwd_varlen.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace AscendC;

namespace KdaFinalize {
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
constexpr uint32_t KDA_CUBE_MIN_REDUCTION = 16;

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
class ChunkKdaFwdFinalizeKernel {
public:
    using OUT_T = float;
    using AKK_T = float;
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
        usedCoreNum_ = tiling.outputUsedCoreNum;
        const uint64_t outputElements = B_ * HV_ * T_ * V_;
        o_.SetGlobalBuffer((__gm__ OUT_T *)workspace);
        u_.SetGlobalBuffer((__gm__ OUT_T *)workspace + outputElements);
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
        ProcessOutAiv();
        ReleaseVectorEvents();
    }

    __aicore__ inline void ProcessAic()
    {
        ProcessOutAic();
    }

private:
    __aicore__ inline void AllocVectorEvents()
    {
        mte2ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte2Event_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        mte3ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
        mte2ToMte3Event_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
        mte3ToMte2Event_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
        sToVEvent_ = pipe_->AllocEventID<HardEvent::S_V>();
        sToMte2Event_ = pipe_->AllocEventID<HardEvent::S_MTE2>();
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
        pipe_->ReleaseEventID<HardEvent::S_V>(sToVEvent_);
        pipe_->ReleaseEventID<HardEvent::S_MTE2>(sToMte2Event_);
        vectorEventsAllocated_ = false;
    }

    __aicore__ inline uint64_t QOffset(uint64_t b, uint64_t h, uint64_t t, uint64_t d) const
    {
        return ((b * H_ + h) * T_ + t) * K_ + d;
    }

    __aicore__ inline uint64_t KVOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d, uint64_t dim) const
    {
        return ((b * HV_ + hv) * T_ + t) * dim + d;
    }

    __aicore__ inline uint64_t OutputOffset(uint64_t b, uint64_t hv, uint64_t t, uint64_t d) const
    {
        return ((b * T_ + t) * HV_ + hv) * V_ + d;
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
    __aicore__ inline void CopyRowsOut(GlobalTensor<CopyT> &dst, uint64_t offset, LocalTensor<CopyT> &src,
                                       uint64_t rows, uint64_t cols, uint64_t dstStride)
    {
        if (cols == dstStride) {
            CopyVectorOut(dst, offset, src, rows * cols);
            return;
        }
        constexpr uint64_t blockBytes = 32;
        const uint64_t rowBytes = cols * sizeof(CopyT);
        const uint64_t gapBytes = (dstStride - cols) * sizeof(CopyT);
        DataCopyParams params{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            1,
#else
            static_cast<uint16_t>(rows),
#endif
            static_cast<uint16_t>(rowBytes / blockBytes),
            0,
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
            0
#else
            static_cast<uint16_t>(gapBytes / blockBytes)
#endif
        };
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        const uint64_t dstRowBytes = dstStride * sizeof(CopyT);
        LoopModeParams loopParams{
            static_cast<uint32_t>(rows), 1, rowBytes, dstRowBytes, 0, 0};
        // Loop-mode registers are core-local state and must not leak across DMA calls.
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
        DataCopy(dst[offset], src, params);
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
#else
        DataCopy(dst[offset], src, params);
#endif
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

    __aicore__ inline void ComputeTailLocalRows(LocalTensor<float> &dst, uint64_t b, uint64_t hv,
                                                uint64_t start, uint64_t curT, uint64_t rowBegin,
                                                uint64_t rows)
    {
        LocalTensor<float> vRow = exp2Buf_.Get<float>();
        LocalTensor<T> coefficientTyped = gateWritebackBuf_.Get<T>();
        LocalTensor<float> coefficients = gateWritebackBuf_.Get<float>()[BT_];
        for (uint64_t localRow = 0; localRow < rows; ++localRow) {
            LocalTensor<float> dstRow = dst[localRow * V_];
            CopyVectorIn(
                coefficientTyped, preparedAqk_,
                AOffset(b, hv, start + rowBegin + localRow, 0), curT);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(
                coefficients, coefficientTyped, RoundMode::CAST_NONE,
                static_cast<uint32_t>(curT));
            // Earlier megakernel stages reuse V_S event IDs. Drain the cast
            // before the first scalar coefficient read in this tail row.
            PipeBarrier<PIPE_ALL>();
            Duplicate(dstRow, 0.0f, static_cast<uint32_t>(V_));
            PipeBarrier<PIPE_V>();
            for (uint64_t j = 0; j < curT; ++j) {
                LoadAsFloatRow(
                    propagatedVNew_, KVOffset(b, hv, start + j, 0, V_), vRow, V_);
                float weight = coefficients.GetValue(j);
                SetFlag<HardEvent::S_V>(sToVEvent_);
                WaitFlag<HardEvent::S_V>(sToVEvent_);
                Muls(vRow, vRow, weight, static_cast<uint32_t>(V_));
                PipeBarrier<PIPE_V>();
                Add(dstRow, dstRow, vRow, static_cast<uint32_t>(V_));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
                WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
            }
            SetFlag<HardEvent::S_MTE2>(sToMte2Event_);
            WaitFlag<HardEvent::S_MTE2>(sToMte2Event_);
        }
    }

    __aicore__ inline void ComputeTailStateRows(LocalTensor<float> &dst, uint64_t b, uint64_t hv,
                                                uint64_t chunkIdx, uint64_t start, uint64_t rowBegin,
                                                uint64_t rows)
    {
        LocalTensor<float> hRow = exp2Buf_.Get<float>();
        LocalTensor<T> coefficientTyped = gateWritebackBuf_.Get<T>();
        LocalTensor<float> coefficients = gateWritebackBuf_.Get<float>()[BT_];
        for (uint64_t localRow = 0; localRow < rows; ++localRow) {
            LocalTensor<float> dstRow = dst[localRow * V_];
            CopyVectorIn(
                coefficientTyped, preparedQG_,
                KVOffset(b, hv, start + rowBegin + localRow, 0, K_), K_);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            Cast(
                coefficients, coefficientTyped, RoundMode::CAST_NONE,
                static_cast<uint32_t>(K_));
            // Earlier megakernel stages reuse V_S event IDs. Drain the cast
            // before the first scalar coefficient read in this tail row.
            PipeBarrier<PIPE_ALL>();
            Duplicate(dstRow, 0.0f, static_cast<uint32_t>(V_));
            PipeBarrier<PIPE_V>();
            for (uint64_t d = 0; d < K_; ++d) {
                LoadAsFloatRow(
                    propagatedH_, HOffset(b, hv, chunkIdx, d, 0), hRow, V_);
                float weight = coefficients.GetValue(d);
                SetFlag<HardEvent::S_V>(sToVEvent_);
                WaitFlag<HardEvent::S_V>(sToVEvent_);
                Muls(hRow, hRow, weight, static_cast<uint32_t>(V_));
                PipeBarrier<PIPE_V>();
                Add(dstRow, dstRow, hRow, static_cast<uint32_t>(V_));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
                WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
            }
            SetFlag<HardEvent::S_MTE2>(sToMte2Event_);
            WaitFlag<HardEvent::S_MTE2>(sToMte2Event_);
        }
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
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    __aicore__ inline void ComputeOutputCubeStagedArch35(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                     uint64_t curT)
    {
        SetMMLayoutTransform(true);
        using ElementA = T;
        using ElementB = T;
        using ElementC = OUT_T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, ElementA, LayoutTagL1A>;

        constexpr uint16_t kMte2Event = 0;
        constexpr uint16_t kMte1Event = 0;
        constexpr uint16_t kMmadEvent = 0;
        constexpr uint32_t kL1A0Offset = 0;
        constexpr uint32_t kL1B0Offset = 64 * 128 * sizeof(ElementA);
        constexpr uint32_t kL1A1Offset = kL1B0Offset + 128 * 128 * sizeof(ElementB);
        constexpr uint32_t kL1B1Offset = kL1A1Offset + 64 * 64 * sizeof(ElementA);

        Catlass::Arch::Resource<KdaArchTag> resource;
        LocalTensor<ElementA> l1A0 = resource.l1Buf.template GetBufferByByte<ElementA>(kL1A0Offset);
        LocalTensor<ElementB> l1B0 = resource.l1Buf.template GetBufferByByte<ElementB>(kL1B0Offset);
        LocalTensor<ElementA> l1A1 = resource.l1Buf.template GetBufferByByte<ElementA>(kL1A1Offset);
        LocalTensor<ElementB> l1B1 = resource.l1Buf.template GetBufferByByte<ElementB>(kL1B1Offset);
        LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        LocalTensor<ElementC> l0C = resource.l0CBuf.template GetBufferByByte<ElementC>(0);

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        auto layoutQ = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutH = tla::MakeLayout<ElementB, LayoutTagB>(K_, V_);
        auto layoutO = tla::MakeLayout<ElementC, LayoutTagC>(BT_, V_);
        auto layoutAqk = tla::MakeLayout<ElementA, LayoutTagA>(BT_, BT_);
        auto layoutV = tla::MakeLayout<ElementB, LayoutTagB>(BT_, V_);

        for (uint64_t nOffset = 0; nOffset < V_; nOffset += 128) {
            const uint32_t curN = static_cast<uint32_t>((V_ - nOffset) > 128 ? 128 : (V_ - nOffset));
            auto tensorH = tla::MakeTensor(propagatedH_[HOffset(b, hv, chunkIdx, 0, nOffset)], layoutH,
                                           Catlass::Arch::PositionGM{});
            auto tensorVNew = tla::MakeTensor(propagatedVNew_[KVOffset(b, hv, start, nOffset, V_)], layoutV,
                                              Catlass::Arch::PositionGM{});

            for (uint64_t mOffset = 0; mOffset < curT; mOffset += 64) {
                const uint32_t curM = static_cast<uint32_t>((curT - mOffset) > 64 ? 64 : (curT - mOffset));
                auto tensorQ = tla::MakeTensor(preparedQG_[KVOffset(b, hv, start + mOffset, 0, K_)], layoutQ,
                                               Catlass::Arch::PositionGM{});
                auto tensorAqk = tla::MakeTensor(preparedAqk_[AOffset(b, hv, start + mOffset, 0)], layoutAqk,
                                                 Catlass::Arch::PositionGM{});
                auto tensorO = tla::MakeTensor(o_[KVOffset(b, hv, start + mOffset, nOffset, V_)], layoutO,
                                               Catlass::Arch::PositionGM{});
                auto tensorLocal = tla::MakeTensor(u_[KVOffset(b, hv, start + mOffset, nOffset, V_)], layoutO,
                                                   Catlass::Arch::PositionGM{});

                auto blockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(curM, K_));
                auto blockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(K_, curN));
                auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(curM, curT));
                auto blockVNew = GetTile(tensorVNew, tla::MakeCoord(0, 0), tla::MakeShape(curT, curN));
                auto blockO = GetTile(tensorO, tla::MakeCoord(0, 0), tla::MakeShape(curM, curN));
                auto blockLocal =
                    GetTile(tensorLocal, tla::MakeCoord(0, 0), tla::MakeShape(curM, curN));

                using CopyGmToL1A0 = typename TileCopy::template CopyGmToL1A<decltype(blockQ)>;
                using CopyGmToL1B0 = typename TileCopy::template CopyGmToL1B<decltype(blockH)>;
                using CopyGmToL1A1 = typename TileCopy::template CopyGmToL1A<decltype(blockAqk)>;
                using CopyGmToL1B1 = typename TileCopy::template CopyGmToL1B<decltype(blockVNew)>;
                using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(blockO)>;
                CopyGmToL1A0 copyGmToL1A0;
                CopyGmToL1B0 copyGmToL1B0;
                CopyGmToL1A1 copyGmToL1A1;
                CopyGmToL1B1 copyGmToL1B1;
                CopyL0CToDst copyL0CToDst;

                auto layoutL1A0 = tla::MakeLayout<ElementA, LayoutTagL1A>(curM, K_);
                auto layoutL1B0 = tla::MakeLayout<ElementB, LayoutTagL1B>(K_, curN);
                auto layoutL1A1 = tla::MakeLayout<ElementA, LayoutTagL1A>(curM, curT);
                auto layoutL1B1 = tla::MakeLayout<ElementB, LayoutTagL1B>(curT, curN);
                auto layoutL0A0 = tla::MakeLayout<ElementA, LayoutTagL0A>(curM, K_);
                auto layoutL0B0 = tla::MakeLayout<ElementB, LayoutTagL0B>(K_, curN);
                auto layoutL0A1 = tla::MakeLayout<ElementA, LayoutTagL0A>(curM, curT);
                auto layoutL0B1 = tla::MakeLayout<ElementB, LayoutTagL0B>(curT, curN);
                auto layoutL0C = tla::MakeLayoutL0C(curM, curN);

                auto tensorL1A0 = tla::MakeTensor(l1A0, layoutL1A0, Catlass::Arch::PositionL1{});
                auto tensorL1B0 = tla::MakeTensor(l1B0, layoutL1B0, Catlass::Arch::PositionL1{});
                auto tensorL1A1 = tla::MakeTensor(l1A1, layoutL1A1, Catlass::Arch::PositionL1{});
                auto tensorL1B1 = tla::MakeTensor(l1B1, layoutL1B1, Catlass::Arch::PositionL1{});
                auto tensorL0A0 = tla::MakeTensor(l0A, layoutL0A0, Catlass::Arch::PositionL0A{});
                auto tensorL0B0 = tla::MakeTensor(l0B, layoutL0B0, Catlass::Arch::PositionL0B{});
                auto tensorL0A1 = tla::MakeTensor(l0A, layoutL0A1, Catlass::Arch::PositionL0A{});
                auto tensorL0B1 = tla::MakeTensor(l0B, layoutL0B1, Catlass::Arch::PositionL0B{});
                auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
                uint32_t localRow = 0;
                uint32_t localColumn = 0;
                auto tileL1A0 = GetTile(tensorL1A0, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curM, K_));
                auto tileL1B0 = GetTile(tensorL1B0, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(K_, curN));
                auto tileL1A1 = GetTile(tensorL1A1, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curM, curT));
                auto tileL1B1 = GetTile(tensorL1B1, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curT, curN));
                auto tileL0A0 = GetTile(tensorL0A0, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curM, K_));
                auto tileL0B0 = GetTile(tensorL0B0, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(K_, curN));
                auto tileL0A1 = GetTile(tensorL0A1, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curM, curT));
                auto tileL0B1 = GetTile(tensorL0B1, tla::MakeCoord(localRow, localColumn),
                                        tla::MakeShape(curT, curN));
                auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(localRow, localColumn),
                                       tla::MakeShape(curM, curN));

                copyGmToL1A0(tensorL1A0, blockQ);
                copyGmToL1B0(tensorL1B0, blockH);
                copyGmToL1A1(tensorL1A1, blockAqk);
                copyGmToL1B1(tensorL1B1, blockVNew);
                SetFlag<HardEvent::MTE2_MTE1>(kMte2Event);
                WaitFlag<HardEvent::MTE2_MTE1>(kMte2Event);

                copyL1ToL0A(tileL0A0, tileL1A0);
                copyL1ToL0B(tileL0B0, tileL1B0);
                SetFlag<HardEvent::MTE1_M>(kMte1Event);
                WaitFlag<HardEvent::MTE1_M>(kMte1Event);
                tileMmad(tileL0C, tileL0A0, tileL0B0, curM, curN, static_cast<uint32_t>(K_), true, 0b11);
                SetFlag<HardEvent::M_MTE1>(kMmadEvent);
                WaitFlag<HardEvent::M_MTE1>(kMmadEvent);
                copyL0CToDst(blockO, tileL0C, 0b11);
                PipeBarrier<PIPE_ALL>();

                copyL1ToL0A(tileL0A1, tileL1A1);
                copyL1ToL0B(tileL0B1, tileL1B1);
                SetFlag<HardEvent::MTE1_M>(kMte1Event);
                SetFlag<HardEvent::MTE1_MTE2>(kMte2Event);
                WaitFlag<HardEvent::MTE1_M>(kMte1Event);
                WaitFlag<HardEvent::MTE1_MTE2>(kMte2Event);
                tileMmad(tileL0C, tileL0A1, tileL0B1, curM, curN, static_cast<uint32_t>(curT), true, 0b11);
                SetFlag<HardEvent::M_MTE1>(kMmadEvent);
                WaitFlag<HardEvent::M_MTE1>(kMmadEvent);
                copyL0CToDst(blockLocal, tileL0C, 0b11);
                PipeBarrier<PIPE_ALL>();
            }
        }
        SetMMLayoutTransform(false);
    }

    __aicore__ inline void PrefetchOutputTileArch35(Catlass::Arch::Resource<KdaArchTag> &resource, uint32_t slot,
                                                uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                                uint64_t curT, uint64_t nOffset, bool reuseSlot)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = OUT_T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;

        constexpr uint32_t kL1SlotBytes = 96 * 1024;
        constexpr uint32_t kL1A0Offset = 0;
        constexpr uint32_t kL1B0Offset = 64 * 128 * sizeof(ElementA);
        constexpr uint32_t kL1A1Offset = kL1B0Offset + 128 * 128 * sizeof(ElementB);
        constexpr uint32_t kL1B1Offset = kL1A1Offset + 64 * 64 * sizeof(ElementA);
        const uint32_t slotBase = slot * kL1SlotBytes;
        const uint32_t curM = static_cast<uint32_t>(curT);
        const uint32_t curN = static_cast<uint32_t>((V_ - nOffset) > 128 ? 128 : (V_ - nOffset));

        auto layoutQ = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutH = tla::MakeLayout<ElementB, LayoutTagB>(K_, V_);
        auto layoutAqk = tla::MakeLayout<ElementA, LayoutTagA>(BT_, BT_);
        auto layoutV = tla::MakeLayout<ElementB, LayoutTagB>(BT_, V_);
        auto tensorQ = tla::MakeTensor(preparedQG_[KVOffset(b, hv, start, 0, K_)], layoutQ,
                                       Catlass::Arch::PositionGM{});
        auto tensorH = tla::MakeTensor(propagatedH_[HOffset(b, hv, chunkIdx, 0, nOffset)], layoutH,
                                       Catlass::Arch::PositionGM{});
        auto tensorAqk = tla::MakeTensor(preparedAqk_[AOffset(b, hv, start, 0)], layoutAqk,
                                         Catlass::Arch::PositionGM{});
        auto tensorVNew = tla::MakeTensor(propagatedVNew_[KVOffset(b, hv, start, nOffset, V_)], layoutV,
                                          Catlass::Arch::PositionGM{});
        auto blockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(curM, K_));
        auto blockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(K_, curN));
        auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(curM, curT));
        auto blockVNew = GetTile(tensorVNew, tla::MakeCoord(0, 0), tla::MakeShape(curT, curN));

        using CopyGmToL1A0 = typename TileCopy::template CopyGmToL1A<decltype(blockQ)>;
        using CopyGmToL1B0 = typename TileCopy::template CopyGmToL1B<decltype(blockH)>;
        using CopyGmToL1A1 = typename TileCopy::template CopyGmToL1A<decltype(blockAqk)>;
        using CopyGmToL1B1 = typename TileCopy::template CopyGmToL1B<decltype(blockVNew)>;
        CopyGmToL1A0 copyGmToL1A0;
        CopyGmToL1B0 copyGmToL1B0;
        CopyGmToL1A1 copyGmToL1A1;
        CopyGmToL1B1 copyGmToL1B1;

        LocalTensor<ElementA> l1A0 =
            resource.l1Buf.template GetBufferByByte<ElementA>(slotBase + kL1A0Offset);
        LocalTensor<ElementB> l1B0 =
            resource.l1Buf.template GetBufferByByte<ElementB>(slotBase + kL1B0Offset);
        LocalTensor<ElementA> l1A1 =
            resource.l1Buf.template GetBufferByByte<ElementA>(slotBase + kL1A1Offset);
        LocalTensor<ElementB> l1B1 =
            resource.l1Buf.template GetBufferByByte<ElementB>(slotBase + kL1B1Offset);
        auto tensorL1A0 = tla::MakeTensor(
            l1A0, tla::MakeLayout<ElementA, LayoutTagL1A>(curM, K_), Catlass::Arch::PositionL1{});
        auto tensorL1B0 = tla::MakeTensor(
            l1B0, tla::MakeLayout<ElementB, LayoutTagL1B>(K_, curN), Catlass::Arch::PositionL1{});
        auto tensorL1A1 = tla::MakeTensor(
            l1A1, tla::MakeLayout<ElementA, LayoutTagL1A>(curM, curT), Catlass::Arch::PositionL1{});
        auto tensorL1B1 = tla::MakeTensor(
            l1B1, tla::MakeLayout<ElementB, LayoutTagL1B>(curT, curN), Catlass::Arch::PositionL1{});

        if (reuseSlot) {
            WaitFlag<HardEvent::MTE1_MTE2>(slot);
        }
        copyGmToL1A0(tensorL1A0, blockQ);
        copyGmToL1B0(tensorL1B0, blockH);
        copyGmToL1A1(tensorL1A1, blockAqk);
        copyGmToL1B1(tensorL1B1, blockVNew);
        SetFlag<HardEvent::MTE2_MTE1>(slot);
    }

    __aicore__ inline void ComputePrefetchedOutputTileArch35(Catlass::Arch::Resource<KdaArchTag> &resource,
                                                         uint32_t slot, uint64_t b, uint64_t hv, uint64_t start,
                                                         uint64_t curT, uint64_t nOffset)
    {
        using ElementA = T;
        using ElementB = T;
        using ElementC = OUT_T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<KdaArchTag, ElementA, LayoutTagL1A>;

        constexpr uint16_t kMte1Event = 0;
        constexpr uint16_t kMmadEvent = 0;
        constexpr uint16_t kFixEvent = 0;
        constexpr uint32_t kL1SlotBytes = 96 * 1024;
        constexpr uint32_t kL1A0Offset = 0;
        constexpr uint32_t kL1B0Offset = 64 * 128 * sizeof(ElementA);
        constexpr uint32_t kL1A1Offset = kL1B0Offset + 128 * 128 * sizeof(ElementB);
        constexpr uint32_t kL1B1Offset = kL1A1Offset + 64 * 64 * sizeof(ElementA);
        const uint32_t slotBase = slot * kL1SlotBytes;
        const uint32_t curM = static_cast<uint32_t>(curT);
        const uint32_t curN = static_cast<uint32_t>((V_ - nOffset) > 128 ? 128 : (V_ - nOffset));

        LocalTensor<ElementA> l1A0 =
            resource.l1Buf.template GetBufferByByte<ElementA>(slotBase + kL1A0Offset);
        LocalTensor<ElementB> l1B0 =
            resource.l1Buf.template GetBufferByByte<ElementB>(slotBase + kL1B0Offset);
        LocalTensor<ElementA> l1A1 =
            resource.l1Buf.template GetBufferByByte<ElementA>(slotBase + kL1A1Offset);
        LocalTensor<ElementB> l1B1 =
            resource.l1Buf.template GetBufferByByte<ElementB>(slotBase + kL1B1Offset);
        LocalTensor<ElementA> l0A = resource.l0ABuf.template GetBufferByByte<ElementA>(0);
        LocalTensor<ElementB> l0B = resource.l0BBuf.template GetBufferByByte<ElementB>(0);
        LocalTensor<ElementC> l0C = resource.l0CBuf.template GetBufferByByte<ElementC>(0);

        auto tensorL1A0 = tla::MakeTensor(
            l1A0, tla::MakeLayout<ElementA, LayoutTagL1A>(curM, K_), Catlass::Arch::PositionL1{});
        auto tensorL1B0 = tla::MakeTensor(
            l1B0, tla::MakeLayout<ElementB, LayoutTagL1B>(K_, curN), Catlass::Arch::PositionL1{});
        auto tensorL1A1 = tla::MakeTensor(
            l1A1, tla::MakeLayout<ElementA, LayoutTagL1A>(curM, curT), Catlass::Arch::PositionL1{});
        auto tensorL1B1 = tla::MakeTensor(
            l1B1, tla::MakeLayout<ElementB, LayoutTagL1B>(curT, curN), Catlass::Arch::PositionL1{});
        auto tensorL0A0 = tla::MakeTensor(
            l0A, tla::MakeLayout<ElementA, LayoutTagL0A>(curM, K_), Catlass::Arch::PositionL0A{});
        auto tensorL0B0 = tla::MakeTensor(
            l0B, tla::MakeLayout<ElementB, LayoutTagL0B>(K_, curN), Catlass::Arch::PositionL0B{});
        auto tensorL0A1 = tla::MakeTensor(
            l0A, tla::MakeLayout<ElementA, LayoutTagL0A>(curM, curT), Catlass::Arch::PositionL0A{});
        auto tensorL0B1 = tla::MakeTensor(
            l0B, tla::MakeLayout<ElementB, LayoutTagL0B>(curT, curN), Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(curM, curN), Catlass::Arch::PositionL0C{});

        uint32_t localRow = 0;
        uint32_t localColumn = 0;
        auto tileL1A0 = GetTile(tensorL1A0, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curM, K_));
        auto tileL1B0 = GetTile(tensorL1B0, tla::MakeCoord(localRow, localColumn), tla::MakeShape(K_, curN));
        auto tileL1A1 = GetTile(tensorL1A1, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curM, curT));
        auto tileL1B1 = GetTile(tensorL1B1, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curT, curN));
        auto tileL0A0 = GetTile(tensorL0A0, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curM, K_));
        auto tileL0B0 = GetTile(tensorL0B0, tla::MakeCoord(localRow, localColumn), tla::MakeShape(K_, curN));
        auto tileL0A1 = GetTile(tensorL0A1, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curM, curT));
        auto tileL0B1 = GetTile(tensorL0B1, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curT, curN));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(localRow, localColumn), tla::MakeShape(curM, curN));

        auto layoutO = tla::MakeLayout<ElementC, LayoutTagC>(BT_, V_);
        auto tensorO = tla::MakeTensor(o_[KVOffset(b, hv, start, nOffset, V_)], layoutO,
                                       Catlass::Arch::PositionGM{});
        auto tensorLocal = tla::MakeTensor(u_[KVOffset(b, hv, start, nOffset, V_)], layoutO,
                                           Catlass::Arch::PositionGM{});
        auto blockO = GetTile(tensorO, tla::MakeCoord(0, 0), tla::MakeShape(curM, curN));
        auto blockLocal = GetTile(tensorLocal, tla::MakeCoord(0, 0), tla::MakeShape(curM, curN));
        using CopyL0CToDst = typename TileCopy::template CopyL0CToDst<decltype(blockO)>;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToDst copyL0CToDst;
        TileMmad tileMmad;

        WaitFlag<HardEvent::MTE2_MTE1>(slot);
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            WaitFlag<HardEvent::FIX_M>(kFixEvent);
        }
        copyL1ToL0A(tileL0A0, tileL1A0);
        copyL1ToL0B(tileL0B0, tileL1B0);
        SetFlag<HardEvent::MTE1_M>(kMte1Event);
        WaitFlag<HardEvent::MTE1_M>(kMte1Event);
        tileMmad(tileL0C, tileL0A0, tileL0B0, curM, curN, static_cast<uint32_t>(K_), true, 0b11);
        SetFlag<HardEvent::M_MTE1>(kMmadEvent);
        WaitFlag<HardEvent::M_MTE1>(kMmadEvent);
        if constexpr (!IsSameType<T, bfloat16_t>::value) {
            copyL0CToDst(blockO, tileL0C, 0b11);
            PipeBarrier<PIPE_ALL>();
        }

        copyL1ToL0A(tileL0A1, tileL1A1);
        copyL1ToL0B(tileL0B1, tileL1B1);
        SetFlag<HardEvent::MTE1_M>(kMte1Event);
        SetFlag<HardEvent::MTE1_MTE2>(slot);
        WaitFlag<HardEvent::MTE1_M>(kMte1Event);
        tileMmad(tileL0C, tileL0A1, tileL0B1, curM, curN, static_cast<uint32_t>(curT),
                 !IsSameType<T, bfloat16_t>::value, 0b11);
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            SetFlag<HardEvent::M_FIX>(kFixEvent);
            WaitFlag<HardEvent::M_FIX>(kFixEvent);
            auto fixParams = FixpipeParamsV220(
                curN, curM, curN, static_cast<uint32_t>(HV_ * V_), false);
            fixParams.quantPre = QuantMode_t::F322BF16;
            Fixpipe<T, float, CFG_ROW_MAJOR>(
                vNew_[OutputOffset(b, hv, start, nOffset)], l0C, fixParams);
            SetFlag<HardEvent::FIX_M>(kFixEvent);
        } else {
            SetFlag<HardEvent::M_MTE1>(kMmadEvent);
            WaitFlag<HardEvent::M_MTE1>(kMmadEvent);
            copyL0CToDst(blockLocal, tileL0C, 0b11);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessOutAicPipelinedArch35()
    {
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        SetMMLayoutTransform(true);
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t currentTask = static_cast<uint64_t>(GetBlockIdx());
        uint64_t seq = 0;
        uint64_t b = 0;
        uint64_t h = 0;
        uint64_t hv = 0;
        uint64_t chunkIdx = 0;
        uint64_t start = 0;
        uint64_t end = 0;
        while (currentTask < taskNum &&
               !ResolveFlatChunk(currentTask, seq, b, h, hv, chunkIdx, start, end)) {
            currentTask += coreNum;
        }
        if (currentTask >= taskNum) {
            SetMMLayoutTransform(false);
            return;
        }

        Catlass::Arch::Resource<KdaArchTag> resource;
        uint64_t nOffset = 0;
        uint32_t slot = 0;
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            SetFlag<HardEvent::FIX_M>(0);
        }
        PrefetchOutputTileArch35(resource, slot, b, hv, chunkIdx, start, end - start, nOffset, false);
        uint64_t outputTileIdx = 0;

        while (true) {
            uint64_t nextTask = currentTask;
            uint64_t nextSeq = seq;
            uint64_t nextB = b;
            uint64_t nextH = h;
            uint64_t nextHv = hv;
            uint64_t nextChunkIdx = chunkIdx;
            uint64_t nextStart = start;
            uint64_t nextEnd = end;
            uint64_t nextNOffset = nOffset + 128;
            bool hasNext = nextNOffset < V_;
            if (!hasNext) {
                nextTask += coreNum;
                nextNOffset = 0;
                while (nextTask < taskNum &&
                       !ResolveFlatChunk(nextTask, nextSeq, nextB, nextH, nextHv, nextChunkIdx, nextStart,
                                         nextEnd)) {
                    nextTask += coreNum;
                }
                hasNext = nextTask < taskNum;
            }

            const uint32_t nextSlot = slot ^ 1U;
            if (hasNext) {
                PrefetchOutputTileArch35(resource, nextSlot, nextB, nextHv, nextChunkIdx, nextStart,
                                     nextEnd - nextStart, nextNOffset,
                                     outputTileIdx + 1 >= 2);
            }
            ComputePrefetchedOutputTileArch35(resource, slot, b, hv, start, end - start, nOffset);
            if constexpr (!IsSameType<T, bfloat16_t>::value) {
                if (nOffset + 128 >= V_) {
                    Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
                }
            }
            if (!hasNext) {
                break;
            }
            ++outputTileIdx;

            currentTask = nextTask;
            seq = nextSeq;
            b = nextB;
            h = nextH;
            hv = nextHv;
            chunkIdx = nextChunkIdx;
            start = nextStart;
            end = nextEnd;
            nOffset = nextNOffset;
            slot = nextSlot;
        }
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            WaitFlag<HardEvent::FIX_M>(0);
        }
        SetMMLayoutTransform(false);
    }
#endif

    __aicore__ inline void ComputeOutputCube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                             uint64_t curT)
    {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (curT < KDA_CUBE_MIN_REDUCTION) {
            return;
        }
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        if (BT_ == 64 && curT == BT_) {
            ComputeOutputCubeStagedArch35(b, hv, chunkIdx, start, curT);
            return;
        }
#endif
        using ElementA = T;
        using ElementB = T;
        using ElementC = OUT_T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using TileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                LayoutTagB, ElementC, LayoutTagC>;
        using BlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, KdaL1TileShape, KdaL0TileShape,
                                                              ElementA, ElementB, ElementC, void, TileCopy>;

        Catlass::Arch::Resource<KdaArchTag> resource;
        BlockMmad blockMmad(resource);

        auto layoutQ = tla::MakeLayout<ElementA, LayoutTagA>(BT_, K_);
        auto layoutH = tla::MakeLayout<ElementB, LayoutTagB>(K_, V_);
        auto layoutO = tla::MakeLayout<ElementC, LayoutTagC>(BT_, V_);
        for (uint64_t nOffset = 0; nOffset < V_; nOffset += 128) {
            uint32_t curN = static_cast<uint32_t>((V_ - nOffset) > 128 ? 128 : (V_ - nOffset));
            auto tensorH = tla::MakeTensor(propagatedH_[HOffset(b, hv, chunkIdx, 0, nOffset)], layoutH,
                                           Catlass::Arch::PositionGM{});
            for (uint64_t mOffset = 0; mOffset < curT; mOffset += 64) {
                uint32_t curM = static_cast<uint32_t>((curT - mOffset) > 64 ? 64 : (curT - mOffset));
                Catlass::GemmCoord shapeQH{curM, curN, static_cast<uint32_t>(K_)};
                auto tensorQ = tla::MakeTensor(preparedQG_[KVOffset(b, hv, start + mOffset, 0, K_)], layoutQ,
                                               Catlass::Arch::PositionGM{});
                auto tensorO = tla::MakeTensor(o_[KVOffset(b, hv, start + mOffset, nOffset, V_)], layoutO,
                                               Catlass::Arch::PositionGM{});
                auto blockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.m(), shapeQH.k()));
                auto blockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.k(), shapeQH.n()));
                auto blockO = GetTile(tensorO, tla::MakeCoord(0, 0), tla::MakeShape(shapeQH.m(), shapeQH.n()));
                blockMmad(blockQ, blockH, blockO, shapeQH);
                PipeBarrier<PIPE_ALL>();
            }
        }

        if (curT < KDA_CUBE_MIN_REDUCTION) {
            return;
        }

        auto layoutAqk = tla::MakeLayout<ElementA, LayoutTagA>(BT_, BT_);
        auto layoutV = tla::MakeLayout<ElementB, LayoutTagB>(BT_, V_);
        for (uint64_t nOffset = 0; nOffset < V_; nOffset += 128) {
            uint32_t curN = static_cast<uint32_t>((V_ - nOffset) > 128 ? 128 : (V_ - nOffset));
            auto tensorVNew = tla::MakeTensor(propagatedVNew_[KVOffset(b, hv, start, nOffset, V_)], layoutV,
                                              Catlass::Arch::PositionGM{});
            for (uint64_t mOffset = 0; mOffset < curT; mOffset += 64) {
                uint32_t curM = static_cast<uint32_t>((curT - mOffset) > 64 ? 64 : (curT - mOffset));
                Catlass::GemmCoord shapeAV{curM, curN, static_cast<uint32_t>(curT)};
                auto tensorAqk = tla::MakeTensor(preparedAqk_[AOffset(b, hv, start + mOffset, 0)], layoutAqk,
                                                 Catlass::Arch::PositionGM{});
                auto tensorLocal = tla::MakeTensor(u_[KVOffset(b, hv, start + mOffset, nOffset, V_)], layoutO,
                                                   Catlass::Arch::PositionGM{});
                auto blockAqk = GetTile(tensorAqk, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.m(), shapeAV.k()));
                auto blockVNew = GetTile(tensorVNew, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.k(), shapeAV.n()));
                auto blockLocal = GetTile(tensorLocal, tla::MakeCoord(0, 0), tla::MakeShape(shapeAV.m(), shapeAV.n()));
                blockMmad(blockAqk, blockVNew, blockLocal, shapeAV);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void FinalizeOutputRows(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t curT, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        if (subBlockNum == 0 || subBlockIdx >= subBlockNum || V_ == 0) {
            return;
        }
        const uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        const uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        const uint64_t gateWritebackRows =
            ScoreVectorMaxRows(5 * sizeof(float) + 2 * sizeof(T) + sizeof(GK_T));
        const uint64_t gateWritebackBytes =
            gateWritebackRows * K_ * (3 * sizeof(T) + sizeof(GK_T));
        uint64_t maxRows = KDA_VEC_ARENA_ELEMENTS / (3 * V_);
        const uint64_t typedMaxRows = gateWritebackBytes / (V_ * sizeof(T));
        if (maxRows > typedMaxRows) {
            maxRows = typedMaxRows;
        }
        if (maxRows == 0) {
            return;
        }

        for (uint64_t tileRow = rowBegin; tileRow < rowEnd; tileRow += maxRows) {
            uint64_t tileRows = rowEnd - tileRow;
            if (tileRows > maxRows) {
                tileRows = maxRows;
            }
            const uint64_t elems = tileRows * V_;
            const uint64_t ti = start + tileRow;
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> stateLocal = arena;
            LocalTensor<float> localLocal = arena[elems];
            LocalTensor<float> outLocal = arena[2 * elems];
            LocalTensor<T> outTyped = gateWritebackBuf_.Get<T>();

            if (curT < KDA_CUBE_MIN_REDUCTION) {
                ComputeTailStateRows(
                    stateLocal, b, hv, chunkIdx, start, tileRow, tileRows);
                ComputeTailLocalRows(localLocal, b, hv, start, curT, tileRow, tileRows);
            } else {
                CopyVectorIn(stateLocal, o_, KVOffset(b, hv, ti, 0, V_), elems);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                CopyVectorIn(localLocal, u_, KVOffset(b, hv, ti, 0, V_), elems);
                SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            }
            Add(outLocal, stateLocal, localLocal, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(outLocal, static_cast<uint32_t>(elems));
            Cast(outTyped, outLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(elems));
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyRowsOut(vNew_, OutputOffset(b, hv, ti, 0), outTyped, tileRows, V_, HV_ * V_);
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2Event_);
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

    __aicore__ inline void ProcessChunkOutAiv(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                              uint64_t end, uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(syncDoneFlag_);
        FinalizeOutputRows(b, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum);
    }

    __aicore__ inline void ProcessChunkOutAic(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                             uint64_t end)
    {
        uint64_t curT = end - start;
        if (curT == 0) {
            return;
        }
        ComputeOutputCube(b, hv, chunkIdx, start, curT);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
    }

    __aicore__ inline void ProcessOutAiv()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            if (!isVarLen_ && T_ % BT_ == 0 && BT_ == 64 && K_ == 128 && V_ == 128) {
                return;
            }
        }
#endif
        uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        if (subBlockNum == 0) {
            return;
        }
        uint64_t subBlockIdx = static_cast<uint64_t>(GetSubBlockIdx());
        uint64_t coreNum = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
        uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        uint64_t taskNum = static_cast<uint64_t>((isVarLen_ ? NT_ : B_ * NT_) * HV_);
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
                (void)h;
                (void)chunkIdx;
                ProcessChunkOutAiv(b, hv, chunkIdx, start, end, subBlockIdx, subBlockNum);
            }
        }
    }

    __aicore__ inline void ProcessOutAic()
    {
        if constexpr (IsSameType<T, float>::value) {
            return;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
        if (!isVarLen_ && T_ % BT_ == 0 && BT_ == 64 && K_ == 128 && V_ == 128) {
            ProcessOutAicPipelinedArch35();
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
                ProcessChunkOutAic(b, hv, chunkIdx, start, end);
            }
        }
    }

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
    TEventID sToVEvent_ = 0;
    TEventID sToMte2Event_ = 0;
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
    uint64_t usedCoreNum_ = 1;
    uint64_t solveCoreIdx_ = 0;
    __gm__ int64_t *chunkIndicesAddr_ = nullptr;
    __gm__ int64_t *cuSeqlensAddr_ = nullptr;
};
} // namespace

template <typename T, typename GK_T, typename BETA_T, typename TilingData>
__aicore__ inline void RunChunkKdaOutput(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR qgScaled, GM_ADDR aqk,
    GM_ADDR propagatedVNew, GM_ADDR propagatedH, GM_ADDR o, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe)
{
    GM_ADDR outputScratch = userWorkspace + tiling.outputScratchOffset;
    uint64_t outputElements = static_cast<uint64_t>(tiling.batch) *
                              static_cast<uint64_t>(tiling.vHeadNum) *
                              static_cast<uint64_t>(tiling.seqlen) *
                              static_cast<uint64_t>(tiling.vHeadDim);
    GM_ADDR stateScratch = outputScratch;
    GM_ADDR localScratch = outputScratch + outputElements * sizeof(float);
    if ASCEND_IS_AIC {
        ChunkKdaFwdFinalizeKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                qgScaled, aqk, propagatedVNew, propagatedH, stateScratch, userWorkspace, aqk, userWorkspace,
                userWorkspace, localScratch, userWorkspace, userWorkspace, o, propagatedH,
                outputScratch, tiling, &pipe, false);
        op.ProcessAic();
    }
    if ASCEND_IS_AIV {
        ChunkKdaFwdFinalizeKernel<T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                qgScaled, aqk, propagatedVNew, propagatedH, stateScratch, userWorkspace, aqk, userWorkspace,
                userWorkspace, localScratch, userWorkspace, userWorkspace, o, propagatedH,
                outputScratch, tiling, &pipe);
        op.ProcessAiv();
    }
}

} // namespace KdaFinalize
