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
#define CATLASS_ARCH 2201
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
#include "chunk_kda_fwd_varlen.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using namespace AscendC;

namespace KdaPrepare {
namespace {
using KdaInt64 = tla::Int<64>;
using KdaInt128 = tla::Int<128>;
constexpr float LN2 = 0.69314718055994530942f;
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
constexpr uint32_t KDA_SCORE_QUEUE_DEPTH = 2;
constexpr uint32_t KDA_SCORE_LANES = 2;
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
using KdaArchTag = Catlass::Arch::AtlasA2;
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

template <bool SAFE_GATE, typename T, typename GK_T = float, typename BETA_T = float>
class ChunkKdaFwdPrepareKernel {
public:
    using OUT_T = T;
    using AKK_T = float;
    using SCORE_T =
        std::conditional_t<SAFE_GATE && IsSameType<T, half>::value, bfloat16_t, T>;
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
        usedCoreNum_ = tiling.prepareUsedCoreNum;
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
            const uint64_t gateInputSlotBytes = gateStageElems * (2 * sizeof(T) + sizeof(GK_T));
            const uint64_t gatePipelineBytes =
                KDA_GATE_PIPELINE_DEPTH * (gateInputSlotBytes + gateStageElems * sizeof(T));
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

private:
    __aicore__ inline void AllocVectorEvents()
    {
        mte2ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vToMte2Event_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        vToMte3Event_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        mte3ToVEvent_ = pipe_->AllocEventID<HardEvent::MTE3_V>();
        mte2ToMte3Event_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
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
        uint64_t bytesPerRow =
            K_ * KDA_GATE_PIPELINE_DEPTH * (3 * sizeof(T) + sizeof(GK_T));
        uint64_t rows = bytesPerRow == 0 ? 0 : availableBytes / bytesPerRow;
        return rows < KDA_GATE_TILE_ROWS ? rows : KDA_GATE_TILE_ROWS;
    }

    __aicore__ inline uint64_t GateInputSlotBytes() const
    {
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

    __aicore__ inline LocalTensor<T> GateKgTyped(uint64_t slot)
    {
        uint64_t byteOffset = KDA_GATE_PIPELINE_DEPTH * GateInputSlotBytes() +
                              slot * GateStageElems() * sizeof(T);
        return gateWritebackBuf_.Get<T>()[byteOffset / sizeof(T)];
    }

    __aicore__ inline void PrefetchQKGate(uint64_t slot, uint64_t b, uint64_t h, uint64_t hv,
                                          uint64_t token, uint64_t elems)
    {
        const uint64_t rows = elems / K_;
        LocalTensor<T> qTyped = GateQTyped(slot);
        LocalTensor<T> kTyped = GateKTyped(slot);
        LocalTensor<GK_T> gateTyped = GateGTyped(slot);
        CopyRowsIn(qTyped, q_, QOffset(b, h, token, 0), rows, K_,
                   inputSequenceMajor_ ? H_ * K_ : K_);
        CopyRowsIn(kTyped, k_, QOffset(b, h, token, 0), rows, K_,
                   inputSequenceMajor_ ? H_ * K_ : K_);
        CopyVectorIn(gateTyped, gk_, KVOffset(b, hv, token, 0, K_), elems);
        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline void PrefetchKGate(uint64_t slot, uint64_t b, uint64_t h, uint64_t hv,
                                         uint64_t token, uint64_t elems)
    {
        const uint64_t rows = elems / K_;
        LocalTensor<T> kTyped = GateQTyped(slot);
        LocalTensor<GK_T> gateTyped = GateGTyped(slot);
        CopyRowsIn(kTyped, k_, QOffset(b, h, token, 0), rows, K_,
                   inputSequenceMajor_ ? H_ * K_ : K_);
        CopyVectorIn(gateTyped, gk_, KVOffset(b, hv, token, 0, K_), elems);
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
                                                    uint64_t scoreSlot)
    {
        LocalTensor<float> refFp32 = exp2Buf_.Get<float>();
        LoadAsFloatRow(gk_, KVOffset(b, hv, refToken, 0, K_), refFp32, K_);

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
            LocalTensor<GK_T> gateTyped = GateGTyped(qwSlot);
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> qFp32 = arena;
            LocalTensor<float> kFp32 = arena[elems];
            LocalTensor<float> gFp32 = arena[2 * elems];
            LocalTensor<float> expFp32 = arena[3 * elems];
            LocalTensor<float> outFp32 = arena[4 * elems];

            WaitGateInputReady();
            Cast(qFp32, qTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            if constexpr (IsSameType<GK_T, float>::value) {
                gFp32 = gateTyped;
            } else {
                Cast(gFp32, gateTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            }
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

            if (qwOutputPending) {
                WaitGateOutputForVector();
            }
            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG, tileRow),
                          qScore, elems);
            CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W, tileRow),
                          kScore, elems);
            SignalGateOutputDone();
            qwOutputPending = true;
            qwSlot ^= 1;
        }
        if (qwOutputPending) {
            WaitGateOutputForMte2();
            WaitGateOutputForVector();
        }

        uint64_t kgBegin = (validColEnd * subBlockIdx) / subBlockNum;
        uint64_t kgEnd = (validColEnd * (subBlockIdx + 1)) / subBlockNum;
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
            LocalTensor<GK_T> gateTyped = GateGTyped(kgSlot);
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> kFp32 = arena;
            LocalTensor<float> gFp32 = arena[elems];
            LocalTensor<float> expFp32 = arena[2 * elems];
            LocalTensor<float> outFp32 = arena[3 * elems];

            WaitGateInputReady();
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            if constexpr (IsSameType<GK_T, float>::value) {
                gFp32 = gateTyped;
            } else {
                Cast(gFp32, gateTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            }
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
            LoadAsFloatRow(gk_, KVOffset(b, hv, refToken, 0, K_), refFp32, K_);
        }

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
            LocalTensor<GK_T> gateTyped = GateGTyped(gateSlot);
            LocalTensor<float> arena = vecBuf_.Get<float>();
            LocalTensor<float> qFp32 = arena;
            LocalTensor<float> kFp32 = arena[elems];
            LocalTensor<float> gFp32 = arena[2 * elems];
            LocalTensor<float> expFp32 = arena[3 * elems];
            LocalTensor<float> outFp32 = arena[4 * elems];

            uint64_t token = start + tileRow;
            WaitGateInputReady();
            Cast(qFp32, qTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            Cast(kFp32, kTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            if constexpr (IsSameType<GK_T, float>::value) {
                gFp32 = gateTyped;
            } else {
                Cast(gFp32, gateTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(elems));
            }
            uint64_t nextTileRow = tileRow + maxRows;
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

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            if (writeScoreScratch) {
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_QG, tileRow),
                              qScore, elems);
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_W, tileRow),
                              wScore, elems);
                CopyVectorOut(scoreWorkspace_, ScoreScratchOffset(scoreSlot, KDA_SCORE_SCRATCH_KG, tileRow),
                              kgScore, elems);
            } else {
                CopyVectorOut(qg_, KVOffset(b, hv, token, 0, K_), qTyped, elems);
                CopyVectorOut(w_, KVOffset(b, hv, token, 0, K_), kTyped, elems);
                CopyVectorOut(kg_, KVOffset(b, hv, token, 0, K_), kgTyped, elems);
            }
            SignalGateOutputDone();
            outputPending = true;
            gateSlot ^= 1;
        }
        if (outputPending) {
            WaitGateOutputForMte2();
            WaitGateOutputForVector();
        }
        return;
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
        if (writeScoreScratch) {
            PrepareScoreFactorsBulk(b, h, hv, start, subBlockIdx, subBlockNum, refToken, scoreRowBegin,
                                    scoreRowCount, validColEnd, scoreSlot);
            return;
        }
        PrepareGateProductsBulk(b, h, hv, start, curT, subBlockIdx, subBlockNum, useRef, refToken,
                                validColEnd, writeScoreScratch, scoreSlot);
    }

    __aicore__ inline void ComputeRawAqkAkkCube(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT)
    {
        ComputeRawAqkAkkCubeBlock(b, hv, start, curT, 0, curT);
    }

    __aicore__ inline void ComputeRawAqkAkkCubeBlock(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT,
                                                     uint64_t rowBegin, uint64_t rowCount,
                                                     bool readScoreScratch = false, uint64_t scoreSlot = 0,
                                                     uint64_t colCount = 0)
    {
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
        if (colCount == 0 || colCount > curT) {
            colCount = curT;
        }
        Catlass::GemmCoord shape{static_cast<uint32_t>(rowCount), static_cast<uint32_t>(colCount),
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
        auto tensorAqk = tla::MakeTensor(aqk_[AOffset(b, hv, start, 0)], layoutC,
                                         Catlass::Arch::PositionGM{});
        auto tensorAkk = tla::MakeTensor(akk_[AOffset(b, hv, start, 0)], layoutC,
                                         Catlass::Arch::PositionGM{});

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

        if (validRowCount < rowCount) {
            FillLocalFloat(aqkMat, 0.0f, elemCount);
            FillLocalFloat(akkMat, 0.0f, elemCount);
            FillLocalFloat(betaLocal, 0.0f, rowCount);
        }
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);
        if (validRowCount > 0) {
            LoadAsFloatRow(beta_, BetaOffset(b, hv, token), betaLocal, validRowCount);
            DataCopy(aqkMat, aqk_[AOffset(b, hv, token, 0)], static_cast<uint32_t>(validElemCount));
            DataCopy(akkMat, akk_[AOffset(b, hv, token, 0)], static_cast<uint32_t>(validElemCount));
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        }
        Brcb(betaBrcb, betaLocal, static_cast<uint8_t>((rowCount + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();
        uint8_t rowStride = static_cast<uint8_t>(BT_ * sizeof(float) / 32);
        for (uint64_t col = 0; col < BT_; col += 8) {
            Mul(akkMat[col], akkMat[col], betaBrcb, 8, static_cast<uint8_t>(rowCount),
                {1, 1, 0, rowStride, rowStride, 1});
        }
        PipeBarrier<PIPE_V>();
        if (validRowCount > 0) {
            SelectCausalRows(aqkMat, akkMat, rowBegin, validRowCount);
        }

        Muls(xMat, akkMat, -1.0f, static_cast<uint32_t>(elemCount));
        PipeBarrier<PIPE_V>();
        if constexpr (SAFE_GATE) {
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
        PipeBarrier<PIPE_ALL>();
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

    __aicore__ inline void ComputeAkkInverseMchTail(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                    uint64_t start, uint64_t curT)
    {
        uint64_t xBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_X);
        uint64_t lBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0);
        uint64_t yBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y1);
        uint64_t yNextBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_Y0);
        uint64_t tmpBase = SolveScratchOffset(b, hv, chunkIdx, KDA_SOLVE_SCRATCH_TMP);
        (void)start;
        (void)curT;

        uint32_t diagBlocks = static_cast<uint32_t>(BT_ / KDA_SOLVE_DIAG_BT);
        for (uint32_t block = 0; block < diagBlocks; ++block) {
            uint32_t off = block * KDA_SOLVE_DIAG_BT;
            CubeGemmSolveSub(solveWorkspace_, lBase, off, off, solveWorkspace_, lBase, off, off,
                             solveWorkspace_, yBase, off, off,
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
        ComputeAkkMergeCubeWorkspace(b, hv, chunkIdx);
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
        LoadAsFloatRow(beta_, BetaOffset(b, hv, start + rowBegin), betaLocal, rowCount);
        Brcb(betaBrcb, betaLocal, static_cast<uint8_t>((rowCount + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();
        ScaleRowsByBeta(w_, w_, b, hv, start, rowBegin, rowCount, K_, betaLocal, betaBrcb, matrixLocal);
        ScaleRowsByBeta(v_, vNew_, b, hv, start, rowBegin, rowCount, V_, betaLocal, betaBrcb,
                        matrixLocal, inputSequenceMajor_);
    }

    __aicore__ inline void FinalizePrepareIntermediates(uint64_t b, uint64_t hv, uint64_t start,
                                                        uint64_t curT, uint64_t subBlockIdx,
                                                        uint64_t subBlockNum)
    {
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

            CopyVectorIn(aqkLocal, aqk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
            CopyVectorIn(akkLocal, akk_, AOffset(b, hv, start + tileRow, 0), matrixElems);
            CopyVectorIn(qgTyped, qg_, KVOffset(b, hv, start + tileRow, 0, K_), qgElems);
            SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);

            Muls(aqkLocal, aqkLocal, scale_, static_cast<uint32_t>(matrixElems));
            Cast(qgLocal, qgTyped, RoundMode::CAST_NONE, static_cast<uint32_t>(qgElems));
            PipeBarrier<PIPE_V>();
            Muls(qgLocal, qgLocal, scale_, static_cast<uint32_t>(qgElems));
            PipeBarrier<PIPE_V>();
            ClampFp32ToOutputType(aqkLocal, static_cast<uint32_t>(matrixElems));
            ClampFp32ToOutputType(akkLocal, static_cast<uint32_t>(matrixElems));
            ClampFp32ToOutputType(qgLocal, static_cast<uint32_t>(qgElems));
            Cast(aqkTyped, aqkLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(matrixElems));
            Cast(akkTyped, akkLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(matrixElems));
            Cast(qgTyped, qgLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(qgElems));
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(vToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(vToMte3Event_);
            CopyVectorOut(o_, AOffset(b, hv, start + tileRow, 0), aqkTyped, matrixElems);
            CopyVectorOut(u_, AOffset(b, hv, start + tileRow, 0), akkTyped, matrixElems);
            CopyVectorOut(kg_, KVOffset(b, hv, start + tileRow, 0, K_), qgTyped, qgElems);
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
        bool usePostWuCube = UsePostWuCube(curT);
        bool useAkkCubeSolve = UseAkkCubeSolve(curT);
        uint64_t solveRowBegin = 0;
        uint64_t solveRowEnd = 0;
        GetSolveRowRange(BT_, subBlockIdx, subBlockNum, solveRowBegin, solveRowEnd);
        uint64_t scoreBlockSize = ScoreRefBlockSize();
        uint64_t scoreBlockCount = (curT + scoreBlockSize - 1) / scoreBlockSize;
        uint64_t pipelineBlockCount =
            (scoreBlockCount + KDA_SCORE_QUEUE_DEPTH - 1) / KDA_SCORE_QUEUE_DEPTH * KDA_SCORE_QUEUE_DEPTH;
        for (uint64_t block = 0; block < pipelineBlockCount; ++block) {
            if (block < scoreBlockCount) {
                uint64_t rowBegin = block * scoreBlockSize;
                uint64_t rowCount = ScoreRowBlockCount(curT, rowBegin);
                uint64_t refToken = ScoreRefToken(start, curT, rowBegin, rowCount);
                uint64_t scoreSlot =
                    ScoreScratchSlot(block % KDA_SCORE_QUEUE_DEPTH, scoreLane, pairHeads);
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
            }
        }
        bool fusedScoreWriteback = false;
        if (!fusedScoreWriteback) {
            // The final score MMAD only consumes scoreWorkspace_. Run the
            // independent gate writeback while AIC drains its MMAD/Fixpipe path.
            PrepareGateProducts(b, h, hv, start, curT, subBlockIdx, subBlockNum);
        }
        if (pipelineBlockCount > 0) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(scoreDoneFlag_);
        }
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
                    for (uint64_t rowPart = 0; rowPart < KDA_SCORE_LANES; ++rowPart) {
                        uint64_t pairRowBegin = 0;
                        uint64_t pairRowEnd = 0;
                        GetSolveRowRange(
                            BT_, rowPart, KDA_SCORE_LANES, pairRowBegin, pairRowEnd);
                        PrepareAqkAkkSolveInputRows(
                            b, hv, chunkIdx, start, curT, pairRowBegin, pairRowEnd, false, false);
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
                StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
            } else {
                PrepareAqkAkkSolveInputRows(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd,
                                            fullChunk, !fullChunk);
                uint32_t solveIters = KDA_SOLVE_DIAG_MCH_ITERS;
                RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
                for (uint32_t iter = 0; iter < solveIters; ++iter) {
                    AddSolveTmpToXDiagRows(b, hv, chunkIdx, start, solveRowBegin, solveRowEnd,
                                           fullChunk && iter + 1 == solveIters);
                    RunAicAfterBothAivReady(subBlockIdx, subBlockNum);
                }
                if (!fullChunk) {
                    StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
                }
            }
        }
        // Host validation guarantees every accepted shape has enough workspace for this cube path.
        PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
        FinalizePrepareIntermediates(b, hv, start, curT, subBlockIdx, subBlockNum);
    }

    __aicore__ inline void FinishDeferredSafeChunk(uint64_t b, uint64_t hv, uint64_t chunkIdx,
                                                   uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                                   uint64_t subBlockNum)
    {
        uint64_t curT = end - start;
        uint64_t solveRowBegin = 0;
        uint64_t solveRowEnd = 0;
        GetSolveRowRange(BT_, subBlockIdx, subBlockNum, solveRowBegin, solveRowEnd);
        StoreSolveXRowsToAkk(b, hv, chunkIdx, start, curT, solveRowBegin, solveRowEnd);
        PrepareWuCubeInputs(b, hv, start, curT, subBlockIdx, subBlockNum);
        FinalizePrepareIntermediates(b, hv, start, curT, subBlockIdx, subBlockNum);
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
                ComputeRawAqkAkkCubeBlock(b, hv, start, curT, rowBegin, rowCount, true,
                                          block % KDA_SCORE_QUEUE_DEPTH, rowBegin + rowCount);
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
        }
        bool usePostWuCube = UsePostWuCube(curT);
        bool useAkkCubeSolve = UseAkkCubeSolve(curT);
        if (useAkkCubeSolve) {
            if constexpr (SAFE_GATE) {
                Catlass::Arch::CrossCoreWaitFlag(syncReadyFlag_);
                ComputeAkkMergeCubeWorkspace(b, hv, chunkIdx);
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(syncDoneFlag_);
            } else {
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(mchSyncReadyFlag_);
                if (curT == BT_) {
                    ComputeAkkInverseMchFull(b, hv, chunkIdx, start);
                } else {
                    ComputeAkkInverseMchTail(b, hv, chunkIdx, start, curT);
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
        uint64_t scoreBlockSize = ScoreRefBlockSize();
        uint64_t scoreBlockCount = (curT + scoreBlockSize - 1) / scoreBlockSize;
        uint64_t pipelineBlockCount =
            (scoreBlockCount + KDA_SCORE_QUEUE_DEPTH - 1) / KDA_SCORE_QUEUE_DEPTH * KDA_SCORE_QUEUE_DEPTH;
        for (uint64_t block = 0; block < pipelineBlockCount; ++block) {
            Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_FIX>(scoreReadyFlag_);
            if (block < scoreBlockCount) {
                uint64_t rowBegin = block * scoreBlockSize;
                uint64_t rowCount = ScoreRowBlockCount(curT, rowBegin);
                for (uint64_t lane = 0; lane < KDA_SCORE_LANES; ++lane) {
                    uint64_t hv = hvBase + lane;
                    uint64_t scoreSlot =
                        ScoreScratchSlot(block % KDA_SCORE_QUEUE_DEPTH, lane, true);
                    ComputeRawAqkAkkCubeBlock(b, hv, start, curT, rowBegin, rowCount, true,
                                              scoreSlot, rowBegin + rowCount);
                }
            }
            Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(scoreDoneFlag_);
        }

        if (UseAkkCubeSolve(curT)) {
            Catlass::Arch::CrossCoreWaitFlag(syncReadyFlag_);
            for (uint64_t lane = 0; lane < KDA_SCORE_LANES; ++lane) {
                activeSolveSlot_ =
                    (localTaskIdx % (KDA_SOLVE_PIPELINE_DEPTH / KDA_SCORE_LANES)) * KDA_SCORE_LANES + lane;
                ComputeAkkMergeCubeWorkspace(b, hvBase + lane, chunkIdx);
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

    __aicore__ inline void ProcessPreAiv()
    {
        if constexpr (IsSameType<T, float>::value) {
            isAivOnly_ = true;
        }
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
    bool inputSequenceMajor_ = false;
    bool isAivOnly_ = false;
    uint64_t usedCoreNum_ = 1;
    uint64_t solveCoreIdx_ = 0;
    uint64_t activeSolveSlot_ = 0;
    __gm__ int64_t *chunkIndicesAddr_ = nullptr;
    __gm__ int64_t *cuSeqlensAddr_ = nullptr;
};
} // namespace

template <bool SAFE_GATE, typename T, typename GK_T, typename BETA_T, typename TilingData>
__aicore__ inline void RunChunkKdaPrepare(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR beta, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR akk, GM_ADDR qg,
    GM_ADDR qgScaled, GM_ADDR wSeed, GM_ADDR uSeed, GM_ADDR userWorkspace,
    const TilingData &tiling, TPipe &pipe)
{
    GM_ADDR aqkFp32 = userWorkspace + tiling.prepareAqkFp32Offset;
    GM_ADDR akkFp32 = userWorkspace + tiling.prepareAkkFp32Offset;
    GM_ADDR prepareScratch = userWorkspace + tiling.prepareScratchOffset;

    if ASCEND_IS_AIC {
        ChunkKdaFwdPrepareKernel<SAFE_GATE, T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                nullptr, nullptr, nullptr, nullptr, aqk, userWorkspace, aqkFp32, akkFp32,
                wSeed, akk, qg, qgScaled, uSeed, userWorkspace, prepareScratch, tiling, &pipe, false);
        op.ProcessAic();
    }
    if ASCEND_IS_AIV {
        ChunkKdaFwdPrepareKernel<SAFE_GATE, T, GK_T, BETA_T> op;
        op.Init(q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices,
                nullptr, nullptr, nullptr, nullptr, aqk, userWorkspace, aqkFp32, akkFp32,
                wSeed, akk, qg, qgScaled, uSeed, userWorkspace, prepareScratch, tiling, &pipe);
        op.ProcessAiv();
    }
}

template <bool SAFE_GATE, typename T, typename GK_T, typename BETA_T,
          typename TilingData, uint32_t COMPILE_BT, uint32_t COMPILE_K,
          uint32_t COMPILE_V>
__aicore__ inline void RunChunkKdaPrepare(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR, GM_ADDR,
    GM_ADDR, GM_ADDR beta, GM_ADDR initialState, GM_ADDR cuSeqlens,
    GM_ADDR chunkIndices, GM_ADDR aqk, GM_ADDR akk, GM_ADDR qg,
    GM_ADDR qgScaled, GM_ADDR wSeed, GM_ADDR uSeed, GM_ADDR,
    GM_ADDR userWorkspace, const TilingData &tiling, TPipe &pipe,
    bool = true)
{
    RunChunkKdaPrepare<SAFE_GATE, T, GK_T, BETA_T>(
        q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices, aqk,
        akk, qg, qgScaled, wSeed, uSeed, userWorkspace, tiling, pipe);
}

} // namespace KdaPrepare
