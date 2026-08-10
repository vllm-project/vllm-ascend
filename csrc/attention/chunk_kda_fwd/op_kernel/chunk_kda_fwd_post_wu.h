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
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm_coord.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "chunk_kda_fwd_varlen.h"
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
constexpr uint32_t KDA_POST_EVENT = 3;
constexpr uint32_t KDA_POST_EVENT_NEXT = 4;
constexpr uint32_t KDA_POST_EVENT_FIX = 5;

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


    __aicore__ inline void ComputePostWuCube(uint64_t b, uint64_t hv, uint64_t chunkIdx, uint64_t start,
                                             uint64_t curT)
    {
        using ElementA = AKK_T;
        using ElementB = T;
        using LayoutTagA = Catlass::layout::RowMajor;
        using LayoutTagB = Catlass::layout::RowMajor;
        using LayoutTagC = Catlass::layout::RowMajor;
        using WTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, float, LayoutTagC>;
        using UTileCopy = Catlass::Gemm::Tile::PackedTileCopyTla<KdaArchTag, ElementA, LayoutTagA, ElementB,
                                                                 LayoutTagB, OUT_T, LayoutTagC>;
        using PostL1TileShape128 = tla::Shape<KdaInt128, KdaInt128, tla::_256>;
        using PostL0TileShape128 = tla::Shape<KdaInt128, KdaInt128, KdaInt128>;
        using PostL1TileShape256 = tla::Shape<KdaInt128, tla::_256, tla::_256>;
        using PostL0TileShape256 = tla::Shape<KdaInt128, tla::_256, KdaInt64>;
        using WBlockMmad = Catlass::Gemm::Block::BlockMmadTla<KdaDispatchPolicy, PostL1TileShape128,
                                                               PostL0TileShape128,
                                                               ElementA, ElementB, float, void, WTileCopy>;
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
            LayoutTagC tagC = LayoutTagC::template MakeLayout<float>(BT_, K_);
            auto layoutB = tla::MakeLayoutFromTag(tagB);
            auto layoutC = tla::MakeLayoutFromTag(tagC);
            Catlass::GemmCoord shape{static_cast<uint32_t>(curT), static_cast<uint32_t>(K_),
                                     static_cast<uint32_t>(curT)};
            auto tensorB = tla::MakeTensor(preparedQG_[KVOffset(b, hv, start, 0, K_)], layoutB,
                                           Catlass::Arch::PositionGM{});
            auto tensorC = tla::MakeTensor(h_[WScratchOffset(b, hv, chunkIdx, 0, 0)], layoutC,
                                            Catlass::Arch::PositionGM{});
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

    __aicore__ inline void CopyScratchWAndFinalizeKg(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                                     uint64_t start, uint64_t curT, uint64_t subBlockIdx,
                                                     uint64_t subBlockNum, bool copyScratchW)
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
            uint64_t token = start + tileRow;

            if (copyScratchW) {
                uint64_t scratchBase = WScratchOffset(b, hv, chunkIdx, tileRow, 0);
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
            }

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

    template <typename SrcTensor, typename DstTensor>
    __aicore__ inline void ComputeTailWuRow(GlobalTensor<SrcTensor> &src, GlobalTensor<DstTensor> &dst,
                                            uint64_t akkBase, uint64_t srcBase, uint64_t dstBase, uint64_t curT,
                                            uint64_t dim)
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
            LoadAsFloatVector(src, srcBase + j * dim, value, typed, dim);
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

    __aicore__ inline void ComputeTailWuVector(uint64_t b, uint64_t hv, uint64_t start, uint64_t curT,
                                               uint64_t subBlockIdx, uint64_t subBlockNum)
    {
        uint64_t rowBegin = (curT * subBlockIdx) / subBlockNum;
        uint64_t rowEnd = (curT * (subBlockIdx + 1)) / subBlockNum;
        for (uint64_t row = rowBegin; row < rowEnd; ++row) {
            ComputeTailWuRow(
                preparedQG_, w_, AOffset(b, hv, start + row, 0), KVOffset(b, hv, start, 0, K_),
                KVOffset(b, hv, start + row, 0, K_), curT, K_);
            ComputeTailWuRow(
                propagatedVNew_, u_, AOffset(b, hv, start + row, 0), KVOffset(b, hv, start, 0, V_),
                KVOffset(b, hv, start + row, 0, V_), curT, V_);
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

    __aicore__ inline void ProcessChunkPostAiv(uint64_t b, uint64_t h, uint64_t hv, uint64_t chunkIdx,
                                               uint64_t start, uint64_t end, uint64_t subBlockIdx,
                                               uint64_t subBlockNum)
    {
        uint64_t curT = end - start;
        if (curT == 0 || !UsePostWuCube(curT)) {
            return;
        }
        if (curT < BT_) {
            ComputeTailWuVector(b, hv, start, curT, subBlockIdx, subBlockNum);
            CopyScratchWAndFinalizeKg(
                b, h, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum, false);
            return;
        }
        Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE2>(syncDoneFlag_);
        CopyScratchWAndFinalizeKg(
            b, h, hv, chunkIdx, start, curT, subBlockIdx, subBlockNum, true);
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
        if (curT < BT_) {
            return;
        }
        ComputePostWuCube(b, hv, chunkIdx, start, curT);
        Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(syncDoneFlag_);
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
