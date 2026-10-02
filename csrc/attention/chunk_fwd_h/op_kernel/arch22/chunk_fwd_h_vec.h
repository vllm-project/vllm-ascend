/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH22_CHUNK_FWD_H_VEC_H
#define ARCH22_CHUNK_FWD_H_VEC_H

#include <type_traits>

#include "../chunk_fwd_h_policy.h"
#include "../chunk_fwd_h_utils.h"

namespace GDN {

constexpr float FWD_H_ARCH22_LN2 = 0.69314718055994530942f;

template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
class ChunkFwdHVecArch22 {
public:
    __aicore__ inline void Init(const FwdHKernelArgs &args)
    {
        args_ = args;
        coreIdx_ = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        coreNum_ = AscendC::GetBlockNum();
        aiv_ = AscendC::GetSubBlockIdx();
        InitBuffers();
        InitEvents();
    }

    __aicore__ inline void InitEvents()
    {
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            ioFreeEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
            ioReadyEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE2_V>();
            ioDoneEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_MTE3>();
            scalarReadEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::V_S>();
            scalarWriteEvent_[slot] = GetTPipePtr()->AllocEventID<AscendC::HardEvent::S_V>();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        }
    }

    __aicore__ inline AscendC::TEventID IoFreeEvent(uint32_t slot) const
    {
        return ioFreeEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID IoReadyEvent(uint32_t slot) const
    {
        return ioReadyEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID IoDoneEvent(uint32_t slot) const
    {
        return ioDoneEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID ScalarReadEvent(uint32_t slot) const
    {
        return scalarReadEvent_[slot];
    }

    __aicore__ inline AscendC::TEventID ScalarWriteEvent(uint32_t slot) const
    {
        return scalarWriteEvent_[slot];
    }

    __aicore__ inline void Process()
    {
        const FwdHCoreHeadRange range =
            FwdHResolveCoreHeadRange(args_.tiling, coreIdx_, coreNum_);
        const uint32_t headsPerSequence = static_cast<uint32_t>(args_.tiling.vNumHead);
        uint32_t cachedSequence = FwdHSequenceCount(args_.tiling);
        FwdHSequenceSpan sequenceSpan{};
        for (uint32_t cursor = range.begin; cursor < range.end;) {
            const uint32_t sequence = cursor / headsPerSequence;
            const uint32_t hvBegin = cursor - sequence * headsPerSequence;
            uint32_t unitHeads = range.end - cursor;
            const uint32_t sequenceRemain = headsPerSequence - hvBegin;
            if (unitHeads > sequenceRemain) {
                unitHeads = sequenceRemain;
            }
            if (unitHeads > FWD_H_AIC_HEAD_SLOTS) {
                unitHeads = FWD_H_AIC_HEAD_SLOTS;
            }
            if (sequence != cachedSequence) {
                sequenceSpan = FwdHResolveSequence(args_, sequence);
                cachedSequence = sequence;
            }
            const FwdHWorkUnit unit{
                sequenceSpan,
                FwdHBuildHeadRange<CompilePolicy::GATE_MODE>(args_.tiling, hvBegin, unitHeads)};
            cursor += unitHeads;
            if (unit.sequence.chunkCount != 0 && unit.headRound.activeHeadCount != 0) {
                ProcessWorkUnit(unit, cursor < range.end);
            }
        }
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        }
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::V_S>(ScalarReadEvent(slot));
            GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::S_V>(ScalarWriteEvent(slot));
        }
    }

private:
    using PType = std::conditional_t<CompilePolicy::STATE_FP32, float, bfloat16_t>;
    static constexpr uint32_t TILE_ROWS = 8;
    static constexpr uint32_t TILE_ELEMENTS = TILE_ROWS * FWD_H_V;
    static constexpr uint32_t LOCAL_SLOT_BYTES = 32 * 1024;
    static constexpr uint32_t LOCAL0_BASE = 0;
    static constexpr uint32_t LOCAL1_BASE = LOCAL_SLOT_BYTES;
    static constexpr uint32_t STATE0_BASE = 64 * 1024;
    static constexpr uint32_t STATE1_BASE = 96 * 1024;
    static constexpr uint32_t RAW_BASE = 0;
    static constexpr uint32_t CAST_BASE = 4 * 1024;
    static constexpr uint32_t CALC_BASE = 8 * 1024;
    static constexpr uint32_t IO_BASE = 12 * 1024;
    static constexpr uint32_t RIGHT_BASE = 14 * 1024;
    static constexpr uint32_t GATE_RAW_BASE = 16 * 1024;
    static constexpr uint32_t GATE_FP32_BASE = 17 * 1024;
    static constexpr uint32_t ALPHA_BASE = 18 * 1024;
    static constexpr uint32_t SHARE_BASE = 19 * 1024;

    __aicore__ inline uint32_t LocalBase(uint32_t slot) const
    {
        return slot == 0 ? LOCAL0_BASE : LOCAL1_BASE;
    }

    __aicore__ inline uint32_t TileRows(uint32_t rows) const
    {
        return rows < TILE_ROWS ? rows : TILE_ROWS;
    }

    __aicore__ inline void InitBuffers()
    {
        GetTPipePtr()->InitBuffer(ubBuf_, 192 * 1024);
        AscendC::LocalTensor<uint8_t> ub = ubBuf_.Get<uint8_t>();
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            const uint32_t base = LocalBase(slot);
            raw_[slot] = ub[base + RAW_BASE].template ReinterpretCast<uint8_t>();
            cast_[slot] = ub[base + CAST_BASE].template ReinterpretCast<float>();
            calc_[slot] = ub[base + CALC_BASE].template ReinterpretCast<float>();
            io_[slot] = ub[base + IO_BASE].template ReinterpretCast<bfloat16_t>();
            right_[slot] = ub[base + RIGHT_BASE].template ReinterpretCast<bfloat16_t>();
            gateRaw_[slot] = ub[base + GATE_RAW_BASE].template ReinterpretCast<GateT>();
            gateFp32_[slot] = ub[base + GATE_FP32_BASE].template ReinterpretCast<float>();
            alpha_[slot] = ub[base + ALPHA_BASE].template ReinterpretCast<float>();
            share_[slot] = ub[base + SHARE_BASE].template ReinterpretCast<uint8_t>();
        }
        stateBf16_[0] = ub[STATE0_BASE].template ReinterpretCast<bfloat16_t>();
        stateBf16_[1] = ub[STATE1_BASE].template ReinterpretCast<bfloat16_t>();
    }

    __aicore__ inline uint64_t UOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head, uint32_t row) const
    {
        return FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                               chunk.tokenBegin + row, FWD_H_V);
    }

    __aicore__ inline uint64_t GateOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head) const
    {
        const uint32_t dim = CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G ? 1 : FWD_H_K;
        return FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                               chunk.tokenBegin, dim);
    }

    __aicore__ inline uint64_t HOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head) const
    {
        return FwdHHOffset(args_.tiling, unit.sequence, head.hv, chunk.globalChunk);
    }

    __aicore__ inline uint64_t ScratchByteOffset(int64_t base, uint32_t slot,
                                                  uint32_t slotBytes) const
    {
        return static_cast<uint64_t>(base) +
            (static_cast<uint64_t>(coreIdx_) * FWD_H_AIC_HEAD_SLOTS + slot) * slotBytes;
    }

    __aicore__ inline float ReadScalar(const AscendC::LocalTensor<float> &tensor,
                                       uint32_t index, uint32_t slot) const
    {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(ScalarReadEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(ScalarReadEvent(slot));
        const float value = tensor.GetValue(index);
        AscendC::SetFlag<AscendC::HardEvent::S_V>(ScalarWriteEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(ScalarWriteEvent(slot));
        return value;
    }

    __aicore__ inline void PrepareScalarGate(uint32_t slot, uint32_t validTokens)
    {
        if constexpr (std::is_same<GateT, float>::value) {
            AscendC::Adds(gateFp32_[slot], gateRaw_[slot], 0.0f, validTokens);
        } else {
            AscendC::Cast(gateFp32_[slot], gateRaw_[slot], AscendC::RoundMode::CAST_NONE,
                          validTokens);
        }
        AscendC::PipeBarrier<PIPE_V>();
        const float last = ReadScalar(gateFp32_[slot], validTokens - 1, slot);
        AscendC::Duplicate(cast_[slot], last, validTokens);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(gateFp32_[slot], cast_[slot], gateFp32_[slot], validTokens);
        AscendC::Duplicate(alpha_[slot], last, 1);
        if constexpr (CompilePolicy::USE_EXP2) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(gateFp32_[slot], gateFp32_[slot], FWD_H_ARCH22_LN2, validTokens);
            AscendC::Muls(alpha_[slot], alpha_[slot], FWD_H_ARCH22_LN2, 1);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gateFp32_[slot], gateFp32_[slot], validTokens);
        AscendC::Exp(alpha_[slot], alpha_[slot], 1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepareKeyGate(uint32_t slot)
    {
        if constexpr (std::is_same<GateT, float>::value) {
            AscendC::Adds(gateFp32_[slot], gateRaw_[slot], 0.0f, FWD_H_K);
        } else {
            AscendC::Cast(gateFp32_[slot], gateRaw_[slot], AscendC::RoundMode::CAST_NONE,
                          FWD_H_K);
        }
        if constexpr (CompilePolicy::USE_EXP2) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(gateFp32_[slot], gateFp32_[slot], FWD_H_ARCH22_LN2, FWD_H_K);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(gateFp32_[slot], gateFp32_[slot], FWD_H_K);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RunSMinusOne(const FwdHWorkUnit &unit)
    {
        // S-1：StateT=FP32 且存在 initial_state 时，逐 tile 执行 H0=cast_BF16(R0)，
        // GM 物理顺序原生遵循 state_v_first，不依赖外部转置。
        if constexpr (!CompilePolicy::STATE_FP32) {
            return;
        }
        if (args_.tiling.useInitialState == 0) {
            return;
        }
        AscendC::GlobalTensor<float> initial;
        AscendC::GlobalTensor<bfloat16_t> h;
        initial.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.initialState));
        h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
        const uint32_t pairCount = FwdHMode2PairCount(unit.headRound.activeHeadCount);
        for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
            if (FwdHMode2PairHasHead(unit.headRound.activeHeadCount, pairSlot, aiv_)) {
                const FwdHHeadBinding &head = unit.headRound.heads[pairSlot * 2U + aiv_];
                const uint32_t slot = head.localSlot;
                const uint64_t stateBase = FwdHStateOffset<STATE_V_FIRST>(
                    args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
                const FwdHChunkSpan first = FwdHBuildChunk(unit.sequence, 0);
                for (uint32_t row = 0; row < FWD_H_K; row += TILE_ROWS) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
                    AscendC::DataCopy(calc_[slot], initial[stateBase + row * FWD_H_V], TILE_ELEMENTS);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
                    AscendC::Cast(io_[slot], calc_[slot], AscendC::RoundMode::CAST_RINT, TILE_ELEMENTS);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
                    AscendC::DataCopy(h[HOffset(unit, first, head) + row * FWD_H_V],
                                      io_[slot], TILE_ELEMENTS);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
                }
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                FwdHAivLocalFlag(FWD_H_H_READY_FLAG, pairSlot));
        }
    }

    template <bool HAS_P>
    __aicore__ inline void PrefetchStage1Head(const FwdHWorkUnit &unit,
                                              const FwdHChunkSpan &chunk,
                                              const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        AscendC::GlobalTensor<bfloat16_t> u;
        u.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.u));
        const uint32_t firstRows = TileRows(chunk.validTokens);
        AscendC::DataCopy(io_[slot], u[UOffset(unit, chunk, head, 0)], firstRows * FWD_H_V);
        if constexpr (HAS_P) {
            const uint64_t pByteOffset = ScratchByteOffset(args_.tiling.vWorkspaceOffset,
                                                           head.roundHead,
                                                           FWD_H_TOKEN_MATRIX_FP32_BYTES);
            AscendC::GlobalTensor<PType> p;
            p.SetGlobalBuffer(reinterpret_cast<__gm__ PType *>(args_.workspace + pByteOffset));
            AscendC::LocalTensor<PType> rawP = raw_[slot].template ReinterpretCast<PType>();
            AscendC::DataCopy(rawP, p, firstRows * FWD_H_V);
        }
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            if (args_.tiling.storeFinalState != 0 || !chunk.last) {
                AscendC::GlobalTensor<GateT> gate;
                gate.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(args_.g));
                AscendC::DataCopyPadExtParams<GateT> pad{false, 0, 0, 0};
                const uint32_t gateBytes = static_cast<uint32_t>(chunk.validTokens * sizeof(GateT));
                AscendC::DataCopyExtParams copy{1, gateBytes, 0, 0, 0};
                AscendC::DataCopyPad(gateRaw_[slot], gate[GateOffset(unit, chunk, head)], copy, pad);
            }
        }
        if (chunk.first && !CompilePolicy::STATE_FP32 && args_.tiling.useInitialState != 0) {
            AscendC::GlobalTensor<bfloat16_t> initial;
            initial.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.initialState));
            const uint64_t offset = FwdHStateOffset<STATE_V_FIRST>(
                args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
            AscendC::DataCopy(stateBf16_[slot], initial[offset], FWD_H_K * FWD_H_V);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
    }

    template <bool HAS_P>
    __aicore__ inline void LoadStage1Tile(const FwdHWorkUnit &unit,
                                          const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head,
                                          uint32_t row)
    {
        const uint32_t slot = head.localSlot;
        const uint32_t rows = TileRows(chunk.validTokens - row);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        AscendC::GlobalTensor<bfloat16_t> u;
        u.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.u));
        AscendC::DataCopy(io_[slot], u[UOffset(unit, chunk, head, row)], rows * FWD_H_V);
        if constexpr (HAS_P) {
            const uint64_t pByteOffset = ScratchByteOffset(args_.tiling.vWorkspaceOffset,
                                                           head.roundHead,
                                                           FWD_H_TOKEN_MATRIX_FP32_BYTES);
            AscendC::GlobalTensor<PType> p;
            p.SetGlobalBuffer(reinterpret_cast<__gm__ PType *>(args_.workspace + pByteOffset));
            AscendC::LocalTensor<PType> rawP = raw_[slot].template ReinterpretCast<PType>();
            AscendC::DataCopy(rawP, p[row * FWD_H_V], rows * FWD_H_V);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
    }

    template <bool HAS_P, bool WRITE_RIGHT, bool WAIT_READY>
    __aicore__ inline void ComputeStage1Tile(const FwdHWorkUnit &unit,
                                             const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head,
                                             uint32_t row)
    {
        const uint32_t slot = head.localSlot;
        const uint32_t rows = TileRows(chunk.validTokens - row);
        const uint32_t elements = rows * FWD_H_V;
        if constexpr (WAIT_READY) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
        }
        AscendC::Cast(calc_[slot], io_[slot], AscendC::RoundMode::CAST_NONE, elements);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (HAS_P) {
            if constexpr (std::is_same<PType, float>::value) {
                AscendC::Sub(calc_[slot], calc_[slot], raw_[slot].template ReinterpretCast<float>(), elements);
            } else {
                AscendC::Cast(cast_[slot], raw_[slot].template ReinterpretCast<bfloat16_t>(),
                              AscendC::RoundMode::CAST_NONE, elements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(calc_[slot], calc_[slot], cast_[slot], elements);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(io_[slot], calc_[slot], AscendC::RoundMode::CAST_RINT, elements);
        if constexpr (WRITE_RIGHT) {
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                uint32_t dstShape[2] = {rows, FWD_H_V};
                uint32_t srcShape[2] = {rows, 1};
                AscendC::Broadcast<float, 2, 1>(
                    raw_[slot].template ReinterpretCast<float>(), gateFp32_[slot][row],
                    dstShape, srcShape, share_[slot]);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(calc_[slot], calc_[slot],
                             raw_[slot].template ReinterpretCast<float>(), elements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(right_[slot], calc_[slot], AscendC::RoundMode::CAST_RINT, elements);
            }
        }

        AscendC::GlobalTensor<bfloat16_t> vNew;
        vNew.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.vNew));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        AscendC::DataCopy(vNew[UOffset(unit, chunk, head, row)], io_[slot], elements);
        if constexpr (WRITE_RIGHT) {
            const uint64_t rightOffset = args_.tiling.vUpdateWorkspaceOffset / sizeof(bfloat16_t) +
                FwdHCoreSlotOffset(coreIdx_, head.roundHead, FWD_H_CHUNK * FWD_H_V) +
                row * FWD_H_V;
            AscendC::GlobalTensor<bfloat16_t> rightGm;
            rightGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.workspace));
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                AscendC::DataCopy(rightGm[rightOffset], right_[slot], elements);
            } else {
                AscendC::DataCopy(rightGm[rightOffset], io_[slot], elements);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
    }

    template <bool HAS_P, bool WRITE_RIGHT>
    __aicore__ inline void ConsumeStage1Head(const FwdHWorkUnit &unit,
                                             const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head)
    {
        // Stage1：V_new=cast_BF16(fp32(U)-fp32(P))；g-only 同时生成
        // V_new_g=cast_BF16(E(g_last-g_i)*V_new_fp32)，gk-only 的 right 即 V_new。
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
        if constexpr (WRITE_RIGHT && CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            PrepareScalarGate(slot, chunk.validTokens);
        }
        if (chunk.first && args_.tiling.useInitialState == 0) {
            AscendC::Duplicate(stateBf16_[slot], static_cast<bfloat16_t>(0), FWD_H_K * FWD_H_V);
            AscendC::PipeBarrier<PIPE_V>();
        }
        ComputeStage1Tile<HAS_P, WRITE_RIGHT, false>(unit, chunk, head, 0);
        for (uint32_t row = TILE_ROWS; row < chunk.validTokens; row += TILE_ROWS) {
            LoadStage1Tile<HAS_P>(unit, chunk, head, row);
            ComputeStage1Tile<HAS_P, WRITE_RIGHT, true>(unit, chunk, head, row);
        }
        if (chunk.first && (!CompilePolicy::STATE_FP32 || args_.tiling.useInitialState == 0)) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            AscendC::GlobalTensor<bfloat16_t> h;
            h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
            AscendC::DataCopy(h[HOffset(unit, chunk, head)], stateBf16_[slot], FWD_H_K * FWD_H_V);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        }
    }

    __aicore__ inline void RunStage1(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // Stage1：V_new=cast_BF16(fp32(U)-fp32(P))；g-only 同时生成
        // V_new_g=cast_BF16(E(g_last-g_i)*V_new_fp32)，gk-only 的右矩阵就是 V_new。
        const bool hasP = !(chunk.first && args_.tiling.useInitialState == 0);
        const bool writeRight = args_.tiling.storeFinalState != 0 || !chunk.last;
        const uint32_t pairCount = FwdHMode2PairCount(unit.headRound.activeHeadCount);
        for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
            if (hasP && pairSlot > 0) {
                // A2/A3 的跨核 wait 会阻塞全部流水；先消费已 ready 的 ping，不能让它
                // 等待下一个 pair 的 P_READY。无 P 时保留 pong MTE2 与 ping VEC 的重叠。
                CompleteStage1Pair(unit, chunk, pairSlot - 1U, hasP, writeRight);
            }
            if (hasP) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                    FwdHAivLocalFlag(FWD_H_P_READY_FLAG, pairSlot));
            }
            if (FwdHMode2PairHasHead(unit.headRound.activeHeadCount, pairSlot, aiv_)) {
                const FwdHHeadBinding &head = unit.headRound.heads[pairSlot * 2U + aiv_];
                if (hasP) {
                    PrefetchStage1Head<true>(unit, chunk, head);
                } else {
                    PrefetchStage1Head<false>(unit, chunk, head);
                }
            }
            if (!hasP && pairSlot > 0) {
                CompleteStage1Pair(unit, chunk, pairSlot - 1U, hasP, writeRight);
            }
        }
        if (pairCount > 0) {
            CompleteStage1Pair(unit, chunk, pairCount - 1U, hasP, writeRight);
        }
    }

    __aicore__ inline void CompleteStage1Pair(const FwdHWorkUnit &unit,
                                              const FwdHChunkSpan &chunk,
                                              uint32_t pairSlot, bool hasP,
                                              bool writeRight)
    {
        if (FwdHMode2PairHasHead(unit.headRound.activeHeadCount, pairSlot, aiv_)) {
            const FwdHHeadBinding &head = unit.headRound.heads[pairSlot * 2U + aiv_];
            DispatchStage1(unit, chunk, head, hasP, writeRight);
        }
        if (hasP) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_V>(
                FwdHAivLocalFlag(FWD_H_P_FREE_FLAG, pairSlot));
        }
        if (writeRight) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                FwdHAivLocalFlag(FWD_H_RIGHT_READY_FLAG, pairSlot));
        }
    }

    __aicore__ inline void DispatchStage1(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head, bool hasP, bool writeRight)
    {
        if (hasP) {
            writeRight ? ConsumeStage1Head<true, true>(unit, chunk, head)
                       : ConsumeStage1Head<true, false>(unit, chunk, head);
        } else {
            writeRight ? ConsumeStage1Head<false, true>(unit, chunk, head)
                       : ConsumeStage1Head<false, false>(unit, chunk, head);
        }
    }

    __aicore__ inline void PrefetchStage3Head(const FwdHWorkUnit &unit,
                                              const FwdHChunkSpan &chunk,
                                              const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        const uint64_t dByteOffset = ScratchByteOffset(args_.tiling.hWorkspaceOffset,
                                                       head.roundHead,
                                                       FWD_H_STATE_FP32_BYTES);
        AscendC::GlobalTensor<float> d;
        d.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + dByteOffset));
        AscendC::DataCopy(raw_[slot].template ReinterpretCast<float>(), d, TILE_ELEMENTS);
        if constexpr (CompilePolicy::STATE_FP32) {
            AscendC::GlobalTensor<float> state;
            uint64_t offset = 0;
            if (chunk.first && args_.tiling.useInitialState != 0) {
                state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.initialState));
                offset = FwdHStateOffset<STATE_V_FIRST>(
                    args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
            } else {
                state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace));
                offset = args_.tiling.kDecayWorkspaceOffset / sizeof(float) +
                    FwdHCoreSlotOffset(coreIdx_, head.roundHead, FWD_H_K * FWD_H_V);
            }
            if (!(chunk.first && args_.tiling.useInitialState == 0)) {
                AscendC::DataCopy(calc_[slot], state[offset], TILE_ELEMENTS);
            }
        }
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
            AscendC::GlobalTensor<GateT> gk;
            gk.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(args_.gk));
            const uint64_t offset = GateOffset(unit, chunk, head) +
                static_cast<uint64_t>(chunk.validTokens - 1) * FWD_H_K;
            AscendC::DataCopy(gateRaw_[slot], gk[offset], FWD_H_K);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
    }

    __aicore__ inline void LoadStage3Tile(const FwdHWorkUnit &unit,
                                          const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head,
                                          uint32_t row)
    {
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        const uint64_t dByteOffset = ScratchByteOffset(args_.tiling.hWorkspaceOffset,
                                                       head.roundHead,
                                                       FWD_H_STATE_FP32_BYTES);
        AscendC::GlobalTensor<float> d;
        d.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace + dByteOffset));
        AscendC::DataCopy(raw_[slot].template ReinterpretCast<float>(),
                          d[row * FWD_H_V], TILE_ELEMENTS);
        if constexpr (CompilePolicy::STATE_FP32) {
            AscendC::GlobalTensor<float> state;
            if (chunk.first) {
                state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.initialState));
            } else {
                state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace));
            }
            if (!(chunk.first && args_.tiling.useInitialState == 0)) {
                const uint64_t base = chunk.first
                    ? FwdHStateOffset<STATE_V_FIRST>(
                          args_.tiling, unit.sequence.sequence, head.hv, 0, 0)
                    : args_.tiling.kDecayWorkspaceOffset / sizeof(float) +
                        FwdHCoreSlotOffset(coreIdx_, head.roundHead, FWD_H_K * FWD_H_V);
                AscendC::DataCopy(calc_[slot], state[base + row * FWD_H_V], TILE_ELEMENTS);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
    }

    template <bool WRITE_H, bool WAIT_READY>
    __aicore__ inline void ComputeStage3Tile(const FwdHWorkUnit &unit,
                                             const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head,
                                             uint32_t row)
    {
        const uint32_t slot = head.localSlot;
        if constexpr (WAIT_READY) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
        }
        if constexpr (CompilePolicy::STATE_FP32) {
            if (chunk.first && args_.tiling.useInitialState == 0) {
                AscendC::Duplicate(calc_[slot], 0.0f, TILE_ELEMENTS);
            }
        }
        if constexpr (!CompilePolicy::STATE_FP32) {
            AscendC::Cast(calc_[slot], stateBf16_[slot][row * FWD_H_V],
                          AscendC::RoundMode::CAST_NONE, TILE_ELEMENTS);
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            const float factor = ReadScalar(alpha_[slot], 0, slot);
            AscendC::Muls(calc_[slot], calc_[slot], factor, TILE_ELEMENTS);
        } else if constexpr (STATE_V_FIRST) {
            for (uint32_t localRow = 0; localRow < TILE_ROWS; ++localRow) {
                AscendC::Mul(calc_[slot][localRow * FWD_H_V],
                             calc_[slot][localRow * FWD_H_V], gateFp32_[slot], FWD_H_V);
            }
        } else {
            uint32_t dstShape[2] = {TILE_ROWS, FWD_H_V};
            uint32_t srcShape[2] = {TILE_ROWS, 1};
            AscendC::Broadcast<float, 2, 1>(cast_[slot], gateFp32_[slot][row],
                                            dstShape, srcShape, share_[slot]);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(calc_[slot], calc_[slot], cast_[slot], TILE_ELEMENTS);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(calc_[slot], calc_[slot],
                     raw_[slot].template ReinterpretCast<float>(), TILE_ELEMENTS);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (CompilePolicy::STATE_FP32) {
            AscendC::GlobalTensor<float> state;
            state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace));
            const uint64_t stateOffset = args_.tiling.kDecayWorkspaceOffset / sizeof(float) +
                FwdHCoreSlotOffset(coreIdx_, head.roundHead, FWD_H_K * FWD_H_V) + row * FWD_H_V;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
            AscendC::DataCopy(state[stateOffset], calc_[slot], TILE_ELEMENTS);
            if constexpr (!WRITE_H) {
                if (args_.tiling.storeFinalState != 0) {
                    AscendC::GlobalTensor<float> finalState;
                    finalState.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.finalState));
                    const uint64_t finalOffset = FwdHStateOffset<STATE_V_FIRST>(
                        args_.tiling, unit.sequence.sequence, head.hv, 0, 0) + row * FWD_H_V;
                    AscendC::DataCopy(finalState[finalOffset], calc_[slot], TILE_ELEMENTS);
                }
            }
        } else {
            AscendC::Cast(stateBf16_[slot][row * FWD_H_V], calc_[slot],
                          AscendC::RoundMode::CAST_RINT, TILE_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            if constexpr (!WRITE_H) {
                // Terminal BF16 state has no per-tile MTE3 copy. Bridge V to MTE3 before
                // releasing the slot so the next MTE2 cannot overwrite raw_ early.
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
            }
        }
        if constexpr (WRITE_H) {
            AscendC::Cast(io_[slot], calc_[slot], AscendC::RoundMode::CAST_RINT, TILE_ELEMENTS);
            AscendC::GlobalTensor<bfloat16_t> h;
            h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
            const FwdHChunkSpan next = FwdHBuildChunk(unit.sequence, chunk.chunk + 1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
            AscendC::DataCopy(h[HOffset(unit, next, head) + row * FWD_H_V], io_[slot], TILE_ELEMENTS);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
    }

    template <bool WRITE_H>
    __aicore__ inline void ConsumeStage3Head(const FwdHWorkUnit &unit,
                                             const FwdHChunkSpan &chunk,
                                             const FwdHHeadBinding &head)
    {
        // Stage3：g-only 为 R_next=E(g_last)R+D；gk-only 为
        // R_next[k,v]=E(gk_last[k])R[k,v]+D[k,v]，所有算术均为 FP32。
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
            PrepareKeyGate(slot);
        }
        ComputeStage3Tile<WRITE_H, false>(unit, chunk, head, 0);
        for (uint32_t row = TILE_ROWS; row < FWD_H_K; row += TILE_ROWS) {
            LoadStage3Tile(unit, chunk, head, row);
            ComputeStage3Tile<WRITE_H, true>(unit, chunk, head, row);
        }
        if constexpr (!WRITE_H) {
            if (args_.tiling.storeFinalState != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
                const uint64_t finalOffset = FwdHStateOffset<STATE_V_FIRST>(
                    args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
                if constexpr (!CompilePolicy::STATE_FP32) {
                    AscendC::GlobalTensor<bfloat16_t> finalState;
                    finalState.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.finalState));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
                    AscendC::DataCopy(finalState[finalOffset], stateBf16_[slot], FWD_H_K * FWD_H_V);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            }
        }
    }

    __aicore__ inline void CompleteStage3Pair(const FwdHWorkUnit &unit,
                                              const FwdHChunkSpan &chunk,
                                              uint32_t pairSlot, bool writeH)
    {
        if (FwdHMode2PairHasHead(unit.headRound.activeHeadCount, pairSlot, aiv_)) {
            const FwdHHeadBinding &head = unit.headRound.heads[pairSlot * 2U + aiv_];
            writeH ? ConsumeStage3Head<true>(unit, chunk, head)
                   : ConsumeStage3Head<false>(unit, chunk, head);
        }
        AscendC::CrossCoreSetFlag<0x2, PIPE_V>(
            FwdHAivLocalFlag(FWD_H_D_FREE_FLAG, pairSlot));
        if (writeH) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(
                FwdHAivLocalFlag(FWD_H_H_READY_FLAG, pairSlot));
        }
    }

    __aicore__ inline void RunStage3(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // Stage3：g-only 为 R_next=E(g_last)R+D；gk-only 为
        // R_next[k,v]=E(gk_last[k])R[k,v]+D[k,v]，并按需生成下一 H 或 final_state。
        const bool writeH = !chunk.last;
        const uint32_t pairCount = FwdHMode2PairCount(unit.headRound.activeHeadCount);
        for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
            if (pairSlot > 0) {
                // 先消费已完成 MTE2 的 ping，再等待下一 pair 的 D_READY。
                CompleteStage3Pair(unit, chunk, pairSlot - 1U, writeH);
            }
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(
                FwdHAivLocalFlag(FWD_H_D_READY_FLAG, pairSlot));
            if (FwdHMode2PairHasHead(unit.headRound.activeHeadCount, pairSlot, aiv_)) {
                const FwdHHeadBinding &head = unit.headRound.heads[pairSlot * 2U + aiv_];
                PrefetchStage3Head(unit, chunk, head);
            }
        }
        if (pairCount > 0) {
            CompleteStage3Pair(unit, chunk, pairCount - 1U, writeH);
        }
    }

    __aicore__ inline void ProcessWorkUnit(const FwdHWorkUnit &unit, bool hasNextWorkUnit)
    {
        const bool hasCubeWork = args_.tiling.useInitialState || args_.tiling.storeFinalState ||
            args_.tiling.seqlen > static_cast<int64_t>(FWD_H_CHUNK);
        const uint32_t localHeads = FwdHAivHeadCount(unit.headRound.activeHeadCount, aiv_);
        const uint32_t pairCount = FwdHMode2PairCount(unit.headRound.activeHeadCount);
        if (args_.tiling.useInitialState == 0 && unit.sequence.chunkCount > 1) {
            // 首 chunk 没有 Stage0/P 消费链。为第二个 chunk 的 Stage0 提供一次 P scratch
            // 初始 free credit；后续 chunk 的 credit 都由前一 chunk 的 Stage1 产生。
            for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_V>(
                    FwdHAivLocalFlag(FWD_H_P_FREE_FLAG, pairSlot));
            }
        }
        RunSMinusOne(unit);
        for (uint32_t chunkId = 0; chunkId < unit.sequence.chunkCount; ++chunkId) {
            const FwdHChunkSpan chunk = FwdHBuildChunk(unit.sequence, chunkId);
            if (chunkId > 0) {
                for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(
                        FwdHAivLocalFlag(FWD_H_RIGHT_FREE_FLAG, pairSlot));
                }
            }
            RunStage1(unit, chunk);
            if (args_.tiling.storeFinalState != 0 || !chunk.last) {
                RunStage3(unit, chunk);
            }
        }
        if (args_.tiling.storeFinalState != 0) {
            for (uint32_t pairSlot = 0; pairSlot < pairCount; ++pairSlot) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(
                    FwdHAivLocalFlag(FWD_H_RIGHT_FREE_FLAG, pairSlot));
            }
        }
        // 跨 head_round 前必须把本轮所有 VEC->MTE3 写回收口。AIC 收到 ROUND_DONE 后
        // 才能开始下一轮 kg/H/W 的 MTE2，避免上一轮 MTE3 与下一轮预取跨 round 交叠。
        for (uint32_t slot = 0; slot < localHeads; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        }
        if (hasCubeWork && hasNextWorkUnit) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(FwdHAivLocalFlag(FWD_H_ROUND_DONE_FLAG, 0));
            AscendC::CrossCoreWaitFlag<0x2, PIPE_V>(FwdHAivLocalFlag(FWD_H_ROUND_ACK_FLAG, 0));
        }
    }

    FwdHKernelArgs args_{};
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t aiv_ = 0;
    AscendC::TEventID ioFreeEvent_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::TEventID ioReadyEvent_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::TEventID ioDoneEvent_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::TEventID scalarReadEvent_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::TEventID scalarWriteEvent_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_{};
    AscendC::LocalTensor<uint8_t> raw_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> cast_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> calc_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> io_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> right_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<GateT> gateRaw_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> gateFp32_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<float> alpha_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<uint8_t> share_[FWD_H_AIV_HEAD_SLOTS]{};
    AscendC::LocalTensor<bfloat16_t> stateBf16_[FWD_H_AIV_HEAD_SLOTS]{};
};

} // namespace GDN

#endif // ARCH22_CHUNK_FWD_H_VEC_H
