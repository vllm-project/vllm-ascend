/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#ifndef ARCH35_CHUNK_FWD_H_VEC_H
#define ARCH35_CHUNK_FWD_H_VEC_H

#include <type_traits>

#include "../chunk_fwd_h_policy.h"
#include "../chunk_fwd_h_utils.h"
#include "kernel_utils/vector/regbase.hpp"

namespace GDN {

using namespace AscendC::MicroAPI;

constexpr float FWD_H_LN2 = 0.69314718055994530942f;
constexpr uint32_t FWD_H_ARCH35_GATE_SLOT_BYTES = 512;
constexpr uint32_t FWD_H_ARCH35_GATE_SCALE_OFFSET = 256;
static_assert(FWD_H_CHUNK * sizeof(float) <= FWD_H_ARCH35_GATE_SCALE_OFFSET);
static_assert(FWD_H_ARCH35_GATE_SCALE_OFFSET + FWD_H_CHUNK * sizeof(float) <=
              FWD_H_ARCH35_GATE_SLOT_BYTES);

constexpr CastTrait FWD_H_B16_TO_F32_ZERO = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

__simd_callee__ inline void FwdHLoadFloatPair(RegTensor<float> &zero, RegTensor<float> &one,
                                               __ubuf__ float *src)
{
    LoadAlign<float, LoadDist::DIST_DINTLV_B32>(zero, one, src);
}

__simd_callee__ inline void FwdHStoreFloatPair(__ubuf__ float *dst,
                                                RegTensor<float> &zero, RegTensor<float> &one,
                                                MaskReg &mask32)
{
    StoreAlign<float, StoreDist::DIST_INTLV_B32>(dst, zero, one, mask32);
}

template <typename T>
__simd_callee__ inline void FwdHLoadAsFloat(RegTensor<float> &zero, RegTensor<float> &one,
                                             __ubuf__ T *src, MaskReg &mask16)
{
    RegTensor<T> raw;
    LoadIn<T, false>(raw, src);
    CastHalf2Float<T>(zero, one, raw, mask16);
}

template <typename T>
__simd_callee__ inline void FwdHLoadScalar(RegTensor<float> &dst, __ubuf__ T *src,
                                            MaskReg &mask16, MaskReg &mask32)
{
    if constexpr (std::is_same<T, float>::value) {
        LoadIn<float, true>(dst, src);
    } else {
        RegTensor<T> raw;
        RegTensor<float> unused;
        LoadIn<T, true>(raw, src);
        CastHalf2Float<T>(dst, unused, raw, mask16);
    }
    (void)mask32;
}

template <typename T>
__simd_callee__ inline void FwdHLoadGateVector(RegTensor<float> &dst,
                                                __ubuf__ T *src, MaskReg &mask)
{
    if constexpr (std::is_same<T, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src);
    } else {
        static_assert(std::is_same<T, bfloat16_t>::value,
                      "chunk_fwd_h scalar gate only supports float/bfloat16_t");
        RegTensor<T> raw;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(raw, src);
        Cast<float, T, FWD_H_B16_TO_F32_ZERO>(dst, raw, mask);
    }
}

__simd_callee__ inline void FwdHStoreGateVector(__ubuf__ float *dst,
                                                 RegTensor<float> &src, MaskReg &mask)
{
    DataCopy<float, StoreDist::DIST_NORM_B32>(dst, src, mask);
}

__simd_callee__ inline void FwdHStoreGateScalar(__ubuf__ float *dst,
                                                 RegTensor<float> &src, MaskReg &mask)
{
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dst, src, mask);
}

__simd_callee__ inline void FwdHStoreBf16(__ubuf__ bfloat16_t *dst,
                                          RegTensor<float> &zero, RegTensor<float> &one,
                                          MaskReg &mask32, MaskReg &mask16)
{
    RegTensor<bfloat16_t> output;
    CastFloat2Half<bfloat16_t>(output, zero, one, mask32);
    StoreAlign(dst, output, mask16);
}

template <bool USE_EXP2, typename GateT>
__simd_vf__ inline void FwdHPrepareScalarGateArch35Vf(__ubuf__ float *gateScale,
                                                       __ubuf__ GateT *gate,
                                                       __ubuf__ float *alpha,
                                                       uint16_t validTokens)
{
    MaskReg mask16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    MaskReg mask32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> lastGate;
    FwdHLoadScalar<GateT>(lastGate, gate + validTokens - 1, mask16, mask32);

    RegTensor<float> alphaReg;
    Adds(alphaReg, lastGate, 0.0f, mask32);
    if constexpr (USE_EXP2) {
        Muls(alphaReg, alphaReg, FWD_H_LN2, mask32);
    }
    Exp(alphaReg, alphaReg, mask32);
    uint32_t scalarCount = 1;
    MaskReg scalarMask = UpdateMask<float>(scalarCount);
    FwdHStoreGateVector(alpha, alphaReg, scalarMask);

    RegTensor<float> gateReg;
    if (validTokens == FWD_H_CHUNK) {
        uint32_t gateCount = validTokens;
        MaskReg gateMask = UpdateMask<float>(gateCount);
        FwdHLoadGateVector<GateT>(gateReg, gate, gateMask);
        Sub(gateReg, lastGate, gateReg, gateMask);
        if constexpr (USE_EXP2) {
            Muls(gateReg, gateReg, FWD_H_LN2, gateMask);
        }
        Exp(gateReg, gateReg, gateMask);
        FwdHStoreGateVector(gateScale, gateReg, gateMask);
    } else {
        // DataCopyPad initializes only the valid tail and its final 32-byte block.
        // Keep tail loads scalar so the VF never reads the uninitialized remainder
        // of the 256-byte raw-gate bank. Full chunks retain the single vector Exp.
        for (uint16_t row = 0; row < validTokens; ++row) {
            FwdHLoadScalar<GateT>(gateReg, gate + row, mask16, mask32);
            Sub(gateReg, lastGate, gateReg, mask32);
            if constexpr (USE_EXP2) {
                Muls(gateReg, gateReg, FWD_H_LN2, mask32);
            }
            Exp(gateReg, gateReg, mask32);
            uint32_t one = 1;
            MaskReg oneMask = UpdateMask<float>(one);
            FwdHStoreGateScalar(gateScale + row, gateReg, oneMask);
        }
    }
}

template <bool HAS_P, bool SCALAR_G, bool WRITE_RIGHT, bool ZERO_STATE, typename PType>
__simd_vf__ inline void FwdHStage1Arch35Vf(__ubuf__ bfloat16_t *uAndVNew,
                                           __ubuf__ PType *p,
                                           __ubuf__ float *gateScale,
                                           __ubuf__ bfloat16_t *right,
                                           __ubuf__ bfloat16_t *state,
                                           uint16_t validTokens)
{
    // Stage1:
    //   V_new_fp32[i,:] = fp32(U[i,:]) - fp32(P[i,:])，HAS_P=false 时 P=0；
    //   V_new[i,:] = cast_BF16(V_new_fp32[i,:])；
    //   g-only: right[i,:] = cast_BF16(E(g_last-g_i) * V_new_fp32[i,:])，alpha=E(g_last)；
    //   gk-only: right=V_new。E 由 USE_EXP2 在编译期选择。
    // 本 VF 覆盖当前 head 的全部 MTE2 后向量计算，内部没有运行期分支。
    constexpr uint16_t BF16_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    constexpr uint16_t STATE_ROWS = FWD_H_STATE_BF16_BYTES / sizeof(bfloat16_t) / BF16_PER_REG;
    MaskReg mask16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    MaskReg mask32 = CreateMask<float, MaskPattern::ALL>();

    if constexpr (ZERO_STATE) {
        RegTensor<bfloat16_t> zeroState;
        Duplicate(zeroState, static_cast<bfloat16_t>(0), mask16);
        #pragma unroll 2
        for (uint16_t row = 0; row < STATE_ROWS; ++row) {
            StoreAlign(state + static_cast<uint32_t>(row) * BF16_PER_REG, zeroState, mask16);
        }
    }

    RegTensor<float> u0;
    RegTensor<float> u1;
    RegTensor<float> p0;
    RegTensor<float> p1;
    RegTensor<float> scaleReg;
    #pragma unroll 2
    for (uint16_t row = 0; row < validTokens; ++row) {
        const uint32_t rowBf16Offset = static_cast<uint32_t>(row) * FWD_H_V;
        const uint32_t rowFp32Offset = static_cast<uint32_t>(row) * FWD_H_V;
        if constexpr (SCALAR_G && WRITE_RIGHT) {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(scaleReg, gateScale + row);
        }
        for (uint16_t col = 0; col < FWD_H_V; col += BF16_PER_REG) {
            const uint32_t bf16Offset = rowBf16Offset + col;
            const uint32_t fp32Offset = rowFp32Offset + col;
            FwdHLoadAsFloat<bfloat16_t>(u0, u1, uAndVNew + bf16Offset, mask16);
            if constexpr (HAS_P) {
                if constexpr (std::is_same<PType, float>::value) {
                    FwdHLoadFloatPair(p0, p1, p + fp32Offset);
                } else {
                    FwdHLoadAsFloat<PType>(p0, p1, p + bf16Offset, mask16);
                }
            } else {
                Duplicate(p0, 0.0f, mask32);
                Duplicate(p1, 0.0f, mask32);
            }
            Sub(u0, u0, p0, mask32);
            Sub(u1, u1, p1, mask32);
            FwdHStoreBf16(uAndVNew + bf16Offset, u0, u1, mask32, mask16);

            if constexpr (WRITE_RIGHT) {
                if constexpr (SCALAR_G) {
                    Mul(u0, u0, scaleReg, mask32);
                    Mul(u1, u1, scaleReg, mask32);
                }
                FwdHStoreBf16(right + bf16Offset, u0, u1, mask32, mask16);
            }
        }
    }
}

template <bool STATE_FP32, bool SCALAR_G, bool STATE_V_FIRST, bool WRITE_H,
          bool ZERO_STATE, bool USE_EXP2, typename GateT>
__simd_vf__ inline void FwdHStage3Arch35Vf(__ubuf__ bfloat16_t *stateBf16,
                                           __ubuf__ float *stateFp32,
                                           __ubuf__ float *d,
                                           __ubuf__ GateT *gkLast,
                                           __ubuf__ float *alpha,
                                           __ubuf__ bfloat16_t *hNext)
{
    // Stage3:
    //   g-only: R_next = alpha * R + D，alpha=E(g_last)；
    //   gk-only: R_next[k,v] = E(gk_last[k]) * R[k,v] + D[k,v]；
    //   WRITE_H 时 H_next = cast_BF16(R_next)。
    // STATE_V_FIRST=true 时 state、D 和 H 都以物理 [V,K] 顺序处理，公式语义仍为 [K,V]。
    constexpr uint16_t BF16_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    constexpr uint16_t ROWS = 128;
    MaskReg mask16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    MaskReg mask32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> state0;
    RegTensor<float> state1;
    RegTensor<float> d0;
    RegTensor<float> d1;
    RegTensor<float> gate0;
    RegTensor<float> gate1;

    if constexpr (SCALAR_G) {
        LoadIn<float, true>(gate0, alpha);
        Adds(gate1, gate0, 0.0f, mask32);
    } else if constexpr (STATE_V_FIRST) {
        // 物理 [V,K] 布局的 128 列恰好由一对 FP32 寄存器覆盖。
        // 所有 V 行使用同一份 E(gk_last)，只计算一次并跨行复用。
        if constexpr (std::is_same<GateT, float>::value) {
            FwdHLoadFloatPair(gate0, gate1, gkLast);
        } else {
            FwdHLoadAsFloat<GateT>(gate0, gate1, gkLast, mask16);
        }
        if constexpr (USE_EXP2) {
            Muls(gate0, gate0, FWD_H_LN2, mask32);
            Muls(gate1, gate1, FWD_H_LN2, mask32);
        }
        Exp(gate0, gate0, mask32);
        Exp(gate1, gate1, mask32);
    }

    #pragma unroll 2
    for (uint16_t row = 0; row < ROWS; ++row) {
        for (uint16_t col = 0; col < FWD_H_V; col += BF16_PER_REG) {
            const uint32_t bf16Offset = static_cast<uint32_t>(row) * FWD_H_V + col;
            const uint32_t fp32Offset = static_cast<uint32_t>(row) * FWD_H_V + col;
            if constexpr (STATE_FP32) {
                if constexpr (ZERO_STATE) {
                    Duplicate(state0, 0.0f, mask32);
                    Duplicate(state1, 0.0f, mask32);
                } else {
                    FwdHLoadFloatPair(state0, state1, stateFp32 + fp32Offset);
                }
            } else {
                FwdHLoadAsFloat<bfloat16_t>(state0, state1, stateBf16 + bf16Offset, mask16);
            }
            FwdHLoadFloatPair(d0, d1, d + fp32Offset);

            if constexpr (!SCALAR_G) {
                if constexpr (!STATE_V_FIRST) {
                    FwdHLoadScalar<GateT>(gate0, gkLast + row, mask16, mask32);
                    if constexpr (USE_EXP2) {
                        Muls(gate0, gate0, FWD_H_LN2, mask32);
                    }
                    Exp(gate0, gate0, mask32);
                    Adds(gate1, gate0, 0.0f, mask32);
                }
            }
            Mul(state0, state0, gate0, mask32);
            Mul(state1, state1, gate1, mask32);
            Add(state0, state0, d0, mask32);
            Add(state1, state1, d1, mask32);

            if constexpr (STATE_FP32) {
                FwdHStoreFloatPair(stateFp32 + fp32Offset, state0, state1, mask32);
            } else {
                FwdHStoreBf16(stateBf16 + bf16Offset, state0, state1, mask32, mask16);
            }
            if constexpr (WRITE_H && STATE_FP32) {
                // FP32 state 没有独立的 32 KiB H bank。当前 D 行读入寄存器后，允许把
                // BF16 H 原位写到 D slot 的低半区；写地址只覆盖已经消费的 D 行。
                FwdHStoreBf16(hNext + bf16Offset, state0, state1, mask32, mask16);
            }
        }
    }
}

__simd_vf__ inline void FwdHSMinusOneArch35Vf(__ubuf__ float *state,
                                               __ubuf__ bfloat16_t *h)
{
    // S-1: H0 = cast_BF16(initial_state)。输入和输出保持 state_v_first 指定的同一物理顺序。
    constexpr uint16_t BF16_PER_REG = AscendC::VECTOR_REG_WIDTH / sizeof(bfloat16_t);
    MaskReg mask16 = CreateMask<bfloat16_t, MaskPattern::ALL>();
    MaskReg mask32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> state0;
    RegTensor<float> state1;
    #pragma unroll 2
    for (uint16_t row = 0; row < 128; ++row) {
        for (uint16_t col = 0; col < FWD_H_V; col += BF16_PER_REG) {
            const uint32_t fp32Offset = static_cast<uint32_t>(row) * FWD_H_V + col;
            const uint32_t bf16Offset = static_cast<uint32_t>(row) * FWD_H_V + col;
            FwdHLoadFloatPair(state0, state1, state + fp32Offset);
            FwdHStoreBf16(h + bf16Offset, state0, state1, mask32, mask16);
        }
    }
}

template <typename GateT, typename CompilePolicy, bool STATE_V_FIRST>
class ChunkFwdHVecArch35 {
public:
    __aicore__ inline void Init(const FwdHKernelArgs &args)
    {
        args_ = args;
        coreIdx_ = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        coreNum_ = AscendC::GetBlockNum();
        aiv_ = AscendC::GetSubBlockIdx();
        AscendC::LocalMemAllocator<AscendC::Hardware::UB> allocator;
        ub_ = allocator.Alloc<AscendC::TPosition::VECCALC, uint8_t>(
            AscendC::GetRuntimeUBSize());
        InitLocalEvents();
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
        DrainLocalEvents();
    }

private:
    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> UbAt(uint32_t byteOffset)
    {
        return ub_[byteOffset].template ReinterpretCast<T>();
    }

    // FP32 state 路径让同一 AIV 的两个 head 串行复用数据槽 0；原 localSlot
    // 保留为 stateSlot，使两个 64 KiB FP32 state 可以跨 chunk 常驻。
    __aicore__ inline uint32_t DataSlot(const FwdHHeadBinding &head) const
    {
        if constexpr (CompilePolicy::STATE_FP32) {
            return 0;
        }
        return head.localSlot;
    }

    __aicore__ inline uint32_t ActiveDataSlotCount(uint32_t localHeads) const
    {
        if constexpr (CompilePolicy::STATE_FP32) {
            return localHeads == 0 ? 0 : 1;
        }
        return localHeads;
    }

    __aicore__ inline AscendC::LocalTensor<uint8_t> LocalSlot(uint32_t slot)
    {
        return UbAt<uint8_t>(slot * FWD_H_UB_LOCAL_SLOT_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> RightSlot(uint32_t slot)
    {
        return UbAt<bfloat16_t>(slot * FWD_H_UB_LOCAL_SLOT_BYTES + FWD_H_TOKEN_MATRIX_FP32_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> StateBf16Slot(uint32_t slot)
    {
        return UbAt<bfloat16_t>(FWD_H_UB_BF16_STATE_BASE + slot * FWD_H_STATE_BF16_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> WorkBf16Slot(uint32_t slot)
    {
        return UbAt<bfloat16_t>(FWD_H_UB_BF16_WORK_BASE + slot * FWD_H_TOKEN_MATRIX_BF16_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<bfloat16_t> SminusHSlot(uint32_t slot)
    {
        return UbAt<bfloat16_t>(slot * FWD_H_STATE_BF16_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<float> SminusStateSlot(uint32_t slot)
    {
        return StateFp32(slot);
    }

    __aicore__ inline AscendC::LocalTensor<float> StateFp32(uint32_t slot)
    {
        return UbAt<float>(FWD_H_UB_LOCAL1_BASE + slot * FWD_H_STATE_FP32_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<GateT> GateSlot(uint32_t slot)
    {
        return UbAt<GateT>(FWD_H_UB_GATE_BASE + slot * FWD_H_ARCH35_GATE_SLOT_BYTES);
    }

    __aicore__ inline AscendC::LocalTensor<float> GateScaleSlot(uint32_t slot)
    {
        return UbAt<float>(FWD_H_UB_GATE_BASE + slot * FWD_H_ARCH35_GATE_SLOT_BYTES +
                           FWD_H_ARCH35_GATE_SCALE_OFFSET);
    }

    __aicore__ inline AscendC::LocalTensor<float> AlphaSlot(uint32_t slot)
    {
        return UbAt<float>(FWD_H_UB_GATE_BASE + 1024 + slot * 32);
    }

    __aicore__ inline void InitLocalEvents()
    {
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(WorkFreeEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(GateFreeEvent(slot));
        }
    }

    __aicore__ inline AscendC::TEventID IoFreeEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(slot);
    }

    __aicore__ inline AscendC::TEventID IoReadyEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(slot);
    }

    __aicore__ inline AscendC::TEventID IoDoneEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(slot);
    }

    __aicore__ inline AscendC::TEventID WorkFreeEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(FWD_H_AIV_HEAD_SLOTS + slot);
    }

    __aicore__ inline AscendC::TEventID WorkReadyEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(FWD_H_AIV_HEAD_SLOTS + slot);
    }

    __aicore__ inline AscendC::TEventID WorkDoneEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(FWD_H_AIV_HEAD_SLOTS + slot);
    }

    __aicore__ inline AscendC::TEventID GateFreeEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(slot);
    }

    __aicore__ inline AscendC::TEventID StateMte3ToVEvent(uint32_t slot) const
    {
        return static_cast<AscendC::TEventID>(slot);
    }

    __aicore__ inline void WaitStateWritebackBeforeVectorReuse(uint32_t slot)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(StateMte3ToVEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(StateMte3ToVEvent(slot));
    }

    __aicore__ inline void DrainLocalEvents()
    {
        for (uint32_t slot = 0; slot < FWD_H_AIV_HEAD_SLOTS; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(WorkFreeEvent(slot));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(GateFreeEvent(slot));
        }
        // Io 使用 0/1；跨 chunk Work 使用 2/3；Gate/alpha 的 V->MTE2 credit
        // 使用独立 HardEvent 空间的 0/1。所有代际均在退出前回到 free 状态。
    }

    __aicore__ inline uint64_t HOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head) const
    {
        return FwdHHOffset(args_.tiling, unit.sequence, head.hv, chunk.globalChunk);
    }

    __aicore__ inline uint64_t UOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                       const FwdHHeadBinding &head) const
    {
        return FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                               chunk.tokenBegin, FWD_H_V);
    }

    __aicore__ inline uint64_t GateOffset(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head) const
    {
        const uint32_t dim = CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G ? 1 : FWD_H_K;
        return FwdHInputOffset(args_.tiling, unit.sequence.physicalBatch, head.hv,
                               chunk.tokenBegin, dim);
    }

    __aicore__ inline void CopyGateToUb(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                        const FwdHHeadBinding &head, uint32_t gateSlot)
    {
        AscendC::GlobalTensor<GateT> gateGm;
        gateGm.SetGlobalBuffer(reinterpret_cast<__gm__ GateT *>(
            CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G ? args_.g : args_.gk));
        gateGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            const uint64_t gateOffset = GateOffset(unit, chunk, head);
            if (chunk.validTokens == FWD_H_CHUNK) {
                AscendC::DataCopy(GateSlot(gateSlot), gateGm[gateOffset], FWD_H_CHUNK);
            } else {
                AscendC::DataCopyPadExtParams<GateT> pad{false, 0, 0, 0};
                const uint32_t gateBytes = static_cast<uint32_t>(chunk.validTokens * sizeof(GateT));
                AscendC::DataCopyExtParams copy{1, gateBytes, 0, 0, 0};
                AscendC::DataCopyPad(GateSlot(gateSlot), gateGm[gateOffset], copy, pad);
            }
        } else {
            const uint64_t lastOffset = GateOffset(unit, chunk, head) +
                                        static_cast<uint64_t>(chunk.validTokens - 1) * FWD_H_K;
            AscendC::DataCopy(GateSlot(gateSlot), gateGm[lastOffset], FWD_H_K);
        }
    }

    __aicore__ inline void PrefetchSMinusOneHead(const FwdHWorkUnit &unit,
                                                 const FwdHHeadBinding &head)
    {
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        AscendC::GlobalTensor<float> initial;
        initial.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.initialState));
        const uint64_t stateBase = FwdHStateOffset<STATE_V_FIRST>(
            args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
        AscendC::DataCopy(SminusStateSlot(slot), initial[stateBase], FWD_H_K * FWD_H_V);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
    }

    __aicore__ inline void ConsumeSMinusOneHead(
        const FwdHWorkUnit &unit, const FwdHHeadBinding &head)
    {
        // S-1：H0=cast_BF16(initial_state)。输入输出保持 state_v_first 指定的物理顺序。
        const uint32_t slot = head.localSlot;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
        AscendC::VF_CALL<FwdHSMinusOneArch35Vf>(
            reinterpret_cast<__ubuf__ float *>(SminusStateSlot(slot).GetPhyAddr()),
            reinterpret_cast<__ubuf__ bfloat16_t *>(SminusHSlot(slot).GetPhyAddr()));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        AscendC::GlobalTensor<bfloat16_t> h;
        h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
        const FwdHChunkSpan firstChunk = FwdHBuildChunk(unit.sequence, 0);
        AscendC::DataCopy(h[HOffset(unit, firstChunk, head)], SminusHSlot(slot),
                          FWD_H_K * FWD_H_V);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
    }

    __aicore__ inline void RunSMinusOne(const FwdHWorkUnit &unit)
    {
        // S-1：StateT=FP32 且存在 initial_state 时，H0=cast_BF16(R0)；
        // 输入输出保持 state_v_first 指定的物理顺序，不依赖外部转置。
        if constexpr (!CompilePolicy::STATE_FP32) {
            return;
        }
        if (args_.tiling.useInitialState == 0) {
            return;
        }
        const uint32_t localHeads = FwdHAivHeadCount(unit.headRound.activeHeadCount, aiv_);
        for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
            const FwdHHeadBinding &head = unit.headRound.heads[aiv_ + 2 * coreHeadId];
            PrefetchSMinusOneHead(unit, head);
            if (coreHeadId > 0) {
                ConsumeSMinusOneHead(
                    unit, unit.headRound.heads[aiv_ + 2 * (coreHeadId - 1)]);
            }
        }
        if (localHeads > 0) {
            ConsumeSMinusOneHead(
                unit, unit.headRound.heads[aiv_ + 2 * (localHeads - 1)]);
        }
        // S-1 与主循环复用 [0,192) KiB。先 drain 全部 active bank，再统一发布 H ready。
        for (uint32_t slot = 0; slot < localHeads; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        }
        for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
            const FwdHHeadBinding &head = unit.headRound.heads[aiv_ + 2 * coreHeadId];
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                FwdHAivLocalFlag(FWD_H_H_READY_FLAG, head.localSlot));
        }
    }

    // 保留 leaf stage 调用边界，使各 stage 的临时标量对象复用同一栈区。
    template <bool HAS_P, bool WRITE_RIGHT, bool ZERO_STATE, bool SEPARATE_INPUT>
    __aicore__ inline void ConsumeStage1Head(
        const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk, const FwdHHeadBinding &head,
        uint32_t inputSlot)
    {
        const uint32_t slot = DataSlot(head);
        if constexpr (SEPARATE_INPUT) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(WorkReadyEvent(inputSlot));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
        }
        using PType = std::conditional_t<CompilePolicy::STATE_FP32, float, bfloat16_t>;
        __ubuf__ PType *p = reinterpret_cast<__ubuf__ PType *>(LocalSlot(slot).GetPhyAddr());
        __ubuf__ bfloat16_t *right = reinterpret_cast<__ubuf__ bfloat16_t *>(RightSlot(slot).GetPhyAddr());
        __ubuf__ bfloat16_t *state = reinterpret_cast<__ubuf__ bfloat16_t *>(StateBf16Slot(slot).GetPhyAddr());
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            if constexpr (WRITE_RIGHT) {
                AscendC::VF_CALL<FwdHPrepareScalarGateArch35Vf<CompilePolicy::USE_EXP2, GateT>>(
                    reinterpret_cast<__ubuf__ float *>(GateScaleSlot(inputSlot).GetPhyAddr()),
                    reinterpret_cast<__ubuf__ GateT *>(GateSlot(inputSlot).GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float *>(AlphaSlot(inputSlot).GetPhyAddr()),
                    static_cast<uint16_t>(chunk.validTokens));
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        if constexpr (HAS_P) {
            // Gate decay is independent of P. Queue it before the cross-core P wait so
            // scalar Exp can overlap the producer's Stage0 tail.
            AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                FwdHAivLocalFlag(FWD_H_P_READY_FLAG, head.localSlot));
        }
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            AscendC::VF_CALL<FwdHStage1Arch35Vf<HAS_P, true, WRITE_RIGHT, ZERO_STATE, PType>>(
                reinterpret_cast<__ubuf__ bfloat16_t *>(WorkBf16Slot(inputSlot).GetPhyAddr()), p,
                reinterpret_cast<__ubuf__ float *>(GateScaleSlot(inputSlot).GetPhyAddr()), right, state,
                static_cast<uint16_t>(chunk.validTokens));
        } else {
            AscendC::VF_CALL<FwdHStage1Arch35Vf<HAS_P, false, WRITE_RIGHT, ZERO_STATE, PType>>(
                reinterpret_cast<__ubuf__ bfloat16_t *>(WorkBf16Slot(inputSlot).GetPhyAddr()), p,
                reinterpret_cast<__ubuf__ float *>(GateScaleSlot(inputSlot).GetPhyAddr()), right, state,
                static_cast<uint16_t>(chunk.validTokens));
        }
        if constexpr (HAS_P) {
            AscendC::CrossCoreSetFlag<0x4, PIPE_V>(
                FwdHAivLocalFlag(FWD_H_P_FREE_FLAG, head.localSlot));
        }

        AscendC::GlobalTensor<bfloat16_t> vNew;
        AscendC::GlobalTensor<bfloat16_t> rightGm;
        AscendC::GlobalTensor<bfloat16_t> h;
        vNew.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.vNew));
        rightGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            args_.workspace + args_.tiling.vUpdateWorkspaceOffset));
        h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
        if constexpr (SEPARATE_INPUT) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(WorkDoneEvent(inputSlot));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(WorkDoneEvent(inputSlot));
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        }
        AscendC::DataCopy(vNew[UOffset(unit, chunk, head)], WorkBf16Slot(inputSlot),
                          chunk.validTokens * FWD_H_V);
        if constexpr (WRITE_RIGHT) {
            const uint64_t rightOffset = FwdHCoreSlotOffset(coreIdx_, head.roundHead,
                                                            FWD_H_CHUNK * FWD_H_V);
            AscendC::DataCopy(rightGm[rightOffset], RightSlot(slot), chunk.validTokens * FWD_H_V);
            if constexpr (!CompilePolicy::STATE_FP32) {
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                    FwdHAivLocalFlag(FWD_H_RIGHT_READY_FLAG, head.localSlot));
            }
        }
        if constexpr (!CompilePolicy::STATE_FP32) {
            if (chunk.first) {
                AscendC::DataCopy(h[HOffset(unit, chunk, head)], StateBf16Slot(slot), FWD_H_K * FWD_H_V);
                // Stage3 reads and updates the same BF16 state bank on PIPE_V.
                WaitStateWritebackBeforeVectorReuse(slot);
            }
        } else if constexpr (ZERO_STATE) {
            AscendC::DataCopy(h[HOffset(unit, chunk, head)], StateBf16Slot(slot), FWD_H_K * FWD_H_V);
            // H0 临时区只在首 chunk 使用；对应 FP32 state 在 Stage3 以 ZERO_INIT 生成。
            WaitStateWritebackBeforeVectorReuse(slot);
        }
        if constexpr (SEPARATE_INPUT) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(WorkFreeEvent(inputSlot));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
    }

    __aicore__ inline void PrefetchStage1Inputs(
        const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk, const FwdHHeadBinding &head,
        uint32_t inputSlot, bool writeRight)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(WorkFreeEvent(inputSlot));
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            if (writeRight) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(GateFreeEvent(inputSlot));
            }
        }
        AscendC::GlobalTensor<bfloat16_t> u;
        u.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.u));
        u.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        AscendC::DataCopy(WorkBf16Slot(inputSlot), u[UOffset(unit, chunk, head)],
                          chunk.validTokens * FWD_H_V);
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            if (writeRight) {
                CopyGateToUb(unit, chunk, head, inputSlot);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(WorkReadyEvent(inputSlot));
    }

    __aicore__ inline void AcquireStage1LocalSlot(const FwdHHeadBinding &head)
    {
        // Publish local-slot ownership to V before queuing next-chunk U/g on MTE2;
        // otherwise the current Stage1 would sit behind the lookahead transfer.
        const uint32_t slot = DataSlot(head);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
    }

    __aicore__ inline void RunStage1(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        // Stage1：V_new=cast_BF16(fp32(U)-fp32(P))；g-only 同时生成
        // V_new_g=cast_BF16(E(g_last-g_i)*V_new_fp32)，gk-only 的右矩阵就是 V_new。
        const bool hasP = !(chunk.first && args_.tiling.useInitialState == 0);
        const bool writeRight = args_.tiling.storeFinalState != 0 || !chunk.last;
        const bool zeroState = chunk.first && args_.tiling.useInitialState == 0;
        const uint32_t localHeads = FwdHAivHeadCount(unit.headRound.activeHeadCount, aiv_);
        if constexpr (CompilePolicy::STATE_FP32) {
            // U/g 使用两个独立 input bank 预取；P/D 只使用数据槽 0，并由 ready/free
            // 在两个 head 之间串行传递所有权。这样不增加 round 次数。
            for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
                const FwdHHeadBinding &head = unit.headRound.heads[aiv_ + 2 * coreHeadId];
                PrefetchStage1Inputs(unit, chunk, head, coreHeadId, writeRight);
            }
            for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
                const FwdHHeadBinding &head = unit.headRound.heads[aiv_ + 2 * coreHeadId];
                AcquireStage1LocalSlot(head);
                DispatchStage1<true>(unit, chunk, head, coreHeadId,
                                     hasP, writeRight, zeroState);
            }
            if (writeRight) {
                // 两个 head 的 right 共用数据槽高半区。等最后一次 MTE3 读取完成后，
                // 再按逻辑 slot 发布原有 ready，防止首个 D 提前覆盖第二个 right。
                for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
                    const FwdHHeadBinding &head =
                        unit.headRound.heads[aiv_ + 2 * coreHeadId];
                    AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                        FwdHAivLocalFlag(FWD_H_RIGHT_READY_FLAG, head.localSlot));
                }
            }
        } else {
            for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
                const FwdHHeadBinding &head = unit.headRound.heads[aiv_ + 2 * coreHeadId];
                const uint32_t slot = DataSlot(head);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
                AscendC::GlobalTensor<bfloat16_t> u;
                u.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.u));
                u.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                AscendC::DataCopy(WorkBf16Slot(slot), u[UOffset(unit, chunk, head)],
                                  chunk.validTokens * FWD_H_V);
                if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                    if (writeRight) {
                        CopyGateToUb(unit, chunk, head, slot);
                    }
                }
                if (chunk.first && args_.tiling.useInitialState != 0) {
                    AscendC::GlobalTensor<bfloat16_t> initial;
                    initial.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.initialState));
                    const uint64_t stateOffset = FwdHStateOffset<STATE_V_FIRST>(
                        args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
                    AscendC::DataCopy(StateBf16Slot(slot), initial[stateOffset], FWD_H_K * FWD_H_V);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
                if (coreHeadId > 0) {
                    const FwdHHeadBinding &previous =
                        unit.headRound.heads[aiv_ + 2 * (coreHeadId - 1)];
                    DispatchStage1<false>(unit, chunk, previous, DataSlot(previous),
                                          hasP, writeRight, zeroState);
                }
            }
            if (localHeads > 0) {
                const FwdHHeadBinding &last = unit.headRound.heads[aiv_ + 2 * (localHeads - 1)];
                DispatchStage1<false>(unit, chunk, last, DataSlot(last),
                                      hasP, writeRight, zeroState);
            }
        }
    }

    template <bool SEPARATE_INPUT>
    __aicore__ inline void DispatchStage1(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                          const FwdHHeadBinding &head, uint32_t inputSlot, bool hasP,
                                          bool writeRight, bool zeroState)
    {
        if (hasP) {
            if (writeRight) {
                zeroState ? ConsumeStage1Head<true, true, true, SEPARATE_INPUT>(
                                unit, chunk, head, inputSlot)
                          : ConsumeStage1Head<true, true, false, SEPARATE_INPUT>(
                                unit, chunk, head, inputSlot);
            } else {
                zeroState ? ConsumeStage1Head<true, false, true, SEPARATE_INPUT>(
                                unit, chunk, head, inputSlot)
                          : ConsumeStage1Head<true, false, false, SEPARATE_INPUT>(
                                unit, chunk, head, inputSlot);
            }
        } else if (writeRight) {
            zeroState ? ConsumeStage1Head<false, true, true, SEPARATE_INPUT>(
                            unit, chunk, head, inputSlot)
                      : ConsumeStage1Head<false, true, false, SEPARATE_INPUT>(
                            unit, chunk, head, inputSlot);
        } else {
            zeroState ? ConsumeStage1Head<false, false, true, SEPARATE_INPUT>(
                            unit, chunk, head, inputSlot)
                      : ConsumeStage1Head<false, false, false, SEPARATE_INPUT>(
                            unit, chunk, head, inputSlot);
        }
    }

    template <bool WRITE_H, bool WAIT_MTE2, bool ZERO_INIT, bool RESIDENT_FP32_STATE,
              bool RELEASE_GATE_BANK>
    __aicore__ inline void ConsumeStage3Head(
        const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk, const FwdHHeadBinding &head,
        uint32_t gateSlot)
    {
        static_assert(!RESIDENT_FP32_STATE || CompilePolicy::STATE_FP32,
                      "resident state is only valid for FP32 state");
        const uint32_t slot = DataSlot(head);
        const uint32_t stateSlot = head.localSlot;
        AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
            FwdHAivLocalFlag(FWD_H_D_READY_FLAG, head.localSlot));
        if constexpr (WAIT_MTE2) {
            // 只有本次 Stage3 确实从 GM 搬入 state/gk 时才等待 MTE2；BF16 g-only
            // 直接消费 Stage1 已驻留的 stateBf16，不能等待不存在的 event credit。
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
        }
        __ubuf__ float *d = reinterpret_cast<__ubuf__ float *>(LocalSlot(slot).GetPhyAddr());
        __ubuf__ bfloat16_t *hNext = nullptr;
        if constexpr (CompilePolicy::STATE_FP32) {
            // D 使用完整 64 KiB local slot；right 只是其中高半区的地址别名。
            // VF 逐行加载对应的 FP32 D 后，再从低地址向高地址写 BF16 H；
            // 因此每次写入只复用已经消费完成的 D 行。
            hNext = reinterpret_cast<__ubuf__ bfloat16_t *>(LocalSlot(slot).GetPhyAddr());
        } else {
            hNext = reinterpret_cast<__ubuf__ bfloat16_t *>(StateBf16Slot(slot).GetPhyAddr());
        }
        if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            AscendC::VF_CALL<FwdHStage3Arch35Vf<CompilePolicy::STATE_FP32, true, STATE_V_FIRST,
                                                WRITE_H, ZERO_INIT, CompilePolicy::USE_EXP2, GateT>>(
                reinterpret_cast<__ubuf__ bfloat16_t *>(StateBf16Slot(slot).GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(StateFp32(stateSlot).GetPhyAddr()), d,
                reinterpret_cast<__ubuf__ GateT *>(GateSlot(gateSlot).GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(AlphaSlot(gateSlot).GetPhyAddr()), hNext);
        } else {
            AscendC::VF_CALL<FwdHStage3Arch35Vf<CompilePolicy::STATE_FP32, false, STATE_V_FIRST,
                                                WRITE_H, ZERO_INIT, CompilePolicy::USE_EXP2, GateT>>(
                reinterpret_cast<__ubuf__ bfloat16_t *>(StateBf16Slot(slot).GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(StateFp32(stateSlot).GetPhyAddr()), d,
                reinterpret_cast<__ubuf__ GateT *>(GateSlot(gateSlot).GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(AlphaSlot(gateSlot).GetPhyAddr()), hNext);
        }
        if constexpr (RELEASE_GATE_BANK) {
            static_assert(CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G,
                          "cross-chunk gate banks are only used by scalar-g");
            // Stage3 is the final V consumer of alpha/gateScale. Publish this bank to
            // MTE2 only after its vector reads have completed.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(GateFreeEvent(gateSlot));
        }
        if constexpr (!WRITE_H || !CompilePolicy::STATE_FP32) {
            AscendC::CrossCoreSetFlag<0x4, PIPE_V>(
                FwdHAivLocalFlag(FWD_H_D_FREE_FLAG, head.localSlot));
        }

        AscendC::GlobalTensor<bfloat16_t> h;
        h.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.h));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(IoDoneEvent(slot));
        if constexpr (WRITE_H) {
            if constexpr (CompilePolicy::STATE_FP32) {
                if constexpr (!RESIDENT_FP32_STATE) {
                    AscendC::GlobalTensor<float> state;
                    state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace));
                    const uint64_t stateOffset = args_.tiling.kDecayWorkspaceOffset / sizeof(float) +
                        FwdHCoreSlotOffset(coreIdx_, head.roundHead, FWD_H_K * FWD_H_V);
                    AscendC::DataCopy(state[stateOffset], StateFp32(stateSlot), FWD_H_K * FWD_H_V);
                    if constexpr (ZERO_INIT) {
                        // Zero-init skips the next head's MTE2 state load, so it needs a
                        // direct MTE3->V fence before reusing the shared FP32 bank.
                        WaitStateWritebackBeforeVectorReuse(slot);
                    }
                }
            }
            const FwdHChunkSpan next = FwdHBuildChunk(unit.sequence, chunk.chunk + 1);
            if constexpr (CompilePolicy::STATE_FP32) {
                AscendC::LocalTensor<bfloat16_t> hLocal =
                    LocalSlot(slot).template ReinterpretCast<bfloat16_t>();
                AscendC::DataCopy(h[HOffset(unit, next, head)], hLocal, FWD_H_K * FWD_H_V);
                // PIPE_MTE3 的跨核通知排在 H 写回之后；AIC 收到 D_FREE 时，D slot
                // 已不再作为 MTE3 输入，可以安全进入下一次 Stage0/Stage2。
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                    FwdHAivLocalFlag(FWD_H_D_FREE_FLAG, head.localSlot));
            } else {
                AscendC::DataCopy(h[HOffset(unit, next, head)], StateBf16Slot(slot),
                                  FWD_H_K * FWD_H_V);
            }
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                FwdHAivLocalFlag(FWD_H_H_READY_FLAG, head.localSlot));
        } else {
            if (args_.tiling.storeFinalState != 0) {
                if constexpr (CompilePolicy::STATE_FP32) {
                    AscendC::GlobalTensor<float> finalState;
                    finalState.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.finalState));
                    const uint64_t offset = FwdHStateOffset<STATE_V_FIRST>(
                        args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
                    AscendC::DataCopy(finalState[offset], StateFp32(stateSlot), FWD_H_K * FWD_H_V);
                    if constexpr (ZERO_INIT && !RESIDENT_FP32_STATE) {
                        WaitStateWritebackBeforeVectorReuse(slot);
                    }
                } else {
                    AscendC::GlobalTensor<bfloat16_t> finalState;
                    finalState.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(args_.finalState));
                    const uint64_t offset = FwdHStateOffset<STATE_V_FIRST>(
                        args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
                    AscendC::DataCopy(finalState[offset], StateBf16Slot(slot), FWD_H_K * FWD_H_V);
                }
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
    }

    template <bool RESIDENT_FP32_STATE, bool EXTERNAL_GATE_BANK>
    __aicore__ inline void RunStage3Impl(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk,
                                         uint32_t externalGateSlot)
    {
        static_assert(!RESIDENT_FP32_STATE || CompilePolicy::STATE_FP32,
                      "resident state is only valid for FP32 state");
        static_assert(!EXTERNAL_GATE_BANK ||
                          CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G,
                      "external gate banks are only valid for scalar-g");
        // Stage3：g-only 为 R_next=E(g_last)R+D；gk-only 为
        // R_next[k,v]=E(gk_last[k])R[k,v]+D[k,v]，并按需生成下一 H 或 final_state。
        const bool writeH = !chunk.last;
        const uint32_t localHeads = FwdHAivHeadCount(unit.headRound.activeHeadCount, aiv_);
        if constexpr (CompilePolicy::STATE_FP32) {
            // FP32 路径只有一个 P/D 数据槽；进入 Stage3 前收回该槽的最终本地 credit。
            for (uint32_t slot = 0; slot < ActiveDataSlotCount(localHeads); ++slot) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            }
        }
        for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
            const FwdHHeadBinding &head = unit.headRound.heads[aiv_ + 2 * coreHeadId];
            const uint32_t slot = DataSlot(head);
            // externalGateSlot==FWD_H_AIV_HEAD_SLOTS 表示按逻辑 head slot 选择双 gate bank；
            // 单 head chunk 流水仍直接传入 chunk 奇偶 bank。
            const uint32_t gateSlot = EXTERNAL_GATE_BANK
                                          ? (externalGateSlot == FWD_H_AIV_HEAD_SLOTS
                                                 ? head.localSlot
                                                 : externalGateSlot)
                                          : slot;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            if constexpr (CompilePolicy::STATE_FP32) {
                if (chunk.first) {
                    if (args_.tiling.useInitialState != 0) {
                        AscendC::GlobalTensor<float> initial;
                        initial.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.initialState));
                        const uint64_t initialOffset = FwdHStateOffset<STATE_V_FIRST>(
                            args_.tiling, unit.sequence.sequence, head.hv, 0, 0);
                        AscendC::DataCopy(StateFp32(head.localSlot), initial[initialOffset],
                                          FWD_H_K * FWD_H_V);
                    }
                } else if constexpr (!RESIDENT_FP32_STATE) {
                    AscendC::GlobalTensor<float> state;
                    const uint64_t workspaceBase = args_.tiling.kDecayWorkspaceOffset / sizeof(float);
                    state.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args_.workspace));
                    const uint64_t offset = workspaceBase +
                        FwdHCoreSlotOffset(coreIdx_, head.roundHead, FWD_H_K * FWD_H_V);
                    AscendC::DataCopy(StateFp32(head.localSlot), state[offset], FWD_H_K * FWD_H_V);
                }
            }
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
                CopyGateToUb(unit, chunk, head, slot);
            }
            const bool zeroState = chunk.first && args_.tiling.useInitialState == 0;
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
            } else if constexpr (CompilePolicy::STATE_FP32) {
                const bool stateLoaded = !zeroState && (!RESIDENT_FP32_STATE || chunk.first);
                if (stateLoaded) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IoReadyEvent(slot));
                }
            }
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::KEY_GK) {
                // gk-only 每个 Stage3 都搬入 gk_last，必须等待该 MTE2。
                if (zeroState) {
                    if (writeH) {
                        ConsumeStage3Head<true, true, true, RESIDENT_FP32_STATE,
                                          EXTERNAL_GATE_BANK>(unit, chunk, head, gateSlot);
                    } else {
                        ConsumeStage3Head<false, true, true, RESIDENT_FP32_STATE,
                                          EXTERNAL_GATE_BANK>(unit, chunk, head, gateSlot);
                    }
                } else if (writeH) {
                    ConsumeStage3Head<true, true, false, RESIDENT_FP32_STATE,
                                      EXTERNAL_GATE_BANK>(unit, chunk, head, gateSlot);
                } else {
                    ConsumeStage3Head<false, true, false, RESIDENT_FP32_STATE,
                                      EXTERNAL_GATE_BANK>(unit, chunk, head, gateSlot);
                }
            } else if constexpr (CompilePolicy::STATE_FP32) {
                if (zeroState) {
                    // 首 chunk 无 initial_state 时 state 为 VF 内构造的零值，没有 MTE2 输入。
                    if (writeH) {
                        ConsumeStage3Head<true, false, true, RESIDENT_FP32_STATE,
                                          EXTERNAL_GATE_BANK>(unit, chunk, head, gateSlot);
                    } else {
                        ConsumeStage3Head<false, false, true, RESIDENT_FP32_STATE,
                                          EXTERNAL_GATE_BANK>(unit, chunk, head, gateSlot);
                    }
                } else {
                    const bool waitStateMte2 = !RESIDENT_FP32_STATE || chunk.first;
                    if (waitStateMte2) {
                        if (writeH) {
                            ConsumeStage3Head<true, true, false, RESIDENT_FP32_STATE,
                                              EXTERNAL_GATE_BANK>(unit, chunk, head, gateSlot);
                        } else {
                            ConsumeStage3Head<false, true, false, RESIDENT_FP32_STATE,
                                              EXTERNAL_GATE_BANK>(unit, chunk, head, gateSlot);
                        }
                    } else if (writeH) {
                        ConsumeStage3Head<true, false, false, RESIDENT_FP32_STATE,
                                          EXTERNAL_GATE_BANK>(unit, chunk, head, gateSlot);
                    } else {
                        ConsumeStage3Head<false, false, false, RESIDENT_FP32_STATE,
                                          EXTERNAL_GATE_BANK>(unit, chunk, head, gateSlot);
                    }
                }
            } else if (writeH) {
                // BF16 g-only 的 stateBf16 由 Stage1 写入并由 right-ready 链路保护。
                ConsumeStage3Head<true, false, false, false, false>(unit, chunk, head, gateSlot);
            } else {
                ConsumeStage3Head<false, false, false, false, false>(unit, chunk, head, gateSlot);
            }
        }
    }

    __aicore__ inline void RunStage3(const FwdHWorkUnit &unit, const FwdHChunkSpan &chunk)
    {
        if constexpr (CompilePolicy::STATE_FP32) {
            // 每个 AIV 的两个 64 KiB state bank 分别绑定 localSlot 0/1，四 head
            // 单轮时也可以跨 chunk 常驻。scalar-g 的 alpha 使用对应逻辑 head bank。
            if constexpr (CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
                RunStage3Impl<true, true>(unit, chunk, FWD_H_AIV_HEAD_SLOTS);
            } else {
                RunStage3Impl<true, false>(unit, chunk, 0);
            }
        } else {
            RunStage3Impl<false, false>(unit, chunk, 0);
        }
    }

    __aicore__ inline bool NeedsStage3(const FwdHChunkSpan &chunk) const
    {
        return args_.tiling.storeFinalState != 0 || !chunk.last;
    }

    __aicore__ inline void ProcessSingleHeadPipeline(const FwdHWorkUnit &unit)
    {
        // activeHeadCount==1 时只有 AIV0 持有 head。U/g 使用 chunk 奇偶 bank，
        // 下一 chunk 的 MTE2 在当前 Stage1/Stage3 前发射；P/D/right/state 仍使用
        // 逻辑 localSlot0，并保持原有跨核 ready/free 协议。
        const uint32_t localHeads = FwdHAivHeadCount(unit.headRound.activeHeadCount, aiv_);
        if (localHeads == 0) {
            return;
        }
        const FwdHHeadBinding &head = unit.headRound.heads[aiv_];
        const FwdHChunkSpan first = FwdHBuildChunk(unit.sequence, 0);
        const bool firstWriteRight = NeedsStage3(first);
        PrefetchStage1Inputs(unit, first, head, 0, firstWriteRight);

        for (uint32_t chunkId = 0; chunkId < unit.sequence.chunkCount; ++chunkId) {
            const FwdHChunkSpan chunk = FwdHBuildChunk(unit.sequence, chunkId);
            const uint32_t inputSlot = chunkId & 1U;
            const bool writeRight = NeedsStage3(chunk);
            AcquireStage1LocalSlot(head);
            if (!chunk.last) {
                const FwdHChunkSpan next = FwdHBuildChunk(unit.sequence, chunkId + 1);
                PrefetchStage1Inputs(unit, next, head, inputSlot ^ 1U, NeedsStage3(next));
            }
            if (chunkId > 0) {
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                    FwdHAivLocalFlag(FWD_H_RIGHT_FREE_FLAG, head.localSlot));
            }
            const bool hasP = !(chunk.first && args_.tiling.useInitialState == 0);
            const bool zeroState = chunk.first && args_.tiling.useInitialState == 0;
            DispatchStage1<true>(unit, chunk, head, inputSlot, hasP, writeRight, zeroState);
            if (writeRight) {
                // 单 head 不需要合并等待，保持一份 ready/一次消费的原协议。
                AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(
                    FwdHAivLocalFlag(FWD_H_RIGHT_READY_FLAG, head.localSlot));
            }
            if (writeRight) {
                RunStage3Impl<true, true>(unit, chunk, inputSlot);
            }
        }
    }

    __aicore__ inline void ProcessWorkUnit(const FwdHWorkUnit &unit, bool hasNextWorkUnit)
    {
        const bool hasCubeWork = args_.tiling.useInitialState || args_.tiling.storeFinalState ||
            args_.tiling.seqlen > static_cast<int64_t>(FWD_H_CHUNK);
        const uint32_t localHeads = FwdHAivHeadCount(unit.headRound.activeHeadCount, aiv_);
        RunSMinusOne(unit);
        if constexpr (CompilePolicy::STATE_FP32 &&
                      CompilePolicy::GATE_MODE == FwdHGateMode::SCALAR_G) {
            if (unit.headRound.activeHeadCount == 1) {
                ProcessSingleHeadPipeline(unit);
            } else {
                for (uint32_t chunkId = 0; chunkId < unit.sequence.chunkCount; ++chunkId) {
                    const FwdHChunkSpan chunk = FwdHBuildChunk(unit.sequence, chunkId);
                    if (chunkId > 0) {
                        for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
                            const FwdHHeadBinding &head =
                                unit.headRound.heads[aiv_ + 2 * coreHeadId];
                            AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                                FwdHAivLocalFlag(FWD_H_RIGHT_FREE_FLAG, head.localSlot));
                        }
                    }
                    RunStage1(unit, chunk);
                    if (NeedsStage3(chunk)) {
                        RunStage3(unit, chunk);
                    }
                }
            }
        } else {
            for (uint32_t chunkId = 0; chunkId < unit.sequence.chunkCount; ++chunkId) {
                const FwdHChunkSpan chunk = FwdHBuildChunk(unit.sequence, chunkId);
                if (chunkId > 0) {
                    for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
                        const FwdHHeadBinding &head =
                            unit.headRound.heads[aiv_ + 2 * coreHeadId];
                        AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                            FwdHAivLocalFlag(FWD_H_RIGHT_FREE_FLAG, head.localSlot));
                    }
                }
                RunStage1(unit, chunk);
                if (NeedsStage3(chunk)) {
                    RunStage3(unit, chunk);
                }
            }
        }
        const bool lastHasStage2 = args_.tiling.storeFinalState != 0;
        if (lastHasStage2) {
            for (uint32_t coreHeadId = 0; coreHeadId < localHeads; ++coreHeadId) {
                const FwdHHeadBinding &head = unit.headRound.heads[aiv_ + 2 * coreHeadId];
                AscendC::CrossCoreWaitFlag<0x4, PIPE_V>(
                    FwdHAivLocalFlag(FWD_H_RIGHT_FREE_FLAG, head.localSlot));
            }
        }
        // 跨 head_round 前必须把本轮所有 VEC->MTE3 写回收口。AIC 收到 ROUND_DONE 后
        // 才能开始下一轮 kg/H/W 的 MTE2，避免上一轮 MTE3 与下一轮预取跨 round 交叠。
        for (uint32_t slot = 0; slot < ActiveDataSlotCount(localHeads); ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(IoFreeEvent(slot));
        }
        if (hasCubeWork && hasNextWorkUnit) {
            AscendC::CrossCoreSetFlag<0x4, PIPE_MTE3>(FwdHAivLocalFlag(FWD_H_ROUND_DONE_FLAG, 0));
            // DONE 仍由 MTE3 发布以保证写回完成；ACK 用 PIPE_S gate 下一轮全部指令下发。
            AscendC::CrossCoreWaitFlag<0x4, PIPE_S>(FwdHAivLocalFlag(FWD_H_ROUND_ACK_FLAG, 0));
        }
    }

    FwdHKernelArgs args_{};
    uint32_t coreIdx_ = 0;
    uint32_t coreNum_ = 1;
    uint32_t aiv_ = 0;
    // allocator 从 dynamicStartUB 分配 GetRuntimeUBSize()，保留 split-core AIV 的
    // 高端 8 KiB VF 栈，不假设运行时 UB 起始地址为 0。
    AscendC::LocalTensor<uint8_t> ub_{};
};

} // namespace GDN

#endif // ARCH35_CHUNK_FWD_H_VEC_H
