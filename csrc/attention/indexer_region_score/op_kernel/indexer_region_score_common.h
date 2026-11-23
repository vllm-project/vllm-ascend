// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.

// The standalone paged operator shares only the Vector primitives; the guards are kept separate so the full common implementation can live here later.
#ifndef INDEXER_REGION_SCORE_COMMON_VECTOR_H
#define INDEXER_REGION_SCORE_COMMON_VECTOR_H
#include "kernel_operator.h"
namespace IndexerRegionScoreVector {
using namespace AscendC;

constexpr uint32_t H4_HEADS = 4U;
constexpr uint32_t H4_ROWS_PER_AIV = 16U;
constexpr uint32_t H4_VECTOR_LANES = 64U;
constexpr uint32_t H4_NEG_INF_BITS = 0xFF800000U;

__aicore__ inline void WeightedReduceH4Micro(
    const LocalTensor<float> &dst, const LocalTensor<float> &src,
    uint32_t headStride, uint32_t count,
    float w0, float w1, float w2, float w3)
{
    auto dstPtr = reinterpret_cast<__local_mem__ float *>(dst.GetPhyAddr());
    auto srcPtr = reinterpret_cast<__local_mem__ float *>(src.GetPhyAddr());
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(count);
        MicroAPI::RegTensor<float> input;
        MicroAPI::RegTensor<float> weight;
        MicroAPI::RegTensor<float> accum;

        MicroAPI::LoadAlign<float>(input, srcPtr);
        MicroAPI::Duplicate(weight, w0, mask);
        MicroAPI::Mul(accum, input, weight, mask);

        MicroAPI::LoadAlign<float>(input, srcPtr + headStride);
        MicroAPI::Duplicate(weight, w1, mask);
        MicroAPI::MulAddDst(accum, input, weight, mask);
        MicroAPI::LoadAlign<float>(input, srcPtr + 2U * headStride);
        MicroAPI::Duplicate(weight, w2, mask);
        MicroAPI::MulAddDst(accum, input, weight, mask);
        MicroAPI::LoadAlign<float>(input, srcPtr + 3U * headStride);
        MicroAPI::Duplicate(weight, w3, mask);
        MicroAPI::MulAddDst(accum, input, weight, mask);

        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(dstPtr, accum, mask);
    }
}

// For tail tiles the VF broadcasts the FP32 weights directly, skipping the Scalar fetch and parameter packing; the caller owns the UB synchronization.

// Only the causally visible prefix is written; the caller must initialize the masked suffix.
__aicore__ inline void WeightedReduceH4MaskMicro(
    const LocalTensor<float> &dst, const LocalTensor<float> &src,
    uint32_t headStride, uint32_t count,
    float w0, float w1, float w2, float w3,
    int32_t kStart, int32_t visibleLength)
{
    const int32_t visibleInTile = visibleLength - kStart;
    const uint32_t activeCount = visibleInTile <= 0
        ? 0U
        : (static_cast<uint32_t>(visibleInTile) < count
               ? static_cast<uint32_t>(visibleInTile)
               : count);
    if (activeCount == 0U) {
        return;
    }
    WeightedReduceH4Micro(dst, src, headStride, activeCount,
                          w0, w1, w2, w3);
}

template <uint32_t SEGMENTS, uint32_t WEIGHT_ROW_STRIDE>
__simd_callee__ inline void AccumulateH4RemainingHeads(
    __ubuf__ float *src0, __ubuf__ float *src1, __ubuf__ float *weights,
    uint16_t row, uint32_t headStride, MicroAPI::MaskReg &mask,
    MicroAPI::RegTensor<float> &in00, MicroAPI::RegTensor<float> &in01,
    MicroAPI::RegTensor<float> &in10, MicroAPI::RegTensor<float> &in11,
    MicroAPI::RegTensor<float> &wt0, MicroAPI::RegTensor<float> &wt1,
    MicroAPI::RegTensor<float> &acc00, MicroAPI::RegTensor<float> &acc01,
    MicroAPI::RegTensor<float> &acc10, MicroAPI::RegTensor<float> &acc11)
{
    for (uint16_t head = 1U; head < H4_HEADS; ++head) {
        MicroAPI::LoadAlign(in00, src0 + head * headStride);
        MicroAPI::LoadAlign(in10, src1 + head * headStride);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::LoadAlign(in01, src0 + head * headStride + H4_VECTOR_LANES);
            MicroAPI::LoadAlign(in11, src1 + head * headStride + H4_VECTOR_LANES);
        }
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(
            wt0, weights + row * WEIGHT_ROW_STRIDE + head);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(
            wt1, weights + (row + 1U) * WEIGHT_ROW_STRIDE + head);
        MicroAPI::MulAddDst(acc00, in00, wt0, mask);
        MicroAPI::MulAddDst(acc10, in10, wt1, mask);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::MulAddDst(acc01, in01, wt0, mask);
            MicroAPI::MulAddDst(acc11, in11, wt1, mask);
        }
    }
}

template <uint32_t SEGMENTS, uint32_t WEIGHT_ROW_STRIDE = H4_HEADS>
__simd_vf__ inline void WeightedReduceH4RowsVFImpl(
    __ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *weights,
    uint32_t headStride, uint32_t dstRowStride)
{
    static_assert(SEGMENTS == 1U || SEGMENTS == 2U,
                  "regular H4 VF supports one or two 64-lane segments");
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> in00, in01, in10, in11;
    MicroAPI::RegTensor<float> wt0, wt1;
    MicroAPI::RegTensor<float> acc00, acc01, acc10, acc11;

    for (uint16_t row = 0U; row < H4_ROWS_PER_AIV; row += 2U) {
        __ubuf__ float *src0 = src + row * H4_HEADS * headStride;
        __ubuf__ float *src1 = src0 + H4_HEADS * headStride;
        MicroAPI::LoadAlign(in00, src0);
        MicroAPI::LoadAlign(in10, src1);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::LoadAlign(in01, src0 + H4_VECTOR_LANES);
            MicroAPI::LoadAlign(in11, src1 + H4_VECTOR_LANES);
        }
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(
            wt0, weights + row * WEIGHT_ROW_STRIDE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(
            wt1, weights + (row + 1U) * WEIGHT_ROW_STRIDE);
        MicroAPI::Mul(acc00, in00, wt0, mask);
        MicroAPI::Mul(acc10, in10, wt1, mask);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::Mul(acc01, in01, wt0, mask);
            MicroAPI::Mul(acc11, in11, wt1, mask);
        }

        AccumulateH4RemainingHeads<SEGMENTS, WEIGHT_ROW_STRIDE>(
            src0, src1, weights, row, headStride, mask,
            in00, in01, in10, in11, wt0, wt1, acc00, acc01, acc10, acc11);

        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            dst + row * dstRowStride, acc00, mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            dst + (row + 1U) * dstRowStride, acc10, mask);
        if constexpr (SEGMENTS == 2U) {
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
                dst + row * dstRowStride + H4_VECTOR_LANES, acc01, mask);
            MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
                dst + (row + 1U) * dstRowStride + H4_VECTOR_LANES, acc11, mask);
        }
    }
}

template <uint32_t SEGMENTS, uint32_t WEIGHT_ROW_STRIDE = H4_HEADS>
__aicore__ inline void WeightedReduceH4RowsVF(
    const LocalTensor<float> &dst, const LocalTensor<float> &src,
    const LocalTensor<float> &weights, uint32_t weightOffset,
    uint32_t headStride, uint32_t dstRowStride)
{
    auto dstPtr = reinterpret_cast<__ubuf__ float *>(dst.GetPhyAddr());
    auto srcPtr = reinterpret_cast<__ubuf__ float *>(src.GetPhyAddr());
    auto weightPtr = reinterpret_cast<__ubuf__ float *>(weights.GetPhyAddr());
    asc_vf_call<WeightedReduceH4RowsVFImpl<SEGMENTS, WEIGHT_ROW_STRIDE>>(
        dstPtr, srcPtr, weightPtr + weightOffset, headStride, dstRowStride);
}

} // namespace IndexerRegionScoreVector
#endif // INDEXER_REGION_SCORE_COMMON_VECTOR_H

#ifndef INDEXER_REGION_SCORE_COMMON_VECTOR_ONLY
#ifndef INDEXER_REGION_SCORE_COMMON_H
#define INDEXER_REGION_SCORE_COMMON_H
#include "indexer_region_score_paged_tiling_data.h"
namespace IndexerRegionScorePaged {
using namespace AscendC;

// actualKvLengths holds the valid KV token count including this batch's queries; subtracting the full request's Q count yields the zero-based first position.
__aicore__ inline int64_t RequestFirstTokenPosition(
    int32_t actualKvLength, uint32_t requestBegin, uint32_t requestEnd)
{
    return static_cast<int64_t>(actualKvLength) -
        static_cast<int64_t>(requestEnd - requestBegin);
}

// SDK emits one native kernel variant per query dtype. Q and K share this type.
#ifdef DTYPE_QUERY
using QkInputType = DTYPE_QUERY;
#else
using QkInputType = fp8_e4m3fn_t;
#endif
constexpr uint32_t kQkElementBytes = sizeof(QkInputType);
__aicore__ inline uint32_t QkInputDim(const PagedTilingData &)
{
    return 256U;
}

constexpr uint32_t kPagedKeyMacro = kQkElementBytes == 1U ? 512U : 256U;
constexpr uint32_t kPagedMaxPagesPerMacro = kPagedKeyMacro; // page size S is at least 1
constexpr uint32_t kPagedProxyDim = 256U;
constexpr uint32_t kPagedInputC0 = 32U / kQkElementBytes;
// FP8 NZ uses 32 elements per C0; typed paths derive C0 from the element size.
constexpr uint32_t kPagedFp8C0 = 32U;
constexpr uint32_t kPagedNAlignment = 16U;
constexpr uint32_t kPagedKeyL1Bytes = kPagedKeyMacro * kPagedProxyDim * kQkElementBytes;
constexpr uint32_t kPagedKeyL1Slots = 2U;

// Paged loading and Cube consumption share the L1 double-buffered slots and event lifetimes.
class KeyResidentAdapterBase {
protected:
    enum KeySlotState : uint8_t {
        kKeySlotFree = 0U,
        kKeySlotReady = 1U,
        kKeySlotConsumed = 2U,
        // Release has consumed the last free permit; this state is not reusable.
        kKeySlotDrained = 3U,
    };

public:
    __aicore__ inline void Release()
    {
        if ASCEND_IS_AIC {
            DrainKeyMacroSlots();
            for (uint32_t slot = 0U; slot < kPagedKeyL1Slots; ++slot) {
                pipe_->ReleaseEventID<HardEvent::MTE2_MTE1>(keyReady_[slot]);
                pipe_->ReleaseEventID<HardEvent::MTE1_MTE2>(keyFree_[slot]);
            }
        }
    }

    __aicore__ inline void WaitKeyMacroReady(uint32_t slot)
    {
        if (slot >= kPagedKeyL1Slots || keySlotState_[slot] != kKeySlotReady) {
            return;
        }
        WaitFlag<HardEvent::MTE2_MTE1>(keyReady_[slot]);
        keySlotState_[slot] = kKeySlotConsumed;
    }

    __aicore__ inline void ReleaseKeyMacroSlot(uint32_t slot)
    {
        if (slot >= kPagedKeyL1Slots || keySlotState_[slot] != kKeySlotConsumed) {
            return;
        }
        SetFlag<HardEvent::MTE1_MTE2>(keyFree_[slot]);
        keySlotState_[slot] = kKeySlotFree;
    }

    __aicore__ inline LocalTensor<QkInputType> GetKeyL1(uint32_t slot) const
    {
        return kL1_[slot];
    }

protected:
    __aicore__ inline void InitKeyMacroSlots()
    {
        if ASCEND_IS_AIC {
            pipe_->InitBuffer(kL1PoolBuf_, kPagedKeyL1Slots * kPagedKeyL1Bytes);
            LocalTensor<QkInputType> kL1Pool = kL1PoolBuf_.Get<QkInputType>();
            kL1_[0] = kL1Pool;
            kL1_[1] = kL1Pool[kPagedKeyL1Bytes / kQkElementBytes];
            for (uint32_t slot = 0U; slot < kPagedKeyL1Slots; ++slot) {
                keyReady_[slot] = pipe_->AllocEventID<HardEvent::MTE2_MTE1>();
                keyFree_[slot] = pipe_->AllocEventID<HardEvent::MTE1_MTE2>();
                keySlotState_[slot] = kKeySlotFree;
                // Allow preparing either double-buffered slot first.
                SetFlag<HardEvent::MTE1_MTE2>(keyFree_[slot]);
            }
        }
    }

    __aicore__ inline void DrainKeyMacroSlots()
    {
        for (uint32_t slot = 0U; slot < kPagedKeyL1Slots; ++slot) {
            if (keySlotState_[slot] == kKeySlotReady) {
                WaitKeyMacroReady(slot);
            }
            if (keySlotState_[slot] == kKeySlotConsumed) {
                ReleaseKeyMacroSlot(slot);
            }
            if (keySlotState_[slot] == kKeySlotFree) {
                // Each slot starts with one free permit; the last permits (including those of unused slots) must be drained before releasing the event IDs.
                WaitFlag<HardEvent::MTE1_MTE2>(keyFree_[slot]);
                keySlotState_[slot] = kKeySlotDrained;
            }
        }
    }

    PagedTilingData tiling_{};
    TPipe *pipe_ = nullptr;
    TBuf<TPosition::A1> kL1PoolBuf_;
    LocalTensor<QkInputType> kL1_[kPagedKeyL1Slots];
    TEventID keyReady_[kPagedKeyL1Slots];
    TEventID keyFree_[kPagedKeyL1Slots];
    KeySlotState keySlotState_[kPagedKeyL1Slots];
};
} // namespace IndexerRegionScorePaged

namespace IndexerRegionScorePaged {
using namespace AscendC;

constexpr uint32_t PAGED_H = 4U;
constexpr uint32_t PAGED_D = 256U;
constexpr uint32_t PAGED_MQ_MAX = 32U;
constexpr uint32_t PAGED_Q_MACRO_MAX = 64U;
constexpr uint32_t PAGED_K_MACRO = kPagedKeyMacro;
constexpr uint32_t PAGED_NR = 128U;
constexpr uint32_t PAGED_M_MAX = 128U;
constexpr uint32_t PAGED_Q_ROWS_MAX = 256U;
constexpr uint32_t PAGED_PIPE_SLOTS = 2U;
// 16-bit operands occupy the complete 64 KiB L0A/B. One permit protects the
// shared operand slot; FP8 preserves its original two operand slots.
constexpr uint32_t PAGED_OPERAND_SLOTS = kQkElementBytes == 1U ? 2U : 1U;
static_assert(kQkElementBytes == 1U || kQkElementBytes == 2U, "unsupported Q/K element width");
static_assert(kPagedKeyL1Slots * kPagedKeyL1Bytes +
    PAGED_PIPE_SLOTS * PAGED_Q_ROWS_MAX * PAGED_D * kQkElementBytes <= 512U * 1024U,
    "Q/K resident buffers exceed A5 L1 capacity");
static_assert(PAGED_OPERAND_SLOTS * PAGED_M_MAX * PAGED_D * kQkElementBytes <= 64U * 1024U,
    "Q operand exceeds A5 L0A capacity");
static_assert(PAGED_OPERAND_SLOTS * PAGED_NR * PAGED_D * kQkElementBytes <= 64U * 1024U,
    "K operand exceeds A5 L0B capacity");
constexpr uint32_t PAGED_QUERIES_PER_AIV_MAX = PAGED_MQ_MAX / 2U;
constexpr uint32_t PAGED_RESULT_SLOT_ELEMENTS = PAGED_M_MAX * PAGED_NR;
constexpr uint32_t PAGED_OUTPUT_SLOT_ELEMENTS = PAGED_QUERIES_PER_AIV_MAX * PAGED_NR;
// An aligned H4 weight row occupies 128 FP32 elements, preserving the original pipeline's UB budget.
constexpr uint32_t PAGED_WEIGHT_FP32_ELEMENTS = 144U;
constexpr uint32_t PAGED_H4_WEIGHT_ROW_STRIDE = 32U / sizeof(float);
constexpr uint32_t PAGED_SYNC_MODE = 4U;
constexpr uint32_t PAGED_C_FREE_EVENT = 0U;
constexpr uint32_t PAGED_C_READY_EVENT = 2U;
constexpr uint32_t PAGED_WORK_DRAINED_EVENT = 10U;
constexpr uint32_t PAGED_AIV1_EVENT_OFFSET = 16U;
constexpr FixpipeConfig PAGED_FIX_TO_UB_ROW_MAJOR = {CO2Layout::ROW_MAJOR, true};

__aicore__ inline uint32_t PagedMin(uint32_t left, uint32_t right)
{
    return left < right ? left : right;
}

__aicore__ inline uint32_t PagedMax(uint32_t left, uint32_t right)
{
    return left > right ? left : right;
}

__aicore__ inline uint32_t PagedAlignUp(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

template <typename KeyAdapter, bool FOUR_HEADS = true>
class KeyResidentCompute final {
public:
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProcessCube();
        } else {
            ProcessVector();
        }
    }

    __aicore__ inline void Init(GM_ADDR indexQ, GM_ADDR weights,
                                GM_ADDR actualKvLengths, GM_ADDR output,
                                const PagedTilingData *tiling, TPipe *pipe,
                                KeyAdapter *keyAdapter)
    {
        tiling_ = *tiling;
        requestKCarry_ = 0U;
        pipe_ = pipe;
        keyAdapter_ = keyAdapter;
        indexQGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QkInputType *>(indexQ));
        weightsFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weights));
        weightsBf16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(weights));
        weightsFp16Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weights));
        actualKvLengthsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(actualKvLengths));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));

        if ASCEND_IS_AIC {
            InitCubeBuffers();
        } else {
            InitVectorBuffers();
        }
    }

    __aicore__ inline void Release()
    {
        if ASCEND_IS_AIC {
            ReleaseCubeEvents();
        } else {
            DrainOutputStores();
            pipe_->ReleaseEventID<HardEvent::MTE2_S>(mte2ToScalar_);
            pipe_->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVector_);
            pipe_->ReleaseEventID<HardEvent::V_S>(vectorToScalar_);
            pipe_->ReleaseEventID<HardEvent::V_MTE2>(vectorToMte2_);
            pipe_->ReleaseEventID<HardEvent::V_MTE3>(vectorToMte3_);
            for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
                pipe_->ReleaseEventID<HardEvent::MTE3_V>(outputStoreDone_[slot]);
            }
        }
    }

private:
    __aicore__ inline void InitCubeBuffers()
    {
        SetMMRowMajor();
        pipe_->InitBuffer(qL1Buf_, PAGED_PIPE_SLOTS * PAGED_Q_ROWS_MAX * PAGED_D * kQkElementBytes);
        pipe_->InitBuffer(qL0Buf_, PAGED_OPERAND_SLOTS * PAGED_M_MAX * PAGED_D * kQkElementBytes);
        pipe_->InitBuffer(kL0Buf_, PAGED_OPERAND_SLOTS * PAGED_NR * PAGED_D * kQkElementBytes);
        pipe_->InitBuffer(cL0Buf_,
                          PAGED_PIPE_SLOTS * PAGED_M_MAX * PAGED_NR * sizeof(float));
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            pipe_->InitBuffer(mmOutBuf_[slot],
                              PAGED_RESULT_SLOT_ELEMENTS * sizeof(float));
            mmOut_[slot] = mmOutBuf_[slot].Get<float>();
        }

        LocalTensor<QkInputType> qL1Pool = qL1Buf_.Get<QkInputType>();
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            qL1_[slot] = qL1Pool[slot * PAGED_Q_ROWS_MAX * PAGED_D];
            qReady_[slot] = pipe_->AllocEventID<HardEvent::MTE2_MTE1>();
            qFree_[slot] = pipe_->AllocEventID<HardEvent::MTE1_MTE2>();
            if (slot < PAGED_OPERAND_SLOTS) {
                mToMte1_[slot] = pipe_->AllocEventID<HardEvent::M_MTE1>();
                SetFlag<HardEvent::M_MTE1>(mToMte1_[slot]);
            }
            fixToM_[slot] = pipe_->AllocEventID<HardEvent::FIX_M>();
            SetFlag<HardEvent::MTE1_MTE2>(qFree_[slot]);
            SetFlag<HardEvent::FIX_M>(fixToM_[slot]);
        }
        qL0_ = qL0Buf_.Get<QkInputType>();
        kL0_ = kL0Buf_.Get<QkInputType>();
        cL0_ = cL0Buf_.Get<float>();
        mte1ToM_ = pipe_->AllocEventID<HardEvent::MTE1_M>();
        mToFix_ = pipe_->AllocEventID<HardEvent::M_FIX>();
    }

    __aicore__ inline void InitVectorBuffers()
    {
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            pipe_->InitBuffer(mmOutBuf_[slot],
                              PAGED_RESULT_SLOT_ELEMENTS * sizeof(float));
            mmOut_[slot] = mmOutBuf_[slot].Get<float>();
        }
        pipe_->InitBuffer(accBuf_,
                          PAGED_PIPE_SLOTS * PAGED_OUTPUT_SLOT_ELEMENTS * sizeof(float));
        // Strided GM rows are moved into individually 32-byte-aligned UB rows.
        pipe_->InitBuffer(weightUbBuf_, PAGED_WEIGHT_FP32_ELEMENTS * sizeof(float));
        pipe_->InitBuffer(weightInputUbBuf_, PAGED_M_MAX * sizeof(float));
        pipe_->InitBuffer(cuSeqLensUbBuf_, 2U * sizeof(int32_t));
        pipe_->InitBuffer(visibleUbBuf_, PAGED_QUERIES_PER_AIV_MAX * sizeof(int32_t));
        acc_ = accBuf_.Get<float>();
        weightUb_ = weightUbBuf_.Get<float>();
        weightInputBf16Ub_ = weightInputUbBuf_.Get<bfloat16_t>();
        weightInputFp16Ub_ = weightInputUbBuf_.Get<half>();
        cuSeqLensUb_ = cuSeqLensUbBuf_.Get<int32_t>();
        visibleUb_ = visibleUbBuf_.Get<int32_t>();
        mte2ToScalar_ = pipe_->AllocEventID<HardEvent::MTE2_S>();
        mte2ToVector_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        vectorToScalar_ = pipe_->AllocEventID<HardEvent::V_S>();
        vectorToMte2_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        vectorToMte3_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            outputStoreDone_[slot] = pipe_->AllocEventID<HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline LocalTensor<float> MmOut(uint32_t flatOffset)
    {
        const uint32_t slot = flatOffset / PAGED_RESULT_SLOT_ELEMENTS;
        return mmOut_[slot][flatOffset - slot * PAGED_RESULT_SLOT_ELEMENTS];
    }

    __aicore__ inline void ReleaseCubeEvents()
    {
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            WaitFlag<HardEvent::MTE1_MTE2>(qFree_[slot]);
            if (slot < PAGED_OPERAND_SLOTS) {
                WaitFlag<HardEvent::M_MTE1>(mToMte1_[slot]);
                pipe_->ReleaseEventID<HardEvent::M_MTE1>(mToMte1_[slot]);
            }
            WaitFlag<HardEvent::FIX_M>(fixToM_[slot]);
            pipe_->ReleaseEventID<HardEvent::MTE2_MTE1>(qReady_[slot]);
            pipe_->ReleaseEventID<HardEvent::MTE1_MTE2>(qFree_[slot]);
            pipe_->ReleaseEventID<HardEvent::FIX_M>(fixToM_[slot]);
        }
        pipe_->ReleaseEventID<HardEvent::MTE1_M>(mte1ToM_);
        pipe_->ReleaseEventID<HardEvent::M_FIX>(mToFix_);
        SetMMColumnMajor();
    }

    __aicore__ inline uint32_t PairCore() const
    {
        uint32_t pair = GetBlockIdx();
        if ASCEND_IS_AIV {
            pair /= GetTaskRation();
        }
        return pair;
    }

    __aicore__ inline uint32_t Heads() const
    {
        return FOUR_HEADS ? PAGED_H : tiling_.headsPerGroup;
    }

    __aicore__ inline uint32_t WeightFp32RowStride() const
    {
        return PagedAlignUp(Heads() * sizeof(float), 32U) / sizeof(float);
    }

    __aicore__ inline uint32_t WeightNativeRowStride() const
    {
        return PagedAlignUp(Heads() * sizeof(half), 32U) / sizeof(half);
    }

    __aicore__ inline uint32_t QMicroTile() const
    {
        return tiling_.qTile;
    }

    __aicore__ inline uint32_t QMacroTile() const
    {
        return (QMicroTile() * Heads()) % 16U == 0U
                   ? 2U * QMicroTile()
                   : QMicroTile();
    }

    __aicore__ inline void GetLocalQueryRange(uint32_t qCount, uint32_t subBlock,
                                               uint32_t &localQStart,
                                               uint32_t &localQCount) const
    {
        const uint32_t heads = Heads();
        const uint32_t queryAlignment = heads % 16U == 0U ? 1U :
                                        heads % 8U == 0U ? 2U : 4U;
        const uint32_t capacity = (PAGED_M_MAX / 2U) / heads;
        const uint32_t alignedCapacity = capacity / queryAlignment * queryAlignment;
        const uint32_t firstCount = qCount * heads <= PAGED_M_MAX / 2U || alignedCapacity == 0U
                                        ? qCount
                                        : PagedMin(qCount, alignedCapacity);
        localQStart = subBlock == 0U ? 0U : firstCount;
        localQCount = subBlock == 0U ? firstCount : qCount - firstCount;
    }

    __aicore__ inline void GetCoreWorkRange(uint32_t &qBegin, uint32_t &qEnd,
                                            uint32_t &kBegin, uint32_t &kEnd) const
    {
        const uint32_t core = PairCore();
        const uint32_t rangeUnit = tiling_.qTile;
        const uint32_t tokens = tiling_.batchTokens != 0U ? tiling_.batchCount : tiling_.totalTokens;
        const uint32_t totalQRanges =
            (tokens + rangeUnit - 1U) / rangeUnit;
        const uint32_t totalKRanges =
            (tiling_.seqK + tiling_.kTile - 1U) / tiling_.kTile;
        const uint32_t qShard = core / tiling_.kShardCount;
        const uint32_t kShard = core % tiling_.kShardCount;

        const uint32_t qRangesPerShard = totalQRanges / tiling_.qShardCount;
        const uint32_t qExtra = totalQRanges % tiling_.qShardCount;
        const uint32_t firstQRange =
            qShard * qRangesPerShard + PagedMin(qShard, qExtra);
        const uint32_t localQRanges =
            qRangesPerShard + (qShard < qExtra ? 1U : 0U);
        qBegin = tiling_.batchBegin + firstQRange * rangeUnit;
        qEnd = tiling_.batchBegin + PagedMin(tokens,
                        (firstQRange + localQRanges) * rangeUnit);

        const uint32_t kRangesPerShard = totalKRanges / tiling_.kShardCount;
        const uint32_t kExtra = totalKRanges % tiling_.kShardCount;
        const uint32_t firstKRange =
            kShard * kRangesPerShard + PagedMin(kShard, kExtra);
        const uint32_t localKRanges =
            kRangesPerShard + (kShard < kExtra ? 1U : 0U);
        kBegin = firstKRange * tiling_.kTile;
        kEnd = PagedMin(tiling_.seqK,
                        (firstKRange + localKRanges) * tiling_.kTile);
    }

    __aicore__ inline bool IntersectQueryRange(uint32_t requestBegin, uint32_t requestEnd,
                                                uint32_t coreBegin, uint32_t coreEnd,
                                                uint32_t &begin, uint32_t &end) const
    {
        begin = PagedMax(requestBegin, coreBegin);
        end = PagedMin(requestEnd, coreEnd);
        return begin < end;
    }

    __aicore__ inline void GatherQueryToL1(uint32_t slot, uint32_t group,
                                           uint32_t qStart, uint32_t qCount,
                                           uint32_t qRowsAlign)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(qFree_[slot]);
        // Treat each contiguous [H,D] slice of Q as an ND matrix and assemble them into the same NZ matrix as the contiguous cache.
        Nd2NzParams nd2nz;
        nd2nz.ndNum = qCount;
        nd2nz.nValue = Heads();
        nd2nz.dValue = QkInputDim(tiling_);
        nd2nz.srcNdMatrixStride = tiling_.qStrideToken;
        nd2nz.srcDValue = QkInputDim(tiling_);
        nd2nz.dstNzC0Stride = qRowsAlign;
        nd2nz.dstNzNStride = 1U;
        nd2nz.dstNzMatrixStride = Heads() * kPagedInputC0;
        const uint64_t sourceOffset = static_cast<uint64_t>(qStart) * tiling_.qStrideToken +
                                      static_cast<uint64_t>(group) * tiling_.qStrideGroup;
        DataCopy(qL1_[slot], indexQGm_[sourceOffset], nd2nz);
        SetFlag<HardEvent::MTE2_MTE1>(qReady_[slot]);
    }

    __aicore__ inline void GatherFlatQueryMacro(uint64_t flatQ,
                                                uint32_t queryMacrosPerGroup,
                                                uint32_t qBegin, uint32_t qEnd)
    {
        const uint32_t group = static_cast<uint32_t>(flatQ / queryMacrosPerGroup);
        const uint32_t queryInGroup = static_cast<uint32_t>(flatQ % queryMacrosPerGroup);
        const uint32_t qStart = qBegin + queryInGroup * QMacroTile();
        const uint32_t qCount = PagedMin(QMacroTile(), qEnd - qStart);
        const uint32_t qRowsAlign = PagedAlignUp(qCount * Heads(), 16U);
        const uint32_t slot = static_cast<uint32_t>(flatQ % PAGED_PIPE_SLOTS);

        GatherQueryToL1(slot, group, qStart, qCount, qRowsAlign);
    }

    __aicore__ inline void LoadOperandsToL0(LocalTensor<QkInputType> keyL1,
                                             uint32_t qRowsAlign, uint32_t kRowsAlign,
                                             uint32_t qRowStart, uint32_t kRowStart,
                                             uint32_t mAlign, uint32_t nAlign,
                                             uint32_t resultSlot, uint32_t querySlot)
    {
        WaitFlag<HardEvent::M_MTE1>(mToMte1_[resultSlot % PAGED_OPERAND_SLOTS]);

        LoadData2DParamsV2 qParams;
        qParams.mStartPosition = qRowStart / 16U;
        qParams.kStartPosition = 0U;
        qParams.mStep = mAlign / 16U;
        qParams.kStep = QkInputDim(tiling_) / kPagedInputC0;
        qParams.srcStride = qRowsAlign / 16U;
        qParams.dstStride = mAlign / 16U;
        qParams.ifTranspose = false;
        LoadData(qL0_[(resultSlot % PAGED_OPERAND_SLOTS) * PAGED_M_MAX * PAGED_D], qL1_[querySlot], qParams);

        LoadData2DParamsV2 kParams;
        kParams.mStartPosition = kRowStart / 16U;
        kParams.kStartPosition = 0U;
        kParams.mStep = nAlign / 16U;
        kParams.kStep = QkInputDim(tiling_) / kPagedInputC0;
        kParams.srcStride = kRowsAlign / 16U;
        kParams.dstStride = nAlign / 16U;
        kParams.ifTranspose = false;
        LoadData(kL0_[(resultSlot % PAGED_OPERAND_SLOTS) * PAGED_NR * PAGED_D], keyL1, kParams);

        SetFlag<HardEvent::MTE1_M>(mte1ToM_);
        WaitFlag<HardEvent::MTE1_M>(mte1ToM_);
    }

    __aicore__ inline void RunTileMmad(
        uint32_t mAlign, uint32_t nAlign, uint32_t resultSlot)
    {
        WaitFlag<HardEvent::FIX_M>(fixToM_[resultSlot]);
        MmadParams mm;
        mm.m = mAlign;
        mm.n = nAlign;
        mm.k = QkInputDim(tiling_);
        mm.cmatrixInitVal = true;
        mm.cmatrixSource = false;
        Mmad(cL0_[resultSlot * PAGED_M_MAX * PAGED_NR],
             qL0_[(resultSlot % PAGED_OPERAND_SLOTS) * PAGED_M_MAX * PAGED_D],
             kL0_[(resultSlot % PAGED_OPERAND_SLOTS) * PAGED_NR * PAGED_D], mm);
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(mToMte1_[resultSlot % PAGED_OPERAND_SLOTS]);
        SetFlag<HardEvent::M_FIX>(mToFix_);
        WaitFlag<HardEvent::M_FIX>(mToFix_);
    }

    __aicore__ inline void ConfigureCubeFixpipe(
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> &fp,
        uint32_t mAlign, uint32_t nAlign, uint32_t qCount,
        bool &singleAiv, uint32_t &firstRows, uint32_t &secondRows)
    {
        fp.nSize = nAlign;
        const uint32_t totalRows = mAlign;
        const uint32_t heads = Heads();
        const uint32_t queryAlignment = heads % 16U == 0U ? 1U :
                                        heads % 8U == 0U ? 2U : 4U;
        const uint32_t alignedCapacity = ((PAGED_M_MAX / 2U) / heads) /
                                         queryAlignment * queryAlignment;
        singleAiv = totalRows <= PAGED_M_MAX / 2U || alignedCapacity == 0U;
        firstRows = singleAiv
                        ? PagedMin(PAGED_M_MAX / 2U, totalRows)
                        : PagedMin(qCount, alignedCapacity) * heads;
        secondRows = totalRows - firstRows;
        fp.mSize = firstRows;
        fp.srcStride = mAlign;
        fp.dstStride = PAGED_NR;
        fp.params.ndNum = 1U;
        fp.params.srcNdStride = 0U;
        fp.params.dstNdStride = 0U;
        fp.dualDstCtl = 0U;
        fp.reluEn = true;
        fp.subBlockId = 0U;
    }

    __aicore__ inline void PublishCubeResult(
        uint32_t mAlign, uint32_t nAlign, uint32_t qCount, uint32_t kCount,
        uint32_t resultSlot)
    {
        CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_C_FREE_EVENT + resultSlot);

        CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_FIX>(
            PAGED_C_FREE_EVENT + resultSlot + PAGED_AIV1_EVENT_OFFSET);

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fp;
        bool singleAiv = false;
        uint32_t firstRows = 0U;
        uint32_t secondRows = 0U;
        ConfigureCubeFixpipe(fp, mAlign, nAlign, qCount, singleAiv, firstRows, secondRows);
        Fixpipe<float, float, PAGED_FIX_TO_UB_ROW_MAJOR>(
            MmOut(resultSlot * PAGED_RESULT_SLOT_ELEMENTS),
            cL0_[resultSlot * PAGED_M_MAX * PAGED_NR], fp);
        if (!singleAiv) {
            CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_C_READY_EVENT + resultSlot);
        }
        if (secondRows != 0U) {
            fp.mSize = secondRows;
            fp.subBlockId = singleAiv ? 0U : 1U;
            Fixpipe<float, float, PAGED_FIX_TO_UB_ROW_MAJOR>(
                singleAiv
                    ? MmOut(resultSlot * PAGED_RESULT_SLOT_ELEMENTS + firstRows * PAGED_NR)
                    : MmOut(resultSlot * PAGED_RESULT_SLOT_ELEMENTS),
                cL0_[resultSlot * PAGED_M_MAX * PAGED_NR + firstRows * 16U], fp);
        }
        if (singleAiv) {
            CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_C_READY_EVENT + resultSlot);
        }
        CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(
            PAGED_C_READY_EVENT + resultSlot + PAGED_AIV1_EVENT_OFFSET);

        SetFlag<HardEvent::FIX_M>(fixToM_[resultSlot]);
    }

    __aicore__ inline void ComputeOneTile(LocalTensor<QkInputType> keyL1,
                                           uint32_t qRowsAlign, uint32_t kRowsAlign,
                                           uint32_t qRowStart, uint32_t kRowStart,
                                           uint32_t qCount, uint32_t kCount,
                                           uint32_t resultSlot, uint32_t querySlot,
                                           bool fullyMasked)
    {
        // A fully masked tile still follows the two-AIV free/ready protocol; the AIV only writes negative infinity.
        if (fullyMasked) {
            CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_C_FREE_EVENT + resultSlot);
            CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_FIX>(
                PAGED_C_FREE_EVENT + resultSlot + PAGED_AIV1_EVENT_OFFSET);
            CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_C_READY_EVENT + resultSlot);
            CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(
                PAGED_C_READY_EVENT + resultSlot + PAGED_AIV1_EVENT_OFFSET);
            return;
        }
        const uint32_t mAlign = PagedAlignUp(qCount * Heads(), 16U);
        const uint32_t nAlign = PagedAlignUp(kCount, 16U);
        LoadOperandsToL0(keyL1, qRowsAlign, kRowsAlign, qRowStart, kRowStart,
                         mAlign, nAlign, resultSlot, querySlot);

        RunTileMmad(mAlign, nAlign, resultSlot);

        PublishCubeResult(mAlign, nAlign, qCount, kCount, resultSlot);
    }

    __aicore__ inline int32_t VisibleRegions(uint32_t request,
                                              uint32_t requestBegin,
                                              uint32_t token) const
    {
        const int64_t position = requestFirstPosition_ +
            static_cast<int64_t>(token - requestBegin);
        if (position <= 0) {
            return 0;
        }
        const uint64_t completed = static_cast<uint64_t>(position) / tiling_.regionSize;
        return static_cast<int32_t>(
            completed > tiling_.seqK ? tiling_.seqK : completed);
    }

    __aicore__ inline int32_t GetMaxVisibleLength(uint32_t request,
                                                   uint32_t requestBegin,
                                                   uint32_t qStart,
                                                   uint32_t qCount) const
    {
        return VisibleRegions(request, requestBegin, qStart + qCount - 1U);
    }

    __aicore__ inline bool GetRequestKeyRange(
        uint32_t request, uint32_t requestBegin, uint32_t qEnd,
        uint32_t &kBegin, uint32_t &kEnd)
    {

        // Both AIVs use the pair's final Q token so producer and consumers
        // traverse exactly the same K macros. TopK only reads visible scores.
        const uint32_t visible = static_cast<uint32_t>(
            VisibleRegions(request, requestBegin, qEnd - 1U));
        const uint32_t macroCount =
            (visible + PAGED_K_MACRO - 1U) / PAGED_K_MACRO;
        // Keep the original shard range when every K macro is needed.
        if (macroCount == (tiling_.seqK + PAGED_K_MACRO - 1U) / PAGED_K_MACRO) {
            return kBegin < kEnd;
        }
        const uint32_t physicalKShard = PairCore() % tiling_.kShardCount;
        const uint32_t kShard = (physicalKShard + tiling_.kShardCount - requestKCarry_) %
            tiling_.kShardCount;
        const uint32_t macrosPerShard = macroCount / tiling_.kShardCount;
        const uint32_t extra = macroCount % tiling_.kShardCount;
        // Every pair participant advances before an empty local range exits.
        requestKCarry_ = (requestKCarry_ + extra) % tiling_.kShardCount;
        const uint32_t firstMacro =
            kShard * macrosPerShard + PagedMin(kShard, extra);
        const uint32_t localMacros =
            macrosPerShard + (kShard < extra ? 1U : 0U);
        kBegin = firstMacro * PAGED_K_MACRO;
        // Retain the existing last-macro mask handling and avoid tail
        // work beyond it, even when the output capacity is much larger.
        kEnd = PagedMin(tiling_.seqK,
            (firstMacro + localMacros) * PAGED_K_MACRO);
        return kBegin < kEnd;

    }


    __aicore__ inline void PrepareNextKeyMacro(
        uint32_t request, uint32_t kStart, uint32_t coreKEnd,
        uint32_t keyMacroIndex, bool &nextKeyPrepared)
    {
        if (!nextKeyPrepared && kStart + PAGED_K_MACRO < coreKEnd) {
            const uint32_t nextKStart = kStart + PAGED_K_MACRO;
            const uint32_t nextKCount =
                PagedMin(PAGED_K_MACRO, coreKEnd - nextKStart);
            const uint32_t nextKeySlot =
                (keyMacroIndex + 1U) % kPagedKeyL1Slots;
            keyAdapter_->PrepareKeyMacroToL1(
                request, nextKStart, nextKCount, nextKeySlot,
                PagedAlignUp(nextKCount, kPagedNAlignment));
            nextKeyPrepared = true;
        }
    }

    __aicore__ inline void ComputeCubeQueryMicro(
        LocalTensor<QkInputType> keyL1, uint32_t qRowsAlign, uint32_t kRowsAlign,
        uint32_t qMicro, uint32_t qMicroCount, uint32_t request, uint32_t group,
        uint32_t qStart, uint32_t kStart, uint32_t kMacroCount, uint32_t querySlot,
        int32_t maxVisible, uint32_t &streamLoop, bool &nextKeyPrepared,
        uint32_t coreKEnd, uint32_t keyMacroIndex)
    {
        for (uint32_t kMicro = 0U; kMicro < kMacroCount;
             kMicro += PAGED_NR) {
            const uint32_t kMicroCount =
                PagedMin(PAGED_NR, kMacroCount - kMicro);
            const uint32_t resultSlot = streamLoop % PAGED_PIPE_SLOTS;
            bool fullyMasked = false;

                fullyMasked = qMicroCount == QMicroTile() &&
                    kMicroCount == PAGED_NR &&
                    static_cast<int32_t>(kStart + kMicro) >= maxVisible;

            ComputeOneTile(keyL1, qRowsAlign, kRowsAlign,
                           qMicro * Heads(), kMicro,
                           qMicroCount, kMicroCount,
                           resultSlot, querySlot, fullyMasked);
            ++streamLoop;

            PrepareNextKeyMacro(request, kStart, coreKEnd, keyMacroIndex, nextKeyPrepared);
        }
    }

    __aicore__ inline void ComputeCubeQueryMacro(
        LocalTensor<QkInputType> keyL1, uint32_t qRowsAlign, uint32_t kRowsAlign,
        uint32_t qMacroCount, uint32_t request, uint32_t requestQBegin, uint32_t group,
        uint32_t qStart, uint32_t kStart, uint32_t kMacroCount, uint32_t querySlot,
        uint32_t &streamLoop, bool &nextKeyPrepared, uint32_t coreKEnd, uint32_t keyMacroIndex)
    {
        for (uint32_t qMicro = 0U; qMicro < qMacroCount; qMicro += QMicroTile()) {
            const uint32_t qMicroCount =
                PagedMin(QMicroTile(), qMacroCount - qMicro);
            int32_t maxVisible = 0;

                if (qMicroCount == QMicroTile()) {
                    maxVisible = GetMaxVisibleLength(
                        request, requestQBegin,
                        qStart + qMicro, qMicroCount);
                }

            ComputeCubeQueryMicro(keyL1, qRowsAlign, kRowsAlign, qMicro, qMicroCount,
                                request, group, qStart, kStart, kMacroCount, querySlot,
                                maxVisible, streamLoop, nextKeyPrepared, coreKEnd, keyMacroIndex);
        }
    }

    __aicore__ inline void ConsumeCubeKeyMacro(
        LocalTensor<QkInputType> keyL1, uint32_t kRowsAlign, uint64_t flatQueryMacros,
        uint32_t queryMacrosPerGroup, uint32_t qBegin, uint32_t qEnd, uint32_t request,
        uint32_t requestQBegin, uint32_t kStart, uint32_t kMacroCount, uint32_t coreKEnd,
        uint32_t keyMacroIndex, uint32_t &streamLoop, bool &nextKeyPrepared)
    {
        for (uint64_t flatQ = 0U; flatQ < flatQueryMacros; ++flatQ) {
            const uint32_t group =
                static_cast<uint32_t>(flatQ / queryMacrosPerGroup);
            const uint32_t queryInGroup =
                static_cast<uint32_t>(flatQ % queryMacrosPerGroup);
            const uint32_t qStart = qBegin + queryInGroup * QMacroTile();
            const uint32_t qMacroCount = PagedMin(QMacroTile(), qEnd - qStart);
            const uint32_t querySlot =
                static_cast<uint32_t>(flatQ % PAGED_PIPE_SLOTS);
            const uint32_t qRowsAlign = PagedAlignUp(qMacroCount * Heads(), 16U);
            WaitFlag<HardEvent::MTE2_MTE1>(qReady_[querySlot]);

            if (flatQ + 1U < flatQueryMacros) {
                GatherFlatQueryMacro(flatQ + 1U, queryMacrosPerGroup, qBegin, qEnd);
            }
            ComputeCubeQueryMacro(keyL1, qRowsAlign, kRowsAlign, qMacroCount,
                                request, requestQBegin, group, qStart, kStart, kMacroCount,
                                querySlot, streamLoop, nextKeyPrepared, coreKEnd, keyMacroIndex);
            // The producer may reuse a Q slot only after all LoadData operations of the current macro block complete.
            SetFlag<HardEvent::MTE1_MTE2>(qFree_[querySlot]);
        }
    }

    __aicore__ inline void ProcessCubeRequestKeys(
        uint32_t request, uint32_t requestQBegin, uint32_t qBegin, uint32_t qEnd,
        uint32_t queryMacrosPerGroup, uint64_t flatQueryMacros,
        uint32_t coreKBegin, uint32_t coreKEnd, uint32_t firstKeyMacroIndex, uint32_t &streamLoop)
    {
        for (uint32_t kStart = coreKBegin, keyMacroIndex = firstKeyMacroIndex;
             kStart < coreKEnd; kStart += PAGED_K_MACRO, ++keyMacroIndex) {
            const uint32_t kMacroCount =
                PagedMin(PAGED_K_MACRO, coreKEnd - kStart);
            const uint32_t keySlot = keyMacroIndex % kPagedKeyL1Slots;
            const uint32_t kRowsAlign = PagedAlignUp(kMacroCount, kPagedNAlignment);
            keyAdapter_->WaitKeyMacroReady(keySlot);

            LocalTensor<QkInputType> keyL1 = keyAdapter_->GetKeyL1(keySlot);

            GatherFlatQueryMacro(0U, queryMacrosPerGroup, qBegin, qEnd);
            bool nextKeyPrepared = false;
            ConsumeCubeKeyMacro(keyL1, kRowsAlign, flatQueryMacros, queryMacrosPerGroup,
                                   qBegin, qEnd, request, requestQBegin, kStart, kMacroCount,
                                   coreKEnd, keyMacroIndex, streamLoop, nextKeyPrepared);
            // The current K macro block must stay resident until all groups and Q macro blocks of the request finish consuming it.
            keyAdapter_->ReleaseKeyMacroSlot(keySlot);
        }
    }

    __aicore__ inline void ProcessCubeRequests(
        uint32_t coreQBegin, uint32_t coreQEnd, uint32_t coreKBegin, uint32_t coreKEnd,
        uint32_t &streamLoop)
    {
        for (uint32_t request = 0U; request < tiling_.numRequests; ++request) {
            uint32_t requestQBegin = 0U;
            uint32_t requestQEnd = 0U;
            if (!LoadRequestSpan(request, requestQBegin, requestQEnd)) {
                continue;
            }
            uint32_t qBegin = 0U;
            uint32_t qEnd = 0U;
            if (!IntersectQueryRange(requestQBegin, requestQEnd,
                                     coreQBegin, coreQEnd, qBegin, qEnd)) {
                continue;
            }
            uint32_t requestKBegin = coreKBegin;
            uint32_t requestKEnd = coreKEnd;
            if (!GetRequestKeyRange(request, requestQBegin, qEnd,
                                          requestKBegin, requestKEnd)) {
                continue;
            }
            const uint32_t queryMacrosPerGroup =
                (qEnd - qBegin + QMacroTile() - 1U) / QMacroTile();
            const uint64_t flatQueryMacros =
                static_cast<uint64_t>(tiling_.groups) * queryMacrosPerGroup;
            if (flatQueryMacros == 0U) {
                continue;
            }

            const uint32_t firstKeyMacroIndex = requestKBegin / PAGED_K_MACRO;
            keyAdapter_->PrepareKeyMacroToL1(
                request, requestKBegin, PagedMin(PAGED_K_MACRO, requestKEnd - requestKBegin),
                firstKeyMacroIndex % kPagedKeyL1Slots,
                PagedAlignUp(PagedMin(PAGED_K_MACRO, requestKEnd - requestKBegin),
                                  kPagedNAlignment));

            ProcessCubeRequestKeys(request, requestQBegin, qBegin, qEnd,
                                          queryMacrosPerGroup, flatQueryMacros,
                                          requestKBegin, requestKEnd, firstKeyMacroIndex, streamLoop);
        }
    }

    __aicore__ inline void ProcessCube()
    {
        uint32_t streamLoop = 0U;
        uint32_t coreQBegin = 0U;
        uint32_t coreQEnd = 0U;
        uint32_t coreKBegin = 0U;
        uint32_t coreKEnd = 0U;
        GetCoreWorkRange(coreQBegin, coreQEnd, coreKBegin, coreKEnd);
        ProcessCubeRequests(coreQBegin, coreQEnd, coreKBegin, coreKEnd, streamLoop);

        // Drain the result-slot permits of both AIVs and close out the last-generation permits of unused slots.
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_C_FREE_EVENT + slot);
            CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_FIX>(
                PAGED_C_FREE_EVENT + slot + PAGED_AIV1_EVENT_OFFSET);
        }
        CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(PAGED_WORK_DRAINED_EVENT);
        CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_FIX>(
            PAGED_WORK_DRAINED_EVENT + PAGED_AIV1_EVENT_OFFSET);

    }

    __aicore__ inline bool LoadRequestSpan(uint32_t request,
                                            uint32_t &begin, uint32_t &end)
    {
        if (!keyAdapter_->GetRequestTokenSpan(request, begin, end)) {
            return false;
        }

            requestFirstPosition_ = RequestFirstTokenPosition(
                actualKvLengthsGm_.GetValue(request), begin, end);

        return true;
    }

    __aicore__ inline void LoadWeights(uint32_t group, uint32_t qStart,
                                        uint32_t localQStart, uint32_t localQCount)
    {
        if (localQCount == 0U) {
            return;
        }
        const uint64_t sourceOffset =
            static_cast<uint64_t>(qStart + localQStart) * tiling_.weightStrideToken +
            static_cast<uint64_t>(group) * tiling_.weightStrideGroup;
        // The [T,G,H] source layout has gaps between rows; a single 2D move pads each UB row up to 32 bytes.
        if (tiling_.weightDtype == 0U) {
            const uint32_t rowBytes = Heads() * sizeof(float);
            const uint32_t sourceGap =
                (tiling_.weightStrideToken - Heads()) * sizeof(float);
            DataCopyExtParams params{
                static_cast<uint16_t>(localQCount), rowBytes, sourceGap, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
            DataCopyPad(weightUb_, weightsFp32Gm_[sourceOffset], params, pad);
        } else if (tiling_.weightDtype == 1U) {
            const uint32_t rowBytes = Heads() * sizeof(bfloat16_t);
            const uint32_t sourceGap =
                (tiling_.weightStrideToken - Heads()) * sizeof(bfloat16_t);
            DataCopyExtParams params{
                static_cast<uint16_t>(localQCount), rowBytes, sourceGap, 0U, 0U};
            DataCopyPadExtParams<bfloat16_t> pad{false, 0U, 0U, 0.0F};
            DataCopyPad(weightInputBf16Ub_, weightsBf16Gm_[sourceOffset], params, pad);
        } else {
            const uint32_t rowBytes = Heads() * sizeof(half);
            const uint32_t sourceGap =
                (tiling_.weightStrideToken - Heads()) * sizeof(half);
            DataCopyExtParams params{
                static_cast<uint16_t>(localQCount), rowBytes, sourceGap, 0U, 0U};
            DataCopyPadExtParams<half> pad{false, 0U, 0U, 0.0F};
            DataCopyPad(weightInputFp16Ub_, weightsFp16Gm_[sourceOffset], params, pad);
        }
        SetFlag<HardEvent::MTE2_V>(mte2ToVector_);
    }

    __aicore__ inline void FinishWeights(uint32_t localQCount)
    {
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector_);
        // The FP32 weights serve both Vector and Scalar; each must separately synchronize the latest UB data.
        if (tiling_.weightDtype == 0U) {
            SetFlag<HardEvent::MTE2_S>(mte2ToScalar_);
            WaitFlag<HardEvent::MTE2_S>(mte2ToScalar_);
        }
        const uint32_t fp32Stride = WeightFp32RowStride();
        const uint32_t nativeStride = WeightNativeRowStride();
        for (uint32_t row = 0U; row < localQCount; ++row) {
            if (tiling_.weightDtype == 1U) {
                Cast(weightUb_[row * fp32Stride], weightInputBf16Ub_[row * nativeStride],
                     RoundMode::CAST_NONE, Heads());
            } else if (tiling_.weightDtype == 2U) {
                Cast(weightUb_[row * fp32Stride], weightInputFp16Ub_[row * nativeStride],
                     RoundMode::CAST_NONE, Heads());
            }
            if (tiling_.headScale != 1.0F) {
                Muls(weightUb_[row * fp32Stride], weightUb_[row * fp32Stride],
                     tiling_.headScale, Heads());
            }
        }
        if (tiling_.weightDtype != 0U || tiling_.headScale != 1.0F) {
            SetFlag<HardEvent::V_S>(vectorToScalar_);
            WaitFlag<HardEvent::V_S>(vectorToScalar_);
        }
    }

    __aicore__ inline float LoadScaledWeight(uint32_t row, uint32_t head) const
    {
        return weightUb_.GetValue(row * WeightFp32RowStride() + head);
    }

    __aicore__ inline void LoadVisibleLengths(uint32_t request,
                                               uint32_t requestBegin,
                                               uint32_t qStart,
                                               uint32_t localQStart,
                                               uint32_t localQCount)
    {
        if (localQCount == 0U) {
            return;
        }
        for (uint32_t row = 0U; row < localQCount; ++row) {
            visibleUb_.SetValue(row, VisibleRegions(
                request, requestBegin, qStart + localQStart + row));
        }
    }

    __aicore__ inline void GetLocalQueryRange(uint32_t qCount,
                                               uint32_t &localQStart,
                                               uint32_t &localQCount) const
    {
        GetLocalQueryRange(qCount, GetSubBlockIdx(), localQStart, localQCount);
    }

    __aicore__ inline bool CanUseOutput2D(uint32_t rowElements) const
    {
        return tiling_.outputStrideToken >= rowElements &&
               (tiling_.outputStrideToken - rowElements) <=
                   static_cast<uint64_t>(UINT32_MAX) / sizeof(float);
    }

    __aicore__ inline uint32_t AcquireOutputStore()
    {
        const uint32_t slot = outputStoreCount_ % PAGED_PIPE_SLOTS;
        if (outputStoreCount_ >= PAGED_PIPE_SLOTS) {
            WaitFlag<HardEvent::MTE3_V>(outputStoreDone_[slot]);
        }
        return slot;
    }

    __aicore__ inline void CommitOutputStore(uint32_t slot)
    {
        SetFlag<HardEvent::MTE3_V>(outputStoreDone_[slot]);
        ++outputStoreCount_;
    }

    __aicore__ inline void DrainOutputStores()
    {
        const uint32_t pending = PagedMin(outputStoreCount_, PAGED_PIPE_SLOTS);
        for (uint32_t slot = 0U; slot < pending; ++slot) {
            WaitFlag<HardEvent::MTE3_V>(outputStoreDone_[slot]);
        }
    }

    __aicore__ inline void StoreRows(uint32_t group, uint32_t firstQ,
                                      uint32_t kStart, uint32_t kCount,
                                      uint32_t rowCount, uint32_t rowStride,
                                      LocalTensor<float> source)
    {
        SetFlag<HardEvent::V_MTE3>(vectorToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vectorToMte3_);
        const uint64_t outputOffset =
            static_cast<uint64_t>(group) * tiling_.outputStrideGroup +
            static_cast<uint64_t>(firstQ - tiling_.batchBegin) * tiling_.outputStrideToken + kStart;
        if (CanUseOutput2D(kCount)) {
            DataCopyExtParams params{
                static_cast<uint16_t>(rowCount),
                static_cast<uint32_t>(kCount * sizeof(float)),
                static_cast<uint32_t>((rowStride - kCount) * sizeof(float) / 32U),
                static_cast<uint32_t>((tiling_.outputStrideToken - kCount) * sizeof(float)), 0U};
            DataCopyPad(outputGm_[outputOffset], source, params);
            return;
        }
        for (uint32_t row = 0U; row < rowCount; ++row) {
            DataCopyExtParams params{
                1U, static_cast<uint32_t>(kCount * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(
                outputGm_[outputOffset + static_cast<uint64_t>(row) * tiling_.outputStrideToken],
                source[row * rowStride], params);
        }
    }

    __aicore__ inline void ReduceFullRows(
        const LocalTensor<float> &accSlot,
        const LocalTensor<float> &mmSlot)
    {
        // One VF reduces two 64-column segments over the 16 H4 weight rows; the moves and events are handled by the caller.
        IndexerRegionScoreVector::WeightedReduceH4RowsVF<
            2U, PAGED_H4_WEIGHT_ROW_STRIDE>(
            accSlot, mmSlot, weightUb_, 0U, PAGED_NR, PAGED_NR);
    }

    __aicore__ inline void ReduceCausalFullTile(
        LocalTensor<float> accSlot, LocalTensor<float> mmSlot,
        uint32_t localQCount, int32_t tileBegin,
        bool fullyVisible, bool fullyMasked)
    {
        if (fullyVisible && (tiling_.seqK % PAGED_NR) == 0U) {
            ReduceFullRows(accSlot, mmSlot);
            return;
        }
        if (fullyMasked) {
            return;
        }
        for (uint32_t row = 0U; row < localQCount; ++row) {
            for (uint32_t segment = 0U; segment < 2U; ++segment) {
                const uint32_t colStart = segment * 64U;
                LocalTensor<float> rowOutput =
                    accSlot[row * PAGED_NR + colStart];
                LocalTensor<float> rowInput =
                    mmSlot[row * PAGED_H * PAGED_NR + colStart];
                if (fullyVisible) {
                    IndexerRegionScoreVector::WeightedReduceH4Micro(
                        rowOutput, rowInput, PAGED_NR, 64U,
                        LoadScaledWeight(row, 0U), LoadScaledWeight(row, 1U),
                        LoadScaledWeight(row, 2U), LoadScaledWeight(row, 3U));
                } else {
                    IndexerRegionScoreVector::WeightedReduceH4MaskMicro(
                        rowOutput, rowInput, PAGED_NR, 64U,
                        LoadScaledWeight(row, 0U), LoadScaledWeight(row, 1U),
                        LoadScaledWeight(row, 2U), LoadScaledWeight(row, 3U),
                        tileBegin + static_cast<int32_t>(colStart),
                        visibleUb_.GetValue(row));
                }
            }
        }
    }

    __aicore__ inline void StoreCausalFullTile(
        uint32_t group, uint32_t qStart, uint32_t localQStart, uint32_t localQCount,
        uint32_t kStart, uint32_t kCount, uint32_t resultSlot, LocalTensor<float> mmSlot)
    {
        int32_t minVisible = visibleUb_.GetValue(0U);
        int32_t maxVisible = minVisible;
        for (uint32_t row = 1U; row < localQCount; ++row) {
            const int32_t visible = visibleUb_.GetValue(row);
            minVisible = visible < minVisible ? visible : minVisible;
            maxVisible = visible > maxVisible ? visible : maxVisible;
        }
        const int32_t tileBegin = static_cast<int32_t>(kStart);
        const int32_t tileEnd = static_cast<int32_t>(kStart + kCount);
        const bool fullyVisible = tileEnd <= minVisible;
        const bool fullyMasked = tileBegin >= maxVisible;
        const uint32_t storeSlot = AcquireOutputStore();
        LocalTensor<float> accSlot = acc_[storeSlot * PAGED_OUTPUT_SLOT_ELEMENTS];
        if (!fullyVisible) {
            Duplicate(accSlot.ReinterpretCast<uint32_t>(),
                      IndexerRegionScoreVector::H4_NEG_INF_BITS,
                      localQCount * PAGED_NR);
        }
        CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_V>(
            PAGED_C_READY_EVENT + resultSlot);
        ReduceCausalFullTile(accSlot, mmSlot, localQCount, tileBegin,
                             fullyVisible, fullyMasked);
        StoreRows(group, qStart + localQStart, kStart, kCount,
                  localQCount, PAGED_NR, accSlot);
        CommitOutputStore(storeSlot);
    }

    __aicore__ inline void ReduceRuntimeHeads(
        LocalTensor<float> dst, LocalTensor<float> src, uint32_t count, uint32_t row)
    {
        // Native weights have already been cast; reuse their UB as reduction scratch.
        auto tmp = weightInputFp16Ub_.ReinterpretCast<float>();
        for (uint32_t head = 0U; head < Heads(); head += PAGED_H) {
            auto partial = head == 0U ? dst : tmp;
            IndexerRegionScoreVector::WeightedReduceH4Micro(
                partial, src[head * PAGED_NR], PAGED_NR, count,
                LoadScaledWeight(row, head), LoadScaledWeight(row, head + 1U),
                LoadScaledWeight(row, head + 2U), LoadScaledWeight(row, head + 3U));
            if (head != 0U) {
                Add(dst, dst, tmp, count);
            }
        }
    }

    __aicore__ inline void StoreCausalTail(
        uint32_t group, uint32_t qStart, uint32_t localQStart, uint32_t localQCount,
        uint32_t kStart, uint32_t kCount, LocalTensor<float> mmSlot)
    {
        for (uint32_t segment = 0U; segment < 2U; ++segment) {
            const uint32_t colStart = segment * 64U;
            if (colStart >= kCount) {
                break;
            }
            const uint32_t segmentCount = PagedMin(64U, kCount - colStart);
            const uint32_t rowStride = PagedAlignUp(segmentCount, 8U);
            const uint32_t storeSlot = AcquireOutputStore();
            LocalTensor<float> accSlot = acc_[storeSlot * PAGED_OUTPUT_SLOT_ELEMENTS];
            Duplicate(accSlot.ReinterpretCast<uint32_t>(),
                      IndexerRegionScoreVector::H4_NEG_INF_BITS,
                      localQCount * rowStride);
            for (uint32_t row = 0U; row < localQCount; ++row) {
                if constexpr (FOUR_HEADS) {
                    IndexerRegionScoreVector::WeightedReduceH4MaskMicro(
                        accSlot[row * rowStride],
                        mmSlot[row * PAGED_H * PAGED_NR + colStart],
                        PAGED_NR, segmentCount,
                        LoadScaledWeight(row, 0U), LoadScaledWeight(row, 1U),
                        LoadScaledWeight(row, 2U), LoadScaledWeight(row, 3U),
                        static_cast<int32_t>(kStart + colStart),
                        visibleUb_.GetValue(row));
                } else {
                    const int32_t visible = visibleUb_.GetValue(row) - static_cast<int32_t>(kStart + colStart);
                    if (visible > 0) {
                        ReduceRuntimeHeads(accSlot[row * rowStride],
                            mmSlot[row * Heads() * PAGED_NR + colStart],
                            PagedMin(segmentCount, static_cast<uint32_t>(visible)), row);
                    }
                }
            }
            StoreRows(group, qStart + localQStart, kStart + colStart,
                      segmentCount, localQCount, rowStride, accSlot);
            CommitOutputStore(storeSlot);
        }
    }

    __aicore__ inline void PostprocessCausalTile(
        uint32_t group, uint32_t qStart, uint32_t qCount, uint32_t kStart,
        uint32_t kCount, uint32_t resultSlot)
    {
        uint32_t localQStart = 0U;
        uint32_t localQCount = 0U;
        GetLocalQueryRange(qCount, localQStart, localQCount);
        if (localQCount == 0U) {
            CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_V>(
                PAGED_C_READY_EVENT + resultSlot);
            return;
        }
        LocalTensor<float> mmSlot = MmOut(resultSlot * PAGED_RESULT_SLOT_ELEMENTS);

        if constexpr (FOUR_HEADS) {
            if (qCount == PAGED_MQ_MAX && kCount == PAGED_NR &&
                localQCount == PAGED_QUERIES_PER_AIV_MAX) {
                StoreCausalFullTile(group, qStart, localQStart, localQCount,
                                    kStart, kCount, resultSlot, mmSlot);
                return;
            }
        }

        CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_V>(
            PAGED_C_READY_EVENT + resultSlot);
        StoreCausalTail(group, qStart, localQStart, localQCount, kStart, kCount, mmSlot);
    }

    __aicore__ inline void ConsumeVectorResult(
        uint32_t group, uint32_t qStart, uint32_t qMicro, uint32_t qMicroCount,
        uint32_t kStart, uint32_t kMicro, uint32_t kMicroCount, uint32_t resultSlot,
        uint32_t request, uint32_t localQStart, uint32_t localQCount, uint32_t &streamLoop)
    {

            PostprocessCausalTile(
                group, qStart + qMicro, qMicroCount,
                kStart + kMicro, kMicroCount, resultSlot);

        PipeBarrier<PIPE_V>();
        CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_V>(
            PAGED_C_FREE_EVENT + resultSlot);

        ++streamLoop;
    }

    __aicore__ inline void ProcessVectorQueryMacro(
        uint32_t request, uint32_t requestQBegin, uint32_t group, uint32_t qStart,
        uint32_t qMacroCount, uint32_t kStart, uint32_t kMacroCount, uint32_t &streamLoop)
    {
        for (uint32_t qMicro = 0U; qMicro < qMacroCount;
             qMicro += QMicroTile()) {
            const uint32_t qMicroCount =
                PagedMin(QMicroTile(), qMacroCount - qMicro);
            uint32_t localQStart = 0U;
            uint32_t localQCount = 0U;
            GetLocalQueryRange(qMicroCount, localQStart, localQCount);
            if (localQCount != 0U) {
                LoadWeights(group, qStart + qMicro,
                            localQStart, localQCount);
            }

                LoadVisibleLengths(request, requestQBegin,
                                   qStart + qMicro,
                                   localQStart, localQCount);

            if (localQCount != 0U) {
                FinishWeights(localQCount);
            }
            for (uint32_t kMicro = 0U; kMicro < kMacroCount;
                 kMicro += PAGED_NR) {
                const uint32_t kMicroCount =
                    PagedMin(PAGED_NR, kMacroCount - kMicro);
                const uint32_t resultSlot = streamLoop % PAGED_PIPE_SLOTS;
                ConsumeVectorResult(group, qStart, qMicro, qMicroCount, kStart, kMicro,
                        kMicroCount, resultSlot, request, localQStart, localQCount, streamLoop);
            }
            // There is only one weight UB; the next move may start only after all VFs of the current generation finish consuming it.
            if (localQCount != 0U) {
                SetFlag<HardEvent::V_MTE2>(vectorToMte2_);
                WaitFlag<HardEvent::V_MTE2>(vectorToMte2_);
            }
        }
    }

    __aicore__ inline void ConsumeVectorKeyMacro(
        uint32_t request, uint32_t requestQBegin, uint32_t qBegin, uint32_t qEnd,
        uint32_t kStart, uint32_t kMacroCount, uint32_t &streamLoop)
    {
        for (uint32_t group = 0U; group < tiling_.groups; ++group) {
            for (uint32_t qStart = qBegin; qStart < qEnd; qStart += QMacroTile()) {
                const uint32_t qMacroCount =
                    PagedMin(QMacroTile(), qEnd - qStart);
                ProcessVectorQueryMacro(request, requestQBegin, group, qStart,
                                      qMacroCount, kStart, kMacroCount, streamLoop);
            }
        }
    }

    __aicore__ inline void ProcessVectorRequests(
        uint32_t coreQBegin, uint32_t coreQEnd, uint32_t coreKBegin, uint32_t coreKEnd,
        uint32_t &streamLoop)
    {
        for (uint32_t request = 0U; request < tiling_.numRequests; ++request) {
            uint32_t requestQBegin = 0U;
            uint32_t requestQEnd = 0U;
            if (!LoadRequestSpan(request, requestQBegin, requestQEnd)) {
                continue;
            }
            uint32_t qBegin = 0U;
            uint32_t qEnd = 0U;
            if (!IntersectQueryRange(requestQBegin, requestQEnd,
                                     coreQBegin, coreQEnd, qBegin, qEnd)) {
                continue;
            }
            uint32_t requestKBegin = coreKBegin;
            uint32_t requestKEnd = coreKEnd;
            if (!GetRequestKeyRange(request, requestQBegin, qEnd,
                                          requestKBegin, requestKEnd)) {
                continue;
            }

            for (uint32_t kStart = requestKBegin; kStart < requestKEnd; kStart += PAGED_K_MACRO) {
                const uint32_t kMacroCount =
                    PagedMin(PAGED_K_MACRO, requestKEnd - kStart);
                ConsumeVectorKeyMacro(request, requestQBegin, qBegin, qEnd,
                                          kStart, kMacroCount, streamLoop);
            }
        }
    }

    __aicore__ inline void ProcessVector()
    {
        for (uint32_t slot = 0U; slot < PAGED_PIPE_SLOTS; ++slot) {
            CrossCoreSetFlag<PAGED_SYNC_MODE, PIPE_V>(PAGED_C_FREE_EVENT + slot);
        }

        uint32_t streamLoop = 0U;
        uint32_t coreQBegin = 0U;
        uint32_t coreQEnd = 0U;
        uint32_t coreKBegin = 0U;
        uint32_t coreKEnd = 0U;
        GetCoreWorkRange(coreQBegin, coreQEnd, coreKBegin, coreKEnd);
        ProcessVectorRequests(coreQBegin, coreQEnd, coreKBegin, coreKEnd, streamLoop);
        CrossCoreWaitFlag<PAGED_SYNC_MODE, PIPE_V>(PAGED_WORK_DRAINED_EVENT);

    }

    PagedTilingData tiling_{};
    TPipe *pipe_ = nullptr;
    KeyAdapter *keyAdapter_ = nullptr;
    int64_t requestFirstPosition_ = 0;
    uint32_t requestKCarry_ = 0U;

    GlobalTensor<QkInputType> indexQGm_;
    GlobalTensor<float> weightsFp32Gm_;
    GlobalTensor<bfloat16_t> weightsBf16Gm_;
    GlobalTensor<half> weightsFp16Gm_;
    GlobalTensor<int32_t> actualKvLengthsGm_;
    GlobalTensor<float> outputGm_;

    TEventID qReady_[PAGED_PIPE_SLOTS];
    TEventID qFree_[PAGED_PIPE_SLOTS];
    TEventID mToMte1_[PAGED_PIPE_SLOTS];
    TEventID fixToM_[PAGED_PIPE_SLOTS];
    TEventID mte1ToM_;
    TEventID mToFix_;
    TEventID mte2ToScalar_;
    TEventID mte2ToVector_;
    TEventID vectorToScalar_;
    TEventID vectorToMte2_;
    TEventID vectorToMte3_;
    TEventID outputStoreDone_[PAGED_PIPE_SLOTS];

    TBuf<TPosition::A1> qL1Buf_;
    TBuf<TPosition::A2> qL0Buf_;
    TBuf<TPosition::B2> kL0Buf_;
    TBuf<TPosition::CO1> cL0Buf_;
    TBuf<TPosition::VECCALC> mmOutBuf_[PAGED_PIPE_SLOTS];
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> weightUbBuf_;
    TBuf<TPosition::VECCALC> weightInputUbBuf_;
    TBuf<TPosition::VECCALC> cuSeqLensUbBuf_;
    TBuf<TPosition::VECCALC> visibleUbBuf_;

    LocalTensor<QkInputType> qL1_[PAGED_PIPE_SLOTS];
    LocalTensor<QkInputType> qL0_;
    LocalTensor<QkInputType> kL0_;
    LocalTensor<float> cL0_;
    LocalTensor<float> mmOut_[PAGED_PIPE_SLOTS];
    LocalTensor<float> acc_;
    LocalTensor<float> weightUb_;
    LocalTensor<bfloat16_t> weightInputBf16Ub_;
    LocalTensor<half> weightInputFp16Ub_;
    LocalTensor<int32_t> cuSeqLensUb_;
    LocalTensor<int32_t> visibleUb_;
    uint32_t outputStoreCount_ = 0U;
};

} // namespace IndexerRegionScorePaged

#endif
#endif
