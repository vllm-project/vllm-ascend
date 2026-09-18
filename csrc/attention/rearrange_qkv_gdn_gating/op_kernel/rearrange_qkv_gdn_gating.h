// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vllm-ascend project

#pragma once

#include "kernel_operator.h"
// The generated mix kernel wrapper references matmul::clearWorkspace.
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace rearrange_qkv_gdn_gating_impl {
constexpr uint32_t BF16_BYTES = sizeof(uint16_t);
constexpr uint32_t FP32_BYTES = sizeof(float);
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t DATABLOCK_BYTES = 32;
constexpr uint32_t BF16_PER_DATABLOCK = DATABLOCK_BYTES / BF16_BYTES;
constexpr uint32_t ALIGNED_HEADS = DATABLOCK_BYTES / BF16_BYTES;
constexpr uint32_t DMA_TILE_BYTES = 160 * 1024;
constexpr uint32_t DMA_TILE_ELEMENTS = DMA_TILE_BYTES / BF16_BYTES;

// Event ids are shared per pipeline pair, every id below is set and waited
// exactly once per tile so that the hardware counters stay balanced.
constexpr int32_t EVENT_DMA_LOAD_TO_STORE = 0;
constexpr int32_t EVENT_DMA_STORE_TO_LOAD = 1;
constexpr int32_t EVENT_HEAD_LOAD_TO_VEC = 2;
constexpr int32_t EVENT_TILE_LOAD_TO_VEC = 3;
constexpr int32_t EVENT_VEC_TO_LOAD = 4;
constexpr int32_t EVENT_VEC_TO_STORE = 5;
constexpr int32_t EVENT_STORE_TO_VEC = 6;
constexpr int32_t EVENT_STORE_TO_LOAD = 7;

/**
 * Cube (AIC) stage: rearrange the token major mixed QKV into three packed
 * [query | key | value] blocks with plain MTE copies, i.e. without touching
 * the vector pipes.
 */
class RearrangeQkvDmaStage {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RearrangeQkvGdnGatingTilingData& tiling, TPipe* pipe)
    {
        tokens_ = tiling.tokens;
        qDim_ = tiling.qDim;
        kDim_ = tiling.kDim;
        vDim_ = tiling.vDim;
        rowDim_ = tiling.rowDim;
        tileRows_ = tiling.dmaTileRows;
        coreNum_ = GetBlockNum();
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
        x_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        y_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        pipe->InitBuffer(copyBuffer_, DMA_TILE_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (coreNum_ == 0 || tokens_ == 0 || rowDim_ == 0) {
            return;
        }
        const uint64_t coreId = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t usedCoreNum = static_cast<uint64_t>(coreNum_);
        const uint64_t baseRows = tokens_ / usedCoreNum;
        const uint64_t extraRows = tokens_ % usedCoreNum;
        const uint64_t coreRows = baseRows + (coreId < extraRows ? 1 : 0);
        const uint64_t rowStart = coreId * baseRows + (coreId < extraRows ? coreId : extraRows);
        LocalTensor<bfloat16_t> local = copyBuffer_.Get<bfloat16_t>();

        uint64_t row = rowStart;
        const uint64_t rowEnd = rowStart + coreRows;
        while (row < rowEnd) {
            const uint64_t qDst = row * qDim_;
            const uint64_t kDst = tokens_ * qDim_ + row * kDim_;
            const uint64_t vDst = tokens_ * (qDim_ + kDim_) + row * vDim_;
            if (tileRows_ == 0) {
                const uint64_t src = row * rowDim_;
                CopySegment(local, src, qDst, qDim_);
                CopySegment(local, src + qDim_, kDst, kDim_);
                CopySegment(local, src + qDim_ + kDim_, vDst, vDim_);
                ++row;
                continue;
            }

            uint64_t rows = rowEnd - row;
            if (rows > tileRows_) {
                rows = tileRows_;
            }
            DataCopy(local, x_[row * rowDim_], static_cast<uint32_t>(rows * rowDim_));
            WaitLoadBeforeStore();

            const DataCopyParams qParams{
                static_cast<uint16_t>(rows), static_cast<uint16_t>(qDim_ / BF16_PER_DATABLOCK),
                static_cast<uint16_t>((rowDim_ - qDim_) / BF16_PER_DATABLOCK), 0};
            const DataCopyParams kParams{
                static_cast<uint16_t>(rows), static_cast<uint16_t>(kDim_ / BF16_PER_DATABLOCK),
                static_cast<uint16_t>((rowDim_ - kDim_) / BF16_PER_DATABLOCK), 0};
            const DataCopyParams vParams{
                static_cast<uint16_t>(rows), static_cast<uint16_t>(vDim_ / BF16_PER_DATABLOCK),
                static_cast<uint16_t>((rowDim_ - vDim_) / BF16_PER_DATABLOCK), 0};
            DataCopy(y_[qDst], local, qParams);
            DataCopy(y_[kDst], local[qDim_], kParams);
            DataCopy(y_[vDst], local[qDim_ + kDim_], vParams);
            WaitStoreBeforeLoad();
            row += rows;
        }
    }

private:
    __aicore__ inline void WaitLoadBeforeStore()
    {
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_DMA_LOAD_TO_STORE);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_DMA_LOAD_TO_STORE);
    }

    __aicore__ inline void WaitStoreBeforeLoad()
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_DMA_STORE_TO_LOAD);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_DMA_STORE_TO_LOAD);
    }

    __aicore__ inline void CopySegment(LocalTensor<bfloat16_t> local, uint64_t src, uint64_t dst, uint64_t width)
    {
        for (uint64_t offset = 0; offset < width; offset += DMA_TILE_ELEMENTS) {
            uint64_t count = width - offset;
            if (count > DMA_TILE_ELEMENTS) {
                count = DMA_TILE_ELEMENTS;
            }
            DataCopy(local, x_[src + offset], static_cast<uint32_t>(count));
            WaitLoadBeforeStore();
            const DataCopyParams params{
                1, static_cast<uint16_t>(count / BF16_PER_DATABLOCK), 0, 0};
            DataCopy(y_[dst + offset], local, params);
            WaitStoreBeforeLoad();
        }
    }

    GlobalTensor<bfloat16_t> x_;
    GlobalTensor<bfloat16_t> y_;
    TBuf<TPosition::A1> copyBuffer_;
    uint64_t tokens_ = 0;
    uint64_t qDim_ = 0;
    uint64_t kDim_ = 0;
    uint64_t vDim_ = 0;
    uint64_t rowDim_ = 0;
    uint32_t tileRows_ = 0;
    uint32_t coreNum_ = 1;
};

/**
 * Vector (AIV) stage: the gating math that runs while the cube cores are busy
 * with the rearrange.
 *
 *   u = beta * (a + dt_bias)
 *   g = -exp(alog) / beta * (max(u, 0) + log(1 + exp(-|u|)))
 *   beta_out = sigmoid(b)
 *
 * `max(u, 0) + log(1 + exp(-|u|))` is the numerically stable softplus, equal
 * to log(1 + exp(u)) without the overflow of the naive form.
 */
template <typename THead>
class GdnGatingStage {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR aLog, GM_ADDR dtBias, GM_ADDR g, GM_ADDR betaOut,
                                const RearrangeQkvGdnGatingTilingData& tiling, TPipe* pipe)
    {
        tokens_ = tiling.tokens;
        heads_ = tiling.numHeads;
        tileRows_ = tiling.gatingTileRows;
        beta_ = tiling.beta;
        threshold_ = tiling.threshold;
        // The stable softplus below is exact for every input, so the
        // threshold branch of the reference kernel is not needed here.
        (void)threshold_;
        if (tileRows_ == 0) {
            tileRows_ = 1;
        }
        // BuildHeadVectors grows row replicas with vector stores at
        // `filled * heads_`; keep every such UB destination 32-byte aligned.
        if ((heads_ * FP32_BYTES) % ALIGN_BYTES != 0) {
            tileRows_ = 1;
        }
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(a));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(b));
        aLog_.SetGlobalBuffer(reinterpret_cast<__gm__ THead*>(aLog));
        dtBias_.SetGlobalBuffer(reinterpret_cast<__gm__ THead*>(dtBias));
        g_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g));
        betaOut_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(betaOut));
        // A row of `heads_` bf16 values covers a whole number of 32-byte
        // blocks only when the head count is a multiple of 16. GDN layers such
        // as Qwen3-Next expose 24 heads per rank, hence the padded copy path.
        aligned_ = (heads_ % ALIGNED_HEADS) == 0;

        const uint32_t tileElements = tileRows_ * heads_;
        pipe->InitBuffer(stagingBuf_, AlignUp(tileElements * BF16_BYTES, ALIGN_BYTES));
        pipe->InitBuffer(vecBuf_, AlignUp(tileElements * FP32_BYTES, ALIGN_BYTES));
        pipe->InitBuffer(scratchBuf_, AlignUp(tileElements * FP32_BYTES, ALIGN_BYTES));
        pipe->InitBuffer(biasRepeatBuf_, AlignUp(tileElements * FP32_BYTES, ALIGN_BYTES));
        pipe->InitBuffer(scaleRepeatBuf_, AlignUp(tileElements * FP32_BYTES, ALIGN_BYTES));
        const uint32_t headRawBytes = 2 * AlignUp(heads_ * static_cast<uint32_t>(sizeof(THead)), ALIGN_BYTES);
        const uint32_t headVecBytes = 2 * AlignUp(heads_ * FP32_BYTES, ALIGN_BYTES);
        pipe->InitBuffer(headRawBuf_, headRawBytes);
        pipe->InitBuffer(headVecBuf_, headVecBytes);
    }

    __aicore__ inline void Process()
    {
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        const uint64_t coreNum = static_cast<uint64_t>(GetBlockNum());
        if (subBlockNum == 0 || coreNum == 0 || tokens_ == 0 || heads_ == 0) {
            return;
        }
        // In a MIX_AIC_1_2 launch GetBlockNum() already reports the total
        // number of AIV workers.  Pair the group index with the AIV sub-block
        // id instead of multiplying the worker count by the task ratio again.
        const uint64_t coreId = (static_cast<uint64_t>(GetBlockIdx()) / subBlockNum) * subBlockNum +
                                static_cast<uint64_t>(GetSubBlockIdx());
        const uint64_t baseRows = tokens_ / coreNum;
        const uint64_t extraRows = tokens_ % coreNum;
        const uint64_t coreRows = baseRows + (coreId < extraRows ? 1 : 0);
        const uint64_t rowStart = coreId * baseRows + (coreId < extraRows ? coreId : extraRows);
        if (coreRows == 0) {
            return;
        }
        const uint32_t repeatRows = static_cast<uint32_t>(coreRows < tileRows_ ? coreRows : tileRows_);
        BuildHeadVectors(repeatRows);

        uint64_t row = rowStart;
        const uint64_t rowEnd = rowStart + coreRows;
        while (row < rowEnd) {
            uint64_t rows = rowEnd - row;
            if (rows > tileRows_) {
                rows = tileRows_;
            }
            ProcessTile(row, static_cast<uint32_t>(rows));
            row += rows;
        }
    }

private:
    __aicore__ inline void BuildHeadVectors(uint32_t repeatRows)
    {
        LocalTensor<THead> raw = headRawBuf_.Get<THead>();
        const uint32_t headRawStride =
            AlignUp(heads_ * static_cast<uint32_t>(sizeof(THead)), ALIGN_BYTES) / sizeof(THead);
        LocalTensor<float> headVec = headVecBuf_.Get<float>();
        const uint32_t headVecStride = AlignUp(heads_ * FP32_BYTES, ALIGN_BYTES) / FP32_BYTES;
        LocalTensor<float> biasHead = headVec;
        LocalTensor<float> scaleHead = headVec[headVecStride];
        LoadHead(raw, dtBias_);
        LoadHead(raw[headRawStride], aLog_);
        SetFlag<HardEvent::MTE2_V>(EVENT_HEAD_LOAD_TO_VEC);
        WaitFlag<HardEvent::MTE2_V>(EVENT_HEAD_LOAD_TO_VEC);
        if constexpr (IsSameType<THead, float>::value) {
            Adds(biasHead, raw, 0.0f, heads_);
            Adds(scaleHead, raw[headRawStride], 0.0f, heads_);
        } else {
            Cast(biasHead, raw, RoundMode::CAST_NONE, heads_);
            Cast(scaleHead, raw[headRawStride], RoundMode::CAST_NONE, heads_);
        }
        PipeBarrier<PIPE_V>();
        Exp(scaleHead, scaleHead, heads_);
        PipeBarrier<PIPE_V>();
        Muls(scaleHead, scaleHead, -1.0f / beta_, heads_);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> biasRepeat = biasRepeatBuf_.Get<float>();
        LocalTensor<float> scaleRepeat = scaleRepeatBuf_.Get<float>();
        Adds(biasRepeat, biasHead, 0.0f, heads_);
        Adds(scaleRepeat, scaleHead, 0.0f, heads_);
        PipeBarrier<PIPE_V>();
        // Grow the per row copies by doubling the already filled prefix instead
        // of issuing one copy per row.
        uint32_t filled = 1;
        while (filled < repeatRows) {
            const uint32_t remaining = repeatRows - filled;
            const uint32_t chunk = filled < remaining ? filled : remaining;
            Adds(biasRepeat[filled * heads_], biasRepeat, 0.0f, chunk * heads_);
            Adds(scaleRepeat[filled * heads_], scaleRepeat, 0.0f, chunk * heads_);
            PipeBarrier<PIPE_V>();
            filled += chunk;
        }
    }

    __aicore__ inline void ProcessTile(uint64_t rowStart, uint32_t rows)
    {
        const uint32_t count = rows * heads_;
        const uint64_t offset = rowStart * static_cast<uint64_t>(heads_);
        LocalTensor<bfloat16_t> staging = stagingBuf_.Get<bfloat16_t>();
        LocalTensor<float> value = vecBuf_.Get<float>();
        LocalTensor<float> scratch = scratchBuf_.Get<float>();
        LocalTensor<float> biasRepeat = biasRepeatBuf_.Get<float>();
        LocalTensor<float> scaleRepeat = scaleRepeatBuf_.Get<float>();

        // a -> g
        LoadBf16(staging, a_, offset, count);
        SetFlag<HardEvent::MTE2_V>(EVENT_TILE_LOAD_TO_VEC);
        WaitFlag<HardEvent::MTE2_V>(EVENT_TILE_LOAD_TO_VEC);
        Cast(value, staging, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_VEC_TO_LOAD);
        Add(value, value, biasRepeat, count);
        PipeBarrier<PIPE_V>();
        Muls(value, value, beta_, count);
        PipeBarrier<PIPE_V>();
        Abs(scratch, value, count);
        PipeBarrier<PIPE_V>();
        Muls(scratch, scratch, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Exp(scratch, scratch, count);
        PipeBarrier<PIPE_V>();
        Adds(scratch, scratch, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Ln(scratch, scratch, count);
        PipeBarrier<PIPE_V>();
        Maxs(value, value, 0.0f, count);
        PipeBarrier<PIPE_V>();
        Add(scratch, scratch, value, count);
        PipeBarrier<PIPE_V>();
        Mul(scratch, scratch, scaleRepeat, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_VEC_TO_STORE);
        WaitFlag<HardEvent::V_MTE3>(EVENT_VEC_TO_STORE);
        StoreFp32(g_, scratch, offset, count);
        SetFlag<HardEvent::MTE3_V>(EVENT_STORE_TO_VEC);

        // b -> beta
        WaitFlag<HardEvent::V_MTE2>(EVENT_VEC_TO_LOAD);
        LoadBf16(staging, b_, offset, count);
        SetFlag<HardEvent::MTE2_V>(EVENT_TILE_LOAD_TO_VEC);
        WaitFlag<HardEvent::MTE2_V>(EVENT_TILE_LOAD_TO_VEC);
        Cast(value, staging, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_VEC_TO_LOAD);
        WaitFlag<HardEvent::V_MTE2>(EVENT_VEC_TO_LOAD);
        WaitFlag<HardEvent::MTE3_V>(EVENT_STORE_TO_VEC);
        Muls(scratch, value, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Exp(scratch, scratch, count);
        PipeBarrier<PIPE_V>();
        Adds(scratch, scratch, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Duplicate(value, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Div(scratch, value, scratch, count);
        PipeBarrier<PIPE_V>();
        Cast(staging, scratch, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_VEC_TO_STORE);
        WaitFlag<HardEvent::V_MTE3>(EVENT_VEC_TO_STORE);
        StoreBf16(betaOut_, staging, offset, count);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_STORE_TO_LOAD);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_STORE_TO_LOAD);
    }

    __aicore__ inline void LoadHead(const LocalTensor<THead>& dst, const GlobalTensor<THead>& src)
    {
        if (aligned_) {
            DataCopy(dst, src, heads_);
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = heads_ * sizeof(THead);
            params.srcStride = 0;
            params.dstStride = 0;
            DataCopyPad(dst, src, params, {});
        }
    }

    __aicore__ inline void LoadBf16(const LocalTensor<bfloat16_t>& dst, const GlobalTensor<bfloat16_t>& src,
                                    uint64_t offset, uint32_t elements)
    {
        if (aligned_) {
            DataCopy(dst, src[offset], elements);
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = elements * BF16_BYTES;
            params.srcStride = 0;
            params.dstStride = 0;
            DataCopyPad(dst, src[offset], params, {});
        }
    }

    __aicore__ inline void StoreBf16(const GlobalTensor<bfloat16_t>& dst, const LocalTensor<bfloat16_t>& src,
                                     uint64_t offset, uint32_t elements)
    {
        if (aligned_) {
            DataCopy(dst[offset], src, elements);
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = elements * BF16_BYTES;
            params.srcStride = 0;
            params.dstStride = 0;
            DataCopyPad(dst[offset], src, params);
        }
    }

    __aicore__ inline void StoreFp32(const GlobalTensor<float>& dst, const LocalTensor<float>& src, uint64_t offset,
                                     uint32_t elements)
    {
        if (aligned_) {
            DataCopy(dst[offset], src, elements);
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = elements * FP32_BYTES;
            params.srcStride = 0;
            params.dstStride = 0;
            DataCopyPad(dst[offset], src, params);
        }
    }

    GlobalTensor<bfloat16_t> a_;
    GlobalTensor<bfloat16_t> b_;
    GlobalTensor<THead> aLog_;
    GlobalTensor<THead> dtBias_;
    GlobalTensor<float> g_;
    GlobalTensor<bfloat16_t> betaOut_;
    TBuf<TPosition::VECCALC> stagingBuf_;
    TBuf<TPosition::VECCALC> vecBuf_;
    TBuf<TPosition::VECCALC> scratchBuf_;
    TBuf<TPosition::VECCALC> biasRepeatBuf_;
    TBuf<TPosition::VECCALC> scaleRepeatBuf_;
    TBuf<TPosition::VECCALC> headRawBuf_;
    TBuf<TPosition::VECCALC> headVecBuf_;
    uint64_t tokens_ = 0;
    uint32_t heads_ = 0;
    uint32_t tileRows_ = 0;
    float beta_ = 1.0f;
    float threshold_ = 20.0f;
    bool aligned_ = true;
};
}  // namespace rearrange_qkv_gdn_gating_impl
