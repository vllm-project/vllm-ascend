/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FUSED_GDN_DECODE_KERNEL_H
#define FUSED_GDN_DECODE_KERNEL_H

#include "kernel_operator.h"
#include "fused_gdn_decode_tiling_data.h"

namespace FusedGdnDecode {

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BF16_NUM_PER_BLOCK = 16;
constexpr uint32_t FP32_NUM_PER_BLOCK = 8;
constexpr uint32_t REPEAT_LEN = 64;
constexpr uint32_t UB_ALIGN_BYTES = 256;
constexpr float EPS = 1.0e-6f;
constexpr uint32_t GATING_A_LOG_OFFSET = 0;
constexpr uint32_t GATING_DT_BIAS_OFFSET = 8;
constexpr uint32_t GATING_X_OFFSET = 16;
constexpr uint32_t GATING_TMP_OFFSET = 24;
constexpr uint32_t GATING_RELU_OFFSET = 32;
constexpr uint32_t GATING_BETA_OFFSET = 40;
constexpr uint32_t GATING_EXP_G_OFFSET = 48;
constexpr uint32_t SCALAR_BETA_BRCB_OFFSET = 64;
constexpr uint32_t SCALAR_EXP_G_BRCB_OFFSET = 128;
constexpr uint32_t SCALAR_UB_SIZE = 192;

template <typename InType, typename StateType>
class KernelFusedGdnDecode {
public:
    __aicore__ inline KernelFusedGdnDecode() {}

    __aicore__ inline void Init(GM_ADDR mixedQkv, GM_ADDR a, GM_ADDR b, GM_ADDR aLog, GM_ADDR dtBias,
                                GM_ADDR state, GM_ADDR stateIndices, GM_ADDR out, GM_ADDR stateOut,
                                const FusedGdnDecodeTilingData *tilingData, TPipe *pipe)
    {
        pipe_ = pipe;
        td_ = tilingData;
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        alignK_ = Ceil(td_->k, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        alignBV_ = Ceil(td_->bv, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        stateAuxOffset_ = alignK_ * alignBV_ * sizeof(StateType) / sizeof(InType);

        mixedQkvGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InType *>(mixedQkv));
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InType *>(a));
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InType *>(b));
        aLogGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(aLog));
        dtBiasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(dtBias));
        stateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateType *>(state));
        stateOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateType *>(stateOut));
        stateIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(stateIndices));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InType *>(out));
        InitLocalBuffers();
    }

    __aicore__ inline void Process()
    {
        const uint32_t tasksPerBlock = td_->totalTasks / blockNum_;
        const uint32_t remainder = td_->totalTasks % blockNum_;
        const uint32_t taskCount = tasksPerBlock + (blockIdx_ < remainder ? 1 : 0);
        const uint32_t taskStart = blockIdx_ * tasksPerBlock + MinU32(blockIdx_, remainder);
        uint32_t taskOffset = 0;
        while (taskOffset < taskCount) {
            const uint32_t firstTask = taskStart + taskOffset;
            const uint32_t firstHv = firstTask % td_->hv;
            const uint32_t remainingInBatch = td_->hv - firstHv;
            const uint32_t segmentSize = MinU32(
                FP32_NUM_PER_BLOCK, MinU32(taskCount - taskOffset, remainingInBatch));
            ProcessBatchSegment(firstTask, segmentSize);
            taskOffset += segmentSize;
        }
    }

private:
    __aicore__ inline uint32_t Ceil(uint32_t x, uint32_t y) const
    {
        return (x + y - 1) / y;
    }

    __aicore__ inline uint32_t MinU32(uint32_t x, uint32_t y) const
    {
        return x < y ? x : y;
    }

    __aicore__ inline uint32_t AlignBytes(uint32_t x) const
    {
        return Ceil(x, UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
    }

    __aicore__ inline void InitLocalBuffers()
    {
        pipe_->InitBuffer(qQueue_, BUFFER_NUM, 2 * alignK_ * sizeof(InType));
        const uint32_t stateSlotBytes =
            alignK_ * alignBV_ * sizeof(StateType) + alignBV_ * sizeof(InType);
        const uint8_t stateBufferNum = static_cast<uint8_t>(td_->stateBufferNum);
        pipe_->InitBuffer(stateQueue_, stateBufferNum, stateSlotBytes);
        pipe_->InitBuffer(stateOutQueue_, stateBufferNum, stateSlotBytes);
        pipe_->InitBuffer(tmpBuf_, td_->ubRestBytes);

        uint32_t offset = 0;
        offset = AlignBytes(offset);
        qUb_ = tmpBuf_.GetWithOffset<float>(2 * alignK_, offset);
        kUb_ = qUb_[alignK_];
        offset += 2 * alignK_ * sizeof(float);
        offset = AlignBytes(offset);
        vUb_ = tmpBuf_.GetWithOffset<float>(alignBV_, offset);
        offset += alignBV_ * sizeof(float);
        if constexpr (!std::is_same<StateType, float>::value) {
            offset = AlignBytes(offset);
            hUb_ = tmpBuf_.GetWithOffset<float>(alignK_ * alignBV_, offset);
            offset += alignK_ * alignBV_ * sizeof(float);
            offset = AlignBytes(offset);
            hTmpUb_ = tmpBuf_.GetWithOffset<float>(alignK_ * alignBV_, offset);
            offset += alignK_ * alignBV_ * sizeof(float);
        }
        offset = AlignBytes(offset);
        reduceWorkUb_ = tmpBuf_.GetWithOffset<float>(alignK_, offset);
        offset += alignK_ * sizeof(float);
        offset = AlignBytes(offset);
        deltaUb_ = tmpBuf_.GetWithOffset<float>(alignBV_, offset);
        offset += alignBV_ * sizeof(float);
        offset = AlignBytes(offset);
        outUb_ = tmpBuf_.GetWithOffset<float>(alignBV_, offset);
        offset += alignBV_ * sizeof(float);
        offset = AlignBytes(offset);
        scalarUb_ = tmpBuf_.GetWithOffset<float>(SCALAR_UB_SIZE, offset);
    }

    __aicore__ inline void LoadQK(uint32_t batch, uint32_t h)
    {
        const uint64_t base = static_cast<uint64_t>(batch) * td_->mixedStride;
        const uint64_t qOffset = base + static_cast<uint64_t>(h) * td_->k;
        const uint64_t kOffset = base + static_cast<uint64_t>(td_->h + h) * td_->k;

        auto qkLocal = qQueue_.AllocTensor<InType>();
        DataCopyExtParams qkParams{1, static_cast<uint32_t>(td_->k * sizeof(InType)), 0, 0, 0};
        DataCopyPadExtParams<InType> qkPad{true, 0, static_cast<uint8_t>(alignK_ - td_->k), 0};
        DataCopyPad(qkLocal, mixedQkvGm_[qOffset], qkParams, qkPad);
        DataCopyPad(qkLocal[alignK_], mixedQkvGm_[kOffset], qkParams, qkPad);
        qQueue_.EnQue(qkLocal);
        qkLocal = qQueue_.DeQue<InType>();
        Cast(qUb_, qkLocal, RoundMode::CAST_NONE, 2 * alignK_);
        PipeBarrier<PIPE_V>();
        qQueue_.FreeTensor(qkLocal);
    }

    __aicore__ inline void LoadGatingInputs(uint32_t batch, uint32_t firstHv, uint32_t groupSize)
    {
        const uint64_t offset = static_cast<uint64_t>(batch) * td_->hv + firstHv;
        auto abLocal = qQueue_.AllocTensor<InType>();
        DataCopyExtParams abParams{1, static_cast<uint32_t>(groupSize * sizeof(InType)), 0, 0, 0};
        DataCopyPadExtParams<InType> abPad{
            true, 0, static_cast<uint8_t>(BF16_NUM_PER_BLOCK - groupSize), 0};
        DataCopyPad(abLocal, aGm_[offset], abParams, abPad);
        DataCopyPad(abLocal[BF16_NUM_PER_BLOCK], bGm_[offset], abParams, abPad);
        qQueue_.EnQue(abLocal);
        abLocal = qQueue_.DeQue<InType>();
        Cast(reduceWorkUb_, abLocal, RoundMode::CAST_NONE, 2 * BF16_NUM_PER_BLOCK);
        qQueue_.FreeTensor(abLocal);
        PipeBarrier<PIPE_V>();

        auto paramsLocal = qQueue_.AllocTensor<float>();
        DataCopyExtParams floatParams{1, static_cast<uint32_t>(groupSize * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> floatPad{
            true, 0, static_cast<uint8_t>(FP32_NUM_PER_BLOCK - groupSize), 0.0f};
        DataCopyPad(paramsLocal, aLogGm_[firstHv], floatParams, floatPad);
        DataCopyPad(paramsLocal[FP32_NUM_PER_BLOCK], dtBiasGm_[firstHv], floatParams, floatPad);
        qQueue_.EnQue(paramsLocal);
        paramsLocal = qQueue_.DeQue<float>();
        DataCopy(scalarUb_[GATING_A_LOG_OFFSET], paramsLocal, 2 * FP32_NUM_PER_BLOCK);
        qQueue_.FreeTensor(paramsLocal);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrefetchState(
        int64_t stateIdx, uint32_t batch, uint32_t hv, uint32_t vStart, uint32_t curBV)
    {
        auto stateLocal = stateQueue_.AllocTensor<StateType>();
        const uint64_t stateOffset = static_cast<uint64_t>(stateIdx) * td_->stateSlotStride +
                                     static_cast<uint64_t>(hv) * td_->stateHeadStride +
                                     static_cast<uint64_t>(vStart) * td_->k;
        DataCopyExtParams stateParams{static_cast<uint16_t>(curBV),
                                       static_cast<uint32_t>(td_->k * sizeof(StateType)), 0, 0, 0};
        DataCopyPadExtParams<StateType> statePad{true, 0, static_cast<uint8_t>(alignK_ - td_->k), 0};
        DataCopyPad(stateLocal, stateGm_[stateOffset], stateParams, statePad);

        const uint64_t mixedBase = static_cast<uint64_t>(batch) * td_->mixedStride;
        const uint64_t vOffset = mixedBase + static_cast<uint64_t>(2 * td_->h * td_->k) +
                                 static_cast<uint64_t>(hv) * td_->v + vStart;
        auto vLocal = stateLocal.template ReinterpretCast<InType>()[stateAuxOffset_];
        DataCopyExtParams vParams{1, static_cast<uint32_t>(curBV * sizeof(InType)), 0, 0, 0};
        const uint32_t alignedCurBV = Ceil(curBV, BF16_NUM_PER_BLOCK) * BF16_NUM_PER_BLOCK;
        DataCopyPadExtParams<InType> vPad{true, 0, static_cast<uint8_t>(alignedCurBV - curBV), 0};
        DataCopyPad(vLocal, mixedQkvGm_[vOffset], vParams, vPad);
        stateQueue_.EnQue(stateLocal);
    }

    __aicore__ inline LocalTensor<StateType> DequeueState(uint32_t curBV)
    {
        auto stateLocal = stateQueue_.DeQue<StateType>();
        auto vLocal = stateLocal.template ReinterpretCast<InType>()[stateAuxOffset_];
        Cast(vUb_, vLocal, RoundMode::CAST_NONE, alignBV_);
        if constexpr (!std::is_same<StateType, float>::value) {
            Cast(hUb_, stateLocal, RoundMode::CAST_NONE, alignK_ * curBV);
        }
        PipeBarrier<PIPE_V>();
        return stateLocal;
    }

    __aicore__ inline void MulBlockScalar(LocalTensor<float> &dst, const LocalTensor<float> &src,
                                          const LocalTensor<float> &scalarBlock, uint32_t count)
    {
        constexpr uint8_t repeatStride = REPEAT_LEN / FP32_NUM_PER_BLOCK;
        uint32_t offset = 0;
        while (count - offset >= REPEAT_LEN) {
            const uint32_t repeatTime = MinU32(255, (count - offset) / REPEAT_LEN);
            Mul(dst[offset], src[offset], scalarBlock, REPEAT_LEN, static_cast<uint8_t>(repeatTime),
                {1, 1, 0, repeatStride, repeatStride, 0});
            offset += repeatTime * REPEAT_LEN;
        }
        if (offset < count) {
            Mul(dst[offset], src[offset], scalarBlock, count - offset, 1,
                {1, 1, 0, repeatStride, repeatStride, 0});
        }
    }

    __aicore__ inline void Normalize(LocalTensor<float> &x, uint32_t len)
    {
        Mul(reduceWorkUb_, x, x, len);
        PipeBarrier<PIPE_V>();
        for (uint32_t offset = REPEAT_LEN; offset < len; offset += REPEAT_LEN) {
            const uint32_t curLen = MinU32(REPEAT_LEN, len - offset);
            Add(reduceWorkUb_, reduceWorkUb_, reduceWorkUb_[offset], curLen);
            PipeBarrier<PIPE_V>();
        }
        WholeReduceSum(scalarUb_, reduceWorkUb_, REPEAT_LEN, 1, 1, 1, FP32_NUM_PER_BLOCK);
        PipeBarrier<PIPE_V>();
        Adds(scalarUb_, scalarUb_, EPS, 1);
        PipeBarrier<PIPE_V>();
        Rsqrt(scalarUb_, scalarUb_, 1);
        PipeBarrier<PIPE_V>();
        Brcb(reduceWorkUb_, scalarUb_, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        MulBlockScalar(x, x, reduceWorkUb_, len);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void MatVecMul(const LocalTensor<float> &matrix, const LocalTensor<float> &vec,
                                     LocalTensor<float> &dst, uint32_t rows)
    {
        const uint8_t repeatStride = alignK_ / FP32_NUM_PER_BLOCK;
        const uint8_t repeatTime = static_cast<uint8_t>(rows);
        for (uint32_t i = 0; i < alignK_; i += REPEAT_LEN) {
            const uint64_t mask = MinU32(REPEAT_LEN, alignK_ - i);
            Mul(dst[i], matrix[i], vec[i], mask, repeatTime,
                {1, 1, 1, repeatStride, repeatStride, 0});
        }
    }

    __aicore__ inline void RankOneUpdate(LocalTensor<float> &dst, const LocalTensor<float> &vec,
                                         const LocalTensor<float> &rowScalars, uint32_t rows)
    {
        const uint8_t dstRepeatStride = alignK_ / FP32_NUM_PER_BLOCK;
        for (uint32_t i = 0; i < alignK_; i += REPEAT_LEN) {
            const uint64_t mask = MinU32(REPEAT_LEN, alignK_ - i);
            MulAddDst(dst[i], vec[i], rowScalars, mask, static_cast<uint8_t>(rows),
                      {1, 1, 0, dstRepeatStride, 0, 1});
        }
    }

    __aicore__ inline void ReduceRows(LocalTensor<float> &dst, LocalTensor<float> &src, uint32_t rows)
    {
        uint32_t shape[2] = {rows, alignK_};
        ReduceSum<float, Pattern::Reduce::AR, true>(dst, src, shape, true);
    }

    __aicore__ inline void PrepareQK()
    {
        Normalize(qUb_, td_->k);
        Normalize(kUb_, td_->k);
        Muls(qUb_, qUb_, td_->scale, alignK_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepareGatingGroup(uint32_t groupSize)
    {
        Add(scalarUb_[GATING_X_OFFSET], reduceWorkUb_, scalarUb_[GATING_DT_BIAS_OFFSET], groupSize);
        Mins(scalarUb_[GATING_TMP_OFFSET], scalarUb_[GATING_X_OFFSET], td_->softplusThreshold, groupSize);
        Exp(scalarUb_[GATING_TMP_OFFSET], scalarUb_[GATING_TMP_OFFSET], groupSize);
        Adds(scalarUb_[GATING_TMP_OFFSET], scalarUb_[GATING_TMP_OFFSET], 1.0f, groupSize);
        Ln(scalarUb_[GATING_TMP_OFFSET], scalarUb_[GATING_TMP_OFFSET], groupSize);
        Max(scalarUb_[GATING_X_OFFSET], scalarUb_[GATING_X_OFFSET], scalarUb_[GATING_TMP_OFFSET], groupSize);

        Exp(scalarUb_[GATING_A_LOG_OFFSET], scalarUb_[GATING_A_LOG_OFFSET], groupSize);
        Muls(scalarUb_[GATING_A_LOG_OFFSET], scalarUb_[GATING_A_LOG_OFFSET], -1.0f, groupSize);
        Mul(scalarUb_[GATING_X_OFFSET], scalarUb_[GATING_X_OFFSET], scalarUb_[GATING_A_LOG_OFFSET], groupSize);
        Exp(scalarUb_[GATING_EXP_G_OFFSET], scalarUb_[GATING_X_OFFSET], groupSize);

        Sigmoid(scalarUb_[GATING_BETA_OFFSET], reduceWorkUb_[BF16_NUM_PER_BLOCK], groupSize);
        PipeBarrier<PIPE_V>();
        Brcb(scalarUb_[SCALAR_BETA_BRCB_OFFSET], scalarUb_[GATING_BETA_OFFSET], 1, {1, 8});
        Brcb(scalarUb_[SCALAR_EXP_G_BRCB_OFFSET], scalarUb_[GATING_EXP_G_OFFSET], 1, {1, 8});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeTile(uint32_t curBV, const LocalTensor<float> &betaBlock,
                                       const LocalTensor<float> &expGBlock)
    {
        if constexpr (std::is_same<StateType, float>::value) {
            MulBlockScalar(hUb_, hTmpUb_, expGBlock, alignK_ * curBV);
        } else {
            MulBlockScalar(hUb_, hUb_, expGBlock, alignK_ * curBV);
        }
        PipeBarrier<PIPE_V>();

        MatVecMul(hUb_, kUb_, hTmpUb_, curBV);
        PipeBarrier<PIPE_V>();
        ReduceRows(deltaUb_, hTmpUb_, curBV);
        PipeBarrier<PIPE_V>();
        Sub(deltaUb_, vUb_, deltaUb_, curBV);
        PipeBarrier<PIPE_V>();
        MulBlockScalar(deltaUb_, deltaUb_, betaBlock, curBV);
        PipeBarrier<PIPE_V>();

        Brcb(hTmpUb_, deltaUb_, static_cast<uint8_t>(Ceil(curBV, FP32_NUM_PER_BLOCK)), {1, 8});
        PipeBarrier<PIPE_V>();
        RankOneUpdate(hUb_, kUb_, hTmpUb_, curBV);
        PipeBarrier<PIPE_V>();
        MatVecMul(hUb_, qUb_, hTmpUb_, curBV);
        PipeBarrier<PIPE_V>();
        ReduceRows(outUb_, hTmpUb_, curBV);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Store(int64_t stateIdx, uint32_t batch, uint32_t hv, uint32_t vStart, uint32_t curBV,
                                 LocalTensor<StateType> stateLocal)
    {
        auto outLocal = stateLocal.template ReinterpretCast<InType>()[stateAuxOffset_];
        Cast(outLocal, outUb_, RoundMode::CAST_RINT, alignBV_);
        if constexpr (!std::is_same<StateType, float>::value) {
            Cast(stateLocal, hUb_, RoundMode::CAST_RINT, alignK_ * curBV);
        }
        stateOutQueue_.EnQue(stateLocal);
        stateLocal = stateOutQueue_.DeQue<StateType>();

        outLocal = stateLocal.template ReinterpretCast<InType>()[stateAuxOffset_];
        const uint64_t outOffset = static_cast<uint64_t>(batch) * td_->outBatchStride +
                                   static_cast<uint64_t>(hv) * td_->v + vStart;
        DataCopyExtParams outParams{
            1, static_cast<uint32_t>(curBV * sizeof(InType)), 0, 0, 0};
        DataCopyPad(outGm_[outOffset], outLocal, outParams);

        const uint64_t stateOffset = static_cast<uint64_t>(stateIdx) * td_->stateSlotStride +
                                     static_cast<uint64_t>(hv) * td_->stateHeadStride +
                                     static_cast<uint64_t>(vStart) * td_->k;
        DataCopyExtParams stateParams{static_cast<uint16_t>(curBV),
                                       static_cast<uint32_t>(td_->k * sizeof(StateType)), 0, 0, 0};
        DataCopyPad(stateOutGm_[stateOffset], stateLocal, stateParams);

        stateOutQueue_.FreeTensor(stateLocal);
    }

    __aicore__ inline void StoreZeroOut(uint32_t batch, uint32_t hv, uint32_t vStart, uint32_t curBV)
    {
        Duplicate(outUb_, 0.0f, alignBV_);
        auto stateLocal = stateOutQueue_.AllocTensor<StateType>();
        auto outLocal = stateLocal.template ReinterpretCast<InType>()[stateAuxOffset_];
        Cast(outLocal, outUb_, RoundMode::CAST_RINT, alignBV_);
        stateOutQueue_.EnQue(stateLocal);
        stateLocal = stateOutQueue_.DeQue<StateType>();
        outLocal = stateLocal.template ReinterpretCast<InType>()[stateAuxOffset_];
        const uint64_t outOffset = static_cast<uint64_t>(batch) * td_->outBatchStride +
                                   static_cast<uint64_t>(hv) * td_->v + vStart;
        DataCopyExtParams outParams{
            1, static_cast<uint32_t>(curBV * sizeof(InType)), 0, 0, 0};
        DataCopyPad(outGm_[outOffset], outLocal, outParams);
        stateOutQueue_.FreeTensor(stateLocal);
    }

    __aicore__ inline void ProcessValueHead(uint32_t batch, uint32_t hv, int64_t stateIdx, uint32_t groupOffset)
    {
        const auto betaBlock = scalarUb_[SCALAR_BETA_BRCB_OFFSET + groupOffset * FP32_NUM_PER_BLOCK];
        const auto expGBlock = scalarUb_[SCALAR_EXP_G_BRCB_OFFSET + groupOffset * FP32_NUM_PER_BLOCK];
        PrefetchState(stateIdx, batch, hv, 0, MinU32(td_->bv, td_->v));
        for (uint32_t vTile = 0; vTile < td_->vTiles; ++vTile) {
            const uint32_t vStart = vTile * td_->bv;
            const uint32_t curBV = MinU32(td_->bv, td_->v - vStart);
            auto stateLocal = DequeueState(curBV);
            if (vTile + 1 < td_->vTiles) {
                const uint32_t nextVStart = vStart + curBV;
                const uint32_t nextBV = MinU32(td_->bv, td_->v - nextVStart);
                PrefetchState(stateIdx, batch, hv, nextVStart, nextBV);
            }
            auto stateOutLocal = stateOutQueue_.AllocTensor<StateType>();
            if constexpr (std::is_same<StateType, float>::value) {
                hUb_ = stateOutLocal;
                hTmpUb_ = stateLocal;
            }
            ComputeTile(curBV, betaBlock, expGBlock);
            Store(stateIdx, batch, hv, vStart, curBV, stateOutLocal);
            stateQueue_.FreeTensor(stateLocal);
        }
    }

    __aicore__ inline void ProcessBatchSegment(uint32_t firstTask, uint32_t segmentSize)
    {
        const uint32_t firstHv = firstTask % td_->hv;
        const uint32_t batch = firstTask / td_->hv;
        const int64_t stateIdx = stateIndicesGm_.GetValue(batch);
        if (stateIdx <= 0) {
            for (uint32_t taskOffset = 0; taskOffset < segmentSize; ++taskOffset) {
                const uint32_t hv = firstHv + taskOffset;
                for (uint32_t vTile = 0; vTile < td_->vTiles; ++vTile) {
                    const uint32_t vStart = vTile * td_->bv;
                    const uint32_t curBV = MinU32(td_->bv, td_->v - vStart);
                    StoreZeroOut(batch, hv, vStart, curBV);
                }
            }
            return;
        }

        LoadGatingInputs(batch, firstHv, segmentSize);
        PrepareGatingGroup(segmentSize);
        const uint32_t headsPerH = td_->hv / td_->h;
        uint32_t segmentOffset = 0;
        while (segmentOffset < segmentSize) {
            const uint32_t hv = firstHv + segmentOffset;
            const uint32_t h = hv / headsPerH;
            const uint32_t remainingInH = headsPerH - hv % headsPerH;
            const uint32_t groupSize =
                MinU32(segmentSize - segmentOffset, remainingInH);
            LoadQK(batch, h);
            PrepareQK();
            for (uint32_t groupOffset = 0; groupOffset < groupSize; ++groupOffset) {
                ProcessValueHead(
                    batch, hv + groupOffset, stateIdx, segmentOffset + groupOffset);
            }
            segmentOffset += groupSize;
        }
    }

    GlobalTensor<InType> mixedQkvGm_;
    GlobalTensor<InType> aGm_;
    GlobalTensor<InType> bGm_;
    GlobalTensor<float> aLogGm_;
    GlobalTensor<float> dtBiasGm_;
    GlobalTensor<StateType> stateGm_;
    GlobalTensor<StateType> stateOutGm_;
    GlobalTensor<int64_t> stateIndicesGm_;
    GlobalTensor<InType> outGm_;
    TPipe *pipe_;
    const FusedGdnDecodeTilingData *td_;
    uint32_t blockIdx_;
    uint32_t blockNum_;
    uint32_t alignK_;
    uint32_t alignBV_;
    uint32_t stateAuxOffset_;
    TQue<QuePosition::VECIN, 1> qQueue_;
    TQue<QuePosition::VECIN, 1> stateQueue_;
    TQue<QuePosition::VECOUT, 1> stateOutQueue_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    LocalTensor<float> qUb_;
    LocalTensor<float> kUb_;
    LocalTensor<float> vUb_;
    LocalTensor<float> hUb_;
    LocalTensor<float> hTmpUb_;
    LocalTensor<float> reduceWorkUb_;
    LocalTensor<float> deltaUb_;
    LocalTensor<float> outUb_;
    LocalTensor<float> scalarUb_;
};

} // namespace FusedGdnDecode

#endif // FUSED_GDN_DECODE_KERNEL_H
