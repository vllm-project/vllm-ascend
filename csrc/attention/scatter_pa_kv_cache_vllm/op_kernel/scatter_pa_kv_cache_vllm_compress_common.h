/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_pa_kv_cache_vllm_compress_common.h
 * \brief
 */
#ifndef SCATTER_PA_KV_CACHE_VLLM_COMPRESS_COMMON_H
#define SCATTER_PA_KV_CACHE_VLLM_COMPRESS_COMMON_H

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_vllm_common.h"

namespace ScatterPaKvCacheVllm {
class ScatterPaKvCacheVllmCompressCommon {
public:
    __aicore__ inline ScatterPaKvCacheVllmCompressCommon()
    {
    }

    __aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 16)
    {
        return y == 0 ? 0 : (x + y - 1) / y * y;
    }

    __aicore__ inline void Init(AscendC::TPipe *pipe, const ScatterPaKvCacheVllmTilingData *__restrict tilingData)
    {
        pipe_ = pipe;
        // init tiling
        numTokens_ = tilingData->numTokens;
        numHeads_ = tilingData->numHead;
        headSizeK_ = tilingData->kHeadSize;
        headSizeV_ = tilingData->vHeadSize;
        numBatchs_ = tilingData->batch;
        strideK_ = tilingData->kStride;
        strideV_ = tilingData->vStride;
        offsetK_ = tilingData->kOffset;
        offsetV_ = tilingData->vOffset;
    }

    template <typename T>
    __aicore__ inline void InitGlobalTensor(AscendC::GlobalTensor<T> &gm, GM_ADDR addr)
    {
        gm.SetGlobalBuffer((__gm__ T *)addr);
    }

    template <typename T, typename B>
    __aicore__ inline void InitTBuf(B &buf, uint32_t len)
    {
        pipe_->InitBuffer(buf, RoundUp(len * sizeof(T), BLOCK_SIZE));
    }

    template <typename T, typename B>
    __aicore__ inline void InitScalarTBuf(B &buf, uint32_t len)
    {
        pipe_->InitBuffer(buf, SCALAR_INPUT_PARAMETERS * RoundUp(len * sizeof(T), BLOCK_SIZE));
    }

    __aicore__ inline void AllocateTask()
    {
        coreNums_ = static_cast<uint32_t>(AscendC::GetBlockNum());
        perCoreTaskNum_ = numTokens_ / coreNums_;
        tailTaskNum_ = numTokens_ % coreNums_;
        blockId_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        startTaskId_ = blockId_ * perCoreTaskNum_;

        if (blockId_ < tailTaskNum_) {
            perCoreTaskNum_++;
            startTaskId_ += blockId_;
        } else {
            startTaskId_ += tailTaskNum_;
        }
    }

    __aicore__ inline void AllocateTaskRope()
    {
        coreNums_ = static_cast<uint32_t>(AscendC::GetBlockNum());
        perCoreTaskNum_ = numTokens_ * TASK_MULTIPLE / coreNums_;
        tailTaskNum_ = numTokens_ * TASK_MULTIPLE % coreNums_;
        blockId_ = static_cast<uint32_t>(AscendC::GetBlockIdx());
        startTaskId_ = blockId_ * perCoreTaskNum_;

        if (blockId_ < tailTaskNum_) {
            perCoreTaskNum_++;
            startTaskId_ += blockId_;
        } else {
            startTaskId_ += tailTaskNum_;
        }
    }

    template <typename T>
    __aicore__ inline void CopyKvCache(AscendC::GlobalTensor<T> &src, AscendC::LocalTensor<T> &ubAddr,
                                       AscendC::GlobalTensor<T> &dst, uint64_t start, uint64_t cacheStart,
                                       AscendC::DataCopyParams &copyParamsIn, AscendC::DataCopyParams &copyParamsOut)
    {
        DataCopy(ubAddr, src[start], copyParamsIn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
        DataCopy(dst[cacheStart], ubAddr, copyParamsOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    template <typename T>
    __aicore__ inline void CopyKvCacheDoubleBuf(AscendC::GlobalTensor<T> &src, AscendC::LocalTensor<T> &ubAddr,
                                                AscendC::GlobalTensor<T> &dst, event_t eventID, uint64_t start,
                                                uint64_t cacheStart, AscendC::DataCopyParams &copyParamsIn,
                                                AscendC::DataCopyParams &copyParamsOut)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
        DataCopy(ubAddr, src[start], copyParamsIn);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventID);
        DataCopy(dst[cacheStart], ubAddr, copyParamsOut);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
    }

    template <typename T>
    __aicore__ inline void RopeInitscalarBuf()
    {
        uint32_t scalarBufStart = 0;
        tokenSize_ = 1 * static_cast<uint64_t>(headSizeK_);
        numBlocks_ = static_cast<uint32_t>(tokenSize_) * sizeof(T) / BLOCK_SIZE;
        copyFp16Num = TOTAL_BUFFER_SIZE / sizeof(T) / BUFFER_NUM; // partA和partC搬运的数据量

        InitScalarTBuf<int32_t>(scalarBuf_, numHeads_ * numBatchs_);
        scalarLocal_ = scalarBuf_.Get<int32_t>();

        winsLocal_ = scalarLocal_[scalarBufStart];
        scalarBufStart += RoundUp(numHeads_ * numBatchs_ * sizeof(int32_t), BLOCK_SIZE) / sizeof(int32_t);

        seqLenLocal_ = scalarLocal_[scalarBufStart];
        scalarBufStart += RoundUp(numBatchs_ * sizeof(int32_t), BLOCK_SIZE) / sizeof(int32_t);

        csumSeqLenLocal_ = scalarLocal_[scalarBufStart];
        scalarBufStart += RoundUp(numBatchs_ * sizeof(int32_t), BLOCK_SIZE) / sizeof(int32_t);

        offsetLocal_ = scalarLocal_[scalarBufStart];
    }

    template <typename T>
    __aicore__ inline void
    RopeInitBuf(AscendC::LocalTensor<T> &copyLocalPing_, AscendC::LocalTensor<T> &copyLocalPong_,
                AscendC::LocalTensor<T> &tokenLocal_, AscendC::LocalTensor<float> &tokenCompressLocal_,
                AscendC::LocalTensor<float> &tokenSumLocal_, uint16_t &repeatDivisor, uint16_t &addNumPerLoop)
    {
        uint32_t computeBufStart = 0;
        uint32_t copyBufStart = 0;

        // 掩码模式一次计算256B，需要考虑对齐
        repeatDivisor = (tokenSize_ <= MAX_MASK_NUM && tokenSize_ % MAX_MASK_NUM == 0) ? MAX_MASK_NUM : MIN_MASK_NUM;
        addNumPerLoop = PER_FLOAT_NUM / tokenSize_;

        InitTBuf<T>(totalBuf_, TOTAL_BUFFER_SIZE / sizeof(T));
        tokenLocal_ = totalBuf_.GetWithOffset<T>(static_cast<uint32_t>(MAX_FLOAT_NUM), computeBufStart);

        computeBufStart += MAX_FLOAT_NUM * sizeof(T);
        tokenCompressLocal_ = totalBuf_.GetWithOffset<float>(static_cast<uint32_t>(MAX_FLOAT_NUM), computeBufStart);

        computeBufStart += MAX_FLOAT_NUM * sizeof(float);
        tokenSumLocal_ = totalBuf_.GetWithOffset<float>(static_cast<uint32_t>(MAX_FLOAT_NUM), computeBufStart);

        copyLocalPing_ = totalBuf_.GetWithOffset<T>(static_cast<uint32_t>(copyFp16Num), copyBufStart);
        copyBufStart += TOTAL_BUFFER_SIZE / BUFFER_NUM;
        copyLocalPong_ = totalBuf_.GetWithOffset<T>(static_cast<uint32_t>(copyFp16Num), copyBufStart);
    }

    template <typename T>
    __aicore__ inline void OmniInitBuf(AscendC::LocalTensor<T> &copyLocalPing_, AscendC::LocalTensor<T> &copyLocalPong_)
    {
        uint32_t copyBufStart = 0;
        InitTBuf<T>(totalBuf_, TOTAL_BUFFER_SIZE / sizeof(T));

        copyLocalPing_ = totalBuf_.GetWithOffset<T>(static_cast<uint32_t>(copyFp16Num), copyBufStart);
        copyBufStart += TOTAL_BUFFER_SIZE / BUFFER_NUM;
        copyLocalPong_ = totalBuf_.GetWithOffset<T>(static_cast<uint32_t>(copyFp16Num), copyBufStart);
    }

    __aicore__ inline void PreProcess(AscendC::GlobalTensor<int32_t> &winsInputGt_,
                                      AscendC::GlobalTensor<int32_t> &seqLenInputGt_,
                                      AscendC::GlobalTensor<int32_t> &offsetInputGt_)
    {
        DataCopy(winsLocal_, winsInputGt_, RoundUp(numHeads_ * numBatchs_, BLOCK_SIZE / sizeof(int32_t)));
        AscendC::PipeBarrier<PIPE_MTE2>();
        DataCopy(seqLenLocal_, seqLenInputGt_, RoundUp(numBatchs_, BLOCK_SIZE / sizeof(int32_t)));
        AscendC::PipeBarrier<PIPE_MTE2>();
        DataCopy(offsetLocal_, offsetInputGt_, RoundUp(numHeads_ * numBatchs_, BLOCK_SIZE / sizeof(int32_t)));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID1);
        csumSeqLenLocal_.SetValue(0, 0);
        for (uint32_t i = 1; i < numBatchs_; i++) { // 获取累加的seqlen
            csumSeqLenLocal_.SetValue(i, csumSeqLenLocal_.GetValue(i - 1) + seqLenLocal_.GetValue(i - 1));
        }
    }

    __aicore__ inline void ParamSet(AscendC::GlobalTensor<int32_t> &slotInputGt_, uint32_t startTaskOffset)
    {
        slotValue = slotInputGt_.GetValue(startTaskOffset);
        batchId = (startTaskOffset) / numHeads_;
        headId = (startTaskOffset) % numHeads_;
        headWin = static_cast<uint32_t>(winsLocal_.GetValue(startTaskOffset));
        seqLen = static_cast<uint32_t>(seqLenLocal_.GetValue(batchId));
        consumSeqLen = static_cast<uint32_t>(csumSeqLenLocal_.GetValue(batchId));
        offsetPerLine = (numHeads_ - 1) * numBlocks_; // 每一个token间隔(num_heads-1)*tokenSize 此处单位为32B
        offsetIdx = offsetLocal_.GetValue(startTaskOffset);
        offsetIdx = (offsetIdx == -1 || headWin == 0) ? seqLen : offsetIdx; // 当等于-1时, 跳过当前头压缩
    }

protected:
    uint32_t numTokens_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t headSizeK_ = 0;
    uint32_t headSizeV_ = 0;
    uint32_t numBatchs_ = 0;
    uint32_t strideK_ = 0;
    uint32_t strideV_ = 0;
    uint32_t offsetK_ = 0;
    uint32_t offsetV_ = 0;

    uint32_t coreNums_ = 0;
    uint32_t perCoreTaskNum_ = 0;
    uint32_t tailTaskNum_ = 0;
    uint32_t blockId_ = 0;
    uint32_t startTaskId_ = 0;

    uint64_t tokenSize_ = 0;
    uint32_t numBlocks_ = 0;
    uint32_t copyFp16Num = 0;
    AscendC::LocalTensor<int32_t> scalarLocal_;     // 存放下面四个标量的总LocalTensor
    AscendC::LocalTensor<int32_t> winsLocal_;       // 临时存放 wins
    AscendC::LocalTensor<int32_t> seqLenLocal_;     // 临时存放 seqLen
    AscendC::LocalTensor<int32_t> csumSeqLenLocal_; // 临时存放 comsumSeqLen
    AscendC::LocalTensor<int32_t> offsetLocal_;     // 临时存放 offset
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> totalBuf_;

    uint32_t slotValue = 0;
    uint32_t batchId = 0;
    uint32_t headId = 0;
    uint32_t headWin = 0;
    uint32_t seqLen = 0;
    uint32_t consumSeqLen = 0;
    uint32_t offsetPerLine = 0;
    int32_t offsetIdx = 0;
    AscendC::TPipe *pipe_;
};
} // namespace ScatterPaKvCacheVllm
#endif
