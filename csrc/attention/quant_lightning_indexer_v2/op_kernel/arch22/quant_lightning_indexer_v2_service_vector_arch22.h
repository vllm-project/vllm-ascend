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
 * \file quant_lightning_indexer_v2_service_vector_arch22.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
#define QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common_arch22.h"
#include "quant_lightning_indexer_v2_vector.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
using namespace QLIV2ServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t BASE_TOPK_VALUE_IDX_SIZE = 4096;
constexpr uint32_t ELE_NUM_32 = 32;
constexpr uint32_t ELE_NUM_128 = 128;
constexpr uint32_t ELE_NUM_512 = 512;

template <typename QLIV2T>
class QLIV2Vector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIV2T::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIV2T::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIV2T::pageAttention;
    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline QLIV2Vector(){};
    __aicore__ inline void ProcessVec0(const QLIV2Common::RunInfo &info);
    __aicore__ inline void ProcessVec1(const QLIV2Common::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLIV2Common::ConstInfo &constInfo,
                                      const struct QLIV2Common::LdSplitCoreInfo &ldInfo,
                                      const QLIV2TilingData *__restrict tilingData);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm, GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                  GlobalTensor<float> vec1ResGm);
    __aicore__ inline void InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexInGm,
                                                  GlobalTensor<int32_t> candidateTopkIndexOutGm,
                                                  GlobalTensor<int32_t> outputIdxOffsetGm,
                                                  bool outputIdxOffsetValid);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<half> weightsGm, GlobalTensor<half> qScaleGm,
                                              GlobalTensor<half> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline int32_t AlignS2(int32_t cuS2Len);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<half> weightsGm;
    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    GlobalTensor<half> vec0OutGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> candidateTopkIndexInGm;
    GlobalTensor<int32_t> candidateTopkIndexOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================

private:
    __aicore__ inline void GetKeyScale(const QLIV2Common::RunInfo &runInfo, const LocalTensor<half> &resUb,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // candidate (two-level topk)
    __aicore__ inline int32_t CountGE(const LocalTensor<float> &sortedDesc, int32_t n, float x);
    __aicore__ inline void BuildCandidateMask(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                              int32_t cuBaseS2Idx, int32_t innerS1Idx);
    // mode=4 (consumer skip): 候选块内紧凑收集, 返回紧凑排序长度 S (0=本 tile 无候选块)
    __aicore__ inline int32_t BuildCandidateCompact(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                                    int32_t cuBaseS2Idx, int32_t innerS1Idx,
                                                    const LocalTensor<float> &mmInUb, int32_t cuS2Len);
    __aicore__ inline void ProcessCandBlockTopk(const QLIV2Common::RunInfo &info, int32_t cuS1Idx, int32_t cuS2Len,
                                                int32_t cuS2LenVecAlign, int32_t cuRealAcSeq, int32_t innerS1Idx);
    __aicore__ inline void CopyOutCandTopkIndex(const QLIV2Common::RunInfo &info, int32_t cuS1Idx, int32_t innerS1Idx);
    // candidate 新增向量 op 的 64 元素分块封装 (本版本 Level-2 count > 64 的 vmax 等指令会触发 aicore 异常)
    __aicore__ inline void VecMaxPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    __aicore__ inline void VecMinPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    __aicore__ inline void VecAbs(const LocalTensor<float> &dst, const LocalTensor<float> &src, int32_t n);
    __aicore__ inline void VecMinsScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src, float v,
                                         int32_t n);
    __aicore__ inline void VecMaxsScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src, float v,
                                         int32_t n);
    __aicore__ inline void VecSubPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    __aicore__ inline void VecAddsScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src, float v,
                                         int32_t n);
    __aicore__ inline void VecMulPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    // tmp buff for candidate (two-level topk)
    TBuf<TPosition::VECCALC> blockSortOutBuf_; // mode=1: 块级 topk 累加器 [CeilDiv(s1BaseSize,2), 2048, 2]
    TBuf<TPosition::VECCALC> candBuf_;         // mode=2: 排序后的候选块索引 [2048, 2] (value+idx)
    TBuf<TPosition::VECCALC> candConstBuf_;    // mode=1/2: negHuge 常量 [2048]

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> globalBlockTopkUb_;
    LocalTensor<float> candIsOut_;       // mode=2: position 级 0/1 (1=候选外), 仅用于分数降级 (R11 leak)
    LocalTensor<float> candNegHuge_;     // -1e30 常量
    GlobalTensor<int32_t> outputIdxOffsetGm_;        // A15: 每行输出索引偏移 (仅 sparse_indices)
    bool isOutputIdxOffsetValid_ = false;            // A15: offset 是否传入 (经 InitVecCandidateTensor 传递)

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;

    // para for LD
    uint32_t mrgListNum_ = 4;

    struct QLIV2Common::ConstInfo constInfo_;
    struct QLIV2Common::LdSplitCoreInfo ldInfo_;
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::GetKeyScale(const QLIV2Common::RunInfo &runInfo,
                                                        const LocalTensor<half> &resUb, int64_t batchId,
                                                        int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    if constexpr (PAGE_ATTENTION) {
        // A11: k_scale 0 轴非连续 — 块基址 = 表值 x keyDequantScaleStride0 (真实 stride);
        // 0 时兜底原紧凑公式 (kCacheBlockSize_), 现网行为不变 (R8)
        // Paged hybrid caches can span more than 2^31 scale elements. Promote
        // before multiplication so high physical block IDs do not wrap.
        int64_t kScaleBlkStride = constInfo_.keyDequantScaleStride0 != 0 ?
                                      static_cast<int64_t>(constInfo_.keyDequantScaleStride0) : kCacheBlockSize_;
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        int32_t blockTableBatchOffset = batchId * maxBlockNumPerBatch_;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(half);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb,
                                 kScaleGm[blockId * kScaleBlkStride + startBlockTableOffset], copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(half);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(half);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb[resUbBaseOffset + i * kCacheBlockSize_],
                                 kScaleGm[blockId * kScaleBlkStride], copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(half);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(resUb, kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inQueue_, 2, s2BaseSize_ * sizeof(float) * 2);                                    // 32KB
    pipe->InitBuffer(outQueue_, 1, BASE_TOPK * sizeof(float));                                         // 8 KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                        // 8 KB
    pipe->InitBuffer(tmpBuf_, 64 * 1024);                                                              // 64KB
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE * sizeof(float)); // 32KB

    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    globalTopkNum_ = 0;

    // candidate (two-level topk) 按模式分配, mode=3 不占用额外 UB
    // UB 预算 (192KB): 基础 144KB + mode=1 blockSortOutBuf 32KB = 176KB / mode=2 candBuf 16KB + candConstBuf 8KB = 168KB
    if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
        pipe->InitBuffer(blockSortOutBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE * sizeof(float)); // 32KB
        globalBlockTopkUb_ = blockSortOutBuf_.Get<float>();
        InitSortOutBuf(globalBlockTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);
    } else if (constInfo_.candidateMode == CANDIDATE_MODE_CONSUMER ||
               constInfo_.candidateMode == CANDIDATE_MODE_CONSUMER_SKIP) {
        // R6 修复: candBuf 按行分区 (每 AIV 处理 CeilDiv(s1BaseSize,2)=2 行, 行间 tile0 重排序
        // 会互相覆盖) — 2 行 x 2048 对 x 8B = 32KB; mode=2/4 UB 预算 144+32+8=184KB <= 192KB
        // (mode=4 的 negHuge 实际不消费, 但保持与 mode=2 相同的分配以最小化分叉)
        pipe->InitBuffer(candBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2 * sizeof(float));
        pipe->InitBuffer(candConstBuf_, BASE_TOPK * sizeof(float)); // 8KB: negHuge
        candNegHuge_ = candConstBuf_.Get<float>();
        Duplicate(candNegHuge_.template ReinterpretCast<int32_t>(), QLIV2ServiceVec::NEG_HUGE_F32, BASE_TOPK);
        PipeBarrier<PIPE_V>();
    }

    // 基本块执行前初始化UB和GM
    // step1. 初始化一个有序索引 0 - s2BaseSize_
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // step2. globalTopkUb_ [CeilDiv(s1BaseSize_, 2), BASE_TOPK, 2]   -inf,-1
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, BASE_TOPK_VALUE_IDX_SIZE * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldTmpBuf_, BASE_TOPK_VALUE_IDX_SIZE * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitParams(const struct QLIV2Common::ConstInfo &constInfo,
                                                       const struct QLIV2Common::LdSplitCoreInfo &ldInfo,
                                                       const QLIV2TilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    this->ldInfo_ = ldInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize; // 4
    s2BaseSize_ = constInfo.s2BaseSize; // 2048
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecInputTensor(GlobalTensor<half> weightsGm,
                                                               GlobalTensor<half> qScaleGm, GlobalTensor<half> kScaleGm,
                                                               GlobalTensor<int32_t> indiceOutGm,
                                                               GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm,
                                                                   GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                                   GlobalTensor<float> vec1ResGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec0OutGm = vec0OutGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexInGm,
                                                                   GlobalTensor<int32_t> candidateTopkIndexOutGm,
                                                                   GlobalTensor<int32_t> outputIdxOffsetGm,
                                                                   bool outputIdxOffsetValid)
{
    this->candidateTopkIndexInGm = candidateTopkIndexInGm;
    this->candidateTopkIndexOutGm = candidateTopkIndexOutGm;
    this->outputIdxOffsetGm_ = outputIdxOffsetGm;  // A15
    this->isOutputIdxOffsetValid_ = outputIdxOffsetValid;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::AllocEventID()
{}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::FreeEventID()
{}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    QLIV2ServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);
    // mode=1: candidate_topk_index 同步填 -1 (行偏移 = sparse 偏移 / sparseCount * candidateTopkBlocks)
    if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
        uint64_t candOffset = static_cast<uint64_t>(invalidS1offset) / constInfo_.sparseCount *
                              constInfo_.candidateTopkBlocks;
        LocalTensor<float> candULocal = outQueue_.AllocTensor<float>();
        LocalTensor<int32_t> candIdxLocal = candULocal.template ReinterpretCast<int32_t>();
        Duplicate(candIdxLocal, constInfo_.INVALID_IDX, constInfo_.candidateTopkBlocks);
        outQueue_.EnQue<float>(candULocal);
        candULocal = outQueue_.DeQue<float>();
        QLIV2ServiceVec::CopyOut(candidateTopkIndexOutGm[candOffset], candIdxLocal, constInfo_.candidateTopkBlocks);
        outQueue_.FreeTensor(candULocal);
    }
}


// 64 元素分块封装: 规避 Level-2 大 count 的 vmax/vmin/vabs 等指令异常
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMaxPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Max(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMinPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Min(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecAbs(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                   int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Abs(dst[off], src[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMinsScalar(const LocalTensor<float> &dst,
                                                          const LocalTensor<float> &src, float v, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Mins(dst[off], src[off], v, 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMaxsScalar(const LocalTensor<float> &dst,
                                                          const LocalTensor<float> &src, float v, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Maxs(dst[off], src[off], v, 64);
    }
    PipeBarrier<PIPE_V>();
}


template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecSubPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Sub(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecAddsScalar(const LocalTensor<float> &dst,
                                                          const LocalTensor<float> &src, float v, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Adds(dst[off], src[off], v, 64);
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMulPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    for (int32_t off = 0; off < n; off += 64) {
        Mul(dst[off], src0[off], src1[off], 64);
    }
    PipeBarrier<PIPE_V>();
}

// 统计降序序列 (偶数 lane 为 value) 中 >= x 的元素个数, 二分实现
template <typename QLIV2T>
__aicore__ inline int32_t QLIV2Vector<QLIV2T>::CountGE(const LocalTensor<float> &sortedDesc, int32_t n, float x)
{
    int32_t lo = 0;
    int32_t hi = n;
    while (lo < hi) {
        int32_t mid = (lo + hi) / 2;
        if (sortedDesc.GetValue(2 * mid) >= x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// mode=2 (use_candidate): 加载/排序候选行 (每行首个S2分片), 判定 tile 内块归属,
// 产出 position 级 isOut (fp32 0/1, 1=候选外) 与其 int32 形式
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::BuildCandidateMask(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                                               int32_t cuBaseS2Idx, int32_t innerS1Idx)
{
    int32_t candBlocks = static_cast<int32_t>(constInfo_.candidateTopkBlocks);
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t tileBlkNum = s2BaseSize_ / blockSize;
    int32_t tileBlockBase = cuBaseS2Idx / blockSize;
    // R6 修复: 同核多行 (每 AIV 处理 CeilDiv(s1BaseSize,2)=2 行) 共享 candBuf 时,
    // 后一行的 tile0 重排序会覆盖前行候选 (s2 内层循环按 gS1 块整体推进) — candBuf 按行分区
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    LocalTensor<float> candPairs = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    if (info.isFirstS2InnerLoop) {
        // 加载候选行到 tmp 尾部, 转 fp32 后与有序索引组成 [values | idx] 对并降序排序
        LocalTensor<int32_t> candInt = tmp[12288].template ReinterpretCast<int32_t>();
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = candBlocks * sizeof(int32_t);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        AscendC::DataCopyPad(candInt,
                             candidateTopkIndexInGm[info.candidateOutOffset +
                                                    cuS1Idx * constInfo_.candidateTopkBlocks],
                             copyInParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        PipeBarrier<PIPE_V>();
        Cast(candPairs, candInt, RoundMode::CAST_NONE, candBlocks);
        PipeBarrier<PIPE_V>();
        DataCopy(candPairs[candBlocks].template ReinterpretCast<int32_t>(), globalTopkIndice_, candBlocks);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> candSortTmp = tmp[4096];
        QLIV2ServiceVec::SortAll(candPairs, candSortTmp, candBlocks);
        PipeBarrier<PIPE_V>();
        // V->S 围栏: 排序结果对后续标量 GetValue (CountGE 二分) 可见
        // (PipeBarrier<PIPE_V> 不足以保证标量读到 V 写数据; 参照 CANN topk_v200 的 SetFlag/WaitFlag<V_S> 模式)
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    }
    // 候选与 tile 块号 [tileBlockBase, tileBlockBase+tileBlkNum) 的最小距离, 0 即命中
    // (candBuf 按行分区, R6: 同核行间覆盖已修复)
    LocalTensor<float> candPairs2 = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    int32_t lo = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase));
    int32_t hi = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase + tileBlkNum));
    LocalTensor<float> blkIdxF = tmp[4096];     // [tileBlkNum]
    LocalTensor<float> acc = tmp[4352];         // [tileBlkNum]
    LocalTensor<float> diff = tmp[4608];        // [tileBlkNum]
    Cast(blkIdxF, globalTopkIndice_, RoundMode::CAST_NONE, tileBlkNum);
    PipeBarrier<PIPE_V>();
    Adds(blkIdxF, blkIdxF, static_cast<float>(tileBlockBase), tileBlkNum);
    PipeBarrier<PIPE_V>();
    Duplicate(acc.ReinterpretCast<int32_t>(), QLIV2ServiceVec::POS_INF_F32, tileBlkNum);
    PipeBarrier<PIPE_V>();
    // 降序排序下: 值 >= base+tileBlkNum 占 [0, hi), 落在 tile 内的候选占 [hi, lo), < base (含 -1 pad) 占 [lo, ...)
    for (int32_t j = hi; j < lo; j++) {
        float v = candPairs2.GetValue(2 * j);
        Adds(diff, blkIdxF, -v, tileBlkNum);
        PipeBarrier<PIPE_V>();
        VecAbs(diff, diff, tileBlkNum);
        VecMinPair(acc, acc, diff, tileBlkNum);
    }
    // 距离 0 → 候选块; Brcb 展开到位置级; 距离 clamp 到 1 得 isOut (0/1)
    LocalTensor<float> posDist = tmp[12288]; // candInt 已释放, 可复用
    Brcb(posDist, acc, tileBlkNum / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    VecMinsScalar(posDist, posDist, 1.0f, s2BaseSize_);
    PipeBarrier<PIPE_V>();
    candIsOut_ = posDist;
    // R11 (leak 语义, §1.6): isOut 仅用于分数降级 (pen 链), 索引保留真实位置号 —
    // 候选外可达位置作为 topk 填充泄漏成有效索引 (对齐模型 where(idxs < compress_lens) 语义),
    // 不再把候选外索引改 -1 (原 isOutI32/CAST_RINT 链已删)。
}

// mode=4 (consumer skip): 候选块内紧凑收集 — 把 mode2 的 "全量计算 + 排序前 mask" 升级为
// "skip 紧凑排序": 仅把本 tile 命中的候选块 (典型 ~16 块 = 128 位置) 的 score/位置搬入
// 紧凑缓冲, SortAll 排序对象从 tile 全长 (2048) 降为 S (64 倍数的 AlignS2 形, ≤2048)。
//
// 数学等价性 (与 mode2 输出集合一致): 候选外可达位置在 mode2 下被降级 NEG_HUGE, 仅当行内
// 候选有效位置 < BASE_TOPK 时才泄漏进输出; 而 mode1 产出的候选列表每块均为本行真实块
// (≥1 个有效位置), 总块数 > topkBlocks 时候选有效位置 ≥ 2048*1 = BASE_TOPK (不泄漏),
// 总块数 ≤ topkBlocks 时全部块均为候选 (候选外为空) — 两种情形 skip 与 mask 的 topk
// 集合均一致 (相等 score 的 tie 顺序可能不同, 多重集等价)。
//
// UB 布局 (tmpBuf_ 64KB 时间分区, 与 mask 路径互不干扰):
//   [0, S) 紧凑 score (fp32) | [S, 2S) 紧凑 idx (int32, 全局位置)
//   [2S, 4S+4096) SortAll/MrgSort 级联 tmp (MergeSort 需 (S+2048)*2, S=2048 时恰为 [4096,12288))
//   [8192, 8512) 尾块修正 scratch (5x64, 仅修正阶段活跃, 先于 SortAll)
//   [12288, 12288+s2BaseSize_) 整 tile 位置向量 (复用 candInt 槽位, 其首 tile 加载已完成)
// 搬运用 int32 视图 Adds(+0) (纯 V 管道, bit 精确 — MergeSortVecCopy 同款; 不用 UB->UB
// DataCopy: 其 MTE 管道不被 PipeBarrier<PIPE_V> fence, 见 qvector.h 竞态注释)。
// 尾块内无效位置经 64 窗口向量修正 (identity 于有效/pad 项, 复用 §S4a/R10 的 int32 位型链)。
template <typename QLIV2T>
__aicore__ inline int32_t QLIV2Vector<QLIV2T>::BuildCandidateCompact(const QLIV2Common::RunInfo &info,
                                                                       int32_t cuS1Idx, int32_t cuBaseS2Idx,
                                                                       int32_t innerS1Idx,
                                                                       const LocalTensor<float> &mmInUb,
                                                                       int32_t cuS2Len)
{
    int32_t candBlocks = static_cast<int32_t>(constInfo_.candidateTopkBlocks);
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t tileBlkNum = s2BaseSize_ / blockSize;
    int32_t tileBlockBase = cuBaseS2Idx / blockSize;
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    LocalTensor<float> candPairs = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    if (info.isFirstS2InnerLoop) {
        // 首 tile: 加载候选行并降序排序 (与 BuildCandidateMask 完全一致)
        LocalTensor<int32_t> candInt = tmp[12288].template ReinterpretCast<int32_t>();
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = candBlocks * sizeof(int32_t);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        AscendC::DataCopyPad(candInt,
                             candidateTopkIndexInGm[info.candidateOutOffset +
                                                    cuS1Idx * constInfo_.candidateTopkBlocks],
                             copyInParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        PipeBarrier<PIPE_V>();
        Cast(candPairs, candInt, RoundMode::CAST_NONE, candBlocks);
        PipeBarrier<PIPE_V>();
        DataCopy(candPairs[candBlocks].template ReinterpretCast<int32_t>(), globalTopkIndice_, candBlocks);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> candSortTmp = tmp[4096];
        QLIV2ServiceVec::SortAll(candPairs, candSortTmp, candBlocks);
        PipeBarrier<PIPE_V>();
        // V->S 围栏: 排序结果对后续标量 GetValue (CountGE 二分/块号读取) 可见
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    }
    LocalTensor<float> candPairs2 = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    int32_t lo = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase));
    int32_t hi = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase + tileBlkNum));
    int32_t candCnt = lo - hi;
    if (candCnt <= 0) {
        return 0; // 本 tile 无候选块: 调用方跳过排序与归并 (acc 保持前序 tile 状态)
    }
    int32_t nPairs = candCnt * blockSize;
    // S: 紧凑排序长度 — 64 的倍数 (向量指令 count 对齐) 且为 AlignS2 形 (SortAll 支持)
    int32_t s;
    if (nPairs <= 64) {
        s = 64;
    } else if (nPairs <= 128) {
        s = 128;
    } else if (nPairs <= 256) {
        s = 256;
    } else if (nPairs <= 384) {
        s = 384;
    } else if (nPairs <= 512) {
        s = 512;
    } else if (nPairs <= 1024) {
        s = 1024;
    } else if (nPairs <= 1536) {
        s = 1536;
    } else {
        s = 2048;
    }
    LocalTensor<int32_t> compactScoreI = tmp[0].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> compactIdx = tmp[s].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> mmInUbI = mmInUb.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> fullPos = tmp[12288].template ReinterpretCast<int32_t>();
    // pad 填充 (-inf/-1): S 为 64 倍数, 与 mode2 尾部填充语义一致 (不足 topk 时泄漏为 -1 槽)
    Duplicate(compactScoreI, QLIV2ServiceVec::NEG_INF, s);
    PipeBarrier<PIPE_V>();
    Duplicate(compactIdx, QLIV2ServiceVec::INVALID_INDEX, s);
    PipeBarrier<PIPE_V>();
    // 整 tile 位置向量 [cuBaseS2Idx, cuBaseS2Idx+cuS2Len) (count 语义同主路径 Adds(sortIndice...))
    Adds(fullPos, globalTopkIndice_, cuBaseS2Idx, cuS2Len);
    PipeBarrier<PIPE_V>();
    // 逐候选块搬运 score 与位置 (int32 Adds(+0), bit 精确); 标量 GetValue 读已排序候选
    // (V_S fence 后稳定, 与 mask 路径 CountGE/GetValue 同款)
    int32_t validTail = cuS2Len % blockSize;
    int32_t boundaryBlk = tileBlockBase + cuS2Len / blockSize; // 全局混合块号 (validTail>0 时)
    int32_t boundaryDst = -1;                                   // 该块在紧凑缓冲内的起始 entry
    for (int32_t j = hi; j < lo; j++) {
        int32_t blkNum = static_cast<int32_t>(candPairs2.GetValue(2 * j));
        int32_t bLocal = blkNum - tileBlockBase;
        int32_t dst = (j - hi) * blockSize;
        AscendC::Adds(compactScoreI[dst], mmInUbI[bLocal * blockSize], 0, blockSize);
        AscendC::Adds(compactIdx[dst], fullPos[bLocal * blockSize], 0, blockSize);
        if (validTail != 0 && blkNum == boundaryBlk) {
            boundaryDst = dst;
        }
    }
    PipeBarrier<PIPE_V>();
    // 尾块修正: 混合块内无效位置 (entry 序号 ∈ [vj, blockSize), vj = validTail) -> (-inf 位型, -1)。
    // 注意不可依赖搬运进来的 pos 值判定 (无效槽位搬运的是 fullPos 的垃圾值, 若垃圾 < thr 会漏修),
    // 改用 entry 序号驱动的 64 窗口向量实现: isInval = (e >= vj) && (e < blockSize), 对窗口内
    // 其余 entry (其他候选块/pad 项) 恒等; 禁标量 SetValue (S 管道写不受 PipeBarrier<PIPE_V>
    // fence, 竞态 — 见 mode1 R6 注释)。
    if (boundaryDst >= 0) {
        int32_t w = boundaryDst / 64 * 64; // w+64 <= S 恒成立 (boundaryDst+blockSize <= nPairs <= S 且 S 为 64 倍数)
        LocalTensor<int32_t> eScr = tmp[8192].template ReinterpretCast<int32_t>();  // 窗口内 entry 序号
        LocalTensor<int32_t> thrI = tmp[8256].template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> m1 = tmp[8320].template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> m2 = tmp[8384].template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> tI = tmp[8448].template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> pRaw = tmp[8512].template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> pNeg = tmp[8576].template ReinterpretCast<int32_t>();
        // eScr: 窗口内 entry 相对 boundary 块首的偏移 e = k - (boundaryDst - w)
        // (k 为窗口内下标, 对应 compactIdx[w + k]; e = 0 即 boundary 块首)
        ArithProgression<int32_t>(eScr, w - boundaryDst, 1, 64);
        PipeBarrier<PIPE_V>();
        Duplicate(thrI, validTail, 64);          // vj
        PipeBarrier<PIPE_V>();
        // 指示函数经 clamp 实现, 需 +1 平移避免边界值被 clamp 抹掉:
        // m1 = min(1, max(0, e - vj + 1)) = (e >= vj)   — e = vj 时必须为 1 (第一个无效 entry)
        AscendC::Sub(m1, eScr, thrI, 64);        // e - vj
        PipeBarrier<PIPE_V>();
        AscendC::Adds(m1, m1, 1, 64);            // e - vj + 1
        PipeBarrier<PIPE_V>();
        AscendC::Maxs(m1, m1, 0, 64);
        PipeBarrier<PIPE_V>();
        AscendC::Mins(m1, m1, 1, 64);            // (e >= vj)
        PipeBarrier<PIPE_V>();
        // m2 = min(1, max(0, blockSize - e)) = (e < blockSize) — e = blockSize-1 时必须为 1,
        // e = blockSize (boundary 块外首个 entry) 必须为 0
        Duplicate(m2, blockSize - 1, 64);        // blockSize-1
        PipeBarrier<PIPE_V>();
        AscendC::Sub(m2, m2, eScr, 64);          // (blockSize-1) - e
        PipeBarrier<PIPE_V>();
        AscendC::Adds(m2, m2, 1, 64);            // blockSize - e
        PipeBarrier<PIPE_V>();
        AscendC::Maxs(m2, m2, 0, 64);
        PipeBarrier<PIPE_V>();
        AscendC::Mins(m2, m2, 1, 64);            // (e < blockSize)
        PipeBarrier<PIPE_V>();
        AscendC::Mul(tI, m1, m2, 64);            // isInval (0/1)
        PipeBarrier<PIPE_V>();
        AscendC::Adds(pRaw, compactIdx[w], 1, 64); // pos + 1
        PipeBarrier<PIPE_V>();
        AscendC::Mul(pRaw, pRaw, tI, 64);          // (pos+1)*isInval
        PipeBarrier<PIPE_V>();
        AscendC::Sub(compactIdx[w], compactIdx[w], pRaw, 64); // 无效 pos -> -1
        PipeBarrier<PIPE_V>();
        AscendC::Duplicate(pRaw, 0, 64);
        PipeBarrier<PIPE_V>();
        AscendC::Sub(pRaw, pRaw, tI, 64);          // -isInval
        PipeBarrier<PIPE_V>();
        AscendC::Adds(pNeg, compactScoreI[w], 8388608, 64); // score - NINF_bits (NINF=0xFF800000)
        PipeBarrier<PIPE_V>();
        AscendC::Mul(pNeg, pNeg, pRaw, 64);
        PipeBarrier<PIPE_V>();
        AscendC::Add(compactScoreI[w], compactScoreI[w], pNeg, 64); // 无效 score -> -inf 位型
        PipeBarrier<PIPE_V>();
    }
    return s;
}

// mode=1 (is_candidate_source): 块内 amax (log2(blockSize) 轮 Max 树) + pin 尾块 + 块级排序/归并
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessCandBlockTopk(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                                                 int32_t cuS2Len, int32_t cuS2LenVecAlign,
                                                                 int32_t cuRealAcSeq, int32_t innerS1Idx)
{
            int32_t blkLen = cuS2LenVecAlign;
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t blockNum = blkLen / blockSize;                         // 含尾块 (pad -inf 后按对齐长度计)
    int32_t realBlockNum = (cuS2Len + blockSize - 1) / blockSize; // 含有效位置的块数
    // 所有向量 count 按 64 对齐 (非对齐 count 的向量指令在本架构触发 aicore 异常);
    // 且块数必须过 AlignS2 (32*(4^n)*m, m<=3): SortAll 的 MrgSort 循环按 mrgGroups/4 缩减,
    // 非 4 幂组数 (如 192 块=6 组) 会整组丢失 (实测 s2=1536 丢块 128..191 段)
    int32_t blockNumPad = (blockNum < 64) ? 64 : AlignS2(blockNum);
    int32_t tileBlockBase = info.s2Idx * s2BaseSize_ / blockSize;
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    // 块内归约: vcgmax 每 32B 块 (8 fp32) 出 1 个 max, 紧凑输出 [blockNum]
    LocalTensor<float> blkScore = tmp[6144];
    // R10: blkLen=96 是 AlignS2 输出中唯一非 64 倍数的值 (≤128 段对齐到 32 的倍数),
    // 96/64 整除截断为 1 只归约 [0,64) — 块 8..11 残留上一 tile/行的 stale 分数
    // (实测 big128k_b2_varlen b1 行 73 tile 56: 块 14344 拿到 tile 55 块 14088 的
    // 3.5568, 虚高挤掉 2048 名边界块 2760; 小 shape 总块数≤2048 集合不变故未暴露)。
    // 改 CeilDiv 覆盖全部块; 多归约的 [96,128) stale 只落在 pad 块槽位, 被 -inf
    // 位型链位精确覆盖, 无害。
    int32_t brmRepeat = CeilDiv(blkLen, 64);
    BlockReduceMax(blkScore, tmp[0], brmRepeat, 64, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    int32_t lastBlk = (cuRealAcSeq - 1) / blockSize;
    int32_t pinLocal = lastBlk - tileBlockBase;
    // 块级 [scores | idx] 对: idx = tileBlockBase + j; j >= realBlockNum 的 pad 块置 -1
    // (禁止标量 SetValue: 标量写不受 PipeBarrier<PIPE_V> 围栏, 与后续 V 管道读存在确定性竞态
    //  (实测 innerS1Idx=1 行的填充被 SortAll 抢跑覆盖); 禁止 s32->f32 Cast (v220 无此组合)。
    //  改为纯 int32 向量算术: idx' = idx - (idx+1)*isPad, isPad = clamp(idx - thr, 0, 1),
    //  thr = base + realBlockNum - 1; 所有指令 count=blockNumPad (64 对齐), 按 64 分块避免 mask 寄存器限制。
    //  scratch 用 mode=1 独占区 [14336,15360) (mode=2 的 isOutI32 已随 R11 leak 改造删除),
    //  避开主路径 tmpSortBuf [4096,12288), 消除跨迭代残留读的隐患)
    LocalTensor<int32_t> idxScr = tmp[14336].template ReinterpretCast<int32_t>();      // 等差源
    LocalTensor<int32_t> thrI = tmp[14592].template ReinterpretCast<int32_t>();        // 阈值
    LocalTensor<int32_t> offI = tmp[14848].template ReinterpretCast<int32_t>();        // idx-thr -> isPad
    LocalTensor<int32_t> tI = tmp[15104].template ReinterpretCast<int32_t>();          // (idx+1)*isPad
    LocalTensor<int32_t> blkIdx = tmp[6144 + blockNumPad].template ReinterpretCast<int32_t>(); // 最终 idx
    ArithProgression<int32_t>(idxScr, static_cast<int32_t>(tileBlockBase), 1, blockNumPad);
    Duplicate(thrI, tileBlockBase + realBlockNum - 1, blockNumPad);
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Sub(offI[off], idxScr[off], thrI[off], 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Maxs(offI[off], offI[off], 0, 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Mins(offI[off], offI[off], 1, 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Adds(tI[off], idxScr[off], 1, 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Mul(tI[off], tI[off], offI[off], 64);
    }
    PipeBarrier<PIPE_V>();
    for (int32_t off = 0; off < blockNumPad; off += 64) {
        Sub(blkIdx[off], idxScr[off], tI[off], 64);
    }
    PipeBarrier<PIPE_V>();
    // pad 块 score 归一为 -inf 位型: 覆盖 [realBlockNum, blockNumPad), 含 stale 区 [blockNum, blockNumPad)
    // (尾 tile cuS2Len 非整块时 BlockReduceMax 只写 [0, blockNum), 其后是脏数据 —
    //  脏 score 的 (-inf,-1) 不同构对会挤进 topk 挤掉真实块, 实测 128K 场景 -1 槽超标)
    // score' = score + (score - NINF_bits)*(-isPad), isPad 于 pad 位为 1; 全 int32 向量
    if (blockNumPad > realBlockNum) {
        LocalTensor<int32_t> pRaw = tmp[15360].template ReinterpretCast<int32_t>(); // -isPad
        LocalTensor<int32_t> pNeg = tmp[15616].template ReinterpretCast<int32_t>(); // (score-NINF)*(-isPad)
        LocalTensor<int32_t> scoreI = blkScore.template ReinterpretCast<int32_t>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Duplicate(pRaw[off], -1, 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mul(pRaw[off], pRaw[off], offI[off], 64); // -isPad
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pNeg[off], scoreI[off], 8388608, 64); // score - NINF_bits (NINF=0xFF800000=-8388608)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mul(pNeg[off], pNeg[off], pRaw[off], 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Add(scoreI[off], scoreI[off], pNeg[off], 64); // pad 块 -> -inf 位型
        }
        PipeBarrier<PIPE_V>();
    }
    // pin 最新 token 所在块 (+inf 位型注入该块 score): 纯 int32 向量实现, 在 isPad 链之后
    // (复用其 scratch; 标量 SetValue 写不受 PipeBarrier<PIPE_V> fence, 与 SortAll 多轮读存在竞态, 已弃用)。
    // flag = -1 于 pin 位, 0 于其余: score' = score + (score - PINF_bits) * flag,
    // pin 位得 PINF_bits (0x7F800000 = +inf), 其余不变 (flag=0 使 pad 位 -inf 的中间溢出无害)。
    if (pinLocal >= 0 && pinLocal < realBlockNum) {
        LocalTensor<int32_t> pRaw = tmp[15360].template ReinterpretCast<int32_t>(); // [15360,15616)
        LocalTensor<int32_t> pNeg = tmp[15616].template ReinterpretCast<int32_t>(); // [15616,15872)
        LocalTensor<int32_t> pA = thrI;   // 复用 isPad 链已释放 scratch
        LocalTensor<int32_t> pB = offI;
        LocalTensor<int32_t> pFlag = tI;
        LocalTensor<int32_t> scoreI = blkScore.template ReinterpretCast<int32_t>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            // idxScr 为全局块号, pin 也必须用全局 id (tileBlockBase + pinLocal) 比较
            Adds(pRaw[off], idxScr[off], -(tileBlockBase + pinLocal), 64); // 0 于 pin 块
        }
        PipeBarrier<PIPE_V>();
        Duplicate(pNeg, 0, blockNumPad);
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Sub(pNeg[off], pNeg[off], pRaw[off], 64); // -(idx-pinLocal)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Maxs(pA[off], pRaw[off], 0, 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mins(pA[off], pA[off], 1, 64); // 1 若 idx > pin
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Maxs(pB[off], pNeg[off], 0, 64);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mins(pB[off], pB[off], 1, 64); // 1 若 idx < pin
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Add(pFlag[off], pA[off], pB[off], 64); // isNotPin (0/1, 互斥)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pFlag[off], pFlag[off], -1, 64); // -1 于 pin, 0 其余
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Adds(pRaw[off], scoreI[off], -2139095040, 64); // score - 0x7F800000 (复用 pRaw)
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Mul(pRaw[off], pRaw[off], pFlag[off], 64); // (score-PINF)*flag
        }
        PipeBarrier<PIPE_V>();
        for (int32_t off = 0; off < blockNumPad; off += 64) {
            Add(scoreI[off], scoreI[off], pRaw[off], 64); // pin -> +inf 位型
        }
        PipeBarrier<PIPE_V>();
    }
    // 块级排序 + 归并到块级累加器
    LocalTensor<float> blkPairs = tmp[6144];   // [scores blockNumPad | idx blockNumPad]
    // MrgSort tmp 需容纳 mrgDst+mrgSrc = 2048*2 + blockNumPad*2 <= 4608 floats;
    // 11776 + 4608 = 16384 恰为 tmpBuf_(64KB) 末尾, 12288 起会越界 2KB (aicore 异常)
    LocalTensor<float> blkSortTmp = tmp[11776]; // 与 blkPairs 不重叠 (MrgSort src/tmp 禁止重叠)
    QLIV2ServiceVec::SortAll(blkPairs, blkSortTmp, blockNumPad);
    PipeBarrier<PIPE_V>();
    // 候选专用合并: 纯 V 回拷 (MergeSort 尾部 UB->UB DataCopy 不受 PipeBarrier<PIPE_V> fence,
    // 多 tile 下一次 MrgSort 读 acc 存在调度敏感竞态, 实测 tile1 读到 stale 数据)
    QLIV2ServiceVec::MergeSortVecCopy(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE],
                                      constInfo_.candidateTopkBlocks, blkPairs, blockNumPad, blkSortTmp);
    PipeBarrier<PIPE_V>();
}

// mode=1: 行末直出 candidate_topk_index 并复位块级累加器
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::CopyOutCandTopkIndex(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                                                 int32_t innerS1Idx)
{
    LocalTensor<uint32_t> candIdxULocal = outQueue_.AllocTensor<uint32_t>();
    ExtractIndex(candIdxULocal,
                 globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE].template ReinterpretCast<uint32_t>(),
                 constInfo_.candidateTopkBlocks);
    PipeBarrier<PIPE_V>();
    InitSortOutBuf(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
    outQueue_.EnQue<uint32_t>(candIdxULocal);
    candIdxULocal = outQueue_.DeQue<uint32_t>();
    QLIV2ServiceVec::CopyOut(candidateTopkIndexOutGm[info.candidateOutOffset +
                                                     cuS1Idx * constInfo_.candidateTopkBlocks],
                             candIdxULocal.template ReinterpretCast<int32_t>(), constInfo_.candidateTopkBlocks);
    outQueue_.FreeTensor(candIdxULocal);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessVec0(const QLIV2Common::RunInfo &info)
{
    // 只需要一个v核做
    if (blockId_ % 2 != 0) {
        return;
    }
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    // 计算输出w基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 4*64 + aic_offset
    int64_t vec0OutGmOffset = (info.loop % 2) * ((s1BaseSize_ * gSize_ * BLOCK_CUBE));
    // 计算输入weight的地址偏移，qScale的地址偏移与weight相同
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * qHeadNum_;
    // 当前需要计算的S1行数，处理尾块场景
    int32_t cuS1ProcNum = cuBaseS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuProcRealNum = cuS1ProcNum * gSize_;
    int32_t cuProcEleNum = QLIV2Common::Align(cuProcRealNum, 32); // 32: UB对齐, 参照v1

    LocalTensor<half> inWeightsUb = inQueue_.AllocTensor<half>();
    LocalTensor<half> inQScaleUb = inWeightsUb[cuProcEleNum];
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = cuProcRealNum * sizeof(half);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    AscendC::DataCopyPad(inWeightsUb, weightsGm[weightGmOffset], copyInParams, padParams);
    AscendC::DataCopyPad(inQScaleUb, qScaleGm[weightGmOffset], copyInParams, padParams);

    inQueue_.EnQue<half>(inWeightsUb);
    inWeightsUb = inQueue_.DeQue<half>();
    AscendC::Mul(inWeightsUb, inWeightsUb, inQScaleUb, cuProcEleNum);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> resUb = outQueue_.AllocTensor<half>();
    AscendC::Brcb(resUb, inWeightsUb, static_cast<uint8_t>(cuProcEleNum / 8), {1, 8});
    inQueue_.FreeTensor(inWeightsUb);

    outQueue_.EnQue<half>(resUb);
    resUb = outQueue_.DeQue<half>();
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = cuProcRealNum * BLOCK_CUBE * sizeof(half);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    AscendC::DataCopyPad(vec0OutGm[vec0OutGmOffset], resUb, copyOutParams);
    outQueue_.FreeTensor(resUb);
}

template <typename QLIV2T>
__aicore__ inline int32_t QLIV2Vector<QLIV2T>::AlignS2(int32_t cuS2Len)
{
    // 限制：当前cuS2Len最大为2048，暂不考虑更长
    // 该函数目的是将cuS2Len对齐到形如 32*(4^n)*m 的形式 (m ∈ [1, 3])，方便后续sort/merge
    if (cuS2Len <= ELE_NUM_128) {
        return Align(cuS2Len, ELE_NUM_32);
    } else if (cuS2Len <= ELE_NUM_512) {
        return Align(cuS2Len, ELE_NUM_128);
    } else {
        return Align(cuS2Len, ELE_NUM_512);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessVec1(const QLIV2Common::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 4*2048 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * (s1BaseSize_ * s2BaseSize_);

    // cuS1BeginIdxPerAiv: 每个AIV的S1起始偏移
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // cuS1ProcNumPerAiv: 每个AIv的S1计算量
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);
    // 基本块基地址偏移奇数核加一个S1地址偏移
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * s2BaseSize_;
    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);
        blockS2StartIdx_ = 0;
        if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
            InitSortOutBuf(globalBlockTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);
        }
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    int32_t cuRealAcSeqCount = 0;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + cuS1BeginIdxPerAiv;
    }
    int32_t cuRealAcSeqIni = cuRealAcSeq;

    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeqCount += 1;
            cuRealAcSeq = (cuRealAcSeqCount + cuRealAcSeqIni) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        // 当前vec1ResGm对应S1的位置
        uint64_t wsOffset = static_cast<uint64_t>(info.saveWorkSpaceIdx) * s1BaseSize_ * BASE_TOPK_VALUE_IDX_SIZE +
                            static_cast<uint64_t>(cuS1Idx - cuBaseS1Idx) * BASE_TOPK_VALUE_IDX_SIZE;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = AlignS2(cuS2Len);
            LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
            LocalTensor<float> kScaleUb = mmInUb[cuS2LenVecAlign];
            LocalTensor<half> kScaleTUb = kScaleUb.template ReinterpretCast<half>()[cuS2LenVecAlign];
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> padTParams{false, 0, 0, 0};
            AscendC::DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = cuS2Len * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            copyInParams.rsv = 0;
            AscendC::DataCopyPad(mmInUb, mm1ResGm[mmGmOffset + innerS1Idx * s2BaseSize_], copyInParams, padParams);
            GetKeyScale(info, kScaleTUb, info.bIdx, cuBaseS2Idx, cuS2Len);
            inQueue_.EnQue<float>(mmInUb);
            mmInUb = inQueue_.DeQue<float>();
            AscendC::Cast(kScaleUb, kScaleTUb, RoundMode::CAST_NONE, cuS2Len);
            PipeBarrier<PIPE_V>();
            AscendC::Mul(mmInUb, mmInUb, kScaleUb, cuS2Len);
            PipeBarrier<PIPE_V>();
            // mode=2 (use_candidate, R11 leak): 候选块外 score 降级 NEG_HUGE (S4a, 纯算术无 NaN);
            // 仅降级排序资格、不取消入选资格 — 不足 topk 时候选外可达位置作为填充泄漏 (§1.6)
            // mode=4 (use_candidate + skip): 候选块内紧凑收集, 排序对象 2048 -> S (输出集合与
            // mode=2 等价, 见 BuildCandidateCompact 头注释); 0 = 本 tile 无候选块, 跳过排序与归并
            int32_t compactSortLen = 0;
            bool candSkip = constInfo_.candidateMode == CANDIDATE_MODE_CONSUMER_SKIP;
            if (constInfo_.candidateMode == CANDIDATE_MODE_CONSUMER) {
                BuildCandidateMask(info, cuS1Idx, cuBaseS2Idx, innerS1Idx);
                PipeBarrier<PIPE_V>();
                LocalTensor<float> pen = tmpBuf_.Get<float>()[4096]; // blkIdxF/acc/d 已释放, 复用
                Sub(pen, candNegHuge_, mmInUb, cuS2Len);
                PipeBarrier<PIPE_V>();
                Mul(pen, pen, candIsOut_, cuS2Len);
                PipeBarrier<PIPE_V>();
                Add(mmInUb, mmInUb, pen, cuS2Len);
                PipeBarrier<PIPE_V>();
            } else if (candSkip) {
                compactSortLen = BuildCandidateCompact(info, cuS1Idx, cuBaseS2Idx, innerS1Idx, mmInUb, cuS2Len);
                inQueue_.FreeTensor(mmInUb);
            }
            if (!candSkip) {
            LocalTensor<float> sortBuff = tmpBuf_.Get<float>();
            LocalTensor<float> sortScoreUb = sortBuff;
            LocalTensor<float> sortIndiceUb = sortBuff[cuS2LenVecAlign];
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), QLIV2ServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, mmInUb, 0.0f, cuS2Len);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(mmInUb);
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // 无效数据索引填充为-1
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
                PipeBarrier<PIPE_V>();
            }
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            PipeBarrier<PIPE_V>();
            // R11 (leak 语义, §1.6): 候选块外索引不再置 -1 —
            // 分数已在 S4a 降级 NEG_HUGE (仍高于 -inf: 不足 topk 时作为填充入选, 排候选内之后),
            // 索引保留真实位置号, 与模型 where(idxs < compress_lens, idxs + offset, -1) 等价:
            // 仅不可达位置 (score -inf 沉底 + 尾部 -1 填充) 输出 -1。
            // mode=1 (is_candidate_source): 块化 amax + pin + 块级排序归并 (S5a/S6a)
            if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
                ProcessCandBlockTopk(info, cuS1Idx, cuS2Len, cuS2LenVecAlign, cuRealAcSeq, innerS1Idx);
            }
            LocalTensor<float> tmpSortBuf = sortBuff[2 * cuS2LenVecAlign];
            QLIV2ServiceVec::SortAll(sortBuff, tmpSortBuf, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            QLIV2ServiceVec::MergeSort(globalTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK, sortBuff,
                                       cuS2LenVecAlign, tmpSortBuf);
            PipeBarrier<PIPE_V>();
            } else if (compactSortLen > 0) {
                // mode=4: 紧凑缓冲已在 BuildCandidateCompact 构建于 tmp[0..2S) / idx tmp[S..2S);
                // MergeSort 的 MrgSort tmp 需容纳 (S+2048)*2 对, 区域 [2S, 4S+4096) <= 12288
                LocalTensor<float> sortBuff = tmpBuf_.Get<float>();
                LocalTensor<float> tmpSortBuf = sortBuff[2 * compactSortLen];
                QLIV2ServiceVec::SortAll(sortBuff, tmpSortBuf, compactSortLen);
                PipeBarrier<PIPE_V>();
                QLIV2ServiceVec::MergeSort(globalTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK, sortBuff,
                                           compactSortLen, tmpSortBuf);
                PipeBarrier<PIPE_V>();
            }
            // compactSortLen == 0: 本 tile 无候选块, acc 保持前序 tile 状态 (行末 tile 必含
            // pin 块, 单 tile 行不会出现 0 候选, needCopyOutGm 读 acc 恒有排序数据)
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            // 中间结果保存
            if (needCopyOutGm && !info.isNeedLD) {
                LocalTensor<uint32_t> idxULocal = outQueue_.AllocTensor<uint32_t>();
                ExtractIndex(idxULocal,
                             globalTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE].template ReinterpretCast<uint32_t>(),
                             BASE_TOPK);
                PipeBarrier<PIPE_V>();
                // A15: output_idx_offset — 输出拷 GM 前逐元素加行偏移 (对齐 arch35 IndicesAddOffset;
                // 零偏移零开销; GM 标量读无 V_S 竞态; int32 整数域 Adds 位精确, -1 槽 +0 不变。
                // candidate_topk_index 不加 (相对块号契约, §11.6))
                if (isOutputIdxOffsetValid_) {
                    int32_t rowOff = outputIdxOffsetGm_.GetValue(info.outputIdxOffsetCoreOffset +
                                                                 cuS1Idx * kHeadNum_);
                    if (rowOff != 0) {
                        LocalTensor<int32_t> idxI32 = idxULocal.template ReinterpretCast<int32_t>();
                        for (int32_t off = 0; off < constInfo_.sparseCount; off += 64) {
                            Adds(idxI32[off], idxI32[off], rowOff, 64);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
                InitSortOutBuf(globalTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
                outQueue_.EnQue<uint32_t>(idxULocal);
                idxULocal = outQueue_.DeQue<uint32_t>();
                QLIV2ServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                                         idxULocal.template ReinterpretCast<int32_t>(), constInfo_.sparseCount);
                outQueue_.FreeTensor(idxULocal);
                // mode=1: 块级 topk 直出 candidate_topk_index
                if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
                    CopyOutCandTopkIndex(info, cuS1Idx, innerS1Idx);
                }
            }
            // LD拷贝到当前vector对应S1的位置
            if (info.isNeedLD && info.isLastS2InnerLoop) { // 当前核存在归约任务 且是最后处理的一段
                AscendC::DataCopyExtParams copyWsParams;
                copyWsParams.blockLen = BASE_TOPK_VALUE_IDX_SIZE * sizeof(float);
                copyWsParams.srcStride = 0;
                copyWsParams.dstStride = 0;
                copyWsParams.blockCount = 1;
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                AscendC::DataCopyPad(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE],
                                     copyWsParams);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                InitSortOutBuf(globalTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
                PipeBarrier<PIPE_V>();
            }
        } else if (cuRealAcSeq <= 0) {
            // 无效长度处理
            if (info.isNeedLD && info.isLastS2InnerLoop) {
                PipeBarrier<PIPE_V>();
                InitSortOutBuf(globalTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                QLIV2ServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE],
                                         BASE_TOPK_VALUE_IDX_SIZE);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            } else {
                CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
            }
        } else if (cuS2Len <= 0) {
            // LD拷贝到当前vector对应S1的位置
            if (info.isNeedLD && info.isLastS2InnerLoop) { // 当前核存在归约任务 且是最后处理的一段
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                QLIV2ServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE],
                                         BASE_TOPK_VALUE_IDX_SIZE);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                InitSortOutBuf(globalTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    // BNSD场景无效S1 输出-1
    if (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2SizeOrig;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessLD()
{
    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
    LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<float> indexValuePadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams indexValueParams;
    indexValueParams.blockLen = BASE_TOPK_VALUE_IDX_SIZE * sizeof(int32_t); // bytes
    indexValueParams.srcStride = 3 * BASE_TOPK_VALUE_IDX_SIZE * sizeof(int32_t);
    indexValueParams.dstStride = 0;

    uint32_t ldProWorkspaceNum = ldInfo_.workspaceNum;
    uint32_t ldProcessLen = 4; // 4: 4块归约任务做一次Merge
    uint32_t ldProcessNum = (ldProWorkspaceNum - 1) / (ldProcessLen - 1);
    uint32_t ldTailLen = ldProWorkspaceNum - (ldProcessNum * (ldProcessLen - 1) + 1); // 尾块长度

    for (uint32_t j = 0; j < ldInfo_.mNum; j++) {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        // 拷贝第一块归约任务
        indexValueParams.blockCount = 1;
        uint64_t wsOffsetIni = static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * BASE_TOPK_VALUE_IDX_SIZE +
                               static_cast<uint64_t>(ldInfo_.mStart + j) * BASE_TOPK_VALUE_IDX_SIZE;
        AscendC::DataCopyPad(curValueIdxUb, vec1ResGm[wsOffsetIni], indexValueParams, indexValuePadParams);
        uint64_t valueOffset = BASE_TOPK_VALUE_IDX_SIZE;
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        // 处理等于4的部分
        for (uint32_t i = 0; i < ldProcessNum; i++) {
            // LD处理偏移
            uint64_t wsOffset =
                static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * BASE_TOPK_VALUE_IDX_SIZE +
                static_cast<uint64_t>(ldInfo_.mStart + j) * BASE_TOPK_VALUE_IDX_SIZE +
                static_cast<uint64_t>(i * (ldProcessLen - 1) + 1) * s1BaseSize_ * BASE_TOPK_VALUE_IDX_SIZE;
            indexValueParams.blockCount = ldProcessLen - 1; // 拷贝4块进行merge

            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            AscendC::DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset], indexValueParams,
                                 indexValuePadParams);
            // merge参数
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            params.validBit = 0b1111;
            params.repeatTimes = 1;

            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[BASE_TOPK_VALUE_IDX_SIZE];
            srcList.src3 = curValueIdxUb[2 * BASE_TOPK_VALUE_IDX_SIZE];
            srcList.src4 = curValueIdxUb[3 * BASE_TOPK_VALUE_IDX_SIZE];
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, BASE_TOPK_VALUE_IDX_SIZE);
            PipeBarrier<PIPE_V>();
        }

        // 处理不等于4的部分
        if (ldTailLen != 0) {
            // 搬运尾块
            uint64_t wsOffsetTail =
                static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * BASE_TOPK_VALUE_IDX_SIZE +
                static_cast<uint64_t>(ldInfo_.mStart + j) * BASE_TOPK_VALUE_IDX_SIZE +
                static_cast<uint64_t>(ldProcessNum * (ldProcessLen - 1) + 1) * s1BaseSize_ * BASE_TOPK_VALUE_IDX_SIZE;
            indexValueParams.blockCount = ldTailLen;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            AscendC::DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffsetTail], indexValueParams,
                                 indexValuePadParams);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            if (ldTailLen == 1) {
                params.validBit = 0b0011;
            } else if (ldTailLen == 2) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[BASE_TOPK_VALUE_IDX_SIZE];
            srcList.src3 = curValueIdxUb[2 * BASE_TOPK_VALUE_IDX_SIZE];
            srcList.src4 = curValueIdxUb[3 * BASE_TOPK_VALUE_IDX_SIZE];
            PipeBarrier<PIPE_V>();
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, BASE_TOPK_VALUE_IDX_SIZE);
            PipeBarrier<PIPE_V>();
        }

        // 搬出
        Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
        PipeBarrier<PIPE_V>();
        InitSortOutBuf(curValueIdxUb, BASE_TOPK_VALUE_IDX_SIZE);
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        // A15: LD(decode) 路径同样加 output_idx_offset (行前缀 = indiceOutCoreOffset/sparseCount, 即
        // batch 前缀 x kHeadNum + n2; 与 ProcessVec1 消费点同语义)
        if (isOutputIdxOffsetValid_) {
            uint32_t rowGlobal = ldInfo_.mStart + j;
            int32_t rowOff = outputIdxOffsetGm_.GetValue(
                ldInfo_.indiceOutCoreOffset / constInfo_.sparseCount + rowGlobal * constInfo_.kHeadNum);
            if (rowOff != 0) {
                for (int32_t off = 0; off < constInfo_.sparseCount; off += 64) {
                    Adds(idxULocal1[off], idxULocal1[off], rowOff, 64);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        uint64_t outOffset =
            ldInfo_.indiceOutCoreOffset + (ldInfo_.mStart + j) * constInfo_.kHeadNum * constInfo_.sparseCount;
        AscendC::DataCopyPad(indiceOutGm[outOffset], idxULocal1, copyOutParams);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
