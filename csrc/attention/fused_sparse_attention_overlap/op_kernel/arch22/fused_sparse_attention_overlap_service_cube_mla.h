/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_sparse_attention_overlap_service_cube_mla.h
 * \brief use 7 buffer for matmul l1, better pipeline
 */
#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_SERVICE_CUBE_MLA_H
#define FUSED_SPARSE_ATTENTION_OVERLAP_SERVICE_CUBE_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fused_sparse_attention_overlap_common.h"

struct PAShape {
    uint32_t blockSize;
    uint32_t headNum;             // Usually the KV head count, corresponding to N2
    uint32_t headDim;             // For MLA, RoPE is 64 and NoPE is 512, corresponding to D
    uint32_t maxblockNumPerBatch; // Maximum number of entries in each block-table row
    uint32_t actHeadDim;          // Actual copied column size after accounting for N partitioning, corresponding to D
    uint32_t copyRowNum;          // Total number of rows to copy
    uint32_t copyRowNumAlign;
};

struct Position {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx;
};

// Scenario: copy query, queryRope, key, and value from GM to L1.
// GM uses ND format.
// L1 uses NZ format.
// The parameters describe the GM row count, column count, and column stride.
template <typename T>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<T> &l1Tensor, GlobalTensor<T> &gmTensor,
                                        uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,       // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct;       // Number of rows in the ND matrix
    // For int4 T, dValue = col / 2 and srcDValue = colStride / 2.
    nd2nzPara.dValue = col;          // Number of columns in the ND matrix
    nd2nzPara.srcDValue = colStride; // Offset between the start addresses of adjacent rows in one ND matrix
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

/*
    Copies page-attention data from GM to L1 and supports ND and NZ formats.
    Page-attention layouts include BNBD (blockNum, N, blockSize, D) and BBH (blockNum, blockSize, N * D).
    BSH, BSND, and TND use BBH storage.
    shape.copyRowNumAlign must be aligned to 16. For example, a 128 * 512 key copy with a 10 * 512
    tail block must be aligned to 16 * 512.
*/
template <typename T, FusedSparseAttentionOverlapLayout SRC_LAYOUT>
__aicore__ inline void DataCopyPA(LocalTensor<T> &dstTensor,  //l1
                                  GlobalTensor<T> &srcTensor, //gm
                                  GlobalTensor<int32_t> &blockTableGm,
                                  const PAShape &shape,       // blockSize, headNum, headDim                           
                                  const Position &startPos)   // bacthIdx nIdx curSeqIdx
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = startPos.bIdx * shape.maxblockNumPerBatch;
    uint32_t curS2Idx = startPos.s2Idx;
    uint32_t blockElementCnt = 32 / sizeof(T);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;   // Obtain the block-table index
        uint64_t reaminRowCnt = curS2Idx % shape.blockSize;    // Obtain the row offset within one block
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // Read the block ID from the block table
        // Calculate the number of rows that can be copied.
        uint32_t copyRowCnt = shape.blockSize - reaminRowCnt;  // Process only one block at a time
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;  // The final copy does not fill one block
        }
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim ;   // Page-attention offset

        uint64_t dStride = shape.headDim;
        if constexpr (SRC_LAYOUT == FusedSparseAttentionOverlapLayout::BSND || SRC_LAYOUT == FusedSparseAttentionOverlapLayout::TND) {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim) +
                      reaminRowCnt * shape.headDim * shape.headNum + startPos.dIdx;
            dStride = shape.headDim * shape.headNum;
        } else {
            offset += (uint64_t)(startPos.n2Idx * shape.headDim * shape.blockSize) +
                      reaminRowCnt * shape.headDim + startPos.dIdx;
        }

        uint32_t dValue = shape.actHeadDim;
        uint32_t srcDValue = dStride;
        LocalTensor<T> tmpDstTensor = dstTensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<T> tmpSrcTensor = srcTensor[offset];

        DataCopyGmNDToL1<T>(tmpDstTensor, tmpSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);                     
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template <typename FusedSparseAttentionOverlapTraits> class FusedSparseAttentionOverlapMatmulService {
public:
    // Use float for intermediate computations in high-precision mode.
    using T = float;
    using Q_T = typename FusedSparseAttentionOverlapTraits::queryType;
    using KV_T = typename FusedSparseAttentionOverlapTraits::kvType;
    using OUT_T = typename FusedSparseAttentionOverlapTraits::outputType;
    using MM_OUT_T = T;

    __aicore__ inline FusedSparseAttentionOverlapMatmulService(){};
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<Q_T> qRopeGm,
                                               GlobalTensor<KV_T> keyGm, GlobalTensor<KV_T> kRopeGm,
                                               GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
                                               GlobalTensor<MM_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitPageAttentionInfo(const GlobalTensor<KV_T>& kvMergeGm,
                                                 GlobalTensor<int32_t> blockTableGm, GlobalTensor<int32_t> topKGm,
                                                 uint32_t blockSize, uint32_t maxBlockNumPerBatch);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void UpdateKey(GlobalTensor<KV_T> keyGm);
    __aicore__ inline void UpdateValue(GlobalTensor<KV_T> valueGm);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void CalcTopKBlockInfo(const RunInfo &info, uint32_t &curTopKIdx,
                                             uint64_t &curOffsetInSparseBlock, uint32_t curSeqIdx,
                                             uint32_t &copyRowCnt, int64_t &idInTopK);
    __aicore__ inline void ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo);
    __aicore__ inline void ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo);

private:
    static constexpr bool PAGE_ATTENTION = FusedSparseAttentionOverlapTraits::pageAttention;
    static constexpr int TEMPLATE_MODE = FusedSparseAttentionOverlapTraits::templateMode;
    static constexpr bool FLASH_DECODE = FusedSparseAttentionOverlapTraits::flashDecode;
    static constexpr FusedSparseAttentionOverlapLayout LAYOUT_T = FusedSparseAttentionOverlapTraits::layout;
    static constexpr FusedSparseAttentionOverlapLayout KV_LAYOUT_T = FusedSparseAttentionOverlapTraits::kvLayout;

    static constexpr uint32_t M_SPLIT_SIZE = 128;     // Partition size along M
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // Partition size along N
    static constexpr uint32_t N_WORKSPACE_SIZE = 512; // Workspace size along N

    static constexpr uint32_t L1_BLOCK_SIZE = (64 * (512 + 64) * sizeof(Q_T));
    static constexpr uint32_t L1_BLOCK_OFFSET = 64 * (512 + 64); // Number of elements in 72 KB

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    // mte2 <> mte1 EventID
    // L1 uses triple buffering with three event IDs.
    static constexpr uint32_t L1_EVENT0 = EVENT_ID2;
    static constexpr uint32_t L1_EVENT1 = EVENT_ID3;
    static constexpr uint32_t L1_EVENT2 = EVENT_ID4;
    static constexpr uint32_t L1_EVENT3 = EVENT_ID5;
    static constexpr uint32_t L1_EVENT4 = EVENT_ID6;
    static constexpr uint32_t L1_EVENT5 = EVENT_ID7;
    static constexpr uint32_t L1_EVENT6 = EVENT_ID1;

    // m <> mte1 EventID
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};                    // isSetFMatrix isSetPadding;
    static constexpr uint32_t mte21QPIds[4] = {L1_EVENT0, L1_EVENT1, L1_EVENT2, L1_EVENT3}; // Reused by MTE1 and MTE2
    static constexpr uint32_t mte21KVIds[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};

    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    ConstInfo constInfo{};

    // Divide L1 into three buffers for tracking.
    uint32_t qpL1BufIter = 0;
    uint32_t kvL1BufIter = -1;
    uint32_t abL0BufIter = 0;
    uint32_t cL0BufIter = 0;

    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<Q_T> qRopeGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> kRopeGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> kvMergeGm_;

    // mm2
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<OUT_T> attentionOutGm;

    // block_table
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;

    TBuf<TPosition::A1> bufQPL1;
    TBuf<TPosition::A1> bufKVL1;
    TBuf<TPosition::A2> tmpBufL0A;
    TBuf<TPosition::B2> tmpBufL0B;
    TBuf<TPosition::CO1> tmpBufL0C;

    LocalTensor<Q_T> l1QPTensor;
    LocalTensor<Q_T> l1KVTensor;
    LocalTensor<KV_T> aL0TensorPingPong;
    LocalTensor<KV_T> bL0TensorPingPong;
    LocalTensor<MM_OUT_T> cL0TensorPingPong;

    // L0AB m <> mte1 EventID
    __aicore__ inline uint32_t Mte1MmABEventId(uint32_t idx)
    {
        return (L0AB_EVENT0 + idx);
    }

    __aicore__ inline uint32_t GetQPL1RealIdx(uint32_t mIdx, uint32_t k1Idx)
    {
        uint32_t idxMap[] = {0, 2}; // Keep blocks 0-1 and 2-3 contiguous so addresses for the same M block remain contiguous
        return idxMap[mIdx % 2] + k1Idx;
    }

    __aicore__ inline void CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                          uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset);
    __aicore__ inline void CopyInMm1ARopeToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                              uint32_t mSizeAct);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t keyGmBaseOffset,
                                               uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                               uint32_t nActCopyRowCount, uint32_t headSize);
    __aicore__ inline void CopyInMm1BRopeToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t keyGmBaseOffset,
                                                   uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                                   uint32_t nActCopyRowCount, uint32_t headSize);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                          uint32_t subMSizeAct, uint32_t nSize, uint32_t nOffset);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t valueGmBaseOffset,
                                               uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                               uint32_t nActCopyRowCount, uint32_t copyStartColumnCount,
                                               uint32_t copyColumnCount);
    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t mSize, uint32_t kSize);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_T> &bL0Tensor, LocalTensor<KV_T> &bL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t kSize, uint32_t nSize);
};

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<Q_T> qRopeGm,
                                                   GlobalTensor<KV_T> keyGm, GlobalTensor<KV_T> kRopeGm,
                                                   GlobalTensor<MM_OUT_T> mm1ResGm)
{
    // mm1
    this->queryGm = queryGm;
    this->qRopeGm = qRopeGm;
    this->keyGm = keyGm;
    this->kRopeGm = kRopeGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
                                                   GlobalTensor<MM_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm)
{
    // mm2
    this->vec1ResGm = vec1ResGm;
    this->valueGm = valueGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::InitPageAttentionInfo(const GlobalTensor<KV_T>& kvMergeGm, GlobalTensor<int32_t> blockTableGm,
		                              GlobalTensor<int32_t> topKGm, uint32_t blockSize, uint32_t maxBlockNumPerBatch)
{
    this->blockTableGm = blockTableGm;
    this->topKGm = topKGm;
    this->kvCacheBlockSize = blockSize;
    this->maxBlockNumPerBatch = maxBlockNumPerBatch;
    this->kvMergeGm_ = kvMergeGm;
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4); // (64K + 8K) * 4
    l1QPTensor = bufQPL1.Get<Q_T>();
    pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3); // (64K + 8K) * 3
    l1KVTensor = bufKVL1.Get<KV_T>();

    // L0A
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2); // 64K
    aL0TensorPingPong = tmpBufL0A.Get<KV_T>();
    // L0B
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2); // 64K
    bL0TensorPingPong = tmpBufL0B.Get<KV_T>();
    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2); // 128K
    cL0TensorPingPong = tmpBufL0C.Get<MM_OUT_T>();
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::UpdateKey(GlobalTensor<KV_T> keyGm)
{
    this->keyGm = keyGm;
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::UpdateValue(GlobalTensor<KV_T> valueGm)
{
    this->valueGm = valueGm;
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::CopyGmToL1(LocalTensor<KV_T> &l1Tensor,
                                                                 GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                                                 uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // Number of rows
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // Align to 16, in blocks
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor, const RunInfo &info,
                                                                     uint32_t mSeqIdx, uint32_t mSizeAct,
                                                                     uint32_t headSize, uint32_t headOffset)
{
    auto srcGm = queryGm[info.tensorAOffset + mSeqIdx * constInfo.headDim + headOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, headSize, constInfo.headDim);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::CopyInMm1ARopeToL1(LocalTensor<KV_T> &l1Tensor,
                                                                         const RunInfo &info, uint32_t mSeqIdx,
                                                                         uint32_t mSizeAct)
{
    auto srcGm = qRopeGm[info.tensorARopeOffset + mSeqIdx * constInfo.headDimRope];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, constInfo.headDimRope, constInfo.headDimRope);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::CopyInMm1BToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t keyGmBaseOffset,
                                                   uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                                   uint32_t nActCopyRowCount, uint32_t headSize)
{
    uint64_t dStride = constInfo.headDim;
    if constexpr (KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::BSND || KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
        dStride = constInfo.headDim * constInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = headSize;
    mm1Nd2NzParamsForB.srcDValue = dStride;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], keyGm[keyGmBaseOffset], mm1Nd2NzParamsForB);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::CopyInMm1BRopeToL1(LocalTensor<KV_T> &bL1Tensor, const uint64_t kRopeGmBaseOffset,
                                                       uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                                       uint32_t nActCopyRowCount, uint32_t headSize)
{
    uint64_t dStride = constInfo.headDimRope;
    if constexpr (KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::BSND || KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
        dStride = constInfo.headDimRope * constInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = headSize;
    mm1Nd2NzParamsForB.srcDValue = dStride;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], kRopeGm[kRopeGmBaseOffset], mm1Nd2NzParamsForB);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor,
                                                                   LocalTensor<KV_T> &aL1Tensor, uint32_t idx,
                                                                   uint32_t kSplitSize, uint32_t mSize, uint32_t kSize)
{
    LocalTensor<KV_T> srcTensor = aL1Tensor[mSize * kSplitSize * idx];
    LoadData3DParamsV2<KV_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = mSize / 16; // Hin=M1=8
    loadData3DParams.l1W = 16;         // Win=M0
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // Tail data does not affect the sliding-window result

    // SetLoadToA0Params
    loadData3DParams.mExtension = mSize; // M
    loadData3DParams.kExtension = kSize; // K
    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;
    loadData3DParams.channelSize = kSize; // Cin=K
    LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, srcTensor, loadData3DParams);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::LoadDataMm1B(LocalTensor<KV_T> &l0Tensor,
                                                                   LocalTensor<KV_T> &l1Tensor, uint32_t idx,
                                                                   uint32_t kSplitSize, uint32_t kSize, uint32_t nSize)
{
    // Load the full N dimension.
    LocalTensor<KV_T> srcTensor = l1Tensor[nSize * kSplitSize * idx];

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = (nSize + 15) / 16 * kSize / (32 / sizeof(KV_T));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0Tensor, srcTensor, loadData2DParams);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info,
                                                                     uint32_t mSeqIdx, uint32_t subMSizeAct,
                                                                     uint32_t nSize, uint32_t nOffset)
{
    auto srcGm = vec1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                           mSeqIdx * info.actualSingleProcessSInnerSizeAlign + nOffset];
    CopyGmToL1(aL1Tensor, srcGm, subMSizeAct, nSize, info.actualSingleProcessSInnerSizeAlign);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::CopyInMm2BToL1(
    LocalTensor<KV_T> &bL1Tensor, const uint64_t valueGmBaseOffset, uint32_t copyTotalRowCntAlign,
    uint32_t copyStartRowCnt, uint32_t nActCopyRowCount, uint32_t copyStartColumnCount, uint32_t copyColumnCount)
{
    uint64_t step = constInfo.headDim;
    if constexpr (KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::BSND || KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
        step = constInfo.headDim * constInfo.kvHeadNum;
    }

    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    Nd2NzParams mm1Nd2NzParamsForB;
    mm1Nd2NzParamsForB.ndNum = 1;
    mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
    mm1Nd2NzParamsForB.dValue = copyColumnCount;
    mm1Nd2NzParamsForB.srcDValue = step;
    mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
    mm1Nd2NzParamsForB.dstNzNStride = 1;
    mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
    mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
    DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], valueGm[valueGmBaseOffset + copyStartColumnCount],
             mm1Nd2NzParamsForB);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::CalcTopKBlockInfo(
    const RunInfo &info, uint32_t &curTopKIdx, uint64_t &curOffsetInSparseBlock, uint32_t curSeqIdx, uint32_t &copyRowCnt, int64_t &idInTopK)
{
    uint64_t blockBegin = idInTopK * constInfo.sparseBlockSize;
    uint64_t blockEnd = (blockBegin + constInfo.sparseBlockSize > info.threshold) ?
                        info.threshold : blockBegin + constInfo.sparseBlockSize;
    uint64_t blockLen = blockEnd - blockBegin;
    if (curOffsetInSparseBlock + copyRowCnt < blockLen) {
        curOffsetInSparseBlock += copyRowCnt;
        copyRowCnt = blockLen - curOffsetInSparseBlock;
    } else {
        for (uint64_t topkidx = curTopKIdx + 1; topkidx < constInfo.sparseBlockCount; topkidx++) {
            int64_t sparseIndices = topKGm.GetValue(info.topKBaseOffset + topkidx);
            if (sparseIndices == -1) {
                break;
            }
            
            uint64_t blockBegin = sparseIndices * constInfo.sparseBlockSize;
            if (blockBegin >= info.threshold) {
                continue;
            }
            uint64_t blockEnd = (blockBegin + constInfo.sparseBlockSize > info.threshold) ?
                                info.threshold : blockBegin + constInfo.sparseBlockSize;
            uint64_t blockLen = blockEnd - blockBegin;
            curTopKIdx = topkidx;
            idInTopK = sparseIndices;
            curOffsetInSparseBlock = 0;
            copyRowCnt = blockLen;
            break;
        }
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    // An additional outer M loop is required.
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mL1Size = M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = FusedSparseAttentionOverlapAlign(M_SPLIT_SIZE, 16U);
    uint32_t mL1Loops = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;

    uint32_t nSize = info.actualSingleProcessSInnerSize;
    uint32_t nL1Size = N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = FusedSparseAttentionOverlapAlign(N_SPLIT_SIZE, 16U);
    uint32_t nL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;

    uint32_t kSize = 576;
    uint32_t kL1Size = 288;
    uint32_t kL1Loops = 2; // 2: 576 / 288, MLA-specific; generalized D is not considered here

    uint32_t kL0Size = 96;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size; // 288 / 96 = 3 kloops

    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> kRopeTensor;
    LocalTensor<KV_T> kTensor;
    // ka selects one of four buffers for the left matrix; kb selects one of three buffers for the right matrix.
    uint32_t ka = 0, kb = 0;
    
    uint32_t curTopKIdx = info.curTopKIdx;
    uint64_t curOffsetInSparseBlock = info.curOffsetInSparseBlock; // Offset within the sparse block
    uint32_t copyRowCnt = 0;
    int64_t idInTopK = topKGm.GetValue(info.topKBaseOffset + curTopKIdx);

    uint32_t curTopKIdxTmp = 0;
    uint64_t curOffsetInSparseBlockTmp = 0;
    uint32_t copyRowCntTmp = 0;
    int64_t idInTopKTmp = 0;

    // Partition N, K, and M in L1.
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // Partition N in L1: 512 / 128 = 4
        if (nL1 == (nL1Loops - 1)) {
            // Recalculate the tail-block size.
            nL1Size = nSize - (nL1Loops - 1) * N_SPLIT_SIZE;
            nL1SizeAlign = FusedSparseAttentionOverlapAlign(nL1Size, 16U);
        }
        curTopKIdxTmp = curTopKIdx;
        curOffsetInSparseBlockTmp = curOffsetInSparseBlock;
        copyRowCntTmp = copyRowCnt;
        idInTopKTmp = idInTopK;

        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) { // Partition K in L1: 576 / 288; generalized D is not considered here
            kvL1BufIter++;
            uint32_t kb = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
            // Select the current block from K.
            bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
                // Main MM1 copy path
 
                uint32_t curSeqIdx = info.s2BatchOffset + nL1 * N_SPLIT_SIZE;
                uint32_t copyFinishRowCnt = 0;
                curTopKIdx = curTopKIdxTmp;
                curOffsetInSparseBlock = curOffsetInSparseBlockTmp;
                copyRowCnt = copyRowCntTmp;
                idInTopK = idInTopKTmp;
                if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                    if (kL1 == 0) {
                        Nd2NzParams nd2nzPara;
                        nd2nzPara.ndNum = 1;
                        nd2nzPara.nValue = nL1Size;                 // Number of rows
                        nd2nzPara.dValue = constInfo.headDim >> 1;  // constInfo.headDim;
                        nd2nzPara.srcDValue = constInfo.headDim;
                        nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                        nd2nzPara.dstNzNStride = 1;
                        nd2nzPara.srcNdMatrixStride = 0;
                        nd2nzPara.dstNzMatrixStride = 0;
                        DataCopy(bL1Tensor,
                                 kvMergeGm_[info.loop % 4 * N_WORKSPACE_SIZE * kSize +
                                            nL1 * N_SPLIT_SIZE * constInfo.headDim],
                                 nd2nzPara);
                        nd2nzPara.dValue = constInfo.headDimRope >> 1;
                        nd2nzPara.srcDValue = constInfo.headDimRope;
                        DataCopy(
                            bL1Tensor[nL1SizeAlign * (constInfo.headDim >> 1)],
                            kvMergeGm_[info.loop % 4 * N_WORKSPACE_SIZE * kSize + N_WORKSPACE_SIZE * constInfo.headDim +
                                       nL1 * N_SPLIT_SIZE * constInfo.headDimRope],
                            nd2nzPara);
                    } else {
                        LocalTensor<Q_T> kTmpTensor = bL1Tensor[(constInfo.headDimRope >> 1) * nL1SizeAlign];
                        Nd2NzParams nd2nzPara;
                        nd2nzPara.ndNum = 1;
                        nd2nzPara.nValue = nL1Size;                 // Number of rows
                        nd2nzPara.dValue = constInfo.headDim >> 1;  // constInfo.headDim;
                        nd2nzPara.srcDValue = constInfo.headDim;
                        nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                        nd2nzPara.dstNzNStride = 1;
                        nd2nzPara.srcNdMatrixStride = 0;
                        nd2nzPara.dstNzMatrixStride = 0;
                        DataCopy(kTmpTensor,
                                 kvMergeGm_[info.loop % 4 * N_WORKSPACE_SIZE * kSize + (constInfo.headDim >> 1) +
                                            nL1 * N_SPLIT_SIZE * constInfo.headDim],
                                 nd2nzPara);
                        nd2nzPara.dValue = constInfo.headDimRope >> 1;
                        nd2nzPara.srcDValue = constInfo.headDimRope;
                        DataCopy(
                            bL1Tensor,
                            kvMergeGm_[info.loop % 4 * N_WORKSPACE_SIZE * kSize + N_WORKSPACE_SIZE * constInfo.headDim +
                                       (constInfo.headDimRope >> 1) + nL1 * N_SPLIT_SIZE * constInfo.headDimRope],
                            nd2nzPara);
                    }
                } else {
                    while (copyFinishRowCnt < nL1Size) {
                        CalcTopKBlockInfo(info, curTopKIdx, curOffsetInSparseBlock, curSeqIdx, copyRowCnt, idInTopK);
                        if (copyFinishRowCnt + copyRowCnt > nL1Size) {
                            copyRowCnt = nL1Size - copyFinishRowCnt;
                        }

                        // BN2-axis offset
                        if constexpr (PAGE_ATTENTION) {
                            Position startPos;
                            startPos.bIdx = info.bIdx;
                            startPos.n2Idx = info.n2Idx;
                            startPos.s2Idx = idInTopK * constInfo.sparseBlockSize + curOffsetInSparseBlock;
                            // Rename 256 and 32 after the seven-buffer naming is updated.
                            startPos.dIdx = kL1 * 256;  // MM1 right matrix is BN2S2D: D is K and is not split; MM2 right matrix uses S2 as K and splits D
                            Position ropeStartPos = startPos;
                            ropeStartPos.dIdx = kL1 * 32;
                            PAShape shape;
                            shape.blockSize = kvCacheBlockSize;
                            shape.headNum = constInfo.kvHeadNum;
                            shape.headDim = constInfo.headDim;
                            shape.actHeadDim = 256;
                            shape.maxblockNumPerBatch = maxBlockNumPerBatch;
                            shape.copyRowNum = copyRowCnt;
                            shape.copyRowNumAlign = nL1SizeAlign;
                            PAShape ropeShape = shape;
                            ropeShape.headDim = constInfo.headDimRope;
                            ropeShape.actHeadDim = 32;
                            if (kL1 == 0) {
                                kTensor = bL1Tensor[copyFinishRowCnt * 16];
                                DataCopyPA<KV_T, KV_LAYOUT_T>(kTensor, keyGm, blockTableGm, shape, startPos);
                                kRopeTensor = bL1Tensor[(nL1SizeAlign * (BlockAlign<KV_T>(constInfo.headDim) >> 1)) +
                                                        copyFinishRowCnt * 16];
                                DataCopyPA<KV_T, KV_LAYOUT_T>(kRopeTensor, kRopeGm, blockTableGm, ropeShape,
                                                              ropeStartPos);
                            } else {
                                kRopeTensor = bL1Tensor[copyFinishRowCnt * 16];
                                DataCopyPA<KV_T, KV_LAYOUT_T>(kRopeTensor, kRopeGm, blockTableGm, ropeShape,
                                                              ropeStartPos);
                                LocalTensor<Q_T> kTmpTensor = bL1Tensor[32 * nL1SizeAlign + copyFinishRowCnt * 16];
                                DataCopyPA<KV_T, KV_LAYOUT_T>(kTmpTensor, keyGm, blockTableGm, shape, startPos);
                            }
                        } else {
                            uint64_t keyOffset = info.tensorBOffset;
                            uint64_t kRopeOffset = info.tensorBRopeOffset;
                            if constexpr (KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::BSND || KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
                                keyOffset += (idInTopK * constInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                             constInfo.kvHeadNum * constInfo.headDim;
                                kRopeOffset += (idInTopK * constInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                               constInfo.kvHeadNum * constInfo.headDimRope;
                            } else {
                                keyOffset += (idInTopK * constInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                             constInfo.headDim;
                                kRopeOffset += (idInTopK * constInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                               constInfo.headDimRope;
                            }

                            if (kL1 == 0) {
                                CopyInMm1BToL1(bL1Tensor, keyOffset, nL1SizeAlign, copyFinishRowCnt, copyRowCnt, 256);
                                kRopeTensor = bL1Tensor[nL1SizeAlign * (BlockAlign<KV_T>(constInfo.headDim) >> 1)];
                                CopyInMm1BRopeToL1(kRopeTensor, kRopeOffset, nL1SizeAlign, copyFinishRowCnt, copyRowCnt,
                                                   32);
                            } else {
                                kRopeTensor = bL1Tensor;
                                CopyInMm1BRopeToL1(kRopeTensor, kRopeOffset + 32, nL1SizeAlign, copyFinishRowCnt,
                                                   copyRowCnt, 32);
                                LocalTensor<Q_T> kTmpTensor = bL1Tensor[nL1SizeAlign * 32];
                                CopyInMm1BToL1(kTmpTensor, keyOffset + 256, nL1SizeAlign, copyFinishRowCnt, copyRowCnt,
                                               256);
                            }
                        }

                        // Update loop variables.
                        copyFinishRowCnt += copyRowCnt;
                        curSeqIdx += copyRowCnt;
                    }
                }

            SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            mL1Size = M_SPLIT_SIZE;
            mL1SizeAlign = FusedSparseAttentionOverlapAlign(M_SPLIT_SIZE, 16U);
            for (uint32_t mL1 = 0; mL1 < mL1Loops; mL1++) {
                uint32_t aL1PaddingSize = 0; // Align the left-matrix tail so the two 32 KB regions remain contiguous
                if (mL1 == (mL1Loops - 1)) {
                    // Recalculate the tail-block size.
                    mL1Size = mSize - (mL1Loops - 1) * M_SPLIT_SIZE;
                    mL1SizeAlign = FusedSparseAttentionOverlapAlign(mL1Size, 16U);
                    // When mL1SizeAlign < 128 and kL1 == 0, apply an offset so half of qRope is copied to
                    // the current tensor and half to the next tensor.
                    aL1PaddingSize = (M_SPLIT_SIZE - mL1SizeAlign) * 288;
                }

                // The M L1 index selects whether the left matrix uses blocks 1-2 or 3-4.
                // The K L1 index selects the first or second block within the selected pair.
                uint32_t mIdx = qpL1BufIter + mL1;
                ka = GetQPL1RealIdx(mIdx, kL1);
                LocalTensor<Q_T> aL1Tensor =
                    l1QPTensor[ka * L1_BLOCK_OFFSET + (1 - kL1) * aL1PaddingSize]; // An offset is required when kL1 == 0
                if (nL1 == 0) { // Runs twice, for mL1 == 0 and mL1 == 1
                    if (kL1 == 0) {
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka + 1]);
                        CopyInMm1AToL1(aL1Tensor, info, mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size, 256, 0);
                        // L1 uses NZ format, so the qRope offset is the size of the full qNoPE block after K
                        // partitioning; 256 is half of headDim.
                        LocalTensor<Q_T> qRopeTensor =
                            aL1Tensor[mL1SizeAlign *
                                      256];
                        CopyInMm1ARopeToL1(qRopeTensor, info, mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size);
                    } else {
                        // 32 is half of the RoPE head dimension.
                        LocalTensor<Q_T> qTmpTensor = aL1Tensor[mL1SizeAlign * 32];
                        CopyInMm1AToL1(qTmpTensor, info, mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size, 256,
                                       256);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                }

                // Synchronize with unitFlag.
                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(cL0BufIter % 2) *
                                      (L0C_PP_SIZE / sizeof(MM_OUT_T))]; // Keep cL0BufIter synchronized with the M step
                for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1A(aL0Tensor, aL1Tensor, kL0, kL0Size, mL1SizeAlign, kL0Size);
                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1B(bL0Tensor, bL1Tensor, kL0, kL0Size, kL0Size, nL1SizeAlign);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));

                    // M == 1 requires special handling.
                    MmadParams mmadParams;
                    mmadParams.m = mL1SizeAlign;
                    mmadParams.n = nL1SizeAlign;
                    mmadParams.k = kL0Size;
                    mmadParams.cmatrixInitVal = (kL1 == 0 && kL0 == 0);
                    mmadParams.cmatrixSource = false;
                    mmadParams.unitFlag =
                        (kL1 == 1 && kL0 == (kL0Loops - 1)) ? 0b11 : 0b10; // Flip the flag on the final accumulation to indicate the result can be copied out
                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);

                    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }

                if (nL1 == (nL1Loops - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]); // Reverse synchronization: A in L1 has been consumed by MTE1
                }

                if (kL1 == 1) { // Final kL1 iteration
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    // Change to nSizeAlign.
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // Gap between two rows in mm1ResGm
                    fixParams.unitFlag = 0b11;
                    fixParams.ndNum = 1; // Output in ND format

                    // Check whether the output offset (info.loop % constInfo.preLoadNum) * mmResUbSize is calculated in matmul.
                    Fixpipe(mm1ResGm[(info.loop % (constInfo.preLoadNum)) * constInfo.mmResUbSize + nL1 * N_SPLIT_SIZE +
                                     (mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE) *
                                         info.actualSingleProcessSInnerSizeAlign],
                            cL0Tensor, fixParams);
                }
                if (mL1Loops == 2) {
                    cL0BufIter++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]); // Reverse synchronization: L1 has been consumed by MTE1
        }
        if (mL1Loops == 1) {
            cL0BufIter++;
        }
    }
    qpL1BufIter += mL1Loops;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapMatmulService<FusedSparseAttentionOverlapTraits>::ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mSizeAlign = (mSize + 16 - 1) / 16;
    uint32_t mL1Loops = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = M_SPLIT_SIZE; // 16-element alignment
    uint32_t mL1Size = M_SPLIT_SIZE;      // Actual M size

    uint32_t nSize = BlockAlign<KV_T>(constInfo.headDim);
    uint32_t nL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = N_SPLIT_SIZE; // 16-element alignment
    uint32_t nL1Size = N_SPLIT_SIZE;      // Actual N size

    uint32_t kSize = info.actualSingleProcessSInnerSize;
    uint32_t kL1Size = 256;
    uint32_t kL1SizeAlign = FusedSparseAttentionOverlapAlign(kL1Size, 16U);
    uint32_t kL1Loops = (kSize + kL1Size - 1) / kL1Size;
    uint32_t kL0Size = 128;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
    uint32_t kL0SizeAlign = kL0Size;
    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> subvTensor;

    // ka selects one of four buffers for the left matrix; kb selects one of three buffers for the right matrix.
    uint32_t ka = 0, kb = 0;
    uint32_t mBaseIdx = qpL1BufIter;
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // Partition N in L1
        if (nL1 == (nL1Loops - 1)) {
            // Tail block
            nL1Size = nSize - (nL1Loops - 1) * N_SPLIT_SIZE;
            nL1SizeAlign = FusedSparseAttentionOverlapAlign(nL1Size, 16U);
        }

        // Implement K partitioning in L1 as a loop, consistent with MM1.
        kL1Size = 256;
        kL1SizeAlign = FusedSparseAttentionOverlapAlign(kL1Size, 16U);

        uint32_t curTopKIdx = info.curTopKIdx;
        uint64_t curOffsetInSparseBlock = info.curOffsetInSparseBlock;
        uint32_t copyRowCnt = 0;
        int64_t idInTopK = topKGm.GetValue(info.topKBaseOffset + curTopKIdx);

        for (uint32_t k1 = 0; k1 < kL1Loops; k1++) { // Partition K in L1, with an inner L0 loop
            if (k1 == (kL1Loops - 1)) {
                // Tail block
                kL1Size = kSize - (kL1Loops - 1) * 256;
                kL1SizeAlign = FusedSparseAttentionOverlapAlign(kL1Size, 16U);
            }
            kvL1BufIter++;
            uint32_t kb = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
            bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
            uint32_t kOffset = k1 * kL0Loops;
            kL0Size = 128;
            // Initialize kL0Size before calculating kL0Loops; the loop modifies kL0Size and would otherwise
            // produce an incorrect kL0Loops value.
            kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
            kL0SizeAlign = kL0Size;
            for (uint32_t kL1 = kOffset; kL1 < kL0Loops + kOffset; kL1++) { // Copy page-attention data in 128-row iterations
                if (kL1 == kOffset + kL0Loops - 1) {
                    // Tail block
                    kL0Size = kL1Size - (kL0Loops - 1) * kL0Size;
                    kL0SizeAlign = FusedSparseAttentionOverlapAlign(kL0Size, 16U);
                }

                uint32_t curSeqIdx = info.s2BatchOffset + (kL1 - kOffset) * 128 + k1 * 256;
                uint32_t copyFinishRowCnt = 0;
                if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
                    Nd2NzParams nd2nzPara;
                    nd2nzPara.ndNum = 1;
                    nd2nzPara.nValue = kL0Size;      // Number of rows
                    nd2nzPara.dValue = N_SPLIT_SIZE; // constInfo.headDim;
                    nd2nzPara.srcDValue = constInfo.headDim;
                    nd2nzPara.dstNzC0Stride = kL0SizeAlign;
                    nd2nzPara.dstNzNStride = 1;
                    nd2nzPara.srcNdMatrixStride = 0;
                    nd2nzPara.dstNzMatrixStride = 0;
                    DataCopy(bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE],
                             kvMergeGm_[info.loop % 4 * N_WORKSPACE_SIZE * 576 + kL1 * 128 * constInfo.headDim +
                                        nL1 * N_SPLIT_SIZE],
                             nd2nzPara);
                } else {
                    while (copyFinishRowCnt < kL0Size) {
                        CalcTopKBlockInfo(info, curTopKIdx, curOffsetInSparseBlock, curSeqIdx, copyRowCnt, idInTopK);

                        if (copyFinishRowCnt + copyRowCnt > kL0Size) {
                            copyRowCnt = kL0Size - copyFinishRowCnt;
                        }

                        if constexpr (PAGE_ATTENTION) {
                            Position startPos;
                            startPos.bIdx = info.bIdx;
                            startPos.n2Idx = info.n2Idx;
                            startPos.s2Idx = idInTopK * constInfo.sparseBlockSize + curOffsetInSparseBlock;
                            startPos.dIdx =
                                nL1 * N_SPLIT_SIZE;  // MM1 right matrix is BN2S2D: D is K and is not split; MM2 right matrix uses S2 as K and splits D
                            PAShape shape;
                            shape.blockSize = kvCacheBlockSize;
                            shape.headNum = constInfo.kvHeadNum;
                            shape.headDim = constInfo.headDim;
                            shape.actHeadDim = nL1Size;
                            shape.maxblockNumPerBatch = maxBlockNumPerBatch;
                            shape.copyRowNum = copyRowCnt;
                            shape.copyRowNumAlign = kL0SizeAlign;
                            subvTensor = bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE + copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(subvTensor, valueGm, blockTableGm, shape, startPos);
                        } else {
                            uint64_t valueOffset = info.tensorBOffset;
                            if constexpr (KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::BSND || KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
                                valueOffset += (idInTopK * constInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                               constInfo.kvHeadNum * constInfo.headDim;
                            } else {
                                valueOffset += (idInTopK * constInfo.sparseBlockSize + curOffsetInSparseBlock) *
                                               constInfo.headDim;
                            }

                            subvTensor = bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE];
                            CopyInMm2BToL1(subvTensor, valueOffset, kL0SizeAlign, copyFinishRowCnt, copyRowCnt,
                                           nL1 * N_SPLIT_SIZE, nL1Size);
                        }
                        // Update loop variables.
                        copyFinishRowCnt += copyRowCnt;
                        curSeqIdx += copyRowCnt;
                    }
                }
            }
            SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            mL1SizeAlign = M_SPLIT_SIZE;
            mL1Size = M_SPLIT_SIZE;      // Actual M size
            for (uint32_t mL1 = 0; mL1 < mL1Loops; mL1++) {
                if (mL1 == (mL1Loops - 1)) {
                    // Tail block
                    mL1Size = mSize - (mL1Loops - 1) * M_SPLIT_SIZE;
                    mL1SizeAlign = FusedSparseAttentionOverlapAlign(mL1Size, 16U);
                }

                uint32_t mIdx = mBaseIdx + mL1;
                ka = GetQPL1RealIdx(mIdx, k1);
                LocalTensor<KV_T> aL1Tensor = l1QPTensor[ka * L1_BLOCK_OFFSET];
                if (nL1 == 0) {
                    WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
                    CopyInMm2AToL1(aL1Tensor, info, mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size, kL1Size,
                                   256 * k1);
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                }

                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(cL0BufIter % 2) *
                                      (L0C_PP_SIZE / sizeof(MM_OUT_T))]; // Keep cL0BufIter synchronized with the M step
                uint32_t baseK = 128;
                uint32_t baseN = 128;
                kL0Size = 128;
                kL0SizeAlign = kL0Size;
                for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                    if (kL0 + 1 == kL0Loops) {
                        kL0Size = kL1Size - (kL0Loops - 1) * kL0Size;
                        kL0SizeAlign = FusedSparseAttentionOverlapAlign(kL0Size, 16U);
                    }
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadData3DParamsV2<KV_T> loadData3DParamsForB;
                    loadData3DParamsForB.l1H = kL0SizeAlign / 16;    // Source operand height
                    loadData3DParamsForB.l1W = 16;                   // Source operand width is 16; destination height is l1H * l1W
                    loadData3DParamsForB.padList[0] = 0;
                    loadData3DParamsForB.padList[1] = 0;
                    loadData3DParamsForB.padList[2] = 0;
                    loadData3DParamsForB.padList[3] = 255;           // Tail data does not affect the sliding-window result

                    loadData3DParamsForB.mExtension = kL0SizeAlign;  // Transfer length along the destination height dimension
                    loadData3DParamsForB.kExtension = nL1SizeAlign;  // Transfer length along the destination width dimension
                    loadData3DParamsForB.mStartPt = 0;               // Kernel start point along the destination width dimension
                    loadData3DParamsForB.kStartPt = 0;               // Kernel start point along the destination height dimension
                    loadData3DParamsForB.strideW = 1;
                    loadData3DParamsForB.strideH = 1;
                    loadData3DParamsForB.filterW = 1;
                    loadData3DParamsForB.filterSizeW = false;        // Whether to add 256 elements to the kernel width based on filterW
                    loadData3DParamsForB.filterH = 1;
                    loadData3DParamsForB.filterSizeH = false;        // Whether to add 256 elements to the kernel height based on filterH
                    loadData3DParamsForB.dilationFilterW = 1;        // Kernel width dilation factor
                    loadData3DParamsForB.dilationFilterH = 1;        // Kernel height dilation factor
                    loadData3DParamsForB.enTranspose = 1;            // Whether transpose is enabled
                    loadData3DParamsForB.fMatrixCtrl = 0;            // Select FMATRIX_LEFT when 0 and FMATRIX_RIGHT when 1
                    loadData3DParamsForB.channelSize = nL1SizeAlign; // Source operand channel count; with dilation 1, destination width is filterW * filterH * channelSize
                    LoadData<KV_T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[kL0 * baseK * baseN], loadData3DParamsForB);

                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadData3DParamsV2<KV_T> loadData3DParamsForA;
                    loadData3DParamsForA.l1H = mL1SizeAlign / 16;    // Source operand height
                    loadData3DParamsForA.l1W = 16;                   // Source operand width
                    loadData3DParamsForA.padList[0] = 0;
                    loadData3DParamsForA.padList[1] = 0;
                    loadData3DParamsForA.padList[2] = 0;
                    loadData3DParamsForA.padList[3] = 255;           // Tail data does not affect the sliding-window result

                    loadData3DParamsForA.mExtension = mL1SizeAlign;  // Transfer length along the destination height dimension
                    loadData3DParamsForA.kExtension = kL0SizeAlign;  // Transfer length along the destination width dimension
                    loadData3DParamsForA.mStartPt = 0;               // Kernel start point along the destination width dimension
                    loadData3DParamsForA.kStartPt = 0;               // Kernel start point along the destination height dimension
                    loadData3DParamsForA.strideW = 1;                // Kernel step along the source operand width dimension
                    loadData3DParamsForA.strideH = 1;                // Kernel step along the source operand height dimension
                    loadData3DParamsForA.filterW = 1;                // Kernel width
                    loadData3DParamsForA.filterSizeW = false;        // Whether to add 256 elements to the kernel width based on filterW
                    loadData3DParamsForA.filterH = 1;                // Kernel height
                    loadData3DParamsForA.filterSizeH = false;        // Whether to add 256 elements to the kernel height based on filterH
                    loadData3DParamsForA.dilationFilterW = 1;        // Kernel width dilation factor
                    loadData3DParamsForA.dilationFilterH = 1;        // Kernel height dilation factor
                    loadData3DParamsForA.enTranspose = 0;            // Whether to transpose the entire destination matrix
                    loadData3DParamsForA.fMatrixCtrl = 0;
                    loadData3DParamsForA.channelSize = kL0SizeAlign; // Source operand channel count; with dilation 1, destination width is filterW * filterH * channelSize
                    LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[kL0 * baseK * mL1SizeAlign],
                                                    loadData3DParamsForA);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));

                    MmadParams mmadParams;
                    mmadParams.m = mL1SizeAlign;
                    mmadParams.n = nL1SizeAlign;
                    mmadParams.k = kL0Size;
                    mmadParams.cmatrixInitVal = (kL0 == 0 && k1 == 0);
                    mmadParams.cmatrixSource = false;
                    mmadParams.unitFlag = ((k1 == (kL1Loops - 1)) && (kL0 == (kL0Loops - 1))) ? 0b11 : 0b10;

                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }

                if (nL1 == (nL1Loops - 1)) {    // Final nL1 iteration; keep B resident in L1 for the next computation
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]); // Reverse synchronization: A in L1 has been consumed by MTE1
                }

                if (k1 == (kL1Loops - 1)) {
                    if (nL1 == 0 && mL1 == 0) { // Wait before the first Fixpipe
                        CrossCoreWaitFlag(constInfo.syncV1NupdateC2);
                    }

                    if (!info.isFirstSInnerLoop) {
                        SetAtomicAdd<MM_OUT_T>();
                    }
                    // ND
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    fixParams.dstStride = nSize; // Gap between two rows in mm2ResGm
                    fixParams.ndNum = 1;         // Output in ND format
                    fixParams.unitFlag = 0b11;

                    uint64_t mm2Offset = (mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE) * nSize + nL1 * N_SPLIT_SIZE;
                    Fixpipe(mm2ResGm[(info.bn2IdxInCurCore % (constInfo.preLoadNum)) *
                            constInfo.bmm2ResUbSize + mm2Offset], cL0Tensor, fixParams);
                    if (!info.isFirstSInnerLoop) {
                        SetAtomicNone();
                    }
                }

                if (mL1Loops == 2) {
                    cL0BufIter++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]); // Reverse synchronization: L1 has been consumed by MTE1
        }
        // cL0BufIter is no longer in use.
        if (mL1Loops == 1) {
            cL0BufIter++;
        }
    }
    qpL1BufIter += mL1Loops;
}

#endif // FUSED_SPARSE_ATTENTION_OVERLAP_SERVICE_CUBE_MLA_H
