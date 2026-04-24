/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attn_sharedkv_scfa_block_cube.h
 * \brief use 7 buffer for matmul l1, better pipeline
 */
#ifndef SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H
#define SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../sparse_attn_sharedkv_common.h"

namespace SASKernel {
template <typename SAST>
class SASCubeBlock {
public:
    // 中间计算数据类型为float, 高精度模式
    using T = float;
    using Q_T = typename SAST::queryType;
    using KV_T = typename SAST::kvType;
    using OUT_T = typename SAST::outputType;
    using MM_OUT_T = T;

    __aicore__ inline SASCubeBlock(){};
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<KV_T> oriKvGm,
                                               GlobalTensor<KV_T> cmpKV, GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM_OUT_T> mm2ResGm,
                                               GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitPageAttentionInfo(GlobalTensor<KV_T> oriKvGm, const GlobalTensor<KV_T> &kvMergeGm,
                                                 GlobalTensor<int32_t> oriBlockTableGm,
                                                 GlobalTensor<int32_t> cmpBlockTableGm);
    __aicore__ inline void InitBuffers(TPipe *pipe);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo);
    __aicore__ inline void ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo);

private:
    static constexpr bool PAGE_ATTENTION = SAST::pageAttention;
    static constexpr int TEMPLATE_MODE = SAST::templateMode;
    static constexpr bool FLASH_DECODE = SAST::flashDecode;
    static constexpr SAS_LAYOUT LAYOUT_T = SAST::layout;
    static constexpr SAS_LAYOUT KV_LAYOUT_T = SAST::kvLayout;

    static constexpr uint32_t M_SPLIT_SIZE = 128;     // m方向切分
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // n方向切分
    static constexpr uint32_t K_L0_SPLIT_SIZE = 128;  // k方向L0切分
    static constexpr uint32_t K_L1_SPLIT_SIZE = 256;  // k方向L1切分
    static constexpr uint32_t N_WORKSPACE_SIZE = 512; // n方向切分

    static constexpr uint32_t L1_BLOCK_SIZE = (64 * 512 * sizeof(Q_T));
    static constexpr uint32_t L1_BLOCK_OFFSET = 64 * 512;

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    // mte2 <> mte1 EventID
    // L1 3buf, 使用3个eventId
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

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};                    // isSetFMatrix isSetPadding
    static constexpr uint32_t mte21QPIds[4] = {L1_EVENT0, L1_EVENT1, L1_EVENT2, L1_EVENT3}; // mte12复用
    static constexpr uint32_t mte21KVIds[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};

    ConstInfo constInfo{};

    // L1分成3块buf, 用于记录
    uint32_t qpL1BufIter = 0;
    uint32_t kvL1BufIter = -1;
    uint32_t abL0BufIter = 0;
    uint32_t cL0BufIter = 0;

    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> oriKvGm;
    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> cmpKvGm;

    // mm2
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<OUT_T> attentionOutGm;

    // block_table
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;

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
        uint32_t idxMap[] = {0, 2}; // 确保0块和1块连在一起, 2和3块连在一起, 来保证同一m块的地址相连
        return idxMap[mIdx % 2] + k1Idx;
    }

    __aicore__ inline void CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                          uint32_t mSizeAct, uint32_t headSize, uint32_t headOffset);

    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                          uint32_t subMSizeAct, uint32_t nSize, uint32_t nOffset);
    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t mSize, uint32_t kSize);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_T> &bL0Tensor, LocalTensor<KV_T> &bL1Tensor, uint32_t idx,
                                        uint32_t kSplitSize, uint32_t kSize, uint32_t nSize);
};

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<KV_T> oriKvGm,
                                                               GlobalTensor<KV_T> cmpKvGm,
                                                               GlobalTensor<MM_OUT_T> mm1ResGm)
{
    // mm1
    this->queryGm = queryGm;
    this->oriKvGm = oriKvGm;
    this->cmpKvGm = cmpKvGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm,
                                                               GlobalTensor<MM_OUT_T> mm2ResGm,
                                                               GlobalTensor<OUT_T> attentionOutGm)
{
    // mm2
    this->vec1ResGm = vec1ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SAST>
__aicore__ inline void
SASCubeBlock<SAST>::InitPageAttentionInfo(GlobalTensor<KV_T> oriKvGm, const GlobalTensor<KV_T> &kvMergeGm,
                                          GlobalTensor<int32_t> oriBlockTableGm, GlobalTensor<int32_t> cmpBlockTableGm)
{
    this->oriKvGm = oriKvGm;
    this->kvMergeGm_ = kvMergeGm;
    this->oriBlockTableGm = oriBlockTableGm;
    this->cmpBlockTableGm = cmpBlockTableGm;
}

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4);
    l1QPTensor = bufQPL1.Get<Q_T>();
    pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3);
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

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::AllocEventID()
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

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::FreeEventID()
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

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor,
                                                      uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor, const RunInfo &info,
                                                          uint32_t mSeqIdx, uint32_t mSizeAct, uint32_t headSize,
                                                          uint32_t headOffset)
{
    auto srcGm = queryGm[info.tensorAOffset + mSeqIdx * constInfo.headDim + headOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, headSize, constInfo.headDim);
}

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor,
                                                        uint32_t idx, uint32_t kSplitSize, uint32_t mSize,
                                                        uint32_t kSize)
{
    LocalTensor<KV_T> srcTensor = aL1Tensor[mSize * kSplitSize * idx];
    LoadData3DParamsV2<KV_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = mSize / 16; // Hin=M1=8
    loadData3DParams.l1W = 16;         // Win=M0
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

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

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::LoadDataMm1B(LocalTensor<KV_T> &l0Tensor, LocalTensor<KV_T> &l1Tensor,
                                                        uint32_t idx, uint32_t kSplitSize, uint32_t kSize,
                                                        uint32_t nSize)
{
    // N 方向全载
    LocalTensor<KV_T> srcTensor = l1Tensor[nSize * kSplitSize * idx];

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = (nSize + 15) / 16 * kSize / (32 / sizeof(KV_T));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0Tensor, srcTensor, loadData2DParams);
}

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info,
                                                          uint32_t mSeqIdx, uint32_t subMSizeAct, uint32_t nSize,
                                                          uint32_t nOffset)
{
    auto srcGm = vec1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                           mSeqIdx * info.actualSingleProcessSInnerSizeAlign + nOffset];
    CopyGmToL1(aL1Tensor, srcGm, subMSizeAct, nSize, info.actualSingleProcessSInnerSizeAlign);
}

template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mL1Size = M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = SASAlign(M_SPLIT_SIZE, 16);
    uint32_t mL1Loops = CeilDiv(mSize, M_SPLIT_SIZE);

    uint32_t nSize = info.actualSingleProcessSInnerSize;
    uint32_t nL1Size = N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = SASAlign(N_SPLIT_SIZE, 16);
    uint32_t nL1Loops = CeilDiv(nSize, N_SPLIT_SIZE);

    uint32_t kSize = 512;
    uint32_t kL1Size = 256;
    uint32_t kL1Loops = 2;
    uint32_t kL0Size = 128;
    uint32_t kL0Loops = CeilDiv(kL1Size, kL0Size);

    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> kTensor;
    uint32_t ka = 0, kb = 0;
    uint32_t copyRowCnt = 0;
    uint32_t copyRowCntTmp = 0;

    // L1 切n切k
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // L1切n, 512/128=4
        if (nL1 == (nL1Loops - 1)) {
            // 尾块重新计算size
            nL1Size = nSize - (nL1Loops - 1) * N_SPLIT_SIZE;
            nL1SizeAlign = SASAlign(nL1Size, 16);
        }
        copyRowCntTmp = copyRowCnt;

        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) {
            kvL1BufIter++;
            uint32_t kb = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
            // 从k当中取当前的块
            bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
            uint32_t curSeqIdx = info.s2BatchOffset + nL1 * N_SPLIT_SIZE;
            uint32_t copyFinishRowCnt = 0;

            if (info.isOri) {
                uint32_t curS2Offset = info.s2Idx * constInfo.s2BaseSize + info.s2StartPoint;
                if constexpr (PAGE_ATTENTION) {
                    while (copyFinishRowCnt < nL1Size) {
                        // printf("oriIn copyFinishRowCnt=%u\n", copyFinishRowCnt);
                        // CalcTopKBlockInfo(info, curTopKIdx, curOffsetInSparseBlock, curSeqIdx, copyRowCnt, idInTopK);
                        copyRowCnt = constInfo.paOriBlockSize - curS2Offset % constInfo.paOriBlockSize; // 由于ori_left的存在， 即使第一块搬运也可能并非是pa_block的零点位
                        // printf("info.s2Idx=%u, constInfo.s2BaseSize=%u, info.s2StartPoint=%u, constInfo.paOriBlockSize=%u, curS2Offset=%u, copyRowCnt=%u\n", info.s2Idx, constInfo.s2BaseSize, info.s2StartPoint, constInfo.paOriBlockSize, curS2Offset, copyRowCnt);
                        if (copyFinishRowCnt + copyRowCnt > nL1Size) {
                            copyRowCnt = nL1Size - copyFinishRowCnt;
                        }

                        Position startPos;
                        startPos.bIdx = info.bIdx;
                        startPos.n2Idx = info.n2Idx;
                        // startPos.s2Idx = idInTopK * constInfo.sparseBlockSize + curOffsetInSparseBlock;
                        startPos.s2Idx = curS2Offset;
                        // 256、32等待7buf命名更改
                        startPos.dIdx = kL1 * 256;  // mm1 右矩阵 bn2s2d, d为k轴不切; mm2 右矩阵, s2为k轴, d轴切分

                        PAShape shape;
                        shape.blockSize = constInfo.paOriBlockSize;
                        shape.headNum = constInfo.kvHeadNum;
                        shape.headDim = constInfo.headDim;
                        shape.actHeadDim = 256;
                        shape.maxblockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
                        shape.copyRowNum = copyRowCnt;
                        shape.copyRowNumAlign = nL1SizeAlign;
                        if (kL1 == 0) {
                            kTensor = bL1Tensor[copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(kTensor, oriKvGm, oriBlockTableGm, shape, startPos);

                        } else {
                            kTensor = bL1Tensor[copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(kTensor, oriKvGm, oriBlockTableGm, shape, startPos);
                        }
                        // 更新循环变量
                        copyFinishRowCnt += copyRowCnt;
                        curSeqIdx += copyRowCnt;
                        curS2Offset += copyRowCnt;
                    }
                } else {
                    if (kL1 == 0) {
                        Nd2NzParams nd2nzPara;
                        nd2nzPara.ndNum = 1;
                        nd2nzPara.nValue = nL1Size;
                        nd2nzPara.dValue = constInfo.headDim >> 1;
                        nd2nzPara.srcDValue = constInfo.headDim;
                        nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                        nd2nzPara.dstNzNStride = 1;
                        nd2nzPara.srcNdMatrixStride = 0;
                        nd2nzPara.dstNzMatrixStride = 0;
                        DataCopy(bL1Tensor, oriKvGm[info.tensorBOffset + curS2Offset * constInfo.headDim +
                                nL1 * N_SPLIT_SIZE * constInfo.headDim], nd2nzPara);
                    } else {
                        Nd2NzParams nd2nzPara;
                        nd2nzPara.ndNum = 1;
                        nd2nzPara.nValue = nL1Size;
                        nd2nzPara.dValue = constInfo.headDim >> 1;
                        nd2nzPara.srcDValue = constInfo.headDim;
                        nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                        nd2nzPara.dstNzNStride = 1;
                        nd2nzPara.srcNdMatrixStride = 0;
                        nd2nzPara.dstNzMatrixStride = 0;
                        DataCopy(bL1Tensor, 
                                    oriKvGm[info.tensorBOffset + curS2Offset * constInfo.headDim + (constInfo.headDim >> 1) +
                                        nL1 * N_SPLIT_SIZE * constInfo.headDim], 
                                    nd2nzPara);
                    }
                }
            } else {
                if (kL1 == 0) {
                    Nd2NzParams nd2nzPara;
                    nd2nzPara.ndNum = 1;
                    nd2nzPara.nValue = nL1Size;
                    nd2nzPara.dValue = constInfo.headDim >> 1;
                    nd2nzPara.srcDValue = constInfo.headDim;
                    nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                    nd2nzPara.dstNzNStride = 1;
                    nd2nzPara.srcNdMatrixStride = 0;
                    nd2nzPara.dstNzMatrixStride = 0;
                    DataCopy(bL1Tensor,
                             kvMergeGm_[info.cmpLoop % 4 * N_WORKSPACE_SIZE * kSize +
                                        nL1 * N_SPLIT_SIZE * constInfo.headDim],
                             nd2nzPara);
                } else {
                    Nd2NzParams nd2nzPara;
                    nd2nzPara.ndNum = 1;
                    nd2nzPara.nValue = nL1Size;
                    nd2nzPara.dValue = constInfo.headDim >> 1;
                    nd2nzPara.srcDValue = constInfo.headDim;
                    nd2nzPara.dstNzC0Stride = nL1SizeAlign;
                    nd2nzPara.dstNzNStride = 1;
                    nd2nzPara.srcNdMatrixStride = 0;
                    nd2nzPara.dstNzMatrixStride = 0;
                    DataCopy(bL1Tensor,
                             kvMergeGm_[info.cmpLoop % 4 * N_WORKSPACE_SIZE * kSize + (constInfo.headDim >> 1) +
                                        nL1 * N_SPLIT_SIZE * constInfo.headDim],
                             nd2nzPara);
                }
            }

            SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            mL1Size = M_SPLIT_SIZE;
            mL1SizeAlign = SASAlign(M_SPLIT_SIZE, 16U);
            for (uint32_t mL1 = 0; mL1 < mL1Loops; mL1++) {
                uint32_t aL1PaddingSize = 0; // 用于使左矩阵对齐到尾部, 以保证两块32K内存连续
                if (mL1 == (mL1Loops - 1)) {
                    mL1Size = mSize - (mL1Loops - 1) * M_SPLIT_SIZE;
                    mL1SizeAlign = SASAlign(mL1Size, 16U);
                    aL1PaddingSize = (M_SPLIT_SIZE - mL1SizeAlign) * 256;
                }
                uint32_t mIdx = qpL1BufIter + mL1;
                ka = GetQPL1RealIdx(mIdx, kL1);
                LocalTensor<Q_T> aL1Tensor = l1QPTensor[ka * L1_BLOCK_OFFSET + (1 - kL1) * aL1PaddingSize];
                if (nL1 == 0) {
                    if (kL1 == 0) {
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka + 1]);
                        CopyInMm1AToL1(aL1Tensor, info, mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size, 256, 0);
                    } else {
                        LocalTensor<Q_T> qTmpTensor = aL1Tensor;
                        CopyInMm1AToL1(qTmpTensor, info, mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE, mL1Size, 256,
                                       256);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                }
                // 使用unitflag同步
                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(cL0BufIter % 2) *
                                      (L0C_PP_SIZE / sizeof(MM_OUT_T))]; // 需要保证cL0BufIter和m步调一致
                for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1A(aL0Tensor, aL1Tensor, kL0, kL0Size, mL1SizeAlign, kL0Size);
                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadDataMm1B(bL0Tensor, bL1Tensor, kL0, kL0Size, kL0Size, nL1SizeAlign);
                    SetFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));
                    WaitFlag<HardEvent::MTE1_M>(Mte1MmABEventId(abL0BufIter % 2));

                    MmadParams mmadParams;
                    mmadParams.m = mL1SizeAlign;
                    mmadParams.n = nL1SizeAlign;
                    mmadParams.k = kL0Size;
                    mmadParams.cmatrixInitVal = (kL1 == 0 && kL0 == 0);
                    mmadParams.cmatrixSource = false;
                    mmadParams.unitFlag =
                        (kL1 == 1 && kL0 == (kL0Loops - 1)) ? 0b11 : 0b10; // 累加最后一次翻转flag, 表示可以搬出
                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    SetFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    abL0BufIter++;
                }

                if (nL1 == (nL1Loops - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]); // 反向同步, 表示L1中的A已经被mte1消费完
                }

                if (kL1 == 1) { // 最后一轮kL1循环
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    // 改成nSizeAlign
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                    fixParams.unitFlag = 0b11;
                    fixParams.ndNum = 1; // 输出ND

                    Fixpipe(mm1ResGm[(info.loop % (constInfo.preLoadNum)) * constInfo.mmResUbSize + nL1 * N_SPLIT_SIZE +
                                     (mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE) *
                                         info.actualSingleProcessSInnerSizeAlign],
                            cL0Tensor, fixParams);
                }
                if (mL1Loops == 2) {
                    cL0BufIter++;
                }
            }

            SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]); // 反向同步, 表示L1已经被mte1消费完
        }
        if (mL1Loops == 1) {
            cL0BufIter++;
        }
    }
    qpL1BufIter += mL1Loops;
}


template <typename SAST>
__aicore__ inline void SASCubeBlock<SAST>::ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mSizeAlign = (mSize + 16 - 1) / 16;
    uint32_t mL1Loops = (mSize + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;
    uint32_t mL1SizeAlign = M_SPLIT_SIZE; // 16对齐
    uint32_t mL1Size = M_SPLIT_SIZE;      // m的实际大小

    uint32_t nSize = BlockAlign<KV_T>(constInfo.headDim);
    uint32_t nL1Loops = (nSize + N_SPLIT_SIZE - 1) / N_SPLIT_SIZE;
    uint32_t nL1SizeAlign = N_SPLIT_SIZE; // 16对齐
    uint32_t nL1Size = N_SPLIT_SIZE;      // n的实际大小

    uint32_t kSize = info.actualSingleProcessSInnerSize;
    uint32_t kL1Size = 256;
    uint32_t kL1SizeAlign = SASAlign(kL1Size, 16U);
    uint32_t kL1Loops = (kSize + kL1Size - 1) / kL1Size;
    uint32_t kL0Size = 128;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
    uint32_t kL0SizeAlign = kL0Size;
    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> subvTensor;

    // ka表示左矩阵4buf选择哪一块buf, kb表示右矩阵3buf选择哪一块buf
    uint32_t ka = 0, kb = 0;
    uint32_t mBaseIdx = qpL1BufIter;
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // n切L1
        if (nL1 == (nL1Loops - 1)) {
            // 尾块
            nL1Size = nSize - (nL1Loops - 1) * N_SPLIT_SIZE;
            nL1SizeAlign = SASAlign(nL1Size, 16U);
        }
        // k l1写成一个循环, 和mm1保持一致
        kL1Size = 256;
        kL1SizeAlign = SASAlign(kL1Size, 16U);
        uint32_t copyRowCnt = 0;

        for (uint32_t k1 = 0; k1 < kL1Loops; k1++) { // k切L1, 这里套了一层l0来操作
            if (k1 == (kL1Loops - 1)) {
                // 尾块
                kL1Size = kSize - (kL1Loops - 1) * 256;
                kL1SizeAlign = SASAlign(kL1Size, 16U);
            }
            kvL1BufIter++;
            uint32_t kb = kvL1BufIter % 3;
            WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
            bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
            uint32_t kOffset = k1 * kL0Loops;
            kL0Size = 128;
            // 此处必须先初始化kL0Size, 再求kL0Loops, 否则由于循环会改变kL0Size大小, 导致kL0Loops错误
            kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
            kL0SizeAlign = kL0Size;
            for (uint32_t kL1 = kOffset; kL1 < kL0Loops + kOffset; kL1++) { // 128 循环搬pa
                if (kL1 == kOffset + kL0Loops - 1) {
                    // 尾块
                    kL0Size = kL1Size - (kL0Loops - 1) * kL0Size;
                    kL0SizeAlign = SASAlign(kL0Size, 16U);
                }

                uint32_t curSeqIdx = info.s2BatchOffset + (kL1 - kOffset) * 128 + k1 * 256;
                uint32_t copyFinishRowCnt = 0;

                if (info.isOri) {
                    uint32_t curS2Offset = info.s2Idx * constInfo.s2BaseSize + info.s2StartPoint;
                    if constexpr (PAGE_ATTENTION) {
                        while (copyFinishRowCnt < kL0Size) {
                            copyRowCnt = constInfo.paOriBlockSize - curS2Offset % constInfo.paOriBlockSize;
                            if (copyFinishRowCnt + copyRowCnt > kL0Size) {
                                copyRowCnt = kL0Size - copyFinishRowCnt;
                            }
                            Position startPos;
                            startPos.bIdx = info.bIdx;
                            startPos.n2Idx = info.n2Idx;
                            startPos.s2Idx = curS2Offset;
                            startPos.dIdx = nL1 * N_SPLIT_SIZE;  // mm1 右矩阵 bn2s2d, d为k轴不切; mm2 右矩阵, s2为k轴, d轴切分
                            PAShape shape;
                            shape.blockSize = constInfo.paOriBlockSize;
                            shape.headNum = constInfo.kvHeadNum;
                            shape.headDim = constInfo.headDim;
                            shape.actHeadDim = nL1Size;
                            shape.maxblockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
                            shape.copyRowNum = copyRowCnt;
                            shape.copyRowNumAlign = kL0SizeAlign;
                            subvTensor = bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE + copyFinishRowCnt * 16];
                            DataCopyPA<KV_T, KV_LAYOUT_T>(subvTensor, oriKvGm, oriBlockTableGm, shape, startPos);

                            // 更新循环变量
                            copyFinishRowCnt += copyRowCnt;
                            curSeqIdx += copyRowCnt;
                            curS2Offset += copyRowCnt;
                        }
                    } else {
                        Nd2NzParams nd2nzPara;
                        nd2nzPara.ndNum = 1;
                        nd2nzPara.nValue = kL0Size;      // 行数
                        nd2nzPara.dValue = N_SPLIT_SIZE; // constInfo.headDim;
                        nd2nzPara.srcDValue = constInfo.headDim;
                        nd2nzPara.dstNzC0Stride = kL0SizeAlign;
                        nd2nzPara.dstNzNStride = 1;
                        nd2nzPara.srcNdMatrixStride = 0;
                        nd2nzPara.dstNzMatrixStride = 0;
                        DataCopy(bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE],
                                oriKvGm[info.tensorBOffset + curS2Offset * constInfo.headDim + kL1 * 128 * constInfo.headDim +
                                nL1 * N_SPLIT_SIZE], nd2nzPara);
                    }
                } else {
                    Nd2NzParams nd2nzPara;
                    nd2nzPara.ndNum = 1;
                    nd2nzPara.nValue = kL0Size;      // 行数
                    nd2nzPara.dValue = N_SPLIT_SIZE; // constInfo.headDim;
                    nd2nzPara.srcDValue = constInfo.headDim;
                    nd2nzPara.dstNzC0Stride = kL0SizeAlign;
                    nd2nzPara.dstNzNStride = 1;
                    nd2nzPara.srcNdMatrixStride = 0;
                    nd2nzPara.dstNzMatrixStride = 0;
                    DataCopy(bL1Tensor[(kL1 - kOffset) * 128 * N_SPLIT_SIZE],
                             kvMergeGm_[info.cmpLoop % 4 * N_WORKSPACE_SIZE * 512 + kL1 * 128 * constInfo.headDim +
                                        nL1 * N_SPLIT_SIZE],
                             nd2nzPara);
                }
            }
            SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            mL1SizeAlign = M_SPLIT_SIZE;
            mL1Size = M_SPLIT_SIZE; // m的实际大小
            for (uint32_t mL1 = 0; mL1 < mL1Loops; mL1++) {
                if (mL1 == (mL1Loops - 1)) {
                    // 尾块
                    mL1Size = mSize - (mL1Loops - 1) * M_SPLIT_SIZE;
                    mL1SizeAlign = SASAlign(mL1Size, 16U);
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
                                      (L0C_PP_SIZE / sizeof(MM_OUT_T))]; // 需要保证cL0BufIter和m步调一致
                uint32_t baseK = 128;
                uint32_t baseN = 128;
                kL0Size = 128;
                kL0SizeAlign = kL0Size;
                for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                    if (kL0 + 1 == kL0Loops) {
                        kL0Size = kL1Size - (kL0Loops - 1) * kL0Size;
                        kL0SizeAlign = SASAlign(kL0Size, 16U);
                    }
                    WaitFlag<HardEvent::M_MTE1>(Mte1MmABEventId(abL0BufIter % 2));
                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LoadData3DParamsV2<KV_T> loadData3DParamsForB;
                    loadData3DParamsForB.l1H = kL0SizeAlign / 16; // 源操作数height
                    loadData3DParamsForB.l1W = 16;                // 源操作数weight=16，目的height=l1H*L1W
                    loadData3DParamsForB.padList[0] = 0;
                    loadData3DParamsForB.padList[1] = 0;
                    loadData3DParamsForB.padList[2] = 0;
                    loadData3DParamsForB.padList[3] = 255; // 尾部数据不影响滑窗的结果

                    loadData3DParamsForB.mExtension = kL0SizeAlign; // 在目的操作数height维度的传输长度
                    loadData3DParamsForB.kExtension = nL1SizeAlign; // 在目的操作数width维度的传输长度
                    loadData3DParamsForB.mStartPt = 0;              // 卷积核在目的操作数width维度的起点
                    loadData3DParamsForB.kStartPt = 0;              // 卷积核在目的操作数height维度的起点
                    loadData3DParamsForB.strideW = 1;
                    loadData3DParamsForB.strideH = 1;
                    loadData3DParamsForB.filterW = 1;
                    loadData3DParamsForB.filterSizeW = false; // 是否在filterW的基础上将卷积核width增加256个元素
                    loadData3DParamsForB.filterH = 1;
                    loadData3DParamsForB.filterSizeH = false; // 是否在filterH的基础上将卷积核height增加256个元素
                    loadData3DParamsForB.dilationFilterW = 1; // 卷积核width膨胀系数
                    loadData3DParamsForB.dilationFilterH = 1; // 卷积核height膨胀系数
                    loadData3DParamsForB.enTranspose = 1;     // 是否启用转置功能
                    loadData3DParamsForB.fMatrixCtrl =
                        0; // 使用FMATRIX_LEFT还是使用FMATRIX_RIGHT，=0使用FMATRIX_LEFT，=1使用FMATRIX_RIGHT 1
                    loadData3DParamsForB.channelSize =
                        nL1SizeAlign; // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
                    LoadData<KV_T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[kL0 * baseK * baseN], loadData3DParamsForB);

                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LoadData3DParamsV2<KV_T> loadData3DParamsForA;
                    loadData3DParamsForA.l1H = mL1SizeAlign / 16; // 源操作数height
                    loadData3DParamsForA.l1W = 16;                // 源操作数weight
                    loadData3DParamsForA.padList[0] = 0;
                    loadData3DParamsForA.padList[1] = 0;
                    loadData3DParamsForA.padList[2] = 0;
                    loadData3DParamsForA.padList[3] = 255; // 尾部数据不影响滑窗的结果

                    loadData3DParamsForA.mExtension = mL1SizeAlign; // 在目的操作数height维度的传输长度
                    loadData3DParamsForA.kExtension = kL0SizeAlign; // 在目的操作数width维度的传输长度
                    loadData3DParamsForA.mStartPt = 0;              // 卷积核在目的操作数width维度的起点
                    loadData3DParamsForA.kStartPt = 0;              // 卷积核在目的操作数height维度的起点
                    loadData3DParamsForA.strideW = 1;         // 卷积核在源操作数width维度滑动的步长
                    loadData3DParamsForA.strideH = 1;         // 卷积核在源操作数height维度滑动的步长
                    loadData3DParamsForA.filterW = 1;         // 卷积核width
                    loadData3DParamsForA.filterSizeW = false; // 是否在filterW的基础上将卷积核width增加256个元素
                    loadData3DParamsForA.filterH = 1;         // 卷积核height
                    loadData3DParamsForA.filterSizeH = false; // 是否在filterH的基础上将卷积核height增加256个元素
                    loadData3DParamsForA.dilationFilterW = 1; // 卷积核width膨胀系数
                    loadData3DParamsForA.dilationFilterH = 1; // 卷积核height膨胀系数
                    loadData3DParamsForA.enTranspose = 0; // 是否启用转置功能，对整个目标矩阵进行转置
                    loadData3DParamsForA.fMatrixCtrl = 0;
                    loadData3DParamsForA.channelSize =
                        kL0SizeAlign; // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
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

                if (nL1 == (nL1Loops - 1)) { // nL1最后一轮, 需要将B驻留在L1中, 用于下一轮的计算？
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]); // 反向同步, 表示L1中的A已经被mte1消费完
                }

                if (k1 == (kL1Loops - 1)) {
                    // ND
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    fixParams.dstStride = nSize; // mm2ResGm两行之间的间隔
                    fixParams.ndNum = 1;         // 输出ND
                    fixParams.unitFlag = 0b11;

                    uint64_t mm2Offset = (mSplitInfo.nBufferStartM + mL1 * M_SPLIT_SIZE) * nSize + nL1 * N_SPLIT_SIZE;
                    Fixpipe(mm2ResGm[(info.loop % (constInfo.preLoadNum)) * constInfo.bmm2ResUbSize + mm2Offset],
                            cL0Tensor, fixParams);
                }

                if (mL1Loops == 2) {
                    cL0BufIter++;
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]); // 反向同步, 表示L1已经被mte1消费完
        }
        // cL0BufIter已经不在使用
        if (mL1Loops == 1) {
            cL0BufIter++;
        }
    }
    qpL1BufIter += mL1Loops;
}
} // namespace SASKernel
#endif // SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H