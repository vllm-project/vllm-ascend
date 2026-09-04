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
 * \file ifa_service_matmul_full_quant.h
 * \brief
 */
#ifndef IFA_SERVICE_MATMUL_FULL_QUANT_H
#define IFA_SERVICE_MATMUL_FULL_QUANT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../ifa_public_define.h"

template <typename IFAT>
class IfaMatmulFullQuant {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;

    using KV_T = typename IFAT::kvType;
    using OUT_T = typename IFAT::outputType;

    using ROPE_T = typename IFAT::orginalType;

    static constexpr bool PAGE_ATTENTION = IFAT::pageAttention;
    static constexpr LAYOUT LAYOUT_T = IFAT::layout;
    static constexpr LAYOUT KV_LAYOUT_T = IFAT::kvLayout;
    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;
    static constexpr bool QUANT = (IsSameType<KV_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);

    using MMAD_OUT_T = typename AscendC::Conditional<QUANT, int32_t, T>::type;
    using MM1_OUT_T = typename AscendC::Conditional<QUANT, int32_t, T>::type;
#ifdef QUANT_MM2_FP16
    using MM2_OUT_T = typename AscendC::Conditional<QUANT, half, T>::type;
#else
    using MM2_OUT_T = typename AscendC::Conditional<QUANT, int32_t, T>::type;
#endif

    __aicore__ inline IfaMatmulFullQuant() {};
    __aicore__ inline void InitParams(uint64_t qHeadNum, uint64_t kvHeadNum, uint64_t headDim, uint64_t headDimRope,
                                      uint64_t qSeqSize, uint32_t mmResUbSize, uint32_t bmm2ResUbSize);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<KV_T> queryGm, GlobalTensor<ROPE_T> qRopeGm,
        GlobalTensor<KV_T> keyGm, GlobalTensor<ROPE_T> kRopeGm, GlobalTensor<MM1_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
        GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitPageAttentionInfo(GlobalTensor<int32_t> blockTableGm, uint32_t blockSize,
        uint32_t maxBlockNumPerBatch);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void UpdateKey(GlobalTensor<KV_T> keyGm);
    __aicore__ inline void UpdateValue(GlobalTensor<KV_T> valueGm);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const ExtraInfoMla &info);
    __aicore__ inline void ComputeMm2(const ExtraInfoMla &info);

protected:
    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    template <typename T> __aicore__ inline size_t BlockAlign(size_t s)
    {
        if constexpr (IsSameType<T, int4b_t>::value) {
            return (s + 63) / 64 * 64;
        }
        size_t n = (32 / sizeof(KV_T));
        return (s + n - 1) / n * n;
    }

    template <typename DT>
    __aicore__ inline void CopyGmToL1(LocalTensor<DT> &l1Tensor, GlobalTensor<DT> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride, uint32_t dstNAlign = 16);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T>& aL1Tensor, const ExtraInfoMla &info, uint32_t mSize, uint64_t mOffset);
    __aicore__ inline void CopyInMm1ARopeToL1(LocalTensor<ROPE_T>& aL1Tensor, const ExtraInfoMla &info, uint32_t mSize, uint64_t mOffset);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<KV_T>& bL1Tensor, const ExtraInfoMla &info, uint32_t subNid, uint32_t subNSize, uint32_t nSplitSize);
    __aicore__ inline void CopyInMm1BRopeToL1(LocalTensor<ROPE_T>& bL1Tensor, const ExtraInfoMla &info, uint32_t subNid, uint32_t subNSize, uint32_t nSplitSize);
    __aicore__ inline void CopyInMm1BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, const uint64_t keyGmBaseOffset,
        uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount);
    __aicore__ inline void CopyInMm1BRopeToL1ForPA(LocalTensor<ROPE_T>& bL1Tensor, const uint64_t keyGmBaseOffset,
        uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, const ExtraInfoMla &info, uint32_t mSize, uint64_t mOffset);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<KV_T>& aL1Tensor, const ExtraInfoMla &info, uint32_t subKid,
        uint32_t kSplitSize, uint32_t subNid, uint32_t nSplitSize, uint32_t subKSize, uint32_t subNSize);
    __aicore__ inline void CopyInMm2BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, const uint64_t valueGmBaseOffset,
        uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount,
        uint32_t copyStartColumnCount, uint32_t copyColumnCount);

    template <typename DT>
    __aicore__ inline void LoadDataL0A(LocalTensor<DT>& aL0Tensor, const LocalTensor<DT>& aL1Tensor, uint32_t kidx, uint32_t kSplitSize, uint32_t mSize, uint32_t subkSize);
    template <typename DT>
    __aicore__ inline void LoadDataMm1B(LocalTensor<DT>& bL0Tensor, LocalTensor<DT>& bL1Tensor, uint32_t idx, uint32_t kSplitSize, uint32_t kSize, uint32_t nSize);
    __aicore__ inline void LoadDataMm2B(LocalTensor<KV_T>& bL0Tensor, LocalTensor<KV_T>& bL1Tensor, uint32_t idx, uint32_t nSize, uint32_t subkSize, uint32_t kSplitSize, uint32_t kSize);

protected:
    // mm1
    GlobalTensor<KV_T> queryGm;
    GlobalTensor<ROPE_T> qRopeGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<ROPE_T> kRopeGm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;

    // mm2
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<OUT_T> attentionOutGm;

    // pageAttention
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // block_table
    GlobalTensor<int32_t> blockTableGm;

    // params
    uint32_t mmResUbSize = 0U;
    uint32_t bmm2ResUbSize = 0U;

    uint64_t qHeadNum = 0ULL;
    static constexpr uint64_t kvHeadNum = 1ULL;
    uint64_t gSize = 0ULL;

    uint64_t qSeqSize = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;

private:
    // L1
    static constexpr uint32_t L1_PQ_SIZE = (128 * 512 + 128 * 64 * 2);   // QP共用Buffer：128*512 + 128*64*2 = 80K   
    static constexpr uint32_t L1_KV_SIZE = (128 * 512 + 128 * 64 * 2);  // KV开DB：128*512 + 128*64*2 = 80K * 3 = 240K

    static constexpr uint32_t BT_PP_SIZE = (2 * 1024);

    // L0
    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    // mte2 <> mte1
    static constexpr uint32_t QP_EVENT0 = EVENT_ID2;
    static constexpr uint32_t QP_EVENT1 = EVENT_ID3;
    static constexpr uint32_t KV_EVENT0 = EVENT_ID4;
    static constexpr uint32_t KV_EVENT1 = EVENT_ID5;
    static constexpr uint32_t KV_EVENT2 = EVENT_ID6;
    static constexpr uint32_t BIAS_EVENT0 = EVENT_ID7;

    static constexpr uint32_t KV_BUFF_NUM = 3;

    // m <> mte1
    static constexpr uint32_t L0A_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0A_EVENT1 = EVENT_ID4;

    static constexpr uint32_t BT_EVENT0 = EVENT_ID5;
    static constexpr uint32_t BT_EVENT1 = EVENT_ID6;

    // fix <> m
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;

    static constexpr uint32_t M_SPLIT_SIZE = 256;

    TBuf<TPosition::A1> qpBufL1;
    LocalTensor<KV_T> qpL1Tensor;

    TBuf<TPosition::A1> kvBufL1;
    LocalTensor<KV_T> kvL1Tensor;

    // bias的buffer,组DB
    TBuf<TPosition::C1> biasBufL1;
    LocalTensor<int32_t> biasL1Tensor;

    TBuf<TPosition::A2> tmpBufL0A;
    LocalTensor<KV_T> aL0TensorPingPong;

    // L0_B
    TBuf<TPosition::B2> tmpBufL0B;
    LocalTensor<KV_T> bL0TensorPingPong;

    // L0_C
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<MMAD_OUT_T> cL0TensorPingPong;

    // BT buffer
    TBuf<TPosition::C2> tmpBufBias;
    LocalTensor<int32_t> biasBTTensorPingPong;

    uint32_t qpL1BufIter = -1;
    uint32_t kvL1BufIter = -1;
    uint32_t aL0BufIter = 0;
    uint32_t bL0BufIter = 0;
    uint32_t cL0BufIter = 0;
    uint32_t biasL1BufIter = 0; // double buffer
    uint32_t btBufIter = 0; // double buffer
    bool isFirstIter = true;  
};

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::InitParams(uint64_t qHeadNum, uint64_t kvHeadNum, uint64_t headDim,
                                                       uint64_t headDimRope, uint64_t qSeqSize, uint32_t mmResUbSize,
                                                       uint32_t bmm2ResUbSize)
{
    this->qHeadNum = qHeadNum;
    this->qSeqSize = qSeqSize;
    this->mmResUbSize = mmResUbSize;
    this->bmm2ResUbSize = bmm2ResUbSize;
    this->gSize = qHeadNum / kvHeadNum;
    this->isFirstIter = true;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::InitMm1GlobalTensor(GlobalTensor<KV_T> queryGm, GlobalTensor<ROPE_T> qRopeGm,
                                                                GlobalTensor<KV_T> keyGm, GlobalTensor<ROPE_T> kRopeGm,
                                                                GlobalTensor<MM1_OUT_T> mm1ResGm)
{
    // mm1
    this->queryGm = queryGm;
    this->qRopeGm = qRopeGm;
    this->keyGm = keyGm;
    this->kRopeGm = kRopeGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::InitMm2GlobalTensor(
    GlobalTensor<KV_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
    GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm)
{
    // mm2
    this->vec1ResGm = vec1ResGm;
    this->valueGm = valueGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::InitPageAttentionInfo(GlobalTensor<int32_t> blockTableGm,
                                                                  uint32_t blockSize, uint32_t maxBlockNumPerBatch)
{
    this->blockTableGm = blockTableGm;
    this->kvCacheBlockSize = blockSize;
    this->maxBlockNumPerBatch = maxBlockNumPerBatch;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(qpBufL1, L1_PQ_SIZE * 2);
    qpL1Tensor = qpBufL1.Get<KV_T>();

    pipe->InitBuffer(kvBufL1, L1_KV_SIZE * KV_BUFF_NUM);
    kvL1Tensor = kvBufL1.Get<KV_T>();

    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2); // 64K
    aL0TensorPingPong = tmpBufL0A.Get<KV_T>();
    // L0B
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2); // 64K
    bL0TensorPingPong = tmpBufL0B.Get<KV_T>();
    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2); // 128K
    cL0TensorPingPong = tmpBufL0C.Get<MMAD_OUT_T>();
    // BIAS
    pipe->InitBuffer(biasBufL1, BT_PP_SIZE * 2);
    biasL1Tensor = biasBufL1.Get<int32_t>();
    // BT 最大4k
    pipe->InitBuffer(tmpBufBias, BT_PP_SIZE * 2);
    biasBTTensorPingPong = tmpBufBias.Get<int32_t>();
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::UpdateKey(GlobalTensor<KV_T> keyGm)
{
    this->keyGm = keyGm;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::UpdateValue(GlobalTensor<KV_T> valueGm)
{
    this->valueGm = valueGm;
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(QP_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(QP_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT2);

    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);

    SetFlag<HardEvent::MTE1_MTE2>(BIAS_EVENT0);

    SetFlag<HardEvent::M_MTE1>(BT_EVENT0);
    SetFlag<HardEvent::M_MTE1>(BT_EVENT1);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT2);

    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);

    WaitFlag<HardEvent::MTE1_MTE2>(BIAS_EVENT0);

    WaitFlag<HardEvent::M_MTE1>(BT_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(BT_EVENT1);
}

template <typename IFAT>
template <typename DT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::CopyGmToL1(LocalTensor<DT>& l1Tensor, GlobalTensor<DT> &gmSrcTensor,
                                                     uint32_t srcN, uint32_t srcD, uint32_t srcDstride, uint32_t dstNAlign)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    if constexpr (KVINT4) {
        nd2nzPara.dValue = srcD / 2;
        nd2nzPara.srcDValue = srcDstride / 2;
    } else {
        nd2nzPara.dValue = srcD;
        nd2nzPara.srcDValue = srcDstride;
    }
    nd2nzPara.dstNzC0Stride = (srcN + dstNAlign - 1) / dstNAlign * dstNAlign; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor, const ExtraInfoMla &info, uint32_t mSize, uint64_t mOffset)
{
    auto srcGm = queryGm[info.tensorAOffset + mOffset];
    CopyGmToL1(l1Tensor, srcGm, mSize, headDim, headDim);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::CopyInMm1ARopeToL1(LocalTensor<ROPE_T> &l1Tensor,
                                                                 const ExtraInfoMla &info, uint32_t mSize, uint64_t mOffset)
{
    auto srcGm = qRopeGm[info.tensorARopeOffset + mOffset];
    CopyGmToL1(l1Tensor, srcGm, mSize, headDimRope, headDimRope);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::CopyInMm1BToL1(LocalTensor<KV_T>& bL1Tensor,
                                                                                const ExtraInfoMla &info, uint32_t subNid, uint32_t subNSize, uint32_t nSplitSize)
{
    uint64_t dStride = headDim;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        dStride = headDim * kvHeadNum;
    }

    auto srcGm = keyGm[info.tensorBOffset + subNid * nSplitSize * dStride];
    CopyGmToL1(bL1Tensor, srcGm, subNSize, headDim, dStride);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::CopyInMm1BRopeToL1(LocalTensor<ROPE_T>& bL1Tensor,
                                                                                    const ExtraInfoMla &info, uint32_t subNid, uint32_t subNSize, uint32_t nSplitSize)
{
    uint64_t dStride = headDimRope;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        dStride = headDimRope * kvHeadNum;
    }
    auto srcGm = kRopeGm[info.tensorBRopeOffset + nSplitSize * subNid * dStride];
    CopyGmToL1(bL1Tensor, srcGm, subNSize, headDimRope, dStride);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::CopyInMm1BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, const uint64_t keyGmBaseOffset, uint32_t copyTotalRowCntAlign,
    uint32_t copyStartRowCnt, uint32_t nActCopyRowCount)
{
    if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        DataCopyParams intriParams;
        intriParams.blockLen = nActCopyRowCount;
        intriParams.blockCount = headDim / blockElementCnt;
        intriParams.dstStride = copyTotalRowCntAlign - nActCopyRowCount;
        intriParams.srcStride = kvCacheBlockSize - nActCopyRowCount;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], keyGm[keyGmBaseOffset], intriParams);
    } else {
        uint64_t dStride = headDim;
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
            dStride = headDim * kvHeadNum;
        }

        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        Nd2NzParams mm1Nd2NzParamsForB;
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
        if constexpr (KVINT4) {
            mm1Nd2NzParamsForB.dValue = headDim / 2;
            mm1Nd2NzParamsForB.srcDValue = dStride / 2;
        } else {
            mm1Nd2NzParamsForB.dValue = headDim;
            mm1Nd2NzParamsForB.srcDValue = dStride;
        }
        mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
        mm1Nd2NzParamsForB.dstNzNStride = 1;
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], keyGm[keyGmBaseOffset], mm1Nd2NzParamsForB);
    }
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::CopyInMm1BRopeToL1ForPA(
    LocalTensor<ROPE_T>& bL1Tensor, const uint64_t kRopeGmBaseOffset, uint32_t copyTotalRowCntAlign,
    uint32_t copyStartRowCnt, uint32_t nActCopyRowCount)
{
    if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
        uint32_t blockElementCnt = 32 / sizeof(ROPE_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        DataCopyParams intriParams;
        intriParams.blockLen = nActCopyRowCount;
        intriParams.blockCount = headDimRope / blockElementCnt;
        intriParams.dstStride = copyTotalRowCntAlign - nActCopyRowCount;
        intriParams.srcStride = kvCacheBlockSize - nActCopyRowCount;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], kRopeGm[kRopeGmBaseOffset], intriParams);
    } else {
        uint64_t dStride = headDimRope;
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
            dStride = headDimRope * kvHeadNum;
        }

        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        Nd2NzParams mm1Nd2NzParamsForB;
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
        if constexpr (KVINT4) {
            mm1Nd2NzParamsForB.dValue = headDimRope / 2;
            mm1Nd2NzParamsForB.srcDValue = dStride / 2;
        } else {
            mm1Nd2NzParamsForB.dValue = headDimRope;
            mm1Nd2NzParamsForB.srcDValue = dStride;
        }
        mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
        mm1Nd2NzParamsForB.dstNzNStride = 1;
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], kRopeGm[kRopeGmBaseOffset], mm1Nd2NzParamsForB);
    }
}

template <typename IFAT>
template <typename DT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::LoadDataL0A(LocalTensor<DT>& aL0Tensor,
    const LocalTensor<DT>& aL1Tensor, uint32_t kIdx, uint32_t kSplitSize, uint32_t mSize, uint32_t subkSize)
{
    LocalTensor<DT> srcTensor = aL1Tensor[mSize * kSplitSize * kIdx];

    LoadData3DParamsV2<DT> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = mSize / 16; // Hin=M1=8
    loadData3DParams.l1W = 16; // Win=M0
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果
    // SetLoadToA0Params
    loadData3DParams.mExtension = mSize; // M
    loadData3DParams.kExtension = subkSize; // K
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
    loadData3DParams.channelSize = subkSize; // Cin=K
    LoadData(aL0Tensor, srcTensor, loadData3DParams);
}

template <typename IFAT>
template <typename DT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::LoadDataMm1B(LocalTensor<DT> &l0Tensor,
    LocalTensor<DT> &l1Tensor, uint32_t idx, uint32_t kSplitSize, uint32_t kSize, uint32_t nSize)
{
    // N 方向全载
    LocalTensor<DT> srcTensor = l1Tensor[nSize * kSplitSize * idx];

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    if constexpr (KVINT4) {
        loadData2DParams.repeatTimes = (nSize + 15) / 16 * kSize / 64;
    } else {
        loadData2DParams.repeatTimes = (nSize + 15) / 16 * kSize / (32 / sizeof(DT));
    }

    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0Tensor, srcTensor, loadData2DParams);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor,
                                                                                const ExtraInfoMla &info, uint32_t mSize, uint64_t mOffset)
{
    // 全量拷贝
    auto srcGm = vec1ResGm[(info.loop % PRE_LOAD_NUM_MLA) * mmResUbSize + mOffset];
    CopyGmToL1(aL1Tensor, srcGm, mSize, info.actualSingleProcessSInnerSize,
               info.actualSingleProcessSInnerSizeAlign);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::CopyInMm2BToL1(LocalTensor<KV_T> &bL1Tensor, const ExtraInfoMla &info,
    uint32_t subKid, uint32_t kSplitSize, uint32_t subNid, uint32_t nSplitSize, uint32_t subKSize, uint32_t subNSize)
{
    uint64_t dStride = headDim;
    if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
        dStride = headDim * kvHeadNum;
    }
    // 拷贝 128 * 256
    auto srcGm = keyGm[info.tensorBOffset + subKid * kSplitSize * dStride + subNid * nSplitSize];
    CopyGmToL1(bL1Tensor, srcGm, subKSize, subNSize, dStride, 32U);
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::CopyInMm2BToL1ForPA(LocalTensor<KV_T>& bL1Tensor,
    const uint64_t valueGmBaseOffset, uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount,
    uint32_t copyStartColumnCount, uint32_t copyColumnCount)
{
    if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
        // copyStartColumnCount和copyColumnCount都需要blockElementCnt对齐
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        DataCopyParams intriParams;
        intriParams.blockLen = nActCopyRowCount;
        intriParams.blockCount = copyColumnCount / blockElementCnt;
        intriParams.dstStride = copyTotalRowCntAlign - nActCopyRowCount;
        intriParams.srcStride = kvCacheBlockSize - nActCopyRowCount;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], valueGm[valueGmBaseOffset + copyStartColumnCount * kvCacheBlockSize], intriParams);
    } else {
        uint64_t step = headDim;
        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
            step = headDim * kvHeadNum;
        }

        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (KVINT4) {
            blockElementCnt = 64;
        }

        Nd2NzParams mm1Nd2NzParamsForB;
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
        if constexpr (KVINT4) {
            mm1Nd2NzParamsForB.dValue = copyColumnCount / 2;
            mm1Nd2NzParamsForB.srcDValue = step / 2;
        } else {
            mm1Nd2NzParamsForB.dValue = copyColumnCount;
            mm1Nd2NzParamsForB.srcDValue = step;
        }
        mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
        mm1Nd2NzParamsForB.dstNzNStride = 1;
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], valueGm[valueGmBaseOffset + copyStartColumnCount], mm1Nd2NzParamsForB);
    }
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::LoadDataMm2B(LocalTensor<KV_T> &bL0Tensor, LocalTensor<KV_T> &bL1Tensor,
                                                           uint32_t idx, uint32_t nSize, uint32_t subkSize,
                                                           uint32_t kSplitSize, uint32_t kSize)
{
    // L1 128 * 256; L0 64 * 128
    uint32_t kloops = subkSize / 32;

    LocalTensor<KV_T> srcTensor = bL1Tensor[idx * kSplitSize * 32 / sizeof(KV_T)];
    for (uint32_t i = 0; i < kloops; i++) {
        LoadData2dTransposeParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        if constexpr (KVINT4) {
            loadData2DParams.repeatTimes = nSize / 64;
        } else {
            loadData2DParams.repeatTimes = nSize / (32 / sizeof(KV_T));
        }

        loadData2DParams.srcStride = kSize / 32;
        loadData2DParams.dstGap = 1;
        loadData2DParams.dstFracGap = 0;
  
        LocalTensor<KV_T> tmpSrcTensor;
        if constexpr (KVINT4) {
            tmpSrcTensor = srcTensor[i * 16 * 64];
        } else {
            tmpSrcTensor = srcTensor[i * 32 * (32 / sizeof(KV_T))];
        }

        LoadDataWithTranspose(bL0Tensor[i * 32 * nSize], tmpSrcTensor, loadData2DParams);
    }
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::ComputeMm1(const ExtraInfoMla &info)
{
    uint32_t mSizeAct = info.gSize * info.s1Size;
    uint32_t mSize = Align(mSizeAct, 16U);
    uint32_t m1Loops = (mSizeAct + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;
    uint32_t m1Tail = mSizeAct - (m1Loops - 1) * M_SPLIT_SIZE;
    uint32_t subM1Size = M_SPLIT_SIZE;
    uint32_t subM1SizeAct = M_SPLIT_SIZE;

    for (uint32_t m1 = 0; m1 < m1Loops; m1++) {
        if (m1 == (m1Loops - 1)) {
            subM1SizeAct = m1Tail;
            subM1Size = Align(m1Tail, 16U);
        }
    
        const uint32_t nSplitSize = 128; // n方向切分
        uint32_t nloops = (info.actualSingleProcessSInnerSize + nSplitSize - 1) / nSplitSize;
        uint32_t nTail = info.actualSingleProcessSInnerSize - (nloops - 1) * nSplitSize;
        uint32_t subNSize = nSplitSize;
        uint32_t subNSizeAct = nSplitSize;

        for (uint32_t n = 0; n < nloops; n++) {
            if (n == nloops - 1) {
                subNSizeAct = nTail;
                subNSize = Align(nTail, 16U);
            }
            kvL1BufIter++;

            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % KV_BUFF_NUM));
            LocalTensor<KV_T> kTensor = kvL1Tensor[(kvL1BufIter % KV_BUFF_NUM) * (L1_KV_SIZE / sizeof(KV_T))];
            LocalTensor<KV_T> tmpkRopeTensor = kTensor[subNSize * BlockAlign<KV_T>(headDim)];
            LocalTensor<ROPE_T> kRopeTensor = tmpkRopeTensor.template ReinterpretCast<ROPE_T>();

            if constexpr (PAGE_ATTENTION) {
                uint64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
                uint32_t curSeqIdx = info.s2BatchOffset + n * nSplitSize;
                uint32_t copyFinishRowCnt = 0;
                while (copyFinishRowCnt < subNSizeAct) {
                    uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize; // 获取blcok table上的索引
                    uint64_t reaminRowCnt = curSeqIdx % kvCacheBlockSize;  // 获取在单个块上超出的行数
                    uint64_t idInBlockTable =
                        blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
                    // 计算可以拷贝行数
                    uint32_t copyRowCnt = kvCacheBlockSize - reaminRowCnt;
                    if (copyFinishRowCnt + copyRowCnt > subNSizeAct) {
                        copyRowCnt = subNSizeAct - copyFinishRowCnt;
                    }
                    uint64_t keyOffset = idInBlockTable * kvCacheBlockSize * headDim * kvHeadNum;
                    uint64_t kRopeOffset = idInBlockTable * kvCacheBlockSize * headDimRope * kvHeadNum;
                    if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
                        uint32_t blockElementCnt = 32 / sizeof(KV_T);
                        uint32_t ropeBlockElementCnt = 32 / sizeof(ROPE_T);

                        keyOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * blockElementCnt;
                        kRopeOffset += (uint64_t)(info.n2Idx * headDimRope * kvCacheBlockSize) + reaminRowCnt * ropeBlockElementCnt;
                    } else {
                        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
                            keyOffset += (uint64_t)(info.n2Idx * headDim) + reaminRowCnt * headDim * kvHeadNum;
                            kRopeOffset += (uint64_t)(info.n2Idx * headDimRope) + reaminRowCnt * headDimRope * kvHeadNum;
                        } else {
                            keyOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * headDim;
                            kRopeOffset += (uint64_t)(info.n2Idx * headDimRope * kvCacheBlockSize) + reaminRowCnt * headDimRope;
                        }
                    }
                    CopyInMm1BToL1ForPA(kTensor, keyOffset, subNSize, copyFinishRowCnt, copyRowCnt);
                    CopyInMm1BRopeToL1ForPA(kRopeTensor, kRopeOffset, subNSize, copyFinishRowCnt, copyRowCnt);

                    // 更新循环变量
                    copyFinishRowCnt += copyRowCnt;
                    curSeqIdx += copyRowCnt;
                }
            } else {
                CopyInMm1BToL1(kTensor, info, n, subNSizeAct, nSplitSize); // 拷贝 128 * 512
                CopyInMm1BRopeToL1(kRopeTensor, info, n, subNSizeAct, nSplitSize); // 拷贝 128 * 64
            }

            SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % KV_BUFF_NUM));
            WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % KV_BUFF_NUM));

            const uint32_t mSplitSizeL0 = 128U;
            uint32_t mLoops = (subM1SizeAct + mSplitSizeL0 - 1) / mSplitSizeL0;
            uint32_t mTail = subM1SizeAct - (mLoops - 1) * mSplitSizeL0;
            
            uint32_t subMSize = mSplitSizeL0;
            uint32_t subMSizeAct = mSplitSizeL0;

            uint32_t mx = mLoops - 1;
            for (uint32_t m = 0; m < mLoops; m++) {
                if (m == mLoops - 1) {
                    subMSizeAct = mTail;
                    subMSize = Align(subMSizeAct, 16U);
                }

                uint32_t ka = 0;
                if (n == 0) {
                    qpL1BufIter++;
                    ka = qpL1BufIter;
                    WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + (ka % 2));
                    uint64_t mOffset = m1 * M_SPLIT_SIZE + m * mSplitSizeL0;
                    LocalTensor<KV_T> qTensor = qpL1Tensor[(ka % 2) * (L1_PQ_SIZE / sizeof(KV_T))];
                    LocalTensor<KV_T> tmpQRopeTensor = qTensor[subMSize * BlockAlign<KV_T>(headDim)];
                    LocalTensor<ROPE_T> qRopeTensor = tmpQRopeTensor.template ReinterpretCast<ROPE_T>();
                    CopyInMm1AToL1(qTensor, info, subMSizeAct, mOffset * headDim);
                    CopyInMm1ARopeToL1(qRopeTensor, info, subMSizeAct, mOffset * headDimRope);
                    SetFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + (ka % 2));
                    WaitFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + (ka % 2));
                } else {
                    ka = qpL1BufIter - mx;
                    mx--;
                }

                LocalTensor<KV_T> qTensor = qpL1Tensor[(ka % 2) * (L1_PQ_SIZE / sizeof(KV_T))];
                LocalTensor<KV_T> tmpQRopeTensor = qTensor[subMSize * BlockAlign<KV_T>(headDim)];
                LocalTensor<ROPE_T> qRopeTensor = tmpQRopeTensor.template ReinterpretCast<ROPE_T>();

#if defined(MM1_RES_ADD) && defined(MM1_DATO)
                // nope 计算
                uint32_t kSplitSize = 256;
                uint32_t kSize = BlockAlign<KV_T>(headDim);
                uint32_t kLoops = (kSize + kSplitSize - 1) / kSplitSize;
                uint32_t subKSize = kSplitSize;

                LocalTensor<int32_t> btC2Tensor = biasBTTensorPingPong[(btBufIter % 2) * (BT_PP_SIZE / sizeof(int32_t))];
                if (isFirstIter) {
                    isFirstIter = false;
                    int32_t biasValue = 1260388352; // 2^21 + 150 * 2^23
                    InitConstValueParams<int32_t> params(1, 16, 0, biasValue); // nSplitSize * 4 / 32B = 16
                    WaitFlag<HardEvent::MTE1_MTE2>(BIAS_EVENT0 + biasL1BufIter % 2);
                    LocalTensor<int32_t> biasTensor = biasL1Tensor[(biasL1BufIter % 2) * (BT_PP_SIZE / sizeof(int32_t))];
                    InitConstValue(biasTensor, params);
                    SetFlag<HardEvent::MTE2_MTE1>(BIAS_EVENT0 + biasL1BufIter % 2);
                    WaitFlag<HardEvent::MTE2_MTE1>(BIAS_EVENT0 + biasL1BufIter % 2);
                    
                    DataCopy(btC2Tensor, biasTensor, nSplitSize);
                    SetFlag<HardEvent::MTE1_MTE2>(BIAS_EVENT0 + biasL1BufIter % 2);
                }

                LocalTensor<MMAD_OUT_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MMAD_OUT_T))];
                LocalTensor<T> cL0TensorFp32 = cL0Tensor.template ReinterpretCast<T>();

                for (uint32_t k = 0; k < kLoops; k++) {
                    if (k + 1 == kLoops) {
                        subKSize = kSize - (kLoops - 1) * kSplitSize;
                    }

                    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];

                    LoadDataL0A(aL0Tensor, qTensor, k, kSplitSize, subMSize, subKSize);

                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];

                    LoadDataMm1B(bL0Tensor, kTensor, k, kSplitSize, subKSize, subNSize);

                    SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);
                    WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);

                    MmadParams mmadParams;
                    mmadParams.m = subMSize;
                    mmadParams.n = subNSize;
                    mmadParams.k = subKSize;
                    mmadParams.cmatrixInitVal = false;
                    mmadParams.cmatrixSource = (k == 0);
                    mmadParams.unitFlag = 0b10;

                    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    if (k == 0) {
                        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, btC2Tensor, mmadParams);
                    } else {
                        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                    }

                    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                    aL0BufIter++;

                    bL0BufIter++;
                }

                // ROPE 计算
                {
                    float aVal = -2097152.0; // -2^21
                    float bVal = 5.0; // 5
                    InitConstValueParams<float> paramsA(1, 32, 0, aVal); // n*k*4/512B=128*32*4/512
                    InitConstValueParams<float> paramsB(1, 32, 0, bVal); // n*k*4/512B=128*32*4/512

                    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LocalTensor<float> aL0TensorFp32 = aL0Tensor.template ReinterpretCast<float>();
                    InitConstValue(aL0TensorFp32, paramsA);

                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LocalTensor<float> bL0TensorFp32 = bL0Tensor.template ReinterpretCast<float>();
                    InitConstValue(bL0TensorFp32, paramsB);

                    LocalTensor<KV_T> tmpaL0Tensor = aL0Tensor[128 * 32 * 4];
                    LocalTensor<ROPE_T> aL0TensorRope = tmpaL0Tensor.template ReinterpretCast<ROPE_T>();
                    LoadDataL0A(aL0TensorRope, qRopeTensor, 0, 0, subMSize, headDimRope);

                    LocalTensor<KV_T> tmpbL0Tensor = bL0Tensor[128 * 32 * 4];
                    LocalTensor<ROPE_T> bL0TensorRope = tmpbL0Tensor.template ReinterpretCast<ROPE_T>();
                    LoadDataMm1B(bL0TensorRope, kRopeTensor, 0, 0, headDimRope, subNSize);

                    SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);
                    WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);

                    {
                        MmadParams mmadParams;
                        mmadParams.m = subMSize;
                        mmadParams.n = subNSize;
                        mmadParams.k = 1;
                        mmadParams.cmatrixInitVal = false;
                        mmadParams.cmatrixSource = false;
                        mmadParams.unitFlag = 0b10;
                        if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                            PipeBarrier<PIPE_M>();
                        }
                        Mmad(cL0TensorFp32, aL0TensorFp32, bL0TensorFp32, mmadParams);
                    }

                    {
                        MmadParams mmadParams;
                        mmadParams.m = subMSize;
                        mmadParams.n = subNSize;
                        mmadParams.k = headDimRope;
                        mmadParams.cmatrixInitVal = false; // true
                        mmadParams.cmatrixSource = false; // Co1
                        mmadParams.unitFlag = 0b11;

                        if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                            PipeBarrier<PIPE_M>();
                        }
                        Mmad(cL0TensorFp32, aL0TensorRope, bL0TensorRope, mmadParams);
                    }

                    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                    aL0BufIter++;

                    bL0BufIter++;
                }

                FixpipeParamsV220 fixParams;
                fixParams.nSize = subNSize;
                fixParams.mSize = subMSizeAct;
                fixParams.srcStride = subMSize;
                fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                fixParams.ndNum = 1;
                fixParams.unitFlag = 0b11;

                size_t baseOffset = (info.loop % (PRE_LOAD_NUM_MLA)) * mmResUbSize;

                Fixpipe(mm1ResGm[baseOffset + (m1 * M_SPLIT_SIZE + m * mSplitSizeL0) * info.actualSingleProcessSInnerSizeAlign + n * nSplitSize], cL0Tensor,
                        fixParams);

                cL0BufIter++;
#else
                // ROPE 计算
                {
                    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
                    LocalTensor<MMAD_OUT_T> tmpcL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MMAD_OUT_T))];
                    LocalTensor<T> cL0Tensor = tmpcL0Tensor.template ReinterpretCast<T>();

                    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                    LocalTensor<KV_T> tmpaL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];
                    LocalTensor<ROPE_T> aL0Tensor = tmpaL0Tensor.template ReinterpretCast<ROPE_T>();
                    LoadDataL0A(aL0Tensor, qRopeTensor[(m * mSplitSizeL0 * 32) / sizeof(ROPE_T)], 0, 0, mSize, headDimRope);

                    LocalTensor<KV_T> tmpbL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
                    LocalTensor<ROPE_T> bL0Tensor = tmpbL0Tensor.template ReinterpretCast<ROPE_T>();
                    LoadDataMm1B(bL0Tensor, kRopeTensor, 0, 0, headDimRope, subNSize);

                    SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);
                    WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);

                    MmadParams mmadParams;
                    mmadParams.m = subMSize;
                    mmadParams.n = subNSize;
                    mmadParams.k = headDimRope; // 64
                    mmadParams.cmatrixInitVal = true;
                    mmadParams.cmatrixSource = false;

                    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);

                    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                    aL0BufIter++;

                    bL0BufIter++;

                    SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
                    WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);

                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = subNSize;
                    fixParams.mSize = subMSizeAct;
                    fixParams.srcStride = subMSize;
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                    fixParams.ndNum = 1;

                    // Rope 保存在下半块内存
                    size_t offset = mSizeAct * info.actualSingleProcessSInnerSizeAlign;
                    auto tmpDstGm = mm1ResGm[(info.loop % (PRE_LOAD_NUM_MLA)) * mmResUbSize + offset];
                    auto ropeGmPtr = tmpDstGm.GetPhyAddr();
                    GlobalTensor<T> dstGm;
                    dstGm.SetGlobalBuffer((__gm__ T *)ropeGmPtr);
                    Fixpipe(dstGm[m * mSplitSizeL0 * info.actualSingleProcessSInnerSizeAlign + n * nSplitSize], cL0Tensor, fixParams);

                    SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
                    cL0BufIter++;
                }

                // nope 计算
                {
                    uint32_t kSplitSize = 256;
                    uint32_t kSize = BlockAlign<KV_T>(headDim);
                    uint32_t kLoops = (kSize + kSplitSize - 1) / kSplitSize;
                    uint32_t subKSize = kSplitSize;

                    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
                    LocalTensor<MMAD_OUT_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MMAD_OUT_T))];
                    for (uint32_t k = 0; k < kLoops; k++) {
                        if (k + 1 == kLoops) {
                            subKSize = kSize - (kLoops - 1) * kSplitSize;
                        }

                        WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                        LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];

                        LoadDataL0A(aL0Tensor, qTensor[(m * mSplitSizeL0 * 32) / sizeof(KV_T)], k, kSplitSize, mSize, subKSize);

                        LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];

                        LoadDataMm1B(bL0Tensor, kTensor, k, kSplitSize, subKSize, subNSize);

                        SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);
                        WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);

                        MmadParams mmadParams;
                        mmadParams.m = subMSize;
                        mmadParams.n = subNSize;
                        mmadParams.k = subKSize;
                        mmadParams.cmatrixInitVal = (k == 0);
                        mmadParams.cmatrixSource = false;

                        if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                            PipeBarrier<PIPE_M>();
                        }
                        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);

                        SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                        aL0BufIter++;

                        bL0BufIter++;
                    }

                    SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);
                    WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + cL0BufIter % 2);

                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = subNSize;
                    fixParams.mSize = subMSizeAct;
                    fixParams.srcStride = subMSize;
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                    fixParams.ndNum = 1;

                    size_t baseOffset = (info.loop % (PRE_LOAD_NUM_MLA)) * mmResUbSize;

                    Fixpipe(mm1ResGm[baseOffset + m * mSplitSizeL0 * info.actualSingleProcessSInnerSizeAlign + n * nSplitSize], cL0Tensor,
                            fixParams);

                    SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
                    cL0BufIter++;
                }
#endif
                if (n == nloops - 1) {
                    SetFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + (ka % 2));
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % KV_BUFF_NUM));
        }
    }
}

template <typename IFAT>
__aicore__ inline void IfaMatmulFullQuant<IFAT>::ComputeMm2(const ExtraInfoMla &info)
{
    uint32_t mSizeAct = info.gSize * info.s1Size;
    
    uint32_t mSize = Align(mSizeAct, 16U);
    uint32_t nSize = BlockAlign<KV_T>(headDim);
    uint32_t m1Loops = (mSizeAct + M_SPLIT_SIZE - 1) / M_SPLIT_SIZE;
    uint32_t m1Tail = mSizeAct - (m1Loops - 1) * M_SPLIT_SIZE;
    uint32_t subM1Size = M_SPLIT_SIZE;
    uint32_t subM1SizeAct = M_SPLIT_SIZE;

    for (uint32_t m1 = 0; m1 < m1Loops; m1++) {
         if (m1 == (m1Loops - 1)) {
            subM1SizeAct = m1Tail;
            subM1Size = Align(m1Tail, 16U);
        }      

        uint32_t nSplitSize = 128;
        uint32_t nLoops = (nSize + nSplitSize - 1) / nSplitSize;
        uint32_t nTail = nSize - (nLoops - 1) * nSplitSize;
        uint32_t subNSize = nSplitSize;

        for (uint32_t n = 0; n < nLoops; n++) {
            if (n == nLoops - 1) {
                subNSize = nTail;
            }

            // copyB：512 * 128
            kvL1BufIter++;
            WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % KV_BUFF_NUM));
            LocalTensor vTensor = kvL1Tensor[(kvL1BufIter % KV_BUFF_NUM) * (L1_KV_SIZE / sizeof(KV_T))];

            uint32_t kSizeAct = info.actualSingleProcessSInnerSize;
            uint32_t kSize = Align(kSizeAct, 32U);

            if constexpr (PAGE_ATTENTION) {
                uint64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
                uint32_t curSeqIdx = info.s2BatchOffset;
                uint32_t copyFinishRowCnt = 0;
                while (copyFinishRowCnt < kSizeAct) {
                    uint64_t blockIdOffset = curSeqIdx / kvCacheBlockSize; // 获取blcok table上的索引
                    uint64_t reaminRowCnt = curSeqIdx % kvCacheBlockSize;  // 获取在单个块上超出的行数
                    uint64_t idInBlockTable =
                        blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
                    // 计算可以拷贝行数
                    uint32_t copyRowCnt = kvCacheBlockSize - reaminRowCnt;
                    if (copyFinishRowCnt + copyRowCnt > kSizeAct) {
                        copyRowCnt = kSizeAct - copyFinishRowCnt;
                    }
                    uint64_t valueOffset = idInBlockTable * kvCacheBlockSize * headDim * kvHeadNum;
                    if constexpr (KV_LAYOUT_T == LAYOUT::NZ) {
                        uint32_t blockElementCnt = 32 / sizeof(KV_T);
                        valueOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * blockElementCnt;
                    } else {
                        if constexpr (LAYOUT_T == LAYOUT::BSH || LAYOUT_T == LAYOUT::BSND) {
                            valueOffset += (uint64_t)(info.n2Idx * headDim) + reaminRowCnt * headDim * kvHeadNum;
                        } else {
                            valueOffset += (uint64_t)(info.n2Idx * headDim * kvCacheBlockSize) + reaminRowCnt * headDim;
                        }
                    }
                    CopyInMm2BToL1ForPA(vTensor, valueOffset, kSize, copyFinishRowCnt, copyRowCnt, n * nSplitSize, subNSize);

                    // 更新循环变量
                    copyFinishRowCnt += copyRowCnt;
                    curSeqIdx += copyRowCnt;
                }
            } else {
                CopyInMm2BToL1(vTensor, info, 0, 0, n, nSplitSize, kSizeAct, subNSize); // 128 * 512
            }

            SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % KV_BUFF_NUM));
            WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + (kvL1BufIter % KV_BUFF_NUM));

            const uint32_t mSplitSizeL0 = 128;
            uint32_t mLoops = (subM1SizeAct + mSplitSizeL0 - 1) / mSplitSizeL0;
            uint32_t mTail = subM1SizeAct - (mLoops - 1) * mSplitSizeL0;
            uint32_t subMSize = mSplitSizeL0;
            uint32_t subMSizeAct = mSplitSizeL0;

            uint32_t mx = mLoops - 1;
            for (uint32_t m = 0; m < mLoops; m++) {
                if (m == mLoops - 1) {
                    subMSizeAct = mTail;
                    subMSize = Align(subMSizeAct, 16U);
                }
                
                uint32_t ka = 0;
                if (n == 0) {
                    qpL1BufIter++;
                    ka = qpL1BufIter;
                    LocalTensor<KV_T> pTensor = qpL1Tensor[(ka % 2) * (L1_PQ_SIZE / sizeof(KV_T))];

                    WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + (ka % 2));
                    uint64_t mOffset = m1 * M_SPLIT_SIZE + m * mSplitSizeL0;
                    CopyInMm2AToL1(pTensor, info, subMSizeAct, mOffset * info.actualSingleProcessSInnerSize);
                    SetFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + (ka % 2));
                    WaitFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + (ka % 2));
                } else {
                    ka = qpL1BufIter - mx;
                    mx--;
                }

                LocalTensor<KV_T> pTensor = qpL1Tensor[(ka % 2) * (L1_PQ_SIZE / sizeof(KV_T))];

                // 切K
                uint32_t kSplitSize = 256;
                uint32_t kloops = (info.actualSingleProcessSInnerSize + kSplitSize - 1) / kSplitSize;
                uint32_t kTail = info.actualSingleProcessSInnerSize - (kloops - 1) * kSplitSize;
                uint32_t subKSize = kSplitSize;
                uint32_t subKSizeAct = kSplitSize;

                LocalTensor<MMAD_OUT_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(MMAD_OUT_T))];

                for (uint32_t k = 0; k < kloops; k++) {
                    if (k == kloops - 1) {
                        subKSizeAct = kTail;
                        subKSize = Align(kTail, 32U);
                    }

                    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                    LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(KV_T))];

                    LoadDataL0A(aL0Tensor, pTensor, k, kSplitSize, subMSize, subKSize); // 64 * 128

                    LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * (L0B_PP_SIZE / sizeof(KV_T))];

                    LoadDataMm2B(bL0Tensor, vTensor, k, subNSize, subKSize, kSplitSize, kSize);

                    SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);
                    WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + aL0BufIter % 2);

                    MmadParams mmadParams;
                    mmadParams.m = subMSize;
                    mmadParams.n = subNSize;
                    mmadParams.k = subKSizeAct;
                    mmadParams.cmatrixInitVal = (k == 0);
                    mmadParams.cmatrixSource = false;
                    mmadParams.unitFlag = (k == kloops - 1) ? 0b11: 0b10;

                    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                        PipeBarrier<PIPE_M>();
                    }
                    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);

                    bL0BufIter++;

                    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + aL0BufIter % 2);
                    aL0BufIter++;
                }

                FixpipeParamsV220 fixParams;
                fixParams.nSize = subNSize; // 实现切片大小
                fixParams.mSize = subMSizeAct; // msdIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
                fixParams.srcStride = subMSize;   // ((fixParams.mSize + 15) / 16) * 16
                fixParams.dstStride = headDim; // headdimAlign mm2ResGm两行之间的间隔
                fixParams.ndNum = 1;
                fixParams.unitFlag = 0b11;
#ifdef QUANT_MM2_FP16
                fixParams.quantPre = QuantMode_t::DEQF16;
                fixParams.deqScalar = 0x3A800000; //117 << (10 + 13)=0x3A800000 float 1/1024  0x3F800000=1.0
#endif

                size_t baseOffset =(info.loop % (PRE_LOAD_NUM_MLA)) * bmm2ResUbSize;
                Fixpipe(mm2ResGm[ baseOffset + (m1 * M_SPLIT_SIZE + m * mSplitSizeL0) * headDim + nSplitSize * n], cL0Tensor, fixParams);

                cL0BufIter++;

                if (n== nLoops - 1) {
                    SetFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + (ka % 2));
                }
            }
            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + (kvL1BufIter % KV_BUFF_NUM));
        }
    }
}

#endif