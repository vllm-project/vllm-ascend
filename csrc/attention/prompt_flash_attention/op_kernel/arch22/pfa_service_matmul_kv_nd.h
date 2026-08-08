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
 * \file pfa_service_matmul_kv_nd.h
 * \brief
 */

#ifndef PFA_SERVICE_MATMUL_KV_ND_H
#define PFA_SERVICE_MATMUL_KV_ND_H

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
#include "prompt_flash_attention_mla_common.h"

template <typename INPUT_T, typename T, bool pageAttention = false>
class PfaMatmulKvNd {
public:
    __aicore__ inline PfaMatmulKvNd() {};
    __aicore__ inline void InitParams(uint32_t dSize, uint32_t valueDSize, uint32_t ropeDSize, uint32_t valueNSize, int64_t mm1Ka, int64_t mm1Kb, int64_t mm1RopeKa, int64_t mm1RopeKb, int64_t mm2Kb);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<INPUT_T> queryGm, GlobalTensor<INPUT_T> keyGm,
                                               GlobalTensor<INPUT_T> qRopeGm, GlobalTensor<INPUT_T> kRopeGm,
                                               GlobalTensor<T> mm1ResGm[2]);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<INPUT_T> vec1ResGm[2], GlobalTensor<INPUT_T> valueGm,
                                               GlobalTensor<T> mm2ResGm[2], GlobalTensor<INPUT_T> attentionOutGm);
    __aicore__ inline void InitPageAttentionInfo(GlobalTensor<int32_t> blockTableGm, uint32_t blockSize, uint32_t maxBlockNumPerBatch);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void UpdateKey(GlobalTensor<INPUT_T> keyGm);
    __aicore__ inline void UpdateValue(GlobalTensor<INPUT_T> valueGm);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    template <CubeFormat OutFormat=CubeFormat::NZ>
    __aicore__ inline void ComputeMm1(const SplitSameABExtraInfo &info);
    template <CubeFormat AFormat, CubeFormat OutFormat=CubeFormat::NZ>
    __aicore__ inline void ComputeMm2(const SplitSameABExtraInfo &info);

    template <uint32_t M_L1_SIZE>
    __aicore__ inline void ResetLoad3DConfig();

protected:
    template <typename T1> static __aicore__ inline T1 Align(T1 num, T1 rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
    }

    template <typename T1> static __aicore__ inline size_t BlockAlign(size_t s)
    {
        if constexpr (IsSameType<T1, int4b_t>::value) {
            return (s + 63) / 64 * 64;
        }
        size_t n = (32 / sizeof(T1));
        return (s + n - 1) / n * n;
    }

    __aicore__ inline void CopyNDGmToL1(LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmSrcTensor, 
                                        uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyNDGmToZZL1(LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmSrcTensor, 
                                        uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyNZGmToL1(LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmSrcTensor, 
                                        uint32_t srcN, uint32_t srcD, uint32_t srcNstride);
    __aicore__ inline void LoadDataAToL0(LocalTensor<INPUT_T> &aL0Tensor, const LocalTensor<INPUT_T> &aL1Tensor, 
                                        uint32_t mL1Size, uint32_t subMStart, uint32_t subMSize,
                                        uint32_t subKStart, uint32_t subKSize);
    __aicore__ inline void LoadDataZZAToL0(LocalTensor<INPUT_T> &aL0Tensor, const LocalTensor<INPUT_T> &aL1Tensor, 
                                        uint32_t kL1Size, uint32_t subMStart, uint32_t subMSize,
                                        uint32_t subKStart, uint32_t subKSize);
    __aicore__ inline void Load3DDataAToL0(LocalTensor<INPUT_T> &aL0Tensor, const LocalTensor<INPUT_T> &aL1Tensor, 
                                        uint32_t mL1Size, uint32_t subMStart, uint32_t subMSize,
                                        uint32_t subKStart, uint32_t subKSize);
    __aicore__ inline void Load3DDataAToL0NonFMatrix(LocalTensor<INPUT_T> &aL0Tensor, const LocalTensor<INPUT_T> &aL1Tensor, 
                                        uint32_t mL1Size, uint32_t subMStart, uint32_t subMSize,
                                        uint32_t subKStart, uint32_t subKSize);
    __aicore__ inline void LoadDataBTransposeToL0(LocalTensor<INPUT_T> &bL0Tensor, const LocalTensor<INPUT_T> &bL1Tensor, 
                                        uint32_t nL1Size, uint32_t subNStart, uint32_t subNSize,
                                        uint32_t subKStart, uint32_t subKSize);
    __aicore__ inline void LoadDataBToL0(LocalTensor<INPUT_T> &bL0Tensor, const LocalTensor<INPUT_T> &bL1Tensor, 
                                        uint32_t kL1Size, uint32_t subKStart, uint32_t subKSize,
                                        uint32_t subNStart, uint32_t subNSize);
    __aicore__ inline void LoadDataZZBToL0(LocalTensor<INPUT_T> &bL0Tensor, const LocalTensor<INPUT_T> &bL1Tensor, 
                                        uint32_t kL1Size, uint32_t subKStart, uint32_t subKSize,
                                        uint32_t subNStart, uint32_t subNSize);
    template <CubeFormat GMFormat>
    __aicore__ inline void FixpipeL0CToGM(GlobalTensor<T> &cGmTensor, const LocalTensor<T> &cL0Tensor, 
                                        uint32_t dstStride, uint32_t subMStart, uint32_t subMSize,
                                        uint32_t subNStart, uint32_t subNSize);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<INPUT_T> &aL1Tensor, const SplitSameABExtraInfo &info);
    __aicore__ inline void CopyInMm1RopeAToL1(LocalTensor<INPUT_T> &aRopeL1Tensor, const SplitSameABExtraInfo &info);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<INPUT_T> &bL1Tensor, const SplitSameABExtraInfo &info, 
                                        uint32_t subNStart, uint32_t subNSize);
    __aicore__ inline void CopyInMm1BToL1ForPA(LocalTensor<INPUT_T> &bL1Tensor, const uint64_t keyGmBaseOffset,
                                            uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                            uint32_t nActCopyRowCount, PaCacheLayoutType paCacheLayoutType);
    __aicore__ inline void CopyInMm1RopeBToL1(LocalTensor<INPUT_T> &bRopeL1Tensor, const SplitSameABExtraInfo &info, 
                                        uint32_t subNStart, uint32_t subNSize);
    __aicore__ inline void CopyInMm1RopeBToL1ForPA(LocalTensor<INPUT_T> &bRopeL1Tensor, const uint64_t kRopeGmBaseOffset,
                                            uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                            uint32_t nActCopyRowCount, PaCacheLayoutType paCacheLayoutType);
    template <CubeFormat AFormat>
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<INPUT_T> &aL1Tensor, const SplitSameABExtraInfo &info, 
                                        uint32_t gmStride, uint32_t subMStart, uint32_t subMSize,
                                        uint32_t subKStart, uint32_t subKSize);
    template <uint32_t KL0_SPLITE_SIZE>
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<INPUT_T> &bL1Tensor, const SplitSameABExtraInfo &info, 
                                        uint32_t subKStart, uint32_t subKSize);
    template <uint32_t KL0_SPLITE_SIZE>
    __aicore__ inline void CopyInMm2BToL1ForPA(LocalTensor<INPUT_T> &bL1Tensor, const uint64_t valueGmBaseOffset,
                                            uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount, uint32_t copyStartColumnCount,
                                            uint32_t copyColumnCount, PaCacheLayoutType paCacheLayoutType);
    
    __aicore__ inline LocalTensor<INPUT_T> CopyQToL1(const SplitSameABExtraInfo &info, uint32_t mL1Size);
    template <uint32_t KL0_SPLITE_SIZE>
    __aicore__ inline LocalTensor<INPUT_T> CopyVToL1(const SplitSameABExtraInfo &info, uint32_t subKStart, uint32_t subKSize);

protected:
    // mm1
    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<INPUT_T> qRopeGm;
    GlobalTensor<INPUT_T> kRopeGm;
    GlobalTensor<T> mm1ResGm[2];

    // mm2
    GlobalTensor<INPUT_T> vec1ResGm[2];
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<T> mm2ResGm[2];
    GlobalTensor<INPUT_T> attentionOutGm;

    // pageAttention
    GlobalTensor<int32_t> blockTableGm;
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;

    // params
    uint32_t dSize = 0;
    uint32_t ropeDSize = 0;
    uint32_t valueDSize = 0;
    uint32_t valueNSize = 0;
    int64_t mm1Ka = 0;
    int64_t mm1Kb = 0;
    int64_t mm1RopeKa = 0;
    int64_t mm1RopeKb = 0;
    int64_t mm2Kb = 0;

private:
    // L1
    static constexpr uint32_t L1_1_SIZE = 192 * 1024; // 192KB
    static constexpr uint32_t L1_2_SIZE = 64 * 1024; // 64KB
    static constexpr uint32_t L1_1_SIZE_WOKSPLIT = 128 * 1024; // D等长=128，不切K轴场景
    static constexpr uint32_t L1_2_SIZE_WOKSPLIT = 128 * 1024; // D等长=128，不切K轴场景

    // L0
    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);

    // mte2 <> mte1
    static constexpr uint32_t Q_EVENT0 = EVENT_ID2;
    static constexpr uint32_t Q_EVENT1 = EVENT_ID3;
    static constexpr uint32_t KV_EVENT0 = EVENT_ID4;
    static constexpr uint32_t KV_EVENT1 = EVENT_ID5;
    static constexpr uint32_t TMP_EVENT0 = EVENT_ID6;
    static constexpr uint32_t TMP_EVENT1 = EVENT_ID7;

    // m <> mte1
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;

    // fix <> m
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {false, false};

    TBuf<TPosition::A1> L1_1;
    TBuf<TPosition::A1> L1_2;

    LocalTensor<INPUT_T> qL1Tensor[2];   // q和v复用L1_1
    LocalTensor<INPUT_T> kvL1Tensor[2];   // k和p复用L1_2

    TBuf<TPosition::A2> tmpBufL0A;
    LocalTensor<INPUT_T> aL0TensorPingPong[2];
    
    // L0_B
    TBuf<TPosition::B2> tmpBufL0B;
    LocalTensor<INPUT_T> bL0TensorPingPong[2];

    // L0_C
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<T> cL0TensorPingPong[2];

    Nd2NzParams copyInMm1AToL1Params;
    Nd2NzParams copyInMm1BToL1Params;
    Nd2NzParams copyInMm2BToL1Params;
    LoadData3DParamsV2<INPUT_T> loadData3DParams;
    LoadData2DParams mm1LoadDataBTransposeToL0Params;
    LoadData2DParams mm2LoadDataBToL0Params;

    uint32_t qL1BufIter = 0;
    uint32_t kvL1BufIter = 0;
    uint32_t abL0BufIter = 0;
    uint32_t cL0BufIter = 0;
    uint64_t qL1BufAddr[2] = {(uint64_t)-1, (uint64_t)-1};
};

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::InitParams(uint32_t dSize, uint32_t valueDSize, uint32_t ropeDSize, uint32_t valueNSize,
                                        int64_t mm1Ka, int64_t mm1Kb, int64_t mm1RopeKa, int64_t mm1RopeKb, int64_t mm2Kb)
{
    this->dSize = dSize;
    this->valueDSize = valueDSize;
    this->ropeDSize = ropeDSize;
    this->valueNSize = valueNSize;

    this->mm1Ka = mm1Ka;
    this->mm1Kb = mm1Kb;
    this->mm1RopeKa = mm1RopeKa;
    this->mm1RopeKb = mm1RopeKb;
    this->mm2Kb = mm2Kb;

    copyInMm1AToL1Params.ndNum = 1;
    copyInMm1AToL1Params.dValue = dSize;
    copyInMm1AToL1Params.srcDValue = mm1Ka;
    copyInMm1AToL1Params.dstNzNStride = 1;
    copyInMm1AToL1Params.srcNdMatrixStride = 0;
    copyInMm1AToL1Params.dstNzMatrixStride = 0;

    copyInMm1BToL1Params.ndNum = 1;
    copyInMm1BToL1Params.dValue = dSize;
    copyInMm1BToL1Params.srcDValue = mm1Kb;
    copyInMm1BToL1Params.dstNzNStride = 1;
    copyInMm1BToL1Params.srcNdMatrixStride = 0;
    copyInMm1BToL1Params.dstNzMatrixStride = 0;

    copyInMm2BToL1Params.ndNum = 1;
    copyInMm2BToL1Params.dValue = valueDSize;
    copyInMm2BToL1Params.srcDValue = mm2Kb;
    copyInMm2BToL1Params.dstNzNStride = 1;
    copyInMm2BToL1Params.srcNdMatrixStride = 0;
    copyInMm2BToL1Params.dstNzMatrixStride = 0;

    loadData3DParams.l1W = GetC0Num<INPUT_T>(); // Win=M0
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = 0;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = 0;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;

    mm1LoadDataBTransposeToL0Params.startIndex = 0;
    mm1LoadDataBTransposeToL0Params.srcStride = 1;
    mm1LoadDataBTransposeToL0Params.dstGap = 0;
    mm1LoadDataBTransposeToL0Params.ifTranspose = false;

    mm2LoadDataBToL0Params.startIndex = 0;
    mm2LoadDataBToL0Params.repeatTimes = valueDSize / GetC0Num<INPUT_T>(); // 列（N轴）方向上的分形数
    mm2LoadDataBToL0Params.dstGap = 0;
    mm2LoadDataBToL0Params.ifTranspose = true; // 将小Z格式转置为小N
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::InitMm1GlobalTensor(GlobalTensor<INPUT_T> queryGm, GlobalTensor<INPUT_T> keyGm,
                                                                      GlobalTensor<INPUT_T> qRopeGm, GlobalTensor<INPUT_T> kRopeGm,
                                                                      GlobalTensor<T> mm1ResGm[2])
{
    // mm1
    this->queryGm = queryGm;
    this->keyGm = keyGm;
    this->qRopeGm = qRopeGm;
    this->kRopeGm = kRopeGm;
    this->mm1ResGm[0] = mm1ResGm[0];
    this->mm1ResGm[1] = mm1ResGm[1];
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::InitMm2GlobalTensor(GlobalTensor<INPUT_T> vec1ResGm[2], GlobalTensor<INPUT_T> valueGm,
                                            GlobalTensor<T> mm2ResGm[2], GlobalTensor<INPUT_T> attentionOutGm)
{
    // mm2
    this->vec1ResGm[0] = vec1ResGm[0];
    this->vec1ResGm[1] = vec1ResGm[1];
    this->valueGm = valueGm;
    this->mm2ResGm[0] = mm2ResGm[0];
    this->mm2ResGm[1] = mm2ResGm[1];
    this->attentionOutGm = attentionOutGm;
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::InitPageAttentionInfo(GlobalTensor<int32_t> blockTableGm, uint32_t blockSize, uint32_t maxBlockNumPerBatch)
{
    this->blockTableGm = blockTableGm;
    this->kvCacheBlockSize = blockSize;
    this->maxBlockNumPerBatch = maxBlockNumPerBatch;
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(L1_1, L1_1_SIZE * 2);
    pipe->InitBuffer(L1_2, L1_2_SIZE * 2);

    qL1Tensor[0] = L1_1.Get<INPUT_T>();
    qL1Tensor[1] = qL1Tensor[0][L1_1_SIZE / sizeof(INPUT_T)];

    kvL1Tensor[0] = L1_2.Get<INPUT_T>();
    kvL1Tensor[1] = kvL1Tensor[0][L1_2_SIZE / sizeof(INPUT_T)];

    // L0A
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2); // 64K
    aL0TensorPingPong[0] = tmpBufL0A.Get<INPUT_T>();
    aL0TensorPingPong[1] = aL0TensorPingPong[0][L0A_PP_SIZE / sizeof(INPUT_T)];

    // L0B
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2); // 64K
    bL0TensorPingPong[0] = tmpBufL0B.Get<INPUT_T>();
    bL0TensorPingPong[1] = bL0TensorPingPong[0][L0B_PP_SIZE / sizeof(INPUT_T)];

    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2); // 128K
    cL0TensorPingPong[0] = tmpBufL0C.Get<T>();
    cL0TensorPingPong[1] = cL0TensorPingPong[0][L0C_PP_SIZE / sizeof(T)];
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);

    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);

    SetFlag<HardEvent::MTE1_MTE2>(TMP_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(TMP_EVENT1);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(TMP_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(TMP_EVENT1);

    WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);

    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyNDGmToL1(LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmSrcTensor, 
                                    uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = Align(srcN, (uint32_t)BLOCK_CUBE);
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyNDGmToZZL1(LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmSrcTensor, 
                                                                 uint32_t srcN, uint32_t srcD, uint32_t srcDstride)
{
    ASCENDC_ASSERT(srcDstride < 65536, { KERNEL_LOG(KERNEL_ERROR, "srcDstride must less than 65536, current value is %u", srcDstride); });
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = CeilDiv(srcN, BLOCK_CUBE);
    nd2nzPara.nValue = BLOCK_CUBE; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = BLOCK_CUBE;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = BLOCK_CUBE * srcDstride;
    nd2nzPara.dstNzMatrixStride = BLOCK_CUBE * srcD;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyNZGmToL1(LocalTensor<INPUT_T> &l1Tensor, const GlobalTensor<INPUT_T> &gmSrcTensor, 
                                                               uint32_t srcN, uint32_t srcD, uint32_t srcNstride)
{
    DataCopyParams param;
    param.blockCount = CeilDiv(srcD, GetC0Num<INPUT_T>());
    param.blockLen = srcN;
    param.srcStride = (srcNstride - srcN);
    param.dstStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, param);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::LoadDataAToL0(LocalTensor<INPUT_T> &aL0Tensor, const LocalTensor<INPUT_T> &aL1Tensor, 
                                                                uint32_t mL1Size, uint32_t subMStart, uint32_t subMSize,
                                                                uint32_t subKStart, uint32_t subKSize)
{
    constexpr uint32_t ROWS_PER_FRAC = BLOCK_CUBE;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = subKSize / GetC0Num<INPUT_T>();
    loadData2DParams.srcStride = mL1Size / ROWS_PER_FRAC;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;

    LocalTensor<INPUT_T> srcTensor = aL1Tensor[mL1Size * subKStart][GetC0Num<INPUT_T>() * subMStart];

    uint32_t mLoops = subMSize / ROWS_PER_FRAC;
    for (uint32_t i = 0; i < mLoops; i++) {
        LoadData(aL0Tensor[i * ROWS_PER_FRAC * subKSize], srcTensor[i * ROWS_PER_FRAC * GetC0Num<INPUT_T>()], loadData2DParams);
    }
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::LoadDataZZAToL0(LocalTensor<INPUT_T> &aL0Tensor, const LocalTensor<INPUT_T> &aL1Tensor, 
                                                                  uint32_t kL1Size, uint32_t subMStart, uint32_t subMSize,
                                                                  uint32_t subKStart, uint32_t subKSize)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = (subMSize / BLOCK_CUBE) * (subKSize / GetC0Num<INPUT_T>()); // L1A为ZZ格式
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;

    LocalTensor<INPUT_T> srcTensor = aL1Tensor[kL1Size * subMStart + subKStart];
    LoadData(aL0Tensor, srcTensor, loadData2DParams);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::Load3DDataAToL0(LocalTensor<INPUT_T> &aL0Tensor, const LocalTensor<INPUT_T> &aL1Tensor, 
                                                                  uint32_t mL1Size, uint32_t subMStart, uint32_t subMSize,
                                                                  uint32_t subKStart, uint32_t subKSize)
{
    loadData3DParams.l1H = mL1Size / BLOCK_CUBE;
    loadData3DParams.mExtension = subMSize;
    loadData3DParams.kExtension = subKSize;
    loadData3DParams.channelSize = subKSize;

    LocalTensor<INPUT_T> srcTensor = aL1Tensor[mL1Size * subKStart][GetC0Num<INPUT_T>() * subMStart];
    LoadData(aL0Tensor, srcTensor, loadData3DParams);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::Load3DDataAToL0NonFMatrix(LocalTensor<INPUT_T> &aL0Tensor, const LocalTensor<INPUT_T> &aL1Tensor, 
                                                                            uint32_t mL1Size, uint32_t subMStart, uint32_t subMSize,
                                                                            uint32_t subKStart, uint32_t subKSize)
{
    loadData3DParams.mExtension = subMSize;
    loadData3DParams.kExtension = subKSize;
    loadData3DParams.channelSize = subKSize;

    LocalTensor<INPUT_T> srcTensor = aL1Tensor[mL1Size * subKStart][GetC0Num<INPUT_T>() * subMStart];
    LoadData<INPUT_T, LOAD3DV2_CONFIG>(aL0Tensor, srcTensor, loadData3DParams);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::LoadDataBTransposeToL0(LocalTensor<INPUT_T> &bL0Tensor, const LocalTensor<INPUT_T> &bL1Tensor, 
                                                                         uint32_t nL1Size, uint32_t subNStart, uint32_t subNSize,
                                                                         uint32_t subKStart, uint32_t subKSize)
{
    LocalTensor<INPUT_T> srcTensor = bL1Tensor[nL1Size * subKStart][GetC0Num<INPUT_T>() * subNStart];
    if (nL1Size == subNSize) {
        mm1LoadDataBTransposeToL0Params.repeatTimes = CeilDiv(subKSize, (uint32_t)BLOCK_CUBE) * CeilDiv(subNSize, GetC0Num<INPUT_T>());
        LoadData(bL0Tensor, srcTensor, mm1LoadDataBTransposeToL0Params);
    } else {
        mm1LoadDataBTransposeToL0Params.repeatTimes = CeilDiv(subNSize, (uint32_t)BLOCK_CUBE);
        uint32_t kLoops = subKSize / GetC0Num<INPUT_T>();
        for (uint32_t i = 0; i < kLoops; i++) {
            LoadData(bL0Tensor[subNSize * i * BLOCK_CUBE], srcTensor[nL1Size * i * GetC0Num<INPUT_T>()], mm1LoadDataBTransposeToL0Params);
        }
    }
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::LoadDataBToL0(LocalTensor<INPUT_T> &bL0Tensor, const LocalTensor<INPUT_T> &bL1Tensor, 
                                                                uint32_t kL1Size, uint32_t subKStart, uint32_t subKSize,
                                                                uint32_t subNStart, uint32_t subNSize)
{
    constexpr uint32_t ROWS_PER_FRAC = BLOCK_CUBE;
    mm2LoadDataBToL0Params.srcStride = kL1Size / ROWS_PER_FRAC;

    LocalTensor<INPUT_T> srcTensor = bL1Tensor[kL1Size * subNStart][GetC0Num<INPUT_T>() * subKStart];

    uint32_t kLoops = subKSize / ROWS_PER_FRAC;
    for (uint32_t i = 0; i < kLoops; i++) {
        LoadData(bL0Tensor[i * ROWS_PER_FRAC * subNSize], srcTensor[i * ROWS_PER_FRAC * GetC0Num<INPUT_T>()], mm2LoadDataBToL0Params);
    }
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::LoadDataZZBToL0(LocalTensor<INPUT_T> &bL0Tensor, const LocalTensor<INPUT_T> &bL1Tensor, 
                                    uint32_t nL1Size, uint32_t subKStart, uint32_t subKSize,
                                    uint32_t subNStart, uint32_t subNSize)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = (subKSize / BLOCK_CUBE) * (subNSize / GetC0Num<INPUT_T>()); // L1B为ZZ格式
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = true;

    LocalTensor<INPUT_T> srcTensor = bL1Tensor[nL1Size * subKStart + subNStart];
    LoadData(bL0Tensor, srcTensor, loadData2DParams);
}

template <typename INPUT_T, typename T, bool pageAttention>
template <CubeFormat GMFormat>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::FixpipeL0CToGM(GlobalTensor<T> &cGmTensor, const LocalTensor<T> &cL0Tensor, 
                                    uint32_t dstStride, uint32_t subMStart, uint32_t subMSize,
                                    uint32_t subNStart, uint32_t subNSize)
{
    FixpipeParamsV220 fixParams;
    fixParams.nSize = subNSize;
    fixParams.mSize = subMSize;
    fixParams.srcStride = Align(subMSize, (uint32_t)BLOCK_CUBE);
    fixParams.ndNum = 1;

    if constexpr (GMFormat == CubeFormat::NZ) {
        ASCENDC_ASSERT(subNSize % BLOCK_CUBE == 0, { KERNEL_LOG(KERNEL_ERROR, "subNSize must be divisible by 16, current value is %u", subNSize); });
        fixParams.dstStride = dstStride * BLOCK_CUBE * sizeof(T) / ONE_BLK_SIZE;
        GlobalTensor<T> dstTensor = cGmTensor[dstStride * subNStart][BLOCK_CUBE * subMStart];
        Fixpipe<T, T, CFG_NZ>(dstTensor, cL0Tensor, fixParams);
    } else {
        fixParams.dstStride = dstStride;
        GlobalTensor<T> dstTensor = cGmTensor[dstStride * subMStart + subNStart];
        Fixpipe<T, T, CFG_ROW_MAJOR>(dstTensor, cL0Tensor, fixParams);
    }
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyInMm1AToL1(LocalTensor<INPUT_T> &aL1Tensor, const SplitSameABExtraInfo &info)
{
    auto srcGm = queryGm[info.qCoreOffset];

    copyInMm1AToL1Params.nValue = info.cubeS1RealSize;
    if (unlikely(info.gBaseSize > 1)) {
        copyInMm1AToL1Params.ndNum = info.gBaseSize;
        copyInMm1AToL1Params.dValue = copyInMm1AToL1Params.dValue;
        copyInMm1AToL1Params.srcNdMatrixStride = copyInMm1AToL1Params.dValue;
        copyInMm1AToL1Params.dstNzC0Stride = Align<uint32_t>(info.cubeS1RealSize * info.gBaseSize, (uint32_t)BLOCK_CUBE);
        copyInMm1AToL1Params.dstNzMatrixStride = info.cubeS1RealSize * 32 / sizeof(INPUT_T);
        DataCopy(aL1Tensor, srcGm, copyInMm1AToL1Params);
        return;
    }
    copyInMm1AToL1Params.dstNzC0Stride = Align<uint32_t>(info.cubeS1RealSize, (uint32_t)BLOCK_CUBE);
    DataCopy(aL1Tensor, srcGm, copyInMm1AToL1Params);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyInMm1RopeAToL1(LocalTensor<INPUT_T> &aRopeL1Tensor, const SplitSameABExtraInfo &info)
{
    auto srcGm = qRopeGm[info.qRopeOffset];
    CopyNDGmToL1(aRopeL1Tensor, srcGm, info.cubeS1RealSize, ropeDSize, this->mm1RopeKa);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyInMm1BToL1(LocalTensor<INPUT_T> &bL1Tensor, const SplitSameABExtraInfo &info, 
                                                                 uint32_t subNStart, uint32_t subNSize)
{
    auto srcGm = keyGm[info.kCoreOffset + subNStart * this->mm1Kb];
    copyInMm1BToL1Params.nValue = subNSize;
    copyInMm1BToL1Params.dstNzC0Stride = Align(subNSize, (uint32_t)BLOCK_CUBE);
    DataCopy(bL1Tensor, srcGm, copyInMm1BToL1Params);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyInMm1RopeBToL1(LocalTensor<INPUT_T> &bRopeL1Tensor, const SplitSameABExtraInfo &info, 
                                                                     uint32_t subNStart, uint32_t subNSize)
{
    auto srcGm = kRopeGm[info.kRopeOffset + subNStart * this->mm1RopeKb];
    CopyNDGmToL1(bRopeL1Tensor, srcGm, subNSize, ropeDSize, this->mm1RopeKb);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyInMm1BToL1ForPA(LocalTensor<INPUT_T> &bL1Tensor, const uint64_t keyGmBaseOffset,
                                            uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                            uint32_t nActCopyRowCount, PaCacheLayoutType paCacheLayoutType)
{
    uint32_t blockElementCnt = 32 / sizeof(INPUT_T); // CUBE basic block size is 32
    if (IsSameType<INPUT_T, int4b_t>::value) {
        blockElementCnt = 64; // Block element count is 64
    }

    if (paCacheLayoutType == PaCacheLayoutType::PA_NZ) {
        DataCopyParams intriParams;
        intriParams.blockLen = nActCopyRowCount;
        intriParams.blockCount = this->dSize / blockElementCnt;
        intriParams.dstStride = copyTotalRowCntAlign - nActCopyRowCount;
        intriParams.srcStride = kvCacheBlockSize - nActCopyRowCount;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], keyGm[keyGmBaseOffset], intriParams);
    } else {
        uint64_t dStride = this->dSize;
        if (paCacheLayoutType == PaCacheLayoutType::PA_BBH) {
            dStride = this->valueNSize * this->dSize;
        }

        Nd2NzParams mm1Nd2NzParamsForB;
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
        if constexpr (IsSameType<INPUT_T, int4b_t>::value) {
            mm1Nd2NzParamsForB.dValue = this->dSize / 2;
            mm1Nd2NzParamsForB.srcDValue = dStride / 2;
        } else {
            mm1Nd2NzParamsForB.dValue = this->dSize;
            mm1Nd2NzParamsForB.srcDValue = dStride;
        }
        mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
        mm1Nd2NzParamsForB.dstNzNStride = 1;
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], keyGm[keyGmBaseOffset], mm1Nd2NzParamsForB);
    }
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyInMm1RopeBToL1ForPA(LocalTensor<INPUT_T> &bRopeL1Tensor, const uint64_t kRopeGmBaseOffset,
                                            uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt,
                                            uint32_t nActCopyRowCount, PaCacheLayoutType paCacheLayoutType)
{
    uint32_t blockElementCnt = 32 / sizeof(INPUT_T); // CUBE basic block size is 32
    if (IsSameType<INPUT_T, int4b_t>::value) {
        blockElementCnt = 64; // Block element count is 64
    }

    if (paCacheLayoutType == PaCacheLayoutType::PA_NZ) {
        DataCopyParams intriParams;
        intriParams.blockLen = nActCopyRowCount;
        intriParams.blockCount = this->ropeDSize / blockElementCnt;
        intriParams.dstStride = copyTotalRowCntAlign - nActCopyRowCount;
        intriParams.srcStride = kvCacheBlockSize - nActCopyRowCount;
        DataCopy(bRopeL1Tensor[copyStartRowCnt * blockElementCnt], kRopeGm[kRopeGmBaseOffset], intriParams);
    } else {
        uint64_t dStride = this->ropeDSize;
        if (paCacheLayoutType == PaCacheLayoutType::PA_BBH) {
            dStride = this->valueNSize * this->ropeDSize;
        }

        Nd2NzParams mm1Nd2NzParamsForB;
        mm1Nd2NzParamsForB.ndNum = 1;
        mm1Nd2NzParamsForB.nValue = nActCopyRowCount;
        if constexpr (IsSameType<INPUT_T, int4b_t>::value) {
            mm1Nd2NzParamsForB.dValue = this->ropeDSize / 2;
            mm1Nd2NzParamsForB.srcDValue = dStride / 2;
        } else {
            mm1Nd2NzParamsForB.dValue = this->ropeDSize;
            mm1Nd2NzParamsForB.srcDValue = dStride;
        }
        mm1Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
        mm1Nd2NzParamsForB.dstNzNStride = 1;
        mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
        mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
        DataCopy(bRopeL1Tensor[copyStartRowCnt * blockElementCnt], kRopeGm[kRopeGmBaseOffset], mm1Nd2NzParamsForB);
    }
}

template <typename INPUT_T, typename T, bool pageAttention>
template <CubeFormat AFormat>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyInMm2AToL1(LocalTensor<INPUT_T> &aL1Tensor, const SplitSameABExtraInfo &info, 
                                    uint32_t gmStride, uint32_t subMStart, uint32_t subMSize,
                                    uint32_t subKStart, uint32_t subKSize)
{
    if constexpr (AFormat == CubeFormat::NZ) {
        auto srcGm = vec1ResGm[info.taskIdMod2][gmStride * subKStart + GetC0Num<INPUT_T>() * subMStart];
        CopyNZGmToL1(aL1Tensor, srcGm, subMSize, subKSize, gmStride);
    } else {
        auto srcGm = vec1ResGm[info.taskIdMod2][gmStride * subMStart + subKStart];
        CopyNDGmToL1(aL1Tensor, srcGm, subMSize, subKSize, gmStride);
    }
}

template <typename INPUT_T, typename T, bool pageAttention>
template <uint32_t KL0_SPLITE_SIZE>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyInMm2BToL1(LocalTensor<INPUT_T> &bL1Tensor, const SplitSameABExtraInfo &info, 
                                                                 uint32_t subKStart, uint32_t subKSize)
{
    auto srcGm = valueGm[info.vCoreOffset + subKStart * this->mm2Kb];
    copyInMm2BToL1Params.nValue = subKSize;
    copyInMm2BToL1Params.dstNzC0Stride = Align<uint32_t>(subKSize, (uint32_t)BLOCK_CUBE);
    DataCopy(bL1Tensor, srcGm, copyInMm2BToL1Params);
}

template <typename INPUT_T, typename T, bool pageAttention>
template <uint32_t KL0_SPLITE_SIZE>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyInMm2BToL1ForPA(LocalTensor<INPUT_T> &bL1Tensor, const uint64_t valueGmBaseOffset,
                                            uint32_t copyTotalRowCntAlign, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount, uint32_t copyStartColumnCount,
                                            uint32_t copyColumnCount, PaCacheLayoutType paCacheLayoutType)
{
    uint32_t blockElementCnt = 32 / sizeof(INPUT_T); // CUBE basic block size is 32
    if (IsSameType<INPUT_T, int4b_t>::value) {
        blockElementCnt = 64; // Block element count is 64
    }

    if (paCacheLayoutType == PaCacheLayoutType::PA_NZ) {
        DataCopyParams intriParams;
        intriParams.blockLen = nActCopyRowCount;
        intriParams.blockCount = copyColumnCount / blockElementCnt;
        intriParams.dstStride = copyTotalRowCntAlign - nActCopyRowCount;
        intriParams.srcStride = kvCacheBlockSize - nActCopyRowCount;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], valueGm[valueGmBaseOffset + copyStartColumnCount * kvCacheBlockSize], intriParams);
    } else {
        uint64_t step = this->valueDSize;
        if (paCacheLayoutType == PaCacheLayoutType::PA_BBH) {
            step = this->valueNSize * this->valueDSize;
        }

        Nd2NzParams mm2Nd2NzParamsForB;
        mm2Nd2NzParamsForB.ndNum = 1;
        mm2Nd2NzParamsForB.nValue = nActCopyRowCount;
        if constexpr (IsSameType<INPUT_T, int4b_t>::value) {
            mm2Nd2NzParamsForB.dValue = copyColumnCount / 2;
            mm2Nd2NzParamsForB.srcDValue = step / 2;
        } else {
            mm2Nd2NzParamsForB.dValue = copyColumnCount;
            mm2Nd2NzParamsForB.srcDValue = step;
        }
        mm2Nd2NzParamsForB.dstNzC0Stride = copyTotalRowCntAlign;
        mm2Nd2NzParamsForB.dstNzNStride = 1;
        mm2Nd2NzParamsForB.srcNdMatrixStride = 0;
        mm2Nd2NzParamsForB.dstNzMatrixStride = 0;
        DataCopy(bL1Tensor[copyStartRowCnt * blockElementCnt], valueGm[valueGmBaseOffset + copyStartColumnCount], mm2Nd2NzParamsForB);
    }
}

template <typename INPUT_T, typename T, bool pageAttention>
template <uint32_t M_L1_SIZE>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::ResetLoad3DConfig()
{
    constexpr uint8_t padList[] = {0, 0, 0, 255};
    SetFmatrix(M_L1_SIZE / BLOCK_CUBE, GetC0Num<INPUT_T>(), padList, FmatrixMode::FMATRIX_LEFT);
    SetLoadDataPaddingValue<INPUT_T>(0);
}

template <typename INPUT_T, typename T, bool pageAttention>
__aicore__ inline LocalTensor<INPUT_T> PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyQToL1(const SplitSameABExtraInfo &info, uint32_t mL1Size)
{
    uint32_t pingpong = this->qL1BufIter % 2;
    if (qL1BufAddr[0] == info.qCoreOffset) {
        qL1BufIter += pingpong;
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + 0);
        return this->qL1Tensor[0];
    } else if (qL1BufAddr[1] == info.qCoreOffset) {
        qL1BufIter += (pingpong - 1);
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + 1);
        return this->qL1Tensor[1];
    }

    WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + pingpong);
    this->CopyInMm1AToL1(this->qL1Tensor[pingpong], info);
    if (info.withRope) {
        auto qRopeL1Tensor = this->qL1Tensor[pingpong][mL1Size * this->dSize];
        this->CopyInMm1RopeAToL1(qRopeL1Tensor, info);
    }

    SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + pingpong);
    WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + pingpong);
    qL1BufAddr[pingpong] = info.qCoreOffset;

    return this->qL1Tensor[pingpong];
}

template <typename INPUT_T, typename T, bool pageAttention>
template <uint32_t KL0_SPLITE_SIZE>
__aicore__ inline LocalTensor<INPUT_T> PfaMatmulKvNd<INPUT_T, T, pageAttention>::CopyVToL1(const SplitSameABExtraInfo &info, uint32_t subKStart, uint32_t subKSize)
{
    uint32_t pingpong = this->qL1BufIter % 2;
    WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + pingpong);
    this->template CopyInMm2BToL1<KL0_SPLITE_SIZE>(this->qL1Tensor[pingpong], info, subKStart, subKSize);
    SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + pingpong);
    WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + pingpong);
    qL1BufAddr[pingpong] = (uint64_t)-1;
    return this->qL1Tensor[pingpong];
}

template <typename INPUT_T, typename T, bool pageAttention>
template <CubeFormat OutFormat>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::ComputeMm1(const SplitSameABExtraInfo &info)
{
    ASCENDC_ASSERT((!info.withRope) || (dSize == BlockAlign<INPUT_T>(dSize)), 
        { KERNEL_LOG(KERNEL_ERROR, "dSize must block aligned if with seprate rope, current value is %u", dSize); });
    constexpr uint32_t mSplitSize = 512;
    constexpr uint32_t nSplitSize = 128;
    constexpr uint32_t mL0SplitSize = 128;
    uint32_t kL0SplitSize = 96;
    if (info.qkvDSameFlag) {
        kL0SplitSize = 128; // qkv D等长=128，不切K轴场景
    }

    uint32_t mSizeAct = info.cubeS1RealSize * info.gBaseSize;
    uint32_t mL1Size = Align(mSizeAct, (uint32_t)BLOCK_CUBE);
    uint32_t kSizeAct = dSize + ropeDSize;
    uint32_t kSize = BlockAlign<INPUT_T>(kSizeAct);
    uint32_t nSizeAct = info.s2RealSize;
    uint32_t nSize = info.s2AlignedSize;

    uint32_t nloops = CeilDiv(nSizeAct, nSplitSize);
    uint32_t nTail = nSizeAct - (nloops - 1) * nSplitSize;
    uint32_t mL0Loops = CeilDiv(mSizeAct, mL0SplitSize);
    uint32_t mL0Tail = mSizeAct - (mL0Loops - 1) * mL0SplitSize;
    uint32_t kL0Loops = CeilDiv(kSize, kL0SplitSize);
    uint32_t kL0Tail = kSize - (kL0Loops - 1) * kL0SplitSize;
    ResetLoad3DConfig<mSplitSize>();

    DEF_LAMBDA_WITHTHIS(CopyKToL1)(const SplitSameABExtraInfo &info, uint32_t subNStart, uint32_t subNSize) -> LocalTensor<INPUT_T> {
        uint32_t pingpong = self_->kvL1BufIter % 2;
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + pingpong);
        self_->CopyInMm1BToL1(self_->kvL1Tensor[pingpong], info, subNStart, subNSize);
        if (info.withRope) {
            auto kRopeL1Tensor = self_->kvL1Tensor[pingpong][Align(subNSize, (uint32_t)BLOCK_CUBE) * self_->dSize];
            self_->CopyInMm1RopeBToL1(kRopeL1Tensor, info, subNStart, subNSize);
        }
        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + pingpong);
        return self_->kvL1Tensor[pingpong];
    } DEF_LAMBDA_END(CopyKToL1);

    DEF_LAMBDA_WITHTHIS(CopyKToL1ForPA)(const SplitSameABExtraInfo &info, uint32_t subNStart, uint32_t subNSize) -> LocalTensor<INPUT_T> {
        uint32_t pingpong = self_->kvL1BufIter % 2;
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + pingpong);
        auto kTensor = self_->kvL1Tensor[pingpong];
        auto kRopeTensor = self_->kvL1Tensor[pingpong][Align(subNSize, (uint32_t)BLOCK_CUBE) * self_->dSize];

        uint64_t blockTableBaseOffset = info.boIdx * self_->maxBlockNumPerBatch;
        uint32_t curSeqIdx = info.s2BaseOffset + subNStart;
        uint32_t copyFinishRowCnt = 0;
        while (copyFinishRowCnt < subNSize) {
            uint64_t blockIdOffset = curSeqIdx / self_->kvCacheBlockSize;
            uint64_t reaminRowCnt = curSeqIdx % self_->kvCacheBlockSize;
            uint64_t idInBlockTable = self_->blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);

            uint32_t copyRowCnt = self_->kvCacheBlockSize - reaminRowCnt;
            if (copyFinishRowCnt + copyRowCnt > subNSize) {
                copyRowCnt = subNSize - copyFinishRowCnt;
            }

            uint64_t keyOffset = idInBlockTable * self_->kvCacheBlockSize * self_->valueNSize * self_->dSize;
            uint64_t kRopeOffset = idInBlockTable * self_->kvCacheBlockSize * self_->valueNSize * self_->ropeDSize;
            if (info.paCacheLayoutType == PaCacheLayoutType::PA_NZ) {
                uint32_t blockElementCnt = 32 / sizeof(INPUT_T); // CUBE basic block size is 32
                if (IsSameType<INPUT_T, int4b_t>::value) {
                    blockElementCnt = 64; // Block element count is 64
                }
                keyOffset += (uint64_t)(info.n2oIdx * self_->dSize * self_->kvCacheBlockSize) + reaminRowCnt * blockElementCnt;
                kRopeOffset += (uint64_t)(info.n2oIdx * self_->ropeDSize * self_->kvCacheBlockSize) + reaminRowCnt * blockElementCnt;
            } else {
                if (info.paCacheLayoutType == PaCacheLayoutType::PA_BBH) {
                    // BBH
                    keyOffset += (uint64_t)(info.n2oIdx * self_->dSize) + reaminRowCnt * self_->valueNSize * self_->dSize;
                    kRopeOffset += (uint64_t)(info.n2oIdx * self_->ropeDSize) + reaminRowCnt * self_->valueNSize * self_->ropeDSize;
                } else {
                    // BNBD
                    keyOffset += (uint64_t)(info.n2oIdx * self_->dSize * self_->kvCacheBlockSize) + reaminRowCnt * self_->dSize;
                    kRopeOffset += (uint64_t)(info.n2oIdx * self_->ropeDSize * self_->kvCacheBlockSize) + reaminRowCnt * self_->ropeDSize;
                }
            }

            self_->CopyInMm1BToL1ForPA(kTensor, keyOffset, Align(subNSize, (uint32_t)BLOCK_CUBE), copyFinishRowCnt, copyRowCnt, info.paCacheLayoutType);
            if (info.withRope) {
                self_->CopyInMm1RopeBToL1ForPA(kRopeTensor, kRopeOffset, Align(subNSize, (uint32_t)BLOCK_CUBE), copyFinishRowCnt, copyRowCnt, info.paCacheLayoutType);
            }

            // 更新循环变量
            copyFinishRowCnt += copyRowCnt;
            curSeqIdx += copyRowCnt;
        }
        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + pingpong);
        return self_->kvL1Tensor[pingpong];
    } DEF_LAMBDA_END(CopyKToL1ForPA);

    DEF_LAMBDA_WITHTHIS(LoadAToL0)(const LocalTensor<INPUT_T> &qL1Tensor, uint32_t mL1Size,
                                   uint32_t subMStart, uint32_t subMSize, uint32_t subKStart, uint32_t subKSize) -> LocalTensor<INPUT_T> {
        uint32_t pingpong = self_->abL0BufIter % 2;
        if (likely(mL1Size == mSplitSize)) {
            self_->Load3DDataAToL0NonFMatrix(self_->aL0TensorPingPong[pingpong], qL1Tensor, mL1Size, subMStart, subMSize, subKStart, subKSize);
        } else {
            self_->Load3DDataAToL0(self_->aL0TensorPingPong[pingpong], qL1Tensor, mL1Size, subMStart, subMSize, subKStart, subKSize);
            self_->template ResetLoad3DConfig<mSplitSize>();
        }
        return self_->aL0TensorPingPong[pingpong];
    } DEF_LAMBDA_END(LoadAToL0);

    DEF_LAMBDA_WITHTHIS(LoadBToL0)(const LocalTensor<INPUT_T> &kL1Tensor, uint32_t nL1Size,
                                   uint32_t subNStart, uint32_t subNSize, uint32_t subKStart, uint32_t subKSize) -> LocalTensor<INPUT_T> {
        uint32_t pingpong = self_->abL0BufIter % 2;
        self_->LoadDataBTransposeToL0(self_->bL0TensorPingPong[pingpong], kL1Tensor, nL1Size, subNStart, subNSize, subKStart, subKSize);
        return self_->bL0TensorPingPong[pingpong];
    } DEF_LAMBDA_END(LoadBToL0);

    DEF_LAMBDA_WITHTHIS(FixpipeCToGM)(const SplitSameABExtraInfo &info, const LocalTensor<T> &cL0Tensor,
                                      uint32_t dstStride, uint32_t subMStart, uint32_t subMSize,
                                      uint32_t subNStart, uint32_t subNSize) {
        uint32_t pingpong = self_->cL0BufIter % 2;
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + pingpong);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + pingpong);
        self_->template FixpipeL0CToGM<OutFormat>(self_->mm1ResGm[info.taskIdMod2], cL0Tensor, dstStride, subMStart, subMSize, subNStart, subNSize);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + pingpong);
    } DEF_LAMBDA_END(FixpipeCToGM);

    auto qTensor = CopyQToL1(info, mL1Size);
    bool l1BufLoaded[2] = {false, false};
    uint32_t subNSize = nSplitSize;
    uint32_t subNSizeAct = nSplitSize;
    for (uint32_t n = 0; n < nloops; n++) {
        if (n == nloops - 1) {
            subNSizeAct = nTail;
            subNSize = Align(nTail, (uint32_t)BLOCK_CUBE);
        }

        if (! l1BufLoaded[kvL1BufIter % 2]) {
            if constexpr (pageAttention) {
                CopyKToL1ForPA(info, n * nSplitSize, subNSizeAct);
            } else {
                CopyKToL1(info, n * nSplitSize, subNSizeAct);
            }
        }

        WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvL1BufIter % 2);
        l1BufLoaded[kvL1BufIter % 2] = false;
        auto kTensor = kvL1Tensor[kvL1BufIter % 2];

        uint32_t subML0Size = mL0SplitSize;
        uint32_t subML0SizeAct = mL0SplitSize;
        for (uint32_t mL0 = 0; mL0 < mL0Loops; mL0++) {
            if (mL0 == mL0Loops - 1) {
                subML0SizeAct = mL0Tail;
                subML0Size = Align(mL0Tail, (uint32_t)BLOCK_CUBE);
            }

            if ((info.qkvDSameFlag == false) && (mL0 == 2)) {
                ++kvL1BufIter;
                if (! l1BufLoaded[kvL1BufIter % 2]) {
                    auto nextN = n + 1;
                    uint32_t nextSubNSizeAct = nSplitSize;
                    if (nextN < nloops) {
                        if (nextN == nloops - 1) {
                            nextSubNSizeAct = nTail;
                        }
                        if constexpr (pageAttention) {
                            CopyKToL1ForPA(info, nextN * nSplitSize, nextSubNSizeAct);
                        } else {
                            CopyKToL1(info, nextN * nSplitSize, nextSubNSizeAct);
                        }
                        l1BufLoaded[kvL1BufIter % 2] = true;
                    }
                }
                --kvL1BufIter;
            }

            WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
            auto cL0Tensor = cL0TensorPingPong[cL0BufIter % 2];

            uint32_t subKL0Size = kL0SplitSize;
            uint32_t subKL0SizeAct = kL0SplitSize;
            for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                bool isLastKL0 = (kL0 == kL0Loops - 1);
                if (unlikely(isLastKL0)) {
                    subKL0SizeAct = kL0Tail;
                    subKL0Size = BlockAlign<INPUT_T>(subKL0SizeAct);
                }

                WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + abL0BufIter % 2);
                auto bL0Tensor = LoadBToL0(kTensor, subNSize, 0, subNSize, kL0 * kL0SplitSize, subKL0Size);
                auto aL0Tensor = LoadAToL0(qTensor, mL1Size, mL0 * mL0SplitSize, subML0Size, kL0 * kL0SplitSize, subKL0Size);
                SetFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + abL0BufIter % 2);
                WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + abL0BufIter % 2);

                MmadParams mmadParams;
                mmadParams.m = subML0Size;
                mmadParams.n = subNSizeAct;
                mmadParams.k = subKL0SizeAct;
                mmadParams.cmatrixInitVal = (kL0 == 0);
                mmadParams.cmatrixSource = false;

                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                if (likely(! isLastKL0)) {
                    PipeBarrier<PIPE_M>();
                }
                SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + abL0BufIter % 2);
                abL0BufIter++;
            }

            FixpipeCToGM(info, cL0Tensor, (OutFormat == CubeFormat::NZ) ? mSizeAct : nSizeAct, 
                         mL0 * mL0SplitSize, subML0SizeAct, n * nSplitSize, (OutFormat == CubeFormat::NZ) ? subNSize : subNSizeAct);
            cL0BufIter++;
        }

        SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvL1BufIter % 2);
        kvL1BufIter++;
    }

    SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qL1BufIter % 2);
    qL1BufIter++;
}

template <typename INPUT_T, typename T, bool pageAttention>
template <CubeFormat AFormat, CubeFormat OutFormat>
__aicore__ inline void PfaMatmulKvNd<INPUT_T, T, pageAttention>::ComputeMm2(const SplitSameABExtraInfo &info)
{
    constexpr uint32_t mSplitSize = 128;
    constexpr uint32_t kSplitSize = 256;
    constexpr uint32_t kL0SplitSize = 128;

    uint32_t mSizeAct = info.cubeS1RealSize * info.gBaseSize;
    uint32_t mInputSize = Align(mSizeAct, (uint32_t)BLOCK_CUBE);
    uint32_t kSizeAct = info.s2RealSize;
    uint32_t kSize = info.s2AlignedSize;
    uint32_t nSizeAct = valueDSize;
    uint32_t nSize = BlockAlign<INPUT_T>(nSizeAct);

    uint32_t mloops = CeilDiv(mSizeAct, mSplitSize);
    uint32_t mTail = mSizeAct - (mloops - 1) * mSplitSize;
    uint32_t kloops = CeilDiv(kSizeAct, kSplitSize);
    uint32_t kTail = kSizeAct - (kloops - 1) * kSplitSize;

    ResetLoad3DConfig<mSplitSize>();

    DEF_LAMBDA_WITHTHIS(CopyPToL1)(const SplitSameABExtraInfo &info, uint32_t gmStride, 
                                   uint32_t subMStart, uint32_t subMSize, uint32_t subKStart, uint32_t subKSize) -> LocalTensor<INPUT_T> {
        uint32_t pingpong = self_->kvL1BufIter % 2;
        WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + pingpong);
        self_->template CopyInMm2AToL1<AFormat>(self_->kvL1Tensor[pingpong], info, gmStride, subMStart, subMSize, subKStart, subKSize);
        SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + pingpong);
        WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + pingpong);
        return self_->kvL1Tensor[pingpong];
    } DEF_LAMBDA_END(CopyPToL1);

    DEF_LAMBDA_WITHTHIS(CopyVToL1ForPA)(const SplitSameABExtraInfo &info, uint32_t subKStart, uint32_t subKSize, uint32_t subDSize) -> LocalTensor<INPUT_T> {
        uint32_t pingpong = self_->qL1BufIter % 2;
        WaitFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + pingpong);
        auto valueTensor = self_->qL1Tensor[pingpong];

        uint64_t blockTableBaseOffset = info.boIdx * self_->maxBlockNumPerBatch;
        uint32_t curSeqIdx = info.s2BaseOffset + subKStart;
        uint32_t copyFinishRowCnt = 0;
        while (copyFinishRowCnt < subKSize) {
            uint64_t blockIdOffset = curSeqIdx / self_->kvCacheBlockSize;
            uint64_t reaminRowCnt = curSeqIdx % self_->kvCacheBlockSize;
            uint64_t idInBlockTable = self_->blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);

            uint32_t copyRowCnt = self_->kvCacheBlockSize - reaminRowCnt;
            if (copyFinishRowCnt + copyRowCnt > subKSize) {
                copyRowCnt = subKSize - copyFinishRowCnt;
            }

            uint64_t valueOffset = idInBlockTable * self_->kvCacheBlockSize * self_->valueNSize * self_->valueDSize;
            if (info.paCacheLayoutType == PaCacheLayoutType::PA_NZ) {
                uint32_t blockElementCnt = 32 / sizeof(INPUT_T); // CUBE basic block size is 32
                if (IsSameType<INPUT_T, int4b_t>::value) {
                    blockElementCnt = 64; // Block element count is 64
                }
                valueOffset += (uint64_t)(info.n2oIdx * self_->valueDSize * self_->kvCacheBlockSize) + reaminRowCnt * blockElementCnt;
            } else {
                if (info.paCacheLayoutType == PaCacheLayoutType::PA_BBH) {
                    // BBH
                    valueOffset += (uint64_t)(info.n2oIdx * self_->valueDSize) + reaminRowCnt * self_->valueNSize * self_->valueDSize;
                } else {
                    // BNBD
                    valueOffset += (uint64_t)(info.n2oIdx * self_->valueDSize * self_->kvCacheBlockSize) + reaminRowCnt * self_->valueDSize;
                }
            }

            self_->template CopyInMm2BToL1ForPA<kL0SplitSize>(valueTensor, valueOffset, Align(subKSize, (uint32_t)BLOCK_CUBE), copyFinishRowCnt, copyRowCnt, 0, subDSize, info.paCacheLayoutType);

            // 更新循环变量
            copyFinishRowCnt += copyRowCnt;
            curSeqIdx += copyRowCnt;
        }
        SetFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + pingpong);
        WaitFlag<HardEvent::MTE2_MTE1>(Q_EVENT0 + pingpong);
        self_->qL1BufAddr[pingpong] = (uint64_t)-1;
        return self_->qL1Tensor[pingpong];
    } DEF_LAMBDA_END(CopyVToL1ForPA);

    DEF_LAMBDA_WITHTHIS(LoadAToL0)(const LocalTensor<INPUT_T> &pL1Tensor, uint32_t mL1Size,
                                   uint32_t subMStart, uint32_t subMSize, uint32_t subKStart, uint32_t subKSize) -> LocalTensor<INPUT_T> {
        uint32_t pingpong = self_->abL0BufIter % 2;
        if (likely(mL1Size == mSplitSize)) {
            self_->Load3DDataAToL0NonFMatrix(self_->aL0TensorPingPong[pingpong], pL1Tensor, mL1Size, subMStart, subMSize, subKStart, subKSize);
        } else {
            self_->Load3DDataAToL0(self_->aL0TensorPingPong[pingpong], pL1Tensor, mL1Size, subMStart, subMSize, subKStart, subKSize);
            self_->template ResetLoad3DConfig<mSplitSize>();
        }
        return self_->aL0TensorPingPong[pingpong];
    } DEF_LAMBDA_END(LoadAToL0);

    DEF_LAMBDA_WITHTHIS(LoadBToL0)(uint32_t nSize, const LocalTensor<INPUT_T> &vL1Tensor, uint32_t kL1Size,
                                   uint32_t subKStart, uint32_t subKSize, uint32_t subNStart, uint32_t subNSize) -> LocalTensor<INPUT_T> {
        uint32_t pingpong = self_->abL0BufIter % 2;
        self_->LoadDataBToL0(self_->bL0TensorPingPong[pingpong], vL1Tensor, kL1Size, subKStart, subKSize, subNStart, subNSize);
        return self_->bL0TensorPingPong[pingpong];
    } DEF_LAMBDA_END(LoadBToL0);

    DEF_LAMBDA_WITHTHIS(FixpipeCToGM)(const SplitSameABExtraInfo &info, const LocalTensor<T> &cL0Tensor,
                                      uint32_t dstStride, uint32_t subMStart, uint32_t subMSize,
                                      uint32_t subNStart, uint32_t subNSize) {
        uint32_t pingpong = self_->cL0BufIter % 2;
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + pingpong);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + pingpong);
        self_->template FixpipeL0CToGM<OutFormat>(self_->mm2ResGm[info.taskIdMod2], cL0Tensor, dstStride, subMStart, subMSize, subNStart, subNSize);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + pingpong);
    } DEF_LAMBDA_END(FixpipeCToGM);

    LocalTensor<INPUT_T> vTensor;
    if constexpr (pageAttention) {
        vTensor = CopyVToL1ForPA(info, 0, kSizeAct, nSize);
    } else {
        vTensor = CopyVToL1<kL0SplitSize>(info, 0, kSizeAct);
    }
    uint32_t subMSize = mSplitSize;
    uint32_t subMSizeAct = mSplitSize;
    for (uint32_t m = 0; m < mloops; m++) {
        if (m == mloops - 1) {
            subMSizeAct = mTail;
            subMSize = Align(mTail, (uint32_t)BLOCK_CUBE);
        }

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + cL0BufIter % 2);
        auto cL0Tensor = cL0TensorPingPong[cL0BufIter % 2];

        uint32_t subKSize = kSplitSize;
        uint32_t subKSizeAct = kSplitSize;
        for (uint32_t k = 0; k < kloops; k++) {
            bool isLastKL1 = (k == kloops - 1);
            if (unlikely(isLastKL1)) {
                subKSizeAct = kTail;
                subKSize = Align(kTail, (uint32_t)BLOCK_CUBE);
            }

            auto pTensor = CopyPToL1(info, (AFormat == CubeFormat::NZ) ? mInputSize : kSize, m * mSplitSize, subMSize, k * kSplitSize, subKSize);

            uint32_t kL0Loops = CeilDiv(subKSizeAct, kL0SplitSize);
            uint32_t kL0Tail = subKSizeAct - (kL0Loops - 1) * kL0SplitSize;
            uint32_t subKL0Size = kL0SplitSize;
            uint32_t subKL0SizeAct = kL0SplitSize;
            for (uint32_t kL0 = 0; kL0 < kL0Loops; kL0++) {
                bool isLastKL0 = (kL0 == kL0Loops - 1);
                if (unlikely(isLastKL0)) {
                    subKL0SizeAct = kL0Tail;
                    subKL0Size = BlockAlign<INPUT_T>(subKL0SizeAct);
                }

                WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + abL0BufIter % 2);
                auto aL0Tensor = LoadAToL0(pTensor, subMSize, 0, subMSize, kL0 * kL0SplitSize, subKL0Size);
                auto bL0Tensor = LoadBToL0(nSize, vTensor, kSize, (k * kSplitSize) + (kL0 * kL0SplitSize), subKL0Size, 0, nSize);
                SetFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + abL0BufIter % 2);
                WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + abL0BufIter % 2);

                MmadParams mmadParams;
                mmadParams.m = subMSize;
                mmadParams.n = nSizeAct;
                mmadParams.k = subKL0SizeAct;
                mmadParams.cmatrixInitVal = (k == 0) && (kL0 == 0);
                mmadParams.cmatrixSource = false;

                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                if (likely(! (isLastKL1 && isLastKL0))) {
                    PipeBarrier<PIPE_M>();
                }
                SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + abL0BufIter % 2);
                abL0BufIter++;
            }

            SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvL1BufIter % 2);
            kvL1BufIter++;
        }

        FixpipeCToGM(info, cL0Tensor, (OutFormat == CubeFormat::NZ) ? mSizeAct : nSizeAct, 
                         m * mSplitSize, subMSizeAct, 0, (OutFormat == CubeFormat::NZ) ? nSize : nSizeAct);
        cL0BufIter++;
    }

    SetFlag<HardEvent::MTE1_MTE2>(Q_EVENT0 + qL1BufIter % 2);
    qL1BufIter++;
}

#endif //PFA_SERVICE_MATMUL_KV_ND_H