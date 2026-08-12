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
 * \file weight_quant_batch_matmul_v2_msd_group.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_MSD_GROUP_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_MSD_GROUP_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "tool.h"
#include "weight_quant_batch_matmul_v2_constant.h"

using AscendC::AIC;
using AscendC::AIV;
using AscendC::BinaryRepeatParams;
using AscendC::BLOCK_CUBE;
using AscendC::BrcbRepeatParams;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::Nd2NzParams;
using AscendC::QuePosition;
using AscendC::SetAtomicAdd;
using AscendC::SetAtomicNone;
using AscendC::SetFlag;
using AscendC::SyncAll;
using AscendC::TBuf;
using AscendC::TEventID;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::WaitFlag;
using matmul::MatmulImpl;
using matmul::MatmulType;

namespace Mc2WeightQuantBatchMatmulV2 {
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
class Mc2WeightQuantBatchMatMulV2MsdGroupKernel {
    static constexpr int32_t SYNC_VECTOR_CUBE_P_FLAG = 2;
    static constexpr int32_t SYNC_VECTOR_CUBE_Q_FLAG = 3;
    static constexpr uint64_t GM_ADDR_ALIGN_BASIC_BLOCK = 512;
    static constexpr uint64_t GROUP_DIM = 32;

public:
    __aicore__ inline Mc2WeightQuantBatchMatMulV2MsdGroupKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2MsdGroupTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInVector1(uint32_t m, uint32_t k);
    __aicore__ inline void ProcessVector1(uint32_t m, uint32_t k);
    __aicore__ inline void UnfoldAMatrix(uint32_t m, uint32_t k);
    __aicore__ inline void ComputeAMatrixSum(uint32_t m, uint32_t k, BinaryRepeatParams &maxAddReatParams,
                                             DataCopyParams &dataCopyParams, BrcbRepeatParams &brcbRepeatParams);
    __aicore__ inline void ComputeAMatrixMax(uint32_t m, uint32_t k, BinaryRepeatParams &maxAddReatParams,
                                             DataCopyParams &dataCopyParams, BrcbRepeatParams &brcbRepeatParams);
    __aicore__ inline void CopyOutVector1(uint32_t m, uint32_t k);
    __aicore__ inline void ProcessCube(uint32_t m, uint32_t cubeSingleCoreN, uint32_t idxN, uint32_t baseN,
                                       uint32_t realBaseN, uint32_t groupIdx, uint32_t nloopIdx);
    __aicore__ inline void CopyInVector3(int32_t vecSingleCoreN, uint32_t groupIdx, uint32_t nBaseOffset,
                                         uint32_t taskId);
    __aicore__ inline void ProcessVector3(int32_t vecSingleCoreN, uint32_t groupIdx, uint32_t nBaseOffset,
                                          uint32_t groupLimit, uint32_t taskId);
    __aicore__ inline void MulOffsetVector3(int32_t vecSingleCoreN, uint32_t groupIdx, uint32_t taskId);

    __aicore__ inline void CopyInAndProcessVector4(int64_t vecSingleCoreN, uint64_t groupIdx, uint64_t nOffset,
                                                   uint32_t taskId, uint32_t groupLimit);
    __aicore__ inline void ProcessVector4MaxScale(int64_t vecSingleCoreN, uint64_t groupIdx);
    __aicore__ inline void CopyOutVector4(int32_t singleCoreRealN, uint32_t groupIdx, uint32_t groupLimit);
    __aicore__ inline void ProcessMatmulResult(uint64_t cOffset, uint64_t v4BaseM, uint64_t v4EndM,
                                               int64_t vecSingleCoreN);
    __aicore__ inline void ProcessMatmulResultHalf(uint64_t cOffset, uint64_t v4EndM, int64_t vecSingleCoreN);
    __aicore__ inline void ProcessVector5(int32_t singleCoreRealN);

    __aicore__ inline void CopyInBL1(uint64_t wOffset, uint64_t singleCoreRealN, Nd2NzParams &nd2nzParams);
    __aicore__ inline void CopyInBL1Nz(const LocalTensor<int8_t> &bL1Tensor, uint64_t kOffset, uint64_t nOffset,
                                       uint64_t l1BaseKb, uint64_t singleCoreRealN);
    __aicore__ inline void CopyInAL1(uint64_t singleCoreRealK, uint64_t kOffset);
    __aicore__ inline void LaunchMatmul(uint64_t singleCoreRealN, uint64_t cOffset, uint64_t aOffset);
    __aicore__ inline void InitVector1Tensor();
    __aicore__ inline void InitVector3ToV5Tensor();
    __aicore__ inline void InitGlobalBuffer(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR bias, GM_ADDR y, GM_ADDR workspace);

    __aicore__ inline void InitScaleFactor()
    {
        if (IsSameType<wType, int4b_t>::value) {
            multiScaleTimes_ = 3;  // int4场景，3:展开3次
            multiFactor1_ = 7.49;  // 7.49：使得round(7.49+0.5)<8
            multiFactor2_ = 14.98; // 14.98: 7.49*2，在展开式中使用
        }
    };

    __aicore__ inline void NotifyCube()
    {
        if ASCEND_IS_AIC {
            return;
        }
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
    }

    __aicore__ inline void NotifyVector()
    {
        if ASCEND_IS_AIV {
            return;
        }
        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
    }

    __aicore__ inline void WaitForVector()
    {
        if ASCEND_IS_AIV {
            return;
        }

        CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
    }

    __aicore__ inline void WaitForCube()
    {
        if ASCEND_IS_AIC {
            return;
        }
        CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
    }

    __aicore__ inline void NotifyCubePing()
    {
        if ASCEND_IS_AIC {
            return;
        }
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_VECTOR_CUBE_P_FLAG);
    }

    __aicore__ inline void WaitForVectorPing()
    {
        if ASCEND_IS_AIV {
            return;
        }
        CrossCoreWaitFlag(SYNC_VECTOR_CUBE_P_FLAG);
    }

    __aicore__ inline void NotifyCubePong()
    {
        if ASCEND_IS_AIC {
            return;
        }
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_VECTOR_CUBE_Q_FLAG);
    }

    __aicore__ inline void WaitForVectorPong()
    {
        if ASCEND_IS_AIV {
            return;
        }
        CrossCoreWaitFlag(SYNC_VECTOR_CUBE_Q_FLAG);
    }

    using InputXType = MatmulType<TPosition::A1, CubeFormat::NZ, wType, aTrans>;
    using InputWType = MatmulType<TPosition::B1, CubeFormat::NZ, wType, bTrans>;
    using OutputYType = MatmulType<TPosition::GM, CubeFormat::ND, preciseType>;
    using InputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, preciseType>;
    MatmulImpl<InputXType, InputWType, OutputYType, InputBiasType> mmObj;

    TPipe *pipe_;
    const Mc2WeightQuantBatchMatmulV2MsdGroupTilingData *tiling_;

    // int8系数默认取127,254，且只需要展开两次
    float multiFactor1_ = 127;
    float multiFactor2_ = 254;
    int32_t multiScaleTimes_ = 2;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<wType> aUnfoldGlobal_;
    GlobalTensor<int8_t> aUnfoldGlobalS8_;
    GlobalTensor<float> workspaceAtomicGlobal_;
    GlobalTensor<preciseType> workspaceCGlobal_;
    GlobalTensor<int8_t> wGlobal_;
    GlobalTensor<xType> antiQuantOffsetGlobal_;
    GlobalTensor<xType> antiQuantScaleGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<uint64_t> quantScaleGlobal_;
    GlobalTensor<yType> yGlobal_;

    GlobalTensor<float> workspaceReduceSumBrc_;
    GlobalTensor<float> workspaceReduceMaxBrc_;

    // vector1
    LocalTensor<xType> aBF16Tensor_;
    LocalTensor<float> aF32Tensor_;
    LocalTensor<float> aF32ReduceSumBrcTensor_;
    LocalTensor<float> aF32MultiScaleTensor_;
    LocalTensor<float> aF32ReduceMaxBrcTensor_;
    LocalTensor<half> aF16Tensor_;
    LocalTensor<wType> aUnfoldLocal_;

    // vector3
    LocalTensor<xType> scaleB16Tensor_;
    LocalTensor<xType> offsetB16Tensor_;
    LocalTensor<float> scaleF32Tensor_;
    LocalTensor<float> offsetF32Tensor_;
    LocalTensor<float> offsetMnF32_;
    LocalTensor<float> aF32ReduceSumTensorV3_;

    // vector4
    LocalTensor<preciseType> cTensor_;
    LocalTensor<half> cF16Tensor_;

    LocalTensor<float> cF32CastTensor_;
    LocalTensor<float> aF32MaxTensor_;

    // vector5
    LocalTensor<float> resF32Tensor_;
    LocalTensor<yType> resF16Tensor_;
    LocalTensor<biasType> biasTensor_;
    LocalTensor<float> biasFp32Tensor_;

    TBuf<> tBuf_;
    TBuf<TPosition::A1> a1Tbuf_;
    TQue<QuePosition::B1, DOUBLE_BUFFER_NUM> inQueueBL1_;

    uint32_t groupNum_ = 1;

    int32_t curBlockIdx_;
    int32_t vecNDimIdx_;
    int32_t vecKDimIdx_;

    TEventID vector1EventIdVToMte3 = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    TEventID vector3EventIdVToMTE21 = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    TEventID vector3EventIdVToMTE22 = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::InitGlobalBuffer(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                   GM_ADDR bias, GM_ADDR y, GM_ADDR workspace)
{
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(x), tiling_->mSize * tiling_->kSize);
    wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(weight), tiling_->kSize * tiling_->nSize);
    antiQuantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantScale), tiling_->nSize);
    antiQuantOffsetGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantOffset), tiling_->nSize);
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType *>(y), tiling_->mSize * tiling_->nSize);
    if (tiling_->hasBias) {
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling_->nSize);
    }

    // int4场景展开4份，但每份的size减半，总体空间占用等于2份的int8
    uint64_t aUnfoldSize =
        CeilDiv(2 * tiling_->mSize * tiling_->kSize, GM_ADDR_ALIGN_BASIC_BLOCK) * GM_ADDR_ALIGN_BASIC_BLOCK;
    uint64_t atomicAddSize =
        CeilDiv(tiling_->mSize * tiling_->nSize * sizeof(float), GM_ADDR_ALIGN_BASIC_BLOCK) * GM_ADDR_ALIGN_BASIC_BLOCK;
    uint64_t reduceSumSize =
        CeilDiv(tiling_->mSize * groupNum_ * GROUP_DIM, GM_ADDR_ALIGN_BASIC_BLOCK) * GM_ADDR_ALIGN_BASIC_BLOCK;

    aUnfoldGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ wType *>(workspace));
    aUnfoldGlobalS8_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspace));
    workspaceReduceSumBrc_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + aUnfoldSize), reduceSumSize);
    workspaceReduceMaxBrc_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace + aUnfoldSize + reduceSumSize),
                                           reduceSumSize);
    workspaceAtomicGlobal_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(workspace + aUnfoldSize + 2 * reduceSumSize), atomicAddSize);

    // reduce sum和reduce max各需要占用一份reduceSumSize的空间
    workspaceCGlobal_.SetGlobalBuffer(
        reinterpret_cast<__gm__ preciseType *>(workspace + aUnfoldSize + atomicAddSize + 2 * reduceSumSize),
        multiScaleTimes_ * tiling_->mSize * tiling_->nSize * DOUBLE_BUFFER_NUM * sizeof(preciseType));

    InitAtomicAddr(workspaceAtomicGlobal_, atomicAddSize, curBlockIdx_);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
                       GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                       const Mc2WeightQuantBatchMatmulV2MsdGroupTilingData *tilingData, TPipe *tPipe)
{
    tiling_ = tilingData;
    pipe_ = tPipe;
    curBlockIdx_ = GetBlockIdx();
    groupNum_ = CeilDiv(tiling_->kSize, tiling_->groupSize);
    InitScaleFactor();

    if ASCEND_IS_AIC {
        mmObj.SetSubBlockIdx(0);
        mmObj.Init(&tiling_->matmulTiling, tPipe);
        if (IsSameType<preciseType, HighPerformanceType>::value) {
            mmObj.SetQuantScalar(0x3F800000); // 设置QuantScalar为1.0
        }
    }

    InitGlobalBuffer(x, weight, antiquantScale, antiquantOffset, bias, y, workspace);

    pipe_->InitBuffer(tBuf_, 192 * 1024 - 256);
    uint32_t mAlignSize = CeilDiv(tiling_->mSize * multiScaleTimes_, GROUP_DIM) * GROUP_DIM;
    if (IsSameType<wType, int4b_t>::value) {
        pipe_->InitBuffer(a1Tbuf_, mAlignSize * tiling_->singleCoreK >> 1);
        pipe_->InitBuffer(inQueueBL1_, 2, tiling_->matmulTiling.singleCoreN * tiling_->groupSize >> 1);
    } else {
        pipe_->InitBuffer(a1Tbuf_, mAlignSize * tiling_->singleCoreK);
        pipe_->InitBuffer(inQueueBL1_, 2, tiling_->matmulTiling.singleCoreN * tiling_->groupSize);
    }

    InitVector1Tensor();
    InitVector3ToV5Tensor();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatMulV2MsdGroupKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                          hasAntiQuantOffset, quantType, weightFormat, preciseType>::InitVector1Tensor()
{
    aF32ReduceSumBrcTensor_ = tBuf_.Get<float>();                           // 0-7k分配给sum的broadCast结果
    aF32ReduceMaxBrcTensor_ = tBuf_.Get<float>()[7 * FLOAT_DATA_BENCHMARK]; // 7-14k分配给max的broadCast结果
    aBF16Tensor_ = tBuf_.Get<xType>()[14 * HALF_DATA_BENCHMARK];            // 14-42k分配给xType的a矩阵
    aF16Tensor_ = tBuf_.Get<half>()[14 * HALF_DATA_BENCHMARK];              // 14-42k分配给half的a矩阵
    aF32Tensor_ = tBuf_.Get<float>()[42 * FLOAT_DATA_BENCHMARK];            // 42-98k分配给float的a矩阵
    aF32MultiScaleTensor_ = tBuf_.Get<float>()[98 * FLOAT_DATA_BENCHMARK];  // 98-154k分配给float的a矩阵
    if (IsSameType<wType, int4b_t>::value) {
        aUnfoldLocal_ = tBuf_.Get<wType>()[154 * INT4_DATA_BENCHMARK]; // 156-184k分配给展开的a矩阵
    } else {
        aUnfoldLocal_ = tBuf_.Get<wType>()[154 * INT8_DATA_BENCHMARK]; // 156-184k分配给展开的a矩阵
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                                 antiQuantType, hasAntiQuantOffset, quantType,
                                                                 weightFormat, preciseType>::InitVector3ToV5Tensor()
{
    uint32_t offset = 0;
    uint32_t elements = 0;
    // vector3 空间重新分配
    offset = 0;
    elements = tiling_->vecSingleCoreN;
    scaleB16Tensor_ = tBuf_.GetWithOffset<xType>(elements, offset);

    offset += elements * sizeof(xType);
    offsetB16Tensor_ = tBuf_.GetWithOffset<xType>(elements, offset);

    offset += elements * sizeof(xType);
    scaleF32Tensor_ = tBuf_.GetWithOffset<float>(elements, offset);

    offset += elements * sizeof(float);
    offsetF32Tensor_ = tBuf_.GetWithOffset<float>(elements, offset);

    offset += elements * sizeof(float);
    elements = tiling_->mSize * tiling_->vecSingleCoreN;
    offsetMnF32_ = tBuf_.GetWithOffset<float>(elements, offset);

    offset += elements * sizeof(float);
    elements = tiling_->mSize * 8;
    aF32ReduceSumTensorV3_ = tBuf_.GetWithOffset<float>(elements, offset);

    // vector4
    offset += elements * sizeof(float);
    elements = tiling_->mSize * tiling_->vecSingleCoreN * multiScaleTimes_;

    if (IsSameType<preciseType, HighPerformanceType>::value) {
        cTensor_ = tBuf_.Get<preciseType>()[80 * HALF_DATA_BENCHMARK];
        cF16Tensor_ = tBuf_.Get<half>()[80 * HALF_DATA_BENCHMARK]; // 高性能模式存fp16 tensor，内存空间与cTensor_一致
        cF32CastTensor_ = tBuf_.Get<float>()[112 * FLOAT_DATA_BENCHMARK];

    } else {
        cTensor_ = tBuf_.Get<preciseType>()[80 * FLOAT_DATA_BENCHMARK];
        cF32CastTensor_ = tBuf_.Get<float>()[80 * FLOAT_DATA_BENCHMARK];
    }

    aF32MaxTensor_ = tBuf_.Get<float>()[184 * FLOAT_DATA_BENCHMARK]; // 184k以后存在aF32Max结果

    // vector5
    resF16Tensor_ = tBuf_.Get<yType>();                            // 0k - 32k存yType类型的y矩阵
    resF32Tensor_ = tBuf_.Get<float>()[32 * FLOAT_DATA_BENCHMARK]; // 32k - 96k存float类型的y矩阵

    if constexpr (IsSameType<biasType, float>::value) {
        biasTensor_ = tBuf_.Get<biasType>()[96 * FLOAT_DATA_BENCHMARK];  // 96k之后存bias类型的y矩阵
        biasFp32Tensor_ = tBuf_.Get<float>()[96 * FLOAT_DATA_BENCHMARK]; // float类型无需重新申请空间存bias
    } else {
        biasTensor_ = tBuf_.Get<biasType>()[96 * HALF_DATA_BENCHMARK];   // 96k - 98k存bias类型的y矩阵
        biasFp32Tensor_ = tBuf_.Get<float>()[98 * FLOAT_DATA_BENCHMARK]; // 98k之后存float类型的y矩阵
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyInBL1(uint64_t wOffset, uint64_t singleCoreRealN, Nd2NzParams &nd2nzParams)
{
    if (IsSameType<wType, int4b_t>::value) {
        nd2nzParams.dValue = singleCoreRealN >> 1;
        nd2nzParams.srcDValue = tiling_->nSize >> 1;
        wOffset = wOffset >> 1;
    } else {
        nd2nzParams.dValue = singleCoreRealN;
        nd2nzParams.srcDValue = tiling_->nSize;
    }
    nd2nzParams.dstNzC0Stride = CeilDiv(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE)) * BLOCK_CUBE;

    // int8/int4场景，nd2nz都只能按照int8形式搬运
    LocalTensor<int8_t> bL1Tensor = inQueueBL1_.AllocTensor<int8_t>();
    DataCopy(bL1Tensor, wGlobal_[wOffset], nd2nzParams);
    inQueueBL1_.EnQue(bL1Tensor);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyInBL1Nz(const LocalTensor<int8_t> &bL1Tensor, uint64_t kOffset, uint64_t nOffset,
                              uint64_t l1BaseKb, uint64_t singleCoreRealN)
{
    uint64_t tilingKSizeNz = CeilAlign(tiling_->kSize, 16UL);

    uint64_t wOffset = nOffset * tilingKSizeNz + kOffset * INT4_BLOCK_SIZE;
    DataCopyParams dmaParams;
    dmaParams.blockCount = singleCoreRealN / INT4_BLOCK_SIZE;
    dmaParams.blockLen = l1BaseKb;
    dmaParams.srcStride = tiling_->kSize - l1BaseKb;
    dmaParams.dstStride = 0;
    wOffset = wOffset >> 1;
    DataCopy(bL1Tensor, wGlobal_[wOffset], dmaParams);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatMulV2MsdGroupKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                          hasAntiQuantOffset, quantType, weightFormat,
                                          preciseType>::CopyInAL1(uint64_t realSingleCoreK, uint64_t kOffset)
{
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = tiling_->mSize * multiScaleTimes_;
    if (IsSameType<wType, int4b_t>::value) {
        nd2nzParams.dValue = realSingleCoreK >> 1;
        nd2nzParams.srcDValue = tiling_->kSize >> 1;
    } else {
        nd2nzParams.dValue = realSingleCoreK;
        nd2nzParams.srcDValue = tiling_->kSize;
    }
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = CeilDiv(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE)) * BLOCK_CUBE;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    LocalTensor<int8_t> aL1Tensor = a1Tbuf_.template Get<int8_t>();

    DataCopy(aL1Tensor, aUnfoldGlobalS8_[kOffset], nd2nzParams);

    TEventID eventIdMte2ToMte1 = GetTPipePtr()->FetchEventID<HardEvent::MTE2_MTE1>();
    SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
    WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                                 antiQuantType, hasAntiQuantOffset, quantType,
                                                                 weightFormat, preciseType>::CopyInVector1(uint32_t m,
                                                                                                           uint32_t k)
{
    if (curBlockIdx_ >= tiling_->mSize) {
        return;
    }

    DataCopyPad2D(aBF16Tensor_, xGlobal_[curBlockIdx_ * k], m, k, tiling_->kSize);
    TEventID eventIdVToMte2 = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();
    SetFlag<HardEvent::MTE2_V>(eventIdVToMte2);
    WaitFlag<HardEvent::MTE2_V>(eventIdVToMte2);
    Cast(aF32Tensor_, aBF16Tensor_, RoundMode::CAST_NONE, m * k);
    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                                 antiQuantType, hasAntiQuantOffset, quantType,
                                                                 weightFormat, preciseType>::ProcessVector1(uint32_t m,
                                                                                                            uint32_t k)
{
    if (curBlockIdx_ >= tiling_->mSize) {
        return;
    }
    BinaryRepeatParams maxAddReatParams;
    maxAddReatParams.dstBlkStride = 1;
    maxAddReatParams.src0BlkStride = 1;
    maxAddReatParams.src1BlkStride = 1;
    maxAddReatParams.dstRepStride = 16;
    maxAddReatParams.src0RepStride = 16;
    maxAddReatParams.src1RepStride = 16;

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = groupNum_ - 1;
    dataCopyParams.blockLen = 8;
    dataCopyParams.srcStride = 8;
    dataCopyParams.dstStride = 0;

    BrcbRepeatParams brcbRepeatParams;
    brcbRepeatParams.dstBlkStride = 1;
    brcbRepeatParams.dstRepStride = 8;

    if constexpr (hasAntiQuantOffset) {
        ComputeAMatrixSum(m, k, maxAddReatParams, dataCopyParams, brcbRepeatParams);
        PipeBarrier<PIPE_V>();
    }

    ComputeAMatrixMax(m, k, maxAddReatParams, dataCopyParams, brcbRepeatParams);
    PipeBarrier<PIPE_V>();

    UnfoldAMatrix(m, k);
    SetFlag<HardEvent::V_MTE3>(vector1EventIdVToMte3);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ComputeAMatrixSum(uint32_t m, uint32_t k, BinaryRepeatParams &maxAddReatParams,
                                    DataCopyParams &dataCopyParams, BrcbRepeatParams &brcbRepeatParams)
{
    if (tiling_->groupSize == 128) {
        // groupSize为128的场景，先将128 reduce 至64(即FP32_MAX_MASK_SIZE)
        Add(aF32MultiScaleTensor_, aF32Tensor_, aF32Tensor_[FP32_MAX_MASK_SIZE], FP32_MAX_MASK_SIZE, groupNum_,
            maxAddReatParams);
        PipeBarrier<PIPE_V>();
        if (groupNum_ > 1) {
            DataCopy(aF32MultiScaleTensor_[FP32_MAX_MASK_SIZE], aF32MultiScaleTensor_[tiling_->groupSize],
                     dataCopyParams);
            PipeBarrier<PIPE_V>();
        }
        BlockReduceSum(aF32MultiScaleTensor_, aF32MultiScaleTensor_, groupNum_, FP32_MAX_MASK_SIZE, 1, 1,
                       VEC_REPEAT_MAX_STRIDE);

    } else {
        BlockReduceSum(aF32MultiScaleTensor_, aF32Tensor_, groupNum_, FP32_MAX_MASK_SIZE, 1, 1, VEC_REPEAT_MAX_STRIDE);
    }

    PipeBarrier<PIPE_V>();
    BlockReduceSum(aF32MultiScaleTensor_, aF32MultiScaleTensor_, CeilDiv(groupNum_, FP32_BLOCK_SIZE),
                   FP32_MAX_MASK_SIZE, 1, 1, VEC_REPEAT_MAX_STRIDE);

    PipeBarrier<PIPE_V>();
    Brcb(aF32ReduceSumBrcTensor_, aF32MultiScaleTensor_, CeilDiv(groupNum_, FP32_BLOCK_SIZE), brcbRepeatParams);

    TEventID eventIdVToMte3V1 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3V1);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3V1);
    DataCopyPad2D(workspaceReduceSumBrc_[curBlockIdx_ * groupNum_ * 8], aF32ReduceSumBrcTensor_, 1, groupNum_ * 8,
                  groupNum_ * 8);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ComputeAMatrixMax(uint32_t m, uint32_t k, BinaryRepeatParams &maxAddReatParams,
                                    DataCopyParams &dataCopyParams, BrcbRepeatParams &brcbRepeatParams)
{
    Abs(aF32MultiScaleTensor_, aF32Tensor_, m * k);
    PipeBarrier<PIPE_V>();
    if (tiling_->groupSize == 128) {
        // groupSize为128的场景，先将128 reduce 至64(即FP32_MAX_MASK_SIZE)
        Max(aF32MultiScaleTensor_, aF32MultiScaleTensor_, aF32MultiScaleTensor_[FP32_MAX_MASK_SIZE], FP32_MAX_MASK_SIZE,
            groupNum_, maxAddReatParams);
        PipeBarrier<PIPE_V>();
        if (groupNum_ > 1) {
            DataCopy(aF32MultiScaleTensor_[FP32_MAX_MASK_SIZE], aF32MultiScaleTensor_[tiling_->groupSize],
                     dataCopyParams);
            PipeBarrier<PIPE_V>();
        }
    }
    BlockReduceMax(aF32MultiScaleTensor_, aF32MultiScaleTensor_, groupNum_, FP32_MAX_MASK_SIZE, 1, 1,
                   VEC_REPEAT_MAX_STRIDE);
    PipeBarrier<PIPE_V>();
    BlockReduceMax(aF32MultiScaleTensor_, aF32MultiScaleTensor_, CeilDiv(groupNum_, FP32_BLOCK_SIZE),
                   FP32_MAX_MASK_SIZE, 1, 1, VEC_REPEAT_MAX_STRIDE);
    PipeBarrier<PIPE_V>();
    Brcb(aF32ReduceMaxBrcTensor_, aF32MultiScaleTensor_, CeilDiv(groupNum_, FP32_BLOCK_SIZE), brcbRepeatParams);
    PipeBarrier<PIPE_V>();

    TEventID eventIdVToMte3V1 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3V1);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3V1);
    DataCopyPad2D(workspaceReduceMaxBrc_[curBlockIdx_ * groupNum_ * FP32_BLOCK_SIZE], aF32ReduceMaxBrcTensor_, 1,
                  groupNum_ * FP32_BLOCK_SIZE, groupNum_ * FP32_BLOCK_SIZE);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                                 antiQuantType, hasAntiQuantOffset, quantType,
                                                                 weightFormat, preciseType>::UnfoldAMatrix(uint32_t m,
                                                                                                           uint32_t k)
{
    uint32_t mainRepeatK = k / tiling_->groupSize;
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = tiling_->groupSize / 8; // float类型，8个block构成一个repeat
    repeatParams.src0RepStride = repeatParams.dstRepStride;
    repeatParams.src1RepStride = 1;
    // (m, k) / (m, 8)
    Div(aF32MultiScaleTensor_, aF32Tensor_, aF32ReduceMaxBrcTensor_, 256 / sizeof(float), mainRepeatK, repeatParams);
    if (tiling_->groupSize == 128) {
        Div(aF32MultiScaleTensor_[64], aF32Tensor_[64], aF32ReduceMaxBrcTensor_, 256 / sizeof(float), mainRepeatK,
            repeatParams);
    }

    uint32_t mK = m * k;

    PipeBarrier<PIPE_V>();
    Muls(aF32MultiScaleTensor_, aF32MultiScaleTensor_, multiFactor1_, mK);
    PipeBarrier<PIPE_V>();
    Cast(aF32Tensor_, aF32MultiScaleTensor_, RoundMode::CAST_ROUND, mK);
    PipeBarrier<PIPE_V>();
    Cast(aF16Tensor_, aF32Tensor_, RoundMode::CAST_NONE, mK);
    PipeBarrier<PIPE_V>();
    Cast(aUnfoldLocal_, aF16Tensor_, RoundMode::CAST_NONE, mK); // A1

    for (uint64_t transATimes = 1; transATimes < multiScaleTimes_; transATimes++) {
        Sub(aF32MultiScaleTensor_, aF32MultiScaleTensor_, aF32Tensor_, mK);
        PipeBarrier<PIPE_V>();
        Muls(aF32MultiScaleTensor_, aF32MultiScaleTensor_, static_cast<float>(multiFactor2_), mK);
        PipeBarrier<PIPE_V>();
        Cast(aF32Tensor_, aF32MultiScaleTensor_, RoundMode::CAST_ROUND, mK);
        PipeBarrier<PIPE_V>();
        Cast(aF16Tensor_, aF32Tensor_, RoundMode::CAST_NONE, mK);
        PipeBarrier<PIPE_V>();
        Cast(aUnfoldLocal_[transATimes * mK], aF16Tensor_, RoundMode::CAST_NONE, mK);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                                 antiQuantType, hasAntiQuantOffset, quantType,
                                                                 weightFormat, preciseType>::CopyOutVector1(uint32_t m,
                                                                                                            uint32_t k)
{
    if (curBlockIdx_ >= tiling_->mSize) {
        return;
    }
    WaitFlag<HardEvent::V_MTE3>(vector1EventIdVToMte3);
    DataCopyPad2D(aUnfoldGlobal_[curBlockIdx_ * multiScaleTimes_ * m * tiling_->kSize], aUnfoldLocal_,
                  multiScaleTimes_ * m, k, tiling_->kSize);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::LaunchMatmul(uint64_t singleCoreRealN, uint64_t cOffset, uint64_t kOffset)
{
    LocalTensor<wType> aL1Tensor = a1Tbuf_.template Get<wType>();
    LocalTensor<wType> bL1Tensor = inQueueBL1_.DeQue<wType>();

    mmObj.SetTensorA(
        aL1Tensor[CeilDiv(multiScaleTimes_ * tiling_->mSize, static_cast<uint64_t>(BLOCK_CUBE)) * BLOCK_CUBE * kOffset],
        aTrans);
    mmObj.SetTensorB(bL1Tensor, bTrans);

    mmObj.SetOrgShape(multiScaleTimes_ * tiling_->mSize, tiling_->nSize, tiling_->kSize, tiling_->kSize,
                      tiling_->nSize);
    mmObj.SetTail(multiScaleTimes_ * tiling_->mSize, singleCoreRealN, tiling_->groupSize);
    mmObj.IterateAll(workspaceCGlobal_[cOffset]);
    inQueueBL1_.FreeTensor(bL1Tensor);
    mmObj.End();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyInVector3(int32_t vecSingleCoreN, uint32_t groupIdx, uint32_t nBaseOffset, uint32_t taskId)
{
    uint64_t globalOffset = groupIdx * tiling_->nSize + nBaseOffset;
    if (taskId > 0) {
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }

    DataCopyPad2D(scaleB16Tensor_, antiQuantScaleGlobal_[globalOffset], 1, vecSingleCoreN, tiling_->nSize);
    if constexpr (hasAntiQuantOffset) {
        DataCopyPad2D(offsetB16Tensor_, antiQuantOffsetGlobal_[globalOffset], 1, vecSingleCoreN, tiling_->nSize);
    }

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    Cast(scaleF32Tensor_, scaleB16Tensor_, RoundMode::CAST_NONE, vecSingleCoreN);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::MulOffsetVector3(int32_t vecSingleCoreN, uint32_t groupIdx, uint32_t taskId)
{
    uint32_t mainRepeatN = vecSingleCoreN / FP32_MAX_MASK_SIZE;
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 8;
    repeatParams.src0RepStride = 0;
    repeatParams.src1RepStride = 8;

    if (taskId > 0) {
        for (uint32_t idxM = 0; idxM < tiling_->mSize; idxM++) {
            MulAddDst(offsetMnF32_[idxM * vecSingleCoreN], aF32ReduceSumTensorV3_[idxM * 8], offsetF32Tensor_,
                      256 / sizeof(float), mainRepeatN, repeatParams);
        }
    } else {
        for (uint32_t idxM = 0; idxM < tiling_->mSize; idxM++) {
            Mul(offsetMnF32_[idxM * vecSingleCoreN], aF32ReduceSumTensorV3_[idxM * 8], offsetF32Tensor_,
                256 / sizeof(float), mainRepeatN, repeatParams);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ProcessVector3(int32_t vecSingleCoreN, uint32_t groupIdx, uint32_t nBaseOffset, uint32_t groupLimit,
                                 uint32_t taskId)
{
    if constexpr (!hasAntiQuantOffset) { // 无offset 无需后续流程
        return;
    }

    DataCopyPad2D(aF32ReduceSumTensorV3_, workspaceReduceSumBrc_[groupIdx * 8], tiling_->mSize, 8, groupNum_ * 8);

    Cast(offsetF32Tensor_, offsetB16Tensor_, RoundMode::CAST_NONE, vecSingleCoreN);
    PipeBarrier<PIPE_V>();
    Mul(offsetF32Tensor_, scaleF32Tensor_, offsetF32Tensor_, vecSingleCoreN);

    TEventID eventIdMte2ToVV3 = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVV3);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVV3);

    // (m, 8) (1, n) -> (m, n)
    PipeBarrier<PIPE_V>();
    MulOffsetVector3(vecSingleCoreN, groupIdx, taskId);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ProcessVector4MaxScale(int64_t vecSingleCoreN, uint64_t groupIdx)
{
    DataCopyPad2D(aF32MaxTensor_, workspaceReduceMaxBrc_[groupIdx * 8], tiling_->mSize, 8, groupNum_ * 8);

    uint32_t mainRepeatN = vecSingleCoreN / FP32_MAX_MASK_SIZE;
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 8;
    repeatParams.src0RepStride = 8;
    repeatParams.src1RepStride = 8;

    for (uint32_t idxM = 0; idxM < tiling_->mSize; idxM++) {
        Mul(cF32CastTensor_[idxM * vecSingleCoreN], cF32CastTensor_[idxM * vecSingleCoreN], scaleF32Tensor_,
            256 / sizeof(float), mainRepeatN, repeatParams);
    }

    TEventID eventIdMte2ToVInV4 = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVInV4);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVInV4);

    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = 8;
    repeatParams.src0RepStride = 8;
    repeatParams.src1RepStride = 0;
    PipeBarrier<PIPE_V>();
    for (uint32_t idxM = 0; idxM < tiling_->mSize; idxM++) {
        Mul(cF32CastTensor_[idxM * vecSingleCoreN], cF32CastTensor_[idxM * vecSingleCoreN], aF32MaxTensor_[idxM * 8],
            256 / sizeof(float), mainRepeatN, repeatParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyInAndProcessVector4(int64_t vecSingleCoreN, uint64_t groupIdx, uint64_t nOffset, uint32_t taskId,
                                          uint32_t groupLimit)
{
    if (vecSingleCoreN <= 0) {
        if (groupIdx + 2 < groupLimit) {
            if (groupIdx % 2 == 0) {
                NotifyCubePing();
            } else {
                NotifyCubePong();
            }
        }
        return;
    }

    uint64_t cOffset = vecKDimIdx_ * 2 * multiScaleTimes_ * tiling_->mSize * tiling_->nSize +
                       (groupIdx % 2) * multiScaleTimes_ * tiling_->mSize * tiling_->nSize +
                       vecNDimIdx_ * tiling_->vecSingleCoreN;

    uint64_t v4BaseM = 0;
    uint64_t v4Size = tiling_->mSize * vecSingleCoreN;
    // 高性能模式计算
    if constexpr (IsSameType<preciseType, HighPerformanceType>::value) {
        ProcessMatmulResultHalf(cOffset, tiling_->mSize, vecSingleCoreN);
    } else {
        // 当前切分第一次可以处理8*n的c矩阵
        uint64_t v4EndM1 = (tiling_->mSize > 8) ? 8 : tiling_->mSize;
        ProcessMatmulResult(cOffset, v4BaseM, v4EndM1, vecSingleCoreN);
        // 当前切分第二次可以处理6*n的c矩阵，两次为14
        uint64_t v4EndM2 = (tiling_->mSize > 14) ? 14 : tiling_->mSize;
        ProcessMatmulResult(cOffset, v4EndM1, v4EndM2, vecSingleCoreN);
        uint64_t v4EndM3 = tiling_->mSize;
        ProcessMatmulResult(cOffset, v4EndM2, v4EndM3, vecSingleCoreN);
    }

    // c矩阵已经消费完毕，通知cube开始下一轮计算
    if (groupIdx + 2 < groupLimit) {
        if (groupIdx % 2 == 0) {
            NotifyCubePing();
        } else {
            NotifyCubePong();
        }
    }

    PipeBarrier<PIPE_V>();
    ProcessVector4MaxScale(vecSingleCoreN, groupIdx);
    PipeBarrier<PIPE_V>();

    if (!hasAntiQuantOffset && taskId == 0) { // 无AntiQuantOffset处理流程
        DataCopy(offsetMnF32_, cF32CastTensor_, v4Size);
    } else {
        Add(offsetMnF32_, offsetMnF32_, cF32CastTensor_, v4Size);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ProcessMatmulResultHalf(uint64_t cOffset, uint64_t v4EndM, int64_t vecSingleCoreN)
{
    uint64_t v4TailCSize = v4EndM * vecSingleCoreN;
    uint64_t mOffset = v4TailCSize;
    DataCopyPad2D(cTensor_, workspaceCGlobal_[cOffset], v4EndM, vecSingleCoreN, multiScaleTimes_ * tiling_->nSize);
    DataCopyPad2D(cTensor_[mOffset], workspaceCGlobal_[cOffset + tiling_->nSize], v4EndM, vecSingleCoreN,
                  multiScaleTimes_ * tiling_->nSize);
    float multiFactor = multiFactor2_;
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    Axpy(cF16Tensor_, cF16Tensor_[mOffset], static_cast<half>(1.0f / multiFactor), v4TailCSize);
    mOffset += v4TailCSize;
    DataCopyPad2D(cTensor_[mOffset], workspaceCGlobal_[cOffset + 2 * tiling_->nSize], v4EndM, vecSingleCoreN,
                  multiScaleTimes_ * tiling_->nSize);
    multiFactor *= multiFactor2_;
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    Axpy(cF16Tensor_, cF16Tensor_[mOffset], static_cast<half>(1.0f / multiFactor), v4TailCSize);
    PipeBarrier<PIPE_V>();
    Cast(cF32CastTensor_, cF16Tensor_, RoundMode::CAST_NONE, v4TailCSize);
    PipeBarrier<PIPE_V>();
    Muls(cF32CastTensor_, cF32CastTensor_, 1.0f / multiFactor1_, v4TailCSize);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ProcessMatmulResult(uint64_t cOffset, uint64_t v4BaseM, uint64_t v4EndM, int64_t vecSingleCoreN)
{
    uint64_t tailM = v4EndM - v4BaseM;
    if (tailM <= 0) {
        return;
    }
    uint64_t v4BaseCSize = v4BaseM * vecSingleCoreN;
    uint64_t v4TailCSize = tailM * vecSingleCoreN;

    float multiFactor = multiFactor1_;
    for (uint32_t multiScaleTimeIdx = 0; multiScaleTimeIdx < multiScaleTimes_; multiScaleTimeIdx++) {
        uint64_t mOffset = v4BaseCSize + multiScaleTimeIdx * v4TailCSize;

        if (v4BaseM != 0) {
            if (multiScaleTimeIdx == 0) {
                WaitFlag<HardEvent::V_MTE2>(vector3EventIdVToMTE21);
            } else if (multiScaleTimeIdx == 1) {
                WaitFlag<HardEvent::V_MTE2>(vector3EventIdVToMTE22);
            }
        }
        DataCopyPad2D(cTensor_[mOffset],
                      workspaceCGlobal_[cOffset + (v4BaseM * multiScaleTimes_ + multiScaleTimeIdx) * tiling_->nSize],
                      tailM, vecSingleCoreN, multiScaleTimes_ * tiling_->nSize);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Cast(cF32CastTensor_[mOffset], cTensor_[mOffset], RoundMode::CAST_NONE, v4TailCSize);
        PipeBarrier<PIPE_V>();
        if (multiScaleTimeIdx == 0) {
            Muls(cF32CastTensor_[mOffset], cF32CastTensor_[mOffset], static_cast<float>(1.0) / multiFactor,
                 v4TailCSize);
        } else {
            Axpy(cF32CastTensor_[v4BaseCSize], cF32CastTensor_[mOffset], static_cast<float>(1.0) / multiFactor,
                 v4TailCSize);
        }

        multiFactor *= multiFactor2_;
        if (v4EndM != tiling_->mSize) {
            if (multiScaleTimeIdx == 1) {
                SetFlag<HardEvent::V_MTE2>(vector3EventIdVToMTE21);
            } else if (multiScaleTimeIdx == 2) {
                SetFlag<HardEvent::V_MTE2>(vector3EventIdVToMTE22);
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatMulV2MsdGroupKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyOutVector4(int32_t singleCoreRealN, uint32_t groupIdx, uint32_t groupLimit)
{
    if (singleCoreRealN <= 0) {
        return;
    }
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    uint64_t atomicWorkspaceOffset = vecNDimIdx_ * tiling_->vecSingleCoreN;
    SetAtomicAdd<float>();
#if defined(__CCE_KT_TEST__)
    bool isUsedProcessLock = false;
    AscendC::ProcessLock::GetProcessLock()->Write();
    isUsedProcessLock = true;
#endif
    DataCopyPad2D(workspaceAtomicGlobal_[atomicWorkspaceOffset], offsetMnF32_, tiling_->mSize, singleCoreRealN,
                  tiling_->nSize);
#if defined(__CCE_KT_TEST__)
    if (isUsedProcessLock == true) {
        isUsedProcessLock = false;
        AscendC::ProcessLock::GetProcessLock()->Unlock();
    }
#endif
    SetAtomicNone();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatMulV2MsdGroupKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                          hasAntiQuantOffset, quantType, weightFormat,
                                          preciseType>::ProcessVector5(int32_t singleCoreRealN)
{
    uint64_t atomicWorkspaceOffset = vecNDimIdx_ * tiling_->vecSingleCoreN;
    DataCopyPad2D(resF32Tensor_, workspaceAtomicGlobal_[atomicWorkspaceOffset], tiling_->mSize, singleCoreRealN,
                  tiling_->nSize);
    if (tiling_->hasBias) {
        DataCopyPad2D(biasTensor_, biasGlobal_[atomicWorkspaceOffset], 1, singleCoreRealN, tiling_->nSize);
    }

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    if (tiling_->hasBias) {
        if constexpr (!IsSameType<biasType, float>::value) {
            Cast(biasFp32Tensor_, biasTensor_, RoundMode::CAST_NONE, singleCoreRealN);
            PipeBarrier<PIPE_V>();
        }

        uint64_t repeatTimes = singleCoreRealN / FP32_MAX_MASK_SIZE;
        BinaryRepeatParams repeatParams;
        repeatParams.dstRepStride = 8;
        repeatParams.src0BlkStride = 1;
        repeatParams.src0RepStride = 8;
        repeatParams.src1BlkStride = 1;
        repeatParams.src1RepStride = 8;

        Add(resF32Tensor_, resF32Tensor_, biasFp32Tensor_, FP32_MAX_MASK_SIZE, repeatTimes, repeatParams);
        for (uint64_t mIdx = 1; mIdx < tiling_->mSize; mIdx++) {
            AscendC::Add<float, false>(resF32Tensor_[mIdx * singleCoreRealN], resF32Tensor_[mIdx * singleCoreRealN],
                                       biasFp32Tensor_, FP32_MAX_MASK_SIZE, repeatTimes, repeatParams);
        }
        PipeBarrier<PIPE_V>();
    }

    if constexpr (IsSameType<yType, int8_t>::value) {
        return;
    } else {
        Cast(resF16Tensor_, resF32Tensor_, RoundMode::CAST_ROUND, tiling_->mSize * singleCoreRealN);
        PipeBarrier<PIPE_V>();
        uint64_t cGlobalOffset = atomicWorkspaceOffset;
        TEventID eventIdVToMte3V5 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3V5);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3V5);
        DataCopyPad2D(yGlobal_[cGlobalOffset], resF16Tensor_, tiling_->mSize, singleCoreRealN, tiling_->nSize);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatMulV2MsdGroupKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                          hasAntiQuantOffset, quantType, weightFormat, preciseType>::Process()
{
    if ASCEND_IS_AIV {
        CopyInVector1(tiling_->vec1SingleCoreM, tiling_->kSize);
        ProcessVector1(tiling_->vec1SingleCoreM, tiling_->kSize);
        CopyOutVector1(tiling_->vec1SingleCoreM, tiling_->kSize);
        SyncAll();
        NotifyCube();

        vecNDimIdx_ = curBlockIdx_ % tiling_->vecBlockDimN;
        vecKDimIdx_ = curBlockIdx_ / tiling_->vecBlockDimN;
        uint64_t kBaseOffset = vecKDimIdx_ * tiling_->singleCoreK;
        uint64_t nBaseOffset = vecNDimIdx_ * tiling_->vecSingleCoreN;
        int64_t singleCoreRealN = tiling_->vecSingleCoreN;
        if (nBaseOffset + singleCoreRealN > tiling_->nSize) {
            singleCoreRealN = tiling_->nSize - nBaseOffset;
        }

        uint64_t groupIdx = kBaseOffset / tiling_->groupSize;
        uint64_t groupLimit = groupIdx + tiling_->singleCoreGroup;
        if (kBaseOffset + tiling_->singleCoreK > tiling_->kSize) {
            groupLimit = tiling_->kSize / tiling_->groupSize;
        }
        uint32_t taskId = 0;
        for (; groupIdx < groupLimit; groupIdx++, taskId++) {
            if (curBlockIdx_ < tiling_->vecBlockDimN * tiling_->cubeBlockDimK && singleCoreRealN > 0) {
                CopyInVector3(singleCoreRealN, groupIdx, nBaseOffset, taskId);
                ProcessVector3(singleCoreRealN, groupIdx, nBaseOffset, groupLimit, taskId);
            }
            WaitForCube();
            if (curBlockIdx_ < tiling_->vecBlockDimN * tiling_->cubeBlockDimK) {
                CopyInAndProcessVector4(singleCoreRealN, groupIdx, nBaseOffset, taskId, groupLimit);
            }
        }
        if (curBlockIdx_ < tiling_->vecBlockDimN * tiling_->cubeBlockDimK && singleCoreRealN > 0) {
            CopyOutVector4(singleCoreRealN, groupIdx, groupLimit);
        }
        SyncAll();
        if (singleCoreRealN > 0) {
            ProcessVector5(singleCoreRealN);
        }
    } else {
        // mte2 流水
        uint64_t cubeKDimIdx_ = curBlockIdx_ / tiling_->cubeBlockDimN;
        uint64_t cubeNDimIdx_ = curBlockIdx_ % tiling_->cubeBlockDimN;
        uint64_t kBaseOffset = cubeKDimIdx_ * tiling_->singleCoreK;
        uint64_t nBaseOffset = cubeNDimIdx_ * tiling_->matmulTiling.singleCoreN;
        uint64_t singleCoreRealN = tiling_->matmulTiling.singleCoreN;
        if (nBaseOffset + singleCoreRealN > tiling_->nSize) {
            singleCoreRealN = tiling_->nSize - nBaseOffset;
        }
        uint64_t singleCoreRealK = tiling_->singleCoreK;
        uint64_t groupIdx = kBaseOffset / tiling_->groupSize;
        uint64_t groupLimit = groupIdx + tiling_->singleCoreGroup;
        if (kBaseOffset + tiling_->singleCoreK > tiling_->kSize) {
            groupLimit = tiling_->kSize / tiling_->groupSize;
            singleCoreRealK = tiling_->kSize - kBaseOffset;
        }
        uint64_t taskId = groupIdx;
        LocalTensor<int8_t> bL1Tensor = inQueueBL1_.AllocTensor<int8_t>();
        uint32_t taskLimit = groupIdx + 3 > groupLimit ? groupLimit : groupIdx + 3;
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = tiling_->groupSize;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        for (; taskId < taskLimit; taskId++) {
            uint64_t kOffset = taskId * tiling_->groupSize;
            if constexpr (weightFormat == CubeFormat::ND) {
                if (IsSameType<wType, int4b_t>::value) {
                    nd2nzParams.dValue = singleCoreRealN >> 1;
                    nd2nzParams.srcDValue = tiling_->nSize >> 1;
                } else {
                    nd2nzParams.dValue = singleCoreRealN;
                    nd2nzParams.srcDValue = tiling_->nSize;
                }
                nd2nzParams.dstNzC0Stride = CeilDiv(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE)) * BLOCK_CUBE;
                if (IsSameType<wType, int4b_t>::value) {
                    DataCopy(bL1Tensor, wGlobal_[(kOffset * tiling_->nSize + nBaseOffset) >> 1], nd2nzParams);
                } else {
                    DataCopy(bL1Tensor, wGlobal_[kOffset * tiling_->nSize + nBaseOffset], nd2nzParams);
                }
            } else {
                CopyInBL1Nz(bL1Tensor, kOffset, nBaseOffset, tiling_->groupSize, singleCoreRealN);
            }
        }
        inQueueBL1_.FreeTensor(bL1Tensor);
        taskId = 0;
        WaitForVector();

        // 多余的核不需要对al1数据读写
        if (likely(groupIdx < groupLimit)) {
            if (IsSameType<wType, int4b_t>::value) {
                CopyInAL1(singleCoreRealK, kBaseOffset >> 1);
            } else {
                CopyInAL1(singleCoreRealK, kBaseOffset);
            }
        }

        if constexpr (weightFormat == CubeFormat::NZ) {
            PipeBarrier<PIPE_MTE2>(); // 隔离preload和cube mte2的影响
        }
        for (; groupIdx < groupLimit; groupIdx++, taskId++) {
            uint64_t kOffset = groupIdx * tiling_->groupSize;
            if (curBlockIdx_ < tiling_->cubeBlockDimN * tiling_->cubeBlockDimK) {
                if constexpr (weightFormat == CubeFormat::ND) {
                    CopyInBL1(kOffset * tiling_->nSize + nBaseOffset, singleCoreRealN, nd2nzParams);
                } else {
                    LocalTensor<int8_t> bL1Tensor = inQueueBL1_.AllocTensor<int8_t>();
                    CopyInBL1Nz(bL1Tensor, kOffset, nBaseOffset, tiling_->groupSize, singleCoreRealN);
                    inQueueBL1_.EnQue(bL1Tensor);
                }

                uint64_t cOffset = cubeKDimIdx_ * 2 * multiScaleTimes_ * tiling_->mSize * tiling_->nSize +
                                   (groupIdx % 2) * multiScaleTimes_ * tiling_->mSize * tiling_->nSize +
                                   cubeNDimIdx_ * tiling_->matmulTiling.singleCoreN;
                if (taskId > 1) {
                    if (groupIdx % 2 == 0) {
                        WaitForVectorPing();
                    } else {
                        WaitForVectorPong();
                    }
                }
                LaunchMatmul(singleCoreRealN, cOffset, kOffset - kBaseOffset);
            }
            NotifyVector();
        }
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vector1EventIdVToMte3);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vector3EventIdVToMTE21);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vector3EventIdVToMTE22);
}
} // namespace Mc2WeightQuantBatchMatmulV2

#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_MSD_GROUP_H
