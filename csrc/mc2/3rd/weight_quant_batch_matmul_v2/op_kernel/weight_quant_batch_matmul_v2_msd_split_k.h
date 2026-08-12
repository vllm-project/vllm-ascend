/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file weight_quant_batch_matmul_v2_msd_split_k.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_MSD_SPLIT_K_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_MSD_SPLIT_K_H

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
using AscendC::BlockReduceMax;
using AscendC::BlockReduceSum;
using AscendC::CopyRepeatParams;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::MaskMode;
using AscendC::Nd2NzParams;
using AscendC::PipeBarrier;
using AscendC::QuePosition;
using AscendC::SetAtomicAdd;
using AscendC::SetAtomicNone;
using AscendC::SetFlag;
using AscendC::SetMaskNorm;
using AscendC::SetVectorMask;
using AscendC::SyncAll;
using AscendC::TBuf;
using AscendC::TEventID;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::WaitFlag;
using matmul::MatmulImpl;
using matmul::MatmulType;

namespace Mc2WeightQuantBatchMatmulV2 {
struct PreProcessParams {
    uint16_t realProcessM;
    uint16_t realProcessK;
};

template <typename wType>
struct CubeProcessParams {
    uint16_t l1BaseKb;
    uint16_t l1BaseKa;
    uint32_t aL1Size;
    LocalTensor<wType> aL1TensorPing;
    LocalTensor<wType> aL1TensorPong;
    LocalTensor<wType> bL1TensorPing;
    LocalTensor<wType> bL1TensorPong;
};

struct MatmulTaskLoopParams {
    uint64_t nOffset;
    uint64_t kOffset;
    uint64_t singleCoreK;
    uint64_t kIdx;
};

struct UnfoldCMatrixParams {
    uint64_t taskSingleCoreNSize;
    uint64_t nOffset;
    uint64_t kIdx;
    uint64_t baseMOffset;
    uint64_t mOffset;
    uint64_t singleCorerealM;
    uint64_t scaleOffset;
    uint64_t nRepeatTimes;
    BinaryRepeatParams &repeatParams;
};

static constexpr MatmulConfig MM_CFG_MSD = {true, false, false, 0, 0, 0, false, false, false, false, 0,
                                            0,    0,     0,     0, 0, 0, false, false, false, false, false};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType = HighPreciseType>
class Mc2WeightQuantBatchMatmulV2MsdSplitKKernel {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2MsdSplitKKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2MsdTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void PreProcess();
    __aicore__ inline void SetPreprocessParams(uint64_t mIdx, uint64_t kIdx, PreProcessParams &preProcessParams);
    __aicore__ inline void CopyInAOrigin(uint64_t mIdx, uint64_t kIdx, PreProcessParams &preProcessParams);

    __aicore__ inline void ComputeSumA(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx,
                                       PreProcessParams &preProcessParams);
    __aicore__ inline void ComputeSumAPerChannel(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx,
                                                 PreProcessParams &preProcessParams);
    __aicore__ inline void GetSumOrMaxAParamPerGroup(BinaryRepeatParams &param);
    __aicore__ inline void ComputeSumAPerGroup(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx,
                                               PreProcessParams &preProcessParams);
    __aicore__ inline uint32_t ComputeSumAOnce(LocalTensor<float> &dst, LocalTensor<float> &src, uint32_t numRepeatK,
                                               PreProcessParams &preProcessParams);
    __aicore__ inline void ComputeMaxA(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx,
                                       PreProcessParams &preProcessParams);
    __aicore__ inline void ComputeMaxAPerChannel(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx,
                                                 PreProcessParams &preProcessParams, uint32_t numRepeat);
    __aicore__ inline void ComputeMaxAPerGroup(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx,
                                               PreProcessParams &preProcessParams);

    __aicore__ inline uint32_t ComputeMaxAOnce(LocalTensor<float> &dst, LocalTensor<float> &src, uint32_t numRepeatK,
                                               PreProcessParams &preProcessParams);
    __aicore__ inline void UnfoldAMatrix(uint64_t mIdx, uint64_t kIdx, PreProcessParams &preProcessParams);
    __aicore__ inline void LaunchMatmul(uint64_t cubeNLoopIdx, uint64_t singleNOffset, uint64_t taskNSize,
                                        const CubeProcessParams<wType> &cubeProcessParams,
                                        SyncProcessor<HardEvent::MTE1_MTE2> &al1SyncProcessor,
                                        SyncProcessor<HardEvent::MTE1_MTE2> &bl1SyncProcessor);
    __aicore__ inline void GetCopyInAL1Params(Nd2NzParams &nd2nzParams, uint64_t &kOffset, uint64_t l1BaseKa,
                                              uint64_t l1BaseM);
    __aicore__ inline void CopyInAL1(const CubeProcessParams<wType> &cubeProcessParams, uint64_t kIdx, uint64_t mOffset,
                                     uint64_t kOffset, uint64_t l1BaseKa, uint64_t l1BaseM,
                                     SyncProcessor<HardEvent::MTE1_MTE2> &al1SyncProcessor);
    __aicore__ inline void BL1PreLoad(const CubeProcessParams<wType> &cubeProcessParams);
    __aicore__ inline void BL1DmaCopy(uint64_t kaOffset, uint64_t stepKbOffset, uint64_t l1RealKa,
                                      uint64_t singleCoreNOffset, uint64_t l1RealN,
                                      const CubeProcessParams<wType> &cubeProcessParamsconst,
                                      const MatmulTaskLoopParams &matmulTaskLoopParams);
    __aicore__ inline void CopyInBL1(const LocalTensor<wType> &bL1Tensor, uint64_t nOffset, uint64_t kOffset,
                                     uint64_t l1RealN, uint64_t l1BaseKb);
    __aicore__ inline void CopyInBL1Nd(const LocalTensor<wType> &bL1Tensor, uint64_t nOffset, uint64_t kOffset,
                                       uint64_t l1RealN, uint64_t l1BaseKb);
    __aicore__ inline void CopyInBL1Nz(const LocalTensor<wType> &bL1Tensor, uint64_t nOffset, uint64_t kOffset,
                                       uint64_t l1RealN, uint64_t l1BaseKb);
    __aicore__ inline void SumMaxMul(uint64_t taskSingleCoreNSize, uint64_t realM);
    __aicore__ inline void ComputeOffsetMn(uint64_t taskSingleCoreNSize, uint64_t taskNSize, uint64_t nOffset,
                                           uint64_t baseMOffset, uint64_t singleCorerealM);
    __aicore__ inline void ProcessC1C2(uint64_t taskSingleCoreNSize, uint64_t nOffset, uint64_t baseMOffset,
                                       uint64_t singleCorerealM);
    __aicore__ inline void ProcessC1C2PerGroup(uint64_t taskSingleCoreNSize, uint64_t nOffset, uint64_t baseMOffset,
                                               uint64_t singleCorerealM, uint64_t nRepeatTimes,
                                               const BinaryRepeatParams &repeatParams);
    __aicore__ inline void CopyOutResult(uint64_t taskSingleCoreNSize, uint64_t nOffset, uint64_t baseMOffset,
                                         uint64_t singleCorerealM);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void InitGlobalTensor(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void ComputeParams();
    __aicore__ inline void ComputeRepeatParams();
    __aicore__ inline void ProcessVector();
    __aicore__ inline void PostProcess();
    __aicore__ inline void PostProcessOneTaskN(uint64_t baseMOffset, uint64_t singleCorerealM,
                                               uint64_t baseSingleCoreNOffset, uint64_t taskSingleCoreNSize,
                                               uint64_t nLoopCnt, int64_t &aivSetTimes);
    __aicore__ inline void CopyInSumMax(uint64_t baseMOffset, uint64_t singleCorerealM);
    __aicore__ inline void CopyInSumMaxPerGroup(uint64_t baseMOffset, uint64_t singleCorerealM);
    __aicore__ inline void ProcessCube();
    __aicore__ inline void InitCubeParams(CubeProcessParams<wType> &cubeProcessParams,
                                          SyncProcessor<HardEvent::MTE1_MTE2> &al1SyncProcessor,
                                          SyncProcessor<HardEvent::MTE1_MTE2> &bl1SyncProcessor);
    __aicore__ inline uint64_t InitMatmulParams();
    __aicore__ inline void ProcessReduceTail(LocalTensor<float> &dst, LocalTensor<float> &src, uint32_t numRepeatK,
                                             uint32_t nextRepeatK, uint32_t numProcessK,
                                             PreProcessParams &preProcessParams);
    __aicore__ inline void SetRepeatParams(BinaryRepeatParams &repeatParams, UnaryRepeatParams &unaryRepeatParams,
                                           UnaryRepeatParams &f32ToF16RepeatParams);
    __aicore__ inline void IterateAllSingleCoreK(uint64_t singleCoreNOffset, uint64_t l1RealM, uint32_t l1RealN,
                                                 uint64_t mOffset, const CubeProcessParams<wType> &cubeProcessParams,
                                                 const MatmulTaskLoopParams &matmulTaskLoopParams,
                                                 SyncProcessor<HardEvent::MTE1_MTE2> &al1SyncProcessor,
                                                 SyncProcessor<HardEvent::MTE1_MTE2> &bl1SyncProcessor);
    __aicore__ inline void LaunchMatmul(const CubeProcessParams<wType> &cubeProcessParams,
                                        SyncProcessor<HardEvent::MTE1_MTE2> &al1SyncProcessor,
                                        SyncProcessor<HardEvent::MTE1_MTE2> &bl1SyncProcessor);
    __aicore__ inline void ProcessUnfoldCMatrix(const UnfoldCMatrixParams &params,
                                                SyncProcessor<HardEvent::V_MTE2> &c1c2SyncProcessor);
    __aicore__ inline void GetPingPongC1C2Tensor(LocalTensor<int32_t> &c1c2S32Tensor, LocalTensor<float> &c1c2Tensor,
                                                 SyncProcessor<HardEvent::V_MTE2> &c1c2SyncProcessor);
    __aicore__ inline void ProcessUnfoldCMatrixPerchannel(uint64_t cMatrixOffset, uint64_t realM,
                                                          uint64_t taskSingleCoreNSize, float divFactors,
                                                          uint64_t nRepeatTimes,
                                                          SyncProcessor<HardEvent::V_MTE2> &c1c2SyncProcessor);
    __aicore__ inline void ProcessUnfoldCMatrixPergroup(uint64_t cMatrixOffset, uint64_t realM,
                                                        uint64_t taskSingleCoreNSize, uint64_t nRepeatTimes,
                                                        SyncProcessor<HardEvent::V_MTE2> &c1c2SyncProcessor);
    __aicore__ inline void CalculateC1C2Pergroup(uint64_t id, uint64_t cnt, uint64_t loadNum,
                                                 uint64_t taskSingleCoreNSize, uint64_t realM, uint64_t nRepeatTimes,
                                                 const BinaryRepeatParams &repeatParams, const event_t &eventIdVToMte2,
                                                 const event_t &eventIdMte2ToV);
    __aicore__ inline void ProcessUnfoldCMatrixPergroupFp16(uint64_t gId, uint64_t realM, half divFactorsFp16[],
                                                            uint64_t repeatTimesAxpyFp32, uint64_t repeatTimesAxpyFp16,
                                                            const CopyRepeatParams &copyParam,
                                                            LocalTensor<half> &c1c2S32Tensor);
    __aicore__ inline void CopyInC1C2TensorPerGroup(uint64_t id, uint64_t loadNum, uint64_t baseMOffset,
                                                    uint64_t nOffset, uint64_t realM, uint64_t taskSingleCoreNSize,
                                                    const event_t &eventIdVToMte2, const event_t &eventIdMte2ToV);
    __aicore__ inline void TransToWType(uint64_t unfoldATimes, uint64_t defaultOffset, uint64_t mainRepeatK,
                                        float multiFactors, const BinaryRepeatParams &f32BinaryRepeatParams,
                                        const UnaryRepeatParams &f32UnaryRepeatParams,
                                        const UnaryRepeatParams &f32ToF16RepeatParams,
                                        const PreProcessParams &preProcessParams);
    __aicore__ inline void TransToWTypeFirstStep(uint64_t defaultOffset, uint64_t mainRepeatK,
                                                 const BinaryRepeatParams &f32BinaryRepeatParams,
                                                 const PreProcessParams &preProcessParams);

    using InputXType = MatmulType<TPosition::A1, CubeFormat::NZ, wType, aTrans>;
    using InputWType = MatmulType<TPosition::B1, CubeFormat::NZ, wType, bTrans>;
    using OutputYType = MatmulType<TPosition::GM, CubeFormat::ND, preciseType>;
    using InputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, preciseType>;
    MatmulImpl<InputXType, InputWType, OutputYType, InputBiasType, MM_CFG_MSD> mmObj_;

    TPipe *pipe_;
    const Mc2WeightQuantBatchMatmulV2MsdTilingData *tiling_;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<int8_t> wGlobal_;
    GlobalTensor<wType> wGlobalWType_;
    GlobalTensor<xType> antiQuantOffsetGlobal_;
    GlobalTensor<xType> antiQuantScaleGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<uint64_t> quantScaleGlobal_;
    GlobalTensor<yType> yGlobal_;

    GlobalTensor<wType> aUnfoldGlobal_;
    GlobalTensor<int8_t> aUnfoldGlobalInt8_; // cube mte2相关指令只能处理s8的输入
    GlobalTensor<preciseType> cUnfoldGlobal_;
    GlobalTensor<float> reduceMaxWorkspaceGm_;
    GlobalTensor<float> reduceSumWorkspaceGm_;

    // 预处理的tensor
    LocalTensor<xType> aOriginTensor_;
    LocalTensor<float> aF32Tensor_;
    LocalTensor<float> aF32TmpTensor_;
    LocalTensor<float> aFp32ReduceTensor_;
    LocalTensor<half> aFp16Tensor_;
    LocalTensor<wType> aUnfoldLocalTensor_;
    LocalTensor<float> aSumTensor_;
    LocalTensor<float> aMaxTensor_;

    LocalTensor<float> antiquantScaleF32Tensor_;
    LocalTensor<float> cF32Tensor_;

    // computeOffsetMn的buffer分配
    LocalTensor<float> antiquantOffsetF32Tensor_;
    LocalTensor<float> scaleOffsetProductTensor_;
    LocalTensor<xType> antiquantScaleF16Tensor_;
    LocalTensor<xType> antiquantOffsetF16Tensor_;
    LocalTensor<float> aSumComputeOffsetTensor_;

    // processC1C2的buffer分配
    LocalTensor<float> c1c2PingTensor_;
    LocalTensor<float> c1c2PongTensor_;
    LocalTensor<int32_t> c1c2S32PingTensor_;
    LocalTensor<int32_t> c1c2S32PongTensor_;
    LocalTensor<half> c1c2Fp16PingTensor_;
    LocalTensor<half> c1c2Fp16PongTensor_;
    LocalTensor<float> c1c2ComputeTensor_;
    LocalTensor<half> c1c2Fp16ComputeTensor_;
    LocalTensor<float> c1c2ScaleMaxComputeTensor_;
    LocalTensor<float> aMaxComputeTensor_;

    // copyOutResult的buffer分配
    LocalTensor<biasType> biasTensor_;
    LocalTensor<float> biasFp32Tensor_;

    LocalTensor<yType> cF16ResultTensor_;

    TBuf<> tBuf_;
    TBuf<TPosition::A1> l1TBuf_;

    LocalTensor<wType> aL1GroupPackTensor_; // pergroup场景A矩阵按照GroupPack维度加载

    BinaryRepeatParams commonRepeatParams_;
    UnaryRepeatParams commonUnaryRepeatParams_;
    UnaryRepeatParams fp16ToFp32UnaryRepeatParams_;

    bool cubeFirstLoop_ = true;
    uint32_t curBlockIdx_ = 0;
    uint32_t multiScaleTimes_ = 2;
    // cache num和同步周期默认为8/4，需要根据具体shape重新计算
    uint32_t c1c2CacheNum_ = 8;
    uint32_t vectorToCubeSyncCycle_ = 4;
    uint64_t kBlockNum_ = 0;
    uint64_t cUnfoldSize_ = 0;
    uint64_t lastAl1Offset_ = 0;
    uint64_t curTaskNSize_ = 0;
    uint64_t curTaskNOffset_ = 0;
    uint64_t cubeSingleCoreN_ = 0;
    uint64_t cubeLoopIdx_ = 0;
    uint64_t aUnfoldSizeS8_ = 0;
    uint64_t aUnfoldSizeWtype_ = 0;
    uint64_t taskStartIdx_ = 0;
    uint64_t taskIdx_ = 0;
    uint64_t taskLimit_ = 0;
    uint64_t kSigncoreTaskNum_ = 0;

    uint32_t v1BaseKGroupNum_ = 1;
    uint32_t groupNum_ = 1;
    uint32_t groupPack_ = 1;
    uint32_t curGroupId_ = 0;
    uint32_t realGroupPack_ = 1;

    float divFactors_[3] = {1.0f, 1.0f, 1.0f};
    float multiFactors_[3] = {1.0f, 1.0f, 1.0f};
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
                       GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                       const Mc2WeightQuantBatchMatmulV2MsdTilingData *tilingData, TPipe *tPipe)
{
    tiling_ = tilingData;
    curBlockIdx_ = GetBlockIdx();

    if ASCEND_IS_AIC {
        mmObj_.SetSubBlockIdx(0);
        mmObj_.Init(&tiling_->matmulTiling, tPipe);
        if constexpr (IsSameType<preciseType, HighPerformanceType>::value) {
            mmObj_.SetQuantScalar(0x3F800000); // 设置QuantScalar为1.0
        }
    }

    pipe_ = tPipe;
    ComputeParams();
    ComputeRepeatParams();
    InitGlobalTensor(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace);
    InitBuffer();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::InitGlobalTensor(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                   GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace)
{
    uint64_t alignedOffset = 0;
    // 将不同的kIdx的数据合并排布，方便后处理一次搬入kBlockNum组m*8
    uint64_t kMaxSumCnt = kBlockNum_;
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        kMaxSumCnt = groupNum_;
    }
    uint64_t reduceMaxSumOffset = CeilAlign(tiling_->mSize * ONE_BLK_SIZE * kMaxSumCnt, 512UL);
    reduceMaxWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    alignedOffset += reduceMaxSumOffset;

    reduceSumWorkspaceGm_.SetGlobalBuffer((__gm__ float *)(workspace + alignedOffset));
    alignedOffset += reduceMaxSumOffset;

    uint64_t aUnfoldSize = aUnfoldSizeS8_ * kBlockNum_;
    aUnfoldGlobal_.SetGlobalBuffer((__gm__ wType *)(workspace + alignedOffset));
    aUnfoldGlobalInt8_.SetGlobalBuffer((__gm__ int8_t *)(workspace + alignedOffset));
    alignedOffset += aUnfoldSize;

    cUnfoldGlobal_.SetGlobalBuffer((__gm__ preciseType *)(workspace + alignedOffset));
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(x));
    wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(weight));
    wGlobalWType_.SetGlobalBuffer(reinterpret_cast<__gm__ wType *>(weight));
    antiQuantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantScale), tiling_->nSize);
    antiQuantOffsetGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantOffset), tiling_->nSize);
    if (tiling_->hasBias) {
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling_->nSize);
    }
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType *>(y), tiling_->mSize * tiling_->nSize);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat, preciseType>::InitBuffer()
{
    // 初始化UB Tbuffer
    pipe_->InitBuffer(tBuf_, 192 * 1024);

    // 预处理分配
    aF32TmpTensor_ = tBuf_.Get<float>();                                // 49k 0-49k分配给float的aTmp矩阵
    aOriginTensor_ = tBuf_.Get<xType>()[49 * HALF_DATA_BENCHMARK];      // 24k 49-73k分配给xType的a矩阵
    aFp32ReduceTensor_ = tBuf_.Get<float>()[73 * FLOAT_DATA_BENCHMARK]; // 24k 73-97分配给float的reduce后的a矩阵
    aFp16Tensor_ = tBuf_.Get<half>()[73 * HALF_DATA_BENCHMARK]; // 24k 73-97k分配给half的a矩阵 空间复用
    // 24k 97-121k分配给unfold的a矩阵
    if constexpr (IsSameType<wType, int4b_t>::value) {
        aUnfoldLocalTensor_ = tBuf_.Get<wType>()[97 * INT4_DATA_BENCHMARK];
    } else {
        aUnfoldLocalTensor_ = tBuf_.Get<wType>()[97 * INT8_DATA_BENCHMARK];
    }
    aMaxTensor_ = tBuf_.Get<float>()[121 * FLOAT_DATA_BENCHMARK]; // 8k 121-129k分配给aMax的a矩阵
    aSumTensor_ = tBuf_.Get<float>()[129 * FLOAT_DATA_BENCHMARK]; // 8k 129-137k分配给aSum的a矩阵
    aF32Tensor_ = tBuf_.Get<float>()[137 * FLOAT_DATA_BENCHMARK]; // 49k 137-186k分配给float的a矩阵

    antiquantScaleF32Tensor_ = tBuf_.Get<float>();                             // 8k 0-8k分配给的fp32的scale矩阵
    aMaxComputeTensor_ = tBuf_.Get<float>()[168 * FLOAT_DATA_BENCHMARK];       // 12k 168-180k分配给aMaxPing矩阵
    aSumComputeOffsetTensor_ = tBuf_.Get<float>()[180 * FLOAT_DATA_BENCHMARK]; // 12k 180-192k分配给的fp32的aSum矩阵

    // 默认64*256
    cF32Tensor_ = tBuf_.Get<float>()[8 * FLOAT_DATA_BENCHMARK]; // 64k 8-72k分配给累加后的c矩阵

    // computeOffsetMn的buffer分配
    antiquantOffsetF32Tensor_ = tBuf_.Get<float>()[136 * FLOAT_DATA_BENCHMARK]; // 8k 136-144k分配给的fp32的offset矩阵
    scaleOffsetProductTensor_ = tBuf_.Get<float>()[144 * FLOAT_DATA_BENCHMARK]; // 8k 144-152k分配给的fp32的offset矩阵
    antiquantScaleF16Tensor_ = tBuf_.Get<xType>()[152 * HALF_DATA_BENCHMARK]; // 4k 152-156k分配给的fp16的scale矩阵
    antiquantOffsetF16Tensor_ = tBuf_.Get<xType>()[156 * HALF_DATA_BENCHMARK]; // 4k 156-160k分配给的fp16的offset矩阵

    // processC1C2的buffer分配
    // 默认32*256
    c1c2S32PingTensor_ = tBuf_.Get<int32_t>()[72 * FLOAT_DATA_BENCHMARK];  // 32k 72-104k分配给int32的c1c2矩阵
    c1c2Fp16PingTensor_ = tBuf_.Get<half>()[72 * HALF_DATA_BENCHMARK];     // 32k 72-104k分配给fp16的c1c2矩阵
    c1c2PingTensor_ = tBuf_.Get<float>()[72 * FLOAT_DATA_BENCHMARK];       // 32k 72-104k分配给float的c1c2矩阵
    c1c2S32PongTensor_ = tBuf_.Get<int32_t>()[104 * FLOAT_DATA_BENCHMARK]; // 32k 104-136k分配给的int32的c1c2矩阵
    c1c2Fp16PongTensor_ = tBuf_.Get<half>()[104 * HALF_DATA_BENCHMARK];    // 32k 104-136k分配给的fp16的c1c2矩阵
    c1c2PongTensor_ = tBuf_.Get<float>()[104 * FLOAT_DATA_BENCHMARK];      // 32k 104-136k分配给的float的c1c2矩阵
    if constexpr (IsSameType<preciseType, HighPreciseType>::value) {
        c1c2ComputeTensor_ = tBuf_.Get<float>()[136 * FLOAT_DATA_BENCHMARK]; // 32k 136-168k分配给的float的c1c2矩阵
    } else {
        c1c2Fp16ComputeTensor_ = tBuf_.Get<half>()[136 * HALF_DATA_BENCHMARK]; // 4K 136K-140K 分配给fp16的c1c2矩阵
        c1c2ScaleMaxComputeTensor_ =
            tBuf_.Get<float>()[140 * FLOAT_DATA_BENCHMARK];                  // 2K 140K-142K 分配给fp16的c1c2矩阵
        c1c2ComputeTensor_ = tBuf_.Get<float>()[142 * FLOAT_DATA_BENCHMARK]; // 26k 142-168k分配给的float的c1c2矩阵
    }

    // copyOutResult的buffer分配
    if constexpr (IsSameType<biasType, float>::value) {
        biasTensor_ = tBuf_.Get<biasType>()[72 * FLOAT_DATA_BENCHMARK]; // 7k 72-79k存bias类型的y矩阵
        biasFp32Tensor_ = tBuf_.Get<float>()[72 * FLOAT_DATA_BENCHMARK]; // 7k 72-79kfloat类型无需重新申请空间存bias
    } else {
        biasTensor_ = tBuf_.Get<biasType>()[72 * HALF_DATA_BENCHMARK];   // 7k 72-79k存bias类型的y矩阵
        biasFp32Tensor_ = tBuf_.Get<float>()[79 * FLOAT_DATA_BENCHMARK]; // 7k 79k-86k之后存float类型的y矩阵
    }
    cF16ResultTensor_ = tBuf_.Get<yType>()[86 * HALF_DATA_BENCHMARK]; // 32k 86-118k分配给的fp16的cResult矩阵
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                                  antiQuantType, hasAntiQuantOffset, quantType,
                                                                  weightFormat, preciseType>::ComputeRepeatParams()
{
    commonRepeatParams_.dstBlkStride = 1;
    commonRepeatParams_.src0BlkStride = 1;
    commonRepeatParams_.src1BlkStride = 1;
    commonRepeatParams_.dstRepStride = FP32_MASK_BLK_NUM;
    commonRepeatParams_.src0RepStride = FP32_MASK_BLK_NUM;
    commonRepeatParams_.src1RepStride = FP32_MASK_BLK_NUM;

    commonUnaryRepeatParams_.dstBlkStride = 1;
    commonUnaryRepeatParams_.srcBlkStride = 1;
    commonUnaryRepeatParams_.dstRepStride = FP32_MASK_BLK_NUM;
    commonUnaryRepeatParams_.srcRepStride = FP32_MASK_BLK_NUM;

    fp16ToFp32UnaryRepeatParams_.dstBlkStride = 1;
    fp16ToFp32UnaryRepeatParams_.srcBlkStride = 1;
    fp16ToFp32UnaryRepeatParams_.dstRepStride = FP32_MASK_BLK_NUM;
    fp16ToFp32UnaryRepeatParams_.srcRepStride = FP32_MASK_BLK_NUM >> 1;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat, preciseType>::ComputeParams()
{
    // pergroup场景kBlockNum_只用于预处理
    kBlockNum_ = CeilDiv(tiling_->kSize, static_cast<uint64_t>(tiling_->v1BaseK));
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        v1BaseKGroupNum_ = CeilDiv(static_cast<uint64_t>(tiling_->v1BaseK), tiling_->groupSize);
        groupNum_ = CeilDiv(tiling_->kSize, tiling_->groupSize);
        groupPack_ = tiling_->groupPack;
    }

    float multiFactor1 = 127.0f;
    float multiFactor2 = 254.0f;
    aUnfoldSizeS8_ = CeilAlign(sizeof(xType) * tiling_->mSize * tiling_->v1BaseK, 512UL);
    aUnfoldSizeWtype_ = aUnfoldSizeS8_;
    if constexpr (IsSameType<wType, int4b_t>::value) {
        multiFactor1 = 7.49f;
        multiFactor2 = 14.98f;
        multiScaleTimes_ = 3;
        // int4场景需要按照unpack的方式去计算偏移
        aUnfoldSizeWtype_ = aUnfoldSizeWtype_ << 1;
    }

    // workspace 空间默认占用32Mb，开pingpong处理，需要计算多少轮n方向可以需要做一次同步
    if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
        vectorToCubeSyncCycle_ =
            16 * 1024 * 256 / (kBlockNum_ * multiScaleTimes_ * tiling_->mSize * tiling_->taskNSize);
        cUnfoldSize_ = CeilAlign(multiScaleTimes_ * tiling_->mSize * tiling_->taskNSize * kBlockNum_, 128UL);
    } else {
        vectorToCubeSyncCycle_ =
            18 * 1024 * 256 / (groupPack_ * multiScaleTimes_ * tiling_->mSize * tiling_->taskNSize);
        cUnfoldSize_ = CeilAlign(multiScaleTimes_ * tiling_->mSize * tiling_->taskNSize * groupPack_, 128UL);
    }
    vectorToCubeSyncCycle_ = vectorToCubeSyncCycle_ <= 0 ? 1 : vectorToCubeSyncCycle_;
    // 硬件同步寄存器最多可连续set16次, 防止由于cacheNum个数过多，使set次数过多，导致卡死
    vectorToCubeSyncCycle_ = vectorToCubeSyncCycle_ > 8 ? 8 : vectorToCubeSyncCycle_;
    c1c2CacheNum_ = vectorToCubeSyncCycle_ << 1;

    divFactors_[0] = 1.0f / multiFactor1;
    if constexpr (IsSameType<preciseType, HighPreciseType>::value) {
        divFactors_[1] = 1.0f / (multiFactor1 * multiFactor2);
        divFactors_[2] = 1.0f / (multiFactor1 * multiFactor2 * multiFactor2);
    } else {
        divFactors_[1] = 1.0f / multiFactor2;
        divFactors_[2] = 1.0f / (multiFactor2 * multiFactor2);
    }
    multiFactors_[0] = multiFactor1;
    multiFactors_[1] = multiFactor2;
    multiFactors_[2] = multiFactor2;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat, preciseType>::Process()
{
    if ASCEND_IS_AIV {
        ProcessVector();
    } else {
        ProcessCube();
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat, preciseType>::ProcessVector()
{
    PreProcess();
    SyncAll();
    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
    PostProcess();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat, preciseType>::PostProcess()
{
    uint64_t postProcessNNum = CeilDiv(tiling_->taskNSize, tiling_->taskSingleCoreNSize);
    uint64_t mIdx = curBlockIdx_ / postProcessNNum;
    uint64_t nIdx = curBlockIdx_ % postProcessNNum;
    uint64_t baseMOffset = mIdx * tiling_->postProcessSingleCoreM;
    uint64_t baseSingleCoreNOffset = nIdx * tiling_->taskSingleCoreNSize;
    uint64_t singleCorerealM =
        (baseMOffset + tiling_->postProcessSingleCoreM > tiling_->mSize && tiling_->mSize > baseMOffset) ?
            tiling_->mSize - baseMOffset :
            tiling_->postProcessSingleCoreM;

    uint64_t nLoopCnt =
        CeilDiv(tiling_->nSize, static_cast<uint64_t>(tiling_->taskNSize)) * CeilDiv(groupNum_, groupPack_);
    if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
        nLoopCnt = CeilDiv(tiling_->nSize, static_cast<uint64_t>(tiling_->taskNSize));
        CopyInSumMax(baseMOffset, singleCorerealM); // perchannel 在task循环外，统一加载sum和max
    }

    int64_t aivSetTimes = 0;
    for (; curTaskNOffset_ < tiling_->nSize; curTaskNOffset_ += tiling_->taskNSize) {
        curTaskNSize_ = tiling_->taskNSize + curTaskNOffset_ > tiling_->nSize ? tiling_->nSize - curTaskNOffset_ :
                                                                                tiling_->taskNSize;
        uint64_t taskSingleCoreNSize = (baseSingleCoreNOffset + tiling_->taskSingleCoreNSize > curTaskNSize_ &&
                                        curTaskNSize_ > baseSingleCoreNOffset) ?
                                           curTaskNSize_ - baseSingleCoreNOffset :
                                           tiling_->taskSingleCoreNSize;
        uint64_t nOffset = curTaskNOffset_ + baseSingleCoreNOffset;

        PostProcessOneTaskN(baseMOffset, singleCorerealM, baseSingleCoreNOffset, taskSingleCoreNSize, nLoopCnt,
                            aivSetTimes);

        if (likely(baseSingleCoreNOffset < curTaskNSize_ && baseMOffset < tiling_->mSize)) {
            CopyOutResult(taskSingleCoreNSize, nOffset, baseMOffset, singleCorerealM);
        }
    }
    if (unlikely(aivSetTimes > 0)) {
        CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL_FLAG);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::PostProcessOneTaskN(uint64_t baseMOffset, uint64_t singleCorerealM, uint64_t baseSingleCoreNOffset,
                                      uint64_t taskSingleCoreNSize, uint64_t nLoopCnt, int64_t &aivSetTimes)
{
    uint64_t nOffset = curTaskNOffset_ + baseSingleCoreNOffset;
    curGroupId_ = 0;
    Duplicate(cF32Tensor_, 0.0f, singleCorerealM * tiling_->taskSingleCoreNSize);
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(FP32_MAX_MASK_SIZE);
    for (; curGroupId_ < groupNum_; curGroupId_ += groupPack_, cubeLoopIdx_++) { // perchannel场景groupNum_为1
        if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) { // pergroup 按照groupPack为单位，统一加载sum和max
            realGroupPack_ = curGroupId_ + groupPack_ > groupNum_ ? groupNum_ - curGroupId_ : groupPack_;
            CopyInSumMaxPerGroup(baseMOffset, singleCorerealM);
        }

        if (likely(baseSingleCoreNOffset < curTaskNSize_ && baseMOffset < tiling_->mSize)) {
            ComputeOffsetMn(taskSingleCoreNSize, curTaskNSize_, nOffset, baseMOffset, singleCorerealM);
        }

        if (unlikely(cubeLoopIdx_ > 0 && cubeLoopIdx_ % vectorToCubeSyncCycle_ == 0)) {
            CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL_FLAG);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);
            aivSetTimes--;
        }
        CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);

        if (likely(baseSingleCoreNOffset < curTaskNSize_ && baseMOffset < tiling_->mSize)) {
            ProcessC1C2(taskSingleCoreNSize, baseSingleCoreNOffset, baseMOffset, singleCorerealM);
            // 需要等待上一轮GroupPack对sum和max的计算结束，才能载入下一轮的sum和max
            event_t eventIdSumMaxVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdSumMaxVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdSumMaxVToMte2);
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(nLoopCnt > 1 && (cubeLoopIdx_ + 1) % vectorToCubeSyncCycle_ == 0)) {
            CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_AIV_ONLY_ALL_FLAG);
            aivSetTimes++;
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat,
                                           preciseType>::CopyInSumMax(uint64_t baseMOffset, uint64_t singleCorerealM)
{
    if constexpr (hasAntiQuantOffset) {
        DataCopyPad2D(aSumComputeOffsetTensor_, reduceSumWorkspaceGm_[baseMOffset * FP32_BLOCK_SIZE], kBlockNum_,
                      singleCorerealM * FP32_BLOCK_SIZE, tiling_->mSize * FP32_BLOCK_SIZE);
    }
    DataCopyPad2D(aMaxComputeTensor_, reduceMaxWorkspaceGm_[baseMOffset * FP32_BLOCK_SIZE], kBlockNum_,
                  singleCorerealM * FP32_BLOCK_SIZE, tiling_->mSize * FP32_BLOCK_SIZE);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyInSumMaxPerGroup(uint64_t baseMOffset, uint64_t singleCorerealM)
{
    uint64_t gmOfffset = baseMOffset * groupNum_ * FP32_BLOCK_SIZE + curGroupId_ * FP32_BLOCK_SIZE;
    if constexpr (hasAntiQuantOffset) {
        DataCopyPad2D(aSumComputeOffsetTensor_, reduceSumWorkspaceGm_[gmOfffset], singleCorerealM,
                      realGroupPack_ * FP32_BLOCK_SIZE, groupNum_ * FP32_BLOCK_SIZE);
    }
    DataCopyPad2D(aMaxComputeTensor_, reduceMaxWorkspaceGm_[gmOfffset], singleCorerealM,
                  realGroupPack_ * FP32_BLOCK_SIZE, groupNum_ * FP32_BLOCK_SIZE);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat, preciseType>::ProcessCube()
{
    LocalTensor<wType> tensorTmp;
    CubeProcessParams<wType> cubeProcessParams = {0, 0, 0, tensorTmp, tensorTmp, tensorTmp, tensorTmp};
    SyncProcessor<HardEvent::MTE1_MTE2> al1SyncProcessor;
    SyncProcessor<HardEvent::MTE1_MTE2> bl1SyncProcessor;
    InitCubeParams(cubeProcessParams, al1SyncProcessor, bl1SyncProcessor);
    BL1PreLoad(cubeProcessParams);
    PipeBarrier<PIPE_MTE2>(); // 隔离preload和cube mte2的影响

    CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);

    for (; curTaskNOffset_ < tiling_->nSize; curTaskNOffset_ += tiling_->taskNSize) {
        curGroupId_ = 0;
        for (; curGroupId_ < groupNum_; curGroupId_ += groupPack_, cubeLoopIdx_++) { // perchannel场景groupNum_为1
            if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
                realGroupPack_ = curGroupId_ + groupPack_ > groupNum_ ? groupNum_ - curGroupId_ : groupPack_;
            }
            if (unlikely(cubeLoopIdx_ > vectorToCubeSyncCycle_ && cubeLoopIdx_ % vectorToCubeSyncCycle_ == 0)) {
                CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
            }
            LaunchMatmul(cubeProcessParams, al1SyncProcessor, bl1SyncProcessor);
            if (likely(cubeLoopIdx_ > 0)) {
                CrossCoreWaitFlag(SYNC_AIC_ONLY_ALL_FLAG);
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
            }
            CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_AIC_ONLY_ALL_FLAG);
        }
    }
    CrossCoreWaitFlag(SYNC_AIC_ONLY_ALL_FLAG);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
    if (unlikely(cubeLoopIdx_ > vectorToCubeSyncCycle_ && cubeLoopIdx_ % vectorToCubeSyncCycle_ == 0)) {
        CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
    }
    al1SyncProcessor.Destory();
    bl1SyncProcessor.Destory();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::InitCubeParams(CubeProcessParams<wType> &cubeProcessParams,
                                 SyncProcessor<HardEvent::MTE1_MTE2> &al1SyncProcessor,
                                 SyncProcessor<HardEvent::MTE1_MTE2> &bl1SyncProcessor)
{
    uint64_t l1Size = 512 * 1024;
    pipe_->InitBuffer(l1TBuf_, l1Size);
    if constexpr (IsSameType<wType, int4b_t>::value) {
        l1Size = l1Size << 1;
    }
    uint16_t aL1DbNum =
        tiling_->matmulTiling.depthA1 / (tiling_->matmulTiling.stepKa * tiling_->matmulTiling.stepM) > 1 ? 2 : 1;
    uint16_t bL1DbNum =
        tiling_->matmulTiling.depthB1 / (tiling_->matmulTiling.stepKb * tiling_->matmulTiling.stepN) > 1 ? 2 : 1;

    uint16_t l1BaseKb = tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKb;
    uint16_t l1BaseKa = tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKa;
    uint32_t bL1Size = tiling_->matmulTiling.baseN * l1BaseKb;
    uint32_t aL1Size = tiling_->matmulTiling.baseM * l1BaseKa;

    cubeProcessParams.l1BaseKb = l1BaseKb, cubeProcessParams.l1BaseKa = l1BaseKa, cubeProcessParams.aL1Size = aL1Size,

    // 为了缓解mte1 bank conflict，l1的空间排布为: bl1Ping, al1Ping, 剩余空间，al1Pong, bl1Pong
        cubeProcessParams.aL1TensorPing = l1TBuf_.Get<wType>()[bL1Size];
    cubeProcessParams.aL1TensorPong =
        l1TBuf_.Get<wType>()[aL1DbNum > 1 ? l1Size - bL1Size * (aL1DbNum - 1) - aL1Size : bL1Size];
    cubeProcessParams.bL1TensorPing = l1TBuf_.Get<wType>()[0];
    cubeProcessParams.bL1TensorPong = l1TBuf_.Get<wType>()[bL1DbNum > 1 ? l1Size - bL1Size : 0],

    al1SyncProcessor.Init(aL1DbNum);
    bl1SyncProcessor.Init(bL1DbNum);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline uint64_t
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat, preciseType>::InitMatmulParams()
{
    curTaskNSize_ =
        tiling_->taskNSize + curTaskNOffset_ > tiling_->nSize ? tiling_->nSize - curTaskNOffset_ : tiling_->taskNSize;
    uint64_t nBlockNum = 0;
    uint64_t kBlockNum = 0;
    if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL)
        kBlockNum = kBlockNum_;
    else {
        // perGroup场景tiling保证singleCoreK为GroupSize
        kBlockNum = realGroupPack_;
    }
    uint64_t totalCoreNum = GetBlockNum();
    if constexpr (bTrans) {
        cubeSingleCoreN_ = CeilDiv(curTaskNSize_, totalCoreNum);
    } else {
        cubeSingleCoreN_ = tiling_->matmulTiling.singleCoreN;
    }
    nBlockNum = CeilDiv(curTaskNSize_, cubeSingleCoreN_);
    uint64_t taskNum = kBlockNum * nBlockNum;
    uint64_t singleCoreTaskNum = CeilDiv(taskNum, totalCoreNum);
    taskStartIdx_ = curBlockIdx_ * singleCoreTaskNum;
    taskLimit_ = (curBlockIdx_ + 1) * singleCoreTaskNum;
    if (unlikely(taskLimit_ > taskNum)) {
        taskLimit_ = taskNum;
    }
    return kBlockNum;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat, preciseType>::PreProcess()
{
    uint64_t mBlockNum = CeilDiv(tiling_->mSize, static_cast<uint64_t>(tiling_->v1BaseM));
    uint64_t taskNum = kBlockNum_ * mBlockNum;
    uint64_t totalCoreNum = GetBlockNum() * 2;
    uint64_t singleCoreTaskNum = CeilDiv(taskNum, totalCoreNum);
    taskStartIdx_ = curBlockIdx_ * singleCoreTaskNum;
    taskLimit_ = (curBlockIdx_ + 1) * singleCoreTaskNum;
    if (unlikely(taskLimit_ > taskNum)) {
        taskLimit_ = taskNum;
    }
    PreProcessParams preProcessParams;
    for (uint64_t taskIdx = taskStartIdx_; taskIdx < taskLimit_; taskIdx++) {
        // 预处理优先处理k方向
        uint64_t mIdx = taskIdx / kBlockNum_;
        uint64_t kIdx = taskIdx % kBlockNum_;
        SetPreprocessParams(mIdx, kIdx, preProcessParams);

        CopyInAOrigin(mIdx, kIdx, preProcessParams);
        PipeBarrier<PIPE_V>();
        if (likely(taskIdx > 0)) {
            event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
        }
        if constexpr (hasAntiQuantOffset) {
            ComputeSumA(taskIdx, mIdx, kIdx, preProcessParams);
            PipeBarrier<PIPE_V>();
        }

        ComputeMaxA(taskIdx, mIdx, kIdx, preProcessParams);
        PipeBarrier<PIPE_V>();

        UnfoldAMatrix(mIdx, kIdx, preProcessParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::SetPreprocessParams(uint64_t mIdx, uint64_t kIdx, PreProcessParams &preProcessParams)
{
    preProcessParams.realProcessM = tiling_->v1BaseM;
    uint64_t mOffset = mIdx * tiling_->v1BaseM;
    if (unlikely(mOffset < tiling_->mSize && mOffset + tiling_->v1BaseM > tiling_->mSize)) {
        preProcessParams.realProcessM = tiling_->mSize - mOffset;
    }

    preProcessParams.realProcessK = tiling_->v1BaseK;
    uint64_t kOffset = kIdx * tiling_->v1BaseK;
    if (unlikely(kOffset < tiling_->kSize && kOffset + tiling_->v1BaseK > tiling_->kSize)) {
        preProcessParams.realProcessK = tiling_->kSize - kOffset;
    }
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        v1BaseKGroupNum_ = CeilDiv(static_cast<uint64_t>(preProcessParams.realProcessK), tiling_->groupSize);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyInAOrigin(uint64_t mIdx, uint64_t kIdx, PreProcessParams &preProcessParams)
{
    DataCopyPad2D(aOriginTensor_, xGlobal_[mIdx * tiling_->v1BaseM * tiling_->kSize + kIdx * tiling_->v1BaseK],
                  preProcessParams.realProcessM, preProcessParams.realProcessK, tiling_->v1BaseK, tiling_->kSize);
    uint64_t duplicateLength =
        tiling_->v1BaseK > preProcessParams.realProcessK ? tiling_->v1BaseK - preProcessParams.realProcessK : 0;
    if (unlikely(duplicateLength > FP16_BLOCK_SIZE)) {
        for (uint64_t mLoopIdx = 0; mLoopIdx < tiling_->v1BaseM; mLoopIdx++) {
            Duplicate(aOriginTensor_[mLoopIdx * tiling_->v1BaseK + preProcessParams.realProcessK],
                      static_cast<xType>(0.0f), duplicateLength);
        }
        PipeBarrier<PIPE_V>();
    }
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    Cast(aF32Tensor_, aOriginTensor_, RoundMode::CAST_NONE, preProcessParams.realProcessM * tiling_->v1BaseK);

    // 避免较大场景同步问题
    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ComputeSumA(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx, PreProcessParams &preProcessParams)
{
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(FP32_MAX_MASK_SIZE);
    if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
        ComputeSumAPerChannel(taskIdx, mIdx, kIdx, preProcessParams);
    } else {
        ComputeSumAPerGroup(taskIdx, mIdx, kIdx, preProcessParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat,
                                           preciseType>::GetSumOrMaxAParamPerGroup(BinaryRepeatParams &param)
{
    param.dstBlkStride = 1;
    param.src0BlkStride = 1;
    param.src1BlkStride = 1;
    param.dstRepStride = FP32_MASK_BLK_NUM;
    param.src0RepStride = FP32_MASK_BLK_NUM * 2;
    param.src1RepStride = FP32_MASK_BLK_NUM * 2;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ComputeSumAPerGroup(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx,
                                      PreProcessParams &preProcessParams)
{
    uint32_t v1GroupNum = preProcessParams.realProcessM * v1BaseKGroupNum_; // tiing保证v1GroupNum小于255
    if (tiling_->groupSize == 128) {
        BinaryRepeatParams param;
        GetSumOrMaxAParamPerGroup(param);
        AscendC::Add<float, false>(aF32TmpTensor_, aF32Tensor_, aF32Tensor_[FP32_MAX_MASK_SIZE], FP32_MAX_MASK_SIZE,
                                   v1GroupNum, param);
        PipeBarrier<PIPE_V>();
        BlockReduceSum(aF32TmpTensor_, aF32TmpTensor_, v1GroupNum, FP32_MAX_MASK_SIZE, 1, 1, FP32_MASK_BLK_NUM);
    } else {
        BlockReduceSum(aF32TmpTensor_, aF32Tensor_, v1GroupNum, FP32_MAX_MASK_SIZE, 1, 1, FP32_MASK_BLK_NUM);
    }

    PipeBarrier<PIPE_V>();
    BlockReduceSum(aF32TmpTensor_, aF32TmpTensor_, CeilDiv(v1GroupNum, FP32_BLOCK_SIZE), FP32_MAX_MASK_SIZE, 1, 1,
                   VEC_REPEAT_MAX_STRIDE);

    PipeBarrier<PIPE_V>();
    Brcb(aSumTensor_, aF32TmpTensor_, CeilDiv(v1GroupNum, FP32_BLOCK_SIZE), {1, 8});

    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    // tiling保证 当K无法在单核全载时，realProcessM == 1, 行优先排列
    // 尾块前面一定是全块所以乘CeilDiv(static_cast<uint64_t>(tiling_->v1BaseK), tiling_->groupSize)
    uint64_t gmOffset = mIdx * tiling_->v1BaseM * groupNum_ * FP32_BLOCK_SIZE +
                        kIdx * CeilDiv(static_cast<uint64_t>(tiling_->v1BaseK), tiling_->groupSize) * FP32_BLOCK_SIZE;
    DataCopyPad2D(reduceSumWorkspaceGm_[gmOffset], aSumTensor_, preProcessParams.realProcessM,
                  v1BaseKGroupNum_ * FP32_BLOCK_SIZE, groupNum_ * FP32_BLOCK_SIZE);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ComputeSumAPerChannel(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx,
                                        PreProcessParams &preProcessParams)
{
    uint32_t numRepeatK = tiling_->v1BaseK / FP32_MAX_MASK_SIZE;
    LocalTensor<float> sumOnceSrcTensor = aF32Tensor_;
    LocalTensor<float> sumOnceDstTensor = aF32TmpTensor_;
    numRepeatK = ComputeSumAOnce(sumOnceDstTensor, sumOnceSrcTensor, numRepeatK, preProcessParams);
    PipeBarrier<PIPE_V>();

    uint32_t computeTimes = 1;
    while (numRepeatK > 1) {
        if ((computeTimes & 1) == 1) {
            sumOnceSrcTensor = aF32TmpTensor_;
            sumOnceDstTensor = aFp32ReduceTensor_;
        } else {
            sumOnceSrcTensor = aFp32ReduceTensor_;
            sumOnceDstTensor = aF32TmpTensor_;
        }
        numRepeatK = ComputeSumAOnce(sumOnceDstTensor, sumOnceSrcTensor, numRepeatK, preProcessParams);
        computeTimes++;
        PipeBarrier<PIPE_V>();
    }

    BlockReduceSum<float, false>(sumOnceDstTensor, sumOnceDstTensor, preProcessParams.realProcessM, FP32_MAX_MASK_SIZE,
                                 1, 1, FP32_MASK_BLK_NUM);
    PipeBarrier<PIPE_V>();
    BlockReduceSum(sumOnceDstTensor, sumOnceDstTensor, 1, FP32_BLOCK_SIZE * preProcessParams.realProcessM,
                   static_cast<uint16_t>(FP32_MASK_BLK_NUM), 1, FP32_MASK_BLK_NUM);
    PipeBarrier<PIPE_V>();
    // brcb一次只能处理8个block
    Brcb(aSumTensor_, sumOnceDstTensor, CeilAlign(preProcessParams.realProcessM, static_cast<uint16_t>(8)), {1, 8});

    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    DataCopyPad2D(
        reduceSumWorkspaceGm_[kIdx * tiling_->mSize * FP32_BLOCK_SIZE + (mIdx * tiling_->v1BaseM) * FP32_BLOCK_SIZE],
        aSumTensor_, preProcessParams.realProcessM, FP32_BLOCK_SIZE, FP32_BLOCK_SIZE);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline uint32_t Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ComputeSumAOnce(LocalTensor<float> &dst, LocalTensor<float> &src, uint32_t numRepeatK,
                                  PreProcessParams &preProcessParams)
{
    uint32_t numProcessK = numRepeatK >> 1;
    uint32_t nextRepeatK = numRepeatK - numProcessK;
    if (likely(numProcessK > 0)) {
        uint32_t offsetPostHalf = numProcessK * FP32_MAX_MASK_SIZE;

        AscendC::Add<float, false>(dst, src, src[offsetPostHalf], FP32_MAX_MASK_SIZE, numProcessK, commonRepeatParams_);
        uint32_t reduceKSize = nextRepeatK * FP32_MAX_MASK_SIZE;
        uint32_t fullKSize = numRepeatK * FP32_MAX_MASK_SIZE;
        for (uint16_t mLoopIdx = 1; mLoopIdx < preProcessParams.realProcessM; mLoopIdx++) {
            AscendC::Add<float, false>(dst[mLoopIdx * reduceKSize], src[mLoopIdx * fullKSize],
                                       src[mLoopIdx * fullKSize + offsetPostHalf], FP32_MAX_MASK_SIZE, numProcessK,
                                       commonRepeatParams_);
        }
    }
    ProcessReduceTail(dst, src, numRepeatK, nextRepeatK, numProcessK, preProcessParams);
    return nextRepeatK;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ComputeMaxA(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx, PreProcessParams &preProcessParams)
{
    SetMaskNorm();
    SetVectorMask<float, MaskMode::NORMAL>(FP32_MAX_MASK_SIZE);
    uint32_t numRepeatK = tiling_->v1BaseK / FP32_MAX_MASK_SIZE;
    AscendC::Abs<float, false>(aF32TmpTensor_, aF32Tensor_, FP32_MAX_MASK_SIZE, numRepeatK, commonUnaryRepeatParams_);
    for (uint16_t mLoopIdx = 1; mLoopIdx < preProcessParams.realProcessM; mLoopIdx++) {
        AscendC::Abs<float, false>(aF32TmpTensor_[mLoopIdx * tiling_->v1BaseK],
                                   aF32Tensor_[mLoopIdx * tiling_->v1BaseK], FP32_MAX_MASK_SIZE, numRepeatK,
                                   commonUnaryRepeatParams_);
    }

    PipeBarrier<PIPE_V>();
    if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
        ComputeMaxAPerChannel(taskIdx, mIdx, kIdx, preProcessParams, numRepeatK);
    } else {
        ComputeMaxAPerGroup(taskIdx, mIdx, kIdx, preProcessParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ComputeMaxAPerGroup(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx,
                                      PreProcessParams &preProcessParams)
{
    uint32_t v1GroupNum = preProcessParams.realProcessM * v1BaseKGroupNum_; // tiing保证v1GroupNum小于255
    if (tiling_->groupSize == 128) {
        BinaryRepeatParams param;
        GetSumOrMaxAParamPerGroup(param);
        AscendC::Max<float, false>(aF32TmpTensor_, aF32TmpTensor_, aF32TmpTensor_[FP32_MAX_MASK_SIZE],
                                   FP32_MAX_MASK_SIZE, v1GroupNum, param);
        PipeBarrier<PIPE_V>();
    }
    BlockReduceMax(aF32TmpTensor_, aF32TmpTensor_, v1GroupNum, FP32_MAX_MASK_SIZE, 1, 1, FP32_MASK_BLK_NUM);

    PipeBarrier<PIPE_V>();
    BlockReduceMax(aF32TmpTensor_, aF32TmpTensor_, CeilDiv(v1GroupNum, FP32_BLOCK_SIZE), FP32_MAX_MASK_SIZE, 1, 1,
                   VEC_REPEAT_MAX_STRIDE);

    PipeBarrier<PIPE_V>();
    Brcb(aMaxTensor_, aF32TmpTensor_, CeilDiv(v1GroupNum, FP32_BLOCK_SIZE), {1, 8});

    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    // tiling保证 当K无法在单核全载时，realProcessM == 1, 行优先排列
    // 尾快前面一定是全块所以乘CeilDiv(static_cast<uint64_t>(tiling_->v1BaseK), tiling_->groupSize)
    uint64_t gmOffset = mIdx * tiling_->v1BaseM * groupNum_ * FP32_BLOCK_SIZE +
                        kIdx * CeilDiv(static_cast<uint64_t>(tiling_->v1BaseK), tiling_->groupSize) * FP32_BLOCK_SIZE;
    DataCopyPad2D(reduceMaxWorkspaceGm_[gmOffset], aMaxTensor_, preProcessParams.realProcessM,
                  v1BaseKGroupNum_ * FP32_BLOCK_SIZE, groupNum_ * FP32_BLOCK_SIZE);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ComputeMaxAPerChannel(uint64_t taskIdx, uint64_t mIdx, uint64_t kIdx,
                                        PreProcessParams &preProcessParams, uint32_t numRepeatK)
{
    uint32_t computeTimes = 0;

    // 循环处理k方向reduce
    LocalTensor<float> maxOnceSrcTensor = aFp32ReduceTensor_;
    LocalTensor<float> maxOnceDstTensor = aF32TmpTensor_;
    while (numRepeatK > 1) {
        if ((computeTimes & 1) == 0) {
            maxOnceSrcTensor = aF32TmpTensor_;
            maxOnceDstTensor = aFp32ReduceTensor_;
        } else {
            maxOnceSrcTensor = aFp32ReduceTensor_;
            maxOnceDstTensor = aF32TmpTensor_;
        }
        numRepeatK = ComputeMaxAOnce(maxOnceDstTensor, maxOnceSrcTensor, numRepeatK, preProcessParams);
        computeTimes++;
        PipeBarrier<PIPE_V>();
    }

    BlockReduceMax<float, false>(maxOnceDstTensor, maxOnceDstTensor, preProcessParams.realProcessM, FP32_MAX_MASK_SIZE,
                                 1, 1, FP32_MASK_BLK_NUM);
    PipeBarrier<PIPE_V>();
    BlockReduceMax(maxOnceDstTensor, maxOnceDstTensor, 1, FP32_BLOCK_SIZE * preProcessParams.realProcessM,
                   FP32_MASK_BLK_NUM, 1, FP32_MASK_BLK_NUM);
    PipeBarrier<PIPE_V>();
    // brcb一次只能处理8个block
    Brcb(aMaxTensor_, maxOnceDstTensor, CeilAlign(preProcessParams.realProcessM, static_cast<uint16_t>(8)), {1, 8});
    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    DataCopyPad2D(
        reduceMaxWorkspaceGm_[kIdx * tiling_->mSize * FP32_BLOCK_SIZE + mIdx * tiling_->v1BaseM * FP32_BLOCK_SIZE],
        aMaxTensor_, preProcessParams.realProcessM, FP32_BLOCK_SIZE, FP32_BLOCK_SIZE);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline uint32_t Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ComputeMaxAOnce(LocalTensor<float> &dst, LocalTensor<float> &src, uint32_t numRepeatK,
                                  PreProcessParams &preProcessParams)
{
    uint32_t numProcessK = numRepeatK >> 1;
    uint32_t nextRepeatK = numRepeatK - numProcessK;
    if (unlikely(numProcessK == 0)) {
        return nextRepeatK;
    }
    uint32_t offsetPostHalf = numProcessK * FP32_MAX_MASK_SIZE;

    AscendC::Max<float, false>(dst, src, src[offsetPostHalf], FP32_MAX_MASK_SIZE, numProcessK, commonRepeatParams_);
    uint32_t reduceKSize = nextRepeatK * FP32_MAX_MASK_SIZE;
    uint32_t fullKSize = numRepeatK * FP32_MAX_MASK_SIZE;
    for (uint16_t mLoopIdx = 1; mLoopIdx < preProcessParams.realProcessM; mLoopIdx++) {
        AscendC::Max<float, false>(dst[mLoopIdx * reduceKSize], src[mLoopIdx * fullKSize],
                                   src[mLoopIdx * fullKSize + offsetPostHalf], FP32_MAX_MASK_SIZE, numProcessK,
                                   commonRepeatParams_);
    }
    ProcessReduceTail(dst, src, numRepeatK, nextRepeatK, numProcessK, preProcessParams);
    return nextRepeatK;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ProcessReduceTail(LocalTensor<float> &dst, LocalTensor<float> &src, uint32_t numRepeatK,
                                    uint32_t nextRepeatK, uint32_t numProcessK, PreProcessParams &preProcessParams)
{
    if ((numRepeatK & 1) == 0) {
        return;
    }
    // 有尾块，额外引入一次DataCopy
    CopyRepeatParams params;
    params.srcStride = 1;
    params.dstStride = 1;
    params.dstRepeatSize = nextRepeatK * FP32_MASK_BLK_NUM;
    params.srcRepeatSize = numRepeatK * FP32_MASK_BLK_NUM;
    AscendC::Copy<float, false>(dst[numProcessK * FP32_MAX_MASK_SIZE], src[(numProcessK << 1) * FP32_MAX_MASK_SIZE],
                                FP32_MAX_MASK_SIZE, preProcessParams.realProcessM, params);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::UnfoldAMatrix(uint64_t mIdx, uint64_t kIdx, PreProcessParams &preProcessParams)
{
    // 避免bank冲突，tensor使用时每次额外往前偏移一个repeat的数据量
    uint32_t defaultOffset = multiScaleTimes_ * FP32_MAX_MASK_SIZE;
    uint32_t mainRepeatK = tiling_->v1BaseK / FP32_MAX_MASK_SIZE;
    BinaryRepeatParams f32BinaryRepeatParams;
    UnaryRepeatParams f32UnaryRepeatParams;
    UnaryRepeatParams f32ToF16RepeatParams;
    SetRepeatParams(f32BinaryRepeatParams, f32UnaryRepeatParams, f32ToF16RepeatParams);
    float multiFactors[3] = {multiFactors_[0], multiFactors_[1], multiFactors_[2]};
    for (uint64_t unfoldATimes = 0; unfoldATimes < multiScaleTimes_;
         unfoldATimes++, defaultOffset -= FP32_MAX_MASK_SIZE) {
        TransToWType(unfoldATimes, defaultOffset, mainRepeatK, multiFactors[unfoldATimes], f32BinaryRepeatParams,
                     f32UnaryRepeatParams, f32ToF16RepeatParams, preProcessParams);

        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        uint64_t gmOffset =
            kIdx * aUnfoldSizeWtype_ + (mIdx * tiling_->v1BaseM + unfoldATimes * tiling_->mSize) * tiling_->v1BaseK;
        DataCopyPad2D(aUnfoldGlobal_[gmOffset],
                      aUnfoldLocalTensor_[unfoldATimes * preProcessParams.realProcessM * tiling_->v1BaseK],
                      preProcessParams.realProcessM, preProcessParams.realProcessK, tiling_->v1BaseK, tiling_->v1BaseK);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::TransToWTypeFirstStep(uint64_t defaultOffset, uint64_t mainRepeatK,
                                        const BinaryRepeatParams &f32BinaryRepeatParams,
                                        const PreProcessParams &preProcessParams)
{
    if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
        Div(aF32TmpTensor_[defaultOffset], aF32Tensor_, aMaxTensor_, FP32_MAX_MASK_SIZE, mainRepeatK,
            f32BinaryRepeatParams);
        for (uint16_t mLoopIdx = 1; mLoopIdx < preProcessParams.realProcessM; mLoopIdx++) {
            AscendC::Div<float, false>(
                aF32TmpTensor_[defaultOffset + mLoopIdx * tiling_->v1BaseK], aF32Tensor_[mLoopIdx * tiling_->v1BaseK],
                aMaxTensor_[mLoopIdx * FP32_BLOCK_SIZE], FP32_MAX_MASK_SIZE, mainRepeatK, f32BinaryRepeatParams);
        }
    } else {
        uint32_t v1GroupNum = preProcessParams.realProcessM * v1BaseKGroupNum_;
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = tiling_->groupSize / FP32_MASK_BLK_NUM; // float类型，8个block构成一个repeat
        repeatParams.src0RepStride = repeatParams.dstRepStride;
        repeatParams.src1RepStride = 1;
        AscendC::Div(aF32TmpTensor_[defaultOffset], aF32Tensor_, aMaxTensor_, FP32_MAX_MASK_SIZE, v1GroupNum,
                     repeatParams);
        if (tiling_->groupSize == 128) {
            AscendC::Div(aF32TmpTensor_[defaultOffset + 64], aF32Tensor_[64], aMaxTensor_, FP32_MAX_MASK_SIZE,
                         v1GroupNum, repeatParams);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::TransToWType(uint64_t unfoldATimes, uint64_t defaultOffset, uint64_t mainRepeatK, float multiFactors,
                               const BinaryRepeatParams &f32BinaryRepeatParams,
                               const UnaryRepeatParams &f32UnaryRepeatParams,
                               const UnaryRepeatParams &f32ToF16RepeatParams, const PreProcessParams &preProcessParams)
{
    if (likely(unfoldATimes > 0)) {
        Sub(aF32TmpTensor_[defaultOffset], aF32Tensor_, aF32TmpTensor_[defaultOffset + FP32_MAX_MASK_SIZE],
            FP32_MAX_MASK_SIZE, mainRepeatK, commonRepeatParams_);
        for (uint16_t mLoopIdx = 1; mLoopIdx < preProcessParams.realProcessM; mLoopIdx++) {
            AscendC::Sub<float, false>(aF32TmpTensor_[defaultOffset + mLoopIdx * tiling_->v1BaseK],
                                       aF32Tensor_[mLoopIdx * tiling_->v1BaseK],
                                       aF32TmpTensor_[defaultOffset + FP32_MAX_MASK_SIZE + mLoopIdx * tiling_->v1BaseK],
                                       FP32_MAX_MASK_SIZE, mainRepeatK, commonRepeatParams_);
        }
    } else {
        TransToWTypeFirstStep(defaultOffset, mainRepeatK, f32BinaryRepeatParams, preProcessParams);
    }
    PipeBarrier<PIPE_V>();
    for (uint16_t mLoopIdx = 0; mLoopIdx < preProcessParams.realProcessM; mLoopIdx++) {
        Muls<float, false>(aF32Tensor_[mLoopIdx * tiling_->v1BaseK],
                           aF32TmpTensor_[defaultOffset + mLoopIdx * tiling_->v1BaseK], multiFactors,
                           FP32_MAX_MASK_SIZE, mainRepeatK, f32UnaryRepeatParams);
    }
    PipeBarrier<PIPE_V>();

    for (uint16_t mLoopIdx = 0; mLoopIdx < preProcessParams.realProcessM; mLoopIdx++) {
        AscendC::Cast<float, float, false>(aF32TmpTensor_[defaultOffset + mLoopIdx * tiling_->v1BaseK],
                                           aF32Tensor_[mLoopIdx * tiling_->v1BaseK], RoundMode::CAST_ROUND,
                                           FP32_MAX_MASK_SIZE, mainRepeatK, f32UnaryRepeatParams);
    }
    PipeBarrier<PIPE_V>();

    for (uint16_t mLoopIdx = 0; mLoopIdx < preProcessParams.realProcessM; mLoopIdx++) {
        AscendC::Cast<half, float, false>(aFp16Tensor_[mLoopIdx * tiling_->v1BaseK],
                                          aF32TmpTensor_[defaultOffset + mLoopIdx * tiling_->v1BaseK],
                                          RoundMode::CAST_NONE, FP32_MAX_MASK_SIZE, mainRepeatK, f32ToF16RepeatParams);
    }
    PipeBarrier<PIPE_V>();
    Cast(aUnfoldLocalTensor_[unfoldATimes * preProcessParams.realProcessM * tiling_->v1BaseK], aFp16Tensor_,
         RoundMode::CAST_NONE, preProcessParams.realProcessM * tiling_->v1BaseK);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::SetRepeatParams(BinaryRepeatParams &repeatParams, UnaryRepeatParams &f32UnaryRepeatParams,
                                  UnaryRepeatParams &f32ToF16RepeatParams)
{
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = FP32_MASK_BLK_NUM;
    repeatParams.src0RepStride = FP32_MASK_BLK_NUM;
    repeatParams.src1RepStride = 0;

    f32UnaryRepeatParams.dstBlkStride = 1;
    f32UnaryRepeatParams.srcBlkStride = 1;
    f32UnaryRepeatParams.dstRepStride = FP32_MASK_BLK_NUM;
    f32UnaryRepeatParams.srcRepStride = FP32_MASK_BLK_NUM;

    f32ToF16RepeatParams.dstBlkStride = 1;
    f32ToF16RepeatParams.srcBlkStride = 1;
    f32ToF16RepeatParams.dstRepStride = FP32_MAX_MASK_SIZE / FP16_BLOCK_SIZE;
    f32ToF16RepeatParams.srcRepStride = FP32_MASK_BLK_NUM;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat,
                                           preciseType>::BL1PreLoad(const CubeProcessParams<wType> &cubeProcessParams)
{
    uint64_t kBlockNum = InitMatmulParams();
    uint64_t preloadTimes = 0;
    MatmulTaskLoopParams matmulTaskLoopParams;
    for (uint64_t taskIdx = taskStartIdx_; taskIdx < taskLimit_; taskIdx++) {
        // 预处理优先处理k方向
        matmulTaskLoopParams.kOffset = taskIdx % kBlockNum * tiling_->matmulTiling.singleCoreK;
        matmulTaskLoopParams.nOffset = taskIdx / kBlockNum * cubeSingleCoreN_;
        matmulTaskLoopParams.singleCoreK =
            matmulTaskLoopParams.kOffset + tiling_->matmulTiling.singleCoreK > tiling_->kSize ?
                tiling_->kSize - matmulTaskLoopParams.kOffset :
                tiling_->matmulTiling.singleCoreK;
        if (unlikely(matmulTaskLoopParams.nOffset >= curTaskNSize_)) {
            continue;
        }

        // n方向遍历
        uint64_t singleCoreRealN = matmulTaskLoopParams.nOffset + cubeSingleCoreN_ > curTaskNSize_ ?
                                       curTaskNSize_ - matmulTaskLoopParams.nOffset :
                                       cubeSingleCoreN_;
        for (uint64_t singleCoreNOffset = 0; singleCoreNOffset < singleCoreRealN;
             singleCoreNOffset += tiling_->matmulTiling.baseN) {
            uint32_t l1RealN = singleCoreNOffset + tiling_->matmulTiling.baseN > singleCoreRealN ?
                                   singleCoreRealN - singleCoreNOffset :
                                   tiling_->matmulTiling.baseN;
            // ka方向遍历
            for (uint64_t kaOffset = 0; kaOffset < matmulTaskLoopParams.singleCoreK;
                 kaOffset += cubeProcessParams.l1BaseKa) {
                uint64_t l1RealKa = kaOffset + cubeProcessParams.l1BaseKa > matmulTaskLoopParams.singleCoreK ?
                                        matmulTaskLoopParams.singleCoreK - kaOffset :
                                        cubeProcessParams.l1BaseKa;
                // stepKa/stepKb方向遍历
                uint64_t realStepKa = CeilDiv(l1RealKa, static_cast<uint64_t>(tiling_->matmulTiling.baseK));
                for (uint64_t stepKbOffset = 0; stepKbOffset < realStepKa;
                     stepKbOffset += tiling_->matmulTiling.stepKb) {
                    BL1DmaCopy(kaOffset, stepKbOffset, l1RealKa, singleCoreNOffset, l1RealN, cubeProcessParams,
                               matmulTaskLoopParams);
                    preloadTimes++;
                    // 经验值，预载入次数保持10次即可
                    if (preloadTimes >= 10) {
                        return;
                    }
                }
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::BL1DmaCopy(uint64_t kaOffset, uint64_t stepKbOffset, uint64_t l1RealKa, uint64_t singleCoreNOffset,
                             uint64_t l1RealN, const CubeProcessParams<wType> &cubeProcessParams,
                             const MatmulTaskLoopParams &matmulTaskLoopParams)
{
    uint64_t singleCoreKbOffset = kaOffset + stepKbOffset * tiling_->matmulTiling.baseK;
    uint64_t l1RealKb = stepKbOffset * tiling_->matmulTiling.baseK + cubeProcessParams.l1BaseKb > l1RealKa ?
                            l1RealKa - stepKbOffset * tiling_->matmulTiling.baseK :
                            cubeProcessParams.l1BaseKb;
    DataCopyParams dmaParams;
    uint64_t bOffset = 0;
    uint64_t nOffset = singleCoreNOffset + matmulTaskLoopParams.nOffset;
    if constexpr (weightFormat != CubeFormat::NZ) {
        if constexpr (bTrans) {
            dmaParams.blockCount = l1RealN;
            dmaParams.blockLen = l1RealKb / ONE_BLK_SIZE;
            dmaParams.srcStride = tiling_->kSize / ONE_BLK_SIZE - dmaParams.blockLen;
            bOffset = nOffset * tiling_->kSize + matmulTaskLoopParams.kOffset + singleCoreKbOffset;
        } else {
            dmaParams.blockCount = l1RealKb;
            dmaParams.blockLen = l1RealN / ONE_BLK_SIZE;
            dmaParams.srcStride = tiling_->nSize / ONE_BLK_SIZE - dmaParams.blockLen;
            bOffset = matmulTaskLoopParams.kOffset * tiling_->nSize + nOffset;
            if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
                bOffset = (curGroupId_ * tiling_->groupSize + matmulTaskLoopParams.kOffset) * tiling_->nSize + nOffset;
            }
        }
        dmaParams.dstStride = 0;
        if constexpr (IsSameType<wType, int4b_t>::value) {
            dmaParams.blockLen = dmaParams.blockLen >> 1;
            dmaParams.srcStride = dmaParams.srcStride >> 1;
            bOffset = bOffset >> 1;
        }
        DataCopy(l1TBuf_.Get<wType>().template ReinterpretCast<int8_t>(), wGlobal_[bOffset], dmaParams);
    } else { // weightNZ msd splitK 暂时只支持perchannel，int8
        uint64_t kOffset = matmulTaskLoopParams.kOffset + singleCoreKbOffset;
        CopyInBL1Nz(l1TBuf_.Get<wType>(), nOffset, kOffset, l1RealN, l1RealKb);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::LaunchMatmul(const CubeProcessParams<wType> &cubeProcessParams,
                               SyncProcessor<HardEvent::MTE1_MTE2> &al1SyncProcessor,
                               SyncProcessor<HardEvent::MTE1_MTE2> &bl1SyncProcessor)
{
    // msd方案暂时只实现stepKa >= stepKb模板
    uint64_t kBlockNum = InitMatmulParams();
    kSigncoreTaskNum_ = taskLimit_ % kBlockNum - taskStartIdx_ % kBlockNum;
    MatmulTaskLoopParams matmulTaskLoopParams;
    for (taskIdx_ = taskStartIdx_; taskIdx_ < taskLimit_; taskIdx_++) {
        // 预处理优先处理k方向
        matmulTaskLoopParams.kIdx = taskIdx_ % kBlockNum;
        matmulTaskLoopParams.kOffset = matmulTaskLoopParams.kIdx * tiling_->matmulTiling.singleCoreK;
        matmulTaskLoopParams.nOffset = taskIdx_ / kBlockNum * cubeSingleCoreN_;
        matmulTaskLoopParams.singleCoreK =
            matmulTaskLoopParams.kOffset + tiling_->matmulTiling.singleCoreK > tiling_->kSize ?
                tiling_->kSize - matmulTaskLoopParams.kOffset :
                tiling_->matmulTiling.singleCoreK;

        if (unlikely(matmulTaskLoopParams.nOffset >= curTaskNSize_)) {
            continue;
        }
        // m方向遍历
        for (uint64_t mOffset = 0; mOffset < tiling_->matmulTiling.M; mOffset += tiling_->matmulTiling.baseM) {
            uint32_t l1RealM = mOffset + tiling_->matmulTiling.baseM > tiling_->matmulTiling.M ?
                                   tiling_->matmulTiling.M - mOffset :
                                   tiling_->matmulTiling.baseM;

            // n方向遍历
            uint64_t singleCoreRealN = matmulTaskLoopParams.nOffset + cubeSingleCoreN_ > curTaskNSize_ ?
                                           curTaskNSize_ - matmulTaskLoopParams.nOffset :
                                           cubeSingleCoreN_;
            for (uint64_t singleCoreNOffset = 0; singleCoreNOffset < singleCoreRealN;
                 singleCoreNOffset += tiling_->matmulTiling.baseN) {
                uint32_t l1RealN = singleCoreNOffset + tiling_->matmulTiling.baseN > singleCoreRealN ?
                                       singleCoreRealN - singleCoreNOffset :
                                       tiling_->matmulTiling.baseN;
                IterateAllSingleCoreK(singleCoreNOffset, l1RealM, l1RealN, mOffset, cubeProcessParams,
                                      matmulTaskLoopParams, al1SyncProcessor, bl1SyncProcessor);
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::IterateAllSingleCoreK(uint64_t singleCoreNOffset, uint64_t l1RealM, uint32_t l1RealN,
                                        uint64_t mOffset, const CubeProcessParams<wType> &cubeProcessParams,
                                        const MatmulTaskLoopParams &matmulTaskLoopParams,
                                        SyncProcessor<HardEvent::MTE1_MTE2> &al1SyncProcessor,
                                        SyncProcessor<HardEvent::MTE1_MTE2> &bl1SyncProcessor)
{
    uint64_t l1RealMAlign = CeilDiv(l1RealM, static_cast<uint64_t>(BLOCK_CUBE)) * BLOCK_CUBE;
    mmObj_.SetOrgShape(tiling_->mSize, tiling_->nSize, cubeProcessParams.l1BaseKa, cubeProcessParams.l1BaseKb,
                       curTaskNSize_);
    // ka方向遍历
    for (uint64_t kaOffset = 0; kaOffset < matmulTaskLoopParams.singleCoreK; kaOffset += cubeProcessParams.l1BaseKa) {
        uint64_t l1RealKa = kaOffset + cubeProcessParams.l1BaseKa > matmulTaskLoopParams.singleCoreK ?
                                matmulTaskLoopParams.singleCoreK - kaOffset :
                                cubeProcessParams.l1BaseKa;
        CopyInAL1(cubeProcessParams, matmulTaskLoopParams.kIdx, mOffset, kaOffset, l1RealKa, l1RealM, al1SyncProcessor);

        // stepKa/stepKb方向遍历
        uint64_t realStepKa = CeilDiv(l1RealKa, static_cast<uint64_t>(tiling_->matmulTiling.baseK));
        for (uint64_t stepKbOffset = 0; stepKbOffset < realStepKa; stepKbOffset += tiling_->matmulTiling.stepKb) {
            uint64_t singleCoreKbOffset = kaOffset + stepKbOffset * tiling_->matmulTiling.baseK;
            uint64_t l1RealKb = stepKbOffset * tiling_->matmulTiling.baseK + cubeProcessParams.l1BaseKb > l1RealKa ?
                                    l1RealKa - stepKbOffset * tiling_->matmulTiling.baseK :
                                    cubeProcessParams.l1BaseKb;
            bl1SyncProcessor.WaitSyncFlag();
            LocalTensor<wType> bL1Tensor =
                bl1SyncProcessor.GetBufferId() == 0 ? cubeProcessParams.bL1TensorPing : cubeProcessParams.bL1TensorPong;
            CopyInBL1(bL1Tensor, curTaskNOffset_ + singleCoreNOffset + matmulTaskLoopParams.nOffset,
                      matmulTaskLoopParams.kOffset + singleCoreKbOffset, l1RealN, l1RealKb);

            TEventID eventIdMte2ToMte1 = GetTPipePtr()->FetchEventID<HardEvent::MTE2_MTE1>();
            SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);
            WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMte1);

            uint64_t aTensorOffset = l1RealMAlign * (stepKbOffset * tiling_->matmulTiling.baseK);
            if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
                aTensorOffset =
                    (matmulTaskLoopParams.kIdx * tiling_->groupSize + stepKbOffset * tiling_->matmulTiling.baseK) *
                        (CeilDiv(tiling_->matmulTiling.M, static_cast<int32_t>(BLOCK_CUBE)) * BLOCK_CUBE) +
                    mOffset;
            }
            mmObj_.SetTensorA(aL1GroupPackTensor_[aTensorOffset], aTrans);
            mmObj_.SetTensorB(bL1Tensor, bTrans);
            mmObj_.SetTail(l1RealM, l1RealN, l1RealKb);
            mmObj_.Iterate(kaOffset != 0 || stepKbOffset != 0);
            if (unlikely(kaOffset + cubeProcessParams.l1BaseKa >= matmulTaskLoopParams.singleCoreK &&
                         stepKbOffset + tiling_->matmulTiling.stepKb >= realStepKa)) {
                // c矩阵的offset计算公式:cube单次循环size * pingpong份数量 + 切kId * matmul一次计算的size + m偏移 * n
                // +n偏移
                mmObj_.GetTensorC(
                    cUnfoldGlobal_[cUnfoldSize_ * (cubeLoopIdx_ % c1c2CacheNum_) +
                                   matmulTaskLoopParams.kIdx * multiScaleTimes_ * tiling_->mSize * curTaskNSize_ +
                                   mOffset * curTaskNSize_ + singleCoreNOffset + matmulTaskLoopParams.nOffset]);
            }
            mmObj_.End();
            bl1SyncProcessor.SetSyncFlag();
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::GetCopyInAL1Params(Nd2NzParams &nd2nzParams, uint64_t &kOffset, uint64_t l1BaseKa, uint64_t l1BaseM)
{
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = l1BaseM;
    if constexpr (IsSameType<wType, int4b_t>::value) {
        nd2nzParams.dValue = l1BaseKa >> 1;
        nd2nzParams.srcDValue = tiling_->v1BaseK >> 1; // 前处理按照v1BaseK连续一行放置
        kOffset = kOffset >> 1;
    } else {
        nd2nzParams.dValue = l1BaseKa;
        nd2nzParams.srcDValue = tiling_->v1BaseK;
    }

    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = CeilDiv(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE)) * BLOCK_CUBE;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyInAL1(const CubeProcessParams<wType> &cubeProcessParams, uint64_t kIdx, uint64_t mOffset,
                            uint64_t kOffset, uint64_t l1BaseKa, uint64_t l1BaseM,
                            SyncProcessor<HardEvent::MTE1_MTE2> &al1SyncProcessor)
{
    if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
        if (taskIdx_ > taskStartIdx_) { // perGroup场景A矩阵按照groupPack维度加载
            return;
        } else {
            l1BaseKa = realGroupPack_ * tiling_->groupSize;
            l1BaseM = CeilDiv(tiling_->matmulTiling.M, static_cast<int32_t>(BLOCK_CUBE)) * BLOCK_CUBE;
            kIdx = 0; // 每个核把所有groupPack全部加载进去
        }
    }
    aL1GroupPackTensor_ =
        al1SyncProcessor.GetBufferId() == 0 ? cubeProcessParams.aL1TensorPing : cubeProcessParams.aL1TensorPong;

    Nd2NzParams nd2nzParams;
    GetCopyInAL1Params(nd2nzParams, kOffset, l1BaseKa, l1BaseM);

    uint64_t aOffset = kIdx * aUnfoldSizeS8_ + mOffset * nd2nzParams.srcDValue + kOffset;
    if (antiQuantType == Mc2QuantType::PER_GROUP) {
        uint64_t curKOffset = (curGroupId_ + kIdx) * tiling_->groupSize;
        if constexpr (IsSameType<wType, int4b_t>::value) {
            aOffset = curKOffset / tiling_->v1BaseK * aUnfoldSizeS8_ + ((mOffset * tiling_->v1BaseK) >> 1) +
                      ((curKOffset % tiling_->v1BaseK) >> 1);
        } else {
            aOffset = curKOffset / tiling_->v1BaseK * aUnfoldSizeS8_ + mOffset * tiling_->v1BaseK +
                      curKOffset % tiling_->v1BaseK;
        }
    }
    if (unlikely(cubeFirstLoop_ || aOffset != lastAl1Offset_)) {
        if (unlikely(!cubeFirstLoop_)) {
            al1SyncProcessor.SetSyncFlag();
        }
        al1SyncProcessor.WaitSyncFlag();
        DataCopy(aL1GroupPackTensor_.template ReinterpretCast<int8_t>(), aUnfoldGlobalInt8_[aOffset], nd2nzParams);
        lastAl1Offset_ = aOffset;
        cubeFirstLoop_ = false;
        return;
    }
    return;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyInBL1(const LocalTensor<wType> &bL1Tensor, uint64_t nOffset, uint64_t kOffset, uint64_t l1RealN,
                            uint64_t l1BaseKb)
{
    if constexpr (weightFormat != CubeFormat::NZ) {
        CopyInBL1Nd(bL1Tensor, nOffset, kOffset, l1RealN, l1BaseKb);
    } else {
        CopyInBL1Nz(bL1Tensor, nOffset, kOffset, l1RealN, l1BaseKb);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyInBL1Nd(const LocalTensor<wType> &bL1Tensor, uint64_t nOffset, uint64_t kOffset, uint64_t l1RealN,
                              uint64_t l1BaseKb)
{
    uint64_t bOffset = 0;
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    if constexpr (bTrans) {
        nd2nzParams.nValue = l1RealN;
        nd2nzParams.dValue = l1BaseKb;
        nd2nzParams.srcDValue = tiling_->kSize;
        bOffset = nOffset * nd2nzParams.srcDValue + kOffset;
    } else {
        nd2nzParams.nValue = l1BaseKb;
        nd2nzParams.dValue = l1RealN;
        nd2nzParams.srcDValue = tiling_->nSize;
        bOffset = kOffset * nd2nzParams.srcDValue + nOffset;
        if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
            bOffset = (curGroupId_ * tiling_->groupSize + kOffset) * nd2nzParams.srcDValue + nOffset;
        }
    }

    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    if constexpr (IsSameType<wType, int4b_t>::value) {
        nd2nzParams.dValue = nd2nzParams.dValue >> 1;
        nd2nzParams.srcDValue = nd2nzParams.srcDValue >> 1;
        bOffset = bOffset >> 1;
    }
    uint16_t nValueAlignSize = static_cast<uint16_t>(BLOCK_CUBE);
    if constexpr (!bTrans) {
        // 不转置场景下，根据load2d transpose指令要求，需要控制mte2上尾块数据间隔
        if constexpr (IsSameType<wType, int4b_t>::value) {
            nValueAlignSize = INT4_BLOCK_SIZE;
        } else {
            nValueAlignSize = INT8_BLOCK_SIZE;
        }
    }
    nd2nzParams.dstNzC0Stride = CeilDiv(nd2nzParams.nValue, nValueAlignSize) * nValueAlignSize;
    DataCopy(bL1Tensor.template ReinterpretCast<int8_t>(), wGlobal_[bOffset], nd2nzParams);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyInBL1Nz(const LocalTensor<wType> &bL1Tensor, uint64_t nOffset, uint64_t kOffset, uint64_t l1RealN,
                              uint64_t l1BaseKb)
{
    uint64_t bOffset = 0;
    DataCopyParams dmaParams;
    if constexpr (bTrans) {
        dmaParams.blockCount = l1BaseKb / ONE_BLK_SIZE;
        dmaParams.blockLen = l1RealN;
        dmaParams.srcStride = tiling_->nSize - l1RealN;
        bOffset = tiling_->nSize * kOffset + nOffset * ONE_BLK_SIZE;
        // NZ trans，n分核后可能搬移不满一个分形，dst需要补齐到16上对齐
        dmaParams.dstStride = CeilAlign(l1RealN, 16) - l1RealN;
    } else {
        uint64_t nzWCnt = ONE_BLK_SIZE;
        if constexpr (IsSameType<wType, int4b_t>::value) {
            nzWCnt = INT4_BLOCK_SIZE;
        }
        bOffset = tiling_->kSize * nOffset + kOffset * nzWCnt;
        if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
            // tiling 保证noffset, koffset分别按照32B和16对齐
            uint64_t tilingKSizeNz = CeilAlign(tiling_->kSize, 16UL);
            kOffset = curGroupId_ * tiling_->groupSize + kOffset;
            bOffset = tilingKSizeNz * nOffset + kOffset * nzWCnt;
        }
        if constexpr (IsSameType<wType, int4b_t>::value) {
            bOffset = bOffset >> 1;
        }
        dmaParams.blockCount = l1RealN / nzWCnt;
        dmaParams.blockLen = l1BaseKb;
        dmaParams.srcStride = tiling_->kSize - l1BaseKb;
        dmaParams.dstStride = 0;
    }
    DataCopy(bL1Tensor.template ReinterpretCast<int8_t>(), wGlobal_[bOffset], dmaParams);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat,
                                           preciseType>::SumMaxMul(uint64_t taskSingleCoreNSize, uint64_t realM)
{
    uint64_t repeatTimes = tiling_->taskSingleCoreNSize / FP32_MAX_MASK_SIZE;
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = FP32_MASK_BLK_NUM;
    repeatParams.src0RepStride = 0;
    repeatParams.src1RepStride = FP32_MASK_BLK_NUM;

    uint32_t kCnt = 0;
    uint32_t sumOffset = 0;
    uint32_t scaleOffset = 0;
    if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
        kCnt = kBlockNum_;
    } else {
        kCnt = realGroupPack_;
    }
    for (uint64_t kIdx = 0; kIdx < kCnt; kIdx++) {
        for (uint32_t idxM = 0; idxM < realM; idxM++) {
            if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
                sumOffset = (kIdx * realM + idxM) * FP32_BLOCK_SIZE;
            } else {
                sumOffset = idxM * realGroupPack_ * FP32_BLOCK_SIZE + kIdx * FP32_BLOCK_SIZE;
                scaleOffset = kIdx * taskSingleCoreNSize;
            }
            AscendC::MulAddDst<float, float, false>(
                cF32Tensor_[idxM * tiling_->taskSingleCoreNSize], aSumComputeOffsetTensor_[sumOffset],
                scaleOffsetProductTensor_[scaleOffset], FP32_MAX_MASK_SIZE, repeatTimes, repeatParams);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ComputeOffsetMn(uint64_t taskSingleCoreNSize, uint64_t taskNSize, uint64_t nOffset,
                                  uint64_t baseMOffset, uint64_t realM)
{
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

    uint64_t gmOffset = curGroupId_ * tiling_->nSize + nOffset; // perchannel下curGroupId_为0
    DataCopyPad2D(antiquantScaleF16Tensor_, antiQuantScaleGlobal_[gmOffset], realGroupPack_, taskSingleCoreNSize,
                  tiling_->nSize);
    if constexpr (hasAntiQuantOffset) {
        DataCopyPad2D(antiquantOffsetF16Tensor_, antiQuantOffsetGlobal_[gmOffset], realGroupPack_, taskSingleCoreNSize,
                      tiling_->nSize);
    }

    PipeBarrier<PIPE_V>();
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    uint64_t repeatTimesGp = CeilDiv(realGroupPack_ * taskSingleCoreNSize, static_cast<uint64_t>(FP32_MAX_MASK_SIZE));
    AscendC::Cast<float, xType, false>(antiquantScaleF32Tensor_, antiquantScaleF16Tensor_, RoundMode::CAST_NONE,
                                       FP32_MAX_MASK_SIZE, repeatTimesGp, fp16ToFp32UnaryRepeatParams_);
    if constexpr (!hasAntiQuantOffset) {
        return;
    }

    AscendC::Cast<float, xType, false>(antiquantOffsetF32Tensor_, antiquantOffsetF16Tensor_, RoundMode::CAST_NONE,
                                       FP32_MAX_MASK_SIZE, repeatTimesGp, fp16ToFp32UnaryRepeatParams_);
    PipeBarrier<PIPE_V>();

    AscendC::Mul<float, false>(scaleOffsetProductTensor_, antiquantScaleF32Tensor_, antiquantOffsetF32Tensor_,
                               FP32_MAX_MASK_SIZE, repeatTimesGp, commonRepeatParams_);
    PipeBarrier<PIPE_V>();

    SumMaxMul(taskSingleCoreNSize, realM);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat,
                                           preciseType>::ProcessC1C2(uint64_t taskSingleCoreNSize, uint64_t nOffset,
                                                                     uint64_t baseMOffset, uint64_t singleCorerealM)
{
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

    uint64_t nRepeatTimes = tiling_->taskSingleCoreNSize / FP32_MAX_MASK_SIZE;

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = FP32_MASK_BLK_NUM;
    repeatParams.src0RepStride = FP32_MASK_BLK_NUM;
    repeatParams.src1RepStride = 0;

    if constexpr (IsSameType<preciseType, HighPreciseType>::value) {
        SyncProcessor<HardEvent::V_MTE2> c1c2SyncProcessor;
        c1c2SyncProcessor.Init(DOUBLE_BUFFER_NUM);
        uint64_t kCnt = 0;
        uint64_t scaleOffset = 0;
        if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
            kCnt = kBlockNum_;
        } else {
            kCnt = realGroupPack_;
        }
        for (uint64_t kIdx = 0; kIdx < kCnt; kIdx++) {
            if constexpr (antiQuantType == Mc2QuantType::PER_GROUP) {
                scaleOffset = kIdx * taskSingleCoreNSize;
            }

            for (uint64_t mOffset = 0; mOffset < singleCorerealM; mOffset += tiling_->postProcessBaseM) {
                UnfoldCMatrixParams params = {taskSingleCoreNSize, nOffset,     kIdx,         baseMOffset, mOffset,
                                              singleCorerealM,     scaleOffset, nRepeatTimes, repeatParams};
                ProcessUnfoldCMatrix(params, c1c2SyncProcessor);
                PipeBarrier<PIPE_V>();
            }
        }
        c1c2SyncProcessor.Destory();
    } else {
        ProcessC1C2PerGroup(taskSingleCoreNSize, nOffset, baseMOffset, singleCorerealM, nRepeatTimes, repeatParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ProcessUnfoldCMatrix(const UnfoldCMatrixParams &params,
                                       SyncProcessor<HardEvent::V_MTE2> &c1c2SyncProcessor)
{
    uint64_t kIdx = params.kIdx;
    uint64_t mOffset = params.mOffset;

    uint64_t realM = mOffset + tiling_->postProcessBaseM > params.singleCorerealM ? params.singleCorerealM - mOffset :
                                                                                    tiling_->postProcessBaseM;
    if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
        float divFactors[3] = {divFactors_[0], divFactors_[1], divFactors_[2]};
        uint64_t repeatTimesDup = (realM * tiling_->taskSingleCoreNSize) / FP32_MAX_MASK_SIZE;
        Duplicate(c1c2ComputeTensor_, 0.0f, FP32_MAX_MASK_SIZE, repeatTimesDup, 1, FP32_MASK_BLK_NUM);
        PipeBarrier<PIPE_V>();

        for (uint16_t unfoldATimes = 0; unfoldATimes < multiScaleTimes_; unfoldATimes++) {
            uint64_t cMatrixOffset =
                cUnfoldSize_ * (cubeLoopIdx_ % c1c2CacheNum_) +
                ((kIdx * multiScaleTimes_ + unfoldATimes) * tiling_->mSize + mOffset + params.baseMOffset) *
                    curTaskNSize_ +
                params.nOffset;
            ProcessUnfoldCMatrixPerchannel(cMatrixOffset, realM, params.taskSingleCoreNSize, divFactors[unfoldATimes],
                                           params.nRepeatTimes, c1c2SyncProcessor);
        }
    } else {
        uint64_t repeatTimesDup = (multiScaleTimes_ * realM * tiling_->taskSingleCoreNSize) / FP32_MAX_MASK_SIZE;
        Duplicate(c1c2ComputeTensor_, 0.0f, FP32_MAX_MASK_SIZE, repeatTimesDup, 1, FP32_MASK_BLK_NUM);
        PipeBarrier<PIPE_V>();

        uint64_t cMatrixOffset =
            cUnfoldSize_ * (cubeLoopIdx_ % c1c2CacheNum_) +
            ((kIdx * multiScaleTimes_) * tiling_->mSize + mOffset + params.baseMOffset) * curTaskNSize_ +
            params.nOffset;
        ProcessUnfoldCMatrixPergroup(cMatrixOffset, realM, params.taskSingleCoreNSize, params.nRepeatTimes,
                                     c1c2SyncProcessor);
    }

    for (uint32_t idxM = 0; idxM < realM; idxM++) {
        AscendC::Mul<float, false>(c1c2ComputeTensor_[idxM * tiling_->taskSingleCoreNSize],
                                   c1c2ComputeTensor_[idxM * tiling_->taskSingleCoreNSize],
                                   antiquantScaleF32Tensor_[params.scaleOffset], FP32_MAX_MASK_SIZE,
                                   params.nRepeatTimes, commonRepeatParams_);
    }
    PipeBarrier<PIPE_V>();

    uint64_t maxOffset = 0;
    for (uint32_t idxM = 0; idxM < realM; idxM++) {
        if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
            maxOffset = (kIdx * params.singleCorerealM + mOffset + idxM) * FP32_BLOCK_SIZE;
        } else {
            maxOffset = idxM * realGroupPack_ * FP32_BLOCK_SIZE + kIdx * FP32_BLOCK_SIZE;
        }
        AscendC::MulAddDst<float, float, false>(cF32Tensor_[(mOffset + idxM) * tiling_->taskSingleCoreNSize],
                                                c1c2ComputeTensor_[idxM * tiling_->taskSingleCoreNSize],
                                                aMaxComputeTensor_[maxOffset], FP32_MAX_MASK_SIZE, params.nRepeatTimes,
                                                params.repeatParams);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ProcessC1C2PerGroup(uint64_t taskSingleCoreNSize, uint64_t nOffset, uint64_t baseMOffset,
                                      uint64_t singleCorerealM, uint64_t nRepeatTimes,
                                      const BinaryRepeatParams &repeatParams)
{
    // 16: 32k / 2 c1c2Fp16PingTensor_空间为32k
    uint64_t loadNum = 16 * 1024 / (multiScaleTimes_ * singleCorerealM * taskSingleCoreNSize * sizeof(preciseType));
    loadNum = loadNum > realGroupPack_ ? realGroupPack_ : loadNum;
    uint64_t cnt = CeilDiv(static_cast<uint64_t>(realGroupPack_), loadNum);

    event_t eventIdsMte2ToV[2] = {
        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()),
        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()),
    };
    event_t eventIdsVToMte2[2] = {
        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()),
        static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()),
    };

    int i = 0;
    CopyInC1C2TensorPerGroup(0, loadNum, baseMOffset, nOffset, singleCorerealM, taskSingleCoreNSize, eventIdsVToMte2[0],
                             eventIdsMte2ToV[0]);
    for (i = 1; i < cnt; i++) {
        CopyInC1C2TensorPerGroup(i, loadNum, baseMOffset, nOffset, singleCorerealM, taskSingleCoreNSize,
                                 eventIdsVToMte2[i & 1], eventIdsMte2ToV[i & 1]);
        CalculateC1C2Pergroup(i - 1, cnt, loadNum, taskSingleCoreNSize, singleCorerealM, nRepeatTimes, repeatParams,
                              eventIdsVToMte2[(i - 1) & 1], eventIdsMte2ToV[(i - 1) & 1]);
    }
    CalculateC1C2Pergroup(i - 1, cnt, loadNum, taskSingleCoreNSize, singleCorerealM, nRepeatTimes, repeatParams,
                          eventIdsVToMte2[(i - 1) & 1], eventIdsMte2ToV[(i - 1) & 1]);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdsMte2ToV[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdsMte2ToV[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdsVToMte2[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdsVToMte2[0]);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CopyInC1C2TensorPerGroup(uint64_t id, uint64_t loadNum, uint64_t baseMOffset, uint64_t nOffset,
                                           uint64_t realM, uint64_t taskSingleCoreNSize, const event_t &eventIdVToMte2,
                                           const event_t &eventIdMte2ToV)
{
    LocalTensor<half> c1c2S32Tensor;
    if ((id & 1) == 0) { // 使用&对2进行取余比较，运算符优先级需加括号
        c1c2S32Tensor = c1c2Fp16PingTensor_;
    } else {
        c1c2S32Tensor = c1c2Fp16PongTensor_;
    }
    uint64_t kIdx = id * loadNum;
    uint64_t realLoadGroupNum = kIdx + loadNum > realGroupPack_ ? realGroupPack_ - kIdx : loadNum;

    if (id > 1) {
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }

    for (int g = 0; g < realLoadGroupNum; g++, kIdx++) {
        uint64_t cMatrixOffset = cUnfoldSize_ * (cubeLoopIdx_ % c1c2CacheNum_) +
                                 ((kIdx * multiScaleTimes_) * tiling_->mSize + baseMOffset) * curTaskNSize_ + nOffset;
        for (uint32_t i = 0; i < multiScaleTimes_; i++) {
            uint64_t gmOffset = cMatrixOffset + i * realM * curTaskNSize_;
            DataCopyPad2D(c1c2S32Tensor[(g * realM * multiScaleTimes_ + realM * i) * tiling_->taskSingleCoreNSize],
                          cUnfoldGlobal_[gmOffset], realM, taskSingleCoreNSize, tiling_->taskSingleCoreNSize,
                          curTaskNSize_);
        }
    }
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ProcessUnfoldCMatrixPergroupFp16(uint64_t gId, uint64_t realM, half divFactorsFp16[],
                                                   uint64_t repeatTimesAxpyFp32, uint64_t repeatTimesAxpyFp16,
                                                   const CopyRepeatParams &copyParam, LocalTensor<half> &c1c2S32Tensor)
{
    SetMaskNorm();
    SetVectorMask<half, MaskMode::NORMAL>(FP16_MASK_SIZE);
    uint64_t c1c2Offset = (gId * realM * multiScaleTimes_) * tiling_->taskSingleCoreNSize;
    AscendC::Copy<half, false>(c1c2Fp16ComputeTensor_, c1c2S32Tensor[c1c2Offset], FP16_MASK_SIZE, repeatTimesAxpyFp16,
                               copyParam);
    PipeBarrier<PIPE_V>();

    for (uint16_t unfoldATimes = 1; unfoldATimes < multiScaleTimes_; unfoldATimes++) {
        uint64_t c1c2Offset = (gId * realM * multiScaleTimes_ + realM * unfoldATimes) * tiling_->taskSingleCoreNSize;
        AscendC::Axpy<half, half, false>(c1c2Fp16ComputeTensor_, c1c2S32Tensor[c1c2Offset],
                                         divFactorsFp16[unfoldATimes], FP16_MASK_SIZE, repeatTimesAxpyFp16,
                                         commonUnaryRepeatParams_);
        PipeBarrier<PIPE_V>();
    }

    SetVectorMask<float, MaskMode::NORMAL>(FP32_MAX_MASK_SIZE);
    // uint64_t castRepeatTimesHalf = ()
    AscendC::Cast<float, half, false>(c1c2ComputeTensor_, c1c2Fp16ComputeTensor_, RoundMode::CAST_NONE,
                                      FP32_MAX_MASK_SIZE, repeatTimesAxpyFp32, fp16ToFp32UnaryRepeatParams_);
    PipeBarrier<PIPE_V>();
    AscendC::Muls<float, false>(c1c2ComputeTensor_, c1c2ComputeTensor_, divFactors_[0], FP32_MAX_MASK_SIZE,
                                repeatTimesAxpyFp32, commonUnaryRepeatParams_);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::CalculateC1C2Pergroup(uint64_t id, uint64_t cnt, uint64_t loadNum, uint64_t taskSingleCoreNSize,
                                        uint64_t realM, uint64_t nRepeatTimes, const BinaryRepeatParams &repeatParams,
                                        const event_t &eventIdVToMte2, const event_t &eventIdMte2ToV)
{
    half divFactorsFp16[3] = {static_cast<half>(divFactors_[0]), static_cast<half>(divFactors_[1]),
                              static_cast<half>(divFactors_[2])};
    LocalTensor<half> c1c2S32Tensor;
    if ((id & 1) == 0) {
        c1c2S32Tensor = c1c2Fp16PingTensor_;
    } else {
        c1c2S32Tensor = c1c2Fp16PongTensor_;
    }

    uint64_t kIdx = id * loadNum;
    uint64_t realLoadGroupNum = kIdx + loadNum > realGroupPack_ ? realGroupPack_ - kIdx : loadNum;
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    uint64_t repeatTimesAxpyFp16 = (realM * tiling_->taskSingleCoreNSize) / FP16_MASK_SIZE;
    uint64_t repeatTimesAxpyFp32 = (realM * tiling_->taskSingleCoreNSize) / FP32_MAX_MASK_SIZE;
    CopyRepeatParams copyParam = {1, 1, 8, 8};

    for (int g = 0; g < realLoadGroupNum; g++, kIdx++) {
        ProcessUnfoldCMatrixPergroupFp16(g, realM, divFactorsFp16, repeatTimesAxpyFp32, repeatTimesAxpyFp16, copyParam,
                                         c1c2S32Tensor);
        PipeBarrier<PIPE_V>();

        uint64_t scaleOffset = kIdx * taskSingleCoreNSize;
        for (uint32_t idxM = 0; idxM < realM; idxM++) {
            AscendC::Mul<float, false>(c1c2ComputeTensor_[idxM * tiling_->taskSingleCoreNSize],
                                       c1c2ComputeTensor_[idxM * tiling_->taskSingleCoreNSize],
                                       antiquantScaleF32Tensor_[scaleOffset], FP32_MAX_MASK_SIZE, nRepeatTimes,
                                       commonRepeatParams_);
        }
        PipeBarrier<PIPE_V>();

        for (uint32_t idxM = 0; idxM < realM; idxM++) {
            uint64_t maxOffset = idxM * realGroupPack_ * FP32_BLOCK_SIZE + kIdx * FP32_BLOCK_SIZE;
            AscendC::MulAddDst<float, float, false>(cF32Tensor_[idxM * tiling_->taskSingleCoreNSize],
                                                    c1c2ComputeTensor_[idxM * tiling_->taskSingleCoreNSize],
                                                    aMaxComputeTensor_[maxOffset], FP32_MAX_MASK_SIZE, nRepeatTimes,
                                                    repeatParams);
        }
        PipeBarrier<PIPE_V>();
    }
    if (cnt > 2 && id < cnt - 2) {
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::GetPingPongC1C2Tensor(LocalTensor<int32_t> &c1c2S32Tensor, LocalTensor<float> &c1c2Tensor,
                                        SyncProcessor<HardEvent::V_MTE2> &c1c2SyncProcessor)
{
    if (c1c2SyncProcessor.GetBufferId() == 0) {
        c1c2Tensor = c1c2PingTensor_;
        c1c2S32Tensor = c1c2S32PingTensor_;
    } else {
        c1c2Tensor = c1c2PongTensor_;
        c1c2S32Tensor = c1c2S32PongTensor_;
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ProcessUnfoldCMatrixPerchannel(uint64_t cMatrixOffset, uint64_t realM, uint64_t taskSingleCoreNSize,
                                                 float divFactors, uint64_t nRepeatTimes,
                                                 SyncProcessor<HardEvent::V_MTE2> &c1c2SyncProcessor)
{
    LocalTensor<int32_t> c1c2S32Tensor;
    LocalTensor<float> c1c2Tensor;
    GetPingPongC1C2Tensor(c1c2S32Tensor, c1c2Tensor, c1c2SyncProcessor);
    c1c2SyncProcessor.WaitSyncFlag();

    DataCopyPad2D(c1c2S32Tensor, cUnfoldGlobal_[cMatrixOffset], realM, taskSingleCoreNSize,
                  tiling_->taskSingleCoreNSize, curTaskNSize_);
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    uint64_t castRepeatTimes = (realM * tiling_->taskSingleCoreNSize) / FP32_MAX_MASK_SIZE;
    AscendC::Cast<float, int32_t, false>(c1c2Tensor, c1c2S32Tensor, RoundMode::CAST_NONE, FP32_MAX_MASK_SIZE,
                                         castRepeatTimes, commonUnaryRepeatParams_);
    PipeBarrier<PIPE_V>();
    for (uint32_t idxM = 0; idxM < realM; idxM++) {
        AscendC::Axpy<float, float, false>(c1c2ComputeTensor_[idxM * tiling_->taskSingleCoreNSize],
                                           c1c2Tensor[idxM * tiling_->taskSingleCoreNSize], divFactors,
                                           FP32_MAX_MASK_SIZE, nRepeatTimes, commonUnaryRepeatParams_);
    }
    PipeBarrier<PIPE_V>();
    c1c2SyncProcessor.SetSyncFlag();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    preciseType>::ProcessUnfoldCMatrixPergroup(uint64_t cMatrixOffset, uint64_t realM, uint64_t taskSingleCoreNSize,
                                               uint64_t nRepeatTimes,
                                               SyncProcessor<HardEvent::V_MTE2> &c1c2SyncProcessor)
{
    float divFactors[3] = {divFactors_[0], divFactors_[1], divFactors_[2]};
    LocalTensor<int32_t> c1c2S32Tensor;
    LocalTensor<float> c1c2Tensor;
    GetPingPongC1C2Tensor(c1c2S32Tensor, c1c2Tensor, c1c2SyncProcessor);
    c1c2SyncProcessor.WaitSyncFlag();

    for (uint32_t idxM = 0; idxM < realM; idxM++) {
        uint64_t gmOffset = cMatrixOffset + idxM * curTaskNSize_;
        DataCopyPad2D(c1c2S32Tensor[idxM * multiScaleTimes_ * tiling_->taskSingleCoreNSize], cUnfoldGlobal_[gmOffset],
                      multiScaleTimes_, taskSingleCoreNSize, tiling_->taskSingleCoreNSize,
                      tiling_->mSize * curTaskNSize_);
    }
    event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

    uint64_t castRepeatTimes = (multiScaleTimes_ * realM * tiling_->taskSingleCoreNSize) / FP32_MAX_MASK_SIZE;
    AscendC::Cast<float, int32_t, false>(c1c2Tensor, c1c2S32Tensor, RoundMode::CAST_NONE, FP32_MAX_MASK_SIZE,
                                         castRepeatTimes, commonUnaryRepeatParams_);
    PipeBarrier<PIPE_V>();

    for (uint16_t unfoldATimes = 0; unfoldATimes < multiScaleTimes_; unfoldATimes++) {
        for (uint32_t idxM = 0; idxM < realM; idxM++) {
            uint64_t c1c2Offset = (idxM * multiScaleTimes_ + unfoldATimes) * tiling_->taskSingleCoreNSize;
            AscendC::Axpy<float, float, false>(c1c2ComputeTensor_[idxM * tiling_->taskSingleCoreNSize],
                                               c1c2Tensor[c1c2Offset], divFactors[unfoldATimes], FP32_MAX_MASK_SIZE,
                                               nRepeatTimes, commonUnaryRepeatParams_);
        }
        PipeBarrier<PIPE_V>();
    }
    c1c2SyncProcessor.SetSyncFlag();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          typename preciseType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdSplitKKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                           hasAntiQuantOffset, quantType, weightFormat,
                                           preciseType>::CopyOutResult(uint64_t taskSingleCoreNSize, uint64_t nOffset,
                                                                       uint64_t baseMOffset, uint64_t singleCorerealM)
{
    if (tiling_->hasBias) {
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

        DataCopyPad2D(biasTensor_, biasGlobal_[nOffset], 1, taskSingleCoreNSize, tiling_->nSize);

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        if constexpr (!IsSameType<biasType, float>::value) {
            Cast(biasFp32Tensor_, biasTensor_, RoundMode::CAST_NONE, taskSingleCoreNSize);
            PipeBarrier<PIPE_V>();
        }

        uint64_t repeatTimes = tiling_->taskSingleCoreNSize / FP32_MAX_MASK_SIZE;
        Add(cF32Tensor_, cF32Tensor_, biasFp32Tensor_, FP32_MAX_MASK_SIZE, repeatTimes, commonRepeatParams_);
        for (uint64_t mLoopIdx = 1; mLoopIdx < singleCorerealM; mLoopIdx++) {
            AscendC::Add<float, false>(cF32Tensor_[mLoopIdx * tiling_->taskSingleCoreNSize],
                                       cF32Tensor_[mLoopIdx * tiling_->taskSingleCoreNSize], biasFp32Tensor_,
                                       FP32_MAX_MASK_SIZE, repeatTimes, commonRepeatParams_);
        }
    }
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<yType, int8_t>::value) {
        return;
    } else {
        Cast(cF16ResultTensor_, cF32Tensor_, RoundMode::CAST_ROUND, singleCorerealM * tiling_->taskSingleCoreNSize);
        PipeBarrier<PIPE_V>();
        TEventID eventIdVToMte3 = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad2D(yGlobal_[baseMOffset * tiling_->nSize + nOffset], cF16ResultTensor_, singleCorerealM,
                      taskSingleCoreNSize, tiling_->taskSingleCoreNSize, tiling_->nSize);
    }
}
} // namespace Mc2WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_MSD_SPLIT_K_H
