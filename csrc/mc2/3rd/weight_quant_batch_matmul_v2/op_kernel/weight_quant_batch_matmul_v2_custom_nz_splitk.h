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
 * \file weight_quant_batch_matmul_v2_custom_nz_splitk.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_CUSTOM_NZ_SPLITK_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_CUSTOM_NZ_SPLITK_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "weight_quant_batch_matmul_v2_constant.h"
#include "tool.h"

using AscendC::AIC;
using AscendC::AIV;
using AscendC::BinaryRepeatParams;
using AscendC::BLOCK_CUBE;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopyPadParams;
using AscendC::DataCopyParams;
using AscendC::FixpipeParamsV220;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::LoadData2DParams;
using AscendC::LocalTensor;
using AscendC::MmadParams;
using AscendC::Nd2NzParams;
using AscendC::ONE_BLK_SIZE;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::UnaryRepeatParams;

struct ComputeParams {
    bool isTailCoreN = false;
    bool isTailCoreK = false;
    bool pingFlag = false;
    uint16_t curSingleK = 0;
    uint16_t vecSingleN = 0;
    uint32_t cubeSingleM = 0;
    uint32_t curCubeSingleN = 0;
    uint32_t curCubeSingleNOrigin = 0; // 按照原始非对齐的shape计算，单次计算量
    uint32_t curSingleCoreK = 0;
    uint32_t curCubeSingleCoreN = 0;
    uint32_t mLoopIdx = 0;
    uint32_t kLoopIdx = 0;
    uint32_t nLoopIdx = 0;
    uint32_t madLoopIdx = 0;
    uint32_t curCubeSingleCoreNLoop = 0;
    uint32_t curSingleCoreKLoop = 0;
    uint32_t mLoopNum = 0;
    uint32_t splitNOffset = 0;
    uint64_t loopNum = 0;
    uint64_t totalLoopIdx = 0;
    uint64_t originNOffset = 0;
    uint64_t kOffset = 0;
    uint64_t cubeSingleCoreNOrigin = 0;
    uint32_t curSingleCoreKOrigin = 0;
    uint16_t curSingleKOrigin = 0;
};

namespace Mc2WeightQuantBatchMatmulV2 {
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
class Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2CustomNzSplitKTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void ComputeGlobalParams();
    __aicore__ inline void InitBuffer();

private:
    __aicore__ inline void InitTensor();
    __aicore__ inline void InitAtomic();
    __aicore__ inline void InitEventIds();
    __aicore__ inline void InitInputOutput(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                           GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y);
    __aicore__ inline void ProcessCube(ComputeParams &computeParams, LocalTensor<xType> &aL1Tensor);
    __aicore__ inline void ComputeMatmulCustom(ComputeParams &computeParams, LocalTensor<xType> &aL1Tensor,
                                               LocalTensor<xType> &bL1Tensor);
    __aicore__ inline void SetMatmulParams(ComputeParams &computeParams);
    __aicore__ inline void CalcComputeParams(ComputeParams &computeParams);
    __aicore__ inline void ProcessVector(ComputeParams &computeParams);
    __aicore__ inline void AntiquantWeight(ComputeParams &computeParams);
    __aicore__ inline void WeightCast(ComputeParams &computeParams);

    __aicore__ inline void CopyInAL1(ComputeParams &computeParams);
    __aicore__ inline void CopyInAL1FullLoad(ComputeParams &computeParams, LocalTensor<xType> &aL1Tensor,
                                             uint32_t aL1Index);
    __aicore__ inline void CopyInBL1(ComputeParams &computeParams);
    __aicore__ inline void WeightCopyIn(ComputeParams &computeParams);

    __aicore__ inline void AntiQuantCompute(ComputeParams &computeParams);
    __aicore__ inline void ComputeMulAddCast(LocalTensor<half> &antiquantWeightTensor, ComputeParams &computeParams);
    __aicore__ inline void ComputeMulAdd(LocalTensor<half> &tmpTensor, LocalTensor<half> &antiquantWeightTensor,
                                         uint32_t loopNum, uint32_t singleCalNum, ComputeParams &computeParams);
    __aicore__ inline void CopyInAntiquantParams(ComputeParams &computeParams);
    __aicore__ inline void WeightCopyOut(ComputeParams &computeParams);
    __aicore__ inline void BroadCastAntiquantParams(LocalTensor<half> &dstTensor, LocalTensor<xType> &srcTensor);
    __aicore__ inline void MainProcess(LocalTensor<xType> &aL1Tensor, ComputeParams &computeParams);
    __aicore__ inline void PostProcess();
    __aicore__ inline void PostCopyInAndCast(LocalTensor<float> &biasComputeTensor,
                                             LocalTensor<float> &scaleComputeTensor, uint32_t realSingleCoreN,
                                             uint32_t nOffset);
    __aicore__ inline void BiasAdd(LocalTensor<float> &biasComputeTensor, LocalTensor<float> &cTensor,
                                   LocalTensor<xType> &outTensor, uint32_t realSingleM, uint32_t realSingleN,
                                   uint32_t nIdx);
    __aicore__ inline void ScaleMul(LocalTensor<float> &scaleComputeTensor, LocalTensor<float> &cTensor,
                                    uint32_t realSingleM, uint32_t realSingleN, uint32_t nIdx);
    __aicore__ inline void PostCast(LocalTensor<xType> &outTensor, LocalTensor<float> &cTensor, uint32_t realSingleM,
                                    uint32_t realSingleN);
    __aicore__ inline void PostMulAndAdd(LocalTensor<float> &biasComputeTensor, LocalTensor<float> &scaleComputeTensor,
                                         uint32_t realSingleCoreN, uint32_t nOffset);
    __aicore__ inline void ReleaseEventIds();

    static constexpr int32_t DOUBLE_BUFFER_NUM = 2;
    static constexpr int32_t BLOCK_SIZE_INT8 = 32;
    static constexpr int32_t WEIGHT_CACHE_COUNT = 4;
    static constexpr int32_t FP32_MASKMAX = 256 / sizeof(float);
    static constexpr int32_t FP16_MASKMAX = 256 / sizeof(half);
    static constexpr int32_t WORKSPACE_N_SIZE = 128;
    static constexpr int32_t L0C_DB_NUM = 2;
    static constexpr int32_t INIT_WORKSPACE_SYNC_ID = 9;
    static constexpr int32_t CUBE_VEC_FLAG_ID = 8;
    static constexpr int32_t VEC_CUBE_FLAG_ID = 6;
    static constexpr int32_t POST_VEC_SYNC_ID = 5;
    static constexpr int32_t POST_VEC_CUBE_SYNC_ID = 7;
    static constexpr uint32_t REPEAT_LINE = 8U; // 256B / 32B
    static constexpr int32_t BLOCK_SIZE = GetBlockSize<wType>();

    TPipe *pipe_;
    const Mc2WeightQuantBatchMatmulV2CustomNzSplitKTilingData *tiling_;

    bool biasFlag_;
    bool hasPostProcessFlag_;

    int32_t curBlockIdx_;
    int32_t cubeNDimIdx_;
    int32_t cubeKDimIdx_;
    int32_t vecKDimIdx_;
    int32_t vecNDimIdx_;
    uint64_t weightCacheSize_;
    uint64_t weightCacheIdx_;

    BrcbRepeatParams brcbParams_;
    DataCopyParams antiquantCopyinParams_;
    DataCopyPadParams antiquantCopyinPadParams_;
    LoadData2DParams bL0Load2dParams_;
    MmadParams mmadParams_;
    FixpipeParamsV220 fixParams_;
    AscendC::LoadData3DParamsV2Pro loadData3DV2_;

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<xType> weightCache_;
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<xType> offsetGlobal_;
    GlobalTensor<xType> scaleGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<yType> yGlobal_;
    GlobalTensor<uint64_t> quantScaleGlobal_;
    GlobalTensor<float> atomicAddWorkspace_;

    LocalTensor<half> offsetComputeTensor_;
    LocalTensor<half> scaleComputeTensor_;

    TBuf<> ubTBuf_;

    TBuf<> offsetPingTBuf_;
    TBuf<> offsetPongTBuf_;
    TBuf<> scalePingTBuf_;
    TBuf<> scalePongTBuf_;

    TBuf<> antiquantParamsFp32TBuf_;
    TBuf<> offsetComputeTBuf_;
    TBuf<> scaleComputeTBuf_;

    TBuf<> biasInputTBuf_;
    TBuf<> biasComputeTBuf_;

    TBuf<> scaleInputTBuf_;
    TBuf<> scaleComputePostTBuf_;

    TBuf<TPosition::A2> a2TBuf_;
    TBuf<TPosition::B2> b2TBuf_;
    TBuf<TPosition::CO1> co1TBuf_;

    TQue<QuePosition::B1, DOUBLE_BUFFER_NUM> inQueueBL1_;
    TQue<QuePosition::A1, 1> inQueueAL1_;

    LocalTensor<wType> weightInPingTensor_;
    LocalTensor<wType> weightInPongTensor_;

    LocalTensor<half> weight16Tensor_;
    LocalTensor<float> weight32Tensor_;
    LocalTensor<half> weightSplitTmpTensor_;
    LocalTensor<xType> weightOutTensor_;

    LocalTensor<xType> offsetPingTensor_;
    LocalTensor<xType> offsetPongTensor_;

    LocalTensor<xType> scalePingTensor_;
    LocalTensor<xType> scalePongTensor_;

    LocalTensor<xType> aL0TensorPing_;
    LocalTensor<xType> aL0TensorPong_;
    LocalTensor<xType> bL0TensorPing_;
    LocalTensor<xType> bL0TensorPong_;
    LocalTensor<float> cL0TensorPing_;
    LocalTensor<float> cL0TensorPong_;

    TEventID mte1WaitMEventIds_[2];
    TEventID mWaitMte1EventIds_[2];
    TEventID mWaitFixEventIds_[2];
    TEventID fixWaitMEventIds_[2];
    TEventID mte2WaitVEventIds_[2];
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::InitInputOutput(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                  GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y)
{
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(x), tiling_->mSize * tiling_->kSize);
    wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ wType *>(weight), tiling_->kSizeAlign * tiling_->nSizeAlign);
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType *>(y), tiling_->mSize * tiling_->nSize);
    biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling_->nSize);
    biasFlag_ = static_cast<bool>(tiling_->hasBias);
    offsetGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantOffset), tiling_->nSize);
    scaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantScale), tiling_->nSize);
    quantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(quantScale), tiling_->nSize);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType, aL1FullLoad>::ComputeGlobalParams()
{
    cubeNDimIdx_ = curBlockIdx_ % tiling_->cubeBlockDimN;
    cubeKDimIdx_ = curBlockIdx_ / tiling_->cubeBlockDimN;
    vecKDimIdx_ = curBlockIdx_ / tiling_->vecBlockDimN;
    vecNDimIdx_ = curBlockIdx_ % tiling_->vecBlockDimN;

    // brcb接口只支持按照Block处理源数据。antiquant操作在目的操作上不需要跳写，不同block间地址步长为1个block
    brcbParams_.dstBlkStride = 1;
    brcbParams_.dstRepStride = 8;
    // antiquant参数搬运进ub时不需要跳写
    antiquantCopyinParams_.dstStride = 0;
    antiquantCopyinParams_.blockCount = 1;
    antiquantCopyinParams_.srcStride = 0;
    hasPostProcessFlag_ = tiling_->cubeBlockDimK > 1 || IsSameType<xType, bfloat16_t>::value || biasFlag_;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType, aL1FullLoad>::InitTensor()
{
    // vector单次计算基本块vecSingleK=512, vecSingleN=32等于16k大小。ub空间划分以16k的倍数存放各块buffer数据。
    weightInPingTensor_ = ubTBuf_.template Get<wType>();        // 0-16K
    weightInPongTensor_ = ubTBuf_.template Get<wType>()[16384]; // 16-32K
    weight32Tensor_ = ubTBuf_.template Get<float>()[8192];      // 32-96K
    weight16Tensor_ = ubTBuf_.template Get<half>()[49152];      // 96-128K
    weightOutTensor_ = ubTBuf_.template Get<xType>()[65536];    // 128-160K
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        weightSplitTmpTensor_ = ubTBuf_.template Get<half>()[24576];
    } else {
        weightSplitTmpTensor_ = ubTBuf_.template Get<half>()[16384]; // 32-64K
    }

    uint32_t tensorSize = tiling_->vecSingleN * tiling_->singleK;
    weightInPingTensor_.SetSize(tensorSize);
    weightInPongTensor_.SetSize(tensorSize);
    weight16Tensor_.SetSize(tensorSize);
    weight32Tensor_.SetSize(tensorSize);
    weightSplitTmpTensor_.SetSize(tensorSize);
    weightOutTensor_.SetSize(tensorSize);

    offsetComputeTensor_ = offsetComputeTBuf_.Get<half>();
    scaleComputeTensor_ = scaleComputeTBuf_.Get<half>();

    offsetPingTensor_ = offsetPingTBuf_.template Get<xType>();
    offsetPongTensor_ = offsetPongTBuf_.template Get<xType>();

    scalePingTensor_ = scalePingTBuf_.template Get<xType>();
    scalePongTensor_ = scalePongTBuf_.template Get<xType>();

    aL0TensorPing_ = a2TBuf_.template Get<xType>();
    aL0TensorPong_ = a2TBuf_.template Get<xType>()[16384];

    bL0TensorPing_ = b2TBuf_.template Get<xType>();
    bL0TensorPong_ = b2TBuf_.template Get<xType>()[16384];

    cL0TensorPing_ = co1TBuf_.template Get<float>();
    cL0TensorPong_ = co1TBuf_.template Get<float>()[16384];
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType, aL1FullLoad>::InitBuffer()
{
    int64_t antiquantShape = 32;
    int64_t antiquantOriSize = antiquantShape * sizeof(xType);

    pipe_->InitBuffer(ubTBuf_, 160 * 1024);
    // 65536分到40核，每核最大1664个数，4K数据量足够使用
    pipe_->InitBuffer(biasInputTBuf_, 8 * 1024);
    pipe_->InitBuffer(biasComputeTBuf_, 8 * 1024);

    pipe_->InitBuffer(scaleInputTBuf_, 4 * 1024);
    pipe_->InitBuffer(scaleComputePostTBuf_, 8 * 1024);

    pipe_->InitBuffer(offsetPingTBuf_, antiquantOriSize);
    pipe_->InitBuffer(offsetPongTBuf_, antiquantOriSize);
    pipe_->InitBuffer(scalePingTBuf_, antiquantOriSize);
    pipe_->InitBuffer(scalePongTBuf_, antiquantOriSize);

    int64_t antiquantBroadCastSize = antiquantShape * ONE_BLK_SIZE;

    pipe_->InitBuffer(offsetComputeTBuf_, antiquantBroadCastSize);
    pipe_->InitBuffer(scaleComputeTBuf_, antiquantBroadCastSize);

    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        int64_t antiquantFp32Size = antiquantShape * sizeof(float);
        pipe_->InitBuffer(antiquantParamsFp32TBuf_, antiquantFp32Size);
    }

    // L0 buffer全部占满
    pipe_->InitBuffer(a2TBuf_, 65536);
    pipe_->InitBuffer(b2TBuf_, 65536);
    pipe_->InitBuffer(co1TBuf_, 131072);

    pipe_->InitBuffer(inQueueBL1_, 2, 131072);
    if constexpr (aL1FullLoad) {
        pipe_->InitBuffer(inQueueAL1_, 1, 262144);
    } else {
        pipe_->InitBuffer(inQueueAL1_, 2, 131072);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType, aL1FullLoad>::InitAtomic()
{
    if (!hasPostProcessFlag_) {
        return;
    }
    uint64_t initTotalSize = tiling_->mSize * tiling_->nSize;
    InitAtomicAddr(atomicAddWorkspace_, initTotalSize, curBlockIdx_);
    CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(INIT_WORKSPACE_SYNC_ID);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType, aL1FullLoad>::InitEventIds()
{
    mte1WaitMEventIds_[0] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
    mte1WaitMEventIds_[1] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();

    mWaitMte1EventIds_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>();
    mWaitMte1EventIds_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>();

    mte2WaitVEventIds_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    mte2WaitVEventIds_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();

    mWaitFixEventIds_[0] = GetTPipePtr()->AllocEventID<HardEvent::FIX_M>();
    mWaitFixEventIds_[1] = GetTPipePtr()->AllocEventID<HardEvent::FIX_M>();

    fixWaitMEventIds_[0] = GetTPipePtr()->AllocEventID<HardEvent::M_FIX>();
    fixWaitMEventIds_[1] = GetTPipePtr()->AllocEventID<HardEvent::M_FIX>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
                       GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                       const Mc2WeightQuantBatchMatmulV2CustomNzSplitKTilingData *tilingData, TPipe *tPipe)
{
    tiling_ = tilingData;
    pipe_ = tPipe;
    curBlockIdx_ = GetBlockIdx();
    InitInputOutput(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y);
    ComputeGlobalParams();

    weightCacheSize_ = WORKSPACE_N_SIZE * tiling_->singleK;

    InitBuffer();
    InitTensor();
    uint64_t workspaceOffset;
    if ASCEND_IS_AIV {
        workspaceOffset = (curBlockIdx_ >> 1) * weightCacheSize_ * WEIGHT_CACHE_COUNT * sizeof(xType);
    } else {
        workspaceOffset = curBlockIdx_ * weightCacheSize_ * WEIGHT_CACHE_COUNT * sizeof(xType);
    }
    weightCache_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(workspace + workspaceOffset));
    atomicAddWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
        workspace + weightCacheSize_ * WEIGHT_CACHE_COUNT * sizeof(xType) * GetBlockNum()));

    if ASCEND_IS_AIV {
        InitAtomic();
    }

    InitEventIds();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType,
                                                aL1FullLoad>::WeightCopyIn(ComputeParams &computeParams)
{
    LocalTensor<wType> originWeight = computeParams.pingFlag ? weightInPingTensor_ : weightInPongTensor_;
    DataCopyParams copyInParams;
    uint64_t wSrcOffset = computeParams.kOffset * tiling_->nSizeAlign + computeParams.originNOffset * BLOCK_SIZE;
    copyInParams.blockCount = CeilDiv(static_cast<int32_t>(computeParams.curSingleK), BLOCK_SIZE);
    copyInParams.blockLen = computeParams.vecSingleN;
    copyInParams.srcStride = tiling_->nSizeAlign - computeParams.vecSingleN;
    copyInParams.dstStride = 0;
    DataCopy(originWeight, wGlobal_[wSrcOffset], copyInParams);
    TEventID eventIdVWaitMte2 = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();
    SetFlag<HardEvent::MTE2_V>(eventIdVWaitMte2);
    WaitFlag<HardEvent::MTE2_V>(eventIdVWaitMte2);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType,
                                                aL1FullLoad>::WeightCast(ComputeParams &computeParams)
{
    LocalTensor<wType> originWeight = computeParams.pingFlag ? weightInPingTensor_ : weightInPongTensor_;
    Cast(weight16Tensor_, originWeight, RoundMode::CAST_NONE, computeParams.vecSingleN * computeParams.curSingleK);
    PipeBarrier<PIPE_V>();
    if (computeParams.pingFlag) {
        SetFlag<HardEvent::V_MTE2>(mte2WaitVEventIds_[0]);
    } else {
        SetFlag<HardEvent::V_MTE2>(mte2WaitVEventIds_[1]);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::ComputeMulAdd(LocalTensor<half> &tmpTensor, LocalTensor<half> &antiquantWeightTensor,
                                uint32_t loopNum, uint32_t singleCalNum, ComputeParams &computeParams)
{
    // 每次分型计算前16个数，一次repeat计算8行
    BinaryRepeatParams splitParams;
    splitParams.src0BlkStride = BLOCK_SIZE / BLOCK_CUBE; // C0 * sizeof(half) / 32B
    splitParams.src1BlkStride = 1;
    splitParams.src0RepStride = REPEAT_LINE * BLOCK_SIZE / BLOCK_CUBE;
    splitParams.src1RepStride = REPEAT_LINE;
    splitParams.dstBlkStride = splitParams.src0BlkStride;
    splitParams.dstRepStride = splitParams.src0RepStride;

    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<half, MaskMode::NORMAL>(FP16_MASKMAX);
    uint32_t repeatTimes = CeilDiv(static_cast<uint32_t>(computeParams.vecSingleN), REPEAT_LINE);

    if constexpr (hasAntiQuantOffset) {
        uint32_t kInnerLoop = BLOCK_SIZE / BLOCK_CUBE;
        for (uint32_t i = 0; i < loopNum; i++) {
            for (uint32_t kInnerLoopIdx = 0; kInnerLoopIdx < kInnerLoop; kInnerLoopIdx++) {
                AscendC::Add<half, false>(
                    tmpTensor[singleCalNum * i + kInnerLoopIdx * ONE_BLK_SIZE / sizeof(half)],
                    antiquantWeightTensor[singleCalNum * i + kInnerLoopIdx * ONE_BLK_SIZE / sizeof(half)],
                    offsetComputeTensor_, AscendC::MASK_PLACEHOLDER, repeatTimes, splitParams);
            }
        }
        PipeBarrier<PIPE_V>();
    } else {
        if constexpr (IsSameType<xType, half>::value) {
            DataCopy(tmpTensor, antiquantWeightTensor, computeParams.vecSingleN * computeParams.curSingleK);
            PipeBarrier<PIPE_V>();
        }
    }

    splitParams.dstBlkStride = 1;
    splitParams.dstRepStride = 8;

    if constexpr (IsSameType<xType, half>::value) {
        uint32_t calSizePerInstr = computeParams.vecSingleN * BLOCK_CUBE;
        uint32_t kInnerLoop = BLOCK_SIZE / BLOCK_CUBE;
        for (uint32_t i = 0; i < loopNum; i++) {
            for (uint32_t kInnerLoopIdx = 0; kInnerLoopIdx < kInnerLoop; kInnerLoopIdx++) {
                AscendC::Mul<half, false>(antiquantWeightTensor[singleCalNum * i + kInnerLoopIdx * calSizePerInstr],
                                          tmpTensor[singleCalNum * i + BLOCK_CUBE * kInnerLoopIdx], scaleComputeTensor_,
                                          AscendC::MASK_PLACEHOLDER, repeatTimes, splitParams);
            }
        }
    }

    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::ComputeMulAddCast(LocalTensor<half> &antiquantWeightTensor, ComputeParams &computeParams)
{
    uint32_t loopNum = computeParams.curSingleK / BLOCK_SIZE;
    uint32_t singleCalNum = computeParams.vecSingleN * BLOCK_SIZE;
    LocalTensor<half> tmpTensor;
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        tmpTensor = antiquantWeightTensor;
    } else {
        tmpTensor = weightSplitTmpTensor_;
    }

    ComputeMulAdd(tmpTensor, antiquantWeightTensor, loopNum, singleCalNum, computeParams);

    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        AscendC::SetVectorMask<half, MaskMode::NORMAL>(FP32_MASKMAX);
        // 按fp32计算，一次计算4行
        UnaryRepeatParams castParams;
        castParams.dstBlkStride = 1;
        castParams.srcBlkStride = BLOCK_SIZE / BLOCK_CUBE;
        castParams.dstRepStride = 8;
        castParams.srcRepStride = 4 * BLOCK_SIZE / BLOCK_CUBE;

        uint32_t calSizePerInstr = computeParams.vecSingleN * BLOCK_CUBE;
        uint32_t kInnerLoop = BLOCK_SIZE / BLOCK_CUBE;
        uint32_t repeatTimes = CeilDiv(static_cast<uint32_t>(computeParams.vecSingleN), 4U);

        for (uint32_t i = 0; i < loopNum; i++) {
            for (uint32_t kInnerLoopIdx = 0; kInnerLoopIdx < kInnerLoop; kInnerLoopIdx++) {
                AscendC::Cast<float, half, false>(weight32Tensor_[singleCalNum * i + kInnerLoopIdx * calSizePerInstr],
                                                  tmpTensor[singleCalNum * i + BLOCK_CUBE * kInnerLoopIdx],
                                                  RoundMode::CAST_NONE, AscendC::MASK_PLACEHOLDER, repeatTimes,
                                                  castParams);
            }
        }
        PipeBarrier<PIPE_V>();
    }
    AscendC::ResetMask();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType,
                                                aL1FullLoad>::AntiQuantCompute(ComputeParams &computeParams)
{
    LocalTensor<half> antiquantWeightTensor = weight16Tensor_;
    ComputeMulAddCast(antiquantWeightTensor, computeParams);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType,
                                                aL1FullLoad>::ProcessVector(ComputeParams &computeParams)
{
    uint64_t nBaseOffset = (vecNDimIdx_ >> 1) * tiling_->cubeSingleCoreN // blockOffset
                           + computeParams.nLoopIdx * tiling_->cubeSingleN;
    uint64_t kOffset = vecKDimIdx_ * tiling_->singleCoreK + computeParams.kLoopIdx * tiling_->singleK;

    computeParams.kOffset = kOffset;
    if (computeParams.totalLoopIdx >= WEIGHT_CACHE_COUNT) {
        CrossCoreWaitFlag(VEC_CUBE_FLAG_ID);
    }
    uint32_t vectorNLoopNum = CeilDiv(computeParams.curCubeSingleN / 2, tiling_->vecSingleN);
    for (uint32_t vectorNloopIdx = 0; vectorNloopIdx < vectorNLoopNum; vectorNloopIdx++) {
        computeParams.splitNOffset =
            GetSubBlockIdx() * (computeParams.curCubeSingleN >> 1) + vectorNloopIdx * tiling_->vecSingleN;
        computeParams.originNOffset = nBaseOffset + computeParams.splitNOffset;
        computeParams.pingFlag = ((computeParams.totalLoopIdx * vectorNLoopNum + vectorNloopIdx) & 1) == 1;
        weightCacheIdx_ = computeParams.totalLoopIdx % WEIGHT_CACHE_COUNT;
        AntiquantWeight(computeParams);
    }
    if (computeParams.totalLoopIdx == 0 && hasPostProcessFlag_) {
        CrossCoreWaitFlag(INIT_WORKSPACE_SYNC_ID);
    }
    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(CUBE_VEC_FLAG_ID);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::PostCopyInAndCast(LocalTensor<float> &biasComputeTensor, LocalTensor<float> &scaleComputeTensor,
                                    uint32_t realSingleCoreN, uint32_t nOffset)
{
    LocalTensor<biasType> biasInput = biasInputTBuf_.template Get<biasType>();
    if (biasFlag_) {
        TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();
        if constexpr (!IsSameType<biasType, float>::value) {
            DataCopyPad2D(biasInput, biasGlobal_[nOffset], 1, realSingleCoreN, tiling_->nSize);
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            Cast(biasComputeTensor, biasInput, RoundMode::CAST_NONE, realSingleCoreN);
            PipeBarrier<PIPE_V>();
        } else {
            DataCopyPad2D(biasComputeTensor, biasGlobal_[nOffset], 1, realSingleCoreN, tiling_->nSize);
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
        }
    }
    TEventID eventId;
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        LocalTensor<xType> scaleInput = scaleInputTBuf_.template Get<xType>();
        DataCopyPad2D(scaleInput, scaleGlobal_[nOffset], 1, realSingleCoreN, tiling_->nSize);
        eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        Cast(scaleComputeTensor, scaleInput, RoundMode::CAST_NONE, realSingleCoreN);
        PipeBarrier<PIPE_V>();
    }

    eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>();
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::BiasAdd(LocalTensor<float> &biasComputeTensor, LocalTensor<float> &cTensor,
                          LocalTensor<xType> &outTensor, uint32_t realSingleM, uint32_t realSingleN, uint32_t nIdx)
{
    if (biasFlag_) {
        BinaryRepeatParams addParams;
        addParams.src0BlkStride = 1;
        addParams.src1BlkStride = 1;
        addParams.src0RepStride = CeilDiv(realSingleN, REPEAT_LINE);
        addParams.src1RepStride = 0;
        addParams.dstBlkStride = 1;
        addParams.dstRepStride = CeilDiv(realSingleN, REPEAT_LINE);
        uint32_t realMask = FP32_MASKMAX;
        uint32_t nLoopNum = CeilDiv(realSingleN, static_cast<uint32_t>(FP32_MASKMAX));
        for (uint32_t nInnerIdx = 0; nInnerIdx < nLoopNum; nInnerIdx++) {
            if (nInnerIdx == nLoopNum - 1) {
                realMask = realSingleN - nInnerIdx * FP32_MASKMAX;
            }
            if (realSingleM <= UINT8_MAX) {
                Add(cTensor[nInnerIdx * FP32_MASKMAX], cTensor[nInnerIdx * FP32_MASKMAX],
                    biasComputeTensor[nIdx * tiling_->postSingleN + nInnerIdx * FP32_MASKMAX], realMask, realSingleM,
                    addParams);
            } else {
                Add(cTensor[nInnerIdx * FP32_MASKMAX], cTensor[nInnerIdx * FP32_MASKMAX],
                    biasComputeTensor[nIdx * tiling_->postSingleN + nInnerIdx * FP32_MASKMAX], realMask, UINT8_MAX,
                    addParams);
                Add(cTensor[realSingleN * UINT8_MAX + nInnerIdx * FP32_MASKMAX],
                    cTensor[realSingleN * UINT8_MAX + nInnerIdx * FP32_MASKMAX],
                    biasComputeTensor[nIdx * tiling_->postSingleN + nInnerIdx * FP32_MASKMAX], realMask, 1, addParams);
            }
        }
        PipeBarrier<PIPE_V>();
    }
    TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(eventId);
    WaitFlag<HardEvent::MTE3_V>(eventId);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::ScaleMul(LocalTensor<float> &scaleComputeTensor, LocalTensor<float> &cTensor, uint32_t realSingleM,
                           uint32_t realSingleN, uint32_t nIdx)
{
    if constexpr (!IsSameType<xType, bfloat16_t>::value) {
        return;
    }
    BinaryRepeatParams mulParams;
    mulParams.src0BlkStride = 1;
    mulParams.src1BlkStride = 1;
    mulParams.src0RepStride = CeilDiv(realSingleN, REPEAT_LINE);
    mulParams.src1RepStride = 0;
    mulParams.dstBlkStride = 1;
    mulParams.dstRepStride = CeilDiv(realSingleN, REPEAT_LINE);
    uint32_t realMask = FP32_MASKMAX;
    uint32_t nLoopNum = CeilDiv(realSingleN, static_cast<uint32_t>(FP32_MASKMAX));
    for (uint32_t nInnerIdx = 0; nInnerIdx < nLoopNum; nInnerIdx++) {
        if (nInnerIdx == nLoopNum - 1) {
            realMask = realSingleN - nInnerIdx * FP32_MASKMAX;
        }
        if (realSingleM <= UINT8_MAX) {
            Mul(cTensor[nInnerIdx * FP32_MASKMAX], cTensor[nInnerIdx * FP32_MASKMAX],
                scaleComputeTensor[nIdx * tiling_->postSingleN + nInnerIdx * FP32_MASKMAX], realMask, realSingleM,
                mulParams);
        } else {
            Mul(cTensor[nInnerIdx * FP32_MASKMAX], cTensor[nInnerIdx * FP32_MASKMAX],
                scaleComputeTensor[nIdx * tiling_->postSingleN + nInnerIdx * FP32_MASKMAX], realMask, UINT8_MAX,
                mulParams);
            Mul(cTensor[realSingleN * UINT8_MAX + nInnerIdx * FP32_MASKMAX],
                cTensor[realSingleN * UINT8_MAX + nInnerIdx * FP32_MASKMAX],
                scaleComputeTensor[nIdx * tiling_->postSingleN + nInnerIdx * FP32_MASKMAX], realMask, 1, mulParams);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::PostCast(LocalTensor<xType> &outTensor, LocalTensor<float> &cTensor, uint32_t realSingleM,
                           uint32_t realSingleN)
{
    UnaryRepeatParams castParams;
    castParams.dstBlkStride = 1;
    castParams.srcBlkStride = 1;
    castParams.dstRepStride = CeilDiv(realSingleN, 16U);
    castParams.srcRepStride = CeilDiv(realSingleN, REPEAT_LINE);
    uint32_t realMask = FP32_MASKMAX;
    uint32_t nLoopNum = CeilDiv(realSingleN, static_cast<uint32_t>(FP32_MASKMAX));
    for (uint32_t nInnerIdx = 0; nInnerIdx < nLoopNum; nInnerIdx++) {
        if (nInnerIdx == nLoopNum - 1) {
            realMask = realSingleN - nInnerIdx * FP32_MASKMAX;
        }
        if (realSingleM <= UINT8_MAX) {
            Cast(outTensor[nInnerIdx * FP32_MASKMAX], cTensor[nInnerIdx * FP32_MASKMAX], RoundMode::CAST_ROUND,
                 realMask, realSingleM, castParams);
        } else {
            Cast(outTensor[nInnerIdx * FP32_MASKMAX], cTensor[nInnerIdx * FP32_MASKMAX], RoundMode::CAST_ROUND,
                 realMask, UINT8_MAX, castParams);
            Cast(outTensor[nInnerIdx * FP32_MASKMAX + CeilAlign(realSingleN, 16U) * UINT8_MAX],
                 cTensor[nInnerIdx * FP32_MASKMAX + CeilAlign(realSingleN, 8U) * UINT8_MAX], RoundMode::CAST_ROUND,
                 realMask, 1, castParams);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::PostMulAndAdd(LocalTensor<float> &biasComputeTensor, LocalTensor<float> &scaleComputeTensor,
                                uint32_t realSingleCoreN, uint32_t nOffset)
{
    LocalTensor<float> cTensorPing = ubTBuf_.template Get<float>();
    LocalTensor<float> cTensorPong = ubTBuf_.template Get<float>()[16384];
    LocalTensor<xType> outTensor = ubTBuf_.template Get<xType>()[65536];
    LocalTensor<float> cTensor;
    uint32_t mLoop = CeilDiv(tiling_->mSize, static_cast<uint64_t>(tiling_->postSingleM));
    uint32_t nLoop = CeilDiv(realSingleCoreN, static_cast<uint32_t>(tiling_->postSingleN));
    uint32_t realSingleN = tiling_->postSingleN;
    uint64_t cOffset = 0;

    for (uint32_t mIdx = 0; mIdx < mLoop; mIdx++) {
        uint32_t realSingleM = tiling_->postSingleM;
        cOffset = mIdx * tiling_->postSingleM * tiling_->nSize + nOffset;
        if (mIdx == mLoop - 1) {
            realSingleM = tiling_->mSize - mIdx * tiling_->postSingleM;
        }
        for (uint32_t nIdx = 0; nIdx < nLoop; nIdx++) {
            realSingleN = tiling_->postSingleN;
            cOffset += nIdx * tiling_->postSingleN;
            if (nIdx == nLoop - 1) {
                realSingleN = realSingleCoreN - nIdx * tiling_->postSingleN;
            }
            if (((mIdx * nLoop + nIdx) & 1) == 1) {
                cTensor = cTensorPing;
            } else {
                cTensor = cTensorPong;
            }
            TEventID eventIdMte2WaitV = GetTPipePtr()->FetchEventID<HardEvent::V_MTE2>();
            SetFlag<HardEvent::V_MTE2>(eventIdMte2WaitV);
            WaitFlag<HardEvent::V_MTE2>(eventIdMte2WaitV);
            DataCopyPad2D(cTensor, atomicAddWorkspace_[cOffset], realSingleM, realSingleN, tiling_->nSize);
            TEventID eventIdVWaitMte2 = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();
            SetFlag<HardEvent::MTE2_V>(eventIdVWaitMte2);
            WaitFlag<HardEvent::MTE2_V>(eventIdVWaitMte2);

            ScaleMul(scaleComputeTensor, cTensor, realSingleM, realSingleN, nIdx);
            BiasAdd(biasComputeTensor, cTensor, outTensor, realSingleM, realSingleN, nIdx);
            PostCast(outTensor, cTensor, realSingleM, realSingleN);
            TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE2_MTE3>();
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            eventId = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
            SetFlag<HardEvent::V_MTE3>(eventId);
            WaitFlag<HardEvent::V_MTE3>(eventId);

            DataCopyPad2D(yGlobal_[cOffset], outTensor, realSingleM, realSingleN, tiling_->nSize);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType, aL1FullLoad>::PostProcess()
{
    if (curBlockIdx_ * tiling_->postSingleCoreN >= tiling_->nSize) {
        return;
    }

    uint32_t realSingleCoreN = tiling_->postSingleCoreN;
    if ((curBlockIdx_ + 1) * tiling_->postSingleCoreN > tiling_->nSize) {
        realSingleCoreN = tiling_->nSize - curBlockIdx_ * tiling_->postSingleCoreN;
    }
    uint32_t nOffset = curBlockIdx_ * tiling_->postSingleCoreN;
    LocalTensor<float> biasComputeTensor = biasComputeTBuf_.template Get<float>();
    LocalTensor<float> scaleComputeTensor = scaleComputePostTBuf_.template Get<float>();

    PostCopyInAndCast(biasComputeTensor, scaleComputeTensor, realSingleCoreN, nOffset);
    PostMulAndAdd(biasComputeTensor, scaleComputeTensor, realSingleCoreN, nOffset);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::BroadCastAntiquantParams(LocalTensor<half> &dstTensor, LocalTensor<xType> &srcTensor)
{
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        LocalTensor<float> fp32Tensor = antiquantParamsFp32TBuf_.Get<float>();
        Cast(fp32Tensor, srcTensor, RoundMode::CAST_NONE, srcTensor.GetSize());
        PipeBarrier<PIPE_V>();
        LocalTensor<half> f16Tensor = scalePingTBuf_.Get<half>(); // 借用scale tensor空间临时用一下
        Cast(f16Tensor, fp32Tensor, RoundMode::CAST_NONE, srcTensor.GetSize());
        PipeBarrier<PIPE_V>();
        Brcb(dstTensor, f16Tensor, srcTensor.GetSize() / (ONE_BLK_FLOAT_NUM), brcbParams_);
    } else {
        Brcb(dstTensor, srcTensor, srcTensor.GetSize() / (ONE_BLK_FLOAT_NUM), brcbParams_);
    }
    PipeBarrier<PIPE_V>();
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType,
                                                aL1FullLoad>::CopyInAntiquantParams(ComputeParams &computeParams)
{
    uint64_t antiquantOffset = computeParams.originNOffset;
    antiquantCopyinParams_.blockLen = computeParams.vecSingleN * sizeof(xType);
    if (computeParams.pingFlag) {
        WaitFlag<HardEvent::V_MTE2>(mte2WaitVEventIds_[0]);
    } else {
        WaitFlag<HardEvent::V_MTE2>(mte2WaitVEventIds_[1]);
    }
    if constexpr (hasAntiQuantOffset) {
        LocalTensor<xType> offsetInput = computeParams.pingFlag ? offsetPingTensor_ : offsetPongTensor_;
        DataCopyPad(offsetInput, offsetGlobal_[antiquantOffset], antiquantCopyinParams_, antiquantCopyinPadParams_);
        TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        BroadCastAntiquantParams(offsetComputeTensor_, offsetInput);
    }

    if constexpr (IsSameType<xType, half>::value) {
        LocalTensor<xType> scaleInput = computeParams.pingFlag ? scalePingTensor_ : scalePongTensor_;
        DataCopyPad(scaleInput, scaleGlobal_[antiquantOffset], antiquantCopyinParams_, antiquantCopyinPadParams_);
        TEventID eventId1 = GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eventId1);
        WaitFlag<HardEvent::MTE2_V>(eventId1);
        BroadCastAntiquantParams(scaleComputeTensor_, scaleInput);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType,
                                                aL1FullLoad>::AntiquantWeight(ComputeParams &computeParams)
{
    CopyInAntiquantParams(computeParams);
    WeightCopyIn(computeParams);
    WeightCast(computeParams);
    AntiQuantCompute(computeParams);
    WeightCopyOut(computeParams);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType,
                                                aL1FullLoad>::WeightCopyOut(ComputeParams &computeParams)
{
    LocalTensor<xType> weightOutput = weightOutTensor_;
    TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(eventId);
    WaitFlag<HardEvent::MTE3_V>(eventId);
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        Cast(weightOutput, weight32Tensor_, RoundMode::CAST_RINT, computeParams.vecSingleN * computeParams.curSingleK);
    } else {
        DataCopy(weightOutput, weight16Tensor_, computeParams.vecSingleN * computeParams.curSingleK);
    }
    DataCopyParams copyoutParams;
    uint64_t wDstOffset = weightCacheIdx_ * weightCacheSize_;
    wDstOffset += (computeParams.splitNOffset) * BLOCK_CUBE;
    copyoutParams.blockCount = computeParams.curSingleK / BLOCK_CUBE;
    copyoutParams.blockLen = computeParams.vecSingleN;
    copyoutParams.dstStride = 128 - computeParams.vecSingleN;
    copyoutParams.srcStride = 0;
    eventId = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    DataCopy(weightCache_[wDstOffset], weightOutput, copyoutParams);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType,
                                                aL1FullLoad>::SetMatmulParams(ComputeParams &computeParams)
{
    bL0Load2dParams_.startIndex = 0;
    bL0Load2dParams_.repeatTimes = tiling_->cubeBaseK / BLOCK_CUBE * computeParams.curCubeSingleN / BLOCK_CUBE;
    bL0Load2dParams_.srcStride = 1;
    bL0Load2dParams_.dstGap = 0;

    mmadParams_.m = CeilAlign(computeParams.cubeSingleM, BLOCK_CUBE); // 防止cubeSingleM=1时芯片使能GEMV
    mmadParams_.n = computeParams.curCubeSingleN;
    mmadParams_.cmatrixInitVal = computeParams.kLoopIdx == 0;
    mmadParams_.cmatrixSource = false;

    fixParams_.nSize = computeParams.curCubeSingleNOrigin;
    fixParams_.mSize = computeParams.cubeSingleM;
    fixParams_.srcStride = CeilAlign(computeParams.cubeSingleM, BLOCK_CUBE);
    fixParams_.dstStride = tiling_->nSize;
    if constexpr (IsSameType<xType, bfloat16_t>::value) {
        fixParams_.quantPre = hasPostProcessFlag_ ? QuantMode_t::NoQuant : QuantMode_t::F322BF16;
    } else {
        fixParams_.quantPre = hasPostProcessFlag_ ? QuantMode_t::NoQuant : QuantMode_t::F322F16;
    }

    fixParams_.ndNum = 1;
    constexpr uint8_t padList[4] = {0, 0, 0, 0};
    AscendC::SetFmatrix(1, CeilAlign(computeParams.cubeSingleM, BLOCK_CUBE), padList,
                        AscendC::FmatrixMode::FMATRIX_LEFT);
    loadData3DV2_.channelSize = computeParams.curSingleK;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::ComputeMatmulCustom(ComputeParams &computeParams, LocalTensor<xType> &aL1Tensor,
                                      LocalTensor<xType> &bL1Tensor)
{
    LocalTensor<float> cL0Tensor = (computeParams.madLoopIdx & 1) == 1 ? cL0TensorPing_ : cL0TensorPong_;
    LocalTensor<xType> aL0Tensor;
    LocalTensor<xType> bL0Tensor;
    uint64_t cOffset = computeParams.mLoopIdx * tiling_->cubeSingleM * tiling_->nSize +
                       cubeNDimIdx_ * tiling_->cubeSingleCoreN + computeParams.nLoopIdx * tiling_->cubeSingleN;
    SetMatmulParams(computeParams);

    uint32_t stepK = CeilDiv(computeParams.curSingleKOrigin, static_cast<uint16_t>(tiling_->cubeBaseK));
    for (uint32_t kL0Idx = 0; kL0Idx < stepK; kL0Idx++) {
        uint32_t realBaseK =
            (kL0Idx == stepK - 1) ? computeParams.curSingleKOrigin - kL0Idx * tiling_->cubeBaseK : tiling_->cubeBaseK;
        aL0Tensor = (kL0Idx & 1) == 1 ? aL0TensorPing_ : aL0TensorPong_;
        bL0Tensor = (kL0Idx & 1) == 1 ? bL0TensorPing_ : bL0TensorPong_;
        WaitFlag<HardEvent::M_MTE1>(mte1WaitMEventIds_[kL0Idx & 1]);
        uint64_t aL1MPos = 0;
        uint64_t aL1KPos = kL0Idx * tiling_->cubeBaseK;
        uint64_t mStep = CeilAlign(computeParams.cubeSingleM, BLOCK_CUBE);
        uint64_t kStep = realBaseK;
        loadData3DV2_.extConfig = (aL1MPos << 48) | (aL1KPos << 32) | (mStep << 16) | kStep;
        LoadData(aL0Tensor, aL1Tensor, loadData3DV2_);
        LoadData(bL0Tensor, bL1Tensor[kL0Idx * tiling_->cubeBaseK * computeParams.curCubeSingleN], bL0Load2dParams_);
        SetFlag<HardEvent::MTE1_M>(mWaitMte1EventIds_[kL0Idx & 1]);
        WaitFlag<HardEvent::MTE1_M>(mWaitMte1EventIds_[kL0Idx & 1]);
        if (computeParams.madLoopIdx > L0C_DB_NUM - 1 && kL0Idx == 0 && computeParams.kLoopIdx == 0) {
            WaitFlag<HardEvent::FIX_M>(mWaitFixEventIds_[computeParams.madLoopIdx & 1]);
        }
        if (kL0Idx > 0) {
            mmadParams_.cmatrixInitVal = false;
        }
        mmadParams_.k = realBaseK;
        Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams_);
        PipeBarrier<PIPE_M>();
        SetFlag<HardEvent::M_MTE1>(mte1WaitMEventIds_[kL0Idx & 1]);
    }

    if (computeParams.kLoopIdx == computeParams.curSingleCoreKLoop - 1) {
        SetFlag<HardEvent::M_FIX>(fixWaitMEventIds_[computeParams.madLoopIdx & 1]);
        WaitFlag<HardEvent::M_FIX>(fixWaitMEventIds_[computeParams.madLoopIdx & 1]);

        if (hasPostProcessFlag_) {
            SetAtomicAdd<float>();
            Fixpipe(atomicAddWorkspace_[cOffset], cL0Tensor, fixParams_);
            SetAtomicNone();
        } else {
            Fixpipe(yGlobal_[cOffset], cL0Tensor, fixParams_);
        }

        if (computeParams.madLoopIdx + L0C_DB_NUM < computeParams.curCubeSingleCoreNLoop * computeParams.mLoopNum) {
            SetFlag<HardEvent::FIX_M>(mWaitFixEventIds_[computeParams.madLoopIdx & 1]);
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::ProcessCube(ComputeParams &computeParams, LocalTensor<xType> &aL1Tensor)
{
    CopyInBL1(computeParams);
    LocalTensor<xType> bL1Tensor = inQueueBL1_.DeQue<xType>();
    // 一次循环最大计算的m为128
    uint32_t mLoopNum = CeilDiv(tiling_->mSize, 128UL);
    computeParams.mLoopNum = mLoopNum;
    LocalTensor<xType> aL1TensorTmp;
    for (uint32_t mLoopIdx = 0; mLoopIdx < mLoopNum; mLoopIdx++) {
        if (mLoopIdx == mLoopNum - 1) {
            computeParams.cubeSingleM = tiling_->mSize - tiling_->cubeSingleM * mLoopIdx;
        } else {
            computeParams.cubeSingleM = tiling_->cubeSingleM;
        }
        computeParams.mLoopIdx = mLoopIdx;
        if constexpr (aL1FullLoad) {
            uint32_t aL1Index =
                computeParams.kLoopIdx * tiling_->singleK * CeilAlign(computeParams.cubeSingleM, BLOCK_CUBE) +
                computeParams.mLoopIdx * tiling_->cubeSingleM * computeParams.curSingleK;
            if (computeParams.nLoopIdx == 0) {
                CopyInAL1FullLoad(computeParams, aL1Tensor, aL1Index);
                inQueueAL1_.EnQue(aL1Tensor);
                aL1Tensor = inQueueAL1_.DeQue<xType>();
            }
            aL1TensorTmp = aL1Tensor[aL1Index];
        } else {
            CopyInAL1(computeParams);
            aL1TensorTmp = inQueueAL1_.DeQue<xType>();
        }
        computeParams.madLoopIdx = computeParams.nLoopIdx * mLoopNum + computeParams.mLoopIdx;
        ComputeMatmulCustom(computeParams, aL1TensorTmp, bL1Tensor);
        if constexpr (!aL1FullLoad) {
            inQueueAL1_.FreeTensor(aL1TensorTmp);
        }
    }
    inQueueBL1_.FreeTensor(bL1Tensor);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType,
                                                aL1FullLoad>::CopyInAL1(ComputeParams &computeParams)
{
    uint64_t aOffset = computeParams.mLoopIdx * tiling_->cubeSingleM * tiling_->kSize +
                       cubeKDimIdx_ * tiling_->singleCoreK + computeParams.kLoopIdx * tiling_->singleK;
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = computeParams.cubeSingleM;
    nd2nzParams.dValue = computeParams.curSingleK;
    nd2nzParams.srcDValue = tiling_->kSize;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzC0Stride = CeilDiv(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE)) * BLOCK_CUBE;
    LocalTensor<xType> aL1Tensor = inQueueAL1_.AllocTensor<xType>();
    DataCopy(aL1Tensor, xGlobal_[aOffset], nd2nzParams);
    inQueueAL1_.EnQue(aL1Tensor);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::CopyInAL1FullLoad(ComputeParams &computeParams, LocalTensor<xType> &aL1Tensor, uint32_t aL1Index)
{
    uint64_t aOffset = computeParams.mLoopIdx * tiling_->cubeSingleM * tiling_->kSize +
                       cubeKDimIdx_ * tiling_->singleCoreK + computeParams.kLoopIdx * tiling_->singleK;
    Nd2NzParams nd2nzParams;
    nd2nzParams.nValue = computeParams.cubeSingleM;
    nd2nzParams.dstNzC0Stride = CeilDiv(nd2nzParams.nValue, static_cast<uint16_t>(BLOCK_CUBE)) * BLOCK_CUBE;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dValue = computeParams.curSingleK;
    nd2nzParams.srcDValue = tiling_->kSize;
    nd2nzParams.ndNum = 1;
    nd2nzParams.dstNzNStride = 1;
    DataCopy(aL1Tensor[aL1Index], xGlobal_[aOffset], nd2nzParams);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType,
                                                aL1FullLoad>::CopyInBL1(ComputeParams &computeParams)
{
    CrossCoreWaitFlag(CUBE_VEC_FLAG_ID);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = computeParams.curSingleK / BLOCK_CUBE;
    dataCopyParams.blockLen = computeParams.curCubeSingleN;
    dataCopyParams.dstStride = 0;
    dataCopyParams.srcStride = 128 - computeParams.curCubeSingleN; // workspace上分配的N为128

    LocalTensor<xType> bL1Tensor = inQueueBL1_.AllocTensor<xType>();
    uint64_t wSrcOffset = computeParams.totalLoopIdx % WEIGHT_CACHE_COUNT * weightCacheSize_;
    DataCopy(bL1Tensor, weightCache_[wSrcOffset], dataCopyParams);

    inQueueBL1_.EnQue(bL1Tensor);

    if (computeParams.totalLoopIdx + WEIGHT_CACHE_COUNT < computeParams.loopNum) {
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(VEC_CUBE_FLAG_ID);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType, aL1FullLoad>::ReleaseEventIds()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(mte1WaitMEventIds_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(mte1WaitMEventIds_[1]);

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(mWaitMte1EventIds_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(mWaitMte1EventIds_[1]);

    GetTPipePtr()->ReleaseEventID<HardEvent::FIX_M>(mWaitFixEventIds_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::FIX_M>(mWaitFixEventIds_[1]);

    GetTPipePtr()->ReleaseEventID<HardEvent::M_FIX>(fixWaitMEventIds_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_FIX>(fixWaitMEventIds_[1]);

    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVEventIds_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(mte2WaitVEventIds_[1]);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType,
                                                aL1FullLoad>::CalcComputeParams(ComputeParams &computeParams)
{
    if ASCEND_IS_AIV {
        computeParams.isTailCoreN =
            (vecNDimIdx_ == tiling_->vecBlockDimN - 1) || (vecNDimIdx_ == tiling_->vecBlockDimN - 2);
        computeParams.isTailCoreK = vecKDimIdx_ == tiling_->vecBlockDimK - 1;
    } else {
        computeParams.isTailCoreN = cubeNDimIdx_ == tiling_->cubeBlockDimN - 1;
        computeParams.isTailCoreK = cubeKDimIdx_ == tiling_->cubeBlockDimK - 1;
    }
    computeParams.curCubeSingleN = tiling_->cubeSingleN;
    computeParams.curSingleK = tiling_->singleK;
    computeParams.curCubeSingleCoreNLoop =
        computeParams.isTailCoreN ? tiling_->cubeSingleCoreNTailLoop : tiling_->cubeSingleCoreNLoop;
    computeParams.curSingleCoreKLoop =
        computeParams.isTailCoreK ? tiling_->singleCoreKTailLoop : tiling_->singleCoreKLoop;
    computeParams.curCubeSingleCoreN =
        computeParams.isTailCoreN ? tiling_->cubeSingleCoreNTail : tiling_->cubeSingleCoreN;
    computeParams.cubeSingleCoreNOrigin =
        computeParams.isTailCoreN ? tiling_->cubeSingleCoreNOriTail : tiling_->cubeSingleCoreN;
    computeParams.curSingleCoreKOrigin = computeParams.isTailCoreK ? tiling_->singleCoreKOriTail : tiling_->singleCoreK;
    computeParams.loopNum = computeParams.curCubeSingleCoreNLoop * computeParams.curSingleCoreKLoop;
    computeParams.curSingleCoreK = computeParams.isTailCoreK ? tiling_->singleCoreKTail : tiling_->singleCoreK;
    computeParams.curCubeSingleCoreNLoop = computeParams.curCubeSingleCoreNLoop;
    computeParams.curSingleCoreKLoop = computeParams.curSingleCoreKLoop;
    computeParams.vecSingleN = tiling_->vecSingleN;
    computeParams.curCubeSingleNOrigin = tiling_->cubeSingleN;
    computeParams.curSingleKOrigin = tiling_->singleK;
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType,
    aL1FullLoad>::MainProcess(LocalTensor<xType> &aL1Tensor, ComputeParams &computeParams)
{
    for (uint32_t nLoopIdx = 0; nLoopIdx < computeParams.curCubeSingleCoreNLoop; nLoopIdx++) {
        computeParams.curSingleK = tiling_->singleK;
        computeParams.curSingleKOrigin = tiling_->singleK;
        computeParams.nLoopIdx = nLoopIdx;
        if (nLoopIdx == computeParams.curCubeSingleCoreNLoop - 1) {
            computeParams.curCubeSingleNOrigin =
                computeParams.cubeSingleCoreNOrigin - nLoopIdx * computeParams.curCubeSingleN;
            computeParams.curCubeSingleN = computeParams.curCubeSingleCoreN - nLoopIdx * computeParams.curCubeSingleN;
            computeParams.vecSingleN = Min(computeParams.curCubeSingleN / 2, tiling_->vecSingleN);
        }
        for (uint32_t kLoopIdx = 0; kLoopIdx < computeParams.curSingleCoreKLoop; kLoopIdx++) {
            computeParams.kLoopIdx = kLoopIdx;
            computeParams.totalLoopIdx = nLoopIdx * computeParams.curSingleCoreKLoop + kLoopIdx;
            if (kLoopIdx == computeParams.curSingleCoreKLoop - 1) {
                computeParams.curSingleKOrigin =
                    computeParams.curSingleCoreKOrigin - computeParams.curSingleK * kLoopIdx;
                computeParams.curSingleK = computeParams.curSingleCoreK - computeParams.curSingleK * kLoopIdx;
            }
            if ASCEND_IS_AIV {
                if (curBlockIdx_ < tiling_->vecBlockDimN * tiling_->vecBlockDimK) {
                    ProcessVector(computeParams);
                }
            } else {
                if (curBlockIdx_ < tiling_->cubeBlockDimN * tiling_->cubeBlockDimK) {
                    ProcessCube(computeParams, aL1Tensor);
                }
            }
        }
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, bool aL1FullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2CustomNzSplitkKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                                hasAntiQuantOffset, quantType, aL1FullLoad>::Process()
{
    ComputeParams computeParams;
    CalcComputeParams(computeParams);

    LocalTensor<xType> aL1Tensor;
    if ASCEND_IS_AIC {
        if constexpr (aL1FullLoad) {
            aL1Tensor = inQueueAL1_.AllocTensor<xType>();
        }
        SetFlag<HardEvent::M_MTE1>(mte1WaitMEventIds_[0]);
        SetFlag<HardEvent::M_MTE1>(mte1WaitMEventIds_[1]);
    } else {
        SetFlag<HardEvent::V_MTE2>(mte2WaitVEventIds_[0]);
        SetFlag<HardEvent::V_MTE2>(mte2WaitVEventIds_[1]);
    }

    MainProcess(aL1Tensor, computeParams);

    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::M_MTE1>(mte1WaitMEventIds_[0]);
        WaitFlag<HardEvent::M_MTE1>(mte1WaitMEventIds_[1]);
        if constexpr (aL1FullLoad) {
            inQueueAL1_.FreeTensor(aL1Tensor);
        }
    } else {
        WaitFlag<HardEvent::V_MTE2>(mte2WaitVEventIds_[0]);
        WaitFlag<HardEvent::V_MTE2>(mte2WaitVEventIds_[1]);
    }

    if (hasPostProcessFlag_) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(POST_VEC_CUBE_SYNC_ID);
            PostProcess();
        } else {
            CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(POST_VEC_SYNC_ID);
            CrossCoreWaitFlag(POST_VEC_SYNC_ID);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(POST_VEC_CUBE_SYNC_ID);
        }
    }
    ReleaseEventIds();
}
} // namespace Mc2WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_CUSTOM_NZ_SPLITK_H
