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
 * \file weight_quant_batch_matmul_v2_fixpipe.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_FIXPIPE_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_FIXPIPE_H

#include "../tool.h"
#include "../weight_quant_batch_matmul_v2_constant.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "static_diag_constant.h"
#include "weight_quant_batch_matmul_v2_fixpipe_stage1.h"
#include "weight_quant_batch_matmul_v2_fixpipe_stage2.h"

using AscendC::DataCopy;
using AscendC::DataCopyParams;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::InitConstValueParams;
using AscendC::LoadData2DParams;
using AscendC::LocalTensor;
using AscendC::Nd2NzParams;
using AscendC::ONE_BLK_SIZE;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;

namespace Mc2WeightQuantBatchMatmulV2 {
struct OffsetParam {
    uint16_t realBaseM = 0;
    uint16_t realBaseN = 0;
    uint16_t realBaseK = 0;

    uint64_t mOffset = 0;
    uint64_t nOffset = 0;
    uint64_t kOffset = 0;
    uint64_t cOffset = 0;
};

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
class Mc2WeightQuantBatchMatmulV2FixpipeKernel {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2FixpipeKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2FixpipeTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitStage(Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1,
                                     Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2,
                                     const OffsetParam &offsetParam);
    __aicore__ inline void ComputeParams();
    __aicore__ inline void InitGlobalTensor(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                            GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                            GM_ADDR workspace);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void LoadDiag(const LocalTensor<int8_t> &diagL0b);
    __aicore__ inline void ReviseOffsetParams(OffsetParam &offsetParam);
    __aicore__ inline void CopyInputsGmToL1(const OffsetParam &offsetParam,
                                            Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1);
    __aicore__ inline void WeightGmToL1(const OffsetParam &offsetParam);
    __aicore__ inline void AGmToL1(const OffsetParam &offsetParam);
    __aicore__ inline void AntiqScaleOffsetGmToL1(const OffsetParam &offsetParam);
    __aicore__ inline void LoadAntiqScaleOffset(const OffsetParam &offsetParam,
                                                Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1,
                                                Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2);
    __aicore__ inline void BiasGmToL1(const OffsetParam &offsetParam);
    __aicore__ inline void Stage1Process(const LocalTensor<DTYPE_WEIGHT> &weightS8L1, const OffsetParam &offsetParam,
                                         Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1);
    __aicore__ inline void MixStage1Stage2(const LocalTensor<DTYPE_WEIGHT> &weightS8L1,
                                           const LocalTensor<DTYPE_X> &aF16L1, const OffsetParam &offsetParam,
                                           Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1,
                                           Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2);
    __aicore__ inline void WaitStage1FixpFlag(const TEventID &stage1FixpId);
    __aicore__ inline void Stage2Process(const LocalTensor<DTYPE_X> &aF16L1, const OffsetParam &offsetParam,
                                         Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2);
    __aicore__ inline bool Iterate(OffsetParam &curOffsetParam, OffsetParam &preOffsetParam);
    __aicore__ inline void DoIterate(LocalTensor<DTYPE_X> &aF16L1, LocalTensor<DTYPE_WEIGHT> &weightS8L1,
                                     const OffsetParam &preOffsetParam,
                                     Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1,
                                     Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2);
    __aicore__ inline void LoadBias16InBT(const OffsetParam &offsetParam,
                                          Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2);
    __aicore__ inline void FinishStage2();
    __aicore__ inline void ReleaseFlag();

    TPipe *pipe_;
    const Mc2WeightQuantBatchMatmulV2FixpipeTilingData *tiling_;
    TBuf<TPosition::A1> l1TBuf_;
    TBuf<TPosition::A2> l0aTBuf_;
    TBuf<TPosition::B2> l0bTBuf_;
    TBuf<TPosition::CO1> l0cTBuf_;
    TBuf<TPosition::C2> biasTableTBuf_;
    TBuf<TPosition::C2PIPE2GM> fixpipeTableTBuf_;

    GlobalTensor<DTYPE_X> xGlobal_;
    GlobalTensor<int8_t> wGlobal_;
    GlobalTensor<int32_t> antiquantOffsetGlobal_;
    GlobalTensor<uint64_t> antiquantScaleGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<DTYPE_Y> yGlobal_;

    LocalTensor<DTYPE_WEIGHT> weightS8L1Ping_;
    LocalTensor<DTYPE_X> weightF16L1Ping_;
    LocalTensor<DTYPE_WEIGHT> weightS8L1Pong_;
    LocalTensor<DTYPE_X> weightF16L1Pong_;
    LocalTensor<uint64_t> antiScaleL1Ping_;
    LocalTensor<uint64_t> antiScaleL1Pong_;
    LocalTensor<antiquantOffsetType> antiOffsetL1Ping_;
    LocalTensor<antiquantOffsetType> antiOffsetL1Pong_;

    LocalTensor<DTYPE_X> aF16L1Ping_;
    LocalTensor<biasType> biasL1Ping_;
    LocalTensor<DTYPE_X> aF16L1Pong_;
    LocalTensor<biasType> biasL1Pong_;

    int32_t curBlockIdx_;
    uint64_t nIdx_;
    uint64_t mIdx_;

    uint64_t initMOffset_;
    uint64_t initNOffset_;
    uint64_t nLimit_;
    uint64_t mLimit_;

    TEventID stage1FixToMte1EventIds_[2] = {6, 7};
    TEventID stage2FixToMte1EventIds_[2] = {4, 5};
    TEventID mToMte1EventIds_[2] = {6, 7};
    event_t constEventIdMte2ToMTE1_ = {EVENT_ID0};

    int64_t stage1SyncCount_ = 0;
    int64_t mixSyncCount_ = 0;
    uint64_t taskId_ = 1;
    uint64_t preTaskId_ = 0;

    // 处理流程的基本块维持n*k = 32*256
    uint64_t processBaseN_ = 32;
    constexpr static uint64_t processBaseK_ = 256;
};

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2FixpipeKernel<
    antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType, hasAntiquantOffset, hasBias, weightFormat,
    aFullLoad>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
                     GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                     const Mc2WeightQuantBatchMatmulV2FixpipeTilingData *tilingData, TPipe *tPipe)
{
    tiling_ = tilingData;
    curBlockIdx_ = GetBlockIdx();
    pipe_ = tPipe;
    ComputeParams();
    InitGlobalTensor(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace);
    InitBuffer();
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat, aFullLoad>::ComputeParams()
{
    nIdx_ = curBlockIdx_ % tiling_->nBlockNum;
    mIdx_ = curBlockIdx_ / tiling_->nBlockNum;
    initMOffset_ = mIdx_ * tiling_->singleCoreM;
    initNOffset_ = nIdx_ * tiling_->singleCoreN;

    nLimit_ = initNOffset_ + tiling_->singleCoreN;
    nLimit_ = nLimit_ > tiling_->nSize ? tiling_->nSize : nLimit_;

    mLimit_ = initMOffset_ + tiling_->singleCoreM;
    mLimit_ = mLimit_ > tiling_->mSize ? tiling_->mSize : mLimit_;
    taskId_ = 0;
    preTaskId_ = 0;
    processBaseN_ = tiling_->singleCoreN < processBaseN_ ? tiling_->singleCoreN : processBaseN_;
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2FixpipeKernel<
    antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType, hasAntiquantOffset, hasBias, weightFormat,
    aFullLoad>::InitGlobalTensor(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                 GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace)
{
    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(x));
    wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(weight));
    antiquantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(antiquantScale));
    antiquantOffsetGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(antiquantOffset));
    if constexpr (hasBias) {
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias));
    }
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(y));
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat, aFullLoad>::InitBuffer()
{
    pipe_->InitBuffer(l0bTBuf_, L0B_MAX_SIZE_910B);
    pipe_->InitBuffer(l1TBuf_, L1_MAX_SIZE_910B);
    pipe_->InitBuffer(l0aTBuf_, L0A_MAX_SIZE_910B);
    pipe_->InitBuffer(l0cTBuf_, L0C_MAX_SIZE_910B);
    pipe_->InitBuffer(biasTableTBuf_, BIAS_TABLE_MAX_SIZE_910B);
    pipe_->InitBuffer(fixpipeTableTBuf_, FIXPIPE_TABLE_MAX_SIZE_910B);

    weightS8L1Ping_ = l1TBuf_.Get<DTYPE_WEIGHT>();                           // 0-64k给weightS8L1Ping
    weightS8L1Pong_ = l1TBuf_.Get<DTYPE_WEIGHT>()[64 * INT8_DATA_BENCHMARK]; // 64-128k给weightS8L1Pong
    // 128k - 129k给antiScalePing
    antiScaleL1Ping_ = l1TBuf_.Get<uint64_t>()[128 * UINT64_DATA_BENCHMARK];
    // 129k - 130k给antiOffsetPing
    antiOffsetL1Ping_ = l1TBuf_.Get<antiquantOffsetType>()[129 * FLOAT_DATA_BENCHMARK];
    // 130k - 131k给biasL1Ping
    biasL1Ping_ = l1TBuf_.Get<biasType>()[130 * HALF_DATA_BENCHMARK];

    uint64_t weightPingOffset = 131 * HALF_DATA_BENCHMARK;
    if constexpr (aFullLoad) {
        aF16L1Ping_ = l1TBuf_.Get<DTYPE_X>()[weightPingOffset];
        aF16L1Pong_ = l1TBuf_.Get<DTYPE_X>()[weightPingOffset];
    } else {
        aF16L1Ping_ = l1TBuf_.Get<DTYPE_X>()[weightPingOffset];          // 131 - 256k给al1Ping
        aF16L1Pong_ = l1TBuf_.Get<DTYPE_X>()[256 * HALF_DATA_BENCHMARK]; // 256 - 381k给al1Pong
    }
    // 381k - 382k给antiScalePong
    antiScaleL1Pong_ = l1TBuf_.Get<uint64_t>()[381 * UINT64_DATA_BENCHMARK];
    // 382k - 383k给antiScalePong
    antiOffsetL1Pong_ = l1TBuf_.Get<antiquantOffsetType>()[382 * FLOAT_DATA_BENCHMARK];
    // 383k - 384k给biasPing
    biasL1Pong_ = l1TBuf_.Get<biasType>()[383 * HALF_DATA_BENCHMARK];

    weightF16L1Ping_ = l1TBuf_.Get<DTYPE_X>()[384 * HALF_DATA_BENCHMARK]; // 384-448k给weightF16l1Ping
    weightF16L1Pong_ = l1TBuf_.Get<DTYPE_X>()[448 * HALF_DATA_BENCHMARK]; // 448-512k给weightF16l1Pong
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat, aFullLoad>::Process()
{
    // 优先启动对角阵的mte2载入，和后续的scalar并行
    LocalTensor<int8_t> diagL0b = l0bTBuf_.Get<int8_t>();
    LoadDiag(diagL0b);

    uint64_t initCOffset = initMOffset_ * tiling_->nSize + initNOffset_;
    OffsetParam offsetParams[DOUBLE_BUFFER_NUM] = {
        {tiling_->baseM, tiling_->baseN, tiling_->baseK, initMOffset_, initNOffset_, 0UL, initCOffset},
        {tiling_->baseM, tiling_->baseN, tiling_->baseK, initMOffset_, initNOffset_, 0UL, initCOffset}};
    ReviseOffsetParams(offsetParams[0]);

    // 优先启动AL1的mte2载入，和后续的初始化scalar并行
    AGmToL1(offsetParams[0]);

    // 依赖类初始化
    Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> stage1;
    Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> stage2;
    InitStage(stage1, stage2, offsetParams[0]);
    TEventID mte1ToMte2EventId = GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>();

    CopyInputsGmToL1(offsetParams[0], stage1);
    SetFlag<HardEvent::MTE2_MTE1>(constEventIdMte2ToMTE1_);

    // 依赖变量初始化
    taskId_++;
    LocalTensor<DTYPE_WEIGHT> weightS8L1;
    LocalTensor<DTYPE_X> aF16L1;
    aF16L1 = aF16L1Ping_;

    while (Iterate(offsetParams[taskId_ & 1], offsetParams[preTaskId_ & 1])) {
        LoadAntiqScaleOffset(offsetParams[preTaskId_ & 1], stage1, stage2);
        ReviseOffsetParams(offsetParams[taskId_ & 1]);

        WaitFlag<HardEvent::MTE2_MTE1>(constEventIdMte2ToMTE1_);

        if (taskId_ > 1) {
            WaitFlag<HardEvent::MTE1_MTE2>(mte1ToMte2EventId);
        }
        // mte2提前载入下一拍数据
        if constexpr (!aFullLoad) {
            AGmToL1(offsetParams[taskId_ & 1]);
        }
        CopyInputsGmToL1(offsetParams[taskId_ & 1], stage1);
        SetFlag<HardEvent::MTE2_MTE1>(constEventIdMte2ToMTE1_);

        DoIterate(aF16L1, weightS8L1, offsetParams[preTaskId_ & 1], stage1, stage2);
        SetFlag<HardEvent::MTE1_MTE2>(mte1ToMte2EventId);
        preTaskId_ = taskId_;
        taskId_++;
    }
    LoadAntiqScaleOffset(offsetParams[preTaskId_ & 1], stage1, stage2);
    WaitFlag<HardEvent::MTE2_MTE1>(constEventIdMte2ToMTE1_);
    DoIterate(aF16L1, weightS8L1, offsetParams[preTaskId_ & 1], stage1, stage2);
    FinishStage2();
    if (taskId_ > 1) {
        WaitFlag<HardEvent::MTE1_MTE2>(mte1ToMte2EventId);
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(mte1ToMte2EventId);
    ReleaseFlag();
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat, aFullLoad>::ReleaseFlag()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(constEventIdMte2ToMTE1_);
    GetTPipePtr()->ReleaseEventID<HardEvent::FIX_MTE1>(stage1FixToMte1EventIds_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::FIX_MTE1>(stage1FixToMte1EventIds_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds_[1]);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(mToMte1EventIds_[0]);
    GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(mToMte1EventIds_[1]);
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2FixpipeKernel<
    antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType, hasAntiquantOffset, hasBias, weightFormat,
    aFullLoad>::DoIterate(LocalTensor<DTYPE_X> &aF16L1, LocalTensor<DTYPE_WEIGHT> &weightS8L1,
                          const OffsetParam &preOffsetParam,
                          Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1,
                          Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2)
{
    weightS8L1 = (preTaskId_ & 1) == 0 ? weightS8L1Ping_ : weightS8L1Pong_;
    Stage1Process(weightS8L1, preOffsetParam, stage1);

    if constexpr (!aFullLoad) {
        aF16L1 = (preTaskId_ & 1) == 0 ? aF16L1Ping_ : aF16L1Pong_;
    }
    MixStage1Stage2(weightS8L1, aF16L1, preOffsetParam, stage1, stage2);

    Stage2Process(aF16L1, preOffsetParam, stage2);
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat, aFullLoad>::FinishStage2()
{
    if (mixSyncCount_ == 1) {
        WaitFlag<HardEvent::M_MTE1>(mToMte1EventIds_[0]);
        WaitFlag<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds_[0]);
    } else if (mixSyncCount_ > 1) {
        WaitFlag<HardEvent::M_MTE1>(mToMte1EventIds_[0]);
        WaitFlag<HardEvent::M_MTE1>(mToMte1EventIds_[1]);
        WaitFlag<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds_[0]);
        WaitFlag<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds_[1]);
    }
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2FixpipeKernel<
    antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType, hasAntiquantOffset, hasBias, weightFormat,
    aFullLoad>::InitStage(Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1,
                          Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2, const OffsetParam &offsetParam)
{
    LocalTensor<int32_t> antiquantOffsetBT = biasTableTBuf_.Get<int32_t>(); // 0-512B给antiquantOffset
    LocalTensor<float> biasBt = biasTableTBuf_.Get<float>()[128];           // 512B-1024B给antiquantOffset

    LocalTensor<int32_t> weightS32L0c = l0cTBuf_.Get<int32_t>();                        // 0-64k给weightS32
    LocalTensor<float> weightF32L0c = l0cTBuf_.Get<float>()[64 * FLOAT_DATA_BENCHMARK]; // 64 -112k给weightF32

    LocalTensor<int8_t> weightS8L0a = l0aTBuf_.Get<int8_t>();                         // 前16k给weightS8
    LocalTensor<DTYPE_X> aF16L0a = l0aTBuf_.Get<DTYPE_X>()[16 * HALF_DATA_BENCHMARK]; // 16-64k给AF16

    LocalTensor<DTYPE_X> weightF16L0B = l0bTBuf_.Get<DTYPE_X>()[16 * HALF_DATA_BENCHMARK]; // 16-48k给weightF16

    stage1.Init(weightS8L0a, l0bTBuf_.Get<int8_t>(), antiquantOffsetBT, fixpipeTableTBuf_.Get<uint64_t>(),
                weightS32L0c);

    stage1.SetOriShape(tiling_->baseN);
    stage1.SetParams(processBaseK_, processBaseN_);

    stage2.Init(aF16L0a, weightF16L0B, biasBt, weightF32L0c, yGlobal_);
    stage2.SetMadRelatedParams(offsetParam.realBaseM, processBaseN_, processBaseK_, tiling_->baseM);
    stage2.SetFixBiasParams(tiling_->nSize, tiling_->baseN);

    // 迭代过程依赖的同步id提前申请
    stage1FixToMte1EventIds_[0] = GetTPipePtr()->AllocEventID<HardEvent::FIX_MTE1>();
    stage1FixToMte1EventIds_[1] = GetTPipePtr()->AllocEventID<HardEvent::FIX_MTE1>();
    stage2FixToMte1EventIds_[0] = GetTPipePtr()->AllocEventID<HardEvent::FIX_MTE1>();
    stage2FixToMte1EventIds_[1] = GetTPipePtr()->AllocEventID<HardEvent::FIX_MTE1>();
    mToMte1EventIds_[0] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
    mToMte1EventIds_[1] = GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>();
    constEventIdMte2ToMTE1_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat,
                                         aFullLoad>::ReviseOffsetParams(OffsetParam &offsetParam)
{
    // 求解真实的baseM
    offsetParam.realBaseM =
        offsetParam.mOffset + tiling_->baseM > mLimit_ ? mLimit_ - offsetParam.mOffset : tiling_->baseM;

    // 求解真实的baseN
    offsetParam.realBaseN =
        offsetParam.nOffset + tiling_->baseN > nLimit_ ? nLimit_ - offsetParam.nOffset : tiling_->baseN;

    // 求解真实的baseK
    offsetParam.realBaseK =
        offsetParam.kOffset + tiling_->baseK > tiling_->kSize ? tiling_->kSize - offsetParam.kOffset : tiling_->baseK;
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat,
                                         aFullLoad>::LoadDiag(const LocalTensor<int8_t> &diagL0b)
{
    // 初始化l0a空间
    InitConstValueParams<int32_t> initDiagParams;
    initDiagParams.repeatTimes = 1;
    initDiagParams.blockNum = 32;
    initDiagParams.dstGap = 0;
    initDiagParams.initValue = 0;
    AscendC::InitConstValue(diagL0b.template ReinterpretCast<int32_t>(), initDiagParams);

    GlobalTensor<int8_t> diagGm;
    diagGm.SetGlobalBuffer((__gm__ int8_t *)(FIXP_EYE_DIAG));
// diag在gm上，需要告知oom框架diag的地址
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    AscendC::OOMCheckAddrRange((__gm__ uint8_t *)(FIXP_EYE_DIAG), 1024);
#endif
    uint64_t bOffset = 0;
    DataCopyParams dmaParams;
    dmaParams.blockCount = 32; // 对角阵默认1024B,总共32个blk
    dmaParams.blockLen = 1;
    dmaParams.srcStride = 0;
    dmaParams.dstStride = 0;
    // diag不在l1常驻，暂用weight的pong暂存
    DataCopy(weightS8L1Pong_, diagGm, dmaParams);

    event_t eventIdMte2ToMTE1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
    SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMTE1);
    WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMTE1);
    LoadData2DParams l1ToL0bParams;
    l1ToL0bParams.startIndex = 0;
    l1ToL0bParams.repeatTimes = 2;
    l1ToL0bParams.srcStride = 1;
    l1ToL0bParams.dstGap = 0;
    LoadData(diagL0b, weightS8L1Pong_, l1ToL0bParams);
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2FixpipeKernel<
    antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType, hasAntiquantOffset, hasBias, weightFormat,
    aFullLoad>::CopyInputsGmToL1(const OffsetParam &offsetParam,
                                 Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1)
{
    if (unlikely(offsetParam.kOffset == 0)) {
        // per channel场景, scale, offset, bias等在n方向复用，仅需要载入一次
        AntiqScaleOffsetGmToL1(offsetParam);
        BiasGmToL1(offsetParam);
    }
    WeightGmToL1(offsetParam);
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat,
                                         aFullLoad>::WeightGmToL1(const OffsetParam &offsetParam)
{
    LocalTensor<DTYPE_WEIGHT> weightS8L1;
    weightS8L1 = (taskId_ & 1) == 0 ? weightS8L1Ping_ : weightS8L1Pong_;

    if constexpr (weightFormat != CubeFormat::NZ) {
        uint64_t bOffset = 0;
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = offsetParam.realBaseN;
        nd2nzParams.dValue = offsetParam.realBaseK;
        nd2nzParams.srcDValue = tiling_->kSize;
        bOffset = offsetParam.nOffset * nd2nzParams.srcDValue + offsetParam.kOffset;

        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;

        nd2nzParams.dstNzC0Stride = tiling_->baseN;
        DataCopy(weightS8L1, wGlobal_[bOffset], nd2nzParams);
    } else {
        uint64_t bOffset = 0;
        DataCopyParams dmaParams;
        dmaParams.blockCount = offsetParam.realBaseK / ONE_BLK_SIZE;
        dmaParams.blockLen = offsetParam.realBaseN;
        dmaParams.srcStride = tiling_->nSize - offsetParam.realBaseN;
        bOffset = tiling_->nSize * offsetParam.kOffset + offsetParam.nOffset * ONE_BLK_SIZE;
        dmaParams.dstStride = 0;
        DataCopy(weightS8L1, wGlobal_[bOffset], dmaParams);
    }
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat,
                                         aFullLoad>::AGmToL1(const OffsetParam &offsetParam)
{
    LocalTensor<DTYPE_X> aF16L1;
    aF16L1 = (taskId_ & 1) == 0 ? aF16L1Ping_ : aF16L1Pong_;
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.srcDValue = tiling_->kSize;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = tiling_->baseM;

    uint64_t aOffset = 0;
    if constexpr (aFullLoad) {
        aOffset = offsetParam.mOffset * nd2nzParams.srcDValue;
        nd2nzParams.nValue = offsetParam.realBaseM;
        nd2nzParams.dValue = tiling_->kSize;
        DataCopy(aF16L1, xGlobal_[aOffset], nd2nzParams);
    } else {
        nd2nzParams.nValue = offsetParam.realBaseM;
        nd2nzParams.dValue = offsetParam.realBaseK;
        aOffset = offsetParam.mOffset * nd2nzParams.srcDValue + offsetParam.kOffset;
        DataCopy(aF16L1, xGlobal_[aOffset], nd2nzParams);
    }
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat,
                                         aFullLoad>::AntiqScaleOffsetGmToL1(const OffsetParam &offsetParam)
{
    LocalTensor<uint64_t> antiquantScaleL1;
    antiquantScaleL1 = (taskId_ & 1) == 0 ? antiScaleL1Ping_ : antiScaleL1Pong_;

    DataCopyParams dmaParams;
    dmaParams.blockCount = offsetParam.realBaseN * sizeof(uint64_t) / ONE_BLK_SIZE;
    dmaParams.blockLen = 1;
    dmaParams.srcStride = 0;
    dmaParams.dstStride = 0;
    DataCopy(antiquantScaleL1, antiquantScaleGlobal_[offsetParam.nOffset], dmaParams);
    event_t eventIdMte2ToFixp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_FIX));
    SetFlag<HardEvent::MTE2_FIX>(eventIdMte2ToFixp);
    WaitFlag<HardEvent::MTE2_FIX>(eventIdMte2ToFixp);
    if constexpr (hasAntiquantOffset) {
        LocalTensor<antiquantOffsetType> antiquantOffsetL1;
        antiquantOffsetL1 = (taskId_ & 1) == 0 ? antiOffsetL1Ping_ : antiOffsetL1Pong_;
        dmaParams.blockCount = offsetParam.realBaseN * sizeof(antiquantOffsetType) / ONE_BLK_SIZE;
        DataCopy(antiquantOffsetL1, antiquantOffsetGlobal_[offsetParam.nOffset], dmaParams);
        event_t eventIdMte2ToMTE1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
        SetFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMTE1);
        WaitFlag<HardEvent::MTE2_MTE1>(eventIdMte2ToMTE1);
    }
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2FixpipeKernel<
    antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType, hasAntiquantOffset, hasBias, weightFormat,
    aFullLoad>::LoadAntiqScaleOffset(const OffsetParam &offsetParam,
                                     Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1,
                                     Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2)
{
    if (unlikely(offsetParam.kOffset == 0)) {
        // gmToL1的过程，已经补充了相关的mte2同步，此处无需补充
        LocalTensor<uint64_t> antiquantScaleL1;
        antiquantScaleL1 = (preTaskId_ & 1) == 0 ? antiScaleL1Ping_ : antiScaleL1Pong_;
        stage1.AntiqScaleToFixpipe(antiquantScaleL1);
        if constexpr (hasAntiquantOffset) {
            LocalTensor<antiquantOffsetType> antiquantOffsetL1;
            antiquantOffsetL1 = (preTaskId_ & 1) == 0 ? antiOffsetL1Ping_ : antiOffsetL1Pong_;
            stage1.AntiqOffsetToBT(antiquantOffsetL1);
        }
    }
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat,
                                         aFullLoad>::BiasGmToL1(const OffsetParam &offsetParam)
{
    if constexpr (!hasBias) {
        return;
    }

    LocalTensor<biasType> biasL1;
    biasL1 = (taskId_ & 1) == 0 ? biasL1Ping_ : biasL1Pong_;

    DataCopyParams dmaParams;
    dmaParams.blockCount = offsetParam.realBaseN * sizeof(biasType) / ONE_BLK_SIZE;
    dmaParams.blockLen = 1;
    dmaParams.srcStride = 0;
    dmaParams.dstStride = 0;
    DataCopy(biasL1, biasGlobal_[offsetParam.nOffset], dmaParams);
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2FixpipeKernel<
    antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType, hasAntiquantOffset, hasBias, weightFormat,
    aFullLoad>::Stage1Process(const LocalTensor<DTYPE_WEIGHT> &weightS8L1, const OffsetParam &offsetParam,
                              Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1)
{
    stage1SyncCount_ = 0;
    TEventID stage1FixToMte1EventIds[2] = {stage1FixToMte1EventIds_[0], stage1FixToMte1EventIds_[1]};
    TEventID stage2FixToMte1EventIds[2] = {stage2FixToMte1EventIds_[0], stage2FixToMte1EventIds_[1]};
    TEventID mToMte1EventIds[2] = {mToMte1EventIds_[0], mToMte1EventIds_[1]};

    // 循环过程中，stage1可能和上一次循环的stage2互相干扰，需要确保上一次的stage2已完成或即将完成
    if (mixSyncCount_ == 1) {
        WaitFlag<HardEvent::M_MTE1>(mToMte1EventIds[0]);
        WaitFlag<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds_[0]);
    } else if (mixSyncCount_ > 1) {
        WaitFlag<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds[mixSyncCount_ & 1]);
        WaitFlag<HardEvent::M_MTE1>(mToMte1EventIds[mixSyncCount_ & 1]);
        mixSyncCount_++;
    }

    // 单次处理的基本块为n * k = 32 * 256， n方向循环四次，可以消费完128*256，即一半的mte2数据
    for (uint64_t nOffset = 0; nOffset < offsetParam.realBaseN; nOffset += processBaseN_) {
        if (stage1SyncCount_ > 1) {
            WaitFlag<HardEvent::FIX_MTE1>(stage1FixToMte1EventIds[stage1SyncCount_ & 1]);
        }
        // 涉及流水为mte1/mmad/fixp
        stage1.Process(weightF16L1Ping_[nOffset * processBaseK_], weightS8L1[nOffset * INT8_BLOCK_SIZE], nOffset,
                       processBaseK_, processBaseN_);
        SetFlag<HardEvent::FIX_MTE1>(stage1FixToMte1EventIds[stage1SyncCount_ & 1]);
        stage1SyncCount_++;
    }

    if (mixSyncCount_ > 1) {
        WaitFlag<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds[mixSyncCount_ & 1]);
        WaitFlag<HardEvent::M_MTE1>(mToMte1EventIds[mixSyncCount_ & 1]);
    }
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2FixpipeKernel<
    antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType, hasAntiquantOffset, hasBias, weightFormat,
    aFullLoad>::MixStage1Stage2(const LocalTensor<DTYPE_WEIGHT> &weightS8L1, const LocalTensor<DTYPE_X> &aF16L1,
                                const OffsetParam &offsetParam,
                                Mc2WeightQuantBatchMatmulV2FixpipeStage1<hasAntiquantOffset> &stage1,
                                Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2)
{
    // k轴变化会导致l0a上m,k排布的数据间隔有变化，此处需要根据baseK值实时设置避免mmad乘脏数据
    uint64_t realBaseK = offsetParam.realBaseK > processBaseK_ ? processBaseK_ : offsetParam.realBaseK;
    stage2.ReSetParams(realBaseK);

    // a矩阵一次载入m*processBaseK_，bias一次载入realBaseN,
    // 因此stage2可以将上一次stage1产生的realBaseN*processBaseK完全消费
    if constexpr (!aFullLoad) {
        stage2.LoadA16InA2(aF16L1);
    } else {
        stage2.LoadA16InA2(aF16L1[tiling_->baseM * offsetParam.kOffset]);
    }
    LoadBias16InBT(offsetParam, stage2);

    TEventID stage1FixToMte1EventIds[2] = {stage1FixToMte1EventIds_[0], stage1FixToMte1EventIds_[1]};
    TEventID stage2FixToMte1EventIds[2] = {stage2FixToMte1EventIds_[0], stage2FixToMte1EventIds_[1]};
    TEventID mToMte1EventIds[2] = {mToMte1EventIds_[0], mToMte1EventIds_[1]};

    // stage2需要等待stage1的fixp结束
    WaitStage1FixpFlag(stage1FixToMte1EventIds[stage1SyncCount_ & 1]);

    uint64_t weightS8BaseOffset = tiling_->baseN * processBaseK_;
    uint64_t stage1DbOffset = 0;
    bool needFixToGm = offsetParam.kOffset + processBaseK_ >= tiling_->kSize;
    mixSyncCount_ = 0;
    for (uint64_t nOffset = 0; nOffset < offsetParam.realBaseN; nOffset += processBaseN_) {
        if (mixSyncCount_ > 1) {
            WaitFlag<HardEvent::M_MTE1>(mToMte1EventIds[mixSyncCount_ & 1]);
            WaitFlag<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds[mixSyncCount_ & 1]);
        }

        stage2.Process1(weightF16L1Ping_[nOffset * processBaseK_]);
        if (likely(offsetParam.realBaseK > processBaseK_)) {
            // 若实际mte2载入的realBaseK大于processBaseK_，则触发第二轮stage1，期望流水和stage2并行
            stage1DbOffset = stage1.Process1(weightS8L1[nOffset * INT8_BLOCK_SIZE + weightS8BaseOffset]);
        }
        event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
        SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
        WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
        if (likely(offsetParam.realBaseK > processBaseK_)) {
            // 涉及流水为mmad/fixp
            stage1.Process2(weightF16L1Pong_[nOffset * processBaseK_], nOffset, stage1DbOffset, processBaseK_,
                            processBaseN_);
        }
        // 涉及流水为mmad/fixp
        stage2.Process2(nOffset, offsetParam.kOffset == 0, needFixToGm, offsetParam.cOffset + nOffset);
        SetFlag<HardEvent::M_MTE1>(mToMte1EventIds[mixSyncCount_ & 1]);
        SetFlag<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds[mixSyncCount_ & 1]);
        mixSyncCount_++;
    }

    if (likely(stage1SyncCount_ > 1)) {
        WaitFlag<HardEvent::FIX_MTE1>(stage1FixToMte1EventIds[stage1SyncCount_ & 1]);
    }
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat,
                                         aFullLoad>::WaitStage1FixpFlag(const TEventID &stage1FixpId)
{
    if (unlikely(stage1SyncCount_ == 1)) {
        // s1只有一拍，属于尾块场景，通过同步隔离s1s2
        WaitFlag<HardEvent::FIX_MTE1>(stage1FixToMte1EventIds_[0]);
    } else if (likely(stage1SyncCount_ > 1)) {
        // s1有两拍以上，避免mix阶段的s1踩踏之前的pingpong buffer
        WaitFlag<HardEvent::FIX_MTE1>(stage1FixpId);
        stage1SyncCount_++;
    }
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2FixpipeKernel<
    antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType, hasAntiquantOffset, hasBias, weightFormat,
    aFullLoad>::LoadBias16InBT(const OffsetParam &offsetParam,
                               Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2)
{
    if (unlikely(offsetParam.kOffset == 0)) {
        if constexpr (hasBias) {
            LocalTensor<biasType> biasL1;
            biasL1 = (preTaskId_ & 1) == 0 ? biasL1Ping_ : biasL1Pong_;
            stage2.LoadBias16InBT(biasL1);
        }
        // 当前l0c的c矩阵无db，c矩阵的mmad之前需要等待写出做完
        event_t eventIdFixToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_M));
        SetFlag<HardEvent::FIX_M>(eventIdFixToM);
        WaitFlag<HardEvent::FIX_M>(eventIdFixToM);
    }
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2FixpipeKernel<
    antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType, hasAntiquantOffset, hasBias, weightFormat,
    aFullLoad>::Stage2Process(const LocalTensor<DTYPE_X> &aF16L1, const OffsetParam &offsetParam,
                              Mc2WeightQuantBatchMatmulV2FixpipeStage2<hasBias> &stage2)
{
    TEventID stage2FixToMte1EventIds[2] = {stage2FixToMte1EventIds_[0], stage2FixToMte1EventIds_[1]};
    TEventID mToMte1EventIds[2] = {mToMte1EventIds_[0], mToMte1EventIds_[1]};
    event_t eventIdMToMTE1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
    SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1);
    WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1);

    if (likely(offsetParam.realBaseK > processBaseK_)) {
        // 若实际mte2载入的realBaseK大于processBaseK_，则触发第二轮stage2，消费第二轮stage1产生的数据
        stage2.ReSetParams(offsetParam.realBaseK - processBaseK_);
        if constexpr (!aFullLoad) {
            stage2.LoadA16InA2(aF16L1[tiling_->baseM * processBaseK_]);
        } else {
            stage2.LoadA16InA2(aF16L1[tiling_->baseM * (offsetParam.kOffset + processBaseK_)]);
        }
        LoadBias16InBT(offsetParam, stage2);
    }

    event_t eventIdMte1ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
    SetFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMTE2);
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdMte1ToMTE2);
    bool needFixToGm = offsetParam.kOffset + tiling_->baseK >= tiling_->kSize;
    for (uint64_t nOffset = 0; nOffset < offsetParam.realBaseN; nOffset += processBaseN_) {
        if (likely(offsetParam.realBaseK > processBaseK_)) {
            if (mixSyncCount_ > 1) {
                WaitFlag<HardEvent::M_MTE1>(mToMte1EventIds[mixSyncCount_ & 1]);
                WaitFlag<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds[mixSyncCount_ & 1]);
            }
            stage2.Process1(weightF16L1Pong_[nOffset * processBaseK_]);
            event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
            SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
            WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
            stage2.Process2(nOffset, false, needFixToGm,
                            offsetParam.cOffset + nOffset); // mte1/mmad/fixp
            SetFlag<HardEvent::M_MTE1>(mToMte1EventIds[mixSyncCount_ & 1]);
            SetFlag<HardEvent::FIX_MTE1>(stage2FixToMte1EventIds[mixSyncCount_ & 1]);
            mixSyncCount_++;
        }
    }
}

template <typename antiquantOffsetType, typename biasType, bool aTrans, bool bTrans, Mc2QuantType antiquantType,
          Mc2QuantType quantType, bool hasAntiquantOffset, bool hasBias, CubeFormat weightFormat, bool aFullLoad>
__aicore__ inline bool
Mc2WeightQuantBatchMatmulV2FixpipeKernel<antiquantOffsetType, biasType, aTrans, bTrans, antiquantType, quantType,
                                         hasAntiquantOffset, hasBias, weightFormat,
                                         aFullLoad>::Iterate(OffsetParam &curOffsetParam, OffsetParam &preOffsetParam)
{
    curOffsetParam.kOffset = preOffsetParam.kOffset + tiling_->baseK;
    if (curOffsetParam.kOffset >= tiling_->kSize) {
        // k轴累加超过K，则切换到下一个n
        curOffsetParam.kOffset = 0;
        curOffsetParam.nOffset = preOffsetParam.nOffset + tiling_->baseN;
    } else {
        curOffsetParam.nOffset = preOffsetParam.nOffset;
    }

    if (curOffsetParam.nOffset >= nLimit_) {
        // n轴累加超过单核实际处理的n，则切换到下一个m
        curOffsetParam.mOffset = preOffsetParam.mOffset + tiling_->baseM;
    } else {
        curOffsetParam.mOffset = preOffsetParam.mOffset;
    }
    curOffsetParam.cOffset = curOffsetParam.mOffset * tiling_->nSize + curOffsetParam.nOffset;

    // 若m轴累加超过单核实际处理的m，则标记无需继续迭代
    return curOffsetParam.mOffset < mLimit_;
}
} // namespace Mc2WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_FIXPIPE_H
