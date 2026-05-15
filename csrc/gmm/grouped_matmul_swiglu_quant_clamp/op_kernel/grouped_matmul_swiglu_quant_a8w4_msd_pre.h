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
 * \file grouped_matmul_swiglu_quant_a8w4_msd_pre.h
 * \brief
 */

#ifndef ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_A8W4_MSD_PRE_H
#define ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_A8W4_MSD_PRE_H
#include "grouped_matmul_swiglu_quant_utils.h"
#include "kernel_operator.h"
#ifdef GMM_SWIGLU_QUANT_A8W4_MSD
namespace GROUPED_MATMUL_SWIGLU_QUANT {
using namespace AscendC;
#define BUFFER_NUM_A8W4_PRE 1
constexpr int TWO = 2;
constexpr int EIGHT = 8;
constexpr size_t LEN_128 = 128; // 16bit operator
constexpr int DATA_BLOCK_SIZE_32 = 32;
class GMMA8W4PreProcess {
public:
    __aicore__ inline GMMA8W4PreProcess(){};
    __aicore__ inline void Init(const GMAddrParams gmAddrParams,
                                const GMMSwigluBaseParams *__restrict gmmSwigluBaseParamsIN);
    __aicore__ inline void CalculateTaskInfoEachCore(uint32_t &curCoreTaskNum_, uint32_t &curCoreStartOffset_);
    __aicore__ inline void Process(WorkSpaceSplitConfig &workspaceSplitConfig, int64_t workspaceSplitLoopIdx,
                                   TPipe *pipe);
    __aicore__ inline void CustomInitBuffer(TPipe *pipe);

private:
    TQue<QuePosition::VECIN, BUFFER_NUM_A8W4_PRE> vecInQueueX, vecInQueueXBak;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueA1;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueA2;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueA3;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueue0F;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueRowSum;
    TBuf<TPosition::VECCALC> tempBuff;
    const GMMSwigluBaseParams *__restrict gmmSwigluBaseParams;
    LocalTensor<int8_t> xTensor;
    LocalTensor<half> xHighHalfTensor;
    LocalTensor<float> xHighFloatTensor;
    LocalTensor<half> xLowHalfTensor;
    LocalTensor<half> xLowHalfTensor2;
    LocalTensor<int4b_t> xHighI4Tensor;
    LocalTensor<int4b_t> xLowI4Tensor;
    LocalTensor<int16_t> xLowI16Tensor;
    LocalTensor<int64_t> groupListTensor;
    LocalTensor<float> groupListFTensor;
    LocalTensor<float> xRowSumTensor;

    GlobalTensor<int8_t> xGM;
    GlobalTensor<int8_t> yGm;
    GlobalTensor<int8_t> yGm1;
    GlobalTensor<int8_t> yGm2;
    GlobalTensor<int64_t> groupListGM;

    uint32_t vK{0};
    uint32_t vKAlign{0};
    uint32_t totalM{0};
    uint32_t blockDim{0};
    uint32_t curCoreId{0};
    uint32_t curCoreTaskNum{0};
    uint32_t curCoreStartOffset{0};
    uint32_t curCoreOuterLoopNum{0};
    uint32_t curCoreInnerTailLoopNum{0};
    uint32_t groupNum{0};
};

__aicore__ inline void GMMA8W4PreProcess::Init(const GMAddrParams gmAddrParams,
                                               const GMMSwigluBaseParams *__restrict gmmSwigluBaseParamsIN)
{
    if ASCEND_IS_AIV {
        xGM.SetGlobalBuffer((__gm__ int8_t *)gmAddrParams.xGM);
        yGm1.SetGlobalBuffer((__gm__ int8_t *)gmAddrParams.workSpaceGM);
        yGm2.SetGlobalBuffer((__gm__ int8_t *)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset1);
        groupListGM.SetGlobalBuffer((__gm__ int64_t *)gmAddrParams.groupListGM);
        gmmSwigluBaseParams = gmmSwigluBaseParamsIN;
        vK = gmmSwigluBaseParams->K;
        groupNum = static_cast<uint32_t>(gmmSwigluBaseParams->groupNum);
        // M * K * 7B (1B + 0.5B + 0.5B + 2B + 4B) <= UBsize - 256B
        blockDim = GetBlockNum() * GetTaskRation();
    }
}

__aicore__ inline void GMMA8W4PreProcess::CustomInitBuffer(TPipe *pipe)
{
    pipe->InitBuffer(vecInQueueX, BUFFER_NUM_A8W4_PRE, vK * sizeof(int8_t));    // K * 1B
    pipe->InitBuffer(vecOutQueueA1, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // K * 0.5B
    pipe->InitBuffer(vecOutQueueA2, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // K * 0.5B
    pipe->InitBuffer(vecOutQueueA3, BUFFER_NUM_A8W4_PRE, vK * sizeof(half));    // K * 2B
    // xLowHalfTensor, xLowHalfTensor2 and xHighFloatTensor share the same buffer
    pipe->InitBuffer(tempBuff, vK * sizeof(float)); // K * 4B
    constexpr int BUFFER_SIZE_256B = 128 * sizeof(int16_t);
    pipe->InitBuffer(vecOutQueue0F, BUFFER_NUM_A8W4_PRE, BUFFER_SIZE_256B); // 256B
}


__aicore__ inline void GMMA8W4PreProcess::CalculateTaskInfoEachCore(uint32_t &curCoreTaskNum_,
                                                                    uint32_t &curCoreStartOffset_)
{
    // 均分任务数
    int64_t eachCoreTaskNum = (totalM + blockDim - 1) / blockDim; // 每个核处理的数据量
    // 尾核任务数
    int64_t taskNumPertailCore = eachCoreTaskNum - 1;
    // 实际使用核数
    int64_t usedCoreNum = totalM >= blockDim ? blockDim : totalM;
    // 尾核起始索引
    uint32_t tailCoreIdx = totalM - (eachCoreTaskNum - 1) * usedCoreNum;
    curCoreId = GetBlockIdx();
    // 每个核处理的任务数量 = 是否为尾核 ？均分任务数 ：(均分任务数 - 1)
    curCoreTaskNum_ = curCoreId < tailCoreIdx ? eachCoreTaskNum : eachCoreTaskNum - 1;
    // 每个核处理的起始偏移地址 = 是否为尾核 ？均分任务数 * blockId : (均分任务数 - 1) * blockId + 尾核起始索引
    curCoreStartOffset_ =
        curCoreId < tailCoreIdx ? eachCoreTaskNum * curCoreId : ((eachCoreTaskNum - 1) * curCoreId + tailCoreIdx);
}

__aicore__ inline void GMMA8W4PreProcess::Process(WorkSpaceSplitConfig &workspaceSplitConfig,
                                                  int64_t workspaceSplitLoopIdx, TPipe *pipe)
{
    if ASCEND_IS_AIV {
        if (workspaceSplitLoopIdx >= workspaceSplitConfig.loopCount) {
            return;
        }
        yGm = (workspaceSplitLoopIdx % 2 == 0 ? yGm1 : yGm2);
        CustomInitBuffer(pipe);
        constexpr int32_t MASK = 128;
        xTensor = vecInQueueX.AllocTensor<int8_t>();
        xHighI4Tensor = vecOutQueueA1.AllocTensor<int4b_t>();
        xLowI4Tensor = vecOutQueueA2.AllocTensor<int4b_t>();
        xHighHalfTensor = vecOutQueueA3.AllocTensor<half>();
        const uint32_t xLowHalfOffset = vK * sizeof(half);
        xLowHalfTensor = tempBuff.GetWithOffset<half>(xLowHalfOffset, 0);
        xLowHalfTensor2 = tempBuff.GetWithOffset<half>(xLowHalfOffset, xLowHalfOffset);
        xLowI16Tensor = vecOutQueue0F.AllocTensor<int16_t>();

        Duplicate(xLowI16Tensor, static_cast<int16_t>(0x0F0F), MASK); // get rid of high 4 bits in every int8
        PipeBarrier<PIPE_V>();
        const size_t LEN_VK = (vK / 2) / 128;
        const size_t LAST_LEN_VK = (vK % 256) / 2;
        const half ONE_SIXTEENTH = static_cast<half>(0.0625f);
        // groupList仅支持count
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        totalM = workspaceSplitLoopIdx < workspaceSplitConfig.loopCount - 1 ? workspaceSplitConfig.notLastTaskSize :
                                                                              workspaceSplitConfig.lastLoopTaskSize;
        SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
        CalculateTaskInfoEachCore(curCoreTaskNum, curCoreStartOffset);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); // 0
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0); // 1
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1); // 2

        for (uint32_t xloop = 0; xloop < curCoreTaskNum; xloop++) {
            uint64_t relStartAddr = (xloop + curCoreStartOffset) * vK;
            uint64_t absStartAddr = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize * vK + relStartAddr;
            // 高四位处理开始
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0); // 0
            DataCopy(xTensor, xGM[absStartAddr], vK);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);  // 3
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0); // 3
            Cast(xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, vK);
            PipeBarrier<PIPE_V>();
            Muls(xHighHalfTensor, xHighHalfTensor, ONE_SIXTEENTH, vK);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID1); // 2
            Cast(xHighI4Tensor, xHighHalfTensor, AscendC::RoundMode::CAST_FLOOR, vK);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);  // 4
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0); // 4
            DataCopy(yGm[relStartAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), vK / 2);
            // 高四位处理结束

            // 低四位处理开始
            SetFlag<HardEvent::MTE3_V>(EVENT_ID1); // 2
            And(xLowHalfTensor.ReinterpretCast<int16_t>(), xTensor.ReinterpretCast<int16_t>(), xLowI16Tensor, LEN_128,
                LEN_VK, {1, 1, 1, 8, 8, 0});
            if (LAST_LEN_VK > 0) {
                And(xLowHalfTensor[LEN_VK * LEN_128].ReinterpretCast<int16_t>(),
                    xTensor[LEN_VK * LEN_128 * TWO].ReinterpretCast<int16_t>(), xLowI16Tensor, LAST_LEN_VK, 1,
                    {1, 1, 1, 8, 8, 0});
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); // 0
            Cast(xLowHalfTensor2.ReinterpretCast<half>(), xLowHalfTensor.ReinterpretCast<int8_t>(),
                 AscendC::RoundMode::CAST_NONE, vK);
            PipeBarrier<PIPE_V>();
            const half MINUS_EIGHT = static_cast<half>(-8);
            Adds(xHighHalfTensor, xLowHalfTensor2, MINUS_EIGHT, vK);
            PipeBarrier<PIPE_V>();
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0); // 1
            Cast(xLowI4Tensor, xHighHalfTensor.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE, vK);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID1);  // 5
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID1); // 5
            DataCopy(yGm[relStartAddr + vK / TWO], xLowI4Tensor.ReinterpretCast<int8_t>(), vK / TWO);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0); // 1
            // 低四位处理结束
        }

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0); // 0
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0); // 1
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1); // 2
        vecInQueueX.FreeTensor(xTensor);
        vecOutQueueA1.FreeTensor(xHighI4Tensor);
        vecOutQueueA2.FreeTensor(xLowI4Tensor);
        vecOutQueueA3.FreeTensor(xHighHalfTensor);
        vecOutQueue0F.FreeTensor(xLowI16Tensor);
    }
}

} // namespace GROUPED_MATMUL_SWIGLU_QUANT
#endif // GMM_SWIGLU_QUANT_A8W4_MSD
#endif // ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_A8W4_MSD_PRE_H