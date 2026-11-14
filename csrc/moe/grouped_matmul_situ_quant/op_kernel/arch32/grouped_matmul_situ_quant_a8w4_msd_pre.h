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
 * \file grouped_matmul_situ_quant_a8w4_msd_pre.h
 * \brief
 */

#ifndef OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_PRE_H
#define OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_PRE_H

#include "grouped_matmul_situ_quant_utils_kernel.h"
#include "kernel_operator.h"

#ifdef GMM_SITU_QUANT_A8W4_MSD

namespace GroupedMatmulDequantSituQuant {
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
                                const GMMSituQuantBaseParams* __restrict gmmSituQuantBaseParamsIN);
    __aicore__ inline void CalculateTaskInfoEachCore(uint32_t& curCoreTaskNum_, uint32_t& curCoreStartOffset_);
    __aicore__ inline void Process(WorkSpaceSplitConfig& workspaceSplitConfig, int64_t workspaceSplitLoopIdx,
                                   TPipe* pipe);
    __aicore__ inline void CustomInitBuffer(TPipe* pipe);

private:
    // 第 e 个专家在 groupList 上的绝对累计行数（type0 是累计末值，type1 是前缀和）。
    __aicore__ inline int64_t GroupCumEnd(uint32_t e, int64_t cumPrev) const
    {
        return groupListType == 0 ? groupListGM.GetValue(e) : cumPrev + groupListGM.GetValue(e);
    }
    // 本 split 内 [0, splitTokens) 里，位于 pos 之后的第一个非空专家块的结束下标；
    // 没有更多块时返回 splitTokens。splitStartAbs 是本 split 在整个 groupList 上的起点。
    __aicore__ inline uint32_t NextChunkEnd(uint32_t pos, uint32_t splitTokens, int64_t splitStartAbs) const
    {
        const int64_t splitEnd = splitStartAbs + splitTokens;
        int64_t cumPrev = 0;
        uint32_t localTok = 0;
        for (uint32_t e = 0; e < groupNum; e++) {
            const int64_t cumE = GroupCumEnd(e, cumPrev);
            const int64_t cs = cumPrev > splitStartAbs ? cumPrev : splitStartAbs;
            const int64_t ce = cumE < splitEnd ? cumE : splitEnd;
            if (ce > cs) {
                localTok += static_cast<uint32_t>(ce - cs);
                if (localTok > pos) {
                    return localTok;
                }
            }
            cumPrev = cumE;
        }
        return splitTokens;
    }
    // 本 split 内 start 之前（含刚好在该点结束的）非空专家块数 —— 也就是该 token 之前
    // 已经插进去的补偿行数。
    __aicore__ inline uint32_t CountAuxBefore(uint32_t start, uint32_t splitTokens, int64_t splitStartAbs) const
    {
        const int64_t splitEnd = splitStartAbs + splitTokens;
        int64_t cumPrev = 0;
        uint32_t localTok = 0;
        uint32_t aux = 0;
        for (uint32_t e = 0; e < groupNum; e++) {
            const int64_t cumE = GroupCumEnd(e, cumPrev);
            const int64_t cs = cumPrev > splitStartAbs ? cumPrev : splitStartAbs;
            const int64_t ce = cumE < splitEnd ? cumE : splitEnd;
            if (ce > cs) {
                localTok += static_cast<uint32_t>(ce - cs);
                if (localTok <= start) {
                    aux++;
                } else {
                    break;
                }
            }
            cumPrev = cumE;
        }
        return aux;
    }
    // 把本核区间 [rangeStart, rangeEnd) 负责的补偿行一次性写出去。
    // 所有补偿行的内容完全相同（一行全 -8 的 int4，即每字节 0x88），所以只 Duplicate 一次，
    // 后面都是同源不同目的的 DMA，MTE3 上背靠背下发，不串行等。
    __aicore__ inline void WriteAssistRows(uint32_t rangeStart, uint32_t rangeEnd, uint32_t splitTokens,
                                           int64_t splitStartAbs, LocalTensor<int4b_t>& assistI4Tensor);
    TQue<QuePosition::VECIN, BUFFER_NUM_A8W4_PRE> vecInQueueX, vecInQueueXBak;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueA1;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueA2;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueA3;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueue0F;
    TQue<QuePosition::VECOUT, BUFFER_NUM_A8W4_PRE> vecOutQueueRowSum;
    TBuf<TPosition::VECCALC> tempBuff;
    const GMMSituQuantBaseParams* __restrict gmmSituQuantBaseParams;
    LocalTensor<int8_t> xTensor;
    LocalTensor<half> xHighHalfTensor;
    LocalTensor<float> xHighFloatTensor;
    LocalTensor<half> xLowHalfTensor;
    LocalTensor<half> xLowHalfTensor2;
    LocalTensor<int4b_t> xHighI4Tensor;
    LocalTensor<int4b_t> xLowI4Tensor;
    LocalTensor<int16_t> xLowI16Tensor;
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
    int64_t groupListType{0};
    // 补偿矩阵自算开关（tiling 的 assistRowsPerExpert）。为 1 时，本 split 内每个非空专家块
    // 的有效行之后再打一行全 -8 的 int4，cube 顺手算出 cAux = -8*sum_k(w)*scale，
    // post 减掉即可。这一行必须贴着专家自己的行放，才能和其它行落在同一个 M tile 里共用 B。
    int64_t assistRows{0};
};

__aicore__ inline void GMMA8W4PreProcess::Init(
    const GMAddrParams gmAddrParams, const GMMSituQuantBaseParams* __restrict gmmSituQuantBaseParamsIN)
{
    if ASCEND_IS_AIV {
        xGM.SetGlobalBuffer((__gm__ int8_t*)gmAddrParams.xGM);
        yGm1.SetGlobalBuffer((__gm__ int8_t*)gmAddrParams.workSpaceGM);
        yGm2.SetGlobalBuffer((__gm__ int8_t*)gmAddrParams.workSpaceGM + gmAddrParams.workSpaceOffset1);
        gmmSituQuantBaseParams = gmmSituQuantBaseParamsIN;
        vK = gmmSituQuantBaseParams->K;
        groupNum = static_cast<uint32_t>(gmmSituQuantBaseParams->groupNum);
        groupListType = gmmSituQuantBaseParams->groupListType;
        assistRows = gmmSituQuantBaseParams->assistRowsPerExpert;
        groupListGM.SetGlobalBuffer((__gm__ int64_t*)gmAddrParams.groupListGM);
        // M * K * 7B (1B + 0.5B + 0.5B + 2B + 4B) <= UBsize - 256B
        blockDim = GetBlockNum() * GetTaskRation();
    }
}

__aicore__ inline void GMMA8W4PreProcess::CustomInitBuffer(TPipe* pipe)
{
    pipe->InitBuffer(vecInQueueX, BUFFER_NUM_A8W4_PRE, vK * sizeof(int8_t));    // K * 1B
    pipe->InitBuffer(vecOutQueueA1, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // K * 0.5B
    pipe->InitBuffer(vecOutQueueA2, BUFFER_NUM_A8W4_PRE, vK * sizeof(int4b_t)); // K * 0.5B
    pipe->InitBuffer(vecOutQueueA3, BUFFER_NUM_A8W4_PRE, vK * SIZE_OF_HALF_2);  // K * 2B
    // xLowHalfTensor, xLowHalfTensor2 and xHighFloatTensor share the same buffer
    pipe->InitBuffer(tempBuff, vK * sizeof(float)); // K * 4B
    constexpr int BUFFER_SIZE_256B = 128 * sizeof(int16_t);
    pipe->InitBuffer(vecOutQueue0F, BUFFER_NUM_A8W4_PRE, BUFFER_SIZE_256B); // 256B
}

__aicore__ inline void GMMA8W4PreProcess::CalculateTaskInfoEachCore(uint32_t& curCoreTaskNum_,
                                                                    uint32_t& curCoreStartOffset_)
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

__aicore__ inline void GMMA8W4PreProcess::WriteAssistRows(uint32_t rangeStart, uint32_t rangeEnd,
                                                          uint32_t splitTokens, int64_t splitStartAbs,
                                                          LocalTensor<int4b_t>& assistI4Tensor)
{
    // 一行 K 个 int4 = K/2 字节，内容全是 -8：每字节两个 nibble 都是 0x8，按 int16 写就是 0x8888。
    Duplicate(assistI4Tensor.ReinterpretCast<int16_t>(), static_cast<int16_t>(-30584), vK / 4);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
    const int64_t splitEnd = splitStartAbs + splitTokens;
    int64_t cumPrev = 0;
    uint32_t localTok = 0;
    uint32_t aux = 0;
    int32_t written = 0;
    for (uint32_t e = 0; e < groupNum; e++) {
        const int64_t cumE = GroupCumEnd(e, cumPrev);
        const int64_t cs = cumPrev > splitStartAbs ? cumPrev : splitStartAbs;
        const int64_t ce = cumE < splitEnd ? cumE : splitEnd;
        if (ce > cs) {
            localTok += static_cast<uint32_t>(ce - cs);
            // 该专家块最后一个 token 落在本核区间内 → 它的补偿行由本核写。
            if (localTok > rangeStart && localTok <= rangeEnd) {
                const uint64_t dstRow = static_cast<uint64_t>(TWO * localTok + aux);
                DataCopy(yGm[dstRow * (vK / 2)], assistI4Tensor.ReinterpretCast<int8_t>(), vK / 2);
                written++;
            }
            aux++;
        }
        cumPrev = cumE;
    }
    if (written > 0) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }
}

__aicore__ inline void GMMA8W4PreProcess::Process(WorkSpaceSplitConfig& workspaceSplitConfig,
                                                  int64_t workspaceSplitLoopIdx, TPipe* pipe)
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
        const uint32_t xLowHalfOffset = vK * SIZE_OF_HALF_2;
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
        // 自算补偿矩阵时：本 split 内每个非空专家块末尾多一行，A 区的行号因此整体后移，
        // 每个 token 的目标行 = 2*token + (它前面已经过去的专家块数)。本核负责的补偿行
        // 先在主循环之前一次性写掉（地址互不相交，先后顺序无所谓）。
        const int64_t splitStartAbs = workspaceSplitLoopIdx * workspaceSplitConfig.notLastTaskSize;
        uint32_t auxBefore = 0;
        uint32_t nextBoundary = totalM;
        if (assistRows != 0) {
            auxBefore = CountAuxBefore(curCoreStartOffset, totalM, splitStartAbs);
            nextBoundary = NextChunkEnd(curCoreStartOffset, totalM, splitStartAbs);
            WriteAssistRows(curCoreStartOffset, curCoreStartOffset + curCoreTaskNum, totalM, splitStartAbs,
                            xHighI4Tensor);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); // 0
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0); // 1
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1); // 2

        for (uint32_t xloop = 0; xloop < curCoreTaskNum; xloop++) {
            const uint32_t tokenIdx = xloop + curCoreStartOffset;
            uint64_t relStartAddr = static_cast<uint64_t>(tokenIdx) * vK;
            uint64_t dstStartAddr = relStartAddr + static_cast<uint64_t>(auxBefore) * (vK / 2);
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
            DataCopy(yGm[dstStartAddr], xHighI4Tensor.ReinterpretCast<int8_t>(), vK / 2);
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
            DataCopy(yGm[dstStartAddr + vK / TWO], xLowI4Tensor.ReinterpretCast<int8_t>(), vK / TWO);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0); // 1
            // 低四位处理结束

            if (assistRows != 0 && tokenIdx + 1 == nextBoundary) {
                auxBefore++;
                nextBoundary = NextChunkEnd(tokenIdx + 1, totalM, splitStartAbs);
            }
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

} // namespace GroupedMatmulDequantSituQuant
#endif // GMM_SITU_QUANT_A8W4_MSD
#endif // OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_PRE_H
