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
 * \file grouped_matmul_situ_quant_v2_a8w4_msd_pipeline.h
 * \brief
 */
#ifndef OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_PIPELINE_H
#define OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_PIPELINE_H

#include <typeinfo>
#include "grouped_matmul_situ_quant_v2_a8w4_msd_pre.h"
#include "grouped_matmul_situ_quant_v2_a8w4_msd_mid.h"
#include "grouped_matmul_situ_quant_v2_a8w4_msd_post.h"
#include "grouped_matmul_situ_quant_v2_utils_kernel.h"

using namespace AscendC;
using namespace matmul;

#ifdef GMM_SITU_QUANT_A8W4_MSD

namespace GroupedMatmulDequantSituQuant {

template <class mmType>
class GMMSituV2QuantPipelineSchedule {
private:
    typename mmType::MT& mm;
    TPipe* pipe;
    const GMMSituV2QuantBaseParams* __restrict gmmSituQuantBaseParams;
    const GMMSituV2Quant* __restrict gmmSituQuant;
    // WorkSpaceSplitConfig控制Workspace切割方式的结构体;
    WorkSpaceSplitConfig workspaceSplitConfig;
    WorkSpaceSplitConfig tempWorkspaceSplitConfig;
    // 记录GM_ADDR的结构体
    GMAddrParams gmAddrParams;
    // 前处理GMMA8W4PreProcess类
    GMMA8W4PreProcess preProcess;
    // 中间处理GMMA8W4MidProcess类
    GMMA8W4MidProcess<mmType> midProcess;
    // 后处理GMMA8W4PostProcess类
    GMMA8W4PostProcess postProcess;
    GlobalTensor<int64_t> groupListGM;
    __aicore__ inline void InitWorkSpaceSplitConfig(WorkSpaceSplitConfig& workspaceSplitConfig);

    __aicore__ inline void UpdateWorkSpaceSplitConfig(WorkSpaceSplitConfig& workspaceSplitConfig,
                                                      int32_t workspaceSplitLoopIdx);

public:
    __aicore__ inline GMMSituV2QuantPipelineSchedule(
        typename mmType::MT& mm_, const GMMSituV2QuantBaseParams* __restrict gmmSituQuantBaseParamsIN,
        const GMMSituV2Quant* __restrict gmmSwigluIN, TPipe* tPipeIN)
        : mm(mm_),
          midProcess(mm),
          gmmSituQuantBaseParams(gmmSituQuantBaseParamsIN),
          gmmSituQuant(gmmSwigluIN),
          pipe(tPipeIN)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                GM_ADDR weightAssistanceMatrix, GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale,
                                GM_ADDR workspace);
    // live 行数：groupList 描述的实际计算行数（已按 x 的 M 截断，见 InitWorkSpaceSplitConfig）。
    // 入口拿它挑 Cube 分块（每专家平均行数少的时候换一套 baseM/baseN），所以要在 Init 之后、
    // Process 之前取。
    __aicore__ inline int64_t GetLiveRows() const
    {
        return workspaceSplitConfig.M;
    }
    __aicore__ inline void Process();
};

template <class mmType>
__aicore__ inline void GMMSituV2QuantPipelineSchedule<mmType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                                                                    GM_ADDR xScale, GM_ADDR weightAssistanceMatrix,
                                                                    GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale,
                                                                    GM_ADDR workspace)
{
    gmAddrParams.xGM = x;
    gmAddrParams.weightGM = weight;
    gmAddrParams.weightScaleGM = weightScale;
    gmAddrParams.xScaleGM = xScale;
    gmAddrParams.weightAuxiliaryMatrixGM = weightAssistanceMatrix;
    gmAddrParams.groupListGM = groupList;
    gmAddrParams.yGM = y;
    gmAddrParams.yScaleGM = yScale;
    gmAddrParams.workSpaceGM = workspace;
    gmAddrParams.workSpaceOffset1 = gmmSituQuantBaseParams->workSpaceOffset1 / 2;
    gmAddrParams.workSpaceOffset2 = gmmSituQuantBaseParams->workSpaceOffset1;
    gmAddrParams.workSpaceOffset3 =
        gmmSituQuantBaseParams->workSpaceOffset1 + gmmSituQuantBaseParams->workSpaceOffset2 / 2;
    groupListGM.SetGlobalBuffer((__gm__ int64_t*)gmAddrParams.groupListGM);
    InitWorkSpaceSplitConfig(workspaceSplitConfig);
}

template <class mmType>
__aicore__ inline void GMMSituV2QuantPipelineSchedule<mmType>::Process()
{
    // 1.对每次workspace切分做大循环。
    preProcess.Init(gmAddrParams, gmmSituQuantBaseParams);
    midProcess.Init(gmAddrParams, gmmSituQuantBaseParams);
    postProcess.Init(gmAddrParams, gmmSituQuantBaseParams, gmmSituQuant);

    // 1.前处理提前下发一次
    preProcess.Process(workspaceSplitConfig, 0, pipe);
    for (int64_t workspaceSplitLoopIdx = 0; workspaceSplitLoopIdx < workspaceSplitConfig.loopCount;
         workspaceSplitLoopIdx++) {
        // 更新workspaceSplitConfig
        UpdateWorkSpaceSplitConfig(workspaceSplitConfig, workspaceSplitLoopIdx);
        if ASCEND_IS_AIV {
            pipe->Reset();
        }

        SyncAll<false>();
        // 2.第n次中处理 && 第n+1次前处理 && 第n-1次后处理 并行
        midProcess.Process(workspaceSplitConfig, workspaceSplitLoopIdx);

        preProcess.Process(workspaceSplitConfig, workspaceSplitLoopIdx + 1, pipe);
        if ASCEND_IS_AIV {
            pipe->Reset();
            // 这道 AIV 内部屏障只服务于「pre(n+1) 与 post(n-1) 同时压在一个核上」的
            // 场景：两者都在 loopCount 之外时（loopCount == 1）它们各自在 Process 里
            // 首行就 return 了，中间没有任何 UB 访存，屏障只剩同步开销。实测 E=2/M=256
            // 这一档去掉两道屏障（本道 + 下面那道）省 4.8 µs，占该档 Task Duration 的 6%，
            // 而这一档的固定开销正是客户机器上「每个 launch 恒定 Wait」的来源。
            if (workspaceSplitConfig.loopCount > 1) {
                SyncAll<true>();
            }
        }
        postProcess.Process(tempWorkspaceSplitConfig, workspaceSplitLoopIdx - 1, pipe);
        // 3.第n-1次后处理需要保留第n次的切分数据
        tempWorkspaceSplitConfig = workspaceSplitConfig;
        // reset
        if ASCEND_IS_AIV {
            pipe->Reset();
        }
        // 最后一轮的这道屏障与循环外那道 SyncAll 紧挨着（中间只有 AIV 的 pipe->Reset），
        // 语义完全重合，跳过它——再一次 launch 少一道屏障。
        if (workspaceSplitLoopIdx != workspaceSplitConfig.loopCount - 1) {
            SyncAll<false>();
        }
        // 3.前一次后处理 && 后一次MM 并行
    }
    // reset
    if ASCEND_IS_AIV {
        pipe->Reset();
    }
    SyncAll<false>();
    // // 4.最后一次后处理
    postProcess.Process(workspaceSplitConfig, workspaceSplitConfig.loopCount - 1, pipe);
    if ASCEND_IS_AIV {
        pipe->Destroy();
    }
}

template <class mmType>
__aicore__ inline void GMMSituV2QuantPipelineSchedule<mmType>::InitWorkSpaceSplitConfig(
    WorkSpaceSplitConfig& workspaceSplitConfig)
{
    if (gmmSituQuantBaseParams->groupListType == 0) {
        workspaceSplitConfig.M = groupListGM.GetValue(gmmSituQuant->groupListLen - 1);
    } else {
        int64_t totalTmp = 0;
        for (uint32_t i = 0; i < gmmSituQuant->groupListLen; i++) {
            totalTmp += groupListGM.GetValue(i);
        }
        workspaceSplitConfig.M = totalTmp;
    }
    // groupList 描述的行数才是实际参与计算的行数，而 x / xScale / y / yScale 都按 **x 的 M**
    // 分配（tiling 里 set_M(m_)，post.h 用它给 xScale/yScale 设 GlobalBuffer）。groupList
    // 末值一旦超过它，下面所有按 workspaceSplitConfig.M 推出的行偏移都会落到分配之外——
    // 读 x 与写 y 同时越界，而且不会有任何报错。tiling 读不到 device 上的 groupList，
    // 拦不住；入口也无法从 shape 上发现（两者本该一致）。这里兜底截断：界外的行在 x 里
    // 根本不存在，本来就不该被计算。
    if (unlikely(workspaceSplitConfig.M > gmmSituQuantBaseParams->M)) {
        workspaceSplitConfig.M = gmmSituQuantBaseParams->M;
    }
    workspaceSplitConfig.loopCount = Ceil(workspaceSplitConfig.M, gmmSituQuantBaseParams->mLimit);
    workspaceSplitConfig.notLastTaskSize = gmmSituQuantBaseParams->mLimit;
    workspaceSplitConfig.lastLoopTaskSize =
        workspaceSplitConfig.M - (workspaceSplitConfig.loopCount - 1) * gmmSituQuantBaseParams->mLimit;
    workspaceSplitConfig.leftMatrixStartIndex = 0;
    workspaceSplitConfig.rightMatrixExpertStartIndex = 0;
    workspaceSplitConfig.rightMatrixExpertNextStartIndex = 0;
    workspaceSplitConfig.isLastLoop = false;
}

template <class mmType>
__aicore__ inline void GMMSituV2QuantPipelineSchedule<mmType>::UpdateWorkSpaceSplitConfig(
    WorkSpaceSplitConfig& workspaceSplitConfig, int32_t workspaceSplitLoopIdx)
{
    if (workspaceSplitLoopIdx < 0)
        return;
    workspaceSplitConfig.leftMatrixStartIndex = workspaceSplitLoopIdx * gmmSituQuantBaseParams->mLimit;
    workspaceSplitConfig.rightMatrixExpertStartIndex = workspaceSplitConfig.rightMatrixExpertNextStartIndex;
    workspaceSplitConfig.rightMatrixExpertEndIndex = workspaceSplitConfig.rightMatrixExpertStartIndex;
    // 计算右专家矩阵的终止索引(rightMatrixExpertEndIndex) 和下一次的起始索引(rightMatrixExpertNextStartIndex)
    int32_t curTaskNum = 0;
    int32_t nextTaskNum = 0;
    int32_t curTaskNumTmp = 0;
    int32_t nextTaskNumTmp = 0;
    if (gmmSituQuantBaseParams->groupListType == 1) {
        for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertEndIndex; i++) {
            curTaskNumTmp += groupListGM.GetValue(i);
        }
        if (workspaceSplitConfig.rightMatrixExpertEndIndex == 0) {
            nextTaskNumTmp = groupListGM.GetValue(0);
        } else {
            for (uint32_t i = 0; i < workspaceSplitConfig.rightMatrixExpertEndIndex; i++) {
                nextTaskNumTmp += groupListGM.GetValue(i);
            }
        }
    }
    while (workspaceSplitConfig.rightMatrixExpertEndIndex < gmmSituQuant->groupListLen) {
        if (gmmSituQuantBaseParams->groupListType == 0) {
            curTaskNum = groupListGM.GetValue(workspaceSplitConfig.rightMatrixExpertEndIndex) -
                         workspaceSplitConfig.leftMatrixStartIndex;
        } else {
            curTaskNumTmp += groupListGM.GetValue(workspaceSplitConfig.rightMatrixExpertEndIndex);
            curTaskNum = curTaskNumTmp - workspaceSplitConfig.leftMatrixStartIndex;
        }
        int32_t nextTaskIdx = workspaceSplitConfig.rightMatrixExpertEndIndex >= gmmSituQuant->groupListLen - 1 ?
                                  gmmSituQuant->groupListLen - 1 :
                                  workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
        if (gmmSituQuantBaseParams->groupListType == 0) {
            nextTaskNum = groupListGM.GetValue(nextTaskIdx) - workspaceSplitConfig.leftMatrixStartIndex;
        } else {
            if (workspaceSplitConfig.rightMatrixExpertEndIndex < gmmSituQuant->groupListLen - 1) {
                nextTaskNumTmp += groupListGM.GetValue(nextTaskIdx);
            }
            nextTaskNum = nextTaskNumTmp - workspaceSplitConfig.leftMatrixStartIndex;
        }
        if (curTaskNum > gmmSituQuantBaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        } else if (curTaskNum == gmmSituQuantBaseParams->mLimit &&
                   nextTaskNum > gmmSituQuantBaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
            break;
        } else if (nextTaskNum > gmmSituQuantBaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertEndIndex++;
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        }
        workspaceSplitConfig.rightMatrixExpertEndIndex++;
    }
    workspaceSplitConfig.isLastLoop = workspaceSplitLoopIdx == workspaceSplitConfig.loopCount - 1 ? true : false;

    if (workspaceSplitConfig.isLastLoop) {
        workspaceSplitConfig.rightMatrixExpertEndIndex =
            workspaceSplitConfig.rightMatrixExpertEndIndex >= gmmSituQuant->groupListLen ?
                gmmSituQuant->groupListLen - 1 :
                workspaceSplitConfig.rightMatrixExpertEndIndex;
    }
}

} // namespace GroupedMatmulDequantSituQuant
#endif // GMM_SITU_QUANT_A8W4_MSD
#endif // OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_PIPELINE_H
