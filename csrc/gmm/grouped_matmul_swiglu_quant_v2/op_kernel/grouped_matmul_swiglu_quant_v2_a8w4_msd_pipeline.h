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
 * \file grouped_matmul_swiglu_quant_v2_a8w4_msd_pipeline.h
 * \brief
 */
#ifndef OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A8W4_MSD_PIPELINE_H
#define OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A8W4_MSD_PIPELINE_H

#include <typeinfo>
#include "grouped_matmul_swiglu_quant_v2_a8w4_msd_pre.h"
#include "grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h"
#include "grouped_matmul_swiglu_quant_v2_a8w4_msd_post.h"
#include "grouped_matmul_swiglu_quant_v2_utils.h"

using namespace AscendC;
using namespace matmul;

#ifdef GMM_SWIGLU_QUANT_V2_A8W4_MSD

namespace GroupedMatmulDequantSwigluQuant {

template <class mmType>
class GMMSwigluQuantPipelineSchedule {
private:
    typename mmType::MT &mm;
    TPipe *pipe;
    const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParams;
    const GMMSwigluQuantV2 *__restrict gmmSwigluQuantV2;
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
    __aicore__ inline void InitWorkSpaceSplitConfig(WorkSpaceSplitConfig &workspaceSplitConfig);

    __aicore__ inline void UpdateWorkSpaceSplitConfig(WorkSpaceSplitConfig &workspaceSplitConfig,
                                                      int32_t workspaceSplitLoopIdx);

public:
    __aicore__ inline GMMSwigluQuantPipelineSchedule(
        typename mmType::MT &mm_, const GMMSwigluQuantV2BaseParams *__restrict gmmSwigluQuantV2BaseParamsIN,
        const GMMSwigluQuantV2 *__restrict gmmSwigluIN, TPipe *tPipeIN)
        : mm(mm_), midProcess(mm), gmmSwigluQuantV2BaseParams(gmmSwigluQuantV2BaseParamsIN),
          gmmSwigluQuantV2(gmmSwigluIN), pipe(tPipeIN)
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                GM_ADDR weightAssistanceMatrix, GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale,
                                GM_ADDR workspace);
    __aicore__ inline void Process();
};

template <class mmType>
__aicore__ inline void GMMSwigluQuantPipelineSchedule<mmType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
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
    gmAddrParams.workSpaceOffset1 = gmmSwigluQuantV2BaseParams->workSpaceOffset1 / 2;
    gmAddrParams.workSpaceOffset2 = gmmSwigluQuantV2BaseParams->workSpaceOffset1;
    gmAddrParams.workSpaceOffset3 =
        gmmSwigluQuantV2BaseParams->workSpaceOffset1 + gmmSwigluQuantV2BaseParams->workSpaceOffset2 / 2;
    groupListGM.SetGlobalBuffer((__gm__ int64_t *)gmAddrParams.groupListGM);
    InitWorkSpaceSplitConfig(workspaceSplitConfig);
}

template <class mmType>
__aicore__ inline void GMMSwigluQuantPipelineSchedule<mmType>::Process()
{
    // 1.对每次workspace切分做大循环。
    preProcess.Init(gmAddrParams, gmmSwigluQuantV2BaseParams);
    midProcess.Init(gmAddrParams, gmmSwigluQuantV2BaseParams);
    postProcess.Init(gmAddrParams, gmmSwigluQuantV2BaseParams, gmmSwigluQuantV2);

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
            SyncAll<true>();
        }
        postProcess.Process(tempWorkspaceSplitConfig, workspaceSplitLoopIdx - 1, pipe);
        // 3.第n-1次后处理需要保留第n次的切分数据
        tempWorkspaceSplitConfig = workspaceSplitConfig;
        // reset
        if ASCEND_IS_AIV {
            pipe->Reset();
        }
        SyncAll<false>();
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
__aicore__ inline void
GMMSwigluQuantPipelineSchedule<mmType>::InitWorkSpaceSplitConfig(WorkSpaceSplitConfig &workspaceSplitConfig)
{
    if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
        workspaceSplitConfig.M = groupListGM.GetValue(gmmSwigluQuantV2->groupListLen - 1);
    } else {
        int64_t totalTmp = 0;
        for (uint32_t i = 0; i < gmmSwigluQuantV2->groupListLen; i++) {
            totalTmp += groupListGM.GetValue(i);
        }
        workspaceSplitConfig.M = totalTmp;
    }
    workspaceSplitConfig.loopCount = Ceil(workspaceSplitConfig.M, gmmSwigluQuantV2BaseParams->mLimit);
    workspaceSplitConfig.notLastTaskSize = gmmSwigluQuantV2BaseParams->mLimit;
    workspaceSplitConfig.lastLoopTaskSize =
        workspaceSplitConfig.M - (workspaceSplitConfig.loopCount - 1) * gmmSwigluQuantV2BaseParams->mLimit;
    workspaceSplitConfig.leftMatrixStartIndex = 0;
    workspaceSplitConfig.rightMatrixExpertStartIndex = 0;
    workspaceSplitConfig.rightMatrixExpertNextStartIndex = 0;
    workspaceSplitConfig.isLastLoop = false;
}

template <class mmType>
__aicore__ inline void
GMMSwigluQuantPipelineSchedule<mmType>::UpdateWorkSpaceSplitConfig(WorkSpaceSplitConfig &workspaceSplitConfig,
                                                                   int32_t workspaceSplitLoopIdx)
{
    if (workspaceSplitLoopIdx < 0)
        return;
    workspaceSplitConfig.leftMatrixStartIndex = workspaceSplitLoopIdx * gmmSwigluQuantV2BaseParams->mLimit;
    workspaceSplitConfig.rightMatrixExpertStartIndex = workspaceSplitConfig.rightMatrixExpertNextStartIndex;
    workspaceSplitConfig.rightMatrixExpertEndIndex = workspaceSplitConfig.rightMatrixExpertStartIndex;
    // 计算右专家矩阵的终止索引(rightMatrixExpertEndIndex) 和下一次的起始索引(rightMatrixExpertNextStartIndex)
    int32_t curTaskNum = 0;
    int32_t nextTaskNum = 0;
    int32_t curTaskNumTmp = 0;
    int32_t nextTaskNumTmp = 0;
    if (gmmSwigluQuantV2BaseParams->groupListType == 1) {
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
    while (workspaceSplitConfig.rightMatrixExpertEndIndex < gmmSwigluQuantV2->groupListLen) {
        if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
            curTaskNum = groupListGM.GetValue(workspaceSplitConfig.rightMatrixExpertEndIndex) -
                         workspaceSplitConfig.leftMatrixStartIndex;
        } else {
            curTaskNumTmp += groupListGM.GetValue(workspaceSplitConfig.rightMatrixExpertEndIndex);
            curTaskNum = curTaskNumTmp - workspaceSplitConfig.leftMatrixStartIndex;
        }
        int32_t nextTaskIdx = workspaceSplitConfig.rightMatrixExpertEndIndex >= gmmSwigluQuantV2->groupListLen - 1 ?
                                  gmmSwigluQuantV2->groupListLen - 1 :
                                  workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
        if (gmmSwigluQuantV2BaseParams->groupListType == 0) {
            nextTaskNum = groupListGM.GetValue(nextTaskIdx) - workspaceSplitConfig.leftMatrixStartIndex;
        } else {
            if (workspaceSplitConfig.rightMatrixExpertEndIndex < gmmSwigluQuantV2->groupListLen - 1) {
                nextTaskNumTmp += groupListGM.GetValue(nextTaskIdx);
            }
            nextTaskNum = nextTaskNumTmp - workspaceSplitConfig.leftMatrixStartIndex;
        }
        if (curTaskNum > gmmSwigluQuantV2BaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        } else if (curTaskNum == gmmSwigluQuantV2BaseParams->mLimit &&
                   nextTaskNum > gmmSwigluQuantV2BaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
            break;
        } else if (nextTaskNum > gmmSwigluQuantV2BaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertEndIndex++;
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        }
        workspaceSplitConfig.rightMatrixExpertEndIndex++;
    }
    workspaceSplitConfig.isLastLoop = workspaceSplitLoopIdx == workspaceSplitConfig.loopCount - 1 ? true : false;

    if (workspaceSplitConfig.isLastLoop) {
        workspaceSplitConfig.rightMatrixExpertEndIndex =
            workspaceSplitConfig.rightMatrixExpertEndIndex >= gmmSwigluQuantV2->groupListLen ?
                gmmSwigluQuantV2->groupListLen - 1 :
                workspaceSplitConfig.rightMatrixExpertEndIndex;
    }
}

} // namespace GroupedMatmulDequantSwigluQuant
#endif // GMM_SWIGLU_QUANT_V2_A8W4_MSD
#endif // OP_KERNEL_GROUPED_MATMUL_SWIGLU_QUANT_V2_A8W4_MSD_PIPELINE_H