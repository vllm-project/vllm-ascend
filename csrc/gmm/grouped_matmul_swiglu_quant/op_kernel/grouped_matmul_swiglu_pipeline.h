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
 * \file grouped_matmul_swiglu_quant.cpp
 * \brief
 */
#ifndef ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_PIPELINE_H
#define ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_PIPELINE_H
#include "grouped_matmul_swiglu_quant.h"
#include <typeinfo>
#include "grouped_matmul_swiglu_quant_a8w4_msd_pre.h"
#include "grouped_matmul_swiglu_quant_a8w4_msd_mid.h"
#include "grouped_matmul_swiglu_quant_a8w4_msd_post.h"
#include "grouped_matmul_swiglu_quant_utils.h"
using namespace AscendC;
using namespace matmul;
#ifdef GMM_SWIGLU_QUANT_A8W4_MSD

namespace GROUPED_MATMUL_SWIGLU_QUANT {

template <class mmType>
class GMMSwigluQuantPipelineSchedule {
private:
    typename mmType::MT &mm;
    TPipe *pipe;
    const GMMSwigluBaseParams *__restrict gmmBaseParams;
    const GMMSwiglu *__restrict gmmSwiglu;
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
    __aicore__ inline GMMSwigluQuantPipelineSchedule(typename mmType::MT &mm_,
                                                     const GMMSwigluBaseParams *__restrict gmmBaseParamsIN,
                                                     const GMMSwiglu *__restrict gmmSwigluIN, TPipe *tPipeIN)
        : mm(mm_), midProcess(mm), gmmBaseParams(gmmBaseParamsIN), gmmSwiglu(gmmSwigluIN), pipe(tPipeIN)
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
    gmAddrParams.workSpaceOffset1 = gmmBaseParams->workSpaceOffset1 / 2;
    gmAddrParams.workSpaceOffset2 = gmmBaseParams->workSpaceOffset1;
    gmAddrParams.workSpaceOffset3 = gmmBaseParams->workSpaceOffset1 + gmmBaseParams->workSpaceOffset2 / 2;
    groupListGM.SetGlobalBuffer((__gm__ int64_t *)gmAddrParams.groupListGM);
    InitWorkSpaceSplitConfig(workspaceSplitConfig);
}

template <class mmType>
__aicore__ inline void GMMSwigluQuantPipelineSchedule<mmType>::Process()
{
    // 1.对每次workspace切分做大循环。
    preProcess.Init(gmAddrParams, gmmBaseParams);
    midProcess.Init(gmAddrParams, gmmBaseParams);
    postProcess.Init(gmAddrParams, gmmBaseParams, gmmSwiglu);

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
    workspaceSplitConfig.M = groupListGM.GetValue(gmmSwiglu->groupListLen - 1);
    workspaceSplitConfig.loopCount = Ceil(workspaceSplitConfig.M, gmmBaseParams->mLimit);
    workspaceSplitConfig.notLastTaskSize = gmmBaseParams->mLimit;
    workspaceSplitConfig.lastLoopTaskSize =
        workspaceSplitConfig.M - (workspaceSplitConfig.loopCount - 1) * gmmBaseParams->mLimit;
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
    workspaceSplitConfig.leftMatrixStartIndex = workspaceSplitLoopIdx * gmmBaseParams->mLimit;
    workspaceSplitConfig.rightMatrixExpertStartIndex = workspaceSplitConfig.rightMatrixExpertNextStartIndex;
    workspaceSplitConfig.rightMatrixExpertEndIndex = workspaceSplitConfig.rightMatrixExpertStartIndex;
    // 计算右专家矩阵的终止索引(rightMatrixExpertEndIndex) 和下一次的起始索引(rightMatrixExpertNextStartIndex)
    int32_t curTaskNum = 0;
    int32_t nextTaskNum = 0;
    while (workspaceSplitConfig.rightMatrixExpertEndIndex < gmmSwiglu->groupListLen) {
        curTaskNum = groupListGM.GetValue(workspaceSplitConfig.rightMatrixExpertEndIndex) -
                     workspaceSplitConfig.leftMatrixStartIndex;
        int32_t nextTaskIdx = workspaceSplitConfig.rightMatrixExpertEndIndex >= gmmSwiglu->groupListLen - 1 ?
                                  gmmSwiglu->groupListLen - 1 :
                                  workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
        nextTaskNum = groupListGM.GetValue(nextTaskIdx) - workspaceSplitConfig.leftMatrixStartIndex;
        if (curTaskNum > gmmBaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        } else if (curTaskNum == gmmBaseParams->mLimit && nextTaskNum > gmmBaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
            break;
        } else if (nextTaskNum > gmmBaseParams->mLimit) {
            workspaceSplitConfig.rightMatrixExpertEndIndex++;
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        }
        workspaceSplitConfig.rightMatrixExpertEndIndex++;
    }
    workspaceSplitConfig.isLastLoop = workspaceSplitLoopIdx == workspaceSplitConfig.loopCount - 1 ? true : false;

    if (workspaceSplitConfig.isLastLoop) {
        workspaceSplitConfig.rightMatrixExpertEndIndex =
            workspaceSplitConfig.rightMatrixExpertEndIndex >= gmmSwiglu->groupListLen ?
                gmmSwiglu->groupListLen - 1 :
                workspaceSplitConfig.rightMatrixExpertEndIndex;
    }
}

} // namespace GROUPED_MATMUL_SWIGLU_QUANT
#endif // GMM_SWIGLU_QUANT_A8W4_MSD
#endif // ASCENDC_GROUPED_MATMUL_SWIGLU_QUANT_PIPELINE_H
