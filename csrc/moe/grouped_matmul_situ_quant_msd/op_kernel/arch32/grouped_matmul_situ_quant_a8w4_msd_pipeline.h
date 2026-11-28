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
 * \file grouped_matmul_situ_quant_a8w4_msd_pipeline.h
 * \brief
 */
#ifndef OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_PIPELINE_H
#define OP_KERNEL_GROUPED_MATMUL_SITU_QUANT_A8W4_MSD_PIPELINE_H

#include <typeinfo>
#include "grouped_matmul_situ_quant_a8w4_msd_pre.h"
#include "grouped_matmul_situ_quant_a8w4_msd_mid.h"
#include "grouped_matmul_situ_quant_a8w4_msd_post.h"
#include "grouped_matmul_situ_quant_utils_kernel.h"

using namespace AscendC;
using namespace matmul;

#ifdef GMM_SITU_QUANT_A8W4_MSD

namespace GroupedMatmulDequantSituQuant {

template <class mmType>
class GMMSituQuantPipelineSchedule {
private:
    typename mmType::MT& mm;
    TPipe* pipe;
    const GMMSituQuantBaseParams* __restrict gmmSituQuantBaseParams;
    const GMMSituQuant* __restrict gmmSituQuant;
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
    __aicore__ inline int64_t GetSplitRowsLimit(int64_t liveRows);
    __aicore__ inline bool CanUseSingleRowDecode() const;
    __aicore__ inline void ProcessSingleRowDecode(
#ifdef GMM_SITU_QUANT_MLP
        bool emitPacked
#endif
    );
    __aicore__ inline void InitWorkSpaceSplitConfig(WorkSpaceSplitConfig& workspaceSplitConfig, int64_t knownRows);

    __aicore__ inline void UpdateWorkSpaceSplitConfig(WorkSpaceSplitConfig& workspaceSplitConfig,
                                                      int32_t workspaceSplitLoopIdx);

public:
    __aicore__ inline GMMSituQuantPipelineSchedule(
        typename mmType::MT& mm_, const GMMSituQuantBaseParams* __restrict gmmSituQuantBaseParamsIN,
        const GMMSituQuant* __restrict gmmSwigluIN, TPipe* tPipeIN)
        : mm(mm_),
          midProcess(mm),
          gmmSituQuantBaseParams(gmmSituQuantBaseParamsIN),
          gmmSituQuant(gmmSwigluIN),
          pipe(tPipeIN)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                GM_ADDR weightAssistanceMatrix, GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale,
                                GM_ADDR workspace, int64_t knownRows = -1);
    // live 行数：groupList 描述的实际计算行数（已按 x 的 M 截断，见 InitWorkSpaceSplitConfig）。
    // 入口拿它挑 Cube 分块（每专家平均行数少的时候换一套 baseM/baseN），所以要在 Init 之后、
    // Process 之前取。
    __aicore__ inline int64_t GetLiveRows() const
    {
        return workspaceSplitConfig.M;
    }
    __aicore__ inline bool FitsWideTile(int64_t defaultBaseM, int64_t wideBaseM);
    __aicore__ inline void Process(
#ifdef GMM_SITU_QUANT_MLP
        bool emitPacked = false
#endif
    );
    __aicore__ inline void ProcessDecode(
#ifdef GMM_SITU_QUANT_MLP
        bool emitPacked = false
#endif
    );
#ifdef GMM_SITU_QUANT_MLP
    __aicore__ inline void ProcessPrepackedDown();
    __aicore__ inline void ProcessPrepackedDownBands();
#endif
    __aicore__ inline void ProcessSingleToken(
#ifdef GMM_SITU_QUANT_MLP
        bool emitPacked = false
#endif
    );
    __aicore__ inline int32_t FindSmallSingleExpert() const;
    __aicore__ inline void ProcessSmallSingleExpert(uint32_t expert
#ifdef GMM_SITU_QUANT_MLP
                                                   , bool emitPacked
#endif
    );
};

template <class mmType>
__aicore__ inline bool GMMSituQuantPipelineSchedule<mmType>::FitsWideTile(int64_t defaultBaseM, int64_t wideBaseM)
{
    int64_t previous = 0;
    for (uint32_t i = 0; i < gmmSituQuant->groupListLen; ++i) {
        const int64_t entry = groupListGM.GetValue(i);
        const int64_t rows = gmmSituQuantBaseParams->groupListType == 0 ? entry - previous : entry;
        previous = entry;
        const int64_t msdRows = DOUBLE_ROW * rows + gmmSituQuantBaseParams->assistRowsPerExpert;
        // An extra M tile rereads that expert's weights.
        if (rows > 0 && Ceil(msdRows, wideBaseM) > Ceil(msdRows, defaultBaseM)) {
            return false;
        }
    }
    return true;
}

template <class mmType>
__aicore__ inline void GMMSituQuantPipelineSchedule<mmType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale,
                                                                    GM_ADDR xScale, GM_ADDR weightAssistanceMatrix,
                                                                    GM_ADDR groupList, GM_ADDR y, GM_ADDR yScale,
                                                                    GM_ADDR workspace, int64_t knownRows)
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
    const bool useTwoSlots = gmmSituQuantBaseParams->M > gmmSituQuantBaseParams->mLimit;
    gmAddrParams.workSpaceOffset1 = useTwoSlots ? gmmSituQuantBaseParams->workSpaceOffset1 / 2 : 0;
    gmAddrParams.workSpaceOffset2 = gmmSituQuantBaseParams->workSpaceOffset1;
    gmAddrParams.workSpaceOffset3 =
        gmAddrParams.workSpaceOffset2 + (useTwoSlots ? gmmSituQuantBaseParams->workSpaceOffset2 / 2 : 0);
    groupListGM.SetGlobalBuffer((__gm__ int64_t*)gmAddrParams.groupListGM);
    InitWorkSpaceSplitConfig(workspaceSplitConfig, knownRows);
}

template <class mmType>
__aicore__ inline bool GMMSituQuantPipelineSchedule<mmType>::CanUseSingleRowDecode() const
{
    constexpr int64_t MAX_DECODE_ROWS = 16;
    return workspaceSplitConfig.loopCount == 1 && workspaceSplitConfig.M <= MAX_DECODE_ROWS &&
           workspaceSplitConfig.M <= 2 * gmmSituQuantBaseParams->coreNum;
}

template <class mmType>
__aicore__ inline void GMMSituQuantPipelineSchedule<mmType>::ProcessSingleRowDecode(
#ifdef GMM_SITU_QUANT_MLP
    bool emitPacked
#endif
    )
{
    // The one workspace band starts at zero and covers all expert weights.
    workspaceSplitConfig.rightMatrixExpertEndIndex = gmmSituQuant->groupListLen - 1;
    workspaceSplitConfig.notLastTaskSize = workspaceSplitConfig.M;
    workspaceSplitConfig.isLastLoop = true;
    preProcess.PackDecodeRow(workspaceSplitConfig.M, pipe);
    if ASCEND_IS_AIV {
        pipe->Reset();
    }
    SyncAll<false>();
    // Compensation and token scale are independent of the Cube result.
    VecConfig row;
    float inputScale = 0;
    postProcess.PrepareDecodeRow(row, inputScale, workspaceSplitConfig, pipe);
    midProcess.Process(workspaceSplitConfig, 0);
    SyncAll<false>();
#ifdef GMM_SITU_QUANT_MLP
    postProcess.FinishDecodeRow(row, inputScale, workspaceSplitConfig, emitPacked);
#else
    postProcess.FinishDecodeRow(row, inputScale, workspaceSplitConfig);
#endif
    if ASCEND_IS_AIV {
#ifdef GMM_SITU_QUANT_MLP
        if (emitPacked) {
            pipe->Reset();
        } else
#endif
        {
            pipe->Destroy();
        }
    }
}

template <class mmType>
__aicore__ inline void GMMSituQuantPipelineSchedule<mmType>::ProcessSingleToken(
#ifdef GMM_SITU_QUANT_MLP
    bool emitPacked
#endif
    )
{
    constexpr uint16_t PACK_PAIR_DONE = 5;
    constexpr uint16_t ALL_PACK_DONE = 6;
    constexpr uint16_t ALL_MATMUL_DONE = 7;
    constexpr uint16_t MATMUL_PAIR_DONE = 8;
    constexpr uint16_t LOCAL_PACK_ONLY = 0;

    workspaceSplitConfig.rightMatrixExpertEndIndex = gmmSituQuant->groupListLen - 1;
    workspaceSplitConfig.notLastTaskSize = workspaceSplitConfig.M;
    workspaceSplitConfig.isLastLoop = true;
    preProcess.Init(gmAddrParams, gmmSituQuantBaseParams);
#ifdef GMM_SITU_QUANT_MLP
    midProcess.Init(gmAddrParams, gmmSituQuantBaseParams, emitPacked);
    postProcess.Init(gmAddrParams, gmmSituQuantBaseParams, gmmSituQuant, emitPacked);
#else
    midProcess.Init(gmAddrParams, gmmSituQuantBaseParams);
    postProcess.Init(gmAddrParams, gmmSituQuantBaseParams, gmmSituQuant);
#endif

    // A private row fits in the existing workspace when each Cube group has a slot.
    const bool useLocalPack = gmmSituQuantBaseParams->mLimit >= gmmSituQuantBaseParams->coreNum;
    if ASCEND_IS_AIV {
        if (useLocalPack) {
            if (GetSubBlockIdx() == 0) {
                preProcess.PackDecodeRow(1, pipe, GetBlockIdx() / GetSubBlockNum());
            }
        } else {
            preProcess.PackDecodeRow(1, pipe);
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(PACK_PAIR_DONE);
        pipe->Reset();

        VecConfig row;
        float inputScale = 0;
        postProcess.PrepareDecodeRow(row, inputScale, workspaceSplitConfig, pipe);
        CrossCoreWaitFlag(MATMUL_PAIR_DONE);
#ifdef GMM_SITU_QUANT_MLP
        postProcess.FinishDecodeRow(row, inputScale, workspaceSplitConfig, emitPacked);
#else
        postProcess.FinishDecodeRow(row, inputScale, workspaceSplitConfig);
#endif
#ifdef GMM_SITU_QUANT_MLP
        if (emitPacked) {
            pipe->Reset();
        } else
#endif
        {
            pipe->Destroy();
        }
    }

    if ASCEND_IS_AIC {
        uint32_t expert = 0;
        while (expert + 1 < gmmSituQuant->groupListLen && groupListGM.GetValue(expert) <= 0) {
            ++expert;
        }
        midProcess.ProcessSingleExpertToken(expert, PACK_PAIR_DONE,
                                      useLocalPack ? LOCAL_PACK_ONLY : ALL_PACK_DONE);
        CrossCoreSetFlag<0, PIPE_FIX>(ALL_MATMUL_DONE);
        CrossCoreWaitFlag(ALL_MATMUL_DONE);
        CrossCoreSetFlag<2, PIPE_MTE3>(MATMUL_PAIR_DONE);
    }
}

template <class mmType>
__aicore__ inline int32_t GMMSituQuantPipelineSchedule<mmType>::FindSmallSingleExpert() const
{
    constexpr int64_t MAX_SINGLE_EXPERT_ROWS = 32;
    constexpr int64_t VECTORS_PER_CUBE = 2;
    if (workspaceSplitConfig.M <= 1 || workspaceSplitConfig.M > MAX_SINGLE_EXPERT_ROWS ||
        workspaceSplitConfig.M > VECTORS_PER_CUBE * gmmSituQuantBaseParams->coreNum ||
        workspaceSplitConfig.loopCount != 1) {
        return -1;
    }
    // In either encoding the first positive entry is the size of the first active expert.
    for (uint32_t expert = 0; expert < gmmSituQuant->groupListLen; ++expert) {
        const int64_t entry = groupListGM.GetValue(expert);
        if (entry > 0) {
            return entry >= workspaceSplitConfig.M ? static_cast<int32_t>(expert) : -1;
        }
    }
    return -1;
}

template <class mmType>
__aicore__ inline void GMMSituQuantPipelineSchedule<mmType>::ProcessSmallSingleExpert(uint32_t expert
#ifdef GMM_SITU_QUANT_MLP
                                                                                    , bool emitPacked
#endif
)
{
    constexpr uint16_t PACK_PAIR_DONE = 5;
    constexpr uint16_t ALL_PACK_DONE = 6;
    constexpr uint16_t ALL_MATMUL_DONE = 7;
    constexpr uint16_t MATMUL_PAIR_DONE = 8;
    constexpr uint32_t VECTORS_PER_CUBE = 2;
    const uint32_t rows = workspaceSplitConfig.M;
    const uint32_t rowPairs = Ceil(rows, VECTORS_PER_CUBE);
    workspaceSplitConfig.rightMatrixExpertEndIndex = gmmSituQuant->groupListLen - 1;
    workspaceSplitConfig.notLastTaskSize = rows;
    workspaceSplitConfig.isLastLoop = true;
    preProcess.Init(gmAddrParams, gmmSituQuantBaseParams);
#ifdef GMM_SITU_QUANT_MLP
    midProcess.Init(gmAddrParams, gmmSituQuantBaseParams, emitPacked);
    postProcess.Init(gmAddrParams, gmmSituQuantBaseParams, gmmSituQuant, emitPacked);
#else
    midProcess.Init(gmAddrParams, gmmSituQuantBaseParams);
    postProcess.Init(gmAddrParams, gmmSituQuantBaseParams, gmmSituQuant);
#endif
    if ASCEND_IS_AIV {
        preProcess.PackDecodeRow(rows, pipe);
        CrossCoreSetFlag<2, PIPE_MTE3>(PACK_PAIR_DONE);
        if (GetBlockIdx() < rowPairs * VECTORS_PER_CUBE) {
            pipe->Reset();
            VecConfig row;
            float inputScale = 0;
            postProcess.PrepareDecodeRow(row, inputScale, workspaceSplitConfig, pipe);
            CrossCoreWaitFlag(MATMUL_PAIR_DONE);
#ifdef GMM_SITU_QUANT_MLP
            postProcess.FinishDecodeRow(row, inputScale, workspaceSplitConfig, emitPacked);
#else
            postProcess.FinishDecodeRow(row, inputScale, workspaceSplitConfig);
#endif
        }
#ifdef GMM_SITU_QUANT_MLP
        if (emitPacked) {
            pipe->Reset();
        } else
#endif
        {
            pipe->Destroy();
        }
    }
    if ASCEND_IS_AIC {
        midProcess.ProcessSingleExpertToken(expert, PACK_PAIR_DONE, ALL_PACK_DONE, rows);
        CrossCoreSetFlag<0, PIPE_FIX>(ALL_MATMUL_DONE);
        CrossCoreWaitFlag(ALL_MATMUL_DONE);
        if (GetBlockIdx() < rowPairs) {
            CrossCoreSetFlag<2, PIPE_MTE3>(MATMUL_PAIR_DONE);
        }
    }
}

template <class mmType>
__aicore__ inline void GMMSituQuantPipelineSchedule<mmType>::ProcessDecode(
#ifdef GMM_SITU_QUANT_MLP
    bool emitPacked
#endif
    )
{
    if (!CanUseSingleRowDecode()) {
#ifdef GMM_SITU_QUANT_MLP
        Process(emitPacked);
#else
        Process();
#endif
        return;
    }
    preProcess.Init(gmAddrParams, gmmSituQuantBaseParams);
#ifdef GMM_SITU_QUANT_MLP
    midProcess.Init(gmAddrParams, gmmSituQuantBaseParams, emitPacked);
    postProcess.Init(gmAddrParams, gmmSituQuantBaseParams, gmmSituQuant, emitPacked);
#else
    midProcess.Init(gmAddrParams, gmmSituQuantBaseParams);
    postProcess.Init(gmAddrParams, gmmSituQuantBaseParams, gmmSituQuant);
#endif
#ifdef GMM_SITU_QUANT_MLP
    ProcessSingleRowDecode(emitPacked);
#else
    ProcessSingleRowDecode();
#endif
}

template <class mmType>
__aicore__ inline void GMMSituQuantPipelineSchedule<mmType>::Process(
#ifdef GMM_SITU_QUANT_MLP
    bool emitPacked
#endif
    )
{
    // 1.对每次workspace切分做大循环。
    preProcess.Init(gmAddrParams, gmmSituQuantBaseParams);
#ifdef GMM_SITU_QUANT_MLP
    midProcess.Init(gmAddrParams, gmmSituQuantBaseParams, emitPacked);
    postProcess.Init(gmAddrParams, gmmSituQuantBaseParams, gmmSituQuant, emitPacked);
#else
    midProcess.Init(gmAddrParams, gmmSituQuantBaseParams);
    postProcess.Init(gmAddrParams, gmmSituQuantBaseParams, gmmSituQuant);
#endif

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
#ifdef GMM_SITU_QUANT_MLP
        postProcess.Process(tempWorkspaceSplitConfig, workspaceSplitLoopIdx - 1, pipe, emitPacked);
#else
        postProcess.Process(tempWorkspaceSplitConfig, workspaceSplitLoopIdx - 1, pipe);
#endif
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
#ifdef GMM_SITU_QUANT_MLP
    postProcess.Process(workspaceSplitConfig, workspaceSplitConfig.loopCount - 1, pipe, emitPacked);
#else
    postProcess.Process(workspaceSplitConfig, workspaceSplitConfig.loopCount - 1, pipe);
#endif
    if ASCEND_IS_AIV {
#ifdef GMM_SITU_QUANT_MLP
        if (emitPacked) {
            pipe->Reset();
        } else
#endif
        {
            pipe->Destroy();
        }
    }
}

#ifdef GMM_SITU_QUANT_MLP
template <class mmType>
__aicore__ inline void GMMSituQuantPipelineSchedule<mmType>::ProcessPrepackedDown()
{
    if (!CanUseSingleRowDecode()) {
        ProcessPrepackedDownBands();
        return;
    }
    workspaceSplitConfig.rightMatrixExpertEndIndex = gmmSituQuant->groupListLen - 1;
    workspaceSplitConfig.notLastTaskSize = workspaceSplitConfig.M;
    workspaceSplitConfig.isLastLoop = true;
    midProcess.Init(gmAddrParams, gmmSituQuantBaseParams, true);
    midProcess.SetPrepackedInput(gmAddrParams.xGM);
    postProcess.Init(gmAddrParams, gmmSituQuantBaseParams, gmmSituQuant, true);
    VecConfig row;
    float inputScale = 0;
    postProcess.PrepareDecodeRow(row, inputScale, workspaceSplitConfig, pipe);
    midProcess.Process(workspaceSplitConfig, 0);
    SyncAll<false>();
    postProcess.FinishPrepackedDownRow(row, inputScale, workspaceSplitConfig, gmAddrParams.yGM);
    if ASCEND_IS_AIV {
        pipe->Destroy();
    }
}

template <class mmType>
__aicore__ inline void GMMSituQuantPipelineSchedule<mmType>::ProcessPrepackedDownBands()
{
    midProcess.Init(gmAddrParams, gmmSituQuantBaseParams, true);
    postProcess.Init(gmAddrParams, gmmSituQuantBaseParams, gmmSituQuant, true);
    const auto loops = workspaceSplitConfig.loopCount;
    for (int64_t index = 0; index < loops; ++index) {
        UpdateWorkSpaceSplitConfig(workspaceSplitConfig, index);
        const uint64_t row = workspaceSplitConfig.leftMatrixStartIndex;
        midProcess.SetPrepackedInput(gmAddrParams.xGM + row * gmmSituQuantBaseParams->K, index % 2 != 0);
        midProcess.Process(workspaceSplitConfig, index);
        postProcess.Process(tempWorkspaceSplitConfig, index - 1, pipe, false, gmAddrParams.yGM);
        if ASCEND_IS_AIV {
            pipe->Reset();
        }
        tempWorkspaceSplitConfig = workspaceSplitConfig;
        SyncAll<false>();
    }
    postProcess.Process(workspaceSplitConfig, loops - 1, pipe, false, gmAddrParams.yGM);
    if ASCEND_IS_AIV {
        pipe->Destroy();
    }
}

#endif

template <class mmType>
__aicore__ inline int64_t GMMSituQuantPipelineSchedule<mmType>::GetSplitRowsLimit(int64_t liveRows)
{
    const int64_t slotRows = gmmSituQuantBaseParams->mLimit;
    if constexpr (mmType::BT::format == CubeFormat::NZ) {
        constexpr int64_t HIDDEN_SIZE = 7168;
        constexpr uint32_t LOCAL_EXPERTS = 14;
        constexpr int64_t MIN_PIPELINE_ROWS = 600;
        constexpr int64_t MAX_PIPELINE_ROWS = 937;
        if (gmmSituQuantBaseParams->K != HIDDEN_SIZE || gmmSituQuantBaseParams->N != HIDDEN_SIZE ||
            gmmSituQuant->groupListLen != LOCAL_EXPERTS || gmmSituQuantBaseParams->quantGroupNum != 1 ||
            gmmSituQuantBaseParams->M <= slotRows || liveRows < MIN_PIPELINE_ROWS || liveRows > MAX_PIPELINE_ROWS) {
            return slotRows;
        }

        // Start postprocessing while Cube computes the second band. A shorter final band reduces the exposed
        // Vector tail, and an expert boundary avoids reloading that expert's weights across the two bands.
        const int64_t target = Ceil(liveRows * 4, 5L);
        const int64_t minFirstRows = Ceil(liveRows * 2, 3L);
        const int64_t minLastRows = Ceil(liveRows, 8L);
        int64_t bestRows = slotRows;
        int64_t bestDistance = liveRows;
        int64_t cumulative = 0;
        const bool prefixGroups = gmmSituQuantBaseParams->groupListType == 0;
        for (uint32_t expert = 0; expert < gmmSituQuant->groupListLen; ++expert) {
            const int64_t entry = groupListGM.GetValue(expert);
            const int64_t end = prefixGroups ? entry : cumulative + entry;
            if (end > cumulative && end >= minFirstRows && end <= slotRows && liveRows - end >= minLastRows) {
                const int64_t distance = end > target ? end - target : target - end;
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestRows = end;
                }
            }
            cumulative = end;
        }
        if (cumulative == liveRows) {
            return bestRows;
        }
    }
    return slotRows;
}

template <class mmType>
__aicore__ inline void GMMSituQuantPipelineSchedule<mmType>::InitWorkSpaceSplitConfig(
    WorkSpaceSplitConfig& workspaceSplitConfig, int64_t knownRows)
{
    if (knownRows > 0) {
        workspaceSplitConfig.M = knownRows;
    } else if (gmmSituQuantBaseParams->groupListType == 0) {
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
    const int64_t splitRows = GetSplitRowsLimit(workspaceSplitConfig.M);
    workspaceSplitConfig.loopCount = Ceil(workspaceSplitConfig.M, splitRows);
    workspaceSplitConfig.notLastTaskSize = splitRows;
    workspaceSplitConfig.lastLoopTaskSize =
        workspaceSplitConfig.M - (workspaceSplitConfig.loopCount - 1) * splitRows;
    workspaceSplitConfig.leftMatrixStartIndex = 0;
    workspaceSplitConfig.rightMatrixExpertStartIndex = 0;
    workspaceSplitConfig.rightMatrixExpertNextStartIndex = 0;
    workspaceSplitConfig.isLastLoop = false;
}

template <class mmType>
__aicore__ inline void GMMSituQuantPipelineSchedule<mmType>::UpdateWorkSpaceSplitConfig(
    WorkSpaceSplitConfig& workspaceSplitConfig, int32_t workspaceSplitLoopIdx)
{
    if (workspaceSplitLoopIdx < 0)
        return;
    const int64_t splitRows = workspaceSplitConfig.notLastTaskSize;
    workspaceSplitConfig.leftMatrixStartIndex = workspaceSplitLoopIdx * splitRows;
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
        if (curTaskNum > splitRows) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex;
            break;
        } else if (curTaskNum == splitRows && nextTaskNum > splitRows) {
            workspaceSplitConfig.rightMatrixExpertNextStartIndex = workspaceSplitConfig.rightMatrixExpertEndIndex + 1;
            break;
        } else if (nextTaskNum > splitRows) {
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
