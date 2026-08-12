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
 * \file mc2_moe_gen_task_ops_utils.cpp
 * \brief
 */

#include "mc2_moe_gen_task_ops_utils.h"

#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "mc2_gen_task_ops_utils.h"
#include "mc2_common_log.h"
#include "mc2_platform_info.h"
#include "op_mc2.h"
#include "platform/platform_info.h"

namespace {
const std::string ALL_TO_ALL_ALL_GATHER_BMM_OP_TYPE = "AlltoAllAllGatherBatchMatMul";
const std::string BMM_REDUCE_SCATTER_ALL_TO_ALL_OP_TYPE = "BatchMatMulReduceScatterAlltoAll";
const std::string MOE_DISTRIBUTE_DISPATCH_OP_TYPE = "MoeDistributeDispatch";
const std::string MOE_DISTRIBUTE_COMBINE_OP_TYPE = "MoeDistributeCombine";
const std::string DISTRIBUTE_BARRIER_OP_TYPE = "DistributeBarrier";
const std::string MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE = "MoeDistributeDispatchV2";
const std::string MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE = "MoeDistributeCombineV2";
const std::string MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_OP_TYPE = "MoeDistributeCombineAddRmsNorm";
const std::string MOE_DISTRIBUTE_COMBINE_SETUP_OP_TYPE = "MoeDistributeCombineSetup";
const std::string MOE_DISTRIBUTE_COMBINE_TEARDOWN_OP_TYPE = "MoeDistributeCombineTeardown";
const std::string MOE_DISTRIBUTE_DISPATCH_SETUP_OP_TYPE = "MoeDistributeDispatchSetup";
const std::string MOE_DISTRIBUTE_DISPATCH_TEARDOWN_OP_TYPE = "MoeDistributeDispatchTeardown";
const std::string ALLTO_ALLV_GROUPED_MAT_MUL_OP_TYPE = "AlltoAllvGroupedMatMul";
const std::string GROUPED_MAT_MUL_ALLTO_ALLV_OP_TYPE = "GroupedMatMulAlltoAllv";
const std::string ALL_TO_ALLV_QUANT_GROUPED_MM_OP_TYPE = "AlltoAllvQuantGroupedMatMul";
const std::string QUANT_GROUPED_MM_ALL_TO_ALLV_OP_TYPE = "QuantGroupedMatMulAlltoAllv";
const std::string ATTENTION_TO_FFN_OP_TYPE = "AttentionToFFN";
const std::string FFN_TO_ATTENTION_OP_TYPE = "FFNToAttention";
const std::string QUANT_ALL_REDUCE_OP_TYPE = "QuantAllReduce";
const std::string QUANT_REDUCE_SCATTER_OP_TYPE = "QuantReduceScatter";
const int32_t GROUP_CNT_OF_DS_TRAINING = 1;
const int32_t GROUP_CNT_OF_MOE_DISTRIBUTE = 2;
const int32_t ONE_GROUP_CNT_OF_MOE_DISTRIBUTE = 1;
const int32_t GROUP_CNT_OF_DISTRIBUTE_BARRIER = 1;
const int32_t GROUP_CNT_OF_ATTENTION_FFN = 1;
const int32_t GROUP_CNT_OF_QUANT_ALL_REDUCE = 1;
const int32_t GROUP_CNT_OF_QUANT_REDUCE_SCATTER = 1;
const int32_t GROUP_CNT = 2;
const int32_t MAX_GROUP_CNT = 16;
const std::string SO_NAME_LEGACY = "libccl_kernel.so";
const std::string SO_NAME_MC2_SERVER = "libmc2_server.so";
const std::string KERNEL_NAME_V1 = "RunAicpuKfcSrvLaunch";
const std::string KERNEL_NAME_MC2_SERVER = "Mc2ServerKernel";

// key: op type
// value: group cnt
static const std::map<const std::string, int32_t> GROUP_CNT_MAP{
    {DISTRIBUTE_BARRIER_OP_TYPE, GROUP_CNT_OF_DISTRIBUTE_BARRIER},
    {ALL_TO_ALL_ALL_GATHER_BMM_OP_TYPE, GROUP_CNT},
    {BMM_REDUCE_SCATTER_ALL_TO_ALL_OP_TYPE, GROUP_CNT},
    {MOE_DISTRIBUTE_DISPATCH_OP_TYPE, GROUP_CNT_OF_MOE_DISTRIBUTE},
    {MOE_DISTRIBUTE_COMBINE_OP_TYPE, GROUP_CNT_OF_MOE_DISTRIBUTE},
    {MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE, GROUP_CNT_OF_MOE_DISTRIBUTE},
    {MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE, GROUP_CNT_OF_MOE_DISTRIBUTE},
    {MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_OP_TYPE, GROUP_CNT_OF_MOE_DISTRIBUTE},
    {MOE_DISTRIBUTE_COMBINE_SETUP_OP_TYPE, ONE_GROUP_CNT_OF_MOE_DISTRIBUTE},
    {MOE_DISTRIBUTE_COMBINE_TEARDOWN_OP_TYPE, ONE_GROUP_CNT_OF_MOE_DISTRIBUTE},
    {MOE_DISTRIBUTE_DISPATCH_SETUP_OP_TYPE, ONE_GROUP_CNT_OF_MOE_DISTRIBUTE},
    {MOE_DISTRIBUTE_DISPATCH_TEARDOWN_OP_TYPE, ONE_GROUP_CNT_OF_MOE_DISTRIBUTE},
    {ALLTO_ALLV_GROUPED_MAT_MUL_OP_TYPE, GROUP_CNT_OF_DS_TRAINING},
    {GROUPED_MAT_MUL_ALLTO_ALLV_OP_TYPE, GROUP_CNT_OF_DS_TRAINING},
    {ALL_TO_ALLV_QUANT_GROUPED_MM_OP_TYPE, GROUP_CNT_OF_DS_TRAINING},
    {QUANT_GROUPED_MM_ALL_TO_ALLV_OP_TYPE, GROUP_CNT_OF_DS_TRAINING},
    {ATTENTION_TO_FFN_OP_TYPE, GROUP_CNT_OF_ATTENTION_FFN},
    {FFN_TO_ATTENTION_OP_TYPE, GROUP_CNT_OF_ATTENTION_FFN},
    {QUANT_ALL_REDUCE_OP_TYPE, GROUP_CNT_OF_QUANT_ALL_REDUCE},
    {QUANT_REDUCE_SCATTER_OP_TYPE, GROUP_CNT_OF_QUANT_REDUCE_SCATTER}};

static const std::unordered_set<std::string> NO_AI_CPU_SET{MOE_DISTRIBUTE_DISPATCH_OP_TYPE,
                                                           MOE_DISTRIBUTE_COMBINE_OP_TYPE,
                                                           MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE,
                                                           MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE,
                                                           MOE_DISTRIBUTE_DISPATCH_TEARDOWN_OP_TYPE,
                                                           MOE_DISTRIBUTE_COMBINE_TEARDOWN_OP_TYPE,
                                                           MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_OP_TYPE,
                                                           DISTRIBUTE_BARRIER_OP_TYPE,
                                                           ATTENTION_TO_FFN_OP_TYPE,
                                                           FFN_TO_ATTENTION_OP_TYPE};

static const std::unordered_set<std::string> NEED_SET_BLOCK_SET{
    MOE_DISTRIBUTE_DISPATCH_OP_TYPE, MOE_DISTRIBUTE_COMBINE_OP_TYPE, MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE,
    MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE};

// 对已有结构的重复定义，只在本文件插入 aicpu desc 的时候使用
struct HcclCommParamDescTmp {
    uint64_t version   : 4;
    uint64_t groupNum  : 4;
    uint64_t hasFfts   : 1;
    uint64_t tilingOff : 7;
    uint64_t isDyn     : 48;
};

static bool IsPlatform910B(const char *nodeName)
{
    fe::PlatFormInfos platform_info;
    fe::OptionalInfos optional_info;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Cannot get platform info!");
        return false;
    }
    static std::set<std::string> supported_soc = {"Ascend910B"};
    std::string short_soc_version;
    if (!platform_info.GetPlatformRes("version", "Short_SoC_version", short_soc_version) || short_soc_version.empty()) {
        OPS_LOG_E(nodeName, "Cannot get short soc version!");
        return false;
    }
    OPS_LOG_D(nodeName, "Get soc version: %s", short_soc_version.c_str());
    return supported_soc.count(short_soc_version) > 0;
}

static bool GetGroupCnt(const char *nodeName, const char *opType, int32_t &cnt)
{
    const std::string opTypeStr = opType;
    if (GROUP_CNT_MAP.find(opTypeStr) == GROUP_CNT_MAP.end()) {
        OPS_LOG_E(nodeName, "Op type [%s] has not registe in group cnt map.", opType);
        return false;
    }
    cnt = GROUP_CNT_MAP.at(opTypeStr);
    if ((cnt <= 0) || (cnt > MAX_GROUP_CNT)) {
        OPS_LOG_E(nodeName, "Group cnt [%d] is invalid, it should in [1, %d].", cnt, MAX_GROUP_CNT);
        return false;
    }

    return true;
}
} // namespace

namespace ops {
ge::Status Mc2MoeGenTaskOpsUtils::McMoeInsertHiddenInputForAicore(const gert::ExeResGenerationContext *context,
                                                                  const int32_t groupCnt,
                                                                  std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();

    // 找到插入位置
    const auto getIdxFunc = [](const std::vector<ge::ArgDescInfo> &argDescInfo) {
        size_t insertIdx = 0U; // 从 0: ffts 开始查找
        for (; insertIdx < argDescInfo.size(); ++insertIdx) {
            if (argDescInfo[insertIdx].GetType() == ge::ArgDescType::kIrInput ||
                argDescInfo[insertIdx].GetType() == ge::ArgDescType::kInputInstance) {
                break;
            }
        }
        return insertIdx;
    };

    ge::KernelLaunchInfo aicoreTask = ge::KernelLaunchInfo::LoadFromData(context, tasks.back()); // 取 aicore task
    if (Mc2GenTaskOpsUtils::InsertHiddenInputsForAicoreTask(context, aicoreTask, getIdxFunc, groupCnt) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Failed to insert hidden input for mix task.");
        return ge::GRAPH_FAILED;
    }
    tasks.back() = aicoreTask.Serialize();

    OPS_LOG_D(nodeName, "Modify AICore task for mc2 node successfully.");
    return ge::GRAPH_SUCCESS;
}

// McMoeInsertHiddenInputForAicoreV2 在新接口中与V1无差别，整合为一

ge::Status Mc2MoeGenTaskOpsUtils::CreateAicpuTaskMc2Moe(const gert::ExeResGenerationContext *context,
                                                        ge::KernelLaunchInfo &aicpuTask, const int32_t groupCnt)
{
    // desc 配置
    union {
        HcclCommParamDescTmp hcclCommParamDesc;
        uint64_t customValue;
    } desc;
    desc.hcclCommParamDesc.version = 1; // 保留参数，暂时保持为 1
    desc.hcclCommParamDesc.groupNum = groupCnt;
    desc.hcclCommParamDesc.hasFfts = 0; // aicpu 暂时用不到此参数
    desc.hcclCommParamDesc.tilingOff = desc.hcclCommParamDesc.hasFfts + desc.hcclCommParamDesc.groupNum + 1;
    desc.hcclCommParamDesc.isDyn = 0; // 现在默认为 0

    // aicpu 参数顺序: {desc}{ffts}{hcom}{tiling}
    std::vector<ge::ArgDescInfo> argsDescInfos;
    argsDescInfos.emplace_back(ge::ArgDescInfo::CreateCustomValue(desc.customValue));

    for (size_t i = 0; i < groupCnt; ++i) {
        argsDescInfos.emplace_back(ge::ArgDescInfo::CreateHiddenInput(ge::HiddenInputSubType::kHcom));
    }
    argsDescInfos.emplace_back(ge::ArgDescInfo(ge::ArgDescType::kTiling));

    auto argsFormatStr = ge::ArgsFormatSerializer::Serialize(argsDescInfos);
    aicpuTask.SetArgsFormat(argsFormatStr.GetString());
    OPS_LOG_I(context->GetNodeName(), "aicpu ArgsFormat: %s", argsFormatStr.GetString());

    return ge::GRAPH_SUCCESS;
}

// 单、双通信域均需要保证填入task顺序 [wait task, aicpu task, record task,
// aicore task]
ge::Status Mc2MoeGenTaskOpsUtils::Mc2MoeInsertTask(const gert::ExeResGenerationContext *context,
                                                   std::vector<std::vector<uint8_t>> &tasks, int32_t groupCnt)
{
    int64_t aicoreIdx = 0; // 直接置0
    const char *nodeName = context->GetNodeName();

    const auto streamInfos = context->GetAttachedStreamInfos();
    if (streamInfos.empty()) {
        OPS_LOG_E(nodeName, "No stream info found.");
        return ge::GRAPH_FAILED;
    }
    const int64_t attachStreamId = streamInfos[0].stream_id;
    const int64_t streamId = context->GetStreamId();

    // 1. wait task
    const char *opType = context->GetNodeType();
    const std::string opTypeStr = opType;
    const char *groupName =
        opTypeStr == DISTRIBUTE_BARRIER_OP_TYPE || opTypeStr == ALLTO_ALLV_GROUPED_MAT_MUL_OP_TYPE ||
                opTypeStr == GROUPED_MAT_MUL_ALLTO_ALLV_OP_TYPE || opTypeStr == ALL_TO_ALLV_QUANT_GROUPED_MM_OP_TYPE ||
                opTypeStr == QUANT_GROUPED_MM_ALL_TO_ALLV_OP_TYPE || opTypeStr == ATTENTION_TO_FFN_OP_TYPE ||
                opTypeStr == FFN_TO_ATTENTION_OP_TYPE ?
            "group" :
            "group_ep";
    ge::KernelLaunchInfo waitTask = ge::KernelLaunchInfo::CreateHcomWaitTask(context, groupName);
    waitTask.SetStreamId(attachStreamId);
    tasks.insert(tasks.begin() + aicoreIdx, waitTask.Serialize());
    ++aicoreIdx;

    // 2. aicpu task
    bool needAicpuTesk = IsPlatform910B(nodeName) || NO_AI_CPU_SET.find(opTypeStr) == NO_AI_CPU_SET.end();
    if (needAicpuTesk) {
        const bool useA5AicpuServer = IsTargetPlatformNpuArch(nodeName, NPUARCH_A5);
        const std::string &soName = useA5AicpuServer ? SO_NAME_MC2_SERVER : SO_NAME_LEGACY;
        const std::string &kernelName = useA5AicpuServer ? KERNEL_NAME_MC2_SERVER : KERNEL_NAME_V1;
        OPS_LOG_I(nodeName, "Create AICPU KFC task for MC2 MOE, so[%s], kernel[%s], useA5AicpuServer[%d].",
                  soName.c_str(), kernelName.c_str(), useA5AicpuServer);
        ge::KernelLaunchInfo aicpuTask =
            ge::KernelLaunchInfo::CreateAicpuKfcTask(context, soName.c_str(), kernelName.c_str());
        aicpuTask.SetStreamId(attachStreamId);
        if (NEED_SET_BLOCK_SET.find(opTypeStr) != NEED_SET_BLOCK_SET.end()) {
            int64_t numBlocks = 6;
            if (!context->GetIntAttrVal("_aicpu_blockdim", numBlocks) || numBlocks <= 0) {
                OPS_LOG_I(nodeName, "Can't get valid aicpu numBlocks, set numBlocks 6.");
                numBlocks = 6;
            }
            aicpuTask.SetBlockDim(numBlocks);
            OPS_LOG_I(nodeName, "Set aicpu numBlocks is %ld.", numBlocks);
        }
        if (CreateAicpuTaskMc2Moe(context, aicpuTask, groupCnt) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        tasks.insert(tasks.begin() + aicoreIdx, aicpuTask.Serialize());
        ++aicoreIdx;
    }

    // 3. record task
    ge::KernelLaunchInfo recordTask = ge::KernelLaunchInfo::CreateHcomRecordTask(context, groupName);
    recordTask.SetStreamId(streamId);
    tasks.insert(tasks.begin() + aicoreIdx, recordTask.Serialize());
    // aicoreIndex += 2L;  // 跳过aicore task

    OPS_LOG_I(nodeName, "Generate task for mc2 node successfully.");
    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(const gert::ExeResGenerationContext *context,
                                                        std::vector<std::vector<uint8_t>> &tasks)
{
    // 先插了aicore task, 再插wait + aicpu + record
    const char *nodeName = context->GetNodeName();

    int32_t groupCnt = -1;
    const char *opType = context->GetNodeType();
    if (opType == nullptr) {
        OPS_LOG_E(nodeName, "Op type is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (!GetGroupCnt(nodeName, opType, groupCnt)) {
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_D(nodeName, "Op [%s] get group [%d] success.", opType, groupCnt);

    if (McMoeInsertHiddenInputForAicore(context, groupCnt, tasks) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Insert hidden input for [%s] failed.", opType);
        return ge::GRAPH_FAILED;
    }

    if (Mc2GenTaskOpsUtils::IsComputationOnly()) {
        OPS_LOG_D(nodeName, "Debug mode is 1 (i.e. computation only), generating task for "
                            "mc2 node ends.");
        return ge::GRAPH_SUCCESS;
    }

    const std::string opTypeStr = opType; // 和V2的差异点，待确认，应该可以整合为一，用V2的if逻辑即可
    if (opTypeStr == MOE_DISTRIBUTE_COMBINE_TEARDOWN_OP_TYPE || opTypeStr == MOE_DISTRIBUTE_DISPATCH_TEARDOWN_OP_TYPE) {
        return ge::GRAPH_SUCCESS;
    }
    return Mc2MoeInsertTask(context, tasks, groupCnt);
}

// 支持静态图在线编译.o
ge::Status Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(const gert::ExeResGenerationContext *context,
                                                          std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    if (tasks.size() <= 0) {
        OPS_LOG_E(nodeName, "Failed to get task for mc2 node.");
        return ge::GRAPH_FAILED;
    }

    int32_t groupCnt = -1;
    const char *opType = context->GetNodeType();
    if (opType == nullptr) {
        OPS_LOG_E(nodeName, "Op type is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (!GetGroupCnt(nodeName, opType, groupCnt)) {
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(nodeName, "Op [%s] get group [%d] success.", opType, groupCnt);

    if (McMoeInsertHiddenInputForAicore(context, groupCnt, tasks) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Insert hidden input for [%s] failed.", opType);
        return ge::GRAPH_FAILED;
    }

    if (Mc2GenTaskOpsUtils::IsComputationOnly()) {
        OPS_LOG_D(nodeName, "Debug mode is 1 (i.e. computation only), generating task for "
                            "mc2 node ends.");
        return ge::GRAPH_SUCCESS;
    }

    const std::string opTypeStr = opType;
    bool useAiCpu =
        ((opTypeStr != MOE_DISTRIBUTE_DISPATCH_V2_OP_TYPE) && (opTypeStr != MOE_DISTRIBUTE_COMBINE_V2_OP_TYPE) &&
         (opTypeStr != DISTRIBUTE_BARRIER_OP_TYPE) && (opTypeStr != MOE_DISTRIBUTE_DISPATCH_OP_TYPE) &&
         (opTypeStr != MOE_DISTRIBUTE_COMBINE_OP_TYPE) && (opTypeStr != ATTENTION_TO_FFN_OP_TYPE) &&
         (opTypeStr != FFN_TO_ATTENTION_OP_TYPE) && (opTypeStr != QUANT_ALL_REDUCE_OP_TYPE) &&
         (opTypeStr != QUANT_REDUCE_SCATTER_OP_TYPE) && (opTypeStr != MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM_OP_TYPE));
    return useAiCpu ? Mc2MoeInsertTask(context, tasks, groupCnt) : ge::GRAPH_SUCCESS;
}
} // namespace ops
