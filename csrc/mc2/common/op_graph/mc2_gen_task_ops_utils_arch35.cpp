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
 * \file mc2_gen_task_ops_utils_arch35.cpp
 * \brief
 */

#include "mc2_gen_task_ops_utils_arch35.h"

namespace ops {

ge::Status Mc2Arch35GenTaskOpsUtils::CreateCCUFusionTask(const gert::ExeResGenerationContext *context,
                                                         std::vector<std::vector<uint8_t>> &tasks)
{
    // 填充groupinfo
    GroupInfo groupInfo;
    if (!GetGroupInfo(context, groupInfo)) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get group info.");
        return ge::GRAPH_FAILED;
    }

    // 添加ccugroup
    std::vector<std::string> ccuGroups;
    for (const std::string &groupAttrName : groupInfo.groupAttrNames) {
        ccuGroups.emplace_back(groupAttrName);
        OPS_LOG_D(context->GetNodeName(), "Set group attr [%s] to ccu task info success.", groupAttrName.c_str());
    }

    // 创建 ccu task
    ge::KernelLaunchInfo ccuTask = ge::KernelLaunchInfo::CreateCcuTask(context, ccuGroups);
    OPS_LOG_I(context->GetNodeName(), "Create ccu task successfully.");

    // 获取aicore task
    ge::KernelLaunchInfo prevAicoreTask = ge::KernelLaunchInfo::LoadFromData(context, tasks.back());

    // 设置aicore task的numBlocks
    int64_t numBlocks = -1;
    if (!context->GetIntAttrVal("tvm_blockdim", numBlocks) || numBlocks <= 0) {
        OPS_LOG_E(context->GetNodeName(), "Can't get valid numBlocks, get numBlocks %ld.", numBlocks);
        return ge::GRAPH_FAILED;
    }
    prevAicoreTask.SetBlockDim(numBlocks);
    OPS_LOG_I(context->GetNodeName(), "aicore task set numBlocks successfully, set numBlocks %ld.", numBlocks);

    // 获取aicore task的args format
    const char *prevArgsFormatStr = prevAicoreTask.GetArgsFormat();
    if (prevArgsFormatStr == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get Args Format from aicore task in previous aicore task.");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(context->GetNodeName(), "Before Args Format is : [%s] in previous aicore task", prevArgsFormatStr);
    std::vector<ge::ArgDescInfo> argDescInfos = ge::ArgsFormatSerializer::Deserialize(prevArgsFormatStr);
    // 填充hcom,当前约定Hcom插入于args format的最前面
    for (size_t i = 0; i < groupInfo.groupCnt; ++i) {
        argDescInfos.insert(argDescInfos.begin() + i,
                            ge::ArgDescInfo::CreateHiddenInput(ge::HiddenInputSubType::kHcom));
    }
    auto argsFormatStr = ge::ArgsFormatSerializer::Serialize(argDescInfos);

    // 创建 fusion task
    std::vector<ge::KernelLaunchInfo> fusionTasks = {ccuTask, prevAicoreTask};
    ge::KernelLaunchInfo fusionTask = ge::KernelLaunchInfo::CreateFusionTask(context, fusionTasks);
    // 设置fusion task的args format
    if (fusionTask.SetArgsFormat(argsFormatStr.GetString()) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to set Args Format for aicore task.");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(context->GetNodeName(), "fusion task Args Format: %s", argsFormatStr.GetString());

    // 填充streamid
    int64_t streamId = context->GetStreamId();
    if (streamId < 0) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get context stream id: %ld.", streamId);
        return ge::GRAPH_FAILED;
    }
    fusionTask.SetStreamId(streamId);

    // 序列化fusion task
    tasks.back() = fusionTask.Serialize();

    return ge::GRAPH_SUCCESS;
}

} // namespace ops
