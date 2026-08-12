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
 * \file matmul_all_reduce_gen_task_utils.cpp
 * \brief
 */
#include "matmul_all_reduce_gen_task_ops_utils.h"

#include "mc2_common_log.h"
#include "platform/platform_info.h"
#include "mc2_gen_task_ops_utils.h"
#include <algorithm>

namespace ops {
const int64_t AICORE_IDX = 0;
const std::string SO_NAME = "libccl_kernel.so";
const std::string KERNEL_NAME_V1 = "RunAicpuKfcSrvLaunch";

static bool IsPlatform310P()
{
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    MC2_CHECK_SUCCESS_RET(
        "MatmulAllReduceGenTaskOpsUtils",
        fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info));
    OPS_ERR_IF(fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
                   ge::GRAPH_SUCCESS,
               OPS_LOG_E("", "Cannot get platform info!"), return false);
    OPS_LOG_D("", "Get soc version: %s", optional_info.soc_version.c_str());
    static const std::vector<std::string> version_with_common_def = {"Ascend310P", "Ascend310P1", "Ascend310P3"};
    return find(version_with_common_def.begin(), version_with_common_def.end(), optional_info.soc_version) !=
           version_with_common_def.end();
}

ge::Status MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(const gert::ExeResGenerationContext *context,
                                                                          std::vector<std::vector<uint8_t>> &tasks)
{
    if (IsPlatform310P()) {
        int64_t aicore_idx = AICORE_IDX;
        // get main/attach stream ids
        const int64_t attach_stream_id = Mc2GenTaskOpsUtils::GetAttachStreamIdByContext(context);
        // aicpu task
        ge::KernelLaunchInfo aicpu_task =
            ge::KernelLaunchInfo::CreateAicpuKfcTask(context, SO_NAME.c_str(), KERNEL_NAME_V1.c_str());
        aicpu_task.SetStreamId(attach_stream_id);
        MC2_CHECK_SUCCESS_RET(context->GetNodeName(), Mc2GenTaskOpsUtils::CreateAicpuTaskV1(context, aicpu_task));
        tasks.insert(tasks.begin() + aicore_idx, aicpu_task.Serialize());
        OPS_LOG_I(context->GetNodeName(), "aicpu task for 310P done.");
        // aicore task
        Mc2GenTaskOpsUtils::InsertHiddenInputForAicoreV1(context, tasks);
        OPS_LOG_I(context->GetNodeName(), "aicore task for 310P done.");
        return ge::GRAPH_SUCCESS;
    }

    return Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(context, tasks);
}
} // namespace ops
