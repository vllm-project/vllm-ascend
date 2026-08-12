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
 * \file mc2_moe_gen_task_ops_utils.h
 * \brief
 */

#ifndef OPS_TRANSFORMER_DEV_MC2_COMMON_INC_OPS_MC2_MOE_GEN_TASK_UTILS
#define OPS_TRANSFORMER_DEV_MC2_COMMON_INC_OPS_MC2_MOE_GEN_TASK_UTILS

#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/kernel_launch_info.h"
#include "graph/arg_desc_info.h"

namespace ops {
class Mc2MoeGenTaskOpsUtils {
public:
    static ge::Status McMoeInsertHiddenInputForAicore(const gert::ExeResGenerationContext *context,
                                                      const int32_t groupCnt, std::vector<std::vector<uint8_t>> &tasks);
    static ge::Status CreateAicpuTaskMc2Moe(const gert::ExeResGenerationContext *context,
                                            ge::KernelLaunchInfo &aicpuTask, const int32_t groupCnt);
    static ge::Status Mc2MoeInsertTask(const gert::ExeResGenerationContext *context,
                                       std::vector<std::vector<uint8_t>> &tasks, int32_t groupCnt);
    static ge::Status Mc2MoeGenTaskCallback(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks);
    static ge::Status Mc2MoeGenTaskCallbackV2(const gert::ExeResGenerationContext *context,
                                              std::vector<std::vector<uint8_t>> &tasks);
};
} // namespace ops
#endif // OPS_TRANSFORMER_DEV_MC2_COMMON_INC_OPS_MC2_MOE_GEN_TASK_UTILS
