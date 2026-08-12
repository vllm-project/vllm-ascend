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
 * \file mc2_gen_task_utils.h
 * \brief
 */

#ifndef OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_GEN_TASK_OPS_UTILS
#define OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_GEN_TASK_OPS_UTILS

#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/kernel_launch_info.h"
#include "graph/arg_desc_info.h"
#include "platform/soc_spec.h"
#include "platform/platform_ascendc.h"

namespace ops {
class Mc2GenTaskOpsUtils {
public:
    static bool IsComputationOnly();
    static int64_t GetAttachStreamIdByContext(const gert::ExeResGenerationContext *context, size_t idx = 0);
    static ge::Status CommonKFCMc2CalcParamFunc(const gert::ExeResGenerationContext *context,
                                                const ge::AscendString &name, const ge::AscendString &reuse_key);
    static ge::Status CommonKFCMc2GenTask(const gert::ExeResGenerationContext *context,
                                          std::vector<std::vector<uint8_t>> &tasks);
    static ge::Status InsertHiddenInputsForAicoreTask(const gert::ExeResGenerationContext *context,
                                                      ge::KernelLaunchInfo &aicore_task,
                                                      size_t (*get_insert_idx)(const std::vector<ge::ArgDescInfo> &),
                                                      size_t input_cnt = 1U);
    static ge::Status InsertHiddenInputForAicoreV1(const gert::ExeResGenerationContext *context,
                                                   std::vector<std::vector<uint8_t>> &tasks);
    static ge::Status CreateAicpuTaskV1(const gert::ExeResGenerationContext *context, ge::KernelLaunchInfo &aicpu_task);
};
} // namespace ops
#endif // OPS_TRANSFORMER_DEV_MC2_COMMON_INC_MC2_GEN_TASK_OPS_UTILS
