/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MAKE_OP_EXECUTOR_H__
#define __MAKE_OP_EXECUTOR_H__
#include <set>
#include "opdev/platform.h"
#include "opdev/op_arg_def.h"
#include "opdev/op_executor.h"

using namespace std;
#define CREATE_EXECUTOR() UniqueExecutor(__func__)

#define INFER_SHAPE(KERNEL_NAME, op_args...) ACLNN_SUCCESS

#define ADD_TO_LAUNCHER_LIST_AICORE(KERNEL_NAME, op_args...) ACLNN_SUCCESS

#define ADD_TO_LAUNCHER_LIST_DSA(KERNEL_NAME, op_args...) \
    do { \
        op::OpArgContext *opArgCtx = GetOpArgContext(op_args); \
        CreatDSAKernelLauncher(#KERNEL_NAME, KERNEL_NAME##OpTypeId(), KERNEL_NAME##TaskType, executor, opArgCtx); \
    } while (0)

#endif
