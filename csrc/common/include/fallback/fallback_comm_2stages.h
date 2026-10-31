/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GRAPH_FALLBACK_COMMON_TWOSTAGES_H_
#define INC_EXTERNAL_GRAPH_FALLBACK_COMMON_TWOSTAGES_H_

#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/op_execute_prepare_context.h"
#include "exe_graph/runtime/op_execute_launch_context.h"
#include "exe_graph/runtime/tensor.h"
#include "register/op_impl_kernel_registry.h"
#include "register/op_impl_registry.h"
#if __has_include("runtime/base.h")
#include "runtime/base.h"
#else
#include "runtime/rt_external_base.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {

using OpApiAnyValueDeleter = void (*)(void *);
typedef struct {
  void *pointer;
  OpApiAnyValueDeleter deleter;
} OpApiAnyValue;

// aclnn算子params结构体，用于传递算子一阶段到二阶段的参数，定义在算子仓，由算子感知，GE框架不感知
using OpApiFunc = int (*)(void *, uint64_t, aclOpExecutor *, const aclrtStream);
struct OpApiParams {
  std::vector<OpApiAnyValue> converted_params; // 算子下发依赖的参数
  aclOpExecutor *executor = nullptr; // aclOpExecutor指针
  OpApiFunc op_api_func = nullptr; // aclnnxx函数指针，实现算子launch下发
};

// aclnn算子注册的二阶段launch func，函数实现可以与算子类型无关，所有算子使用同一个二阶段注册接口
ge::graphStatus ExecuteOpLaunch(gert::OpExecuteLaunchContext *context);
}  // namespace fallback

#ifdef __cplusplus
}
#endif

#endif  // INC_EXTERNAL_GRAPH_FALLBACK_COMMON_TWOSTAGES_H_