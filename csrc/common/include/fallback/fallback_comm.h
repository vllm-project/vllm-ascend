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
 * \file fallback_comm.h
 * \brief
 */

#ifndef INC_EXTERNAL_GRAPH_FALLBACK_COMMON_H_
#define INC_EXTERNAL_GRAPH_FALLBACK_COMMON_H_

#include "aclnn/aclnn_base.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/tensor.h"
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

aclDataType ToAclDataType(ge::DataType dtype);
}  // namespace fallback

#ifdef __cplusplus
}
#endif

#endif  // INC_EXTERNAL_GRAPH_FALLBACK_COMMON_H_
