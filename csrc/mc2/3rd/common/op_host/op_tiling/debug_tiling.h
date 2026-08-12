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
 * \file compile_info.h
 * \brief
 */
#ifndef OPS_MATMUL_COMPILE_INFO_H_
#define OPS_MATMUL_COMPILE_INFO_H_

#include "exe_graph/runtime/tiling_context.h"

namespace Ops {
namespace Transformer {
std::string DebugTilingContext(const gert::TilingContext *context);
std::string DebugTilingData(gert::TilingContext *context);
} // namespace Transformer
} // namespace Ops

#endif // OPS_MATMUL_COMPILE_INFO_H_
