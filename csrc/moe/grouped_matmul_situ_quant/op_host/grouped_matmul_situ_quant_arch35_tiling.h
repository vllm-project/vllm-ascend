/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_situ_quant_arch35_tiling.h
 * \brief arch35（Ascend950）MX A8W4 融合路径的 tiling 入口。
 *
 * 与 A2/A3 的 A8W4 MSD 路径（grouped_matmul_situ_quant_a8w4_base_tiling.*）并列，
 * 由 grouped_matmul_situ_quant_tiling.cpp 按 x 的 dtype 分派。
 */
#ifndef __OP_HOST_GROUPED_MATMUL_SITU_QUANT_ARCH35_TILING_H__
#define __OP_HOST_GROUPED_MATMUL_SITU_QUANT_ARCH35_TILING_H__

#include "exe_graph/runtime/tiling_context.h"
#include "graph/ge_error_codes.h"

namespace optiling {

ge::graphStatus TilingForGMMSituQuantArch35(gert::TilingContext *context);

} // namespace optiling
#endif // __OP_HOST_GROUPED_MATMUL_SITU_QUANT_ARCH35_TILING_H__
