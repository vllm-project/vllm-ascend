/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file matmul_v3_tiling_helper.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_TILING_HELPER_H__
#define __OP_HOST_MATMUL_V3_TILING_HELPER_H__

#include "matmul_v3_common_advanced.h"
#include "matmul_v3_compile_info_advanced.h"
#include "matmul_v3_tiling_key.h"

namespace optiling {
namespace mc2_matmul_v3_advanced {
class Mc2MatMulV3TilingHelper {
public:
    static void ResetBase(const gert::TilingContext *context, const Mc2MatmulV3CompileInfo &compileInfo,
                          const Mc2MatMulV3Args &args, Mc2MatMulV3RunInfo &runInfo);
    static void CalL1Tiling(const Mc2MatmulV3CompileInfo &compileInfo, const Mc2MatMulV3Args &args,
                            Mc2MatMulV3RunInfo &runInfo);
    static Mc2MatMulV3L0C2Out GetL0C2Out(const gert::TilingContext *context, const Mc2MatmulV3CompileInfo &compileInfo,
                                         const Mc2MatMulV3Args &args, const Mc2MatMulV3RunInfo &runInfo);
    static bool CheckIfDoubleAswt(const gert::TilingContext *context, const Mc2MatMulV3Args &args,
                                  const uint64_t batchC);
};
} // namespace mc2_matmul_v3_advanced
} // namespace optiling

#endif // __OP_HOST_MATMUL_V3_TILING_HELPER_H__
