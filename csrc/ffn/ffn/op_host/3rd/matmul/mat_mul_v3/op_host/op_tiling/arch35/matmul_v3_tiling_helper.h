/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#pragma once

#include "matmul_v3_common_advanced.h"
#include "matmul_v3_compile_info_advanced.h"
#include "matmul_v3_tiling_key.h"
#include "../../../op_kernel/arch35/mat_mul_v3_tiling_key_public.h"

namespace optiling {
namespace matmul_v3_advanced {
using StrideIndexPairs = std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>>;
class MatMulV3TilingHelper {
public:
    static void ResetBase(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo);
    static void CalL1Tiling(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo);
    static MatMulV3L0C2Out GetL0C2Out(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                      const MatMulV3RunInfo& runInfo);
    static uint64_t GetStepSmallK(bool isBL1FullLoad, const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                  MatMulV3RunInfo& runInfo);
    static void AdjustBL1TilingCommon(uint64_t aBatchDimAll, const MatmulV3CompileInfo& compileInfo,
                                      const MatMulV3Args& args, MatMulV3RunInfo& runInfo);
    static void AdjustAL1TilingCommon(uint64_t bBatchDimAll, const MatmulV3CompileInfo& compileInfo,
                                      const MatMulV3Args& args, MatMulV3RunInfo& runInfo);
    static void ResetFullLoadLoadBalance(MatMulV3RunInfo& runInfo);
    static bool IsSelfNonContiguous(const gert::TilingContext* context);
    static void GetRebalanceBlock(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                  MatMulV3RunInfo& runInfo, const gert::TilingContext* context);
    static double GetHbmBW(fe::PlatFormInfos* platformInfo);
    static double GetL2BW(fe::PlatFormInfos* platformInfo);
    static double GetCoreFreq(fe::PlatFormInfos* platformInfo);
    static bool IsTransposeNonContiguous(const gert::TilingContext* context, uint64_t idx);
    static bool IsContiguousStride(StrideIndexPairs& strideIndexPairs);
};
} // namespace matmul_v3_advanced
} // namespace optiling
