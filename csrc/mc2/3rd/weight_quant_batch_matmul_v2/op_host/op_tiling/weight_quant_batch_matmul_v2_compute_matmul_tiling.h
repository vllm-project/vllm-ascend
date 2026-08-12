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
 * \file weight_quant_batch_matmul_v2_compute_matmul_tiling.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_COMPUTE_MATMUL_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_COMPUTE_MATMUL_TILING_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "ops_legacy/op_tiling/op_cache_tiling.h"

namespace optiling {

using Ops::Transformer::OpTiling::AiCoreParams;

struct Mc2MatmulMultiCoreResult {
    uint8_t mDim;
    uint8_t nDim;
    uint8_t batchDim;
};

struct Mc2MatmulParams {
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    ge::DataType aDtype;
    ge::DataType bDtype;
    ge::DataType cDtype;
    ge::DataType biasDtype;
    bool transA;
    bool transB;
    bool hasBias;
    ge::Format format_a;
    ge::Format format_b;
    ge::Format format_out;
    Mc2QuantType quantType;
    bool kbAlign;
};

class Mc2ComputeMatmulTiling {
public:
    static bool GetTiling(AscendC::tiling::TCubeTiling &matmulTiling, Mc2MatmulMultiCoreResult &multiCoreResult,
                          const Mc2MatmulParams &params, const AiCoreParams &aicoreParams,
                          gert::TilingContext *context);

private:
    static bool GetCacheTiling(AscendC::tiling::TCubeTiling &matmulTiling, Mc2MatmulMultiCoreResult &multiCoreResult,
                               const Mc2MatmulParams &params, gert::TilingContext *context);

    static void CalcCommonTiling(AscendC::tiling::TCubeTiling &matmulTiling, const Mc2MatmulParams &params,
                                 const AiCoreParams &aicoreParams);

    static void CalcMsdBufferSize(AscendC::tiling::TCubeTiling &matmulTiling, const Mc2MatmulParams &params);

    static bool MsdA16W8CommonTiling(AscendC::tiling::TCubeTiling &matmulTiling,
                                     Mc2MatmulMultiCoreResult &multiCoreResult, const Mc2MatmulParams &params,
                                     const AiCoreParams &aicoreParams);

    static bool SimpleIncreTiling(AscendC::tiling::TCubeTiling &matmulTiling, Mc2MatmulMultiCoreResult &multiCoreResult,
                                  const Mc2MatmulParams &params, const AiCoreParams &aicoreParams);

    static void Convert2AscendCTiling(const CacheTilingData &tbeTiling, AscendC::tiling::TCubeTiling &matmulTiling,
                                      const Mc2MatmulParams &params, Mc2MatmulMultiCoreResult &multiCoreResult);
    static MatrixTraverse GetIteratorOrder(const CacheTilingData &tbeTiling, int32_t singleCoreM, int32_t singleCoreN,
                                           int32_t singleCoreK, ge::DataType aDtype);

    static bool tryComputeSimpleTiling(AscendC::tiling::TCubeTiling &matmulTiling, const Mc2MatmulParams &params,
                                       const AiCoreParams &aicoreParams);

    static bool tryAFullLoad(AscendC::tiling::TCubeTiling &matmulTiling, const Mc2MatmulParams &params,
                             const AiCoreParams &aicoreParams);

    static bool trySimpleTilingNormalLoad(AscendC::tiling::TCubeTiling &matmulTiling, const Mc2MatmulParams &params,
                                          const AiCoreParams &aicoreParams);
};
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_COMPUTE_MATMUL_TILING_H
