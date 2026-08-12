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
 * \file batch_matmul_v3_tiling_advanced.h
 * \brief
 */
#ifndef __OP_HOST_BATCH_MATMUL_V3_TILING_ADVANCED_H__
#define __OP_HOST_BATCH_MATMUL_V3_TILING_ADVANCED_H__

#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_advanced.h"

namespace optiling {
namespace Mc2batch_matmul_v3_advanced {
using namespace mc2_matmul_v3_advanced;
class Mc2BatchMatMulV3Tiling : public Mc2MatMulV3Tiling {
public:
    explicit Mc2BatchMatMulV3Tiling(gert::TilingContext *context) : Mc2MatMulV3Tiling(context) {};

    ~Mc2BatchMatMulV3Tiling() override = default;

    ge::graphStatus DoTiling() override;

protected:
    ge::graphStatus GetBatchInfo(const gert::TilingContext &context, Mc2MatMulV3Args &args,
                                 Mc2MatMulV3BatchInfo &batchInfo);

private:
    void MergeBatchAndMAxis(Mc2MatMulV3Args &args, Mc2MatMulV3BatchInfo &batchInfo);
    ge::graphStatus GetBmmBiasInfo(const gert::TilingContext &context, Mc2MatMulV3Args &args,
                                   Mc2MatMulV3BatchInfo &batchInfo);
};
} // namespace Mc2batch_matmul_v3_advanced
} // namespace optiling
#endif // __OP_HOST_BATCH_MATMUL_V3_TILING_ADVANCED_H__
