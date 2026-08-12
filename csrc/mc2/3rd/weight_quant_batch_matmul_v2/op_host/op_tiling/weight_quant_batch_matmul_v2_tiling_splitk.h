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
 * \file weight_quant_batch_matmul_v2_tiling_splitk.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_SPLITK_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_SPLITK_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"

namespace optiling {
class Mc2WeightQuantBatchMatmulV2TilingSplitK : public Mc2WeightQuantBatchMatmulV2Tiling {
public:
    explicit Mc2WeightQuantBatchMatmulV2TilingSplitK(gert::TilingContext *context)
        : Mc2WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    }
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~Mc2WeightQuantBatchMatmulV2TilingSplitK() override = default;

protected:
    std::unique_ptr<Mc2WeightQuantBatchMatmulV2TilingData> tilingData_;
    void Reset();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus InstantiateTilingData();
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    bool GetMatMulTiling();
    uint64_t GetTilingKey() const override;
};
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_SPLITK_H
