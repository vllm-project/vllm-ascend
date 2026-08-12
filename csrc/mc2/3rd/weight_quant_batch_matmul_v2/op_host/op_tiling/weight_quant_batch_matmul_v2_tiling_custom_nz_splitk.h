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
 * \file weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_NZ_SPLITK_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_NZ_SPLITK_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"
namespace optiling {

class Mc2WeightQuantBatchMatmulV2CustomNzSplitK : public Mc2WeightQuantBatchMatmulV2Tiling {
public:
    explicit Mc2WeightQuantBatchMatmulV2CustomNzSplitK(gert::TilingContext *context)
        : Mc2WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    }
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~Mc2WeightQuantBatchMatmulV2CustomNzSplitK() override = default;

protected:
    uint64_t cubeSingleN_;
    bool al1FullLoad_;
    std::unique_ptr<Mc2WeightQuantBatchMatmulV2CustomNzSplitKTilingData> tilingData_;
    // std::unique_ptr<Mc2WeightQuantBatchMatmulV2CompileInfo> compileInfoPtr_;

    void Reset();
    ge::graphStatus PostTiling() override;
    bool IsCapable() override;
    ge::graphStatus InstantiateTilingData();
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    void GetMatMulTiling();
};
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_NZ_SPLITK_H
