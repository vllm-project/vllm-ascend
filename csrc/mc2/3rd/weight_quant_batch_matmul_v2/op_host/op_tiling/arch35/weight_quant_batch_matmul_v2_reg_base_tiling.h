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
 * \file weight_quant_batch_matmul_v2_reg_base_tiling.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_REG_BASE_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_REG_BASE_TILING_H

#include "weight_quant_batch_matmul_v2_basic_block_tiling.h"
#include "../../../op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"

namespace optiling {

class Mc2WeightQuantBatchMatmulV2RegBase : public Mc2WeightQuantBatchMatmulV2Tiling {
public:
    explicit Mc2WeightQuantBatchMatmulV2RegBase(gert::TilingContext *context)
        : Mc2WeightQuantBatchMatmulV2Tiling(context)
    {
        if (context->GetCompileInfo() == nullptr) {
            InitCompileInfo();
        }
        tilingSolver_.Init();
    }

    ~Mc2WeightQuantBatchMatmulV2RegBase() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    void PrintCVTilingData(bool debugLevel) const;
    ge::graphStatus PostTiling() override;
    const Mc2WeightQuantBatchMatmulV2RegBaseTilingData GetTilingData() const
    {
        return *tilingData_;
    }
    std::unique_ptr<Mc2WeightQuantBatchMatmulV2RegBaseTilingData> tilingData_ = nullptr;

private:
    void SetBubTiling();
    void SetAdditionalParam();
    void GetBubTilingA16W4NZ(int64_t &nBubSize, int64_t &kBubSize) const;
    void GetBubTilingA16W4ND(int64_t &nBubSize, int64_t &kBubSize) const;
    void GetBubTilingA16W8NDPerGroup(int64_t &nBubSize, int64_t &kBubSize) const;
    void SetMatmulTiling();

    Mc2WeightQuantBatchMatmulV2BasicBlockTiling tilingSolver_;

private:
    ge::graphStatus InstantiateTilingData();
};
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_REG_BASE_TILING_H
