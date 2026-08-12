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
 * \file weight_quant_batch_matmul_v2_tiling_custom.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"

namespace optiling {
class Mc2WeightQuantBatchMatmulV2TilingCustom : public Mc2WeightQuantBatchMatmulV2Tiling {
public:
    explicit Mc2WeightQuantBatchMatmulV2TilingCustom(gert::TilingContext *context)
        : Mc2WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    }
    Mc2WeightQuantBatchMatmulV2TilingCustom(gert::TilingContext *context, Mc2WeightQuantBatchMatmulV2TilingData *out)
        : Mc2WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
        tilingData_ = out;
        InitCompileInfo();
        isOutTilingData_ = true;
    }
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~Mc2WeightQuantBatchMatmulV2TilingCustom() override = default;

protected:
    Mc2WeightQuantBatchMatmulV2TilingData *tilingData_ = nullptr;
    std::unique_ptr<Mc2WeightQuantBatchMatmulV2TilingData> tilingDataManager_;
    // mc2信息
    bool isOutTilingData_ = false;
    uint64_t cubeBaseN_;

    bool IsCapable() override;
    void Reset();
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus InstantiateTilingData();
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

    bool GetMatMulTiling();
    void SetShapeSize();
    void AdjustMatmulTiling() const;
    void AdjustL1Size() const;
    void ComputeDefaultBlock(uint64_t &defaultVecSingleK, uint64_t &defaultVecSingleN);
    void ComputeGroupDefaultBlock(uint64_t &defaultVecSingleK, uint64_t &defaultVecSingleN);
    void ReviseGroupDefaultBlockWithTrans(uint64_t &defaultVecSingleK, uint64_t &defaultVecSingleN);
    void ReviseGroupDefaultBlockWithoutTrans(uint64_t &defaultVecSingleK, uint64_t &defaultVecSingleN);
    void ComputeVectorDefaultBlock(uint64_t &defaultVecSingleK, uint64_t &defaultVecSingleN);
    void ComputeInt4VectorDefaultBlock(uint64_t &defaultVecSingleK, uint64_t &defaultVecSingleN);
    uint64_t ComputeAntiquantBuffer(uint64_t &defaultVecSingleK, uint64_t &defaultVecSingleN);
    uint64_t ComputeWeightBuffer(uint64_t defaultVecSingleK, uint64_t defaultVecSingleN);
    void ComputeInt8VectorDefaultBlock(uint64_t &defaultVecSingleK, uint64_t &defaultVecSingleN) const;
    bool GetTilingFromCache();
    bool CheckCacheTiling();
    bool InvokeCacheTiling();
};

} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_H
