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
 * \file weight_quant_batch_matmul_v2_tiling_msd_group.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_GROUP_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_GROUP_H

#include "weight_quant_batch_matmul_v2_tiling.h"
#include "weight_quant_batch_matmul_v2_tiling_key.h"
#include "op_host/tiling_key.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_kernel_tiling_key.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_tiling_data.h"

namespace optiling {

class Mc2WeightQuantBatchMatmulV2TilingMsdGroup : public Mc2WeightQuantBatchMatmulV2Tiling {
public:
    explicit Mc2WeightQuantBatchMatmulV2TilingMsdGroup(gert::TilingContext *context)
        : Mc2WeightQuantBatchMatmulV2Tiling(context)
    {
        Reset();
    };
    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }
    ~Mc2WeightQuantBatchMatmulV2TilingMsdGroup() override = default;

protected:
    std::unique_ptr<Mc2WeightQuantBatchMatmulV2MsdGroupTilingData> tilingData_;

    void Reset();

    ge::graphStatus PostTiling() override;

    bool IsCapable() override;

    bool CheckL1Size() const;

    ge::graphStatus InstantiateTilingData();

    ge::graphStatus DoOpTiling() override;

    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    // 5、计算TilingKey
    uint64_t GetTilingKey() const override
    {
        if (matmulInfoPtr_->bDtype == ge::DT_INT4 && matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP &&
            (matmulInfoPtr_->innerPrecise != 0 || matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ)) {
            // 在A16W4 pergroup 高性能/高精度 tilingkey
            Mc2TilingKeyConfigure tilingKeyConfigure;
            SetCommonTilingKeyElement(tilingKeyConfigure);

            uint64_t socVersionType = tilingKeyConfigure.socVersionType / 10UL;
            uint64_t subSocVersionType = 0UL;
            uint64_t antiquantScenario = tilingKeyConfigure.quantizationScenario;
            uint64_t algorithm = static_cast<uint64_t>(Mc2OptimizationAlgorithmCategory::MULTI_SCALE_DEQUANT);
            uint64_t subAlgorithm = static_cast<uint64_t>(Mc2OptimizationAlgorithmSubCategory::VDEFAULT);
            uint64_t subAlgorithmCustom = 0UL;
            uint64_t innerPrecise = matmulInfoPtr_->innerPrecise;
            uint64_t templateCustom = 0UL;
            uint64_t apiConstexpr = 0UL;
            bool transA = ((tilingKeyConfigure.transposeSituation >> 1) & 1) != 0;
            bool transB = (tilingKeyConfigure.transposeSituation & 1) != 0;
            uint64_t antiquantType = tilingKeyConfigure.antiquantType;
            uint64_t quantType = tilingKeyConfigure.quantType;
            bool hasAntiquantOffset = ((tilingKeyConfigure.optionInputSituation >> 1) & 1) != 0;
            bool hasBias = false;
            bool isBiasFp32 = false;
            bool isWeightNz = (tilingKeyConfigure.weightFormat == 1UL) ? true : false;
            uint64_t templateExtra = 3UL; // 3 means TEMPLATE_EXTRA_NOT_USED
            uint64_t fullLoadMode = 5UL;  // 5 means FULL_LOAD_MODE_NOT_USED
            uint64_t batch = 0UL;
            uint64_t tilingKey_ = GET_TPL_TILING_KEY(
                socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm, subAlgorithmCustom,
                innerPrecise, templateCustom, apiConstexpr, transA, transB, antiquantType, quantType,
                hasAntiquantOffset, hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
            return tilingKey_;
        } else {
            uint64_t socVersionType = 1UL; // 1 means SUPPORT_L0C_TO_OUT
            uint64_t subSocVersionType = 0UL;
            uint64_t antiquantScenario = 0UL;
            uint64_t algorithm = 3UL; // 3 means CUSTOM tilingkey algorithm
            uint64_t subAlgorithm = 0UL;
            uint64_t subAlgorithmCustom = static_cast<uint64_t>(Mc2KernelTemplateType::MSD_GROUP);
            uint64_t innerPrecise = 0UL;
            uint64_t templateCustom = 0UL;
            uint64_t apiConstexpr = 0UL;
            bool transA = matmulInfoPtr_->transA;
            bool transB = matmulInfoPtr_->transB;
            uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
            uint64_t quantType = static_cast<uint64_t>(matmulInfoPtr_->quantType);
            bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
            bool hasBias = false;
            bool isBiasFp32 = false;
            bool isWeightNz = false;
            uint64_t templateExtra = 3UL; // 3 means TEMPLATE_EXTRA_NOT_USED
            uint64_t fullLoadMode = 5UL;  // 5 means FULL_LOAD_MODE_NOT_USED
            uint64_t batch = 0UL;
            uint64_t tilingKey_ = GET_TPL_TILING_KEY(
                socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm, subAlgorithmCustom,
                innerPrecise, templateCustom, apiConstexpr, transA, transB, antiquantType, quantType,
                hasAntiquantOffset, hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
            return tilingKey_;
        }
    }

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        size_t *workspaces = context_->GetWorkspaceSizes(1);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
        workspaces[0] = 64 * 1024 * 1024; // workspace 固定使用 64 * 1024 * 1024
        return ge::GRAPH_SUCCESS;
    }

    bool GetMatMulTiling();
};

} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_MSD_GROUP_H
