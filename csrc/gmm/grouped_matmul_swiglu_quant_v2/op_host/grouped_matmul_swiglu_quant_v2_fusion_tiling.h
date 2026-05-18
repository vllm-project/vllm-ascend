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
 * \file grouped_matmul_swiglu_quant_v2_fusion_tiling.h
 * \brief
 */
#ifndef __OP_HOST_OP_TILING_GROUPED_MATMUL_SWIGLU_QUANT_V2_FUSION_TILING_H__
#define __OP_HOST_OP_TILING_GROUPED_MATMUL_SWIGLU_QUANT_V2_FUSION_TILING_H__

#include "grouped_matmul_swiglu_quant_v2_tiling.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"

namespace optiling {
namespace GroupedMatmulSwigluQuantV2Tiling {

class GroupedMatmulSwigluQuantV2FusionTiling : public GroupedMatmulSwigluQuantV2Tiling {
public:
    explicit GroupedMatmulSwigluQuantV2FusionTiling(gert::TilingContext* context) : GroupedMatmulSwigluQuantV2Tiling(context) {};

    ~GroupedMatmulSwigluQuantV2FusionTiling() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus PostTiling() override;
    ge::graphStatus ParseInputAndAttr();
    void FillTilingData() override;
    void PrintTilingData() override;
private:
    GMMSwigluQuantV2TilingFusionData tilingData_;
    uint64_t workspaceSize_;
    uint32_t blockDim_;
    int64_t k_;
    int64_t m_;
    int64_t n_;
    int32_t groupNum_;
    int32_t aicCoreNum_;
    int32_t aivCoreNum_;
    int64_t ubFactorDimx_;
    int64_t groupListType_ = 0;
    int8_t isSingleTensor_;
};

}
}
#endif // __OP_HOST_OP_TILING_GROUPED_MATMUL_SWIGLU_QUANT_V2_FUSION_TILING_H__
