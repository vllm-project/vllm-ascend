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
 * \file grouped_matmul_swiglu_quant_v2_base_tiling.h
 * \brief
 */
#ifndef __OP_HOST_OP_TILING_GROUPED_MATMUL_SWIGLU_QUANT_V2_BASE_TILING_H__
#define __OP_HOST_OP_TILING_GROUPED_MATMUL_SWIGLU_QUANT_V2_BASE_TILING_H__

#include "grouped_matmul_swiglu_quant_v2_tiling.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"

namespace optiling {
namespace GroupedMatmulSwigluQuantV2Tiling {

class GroupedMatmulSwigluQuantV2BaseTiling : public GroupedMatmulSwigluQuantV2Tiling {
public:
    explicit GroupedMatmulSwigluQuantV2BaseTiling(gert::TilingContext* context) : GroupedMatmulSwigluQuantV2Tiling(context) {};

    ~GroupedMatmulSwigluQuantV2BaseTiling() override = default;

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    ge::graphStatus PostTiling() override;

    void FillTilingData() override;
    void PrintTilingData() override;
    void SetTilingKeyAndScheMode(void);
    ge::graphStatus ParseInputAndAttr();
    int64_t CalMaxRowInUbA8W4(const uint64_t ubSize, const uint64_t n) const;
    int64_t CalMaxRowInUb(const uint64_t ubSize, const uint64_t n) const;
    int32_t FindBestSingleN(const uint32_t &aicNum, int64_t baseM, int64_t baseN) const;
    bool TryFullLoadA(int32_t baseM, int64_t baseN, int64_t baseK, uint64_t l1Size);
    ge::graphStatus DynamicTilingSingleN(gert::TilingContext *context, const uint32_t &aicNum,
                    int64_t baseM, int64_t baseN, int64_t baseK);

private:
    GMMSwigluQuantV2TilingData tilingData_;
    int64_t k_ = 0;
    int64_t m_ = 0;
    int64_t n_ = 0;
    int64_t quantGroupNum_ = 0;
    int64_t mLimit_ = 0;
    int64_t blockDim_ = 0;
    int64_t maxProcessRowNum_ = 0;
    int64_t groupNum_ = 0;
    int64_t isSingleTensor_ = 1;
    int64_t groupListType_ = 0;
    int64_t smoothScaleDimNum_ = 0;
    int64_t usrWorkspaceLimit_ = 0;
    uint64_t workspaceSize_ = 0;
    int64_t tuningConfig_ = 0;
    float swigluLimtPtr_ = 1000000.0f;
    bool isA8W4MSD_ = false;
    bool isA4W4_ = false;
    bool isNz_ = false;
    bool isWeightTrans_ = false;
    bool isSplitWorkSpace_ = false;
};

}
}
#endif // __OP_HOST_OP_TILING_GROUPED_MATMUL_SWIGLU_QUANT_V2_BASE_TILING_H__
