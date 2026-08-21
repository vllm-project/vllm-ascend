/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file situ_mx_quant_tiling_arch35.h
 * \brief Tiling data structure and parameters for Situ + MX quantization
 */

#ifndef QUANT_SITU_MX_QUANT_TILING_ARCH35_H
#define QUANT_SITU_MX_QUANT_TILING_ARCH35_H

#include <cstdint>
#include <vector>
#include <string>
#include <set>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/situ_mx_quant_tiling_data.h"

namespace optiling {

// ==================== CompileInfo ====================
struct SituMxQuantCompileInfo {
    int64_t totalCoreNum{0};
    int64_t ubSize{0};
};

// ==================== Input Info ====================
struct SituMxQuantInputInfo {
    ge::DataType xDtype{ge::DT_UNDEFINED};
    int64_t dimNum{0};
    int64_t inputDim0{1};   // batch dim (collapsed)
    int64_t inputDim1{1};   // row dim (collapsed)
    int64_t inputDim2{0};   // 2H (input last dim)
};

// ==================== Output Info ====================
struct SituMxQuantOutputInfo {
    ge::DataType yDtype{ge::DT_UNDEFINED};
    ge::DataType mxscaleDtype{ge::DT_UNDEFINED};
    int64_t outputDim2{0};  // H (output last dim)
    int64_t outputDim1{1};  // row dim (collapsed)
};

// ==================== Attr Params ====================
struct SituMxQuantAttrParam {
    float beta{1.0f};
    float linearBeta{0.0f};
    bool activateLeft{false};
    int64_t axis{-1};
    int64_t dstType{36};
    bool hasLinearBeta{false};
};

// ==================== Tiling Result ====================
struct SituMxQuantTilingResult {
    int64_t basicDim2{256};
    int64_t basicDim1{1};
    int64_t dimNBlockNum{0};
    int64_t maxBasicNumUbDim2{0};
    int64_t maxBasicNumUbDim1{0};
    int64_t usedCoreNum{0};
    int64_t nCoreNum{1};
    int64_t mCorePerB{1};
};

// ==================== Tiling Class ====================
class SituMxQuantRegbaseTiling {
public:
    explicit SituMxQuantRegbaseTiling(gert::TilingContext* context) : context_(context){};

    ge::graphStatus GetNpuInfo();
    ge::graphStatus ParseAttrs();
    ge::graphStatus ValidateInput();
    ge::graphStatus ValidateOutput();
    ge::graphStatus PreProcess();
    ge::graphStatus CalculateTiling();
    ge::graphStatus FillTilingData();
    void SetTilingKeyAndCore();
    void PrintTilingData() const;

private:
    SituMxQuantCompileInfo compileInfo_;
    SituMxQuantInputInfo inputInfo_;
    SituMxQuantOutputInfo outputInfo_;
    SituMxQuantAttrParam attrParam_;
    SituMxQuantTilingResult tilingResult_;

    uint64_t hasLinearBeta_ = 0;
    uint64_t dstTypeIndex_ = 0;

    gert::TilingContext* context_ = nullptr;
    SituMxQuantTilingData* tilingData_ = nullptr;
};

// ==================== Main function declarations ====================
ge::graphStatus Tiling4SituMxQuant(gert::TilingContext* context);
ge::graphStatus TilingPrepare4SituMxQuant(gert::TilingParseContext* context);

} // namespace optiling
#endif // QUANT_SITU_MX_QUANT_TILING_ARCH35_H
