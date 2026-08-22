/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dequant_situ_quant_tiling.h
 * \brief
 */

#ifndef DEQUANT_SITU_QUANT_TILING_H
#define DEQUANT_SITU_QUANT_TILING_H

#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../op_graph/dequant_situ_quant_proto.h"
#include "../../dequant_swiglu_quant/tiling_base/tiling_base.h"
#include "../../dequant_swiglu_quant/tiling_base/tiling_templates_registry.h"

using namespace Ops::NN::Optiling;
namespace optiling {

BEGIN_TILING_DATA_DEF(DequantSituQuantTilingData)
TILING_DATA_FIELD_DEF(uint32_t, is32BAligned);
TILING_DATA_FIELD_DEF(uint32_t, isDoubleBuffer);
TILING_DATA_FIELD_DEF(uint64_t, rowLen);
TILING_DATA_FIELD_DEF(uint64_t, colLen);
TILING_DATA_FIELD_DEF(uint32_t, baseRowLen);
TILING_DATA_FIELD_DEF(uint32_t, baseColLen);
TILING_DATA_FIELD_DEF(uint32_t, activateLeft);
TILING_DATA_FIELD_DEF(uint32_t, dequantBiasIsEmpty);
TILING_DATA_FIELD_DEF(uint32_t, quantScaleIsEmpty);
TILING_DATA_FIELD_DEF(uint32_t, quantOffsetIsEmpty);
TILING_DATA_FIELD_DEF(uint32_t, quantIsOne);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(float, beta);
TILING_DATA_FIELD_DEF(float, linearBeta);
TILING_DATA_FIELD_DEF(uint32_t, expertNum);
TILING_DATA_FIELD_DEF(uint32_t, hasGroupIndex);
TILING_DATA_FIELD_DEF(uint32_t, hasActivationScale);
TILING_DATA_FIELD_DEF(uint32_t, isPreDequantized);
TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DequantSituQuant, DequantSituQuantTilingData)

struct DequantSituQuantCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

constexpr int64_t DSQ_STATIC_QUANT_ONE = 10000;
constexpr int64_t DSQ_STATIC_QUANT_VEC = 10001;
constexpr int64_t DSQ_STATIC_QUANT_ONE_BIAS = 10002;
constexpr int64_t DSQ_STATIC_QUANT_VEC_BIAS = 10003;
constexpr int64_t DSQ_DYNAMIC_QUANT_NO_SMOOTH = 20000;
constexpr int64_t DSQ_DYNAMIC_QUANT_SMOOTH = 20001;
constexpr int64_t DSQ_DYNAMIC_QUANT_NO_SMOOTH_BIAS = 20002;
constexpr int64_t DSQ_DYNAMIC_QUANT_SMOOTH_BIAS = 20003;
constexpr int64_t DSQ_INT32_DYNAMIC = 30000;
constexpr int64_t DSQ_BF16_DYNAMIC = 40000;

class DequantSituQuantTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit DequantSituQuantTiling(gert::TilingContext* cont) : TilingBaseClass(cont) { Reset(); }
    ~DequantSituQuantTiling() override = default;

    void Reset(gert::TilingContext* cont) override
    {
        TilingBaseClass::Reset(cont);
        Reset();
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    void ShowTilingData();
    ge::graphStatus GetShapeAttrsInfoInner();
    ge::graphStatus CheckInputShapes();
    ge::graphStatus CheckInputShapesInt8(int64_t dimNum, int64_t inDimy, int64_t outDimy);
    ge::graphStatus CheckInputShapesInt32(int64_t inDimy, int64_t outDimy);
    ge::graphStatus CheckInputShapesBF16();
    bool SetAttrs(const gert::RuntimeAttrs* attrs);
    bool CalcTiling(const uint32_t totalCores, const uint64_t ubSize);
    bool CalcUbMaxTileLen(const uint64_t ubSize, uint32_t& maxTileLen);
    bool CalcOptBaseShape(uint32_t maxTileLen);
    ge::graphStatus ValidateInt32Contract();

    const char* opName = "";
    uint32_t totalCore = 0;
    uint32_t totalUsedCoreNum = 0;
    uint32_t inputDTypeLen = 1;
    uint32_t ubMinBlockLen = 32;
    uint32_t cacheLineLen = 512;
    uint32_t maxTileLen = 0;
    uint32_t optBaseRowLen = 0;
    uint32_t optBaseColLen = 0;
    uint64_t workspaceSize_ = 0;

    bool hasDequantBias = false;
    bool hasQuantScale = false;
    bool hasQuantOffset = false;
    bool quantIsOne = false;
    uint32_t activateLeft = 0;
    int64_t quantMode = 0;
    float beta = 4.0f;
    float linearBeta = 25.0f;
    uint64_t quantScaleShapeSize = 0;

    int64_t inDimx = 0;
    int64_t inDimy = 0;
    int64_t outDimy = 0;

    ge::DataType xDtype_ = ge::DT_INT8;
    bool isPreDequantized_ = false;
    bool hasWeightScale_ = false;
    bool hasActivationScale_ = false;
    bool hasGroupIndex_ = false;
    uint32_t expertNum_ = 1;

    DequantSituQuantTilingData tilingData;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

} // namespace optiling
#endif // DEQUANT_SITU_QUANT_TILING_H
