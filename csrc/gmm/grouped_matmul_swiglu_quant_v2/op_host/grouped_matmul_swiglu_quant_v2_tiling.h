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
 * \file grouped_matmul_swiglu_quant_v2_tiling.h
 * \brief
 */
#ifndef __OP_HOST_OP_TILING_GROUPED_MATMUL_SWIGLU_QUANT_V2_TILING_H__
#define __OP_HOST_OP_TILING_GROUPED_MATMUL_SWIGLU_QUANT_V2_TILING_H__

#include <set>
#include "tiling_base/tiling_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

// GMM 基本信息
BEGIN_TILING_DATA_DEF(GMMSwigluQuantV2BaseParams)
TILING_DATA_FIELD_DEF(int64_t, groupNum);
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, K);
TILING_DATA_FIELD_DEF(int64_t, N);
TILING_DATA_FIELD_DEF(int64_t, M);
TILING_DATA_FIELD_DEF(int64_t, baseM);
TILING_DATA_FIELD_DEF(int64_t, baseN);
TILING_DATA_FIELD_DEF(int64_t, mLimit);
TILING_DATA_FIELD_DEF(int64_t, workSpaceOffset1);
TILING_DATA_FIELD_DEF(int64_t, workSpaceOffset2);
TILING_DATA_FIELD_DEF(int64_t, quantGroupNum);
TILING_DATA_FIELD_DEF(int64_t, isSingleTensor);
TILING_DATA_FIELD_DEF(int64_t, groupListType);
TILING_DATA_FIELD_DEF(int64_t, smoothScaleDimNum);
TILING_DATA_FIELD_DEF(int64_t, singleN);
TILING_DATA_FIELD_DEF(float, swigluLimit);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMSwigluQuantV2BaseParamsOp, GMMSwigluQuantV2BaseParams)

// SwigluQuant部分tiling 基本信息
BEGIN_TILING_DATA_DEF(GMMSwigluQuantV2)
TILING_DATA_FIELD_DEF(int64_t, maxProcessRowNum);
TILING_DATA_FIELD_DEF(int64_t, groupListLen);
TILING_DATA_FIELD_DEF(int64_t, tokenLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMSwigluQuantV2Op, GMMSwigluQuantV2)

// 结构体集合
BEGIN_TILING_DATA_DEF(GMMSwigluQuantV2TilingData)
TILING_DATA_FIELD_DEF_STRUCT(GMMSwigluQuantV2BaseParams, gmmSwigluQuantV2BaseParams);
TILING_DATA_FIELD_DEF_STRUCT(GMMSwigluQuantV2, gmmSwigluQuantV2);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(GMMSwigluQuantV2TilingFusionData)
TILING_DATA_FIELD_DEF(int64_t, cubeBlockDim);
TILING_DATA_FIELD_DEF(int64_t, vectorBlockDim);
TILING_DATA_FIELD_DEF(int64_t, groupNum);
TILING_DATA_FIELD_DEF(int64_t, K);
TILING_DATA_FIELD_DEF(int64_t, N);
TILING_DATA_FIELD_DEF(int64_t, M);
// vector
TILING_DATA_FIELD_DEF(int64_t, ubFactorDimx);
TILING_DATA_FIELD_DEF(int64_t, ubFactorDimy);
TILING_DATA_FIELD_DEF(int64_t, actRight);
TILING_DATA_FIELD_DEF(int64_t, groupListType);
TILING_DATA_FIELD_DEF(int8_t, isSingleTensor);
TILING_DATA_FIELD_DEF(float, swigluLimit);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(GMMSwigluQuantParams)
TILING_DATA_FIELD_DEF(uint32_t, groupNum);
TILING_DATA_FIELD_DEF(uint8_t, groupListType);
TILING_DATA_FIELD_DEF(uint8_t, quantDtype);
TILING_DATA_FIELD_DEF(uint8_t, reserved1);
TILING_DATA_FIELD_DEF(uint8_t, dequantDtype);
TILING_DATA_FIELD_DEF(uint32_t, rowLen);
TILING_DATA_FIELD_DEF(uint32_t, ubAvail);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GMMSwigluQuantParamsOp, GMMSwigluQuantParams)

BEGIN_TILING_DATA_DEF(GMMSwigluQuantTilingDataParams)
TILING_DATA_FIELD_DEF_STRUCT(GMMSwigluQuantParams, gmmSwigluQuantParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupedMatmulSwigluQuantV2_0, GMMSwigluQuantTilingDataParams)
REGISTER_TILING_DATA_CLASS(GroupedMatmulSwigluQuantV2_1, GMMSwigluQuantTilingDataParams)

REGISTER_TILING_DATA_CLASS(GroupedMatmulSwigluQuantV2, GMMSwigluQuantV2TilingData)
REGISTER_TILING_DATA_CLASS(GroupedMatmulSwigluQuantV2_3, GMMSwigluQuantV2TilingFusionData)

struct GMMSwigluV2CompileInfo {
    uint64_t ubSize_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    uint32_t baseM_ = 128;
    uint32_t baseN_ = 256;
    bool supportL12BtBf16;
};

namespace GroupedMatmulSwigluQuantV2Tiling {
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 3;
constexpr uint32_t WEIGHT_SCALE_INDEX = 4;
constexpr uint32_t GROUPLIST_INDEX = 2;
constexpr uint32_t SMOOTH_SCALE_INDEX = 7;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
constexpr uint32_t ATTR_INDEX_DEQUANT_MODE = 0;
constexpr uint32_t ATTR_INDEX_GROUPLIST_TYPE = 5;
constexpr uint32_t ATTR_INDEX_TUNING_CONFIG = 6;
constexpr uint32_t ATTR_INDEX_SWIGLU_LIMIT = 7;
constexpr uint32_t ATTR_INDEX_TRANSPOSE_WEIGHT = 4;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
constexpr uint32_t DIM_4 = 4;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t NUM_EIGHT = 8;
constexpr uint32_t SYS_WORKSPACE_SIZE = static_cast<uint32_t>(16 * 1024 * 1024);
constexpr int64_t USER_WORKSPACE_LIMIT = static_cast<int64_t>(64 * 1024 * 1024);
constexpr int64_t DOUBLE_WORKSPACE_SPLIT = 2;
constexpr int64_t INT32_DTYPE_SIZE = 4;
constexpr int64_t FP32_DTYPE_SIZE = 4;
constexpr int64_t FP32_BLOCK_SIZE = 8;
constexpr int64_t BLOCK_BYTE = 32;
constexpr int64_t SWIGLU_REDUCE_FACTOR = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t ND_WEIGHT_DIM_LIMIT = 3;
constexpr int64_t NZ_WEIGHT_DIM_LIMIT = 5;
constexpr int64_t DOUBLE_ROW = 2;
constexpr int64_t PERCHANNEL_WSCALE_DIM_LIMIT = 2;
constexpr int64_t PERGROUP_WSCALE_DIM_LIMIT = 3;
constexpr int64_t A4W4_WEIGHT_NOTRANS_TILING_KEY_MODE = 4;
constexpr int64_t A4W4_WEIGHT_TRANS_TILING_KEY_MODE = 5;
constexpr int64_t A8W8_FUSION_KEY_MODE = 3;
constexpr int64_t A8W4_MSD_TILING_KEY_MODE = 2;
constexpr int64_t SPLITWORKSPACE_TILING_KEY_MODE = 1;
constexpr int64_t COMMON_TILING_KEY_MODE = 0;
constexpr int64_t A8W4_BASEM = 128;
constexpr int64_t A8W4_BASEK = 256;
constexpr int64_t A8W4_BASEN = 256;
constexpr int64_t SIZE_OF_HALF_2 = 2;

class GroupedMatmulSwigluQuantV2Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GroupedMatmulSwigluQuantV2Tiling(gert::TilingContext* context) : Ops::Transformer::OpTiling::TilingBaseClass(context) {};

    ~GroupedMatmulSwigluQuantV2Tiling() override = default;

protected:
    ge::graphStatus GetPlatformInfo() override {return ge::GRAPH_SUCCESS;};

    ge::graphStatus GetShapeAttrsInfo() override {return ge::GRAPH_SUCCESS;};

    ge::graphStatus DoLibApiTiling() override {return ge::GRAPH_SUCCESS;};

    ge::graphStatus GetWorkspaceSize() override {return ge::GRAPH_SUCCESS;};

    virtual void FillTilingData() = 0;
    virtual void PrintTilingData() = 0;
};

} // namespace GroupedMatmulSwigluQuantV2Tiling
} // namespace optiling

#endif // __OP_HOST_OP_TILING_GROUPED_MATMUL_SWIGLU_QUANT_V2_TILING_H__