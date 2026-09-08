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
 * \file quant_lightning_indexer_v2_tiling.h
 * \brief
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_TILING_H
#define QUANT_LIGHTNING_INDEXER_V2_TILING_H

#include "err/ops_err.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

namespace optiling {
// ------------------公共定义--------------------------
struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class DataLayout : uint32_t { BSND = 0, TND = 1, PA_BBND = 2 };

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGTHS_INDEX = 2;
constexpr uint32_t QUERY_DEQUANT_SCALE_INDEX = 3;
constexpr uint32_t KEY_DEQUANT_SCALE_INDEX = 4;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 5;
constexpr uint32_t CU_SEQLENS_K_INDEX = 6;
constexpr uint32_t SEQUSED_Q_INDEX = 7;
constexpr uint32_t SEQUSED_K_INDEX = 8;
constexpr uint32_t CMP_RESIDUAL_K_INDEX = 9;
constexpr uint32_t BLOCK_TABLE_INDEX = 10;
constexpr uint32_t OUTPUT_IDX_OFFSET_INDEX = 11;
constexpr uint32_t METADATA_INDEX = 12;
constexpr uint32_t SPARSE_INDICES_INDEX = 0;
constexpr uint32_t SPARSE_VALUES_INDEX = 1;
// Attributes Index
constexpr uint32_t ATTR_TOPK_INDEX = 0;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 1;
constexpr uint32_t ATTR_MAX_SEQLEN_Q_INDEX = 2;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_MASK_MODE_INDEX = 5;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 6;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 7;

// Dim Index
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
// Dim Num
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_LIMIT = 2048;
constexpr uint32_t SPARSE_LIMIT_8K = 8192;
constexpr uint32_t G_SIZE_LIMIT = 64;
constexpr uint32_t G_SIZE_LIMIT_32_950 = 32;
constexpr uint32_t BLOCK_SIZE_LIMIT = 1024;
constexpr uint32_t BLOCK_SIZE_FACTOR = 16;
constexpr uint32_t SPARSE_MODE_LOWER = 3;
constexpr uint32_t METADATA_LIMIT = 1024;
constexpr int32_t QUANT_MODE_FP8 = 1;
constexpr int32_t QUANT_MODE_INT8 = 2;
constexpr int32_t QUANT_MODE_MXFP8 = 3;
constexpr int32_t QUANT_MODE_HIFLOAT8 = 4;
constexpr int32_t QUANT_MODE_MXFP4 = 5;
constexpr uint32_t MX_SCALE_SHAPE_ALIGN = 64;
constexpr uint32_t MX_E8M0_SCALE_PACK_NUM = 2; // MX的E8M0 scale形状最后一维打包数为2
constexpr uint32_t MXFP4_PACK_NUM = 2;         // 每个uint8承载2个FP4 E2M1逻辑元素
constexpr uint32_t MX_SCALE_GROUP_SIZE = 32;   // MX量化每32个D维元素对应1个E8M0 scale

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(QLIV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint32_t, cmpRatio)
TILING_DATA_FIELD_DEF(uint32_t, returnValue)
TILING_DATA_FIELD_DEF(int32_t, maxSeqlenQ)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, keyDequantScaleStride0)
TILING_DATA_FIELD_DEF(uint32_t, quantMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantLightningIndexerV2, QLIV2TilingData)

// -----------算子CompileInfo定义-------------------
struct QLIV2CompileInfo {};

// -----------算子Tiling入参结构体定义---------------
struct QLIV2ParaInfo {
    TilingRequiredParaInfo query = {nullptr, nullptr};
    TilingRequiredParaInfo key = {nullptr, nullptr};
    TilingRequiredParaInfo weights = {nullptr, nullptr};
    TilingRequiredParaInfo query_dequant_scale = {nullptr, nullptr};
    TilingRequiredParaInfo key_dequant_scale = {nullptr, nullptr};
    TilingOptionalParaInfo cuSeqLensQ = {nullptr, nullptr};
    TilingOptionalParaInfo cuSeqLensK = {nullptr, nullptr};
    TilingOptionalParaInfo sequsedQ = {nullptr, nullptr};
    TilingOptionalParaInfo sequsedK = {nullptr, nullptr};
    TilingOptionalParaInfo cmpResidualK = {nullptr, nullptr};
    TilingOptionalParaInfo blockTable = {nullptr, nullptr};
    TilingOptionalParaInfo outputIdxOffset = {nullptr, nullptr};
    TilingOptionalParaInfo metadata = {nullptr, nullptr};
    TilingRequiredParaInfo attenOut = {nullptr, nullptr};
    TilingRequiredParaInfo sparseValues = {nullptr, nullptr};

    const int32_t *quantMode = nullptr;
    const int32_t *maxSeqlenQ = nullptr;
    const char *layOutQuery = nullptr;
    const char *layOutKey = nullptr;
    const int32_t *blockSize = nullptr;
    const int32_t *sparseMode = nullptr;
    const int32_t *sparseCount = nullptr;
    const int32_t *cmpRatio = nullptr;
    const int32_t *returnValue = nullptr;
};

// -----------算子Tiling入参信息类---------------
class QLIV2TilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    QLIV2ParaInfo opParamInfo;
    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch = NpuArch::DAV_2201;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t gSize = 0;
    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    uint32_t sparseCount = 0;
    uint32_t cmpRatio = 1;
    bool returnValue = false;
    uint32_t keyStride0 = 0;
    uint32_t keyDequantScaleStride0 = 0;
    std::vector<uint32_t> keyStridesVec;
    std::vector<uint32_t> keyDequantScaleStridesVec;
    int32_t maxSeqlenQ = -1;
    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout
    DataLayout inputQLayout = DataLayout::BSND;
    DataLayout inputKLayout = DataLayout::PA_BBND;
};

// -----------算子Tiling入参信息解析及Check类---------------
class QLIV2InfoParser {
public:
    explicit QLIV2InfoParser(gert::TilingContext *context) : context_(context) {}
    ~QLIV2InfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                        const std::string &actualSeqLenName) const;
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus CheckAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus CheckScaleShape();
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetAndCheckBlockSize();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetQueryKeyAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetAttenMaskInfo();
    ge::graphStatus GetActualSeqInfo();
    ge::graphStatus CheckKeyContiguous() const;
    void GenerateInfo(QLIV2TilingInfo &QLIV2Info);
    ge::graphStatus ParseAndCheck(QLIV2TilingInfo &QLIV2Info);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    QLIV2ParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    // Layout
    DataLayout qLayout_ = DataLayout::BSND;
    DataLayout kLayout_ = DataLayout::PA_BBND;
    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch_ = NpuArch::DAV_2201;
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKType_ = ge::DT_FLOAT16;
    ge::DataType weightsType_ = ge::DT_FLOAT16;
    ge::DataType inputQueryScaleType_ = ge::DT_FLOAT16;
    ge::DataType inputKeyScaleType_ = ge::DT_FLOAT16;
    ge::DataType blockTableType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType valuesOutType_ = ge::DT_BF16;
    std::vector<uint32_t> keyStridesVec_;
    std::vector<uint32_t> keyDequantScaleStridesVec_;
};

// ---------------算子Tiling类---------------
class QuantLightningIndexerV2Tiling {
public:
    explicit QuantLightningIndexerV2Tiling(gert::TilingContext *context) : context_(context) {};
    ge::graphStatus DoTiling(QLIV2TilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    QLIV2TilingData tilingData_;
};

} // namespace optiling
#endif // QUANT_LIGHTNING_INDEXER_V2_TILING_H
