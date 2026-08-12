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
 * \file quant_batch_matmul_v3_tiling_base.cc
 * \brief
 */

#include "quant_batch_matmul_v3_tiling_base.h"
#include "quant_batch_matmul_info_factory.h"
#include "mc2_log.h"
#include "common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"


namespace {

constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_BATCH_DIM_INDEX = 3;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t MIN_DIM_NUM_ND = 2;
constexpr size_t MAX_DIM_NUM_ND = 6;
constexpr size_t A4W4_X2_LEN = 2;
constexpr int32_t IDX_K_LOW = 2;
constexpr int32_t IDX_K_HIGH = 3;
constexpr int32_t IDX_N_LOW = 4;
constexpr int32_t IDX_N_HIGH = 5;
constexpr int32_t IDX_B_LOW = 6;

constexpr uint64_t L2_REAL_SIZE = 168; // B4真实的L2Size大小
constexpr uint64_t L2_FAKE_SIZE = 96;  // B4被上层修改后的L2Size大小
constexpr uint64_t GROUP_MKN_BIT_SIZE = 0xFFFF;
constexpr size_t SCALE_PERGROUP_MIN_DIM_NUM = 2;

static constexpr int8_t OUTPUT_INFER_FAIL = static_cast<int8_t>(-1);
static constexpr int8_t OUTPUT_INFER_SUCCESS = 1;

// Mc2QuantBatchMatmulV3 input index, mc2 is not same
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t SCALE_INDEX = 2;
constexpr uint32_t OFFSET_INDEX = 3;
constexpr uint32_t BIAS_INDEX = 4;
constexpr uint32_t PERTOKEN_SCALE_INDEX = 5;

static optiling::Mc2QuantBatchMatmulInfoFactory g_quantBatchMatmulInfoFactory;
} // namespace

namespace optiling {
using namespace ge;

const std::map<ge::DataType, std::vector<BasicQuantMode>> X2_QUANT_MODE_MAP = {
    {ge::DT_FLOAT8_E4M3FN,
     {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERCHANNEL_MODE, BasicQuantMode::PERBLOCK_MODE,
      BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_FLOAT8_E5M2,
     {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERCHANNEL_MODE, BasicQuantMode::PERBLOCK_MODE,
      BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_FLOAT4_E2M1, {BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_FLOAT4_E1M2, {BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_HIFLOAT8, {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERCHANNEL_MODE, BasicQuantMode::PERBLOCK_MODE}},
    {ge::DT_INT8, {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERCHANNEL_MODE}},

};

const std::map<ge::DataType, std::vector<BasicQuantMode>> X1_QUANT_MODE_MAP = {
    {ge::DT_FLOAT8_E4M3FN,
     {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERTOKEN_MODE, BasicQuantMode::PERBLOCK_MODE,
      BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_FLOAT8_E5M2,
     {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERTOKEN_MODE, BasicQuantMode::PERBLOCK_MODE,
      BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_FLOAT4_E2M1, {BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_FLOAT4_E1M2, {BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_HIFLOAT8, {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERTOKEN_MODE, BasicQuantMode::PERBLOCK_MODE}},
    {ge::DT_INT8, {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERTOKEN_MODE}},

};

Mc2QuantBatchMatmulV3TilingBase::Mc2QuantBatchMatmulV3TilingBase(gert::TilingContext *context, bool isTilingOut)
    : TilingBaseClass(context), inputParams_(*(g_quantBatchMatmulInfoFactory.Get())), isTilingOut_(isTilingOut)
{
}

ge::graphStatus Mc2QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo()
{
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "Set PlatformInfoFortiling fail");
    if (inputParams_.initFlag) {
        OP_LOGD(inputParams_.opName, "No need to get shape and attrs from tiling context again");
        return ge::GRAPH_SUCCESS;
    }

    inputParams_.opName = context_->GetNodeName();
    OPS_LOG_D(inputParams_.opName, "TilingContext: %s", Ops::Transformer::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE(inputParams_.opName, "Invalid context."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeInputs(),
                    OP_LOGE(inputParams_.opName, "Fail to analyze context info."), return ge::GRAPH_FAILED);
    auto transA_str = inputParams_.transA ? "true" : "false";
    auto transB_str = inputParams_.transB ? "true" : "false";
    auto hasBias_str = inputParams_.hasBias ? "true" : "false";
    OP_LOGD(inputParams_.opName, "Input params: MKN[%ld, %ld, %ld], transA[%s], transB[%s], bias[%s].",
            inputParams_.mSize, inputParams_.kSize, inputParams_.nSize, transA_str, transB_str, hasBias_str);

    inputParams_.initFlag = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2QuantBatchMatmulV3TilingBase::CheckContext()
{
    auto x1Shape = context_->GetInputShape(X1_INDEX);
    auto x1Desc = context_->GetInputDesc(X1_INDEX);
    auto x2Shape = context_->GetInputShape(X2_INDEX);
    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    auto scaleShape = context_->GetInputShape(SCALE_INDEX);
    auto scaleDesc = context_->GetInputDesc(SCALE_INDEX);
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(inputParams_.opName, "Function context_->GetAttrs() failed!"),
                    return ge::GRAPH_FAILED);
    auto dtypeAttr = attrs->GetAttrPointer<int64_t>(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dtypeAttr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
                    OP_LOGE(inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu.",
                            context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool Mc2QuantBatchMatmulV3TilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    // if op calls Mc2QuantBatchMatmulV3 with attrs, it could rewrite as GetXXXAttr()
    if (attrs) {
        size_t idx = 0;
        auto dtypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto transposeX1Ptr = attrs->GetAttrPointer<bool>(idx++);
        auto transposeX2Ptr = attrs->GetAttrPointer<bool>(idx++);
        const int64_t *groupSizePtr = nullptr;
        if (attrs->GetAttrNum() > idx) {
            groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
        }
        OP_TILING_CHECK(!dtypePtr, OP_LOGE_WITH_INVALID_INPUT(inputParams_.opName, "dtype attr"), return false);
        inputParams_.outDtype = *dtypePtr;
        inputParams_.transA = transposeX1Ptr ? *transposeX1Ptr : false;
        inputParams_.transB = transposeX2Ptr ? *transposeX2Ptr : false;
        inputParams_.groupSize = groupSizePtr ? static_cast<uint64_t>(*groupSizePtr) : 0ULL;
        if (inputParams_.groupSize != 0ULL) {
            inputParams_.groupSizeK = inputParams_.groupSize & GROUP_MKN_BIT_SIZE;
            inputParams_.groupSizeN =
                (inputParams_.groupSize >> 16U) & GROUP_MKN_BIT_SIZE; // 16 is the bit size of MKN group size
            inputParams_.groupSizeM =
                (inputParams_.groupSize >> 32U) & GROUP_MKN_BIT_SIZE; // groupSizeM start at 32 bit of groupSize
        }
    }

    Mc2QuantBatchMatmulV3Trans trans = Mc2QuantBatchMatmulV3Trans::NO_TRANS;
    SetTransAttr(trans);
    if (!compileInfo_.supportL0c2Out) {
        OP_TILING_CHECK(trans != Mc2QuantBatchMatmulV3Trans::B_TRANS,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "x1 and x2", inputParams_.transA ? "true" : "false",
                            "The value of x1 and x2 transpose must be [false, true]."),
                        return false);
    }
    return true;
}

bool Mc2QuantBatchMatmulV3TilingBase::AnalyzeDtype()
{
    inputParams_.aDtype = context_->GetInputDesc(X1_INDEX)->GetDataType();
    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    inputParams_.bDtype = x2Desc->GetDataType();
    inputParams_.scaleDtype = context_->GetInputDesc(SCALE_INDEX)->GetDataType();
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX);
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    inputParams_.biasDtype = biasDesc != nullptr ? biasDesc->GetDataType() : ge::DT_INT32;
    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    isUbQuant_ = inputParams_.cDtype == ge::DT_BF16 || pertokenScaleDesc != nullptr;
    SetFormat();

    OP_TILING_CHECK(!CheckDtype(), OP_LOGE(inputParams_.opName, "CheckDtype failed!"), return false);
    return true;
}

uint64_t Mc2QuantBatchMatmulV3TilingBase::GetBatchSize(const gert::Shape &shape) const
{
    uint64_t batch = 1;
    auto shapeLen = shape.GetDimNum();
    for (size_t i = LAST_BATCH_DIM_INDEX; i <= shape.GetDimNum(); i++) {
        batch = batch * shape.GetDim(shapeLen - i);
    }
    return batch;
}

bool Mc2QuantBatchMatmulV3TilingBase::InferOutBatchDim(const gert::Shape &x1Shape, const gert::Shape &x2Shape)
{
    inputParams_.batchC = 1U;
    auto x1DimNum = x1Shape.GetDimNum();
    auto x2DimNum = x2Shape.GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    const gert::Shape &shapeLong = x1DimNum > x2DimNum ? x1Shape : x2Shape;
    const gert::Shape &shapeShort = x1DimNum > x2DimNum ? x2Shape : x1Shape;
    size_t validOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    for (size_t i = 0; i < outDimNum - LAST_SECOND_DIM_INDEX; i++) {
        auto shortDim = i < validOffset ? 1 : shapeShort.GetDim(i - validOffset);
        auto longDim = shapeLong.GetDim(i);
        if (shortDim > 1 && longDim > 1 && shortDim != longDim) {
            return false;
        }
        inputParams_.batchC = inputParams_.batchC * static_cast<uint64_t>(std::max(shortDim, longDim));
    }
    return true;
}

int8_t Mc2QuantBatchMatmulV3TilingBase::CheckFusionBatchA(const gert::Shape &x1Shape, const gert::Shape &x2Shape,
                                                          const gert::Shape &biasShape, uint64_t fusedDimValue) const
{
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto x1BatchLen = x1ShapeLen - 2;
    if (inputParams_.isPertoken) { // pertoken dim equals to original m dim, cannot fuse
        return 0;
    }
    if (inputParams_.hasBias && biasShape.GetDimNum() > 1) { // 3 dim batch need original batch dim value, cannot fuse
        return 0;
    }
    if (static_cast<int64_t>(fusedDimValue) > static_cast<int64_t>(INT32_MAX)) { // exceed axis limit, cannot fuse
        if (inputParams_.aDtype == ge::DT_INT4) {
            return OUTPUT_INFER_FAIL;
        }
        return 0;
    }
    // only fusion when x2 shape length is 2
    if (x2ShapeLen == 2 && inputParams_.transA == false && x1BatchLen >= 1) {
        // perBlock量化模式限制，groupSizeM不能整除m轴时，不能合轴
        if (inputParams_.groupSizeM > 1 && x1Shape.GetDim(x1ShapeLen - MIN_DIM_NUM_ND) % inputParams_.groupSizeM != 0) {
            return 0;
        }
        // mx量化模式不能batch合轴
        if (IsMicroScaling()) {
            return 0;
        }
        OP_LOGD("Mc2QuantBatchMatmulV3", "CheckFusionBatchA success, start fusion batch A to m dimendion.");
        return OUTPUT_INFER_SUCCESS;
    }
    return 0;
}

void Mc2QuantBatchMatmulV3TilingBase::DoBatchFusion(uint64_t fusedDimValue)
{
    inputParams_.mSize = fusedDimValue;
    inputParams_.batchC = 1UL;
    inputParams_.batchC1 = 1UL;
    inputParams_.batchC2 = 1UL;
    inputParams_.batchC3 = 1UL;
    inputParams_.batchC4 = 1UL;
    inputParams_.batchA = 1UL;
    inputParams_.batchA1 = 1UL;
    inputParams_.batchA2 = 1UL;
    inputParams_.batchA3 = 1UL;
    inputParams_.batchA4 = 1UL;
}

bool Mc2QuantBatchMatmulV3TilingBase::CheckShapeInRangeForMandtoryInputs(size_t x1ShapeLen, size_t x2ShapeLen) const
{
    OP_TILING_CHECK(x1ShapeLen < MIN_DIM_NUM_ND || x2ShapeLen < MIN_DIM_NUM_ND,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "x1 and x2",
                                                          std::to_string(x1ShapeLen).c_str(),
                                                          "The value of x1 and x2 dimension must be greater than 1."),
                    return false);
    OP_TILING_CHECK(x1ShapeLen > MAX_DIM_NUM_ND || x2ShapeLen > MAX_DIM_NUM_ND,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "x1 and x2",
                                                          std::to_string(x1ShapeLen).c_str(),
                                                          "The value of x1 and x2 dimension must be less than 7."),
                    return false);

    if (inputParams_.aDtype == ge::DT_INT4 && !inputParams_.isPertoken) {
        OP_TILING_CHECK(x2ShapeLen != A4W4_X2_LEN,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            inputParams_.opName, "x2", std::to_string(x2ShapeLen).c_str(),
                            "When the dtype of input is int4 without pertokenScale, the shape dim of x2 must be 2."),
                        return false);
    }
    return true;
}

bool Mc2QuantBatchMatmulV3TilingBase::IsMicroScaling() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0;
}

std::string Mc2QuantBatchMatmulV3TilingBase::QuantModeToString(BasicQuantMode quantMode) const
{
    std::string mode = "default";
    switch (quantMode) {
        case BasicQuantMode::PERTENSOR_MODE:
            mode = "pertensor";
            break;
        case BasicQuantMode::PERCHANNEL_MODE:
            mode = "perchannel";
            break;
        case BasicQuantMode::PERTOKEN_MODE:
            mode = "pertoken";
            break;
        case BasicQuantMode::MX_PERGROUP_MODE:
            mode = "mx_pergroup";
            break;
        case BasicQuantMode::PERBLOCK_MODE:
            mode = "perblock";
            break;
        default:
            break;
    }
    return mode;
}

bool Mc2QuantBatchMatmulV3TilingBase::SetX2QuantMode(BasicQuantMode &x2QuantMode, bool isFp8Hif8Input,
                                                     const gert::Shape &scaleShape)
{
    auto scaleShapeLen = scaleShape.GetDimNum();
    if (IsMicroScaling()) {
        x2QuantMode = BasicQuantMode::MX_PERGROUP_MODE;
    } else if (isFp8Hif8Input && inputParams_.scaleDtype == ge::DT_FLOAT && scaleShapeLen > 1 &&
               (scaleShapeLen > SCALE_PERGROUP_MIN_DIM_NUM || scaleShape.GetDim(0) != 1 ||
                inputParams_.groupSizeN > 1)) {
        x2QuantMode = BasicQuantMode::PERBLOCK_MODE;
    } else if (scaleShape.GetDim(scaleShapeLen - LAST_FIRST_DIM_INDEX) == 1) {
        x2QuantMode = BasicQuantMode::PERTENSOR_MODE;
    } else if (static_cast<uint64_t>(scaleShape.GetDim(scaleShapeLen - LAST_FIRST_DIM_INDEX)) == inputParams_.nSize) {
        x2QuantMode = BasicQuantMode::PERCHANNEL_MODE;
    } else {
        auto x2QuantModeList = X2_QUANT_MODE_MAP.find(inputParams_.bDtype);
        OP_TILING_CHECK(
            x2QuantModeList == X2_QUANT_MODE_MAP.end(),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x2",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                                                  "The dtype of x2 must be within the supported range."),
            return false);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x2",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                                              "The dtype of x2 and the shape of x2 and scale do not match any "
                                              "supported quantification, dtype: %s, quantification: %s.");
        return false;
    }
    return true;
}
bool Mc2QuantBatchMatmulV3TilingBase::SetX1QuantMode(BasicQuantMode &x1QuantMode, bool isFp8Hif8Input,
                                                     const gert::StorageShape *pertokenShape)
{
    if (pertokenShape == nullptr) {
        return true;
    }
    auto &pertoken = pertokenShape->GetStorageShape();
    auto pertokenLen = pertoken.GetDimNum();
    if (IsMicroScaling()) {
        x1QuantMode = BasicQuantMode::MX_PERGROUP_MODE;
    } else if (isFp8Hif8Input && inputParams_.perTokenScaleDtype == ge::DT_FLOAT && pertokenLen > 1) {
        x1QuantMode = BasicQuantMode::PERBLOCK_MODE;
    } else if (isFp8Hif8Input && pertokenLen == 1 && pertoken.GetDim(pertokenLen - LAST_FIRST_DIM_INDEX) == 1) {
        x1QuantMode = BasicQuantMode::PERTENSOR_MODE;
    } else if (pertokenLen == 1 &&
               static_cast<uint64_t>(pertoken.GetDim(pertokenLen - LAST_FIRST_DIM_INDEX)) == inputParams_.mSize) {
        x1QuantMode = BasicQuantMode::PERTOKEN_MODE;
    } else {
        auto x1QuantModeList = X1_QUANT_MODE_MAP.find(inputParams_.aDtype);
        OP_TILING_CHECK(
            x1QuantModeList == X1_QUANT_MODE_MAP.end(),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x1",
                                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                                  "The dtype of x1 must be within the supported range."),
            return false);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "x1",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                              "The dtype of x1 and the shape of x1 and pertokenScale do not match any "
                                              "supported quantification, dtype: %s, quantification: %s.");
        return false;
    }
    return true;
}

std::string
Mc2QuantBatchMatmulV3TilingBase::QuantModeMapToString(const std::vector<BasicQuantMode> &quantModeList) const
{
    std::stringstream ss;
    ss << "[";
    for (auto it = quantModeList.begin(); it != quantModeList.end(); it++) {
        ss << QuantModeToString(*it);
        if (it + 1 != quantModeList.end()) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

bool Mc2QuantBatchMatmulV3TilingBase::SetQuantMode(const gert::Shape &scaleShape,
                                                   const gert::StorageShape *pertokenShape)
{
    if (!compileInfo_.supportL12BtBf16) {
        inputParams_.isPerTensor = scaleShape.GetDim(0) == 1;
        return true;
    }
    BasicQuantMode x1QuantMode = BasicQuantMode::DEFAULT;
    BasicQuantMode x2QuantMode = BasicQuantMode::DEFAULT;
    bool isFp8Input = (inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2);
    bool isFp8Hif8Input = inputParams_.aDtype == ge::DT_HIFLOAT8 || isFp8Input;
    if (!SetX1QuantMode(x1QuantMode, isFp8Hif8Input, pertokenShape) ||
        !SetX2QuantMode(x2QuantMode, isFp8Hif8Input, scaleShape)) {
        OP_LOGE(inputParams_.opName, "Get quantification of x1/x2 failed.");
        return false;
    }
    inputParams_.isPerChannel = x2QuantMode == BasicQuantMode::PERCHANNEL_MODE;
    inputParams_.isPerTensor = x2QuantMode == BasicQuantMode::PERTENSOR_MODE;
    inputParams_.isDoubleScale = x1QuantMode == BasicQuantMode::PERTENSOR_MODE;
    inputParams_.isPertoken = x1QuantMode == BasicQuantMode::PERTOKEN_MODE;
    inputParams_.isMxPerGroup = x1QuantMode == BasicQuantMode::MX_PERGROUP_MODE && x1QuantMode == x2QuantMode;
    inputParams_.isPerBlock = x1QuantMode == BasicQuantMode::PERBLOCK_MODE;
    OP_TILING_CHECK(!(inputParams_.isPerChannel || inputParams_.isDoubleScale || inputParams_.isPerTensor ||
                      inputParams_.isPertoken || inputParams_.isMxPerGroup || inputParams_.isPerBlock),
                    OP_LOGE(inputParams_.opName,
                            "Unexpected quantification, quantification of x1: %s, quantification of x2: %s",
                            QuantModeToString(x1QuantMode).c_str(), QuantModeToString(x2QuantMode).c_str()),
                    return false);
    return true;
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
bool Mc2QuantBatchMatmulV3TilingBase::AnalyzeInputs()
{
    auto x1Shape = GetX1Shape(X1_INDEX);
    auto x2Shape = GetX2Shape(X2_INDEX);
    auto scaleShape = GetScaleShape(SCALE_INDEX);
    auto pertokenShape = GetPertokenShape(PERTOKEN_SCALE_INDEX);
    inputParams_.isPertoken = pertokenShape != nullptr;
    auto biasShape = GetBiasShape(BIAS_INDEX);
    inputParams_.hasBias = biasShape != nullptr;
    inputParams_.batchBias = inputParams_.hasBias ? GetBatchSize(biasShape->GetStorageShape()) : 1;
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    if (!isTilingOut_ && !CheckShapeInRangeForMandtoryInputs(x1ShapeLen, x2ShapeLen)) {
        return false;
    }
    auto x1Inner = x1Shape.GetDim(x1ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - LAST_SECOND_DIM_INDEX);
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - LAST_SECOND_DIM_INDEX);
    const std::vector<int64_t> dimValueOfMKN = {x1Inner, x1Outer, x2Inner, x2Outer};
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? x1Inner : x1Outer);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? x1Outer : x1Inner);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? x2Outer : x2Inner);
    const std::vector<gert::Shape *> mandtoryShape = {&x1Shape, &x2Shape, &scaleShape};
    if (!AnalyzeGroupInfo(scaleShape, pertokenShape)) {
        return false;
    }
    inputParams_.batchA = GetBatchSize(x1Shape);
    inputParams_.batchB = GetBatchSize(x2Shape);
    AnalyzeBatchInfo(context_->GetInputShape(0)->GetOriginShape(), context_->GetInputShape(1)->GetOriginShape());
    OP_TILING_CHECK(!InferOutBatchDim(x1Shape, x2Shape),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "input",
                        (std::to_string(inputParams_.batchA) + ", " + std::to_string(inputParams_.batchB)).c_str(),
                        "The value of the batch dimension must be broadcastable."),
                    return false);
    if (!SetQuantMode(scaleShape, pertokenShape)) {
        return false;
    }
    if (!isTilingOut_ && !CheckShape(mandtoryShape, biasShape, pertokenShape, dimValueOfMKN)) {
        return false;
    }
    OP_TILING_CHECK(!CheckOutputShapeAvailable(),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opName, "output",
                        (std::to_string(inputParams_.mSize) + "x" + std::to_string(inputParams_.nSize) + "x" +
                         std::to_string(inputParams_.batchC))
                            .c_str(),
                        "The product of output shape dimensions must be within the boundary of INT64_MAX."),
                    return false);
    uint64_t fusedDimValue = inputParams_.mSize * inputParams_.batchA;
    int8_t resultCheckFusionBatchA = CheckFusionBatchA(x1Shape, x2Shape, biasShape->GetStorageShape(), fusedDimValue);
    OP_TILING_CHECK(resultCheckFusionBatchA == OUTPUT_INFER_FAIL,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "M",
                                                          std::to_string(fusedDimValue).c_str(),
                                                          "The fused M must not exceed INT32_MAX in a4w4 case."),
                    return false);
    if (resultCheckFusionBatchA == OUTPUT_INFER_SUCCESS) {
        DoBatchFusion(fusedDimValue);
    }
    auto isPerTensorStr = inputParams_.isPerTensor ? "true" : "false";
    auto isPertokenStr = inputParams_.isPertoken ? "true" : "false";
    OP_LOGD("Mc2QuantBatchMatmulV3", "batchA: %lu, batchB: %lu, batchC: %lu, isPerTensor: %s, isPertoken: %s",
            inputParams_.batchA, inputParams_.batchB, inputParams_.batchC, isPerTensorStr, isPertokenStr);
    return true;
}

bool Mc2QuantBatchMatmulV3TilingBase::CheckOutputShapeAvailable() const
{
    int64_t m = static_cast<int64_t>(inputParams_.mSize);
    int64_t n = static_cast<int64_t>(inputParams_.nSize);
    int64_t b = static_cast<int64_t>(inputParams_.batchC);
    int64_t mul = m * n;
    if (mul / m != n) {
        return false;
    }
    mul *= b;
    return mul / b == m * n;
}

void Mc2QuantBatchMatmulV3TilingBase::AnalyzeBatchInfo(const gert::Shape &oriShapeA, const gert::Shape &oriShapeB)
{
    int32_t numDimA = static_cast<int32_t>(oriShapeA.GetDimNum());
    int32_t numDimB = static_cast<int32_t>(oriShapeB.GetDimNum());
    inputParams_.batchA4 = numDimA > IDX_K_LOW ? oriShapeA.GetDim(numDimA - IDX_K_HIGH) : 1;
    inputParams_.batchA3 = numDimA > IDX_K_HIGH ? oriShapeA.GetDim(numDimA - IDX_N_LOW) : 1;
    inputParams_.batchA2 = numDimA > IDX_N_LOW ? oriShapeA.GetDim(numDimA - IDX_N_HIGH) : 1;
    inputParams_.batchA1 = numDimA > IDX_N_HIGH ? oriShapeA.GetDim(numDimA - IDX_B_LOW) : 1;
    inputParams_.batchB4 = numDimB > IDX_K_LOW ? oriShapeB.GetDim(numDimB - IDX_K_HIGH) : 1;
    inputParams_.batchB3 = numDimB > IDX_K_HIGH ? oriShapeB.GetDim(numDimB - IDX_N_LOW) : 1;
    inputParams_.batchB2 = numDimB > IDX_N_LOW ? oriShapeB.GetDim(numDimB - IDX_N_HIGH) : 1;
    inputParams_.batchB1 = numDimB > IDX_N_HIGH ? oriShapeB.GetDim(numDimB - IDX_B_LOW) : 1;
    auto outShape = context_->GetOutputShape(0)->GetStorageShape();
    int32_t numDimC = static_cast<int32_t>(outShape.GetDimNum());
    inputParams_.batchC4 = numDimC > IDX_K_LOW ? outShape.GetDim(numDimC - IDX_K_HIGH) : 1UL;
    inputParams_.batchC3 = numDimC > IDX_K_HIGH ? outShape.GetDim(numDimC - IDX_N_LOW) : 1UL;
    inputParams_.batchC2 = numDimC > IDX_N_LOW ? outShape.GetDim(numDimC - IDX_N_HIGH) : 1UL;
    inputParams_.batchC1 = numDimC > IDX_N_HIGH ? outShape.GetDim(numDimC - IDX_B_LOW) : 1UL;
}

bool Mc2QuantBatchMatmulV3TilingBase::ReCalcGroupSize(uint64_t &groupSize, uint64_t inputSize, uint64_t scaleSize,
                                                      const char *dimensionName)
{
    if (groupSize == 0ULL) {
        std::string inputName = "x1";
        std::string scaleName = "pertoken_scale";
        if (strcmp(dimensionName, "n") == 0) {
            inputName = "x2";
            scaleName = "scale";
        }
        OP_TILING_CHECK(scaleSize == 0ULL,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "input",
                                                              std::to_string(scaleSize).c_str(),
                                                              "The shape dimension of input cannot be 0."),
                        return false);
        OP_TILING_CHECK(inputSize % scaleSize != 0ULL,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "input", dimensionName,
                                                              "The group_size in the dimension cannot be inferred."),
                        return false);
        groupSize = inputSize / scaleSize;
    }
    return true;
}

bool Mc2QuantBatchMatmulV3TilingBase::AnalyzeGroupInfo(const gert::Shape &scaleShape,
                                                       const gert::StorageShape *pertokenShape)
{
    if (pertokenShape == nullptr) {
        return true;
    }
    auto &pertoken = pertokenShape->GetStorageShape();
    auto pertokenShapeLen = pertoken.GetDimNum();
    auto scaleShapeLen = scaleShape.GetDimNum();
    if (pertokenShapeLen < 2 ||
        scaleShapeLen < 2) { // scaleLen < 2 means invalid input, leave it to the checher behind.
        return true;
    }
    auto pertokenInner = pertoken.GetDim(1);
    auto pertokenOuter = pertoken.GetDim(0);
    uint64_t pertokenMSize = static_cast<uint64_t>(inputParams_.transA ? pertokenInner : pertokenOuter);
    if (!ReCalcGroupSize(inputParams_.groupSizeM, inputParams_.mSize, pertokenMSize, "m")) {
        return false;
    }
    uint64_t pertokenKSize = static_cast<uint64_t>(inputParams_.transA ? pertokenOuter : pertokenInner);
    if (!ReCalcGroupSize(inputParams_.groupSizeK, inputParams_.kSize, pertokenKSize, "k")) {
        return false;
    }
    auto scaleInner = scaleShape.GetDim(1);
    auto scaleOuter = scaleShape.GetDim(0);
    uint64_t scaleNSize = static_cast<uint64_t>(inputParams_.transB ? scaleOuter : scaleInner);
    if (!ReCalcGroupSize(inputParams_.groupSizeN, inputParams_.nSize, scaleNSize, "n")) {
        return false;
    }
    OP_LOGD(inputParams_.opName, "Infer groupSize success. groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.",
            inputParams_.groupSizeM, inputParams_.groupSizeN, inputParams_.groupSizeK);
    return true;
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::Shape &Mc2QuantBatchMatmulV3TilingBase::GetScaleShape(const size_t index)
{
    return context_->GetInputShape(index)->GetOriginShape();
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::Shape Mc2QuantBatchMatmulV3TilingBase::GetX1Shape(const size_t index)
{
    return context_->GetInputShape(index)->GetOriginShape();
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::Shape Mc2QuantBatchMatmulV3TilingBase::GetX2Shape(const size_t index)
{
    return context_->GetInputShape(index)->GetOriginShape();
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::Shape Mc2QuantBatchMatmulV3TilingBase::GetOutputShape(const size_t index)
{
    return context_->GetOutputShape(index)->GetOriginShape();
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::StorageShape *Mc2QuantBatchMatmulV3TilingBase::GetPertokenShape(const size_t index)
{
    return context_->GetOptionalInputShape(index);
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::StorageShape *Mc2QuantBatchMatmulV3TilingBase::GetBiasShape(const size_t index)
{
    return context_->GetOptionalInputShape(index);
}

void Mc2QuantBatchMatmulV3TilingBase::SetFormat()
{
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    auto x1Desc = context_->GetInputDesc(X1_INDEX);
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat())) == ge::Format::FORMAT_FRACTAL_NZ) {
        inputParams_.aFormat = ge::FORMAT_FRACTAL_NZ;
    }
    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat())) == ge::Format::FORMAT_FRACTAL_NZ) {
        inputParams_.bFormat = ge::FORMAT_FRACTAL_NZ;
    }
}

void Mc2QuantBatchMatmulV3TilingBase::SetTransAttr(Mc2QuantBatchMatmulV3Trans &trans) const
{
    if (!inputParams_.transA && inputParams_.transB) { // most for ND
        trans = Mc2QuantBatchMatmulV3Trans::B_TRANS;
    } else if (!inputParams_.transA && !inputParams_.transB) { // most for weight NZ
        trans = Mc2QuantBatchMatmulV3Trans::NO_TRANS;
    } else if (inputParams_.transA && !inputParams_.transB) {
        trans = Mc2QuantBatchMatmulV3Trans::A_TRANS;
    } else {
        trans = Mc2QuantBatchMatmulV3Trans::AB_TRANS;
    }
}

bool Mc2QuantBatchMatmulV3TilingBase::IsCapable()
{
    return true;
}

void Mc2QuantBatchMatmulV3TilingBase::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo_.workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfoPtr->GetPlatformRes("version", "Soc_version", compileInfo_.socVersionStr);
    compileInfo_.socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo_.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo_.l0bSize);

    std::string platformRes;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", platformRes);
    compileInfo_.supportL0c2Out = !platformRes.empty();
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", platformRes);
    compileInfo_.supportL12BtBf16 = (platformRes.find("bf16") != std::string::npos);
    TilingPrepareForOpCache(context_);
    compileInfoInit_ = true;
}

bool Mc2QuantBatchMatmulV3TilingBase::SetPlatformInfoForTiling()
{
    // mc2和qbmm把compileInfo都赋值给compileInfo_，后续硬件信息可以直接从compileInfo_中获取
    if (!compileInfoInit_) {
        auto mmCompileInfo = reinterpret_cast<const Mc2QuantBatchMatmulV3CompileInfo *>(context_->GetCompileInfo());
        OP_TILING_CHECK(mmCompileInfo == nullptr,
                        OP_LOGE(inputParams_.opName, "quant_batch_matmul_v3_tiling GetCompileInfo is null"),
                        return false);
        compileInfo_ = *mmCompileInfo;
    }
    OP_LOGE_IF(compileInfo_.aicNum <= 0, false, inputParams_.opName, "coreNum <= 0");
    aicoreParams_.aicNum = compileInfo_.aicNum;
    OP_LOGE_IF(compileInfo_.l2Size <= 0, false, inputParams_.opName, "l2Size <= 0");
    // 纠正L2实际物理大小
    compileInfo_.l2Size = compileInfo_.l2Size == L2_FAKE_SIZE * MB_SIZE ? L2_REAL_SIZE * MB_SIZE : compileInfo_.l2Size;
    inputParams_.libApiWorkSpaceSize = compileInfo_.workspaceNum;
    aicoreParams_.ubSize = compileInfo_.ubSize;
    aicoreParams_.l1Size = compileInfo_.l1Size;
    aicoreParams_.l0aSize = compileInfo_.l0aSize;
    aicoreParams_.l0cSize = compileInfo_.l0cSize;
    aicoreParams_.numBlocks = 0;
    return true;
}

bool Mc2QuantBatchMatmulV3TilingBase::GetUbDequantExtreSpace()
{
    return false;
}

bool Mc2QuantBatchMatmulV3TilingBase::CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
                                                 const gert::StorageShape *biasShape,
                                                 const gert::StorageShape *pertokenShape,
                                                 const std::vector<int64_t> &dimValueOfMKN) const
{
    (void)mandtoryShape;
    (void)biasShape;
    (void)pertokenShape;
    (void)dimValueOfMKN;
    return false;
}

bool Mc2QuantBatchMatmulV3TilingBase::CheckDtype() const
{
    return false;
}

ge::graphStatus Mc2QuantBatchMatmulV3TilingBase::CalcUbTiling()
{
    return ge::GRAPH_FAILED;
}

uint64_t Mc2QuantBatchMatmulInfo::GetMatmulApiMSize() const
{
    return mSizePerNpu > 0U ? mSizePerNpu : mSize;
}

uint64_t Mc2QuantBatchMatmulInfo::GetTotalMatmulApiMSize(uint64_t baseM) const
{
    if (mSizePerNpu > 0U) {
        return ops::CeilAlign(mSizePerNpu, baseM) * ops::CeilDiv(mSize, mSizePerNpu);
    } else {
        return mSize;
    }
}

uint64_t Mc2QuantBatchMatmulInfo::GetTotalBaseMCnt(uint64_t baseM) const
{
    return ops::CeilDiv(GetTotalMatmulApiMSize(baseM), baseM); // m方向需要的轮数
}

void Mc2QuantBatchMatmulInfo::Reset()
{
    initFlag = false;
    transA = false;
    transB = false;
    hasBias = false;
    mSize = 0UL;
    mSizePerNpu = 0UL;
    kSize = 0UL;
    nSize = 0UL;
    batchA = 0UL;
    batchA1 = 0UL;
    batchA2 = 0UL;
    batchA3 = 0UL;
    batchA4 = 0UL;
    batchB = 0UL;
    batchB1 = 0UL;
    batchB2 = 0UL;
    batchB3 = 0UL;
    batchB4 = 0UL;
    batchC = 0UL;
    batchC1 = 0UL;
    batchC2 = 0UL;
    batchC3 = 0UL;
    batchC4 = 0UL;
    batchBias = 0UL;
    aDtype = ge::DT_INT8;
    bDtype = ge::DT_INT8;
    cDtype = ge::DT_FLOAT16;
    biasDtype = ge::DT_INT32;
    scaleDtype = ge::DT_UINT64;
    perTokenScaleDtype = ge::DT_FLOAT;
    isPerTensor = false;
    isPerChannel = false;
    isPertoken = false;
    isDoubleScale = false;
    isMxPerGroup = false;
    isPerBlock = false;
    isPerBlockPerToken = false;
    outDtype = 0L;
    libApiWorkSpaceSize = 0U;
    bf16ExtreWorkSpaceSize = 0UL;
    groupSize = 0UL;
    groupSizeM = 0UL;
    groupSizeK = 0UL;
    groupSizeN = 0UL;
    opName = nullptr;
    aFormat = ge::FORMAT_ND;
    bFormat = ge::FORMAT_ND;
    cFormat = ge::FORMAT_ND;
}
} // namespace optiling
