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
 * \file weight_quant_batch_matmul_v2_tiling.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling.h"

#include "ops_legacy/op_tiling/op_cache_tiling.h"
#include "common/op_host/math_util.h"
#include "common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"

namespace optiling {

constexpr size_t BIAS_INDEX = 6UL;
constexpr size_t MM_SHAPE_LEN_ND = 2UL;
constexpr size_t MM_SHAPE_LEN_NZ = 4UL;
constexpr uint64_t MICROSCALING_GROUP_SIZE = 32UL;
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr uint64_t MAX_INT32 = 2147483647UL;
constexpr uint64_t INT4_IN_INT32_NUMS = 8UL;
static const std::initializer_list<ge::DataType> WEIGHT_DTYPE_LIST = {
    ge::DT_INT32, ge::DT_INT8,        ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8,
    ge::DT_INT4,  ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT};
static const std::initializer_list<ge::DataType> BIT8_WEIGHT_DTYPE_LIST = {ge::DT_INT8, ge::DT_FLOAT8_E5M2,
                                                                           ge::DT_FLOAT8_E4M3FN, ge::DT_HIFLOAT8};
static const std::initializer_list<uint64_t> GROUP_SIZE_LIST = {32, 64, 128, 256};

void GetDtype(Mc2WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context)
{
    size_t idx = 0;
    matmulInfo.aDtype = context->GetInputDesc(idx++)->GetDataType();
    matmulInfo.bDtype = context->GetInputDesc(idx++)->GetDataType();
    matmulInfo.antiQuantScaleDtype = context->GetInputDesc(idx++)->GetDataType();
    matmulInfo.cDtype = context->GetOutputDesc(0)->GetDataType();
    auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    if (biasDesc != nullptr) {
        matmulInfo.biasDtype = biasDesc->GetDataType();
    }
}

void GetAttrs(Mc2WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context)
{
    auto attrs = context->GetAttrs();
    size_t idx = 0;
    auto transposeX = attrs->GetAttrPointer<bool>(idx++);
    auto transposeWeight = attrs->GetAttrPointer<bool>(idx++);
    const int64_t *groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    if (groupSizePtr != nullptr) {
        matmulInfo.groupSize = static_cast<uint64_t>(*groupSizePtr);
    }

    idx++; // 跳过dtype属性
    const int64_t *innerPrecisePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        innerPrecisePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    if (innerPrecisePtr != nullptr) {
        // 参数校验阶段已校验innerPrecise只能为0或1， 可转为uint64_t类型
        matmulInfo.innerPrecise = static_cast<uint64_t>(*innerPrecisePtr);
    }

    matmulInfo.transA = transposeX != nullptr && *transposeX;
    matmulInfo.transB = transposeWeight != nullptr && *transposeWeight;
}

void GetInputs(Mc2WeightQuantBatchMatmulInfo &matmulInfo, const gert::TilingContext *context)
{
    size_t idx = 0;
    auto xShape = context->GetInputShape(idx++);
    auto weightShape = context->GetInputShape(idx++);
    auto antiQuantScaleShape = context->GetInputShape(idx++);
    auto antiQuantOffsetShape = context->GetOptionalInputShape(idx++);
    auto quantScaleShape = context->GetOptionalInputShape(idx++);
    auto biasShape = context->GetOptionalInputShape(BIAS_INDEX);
    matmulInfo.bFormat = Mc2GetInputStorageFormat(context, 1);
    uint64_t weightLastDim = weightShape->GetOriginShape().GetDim(1);
    if (matmulInfo.bDtype == ge::DT_INT32 || matmulInfo.bDtype == ge::DT_FLOAT) {
        weightLastDim *= INT4_IN_INT32_NUMS;
    }
    matmulInfo.hasBias = biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0;
    matmulInfo.mSize = static_cast<uint64_t>(matmulInfo.transA ? xShape->GetOriginShape().GetDim(1) :
                                                                 xShape->GetOriginShape().GetDim(0));
    matmulInfo.kSize = static_cast<uint64_t>(matmulInfo.transA ? xShape->GetOriginShape().GetDim(0) :
                                                                 xShape->GetOriginShape().GetDim(1));
    matmulInfo.nSize =
        static_cast<uint64_t>(matmulInfo.transB ? weightShape->GetOriginShape().GetDim(0) : weightLastDim);

    if (Mc2CheckOptionalInputByShape(antiQuantOffsetShape)) {
        matmulInfo.hasAntiQuantOffset = true;
    }
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    if (matmulInfo.antiQuantScaleDtype == ge::DT_FLOAT8_E8M0) {
        matmulInfo.antiQuantType = Mc2QuantType::MX;
    } else {
        if (antiQuantScaleShapeSize == 1) {
            matmulInfo.antiQuantType = Mc2QuantType::PER_TENSOR;
        } else if (matmulInfo.groupSize > 0) {
            matmulInfo.antiQuantType = Mc2QuantType::PER_GROUP;
        } else {
            matmulInfo.antiQuantType = Mc2QuantType::PER_CHANNEL;
        }
    }
    if (Mc2CheckOptionalInputByShape(quantScaleShape)) {
        size_t quantScaleShapeSize = static_cast<size_t>(quantScaleShape->GetStorageShape().GetShapeSize());
        if (quantScaleShapeSize == 0) {
            matmulInfo.quantType = Mc2QuantType::NONE;
        } else if (quantScaleShapeSize == 1) {
            matmulInfo.quantType = Mc2QuantType::PER_TENSOR;
        } else {
            matmulInfo.quantType = Mc2QuantType::PER_CHANNEL;
        }
    }
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2Tiling::GetShapeAttrsInfo()
{
    try {
        matmulInfoPtr_ = std::make_unique<Mc2WeightQuantBatchMatmulInfo>();
    } catch (const std::bad_alloc &e) {
        OP_LOGE(context_->GetNodeName(), "failed to instantiate matmul info");
        return ge::GRAPH_FAILED;
    }

    GetDtype(*matmulInfoPtr_, context_);
    GetAttrs(*matmulInfoPtr_, context_);
    GetInputs(*matmulInfoPtr_, context_);
    opName_ = context_->GetNodeName();
    // int4pack输入场景修正dtype为int4
    if (matmulInfoPtr_->bDtype == ge::DT_INT32) {
        matmulInfoPtr_->bDtype = ge::DT_INT4;
        OP_LOGI(opName_, "The conversion of weight from int32 to int4 is completed.");
    }
    // A16MxF4-NZ场景，PTA图模式下走入在线编译流程，在tiling侧将weight dtype修正为fp4
    if (matmulInfoPtr_->bDtype == ge::DT_FLOAT) {
        matmulInfoPtr_->bDtype = ge::DT_FLOAT4_E2M1;
        OP_LOGI(opName_, "The conversion of weight from fp32 to fp4 is completed.");
    }
    OP_LOGD(opName_,
            "input params: MKN[%lu, %lu, %lu], transA[%s], transB[%s], bias[%s], "
            "group size[%lu]",
            matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize,
            matmulInfoPtr_->transA ? "true" : "false", matmulInfoPtr_->transB ? "true" : "false",
            matmulInfoPtr_->hasBias ? "true" : "false", matmulInfoPtr_->groupSize);
    return ge::GRAPH_SUCCESS;
}

void Mc2WeightQuantBatchMatmulV2Tiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "platformInfo");
        return;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    try {
        compileInfoPtr_ = std::make_unique<Mc2WeightQuantBatchMatmulV2CompileInfo>();
    } catch (const std::bad_alloc &e) {
        OP_LOGE(context_->GetNodeName(), "failed to instantiate compile info");
        return;
    }

    compileInfoPtr_->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr_->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr_->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr_->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr_->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr_->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr_->ubSize);
    compileInfoPtr_->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr_->socVersion = ascendcPlatform.GetSocVersion();
    std::string mmad;
    bool res = platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", mmad);
    compileInfoPtr_->supportMmadS8S4 = res && mmad.find("s8s4") != std::string::npos;

    TilingPrepareForOpCache(context_);
    OP_LOGD(context_->GetNodeName(), "MatmulAllReduce Init Quant Tiling Compile Info Success");
}

void Mc2WeightQuantBatchMatmulV2Tiling::SetCommonTilingKeyElement(Mc2TilingKeyConfigure &tilingKeyConfigure) const
{
    tilingKeyConfigure.socVersionType =
        static_cast<uint8_t>(Mc2SocVersionType::SUPPORT_L0C_TO_OUT) * 10; // 10:乘10第0位
    tilingKeyConfigure.quantizationScenario = static_cast<uint8_t>(Mc2QuantizationScenario::DEFAULT);
    tilingKeyConfigure.transposeSituation =
        (static_cast<uint16_t>(matmulInfoPtr_->transA) << 1) | static_cast<uint16_t>(matmulInfoPtr_->transB);
    tilingKeyConfigure.antiquantType = static_cast<uint8_t>(matmulInfoPtr_->antiQuantType);
    tilingKeyConfigure.quantType = static_cast<uint8_t>(Mc2QuantType::NONE);
    tilingKeyConfigure.optionInputSituation = (static_cast<uint16_t>(matmulInfoPtr_->hasAntiQuantOffset) << 1);
    tilingKeyConfigure.weightFormat = static_cast<uint8_t>(
        matmulInfoPtr_->bFormat == ge::FORMAT_ND ? Mc2WeightFormat::ND : Mc2WeightFormat::FRACTAL_NZ);
    tilingKeyConfigure.apiConstexpr = 0;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2Tiling::GetPlatformInfo()
{
    auto compileInfoPtr =
        compileInfoPtr_ ? compileInfoPtr_.get() :
                          reinterpret_cast<const Mc2WeightQuantBatchMatmulV2CompileInfo *>(context_->GetCompileInfo());
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context_->GetNodeName(), "compileInfoPtr is null");

    if (compileInfoPtr_ == nullptr) {
        compileInfoPtr_ = std::unique_ptr<Mc2WeightQuantBatchMatmulV2CompileInfo>(
            new (std::nothrow) Mc2WeightQuantBatchMatmulV2CompileInfo());
        OP_LOGE_IF(compileInfoPtr_ == nullptr, ge::GRAPH_FAILED, opName_, "compileInfoPtr_ is null");
        compileInfoPtr_->ubSize = compileInfoPtr->ubSize;
        compileInfoPtr_->l1Size = compileInfoPtr->l1Size;
        compileInfoPtr_->l0cSize = compileInfoPtr->l0cSize;
        compileInfoPtr_->l0aSize = compileInfoPtr->l0aSize;
        compileInfoPtr_->l0bSize = compileInfoPtr->l0bSize;
        compileInfoPtr_->workspaceNum = compileInfoPtr->workspaceNum;
        compileInfoPtr_->aivNum = compileInfoPtr->aivNum;
        compileInfoPtr_->aicNum = compileInfoPtr->aicNum;
        compileInfoPtr_->socVersion = compileInfoPtr->socVersion;
        compileInfoPtr_->supportMmadS8S4 = compileInfoPtr->supportMmadS8S4;
    }

    aicoreParams_.numBlocks = 0;
    aicoreParams_.aicNum = compileInfoPtr->aicNum;
    aicoreParams_.ubSize = compileInfoPtr->ubSize;
    aicoreParams_.l1Size = compileInfoPtr->l1Size;
    aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
    aicoreParams_.l0aSize = compileInfoPtr->l0aSize;
    aicoreParams_.l0bSize = compileInfoPtr->l0bSize;

    OP_TILING_CHECK(
        compileInfoPtr->aivNum == 0 || compileInfoPtr->aicNum == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "coreNum",
            (std::to_string(compileInfoPtr->aivNum) + ", " + std::to_string(compileInfoPtr->aicNum)).c_str(),
            "The value of coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);

    OP_LOGI(opName_,
            "get platform: aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) "
            "l0cSize(%lu)  l0aSize(%lu) l0bSize(%lu)",
            compileInfoPtr->aivNum, compileInfoPtr->aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size,
            aicoreParams_.l0cSize, aicoreParams_.l0aSize, aicoreParams_.l0bSize);

    return ge::GRAPH_SUCCESS;
}

bool CheckInputShape(Mc2WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *xShape,
                     const gert::StorageShape *weightShape)
{
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t xOriDimNum = xShape->GetOriginShape().GetDimNum();
    size_t weigthDimNum = weightShape->GetStorageShape().GetDimNum();
    size_t weightOriDimNum = weightShape->GetOriginShape().GetDimNum();
    OP_TILING_CHECK(xOriDimNum != MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams->opName, "x",
                                                             (std::to_string(xOriDimNum) + "D").c_str(),
                                                             "The shape dim of x must be 2D."),
                    return false);
    OP_TILING_CHECK(xDimNum != MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams->opName, "x storage",
                                                             (std::to_string(xDimNum) + "D").c_str(),
                                                             "The shape dim of x storage must be 2D."),
                    return false);
    OP_TILING_CHECK(weightOriDimNum != MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams->opName, "weight original",
                                                             (std::to_string(weightOriDimNum) + "D").c_str(),
                                                             "The shape dim of weight original must be 2D."),
                    return false);
    if (inputParams->bFormat != ge::FORMAT_FRACTAL_NZ) {
        OP_TILING_CHECK(weigthDimNum != MM_SHAPE_LEN_ND,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            inputParams->opName, "weight storage", (std::to_string(weigthDimNum) + "D").c_str(),
                            "The shape dim of weight storage must be 2D in ND format."),
                        return false);
    } else {
        OP_TILING_CHECK(weigthDimNum != MM_SHAPE_LEN_NZ,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                            inputParams->opName, "weight storage", (std::to_string(weigthDimNum) + "D").c_str(),
                            "The shape dim of weight storage must be 4D in NZ format."),
                        return false);
    }
    uint64_t weightLastDim = weightShape->GetOriginShape().GetDim(1);
    if (inputParams->bDtype == ge::DT_INT32 || inputParams->bDtype == ge::DT_FLOAT) {
        weightLastDim *= INT4_IN_INT32_NUMS;
    }
    inputParams->mSize = static_cast<uint64_t>(inputParams->transA ? xShape->GetOriginShape().GetDim(1) :
                                                                     xShape->GetOriginShape().GetDim(0));
    inputParams->kSize = static_cast<uint64_t>(inputParams->transA ? xShape->GetOriginShape().GetDim(0) :
                                                                     xShape->GetOriginShape().GetDim(1));
    inputParams->nSize =
        static_cast<uint64_t>(inputParams->transB ? weightShape->GetOriginShape().GetDim(0) : weightLastDim);
    auto kBSize = static_cast<uint64_t>(inputParams->transB ? weightLastDim : weightShape->GetOriginShape().GetDim(0));
    OP_TILING_CHECK(inputParams->kSize != kBSize,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams->opName, "K dim",
                        (std::to_string(inputParams->kSize) + " vs " + std::to_string(kBSize)).c_str(),
                        "The value of K dim of x and weight must be equal."),
                    return false);
    if (inputParams->bDtype == ge::DT_INT4 || inputParams->bDtype == ge::DT_INT32 ||
        inputParams->bDtype == ge::DT_FLOAT4_E2M1 || inputParams->bDtype == ge::DT_FLOAT4_E1M2 ||
        inputParams->bDtype == ge::DT_FLOAT) {
        uint64_t innerAxisDim = static_cast<uint64_t>(weightLastDim);
        OP_TILING_CHECK((innerAxisDim & 1) != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams->opName, "weight inner dim", std::to_string(innerAxisDim).c_str(),
                            "The value of weight inner dim must be even when dtype is int4/float4."),
                        return false);
    }
    return true;
}

bool CheckAntiQuantScaleShape(Mc2WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *antiQuantScaleShape)
{
    size_t antiQuantScaleDimNum = antiQuantScaleShape->GetStorageShape().GetDimNum();
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(antiQuantScaleDimNum > MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        inputParams->opName, "antiquant scale", (std::to_string(antiQuantScaleDimNum) + "D").c_str(),
                        "The shape dim of antiquant scale must not be more than 2D."),
                    return false);
    if (antiQuantScaleShapeSize != 1) {
        if (antiQuantScaleDimNum == MM_SHAPE_LEN_ND) {
            gert::Shape expectShape;
            uint64_t kNum = inputParams->groupSize > 0 ? ops::CeilDiv(inputParams->kSize, inputParams->groupSize) : 1;
            if (inputParams->transB) {
                expectShape.AppendDim(static_cast<int64_t>(inputParams->nSize));
                expectShape.AppendDim(static_cast<int64_t>(kNum));
            } else {
                expectShape.AppendDim(static_cast<int64_t>(kNum));
                expectShape.AppendDim(static_cast<int64_t>(inputParams->nSize));
            }
            OP_TILING_CHECK(
                expectShape != antiQuantScaleShape->GetStorageShape(),
                OP_LOGE_FOR_INVALID_SHAPE(inputParams->opName, "antiquant",
                                          Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str(),
                                          Ops::Base::ToString(expectShape).c_str()),
                return false);
            if (inputParams->antiQuantScaleDtype == ge::DT_FLOAT8_E8M0) {
                inputParams->antiQuantType = Mc2QuantType::MX;
            } else if (inputParams->groupSize > 0) {
                inputParams->antiQuantType = Mc2QuantType::PER_GROUP;
            } else {
                inputParams->antiQuantType = Mc2QuantType::PER_CHANNEL;
            }
        } else {
            OP_TILING_CHECK(antiQuantScaleShapeSize != inputParams->nSize,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                inputParams->opName, "antiquant scale",
                                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str(),
                                (std::string("The shape of antiquant scale must be ") +
                                 std::to_string(inputParams->nSize) + " when perchannel.")
                                    .c_str()),
                            return false);
            inputParams->antiQuantType = Mc2QuantType::PER_CHANNEL;
        }
    } else {
        inputParams->antiQuantType = Mc2QuantType::PER_TENSOR;
    }
    return true;
}

bool CheckAntiQuantOffsetShape(Mc2WeightQuantBatchMatmulInfo *inputParams,
                               const gert::StorageShape *antiQuantScaleShape,
                               const gert::StorageShape *antiQuantOffsetShape)
{
    if (Mc2CheckOptionalInputByShape(antiQuantOffsetShape)) {
        inputParams->hasAntiQuantOffset = true;
    }
    if (inputParams->hasAntiQuantOffset) {
        OP_TILING_CHECK(
            inputParams->bDtype == ge::DT_FLOAT4_E2M1 || inputParams->bDtype == ge::DT_FLOAT4_E1M2 ||
                inputParams->bDtype == ge::DT_FLOAT8_E5M2 || inputParams->bDtype == ge::DT_FLOAT8_E4M3FN ||
                inputParams->bDtype == ge::DT_HIFLOAT8 || inputParams->bDtype == ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams->opName, "bDtype", ge::TypeUtils::DataTypeToAscendString(inputParams->bDtype).GetString(),
                "The value of bDtype is not supported with AntiQuantOffset for FP32/FP4/FP8/HIFLOAT8 weights."),
            return false);
    }
    if (inputParams->hasAntiQuantOffset) {
        OP_TILING_CHECK(
            antiQuantScaleShape->GetStorageShape() != antiQuantOffsetShape->GetStorageShape(),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(inputParams->opName, "antiquant scale and offset",
                                                   (Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()) + ", " +
                                                    Ops::Base::ToString(antiQuantOffsetShape->GetStorageShape()))
                                                       .c_str(),
                                                   "The shapes of antiquant scale and offset must be the same."),
            return false);
    }
    return true;
}

bool CheckAntiQuantShape(Mc2WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *antiQuantScaleShape,
                         const gert::StorageShape *antiQuantOffsetShape)
{
    OP_TILING_CHECK(
        !CheckAntiQuantScaleShape(inputParams, antiQuantScaleShape),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "antiquant scale",
                                              Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str(),
                                              "The shape of antiquant scale must be valid."),
        return false);
    OP_TILING_CHECK(
        !CheckAntiQuantOffsetShape(inputParams, antiQuantScaleShape, antiQuantOffsetShape),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "antiquant offset",
                                              Ops::Base::ToString(antiQuantOffsetShape->GetStorageShape()).c_str(),
                                              "The shape of antiquant offset must be valid."),
        return false);
    return true;
}

bool CheckQuantShape(Mc2WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *quantScaleShape,
                     const gert::StorageShape *quantOffsetShape)
{
    if (!Mc2CheckOptionalInputByShape(quantScaleShape)) {
        if (Mc2CheckOptionalInputByShape(quantOffsetShape)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "quant offset",
                                                  Ops::Base::ToString(quantOffsetShape->GetStorageShape()).c_str(),
                                                  "The value of quant offset must exist when quant scale is provided.");
            return false;
        }
        if (inputParams->cDtype != inputParams->aDtype) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams->opName, "output y", ge::TypeUtils::DataTypeToAscendString(inputParams->cDtype).GetString(),
                (std::string("The dtype of output y must be x dtype ") +
                 ge::TypeUtils::DataTypeToAscendString(inputParams->aDtype).GetString() + " without quant scale.")
                    .c_str());
            return false;
        }
        return true;
    } else {
        if (inputParams->cDtype != ge::DT_INT8) {
            OP_LOGE_FOR_INVALID_DTYPE(inputParams->opName, "output y",
                                      ge::TypeUtils::DataTypeToAscendString(inputParams->cDtype).GetString(),
                                      "DT_INT8");
            return false;
        }
    }
    size_t quantScaleDimNum = quantScaleShape->GetStorageShape().GetDimNum();
    size_t quantScaleShapeSize = static_cast<size_t>(quantScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(quantScaleDimNum > MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams->opName, "quant scale",
                                                             (std::to_string(quantScaleDimNum) + "D").c_str(),
                                                             "The shape dim of quant scale must not be more than 2D."),
                    return false);
    if (quantScaleDimNum == MM_SHAPE_LEN_ND) {
        OP_TILING_CHECK(
            quantScaleShape->GetStorageShape().GetDim(0) != 1 ||
                static_cast<uint64_t>(quantScaleShape->GetStorageShape().GetDim(1)) != inputParams->nSize,
            OP_LOGE_FOR_INVALID_SHAPE(inputParams->opName, "quant scale",
                                      Ops::Base::ToString(quantScaleShape->GetStorageShape()).c_str(),
                                      (std::string("(1, ") + std::to_string(inputParams->nSize) + ")").c_str()),
            return false);
    } else {
        OP_TILING_CHECK(
            quantScaleShapeSize > 1 && quantScaleShapeSize != inputParams->nSize,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams->opName, "quant scale",
                                                  Ops::Base::ToString(quantScaleShape->GetStorageShape()).c_str(),
                                                  (std::string("The shape of quant scale must be ") +
                                                   std::to_string(inputParams->nSize) + " when perchannel.")
                                                      .c_str()),
            return false);
    }
    return true;
}

bool CheckBiasShape(Mc2WeightQuantBatchMatmulInfo *inputParams, const gert::StorageShape *biasShape)
{
    if (biasShape != nullptr) {
        auto biasShapeDimNum = static_cast<uint64_t>(biasShape->GetStorageShape().GetDimNum());
        if (biasShapeDimNum == 1) {
            OP_TILING_CHECK(biasShape->GetStorageShape().GetDim(0) != 0 &&
                                static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0)) != inputParams->nSize,
                            OP_LOGE_FOR_INVALID_SHAPE(
                                inputParams->opName, "bias", Ops::Base::ToString(biasShape->GetStorageShape()).c_str(),
                                (std::string("(0) or (") + std::to_string(inputParams->nSize) + ")").c_str()),
                            return false);
        } else if (biasShapeDimNum == MM_SHAPE_LEN_ND) {
            OP_TILING_CHECK(biasShape->GetStorageShape().GetDim(0) != 1 ||
                                static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(1)) != inputParams->nSize,
                            OP_LOGE_FOR_INVALID_SHAPE(
                                inputParams->opName, "bias", Ops::Base::ToString(biasShape->GetStorageShape()).c_str(),
                                (std::string("(1, ") + std::to_string(inputParams->nSize) + ")").c_str()),
                            return false);
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams->opName, "bias",
                                                     (std::to_string(biasShapeDimNum) + "D").c_str(),
                                                     "The shape dim of bias must be 1D or 2D.");
            return false;
        }
        return true;
    }
    return true;
}

bool CheckShapeDims(Mc2WeightQuantBatchMatmulInfo *inputParams, NpuArch npuArch)
{
    OP_TILING_CHECK((npuArch != NpuArch::DAV_3510) &&
                        (inputParams->kSize > MAX_SHAPE_DIM || inputParams->nSize > MAX_SHAPE_DIM),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams->opName, "k/n dims",
                        (std::to_string(inputParams->kSize) + ", " + std::to_string(inputParams->nSize)).c_str(),
                        "The value of k/n dims must not be more than 65535."),
                    return false);
    uint64_t batchMax = inputParams->transA ? MAX_SHAPE_DIM : MAX_INT32;
    OP_TILING_CHECK((npuArch != NpuArch::DAV_3510) && (inputParams->mSize > batchMax),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams->opName, "m", std::to_string(inputParams->mSize).c_str(),
                        (std::string("The value of m must not be more than ") + std::to_string(batchMax)).c_str()),
                    return false);
    OP_TILING_CHECK(inputParams->groupSize >= inputParams->kSize || inputParams->groupSize % MIN_GROUP_SIZE != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams->opName, "groupSize", std::to_string(inputParams->groupSize).c_str(),
                        (std::string("The value of groupSize must not be more than ") +
                         std::to_string(inputParams->kSize) + " and align to 32.")
                            .c_str()),
                    return false);
    OP_TILING_CHECK(
        inputParams->antiQuantScaleDtype == ge::DT_FLOAT8_E8M0 && inputParams->groupSize != MICROSCALING_GROUP_SIZE,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "groupSize/MX",
                                              std::to_string(inputParams->groupSize).c_str(),
                                              "The value of groupSize must be 32 in microscaling formats context."),
        return false);

    return true;
}

/*
The function is check the shape limit:
    1. the input x shape dims should be 2, weight shape dims should be 2(ND) or 4(NZ)
    2. the k of x and weight must be same; the inner axis must be even when weight is int4
    3. antiquant shape must be:
        per_group:  not trans_b: (ceil(k, group_size), n); trans_b: (n, ceil(k, group_size))
        per_channel: not trans_n: (1, n) or (n); trans_b: (n, 1) or (n)
        per_tensor: (1,)
    4. quant shape must be (1,), (1, n), (n) or empty tensor
    5. bias shape must be (n,), (1, n) or empty tensor
    6. nk must <= 65535, m <= 65535(trans_a) or int32_max(not trans_a);
    7. group_size < k, align to 32
*/
bool CheckShape(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams,
                platform_ascendc::SocVersion socVersion)
{
    size_t idx = 0;
    auto xShape = context->GetInputShape(idx++);
    auto weightShape = context->GetInputShape(idx++);
    auto antiQuantScaleShape = context->GetInputShape(idx++);
    auto antiQuantOffsetShape = context->GetOptionalInputShape(idx++);
    auto quantScaleShape = context->GetOptionalInputShape(idx++);
    auto quantOffsetShape = context->GetOptionalInputShape(idx++);
    auto biasShape = context->GetOptionalInputShape(idx++);
    auto outputShape = context->GetOutputShape(0);

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();

    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, antiQuantScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    // not yet support empty tensor for input
    OP_TILING_CHECK(xShape->GetStorageShape().GetShapeSize() == 0 ||
                        weightShape->GetStorageShape().GetShapeSize() == 0 ||
                        antiQuantScaleShape->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "inputs", "empty",
                                                          "The value of inputs must not be an empty tensor."),
                    return false);
    inputParams->bFormat = Mc2GetInputStorageFormat(context, 1);
    OP_TILING_CHECK(inputParams->bFormat == ge::FORMAT_NULL,
                    OP_LOGE_FOR_INVALID_FORMAT(inputParams->opName, "weight", "FORMAT_NULL", "FRACTAL_NZ or ND"),
                    return false);
    OP_TILING_CHECK(!CheckInputShape(inputParams, xShape, weightShape),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "x/weight shape",
                                                          (Ops::Base::ToString(xShape->GetStorageShape()) + ", " +
                                                           Ops::Base::ToString(weightShape->GetStorageShape()))
                                                              .c_str(),
                                                          "The shape of x/weight must be valid."),
                    return false);
    OP_TILING_CHECK(
        !CheckAntiQuantShape(inputParams, antiQuantScaleShape, antiQuantOffsetShape),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "antiquant shape",
                                              Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str(),
                                              "The shape of antiquant must be valid."),
        return false);
    OP_TILING_CHECK(
        !CheckQuantShape(inputParams, quantScaleShape, quantOffsetShape),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "quant shape",
                                              Ops::Base::ToString(quantScaleShape->GetStorageShape()).c_str(),
                                              "The shape of quant must be valid."),
        return false);
    OP_TILING_CHECK(!CheckBiasShape(inputParams, biasShape),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "bias shape",
                                                          Ops::Base::ToString(biasShape->GetStorageShape()).c_str(),
                                                          "The shape of bias must be valid."),
                    return false);
    OP_TILING_CHECK(!CheckShapeDims(inputParams, npuArch),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "shape dims",
                                                          Ops::Base::ToString(xShape->GetStorageShape()).c_str(),
                                                          "The shape dims must be valid."),
                    return false);
    return true;
}

bool CheckInputDtype(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams,
                     platform_ascendc::SocVersion socVersion)
{
    OP_TILING_CHECK(inputParams->aDtype != ge::DT_FLOAT16 && inputParams->aDtype != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams->opName, "x",
                                              ge::TypeUtils::DataTypeToAscendString(inputParams->aDtype).GetString(),
                                              "DT_FLOAT16 or DT_BF16"),
                    return false);
    OP_TILING_CHECK(
        std::find(WEIGHT_DTYPE_LIST.begin(), WEIGHT_DTYPE_LIST.end(), inputParams->bDtype) == WEIGHT_DTYPE_LIST.end(),
        OP_LOGE_FOR_INVALID_DTYPE(inputParams->opName, "weight",
                                  ge::TypeUtils::DataTypeToAscendString(inputParams->bDtype).GetString(),
                                  "INT8, FLOAT8_E5M2, FLOAT8_E4M3FN, HIFLOAT8, INT4, FLOAT4_E2M1, FLOAT4_E1M2"),
        return false);
    // the bias is the 6th input
    auto biasDesc = context->GetOptionalInputDesc(6);
    if (biasDesc != nullptr) {
        auto biasDtype = biasDesc->GetDataType();
        if (inputParams->aDtype == ge::DT_BF16) {
            if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
                OP_TILING_CHECK(biasDtype != ge::DT_FLOAT,
                                OP_LOGE_FOR_INVALID_DTYPE(inputParams->opName, "bias",
                                                          ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString(),
                                                          "DT_FLOAT"),
                                return false);
            } else {
                OP_TILING_CHECK(biasDtype != ge::DT_FLOAT && biasDtype != ge::DT_BF16,
                                OP_LOGE_FOR_INVALID_DTYPE(inputParams->opName, "bias",
                                                          ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString(),
                                                          "DT_FLOAT or DT_BF16"),
                                return false);
            }
        } else {
            OP_TILING_CHECK(biasDtype != inputParams->aDtype,
                            OP_LOGE_FOR_INVALID_DTYPE(inputParams->opName, "bias",
                                                      ge::TypeUtils::DataTypeToAscendString(biasDtype).GetString(),
                                                      "DT_FLOAT16"),
                            return false);
        }
    }
    return true;
}

bool CheckAntiQuantDtype(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams,
                         platform_ascendc::SocVersion socVersion)
{
    // the antiquant scale is the 3nd input
    auto antiQuantOffsetDesc = context->GetOptionalInputDesc(3);
    if (inputParams->antiQuantScaleDtype == ge::DT_UINT64 || inputParams->antiQuantScaleDtype == ge::DT_INT64) {
        OP_TILING_CHECK(socVersion != platform_ascendc::SocVersion::ASCEND910B,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams->opName, "antiQuantScaleDtype",
                            ge::TypeUtils::DataTypeToAscendString(inputParams->antiQuantScaleDtype).GetString(),
                            "The value of antiQuantScaleDtype must be within the supported range for int64/uint64 on "
                            "this SoC version."),
                        return false);
        OP_TILING_CHECK(
            inputParams->aDtype != ge::DT_FLOAT16,
            OP_LOGE_FOR_INVALID_DTYPE(inputParams->opName, "x",
                                      ge::TypeUtils::DataTypeToAscendString(inputParams->aDtype).GetString(),
                                      "DT_FLOAT16 when antiquant scale dtype is int64/uint64"),
            return false);
        if (antiQuantOffsetDesc != nullptr) {
            auto antiQuantOffsetDtype = antiQuantOffsetDesc->GetDataType();
            OP_TILING_CHECK(
                antiQuantOffsetDtype != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(inputParams->opName, "antiquant offset",
                                          ge::TypeUtils::DataTypeToAscendString(antiQuantOffsetDtype).GetString(),
                                          "DT_INT32 when scale dtype is int64/uint64"),
                return false);
        }
    } else if (inputParams->antiQuantScaleDtype == ge::DT_FLOAT8_E8M0) {
        OP_TILING_CHECK(inputParams->hasAntiQuantOffset,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams->opName, "antiQuantScaleDtype",
                            ge::TypeUtils::DataTypeToAscendString(inputParams->antiQuantScaleDtype).GetString(),
                            "The value of antiQuantScaleDtype is not supported when antiquant offset is present in "
                            "microscaling formats context."),
                        return false);
    } else if (inputParams->antiQuantScaleDtype == inputParams->aDtype) {
        if (antiQuantOffsetDesc != nullptr) {
            auto antiQuantOffsetDtype = antiQuantOffsetDesc->GetDataType();
            OP_TILING_CHECK(
                antiQuantOffsetDtype != inputParams->aDtype,
                OP_LOGE_FOR_INVALID_DTYPE(inputParams->opName, "antiquant offset",
                                          ge::TypeUtils::DataTypeToAscendString(antiQuantOffsetDtype).GetString(),
                                          (std::string("same as x dtype ") +
                                           ge::TypeUtils::DataTypeToAscendString(inputParams->aDtype).GetString())
                                              .c_str()),
                return false);
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams->opName, "antiquant scale",
            ge::TypeUtils::DataTypeToSerialString(inputParams->antiQuantScaleDtype).c_str(),
            (std::string("The dtype of antiquant scale must be int64/uint64 or the same as x dtype ") +
             ge::TypeUtils::DataTypeToSerialString(inputParams->aDtype).c_str())
                .c_str());
        return false;
    }
    return true;
}

bool CheckQuantDtype(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams)
{
    // the quant_scale is the 4th input
    auto quantScaleDesc = context->GetOptionalInputDesc(4);
    if (quantScaleDesc != nullptr) {
        auto quantScaleDtype = quantScaleDesc->GetDataType();
        OP_TILING_CHECK(quantScaleDtype != ge::DT_UINT64,
                        OP_LOGE_FOR_INVALID_DTYPE(inputParams->opName, "quant scale",
                                                  ge::TypeUtils::DataTypeToAscendString(quantScaleDtype).GetString(),
                                                  "DT_UINT64"),
                        return false);
    } else {
        OP_TILING_CHECK(inputParams->cDtype != inputParams->aDtype,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams->opName, "output y",
                            ge::TypeUtils::DataTypeToAscendString(inputParams->cDtype).GetString(),
                            (std::string("The dtype of output y must be the same as x dtype ") +
                             ge::TypeUtils::DataTypeToAscendString(inputParams->aDtype).GetString() + " without quant.")
                                .c_str()),
                        return false);
    }
    return true;
}

/*
The function is check the dtype limit:
    1. Input x dtype should be fp16 or bf16, weight should be int8 or int4
    2. Output dtype should be same with x dtype without quant
    3. bias dtype should be fp16 when x dtype is fp16, fp32 when x dype is bf16
    4. antiquant scale dtype should be x dtype or int64/uint64, antiquant offset dtype should be x dtype or int32;
    5. quant scale dtype should be uint64
*/
bool CheckDtype(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams,
                platform_ascendc::SocVersion socVersion)
{
    size_t idx = 0;
    inputParams->aDtype = context->GetInputDesc(idx++)->GetDataType();
    inputParams->bDtype = context->GetInputDesc(idx++)->GetDataType();
    inputParams->antiQuantScaleDtype = context->GetInputDesc(idx++)->GetDataType();
    inputParams->cDtype = context->GetOutputDesc(0)->GetDataType();
    OP_TILING_CHECK(
        !CheckInputDtype(context, inputParams, socVersion),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "input dtype",
                                              ge::TypeUtils::DataTypeToAscendString(inputParams->aDtype).GetString(),
                                              "The dtype of input must be within the supported range."),
        return false);
    OP_TILING_CHECK(!CheckAntiQuantDtype(context, inputParams, socVersion),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams->opName, "antiquant dtype",
                        ge::TypeUtils::DataTypeToAscendString(inputParams->antiQuantScaleDtype).GetString(),
                        "The dtype of antiquant must be within the supported range."),
                    return false);
    OP_TILING_CHECK(
        !CheckQuantDtype(context, inputParams),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "quant dtype",
                                              ge::TypeUtils::DataTypeToAscendString(inputParams->cDtype).GetString(),
                                              "The dtype of quant must be within the supported range."),
        return false);
    return true;
}

bool CheckAttr(gert::TilingContext *context, Mc2WeightQuantBatchMatmulInfo *inputParams)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    size_t idx = 0;
    auto transposeX = attrs->GetAttrPointer<bool>(idx++);
    auto transposeWeight = attrs->GetAttrPointer<bool>(idx++);
    inputParams->transA = transposeX != nullptr && *transposeX;
    inputParams->transB = transposeWeight != nullptr && *transposeWeight;
    const int64_t *groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    // the group size not less than 0
    if (groupSizePtr != nullptr) {
        OP_TILING_CHECK(*groupSizePtr < 0,
                        OP_LOGE_FOR_INVALID_VALUE(inputParams->opName, "groupSize",
                                                  std::to_string(*groupSizePtr).c_str(), "not less than 0"),
                        return false);
        inputParams->groupSize = static_cast<uint64_t>(*groupSizePtr);
    }
    idx++; // 跳过dtype属性
    const int64_t *innerPrecisePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        innerPrecisePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    if (innerPrecisePtr != nullptr) {
        OP_TILING_CHECK((*innerPrecisePtr != 0) && (*innerPrecisePtr != 1),
                        OP_LOGE_FOR_INVALID_VALUE(inputParams->opName, "innerPrecise",
                                                  std::to_string(*innerPrecisePtr).c_str(), "0 or 1"),
                        return false);
        inputParams->innerPrecise = static_cast<uint64_t>(*innerPrecisePtr);
    }
    return true;
}

bool CheckTempLimit(Mc2WeightQuantBatchMatmulInfo *inputParams)
{
    // not support transposeA and int8 out
    OP_TILING_CHECK(inputParams->transA,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "transA", "true",
                                                          "The value of transA must be false."),
                    return false);
    OP_TILING_CHECK(inputParams->cDtype == ge::DT_INT8,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "output dtype", "DT_INT8",
                                                          "The value of output dtype must not be DT_INT8."),
                    return false);
    // shape dim of n and k must align to 32B
    int32_t alignDim = 32;
    if (std::find(BIT8_WEIGHT_DTYPE_LIST.begin(), BIT8_WEIGHT_DTYPE_LIST.end(), inputParams->bDtype) ==
        BIT8_WEIGHT_DTYPE_LIST.end()) {
        // when dtype is int4, 64 means 32B
        alignDim = 64;
    }
    // k, n not align to 32B, only not support A16W8/W4/F8/F4 pergroup
    if (inputParams->kSize % alignDim != 0 || inputParams->nSize % alignDim != 0) {
        OP_TILING_CHECK(inputParams->antiQuantType == Mc2QuantType::PER_GROUP,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams->opName, "k/n",
                            (std::to_string(inputParams->kSize) + ", " + std::to_string(inputParams->nSize)).c_str(),
                            "The value of k/n must be aligned to 32B in A16F4/W4/W8/F8 pergroup."),
                        return false);
    }

    // A16F8 only support perchannel
    if (std::find(BIT8_WEIGHT_DTYPE_LIST.begin(), BIT8_WEIGHT_DTYPE_LIST.end(), inputParams->bDtype) !=
            BIT8_WEIGHT_DTYPE_LIST.end() &&
        inputParams->bDtype != ge::DT_INT8) {
        OP_TILING_CHECK(
            inputParams->antiQuantType != Mc2QuantType::PER_CHANNEL,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "antiQuantType",
                                                  std::to_string(static_cast<int>(inputParams->antiQuantType)).c_str(),
                                                  "The value of antiQuantType must be perchannel for A16F8."),
            return false);
    }
    // A16F4 support Mx and pergroup
    if (inputParams->bDtype == ge::DT_FLOAT4_E2M1 || inputParams->bDtype == ge::DT_FLOAT4_E1M2 ||
        inputParams->bDtype == ge::DT_FLOAT) {
        OP_TILING_CHECK(
            inputParams->antiQuantType == Mc2QuantType::PER_TENSOR ||
                inputParams->antiQuantType == Mc2QuantType::PER_CHANNEL,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams->opName, "antiQuantType",
                                                  std::to_string(static_cast<int>(inputParams->antiQuantType)).c_str(),
                                                  "The value of antiQuantType must be Mx or pergroup for A16F4."),
            return false);
    }

    if (inputParams->antiQuantType == Mc2QuantType::PER_GROUP &&
        (inputParams->bDtype == ge::DT_FLOAT4_E2M1 || inputParams->bDtype == ge::DT_FLOAT4_E1M2 ||
         inputParams->bDtype == ge::DT_FLOAT)) {
        OP_TILING_CHECK(std::find(GROUP_SIZE_LIST.begin(), GROUP_SIZE_LIST.end(), inputParams->groupSize) ==
                            GROUP_SIZE_LIST.end(),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams->opName, "groupSize", std::to_string(inputParams->groupSize).c_str(),
                            "The value of groupSize must be 32, 64, 128, or 256 in A16Fp4 pergroup scenario."),
                        return false);
    }

    return true;
}

bool CheckNzSupportedScenarios(Mc2WeightQuantBatchMatmulInfo *inputParams, NpuArch npuArch)
{
    if (npuArch == NpuArch::DAV_3510) {
        // WeightNZ only support the following scenarios:
        // (1) Weight in int4 dtye with per-channel or per-group quantization without transA, transB or C8.
        // (2) Weight in fp4 dtye with per-group or MX quantization without transA, transB or C8.
        OP_TILING_CHECK(
            (inputParams->transB || inputParams->transA || inputParams->cDtype == ge::DT_INT8 ||
             !(((inputParams->antiQuantType == Mc2QuantType::PER_GROUP ||
                 inputParams->antiQuantType == Mc2QuantType::PER_CHANNEL) &&
                (inputParams->bDtype == ge::DT_INT4 || inputParams->bDtype == ge::DT_INT32)) ||
               ((inputParams->antiQuantType == Mc2QuantType::MX ||
                 inputParams->antiQuantType == Mc2QuantType::PER_GROUP) &&
                (inputParams->bDtype == ge::DT_FLOAT4_E2M1 || inputParams->bDtype == ge::DT_FLOAT4_E1M2 ||
                 inputParams->bDtype == ge::DT_FLOAT)))),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams->opName, "weightNz scenarios",
                (std::string("transA=") + std::to_string(inputParams->transA) +
                 ", transB=" + std::to_string(inputParams->transB) +
                 ", bDtype=" + std::to_string(static_cast<int>(inputParams->bDtype)) +
                 ", antiQuantType=" + std::to_string(static_cast<int>(inputParams->antiQuantType)))
                    .c_str(),
                "The value of weightNz scenarios must be S4/FP4 with non-transposed A/B, no C8, and non-per-tensor."),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK(
            (inputParams->antiQuantType == Mc2QuantType::PER_GROUP && inputParams->transA &&
             inputParams->bDtype != ge::DT_INT4) ||
                inputParams->cDtype == ge::DT_INT8 || inputParams->antiQuantType == Mc2QuantType::PER_TENSOR,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams->opName, "weightNz scenarios",
                (std::string("transA=") + std::to_string(inputParams->transA) +
                 ", bDtype=" + std::to_string(static_cast<int>(inputParams->bDtype)) +
                 ", cDtype=" + std::to_string(static_cast<int>(inputParams->cDtype)) +
                 ", antiQuantType=" + std::to_string(static_cast<int>(inputParams->antiQuantType)))
                    .c_str(),
                "The value of weightNz scenarios must not be per-group with transA int8, int8 output or per-tensor."),
            return ge::GRAPH_FAILED);
    }
    return true;
}

ge::graphStatus Mc2CheckPara(gert::TilingContext *context, platform_ascendc::SocVersion socVersion)
{
    // check Raw TilingData
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetRawTilingData()->GetData());
    // check the Required input and output desc
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));
    Mc2WeightQuantBatchMatmulInfo inputParams;
    inputParams.opName = context->GetNodeName();
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    // OP_LOG_FULL
    OPS_LOG_I(inputParams.opName, "TilingContext: %s", Ops::Transformer::DebugTilingContext(context).c_str());
    // check the input and output dtype
    OP_TILING_CHECK(!CheckDtype(context, &inputParams, socVersion),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams.opName, "input dtype",
                                                          std::to_string(static_cast<int>(inputParams.aDtype)).c_str(),
                                                          "The dtype of input must be within the supported range."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckAttr(context, &inputParams),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams.opName, "attr",
                                                          (std::string("transA=") + std::to_string(inputParams.transA) +
                                                           ", transB=" + std::to_string(inputParams.transB))
                                                              .c_str(),
                                                          "The value of attr must be within the supported range."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckShape(context, &inputParams, socVersion),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams.opName, "shape",
                                                          (std::to_string(inputParams.mSize) + "x" +
                                                           std::to_string(inputParams.kSize) + "x" +
                                                           std::to_string(inputParams.nSize))
                                                              .c_str(),
                                                          "The shape must be within the supported range."),
                    return ge::GRAPH_FAILED);
    if (inputParams.bFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_TILING_CHECK(!CheckNzSupportedScenarios(&inputParams, npuArch),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams.opName, "weightNz scenario",
                            (std::string("bFormat=") + std::to_string(static_cast<int>(inputParams.bFormat)) +
                             ", bDtype=" + std::to_string(static_cast<int>(inputParams.bDtype)) +
                             ", antiQuantType=" + std::to_string(static_cast<int>(inputParams.antiQuantType)))
                                .c_str(),
                            "The value of weightNz scenario must be supported in this scenario."),
                        return ge::GRAPH_FAILED);
    }
    if (inputParams.bDtype == ge::DT_FLOAT8_E5M2 || inputParams.bDtype == ge::DT_FLOAT8_E4M3FN ||
        inputParams.bDtype == ge::DT_HIFLOAT8) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            inputParams.opName, "weight FP8",
            (std::string("bDtype=") + std::to_string(static_cast<int>(inputParams.bDtype)) + ", transA=" +
             std::to_string(inputParams.transA) + ", cDtype=" + std::to_string(static_cast<int>(inputParams.cDtype)) +
             ", bFormat=" + std::to_string(static_cast<int>(inputParams.bFormat)))
                .c_str(),
            "The value of weight FP8 must not be used with transA, int8 output or weightNz.");
        return ge::GRAPH_FAILED;
    }
    if (npuArch == NpuArch::DAV_3510) {
        OP_TILING_CHECK(!CheckTempLimit(&inputParams),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams.opName, "input params",
                            (std::to_string(inputParams.mSize) + "x" + std::to_string(inputParams.kSize) + "x" +
                             std::to_string(inputParams.nSize))
                                .c_str(),
                            "The value of input params must be within the supported range for this version."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
