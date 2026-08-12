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
 * \file weight_quant_batch_matmul_v2_checker_for_mmads8s4.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_checker_for_mmads8s4.h"
#include "common/op_host/math_util.h"
#include "common/op_host/op_tiling/debug_tiling.h"

namespace {
constexpr size_t MM_SHAPE_LEN_ND = 2UL;
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr uint64_t MAX_INT32 = 2147483647UL;
} // namespace

namespace optiling {
ge::graphStatus Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckContext()
{
    // check Raw TilingData
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    // check the Required input and output desc
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx++));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    inputParams_.opName = context_->GetNodeName();
    // OP_LOG_FULL
    OPS_LOG_I(inputParams_.opName, "TilingContext: %s", Ops::Transformer::DebugTilingContext(context_).c_str());
    return ge::GRAPH_SUCCESS;
}

/*
The function is check the dtype limit:
    1. Input x dtype should be fp16, weight should be int8
    2. Output dtype should be same with x dtype without quant
    3. antiquant scale dtype should be uint64
    4. other params should be null
*/
bool Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckDtype()
{
    size_t idx = 0;
    inputParams_.aDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.bDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.antiQuantScaleDtype = context_->GetInputDesc(idx++)->GetDataType();
    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opName, "x",
                                              ge::TypeUtils::DataTypeToAscendString(inputParams_.aDtype).GetString(),
                                              "DT_FLOAT16"),
                    return false);

    OP_TILING_CHECK(inputParams_.bDtype != ge::DT_INT8,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opName, "weight",
                                              ge::TypeUtils::DataTypeToAscendString(inputParams_.bDtype).GetString(),
                                              "DT_INT8"),
                    return false);

    OP_TILING_CHECK(
        inputParams_.antiQuantScaleDtype != ge::DT_UINT64 && inputParams_.antiQuantScaleDtype != ge::DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opName, "antiquant scale",
                                  ge::TypeUtils::DataTypeToAscendString(inputParams_.antiQuantScaleDtype).GetString(),
                                  "DT_UINT64 or DT_INT64"),
        return false);

    OP_TILING_CHECK(inputParams_.cDtype != ge::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opName, "y",
                                              ge::TypeUtils::DataTypeToAscendString(inputParams_.cDtype).GetString(),
                                              "DT_FLOAT16"),
                    return false);

    return true;
}

/*
The function is check the attr limit:
    1. transposeX and transposeWeight should be false
    2. Group size should be 0
    3. innerPrecise only support 0 or 1
*/
bool Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckAttr()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto transposeX = attrs->GetAttrPointer<bool>(idx++);
    auto transposeWeight = attrs->GetAttrPointer<bool>(idx++);
    const int64_t *groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    idx++; // 跳过dtype属性
    const int64_t *innerPrecisePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        innerPrecisePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    inputParams_.transA = transposeX != nullptr && *transposeX;
    inputParams_.transB = transposeWeight != nullptr && *transposeWeight;

    OP_TILING_CHECK(inputParams_.transA, OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "transA", "true", "false"),
                    return false);
    OP_TILING_CHECK(inputParams_.transB, OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "transB", "true", "false"),
                    return false);

    if (groupSizePtr != nullptr) {
        OP_TILING_CHECK(
            *groupSizePtr != 0,
            OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "groupSize", std::to_string(*groupSizePtr).c_str(), "0"),
            return false);
        inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    }
    if (innerPrecisePtr != nullptr) {
        OP_TILING_CHECK((*innerPrecisePtr != 0) && (*innerPrecisePtr != 1),
                        OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "innerPrecise",
                                                  std::to_string(*innerPrecisePtr).c_str(), "0 or 1"),
                        return false);
        inputParams_.innerPrecise = static_cast<uint64_t>(*innerPrecisePtr);
    }
    return true;
}

/*
The function is check the shape limit:
    1. the input x shape dims should be 2(ND), weight shape dims should be 2(ND)
    2. the k of x and weight must be same
    3. antiquant shape must be:
        per_channel: not trans_n: (1, n) or (n); trans_b: (n, 1) or (n)
        per_tensor: (1,)
    4. quant shape must be empty tensor
    5. bias shape must be empty tensor
    6. nk must <= 65535, m <= 65535(trans_a) or int32_max(not trans_a);
*/
bool Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckShape()
{
    size_t idx = 0;
    auto xShape = context_->GetInputShape(idx++);
    auto weightShape = context_->GetInputShape(idx++);
    auto antiQuantScaleShape = context_->GetInputShape(idx++);
    auto antiQuantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto quantScaleShape = context_->GetOptionalInputShape(idx++);
    auto quantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto biasShape = context_->GetOptionalInputShape(idx++);
    auto outputShape = context_->GetOutputShape(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, antiQuantScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    // not yet support empty tensor for input
    OP_TILING_CHECK(xShape->GetStorageShape().GetShapeSize() == 0 ||
                        weightShape->GetStorageShape().GetShapeSize() == 0 ||
                        antiQuantScaleShape->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "inputs", "empty",
                                                          "The value of inputs must not be an empty tensor."),
                    return false);
    // check x, weight
    inputParams_.aFormat = Mc2GetInputStorageFormat(context_, 0);
    inputParams_.bFormat = Mc2GetInputStorageFormat(context_, 1);
    OP_TILING_CHECK(!CheckInputShape(xShape, weightShape),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "x/weight shape",
                                                          (Ops::Base::ToString(xShape->GetStorageShape()) + ", " +
                                                           Ops::Base::ToString(weightShape->GetStorageShape()))
                                                              .c_str(),
                                                          "The shape of x/weight must be valid."),
                    return false);
    // check antiquant scale, antiquant offset
    OP_TILING_CHECK(
        !CheckAntiQuantShape(antiQuantScaleShape, antiQuantOffsetShape),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "antiquant shape",
                                              Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str(),
                                              "The shape of antiquant must be valid."),
        return false);
    // check quant scale, quant offset
    OP_TILING_CHECK(quantScaleShape != nullptr, OP_LOGE_WITH_INVALID_INPUT(inputParams_.opName, "quant scale"),
                    return false);
    OP_TILING_CHECK(quantOffsetShape != nullptr, OP_LOGE_WITH_INVALID_INPUT(inputParams_.opName, "quant offset"),
                    return false);
    // check bias
    OP_TILING_CHECK(biasShape != nullptr, OP_LOGE_WITH_INVALID_INPUT(inputParams_.opName, "bias"), return false);
    // check dim value
    OP_TILING_CHECK(inputParams_.kSize > MAX_SHAPE_DIM || inputParams_.nSize > MAX_SHAPE_DIM,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "k/n",
                        (std::to_string(inputParams_.kSize) + ", " + std::to_string(inputParams_.nSize)).c_str(),
                        "The value of k/n must not be more than 65535."),
                    return false);
    uint64_t batchMax = inputParams_.transA ? MAX_SHAPE_DIM : MAX_INT32;
    OP_TILING_CHECK(inputParams_.mSize > batchMax,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "m", std::to_string(inputParams_.mSize).c_str(),
                        (std::string("The value of m must not be more than ") + std::to_string(batchMax)).c_str()),
                    return false);
    OP_TILING_CHECK(inputParams_.groupSize >= inputParams_.kSize || inputParams_.groupSize % MIN_GROUP_SIZE != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "groupSize", std::to_string(inputParams_.groupSize).c_str(),
                        (std::string("The value of groupSize must not be more than ") +
                         std::to_string(inputParams_.kSize) + " and align to 32.")
                            .c_str()),
                    return false);
    return true;
}

bool Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckInputShape(const gert::StorageShape *xShape,
                                                                  const gert::StorageShape *weightShape)
{
    OP_TILING_CHECK(inputParams_.aFormat != ge::Format::FORMAT_ND && inputParams_.aFormat != ge::Format::FORMAT_NCHW,
                    OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, "x",
                                               Ops::Base::ToString(inputParams_.aFormat).c_str(), "ND, NCHW"),
                    return false);

    OP_TILING_CHECK(inputParams_.bFormat != ge::Format::FORMAT_ND && inputParams_.bFormat != ge::Format::FORMAT_NCHW,
                    OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, "weight",
                                               Ops::Base::ToString(inputParams_.bFormat).c_str(), "ND, NCHW"),
                    return false);

    size_t xOriDimNum = xShape->GetOriginShape().GetDimNum();
    size_t weightOriDimNum = weightShape->GetOriginShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t weightDimNum = weightShape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(xOriDimNum != MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "x original",
                                                             (std::to_string(xOriDimNum) + "D").c_str(),
                                                             "The shape dim of x original must be 2D."),
                    return false);

    OP_TILING_CHECK(weightOriDimNum != MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "weight original",
                                                             (std::to_string(weightOriDimNum) + "D").c_str(),
                                                             "The shape dim of weight original must be 2D."),
                    return false);

    OP_TILING_CHECK(xDimNum != MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "x storage",
                                                             (std::to_string(xDimNum) + "D").c_str(),
                                                             "The shape dim of x storage must be 2D."),
                    return false);

    OP_TILING_CHECK(weightDimNum != MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "weight storage",
                                                             (std::to_string(weightDimNum) + "D").c_str(),
                                                             "The shape dim of weight storage must be 2D."),
                    return false);
    uint64_t weightLastDim = weightShape->GetOriginShape().GetDim(1);
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? xShape->GetOriginShape().GetDim(1) :
                                                                     xShape->GetOriginShape().GetDim(0));
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? xShape->GetOriginShape().GetDim(0) :
                                                                     xShape->GetOriginShape().GetDim(1));
    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? weightLastDim : weightShape->GetOriginShape().GetDim(0));
    inputParams_.nSize =
        static_cast<uint64_t>(inputParams_.transB ? weightShape->GetOriginShape().GetDim(0) : weightLastDim);
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "K dim",
                        (std::to_string(inputParams_.kSize) + " vs " + std::to_string(kBSize)).c_str(),
                        "The value of K dim of x and weight must be equal."),
                    return false);
    return true;
}

bool Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::CheckAntiQuantShape(const gert::StorageShape *antiQuantScaleShape,
                                                                      const gert::StorageShape *antiQuantOffsetShape)
{
    size_t antiQuantScaleDimNum = antiQuantScaleShape->GetStorageShape().GetDimNum();
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    OP_TILING_CHECK(antiQuantScaleDimNum > MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        inputParams_.opName, "antiquant scale", (std::to_string(antiQuantScaleDimNum) + "D").c_str(),
                        "The shape dim of antiquant scale must not be more than 2D."),
                    return false);
    if (antiQuantScaleShapeSize != 1) {
        if (antiQuantScaleDimNum == MM_SHAPE_LEN_ND) {
            gert::Shape expectShape;
            uint64_t kNum = inputParams_.groupSize > 0 ? ops::CeilDiv(inputParams_.kSize, inputParams_.groupSize) : 1;
            if (inputParams_.transB) {
                expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
                expectShape.AppendDim(static_cast<int64_t>(kNum));
            } else {
                expectShape.AppendDim(static_cast<int64_t>(kNum));
                expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
            }
            OP_TILING_CHECK(
                expectShape != antiQuantScaleShape->GetStorageShape(),
                OP_LOGE_FOR_INVALID_SHAPE(inputParams_.opName, "antiquant scale",
                                          Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str(),
                                          Ops::Base::ToString(expectShape).c_str()),
                return false);
            inputParams_.antiQuantType =
                inputParams_.groupSize > 0 ? Mc2QuantType::PER_GROUP : Mc2QuantType::PER_CHANNEL;
        } else {
            OP_TILING_CHECK(antiQuantScaleShapeSize != inputParams_.nSize,
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                inputParams_.opName, "antiquant scale",
                                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).c_str(),
                                (std::string("The shape of antiquant scale must be ") +
                                 std::to_string(inputParams_.nSize) + " when perchannel.")
                                    .c_str()),
                            return false);
            inputParams_.antiQuantType = Mc2QuantType::PER_CHANNEL;
        }
    } else {
        inputParams_.antiQuantType = Mc2QuantType::PER_TENSOR;
    }

    OP_TILING_CHECK(antiQuantOffsetShape != nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(inputParams_.opName, "antiquant offset"), return false);
    return true;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2Checker4MmadS8S4::Check()
{
    auto ret = CheckContext();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // check the input and output dtype: x, weight, antiquant_scale, y.
    OP_TILING_CHECK(
        !CheckDtype(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "input dtype",
                                              ge::TypeUtils::DataTypeToAscendString(inputParams_.aDtype).GetString(),
                                              "The dtype of input must be within the supported range."),
        return ge::GRAPH_FAILED);
    // check attrs: transA, transB, group_size, dtype, innerPrecise
    OP_TILING_CHECK(
        !CheckAttr(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "attr",
                                              (std::string(inputParams_.transA ? "transA=true" : "transA=false") +
                                               ", " + std::string(inputParams_.transB ? "transB=true" : "transB=false"))
                                                  .c_str(),
                                              "The value of attr must be within the supported range."),
        return ge::GRAPH_FAILED);
    // check the input and output shape: all
    OP_TILING_CHECK(!CheckShape(),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "shape",
                        (std::to_string(inputParams_.mSize) + "x" + std::to_string(inputParams_.kSize) + ", " +
                         std::to_string(inputParams_.kSize) + "x" + std::to_string(inputParams_.nSize))
                            .c_str(),
                        "The shape must be valid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
