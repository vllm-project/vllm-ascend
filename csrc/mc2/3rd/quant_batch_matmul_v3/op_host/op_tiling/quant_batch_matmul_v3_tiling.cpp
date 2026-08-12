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
 * \file quant_batch_matmul_v3_tiling.cc
 * \brief
 */

#include "quant_batch_matmul_v3_tiling.h"
#include "ops_legacy/op_tiling/op_cache_tiling.h"
#include <map>
#include <numeric>

#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_type.h"
#include "graph/utils/type_utils.h"
#include "mc2_log_compat.h"
#include "register/op_impl_registry.h"
#include "arch35/adaptive_sliding_window_tiling.h"
#include "mc2_log.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/quant_batch_matmul_v3_tiling_key.h"

using AscendC::BLOCK_CUBE;   // uint32_t 16
using AscendC::ONE_BLK_SIZE; // uint32_t 32
using ge::float32_t;
using Ops::Transformer::OpTiling::TilingRegistry;

namespace {
constexpr uint64_t INT_REDUCE_FACTOR = 32;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t BIAS_THREE_DIM = 3;
constexpr uint32_t WORKSPACE_LIMIT = 50U * 1024U * 1024U; // workspaca limit 50M
constexpr int32_t BANK_LEN = 512;
constexpr int64_t LAST_AXIS_LIMIT = 65535;

// Mc2QuantBatchMatmulV3 input index, mc2 is not same
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t SCALE_INDEX = 2;
constexpr uint32_t OFFSET_INDEX = 3;
constexpr uint32_t BIAS_INDEX = 4;
constexpr uint32_t PERTOKEN_SCALE_INDEX = 5;

constexpr uint64_t BIAS_TABLE_NUM = 256;
constexpr uint64_t DATA_SIZE_FP32 = 4;
constexpr uint64_t UB_EXTRE_BYTE = 8;

const std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP = {
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16}, {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},       {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
    {ge::DT_INT4, matmul_tiling::DataType::DT_INT4},       {ge::DT_INT32, matmul_tiling::DataType::DT_INT32},
};

matmul_tiling::DataType GetMatmulTilingDtype(ge::DataType dtype)
{
    auto it = DTYPE_MAP.find(dtype);
    return it != DTYPE_MAP.end() ? it->second : matmul_tiling::DataType::DT_UNDEFINED;
}

template <typename T>
static T CalcTailSize(T num1, T num2)
{
    OP_TILING_CHECK(num2 == 0, OP_LOGE("Nil", "cannot divide by zero"), return 0);

    T mod = num1 % num2;
    if (mod != 0) {
        return mod;
    } else {
        return num2;
    }
}

std::string DType2Str(const ge::DataType dataType)
{
    std::string serialString = ge::TypeUtils::DataTypeToSerialString(dataType);
    std::string prefix = "DT_";
    size_t pos = serialString.find(prefix);
    if (pos != std::string::npos) {
        serialString.erase(pos, prefix.length());
    }
    return serialString;
}
} // namespace

namespace optiling {
Mc2QuantBatchMatmulV3Tiling::Mc2QuantBatchMatmulV3Tiling(gert::TilingContext *context)
    : Mc2QuantBatchMatmulV3TilingBase(context, false), tilingData_(tilingDataSelf_)
{
    Reset();
}

Mc2QuantBatchMatmulV3Tiling::Mc2QuantBatchMatmulV3Tiling(gert::TilingContext *context,
                                                         Mc2QuantBatchMatmulV3TilingData *out)
    : Mc2QuantBatchMatmulV3TilingBase(context, true), tilingData_(*out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
}

void Mc2QuantBatchMatmulV3Tiling::Reset()
{
    isBf16Opt_ = false;
    isUbQuant_ = false;

    if (!isTilingOut_) {
        tilingData_ = Mc2QuantBatchMatmulV3TilingData();
        OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                 0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                        OP_LOGE(inputParams_.opName, "Fail to clear tiling data"), return);
    }
}

ge::graphStatus Mc2QuantBatchMatmulV3Tiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2QuantBatchMatmulV3Tiling::GetShapeAttrsInfo()
{
    tilingDataSize_ = sizeof(Mc2QuantBatchMatmulV3TilingData);
    return Mc2QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
}

bool Mc2QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2ub() const
{
    OP_TILING_CHECK(
        inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x1 and x2",
            (std::string(DType2Str(inputParams_.aDtype)) + " and " + DType2Str(inputParams_.bDtype)).c_str(),
            "The dtype of x1 and x2 must both be INT8."),
        return false);
    OP_TILING_CHECK(inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                                                          DType2Str(inputParams_.scaleDtype).c_str(),
                                                          "The dtype of scale must be UINT64 or INT64."),
                    return false);
    OP_TILING_CHECK(inputParams_.cDtype != ge::DT_INT8 && inputParams_.cDtype != ge::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                                          DType2Str(inputParams_.cDtype).c_str(),
                                                          "The dtype of output must be INT8 or FLOAT16."),
                    return false);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr && inputParams_.biasDtype != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                                                          DType2Str(inputParams_.biasDtype).c_str(),
                                                          "The dtype of bias must be INT32."),
                    return false);
    auto x1Desc = context_->GetInputDesc(X1_INDEX);
    auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    OP_TILING_CHECK(x1Format != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, "x1",
                                               ge::TypeUtils::FormatToSerialString(x1Format).c_str(),
                                               "Input x1 format should be ND."),
                    return false);
    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(x2Format != ge::Format::FORMAT_FRACTAL_NZ,
                    OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, "x2",
                                               ge::TypeUtils::FormatToSerialString(x2Format).c_str(),
                                               "Input x2 format should be FRACTAL_NZ."),
                    return false);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) != nullptr &&
                        context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) != nullptr,
                    OP_LOGE_WITH_INVALID_INPUT(inputParams_.opName, "pertokenScale"), return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2outForSupportedList() const
{
    // x1 x2 scale bias y 合法dtype
    OP_TILING_CHECK(
        !(inputParams_.aDtype == ge::DT_INT8 || inputParams_.aDtype == ge::DT_INT4) ||
            !(inputParams_.bDtype == ge::DT_INT8 || inputParams_.bDtype == ge::DT_INT4),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            inputParams_.opName, "x1 and x2",
            (std::string(DType2Str(inputParams_.aDtype)) + " and " + DType2Str(inputParams_.bDtype)).c_str(),
            "The dtype of x1 and x2 must both be INT8 or INT4."),
        return false);
    OP_TILING_CHECK(!(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_BF16 ||
                      inputParams_.scaleDtype == ge::DT_INT64 || inputParams_.scaleDtype == ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "scale",
                                                          DType2Str(inputParams_.scaleDtype).c_str(),
                                                          "The dtype of scale must be UINT64, BF16, INT64 or FLOAT."),
                    return false);
    OP_TILING_CHECK(context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr &&
                        !(inputParams_.biasDtype == ge::DT_INT32 || inputParams_.biasDtype == ge::DT_BF16 ||
                          inputParams_.biasDtype == ge::DT_FLOAT16 || inputParams_.biasDtype == ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                                                          DType2Str(inputParams_.biasDtype).c_str(),
                                                          "The dtype of bias must be INT32, BF16, FLOAT16 or FLOAT."),
                    return false);
    OP_TILING_CHECK(!(inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16 ||
                      inputParams_.cDtype == ge::DT_BF16 || inputParams_.cDtype == ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output",
                                                          DType2Str(inputParams_.cDtype).c_str(),
                                                          "The dtype of output must be INT8, FLOAT16, BF16 or INT32."),
                    return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2outForA4W4() const
{
    // int4
    if (inputParams_.aDtype == ge::DT_INT4) {
        OP_TILING_CHECK(inputParams_.cDtype != ge::DT_FLOAT16 && inputParams_.cDtype != ge::DT_BF16,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "output", DType2Str(inputParams_.cDtype).c_str(),
                            "If the dtype of input is int4, the dtype of output must be FLOAT16 or BF16."),
                        return false);
        // a4w4场景，x1必须为ND
        auto x1Desc = context_->GetInputDesc(X1_INDEX);
        auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
        OP_TILING_CHECK(x1Format != ge::Format::FORMAT_ND,
                        OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, "x1",
                                                   ge::TypeUtils::FormatToSerialString(x1Format).c_str(),
                                                   "Input x1 format should be ND."),
                        return false);
        if (context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) == nullptr ||
            context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) == nullptr) {
            OP_TILING_CHECK(
                inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    inputParams_.opName, "scale", DType2Str(inputParams_.scaleDtype).c_str(),
                    "If the dtype of input is int4 without pertokenScale, the dtype of scale must be UINT64 or INT64."),
                return false);
            OP_TILING_CHECK(
                context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr && inputParams_.biasDtype != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    inputParams_.opName, "bias", DType2Str(inputParams_.biasDtype).c_str(),
                    "If the dtype of input is int4 without pertokenScale, the dtype of bias must be INT32."),
                return false);
        } else if (!CheckDtypeOnOnlyL0c2outForPertoken()) {
            return false;
        }
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2outForPertoken() const
{
    if (context_->GetOptionalInputDesc(PERTOKEN_SCALE_INDEX) != nullptr &&
        context_->GetOptionalInputShape(PERTOKEN_SCALE_INDEX) != nullptr) {
        // 当bias为FLOAT16,并且有pertoken时，y必须是FLOAT16
        OP_TILING_CHECK(context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr &&
                            inputParams_.biasDtype == ge::DT_FLOAT16 && inputParams_.cDtype != ge::DT_FLOAT16,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "output", DType2Str(inputParams_.cDtype).c_str(),
                            "If the dtype of bias is FLOAT16 with pertokenScale, the dtype of output must be FLOAT16."),
                        return false);
        // 有pertoken时，y必须是FLOAT16/BF16
        OP_TILING_CHECK(!(inputParams_.cDtype == ge::DT_FLOAT16 || inputParams_.cDtype == ge::DT_BF16),
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "output", DType2Str(inputParams_.cDtype).c_str(),
                            "When pertokenScale is not null, the dtype of output must be FLOAT16 or BF16."),
                        return false);
        // 当y为FLOAT16,并且有pertoken时，scale必须是FLOAT
        OP_TILING_CHECK(inputParams_.cDtype == ge::DT_FLOAT16 && inputParams_.scaleDtype != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "scale", DType2Str(inputParams_.scaleDtype).c_str(),
                            "If the dtype of output is FLOAT16 with pertokenScale, the dtype of scale must be FLOAT."),
                        return false);
    } else {
        // 当无pertoken时，bias不能是FLOAT16
        OP_TILING_CHECK(
            context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr && inputParams_.biasDtype == ge::DT_FLOAT16,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias",
                                                  DType2Str(inputParams_.biasDtype).c_str(),
                                                  "When pertokenScale is null, the dtype of bias cannot be FLOAT16."),
            return false);
        // 当bias为FLOAT,并且无pertoken时，y必须是BF16
        OP_TILING_CHECK(context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr &&
                            inputParams_.biasDtype == ge::DT_FLOAT && inputParams_.cDtype != ge::DT_BF16,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "output", DType2Str(inputParams_.cDtype).c_str(),
                            "If the dtype of bias is FLOAT without pertokenScale, the dtype of output must be BF16."),
                        return false);
        // 当y为INT8或FLOAT16,并且无pertoken时，scale不能为FLOAT
        OP_TILING_CHECK(
            (inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16) &&
                inputParams_.scaleDtype == ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "scale", DType2Str(inputParams_.scaleDtype).c_str(),
                "When output dtype is int8 or float16 without pertokenScale, scale dtype must not be float."),
            return false);
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2outForX1NZ() const
{
    // 当y为int8时，x1必须为ND
    auto x1Desc = context_->GetInputDesc(X1_INDEX);
    auto x1Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat()));
    OP_TILING_CHECK(inputParams_.cDtype == ge::DT_INT8 && x1Format != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, "x1",
                                               ge::TypeUtils::FormatToSerialString(x1Format).c_str(),
                                               "When out dtype is INT8, X1 format must be ND."),
                    return false);
    // 当x2为ND时，x1必须为ND
    auto x2Desc = context_->GetInputDesc(X2_INDEX);
    auto x2Format = static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat()));
    OP_TILING_CHECK(x2Format == ge::Format::FORMAT_ND && x1Format != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, "x1",
                                               ge::TypeUtils::FormatToSerialString(x1Format).c_str(),
                                               "When X2 format is ND, X1 format must be ND."),
                    return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2outForUnclassified() const
{
    // dtype 约束条件
    // 当bias为BF16时，y必须为BF16
    OP_TILING_CHECK(
        context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr && inputParams_.biasDtype == ge::DT_BF16 &&
            inputParams_.cDtype != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "output", DType2Str(inputParams_.cDtype).c_str(),
                                              "If the dtype of bias is BF16, the dtype of output must be BF16."),
        return false);

    // 当scale为BF16时，y必须为BF16或INT32
    OP_TILING_CHECK(inputParams_.scaleDtype == ge::DT_BF16 &&
                        !(inputParams_.cDtype == ge::DT_BF16 || inputParams_.cDtype == ge::DT_INT32),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "output", DType2Str(inputParams_.cDtype).c_str(),
                        "If the dtype of scale is BF16, the dtype of output must be BF16 or INT32."),
                    return false);
    // 当y为BF16时，scale必须为BF16或FLOAT
    OP_TILING_CHECK(inputParams_.cDtype == ge::DT_BF16 &&
                        !(inputParams_.scaleDtype == ge::DT_BF16 || inputParams_.scaleDtype == ge::DT_FLOAT),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "scale", DType2Str(inputParams_.scaleDtype).c_str(),
                        "If the dtype of output is BF16, the dtype of scale must be BF16 or FLOAT."),
                    return false);
    // 当y为INT8时，scale必须为INT64或UINT64
    OP_TILING_CHECK(inputParams_.cDtype == ge::DT_INT8 &&
                        !(inputParams_.scaleDtype == ge::DT_UINT64 || inputParams_.scaleDtype == ge::DT_INT64),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "scale", DType2Str(inputParams_.scaleDtype).c_str(),
                        "If the dtype of output is INT8, the dtype of scale must be UINT64 or INT64."),
                    return false);

    // 当y为INT32时，bias必须为INT32
    OP_TILING_CHECK(
        inputParams_.cDtype == ge::DT_INT32 && context_->GetOptionalInputDesc(BIAS_INDEX) != nullptr &&
            inputParams_.biasDtype != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "bias", DType2Str(inputParams_.biasDtype).c_str(),
                                              "If the dtype of output is INT32, the dtype of bias must be INT32."),
        return false);
    // 当y为INT32时，scale必须为FLOAT或BF16
    OP_TILING_CHECK(inputParams_.cDtype == ge::DT_INT32 &&
                        !(inputParams_.scaleDtype == ge::DT_FLOAT || inputParams_.scaleDtype == ge::DT_BF16),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams_.opName, "scale", DType2Str(inputParams_.scaleDtype).c_str(),
                        "If the dtype of output is INT32, the dtype of scale must be FLOAT or BF16."),
                    return false);
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::CheckDtypeOnOnlyL0c2out() const
{
    // 对可选类型进行校验
    if (!CheckDtypeOnOnlyL0c2outForSupportedList()) {
        return false;
    }
    // 对A4W4场景/非A4W4场景进行校验
    if (inputParams_.aDtype == ge::DT_INT4) {
        if (!CheckDtypeOnOnlyL0c2outForA4W4()) {
            return false;
        }
    } else {
        if (!CheckDtypeOnOnlyL0c2outForX1NZ() || !CheckDtypeOnOnlyL0c2outForUnclassified() ||
            !CheckDtypeOnOnlyL0c2outForPertoken()) {
            return false;
        }
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::CheckShapeInRangeForOptionalInputs(const gert::StorageShape *biasShape,
                                                                     const gert::StorageShape *pertokenShape) const
{
    if (biasShape != nullptr) {
        auto biasDimNum = biasShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(!(biasDimNum == 1 || biasDimNum == BIAS_THREE_DIM),
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "bias",
                                                                 std::to_string(biasDimNum).c_str(),
                                                                 "The shape dim of bias must be 1 or 3."),
                        return false);
    }
    if (pertokenShape != nullptr) {
        auto pertokenDimNum = pertokenShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(pertokenDimNum != 1,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "pertoken",
                                                                 std::to_string(pertokenDimNum).c_str(),
                                                                 "The shape dim of pertoken must be 1."),
                        return false);
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::BiasShapeCheck(const gert::Shape &biasShape) const
{
    auto biasDimNum = biasShape.GetDimNum();
    if (biasDimNum == 1) {
        OP_TILING_CHECK(static_cast<uint64_t>(biasShape.GetDim(0)) != inputParams_.nSize,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "bias",
                                                              std::to_string(biasShape.GetDim(0)).c_str(),
                                                              "The dimension of bias must be equal to n."),
                        return false);
    }
    // 3 dim bias case
    if (biasDimNum == BIAS_THREE_DIM) {
        auto biasFirstDim = static_cast<uint64_t>(biasShape.GetDim(0));  // using index 0 to get bias first dim value
        auto biasSecondDim = static_cast<uint64_t>(biasShape.GetDim(1)); // using index 1 to get bias second dim value
        auto biasThirdDim = static_cast<uint64_t>(biasShape.GetDim(2));  // using index 2 to get bias third dim value
        OP_TILING_CHECK(biasFirstDim != inputParams_.batchC,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "bias",
                                                              std::to_string(biasFirstDim).c_str(),
                                                              "The 1st dimension of bias must be equal to batchC."),
                        return false);
        OP_TILING_CHECK(biasSecondDim != 1,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "bias",
                                                              std::to_string(biasSecondDim).c_str(),
                                                              "The 2nd dimension of bias must be 1."),
                        return false);
        OP_TILING_CHECK(biasThirdDim != inputParams_.nSize,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "bias",
                                                              std::to_string(biasThirdDim).c_str(),
                                                              "The 3rd dimension of bias must be equal to nSize."),
                        return false);
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::CheckDimValue(const gert::Shape &scaleShape, const gert::StorageShape *biasShape,
                                                const gert::StorageShape *pertokenShape,
                                                const std::vector<int64_t> &dimValueOfMKN) const
{
    auto x1Inner = dimValueOfMKN[0]; // using index 0 to get x1Inner
    auto x2Inner = dimValueOfMKN[2]; // using index 2 to get x2Inner
    auto x2Outer = dimValueOfMKN[3]; // using index 3 to get x2Outer
    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? x2Inner : x2Outer);
    auto scaleDimValue = static_cast<uint64_t>(scaleShape.GetDim(0));
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opName, "k dimension", std::to_string(inputParams_.kSize).c_str(),
                        "The k dimension of x1 must be equal to the k dimension of x2."),
                    return false);
    OP_TILING_CHECK(scaleDimValue != 1 && scaleDimValue != inputParams_.nSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "scale",
                                                          std::to_string(scaleDimValue).c_str(),
                                                          "The dimension of scale must be 1 or n."),
                    return false);
    if (biasShape != nullptr && !BiasShapeCheck(biasShape->GetStorageShape())) {
        return false;
    }
    if (inputParams_.isPertoken && !isTilingOut_) {
        auto pertoken = pertokenShape->GetStorageShape();
        OP_TILING_CHECK(static_cast<uint64_t>(pertoken.GetDim(0)) != inputParams_.mSize,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, "pertoken",
                                                              std::to_string(pertoken.GetDim(0)).c_str(),
                                                              "The shape of pertoken must be equal to m."),
                        return false);
    }
    if (inputParams_.aDtype == ge::DT_INT4) {
        // remainder by 2 to check if it is a even number
        OP_TILING_CHECK(x1Inner < 0 || x1Inner % 2 != 0 || x2Inner < 0 || x2Inner % 2 != 0,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "x1 and x2",
                            (std::to_string(x1Inner) + " and " + std::to_string(x2Inner)).c_str(),
                            "If input dtype is int4, last axis of x1 and x2 must be a positive even number."),
                        return false);
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
                                             const gert::StorageShape *biasShape,
                                             const gert::StorageShape *pertokenShape,
                                             const std::vector<int64_t> &dimValueOfMKN) const
{
    auto x1Shape = *mandtoryShape[0];    // using index 0 to get x1Shape
    auto x2Shape = *mandtoryShape[1];    // using index 1 to get x2Shape
    auto scaleShape = *mandtoryShape[2]; // using index 2 to get scaleShape

    OP_TILING_CHECK(scaleShape.GetDimNum() != 1 && scaleShape.GetDimNum() != 2,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, "scale",
                                                             std::to_string(scaleShape.GetDimNum()).c_str(),
                                                             "The shape dim of scale must be 1 or 2."),
                    return false);

    if (!CheckShapeInRangeForOptionalInputs(biasShape, pertokenShape)) {
        return false;
    }
    if (!CheckDimValue(scaleShape, biasShape, pertokenShape, dimValueOfMKN)) {
        return false;
    }
    if (!CheckShapeInBoundary(x1Shape, X1_INDEX) || !CheckShapeInBoundary(x2Shape, X2_INDEX)) {
        return false;
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::CheckDtype() const
{
    // 无芯片差异的公共校验
    OP_TILING_CHECK(
        inputParams_.aDtype != inputParams_.bDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "x1 and x2",
            (std::string(DType2Str(inputParams_.aDtype)) + " and " + DType2Str(inputParams_.bDtype)).c_str(),
            "The dtype of x1 and x2 must be the same."),
        return false);

    if (!compileInfo_.supportL0c2Out && !CheckDtypeOnOnlyL0c2ub()) {
        return false;
    } else if (compileInfo_.supportL0c2Out && !compileInfo_.supportL12BtBf16 && !CheckDtypeOnOnlyL0c2out()) {
        return false;
    }
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::CheckShapeInBoundary(const gert::Shape &shape, uint32_t shapeIdx) const
{
    int64_t mul = 1;
    int64_t mulBound = 1;
    const char *dimName = shapeIdx == X1_INDEX ? "x1" : "x2";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        int64_t curDim = shape.GetDim(i);

        OP_TILING_CHECK(i == shape.GetDimNum() - LAST_FIRST_DIM_INDEX && curDim > LAST_AXIS_LIMIT,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, dimName,
                                                              std::to_string(curDim).c_str(),
                                                              "The last dimension must not be larger than 65535."),
                        return false);

        OP_TILING_CHECK(curDim <= 0 || curDim > static_cast<int64_t>(INT32_MAX),
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(inputParams_.opName, dimName,
                                                              std::to_string(curDim).c_str(),
                                                              "The shape must be within the range [1, INT32_MAX]."),
                        return false);

        mulBound = curDim * mul;
        OP_TILING_CHECK(mulBound / curDim != mul,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            inputParams_.opName, dimName, std::to_string(mulBound).c_str(),
                            "The product of shape dimensions must be within the boundary of INT64_MAX."),
                        return false);
        mul = mulBound;
    }
    return true;
}

void Mc2QuantBatchMatmulV3Tiling::SetQuantBatchMatmulRunParas(QuantBatchMatmulRunParas &runParams,
                                                              const optiling::Mc2QuantBatchMatmulInfo &inputParams)
{
    runParams.mSize = inputParams.mSize;
    runParams.kSize = inputParams.kSize;
    runParams.nSize = inputParams.nSize;
    runParams.transA = inputParams.transA;
    runParams.transB = inputParams.transB;
    runParams.hasBias = inputParams.hasBias;
    runParams.isPertoken = inputParams.isPertoken;
    runParams.aFormat = inputParams.aFormat;
    runParams.bFormat = inputParams.bFormat;
    runParams.aDtype = inputParams.aDtype;
    runParams.bDtype = inputParams.bDtype;
    runParams.cDtype = inputParams.cDtype;
    runParams.batchC = inputParams.batchC;
}

void Mc2QuantBatchMatmulV3Tiling::ProcessMSmall()
{
    QuantBatchMatmulRunParas runParams_;
    SetQuantBatchMatmulRunParas(runParams_, inputParams_);
    bool isAllMix =
        CheckSupportConditionQbmm(QbmmType::KN, runParams_, aicoreParams_.aicNum, compileInfo_.supportL0c2Out) &&
        isUbQuant_ && inputParams_.batchC == 1;
    uint64_t baseM = static_cast<uint64_t>(tbeTiling_.m_l0) * BLOCK_CUBE;
    uint64_t baseN = static_cast<uint64_t>(tbeTiling_.n_l0) * BLOCK_CUBE;
    uint64_t needWorkspace =
        ops::CeilAlign(ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(tbeTiling_.n_dim)), baseN) * baseM *
        sizeof(int32_t) * aicoreParams_.aicNum;
    // 控制m方向无循环的进入增量优化模板且workspace不能超过50M限制。
    bool isPertoken = (needWorkspace < WORKSPACE_LIMIT) && inputParams_.isPertoken;
    bool isDecode = inputParams_.mSize <= baseM;
    isBf16Opt_ = isDecode && (isAllMix || isPertoken);
    isBf16Opt_ = isBf16Opt_ || CheckSupportConditionQbmm(QbmmType::Perchannel, runParams_, aicoreParams_.aicNum,
                                                         compileInfo_.supportL0c2Out);
    isBf16Opt_ = isBf16Opt_ || (CheckSupportConditionQbmm(QbmmType::Pertoken, runParams_, aicoreParams_.aicNum,
                                                          compileInfo_.supportL0c2Out) &&
                                isPertoken);
    isBf16Opt_ = isBf16Opt_ && !isTilingOut_; // isTilingOut_表示MC2场景，不走opt模板
    if (isBf16Opt_ && inputParams_.isPertoken) {
        // cv并行，每次base块
        tbeTiling_.m_al1 = 1;
        tbeTiling_.n_bl1 = 1;
    }
}

void Mc2QuantBatchMatmulV3Tiling::UpdateSmallMTbeTiling()
{
    // 通常有24个核，但是micro batch场景可能会使用16核
    bool isValidAicNum = aicoreParams_.aicNum == 24 || aicoreParams_.aicNum == 16;
    bool is910B2or910C = compileInfo_.supportL0c2Out && isValidAicNum;
    bool isNotMatch = inputParams_.transA || inputParams_.transB || inputParams_.hasBias ||
                      inputParams_.aFormat != ge::FORMAT_ND || inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ ||
                      inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8 ||
                      inputParams_.cDtype != ge::DT_BF16 || inputParams_.batchC != 1 || !inputParams_.isPertoken;
    QuantBatchMatmulRunParas runParams_;
    SetQuantBatchMatmulRunParas(runParams_, inputParams_);
    if (!is910B2or910C ||
        (isNotMatch && !CheckSupportConditionQbmm(QbmmType::Perchannel, runParams_, aicoreParams_.aicNum,
                                                  compileInfo_.supportL0c2Out))) {
        return;
    }

    // 仅处理M不大于256且N不小于7168且K不小于4096的场景。Cube和vector的计算比为k，k小于一定值的时候，如果baseN为非128对齐
    // 影响MTE3的带宽，可能会导致vector成为瓶颈。后续充分测试后再放开。
    if (inputParams_.kSize < 4096UL || inputParams_.nSize < 7168UL || inputParams_.mSize > 256UL) {
        return;
    }

    /* 先计算baseM，尽量不切分 */
    uint64_t m_l0 = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t baseM = m_l0 * static_cast<uint64_t>(BLOCK_CUBE);
    uint64_t baseN = 128UL; // 基准baseN为128
    uint64_t baseK = 128UL; // 基准baseK取128

    auto l0aElements = aicoreParams_.l0aSize / 2UL; // 2UL表示开启double buffer，sizeof(int8) = 1
    auto l0bElements = aicoreParams_.l0aSize / 2UL; // 2UL表示开启double buffer，sizeof(int8) = 1
    if (inputParams_.mSize <= 96UL) {
        /* 根据L0C、L0A、L0B大小求解出baseN , 计算访存比最大 */
        auto l0cElements = aicoreParams_.l0cSize / 4UL;            // L0C不开启db，sizeof(int32) = 4UL
        baseN = std::min(l0cElements / baseM, l0bElements / 64UL); // l0c用满，最大化计算访存比，基准baseK取64
        baseN = baseN / 32UL * 32UL;                               // baseN向下32对齐
        uint64_t maxN = baseN * aicoreParams_.aicNum;
        if (inputParams_.nSize > maxN) {
            return; // 1轮计算不完的，L0C开启db会有收益
        }
        /* 根据L0A、L0B计算baseK */
        baseK = std::min(l0aElements / baseM, l0bElements / baseN) / 64UL * 64UL; // 使baseK向下64对齐
        baseK = std::min(baseK, 128UL);                                           // 使baseK不超过128
    } else {
        // 调整baseN使得计算的轮次最少，其次核数最大，最后是计算访存比。
        uint64_t curBaseN = 128UL;
        int64_t maxScore = -2400L;
        uint64_t maxBaseMN = 128UL * 256UL;                          // L0C可容纳 128 x 256 个int32
        while (curBaseN * baseM <= maxBaseMN && curBaseN <= 256UL) { // 限制baseN不超过256
            uint64_t blockNum = ops::CeilDiv(inputParams_.nSize, curBaseN);
            int64_t blockNumPerCore = static_cast<int64_t>(ops::CeilDiv(blockNum, aicoreParams_.aicNum));
            int64_t usedCoreNum = ops::CeilDiv(static_cast<int64_t>(blockNum), blockNumPerCore);
            int64_t computeRatio = static_cast<int64_t>((curBaseN * baseM) / (curBaseN + baseM));
            // 分别设置计算访存比、使用核数和计算轮次的重要度为7、10核100，后续可继续调优
            int64_t curScore = 7 * computeRatio + 10 * usedCoreNum - 100 * blockNumPerCore;
            if (curScore > maxScore) {
                maxScore = curScore;
                baseN = curBaseN;
            }
            curBaseN += 64UL; // baseN 64 字节对齐
        }
        baseK = 128UL; // L0A/L0B可容纳 128 x 256 个int8
    }

    UpdateSmallMTbeTiling(baseM, baseN, baseK);
}

void Mc2QuantBatchMatmulV3Tiling::UpdateSmallMTbeTiling(uint64_t baseM, uint64_t baseN, uint64_t baseK)
{
    constexpr uint64_t enableDoubleBuffer = 2UL;
    auto blockNum = ops::CeilDiv(inputParams_.nSize, baseN);
    auto blockNumPerCore = ops::CeilDiv(blockNum, aicoreParams_.aicNum);
    uint64_t l1Size = aicoreParams_.l1Size - 256UL; // 保留256UL，不用满l1Size

    uint64_t m_l0 = ops::CeilDiv(baseM, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t n_l0 = ops::CeilDiv(baseN, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t k_l0 = baseK / 32UL; // basek需要32字节对齐

    /* 仅在baseM/baseN/baseK调整的情况下重新调整其他tiling信息 */
    if ((m_l0 != static_cast<uint64_t>(tbeTiling_.m_l0)) || (n_l0 != static_cast<uint64_t>(tbeTiling_.n_l0)) ||
        (k_l0 != static_cast<uint64_t>(tbeTiling_.k_l0))) {
        tbeTiling_.m_l0 = m_l0;
        tbeTiling_.n_l0 = n_l0;
        tbeTiling_.k_l0 = k_l0;
        tbeTiling_.db_al1 = enableDoubleBuffer;
        tbeTiling_.db_bl1 = enableDoubleBuffer;
        tbeTiling_.db_l0c = 1; // L0C不开启db
        tbeTiling_.m_dim = 1;  // m方向不分核
        tbeTiling_.k_dim = 1;  // k方向不分核
        tbeTiling_.m_al1 = 1;  // 配置stepM
        tbeTiling_.n_bl1 = 1;  // 配置stepN

        /* (stepKa, stepKb)的优先级为：(8, 4) > (4, 4) > (4, 2) > (2, 4) > (2, 2) > (2, 1) > (1, 1)*/
        uint64_t stepKa = inputParams_.mSize > 96UL ? 4UL : 8UL; // M大于96时，stepKa从4开始，8会导致头开销大
        uint64_t stepKb = 4UL;                                   // 最优选的stepKb为4
        bool reduceStepKa = true;
        while ((stepKa * baseM + stepKb * baseN) * baseK * enableDoubleBuffer > l1Size) {
            if (reduceStepKa) {
                stepKa = stepKa / 2UL; // 以2的倍数递减
            } else {
                stepKb = stepKb / 2UL; // 以2的倍数递减
            }
            reduceStepKa = !reduceStepKa;
        }

        tbeTiling_.kal1_16 = stepKa * tbeTiling_.k_l0;
        tbeTiling_.kbl1_16 = stepKb * tbeTiling_.k_l0;
        tbeTiling_.n_dim = ops::CeilDiv(blockNum, blockNumPerCore);
    }
}

ge::graphStatus Mc2QuantBatchMatmulV3Tiling::DoOpTiling()
{
    isUbQuant_ = inputParams_.cDtype == ge::DT_BF16 || inputParams_.isPertoken;
    // 需要给aicoreParams_ 和libApiWorkSpaceSize赋值
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfoForTiling fail");
    if (!GetTbeTiling()) {
        OP_LOGE(inputParams_.opName, "GetTbeTiling fail");
        return ge::GRAPH_FAILED;
    }
    UpdateSmallMTbeTiling();
    PrintTbeTiling();
    ProcessMSmall();
    tilingData_.params.batchA = inputParams_.batchA;
    tilingData_.params.batchB = inputParams_.batchB;
    tilingData_.params.batchC = inputParams_.batchC;
    tilingData_.params.singleCoreBatch = ops::CeilDiv(inputParams_.batchC, static_cast<uint64_t>(tbeTiling_.batch_dim));
    tilingData_.params.biasThreeDim = static_cast<uint32_t>(inputParams_.batchBias > 1UL);
    tilingData_.params.isPerTensor = static_cast<uint32_t>(inputParams_.isPerTensor);
    tilingData_.params.isPertoken = static_cast<uint32_t>(inputParams_.isPertoken);
    tilingData_.params.biasDtype = static_cast<uint32_t>(inputParams_.biasDtype);
    if (isUbQuant_) {
        return CalcUbTiling();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2QuantBatchMatmulV3Tiling::InitTilingData(matmul_tiling::MatmulApiTilingBase &mm, bool fallback)
{
    auto aFormat =
        inputParams_.aFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto bFormat =
        inputParams_.bFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto cFormat =
        inputParams_.cFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
    auto aDtype = GetMatmulTilingDtype(inputParams_.aDtype);
    auto bDtype = GetMatmulTilingDtype(inputParams_.bDtype);
    auto cDtype = GetMatmulTilingDtype(inputParams_.cDtype);
    OP_LOGE_IF(aDtype == matmul_tiling::DataType::DT_UNDEFINED, ge::GRAPH_FAILED, inputParams_.opName,
               "invalid A dtype %s", DType2Str(inputParams_.aDtype).c_str());
    OP_LOGE_IF(bDtype == matmul_tiling::DataType::DT_UNDEFINED, ge::GRAPH_FAILED, inputParams_.opName,
               "invalid B dtype %s", DType2Str(inputParams_.bDtype).c_str());
    OP_LOGE_IF(cDtype == matmul_tiling::DataType::DT_UNDEFINED, ge::GRAPH_FAILED, inputParams_.opName,
               "invalid C dtype %s", DType2Str(inputParams_.cDtype).c_str());
    mm.SetAType(matmul_tiling::TPosition::GM, aFormat, aDtype, inputParams_.transA);
    mm.SetBType(matmul_tiling::TPosition::GM, bFormat, bDtype, inputParams_.transB);
    mm.SetCType(matmul_tiling::TPosition::GM, cFormat, cDtype);
    mm.SetShape(inputParams_.mSize, inputParams_.nSize, inputParams_.kSize);
    if (fallback) {
        mm.SetShape(ops::CeilDiv<uint64_t>(inputParams_.mSize, tbeTiling_.m_dim),
                    ops::CeilDiv<uint64_t>(inputParams_.nSize, tbeTiling_.n_dim),
                    ops::CeilDiv<uint64_t>(inputParams_.kSize, tbeTiling_.k_dim));
        mm.SetMatmulConfigParams(1, false, matmul_tiling::ScheduleType::INNER_PRODUCT,
                                 matmul_tiling::MatrixTraverse::NOSET, true);
    }
    mm.SetOrgShape(inputParams_.mSize, inputParams_.nSize, inputParams_.kSize);
    if (inputParams_.hasBias) {
        mm.SetBias(true);
        auto biasDtype = GetMatmulTilingDtype(inputParams_.biasDtype);
        OP_LOGE_IF(biasDtype == matmul_tiling::DataType::DT_UNDEFINED, ge::GRAPH_FAILED, inputParams_.opName,
                   "invalid bias dtype %s", DType2Str(inputParams_.biasDtype).c_str());
        mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, biasDtype);
    }
    mm.SetBufferSpace(compileInfo_.l1Size, compileInfo_.l0cSize, compileInfo_.ubSize);
    if (mm.GetTiling(tilingData_.matmulTiling) == -1) {
        OP_LOGE(inputParams_.opName, "Quant Mc2MatmulV3 Get Tiling Failed!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void Mc2QuantBatchMatmulV3Tiling::ConstructCacheParams(BatchmatmulCompileParas &compileParams,
                                                       BatchmatmulRunParas &runParams) const
{
    compileParams.binary_mode_flag = true;
    compileParams.bias_flag = inputParams_.hasBias && (inputParams_.biasDtype == ge::DT_INT32);
    compileParams.pattern_flag = compileInfo_.supportL0c2Out;
    compileParams.zero_flag = false;
    compileParams.aub_double_num = 1;
    compileParams.bub_double_num = 1;

    runParams.trans_a_flag = inputParams_.transA;
    runParams.trans_b_flag = inputParams_.transB;
    // 知识库的匹配用真实的format a和format b, 后面tiling计算全部使用FORMAT_ND
    runParams.format_a_nd = (inputParams_.aFormat == ge::FORMAT_ND);
    runParams.format_b_nd = (inputParams_.bFormat == ge::FORMAT_ND);
    runParams.format_out_nd = true;
    runParams.format_a = inputParams_.aFormat;
    runParams.format_b = inputParams_.bFormat;
    runParams.format_out = ge::FORMAT_ND;
    runParams.reserved_bool = !inputParams_.isPerTensor;
    runParams.nd_flag = runParams.format_a_nd && runParams.format_b_nd;
    runParams.use_pre_ub = runParams.nd_flag && !compileInfo_.supportL0c2Out;
    runParams.weight_nz_flag = !runParams.format_b_nd;
    runParams.batch_a1 = inputParams_.batchA1;
    runParams.batch_a2 = inputParams_.batchA2;
    runParams.batch_a3 = inputParams_.batchA3;
    runParams.batch_a4 = inputParams_.batchA4;
    runParams.batch_b1 = inputParams_.batchB1;
    runParams.batch_b2 = inputParams_.batchB2;
    runParams.batch_b3 = inputParams_.batchB3;
    runParams.batch_b4 = inputParams_.batchB4;

    runParams.b_have_batch = inputParams_.batchB != 1 && inputParams_.batchC > 1;
    runParams.is_batch_matmul_mode = inputParams_.batchC > 1;
    runParams.is_batch_matmul_op = inputParams_.batchC > 1;
    bool alignedMKN = inputParams_.mSize % BLOCK_CUBE == 0 && inputParams_.kSize % INT_REDUCE_FACTOR == 0 &&
                      inputParams_.nSize % BLOCK_CUBE == 0;
    runParams.used_aligned_pattern = alignedMKN && runParams.nd_flag;
    runParams.bias_flag = inputParams_.hasBias && (inputParams_.biasDtype == ge::DT_INT32);
    runParams.pattern_flag = compileParams.pattern_flag;
    runParams.unaligned_flag = !alignedMKN;
    runParams.zero_flag = compileParams.zero_flag;
    runParams.hf32_flag = 0;
    runParams.dtype_a = static_cast<int32_t>(inputParams_.aDtype);
    runParams.dtype_b = static_cast<int32_t>(inputParams_.bDtype);
    // scale为必选输入，因scale导致mix和纯cube场景的baseN可能不同，需要区分
    // 后面tiling计算全部使用fp16，特别的：int32场景下要用int32匹配知识库，用fp16去计算tiling
    runParams.dtype_out = static_cast<int32_t>(inputParams_.cDtype);
    runParams.dtype_bias = ge::GetSizeByDataType(inputParams_.biasDtype);
    runParams.m = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.k = ops::CeilDiv(inputParams_.kSize, INT_REDUCE_FACTOR);
    runParams.n = ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(BLOCK_CUBE));
    runParams.batch = static_cast<int64_t>(inputParams_.batchC);
    runParams.ori_shape_m = inputParams_.mSize;
    runParams.ori_shape_k = inputParams_.kSize;
    runParams.ori_shape_n = inputParams_.nSize;
    runParams.m_quant_check = inputParams_.transA;
    runParams.n_quant_check = !inputParams_.transB;
    runParams.bias_dtype = inputParams_.biasDtype;
    runParams.vector_pre_conv_mode = !inputParams_.isPerTensor && !isUbQuant_;
    runParams.is_quant_batch_matmul_v3 = true;
    runParams.is_pertoken = inputParams_.isPertoken;
    ModifyCacheParams(runParams);
}

// 当输入的M轴或N轴dim大小16对齐后超过INT32_MAX
// 需要在传入TBE Tiling前对batch轴解除多核的绑定
// 使得M/N轴上能有更多的核数参与tiling
// 避免singleCoreM/N超过INT32_MAX
void Mc2QuantBatchMatmulV3Tiling::ModifyCacheParams(BatchmatmulRunParas &runParams) const
{
    if (runParams.m * static_cast<int64_t>(BLOCK_CUBE) > static_cast<int64_t>(INT32_MAX) ||
        runParams.n * static_cast<int64_t>(BLOCK_CUBE) > static_cast<int64_t>(INT32_MAX)) {
        runParams.batch = 1;
        runParams.batch_a1 = 1;
        runParams.batch_a2 = 1;
        runParams.batch_a3 = 1;
        runParams.batch_a4 = 1;
        runParams.batch_b1 = 1;
        runParams.batch_b2 = 1;
        runParams.batch_b3 = 1;
        runParams.batch_b4 = 1;
    }
}

ge::graphStatus Mc2QuantBatchMatmulV3Tiling::DoLibApiTiling()
{
    OP_TILING_CHECK(!SetMatmulTilingFromTbeTiling(), OP_LOGE(inputParams_.opName, "Failed to get tbe tiling!"),
                    return ge::GRAPH_FAILED);
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

/**
 * 1.原则：同一版本修改tilingKey需tiling和kernel匹配
 * 2.mc2涉及tilingkey的文件(目前mc2调用量化MM的tilingKey跟随该设置方法)有：
 *  2.1 matmul_all_reduce_add_rms_norm.cpp
 *  2.2 inplace_matmul_all_reduce_add_rms_norm.cpp
 *  2.3 matmul_all_reduce.cpp
 *  2.4 quant_matmul_all_reduce_tiling.h
 * 3.如何搜索：tiling文件搜索：quant_batch_matmul_v3_tiling.h, matmul_all_reduce_tiling_base.h
 */
uint64_t Mc2QuantBatchMatmulV3Tiling::GetTilingKey(bool isBasicTiling) const
{
    uint64_t trans = (static_cast<uint64_t>(inputParams_.transA) << 1) | static_cast<uint64_t>(inputParams_.transB);
    uint64_t kernelTemplateType = (static_cast<uint64_t>(isBf16Opt_) << 1) | static_cast<uint64_t>(isBasicTiling);
    uint64_t optionAttrs = 0;
    if (inputParams_.cDtype != ge::DT_BF16) {
        optionAttrs = NeedAtomiClean();
    }
    return GET_TPL_TILING_KEY(trans, kernelTemplateType, static_cast<uint64_t>(inputParams_.isPertoken), optionAttrs);
}

uint64_t Mc2QuantBatchMatmulV3Tiling::GetTilingKey() const
{
    return GetTilingKey(false);
}

ge::graphStatus Mc2QuantBatchMatmulV3Tiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize;
    if (isUbQuant_) {
        auto ret = GetUbDequantExtreSpace();
        OP_TILING_CHECK(!ret, OP_LOGE(inputParams_.opName, "GetUbDequantExtreSpace is failed"),
                        return ge::GRAPH_FAILED);
        workspaceSize_ += inputParams_.bf16ExtreWorkSpaceSize;
    }

    if (NeedAtomiClean() && !compileInfo_.supportL0c2Out) {
        workspaceSize_ += aicoreParams_.aicNum * ONE_BLK_SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2QuantBatchMatmulV3Tiling::PostTiling()
{
    size_t tilingDataSize = sizeof(Mc2QuantBatchMatmulV3TilingData);
    OP_LOGD(inputParams_.opName, "Final tiling data size: %zu.", tilingDataSize);

    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(inputParams_.opName, "Tiling data size[%zu] is not aligned to 8.", tilingDataSize),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    auto blockDim = tilingData_.matmulTiling.usedCoreNum;
    context_->SetBlockDim(blockDim);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    if (NeedAtomiClean() && compileInfo_.supportL0c2Out) {
        context_->SetScheduleMode(
            1); // 独占全核，设置以后会让所有核空闲以后才启动，有多核同步指令需要做此设置避免影响整网其他算子
    }
    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    PrintTilingParams();
    return ge::GRAPH_SUCCESS;
}

bool Mc2QuantBatchMatmulV3Tiling::GetTbeTiling()
{
    // 在TilingParse回调函数中初始化的(编译态传过来的compile_info): GemmParseFunc->AnalyzeCompileInfo->AnalyzeExtendInfo
    BatchmatmulCompileParas compileParams;
    BatchmatmulRunParas runParams;
    ConstructCacheParams(compileParams, runParams);

    tbeTiling_.tiling_id = std::numeric_limits<uint64_t>::max();
    return GenTiling("Mc2QuantBatchMatmulV3", compileParams, runParams, tbeTiling_, context_);
}

int32_t Mc2QuantBatchMatmulV3Tiling::GetIteratorOrder()
{
    const int32_t singleCoreM = tilingData_.matmulTiling.singleCoreM;
    const int32_t singleCoreN = tilingData_.matmulTiling.singleCoreN;
    const int32_t singleCoreK = tilingData_.matmulTiling.singleCoreK;
    int32_t reduceSize = GetShapeWithDataType(ONE_BLK_SIZE, inputParams_.aDtype);
    OP_TILING_CHECK(tbeTiling_.kal1_16 * reduceSize == 0 || tbeTiling_.kbl1_16 == 0,
                    OP_LOGE(inputParams_.opName, "Tiling kal1(%ld), kbl1(%ld) or reduceSize(%d) is 0.",
                            tbeTiling_.kal1_16, tbeTiling_.kbl1_16, reduceSize),
                    return -1);
    bool fullkAL1Load = (static_cast<float>(singleCoreK) / (tbeTiling_.kal1_16 * reduceSize)) > 1.0 ? false : true;
    bool fullkBL1Load = (static_cast<float>(singleCoreK) / (tbeTiling_.kbl1_16 * reduceSize)) > 1.0 ? false : true;

    // if KAL1 and KBL1 both can not be full loaded, then select m or n which is no matter
    if (!fullkAL1Load && !fullkBL1Load) {
        return 0;
    } else if (fullkAL1Load && !fullkBL1Load) { // if KAL1 is full loaded, then select the order N fist
        return 1;
    } else if (!fullkAL1Load && fullkBL1Load) { // if KBL1 is full loaded, then select the order M fist
        return 0;
    } else {
        // if AL1LoadSize less then BL1LoadSize, then select order N first, vice versa.
        int64_t mLoop =
            ops::CeilDiv(static_cast<int64_t>(singleCoreM), tbeTiling_.m_al1 * tbeTiling_.m_l0 * BLOCK_CUBE);
        int64_t nLoop =
            ops::CeilDiv(static_cast<int64_t>(singleCoreN), tbeTiling_.n_bl1 * tbeTiling_.n_l0 * BLOCK_CUBE);
        int64_t aL1LoadSize = singleCoreM + singleCoreN * mLoop;
        int64_t bL1LoadSize = singleCoreN + singleCoreM * nLoop;
        return aL1LoadSize < bL1LoadSize ? 1 : 0;
    }
}

bool Mc2QuantBatchMatmulV3Tiling::SetBlockDimsAndSingleCore(AscendC::tiling::TCubeTiling &mt)
{
    auto mFactor = (inputParams_.transA && inputParams_.mSize >= static_cast<uint64_t>(mt.baseM) ?
                        static_cast<uint64_t>(mt.baseM) :
                        BLOCK_CUBE);
    auto nFactor = (!inputParams_.transB && inputParams_.nSize >= static_cast<uint64_t>(mt.baseN) ?
                        static_cast<uint64_t>(mt.baseN) :
                        BLOCK_CUBE);
    auto singleCoreM =
        ops::CeilDiv(ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(tbeTiling_.m_dim)), mFactor) * mFactor;
    auto singleCoreN =
        ops::CeilDiv(ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(tbeTiling_.n_dim)), nFactor) * nFactor;
    auto singleCoreK = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(tbeTiling_.k_dim));
    singleCoreK =
        tbeTiling_.k_dim == 1 ? singleCoreK : ops::CeilAlign(singleCoreK, static_cast<uint64_t>(ONE_BLK_SIZE));
    OP_TILING_CHECK(
        singleCoreM > static_cast<uint64_t>(std::numeric_limits<int>::max()),
        OP_LOGE(inputParams_.opName, "Cache tiling inner error: singleCoreM exceeds the expression range of the int."),
        return false);

    OP_TILING_CHECK(
        singleCoreN > static_cast<uint64_t>(std::numeric_limits<int>::max()),
        OP_LOGE(inputParams_.opName, "Cache tiling inner error: singleCoreN exceeds the expression range of the int."),
        return false);
    mt.singleCoreM = singleCoreM;
    mt.singleCoreN = singleCoreN;
    mt.singleCoreK = singleCoreK;
    if (isBf16Opt_ && inputParams_.isPertoken) {
        mt.singleCoreM = mt.baseM;
        mt.singleCoreN = mt.baseN;
    }
    if (isUbQuant_) {
        tilingData_.params.realSingleCoreM = singleCoreM;
        tilingData_.params.realSingleCoreN = singleCoreN;
    }

    auto mDim = ops::CeilDiv(inputParams_.mSize, static_cast<uint64_t>(singleCoreM));
    auto nDim = ops::CeilDiv(inputParams_.nSize, static_cast<uint64_t>(singleCoreN));
    auto batchDim = ops::CeilDiv(inputParams_.batchC, static_cast<uint64_t>(tilingData_.params.singleCoreBatch));
    auto kDim = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(tilingData_.matmulTiling.singleCoreK));
    auto blockDim = mDim * nDim * batchDim * kDim;
    OP_TILING_CHECK(
        blockDim > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
        OP_LOGE(inputParams_.opName, "Cache tiling inner error: blockDim exceeds the expression range of the int."),
        return false);
    mt.usedCoreNum = std::min(blockDim, aicoreParams_.aicNum);
    return true;
}

bool Mc2QuantBatchMatmulV3Tiling::SetMatmulTilingFromTbeTiling()
{
    auto &mt = tilingData_.matmulTiling;
    mt.M = inputParams_.mSize;
    mt.N = inputParams_.nSize;
    mt.Ka = inputParams_.kSize;
    mt.Kb = inputParams_.kSize;
    mt.baseM = tbeTiling_.m_l0 * BLOCK_CUBE;
    mt.baseN = tbeTiling_.n_l0 * BLOCK_CUBE;
    OP_TILING_CHECK(!SetBlockDimsAndSingleCore(mt),
                    OP_LOGE(inputParams_.opName, "Set usedCoreNum or singleCoreM/N failed when m(%lu) and n(%lu).",
                            inputParams_.mSize, inputParams_.nSize),
                    return false);

    int32_t reduceSize = ONE_BLK_SIZE / ge::GetSizeByDataType(inputParams_.aDtype);
    mt.baseK = tbeTiling_.k_l0 * reduceSize;

    mt.depthA1 = ops::CeilDiv(tbeTiling_.kal1_16, tbeTiling_.k_l0) * tbeTiling_.m_al1 * tbeTiling_.db_al1;
    mt.depthB1 = ops::CeilDiv(tbeTiling_.kbl1_16, tbeTiling_.k_l0) * tbeTiling_.n_bl1 * tbeTiling_.db_bl1;
    mt.stepM = tbeTiling_.m_al1;
    mt.stepN = tbeTiling_.n_bl1;
    mt.stepKa = ops::CeilDiv(tbeTiling_.kal1_16, tbeTiling_.k_l0);
    mt.stepKb = ops::CeilDiv(tbeTiling_.kbl1_16, tbeTiling_.k_l0);

    mt.isBias = inputParams_.hasBias ? 1 : 0;

    int32_t a1Length = mt.baseM * mt.baseK * ge::GetSizeByDataType(inputParams_.aDtype);
    int32_t b1Length = mt.baseN * mt.baseK * ge::GetSizeByDataType(inputParams_.bDtype);
    int32_t c1Length = mt.baseN * mt.baseM * 4; // L0C
    mt.transLength = std::max(std::max(a1Length, b1Length), c1Length);
    auto iteratorOrder = GetIteratorOrder();
    OP_TILING_CHECK(iteratorOrder < 0, OP_LOGE(inputParams_.opName, "Get iteratorOrder failed."), return false);
    mt.iterateOrder = iteratorOrder;
    mt.shareMode = 0;
    mt.dbL0A = 2; // db switch, 1: off, 2: on
    mt.dbL0B = 2; // db switch, 1: off, 2: on
    mt.dbL0C = tbeTiling_.db_l0c;

    bool fallback = false;
    OP_TILING_CHECK(!CalcUsedL1AndUBSize(a1Length * mt.depthA1, b1Length * mt.depthB1, fallback),
                    OP_LOGE(inputParams_.opName, "Cache tiling inner error"), return false);
    if (!fallback) {
        OP_LOGD(inputParams_.opName, "SetMatmulTilingFromTbeTiling !fallback");
        mt.shareL0CSize = c1Length;
        mt.batchM = 1;
        mt.batchN = 1;
        mt.singleBatchM = 1;
        mt.singleBatchN = 1;
    }

    tilingData_.tileL2cacheTiling.isBasicTiling = 0U; // kernel分支控制
    return true;
}

uint32_t Mc2QuantBatchMatmulV3Tiling::GetABankConflictSize()
{
    auto &mt = tilingData_.matmulTiling;
    uint32_t ret = 0;
    if (inputParams_.transA) {
        bool isABankConflict = ops::CeilDiv<uint64_t>(mt.stepM * mt.baseM, ONE_BLK_SIZE) * 32 % 512 == 0;
        ret = isABankConflict ? mt.baseK * ONE_BLK_SIZE * mt.stepKa : 0;
    } else {
        bool isABankConflict = ops::CeilDiv<uint64_t>(mt.stepKa * mt.baseK, ONE_BLK_SIZE) * 32 % 512 == 0;
        ret = isABankConflict ? mt.baseM * ONE_BLK_SIZE * mt.stepM : 0;
    }
    return ret;
}

bool Mc2QuantBatchMatmulV3Tiling::CalcUsedL1AndUBSize(int32_t aL1Size, int32_t bL1Size, bool &fallback)
{
    auto &mt = tilingData_.matmulTiling;
    int32_t biasL1Size = (inputParams_.hasBias && (inputParams_.biasDtype == ge::DT_INT32)) ?
                             mt.baseN * ge::GetSizeByDataType(inputParams_.biasDtype) * mt.stepN :
                             0;
    uint32_t ubSize = 0;
    if (!compileInfo_.supportL0c2Out) {
        biasL1Size = 0;
        ubSize = static_cast<uint32_t>(aicoreParams_.ubSize);
        // ND/NZ trans tensor, bias and scale tensor can reuse UB, and they use tow buffer in UB at the same time
        // so set transLength as half of UB
        mt.transLength = static_cast<int32_t>(ubSize >> 1);

        while (CalcND2NZSpace() > mt.transLength) {
            OP_LOGD(inputParams_.opName, "baseM*baseK*stepKa*stepM > half of UB, decrease stepM");
            mt.stepM = mt.stepM - 1;
            mt.depthA1 = mt.stepKa * mt.stepM * tbeTiling_.db_al1;
            aL1Size = mt.baseM * mt.baseK * ge::GetSizeByDataType(inputParams_.aDtype) * mt.depthA1;
        }

        OP_TILING_CHECK(mt.stepM <= 0, OP_LOGE(inputParams_.opName, "get invalid tiling(stepM <= 0)"), return false);
        uint32_t aBankConflictSize = GetABankConflictSize();
        uint32_t rqdL1Size = aL1Size + bL1Size + aBankConflictSize * tbeTiling_.db_al1;
        fallback = rqdL1Size > compileInfo_.l1Size;
        OP_LOGD(inputParams_.opName, "aBankConflictSize: %d", aBankConflictSize);
        OP_LOGD(inputParams_.opName, "rqdL1Size: %d", rqdL1Size);
        auto usedCoreNum = mt.usedCoreNum;
        if (fallback) {
            matmul_tiling::PlatformInfo platformInfo = {
                platform_ascendc::SocVersion::ASCEND310P,    static_cast<uint64_t>(compileInfo_.l1Size),
                static_cast<uint64_t>(compileInfo_.l0cSize), static_cast<uint64_t>(compileInfo_.ubSize),
                static_cast<uint64_t>(compileInfo_.l0aSize), static_cast<uint64_t>(compileInfo_.l0bSize)};
            matmul_tiling::MatmulApiTiling mm(platformInfo);
            OP_TILING_CHECK(InitTilingData(mm, fallback) != ge::GRAPH_SUCCESS,
                            OP_LOGE(inputParams_.opName, "Init tilingdata (fallback) failed."), return false);
            tilingData_.params.ubSize = ubSize;
            mt.transLength = static_cast<int32_t>(ubSize >> 1);
            mt.usedCoreNum = usedCoreNum;
            return true;
        }
    }

    mt.shareL1Size = aL1Size + bL1Size + biasL1Size;
    mt.shareUbSize = 0;
    tilingData_.params.ubSize = ubSize;
    return true;
}

int32_t Mc2QuantBatchMatmulV3Tiling::CalcND2NZSpace() const
{
    auto &mt = tilingData_.matmulTiling;
    auto aDtypeSize = ge::GetSizeByDataType(inputParams_.aDtype);
    int32_t nd2nzSpace = mt.baseM * mt.baseK * mt.stepM * mt.stepKa * aDtypeSize;
    // ub bank confict
    if (ops::CeilAlign(mt.baseK * mt.stepKa, static_cast<int32_t>(ONE_BLK_SIZE)) % BANK_LEN == 0 &&
        ops::CeilDiv(static_cast<uint32_t>(mt.baseK * mt.stepKa), ONE_BLK_SIZE) < ONE_BLK_SIZE) {
        nd2nzSpace += mt.baseM * mt.stepM * aDtypeSize;
    }

    return nd2nzSpace;
}

void Mc2QuantBatchMatmulV3Tiling::PrintTilingData()
{
    auto &tiling = tilingData_.matmulTiling;
    std::stringstream ss;
    ss << " usedCoreNum: " << tiling.usedCoreNum << " M: " << tiling.M << " N: " << tiling.N << " Ka: " << tiling.Ka
       << " Kb: " << tiling.Kb << " singleCoreM: " << tiling.singleCoreM << " singleCoreN: " << tiling.singleCoreN
       << " singleCoreK: " << tiling.singleCoreK << " baseM: " << tiling.baseM << " baseN: " << tiling.baseN
       << " baseK: " << tiling.baseK << " depthA1: " << tiling.depthA1 << " depthB1: " << tiling.depthB1
       << " stepM: " << tiling.stepM << " stepN: " << tiling.stepN << " stepka: " << tiling.stepKa
       << " stepkb: " << tiling.stepKb << " isBias: " << tiling.isBias << " transLength: " << tiling.transLength
       << " iterateOrder: " << ((tiling.iterateOrder == 1) ? "orderM" : "orderN") << " shareMode: " << tiling.shareMode
       << " dbL0A: " << tiling.dbL0A << " dbL0B: " << tiling.dbL0B << " dbL0C: " << tiling.dbL0C
       << " usedL1Size: " << tiling.shareL1Size << " usedL0CSize: " << tiling.shareL0CSize
       << " usedUBSize: " << tiling.shareUbSize << " batchM: " << tiling.batchM << " batchN: " << tiling.batchN
       << " singleBatchM: " << tiling.singleBatchM << " singleBatchN: " << tiling.singleBatchN;
    OP_LOGD(inputParams_.opName, "%s", ss.str().c_str());
}

void Mc2QuantBatchMatmulV3Tiling::PrintTbeTiling()
{
    optiling::CacheTilingData &tiling = tbeTiling_;
    std::stringstream ss;
    ss << "tiling_id: " << tiling.tiling_id << " n_cub: " << tiling.n_cub << " db_cub: " << tiling.db_cub
       << " m_l0: " << tiling.m_l0 << " k_l0: " << tiling.k_l0 << " n_l0: " << tiling.n_l0
       << " batch_dim: " << tiling.batch_dim << " n_dim: " << tiling.n_dim << " m_dim: " << tiling.m_dim
       << " k_dim: " << tiling.k_dim << " kal1_16: " << tiling.kal1_16 << " kbl1_16: " << tiling.kbl1_16
       << " kal1_factor: " << tiling.kal1_factor << " kbl1_factor: " << tiling.kbl1_factor << " m_al1: " << tiling.m_al1
       << " n_bl1: " << tiling.n_bl1 << " db_al1: " << tiling.db_al1 << " db_bl1: " << tiling.db_bl1
       << " k_aub: " << tiling.k_aub << " m_aub: " << tiling.m_aub << " db_aub: " << tiling.db_aub
       << " k_bub: " << tiling.k_bub << " n_bub: " << tiling.n_bub << " db_bub: " << tiling.db_bub
       << " aub_dim: " << tiling.aub_dim << " bub_dim: " << tiling.bub_dim << " m1_aub: " << tiling.m1_aub
       << " n1_bub: " << tiling.n1_bub << " k1_aub: " << tiling.k1_aub << " k1_bub: " << tiling.k1_bub
       << " m_aub_dim: " << tiling.m_aub_dim << " n_bub_dim: " << tiling.n_bub_dim << " k_aub_dim: " << tiling.k_aub_dim
       << " k_bub_dim: " << tiling.k_bub_dim << " k_org_dim: " << tiling.k_org_dim << " db_l0c: " << tiling.db_l0c
       << " batch_l0: " << tiling.batch_l0 << " batch_aub: " << tiling.batch_aub << " batch_bub: " << tiling.batch_bub
       << " batch_cub: " << tiling.batch_cub << " out_branch_flag: " << tiling.out_branch_flag
       << " bias_flag: " << tiling.bias_flag << " aub_multi_flag: " << tiling.aub_multi_flag
       << " bub_multi_flag: " << tiling.bub_multi_flag << " a_align_value: " << tiling.a_align_value
       << " b_align_value: " << tiling.b_align_value << " aub_align_bound: " << tiling.aub_align_bound
       << " bub_align_bound: " << tiling.bub_align_bound << " min_kl1_cmp_kl0: " << tiling.min_kl1_cmp_kl0
       << " al1_attach_flag: " << tiling.al1_attach_flag << " bl1_attach_flag: " << tiling.bl1_attach_flag
       << " abkl1_attach_flag: " << tiling.abkl1_attach_flag << " l0c_multi_batch: " << tiling.l0c_multi_batch
       << " m_single_core: " << tiling.m_single_core << " n_single_core: " << tiling.n_single_core
       << " flag_cub_solving_bank_conflict: " << tiling.flag_cub_solving_bank_conflict
       << " al1_full_load: " << tiling.al1_full_load << " bl1_full_load: " << tiling.bl1_full_load
       << " hf32_flag: " << tiling.hf32_flag << " zero_flag: " << tiling.zero_flag
       << " datatype_bf16: " << tiling.datatype_bf16 << " deq_scale_var: " << tiling.deq_scale_var;
    OP_LOGD(inputParams_.opName, "%s", ss.str().c_str());
}

void Mc2QuantBatchMatmulV3Tiling::PrintTilingParams() const
{
    Mc2QuantBatchMatmulV3Params &params = tilingData_.params;
    std::stringstream ss;
    ss << " batchA: " << params.batchA << " batchB: " << params.batchB << " batchC: " << params.batchC
       << " singleCoreBatch: " << params.singleCoreBatch << " isPerTensor: " << params.isPerTensor
       << " isPertoken: " << params.isPertoken << " biasThreeDim: " << params.biasThreeDim
       << " ubCalcM: " << params.ubCalcM << " ubCalcN: " << params.ubCalcN << " needUbBuffer: " << params.needUbBuffer
       << " realSingleCoreM: " << params.realSingleCoreM << " realSingleCoreN: " << params.realSingleCoreN
       << " biasDtype: " << params.biasDtype << " ubSize: " << params.ubSize;
    OPS_LOG_D(inputParams_.opName, "Mc2QuantBatchMatmulV3Params params: %s", ss.str().c_str());
}

void Mc2QuantBatchMatmulV3Tiling::SpiltSingleCore(int32_t &singleCoreM, int32_t &singleCoreN)
{
    // 任意m,n方向无循环，KFC mm计算分区内不会S型计算，可以确定每次计算的起始点
    if (tilingData_.matmulTiling.baseM >= singleCoreM || tilingData_.matmulTiling.baseN >= singleCoreN) {
        return;
    }
    bool spiltM = ops::CeilDiv(singleCoreM, tilingData_.matmulTiling.baseM) <=
                  ops::CeilDiv(singleCoreN, tilingData_.matmulTiling.baseN);
    //  spilt singleCore down to baseM/N in one direction
    if (spiltM) {
        tilingData_.matmulTiling.singleCoreM = tilingData_.matmulTiling.baseM;
        singleCoreM = tilingData_.matmulTiling.baseM;
        tbeTiling_.m_al1 = 1;
        tilingData_.matmulTiling.depthA1 =
            ops::CeilDiv(tbeTiling_.kal1_16, tbeTiling_.k_l0) * tbeTiling_.m_al1 * tbeTiling_.db_al1;
        tilingData_.matmulTiling.stepM = tbeTiling_.m_al1;
    } else {
        tilingData_.matmulTiling.singleCoreN = tilingData_.matmulTiling.baseN;
        singleCoreN = tilingData_.matmulTiling.baseN;
        tbeTiling_.n_bl1 = 1;
        tilingData_.matmulTiling.depthB1 =
            ops::CeilDiv(tbeTiling_.kbl1_16, tbeTiling_.k_l0) * tbeTiling_.n_bl1 * tbeTiling_.db_bl1;
        tilingData_.matmulTiling.stepN = tbeTiling_.n_bl1;
    }
}


void Mc2QuantBatchMatmulV3Tiling::SpiltForWorkSpaceLimit(int32_t singleCoreM, int32_t singleCoreN, int32_t blockDim)
{
    int32_t maxSingleCoreM = std::max(singleCoreM, tilingData_.matmulTiling.baseM);
    int32_t maxSingleCoreN = std::max(singleCoreN, tilingData_.matmulTiling.baseN);
    if (isBf16Opt_ && inputParams_.isPertoken) {
        maxSingleCoreN = ops::CeilAlign(singleCoreN, tilingData_.matmulTiling.baseN);
    }
    inputParams_.bf16ExtreWorkSpaceSize = static_cast<uint64_t>(maxSingleCoreM) *
                                          static_cast<uint64_t>(maxSingleCoreN) * sizeof(int32_t) *
                                          static_cast<uint64_t>(blockDim);
    if (inputParams_.bf16ExtreWorkSpaceSize <= WORKSPACE_LIMIT) {
        return;
    }
    uint32_t singleCoreLimit = WORKSPACE_LIMIT / blockDim;
    uint64_t singleCoreShapeLimit = static_cast<uint64_t>(singleCoreLimit) / sizeof(int32_t);
    // N after M, minimum is baseM/baseN
    uint64_t spiltFactor = ops::CeilDiv(static_cast<uint64_t>(maxSingleCoreM) * maxSingleCoreN, singleCoreShapeLimit);
    uint64_t newSingleM =
        (ops::CeilDiv(static_cast<uint64_t>(maxSingleCoreM), static_cast<uint64_t>(BLOCK_CUBE)) / spiltFactor) *
        BLOCK_CUBE;
    newSingleM = std::max(newSingleM, static_cast<uint64_t>(tilingData_.matmulTiling.baseM));
    spiltFactor = ops::CeilDiv(static_cast<uint64_t>(newSingleM) * maxSingleCoreN, singleCoreShapeLimit);
    uint64_t newSingleN =
        static_cast<uint64_t>(ops::CeilDiv(ops::CeilDiv(maxSingleCoreN, tilingData_.matmulTiling.baseN),
                                           static_cast<int32_t>(spiltFactor))) *
        tilingData_.matmulTiling.baseN;
    while (newSingleM * newSingleN > singleCoreShapeLimit) {
        newSingleN -= tilingData_.matmulTiling.baseN;
    }
    newSingleN = std::max(newSingleN, static_cast<uint64_t>(tilingData_.matmulTiling.baseN));
    tilingData_.matmulTiling.singleCoreM = newSingleM;
    tilingData_.matmulTiling.singleCoreN = newSingleN;
    if (static_cast<uint32_t>(tilingData_.matmulTiling.baseM * tbeTiling_.m_al1) > newSingleM) {
        tbeTiling_.m_al1 = 1;
        tilingData_.matmulTiling.depthA1 =
            ops::CeilDiv(tbeTiling_.kal1_16, tbeTiling_.k_l0) * tbeTiling_.m_al1 * tbeTiling_.db_al1;
        tilingData_.matmulTiling.stepM = tbeTiling_.m_al1;
    }
    if (static_cast<uint32_t>(tilingData_.matmulTiling.baseN * tbeTiling_.n_bl1) > newSingleN) {
        tbeTiling_.n_bl1 = 1;
        tilingData_.matmulTiling.depthB1 =
            ops::CeilDiv(tbeTiling_.kbl1_16, tbeTiling_.k_l0) * tbeTiling_.n_bl1 * tbeTiling_.db_bl1;

        tilingData_.matmulTiling.stepN = tbeTiling_.n_bl1;
    }
    inputParams_.bf16ExtreWorkSpaceSize = static_cast<uint64_t>(newSingleM) * static_cast<uint64_t>(newSingleN) *
                                          sizeof(int32_t) * static_cast<uint64_t>(blockDim);
}

bool Mc2QuantBatchMatmulV3Tiling::GetUbDequantExtreSpace()
{
    int32_t singleCoreM = static_cast<int32_t>(tilingData_.params.realSingleCoreM);
    int32_t singleCoreN = static_cast<int32_t>(tilingData_.params.realSingleCoreN);

    // when M, N both have loops in singlecore/base, fixpipe order would be complex, split singlecore to
    // have only one direction with loop for now
    SpiltSingleCore(singleCoreM, singleCoreN);

    SpiltForWorkSpaceLimit(singleCoreM, singleCoreN, static_cast<int32_t>(tilingData_.matmulTiling.usedCoreNum));
    return true;
}

ge::graphStatus Mc2QuantBatchMatmulV3Tiling::CalcPertokenOptUbTiling()
{
    uint64_t ubSize = aicoreParams_.ubSize;
    uint64_t baseM = tbeTiling_.m_l0 * BLOCK_CUBE;
    uint32_t ubCalcN = static_cast<uint32_t>(tbeTiling_.n_l0) * BLOCK_CUBE;
    // input and ub out: mm out int32, ub out bf16/fp16
    uint64_t ubCalc = NUM_DB * ubCalcN * (sizeof(int32_t) + sizeof(int16_t));
    // BroadCast需要的临时空间，最小为256b，最大为：baseM * 32, baseM不会超过2048，不需要乘法溢出校验
    uint64_t needUbSize = baseM * ONE_BLK_SIZE;
    // veccalc: pertokenScale * scale -> (baseM * baseN) fp32, maxSize: 128KB, in ub each mLoop iteration
    needUbSize += baseM * ubCalcN * sizeof(float32_t);
    // input: pertokenScale, in ub each mLoop's iteration.
    needUbSize += ops::CeilAlign(baseM, 8UL) * sizeof(float32_t); // 8: 32 / sizeof(fp32)
    if (!inputParams_.isPerTensor) {
        // input: scale, fp32: db * sizeof(fp32); bf16：db * sizeof(bf16) + veccalc sizeof(fp32) -> 2 * 4
        needUbSize += NUM_DB * ubCalcN * sizeof(float32_t);
    }
    if (inputParams_.biasDtype != ge::DT_INT32) {
        // input: bias bf16 fp16 fp32
        needUbSize += NUM_DB * ubCalcN * ge::GetSizeByDataType(inputParams_.biasDtype);
        // veccalc: bias fp32
        needUbSize += ubCalcN * sizeof(float32_t);
    }
    OP_TILING_CHECK(
        needUbSize >= ubSize,
        OP_LOGE(inputParams_.opName, "there is no proper ub tiling when m(%lu) pertoken opt", inputParams_.mSize),
        return ge::GRAPH_FAILED);

    ubSize -= needUbSize;
    // 已知ubCalcN, 求解还能放得下的ubCalcM
    // veccalc: int32 -> fp32
    ubCalc += ubCalcN * sizeof(float32_t);
    uint32_t ubCalcM = std::min(std::min(ubSize / ubCalc, baseM), inputParams_.mSize);
    OP_TILING_CHECK(ubCalcM == 0, OP_LOGE(inputParams_.opName, "failed to calc ubCalcM(0) with ubCalcN(%u)", ubCalcN),
                    return ge::GRAPH_FAILED);
    tilingData_.params.ubCalcM = ubCalcM;
    tilingData_.params.ubCalcN = ubCalcN;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2QuantBatchMatmulV3Tiling::CalcUbTiling()
{
    if (isBf16Opt_ && inputParams_.isPertoken) {
        return CalcPertokenOptUbTiling();
    }
    return CalcUbTiling(static_cast<uint32_t>(tbeTiling_.n_l0) * BLOCK_CUBE,
                        static_cast<uint32_t>(tbeTiling_.m_l0) * BLOCK_CUBE);
}

ge::graphStatus Mc2QuantBatchMatmulV3Tiling::CalcUbTiling(uint32_t baseN, uint32_t baseM)
{
    uint64_t ubSize = aicoreParams_.ubSize;
    uint64_t needUbSize = 0;
    uint32_t ubCalcN = baseN;
    // src(int32) + scale(fp32/bf16) + pertoken(fp32) + out(fp16/bf16) + veccalc, in and out need double buffer
    // int16_t reprersent bf16, input src + output dst + veccalc dequant api
    uint64_t ubCalc = (NUM_DB * (sizeof(int32_t) + sizeof(int16_t)) + UB_EXTRE_BYTE) * ubCalcN;
    // input: scale perchannel
    if (!inputParams_.isPerTensor) {
        ubCalc += NUM_DB * ge::GetSizeByDataType(inputParams_.scaleDtype) * ubCalcN;
    }
    if (inputParams_.isPertoken || (inputParams_.biasDtype != ge::DT_INT32)) {
        // veccalc: dequant api dst fp32
        ubCalc += sizeof(float) * ubCalcN;
    }
    if (inputParams_.isPertoken) {
        // veccalc: BroadCast需要的临时空间，最小为256b，最大为align(ubM, 8) * 32b, 按照baseM先算
        // baseM不会超过2048，不需要乘法溢出校验
        needUbSize += baseM * ONE_BLK_SIZE;
    }
    if (inputParams_.isPertoken) {
        // input: pertokenScale fp32
        ubCalc += NUM_DB * sizeof(float);
        // 7: to comfirm that pertokenScale 32B(8, fp32) aligned, up to 7, eg: 1->8
        needUbSize += NUM_DB * sizeof(float) * 7;
        // veccalc: mul(* pertokenScale) fp32 m * n, res of broadcast
        ubCalc += sizeof(float) * ubCalcN;
    }
    if (inputParams_.biasDtype != ge::DT_INT32) {
        // veccalc: fp32 out muls fp32 bias
        ubCalc += sizeof(float) * ubCalcN;
        // input: bias bf16/fp16/fp32, veccalc: bias fp32
        needUbSize += NUM_DB * ge::GetSizeByDataType(inputParams_.biasDtype) * ubCalcN + sizeof(float) * ubCalcN;
    }
    OP_TILING_CHECK(needUbSize >= ubSize,
                    OP_LOGE(inputParams_.opName, "there is no proper ub tiling when m(%lu) n(%lu) baseM(%u) baseN(%u)",
                            inputParams_.mSize, inputParams_.nSize, baseM, baseN),
                    return ge::GRAPH_FAILED);
    ubSize -= needUbSize;
    uint32_t ubCalcM = std::min(std::min(ubSize / ubCalc, static_cast<uint64_t>(baseM)), inputParams_.mSize);
    OP_TILING_CHECK(ubCalcM == 0, OP_LOGE(inputParams_.opName, "failed to calc ubCalcM(0) with ubCalcN(%u)", ubCalcN),
                    return ge::GRAPH_FAILED);
    tilingData_.params.ubCalcN = ubCalcN;
    tilingData_.params.ubCalcM = ubCalcM;
    tilingData_.params.needUbBuffer = ubCalcN * ubCalcM * UB_EXTRE_BYTE;
    return ge::GRAPH_SUCCESS;
}

bool Mc2QuantBatchMatmulV3Tiling::NeedAtomiClean() const
{
    if (!compileInfo_.supportL0c2Out) {
        uint32_t alignSize = ONE_BLK_SIZE / ge::GetSizeByDataType(inputParams_.cDtype);

        uint32_t baseN = static_cast<uint32_t>(tilingData_.matmulTiling.baseN);
        uint32_t singleCoreN = static_cast<uint32_t>(tilingData_.matmulTiling.singleCoreN);
        if (baseN < alignSize || CalcTailSize(singleCoreN, baseN) < alignSize) {
            return true;
        }

        uint32_t nDim = ops::CeilDiv(static_cast<uint32_t>(inputParams_.nSize), singleCoreN);
        uint32_t tailSingleCoreN = inputParams_.nSize - (nDim - 1) * singleCoreN;
        return CalcTailSize(tailSingleCoreN, baseN) < alignSize;
    } else {
        uint32_t singleCoreK = static_cast<uint32_t>(tilingData_.matmulTiling.singleCoreK);
        return singleCoreK < static_cast<uint32_t>(inputParams_.kSize);
    }
}

REGISTER_OPS_TILING_TEMPLATE(Mc2QuantBatchMatmulV3, Mc2QuantBatchMatmulV3Tiling, 1);
REGISTER_OPS_TILING_TEMPLATE(Mc2QuantBatchMatmulV3, Mc2AdaptiveSlidingWindowTiling, 2);

static ge::graphStatus Mc2QuantBatchMatmulV3TilingFunc(gert::TilingContext *context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "Mc2QuantBatchMatmulV3", "TilingContext is null!");
    auto compileInfoPtr = context->GetCompileInfo<Mc2QuantBatchMatmulV3CompileInfo>();
    if (compileInfoPtr->supportL12BtBf16) {
        std::vector<int32_t> registerList = {2};
        OP_LOGD("NO_OP_NAME", "Adaptive sliding window tiling process.");
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    } else {
        std::vector<int32_t> registerList = {0, 1};
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    }
}

static ge::graphStatus TilingParseForQuantBatchMatmulV3(gert::TilingParseContext *context)
{
    OP_LOGE_IF(context == nullptr, ge::GRAPH_FAILED, "Mc2QuantBatchMatmulV3", "TilingParseContext is null!");
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "The platformInfoPtr is null!");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<Mc2QuantBatchMatmulV3CompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "The compileInfoPtr is null!");
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    compileInfoPtr->workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();

    std::string platformRes;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", platformRes);
    compileInfoPtr->supportL0c2Out = !platformRes.empty();
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", platformRes);
    compileInfoPtr->supportL12BtBf16 = (platformRes.find("bf16") != std::string::npos);
    platformInfoPtr->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    if (!TilingPrepareForOpCache(context)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Mc2QuantBatchMatmulV3)
    .Tiling(Mc2QuantBatchMatmulV3TilingFunc)
    .TilingParse<Mc2QuantBatchMatmulV3CompileInfo>(TilingParseForQuantBatchMatmulV3);
} // namespace optiling
