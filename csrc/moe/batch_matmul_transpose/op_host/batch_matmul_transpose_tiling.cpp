/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_matmul_transpose_tiling.cpp
 * \brief
 */
#include "batch_matmul_transpose_tiling.h"

#include <map>
#include <string>

#include "batch_matmul_transpose_tiling_data.h"
#include "common.h"
#include "common_tiling.h"
#include "tiling_base/error_log.h"
#include "log/ops_log.h"

namespace optiling {
namespace {
constexpr int32_t INPUT_A_INDEX = 0;
constexpr int32_t INPUT_B_INDEX = 1;
constexpr int32_t OUTPUT_C_INDEX = 0;
constexpr int32_t ATTR_FORMAT_MODE_INDEX = 0;
constexpr int32_t ATTR_QUANT_MODE_INDEX = 1;

const std::map<std::string, uint16_t> QUANT_MODE_MAP = {
    {"per_channel_symm", 0},
    {"per_channel_asymm", 1},
    {"per_token_symm", 2},
};
const std::map<std::string, uint16_t> FORMAT_MODE_MAP = {
    {"ND", 0},
    {"NZ", 1},
};
const std::map<ge::DataType, pp_matmul::TensorDType> GE_TYPE_MAP = {
    {ge::DT_BF16, pp_matmul::TensorDType::TENSOR_DTYPE_BF16},
    {ge::DT_FLOAT16, pp_matmul::TensorDType::TENSOR_DTYPE_FLOAT16},
};

ge::graphStatus UpdatePlatformInfo(gert::TilingContext* context)
{
    auto compileInfo = context->GetCompileInfo<BatchMatmulTransposeCompileInfo>();
    if (compileInfo != nullptr) {
        host_utils::PlatformInfo::Instance().Update(
            compileInfo->coreNumAic, compileInfo->coreNumAic, compileInfo->coreNumAiv,
            compileInfo->ubSize, compileInfo->l1Size, compileInfo->l2Size,
            compileInfo->l0aSize, compileInfo->l0bSize, compileInfo->l0cSize,
            compileInfo->socVersion);
        return ge::GRAPH_SUCCESS;
    }

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    host_utils::PlatformInfo::Instance().Update(ascendcPlatform);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetModeVal(const std::map<std::string, uint16_t> &modeMap, const std::string &mode,
                           const std::string &defaultMode, const char *modeName, gert::TilingContext* context,
                           uint16_t &modeVal)
{
    std::string modeStr = mode.empty() ? defaultMode : mode;
    auto it = modeMap.find(modeStr);
    OP_CHECK_IF(it == modeMap.end(), OP_LOGE(context, "%s: Unsupported mode value %s", modeName, modeStr.c_str()),
                return ge::GRAPH_FAILED);
    modeVal = it->second;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4BatchMatmulTranspose(gert::TilingContext* context)
{
    OP_LOGI("Tiling4BatchMatmulTranspose", "Enter Tiling4BatchMatmulTranspose");
    OPS_LOG_D(context, "Tiling4BatchMatmulTranspose running.\n");

    const gert::StorageShape* aStorageShapePtr = context->GetInputShape(INPUT_A_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, aStorageShapePtr);
    const gert::StorageShape* bStorageShapePtr = context->GetInputShape(INPUT_B_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, bStorageShapePtr);

    const gert::Shape aShape = aStorageShapePtr->GetStorageShape();
    const gert::Shape bShape = bStorageShapePtr->GetStorageShape();
    size_t aDimNum = aShape.GetDimNum();
    size_t bDimNum = bShape.GetDimNum();
    OP_CHECK_IF(aDimNum != 3, OP_LOGE(context, "tensor_a should be dim3."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(bDimNum != 3 && bDimNum != 4,
                OP_LOGE(context, "tensor_b should be dim3 in ND or dim4 in NZ."), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* formatModePtr = attrs->GetAttrPointer<char>(ATTR_FORMAT_MODE_INDEX);
    const char* quantModePtr = attrs->GetAttrPointer<char>(ATTR_QUANT_MODE_INDEX);
    std::string formatMode = (formatModePtr != nullptr) ? formatModePtr : "ND";
    std::string quantMode = (quantModePtr != nullptr) ? quantModePtr : "per_channel_symm";

    uint16_t formatModeVal = 0;
    uint16_t quantModeVal = 0;
    OP_CHECK_IF(GetModeVal(FORMAT_MODE_MAP, formatMode, "ND", "format_mode", context, formatModeVal) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "format_mode invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetModeVal(QUANT_MODE_MAP, quantMode, "per_channel_symm", "quant_mode", context, quantModeVal) !=
                    ge::GRAPH_SUCCESS,
                OP_LOGE(context, "quant_mode invalid."), return ge::GRAPH_FAILED);

    ge::DataType aDtype = context->GetInputDesc(INPUT_A_INDEX)->GetDataType();
    ge::DataType bDtype = context->GetInputDesc(INPUT_B_INDEX)->GetDataType();
    ge::DataType cDtype = context->GetOutputDesc(OUTPUT_C_INDEX)->GetDataType();
    OP_CHECK_IF(aDtype != bDtype || bDtype != cDtype,
                OP_LOGE(context, "tensor type is not the same."), return ge::GRAPH_FAILED);
    auto dtypeIt = GE_TYPE_MAP.find(aDtype);
    OP_CHECK_IF(dtypeIt == GE_TYPE_MAP.end(),
                OP_LOGE(context, "tensor type only support half or bf16."), return ge::GRAPH_FAILED);

    uint32_t n = 0;
    pp_matmul::TensorFormat formatModeEnum = pp_matmul::TensorFormat::TENSOR_FORMAT_ND;
    if (formatModeVal == 0) {
        OP_CHECK_IF(bDimNum != 3, OP_LOGE(context, "tensor_b shape should be dim3 in ND format."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(aShape.GetDim(2) != bShape.GetDim(1), OP_LOGE(context, "tensor shape is wrong."),
                    return ge::GRAPH_FAILED);
        n = static_cast<uint32_t>(bShape.GetDim(2));
    } else {
        formatModeEnum = pp_matmul::TensorFormat::TENSOR_FORMAT_NZ;
        OP_CHECK_IF(bDimNum != 4, OP_LOGE(context, "tensor_b shape should be dim4 in NZ format."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(aShape.GetDim(2) != bShape.GetDim(2), OP_LOGE(context, "tensor shape is wrong."),
                    return ge::GRAPH_FAILED);
        n = static_cast<uint32_t>(bShape.GetDim(1) * bShape.GetDim(3));
    }
    OP_CHECK_IF(aShape.GetDim(1) != bShape.GetDim(0), OP_LOGE(context, "tensor shape is wrong."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(UpdatePlatformInfo(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "update platform info failed."),
                return ge::GRAPH_FAILED);

    pp_matmul::OpShape opShape = {
        .batchSize = static_cast<uint32_t>(aShape.GetDim(1)),
        .m = static_cast<uint32_t>(aShape.GetDim(0)),
        .k = static_cast<uint32_t>(aShape.GetDim(2)),
        .n = n,
    };
    pp_matmul::PpMatmulTilingData matmulTilingData = {
        .opShape = opShape,
    };
    pp_matmul::MatMulInfo mmInfo = {
        .batchSize = opShape.batchSize,
        .m = opShape.m,
        .k = opShape.k,
        .n = opShape.n,
        .dtypeA = dtypeIt->second,
        .dtypeB = dtypeIt->second,
        .dtypeC = dtypeIt->second,
        .formatB = formatModeEnum,
        .mmType = pp_matmul::MatMul::MatMulType::MATMUL_EIN_SUM,
        .inDtype = 2.0,
        .outDtype = 2.0,
        .quantMode = static_cast<pp_matmul::MatMul::QuantMode>(quantModeVal),
    };
    pp_matmul::HardwareInfo hwInfo;
    uint32_t blockDim = 0;
    pp_matmul::GetPpMatmulTiling(mmInfo, hwInfo, blockDim, matmulTilingData);
    OP_CHECK_IF(!host_utils::PpMatmulTilingCheck(matmulTilingData),
                OP_LOGE(context, "pp matmul tiling data is invalid."), return ge::GRAPH_FAILED);

    BatchMatmulTransposeTilingData tiling;
    tiling.set_batchSize(matmulTilingData.opShape.batchSize);
    tiling.set_m(matmulTilingData.opShape.m);
    tiling.set_k(matmulTilingData.opShape.k);
    tiling.set_n(matmulTilingData.opShape.n);
    tiling.set_m0(matmulTilingData.opShape.m0);
    tiling.set_k0(matmulTilingData.opShape.k0);
    tiling.set_n0(matmulTilingData.opShape.n0);
    tiling.set_mLoop(matmulTilingData.mLoop);
    tiling.set_kLoop(matmulTilingData.kLoop);
    tiling.set_nLoop(matmulTilingData.nLoop);
    tiling.set_coreLoop(matmulTilingData.coreLoop);
    tiling.set_swizzlCount(matmulTilingData.swizzlCount);
    tiling.set_tilingKey(matmulTilingData.tilingKey);
    tiling.set_blockDim(matmulTilingData.blockDim);
    tiling.set_swizzlDirect(matmulTilingData.swizzlDirect);
    tiling.set_splitk(matmulTilingData.splitk);
    tiling.set_enShuffleK(matmulTilingData.enShuffleK);
    tiling.set_quantMode(matmulTilingData.quantMode);

    context->SetBlockDim(blockDim);
    // The kernel binary only registers the default tiling key; the real key is
    // carried in the tiling data and dispatched inside the kernel.
    context->SetTilingKey(0);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    OPS_LOG_I(context, "BatchMatmulTranspose tiling key: %u, block dim: %u", matmulTilingData.tilingKey, blockDim);
    OP_LOGI(context, "End Tiling4BatchMatmulTranspose");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4BatchMatmulTranspose(gert::TilingParseContext* context)
{
    OP_LOGI(context, "TilingPrepare4BatchMatmulTranspose running.");
    auto compileInfo = context->GetCompiledInfo<BatchMatmulTransposeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->coreNumAic = ascendcPlatform.GetCoreNumAic();
    compileInfo->coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo->l0cSize);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(BatchMatmulTranspose).Tiling(Tiling4BatchMatmulTranspose)
    .TilingParse<BatchMatmulTransposeCompileInfo>(TilingPrepare4BatchMatmulTranspose);
} // namespace optiling
