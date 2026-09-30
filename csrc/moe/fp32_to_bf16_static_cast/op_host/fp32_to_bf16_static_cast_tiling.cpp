/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <limits>

#include "fp32_to_bf16_static_cast_tiling.h"
#include "error/ops_error.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr int64_t kInputX = 0;
constexpr int64_t kMinDimension = 1;
constexpr int64_t kCoreMinInputElements = 2048;
constexpr int64_t kBlockAlignElements = 512;
constexpr int64_t kTileAlignElements = 128;
constexpr uint64_t kInputBytes = 4;
constexpr uint64_t kOutputBytes = 2;
constexpr uint64_t kBufferBytesPerElement = 2 * (kInputBytes + kOutputBytes);

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

bool AlignUp(int64_t value, int64_t alignment, int64_t& result)
{
    if (value > std::numeric_limits<int64_t>::max() - (alignment - 1)) {
        return false;
    }
    result = CeilDiv(value, alignment) * alignment;
    return true;
}

bool CheckedProduct(const gert::Shape& shape, int64_t& elementNum)
{
    elementNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < kMinDimension ||
            elementNum > std::numeric_limits<int64_t>::max() / dim) {
            return false;
        }
        elementNum *= dim;
    }
    return true;
}
} // namespace

static ge::graphStatus TilingForFp32ToBf16StaticCast(gert::TilingContext* context)
{
    OPS_ERR_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("Fp32ToBf16StaticCast", "Tiling context is null"),
        return ge::GRAPH_FAILED);

    const gert::Tensor* xTensor = context->GetInputTensor(kInputX);
    OPS_LOG_E_IF_NULL(context, xTensor, return ge::GRAPH_FAILED);
    OPS_ERR_IF(xTensor->GetDataType() != ge::DT_FLOAT,
        OPS_LOG_E(context, "Fp32ToBf16StaticCast requires FP32 input."),
        return ge::GRAPH_FAILED);

    const gert::Shape& xShape = xTensor->GetStorageShape();

    int64_t elementNum = 0;
    OPS_ERR_IF(!CheckedProduct(xShape, elementNum),
        OPS_LOG_E(context, "Input dimensions must be positive and have an int64 element count."),
        return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platformInfo, return ge::GRAPH_FAILED);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t availableCores = platform.GetCoreNumAiv();
    OPS_ERR_IF(availableCores == 0,
        OPS_LOG_E(context, "No AIV cores are available."),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OPS_ERR_IF(ubSize == 0,
        OPS_LOG_E(context, "UB size is zero."),
        return ge::GRAPH_FAILED);

    const int64_t requestedCores = CeilDiv(elementNum, kCoreMinInputElements);
    const int64_t coreCount = std::max<int64_t>(
        1, std::min<int64_t>(availableCores, requestedCores));
    int64_t blockFormer = 0;
    OPS_ERR_IF(!AlignUp(CeilDiv(elementNum, coreCount), kBlockAlignElements, blockFormer),
        OPS_LOG_E(context, "Block size overflows int64."),
        return ge::GRAPH_FAILED);
    const int64_t blockNum = CeilDiv(elementNum, blockFormer);

    const uint64_t rawTileElements = ubSize / kBufferBytesPerElement;
    const uint64_t tileElements = rawTileElements / kTileAlignElements * kTileAlignElements;
    OPS_ERR_IF(tileElements == 0 || tileElements > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
        OPS_LOG_E(context, "UB is insufficient for a valid BF16-to-FP32 double-buffer tile."),
        return ge::GRAPH_FAILED);

    Fp32ToBf16StaticCastTilingData tiling;
    tiling.set_elementNum(elementNum);
    tiling.set_blockFormer(blockFormer);
    tiling.set_ubFormer(static_cast<int64_t>(tileElements));
    // The only compiled kernelList entry has suffix _0, so the host key must match.
    context->SetTilingKey(0);
    context->SetBlockDim(blockNum);
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForFp32ToBf16StaticCast(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Fp32ToBf16StaticCast)
    .Tiling(TilingForFp32ToBf16StaticCast)
    .TilingParse<Fp32ToBf16StaticCastCompileInfo>(TilingPrepareForFp32ToBf16StaticCast);
} // namespace optiling
