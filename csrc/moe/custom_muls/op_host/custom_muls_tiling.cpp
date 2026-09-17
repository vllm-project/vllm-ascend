#include "custom_muls_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_impl_registry.h"
#include "op_common/log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/custom_muls_tiling_key.h"

namespace optiling {
namespace {
template <class T>
T CeilDiv(T x, T y)
{
    return (x / y) + static_cast<T>((x % y) != 0);
}

template <class T>
T AlignDown(T x, T align)
{
    return (x / align) * align;
}

ge::graphStatus SaveTilingData(gert::TilingContext* context, CustomMulsTilingData& tilingData)
{
    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CustomMulsTilingFunc(gert::TilingContext* context)
{
    constexpr uint64_t BLOCK_ALIGN_ELEMS = 512;
    constexpr uint64_t MIN_BLOCK_BYTES = 16 * 1024;
    constexpr uint64_t BYTES_PER_ELEMENT = 16;
    constexpr uint64_t UB_ALIGN_BYTES = 256;
    constexpr uint64_t DMA_ALIGN_BYTES = 32;
    constexpr uint64_t UB_USAGE_DENOMINATOR = 16;
    constexpr uint32_t MAX_SUPPORTED_RANK = 8;
    constexpr size_t WORKSPACE_NUM = 1;

    auto* info = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, info);
    platform_ascendc::PlatformAscendC platform(info);
    const uint32_t availableAiv = platform.GetCoreNumAiv();
    OP_CHECK_IF(availableAiv == 0,
                OP_LOGE(context->GetNodeName(), "customMuls: no AIV core"),
                return ge::GRAPH_FAILED);

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    OP_CHECK_IF(ubBytes == 0,
                OP_LOGE(context->GetNodeName(), "customMuls: UB size is zero"),
                return ge::GRAPH_FAILED);

    const auto* xShape = context->GetInputShape(0);
    const auto* xDesc = context->GetInputDesc(0);
    const auto* yShape = context->GetOutputShape(0);
    const auto* yDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    const ge::DataType dtype = xDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT16 && dtype != ge::DT_FLOAT,
                OP_LOGE(context->GetNodeName(), "customMuls only supports BF16, FP16 and FP32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yDesc->GetDataType() != dtype,
                OP_LOGE(context->GetNodeName(), "customMuls output dtype must equal input dtype"),
                return ge::GRAPH_FAILED);

    const uint64_t typeBytes = (dtype == ge::DT_FLOAT) ? sizeof(float) : sizeof(uint16_t);
    const auto& xStorageShape = xShape->GetStorageShape();
    const auto& yStorageShape = yShape->GetStorageShape();
    OP_CHECK_IF((xStorageShape.GetDimNum() > MAX_SUPPORTED_RANK) ||
                    (yStorageShape.GetDimNum() > MAX_SUPPORTED_RANK),
                OP_LOGE(context->GetNodeName(), "customMuls: input or output rank exceeds %u", MAX_SUPPORTED_RANK),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xStorageShape != yStorageShape,
                OP_LOGE(context->GetNodeName(), "customMuls: output shape must equal input shape"),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < xStorageShape.GetDimNum(); ++i) {
        OP_CHECK_IF(xStorageShape.GetDim(i) < 0,
                    OP_LOGE(context->GetNodeName(), "customMuls: input dimension %zu is negative: %ld", i,
                            xStorageShape.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* scalar = attrs->GetAttrPointer<float>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalar);

    const int64_t shapeSize = xStorageShape.GetShapeSize();
    OP_CHECK_IF(shapeSize < 0,
                OP_LOGE(context->GetNodeName(), "customMuls: shape size is invalid: %ld", shapeSize),
                return ge::GRAPH_FAILED);
    const uint64_t total = static_cast<uint64_t>(shapeSize);
    size_t* workspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);

    const uint64_t tilingKey = (dtype == ge::DT_BF16) ? CUSTOM_MULS_KEY_BF16_PROMOTE_FP32 :
        ((dtype == ge::DT_FLOAT16) ? CUSTOM_MULS_KEY_FP16_PROMOTE_FP32 : CUSTOM_MULS_KEY_FP32_DIRECT);

    CustomMulsTilingData tilingData;
    tilingData.set_totalLength(shapeSize);
    tilingData.set_blockFormer(0);
    tilingData.set_blockTail(0);
    tilingData.set_blockNum(0);
    tilingData.set_ubFormer(0);
    tilingData.set_ubLoopFormer(0);
    tilingData.set_ubTailFormer(0);
    tilingData.set_ubLoopTail(0);
    tilingData.set_ubTailTail(0);
    tilingData.set_scalarValue(*scalar);

    context->SetTilingKey(tilingKey);
    workspace[0] = 0;

    if (total == 0) {
        context->SetBlockDim(1);
        return SaveTilingData(context, tilingData);
    }

    OP_CHECK_IF(total > (std::numeric_limits<uint64_t>::max() / typeBytes),
                OP_LOGE(context->GetNodeName(),
                        "customMuls: tensor byte size overflows for %lu elements of %lu bytes", total, typeBytes),
                return ge::GRAPH_FAILED);

    const uint64_t tensorBytes = total * typeBytes;
    const uint64_t desired = CeilDiv(tensorBytes, MIN_BLOCK_BYTES);
    const uint64_t candidates = std::min<uint64_t>(std::max<uint64_t>(desired, 1), availableAiv);
    const uint64_t perCoreElements = CeilDiv(total, candidates);
    const uint64_t alignedUnits = CeilDiv(perCoreElements, BLOCK_ALIGN_ELEMS);
    OP_CHECK_IF(alignedUnits > (std::numeric_limits<uint64_t>::max() / BLOCK_ALIGN_ELEMS),
                OP_LOGE(context->GetNodeName(), "customMuls: aligned block length overflows uint64"),
                return ge::GRAPH_FAILED);

    const uint64_t alignedPerCore = alignedUnits * BLOCK_ALIGN_ELEMS;
    const uint64_t minBlockElements = CeilDiv(MIN_BLOCK_BYTES, typeBytes);
    const uint64_t former = std::max(alignedPerCore, minBlockElements);
    OP_CHECK_IF(former > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context->GetNodeName(), "customMuls: block length %lu exceeds int64 range", former),
                return ge::GRAPH_FAILED);

    const uint64_t blocks64 = CeilDiv(total, former);
    OP_CHECK_IF((blocks64 == 0) || (blocks64 > std::numeric_limits<uint32_t>::max()),
                OP_LOGE(context->GetNodeName(), "customMuls: logical block count %lu is outside uint32 range", blocks64),
                return ge::GRAPH_FAILED);

    const uint64_t blockRemainder = total % former;
    const uint64_t tail = (blockRemainder == 0) ? former : blockRemainder;
    OP_CHECK_IF(tail > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                OP_LOGE(context->GetNodeName(), "customMuls: tail block length %lu exceeds int64 range", tail),
                return ge::GRAPH_FAILED);

    const uint64_t usable = AlignDown(ubBytes - CeilDiv(ubBytes, UB_USAGE_DENOMINATOR), DMA_ALIGN_BYTES);
    const uint64_t ubAlignElements = UB_ALIGN_BYTES / typeBytes;
    const uint64_t ub64 = AlignDown(usable / BYTES_PER_ELEMENT, ubAlignElements);
    OP_CHECK_IF((ub64 == 0) || (ub64 > std::numeric_limits<uint32_t>::max()) ||
                    (ub64 > (std::numeric_limits<uint32_t>::max() / typeBytes)),
                OP_LOGE(context->GetNodeName(), "customMuls: UB tile %lu is outside the valid uint32 byte range", ub64),
                return ge::GRAPH_FAILED);

    const uint64_t ubLoopFormer64 = CeilDiv(former, ub64);
    const uint64_t ubFormerRemainder = former % ub64;
    const uint64_t ubTailFormer64 = (ubFormerRemainder == 0) ? ub64 : ubFormerRemainder;
    const uint64_t ubLoopTail64 = CeilDiv(tail, ub64);
    const uint64_t ubTailRemainder = tail % ub64;
    const uint64_t ubTailTail64 = (ubTailRemainder == 0) ? ub64 : ubTailRemainder;
    OP_CHECK_IF((ubLoopFormer64 > std::numeric_limits<uint32_t>::max()) ||
                    (ubTailFormer64 > std::numeric_limits<uint32_t>::max()) ||
                    (ubLoopTail64 > std::numeric_limits<uint32_t>::max()) ||
                    (ubTailTail64 > std::numeric_limits<uint32_t>::max()),
                OP_LOGE(context->GetNodeName(), "customMuls: UB loop or tail value exceeds uint32 range"),
                return ge::GRAPH_FAILED);

    const uint32_t blocks = static_cast<uint32_t>(blocks64);
    context->SetBlockDim(blocks);
    tilingData.set_blockFormer(static_cast<int64_t>(former));
    tilingData.set_blockTail(static_cast<int64_t>(tail));
    tilingData.set_blockNum(blocks);
    tilingData.set_ubFormer(static_cast<uint32_t>(ub64));
    tilingData.set_ubLoopFormer(static_cast<uint32_t>(ubLoopFormer64));
    tilingData.set_ubTailFormer(static_cast<uint32_t>(ubTailFormer64));
    tilingData.set_ubLoopTail(static_cast<uint32_t>(ubLoopTail64));
    tilingData.set_ubTailTail(static_cast<uint32_t>(ubTailTail64));

    return SaveTilingData(context, tilingData);
}

ge::graphStatus Parse(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct CustomMulsCompileInfo {};
} // namespace

IMPL_OP_OPTILING(customMuls).Tiling(CustomMulsTilingFunc).TilingParse<CustomMulsCompileInfo>(Parse);
} // namespace optiling
