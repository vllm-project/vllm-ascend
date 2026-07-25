/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fused_gdn_decode_tiling.h"

#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling_base/error_log.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fused_gdn_decode_tiling_data.h"

using namespace FusedGdnDecode;

namespace optiling {
namespace {
constexpr size_t MIXED_INDEX = 0;
constexpr size_t A_LOG_INDEX = 3;
constexpr size_t DT_BIAS_INDEX = 4;
constexpr size_t STATE_INDEX = 5;
constexpr uint64_t TILING_BF16_STATE_FP32 = 1;
constexpr uint64_t TILING_FP16_STATE_FP32 = 2;
constexpr uint64_t TILING_BF16_STATE_BF16 = 3;
constexpr uint64_t TILING_FP16_STATE_FP16 = 4;
constexpr uint64_t UB_ALIGN_BYTES = 256;
constexpr uint64_t SCALAR_UB_ELEMS = 192;

uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return (x + y - 1) / y;
}

uint32_t Align(uint32_t x, uint32_t y)
{
    return CeilDiv(x, y) * y;
}

uint64_t EstimateQueueBytes(uint32_t k, uint32_t v, uint32_t bv, uint32_t stateBytes)
{
    const uint32_t alignK = Align(k, 16);
    const uint32_t alignBV = Align(bv, 16);
    const uint64_t qkBytes = 2ULL * alignK * sizeof(uint16_t);
    const uint64_t stateSlotBytes =
        static_cast<uint64_t>(alignK) * alignBV * stateBytes +
        static_cast<uint64_t>(alignBV) * sizeof(uint16_t);
    const uint64_t stateBufferNum = bv >= v ? 1ULL : 2ULL;
    return qkBytes + 2ULL * stateBufferNum * stateSlotBytes;
}

uint64_t EstimateTmpBytes(uint32_t k, uint32_t bv, uint32_t stateBytes)
{
    const uint32_t alignK = Align(k, 16);
    const uint32_t alignBV = Align(bv, 16);
    const uint64_t computeMatrixBytes =
        stateBytes == sizeof(float) ? 0ULL : 2ULL * alignK * alignBV * sizeof(float);
    return
        (3ULL * alignK + 3ULL * alignBV + SCALAR_UB_ELEMS) * sizeof(float) +
        computeMatrixBytes +
        9ULL * UB_ALIGN_BYTES;
}

uint64_t EstimateUbBytes(uint32_t k, uint32_t v, uint32_t bv, uint32_t stateBytes)
{
    return EstimateQueueBytes(k, v, bv, stateBytes) + EstimateTmpBytes(k, bv, stateBytes);
}

uint32_t SelectBv(uint32_t k, uint32_t v, uint32_t stateBytes, uint64_t ubSize)
{
    const uint32_t candidates[] = {128, 64, 32, 16, 8};
    for (uint32_t candidate : candidates) {
        if (candidate > v) {
            continue;
        }
        if (EstimateUbBytes(k, v, candidate, stateBytes) < ubSize) {
            return candidate;
        }
    }
    return 8;
}
} // namespace

ge::graphStatus FusedGdnDecodeTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE("FusedGdnDecode", "tiling context is null");
        return ge::GRAPH_FAILED;
    }
    auto *mixedShapePtr = context->GetInputShape(MIXED_INDEX);
    if (mixedShapePtr == nullptr) {
        OP_LOGE("FusedGdnDecode", "mixed shape ptr null");
        return ge::GRAPH_FAILED;
    }
    const auto &mixedShape = mixedShapePtr->GetOriginShape();
    auto *stateShapePtr = context->GetInputShape(STATE_INDEX);
    if (stateShapePtr == nullptr) {
        OP_LOGE("FusedGdnDecode", "state shape ptr null");
        return ge::GRAPH_FAILED;
    }
    const auto &stateShape = stateShapePtr->GetOriginShape();
    if (mixedShape.GetDimNum() < 2 || stateShape.GetDimNum() < 4) {
        OP_LOGE("FusedGdnDecode", "invalid dim num: mixed=%zu state=%zu", mixedShape.GetDimNum(),
                stateShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    const uint32_t batch = static_cast<uint32_t>(mixedShape.GetDim(0));
    const uint32_t mixedDim = static_cast<uint32_t>(mixedShape.GetDim(1));
    auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE("FusedGdnDecode", "attrs null");
        return ge::GRAPH_FAILED;
    }
    const uint32_t hv = static_cast<uint32_t>(stateShape.GetDim(1));
    const uint32_t v = static_cast<uint32_t>(stateShape.GetDim(2));
    const uint32_t k = static_cast<uint32_t>(stateShape.GetDim(3));
    if (batch == 0 || hv == 0 || v == 0 || k == 0 || mixedDim <= hv * v) {
        OP_LOGE("FusedGdnDecode", "invalid shape values: batch=%u mixedDim=%u hv=%u v=%u k=%u", batch, mixedDim, hv,
                v, k);
        return ge::GRAPH_FAILED;
    }
    const uint32_t qkDim = mixedDim - hv * v;
    if ((qkDim % (2 * k)) != 0) {
        OP_LOGE("FusedGdnDecode", "invalid qkDim/k: qkDim=%u k=%u", qkDim, k);
        return ge::GRAPH_FAILED;
    }
    const uint32_t h = qkDim / (2 * k);
    if (h == 0 || (hv % h) != 0) {
        OP_LOGE("FusedGdnDecode", "invalid head relation: h=%u hv=%u", h, hv);
        return ge::GRAPH_FAILED;
    }

    auto *platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE("FusedGdnDecode", "platform info null");
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t aivNum = platform.GetCoreNumAiv();
    if (aivNum == 0) {
        aivNum = 1;
    }

    auto *mixedDesc = context->GetInputDesc(MIXED_INDEX);
    auto *stateDesc = context->GetInputDesc(STATE_INDEX);
    auto *aLogDesc = context->GetInputDesc(A_LOG_INDEX);
    auto *dtBiasDesc = context->GetInputDesc(DT_BIAS_INDEX);
    if (mixedDesc == nullptr || stateDesc == nullptr || aLogDesc == nullptr || dtBiasDesc == nullptr) {
        OP_LOGE("FusedGdnDecode", "desc null: mixed=%p state=%p aLog=%p dtBias=%p", mixedDesc, stateDesc, aLogDesc,
                dtBiasDesc);
        return ge::GRAPH_FAILED;
    }
    if (aLogDesc->GetDataType() != ge::DT_FLOAT || dtBiasDesc->GetDataType() != ge::DT_FLOAT) {
        OP_LOGE("FusedGdnDecode", "invalid aLog/dtBias dtype: aLog=%d dtBias=%d", aLogDesc->GetDataType(),
                dtBiasDesc->GetDataType());
        return ge::GRAPH_FAILED;
    }
    const ge::DataType mixedDtype = mixedDesc->GetDataType();
    const ge::DataType stateDtype = stateDesc->GetDataType();
    const uint32_t stateBytes = stateDtype == ge::DT_FLOAT ? 4 : 2;

    uint64_t tilingKey = TILING_BF16_STATE_FP32;
    if (mixedDtype == ge::DT_FLOAT16 && stateDtype == ge::DT_FLOAT) {
        tilingKey = TILING_FP16_STATE_FP32;
    } else if (mixedDtype == ge::DT_BF16 && stateDtype == ge::DT_BF16) {
        tilingKey = TILING_BF16_STATE_BF16;
    } else if (mixedDtype == ge::DT_FLOAT16 && stateDtype == ge::DT_FLOAT16) {
        tilingKey = TILING_FP16_STATE_FP16;
    }

    float scale = 1.0f;
    float threshold = 20.0f;
    if (attrs != nullptr) {
        const float *scaleAttr = attrs->GetAttrPointer<float>(0);
        if (scaleAttr != nullptr) {
            scale = *scaleAttr;
        }
        const float *thresholdAttr = attrs->GetAttrPointer<float>(1);
        if (thresholdAttr != nullptr) {
            threshold = *thresholdAttr;
        }
    }

    const uint32_t bv = SelectBv(k, v, stateBytes, ubSize);
    const uint32_t vTiles = CeilDiv(v, bv);
    const uint32_t totalTasks = batch * hv;
    const uint32_t maxTasksPerBlock = CeilDiv(totalTasks, aivNum);
    uint32_t blockDim = CeilDiv(totalTasks, maxTasksPerBlock);
    if (blockDim == 0) {
        blockDim = 1;
    }

    FusedGdnDecodeTilingData td{};
    td.b = batch;
    td.h = h;
    td.hv = hv;
    td.k = k;
    td.v = v;
    td.bv = bv;
    td.vTiles = vTiles;
    td.stateBufferNum = bv >= v ? 1 : 2;
    td.totalTasks = totalTasks;
    td.mixedStride = mixedDim;
    td.stateSlotStride = hv * v * k;
    td.stateHeadStride = v * k;
    td.outBatchStride = hv * v;
    td.scale = scale;
    td.softplusThreshold = threshold;
    td.ubRestBytes = static_cast<uint32_t>(EstimateTmpBytes(k, bv, stateBytes));

    auto *rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetCapacity() < sizeof(FusedGdnDecodeTilingData)) {
        OP_LOGE("FusedGdnDecode", "raw tiling invalid: ptr=%p capacity=%zu need=%zu", rawTilingData,
                rawTilingData == nullptr ? 0UL : rawTilingData->GetCapacity(), sizeof(FusedGdnDecodeTilingData));
        return ge::GRAPH_FAILED;
    }
    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(), &td, sizeof(td));
    if (ret != EOK) {
        OP_LOGE("FusedGdnDecode", "tiling memcpy_s failed: ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(sizeof(td));
    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForFusedGdnDecode(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

IMPL_OP_OPTILING(FusedGdnDecode)
    .Tiling(optiling::FusedGdnDecodeTilingFunc)
    .TilingParse<optiling::FusedGdnDecodeCompileInfo>(optiling::TilingPrepareForFusedGdnDecode);
