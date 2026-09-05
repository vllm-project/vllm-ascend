/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstddef>
#include <cstdint>
#include <limits>

#include "mla_preprocess_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {
namespace {

constexpr size_t INPUT_HIDDEN_STATE = 0;
constexpr size_t INPUT_WDQKV = 3;
constexpr size_t INPUT_GAMMA1 = 5;
constexpr size_t INPUT_KV_CACHE = 14;
constexpr size_t INPUT_WUK = 18;
constexpr size_t OUTPUT_KV_CACHE_ROPE = 3;

constexpr size_t ATTR_CACHE_MODE = 0;
constexpr size_t ATTR_QUANT_MODE = 1;
constexpr size_t ATTR_ENABLE_INNER_OUT = 2;
constexpr size_t ATTR_ENABLE_ROPE = 3;
constexpr size_t ATTR_KV_CACHE_STRIDE0 = 4;
constexpr size_t ATTR_KV_CACHE_ROPE_STRIDE0 = 5;

constexpr int64_t CACHE_MODE_MIN = 0;
constexpr int64_t CACHE_MODE_MAX = 3;
constexpr int64_t QUANT_MODE_MIN = 0;
constexpr int64_t QUANT_MODE_MAX = 3;
constexpr uint64_t GENERIC_KERNEL_TILING_KEY = 0;

struct MlaPreprocessCompileInfo {};

bool IsPositiveU32(int64_t value)
{
    return value > 0 && static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max();
}

bool IsPhysicalNzCache(const gert::Shape &shape, int64_t cacheMode)
{
    const bool isNzCache = cacheMode == 2 || cacheMode == 3;
    return isNzCache && shape.GetDimNum() == 4 && shape.GetDim(2) != 1;
}

ge::graphStatus ParseShapesAndAttrs(gert::TilingContext *context, mlapo::OpParam &opParam)
{
    const gert::StorageShape *hiddenStorageShape = context->GetInputShape(INPUT_HIDDEN_STATE);
    const gert::StorageShape *wdqkvStorageShape = context->GetInputShape(INPUT_WDQKV);
    const gert::StorageShape *gamma1StorageShape = context->GetInputShape(INPUT_GAMMA1);
    const gert::StorageShape *kvCacheStorageShape = context->GetInputShape(INPUT_KV_CACHE);
    const gert::StorageShape *wukStorageShape = context->GetInputShape(INPUT_WUK);
    const gert::StorageShape *kvCacheRopeStorageShape = context->GetOutputShape(OUTPUT_KV_CACHE_ROPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, hiddenStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, wdqkvStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma1StorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, wukStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheRopeStorageShape);

    const gert::Shape &hiddenShape = hiddenStorageShape->GetOriginShape();
    const gert::Shape &gamma1Shape = gamma1StorageShape->GetOriginShape();
    const gert::Shape &kvCacheShape = kvCacheStorageShape->GetOriginShape();
    const gert::Shape &wukShape = wukStorageShape->GetOriginShape();
    const gert::Shape &kvCacheRopeShape = kvCacheRopeStorageShape->GetOriginShape();

    OP_CHECK_IF(hiddenShape.GetDimNum() != 2,
                OP_LOGE(context, "hiddenState must have rank 2, got %zu.", hiddenShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gamma1Shape.GetDimNum() < 1,
                OP_LOGE(context, "gamma1 must have at least one dimension."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(wukShape.GetDimNum() != 3,
                OP_LOGE(context, "wuk must have logical rank 3, got %zu.", wukShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kvCacheShape.GetDimNum() < 2,
                OP_LOGE(context, "kv_cache must have at least two dimensions."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kvCacheRopeShape.GetDimNum() < 1,
                OP_LOGE(context, "kv_cache_rope must not be a scalar."), return ge::GRAPH_FAILED);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *cacheModePtr = attrs->GetInt(ATTR_CACHE_MODE);
    const int64_t *quantModePtr = attrs->GetInt(ATTR_QUANT_MODE);
    const bool *enableInnerOutPtr = attrs->GetBool(ATTR_ENABLE_INNER_OUT);
    const bool *enableRopePtr = attrs->GetBool(ATTR_ENABLE_ROPE);
    const int64_t *kvCacheStride0Ptr = attrs->GetInt(ATTR_KV_CACHE_STRIDE0);
    const int64_t *kvCacheRopeStride0Ptr = attrs->GetInt(ATTR_KV_CACHE_ROPE_STRIDE0);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantModePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, enableInnerOutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, enableRopePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheStride0Ptr);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvCacheRopeStride0Ptr);

    const int64_t cacheMode = *cacheModePtr;
    const int64_t quantMode = *quantModePtr;
    OP_CHECK_IF(cacheMode < CACHE_MODE_MIN || cacheMode > CACHE_MODE_MAX,
                OP_LOGE(context, "cacheMode must be in [0, 3], got %ld.", cacheMode),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(quantMode < QUANT_MODE_MIN || quantMode > QUANT_MODE_MAX,
                OP_LOGE(context, "quantMode must be in [0, 3], got %ld.", quantMode),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*kvCacheStride0Ptr <= 0 || *kvCacheRopeStride0Ptr <= 0,
                OP_LOGE(context, "cache dim0 strides must be positive."), return ge::GRAPH_FAILED);

    const int64_t tokenNum = hiddenShape.GetDim(0);
    const int64_t hiddenStateDim = hiddenShape.GetDim(1);
    const int64_t headNum = wukShape.GetDim(0);
    const int64_t qkNopeHeadDim = wukShape.GetDim(1);
    const int64_t kvLoraRank = wukShape.GetDim(2);
    const int64_t qLoraRank = gamma1Shape.GetDim(0);
    OP_CHECK_IF(!IsPositiveU32(tokenNum) || !IsPositiveU32(hiddenStateDim) || !IsPositiveU32(headNum) ||
                    !IsPositiveU32(qkNopeHeadDim) || !IsPositiveU32(kvLoraRank) || !IsPositiveU32(qLoraRank),
                OP_LOGE(context, "MLA model and token dimensions must be positive uint32 values."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokenNum > static_cast<int64_t>(mlapo::MAX_SUPPORT_TOKEN_NUMS),
                OP_LOGE(context, "token count must be in [1, %u], got %ld.",
                        mlapo::MAX_SUPPORT_TOKEN_NUMS, tokenNum),
                return ge::GRAPH_FAILED);

    const int64_t kvCacheBlockSize = IsPhysicalNzCache(kvCacheShape, cacheMode)
                                         ? kvCacheShape.GetDim(2)
                                         : kvCacheShape.GetDim(1);
    int64_t qkRopeHeadDim = kvCacheRopeShape.GetDim(kvCacheRopeShape.GetDimNum() - 1);
    if (IsPhysicalNzCache(kvCacheRopeShape, cacheMode)) {
        qkRopeHeadDim = kvCacheRopeShape.GetDim(1) * kvCacheRopeShape.GetDim(3);
    }
    OP_CHECK_IF(!IsPositiveU32(kvCacheBlockSize) || !IsPositiveU32(qkRopeHeadDim),
                OP_LOGE(context, "cache block size and RoPE head dimension must be positive uint32 values."),
                return ge::GRAPH_FAILED);

    const auto *hiddenDesc = context->GetInputDesc(INPUT_HIDDEN_STATE);
    const auto *wdqkvDesc = context->GetInputDesc(INPUT_WDQKV);
    OP_CHECK_NULL_WITH_CONTEXT(context, hiddenDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, wdqkvDesc);
    const ge::DataType hiddenDtype = hiddenDesc->GetDataType();
    OP_CHECK_IF(hiddenDtype != ge::DT_FLOAT16 && hiddenDtype != ge::DT_BF16,
                OP_LOGE(context, "hiddenState must be float16 or bfloat16."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(quantMode == static_cast<int64_t>(mlapo::QuantMode::PER_TOKEN_SYMM_QUANT) &&
                    hiddenDtype != ge::DT_BF16,
                OP_LOGE(context, "per_token_quant_symm supports only bfloat16 hiddenState."),
                return ge::GRAPH_FAILED);

    opParam.hiddenStateDim = static_cast<uint32_t>(hiddenStateDim);
    opParam.N = static_cast<uint32_t>(tokenNum);
    opParam.headNum = static_cast<uint32_t>(headNum);
    opParam.cacheMode = static_cast<int32_t>(cacheMode);
    opParam.quantMode = static_cast<mlapo::QuantMode>(quantMode);
    opParam.inDtype = hiddenDtype == ge::DT_BF16 ? mlapo::OpParam::InputDtype::BFLOAT16
                                                 : mlapo::OpParam::InputDtype::FLOAT16;
    opParam.enableInnerOut = *enableInnerOutPtr;
    opParam.enableRope = *enableRopePtr;
    opParam.qLoraRank = static_cast<uint32_t>(qLoraRank);
    opParam.qkNopeHeadDim = static_cast<uint32_t>(qkNopeHeadDim);
    opParam.qkRopeHeadDim = static_cast<uint32_t>(qkRopeHeadDim);
    opParam.kvLoraRank = static_cast<uint32_t>(kvLoraRank);
    opParam.kvCacheBlockSize = static_cast<uint64_t>(kvCacheBlockSize);
    opParam.kvCacheStride0 = static_cast<uint64_t>(*kvCacheStride0Ptr);
    opParam.kvCacheRopeStride0 = static_cast<uint64_t>(*kvCacheRopeStride0Ptr);
    const ge::DataType weightDtype = wdqkvDesc->GetDataType();
    opParam.isWeightQuantized = weightDtype == ge::DT_FLOAT16 || weightDtype == ge::DT_BF16 ? 0U : 1U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillPlatformInfo(gert::TilingContext *context, mlapo::PlatformInfo &platformInfo,
                                 uint64_t &systemWorkspaceSize)
{
    fe::PlatFormInfos *platform = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platform);
    const platform_ascendc::PlatformAscendC ascendcPlatform(platform);
    platformInfo.coreNum = ascendcPlatform.GetCoreNum();
    platformInfo.coreNumAic = ascendcPlatform.GetCoreNumAic();
    platformInfo.coreNumAiv = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo.l0cSize);
    systemWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_CHECK_IF(platformInfo.coreNumAic == 0 || platformInfo.coreNumAiv == 0 || platformInfo.ubSize == 0 ||
                    platformInfo.l0cSize == 0,
                OP_LOGE(context, "invalid platform information for MLA preprocessing."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMlaPreprocess(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MlaPreprocess", "TilingContext is null."),
                return ge::GRAPH_FAILED);

    mlapo::OpParam opParam{};
    OP_CHECK_IF(ParseShapesAndAttrs(context, opParam) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to parse MLA shapes, dtypes or attrs."),
                return ge::GRAPH_FAILED);

    mlapo::PlatformInfo platformInfo{};
    uint64_t systemWorkspaceSize = 0;
    OP_CHECK_IF(FillPlatformInfo(context, platformInfo, systemWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "failed to query platform information."), return ge::GRAPH_FAILED);

    MlaTilingData *tilingData = context->GetTilingData<MlaTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    *tilingData = MlaTilingData{};
    mlapo::MlaPreprocessTiling mlaTiling(platformInfo, opParam, tilingData);
    mlaTiling.Init();

    OP_CHECK_IF(tilingData->userWorkspaceSize > std::numeric_limits<uint64_t>::max() - systemWorkspaceSize,
                OP_LOGE(context, "workspace size overflow."), return ge::GRAPH_FAILED);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = static_cast<size_t>(systemWorkspaceSize + tilingData->userWorkspaceSize);
    context->SetBlockDim(platformInfo.coreNumAic);
    // The MLA dispatch key remains in MlaTilingData::tilingKey. The CANN key
    // selects the single generic mixed AIC/AIV entry emitted by the kernel.
    context->SetTilingKey(GENERIC_KERNEL_TILING_KEY);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseMlaPreprocess(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(MlaPreprocess)
    .Tiling(TilingMlaPreprocess)
    .TilingParse<MlaPreprocessCompileInfo>(TilingParseMlaPreprocess);

}  // namespace optiling
