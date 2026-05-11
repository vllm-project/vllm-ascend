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
 * \file gather_pa_kv_cache_vllm.cc
 * \brief
 */

#include <climits>
#include "log/ops_log.h"
#include "platform/platform_info_def.h"
#include "tiling/tiling_api.h"
#include "gather_pa_kv_cache_vllm_tiling.h"

using namespace Ops::Transformer::OpTiling;
namespace optiling {
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
constexpr uint32_t DIM_6 = 6;
constexpr uint32_t CACHE_MODE_INDEX = 0;
constexpr uint32_t IS_SEQ_LENS_CUNSUM_INDEX = 1;
constexpr uint32_t KV_CACHE_STRIDES_INDEX = 2;
constexpr uint64_t ASCENDC_TOOLS_WORKSPACE = static_cast<uint64_t>(16) * 1024 * 1024;
static constexpr uint32_t TILING_KEY_NZ = 577;
static constexpr uint32_t TILING_KEY_ND = 617;

void PrintTilingDate(gert::TilingContext *context, GatherPaKvCacheVllmTilingData *tilingDataPtr)
{
    OP_LOGD(context, "Start GatherPaKvCacheVllmTilingData priting");
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "blockSize is %u", tilingDataPtr->get_blockSize());
    OP_LOGD(context, "numTokens is %u", tilingDataPtr->get_numTokens());
    OP_LOGD(context, "numblkTabCol is %u", tilingDataPtr->get_numblkTabCol());
    OP_LOGD(context, "tokenSizeK is %u", tilingDataPtr->get_tokenSizeK());
    OP_LOGD(context, "tokenSizeV is %u", tilingDataPtr->get_tokenSizeV());
    OP_LOGD(context, "typeByte is %u", tilingDataPtr->get_typeByte());
    OP_LOGD(context, "hasSeqStarts is %u", tilingDataPtr->get_hasSeqStarts());
    OP_LOGD(context, "isSeqLensCumsum is %u", tilingDataPtr->get_isSeqLensCumsum());
    OP_LOGD(context, "kCacheBlockStride is %ld", tilingDataPtr->get_kCacheBlockStride());
    OP_LOGD(context, "vCacheBlockStride is %ld", tilingDataPtr->get_vCacheBlockStride());
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "End GatherPaKvCacheVllmTilingData priting");
}

static const std::map<ge::DataType, size_t> MAP_OF_DTYPE_SIZE = {{ge::DT_UNDEFINED, 0},
                                                                 {ge::DT_FLOAT, sizeof(float)},
                                                                 {ge::DT_FLOAT16, sizeof(int16_t)},
                                                                 {ge::DT_INT8, sizeof(int8_t)},
                                                                 {ge::DT_BF16, sizeof(int16_t)}};

size_t GetTensorElementSizes(ge::DataType dType)
{
    auto iter = MAP_OF_DTYPE_SIZE.find(dType);
    return iter->second;
}

uint32_t GatherPaKvCacheVllmGetBlockDim(gert::TilingContext *context)
{
    auto kShape = context->GetOutputShape(DIM_0);
    uint32_t sumContextLens = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(DIM_0));
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t blockDim = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    auto attrs = context->GetAttrs();
    const char *mode = attrs->GetAttrPointer<char>(CACHE_MODE_INDEX);
    if (strcmp(mode, "PA_NZ") == 0) {
        return sumContextLens < blockDim ? sumContextLens : blockDim;
    }
    return blockDim;
}


bool CommonGatherPaKvCacheVllmTiling(gert::TilingContext *context)
{
    auto kCacheShape = context->GetInputShape(DIM_0);
    auto blockTablesShape = context->GetInputShape(DIM_2);
    auto kShape = context->GetOutputShape(DIM_0);
    auto vShape = context->GetOutputShape(DIM_1);
    auto attrs = context->GetAttrs();
    const char *mode = attrs->GetAttrPointer<char>(CACHE_MODE_INDEX);
    OP_CHECK_IF((strcmp(mode, "PA_NZ") != 0 && strcmp(mode, "Norm") != 0),
                OP_LOGE(context, "%s is not supported", mode), return ge::GRAPH_FAILED);
    int32_t blockSize;
    int32_t tokenSizeK;
    int32_t tokenSizeV;
    auto inDtype = context->GetInputDesc(DIM_0)->GetDataType();
    uint32_t typeByte = static_cast<uint32_t>(GetTensorElementSizes(inDtype));
    if (strcmp(mode, "PA_NZ") == 0) {
        blockSize = static_cast<int32_t>(kCacheShape->GetStorageShape().GetDim(DIM_2));
        tokenSizeK = static_cast<int32_t>(kShape->GetStorageShape().GetDim(DIM_1));
        tokenSizeV = static_cast<int32_t>(vShape->GetStorageShape().GetDim(DIM_1));
        context->SetTilingKey(TILING_KEY_NZ);
    } else if (strcmp(mode, "Norm") == 0) {
        blockSize = static_cast<int32_t>(kCacheShape->GetStorageShape().GetDim(DIM_1));
        tokenSizeK =
            static_cast<int32_t>(kShape->GetStorageShape().GetDim(DIM_1) * kShape->GetStorageShape().GetDim(DIM_2));
        tokenSizeV =
            static_cast<int32_t>(vShape->GetStorageShape().GetDim(DIM_1) * vShape->GetStorageShape().GetDim(DIM_2));
        context->SetTilingKey(TILING_KEY_ND + typeByte);
    } else {
        return false;
    }

    int32_t numTokens = static_cast<int32_t>(blockTablesShape->GetStorageShape().GetDim(DIM_0)); // block tables row
    int32_t numblkTabCol =
        static_cast<int32_t>(blockTablesShape->GetStorageShape().GetDim(DIM_1)); // block tables column

    OP_CHECK_IF((blockSize <= 0 || blockSize > INT_MAX), OP_LOGE(context, "blockSize is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((numTokens <= 0 || numTokens > INT_MAX), OP_LOGE(context, "numTokens is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((numblkTabCol <= 0 || numblkTabCol > INT_MAX), OP_LOGE(context, "numblkTabCol is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((tokenSizeK <= 0 || tokenSizeK > INT_MAX), OP_LOGE(context, "tokenSizeK is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((tokenSizeV <= 0 || tokenSizeV > INT_MAX), OP_LOGE(context, "tokenSizeV is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((typeByte <= 0 || typeByte > INT_MAX), OP_LOGE(context, "typeByte is invalid."),
                return ge::GRAPH_FAILED);
    GatherPaKvCacheVllmTilingData tilingData;

    int32_t hasSeqStarts;
    auto seqStartsTensor = context->GetOptionalInputTensor(DIM_6);
    seqStartsTensor == nullptr ? hasSeqStarts = 0 : hasSeqStarts = 1;

    auto isSeqLensCumsum = attrs->GetAttrPointer<bool>(IS_SEQ_LENS_CUNSUM_INDEX);
    tilingData.set_blockSize(blockSize);
    tilingData.set_numTokens(numTokens);
    tilingData.set_numblkTabCol(numblkTabCol);
    tilingData.set_tokenSizeK(tokenSizeK);
    tilingData.set_tokenSizeV(tokenSizeV);
    tilingData.set_typeByte(typeByte);
    tilingData.set_hasSeqStarts(hasSeqStarts);
    tilingData.set_isSeqLensCumsum(*isSeqLensCumsum);

    int64_t kCacheBlockStride = 0;
    int64_t vCacheBlockStride = 0;
    auto kvCacheStrides = attrs->GetListInt(KV_CACHE_STRIDES_INDEX);
    if (kvCacheStrides != nullptr && kvCacheStrides->GetSize() >= 2) {
        kCacheBlockStride = kvCacheStrides->GetData()[0];
        vCacheBlockStride = kvCacheStrides->GetData()[1];
    }
    tilingData.set_kCacheBlockStride(kCacheBlockStride);
    tilingData.set_vCacheBlockStride(vCacheBlockStride);
    
    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    *workspaceSize = ASCENDC_TOOLS_WORKSPACE;
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    PrintTilingDate(context, &tilingData);

    return true;
}

ge::graphStatus Tiling4GatherPaKvCacheVllm(gert::TilingContext *context)
{
    GatherPaKvCacheVllmTilingData tilingData;
    auto ret = CommonGatherPaKvCacheVllmTiling(context);
    OP_CHECK_IF((!ret), OP_LOGE(context, "value is invalid."), return ge::GRAPH_FAILED);
    context->SetBlockDim(GatherPaKvCacheVllmGetBlockDim(context));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4GatherPaKvCacheVllm(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepare4GatherPaKvCacheVllm enter.");
    auto compileInfo = context->GetCompiledInfo<Tiling4GatherPaKvCacheVllmCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                OP_LOGE(context, "Get core num failed, core num: %u", static_cast<uint32_t>(compileInfo->coreNum)),
                return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0),
                OP_LOGE(context, "Get ub size failed, ub size: %u", static_cast<uint32_t>(compileInfo->ubSize)),
                return ge::GRAPH_FAILED);

    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context, "TilingPrepare4GatherPaKvCacheVllm exit.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGatherPaKvCacheVllm(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE("TilingForGatherPaKvCacheVllm", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingForGatherPaKvCacheVllm enter.");

    return Tiling4GatherPaKvCacheVllm(context);
}

ge::graphStatus TilingPrepareForGatherPaKvCacheVllm(gert::TilingParseContext *context)
{
        if (context == nullptr) {
        OP_LOGE("TilingPrepareForGatherPaKvCache", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingPrepareForGatherPaKvCache enter");
    return TilingPrepare4GatherPaKvCacheVllm(context);
}

// register tiling interface of the GatherPaKvCacheVllm op.
IMPL_OP_OPTILING(GatherPaKvCacheVllm)
    .Tiling(TilingForGatherPaKvCacheVllm)
    .TilingParse<GatherPaKvCacheVllmCompileInfo>(TilingPrepareForGatherPaKvCacheVllm);

} // namespace optiling