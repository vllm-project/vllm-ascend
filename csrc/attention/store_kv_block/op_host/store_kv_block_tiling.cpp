/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "store_kv_block_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {

constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr int32_t MAX_UB_USE_SIZE = 180 * 1024;

struct StoreKVBlockParams {
    uint32_t numTokens{0};
    uint32_t numCache{0};
    uint32_t numHeads{1};
    uint32_t headSize[5]{1, 1, 1, 1, 1};
    uint32_t blockTableSize{0};
    uint32_t typeByte{0};
    uint32_t tokenSize{1};
    uint32_t tilingKey{0};
    uint64_t workspaceSize{0};
    uint64_t groupInfoLen{0};
    uint32_t corepernum{0};
    uint32_t coretail{0};
    uint64_t sysWorkspaceSize{0};
    uint32_t coreNum{0};
};

static ge::graphStatus DoCommonTiling(gert::TilingContext* context, StoreKVBlockParams& params) {
    auto kShape = context->GetInputShape(DIM_0);
    auto kDimNum = kShape->GetStorageShape().GetDimNum();
    if (kDimNum < 2 || kDimNum > 7) {
        OP_LOGE(context->GetNodeName(), "StoreKVBlock Input kDimNum dim < 2 || kDimNum>7");
        return ge::GRAPH_FAILED;
    }

    for (int i = 0; i < kDimNum; i++) {
        if (i == 0) params.numTokens = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(i));
        else if (i == 1) params.numHeads = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(i));
        else if (static_cast<uint32_t>(kShape->GetStorageShape().GetDim(i)) != 0)
            params.headSize[i - 2] = static_cast<uint32_t>(kShape->GetStorageShape().GetDim(i));
    }

    auto kCacheShape = context->GetInputShape(DIM_1);
    auto kCacheDimNum = kCacheShape->GetStorageShape().GetDimNum();
    if (kCacheDimNum < 2 || kCacheDimNum > 7) {
        OP_LOGE(context->GetNodeName(), "StoreKVBlock Input kCacheDimNum < 2");
        return ge::GRAPH_FAILED;
    }
    params.numCache = kCacheShape->GetStorageShape().GetDim(0) * kCacheShape->GetStorageShape().GetDim(1);

    const int64_t* blockSizePtr = context->GetAttrs()->GetInt(0);
    uint32_t blockSize = static_cast<uint32_t>(*blockSizePtr);
    params.tokenSize = params.numHeads * params.headSize[0] * params.headSize[1] * params.headSize[2] * params.headSize[3] * params.headSize[4];
    params.blockTableSize = blockSize;

    uint32_t typeByte = 0;
    auto xDataType = context->GetInputDesc(DIM_0)->GetDataType();
    if (xDataType == ge::DataType::DT_INT8) {
        typeByte = sizeof(int8_t);
        params.tilingKey = 1;
    } else if (xDataType == ge::DataType::DT_FLOAT16 || xDataType == ge::DataType::DT_BF16) {
        typeByte = sizeof(uint16_t);
        params.tilingKey = 2;
    } else if (xDataType == ge::DataType::DT_INT32 || xDataType == ge::DataType::DT_UINT32) {
        typeByte = sizeof(uint32_t);
        params.tilingKey = 4;
    } else {
        OP_LOGE(context->GetNodeName(), "Unsupported type.");
        return ge::GRAPH_FAILED;
    }

    params.typeByte = typeByte;

    auto groupInfoShape = context->GetInputShape(DIM_2);
    params.groupInfoLen = static_cast<uint32_t>(groupInfoShape->GetStorageShape().GetDim(0));
    params.corepernum = params.groupInfoLen / params.coreNum;
    params.coretail = params.groupInfoLen % params.coreNum;

    uint32_t pageBlockEleSize = params.blockTableSize * params.tokenSize;
    if (pageBlockEleSize > MAX_UB_USE_SIZE) {
        OP_LOGE(context->GetNodeName(), "pageBlockEleSize > MaxUBSize");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus StoreKVBlockTilingFunc(gert::TilingContext* context) {
    StoreKVBlockParams params;

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    params.coreNum = ascendcPlatform.GetCoreNum();
    if (params.coreNum == 0) {
        OP_LOGE(context->GetNodeName(), "Failed to get core num.");
        return ge::GRAPH_FAILED;
    }
    params.sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    auto ret = DoCommonTiling(context, params);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    StoreKVBlockTilingData tilingData;
    if (params.blockTableSize > 0) tilingData.set_blockTableSize(params.blockTableSize);
    if (params.typeByte > 0) tilingData.set_typeByte(params.typeByte);
    if (params.tokenSize > 0) tilingData.set_tokenSize(params.tokenSize);
    if (params.corepernum > 0 || params.coretail != 0) tilingData.set_corePerNum(params.corepernum);
    if (params.coretail < 48) tilingData.set_coreTail(params.coretail);
    if (params.numTokens > 0) tilingData.set_numTokens(params.numTokens);
    if (params.numCache > 0) tilingData.set_numCache(params.numCache);
    if (params.groupInfoLen > 0) tilingData.set_groupInfoLen(params.groupInfoLen);

    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    *workspaceSize = params.workspaceSize + params.sysWorkspaceSize;
    context->SetTilingKey(params.tilingKey);
    if (params.coreNum > 0) context->SetBlockDim(params.coreNum);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForStoreKVBlock(gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StoreKVBlock)
    .Tiling(StoreKVBlockTilingFunc)
    .TilingParse<StoreKVBlockCompileInfo>(TilingParseForStoreKVBlock);

} // namespace optiling
