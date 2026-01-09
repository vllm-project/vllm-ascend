/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file dispatch_ffn_tiling.cpp
 * \brief
 */
#include "vector"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error_log.h"
#include "hcom_topo_info.h"
#include "register/op_def_registry.h"
#include "dispatch_ffn_combine_tiling.h"
#include <vector>
#include <map>
#include <algorithm>
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

using namespace AscendC;
using namespace ge;

namespace {
    // 1. Constant definitions
    const char *K_INNER_DEBUG = "DispatchFFNCombine Tiling Debug";
    constexpr uint32_t ATTR_GROUP_INDEX = 0;
    constexpr uint32_t ATTR_MAX_OUTPUT_SIZE_INDEX = 1;
    constexpr uint32_t ATTR_IS_TRANS_B = 2;
    constexpr uint32_t ATTR_WEIGHT_NZ = 3;
    constexpr uint64_t INIT_TILINGKEY = 1000000;
    constexpr uint64_t TILINGKEY_TRANS_B = 1U;
    constexpr uint64_t TILINGKEY_WEIGHT_NZ = 10;
    constexpr uint32_t X_INDEX = 0;
    constexpr uint32_t WEIGHT_INDEX = 1;
    constexpr uint32_t WEIGHT2_INDEX = 2;
    constexpr uint32_t EXPERTID_INDEX = 3;
    constexpr uint32_t BLOCK_NUM = 20;
    constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
}

namespace optiling {

static int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

// Parse and validate rankId, group, worldSize, and isTransB attributes
static ge::graphStatus DispatchFFNCombineCheckAttrAndSetTiling(gert::TilingContext *context, DispatchFFNCombineInfo& info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    // TODO: set, validate, and print tiling data related to attributes
    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto maxOutputSizePtr = attrs->GetAttrPointer<int>(ATTR_MAX_OUTPUT_SIZE_INDEX);
    auto is_trans_b = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_B);
    auto weight_nz = attrs->GetAttrPointer<bool>(ATTR_WEIGHT_NZ);
    OP_TILING_CHECK(groupPtr == nullptr || strlen(groupPtr) == 0,
    OP_LOGE(K_INNER_DEBUG, "group is invalid."), return GRAPH_FAILED);

    OP_TILING_CHECK(is_trans_b == nullptr,
        OP_LOGE(K_INNER_DEBUG, "is_trans_b is invalid."), return GRAPH_FAILED);
    OP_TILING_CHECK(weight_nz == nullptr,
        OP_LOGE(K_INNER_DEBUG, "weight_nz is invalid."), return GRAPH_FAILED);

    info.maxOutputSize = *maxOutputSizePtr;
    info.isTransposeB = *is_trans_b;
    info.isWeightNz = *weight_nz;

    int64_t rankSize;
    (void)ge::HcomTopoInfo::Instance().GetGroupRankSize(groupPtr, rankSize);
    info.worldSize = rankSize;

    OP_LOGD(K_INNER_DEBUG, "maxOutputSize=%d ", info.maxOutputSize);
    OP_LOGD(K_INNER_DEBUG, "rankSize=%d ", info.worldSize);

    return ge::GRAPH_SUCCESS;
}

// Extract shapes of input tensors A and B to compute M, K, N
static ge::graphStatus DispatchFFNCombineCheckShapeAndSetTiling(gert::TilingContext *context, DispatchFFNCombineInfo &info)
{
    const char *nodeName = context->GetNodeName();

    const gert::StorageShape *aStorageShape = context->GetInputShape(X_INDEX);
    auto expertIdxTensor = context->GetDynamicInputTensor(EXPERTID_INDEX, 0);
    uint32_t M = aStorageShape->GetStorageShape().GetDim(0);
    uint32_t K = aStorageShape->GetStorageShape().GetDim(1);

    auto wTensor = context->GetDynamicInputTensor(WEIGHT_INDEX, 0);
    uint32_t wTensorDims = wTensor->GetOriginShape().GetDimNum();
    uint32_t N = wTensor->GetStorageShape().GetDim(wTensorDims - 1);

    uint32_t topK = expertIdxTensor->GetStorageShape().GetDim(1);
    uint32_t listLen = 0;
    while (true) {
        auto wTensorT = context->GetDynamicInputTensor(WEIGHT_INDEX, ++listLen);
        if (wTensorT == nullptr) {break;}
    }

    uint32_t expertPerRank;
    if (listLen == 1) {
        expertPerRank = wTensor->GetStorageShape().GetDim(0);
    } else {
        expertPerRank = listLen;
    }

    info.M = M;
    info.N = N;
    info.K = K;
    info.expertPerRank = expertPerRank;
    info.topK = topK;
    info.listLen = listLen;
    OP_LOGD(K_INNER_DEBUG, "M=%d ", info.M);
    OP_LOGD(K_INNER_DEBUG, "K=%d ", info.K);
    OP_LOGD(K_INNER_DEBUG, "N=%d ", info.N);
    OP_LOGD(K_INNER_DEBUG, "expertPerRank=%d ", info.expertPerRank);
    OP_LOGD(K_INNER_DEBUG, "topK=%d ", info.topK);
    OP_LOGD(K_INNER_DEBUG, "listLen=%d ", info.listLen);

    return ge::GRAPH_SUCCESS;
}

// Get hardware info such as AI Core count and UB capacity for the current chip platform.
static ge::graphStatus DispatchFFNCombineGetPlatformInfoAndSetTiling(gert::TilingContext *context, DispatchFFNCombineInfo& info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(K_INNER_DEBUG, "aivNum=%d", info.aivNum);
    OP_LOGD(K_INNER_DEBUG, "ubSize=%lu", info.totalUbSize);

    return ge::GRAPH_SUCCESS;
}

void SetTilingData(CoCTiling &cocTilingData, DispatchFFNCombineInfo &info)
{
    cocTilingData.m0 = 128;
    cocTilingData.k0 = 256;
    cocTilingData.n0 = 256;
    cocTilingData.swizzleDirect = 1;
    cocTilingData.swizzleOffset = 7;
    cocTilingData.ubMoveNum = 16 * 1024;
    cocTilingData.pValue = 1;
    cocTilingData.commNpuSplit = info.worldSize;
    cocTilingData.commDataSplit = 1;
    cocTilingData.lenPerLoop = cocTilingData.m0 * cocTilingData.n0 / 2;
}

// Main scheduling function:
// Get tilingData ➝ check Attr ➝ check Shape ➝ get platform info 
// ➝ call SetTilingData (based on rank count) ➝ set blockDim ➝ set tilingKey ➝ set workspace ➝ configure communication parameters

static ge::graphStatus DispatchFFNCombineTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter DispatchFFNCombine tiling func.");

    // 1. tilingData
    DispatchFFNCombineTilingData *tilingData = context->GetTilingData<DispatchFFNCombineTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "DispatchFFNCombine get tilingData.");
    DispatchFFNCombineInfo& info = tilingData->dispatchFFNCombineInfo;
    OP_LOGI(nodeName, "DispatchFFNCombine get tilingData info.");

    OP_TILING_CHECK(DispatchFFNCombineCheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombine CheckAttrAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineCheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombine CheckShapeAndSetTiling Failed"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineGetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "DispatchFFNCombine GetPlatformInfoAndSetTiling Failed"),
        return ge::GRAPH_FAILED);

    SetTilingData(tilingData->cocTiling, info);

    // 2. set blockDim
    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto aicNum = ascendcPlatform.GetCoreNumAic();
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // 3. set tilingKey
    uint64_t tilingKey = INIT_TILINGKEY;
    tilingKey += info.isTransposeB ? TILINGKEY_TRANS_B : 0;
    tilingKey += info.isWeightNz ? TILINGKEY_WEIGHT_NZ : 0;
    context->SetTilingKey(tilingKey);

    OP_LOGD(K_INNER_DEBUG, "tilingKey=%d", tilingKey);

    optiling::MoeInitRoutingQuantV2TilingBase moeInitRoutingQuantV2TilingBase;
    int64_t inuptXDtypeSize = sizeof(int16_t);
    int64_t scaleDim0 = 0;
    int64_t ubSize = 196352;
    int64_t expertCapacity = 0;
    int64_t expertNum = info.expertPerRank * info.worldSize;
    int64_t activeNum = 0;
    int64_t dropPadMode = 0;
     int64_t expertTokensCountOrCumsumFlag = 2;
     bool expertTokensBeforeCapacityFlag = false;
     int64_t quantMode = 1;
     uint32_t aivNumInitRouting = 2 * BLOCK_NUM;
    moeInitRoutingQuantV2TilingBase.DoTiling(info.M, info.K, info.topK, expertCapacity, expertNum, activeNum, dropPadMode, 
        expertTokensCountOrCumsumFlag, expertTokensBeforeCapacityFlag, inuptXDtypeSize, quantMode, scaleDim0, aivNumInitRouting, ubSize);
    uint64_t initRoutingQuantTilingKey = moeInitRoutingQuantV2TilingBase.tilingKey_;
    size_t initRoutingWorkspace = moeInitRoutingQuantV2TilingBase.workspaceSize_;

    tilingData->cocTiling.moeInitRoutingQuantV2TilingData = moeInitRoutingQuantV2TilingBase.quantTilingData;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vbsComputeParamsOp = moeInitRoutingQuantV2TilingBase.quantTilingData.vbsComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.vmsMiddleComputeParamsOp = moeInitRoutingQuantV2TilingBase.quantTilingData.vmsMiddleComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.sortOutComputeParamsOp = moeInitRoutingQuantV2TilingBase.quantTilingData.sortOutComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstComputeParamsOp = moeInitRoutingQuantV2TilingBase.quantTilingData.srcToDstComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.srcToDstCapacityComputeParamsOp = moeInitRoutingQuantV2TilingBase.quantTilingData.srcToDstCapacityComputeParamsOp;
    tilingData->cocTiling.moeInitRoutingQuantV2TilingData.gatherOutComputeParamsOp = moeInitRoutingQuantV2TilingBase.quantTilingData.gatherOutComputeParamsOp;
    tilingData->cocTiling.initRoutingQuantTilingKey = initRoutingQuantTilingKey;

    // 4. workspace
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);

    uint32_t n2 = info.K;
    uint32_t k2 = info.N / 2;

    uint64_t cocWorkspace = (info.M + 256 - 1) / 256 * 256 * info.topK *sizeof(int32_t) +
                            info.worldSize * info.worldSize * info.expertPerRank * sizeof(int32_t) * 3 +
                            info.maxOutputSize * sizeof(float) * 2 +
                            std::max(info.maxOutputSize * info.N * sizeof(int16_t), info.maxOutputSize * n2 * sizeof(int16_t)) +
                            std::max(info.maxOutputSize * info.K * sizeof(int8_t), info.maxOutputSize * k2 * sizeof(int8_t));

    workSpaces[0] = SYSTEM_NEED_WORKSPACE + std::max(cocWorkspace, initRoutingWorkspace);


    // 5. communication
    auto attrs = context->GetAttrs();
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    uint32_t opType = 8U;
    std::string algConfig = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfig);
    mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);

    OP_LOGI(nodeName, "Leave DispatchFFNCombine tiling func.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineTilingFunc(gert::TilingContext* context)
{
    return DispatchFFNCombineTilingFuncImpl(context);
}

struct DispatchFFNCombineCompileInfo {};
ge::graphStatus TilingParseForDispatchFFNCombine(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DispatchFFNCombine)
    .Tiling(DispatchFFNCombineTilingFunc)
    .TilingParse<DispatchFFNCombineCompileInfo>(TilingParseForDispatchFFNCombine);
} // namespace optiling
