/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_tiling.cpp
 * \brief
 */

#include <vector>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "../mega_moe.h"
#include "../../../op_kernel/arch35/mega_moe_tiling.h"
#include "../../../op_kernel/arch35/mega_moe_tiling_key.h"
#include "../../../op_kernel/arch35/mega_moe_workspace_info.h"

using namespace Mc2Tiling;
using namespace AscendC;
using namespace ge;

namespace optiling {
namespace {
const static int64_t NUM_TWO = 2LL;
const static int64_t NUM_FOUR = 4LL;
const static int64_t UB_BLOCK_SIZE = 32LL;

const static int64_t FOUR_DIMS = 4LL;
const static int64_t THREE_DIMS = 3LL;
const static int64_t TWO_DIMS = 2LL;
const static int64_t ONE_DIM = 1LL;
const static int64_t MIN_TOPK = 1LL;
const static int64_t MAX_TOPK = 32LL;
const static int64_t MIN_EXPERT_PER_RANK = 1LL;
const static int64_t MAX_EXPERT_PER_RANK = 1024LL;
const static int64_t MIN_H = 1024LL;
const static int64_t MAX_H = 8LL * 1024LL; // 8K
const static int64_t H_ALIGN = 32LL;
const static int64_t W4_K_ALIGN = 64LL;
const static int64_t HIDDEN_DIM_BASE = 1024LL;
const static int64_t MAX_HIDDEN_DIM = 8LL * 1024LL; // 8K
const static int64_t MIN_EP_WORLD_SIZE = 2LL;
const static int64_t MAX_EP_WORLD_SIZE = 1024LL;
const static int64_t MAX_MOE_EXPERT_NUM = 2048LL;
const static int64_t INPUT_WEIGHT_SCALES_CEIL_ALIGN = 64LL;
const static int64_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 50LL;
constexpr float DEFAULT_ACTIVATION_CLAMP = std::numeric_limits<float>::max();

constexpr uint32_t GMM1_TILE_M = 256U;
constexpr uint32_t GMM2_GM_TILE_M = 256U;
constexpr uint32_t GMM_TILE_N = 256U;

uint32_t CalcExpertsPerBatch(const MegaMoeTilingData *tilingData, uint32_t aicNum)
{
    uint32_t moeExpertPerRank = tilingData->moeExpertPerRank;
    if (moeExpertPerRank == 0U || tilingData->h == 0U || aicNum == 0U) {
        return 1U;
    }

    uint64_t expectedTokens =
        ops::CeilDiv<uint64_t>(
            static_cast<uint64_t>(tilingData->bs) * static_cast<uint64_t>(tilingData->topK), moeExpertPerRank);
    uint64_t expectedMWaves =
        std::max<uint64_t>(1U, ops::CeilDiv<uint64_t>(expectedTokens, GMM1_TILE_M));
    // One GMM1 scheduler N step covers gate+up, hence 2 * GMM_TILE_N output columns.
    uint64_t gmm1TilesPerExpert =
        expectedMWaves *
        ops::CeilDiv<uint64_t>(tilingData->hiddenDim, static_cast<uint64_t>(GMM_TILE_N) * 2U);
    uint64_t gmm2TilesPerExpert =
        ops::CeilDiv<uint64_t>(expectedTokens, GMM2_GM_TILE_M) *
        ops::CeilDiv<uint64_t>(tilingData->h, GMM_TILE_N);
    uint64_t limitingTilesPerExpert =
        std::max<uint64_t>(1U, std::min(gmm1TilesPerExpert, gmm2TilesPerExpert));
    uint64_t expertsToFillAic = ops::CeilDiv<uint64_t>(aicNum, limitingTilesPerExpert);
    uint64_t expertsPerBatch =
        std::max<uint64_t>(1U, std::min<uint64_t>(moeExpertPerRank, expertsToFillAic));
    // Deepen the wave when GMM1's gate+up width exceeds GMM2's output width.
    expertsPerBatch *= ops::CeilDiv<uint64_t>(tilingData->hiddenDim, tilingData->h);
    return static_cast<uint32_t>(
        std::max<uint64_t>(1U, std::min<uint64_t>(moeExpertPerRank, expertsPerBatch)));
}

const static uint32_t WEIGHT_MATRIX_ROW_DIM_INDEX = 0U;
const static uint32_t WEIGHT_MATRIX_COLUMN_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MATRIX_DIM_INDEX = 0U;
const static uint32_t WEIGHT_SCALE_GROUP_DIM_INDEX = 1U;
const static uint32_t WEIGHT_SCALE_MULTI_BASE_DIM_INDEX = 2U;

/*
 * 统计指定动态输入 (tensor list) 中的 tensor 数量。
 */
static uint32_t GetDynamicInputTensorCount(const gert::TilingContext *context, uint32_t inputIndex)
{
    uint32_t tensorCount = 0;
    while (context->GetDynamicInputShape(inputIndex, tensorCount) != nullptr) {
        ++tensorCount;
    }
    return tensorCount;
}

/*
 * 按权重布局返回专家数：逐专家布局取 TensorList 长度，堆叠布局取首个 tensor 的 dim0。
 * 可选输入未传入或首维为 0 时返回 0。
 */
static int64_t GetWeightExpertCount(const gert::TilingContext *context, uint32_t inputIndex,
                                    bool isPerExpertWeightTensor)
{
    const auto *firstTensorShape = context->GetDynamicInputShape(inputIndex, 0);
    if (firstTensorShape == nullptr || firstTensorShape->GetStorageShape().GetDimNum() == 0U ||
        firstTensorShape->GetStorageShape().GetDim(0) <= 0) {
        return 0;
    }
    if (isPerExpertWeightTensor) {
        return static_cast<int64_t>(GetDynamicInputTensorCount(context, inputIndex));
    }
    return firstTensorShape->GetStorageShape().GetDim(0);
}

/*
 * 以单专家视图读取指定维度；堆叠布局会自动跳过最外层的专家维。
 */
static int64_t GetSingleExpertTensorDimSize(const gert::StorageShape *tensorShape, uint32_t singleExpertDimIndex,
                                            bool isPerExpertTensor)
{
    uint32_t expertDimOffset = isPerExpertTensor ? 0U : 1U;
    return tensorShape->GetStorageShape().GetDim(expertDimOffset + singleExpertDimIndex);
}
} // namespace

void PrintMegaMoeTilingData(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);

    OP_LOGD(nodeName, "========== MegaMoeTilingData ==========");
    OP_LOGD(nodeName, "shape: bs=%u, h=%u, hiddenDim=%u, topK=%u, maxOutputSize=%u, isPerExpertWeightTensor=%d",
            tilingData->bs, tilingData->h, tilingData->hiddenDim, tilingData->topK, tilingData->maxOutputSize,
            tilingData->isPerExpertWeightTensor);
    OP_LOGD(nodeName,
            "topology: moeExpertPerRank=%u, sharedExpertNum=%u, epWorldSize=%u, aicNum=%u, blockAivNum=%u, "
            "blockNumPerEP=%u, topoType=%ld",
            tilingData->moeExpertPerRank, tilingData->sharedExpertNum, tilingData->epWorldSize, tilingData->aicNum,
            tilingData->blockAivNum, tilingData->blockNumPerEP, tilingData->topoType);
    OP_LOGD(nodeName, "mode: groupedMatmulMode=%u, combineQuantMode=%ld, clampLimit=%f",
            static_cast<uint32_t>(tilingData->groupedMatmulMode), tilingData->combineQuantMode, tilingData->clampLimit);
    OP_LOGD(nodeName, "combineSync: slotCountPerExpert=%lu", tilingData->combineSyncSlotCountPerExpert);

    const auto &dispatchConfig = tilingData->dispatchBufferConfig;
    OP_LOGD(nodeName, "dispatch: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, copyBufferBytes=%u",
            dispatchConfig.routeItemsPerBatch, dispatchConfig.routeBatchCount, dispatchConfig.bufferCount,
            dispatchConfig.copyBufferBytes);

    const auto &sendMaskConfigWithExtraExpert = tilingData->sendMaskConfigForCoreWithExtraExpert;
    const auto &sendMaskConfigWithoutExtraExpert = tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    OP_LOGD(nodeName, "sendMask: coreCountWithExtraExpert=%u", tilingData->sendMaskCoreCountWithExtraExpert);
    OP_LOGD(nodeName,
            "sendMaskWithExtraExpert: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, bufferBytes=%u",
            sendMaskConfigWithExtraExpert.routeItemsPerBatch, sendMaskConfigWithExtraExpert.routeBatchCount,
            sendMaskConfigWithExtraExpert.bufferCount, sendMaskConfigWithExtraExpert.bufferBytes);
    OP_LOGD(nodeName,
            "sendMaskWithoutExtraExpert: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, bufferBytes=%u",
            sendMaskConfigWithoutExtraExpert.routeItemsPerBatch, sendMaskConfigWithoutExtraExpert.routeBatchCount,
            sendMaskConfigWithoutExtraExpert.bufferCount, sendMaskConfigWithoutExtraExpert.bufferBytes);

    const auto &unpermuteFullChunkConfig = tilingData->unpermuteConfigForFullTokenChunk;
    const auto &unpermuteTailChunkConfig = tilingData->unpermuteConfigForTailTokenChunk;
    OP_LOGD(nodeName, "unpermute: fullTokenChunkCoreCount=%u", tilingData->unpermuteFullTokenChunkCoreCount);
    OP_LOGD(nodeName,
            "unpermuteFullChunk: tokensPerBatch=%d, inputBufferCount=%d, bf16SlotElements=%u, fp32SlotElements=%u, "
            "weightBufferBytes=%u, conversionBufferBytes=%u",
            unpermuteFullChunkConfig.tokensPerBatch, unpermuteFullChunkConfig.inputBufferCount,
            unpermuteFullChunkConfig.bf16SlotElementCount, unpermuteFullChunkConfig.fp32SlotElementCount,
            unpermuteFullChunkConfig.topKWeightsBufferBytes, unpermuteFullChunkConfig.topKWeightsConversionBufferBytes);
    OP_LOGD(nodeName,
            "unpermuteTailChunk: tokensPerBatch=%d, inputBufferCount=%d, bf16SlotElements=%u, fp32SlotElements=%u, "
            "weightBufferBytes=%u, conversionBufferBytes=%u",
            unpermuteTailChunkConfig.tokensPerBatch, unpermuteTailChunkConfig.inputBufferCount,
            unpermuteTailChunkConfig.bf16SlotElementCount, unpermuteTailChunkConfig.fp32SlotElementCount,
            unpermuteTailChunkConfig.topKWeightsBufferBytes, unpermuteTailChunkConfig.topKWeightsConversionBufferBytes);
    OP_LOGD(nodeName, "topkWeightsPrefetch is %d", tilingData->topkWeightsPrefetch);
    OP_LOGD(nodeName, "expertsPerBatch is %u", tilingData->expertsPerBatch);
}

void PrintWorkspaceInfo(const struct WorkspaceInfo *info, const char *nodeName)
{
    OP_LOGD(nodeName, "dispatchRevDataPtr:         %ld\n", info->dispatchRevDataPtr);
    OP_LOGD(nodeName, "dispatchRevScalePtr:        %ld\n", info->dispatchRevScalePtr);
    OP_LOGD(nodeName, "swigluQuantDataPtr:         %ld\n", info->swigluQuantDataPtr);
    OP_LOGD(nodeName, "swigluQuantScalePtr:        %ld\n", info->swigluQuantScalePtr);
    OP_LOGD(nodeName, "expertRevTokenNumsPtr:      %ld\n", info->expertRevTokenNumsPtr);
    OP_LOGD(nodeName, "metaInfoPtr:                %ld\n", info->metaInfoPtr);
    OP_LOGD(nodeName, "flagSwiGluToGmm2Ptr:        %ld\n", info->flagSwiGluToGmm2Ptr);
    OP_LOGD(nodeName, "flagDispatchToGmm1Ptr:      %ld\n", info->flagDispatchToGmm1Ptr);
    OP_LOGD(nodeName, "flagSendCntCalToUpdParamsPtr:      %ld\n", info->flagSendCntCalToUpdParamsPtr);
    OP_LOGD(nodeName, "flagGmmToEpiloguePtr: %ld\n", info->flagGmmToEpiloguePtr);
    OP_LOGD(nodeName, "gmm2ReadyPtr:               %ld\n", info->gmm2ReadyPtr);
    OP_LOGD(nodeName, "gmm2CombineSyncCounterPtr:  %ld\n", info->gmm2CombineSyncCounterPtr);
    OP_LOGD(nodeName, "gmm2MmadResPtr:             %ld\n", info->gmm2MmadResPtr);
    OP_LOGD(nodeName, "workspaceSize:              %ld\n", info->workspaceSize);
}

void PrintPeermemInfo(const MegaMoeTilingData *tilingData, const char *nodeName)
{
    OP_LOGD(nodeName, "========== PeermemInfo ==========");
    int64_t rankSyncInWorldSize = PEERMEM_DATA_OFFSET;
    OP_LOGD(nodeName, "rankSyncInWorldSize: {%ld}\n", rankSyncInWorldSize);
    int64_t sendTotalNum = static_cast<int64_t>(tilingData->bs) * tilingData->topK;
    int64_t compareCount =
        ops::CeilAlign(sendTotalNum * (int64_t)sizeof(int32_t), (int64_t)ALIGN_256) / (int64_t)sizeof(int32_t);
    int64_t maskAlignSize = ops::CeilAlign(compareCount / 8, (int64_t)ALIGN_32);
    int64_t maskSlotSize = maskAlignSize + (int64_t)ALIGN_32; // mask + 32B count
    int64_t maskRecvSize = ops::CeilAlign(
        (int64_t)tilingData->moeExpertPerRank * tilingData->epWorldSize * maskSlotSize, (int64_t)ALIGN_512);
    OP_LOGD(nodeName, "maskRecvSize: {%ld}\n", maskRecvSize);
    uint32_t mxScaleNum = ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32));
    uint32_t dataBytes = ops::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_256)) * sizeof(int8_t);
    uint32_t scaleAlignBytes =
        ops::CeilAlign(mxScaleNum * static_cast<uint32_t>(sizeof(int8_t)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tokenBytes = dataBytes + scaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        uint32_t weightBytes =
            ops::CeilAlign(tilingData->topK * static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        tokenBytes += weightBytes;
    }
    int64_t quantTokenScaleSize =
        ops::CeilAlign((int64_t)((int64_t)(tilingData->bs) * tokenBytes * sizeof(int8_t)), (int64_t)ALIGN_512);
    OP_LOGD(nodeName, "quantTokenScaleSize: {%ld}\n", quantTokenScaleSize);
    // Non-quantized Combine stores one BF16 value (2 bytes) for each hidden-dimension element.
    int64_t combineTokenBytes = static_cast<int64_t>(tilingData->h) * 2;
    if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        int64_t tokenStorageBytes =
            ops::CeilAlign(static_cast<int64_t>(tilingData->h), static_cast<int64_t>(ALIGN_256));
        int64_t scaleCount =
            ops::CeilDiv(static_cast<int64_t>(tilingData->h), static_cast<int64_t>(MXFP_SCALE_GROUP_NUM));
        int64_t storedScaleBytes = ops::CeilAlign(scaleCount, static_cast<int64_t>(MXFP_MULTI_BASE_SIZE));
        combineTokenBytes = ops::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<int64_t>(ALIGN_32));
    }
    int64_t combineSendSize = ops::CeilAlign(sendTotalNum * combineTokenBytes, static_cast<int64_t>(ALIGN_512));
    OP_LOGD(nodeName, "combineSendSize: {%ld}\n", combineSendSize);
    OP_LOGD(nodeName, "total PeermemInfo Size: {%ld}\n",
            rankSyncInWorldSize + maskRecvSize + quantTokenScaleSize + combineSendSize);
}

static ge::DataType GetDataTypeByOpQuantMode(const int64_t opQuantMode)
{
    // unsupport UNQUANT, STATIC, DYNAMIC currently
    switch (opQuantMode) {
        case DISPATCH_QUANT_OUT_DTYPE_E5M2:
            return ge::DT_FLOAT8_E5M2;
        case DISPATCH_QUANT_OUT_DTYPE_E4M3FN:
            return ge::DT_FLOAT8_E4M3FN;
        case DISPATCH_QUANT_OUT_DTYPE_E2M1:
            return ge::DT_FLOAT4_E2M1;
        default:
            return ge::DT_UNDEFINED;
    }
    return ge::DT_UNDEFINED;
}

static int64_t GetOpQuantModeByAttrDispatchOutType(const gert::TilingContext *context, MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    int64_t dispatchQuantOutDtype = static_cast<int64_t>(*dispatchQuantOutDtypePtr);

    int64_t opQuantMode;
    if (dispatchQuantOutDtype == static_cast<int64_t>(ge::DT_FLOAT8_E5M2)) {
        opQuantMode = DISPATCH_QUANT_OUT_DTYPE_E5M2;
    } else if (dispatchQuantOutDtype == static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) {
        opQuantMode = DISPATCH_QUANT_OUT_DTYPE_E4M3FN;
    } else {
        opQuantMode = DISPATCH_QUANT_OUT_DTYPE_E2M1;
    }

    return opQuantMode;
}

static int64_t GetCombineQuantModeByAttr(const gert::TilingContext *context, MegaMoeConfig &config)
{
    auto attrs = context->GetAttrs();
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    if (combineQuantModePtr == nullptr) {
        return COMBINE_QUANT_OUT_TYPE_NO_QUANT;
    }
    return static_cast<int64_t>(*combineQuantModePtr);
}

static uint64_t CalTilingKey(const gert::TilingContext *context, MegaMoeConfig &config, MegaMoeTilingData *tilingData,
                             const char *nodeName)
{
    auto attrs = context->GetAttrs();

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    int64_t combineQuantMode = GetCombineQuantModeByAttr(context, config);
    int64_t topoType = TILINGKEY_TPL_MTE;
    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));

    return GET_TPL_TILING_KEY(static_cast<int64_t>(*dispatchQuantModePtr), opQuantMode, combineQuantMode, topoType,
                              topkWeightsType);
}

static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));
    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    auto activationPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrActivationIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);

    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr || *cclBufferSizePtr < 0,
                    OP_LOGE_WITH_INVALID_INPUT(nodeName, "cclBufferSize"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxRecvTokenNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "maxRecvTokenNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dispatchQuantMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dispatchQuantOutDtypePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dispatchQuantOutDtype"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(combineQuantModePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "combineQuantMode"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commAlgPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "commAlg"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numMaxTokensPerRankPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "numMaxTokensPerRank"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(activationPtr == nullptr || std::strcmp(activationPtr, "swiglu") != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activation", activationPtr == nullptr ? "null" : activationPtr,
                                              "A5 only supports swiglu"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(activationParamsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "activationParams"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(activationParamsPtr->GetSize() > 1U,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationParamsSize",
                                              std::to_string(activationParamsPtr->GetSize()).c_str(),
                                              "should be 0 or 1 for swiglu"),
                    return ge::GRAPH_FAILED);
    if (activationParamsPtr->GetSize() == 1U) {
        const float *activationParams = activationParamsPtr->GetData();
        OP_TILING_CHECK(activationParams == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "activationParamsData"),
                        return ge::GRAPH_FAILED);
        const float activationClamp = activationParams[0];
        OP_TILING_CHECK(activationClamp < 0 || std::isnan(activationClamp),
                        OP_LOGE_FOR_INVALID_VALUE(nodeName, "activationClamp",
                                                  std::to_string(activationClamp).c_str(),
                                                  "should be >= 0 and not NAN"),
                        return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    auto attrs = context->GetAttrs();

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    auto yDesc = context->GetOutputDesc(config.yIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);
    bool isPerExpertWeightTensor = weightOneStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS;
    int64_t weightMoeExpertCount = GetWeightExpertCount(context, config.weight1Index, isPerExpertWeightTensor);

    ge::DataType yDtype = yDesc->GetDataType();
    int64_t yDtypeSize = ge::GetSizeByDataType(yDtype);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    int64_t epWorldSize = static_cast<int64_t>(*epWorldSizePtr);
    OP_TILING_CHECK(epWorldSize < MIN_EP_WORLD_SIZE || epWorldSize > MAX_EP_WORLD_SIZE,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "epWorldSize", std::to_string(epWorldSize).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EP_WORLD_SIZE) + ", " +
                                               std::to_string(MAX_EP_WORLD_SIZE) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    OP_TILING_CHECK((moeExpertNum < epWorldSize || moeExpertNum > MAX_MOE_EXPERT_NUM) || (moeExpertNum % epWorldSize),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertNum", std::to_string(moeExpertNum).c_str(),
                                              (std::string("should in [") + std::to_string(epWorldSize) + ", " +
                                               std::to_string(MAX_MOE_EXPERT_NUM) + "] and mod(..., epWorldSize(" +
                                               std::to_string(epWorldSize) + ")) == 0")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    int64_t moeExpertPerRank = moeExpertNum / epWorldSize;
    OP_TILING_CHECK(
        weightMoeExpertCount != moeExpertPerRank,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "weight1 expert count", std::to_string(weightMoeExpertCount).c_str(),
            (std::string("should equal the local MoE expert count (moeExpertNum / epWorldSize) = ") +
             std::to_string(moeExpertPerRank))
                .c_str()),
        return ge::GRAPH_FAILED);

    // maskRecv Size
    int64_t compareCount =
        ops::CeilAlign((int64_t)(bs * topK * sizeof(int32_t)), (int64_t)(ALIGN_256)) / (int64_t)sizeof(int32_t);
    int64_t maskAlignSize = ops::CeilAlign(compareCount / 8, (int64_t)ALIGN_32); // 8 = block_32 / sizeof(int32_t)
    int64_t maskSlotSize = maskAlignSize + ALIGN_32;                             // mask + 32B count
    int64_t maskRecvSize = ops::CeilAlign(moeExpertPerRank * epWorldSize * maskSlotSize, ALIGN_512);
    // quantTokenScale Size
    uint32_t mxScaleNum = ops::CeilDiv(h, static_cast<int64_t>(ALIGN_32));
    uint32_t dataBytes = ops::CeilAlign(h, static_cast<int64_t>(ALIGN_256)) * sizeof(int8_t);
    uint32_t scaleAlignBytes =
        ops::CeilAlign(mxScaleNum * static_cast<uint32_t>(sizeof(int8_t)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tokenBytes = dataBytes + scaleAlignBytes;
    if (*attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex)) == 1) {
        uint32_t weightBytes =
            ops::CeilAlign(static_cast<uint32_t>(topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        tokenBytes = ops::CeilAlign(tokenBytes + weightBytes, static_cast<uint32_t>(ALIGN_32));
    }
    int64_t quantTokenScaleSize = ops::CeilAlign((int64_t)(bs * tokenBytes * sizeof(int8_t)), (int64_t)ALIGN_512);
    // 必须与 kernel InitCombineBuffers 的 combine record 布局一致。
    int64_t combineTokenBytes = h * yDtypeSize;
    if (GetCombineQuantModeByAttr(context, config) != COMBINE_NO_QUANT) {
        int64_t tokenStorageBytes = ops::CeilAlign(h, static_cast<int64_t>(ALIGN_256));
        int64_t scaleCount = ops::CeilDiv(h, static_cast<int64_t>(MXFP_SCALE_GROUP_NUM));
        int64_t storedScaleBytes = ops::CeilAlign(scaleCount, static_cast<int64_t>(MXFP_MULTI_BASE_SIZE));
        combineTokenBytes = ops::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<int64_t>(ALIGN_32));
    }
    int64_t combineSendSize = ops::CeilAlign(bs * topK * combineTokenBytes, static_cast<int64_t>(ALIGN_512));
    int64_t leastCclBufferSize = PEERMEM_DATA_OFFSET + maskRecvSize + quantTokenScaleSize + combineSendSize;
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>((config.attrCclBufferSizeIndex));
    int64_t cclBufferSize = static_cast<int64_t>(*cclBufferSizePtr);
    OP_TILING_CHECK(cclBufferSize < leastCclBufferSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "cclBufferSize", std::to_string(cclBufferSize).c_str(),
                                              (std::string("should >= ") + std::to_string(leastCclBufferSize)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "cclBufferSize is %ld, leastCclBufferSize is %ld", cclBufferSize, leastCclBufferSize);

    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    int64_t maxRecvTokenNum = static_cast<int64_t>(*maxRecvTokenNumPtr);
    OP_TILING_CHECK(maxRecvTokenNum < 0 || maxRecvTokenNum > bs * epWorldSize * std::min(topK, moeExpertPerRank),
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "maxRecvTokenNum", std::to_string(maxRecvTokenNum).c_str(),
                                              (std::string("should in [0, ") +
                                               std::to_string(bs * epWorldSize * std::min(topK, moeExpertPerRank)) +
                                               "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    auto dispatchQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantModeIndex));
    int64_t dispatchQuantMode = static_cast<int64_t>(*dispatchQuantModePtr);
    OP_TILING_CHECK(dispatchQuantMode != DISPATCH_QUANT_MODE_MXFP,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "dispatchQuantMode", std::to_string(dispatchQuantMode).c_str(),
                        (std::string("only support mxfp(") + std::to_string(DISPATCH_QUANT_MODE_MXFP) + ")").c_str()),
                    return ge::GRAPH_FAILED);

    auto dispatchQuantOutDtypePtr = attrs->GetAttrPointer<int64_t>((config.attrDispatchQuantOutDtypeIndex));
    int64_t dispatchQuantOutDtype = static_cast<int64_t>(*dispatchQuantOutDtypePtr);
    OP_TILING_CHECK(
        dispatchQuantOutDtype != (static_cast<int64_t>(ge::DT_FLOAT8_E5M2)) &&
            dispatchQuantOutDtype != (static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN)) &&
            dispatchQuantOutDtype != (static_cast<int64_t>(ge::DT_FLOAT4_E2M1)),
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "dispatchQuantOutDtype", std::to_string(dispatchQuantOutDtype).c_str(),
                                  "only support fp8_e5m2, fp8_e4m3fn and fp4_e2m1"),
        return ge::GRAPH_FAILED);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    ge::DataType refWeightDataType = GetDataTypeByOpQuantMode(opQuantMode);
    OP_TILING_CHECK(
        refWeightDataType == ge::DT_UNDEFINED,
        OP_LOGE(nodeName, "unsupported dispatchQuantMode(%ld), leading out data type to being DT_UNDEFINED.",
                dispatchQuantMode),
        return ge::GRAPH_FAILED);
    // weight1 must match dispatch quant dtype; the only allowed mismatch is A8W4 (fp4_e2m1 + fp8_e4m3fn).
    if (refWeightDataType != weightOneDesc->GetDataType()) {
        std::string weightDtypeErrMsg = std::string("The dtype of weightOne (") +
                                        Ops::Base::ToString(weightOneDesc->GetDataType()) +
                                        ") must match dispatch quant dtype (" + Ops::Base::ToString(refWeightDataType) +
                                        "), or be fp4_e2m1 with fp8_e4m3fn dispatch quant.";
        OP_TILING_CHECK(
            weightOneDesc->GetDataType() != ge::DT_FLOAT4_E2M1 || opQuantMode != DISPATCH_QUANT_OUT_DTYPE_E4M3FN,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "dispatchQuantOutDtype/weight1", weightDtypeErrMsg.c_str(),
                                                  "weight1 dtype mismatch."),
            return ge::GRAPH_FAILED);
    }

    auto combineQuantModePtr = attrs->GetAttrPointer<int64_t>((config.attrCombineQuantModeIndex));

    OP_TILING_CHECK(
        *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_NO_QUANT &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E5M2 &&
            *combineQuantModePtr != COMBINE_QUANT_OUT_TYPE_E4M3FN,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "combineQuantMode", std::to_string(*combineQuantModePtr).c_str(),
                                  "only support no_quant(0), fp8_e5m2(3) and fp8_e4m3fn(4)"),
        return ge::GRAPH_FAILED);

    auto commAlgPtr = attrs->GetAttrPointer<char>(static_cast<int>(config.attrCommAlgIndex));
    OP_TILING_CHECK(std::strcmp(commAlgPtr, "") != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "commAlg", commAlgPtr, "not support, need empty string"),
                    return ge::GRAPH_FAILED);

    auto numMaxTokensPerRankPtr = attrs->GetAttrPointer<int64_t>((config.attrNumMaxTokensPerRankIndex));
    int64_t numMaxTokensPerRank = static_cast<int64_t>(*numMaxTokensPerRankPtr);
    if (numMaxTokensPerRank != 0) {
        OP_TILING_CHECK(
            bs != numMaxTokensPerRank,
            OP_LOGE_FOR_INVALID_VALUE(nodeName, "numMaxTokensPerRank", std::to_string(numMaxTokensPerRank).c_str(),
                                      (std::string("should be 0 or Bs, Bs is ") + std::to_string(bs).c_str())),
            return ge::GRAPH_FAILED);
    }

    int64_t sharedExpertNum = GetWeightExpertCount(context, config.sharedWeight1Index, isPerExpertWeightTensor);
    OP_TILING_CHECK(sharedExpertNum > NUM_FOUR,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "sharedExpertNum", std::to_string(sharedExpertNum).c_str(),
                                              "only support 0-4"),
                    return ge::GRAPH_FAILED);

    int64_t topkWeightsType = *attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex));
    OP_TILING_CHECK(topkWeightsType != 0 && topkWeightsType != 1,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "topkWeightsType", std::to_string(topkWeightsType).c_str(),
                                              "only support 0(disabled) or 1(enabled)"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttrParams(const gert::TilingContext *context, MegaMoeConfig &config,
                                     MegaMoeTilingData *tilingData, const char *nodeName, const uint32_t aicNum)
{
    auto attrs = context->GetAttrs();

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    auto maxRecvTokenNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMaxRecvTokenNumIndex));
    auto activationParamsPtr = attrs->GetListFloat(config.attrActivationParamsIndex);

    tilingData->epWorldSize = *epWorldSizePtr;
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    int64_t moeExpertNum = static_cast<int64_t>(*moeExpertNumPtr);
    int64_t moeExpertPerRank = moeExpertNum / static_cast<int64_t>(tilingData->epWorldSize);
    tilingData->moeExpertPerRank = static_cast<uint32_t>(moeExpertPerRank);
    tilingData->maxOutputSize = *maxRecvTokenNumPtr != 0 ? *maxRecvTokenNumPtr :
                                                           tilingData->bs * tilingData->epWorldSize *
                                                               std::min(tilingData->topK,
                                                                        tilingData->moeExpertPerRank);
    tilingData->blockNumPerEP = std::max(static_cast<uint32_t>(1), aicNum / tilingData->epWorldSize);
    tilingData->combineQuantMode = GetCombineQuantModeByAttr(context, config);
    tilingData->clampLimit = activationParamsPtr->GetSize() == 0U ? DEFAULT_ACTIVATION_CLAMP :
                                                                    activationParamsPtr->GetData()[0];
    tilingData->actMode = 0U;    // ACT_MODE_SWIGLU, 功能关闭
    tilingData->actSubMode = 0U; // ACT_SUB_MODE_DEFAULT
    tilingData->activationAlpha = 1.0f;
    tilingData->activationBeta = 1.0f;

    tilingData->sharedExpertNum = static_cast<uint32_t>(
        GetWeightExpertCount(context, config.sharedWeight1Index, tilingData->isPerExpertWeightTensor));

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    int64_t opQuantMode = GetOpQuantModeByAttrDispatchOutType(context, config);
    if (weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1 && opQuantMode == DISPATCH_QUANT_OUT_DTYPE_E4M3FN) {
        // A8W4: fp4_e2m1 weight in NZ_C0_32, dispatched via separate template instantiation
        tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A8W4;
    } else if (opQuantMode == DISPATCH_QUANT_OUT_DTYPE_E4M3FN &&
               weightOneDesc->GetDataType() == GetDataTypeByOpQuantMode(opQuantMode) &&
               static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat())) ==
                   ge::FORMAT_FRACTAL_NZ) {
        // A8W8_NZ: fp8_e4m3fn activation × fp8_e4m3fn weight in NZ format (E5M2 not supported)
        tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A8W8_NZ;
    } else if (weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1 && opQuantMode == DISPATCH_QUANT_OUT_DTYPE_E2M1) {
        // A4W4: weight1 和 activation 都是 fp4，GMM1 走 generic，GMM2 走 A8W4。
        // NZ format: weight1 为 FRACTAL_NZ → A4W4_NZ；否则为 A4W4（ND 格式）。
        if (static_cast<ge::Format>(ge::GetPrimaryFormat(weightOneDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ) {
            tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A4W4_NZ;
        } else {
            tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_A4W4;
        }
    } else {
        // Generic: fp8 activation × fp8 weight in ND format
        tilingData->groupedMatmulMode = GROUPED_MATMUL_MODE_GENERAL;
    }

    // GMM1 tile 状态位区每 expert 的 tile 上限（仅 prefetch 软同步路径使用）。
    // 非交织调度只遍历 hiddenDim / 2，交织调度会遍历完整 hiddenDim；host 无法感知 kernel
    // 编译期开关，因此按完整 hiddenDim 预留，避免交织模式覆盖下一 expert 的状态槽。
    tilingData->maxTilesPerExpert = 0;
    if (tilingData->topkWeightsPrefetch == 1) {
        int64_t maxSchedulerN = static_cast<int64_t>(tilingData->hiddenDim);
        int64_t maxTilesM = ops::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), static_cast<int64_t>(256));
        int64_t maxTilesN = ops::CeilDiv(maxSchedulerN, static_cast<int64_t>(256));
        tilingData->maxTilesPerExpert =
            static_cast<uint32_t>(ops::CeilAlign(maxTilesM * maxTilesN, static_cast<int64_t>(16)));
    }

    tilingData->expertsPerBatch = CalcExpertsPerBatch(tilingData, aicNum);

    return ge::GRAPH_SUCCESS;
}

static MegaMoeDispatchBufferConfig CalcDispatchBufferConfig(const MegaMoeTilingData *tilingData,
                                                            uint32_t activationElementsPerByte,
                                                            uint32_t availableUbBytes)
{
    MegaMoeDispatchBufferConfig bufferConfig{};
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->bs) * tilingData->topK;
    uint64_t alignedTotalRouteItems =
        (sendTotalNum + static_cast<uint64_t>(ALIGN_256) - 1U) / static_cast<uint64_t>(ALIGN_256) * ALIGN_256;

    // Stage 1：使用基准 batch 确定 dispatch ring 深度。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_RECV_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) / bufferConfig.routeItemsPerBatch);

    // copyBufferBytes 是每个 dispatch slot 中量化 token 和 scale 的拷贝区大小。
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    uint32_t copyBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        uint32_t weightBytes =
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        copyBufferBytes += weightBytes;
    }
    bufferConfig.copyBufferBytes = copyBufferBytes;

    // fixedBufferBytes 包含 expertTokenCntTensor_、cumsumInfoTensor_、sendCntTensor_ 和 expertTokenNumsOutTensor_。
    uint32_t fixedBufferBytes =
        static_cast<uint32_t>(ALIGN_32) +
        static_cast<uint32_t>(
            ops::CeilAlign(static_cast<uint64_t>(tilingData->epWorldSize) * tilingData->moeExpertPerRank *
                               sizeof(int32_t),
                           static_cast<uint64_t>(ALIGN_32))) +
        tilingData->epWorldSize * static_cast<uint32_t>(ALIGN_32) +
        static_cast<uint32_t>(ops::CeilAlign(static_cast<uint64_t>(tilingData->moeExpertPerRank) * sizeof(int32_t),
                                             static_cast<uint64_t>(ALIGN_32)));
    uint32_t maskBufferBytes = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8);
    // routeIndexBufferBytes 是单个 int32 route index tensor 的大小。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    // bytesWithoutDispatchSlots 包含固定 tensor、maskBatchTensor_、topkIndexTensor_ 和
    // validTopkIndexTensor_（此二者等大）。
    uint32_t bytesWithoutDispatchSlots = fixedBufferBytes + maskBufferBytes + 2U * routeIndexBufferBytes;
    // dispatchSlotBytes 表示一个 ring slot：一份 token/scale copy buffer 加一条 32B triple。
    uint32_t dispatchSlotBytes = copyBufferBytes + static_cast<uint32_t>(ALIGN_32);
    // dispatchSlotBudgetBytes 是扣除非 ring tensor 后可用于分配 ring slot 的 UB。
    uint32_t dispatchSlotBudgetBytes =
        availableUbBytes > bytesWithoutDispatchSlots ? availableUbBytes - bytesWithoutDispatchSlots : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(dispatchSlotBudgetBytes / dispatchSlotBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_DISPATCH_BUFFER_COUNT);
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_DISPATCH_BUFFER_COUNT);

    // Stage 2：固定 ring 深度，用剩余 UB 扩大 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // fixedBytesWithDispatchSlots 包含固定 tensor 和已选中的 dispatch slot。
        uint32_t fixedBytesWithDispatchSlots =
            fixedBufferBytes + static_cast<uint32_t>(bufferConfig.bufferCount) * dispatchSlotBytes;
        // routeItemBudgetBytes 是 maskBatchTensor_ 和两个 route index tensor 可使用的 UB。
        uint32_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithDispatchSlots ? availableUbBytes - fixedBytesWithDispatchSlots : 0U;
        // maskBatchTensor_ 占 1bit，topkIndexTensor_ 和 validTopkIndexTensor_ 各占一个 int32：
        // bytesPerItem = 1/8 + 4 + 4 = 65/8B，因此 maxItems = budgetBytes * 8 / 65。
        uint32_t expandedRouteItems = routeItemBudgetBytes * 8U / 65U;
        expandedRouteItems = expandedRouteItems / static_cast<uint32_t>(ALIGN_256) * ALIGN_256;
        expandedRouteItems =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(expandedRouteItems), alignedTotalRouteItems));
        if (expandedRouteItems > static_cast<uint32_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) /
                                                                bufferConfig.routeItemsPerBatch);
        }
    }
    return bufferConfig;
}

static MegaMoeSendMaskBufferConfig CalcSendMaskBufferConfig(const MegaMoeTilingData *tilingData,
                                                            uint32_t fixedBufferBytes, uint32_t ownedExpertCount,
                                                            uint32_t availableUbBytes)
{
    MegaMoeSendMaskBufferConfig bufferConfig{};
    uint64_t sendTotalNum = static_cast<uint64_t>(tilingData->bs) * tilingData->topK;
    uint64_t alignedTotalRouteItems =
        (sendTotalNum + static_cast<uint64_t>(ALIGN_256) - 1U) / static_cast<uint64_t>(ALIGN_256) * ALIGN_256;

    // Stage 1：使用基准 batch 确定 mask push ring 深度。
    bufferConfig.routeItemsPerBatch =
        static_cast<int32_t>(std::min(alignedTotalRouteItems, static_cast<uint64_t>(BASE_SEND_ROUTE_ITEMS_PER_BATCH)));
    bufferConfig.routeBatchCount =
        static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) / bufferConfig.routeItemsPerBatch);
    bufferConfig.bufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8) + static_cast<uint32_t>(ALIGN_32);

    // routeIndexBufferBytes 是单个 int32 route tensor 的大小。
    uint32_t routeIndexBufferBytes =
        static_cast<uint32_t>(bufferConfig.routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    // bytesWithoutMaskBuffers 包含固定 tensor、topkIdsTensor_ 和 sendGatherOutTensor_。
    uint32_t bytesWithoutMaskBuffers = fixedBufferBytes + 2U * routeIndexBufferBytes;
    // maskBufferBudgetBytes 是扣除非 ring tensor 后可用于分配 mask slot 的 UB。
    uint32_t maskBufferBudgetBytes =
        availableUbBytes > bytesWithoutMaskBuffers ? availableUbBytes - bytesWithoutMaskBuffers : 0U;
    bufferConfig.bufferCount = static_cast<int32_t>(maskBufferBudgetBytes / bufferConfig.bufferBytes);
    bufferConfig.bufferCount = std::min(bufferConfig.bufferCount, MAX_SEND_MASK_BUFFER_COUNT);

    // maskPushCount 是当前 core 实际执行的 mask push 次数，更多 slot 不会增加流水重叠。
    uint64_t maskPushCount = static_cast<uint64_t>(bufferConfig.routeBatchCount) * ownedExpertCount;
    if (maskPushCount > 0U && static_cast<uint64_t>(bufferConfig.bufferCount) > maskPushCount) {
        bufferConfig.bufferCount = static_cast<int32_t>(maskPushCount);
    }
    bufferConfig.bufferCount = std::max(bufferConfig.bufferCount, MIN_SEND_MASK_BUFFER_COUNT);

    // Stage 2：固定 ring 深度，用剩余 UB 扩大 route batch。
    if (static_cast<uint64_t>(bufferConfig.routeItemsPerBatch) < sendTotalNum) {
        // fixedBytesWithMaskCount 包含固定 tensor 和所有 slot 的 32B count 尾部。
        uint32_t fixedBytesWithMaskCount =
            fixedBufferBytes + static_cast<uint32_t>(bufferConfig.bufferCount) * static_cast<uint32_t>(ALIGN_32);
        // routeItemBudgetBytes 是两个 route tensor 和所有 slot 的 mask 可使用的 UB。
        uint32_t routeItemBudgetBytes =
            availableUbBytes > fixedBytesWithMaskCount ? availableUbBytes - fixedBytesWithMaskCount : 0U;
        // topkIdsTensor_ 和 sendGatherOutTensor_ 各占一个 int32，N 个 sendMask slot 各占 1bit：
        // bytesPerItem = 4 + 4 + N/8 = (64 + N)/8B。
        uint32_t expandedRouteItems =
            routeItemBudgetBytes * 8U / (64U + static_cast<uint32_t>(bufferConfig.bufferCount));
        expandedRouteItems = expandedRouteItems / static_cast<uint32_t>(ALIGN_256) * ALIGN_256;
        expandedRouteItems =
            static_cast<uint32_t>(std::min(static_cast<uint64_t>(expandedRouteItems), alignedTotalRouteItems));
        if (expandedRouteItems > static_cast<uint32_t>(bufferConfig.routeItemsPerBatch)) {
            bufferConfig.routeItemsPerBatch = static_cast<int32_t>(expandedRouteItems);
            bufferConfig.routeBatchCount = static_cast<int32_t>((sendTotalNum + bufferConfig.routeItemsPerBatch - 1U) /
                                                                bufferConfig.routeItemsPerBatch);
            bufferConfig.bufferBytes =
                static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8) + static_cast<uint32_t>(ALIGN_32);
        }
    }
    return bufferConfig;
}

static MegaMoeUnpermuteBufferConfig CalcUnpermuteBufferConfig(const MegaMoeTilingData *tilingData,
                                                              uint32_t coreTokenCount,
                                                              uint32_t topKWeightsConversionElementBytes,
                                                              uint32_t availableUbBytes)
{
    MegaMoeUnpermuteBufferConfig bufferConfig{};
    if (coreTokenCount == 0U) {
        return bufferConfig;
    }

    uint32_t bf16SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(uint16_t), static_cast<uint64_t>(ALIGN_32)));
    uint32_t fp32SlotBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(tilingData->h) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    // dataSlotBytes 是同一 token 的 BF16 搬入区和 FP32 计算区之和。
    uint32_t dataSlotBytes = bf16SlotBytes + fp32SlotBytes;
    bufferConfig.bf16SlotElementCount = bf16SlotBytes / sizeof(uint16_t);
    bufferConfig.fp32SlotElementCount = fp32SlotBytes / sizeof(float);

    // scaleBytes 是 combine quant 使用的 BF16/FP32 scale 展开区大小。
    uint32_t scaleBytes = 0U;
    if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleElementCount = (tilingData->h + ALIGN_32 - 1U) / ALIGN_32;
        scaleBytes = static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(uint16_t) * DEQUANT_BF16_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32))) +
                     static_cast<uint32_t>(ops::CeilAlign(
                         static_cast<uint64_t>(scaleElementCount) * sizeof(float) * DEQUANT_FP32_SCALE_EXPANSION,
                         static_cast<uint64_t>(ALIGN_32)));
    }

    // Stage 1：构造基准 weight batch，再选择输入 ring 深度。
    uint32_t baseTokensPerBatch = UNPERMUTE_WEIGHT_ITEMS_PER_BATCH / tilingData->topK;
    bufferConfig.tokensPerBatch = static_cast<int32_t>(std::min(baseTokensPerBatch, coreTokenCount));
    uint32_t weightElementCount = static_cast<uint32_t>(bufferConfig.tokensPerBatch) * tilingData->topK;
    bufferConfig.topKWeightsBufferBytes = static_cast<uint32_t>(
        ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
    if (topKWeightsConversionElementBytes > 0U) {
        bufferConfig.topKWeightsConversionBufferBytes = static_cast<uint32_t>(
            ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * topKWeightsConversionElementBytes,
                           static_cast<uint64_t>(ALIGN_32)));
    }

    // bytesBeforeInputBuffers 包含 weight、scale 和一个累加/输出 data slot。
    uint32_t bytesBeforeInputBuffers = bufferConfig.topKWeightsBufferBytes +
                                       bufferConfig.topKWeightsConversionBufferBytes + scaleBytes + dataSlotBytes;
    uint32_t inputBufferBudgetBytes =
        availableUbBytes > bytesBeforeInputBuffers ? availableUbBytes - bytesBeforeInputBuffers : 0U;
    bufferConfig.inputBufferCount = static_cast<int32_t>(inputBufferBudgetBytes / dataSlotBytes);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, MAX_UNPERMUTE_INPUT_BUFFER_COUNT);
    int32_t accumulationItemCount =
        bufferConfig.tokensPerBatch * static_cast<int32_t>(tilingData->topK + tilingData->sharedExpertNum);
    bufferConfig.inputBufferCount = std::min(bufferConfig.inputBufferCount, accumulationItemCount);
    bufferConfig.inputBufferCount = std::max(bufferConfig.inputBufferCount, MIN_UNPERMUTE_INPUT_BUFFER_COUNT);

    // Stage 2：固定 ring 深度，用剩余 UB 扩大 weight batch。
    if (baseTokensPerBatch < coreTokenCount) {
        // fixedBytes 包含 scale、一个累加/输出 slot 和已经选中的所有输入 slot。
        uint32_t fixedBytes = scaleBytes + (static_cast<uint32_t>(bufferConfig.inputBufferCount) + 1U) * dataSlotBytes;
        // weightBudgetBytes 是 FP32 weight 及可选转换中转区可使用的 UB。
        uint32_t weightBudgetBytes = availableUbBytes - fixedBytes - UNPERMUTE_WEIGHT_ALIGNMENT_RESERVE_BYTES;
        uint32_t weightBytesPerToken =
            tilingData->topK * (static_cast<uint32_t>(sizeof(float)) + topKWeightsConversionElementBytes);
        uint32_t expandedTokensPerBatch = std::min(weightBudgetBytes / weightBytesPerToken, coreTokenCount);
        if (expandedTokensPerBatch > static_cast<uint32_t>(bufferConfig.tokensPerBatch)) {
            bufferConfig.tokensPerBatch = static_cast<int32_t>(expandedTokensPerBatch);
            weightElementCount = expandedTokensPerBatch * tilingData->topK;
            bufferConfig.topKWeightsBufferBytes = static_cast<uint32_t>(ops::CeilAlign(
                static_cast<uint64_t>(weightElementCount) * sizeof(float), static_cast<uint64_t>(ALIGN_32)));
            if (topKWeightsConversionElementBytes > 0U) {
                bufferConfig.topKWeightsConversionBufferBytes = static_cast<uint32_t>(
                    ops::CeilAlign(static_cast<uint64_t>(weightElementCount) * topKWeightsConversionElementBytes,
                                   static_cast<uint64_t>(ALIGN_32)));
            }
        }
    }
    return bufferConfig;
}

static uint64_t CalcCombineSyncSlotCountPerExpert(const MegaMoeTilingData *tilingData)
{
    if (tilingData->combineQuantMode == COMBINE_NO_QUANT && tilingData->topoType != TOPO_TYPE_URMA ||
        tilingData->moeExpertPerRank == 0U) {
        return 0U;
    }

    uint64_t logicalCoreCount = tilingData->blockAivNum;
    bool useAiv1OnlyWaveCombine =
        tilingData->topoType == TOPO_TYPE_MTE &&
        (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_GENERAL ||
         tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ);
    if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ || useAiv1OnlyWaveCombine) {
        // 配对路径和 A8W8 wave 均只让 subBlockIdx=1 参与 Combine，因此逻辑核数是物理 AIV 数的一半。
        logicalCoreCount /= 2U;
    }
    // 同一 token 的 topK expert id 不重复，因此单 expert 从每张卡最多接收 bs 个 token。
    uint64_t maxTokenCountForOneExpert = static_cast<uint64_t>(tilingData->bs) * tilingData->epWorldSize;
    uint64_t maxTokenGroupCountForOneExpert =
        ops::CeilDiv(maxTokenCountForOneExpert, static_cast<uint64_t>(COMBINE_TOKEN_GROUP_SIZE));
    // Workspace 在路由结果产生前分配，因此每个本卡 MoE expert 都按独立最坏情况预留 slot。
    return std::max(maxTokenGroupCountForOneExpert, logicalCoreCount);
}

static ge::graphStatus SetAdaptiveBufferConfigs(const gert::TilingContext *context, MegaMoeConfig &config,
                                                MegaMoeTilingData *tilingData, uint32_t availableUbBytes)
{
    const char *nodeName = context->GetNodeName();
    auto topKWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);
    OP_TILING_CHECK(topKWeightsDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkWeights"),
                    return ge::GRAPH_FAILED);

    uint32_t activationElementsPerByte =
        GetOpQuantModeByAttrDispatchOutType(context, config) == DISPATCH_QUANT_OUT_DTYPE_E2M1 ? 2U : 1U;
    // DispatchBuffInit 的 UB 布局不依赖 AIV 分核结果，所有 AIV core 使用同一套 dispatch 配置。
    // 若 kernel 后续按 core 拆分 DispatchBuffInit 的 tensor 或改变 copyTmp slot 布局，需同步调整此处。
    tilingData->dispatchBufferConfig =
        CalcDispatchBufferConfig(tilingData, activationElementsPerByte, availableUbBytes);

    tilingData->combineSyncSlotCountPerExpert = CalcCombineSyncSlotCountPerExpert(tilingData);

    uint64_t totalFlagElementCount =
        static_cast<uint64_t>(CalcMegaMoeFlagWorkspaceSize(tilingData) / sizeof(int32_t));
    uint32_t resetElementCountPerCore =
        static_cast<uint32_t>(ops::CeilDiv(totalFlagElementCount, static_cast<uint64_t>(tilingData->blockAivNum)));
    uint32_t resetBatchElementCount = std::min(resetElementCountPerCore, static_cast<uint32_t>(DISPATCH_RESET_BATCH));
    uint32_t resetTensorBytes =
        ops::CeilAlign(resetBatchElementCount, static_cast<uint32_t>(INT32_PER_256B)) * sizeof(int32_t);
    uint32_t quantTokenBytes =
        ops::CeilAlign(tilingData->h / activationElementsPerByte, static_cast<uint32_t>(ALIGN_256));
    uint32_t quantScaleAlignBytes = ops::CeilAlign(
        ops::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32)) * static_cast<uint32_t>(sizeof(int8_t)),
        static_cast<uint32_t>(ALIGN_32));
    // 与 kernel xOutTensorSize 一致：token 和 scale 分别对齐，prefetch 模式再追加对齐后的 weight。
    uint32_t quantOutputBufferBytes = quantTokenBytes + quantScaleAlignBytes;
    if (tilingData->topkWeightsPrefetch == 1) {
        quantOutputBufferBytes +=
            ops::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    }
    uint32_t quantInputBufferBytes = ops::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_128)) * sizeof(uint16_t);
    // sendCntAccTensor_ 按本卡 routed MoE expert 数分配，与 kernel 地址布局一致。
    uint32_t maxExpertCountPerCore =
        ops::CeilDiv(tilingData->epWorldSize * tilingData->moeExpertPerRank, tilingData->blockAivNum);
    uint32_t sendCountAccumulatorBytes = static_cast<uint32_t>(ops::CeilAlign(
        static_cast<uint64_t>(maxExpertCountPerCore) * sizeof(int32_t), static_cast<uint64_t>(ALIGN_32)));
    // mxTempTensor_ 占 2KB，xOutTensor_ 和 xInTensor_ 各使用双 buffer。
    uint32_t sendMaskFixedBufferBytes = resetTensorBytes + 2U * 1024U + 2U * quantOutputBufferBytes +
                                        2U * quantInputBufferBytes + sendCountAccumulatorBytes;

    /*
     * 与 kernel SendMaskCal 的 expert 遍历一一对应：
     *   globalExpertId = aivCoreIdx + ownedIdx * blockAivNum。
     * totalExpertCount 除以 blockAivNum 后，前 remainder 个 AIV core 各多处理一个 expert，其余 core
     * 处理 quotient 个 expert，因此这里只需预计算两套配置。
     *
     * 若修改 SendMaskCal 的 expert 分核方式、ownedExpertCount 或 maskPushCount 计算，必须同步更新
     * 这里的两类 core 划分和 kernel 配置选择条件。
     */
    // SendMaskCal 只遍历 routed MoE 专家，不包含独立处理的共享专家。
    uint32_t totalExpertCount = tilingData->epWorldSize * tilingData->moeExpertPerRank;
    uint32_t expertCountPerCoreWithoutExtraExpert = totalExpertCount / tilingData->blockAivNum;
    tilingData->sendMaskCoreCountWithExtraExpert = totalExpertCount % tilingData->blockAivNum;
    uint32_t expertCountPerCoreWithExtraExpert = expertCountPerCoreWithoutExtraExpert + 1U;
    tilingData->sendMaskConfigForCoreWithExtraExpert = CalcSendMaskBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithExtraExpert, availableUbBytes);
    tilingData->sendMaskConfigForCoreWithoutExtraExpert = CalcSendMaskBufferConfig(
        tilingData, sendMaskFixedBufferBytes, expertCountPerCoreWithoutExtraExpert, availableUbBytes);

    /*
     * 与 kernel Unpermute 开头的 TilingByCore(m_, ..., align=1) 一一对应。TilingByCore 使用：
     *   fullTokenChunkSize = ceil(bs / blockAivNum)
     * 为连续 core 分配等长完整 chunk，最后一个活跃 core 可能只处理 tail，后续 core 的 coreLen 为 0
     * 并在读取配置前返回。因此 host 只需预计算“完整 chunk”和“tail chunk”两套配置，并记录完整
     * chunk 对应的 core 数作为 kernel 选择边界。
     *
     * 若修改 TilingByCore、Unpermute 的 align 参数或分核方式，必须同步更新下面的 chunk 推导以及
     * kernel 中 UnpermuteBuffInit 的配置选择条件。
     */
    ge::DataType topKWeightsDataType = topKWeightsDesc->GetDataType();
    // FP32 可直接搬入计算 buffer；其他已支持类型按实际元素大小预留转换中转区。
    uint32_t topKWeightsConversionElementBytes =
        topKWeightsDataType == ge::DT_FLOAT ? 0U : static_cast<uint32_t>(ge::GetSizeByDataType(topKWeightsDataType));
    uint32_t fullTokenChunkSize = ops::CeilDiv(tilingData->bs, tilingData->blockAivNum);
    uint32_t activeCoreCount = ops::CeilDiv(tilingData->bs, fullTokenChunkSize);
    uint32_t tailTokenChunkSize = tilingData->bs - (activeCoreCount - 1U) * fullTokenChunkSize;
    bool tailIsFullTokenChunk = tailTokenChunkSize == fullTokenChunkSize;
    tilingData->unpermuteFullTokenChunkCoreCount = tailIsFullTokenChunk ? activeCoreCount : activeCoreCount - 1U;
    tilingData->unpermuteConfigForFullTokenChunk =
        CalcUnpermuteBufferConfig(tilingData, fullTokenChunkSize, topKWeightsConversionElementBytes, availableUbBytes);
    tilingData->unpermuteConfigForTailTokenChunk =
        tailIsFullTokenChunk ? MegaMoeUnpermuteBufferConfig{} :
                               CalcUnpermuteBufferConfig(tilingData, tailTokenChunkSize,
                                                         topKWeightsConversionElementBytes, availableUbBytes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrAndSetTilingData(const gert::TilingContext *context, MegaMoeConfig &config,
                                                 MegaMoeTilingData *tilingData, const uint32_t aicNum)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckAttrPtrNullptr(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrParams(context, config, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check attr params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetAttrParams(context, config, tilingData, nodeName, aicNum) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "set attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkspace(gert::TilingContext *context, WorkspaceInfo &workspaceInfo, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspace == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "workspace"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(workspaceInfo.workspaceSize == 0LL,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "workspaceSize",
                                              std::to_string(workspaceInfo.workspaceSize).c_str(), "non-zero"),
                    return ge::GRAPH_FAILED);

    int64_t workspaceSize = sysWorkspaceSize + workspaceInfo.workspaceSize + RESERVED_WORKSPACE_SIZE;
    workspace[0] = workspaceSize;

    OP_LOGD(nodeName, "sysWorkspaceSize: %ld \n", sysWorkspaceSize);
    OP_LOGD(nodeName, "mega_moe_tiling workspaceSize: %ld \n", workspaceSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context, MegaMoeConfig &config,
                                             const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScales1Index, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScales2Index, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_CHECK_NULL_WITH_CONTEXT(context, contextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsDesc);

    auto xActiveMaskDesc = context->GetOptionalInputDesc(config.xActiveMaskIndex);
    auto scalesDesc = context->GetOptionalInputDesc(config.scalesIndex);
    OP_TILING_CHECK(xActiveMaskDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "xActiveMask", "not null", "null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(scalesDesc != nullptr, OP_LOGE_FOR_INVALID_VALUE(nodeName, "scales", "not null", "null"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

struct NamedIndex {
    uint32_t index;
    const char *name;
};

// 组合一组专家的两个权重和对应 scale 输入，供 MoE 专家与共享专家复用同一组内校验。
struct ExpertWeightTensorGroupInputs {
    NamedIndex weightOne;
    NamedIndex weightTwo;
    NamedIndex weightScalesOne;
    NamedIndex weightScalesTwo;
    const char *expertTypeName;
};

/*
 * 校验同一 TensorList 内所有 tensor 均为预期维数，且 shape、dtype 和 format 一致。
 * 以第 0 项为基准，从第 1 项开始逐项比较。
 */
static ge::graphStatus CheckTensorPropertiesWithinList(const gert::TilingContext *context,
                                                       const NamedIndex &input, uint32_t expectedDimNum,
                                                       const char *nodeName)
{
    uint32_t tensorCount = GetDynamicInputTensorCount(context, input.index);
    OP_TILING_CHECK(tensorCount == 0U, OP_LOGE_WITH_INVALID_INPUT(nodeName, input.name), return ge::GRAPH_FAILED);

    auto referenceShape = context->GetDynamicInputShape(input.index, 0);
    auto referenceDesc = context->GetDynamicInputDesc(input.index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, referenceDesc);
    uint32_t referenceDimNum = referenceShape->GetStorageShape().GetDimNum();
    OP_TILING_CHECK(referenceDimNum != expectedDimNum,
                    OP_LOGE(nodeName, "%s[0] must be %uD.", input.name, expectedDimNum),
                    return ge::GRAPH_FAILED);

    for (uint32_t tensorIdx = 1; tensorIdx < tensorCount; ++tensorIdx) {
        auto currentShape = context->GetDynamicInputShape(input.index, tensorIdx);
        auto currentDesc = context->GetDynamicInputDesc(input.index, tensorIdx);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentDesc);

        uint32_t currentDimNum = currentShape->GetStorageShape().GetDimNum();
        OP_TILING_CHECK(currentDimNum != expectedDimNum,
                        OP_LOGE(nodeName, "%s[%u] must be %uD.", input.name, tensorIdx, expectedDimNum),
                        return ge::GRAPH_FAILED);
        bool shapeMismatch = false;
        for (uint32_t dimIdx = 0; dimIdx < referenceDimNum && !shapeMismatch; ++dimIdx) {
            shapeMismatch = currentShape->GetStorageShape().GetDim(dimIdx) !=
                            referenceShape->GetStorageShape().GetDim(dimIdx);
        }
        OP_TILING_CHECK(shapeMismatch,
                        OP_LOGE(nodeName, "%s[%u] must have the same shape as %s[0].", input.name, tensorIdx,
                                input.name),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(currentDesc->GetDataType() != referenceDesc->GetDataType(),
                        OP_LOGE(nodeName, "The dtype of %s[%u] must match %s[0].", input.name, tensorIdx, input.name),
                        return ge::GRAPH_FAILED);
        OP_TILING_CHECK(currentDesc->GetStorageFormat() != referenceDesc->GetStorageFormat(),
                        OP_LOGE(nodeName, "The format of %s[%u] must match %s[0].", input.name, tensorIdx, input.name),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

/*
 * 校验 MoE 专家或共享专家的 weight1、weight2、weightScales1 和 weightScales2 四个 TensorList。
 * 先校验每个 TensorList 内部的 tensor 属性一致，再校验四个 TensorList 的长度及堆叠布局下的专家数一致。
 */
static ge::graphStatus CheckExpertWeightInputGroupLayout(const gert::TilingContext *context,
                                                         const ExpertWeightTensorGroupInputs &inputs,
                                                         uint32_t weightDimNum, const char *nodeName)
{
    // scale 比对应的 weight 多一个 multi-base 维度。
    uint32_t scaleDimNum = weightDimNum + 1U;
    OP_TILING_CHECK(
        CheckTensorPropertiesWithinList(context, inputs.weightOne, weightDimNum, nodeName) != ge::GRAPH_SUCCESS ||
            CheckTensorPropertiesWithinList(context, inputs.weightTwo, weightDimNum, nodeName) != ge::GRAPH_SUCCESS ||
            CheckTensorPropertiesWithinList(context, inputs.weightScalesOne, scaleDimNum, nodeName) !=
                ge::GRAPH_SUCCESS ||
            CheckTensorPropertiesWithinList(context, inputs.weightScalesTwo, scaleDimNum, nodeName) !=
                ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "%s weight layout is invalid.", inputs.expertTypeName), return ge::GRAPH_FAILED);

    // 四个 TensorList 以相同下标表示同一专家，因此长度必须一致；堆叠布局只允许一个 tensor。
    uint32_t weightOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightOne.index);
    uint32_t weightTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightTwo.index);
    uint32_t weightScalesOneTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesOne.index);
    uint32_t weightScalesTwoTensorCount = GetDynamicInputTensorCount(context, inputs.weightScalesTwo.index);
    bool tensorCountsMismatch = weightTwoTensorCount != weightOneTensorCount ||
                                weightScalesOneTensorCount != weightOneTensorCount ||
                                weightScalesTwoTensorCount != weightOneTensorCount;
    OP_TILING_CHECK(
        tensorCountsMismatch || (weightDimNum == THREE_DIMS && weightOneTensorCount != 1U),
        OP_LOGE(nodeName, "%s, %s, %s and %s must contain the same number of tensors; stacked layout requires "
                          "exactly one tensor.",
                inputs.weightOne.name, inputs.weightTwo.name, inputs.weightScalesOne.name, inputs.weightScalesTwo.name),
        return ge::GRAPH_FAILED);

    if (weightDimNum == THREE_DIMS) {
        // 堆叠布局中四个 tensor 的 dim0 都表示专家数，必须一致。
        auto weightOneShape = context->GetDynamicInputShape(inputs.weightOne.index, 0);
        auto weightTwoShape = context->GetDynamicInputShape(inputs.weightTwo.index, 0);
        auto weightScalesOneShape = context->GetDynamicInputShape(inputs.weightScalesOne.index, 0);
        auto weightScalesTwoShape = context->GetDynamicInputShape(inputs.weightScalesTwo.index, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoShape);
        int64_t expertCount = weightOneShape->GetStorageShape().GetDim(0);
        OP_TILING_CHECK(
            weightTwoShape->GetStorageShape().GetDim(0) != expertCount ||
                weightScalesOneShape->GetStorageShape().GetDim(0) != expertCount ||
                weightScalesTwoShape->GetStorageShape().GetDim(0) != expertCount,
            OP_LOGE(nodeName, "Dim0 of %s, %s, %s and %s must have matching expert counts for %s inputs.",
                    inputs.weightOne.name, inputs.weightTwo.name, inputs.weightScalesOne.name,
                    inputs.weightScalesTwo.name, inputs.expertTypeName),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

/*
 * 以 MoE weight1[0] 的维数确定公共权重布局，2D 表示逐专家 TensorList，3D 表示单 tensor 堆叠。
 * 校验 MoE 的 weight1、weight2、weightScales1 和 weightScales2。
 * 检查共享专家的对应输入是否同时缺省或同时存在，并在存在时校验其布局与 MoE 一致。
 */
static ge::graphStatus CheckMoeAndSharedWeightInputLayouts(const gert::TilingContext *context,
                                                           const MegaMoeConfig &config, const char *nodeName)
{
    auto weightOneShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneShape);
    uint32_t weightDimNum = weightOneShape->GetStorageShape().GetDimNum();
    bool isPerExpertWeightTensor = weightDimNum == TWO_DIMS;
    OP_TILING_CHECK(
        weightDimNum != TWO_DIMS && weightDimNum != THREE_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "weight1", (std::to_string(weightDimNum) + "D").c_str(),
            "weight1 must be one stacked 3D tensor or a list of 2D tensors."),
        return ge::GRAPH_FAILED);

    const ExpertWeightTensorGroupInputs moeInputs{{config.weight1Index, "weight1"},
                                                  {config.weight2Index, "weight2"},
                                                  {config.weightScales1Index, "weight_scales1"},
                                                  {config.weightScales2Index, "weight_scales2"},
                                                  "MoE expert"};
    const ExpertWeightTensorGroupInputs sharedInputs{{config.sharedWeight1Index, "shared_weight1"},
                                                     {config.sharedWeight2Index, "shared_weight2"},
                                                     {config.sharedWeightScales1Index, "shared_weight_scales1"},
                                                     {config.sharedWeightScales2Index, "shared_weight_scales2"},
                                                     "shared expert"};
    OP_TILING_CHECK(CheckExpertWeightInputGroupLayout(context, moeInputs, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "MoE expert weight layout is invalid."), return ge::GRAPH_FAILED);

    // 检查共享专家的四个 weight/scale 输入是否同时存在或同时缺省。
    bool hasSharedWeightOne =
        GetWeightExpertCount(context, config.sharedWeight1Index, isPerExpertWeightTensor) > 0;
    bool hasSharedWeightTwo =
        GetWeightExpertCount(context, config.sharedWeight2Index, isPerExpertWeightTensor) > 0;
    bool hasSharedWeightScalesOne =
        GetWeightExpertCount(context, config.sharedWeightScales1Index, isPerExpertWeightTensor) > 0;
    bool hasSharedWeightScalesTwo =
        GetWeightExpertCount(context, config.sharedWeightScales2Index, isPerExpertWeightTensor) > 0;
    OP_TILING_CHECK(
        hasSharedWeightOne != hasSharedWeightTwo || hasSharedWeightOne != hasSharedWeightScalesOne ||
            hasSharedWeightOne != hasSharedWeightScalesTwo,
        OP_LOGE(nodeName, "Shared expert weights and weight scales must be provided together."),
        return ge::GRAPH_FAILED);
    if (hasSharedWeightOne) {
        OP_TILING_CHECK(
            CheckExpertWeightInputGroupLayout(context, sharedInputs, weightDimNum, nodeName) != ge::GRAPH_SUCCESS,
            OP_LOGE(nodeName, "Shared expert weight layout is invalid."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                            const char *nodeName)
{
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    auto weightTwoStorageShape = context->GetDynamicInputShape(config.weight2Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);

    uint32_t weightOneDimNum = weightOneStorageShape->GetStorageShape().GetDimNum();
    bool isPerExpertWeightTensor = weightOneDimNum == TWO_DIMS;

    // 去掉可选的专家维后，weight1 和 weight2 均按单专家二维矩阵读取行数和列数。
    const int64_t weightOneRowCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightOneColumnCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightTwoRowCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightTwoColumnCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);

    // 单专家 GMM 形状：weight1=[N, H]，weight2=[H, N/2]，x=[BS, H]。
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    const int64_t xColumnCount = xStorageShape->GetStorageShape().GetDim(1);
    const std::string commonMatrixDimensionsString = "[" + std::to_string(weightOneColumnCount) + ", " +
                                                     std::to_string(weightTwoRowCount) + ", " +
                                                     std::to_string(xColumnCount) + "]";
    OP_TILING_CHECK(
        weightOneColumnCount != weightTwoRowCount || weightOneColumnCount != xColumnCount,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "weight1, weight2 and x", commonMatrixDimensionsString.c_str(),
            "The column count of weight1 and the row count of weight2 must equal the column count of x."),
        return ge::GRAPH_FAILED);

    const std::string weightRowColumnDimensionsString =
        "[" + std::to_string(weightOneRowCount) + ", " + std::to_string(weightTwoColumnCount) + "]";
    OP_TILING_CHECK(
        weightOneRowCount != weightTwoColumnCount * NUM_TWO,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "weight1 and weight2", weightRowColumnDimensionsString.c_str(),
            "The row count of weight1 must equal the column count of weight2 multiplied by 2."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/*
 * 校验共享专家 TensorList 与对应 MoE TensorList 的 dtype 一致，并按 dimensions 逐项校验单专家 shape。
 * 两个 TensorList 的内部一致性已在前序校验，因此只需校验第 0 项。
 */
static ge::graphStatus CheckSharedTensorMatchesMoeTensor(
    const gert::TilingContext *context, const NamedIndex &sharedInput, const NamedIndex &moeInput,
    bool isPerExpertWeightTensor, const NamedIndex *dimensions, uint32_t dimensionCount, const char *nodeName)
{
    auto sharedDesc = context->GetDynamicInputDesc(sharedInput.index, 0);
    auto sharedShape = context->GetDynamicInputShape(sharedInput.index, 0);
    auto moeDesc = context->GetDynamicInputDesc(moeInput.index, 0);
    auto moeShape = context->GetDynamicInputShape(moeInput.index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sharedDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, sharedShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, moeDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, moeShape);

    OP_TILING_CHECK(sharedDesc->GetDataType() != moeDesc->GetDataType(),
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, (std::string(sharedInput.name) + " dtype").c_str(),
                        std::to_string(sharedDesc->GetDataType()).c_str(),
                        (std::string("must be same as ") + moeInput.name).c_str()),
                    return ge::GRAPH_FAILED);

    for (uint32_t dimensionIdx = 0; dimensionIdx < dimensionCount; ++dimensionIdx) {
        const auto &dimension = dimensions[dimensionIdx];
        const int64_t sharedDimensionSize =
            GetSingleExpertTensorDimSize(sharedShape, dimension.index, isPerExpertWeightTensor);
        const int64_t moeDimensionSize =
            GetSingleExpertTensorDimSize(moeShape, dimension.index, isPerExpertWeightTensor);
        OP_TILING_CHECK(
            sharedDimensionSize != moeDimensionSize,
            OP_LOGE_FOR_INVALID_VALUE(
                nodeName, (std::string(sharedInput.name) + " " + dimension.name).c_str(),
                std::to_string(sharedDimensionSize).c_str(),
                (std::string("must be equal to ") + moeInput.name + " " + dimension.name).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSharedExpertInputs(const gert::TilingContext *context, MegaMoeConfig &config,
                                               const char *nodeName)
{
    const NamedIndex sharedWeightOneInput{config.sharedWeight1Index, "shared_weight1"};
    const NamedIndex sharedWeightTwoInput{config.sharedWeight2Index, "shared_weight2"};
    const NamedIndex sharedScaleOneInput{config.sharedWeightScales1Index, "shared_weight_scales1"};
    const NamedIndex sharedScaleTwoInput{config.sharedWeightScales2Index, "shared_weight_scales2"};
    const NamedIndex moeWeightOneInput{config.weight1Index, "weight1"};
    const NamedIndex moeWeightTwoInput{config.weight2Index, "weight2"};
    const NamedIndex moeScaleOneInput{config.weightScales1Index, "weight_scales1"};
    const NamedIndex moeScaleTwoInput{config.weightScales2Index, "weight_scales2"};

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);

    bool isPerExpertWeightTensor = weightOneStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS;
    if (GetWeightExpertCount(context, config.sharedWeight1Index, isPerExpertWeightTensor) == 0) {
        return ge::GRAPH_SUCCESS;
    }
    // 组内一致性已校验，跨组关系只需比较第 0 项。
    const NamedIndex weightDimensions[] = {
        {WEIGHT_MATRIX_ROW_DIM_INDEX, "row count"},
        {WEIGHT_MATRIX_COLUMN_DIM_INDEX, "column count"},
    };
    OP_TILING_CHECK(CheckSharedTensorMatchesMoeTensor(context, sharedWeightOneInput, moeWeightOneInput,
                                                      isPerExpertWeightTensor, weightDimensions, TWO_DIMS,
                                                      nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check shared_weight1 failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckSharedTensorMatchesMoeTensor(context, sharedWeightTwoInput, moeWeightTwoInput,
                                                      isPerExpertWeightTensor, weightDimensions, TWO_DIMS,
                                                      nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check shared_weight2 failed."), return ge::GRAPH_FAILED);

    const NamedIndex scaleDimensions[] = {
        {WEIGHT_SCALE_MATRIX_DIM_INDEX, "matrix dimension"},
        {WEIGHT_SCALE_GROUP_DIM_INDEX, "group dimension"},
        {WEIGHT_SCALE_MULTI_BASE_DIM_INDEX, "multi-base dimension"},
    };
    OP_TILING_CHECK(CheckSharedTensorMatchesMoeTensor(context, sharedScaleOneInput, moeScaleOneInput,
                                                      isPerExpertWeightTensor, scaleDimensions, THREE_DIMS,
                                                      nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check shared_weight_scales1 failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckSharedTensorMatchesMoeTensor(context, sharedScaleTwoInput, moeScaleTwoInput,
                                                      isPerExpertWeightTensor, scaleDimensions, THREE_DIMS,
                                                      nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "check shared_weight_scales2 failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                            const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);

    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);
    auto yStorageShape = context->GetOutputShape(config.yIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorageShape);
    OP_TILING_CHECK(
        yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "y", (std::to_string(yStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);
    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "y dim0 = %ld", yDim0);
    OP_LOGD(nodeName, "y dim1 = %ld", yDim1);

    OP_TILING_CHECK(
        yDim0 != bs || yDim1 != h,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "y", (std::string("[") + std::to_string(yDim0) + ", " + std::to_string(yDim1) + "]").c_str(),
            (std::string("The shape of y must be [bs, h] = [") + std::to_string(bs) + ", " + std::to_string(h) + "].")
                .c_str()),
        return ge::GRAPH_FAILED);

    auto expertTokenNumsStorageShape = context->GetOutputShape(config.expertTokenNumsIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsStorageShape);
    OP_TILING_CHECK(
        expertTokenNumsStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "expert_token_nums",
            (std::to_string(expertTokenNumsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);
    const int64_t expertTokenNumsDim0 = expertTokenNumsStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "expertTokenNums dim0 = %ld", expertTokenNumsDim0);

    // expertTokenNums 仅报告路由专家的 token 数，不包含共享专家.
    auto attrs = context->GetAttrs();
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>((config.attrMoeExpertNumIndex));
    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>((config.attrEpWorldSizeIndex));
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "moeExpertNum"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "epWorldSize"),
                    return ge::GRAPH_FAILED);
    int64_t moeExpertPerRank = static_cast<int64_t>(*moeExpertNumPtr) / static_cast<int64_t>(*epWorldSizePtr);
    OP_TILING_CHECK(
        expertTokenNumsDim0 != moeExpertPerRank,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "expertTokenNums", (std::string("dim0=") + std::to_string(expertTokenNumsDim0)).c_str(),
            (std::string("The shape [dim0] of expertTokenNums must be equal to moeExpertPerRank(") +
             std::to_string(moeExpertPerRank) + ").")
                .c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightScalesTensorDim(const gert::TilingContext *context, MegaMoeConfig &config,
                                                  const char *nodeName)
{
    auto weightScalesOneStorageShape = context->GetDynamicInputShape(config.weightScales1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesOneStorageShape);
    auto weightScalesTwoStorageShape = context->GetDynamicInputShape(config.weightScales2Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScalesTwoStorageShape);

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    auto weightTwoStorageShape = context->GetDynamicInputShape(config.weight2Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightTwoStorageShape);
    bool isPerExpertWeightTensor = weightOneStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS;

    const int64_t weightScalesOneMatrixDimSize =
        GetSingleExpertTensorDimSize(weightScalesOneStorageShape, WEIGHT_SCALE_MATRIX_DIM_INDEX,
                                     isPerExpertWeightTensor);
    const int64_t weightScalesOneGroupDimSize =
        GetSingleExpertTensorDimSize(weightScalesOneStorageShape, WEIGHT_SCALE_GROUP_DIM_INDEX,
                                     isPerExpertWeightTensor);
    const int64_t weightScalesOneMultiBaseDimSize =
        GetSingleExpertTensorDimSize(weightScalesOneStorageShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX,
                                     isPerExpertWeightTensor);
    const int64_t weightScalesTwoMatrixDimSize =
        GetSingleExpertTensorDimSize(weightScalesTwoStorageShape, WEIGHT_SCALE_MATRIX_DIM_INDEX,
                                     isPerExpertWeightTensor);
    const int64_t weightScalesTwoGroupDimSize =
        GetSingleExpertTensorDimSize(weightScalesTwoStorageShape, WEIGHT_SCALE_GROUP_DIM_INDEX,
                                     isPerExpertWeightTensor);
    const int64_t weightScalesTwoMultiBaseDimSize =
        GetSingleExpertTensorDimSize(weightScalesTwoStorageShape, WEIGHT_SCALE_MULTI_BASE_DIM_INDEX,
                                     isPerExpertWeightTensor);

    const int64_t weightOneRowCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightOneColumnCount =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightTwoRowCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    const int64_t weightTwoColumnCount =
        GetSingleExpertTensorDimSize(weightTwoStorageShape, WEIGHT_MATRIX_COLUMN_DIM_INDEX, isPerExpertWeightTensor);

    OP_TILING_CHECK(
        weightScalesOneMatrixDimSize != weightOneRowCount,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "weightScales1",
            (std::string("matrix dimension=") + std::to_string(weightScalesOneMatrixDimSize)).c_str(),
            (std::string("The matrix dimension of weightScales1 must equal the row count of weight1(") +
             std::to_string(weightOneRowCount) + ").")
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        weightScalesOneGroupDimSize != ops::CeilDiv(weightOneColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "weightScales1",
            (std::string("group dimension=") + std::to_string(weightScalesOneGroupDimSize)).c_str(),
            (std::string("The group dimension of weightScales1 must equal CeilDiv(weight1 column count, "
                         "INPUT_WEIGHT_SCALES_CEIL_ALIGN) = ") +
             std::to_string(ops::CeilDiv(weightOneColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN)) + ".")
                .c_str()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        weightScalesTwoMatrixDimSize != weightTwoRowCount,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "weightScales2",
            (std::string("matrix dimension=") + std::to_string(weightScalesTwoMatrixDimSize)).c_str(),
            (std::string("The matrix dimension of weightScales2 must equal the row count of weight2(") +
             std::to_string(weightTwoRowCount) + ").")
                .c_str()),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        weightScalesTwoGroupDimSize != ops::CeilDiv(weightTwoColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            nodeName, "weightScales2",
            (std::string("group dimension=") + std::to_string(weightScalesTwoGroupDimSize)).c_str(),
            (std::string("The group dimension of weightScales2 must equal CeilDiv(weight2 column count, "
                         "INPUT_WEIGHT_SCALES_CEIL_ALIGN) = ") +
             std::to_string(ops::CeilDiv(weightTwoColumnCount, INPUT_WEIGHT_SCALES_CEIL_ALIGN)) + ".")
                .c_str()),
        return ge::GRAPH_FAILED);
    const std::string scaleMultiBaseDimensionsString =
        "[" + std::to_string(weightScalesOneMultiBaseDimSize) + ", " +
        std::to_string(weightScalesTwoMultiBaseDimSize) + "]";
    OP_TILING_CHECK(
        weightScalesOneMultiBaseDimSize != NUM_TWO || weightScalesTwoMultiBaseDimSize != NUM_TWO,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "weightScales1, weightScales2", scaleMultiBaseDimensionsString.c_str(),
            "The per-expert trailing dimension of weightScales1 and weightScales2 must be 2."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(config.contextIndex);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "context"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "context", (std::to_string(contextStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of context must be 1D."),
        return ge::GRAPH_FAILED);
    int64_t contextDim0 = contextStorageShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "context dim0 = %ld", contextDim0);

    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "x", (std::to_string(xStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of x must be 2D."),
                    return ge::GRAPH_FAILED);
    int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "x dim0 = %ld", xDim0);
    OP_LOGD(nodeName, "x dim1 = %ld", xDim1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    OP_TILING_CHECK(topkIdsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkIds"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        topkIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "topkIds", (std::to_string(topkIdsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of topkIds must be 2D."),
        return ge::GRAPH_FAILED);
    const int64_t topkIdsDim0 = topkIdsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkIds dim0 = %ld", topkIdsDim0);
    OP_LOGD(nodeName, "topkIds dim1 = %ld", topkIdsDim1);

    const gert::StorageShape *topkWeightsStorageShape = context->GetInputShape(config.topkWeightsIndex);
    OP_TILING_CHECK(topkWeightsStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "topkWeights"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        nodeName, "topkWeights",
                        (std::to_string(topkWeightsStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(),
                        "The shape dim of topkWeights must be 2D."),
                    return ge::GRAPH_FAILED);
    const int64_t topkWeightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkWeightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "topkWeights dim0 = %ld", topkWeightsDim0);
    OP_LOGD(nodeName, "topkWeights dim1 = %ld", topkWeightsDim1);

    OP_TILING_CHECK(xDim0 != topkIdsDim0 && xDim0 != topkWeightsDim0 && topkIdsDim0 != topkWeightsDim0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        nodeName, "x, topkIds, topkWeights",
                        (std::string("[") + std::to_string(xDim0) + ", " + std::to_string(topkIdsDim0) + ", " +
                         std::to_string(topkWeightsDim0) + "]")
                            .c_str(),
                        "The shape [dim0] of x, topkIds, and topkWeights must be equal."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        topkIdsDim1 != topkWeightsDim1,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            nodeName, "topkIds, topkWeights",
            (std::string("[") + std::to_string(topkIdsDim1) + ", " + std::to_string(topkWeightsDim1) + "]").c_str(),
            "The shape [dim1] of topkIds and topkWeights must be equal."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckMoeAndSharedWeightInputLayouts(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "expert weight tensor layout is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "weight params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWeightScalesTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check weight scales tensor dim failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckSharedExpertInputs(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check shared expert inputs failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckOutputTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "output params shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDataType(const gert::TilingContext *context, MegaMoeConfig &config,
                                           const char *nodeName)
{
    auto contextDesc = context->GetInputDesc(config.contextIndex);
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScales1Index, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScales2Index, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "context",
                                                          Ops::Base::ToString(contextDesc->GetDataType()).c_str(),
                                                          "The dtype of context must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        xDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "x", Ops::Base::ToString(xDesc->GetDataType()).c_str(),
                                              "The dtype of x must be DT_BF16."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(topkIdsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "topkIds",
                                                          Ops::Base::ToString(topkIdsDesc->GetDataType()).c_str(),
                                                          "The dtype of topkIds must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((topkWeightsDesc->GetDataType() != ge::DT_BF16) && (topkWeightsDesc->GetDataType() != ge::DT_FLOAT)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "topkWeights",
                                              Ops::Base::ToString(topkWeightsDesc->GetDataType()).c_str(),
                                              "The dtype of topkWeights must be DT_FLOAT or DT_BF16."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((weightOneDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
         (weightOneDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
         (weightOneDesc->GetDataType() != ge::DT_FLOAT4_E2M1)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            nodeName, "weightOne", Ops::Base::ToString(weightOneDesc->GetDataType()).c_str(),
            "The dtype of weightOne must be within the range DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN or DT_FLOAT4_E2M1."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        ((weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E5M2) &&
         (weightTwoDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN) &&
         (weightTwoDesc->GetDataType() != ge::DT_FLOAT4_E2M1)),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            nodeName, "weightTwo", Ops::Base::ToString(weightTwoDesc->GetDataType()).c_str(),
            "The dtype of weightTwo must be within the range DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN or DT_FLOAT4_E2M1."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        weightOneDesc->GetDataType() != weightTwoDesc->GetDataType(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(nodeName, "weightOne, weightTwo",
                                               (std::string("[") + Ops::Base::ToString(weightOneDesc->GetDataType()) +
                                                ", " + Ops::Base::ToString(weightTwoDesc->GetDataType()) + "]")
                                                   .c_str(),
                                               "The dtypes of weightOne and weightTwo must be the same."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesOneDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "weightScalesOne", Ops::Base::ToString(weightScalesOneDesc->GetDataType()).c_str(),
                        "The dtype of weightScalesOne must be DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(weightScalesTwoDesc->GetDataType() != ge::DT_FLOAT8_E8M0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "weightScalesTwo", Ops::Base::ToString(weightScalesTwoDesc->GetDataType()).c_str(),
                        "The dtype of weightScalesTwo must be DT_FLOAT8_E8M0."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        yDesc->GetDataType() != ge::DT_BF16,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "y", Ops::Base::ToString(yDesc->GetDataType()).c_str(),
                                              "The dtype of y must be DT_BF16."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(expertTokenNumsDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        nodeName, "expertTokenNums", Ops::Base::ToString(expertTokenNumsDesc->GetDataType()).c_str(),
                        "The dtype of expertTokenNums must be DT_INT32."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context, MegaMoeConfig &config,
                                         const char *nodeName)
{
    auto xDesc = context->GetInputDesc(config.xIndex);
    auto topkIdsDesc = context->GetInputDesc(config.topkIdsIndex);
    auto topkWeightsDesc = context->GetInputDesc(config.topkWeightsIndex);

    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    auto weightTwoDesc = context->GetDynamicInputDesc(config.weight2Index, 0);
    auto weightScalesOneDesc = context->GetDynamicInputDesc(config.weightScales1Index, 0);
    auto weightScalesTwoDesc = context->GetDynamicInputDesc(config.weightScales2Index, 0);

    auto yDesc = context->GetOutputDesc(config.yIndex);
    auto expertTokenNumsDesc = context->GetOutputDesc(config.expertTokenNumsIndex);

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "x format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkIdsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "topkIds format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(topkWeightsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "topkWeights format is invalid."), return ge::GRAPH_FAILED);

    // A8W4 path: weight1 must use NZ_C0_32 format now.
    if (weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1 &&
        GetOpQuantModeByAttrDispatchOutType(context, config) == DISPATCH_QUANT_OUT_DTYPE_E4M3FN) {
        OP_TILING_CHECK(weightOneDesc->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ_C0_32,
                        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "weight1",
                                                   Ops::Base::ToString(weightOneDesc->GetStorageFormat()).c_str(),
                                                   "FORMAT_FRACTAL_NZ_C0_32"),
                        return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesOneDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "weightScalesOne format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(weightScalesTwoDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "weightScalesTwo format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "y format is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(expertTokenNumsDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(nodeName, "expertTokenNums format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckMegaMoe(const gert::TilingContext *context, MegaMoeConfig &config,
                                          const char *nodeName)
{
    OP_TILING_CHECK(CheckTensorPtrNullptr(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params check nulld failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDim(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorDataType(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorFormat(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputParam(const gert::TilingContext *context, MegaMoeConfig &config, const char *nodeName)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topkIdsDim1 = topkIdsStorageShape->GetStorageShape().GetDim(1);

    // topk范围
    OP_TILING_CHECK(
        topkIdsDim1 < MIN_TOPK || topkIdsDim1 > MAX_TOPK,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "topK", std::to_string(topkIdsDim1).c_str(), "only support [1, 32]"),
        return ge::GRAPH_FAILED);

    int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    // 检查 H 范围 [1K, 8K]
    OP_TILING_CHECK(
        xDim1 < MIN_H || xDim1 > MAX_H,
        OP_LOGE_FOR_INVALID_VALUE(
            nodeName, "H", std::to_string(xDim1).c_str(),
            (std::string("should in [") + std::to_string(MIN_H) + ", " + std::to_string(MAX_H) + "]").c_str()),
        return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    auto topoTypePtr = attrs->GetAttrPointer<int64_t>(config.attrTopoTypeIndex);
    auto weightOneDesc = context->GetDynamicInputDesc(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneDesc);
    bool isW4NzC0_32 = weightOneDesc->GetDataType() == ge::DT_FLOAT4_E2M1 &&
                       weightOneDesc->GetStorageFormat() == ge::FORMAT_FRACTAL_NZ_C0_32;
    // URMA/Layered 保持 1K 对齐；W4 的 NZ_C0_32 分形要求 GMM K 按 64 对齐；其余 MTE 路径按 32 对齐。
    int64_t requiredHAlignment =
        *topoTypePtr == TOPO_TYPE_URMA ? HIDDEN_DIM_BASE : (isW4NzC0_32 ? W4_K_ALIGN : H_ALIGN);
    OP_TILING_CHECK(
        xDim1 % requiredHAlignment != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "H", std::to_string(xDim1).c_str(),
                                  (std::string("multiple of ") + std::to_string(requiredHAlignment)).c_str()),
        return ge::GRAPH_FAILED);
    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    bool isPerExpertWeightTensor = weightOneStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS;
    int64_t moeExpertPerRank = GetWeightExpertCount(context, config.weight1Index, isPerExpertWeightTensor);
    OP_TILING_CHECK(moeExpertPerRank < MIN_EXPERT_PER_RANK || moeExpertPerRank > MAX_EXPERT_PER_RANK,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "moeExpertPerRank", std::to_string(moeExpertPerRank).c_str(),
                                              (std::string("should in [") + std::to_string(MIN_EXPERT_PER_RANK) + ", " +
                                               std::to_string(MAX_EXPERT_PER_RANK) + "]")
                                                  .c_str()),
                    return ge::GRAPH_FAILED);

    int64_t hiddenDim =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);
    OP_TILING_CHECK(hiddenDim < HIDDEN_DIM_BASE || hiddenDim > MAX_HIDDEN_DIM,
                    OP_LOGE_FOR_INVALID_VALUE(
                        nodeName, "hiddenDim", std::to_string(hiddenDim).c_str(),
                        (std::string("should in [") + std::to_string(HIDDEN_DIM_BASE) + ", " +
                         std::to_string(MAX_HIDDEN_DIM) + "]")
                            .c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hiddenDim % HIDDEN_DIM_BASE != 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "hiddenDim", std::to_string(hiddenDim).c_str(),
                                              (std::string("multiple of ") + std::to_string(HIDDEN_DIM_BASE)).c_str()),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetInputParam(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                     MegaMoeConfig &config)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(config.xIndex);
    int64_t bs = xStorageShape->GetStorageShape().GetDim(0);
    int64_t h = xStorageShape->GetStorageShape().GetDim(1);

    const gert::StorageShape *topkIdsStorageShape = context->GetInputShape(config.topkIdsIndex);
    int64_t topK = topkIdsStorageShape->GetStorageShape().GetDim(1);

    auto weightOneStorageShape = context->GetDynamicInputShape(config.weight1Index, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightOneStorageShape);
    bool isPerExpertWeightTensor = weightOneStorageShape->GetStorageShape().GetDimNum() == TWO_DIMS;
    int64_t hiddenDim =
        GetSingleExpertTensorDimSize(weightOneStorageShape, WEIGHT_MATRIX_ROW_DIM_INDEX, isPerExpertWeightTensor);

    tilingData->bs = static_cast<uint32_t>(bs);
    tilingData->h = static_cast<uint32_t>(h);
    tilingData->hiddenDim = static_cast<uint32_t>(hiddenDim);
    tilingData->topK = static_cast<uint32_t>(topK);
    tilingData->isPerExpertWeightTensor = isPerExpertWeightTensor;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAndSetInput(const gert::TilingContext *context, MegaMoeTilingData *tilingData,
                                        MegaMoeConfig &config, const char *nodeName)
{
    OP_TILING_CHECK(TilingCheckMegaMoe(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check input failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputParam(context, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "check input param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetInputParam(context, tilingData, config) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "set input param failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext *context, MegaMoeConfig &config)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGI(nodeName, "Enter MegaMoe tiling check func.");

    MegaMoeTilingData *tilingData = context->GetTilingData<MegaMoeTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    // Input check & set
    OP_TILING_CHECK(CheckAndSetInput(context, tilingData, config, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Check and set input failed."), return ge::GRAPH_FAILED);

    // Platform Info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    tilingData->aicNum = aicNum;
    tilingData->blockAivNum = aivNum;
    auto attrs = context->GetAttrs();
    tilingData->topoType = *attrs->GetAttrPointer<int64_t>((config.attrTopoTypeIndex));
    tilingData->topkWeightsPrefetch =
        static_cast<int32_t>(*attrs->GetAttrPointer<int64_t>((config.attrTopkWeightsTypeIndex)));
    OP_TILING_CHECK(aivNum <= 0 || aicNum <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "aivNum/aicNum",
                                              (std::to_string(aivNum) + ", " + std::to_string(aicNum)).c_str(),
                                              "should both be > 0"),
                    return ge::GRAPH_FAILED);

    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
    context->SetScheduleMode(1); // batch model, all cores start at the same time
    OP_LOGI(nodeName, "TilingData Init: aivNum: %u, aicNum: %u, ubSize:%lu \n", aivNum, aicNum, ubSize);

    // Attr check & set
    OP_TILING_CHECK(CheckAttrAndSetTilingData(context, config, tilingData, aicNum) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Getting attr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        SetAdaptiveBufferConfigs(context, config, tilingData, static_cast<uint32_t>(ubSize)) == ge::GRAPH_FAILED,
        OP_LOGE(nodeName, "Setting adaptive buffer configs failed."), return ge::GRAPH_FAILED);

    // Cal TilingKey
    uint64_t tilingKey = CalTilingKey(context, config, tilingData, nodeName);
    OP_LOGI(nodeName, "OP TilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    // WorkspaceSize
    WorkspaceInfo workspaceInfo((uint8_t *)0, tilingData);
    OP_TILING_CHECK(SetWorkspace(context, workspaceInfo, nodeName) == ge::GRAPH_FAILED,
                    OP_LOGE(nodeName, "Tiling set workspace Failed"), return ge::GRAPH_FAILED);

    // Print Info
    PrintMegaMoeTilingData(tilingData, nodeName);
    PrintWorkspaceInfo(&workspaceInfo, nodeName);
    PrintPeermemInfo(tilingData, nodeName);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MegaMoeTilingFunc(gert::TilingContext *context)
{
    MegaMoeConfig config;
    ge::graphStatus ret;

    ret = MegaMoeTilingFuncImplPublic(context, config);

    return ret;
}

struct MegaMoeCompileInfo {};
static ge::graphStatus TilingParseForMegaMoe(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MegaMoe).Tiling(MegaMoeTilingFunc).TilingParse<MegaMoeCompileInfo>(TilingParseForMegaMoe);
} // namespace optiling
