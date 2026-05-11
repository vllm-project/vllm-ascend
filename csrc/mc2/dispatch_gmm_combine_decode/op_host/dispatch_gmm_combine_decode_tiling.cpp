/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdio>
#include <cstdint>
#include <string>

#include "log/ops_log.h"
#include "error/ops_error.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../op_kernel/dispatch_gmm_combine_decode_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"

using namespace ge;
namespace {
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t GM_ALIGN_SIZE = 512;
constexpr uint32_t TOKEN_DTYPE_BYTE_SIZE = 2;
constexpr uint32_t L1_TILE_BYTE_SIZE = 32 * 1024;
constexpr uint32_t CUBE_WORKSPACE_STAGE = 4;
constexpr uint32_t RESERVED_WORKSPACE_SIZE = 256 * 1024;

constexpr uint32_t INPUT_X_INDEX = 0;
constexpr uint32_t INPUT_EXPERT_IDS_INDEX = 1;
constexpr uint32_t INPUT_GMM1_WEIGHT_INDEX = 2;
constexpr uint32_t INPUT_GMM1_WEIGHT_SCALE_INDEX = 3;
constexpr uint32_t INPUT_GMM2_WEIGHT_INDEX = 4;
constexpr uint32_t INPUT_GMM2_WEIGHT_SCALE_INDEX = 5;
constexpr uint32_t INPUT_EXPERT_SCALE_INDEX = 6;
constexpr uint32_t INPUT_SMOOTH_SCALE_INDEX = 7;
constexpr uint32_t INPUT_SHARE_X_ACTIVE_MASK_INDEX = 8;

constexpr uint32_t ATTR_GROUP_EP_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_SIZE_INDEX = 1;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 2;
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 3;
constexpr uint32_t ATTR_SHARE_EXPERT_NUM_INDEX = 4;
constexpr uint32_t ATTR_SHARE_EXPERT_RANK_NUM_INDEX = 5;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 6;
constexpr uint32_t ATTR_GLOBAL_BS_INDEX = 7;

constexpr uint32_t MIN_BATCH_SIZE = 1;
constexpr uint32_t MAX_BATCH_SIZE = 256;
constexpr uint32_t MAX_MOE_EXERT_NUM = 512;
constexpr uint32_t SUPPORT_TOP_K = 12;
constexpr uint32_t ONE_DIMS = 1;
constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t MIN_TOKEN_LENGTH = 512;
constexpr uint32_t MAX_TOKEN_LENGTH = 7168;
constexpr uint32_t MIN_GMM1_HIDDEN = 1024;
constexpr uint32_t MAX_GMM1_HIDDEN = 6144;
constexpr uint32_t TENSOR_HIDDEN_INDEX = 1;
constexpr uint32_t SINGLE_HIDDEN_INDEX = 2;
constexpr uint32_t MAX_TENSOR_COUNT = 256;
}  // namespace

namespace optiling {
static size_t CeilUp(size_t x, size_t y)
{
    return (x + y - 1) / y * y;
}

static uint32_t CountTensorListLen(gert::TilingContext *context, int descIndex)
{
    int count = 0;
    for (uint32_t i = 0; i < MAX_TENSOR_COUNT; i++) {
        auto tensorElement = context->GetDynamicInputTensor(descIndex, i);
        if (tensorElement == nullptr) {
            break;
        }
        count++;
    }
    return count;
}

static ge::graphStatus CheckGmm1Shape(gert::TilingContext *context, DispatchGmmCombineDecodeTilingData *tilingData)
{
    const char *nodeName = context->GetNodeName();
    uint32_t moeExpertNumPerRank = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    uint32_t h = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.h;
    uint32_t gmm1ListLen = CountTensorListLen(context, INPUT_GMM1_WEIGHT_INDEX);
    auto gmm1FirstTensorElement = context->GetDynamicInputTensor(INPUT_GMM1_WEIGHT_INDEX, 0);
    auto gmm1FirstTensorElementShape = gmm1FirstTensorElement->GetOriginShape();
    uint32_t elementDims = gmm1FirstTensorElementShape.GetDimNum();
    ge::DataType gmm1DataType = gmm1FirstTensorElement->GetDataType();
    if (gmm1DataType == ge::DT_BF16 || gmm1DataType == ge::DT_FLOAT16) {
        tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.isBf16Fp16W = true;
    } else {
        tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.isBf16Fp16W = false;
    }
    auto gmm1WeightDesc = context->GetDynamicInputDesc(INPUT_GMM1_WEIGHT_INDEX, 0);
    if (GetPrimaryFormat(gmm1WeightDesc->GetStorageFormat()) == ge::FORMAT_ND) {
        tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.isNDFormat = true;
    }

    OPS_ERR_IF(elementDims != 2 && elementDims != 3, OPS_LOG_E(nodeName, "gmm1Weight shape is invalid."),
            return ge::GRAPH_FAILED);
    if (gmm1ListLen > 1) { // List
        OPS_ERR_IF(h != gmm1FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Weight input length does not equals to token hidden size."),
                return ge::GRAPH_FAILED);
        OPS_ERR_IF(gmm1ListLen != moeExpertNumPerRank,
                OPS_LOG_E(nodeName, "gmm1Weight does not match local expert number perRank."),
                return ge::GRAPH_FAILED);
        tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen = 
                                                gmm1FirstTensorElementShape.GetDim(TENSOR_HIDDEN_INDEX);
        tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.isTensorList = true;
    } else { // Single
        if (elementDims == 2) {  // one localExpert perRank
            OPS_ERR_IF(h != gmm1FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Weight input length does not equals to token hidden size."),
                return ge::GRAPH_FAILED);
            tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen = 
                                                gmm1FirstTensorElementShape.GetDim(SINGLE_HIDDEN_INDEX - 1);
        } else {    // multi localExperts perRank
            OPS_ERR_IF(moeExpertNumPerRank != gmm1FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Weight does not match local expert number per rank."),
                return ge::GRAPH_FAILED);
            OPS_ERR_IF(h != gmm1FirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm1Weight input length does not equals to token hidden size."),
                return ge::GRAPH_FAILED);
            tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen = 
                                                gmm1FirstTensorElementShape.GetDim(SINGLE_HIDDEN_INDEX);
        }
        tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.isTensorList = false;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGmm1ScaleShape(gert::TilingContext *context,
                                                DispatchGmmCombineDecodeTilingData *tilingData)
{
    if (tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.isBf16Fp16W) {
        return ge::GRAPH_SUCCESS;
    }
    const char *nodeName = context->GetNodeName();
    uint32_t moeExpertNumPerRank = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    uint32_t n = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen;


    uint32_t gmm1ScaleListLen = CountTensorListLen(context, INPUT_GMM1_WEIGHT_SCALE_INDEX);
    auto gmm1ScaleFirstTensorElement = context->GetDynamicInputTensor(INPUT_GMM1_WEIGHT_SCALE_INDEX, 0);
    auto gmm1ScaleFirstTensorElementShape = gmm1ScaleFirstTensorElement->GetOriginShape();
    uint32_t elementDims = gmm1ScaleFirstTensorElementShape.GetDimNum();
    OPS_ERR_IF(elementDims != 1 && elementDims != 2, OPS_LOG_E(nodeName, "gmm1WeightScale shape is invalid."),
            return ge::GRAPH_FAILED);
    if (gmm1ScaleListLen > 1) { // List
        OPS_ERR_IF(n != gmm1ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Scale length does not equals to gmm1 hidden size."), return ge::GRAPH_FAILED);
    } else { // Single
        if (elementDims == 1) { // one localExpert perRank
            OPS_ERR_IF(n != gmm1ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Scale length does not equals to gmm1 hidden size."), return ge::GRAPH_FAILED);
        } else { // multi localExperts perRank
            OPS_ERR_IF(moeExpertNumPerRank != gmm1ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm1Scale does not match local expert number perRank."), return ge::GRAPH_FAILED);
            OPS_ERR_IF(n != gmm1ScaleFirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm1Scale length does not equals to gmm1 hidden size."), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGmm2Shape(gert::TilingContext *context, DispatchGmmCombineDecodeTilingData *tilingData)
{
    const char *nodeName = context->GetNodeName();
    uint32_t moeExpertNumPerRank = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    uint32_t h = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.h;
    uint32_t n = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen;

    uint32_t gmm2ListLen = CountTensorListLen(context, INPUT_GMM2_WEIGHT_INDEX);
    auto gmm2FirstTensorElement = context->GetDynamicInputTensor(INPUT_GMM2_WEIGHT_INDEX, 0);
    auto gmm2FirstTensorElementShape = gmm2FirstTensorElement->GetOriginShape();
    uint32_t elementDims = gmm2FirstTensorElementShape.GetDimNum();
    OPS_ERR_IF(elementDims != 2 && elementDims != 3, OPS_LOG_E(nodeName, "gmm2Weight shape is invalid."),
            return ge::GRAPH_FAILED);
    auto gmm2WeightDesc = context->GetDynamicInputDesc(INPUT_GMM2_WEIGHT_INDEX, 0);
    if (GetPrimaryFormat(gmm2WeightDesc->GetStorageFormat()) == ge::FORMAT_ND) {
        tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.isNDFormat = true;
    }
    if (gmm2ListLen > 1) { // List
        OPS_ERR_IF(gmm2ListLen != moeExpertNumPerRank,
                OPS_LOG_E(nodeName, "gmm2 does not match local expert number perRank."), return ge::GRAPH_FAILED);
        OPS_ERR_IF(n / 2 != gmm2FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2 does not equals to token hidden size."), return ge::GRAPH_FAILED);
        OPS_ERR_IF(h != gmm2FirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm2 does not match half of gmm1 hidden size."), return ge::GRAPH_FAILED);
    } else { // Single
        if (elementDims == 2) { // one localExpert perRank
            OPS_ERR_IF(n / 2 != gmm2FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2Weight does not equals to token hidden size."), return ge::GRAPH_FAILED);
            OPS_ERR_IF(h != gmm2FirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm2Weight does not match half of gmm1 hidden size."), return ge::GRAPH_FAILED);
        } else { // multi localExperts perRank
            OPS_ERR_IF(moeExpertNumPerRank != gmm2FirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2Weight does not match local expert num perRank."), return ge::GRAPH_FAILED);
            OPS_ERR_IF(n / 2 != gmm2FirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm2Weight does not equals to token hidden size."), return ge::GRAPH_FAILED);
            OPS_ERR_IF(h != gmm2FirstTensorElementShape.GetDim(2),
                OPS_LOG_E(nodeName, "gmm2Weight does not match half of gmm1 hidden size."), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGmm2ScaleShape(gert::TilingContext *context,
                                                DispatchGmmCombineDecodeTilingData *tilingData)
{
    if (tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.isBf16Fp16W) {
        return ge::GRAPH_SUCCESS;
    }
    const char *nodeName = context->GetNodeName();
    uint32_t moeExpertNumPerRank = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    uint32_t h = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.h;

    uint32_t gmm2ScaleListLen = CountTensorListLen(context, INPUT_GMM2_WEIGHT_SCALE_INDEX);
    auto gmm2ScaleFirstTensorElement = context->GetDynamicInputTensor(INPUT_GMM2_WEIGHT_SCALE_INDEX, 0);
    auto gmm2ScaleFirstTensorElementShape = gmm2ScaleFirstTensorElement->GetOriginShape();
    uint32_t elementDims = gmm2ScaleFirstTensorElementShape.GetDimNum();
    OPS_ERR_IF(elementDims != 1 && elementDims != 2, OPS_LOG_E(nodeName, "gmm2WeightScale shape is invalid."),
            return ge::GRAPH_FAILED);
    if (gmm2ScaleListLen > 1) { // List
        OPS_ERR_IF(h != gmm2ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2Scale does not match token hidden size."), return ge::GRAPH_FAILED);
    } else { // Single
        if (elementDims == 1) { // one localExpert perRank
            OPS_ERR_IF(h != gmm2ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2Scale does not match token hidden size."), return ge::GRAPH_FAILED);
        } else { // multi localExperts perRank
            OPS_ERR_IF(moeExpertNumPerRank != gmm2ScaleFirstTensorElementShape.GetDim(0),
                OPS_LOG_E(nodeName, "gmm2Scale does not match local expert number perRank."), return ge::GRAPH_FAILED);
            OPS_ERR_IF(h != gmm2ScaleFirstTensorElementShape.GetDim(1),
                OPS_LOG_E(nodeName, "gmm2Scale does not match token hidden size."), return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckWeightTensorList(gert::TilingContext *context,
                                                DispatchGmmCombineDecodeTilingData *tilingData)
{
    if (CheckGmm1Shape(context, tilingData) == ge::GRAPH_SUCCESS &&
        CheckGmm1ScaleShape(context, tilingData) == ge::GRAPH_SUCCESS &&
        CheckGmm2Shape(context, tilingData) == ge::GRAPH_SUCCESS &&
        CheckGmm2ScaleShape(context, tilingData) == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus CheckXActiveMaskShape(gert::TilingContext *context, const char *nodeName,
                                                DispatchGmmCombineDecodeTilingData &tilingData)
{
    uint32_t epRankId = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankId;
    uint32_t moeExpertNum = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertRankNum;
    uint32_t moeExpertNumPerRank = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    uint32_t batchSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.bs;
    uint32_t h = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.h;
    uint64_t gmm1WeightDim2 = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen;
    uint32_t localExpertNum = epRankId < sharedExpertRankNum ? 1 : moeExpertNumPerRank;
    const gert::StorageShape* xActiveMaskStorageShape = context->GetOptionalInputShape(
                    INPUT_SHARE_X_ACTIVE_MASK_INDEX);
    if (xActiveMaskStorageShape != nullptr) {
        OPS_ERR_IF(xActiveMaskStorageShape->GetStorageShape().GetDimNum() != ONE_DIMS,
                    OPS_LOG_E(nodeName, " xActiveMask scale shape dims must be 1, but current dim num is %lu.",
                            xActiveMaskStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
        const int64_t xActiveMaskDim0 = xActiveMaskStorageShape->GetStorageShape().GetDim(0);
        OPS_ERR_IF(xActiveMaskDim0 != batchSize, OPS_LOG_E(nodeName,
                    "xActiveMask Dim0 must be batchSize(%u), but current dim is %lu.", batchSize, xActiveMaskDim0),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckData(const char *nodeName, DispatchGmmCombineDecodeTilingData &tilingData)
{
    uint32_t batchSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.bs;
    OPS_ERR_IF(batchSize < MIN_BATCH_SIZE, OPS_LOG_E(nodeName, "batchSize(bs) must >= %d.", MIN_BATCH_SIZE),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(batchSize > MAX_BATCH_SIZE, OPS_LOG_E(nodeName, "batchSize(bs) must <= %d.", MAX_BATCH_SIZE),
                    return ge::GRAPH_FAILED);
    uint32_t tokenLength = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.h;
    OPS_ERR_IF(
        tokenLength < MIN_TOKEN_LENGTH || tokenLength > MAX_TOKEN_LENGTH,
        OPS_LOG_E(nodeName, "tokenLength(h) is invalid. Only support [%u, %u].", MIN_TOKEN_LENGTH, MAX_TOKEN_LENGTH),
        return ge::GRAPH_FAILED);
    uint32_t gmm1HLen = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen;
    OPS_ERR_IF(
        gmm1HLen < MIN_GMM1_HIDDEN || gmm1HLen > MAX_GMM1_HIDDEN,
        OPS_LOG_E(nodeName, "gmm1 hidden size is invalid. Only support [%u, %u].", MIN_GMM1_HIDDEN, MAX_GMM1_HIDDEN),
        return ge::GRAPH_FAILED);
    uint32_t topK = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.k;
    OPS_ERR_IF(topK > SUPPORT_TOP_K, OPS_LOG_E(nodeName, "topK(k) must <= %d.", SUPPORT_TOP_K),
                    return ge::GRAPH_FAILED);
    uint32_t globalBatchSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.globalBs;
    uint32_t epRankSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankSize;
    if (globalBatchSize == 0) {
        globalBatchSize = epRankSize * batchSize;
        tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.globalBs = globalBatchSize;
    } else {
        OPS_ERR_IF(globalBatchSize < 0, OPS_LOG_E(nodeName, "globalBatchSize must >= 0."), return ge::GRAPH_FAILED);
        OPS_ERR_IF(globalBatchSize % epRankSize > 0,
                        OPS_LOG_E(nodeName, "globalBatchSize must be divisible by epRankSize."),
                        return ge::GRAPH_FAILED);
    }
    uint32_t moeExpertNumPerRank = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    uint32_t recvAivNum = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.aivNum / 2;
    OPS_ERR_IF(
        moeExpertNumPerRank > recvAivNum,
        OPS_LOG_E(nodeName, "moeExpertNumPerRank must <= (aivNum/2)(%u), but got %u", recvAivNum, moeExpertNumPerRank),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                               DispatchGmmCombineDecodeTilingData &tilingData, std::string &groupEp)
{
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto groupEpPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_EP_INDEX));
    auto epRankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_MOE_EXPERT_NUM_INDEX);
    auto sharedExpertNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARE_EXPERT_NUM_INDEX);
    auto sharedExpertRankNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_SHARE_EXPERT_RANK_NUM_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    auto globalBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_GLOBAL_BS_INDEX);

    uint32_t epRankSize = static_cast<uint32_t>(*epRankSizePtr);
    uint32_t epRankId = static_cast<uint32_t>(*epRankIdPtr);
    uint32_t moeExpertNum = static_cast<uint32_t>(*moeExpertNumPtr);
    uint32_t sharedExpertNum = static_cast<uint32_t>(*sharedExpertNumPtr);
    uint32_t sharedExpertRankNum = static_cast<uint32_t>(*sharedExpertRankNumPtr);
    uint32_t moeExpertNumPerRank = moeExpertNum / (epRankSize - sharedExpertRankNum);

    OPS_ERR_IF(epRankId < 0, OPS_LOG_E(nodeName, "epRankId must >= 0."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(epRankId >= epRankSize, OPS_LOG_E(nodeName, "epRankId must < epRankSize."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(moeExpertNum > MAX_MOE_EXERT_NUM, OPS_LOG_E(nodeName, "moeExpertNum must <= %d.", MAX_MOE_EXERT_NUM),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(moeExpertNum <= 0, OPS_LOG_E(nodeName, "moeExpertNum must > 0."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(sharedExpertNum != 1, OPS_LOG_E(nodeName, "sharedExpertNum must be 1."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(moeExpertNum % (epRankSize - sharedExpertRankNum) != 0,
                    OPS_LOG_E(nodeName, "moeExpertNum must be divisible by (epRankSize - sharedExpertRankNum)."),
                    return ge::GRAPH_FAILED);

    groupEp = std::string(groupEpPtr);
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankSize = epRankSize;
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankId = epRankId;
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNum = moeExpertNum;
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertNum = sharedExpertNum;
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertRankNum = sharedExpertRankNum;
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.globalBs = static_cast<uint32_t>(*globalBsPtr);
    tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank = moeExpertNumPerRank;
    return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg(const gert::TilingContext *context, DispatchGmmCombineDecodeTilingData *tiling, const std::string groupEp)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_D(nodeName, "DispatchGmmCombineDecode groupEp = %s", groupEp.c_str());
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    std::string algConfigAllGatherStr = "AllGather=level0:ring";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupEp, opType, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName,
                                    DispatchGmmCombineDecodeTilingData &tilingData)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OPS_ERR_IF(workSpaces == nullptr, OPS_LOG_E(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    size_t maxTokenNum;
    uint32_t epRankSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankSize;
    uint32_t epRankId = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankId;
    uint32_t sharedExpertRankNum = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertRankNum;
    uint32_t batchSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.bs;
    uint32_t globalBs = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.globalBs;
    uint32_t maxBatchSize = globalBs / epRankSize;
    uint32_t topK = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.k;
    uint32_t moeExpertNumPerRank = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    uint32_t h = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.h;
    uint32_t aicNum = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.aicNum;
    uint64_t gmm1HLen = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen;
    uint64_t gmm2HLen = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen / 2;
    if (epRankId < sharedExpertRankNum) {
        maxTokenNum = maxBatchSize * epRankSize / sharedExpertRankNum;
    } else {
        maxTokenNum = maxBatchSize * epRankSize * std::min(topK, moeExpertNumPerRank);
    }
    uint32_t wTypeSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.isBf16Fp16W ? TOKEN_DTYPE_BYTE_SIZE : sizeof(int8_t);

    // hbm      input                    = x:  float16 or bf16
    // buf1     dispatch (Only AIV)     => x1: float16 or bf16
    // buf2     gmm1 (Only AIC)         => y1: float
    //          sync
    // buf3     swiglu (Only AIV)       => x2: float16 or bf16
    //          sync ?
    // buf4     gmm2 (AIC & AIV)        => y2: float16 or bf16
    // hbm      combine (Only AIV)      => output: float16 or bf16

    size_t x1TokenSize = maxTokenNum * h * wTypeSize; // x1: float16 or bf16
    size_t x2TokenSize = maxTokenNum * gmm2HLen * wTypeSize; // x2: float16 or bf16
    size_t maxTokenSize = x1TokenSize < x2TokenSize ? x2TokenSize : x1TokenSize;
    maxTokenSize = CeilUp(maxTokenSize, GM_ALIGN_SIZE);
    size_t tokenScaleSize = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.isBf16Fp16W ? 0 : CeilUp(maxTokenNum * sizeof(float), GM_ALIGN_SIZE);
    size_t CVSwapBufferSize =
        CeilUp(aicNum * L1_TILE_BYTE_SIZE * CUBE_WORKSPACE_STAGE * sizeof(int32_t), GM_ALIGN_SIZE);
    size_t swigluOutSize = maxTokenNum * gmm1HLen * sizeof(float); // y1: float
    size_t gmm2DepOutSize = maxTokenNum * h * TOKEN_DTYPE_BYTE_SIZE; // y2: float
    size_t maxSwigluGmm2Size = swigluOutSize < gmm2DepOutSize ? gmm2DepOutSize : swigluOutSize;
    maxSwigluGmm2Size = CeilUp(maxSwigluGmm2Size, GM_ALIGN_SIZE);
    size_t groupListSize = CeilUp(moeExpertNumPerRank * sizeof(int64_t), GM_ALIGN_SIZE);
    size_t expandIdxSize = CeilUp(batchSize * topK * sizeof(int32_t), GM_ALIGN_SIZE);
    size_t epSendCountSize = CeilUp(epRankSize * moeExpertNumPerRank * sizeof(int32_t), GM_ALIGN_SIZE);
    size_t resveredSize = CeilUp(RESERVED_WORKSPACE_SIZE, GM_ALIGN_SIZE);
    size_t usrSize = maxTokenSize + tokenScaleSize + CVSwapBufferSize + maxSwigluGmm2Size + groupListSize + expandIdxSize +
                     epSendCountSize + resveredSize;

    workSpaces[0] = SYSTEM_NEED_WORKSPACE + usrSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchGmmCombineDecodeTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    DispatchGmmCombineDecodeTilingData *tilingData = context->GetTilingData<DispatchGmmCombineDecodeTilingData>();
    OPS_ERR_IF(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    std::string groupEp = "";

    const gert::StorageShape *xStorageShape = context->GetInputShape(INPUT_X_INDEX);
    OPS_ERR_IF(xStorageShape == nullptr, OPS_LOG_E(nodeName, "x shape is null."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OPS_LOG_E(nodeName, "x shape dims must be 2, but current dim num is %lu.",
                            xStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t batchSize = xStorageShape->GetStorageShape().GetDim(0);
    tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.bs = batchSize;
    const int64_t hiddenSize = xStorageShape->GetStorageShape().GetDim(1);
    tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.h = hiddenSize;

    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(INPUT_EXPERT_IDS_INDEX);
    OPS_ERR_IF(expertIdsStorageShape == nullptr, OPS_LOG_E(nodeName, "expertIds shape is null."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertIdsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OPS_LOG_E(nodeName, "expertIds shape dims must be 2, but current dim num is %lu.",
                            expertIdsStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t topK = expertIdsStorageShape->GetStorageShape().GetDim(1);
    tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.k = topK;
    OPS_ERR_IF(GetAttrAndSetTilingData(context, nodeName, *tilingData, groupEp) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(CheckWeightTensorList(context, tilingData) != ge::GRAPH_SUCCESS,
           OPS_LOG_E(nodeName, "CheckWeightTensorList failed."), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.aicNum = aicNum;
    tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.aivNum = aivNum;
    OPS_ERR_IF(CheckData(nodeName, *tilingData) != ge::GRAPH_SUCCESS,
            OPS_LOG_E(nodeName, "CheckData failed."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(CheckXActiveMaskShape(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
            OPS_LOG_E(nodeName, "CheckXActiveMaskShape failed."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(SetWorkSpace(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
            OPS_LOG_E(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    SetHcommCfg(context, tilingData, groupEp);
    const gert::StorageShape* xActiveMaskStorageShape = context->GetOptionalInputShape(
                    INPUT_SHARE_X_ACTIVE_MASK_INDEX);
    bool xActiveMaskEnable = (xActiveMaskStorageShape != nullptr);
    uint64_t tilingKey = 0;
    if (xActiveMaskEnable) {
        tilingKey |= EXEC_FLAG_X_ACTIVE_MASK;
    }
    if (tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank != 1) {
        tilingKey |= EXEC_FLAG_DEEP_FUSE;
    }
    if (tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.isTensorList) {
        tilingKey |= EXEC_FLAG_TENSOR_LIST;
    }
    if (tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.isNDFormat) {
        tilingKey |= EXEC_FLAG_ND_FORMAT;
    }

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(aicNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchGmmCombineDecodeTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = DispatchGmmCombineDecodeTilingFuncImpl(context);
    return ret;
}

struct DispatchGmmCombineDecodeCompileInfo {};
ge::graphStatus TilingParseForDispatchGmmCombineDecode(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DispatchGmmCombineDecode)
    .Tiling(DispatchGmmCombineDecodeTilingFunc)
    .TilingParse<DispatchGmmCombineDecodeCompileInfo>(TilingParseForDispatchGmmCombineDecode);
}  // namespace optiling
