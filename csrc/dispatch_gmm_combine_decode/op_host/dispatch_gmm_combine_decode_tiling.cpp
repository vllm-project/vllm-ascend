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
constexpr uint32_t INPUT_SMOOTH_SCALE_INDEX = 6;
constexpr uint32_t INPUT_EXPERT_SCALE_INDEX = 7;

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
constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t MIN_TOKEN_LENGTH = 512;
constexpr uint32_t MAX_TOKEN_LENGTH = 7168;
constexpr uint32_t MIN_GMM1_HIDDEN = 1024;
constexpr uint32_t MAX_GMM1_HIDDEN = 6144;
}  // namespace

namespace optiling {
static size_t CeilUp(size_t x, size_t y)
{
    return (x + y - 1) / y * y;
}

static ge::graphStatus CheckTensorShape(gert::TilingContext *context, const char *nodeName,
                                        DispatchGmmCombineDecodeTilingData &tilingData)
{
    uint32_t epRankId = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.epRankId;
    uint32_t moeExpertNum = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNum;
    uint32_t sharedExpertRankNum = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertRankNum;
    uint32_t moeExpertNumPerRank = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    uint32_t h = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.h;
    uint64_t gmm1WeightDim2 = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen;

    uint32_t localExpertNum = epRankId < sharedExpertRankNum ? 1 : moeExpertNumPerRank;
    const gert::StorageShape *gmm1WeightStorageShape = context->GetInputShape(INPUT_GMM1_WEIGHT_INDEX);
    OPS_ERR_IF(gmm1WeightStorageShape == nullptr, OPS_LOG_E(nodeName, "gmm1 weight shape is null."),
                    return ge::GRAPH_FAILED);
    const int64_t gmm1WeightDim0 = gmm1WeightStorageShape->GetStorageShape().GetDim(0);
    OPS_ERR_IF(gmm1WeightDim0 != localExpertNum,
                    OPS_LOG_E(nodeName, "gmm1Weight Dim0 must be expert number in current rank."),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *gmm1WeightScaleStorageShape = context->GetInputShape(INPUT_GMM1_WEIGHT_SCALE_INDEX);
    OPS_ERR_IF(gmm1WeightScaleStorageShape == nullptr, OPS_LOG_E(nodeName, "gmm1 weight scale shape is null."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(gmm1WeightScaleStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OPS_LOG_E(nodeName, "gmm1 weight scale shape dims must be 2, but current dim num is %lu.",
                            gmm1WeightScaleStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t gmm1WeightScaleDim0 = gmm1WeightScaleStorageShape->GetStorageShape().GetDim(0);
    OPS_ERR_IF(gmm1WeightScaleDim0 != localExpertNum,
                    OPS_LOG_E(nodeName, "gmm1WeightScale Dim0 must be expert number in current rank."),
                    return ge::GRAPH_FAILED);
    const int64_t gmm1WeightScaleDim1 = gmm1WeightScaleStorageShape->GetStorageShape().GetDim(1);
    OPS_ERR_IF(gmm1WeightScaleDim1 != gmm1WeightDim2,
                    OPS_LOG_E(nodeName, "gmm1WeightScale Dim1 must be %lu(gmm1WeightDim2).", gmm1WeightDim2),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *gmm2WeightStorageShape = context->GetInputShape(INPUT_GMM2_WEIGHT_INDEX);
    OPS_ERR_IF(gmm2WeightStorageShape == nullptr, OPS_LOG_E(nodeName, "gmm2 weight shape is null."),
                    return ge::GRAPH_FAILED);
    const int64_t gmm2WeightDim0 = gmm2WeightStorageShape->GetStorageShape().GetDim(0);
    OPS_ERR_IF(gmm2WeightDim0 != localExpertNum,
                    OPS_LOG_E(nodeName, "gmm2Weight Dim0 must be expert number in current rank."),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *gmm2WeightScaleStorageShape = context->GetInputShape(INPUT_GMM2_WEIGHT_SCALE_INDEX);
    OPS_ERR_IF(gmm2WeightScaleStorageShape == nullptr, OPS_LOG_E(nodeName, "gmm2 weight scale shape is null."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(gmm2WeightScaleStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
                    OPS_LOG_E(nodeName, "gmm2 weight scale shape dims must be 2, but current dim num is %lu.",
                            gmm2WeightScaleStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
    const int64_t gmm2WeightScaleDim0 = gmm2WeightScaleStorageShape->GetStorageShape().GetDim(0);
    OPS_ERR_IF(gmm2WeightScaleDim0 != localExpertNum,
                    OPS_LOG_E(nodeName, "gmm2WeightScale Dim0 must be expert number in current rank."),
                    return ge::GRAPH_FAILED);
    const int64_t gmm2WeightScaleDim1 = gmm2WeightScaleStorageShape->GetStorageShape().GetDim(1);
    OPS_ERR_IF(gmm2WeightScaleDim1 != h, OPS_LOG_E(nodeName, "gmm2WeightScale Dim1 must be %u.", h),
                    return ge::GRAPH_FAILED);

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
    uint64_t gmm2HLen = tilingData.disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen / 2;
    if (epRankId < sharedExpertRankNum) {
        maxTokenNum = maxBatchSize * epRankSize / sharedExpertRankNum;
    } else {
        maxTokenNum = maxBatchSize * epRankSize * std::min(topK, moeExpertNumPerRank);
    }

    size_t x2TokenSize = CeilUp(maxTokenNum * gmm2HLen * sizeof(int8_t), GM_ALIGN_SIZE);
    size_t x2ScaleSize = CeilUp(maxTokenNum * sizeof(float), GM_ALIGN_SIZE);
    size_t CVSwapBufferSize =
        CeilUp(aicNum * L1_TILE_BYTE_SIZE * CUBE_WORKSPACE_STAGE * sizeof(int32_t), GM_ALIGN_SIZE);
    size_t swigluOutSize = CeilUp(maxTokenNum * gmm2HLen * sizeof(float), GM_ALIGN_SIZE);
    size_t groupListSize = CeilUp(moeExpertNumPerRank * sizeof(int64_t), GM_ALIGN_SIZE);
    size_t expandIdxSize = CeilUp(batchSize * topK * sizeof(int32_t), GM_ALIGN_SIZE);
    size_t epSendCountSize = CeilUp(epRankSize * moeExpertNumPerRank * sizeof(int32_t), GM_ALIGN_SIZE);
    size_t x1TokenSize = CeilUp(maxTokenNum * h * sizeof(int8_t), GM_ALIGN_SIZE);
    size_t x1ScaleSize = CeilUp(maxTokenNum * sizeof(float), GM_ALIGN_SIZE);
    size_t gmm2DepOutSize = CeilUp(maxTokenNum * h * TOKEN_DTYPE_BYTE_SIZE, GM_ALIGN_SIZE);
    size_t resveredSize = CeilUp(RESERVED_WORKSPACE_SIZE, GM_ALIGN_SIZE);
    size_t usrSize = x2TokenSize + x2ScaleSize + CVSwapBufferSize + swigluOutSize + groupListSize + expandIdxSize +
                     epSendCountSize + x1TokenSize + x1ScaleSize + gmm2DepOutSize + resveredSize;

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
    const gert::StorageShape *gmm1WeightStorageShape = context->GetInputShape(INPUT_GMM1_WEIGHT_INDEX);
    OPS_ERR_IF(gmm1WeightStorageShape == nullptr, OPS_LOG_E(nodeName, "gmm1Weight shape is null."),
                    return ge::GRAPH_FAILED);
    tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen = gmm1WeightStorageShape->GetOriginShape().GetDim(TWO_DIMS);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.aicNum = aicNum;
    tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.aivNum = aivNum;
    OPS_ERR_IF(CheckData(nodeName, *tilingData) != ge::GRAPH_SUCCESS, OPS_LOG_E(nodeName, "CheckData failed."),
                    return ge::GRAPH_FAILED);
    OPS_ERR_IF(SetWorkSpace(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
    SetHcommCfg(context, tilingData, groupEp);
    if (tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank == 1) {
        context->SetTilingKey(0);
    } else {
        context->SetTilingKey(EXEC_FLAG_DEEP_FUSE);
    }
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
