/*
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

#include <cstdio>
#include <cstdint>
#include <string>
#include <cmath>

#include "log/ops_log.h"
#include "error/ops_error.h"

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/fuse_dense_allgather_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"

typedef enum {
    ATTR_TP_INDEX = 0,
    ATTR_RANK_SIZE_INDEX,
    ATTR_RANK_ID_INDEX
} ATTR_TYPE;

static int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}
static constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
static constexpr uint32_t BATCH_SIZE_ONE = 1;
static constexpr uint32_t DEFAULT_ROW = 128;
static constexpr uint32_t DEFAULT_COL = 256;
static constexpr uint32_t DEFAULT_SWIZZLE_COUNT = 4;
static constexpr uint32_t ALLGATHER_EIGHT_RANK_FP16_M0_DEFAULT = 128;
static constexpr int32_t ALLGATHER_EIGHT_RANK_FP16_DATASPLIT_DEFAULT = 16;
static constexpr int32_t ALLGATHER_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT = 100;
static constexpr int32_t HALF_KBYTE = 512;
static constexpr int32_t ALLGATHER_EIGHT_RANK_FP16_PVALUE_DEFAULT = 14;

constexpr int32_t SECOND_TO_MS = 1000;
constexpr double ONE_K = 1024.0;
constexpr double B1_FLOP_PER_MS = (364 * 0.8) * 1e9;
constexpr double DOUBLE = 2.0;
constexpr double HALF_PROB = 0.5;
constexpr int32_t CONDITION_M_ST = 0;
constexpr int32_t CONDITION_M_END = 1;
constexpr int32_t CONDITION_N_ST = 4;
constexpr int32_t CONDITION_N_END = 5;
constexpr int32_t RANKSIZE_EIGHT = 8;
constexpr int32_t MIN_UB_MOVE_NUM = 5120;
constexpr int32_t MAX_UB_NUM = 97280;  // 190 * 1024 / 2

constexpr int32_t DIM_NUM_TWO = 2;
constexpr int32_t DIM_NUM_THREE = 3;
constexpr int32_t DIM_INDEX_ZERO = 0;
constexpr int32_t DIM_INDEX_ONE = 1;
constexpr int32_t DIM_INDEX_TWO = 2;

static std::map<int, std::vector<std::vector<int>>> ALLGATHER_EIGHT_RANK_FP16_M0_MAP = {
    {128,
        {{-1, 31220, -1, 2147483647, -1, 768},
            {31220, 36980, 1280, 2147483647, -1, 768},
            {36980, 2147483647, -1, 2147483647, -1, 768},
            {-1, 2147483647, -1, 2147483647, 768, 2147483647}}},
    {256, {{31220, 36980, -1, 1280, -1, 768}}}};

static std::map<int, std::vector<std::vector<int>>> ALLGATHER_EIGHT_RANK_FP16_UBMOVENUM_MAP = {
    {100,
        {{-1, 3072, -1, 2147483647, -1, 768},
            {3072, 19680, -1, 3072, -1, 768},
            {-1, 3072, -1, 2147483647, 768, 1536},
            {3072, 19680, -1, 3072, 768, 1536},
            {-1, 2147483647, 1792, 2976, 1536, 13312}}},
    {30,
        {{3072, 19680, 3072, 2147483647, -1, 768},
            {19680, 2147483647, -1, 3072, -1, 1536},
            {-1, 2147483647, -1, 1792, 1536, 13312},
            {-1, 768, 2976, 2147483647, 5376, 13312},
            {-1, 768, -1, 2147483647, 13312, 2147483647},
            {26880, 2147483647, -1, 3072, 13312, 2147483647}}},
    {20,
        {{3072, 19680, 3072, 2147483647, 768, 1536},
            {19680, 2147483647, 3072, 2147483647, -1, 1536},
            {-1, 2147483647, 2976, 2147483647, 1536, 5376},
            {768, 2147483647, 2976, 2147483647, 5376, 13312},
            {768, 26880, -1, 2147483647, 13312, 2147483647},
            {26880, 2147483647, 3072, 2147483647, 13312, 2147483647}}}};

int32_t GetValueFromMKNConditionMap(
    int32_t m, int32_t n, int32_t defaultValue, std::map<int, std::vector<std::vector<int>>> conditionMap)
{
    int32_t value = defaultValue;
    for (auto &item : conditionMap) {
        for (auto &condition : item.second) {
            bool inRange = m > condition[CONDITION_M_ST] && m <= condition[CONDITION_M_END] &&
                           n > condition[CONDITION_N_ST] && n <= condition[CONDITION_N_END];
            if (inRange) {
                return item.first;
            }
        }
    }
    return value;
}

void AllgatherEightRankFP16GetDefaultTiling(
    gert::TilingContext *context, PPTilingData &ppTilingData, CommTilingData &commTilingData)
{
    int32_t m = ppTilingData.opShape.m;
    int32_t n = ppTilingData.opShape.n;

    ppTilingData.m0 =
        GetValueFromMKNConditionMap(m, n, ALLGATHER_EIGHT_RANK_FP16_M0_DEFAULT, ALLGATHER_EIGHT_RANK_FP16_M0_MAP);
    ppTilingData.n0 = ppTilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;

    ppTilingData.mLoop = CeilDev(m, ppTilingData.m0);
    ppTilingData.nLoop = CeilDev(n, ppTilingData.n0);

    ppTilingData.coreLoop = ppTilingData.opShape.batchSize * ppTilingData.mLoop * ppTilingData.nLoop;
    ppTilingData.swizzlCount = DEFAULT_SWIZZLE_COUNT;
    ppTilingData.tilingKey = 0;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    ppTilingData.blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);

    commTilingData.ubMoveNum = GetValueFromMKNConditionMap(
            m, n, ALLGATHER_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT, ALLGATHER_EIGHT_RANK_FP16_UBMOVENUM_MAP) *
        HALF_KBYTE;
    commTilingData.is91093 = 0;
}

static void GetDefaultTiling(gert::TilingContext *context, PPTilingData &ppTilingData, CommTilingData &commTilingData)
{
    int32_t m = ppTilingData.opShape.m;
    int32_t n = ppTilingData.opShape.n;

    ppTilingData.m0 = DEFAULT_ROW;
    ppTilingData.n0 = DEFAULT_COL;

    ppTilingData.mLoop = CeilDev(m, ppTilingData.m0);
    ppTilingData.nLoop = CeilDev(n, ppTilingData.n0);
    ppTilingData.coreLoop = ppTilingData.opShape.batchSize * ppTilingData.mLoop * ppTilingData.nLoop;

    ppTilingData.swizzlCount = DEFAULT_SWIZZLE_COUNT;
    ppTilingData.tilingKey = 0;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    ppTilingData.blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);

    commTilingData.ubMoveNum = MIN_UB_MOVE_NUM;
    commTilingData.is91093 = 0;
}

static ge::graphStatus GetAttrAndSetTilingData(
    gert::TilingContext *context, const char *nodeName, FuseDenseAllgatherTilingData &tilingData)

{
    CommTilingData &commTilingData = tilingData.fuseDenseAllgatherInfo.commTilingData;
    PPTilingData &ppTilingData = tilingData.fuseDenseAllgatherInfo.ppTilingData;

    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto RankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_SIZE_INDEX);
    auto RankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_ID_INDEX);

    auto &opShape = ppTilingData.opShape;
    auto &tensor0Shape = context->GetInputTensor(0)->GetOriginShape();
    uint32_t dimNum = tensor0Shape.GetDimNum();
    int64_t bs;
    int64_t rankM;
    int64_t rankN;

    if (dimNum == DIM_NUM_THREE) {
        bs = tensor0Shape.GetDim(DIM_INDEX_ZERO);
        rankM = tensor0Shape.GetDim(DIM_INDEX_ONE);
        rankN = tensor0Shape.GetDim(DIM_INDEX_TWO);
    } else if (dimNum == DIM_NUM_TWO) {
        bs = BATCH_SIZE_ONE;
        rankM = tensor0Shape.GetDim(DIM_INDEX_ZERO);
        rankN = tensor0Shape.GetDim(DIM_INDEX_ONE);
    } else {
        const char *nodeName = context->GetNodeName();
        OPS_LOG_E(nodeName, "Tiling input dim error.");
        return ge::GRAPH_FAILED;
    }

    opShape.batchSize = BATCH_SIZE_ONE;
    opShape.m = bs * rankM;
    opShape.n = rankN;

    commTilingData.rankSize = static_cast<int32_t>(*RankSizePtr);
    commTilingData.rank = static_cast<int32_t>(*RankIdPtr);
    if (commTilingData.rankSize == RANKSIZE_EIGHT) {
        AllgatherEightRankFP16GetDefaultTiling(context, ppTilingData, commTilingData);
    } else {
        GetDefaultTiling(context, ppTilingData, commTilingData);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OPS_ERR_IF(workSpaces == nullptr, OPS_LOG_E(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    workSpaces[0] = systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg(
    const gert::TilingContext *context, FuseDenseAllgatherTilingData *tiling, const std::string groupTp)
{
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupTp, opType, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
}

static ge::graphStatus FuseDenseAllgatherTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    FuseDenseAllgatherTilingData *tilingData = context->GetTilingData<FuseDenseAllgatherTilingData>();
    OPS_ERR_IF(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    OPS_ERR_IF(GetAttrAndSetTilingData(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Get attr and set tiling data failed."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(nodeName, "Tiling set workspace failed."),
        return ge::GRAPH_FAILED);
    SetHcommCfg(context, tilingData, "hcomms");

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platformInfoPtr, return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t aicNum_ = ascendcPlatform.GetCoreNumAic();
    context->SetBlockDim(aicNum_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FuseDenseAllgatherTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = FuseDenseAllgatherTilingFuncImpl(context);
    return ret;
}

struct FuseDenseAllgatherCompileInfo {};
ge::graphStatus TilingParseForFuseDenseAllgather(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FuseDenseAllgather)
    .Tiling(FuseDenseAllgatherTilingFunc)
    .TilingParse<FuseDenseAllgatherCompileInfo>(TilingParseForFuseDenseAllgather);
