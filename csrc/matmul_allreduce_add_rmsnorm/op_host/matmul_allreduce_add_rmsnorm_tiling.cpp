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
#include "../op_kernel/matmul_allreduce_add_rmsnorm_tiling.h"
#include "matmul_allreduce_add_rmsnorm_workspace.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/hccl/hccl_tiling.h"

typedef enum {
    ATTR_TP_INDEX = 0,
    ATTR_RANK_SIZE_INDEX,
    ATTR_RANK_ID_INDEX,
    ATTR_EPSILON_INDEX,
    ATTR_IS_TRANS_B_INDEX,
    ATTR_IS_GATHER_ADD_OUT_INDEX
} ATTR_TYPE;

int32_t CeilDev(int32_t num, int32_t div)
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
static constexpr int32_t VALID_UB_MOVE_NUM = 20480;
static constexpr int32_t COMMDATASPLIT_ONE = 1;
static constexpr int32_t COMM_DATA_DIRECT = 0;
static constexpr uint32_t ALLREDUCE_EIGHT_RANK_FP16_M0_DEFAULT = 128;
static constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_DATASPLIT_DEFAULT = 16;
static constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT = 100;
static constexpr int32_t HALF_KBYTE = 512;
static constexpr int32_t ALLREDUCE_EIGHT_RANK_FP16_PVALUE_DEFAULT = 14;
static constexpr int32_t SWIZZLE_DIRECT_ONE = 1;
static constexpr int32_t COMMNPUSPLIT_ONE = 1;
static constexpr int32_t COMMDATASPLIT_SIXTEEN = 16;
constexpr int32_t SECOND_TO_MS = 1000;
constexpr int64_t MATMUL_BASE_100US = static_cast<int64_t>(1024) * 8192 * 1024;
constexpr int64_t ALLREDUCE_BASE_100US = 4096 * 1024;
constexpr double ONE_K = 1024.0;
constexpr double B1_FLOP_PER_MS = (364 * 0.8) * 1e9;
constexpr double DOUBLE = 2.0;
constexpr double HALF_PROB = 0.5;
constexpr int32_t CONDITION_M_ST = 0;
constexpr int32_t CONDITION_M_END = 1;
constexpr int32_t CONDITION_K_ST = 2;
constexpr int32_t CONDITION_K_END = 3;
constexpr int32_t CONDITION_N_ST = 4;
constexpr int32_t CONDITION_N_END = 5;
constexpr int32_t RANKSIZE_FOUR = 4;
constexpr int32_t RANKSIZE_EIGHT = 8;
constexpr int32_t DIV_TWO = 2;
constexpr int32_t LENPERLOOP_DEFAULT = 5120;
constexpr int32_t MIN_UB_MOVE_NUM = 5120;
constexpr int32_t MAX_UB_NUM = 97280;  // 190 * 1024 / 2
constexpr int32_t MAX_P_VALUE = 15;

constexpr int32_t DIM_NUM_TWO = 2;
constexpr int32_t DIM_NUM_THREE = 3;
constexpr int32_t DIM_INDEX_ZERO = 0;
constexpr int32_t DIM_INDEX_ONE = 1;
constexpr int32_t DIM_INDEX_TWO = 2;

static constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;

static constexpr uint32_t USE_CORE_NUM = 20;

static std::vector<double> ALLREDUCE_UBMOVENUM_COEF = {{-1.72352427e+01,
    2.56887672e-03,
    -8.21819480e+00,
    8.70965589e+01,
    -3.63853858e-01,
    1.27789264e+01,
    1.29782183e+02,
    1.90250023e-02,
    -3.48175441e+00,
    6.18921914e+03,
    3.77072171e+03,
    -5.86895290e+01,
    -8.70740991e-01,
    -1.40262280e-04,
    -2.81910331e-08,
    3.22795486e-05,
    -4.84522320e-03,
    2.94839177e-01,
    2.97260958e-03,
    9.08844709e+01,
    -5.80426209e-10,
    38.183465184603484}};

static std::map<int, std::vector<std::vector<int>>> ALLREDUCE_EIGHT_RANK_FP16_M0_MAP = {
    {128,
        {{-1, 31220, -1, 2147483647, -1, 768},
            {31220, 36980, 1280, 2147483647, -1, 768},
            {36980, 2147483647, -1, 2147483647, -1, 768},
            {-1, 2147483647, -1, 2147483647, 768, 2147483647}}},
    {256, {{31220, 36980, -1, 1280, -1, 768}}}};

static std::map<int, std::vector<std::vector<int>>> ALLREDUCE_EIGHT_RANK_FP16_UBMOVENUM_MAP = {
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

static std::vector<double> ALLREDUCE_PVALUE_COEF = {{-4.23166350e+00,
    6.71137487e-04,
    -1.33434156e+00,
    1.12915884e+01,
    -7.85892737e-02,
    2.59059897e+00,
    3.22129881e+01,
    -5.15776887e-02,
    9.15542742e-01,
    1.56322201e+03,
    3.61977421e+01,
    -5.49544589e-01,
    -2.66903417e-01,
    -3.68521920e-05,
    -6.40666333e-09,
    6.77406054e-06,
    -9.92992099e-04,
    5.60658043e-02,
    2.69372863e-04,
    2.17222337e+01,
    -1.17749660e-10,
    6.100544547671263}};

double GetMTETime(double mknGB, int32_t m0, int32_t n0, double aBindWidth = 3.0, double bBindWidth = 3.0)
{
    // 预估Matmul计算的MTE2搬运时间
    return DOUBLE * mknGB * (SECOND_TO_MS / ONE_K) * (1.0 / (n0 * aBindWidth) + 1.0 / (m0 * bBindWidth));
}

int32_t AllReduceUbMoveNum(int m, int k, int n)
{
    double commPredict = 1.0 * (m / ONE_K) * (n / ONE_K) * (SECOND_TO_MS / ONE_K) / 40;
    double cubePredict = DOUBLE * m * k / B1_FLOP_PER_MS * n;
    double mknGB = (m / ONE_K) * (k / ONE_K) * (n / ONE_K);
    double mteTimePredict1 = GetMTETime(mknGB, DEFAULT_ROW, DEFAULT_COL);
    double mteTimePredict2 = GetMTETime(mknGB, DEFAULT_COL, DEFAULT_ROW);
    double mteTimePredict = std::min(mteTimePredict1, mteTimePredict2);
    double matmulPredict = std::max(cubePredict, mteTimePredict);
    double c0 = matmulPredict / commPredict;
    double c1 = 1.0 * m * n / k;
    double c2 = sqrt(c1);
    double c3 = sqrt(1.0 * m * n) / k;
    double c4 = c3 * c3;
    double c5 = matmulPredict;
    double c6 = commPredict;
    double c7 = 1.0 * n / m;
    double c8 = 1.0 * m * n / sqrt(k);
    double c9 = 1.0 * m * n * sqrt(k);
    double c10 = sqrt(1.0 * m * n) * k;
    double c11 = sqrt(1.0 * m * n * k);
    double c12 = sqrt(1.0 * m * n);
    double c13 = 1.0 * k * k / sqrt(1.0 * m * n);
    double c14 = 1.0 * k * k * sqrt(1.0 * m * n);
    double ubMoveNumDouble = 0;
    std::vector<double> feats_update = {c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        1.0 / c0,
        1.0 / c1,
        1.0 / c2,
        1.0 / c3,
        1.0 / c4,
        c8,
        c9,
        c10,
        c11,
        c12,
        c13,
        1.0 / c13,
        c14,
        1};
    for (uint32_t i = 0; i < feats_update.size(); i++) {
        ubMoveNumDouble += feats_update[i] * ALLREDUCE_UBMOVENUM_COEF[i];
    }

    return std::min(std::max(static_cast<int32_t>(ubMoveNumDouble) * HALF_KBYTE, MIN_UB_MOVE_NUM), MAX_UB_NUM);
}

int32_t AllReducePValue(int m, int k, int n)
{
    double commPredict = 1.0 * (m / ONE_K) * (n / ONE_K) * (SECOND_TO_MS / ONE_K) / 40;
    double cubePredict = DOUBLE * m * k / B1_FLOP_PER_MS * n;
    double mknGB = (m / ONE_K) * (k / ONE_K) * (n / ONE_K);
    double mteTimePredict1 = GetMTETime(mknGB, DEFAULT_ROW, DEFAULT_COL);
    double mteTimePredict2 = GetMTETime(mknGB, DEFAULT_COL, DEFAULT_ROW);
    double mteTimePredict = std::min(mteTimePredict1, mteTimePredict2);
    double matmulPredict = std::max(cubePredict, mteTimePredict);
    double c0 = matmulPredict / commPredict;
    double c1 = 1.0 * m * n / k;
    double c2 = sqrt(c1);
    double c3 = sqrt(1.0 * m * n) / k;
    double c4 = c3 * c3;
    double c5 = matmulPredict;
    double c6 = commPredict;
    double c7 = 1.0 * n / m;
    double c8 = 1.0 * m * n / sqrt(k);
    double c9 = 1.0 * m * n * sqrt(k);
    double c10 = sqrt(1.0 * m * n) * k;
    double c11 = sqrt(1.0 * m * n * k);
    double c12 = sqrt(1.0 * m * n);
    double c13 = 1.0 * k * k / sqrt(1.0 * m * n);
    double c14 = 1.0 * k * k * sqrt(1.0 * m * n);
    double pValueDouble = 0;
    std::vector<double> feats_update = {c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        1.0 / c0,
        1.0 / c1,
        1.0 / c2,
        1.0 / c3,
        1.0 / c4,
        c8,
        c9,
        c10,
        c11,
        c12,
        c13,
        1.0 / c13,
        c14,
        1};
    for (uint32_t i = 0; i < feats_update.size(); i++) {
        pValueDouble += feats_update[i] * ALLREDUCE_PVALUE_COEF[i];
    }

    return std::min(std::max(static_cast<int32_t>(pValueDouble), 1), MAX_P_VALUE);
}

int32_t GetValueFromMKNConditionMap(
    int32_t m, int32_t k, int32_t n, int32_t defaultValue, std::map<int, std::vector<std::vector<int>>> conditionMap)
{
    int32_t value = defaultValue;
    for (auto &item : conditionMap) {
        for (auto &condition : item.second) {
            bool inRange = m > condition[CONDITION_M_ST] && m <= condition[CONDITION_M_END] &&
                           k > condition[CONDITION_K_ST] && k <= condition[CONDITION_K_END] &&
                           n > condition[CONDITION_N_ST] && n <= condition[CONDITION_N_END];
            if (inRange) {
                return item.first;
            }
        }
    }
    return value;
}

void AllReduceEightRankFP16GetDefaultTiling(
    gert::TilingContext *context, PPTilingData &ppTilingData, CommTilingData &commTilingData)
{
    int32_t m = ppTilingData.opShape.m;
    int32_t k = ppTilingData.opShape.k;
    int32_t n = ppTilingData.opShape.n;

    ppTilingData.m0 =
        GetValueFromMKNConditionMap(m, k, n, ALLREDUCE_EIGHT_RANK_FP16_M0_DEFAULT, ALLREDUCE_EIGHT_RANK_FP16_M0_MAP);

    ppTilingData.k0 = DEFAULT_COL;
    ppTilingData.n0 = ppTilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;

    ppTilingData.mLoop = CeilDev(m, ppTilingData.m0);
    ppTilingData.nLoop = CeilDev(n, ppTilingData.n0);
    ppTilingData.kLoop = CeilDev(k, ppTilingData.k0);

    ppTilingData.coreLoop = ppTilingData.opShape.batchSize * ppTilingData.mLoop * ppTilingData.nLoop;
    ppTilingData.swizzlDirect = SWIZZLE_DIRECT_ONE;
    ppTilingData.swizzlCount = DEFAULT_SWIZZLE_COUNT;
    ppTilingData.tilingKey = 0;
    ppTilingData.splitK = 0;

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    ppTilingData.blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);

    commTilingData.ubMoveNum =
        GetValueFromMKNConditionMap(
            m, k, n, ALLREDUCE_EIGHT_RANK_FP16_UBMOVENUM_DEFAULT, ALLREDUCE_EIGHT_RANK_FP16_UBMOVENUM_MAP) *
        HALF_KBYTE;
    commTilingData.pValue = ALLREDUCE_EIGHT_RANK_FP16_PVALUE_DEFAULT;

    commTilingData.commDirect = COMM_DATA_DIRECT;
    commTilingData.commNpuSplit = COMMNPUSPLIT_ONE;
    commTilingData.commDataSplit = COMMDATASPLIT_SIXTEEN;
    commTilingData.is91093 = 0;
    commTilingData.withSerialMode = 0;
    commTilingData.tag = 0;
    commTilingData.write2OtherRank = 0;
}

void GetDefaultTiling(gert::TilingContext *context, PPTilingData &ppTilingData, CommTilingData &commTilingData)
{
    int32_t m = ppTilingData.opShape.m;
    int32_t k = ppTilingData.opShape.k;
    int32_t n = ppTilingData.opShape.n;

    ppTilingData.m0 = DEFAULT_ROW;
    ppTilingData.n0 = DEFAULT_COL;
    ppTilingData.k0 = DEFAULT_COL;

    ppTilingData.mLoop = CeilDev(m, ppTilingData.m0);
    ppTilingData.nLoop = CeilDev(n, ppTilingData.n0);
    ppTilingData.kLoop = CeilDev(k, ppTilingData.k0);
    ppTilingData.coreLoop = ppTilingData.opShape.batchSize * ppTilingData.mLoop * ppTilingData.nLoop;

    ppTilingData.swizzlDirect = m > n ? 0 : 1;
    ppTilingData.swizzlCount = DEFAULT_SWIZZLE_COUNT;
    ppTilingData.tilingKey = 0;
    ppTilingData.splitK = 0;

    uint32_t blockDim = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    ppTilingData.blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);

    commTilingData.ubMoveNum = AllReduceUbMoveNum(m, k, n);
    commTilingData.pValue = AllReducePValue(m, k, n);
    commTilingData.commNpuSplit = commTilingData.rankSize;
    commTilingData.commDataSplit = COMMDATASPLIT_ONE;
    commTilingData.commDirect = COMM_DATA_DIRECT;
    commTilingData.lenPerLoop = ppTilingData.m0 * ppTilingData.n0 * commTilingData.pValue * ppTilingData.blockDim;
    commTilingData.lenPerLoop = commTilingData.lenPerLoop / commTilingData.rankSize;
    commTilingData.is91093 = 0;
    commTilingData.withSerialMode = 0;
    commTilingData.tag = 0;
    commTilingData.write2OtherRank = 0;
}

static inline void GetRmsnormTilingData(RmsNormTilingData &rmsnormtiling, std::vector<int64_t> &shapeVec,
    std::vector<int64_t> &oriShapeVec, uint32_t calcBytes = 0, uint32_t loopCount = 1, float ep = 1e-5)
{
    ge::Shape srcShape(shapeVec);
    ge::Shape oriSrcShape(oriShapeVec);
    uint32_t minValue = 0;
    uint32_t maxValue = 0;
    AscendC::GetRmsNormMaxMinTmpSize(srcShape, sizeof(uint16_t), maxValue, minValue, false);

    if (calcBytes < minValue) {
        rmsnormtiling.calcBytes = minValue;
    } else if (calcBytes > maxValue) {
        rmsnormtiling.calcBytes = maxValue;
    } else {
        rmsnormtiling.calcBytes = calcBytes;
    }

    optiling::RmsNormTiling tilingdata;
    AscendC::GetRmsNormTilingInfo(srcShape, oriSrcShape, rmsnormtiling.calcBytes, sizeof(uint16_t), tilingdata, false);
    size_t tilingSize = tilingdata.GetDataSize();
    tilingdata.SaveToBuffer(&rmsnormtiling.tiling, tilingSize);
    rmsnormtiling.epsilon = ep;
    rmsnormtiling.loopCount = loopCount;
}

static inline void GetQuantTilingData(QuantInfo &quantInfo)
{
    quantInfo.dequantGranularity = QuantGranularity::QUANT_GRANULARITY_UNDEFINED;
    quantInfo.dequantGroupSize = -1;
    quantInfo.quantGranularity = QuantGranularity::QUANT_GRANULARITY_UNDEFINED;
    quantInfo.quantGroupSize = -1;
}

static ge::graphStatus GetAttrAndSetTilingData(
    gert::TilingContext *context, const char *nodeName, MatmulAllreduceAddRmsnormTilingData &tilingData)

{
    CommTilingData &commTilingData = tilingData.matmulAllreduceAddRmsnormInfo.commTilingData;
    PPTilingData &ppTilingData = tilingData.matmulAllreduceAddRmsnormInfo.ppTilingData;
    RmsNormTilingData &rmsnormTilingData = tilingData.matmulAllreduceAddRmsnormInfo.rmsnormTilingData;
    QuantInfo &quantInfo = tilingData.matmulAllreduceAddRmsnormInfo.quantInfo;

    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto RankSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_SIZE_INDEX);
    auto RankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_RANK_ID_INDEX);

    bool isTransB = *(attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_B_INDEX));

    ppTilingData.isTransA = false;
    ppTilingData.isTransB = isTransB;
    ppTilingData.isGatherAddOut = *(attrs->GetAttrPointer<bool>(ATTR_IS_GATHER_ADD_OUT_INDEX));

    auto &opShape = ppTilingData.opShape;
    auto &tensor0Shape = context->GetInputTensor(0)->GetOriginShape();
    uint32_t dimNum = tensor0Shape.GetDimNum();
    int64_t bs;
    int64_t rankM;
    int64_t rankK;

    if (dimNum == DIM_NUM_THREE) {
        bs = tensor0Shape.GetDim(DIM_INDEX_ZERO);
        rankM = tensor0Shape.GetDim(DIM_INDEX_ONE);
        rankK = tensor0Shape.GetDim(DIM_INDEX_TWO);
    } else if (dimNum == DIM_NUM_TWO) {
        bs = BATCH_SIZE_ONE;
        rankM = tensor0Shape.GetDim(DIM_INDEX_ZERO);
        rankK = tensor0Shape.GetDim(DIM_INDEX_ONE);
    } else {
        const char *nodeName = context->GetNodeName();
        OPS_LOG_E(nodeName, "Tiling input dim error.");
        return ge::GRAPH_FAILED;
    }

    int64_t rankN = isTransB ?
        context->GetInputTensor(1)->GetOriginShape().GetDim(DIM_INDEX_ZERO) :
        context->GetInputTensor(1)->GetOriginShape().GetDim(DIM_INDEX_ONE);

    opShape.batchSize = BATCH_SIZE_ONE;
    opShape.m = bs * rankM;
    opShape.n = rankN;
    opShape.k = rankK;

    commTilingData.rankSize = static_cast<int32_t>(*RankSizePtr);
    commTilingData.rank = static_cast<int32_t>(*RankIdPtr);
    if (commTilingData.rankSize == RANKSIZE_EIGHT) {
        AllReduceEightRankFP16GetDefaultTiling(context, ppTilingData, commTilingData);
    } else {
        GetDefaultTiling(context, ppTilingData, commTilingData);
    }

    uint32_t calcBytes = 0;
    uint32_t sLength = 1;
    std::vector<int64_t> shapeVec = {1, 1, rankN};
    std::vector<int64_t> oriShapeVec = shapeVec;
    auto EpsilonPtr = attrs->GetAttrPointer<float>(ATTR_EPSILON_INDEX);
    float epsilon = static_cast<float>(*EpsilonPtr);
    GetRmsnormTilingData(
        rmsnormTilingData, shapeVec, oriShapeVec, calcBytes, commTilingData.rankSize * sLength * rankN, epsilon);
    GetQuantTilingData(quantInfo);

    return ge::GRAPH_SUCCESS;
}

bool IsMatrixAligned(const int64_t &m, const int64_t &n, const bool &transpose, int nElemAlign)
{
    return (transpose ? m : n) % nElemAlign == 0;
}

int64_t GetAlignedMatrixSize(
    const int64_t &batchSize, const int64_t &m, const int64_t &n, const bool &transpose, int nElemAlign)
{
    int64_t nRow = transpose ? n : m;
    int64_t nCol = transpose ? m : n;
    int64_t nColAlign = (nCol + nElemAlign - 1) / nElemAlign * nElemAlign;
    return batchSize * nRow * nColAlign;
}

WorkspaceDetail GetWorkspaceDetail(CoCDataTypeDesc dataType, const MatMulInfo &mmInfo, const QuantInfo &quantInfo)
{
    WorkspaceDetail workspaceDetail;

    int32_t eleSize = COC_TYPE2ELE_SIZE.at(dataType);
    int32_t nElemAlign = ALIGN_BYTES / eleSize;

    bool hasQuant = quantInfo.quantGranularity != QuantGranularity::QUANT_GRANULARITY_UNDEFINED;
    if (hasQuant || (!IsMatrixAligned(mmInfo.m, mmInfo.k, mmInfo.transA, nElemAlign) && mmInfo.m != 1)) {
        workspaceDetail.matrixActivationSize =
            GetAlignedMatrixSize(mmInfo.batchSize, mmInfo.m, mmInfo.k, mmInfo.transA, nElemAlign) * eleSize;
    }

    bool hasDequant = quantInfo.dequantGranularity != QuantGranularity::QUANT_GRANULARITY_UNDEFINED;
    if ((hasDequant && !mmInfo.isInt8) || !IsMatrixAligned(mmInfo.k, mmInfo.n, mmInfo.transB, nElemAlign)) {
        workspaceDetail.matrixWeightSize =
            GetAlignedMatrixSize(mmInfo.batchSize, mmInfo.k, mmInfo.n, mmInfo.transB, nElemAlign) * eleSize;
    }

    bool hasAccum = dataType == CoCDataTypeDesc::INT8INT8_INT32_BF16;
    if (hasAccum) {
        workspaceDetail.matrixIntermediateSize =
        static_cast<int64_t>(mmInfo.batchSize) * mmInfo.m * mmInfo.n * sizeof(int32_t);
    }

    if (mmInfo.isInt8) {
        workspaceDetail.formatDequantParamSize =
        mmInfo.k > mmInfo.n ? mmInfo.k * sizeof(float) : mmInfo.n * sizeof(float);
    }
    return workspaceDetail;
}

void GetMmInfo(gert::TilingContext *context, MatmulAllreduceAddRmsnormTilingData *tiling, MatMulInfo *mmInfo)
{
    PPTilingData tempPPTilingData = tiling->matmulAllreduceAddRmsnormInfo.ppTilingData;
    mmInfo->batchSize = tempPPTilingData.opShape.batchSize;
    mmInfo->m = tempPPTilingData.opShape.m;
    mmInfo->n = tempPPTilingData.opShape.n;
    mmInfo->k = tempPPTilingData.opShape.k;
    auto attrs = context->GetAttrs();
    mmInfo->transA = false;
    mmInfo->transB = *(attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_B_INDEX));
    mmInfo->withBias = false;
    mmInfo->weightNz = false;
    mmInfo->isInt8 = context->GetInputTensor(0)->GetDataType() == ge::DT_INT8;
}

size_t GetUserWorkspaceSize(gert::TilingContext *context, MatmulAllreduceAddRmsnormTilingData *tiling)
{
    MatMulInfo mmInfo;
    GetMmInfo(context, tiling, &mmInfo);
    QuantInfo quantInfo = tiling->matmulAllreduceAddRmsnormInfo.quantInfo;
    return GetWorkspaceDetail(FP16FP16_FP32_FP16, mmInfo, quantInfo).GetSize();
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OPS_ERR_IF(workSpaces == nullptr, OPS_LOG_E(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    MatmulAllreduceAddRmsnormTilingData *tilingData = context->GetTilingData<MatmulAllreduceAddRmsnormTilingData>();
    size_t userWorkspaceSize = GetUserWorkspaceSize(context, tilingData);
    workSpaces[0] = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg(
    const gert::TilingContext *context, MatmulAllreduceAddRmsnormTilingData *tiling, const std::string groupTp)
{
    const char *nodeName = context->GetNodeName();
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(groupTp, opType, algConfigAllToAllStr);
    mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling);
}

static ge::graphStatus MatmulAllreduceAddRmsnormTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    MatmulAllreduceAddRmsnormTilingData *tilingData = context->GetTilingData<MatmulAllreduceAddRmsnormTilingData>();
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

static ge::graphStatus MatmulAllreduceAddRmsnormTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = MatmulAllreduceAddRmsnormTilingFuncImpl(context);
    return ret;
}

struct MatmulAllreduceAddRmsnormCompileInfo {};
ge::graphStatus TilingParseForMatmulAllreduceAddRmsnorm(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulAllreduceAddRmsnorm)
    .Tiling(MatmulAllreduceAddRmsnormTilingFunc)
    .TilingParse<MatmulAllreduceAddRmsnormCompileInfo>(TilingParseForMatmulAllreduceAddRmsnorm);
