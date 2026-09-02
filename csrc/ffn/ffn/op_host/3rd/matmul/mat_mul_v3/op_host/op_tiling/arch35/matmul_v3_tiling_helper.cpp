/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_tiling_helper.cc
 * \brief
 */

#include <cerrno>
#include <cstdlib>
#include <cmath>
#include "matmul_v3_tiling_helper.h"
#include "matmul/common/op_host/math_util.h"
#include "platform/platform_infos_def.h"

using Ops::NN::MathUtil;
namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;
using StrideIndexPairs = std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>>;

int64_t StrToIntWithDefault(const std::string& str, int64_t defaultValue)
{
    if (str.empty()) {
        return defaultValue;
    }
    const char* s = str.c_str();
    char* endPtr = nullptr;
    errno = 0;
    long long parsed = std::strtoll(s, &endPtr, 10);
    if (errno != 0 || endPtr == s || *endPtr != '\0' || parsed < 0) {
        return defaultValue;
    }
    return static_cast<int64_t>(parsed);
}

// ------------------------------ CalL1Tiling -------------------------------------------//
void CalL1TilingDefault(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo)
{
    bool isKInner = !args.isATrans || args.isBTrans;
    uint64_t totalL1Size = compileInfo.l1Size - (args.hasBias ? runInfo.baseN * DB_SIZE * DATA_SIZE_FP32 : 0UL);
    totalL1Size -= args.hasScale ? runInfo.baseN * sizeof(uint64_t) : 0UL;
    // Shape约束 && issue queue约束
    uint64_t maxStepK = std::min(MathUtil::CeilDivision(args.kValue, runInfo.baseK), 8UL);
    uint64_t kAlignUnit = isKInner ? BASIC_BLOCK_K_512_BYTE / args.aDtypeSize : BASIC_BLOCK_SIZE_16;
    uint64_t resKL1 = 0;
    uint64_t singleMteSize = 0;
    for (uint64_t stepK = 1; stepK <= maxStepK; stepK++) {
        uint64_t curKL1 = runInfo.baseK * stepK;
        uint64_t aL1Size = runInfo.baseM * curKL1 * args.aDtypeSize;
        uint64_t bL1Size = runInfo.baseN * curKL1 * args.bDtypeSize;
        if ((aL1Size + bL1Size) * DB_SIZE > totalL1Size ||
            std::max(aL1Size, bL1Size) * DB_SIZE * NUM_TWO > compileInfo.l1Size) {
            break;
        }
        bool condNoRes = resKL1 == 0;
        bool condKAlign256B = curKL1 % (BASIC_BLOCK_K_256_BYTE / args.aDtypeSize) == 0;
        bool condKAlign = resKL1 % kAlignUnit != 0 &&
                          (condKAlign256B || (!condKAlign256B && singleMteSize < L1_SINGLE_SIZE_LIMIT));
        bool condMteSize = resKL1 % kAlignUnit == 0 && curKL1 % kAlignUnit == 0 && singleMteSize < L1_SINGLE_SIZE_LIMIT;
        if (condNoRes || condKAlign || condMteSize) {
            resKL1 = curKL1;
            singleMteSize = std::max(aL1Size, bL1Size);
        }
    }
    runInfo.stepKa = resKL1 / runInfo.baseK;
    runInfo.stepKb = resKL1 / runInfo.baseK;
    runInfo.depthA1 = runInfo.stepKa * DB_SIZE;
    runInfo.depthB1 = runInfo.stepKb * DB_SIZE;
    runInfo.singleCoreM = runInfo.baseM;
    runInfo.singleCoreN = runInfo.baseN;
    return;
}

void CalL1Tiling310P(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo)
{
    runInfo.stepM = 1UL;
    runInfo.stepN = 1UL;
    runInfo.depthA1 = 16UL; // 16 is full use l1 space
    runInfo.depthB1 = 16UL; // 16 is full use l1 space
    runInfo.singleCoreM = runInfo.baseM;
    runInfo.singleCoreN = runInfo.baseN;

    if (args.aFormat == ge::FORMAT_ND || args.bFormat == ge::FORMAT_ND) {
        runInfo.depthA1 = 6UL; // 6为经验值
        runInfo.depthB1 = 6UL; // 6为经验值
    }
    runInfo.stepKa = runInfo.depthA1 / DB_SIZE;
    runInfo.stepKb = runInfo.depthB1 / DB_SIZE;
    if ((args.aFormat == ge::FORMAT_FRACTAL_NZ) && (args.bFormat == ge::FORMAT_FRACTAL_NZ)) {
        // ka全载
        if (args.mValue <= runInfo.baseM) {
            runInfo.baseM = args.mValue;
            if (runInfo.baseM * args.kValue * args.aDtypeSize < (compileInfo.l1Size / NUM_TWO)) {
                runInfo.depthA1 = MathUtil::CeilDivision(args.kValue, runInfo.baseK);
                runInfo.stepKa = runInfo.depthA1;
            }
        }
        // kb全载
        if (args.nValue <= runInfo.baseN) {
            runInfo.baseN = args.nValue;
            if (runInfo.baseN * args.kValue * args.bDtypeSize < (compileInfo.l1Size / NUM_TWO)) {
                runInfo.depthB1 = MathUtil::CeilDivision(args.kValue, runInfo.baseK);
                runInfo.stepKb = runInfo.depthB1;
            }
        }
    }
    return;
}

using CalL1TilingFunc = void (*)(const MatmulV3CompileInfo&, const MatMulV3Args&, MatMulV3RunInfo&);

const static std::map<NpuArch, CalL1TilingFunc> CalL1TilingFuncMap = {
    {NpuArch::DAV_2002, CalL1Tiling310P},
};

// ------------------------------ ResetBase -------------------------------------------//
void ResetBaseDefault(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo)
{
    runInfo.usedCoreNum = compileInfo.aicNum;
    runInfo.baseM = BASIC_BLOCK_SIZE_128;
    runInfo.baseN = BASIC_BLOCK_SIZE_256; // 256 is better base
    runInfo.baseK = BASIC_BLOCK_K_128_BYTE / args.aDtypeSize;
    runInfo.stepM = BASE_STEP;
    runInfo.stepN = BASE_STEP;
    runInfo.iterateOrder = ITER_COL_FIRST;
    runInfo.dbL0C = DB_OFF_SIZE;
    runInfo.singleCoreK = args.kValue;
    runInfo.singleCoreM = runInfo.baseM;
    runInfo.singleCoreN = runInfo.baseN;
    runInfo.mBaseTailSplitCnt = INIT_SPLIT_CNT;
    runInfo.nBaseTailSplitCnt = INIT_SPLIT_CNT;
    runInfo.tailInfo.mTailMain = INIT_SPLIT_VALUE;
    runInfo.tailInfo.nTailMain = INIT_SPLIT_VALUE;
}

void ResetBaseDav3510(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo)
{
    ResetBaseDefault(compileInfo, args, runInfo);
    runInfo.baseM = BASIC_BLOCK_SIZE_256;
    runInfo.singleCoreM = runInfo.baseM;
}

using ResetBaseFunc = void (*)(const MatmulV3CompileInfo&, const MatMulV3Args&, MatMulV3RunInfo&);

const static std::map<NpuArch, ResetBaseFunc> ResetBaseFuncMap = {
    {NpuArch::DAV_3510, ResetBaseDav3510},
};

// ------------------------------ GetL0C2Out -------------------------------------------//
MatMulV3L0C2Out GetL0C2OutDefault(const MatmulV3CompileInfo& /* compileInfo */, const MatMulV3Args& /* args */,
                                  const MatMulV3RunInfo& /* runInfo */)
{
    return MatMulV3L0C2Out::ON_THE_FLY;
}

MatMulV3L0C2Out GetL0C2OutDav3510(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                  const MatMulV3RunInfo& runInfo)
{
    bool isValidMKN = args.kValue <= BASIC_BLOCK_SIZE_256 && args.mValue >= BASIC_BLOCK_SIZE_256;
    uint64_t mCnt = MathUtil::CeilDivision(args.mValue, runInfo.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(args.nValue, runInfo.singleCoreN);
    // make sure the fixpipe stream is large enough to be bound
    bool isMultiRound = mCnt * nCnt > compileInfo.aicNum;
    uint64_t cDtypeSize = ge::GetSizeByDataType(args.cType);
    // 128: SMALL_SHAPE_LOWER_THRES
    bool isUnalignedN = args.nValue * cDtypeSize % 128UL != 0 && args.nValue * cDtypeSize > BASIC_BLOCK_SIZE_256;
    bool fixpipeBound = isValidMKN && isMultiRound && isUnalignedN;
    if (!fixpipeBound || (compileInfo.aivNum != (compileInfo.aicNum * NUM_TWO))) {
        return MatMulV3L0C2Out::ON_THE_FLY;
    }
    if (args.aType == ge::DT_FLOAT16 || args.aType == ge::DT_BF16) {
        return MatMulV3L0C2Out::ND_FIXPIPE_1_1;
    }
    return MatMulV3L0C2Out::ND_FIXPIPE_1_2;
}

using GetL0C2OutFunc = MatMulV3L0C2Out (*)(const MatmulV3CompileInfo&, const MatMulV3Args&, const MatMulV3RunInfo&);

const static std::map<NpuArch, GetL0C2OutFunc> GetL0C2OutFuncMap = {
    {NpuArch::DAV_3510, GetL0C2OutDav3510},
};

uint64_t GetMaxBaseWithLimit(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                             uint64_t baseMNBufferLimit, uint64_t baseAlignUnit, bool isRightMatrix, bool isMemoryBound)
{
    uint64_t shapeValue = isRightMatrix ? args.nValue : args.mValue;
    // baseK限制，cube bound场景需要掩盖fixp，约束baseK至少128B，其他场景不做约束
    uint64_t kAlignValue = ops::CeilAlign(args.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t kLimitValue = isMemoryBound ? BASIC_BLOCK_SIZE_16 : BASIC_BLOCK_K_128_BYTE / args.aDtypeSize;
    uint64_t minKL0 = std::min(kLimitValue, kAlignValue) * args.aDtypeSize;
    // L0 buffer限制
    uint64_t maxBaseMNWithBuffer = baseMNBufferLimit / DATA_SIZE_FP32 / BASIC_BLOCK_SIZE_16;
    uint64_t maxBaseBlock = std::min(compileInfo.l0ASize / DB_SIZE / minKL0, maxBaseMNWithBuffer);
    // bias table约束
    if (isRightMatrix && args.hasBias) {
        maxBaseBlock = std::min(maxBaseBlock, compileInfo.btSize / DB_SIZE / DATA_SIZE_FP32);
    }
    // K内轴时，要求kL1至少256B对齐,目前batchmatmul固定内轴512B对齐
    uint64_t kAlignUnit = !args.isATrans || args.isBTrans ?
                              (isMemoryBound && args.batchInfo == nullptr ? BASIC_BLOCK_K_256_BYTE :
                                                                            BASIC_BLOCK_K_512_BYTE) /
                                  args.aDtypeSize :
                              BASIC_BLOCK_SIZE_16;
    uint64_t maxBaseMNWithKInner = compileInfo.l1Size /
                                   (NUM_TWO * DB_SIZE * args.aDtypeSize * std::min(kAlignUnit, kAlignValue));
    maxBaseBlock = std::min(maxBaseBlock, maxBaseMNWithKInner);
    // 输入shape约束
    maxBaseBlock = std::min(ops::CeilAlign(shapeValue, baseAlignUnit), ops::FloorAlign(maxBaseBlock, baseAlignUnit));
    if (shapeValue < baseAlignUnit) {
        maxBaseBlock = std::min(maxBaseBlock, ops::CeilAlign(shapeValue, BASIC_BLOCK_SIZE_16));
    }
    return maxBaseBlock;
}

static double GetBalanceRateWithTail(const MatMulV3Args& args, uint64_t usedCoreNum, uint64_t baseM, uint64_t baseN)
{
    // 考虑尾轮优化负载均衡率，仅针对cubebound场景生效
    uint64_t batch = args.batchInfo == nullptr ? 1 : args.batchInfo->batchA;
    uint64_t totalRound = batch * MathUtil::CeilDivision(args.mValue, baseM) *
                          MathUtil::CeilDivision(args.nValue, baseN);
    uint64_t mainRound = MathUtil::CeilDivision(totalRound, usedCoreNum) - 1;
    uint64_t totalTailSplit = ops::FloorDiv(usedCoreNum, (totalRound - usedCoreNum * mainRound));
    if (args.mValue <= BASIC_BLOCK_SIZE_16) {
        baseM = args.mValue;
    }
    if (args.nValue <= BASIC_BLOCK_SIZE_16) {
        baseN = args.nValue;
    }
    if (mainRound == 0 || ops::FloorDiv(baseM * baseN, totalTailSplit) < MIN_TATL_BLOCK_SIZE ||
        args.batchInfo != nullptr) {
        return (static_cast<double>(batch) * args.mValue * args.nValue / usedCoreNum) /
               ((mainRound + 1) * baseM * baseN);
    }
    uint64_t tailSplitSqrt = static_cast<uint64_t>(std::sqrt(totalTailSplit));
    uint64_t offset = ops::FloorDiv(totalTailSplit - tailSplitSqrt * tailSplitSqrt, tailSplitSqrt) + 1;
    double tailRound = 1.0 / (tailSplitSqrt * (tailSplitSqrt + offset - 1));
    return (static_cast<double>(args.mValue) * args.nValue / usedCoreNum) / ((mainRound + tailRound) * baseM * baseN);
}

static void GetBaseK(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo)
{
    uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(args.kValue), BASIC_BLOCK_SIZE_16);
    uint64_t maxBaseK = compileInfo.l0ASize / DB_SIZE / args.aDtypeSize / std::max(runInfo.baseM, runInfo.baseN);
    if (kValueAlign <= maxBaseK) {
        runInfo.baseK = kValueAlign;
        return;
    }
    if (args.isATrans && !args.isBTrans) {
        runInfo.baseK = ops::FloorAlign(maxBaseK, BASIC_BLOCK_SIZE_16);
        return;
    }
    if (maxBaseK * args.aDtypeSize >= BASIC_BLOCK_K_256_BYTE) {
        runInfo.baseK = ops::FloorAlign(maxBaseK, BASIC_BLOCK_K_256_BYTE / args.aDtypeSize);
        return;
    }
    std::vector<uint64_t> baseKCandidate = {128, 64, 32, 16};
    for (uint64_t baseK : baseKCandidate) {
        if (maxBaseK >= baseK) {
            runInfo.baseK = baseK;
            return;
        }
    }
}
} // namespace

namespace optiling {
namespace matmul_v3_advanced {
void MatMulV3TilingHelper::ResetBase(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                     MatMulV3RunInfo& runInfo)
{
    auto iter = (ResetBaseFuncMap.find(compileInfo.npuArch) == ResetBaseFuncMap.end()) ?
                    ResetBaseDefault :
                    ResetBaseFuncMap.at(compileInfo.npuArch);
    iter(compileInfo, args, runInfo);
}

void MatMulV3TilingHelper::CalL1Tiling(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                       MatMulV3RunInfo& runInfo)
{
    auto iter = (CalL1TilingFuncMap.find(compileInfo.npuArch) == CalL1TilingFuncMap.end()) ?
                    CalL1TilingDefault :
                    CalL1TilingFuncMap.at(compileInfo.npuArch);
    iter(compileInfo, args, runInfo);
}

MatMulV3L0C2Out MatMulV3TilingHelper::GetL0C2Out(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                                 const MatMulV3RunInfo& runInfo)
{
    auto iter = (GetL0C2OutFuncMap.find(compileInfo.npuArch) == GetL0C2OutFuncMap.end()) ?
                    GetL0C2OutDefault :
                    GetL0C2OutFuncMap.at(compileInfo.npuArch);
    return iter(compileInfo, args, runInfo);
}

bool MatMulV3TilingHelper::IsSelfNonContiguous(const gert::TilingContext* context)
{
    auto selfShape = context->GetInputShape(0)->GetOriginShape();
    auto mat2Shape = context->GetInputShape(1)->GetOriginShape();
    auto selfStorageShape = context->GetInputShape(0)->GetStorageShape();
    size_t selfDimNum = selfShape.GetDimNum();
    size_t mat2DimNum = mat2Shape.GetDimNum();
    // createView with TensorV2 & storageShape 1d ->  non contiguous
    return (context->InputIsView(0) && selfStorageShape.GetDimNum() == NUM_ONE && selfDimNum == NUM_THREE &&
            mat2DimNum == NUM_TWO);
}

bool MatMulV3TilingHelper::IsTransposeNonContiguous(const gert::TilingContext* context, uint64_t idx)
{
    // 获得stride 然后根据stride判断
    auto viewShape = context->GetInputShape(idx)->GetOriginShape();
    auto viewStride = context->GetInputStride(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, viewStride);
    int64_t dimNum = viewStride->GetDimNum();
    StrideIndexPairs strideIndexPairs;
    strideIndexPairs.reserve(dimNum);
    auto lastStride = INT64_MAX;
    bool isTranspose = false;
    for (int64_t i = 0; i < dimNum; i++) {
        int64_t curStride = viewStride->GetStride(i);
        if (curStride == 0 || viewShape[i] == 1) {
            return false;
        }
        if (lastStride < curStride) {
            isTranspose = true;
        }
        lastStride = curStride;
        strideIndexPairs.emplace_back(std::make_pair(curStride, std::make_pair(i, viewShape[i])));
    }
    if (!isTranspose) {
        return false;
    }
    // strides顺序排序
    std::sort(strideIndexPairs.rbegin(), strideIndexPairs.rend());
    if (!IsContiguousStride(strideIndexPairs)) {
        return false;
    }
    std::vector<int> indices;
    for (auto it = strideIndexPairs.begin(); it != strideIndexPairs.end(); it++) {
        indices.push_back(it->second.first);
    }
    // ACLNN已将需要交换末两维的场景归一化，tiling侧仅接收{1, 0, 2}
    const std::vector<int> transposeIndexs = {1, 0, 2};
    return indices == transposeIndexs;
}

bool MatMulV3TilingHelper::IsContiguousStride(StrideIndexPairs& strideIndexPairs)
{
    int64_t expectStride = 1;
    for (auto it = strideIndexPairs.rbegin(); it != strideIndexPairs.rend(); it++) {
        if (it->first != expectStride) {
            return false;
        }
        expectStride *= it->second.second;
    }
    return true;
}

void MatMulV3TilingHelper::GetRebalanceBlock(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args,
                                             MatMulV3RunInfo& runInfo, const gert::TilingContext* context)
{
    // 获取规格，计算相关性能指标
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context->GetNodeName(), "platformInfo is null");
        return;
    }
    double hbmBW = GetHbmBW(platformInfo);
    double l2BW = GetL2BW(platformInfo);
    double singleCoreComputePower = GetCoreFreq(platformInfo) * NUM_EIGHT;
    bool isFP32 = args.aType == ge::DT_FLOAT && !args.isHf32;
    if (compileInfo.npuArch == NpuArch::DAV_3510 && isFP32) {
        singleCoreComputePower /= BASIC_BLOCK_SIZE_16;
    }
    // balanceRateEdge用于判断是否取得最优解，进行减枝, 默认0.9
    double balanceRateEdge = 0.9;
    double cmr = (static_cast<double>(args.mValue) + args.nValue) / (static_cast<double>(args.mValue) * args.nValue);
    double computePower = singleCoreComputePower * compileInfo.aicNum;

    // 切K场景，要求输出size同时小于L0C和UB
    uint64_t baseMNBufferLimit = runInfo.usedCoreNum == compileInfo.aicNum ?
                                     compileInfo.l0CSize :
                                     std::min(compileInfo.l0CSize, compileInfo.ubSize);
    if (args.preferL0cDB2) {
        // swiglu 单 matmul 融合：AIV epilogue 需要 L0C 乒乓（dbL0C=2）才能与 mmad 重叠，
        // 限制 baseM*baseN*4*2 <= l0CSize，即 tile 最多占半个 L0C。
        baseMNBufferLimit = compileInfo.l0CSize / NUM_TWO;
    }

    uint64_t batchNum = args.batchInfo == nullptr ? 1 : args.batchInfo->batchA;
    double l2CacheUsage = std::max(
        static_cast<double>(batchNum * (args.mValue + args.nValue) * args.kValue * args.aDtypeSize) /
            compileInfo.l2Size,
        1.0);
    runInfo.cubeBoundEdge = (l2BW / computePower) + l2CacheUsage * (1 - l2BW / hbmBW) * cmr -
                            (1 + l2BW / hbmBW) / args.kValue;
    uint64_t baseMBest = std::min(ops::CeilAlign(args.mValue, BASIC_BLOCK_SIZE_16), BASIC_BLOCK_SIZE_256);
    uint64_t baseNBest = std::max(
        BASIC_BLOCK_SIZE_16,
        std::min(ops::CeilAlign(args.nValue, BASIC_BLOCK_SIZE_16),
                 ops::FloorAlign(baseMNBufferLimit / DATA_SIZE_FP32 / baseMBest, BASIC_BLOCK_SIZE_16)));
    double cubeBoundParamBest = (1.0 / baseMBest) + (1.0 / baseNBest);
    bool isMemoryBound = cubeBoundParamBest > runInfo.cubeBoundEdge;
    uint64_t innerAlignUnit = isMemoryBound ? BASIC_BLOCK_SIZE_128 : BASIC_BLOCK_SIZE_64;

    // fixpipe bound场景下，要求baseN是256B对齐，发挥搬出带宽
    double fixpBoundEdge = (args.mValue * args.nValue * hbmBW) / ((args.mValue + args.nValue) * l2BW);
    uint64_t baseMAlignUnit = args.isATrans ? innerAlignUnit / args.aDtypeSize : BASIC_BLOCK_SIZE_16;
    uint64_t baseNAlignUnit = (static_cast<double>(args.kValue) < fixpBoundEdge) ?
                                  (BASIC_BLOCK_K_256_BYTE / args.bDtypeSize) :
                                  (args.isBTrans ? BASIC_BLOCK_SIZE_16 : innerAlignUnit / args.bDtypeSize);
    if (args.swigluSingleNAlign32) {
        // swiglu 单 matmul：epilogue 半宽输出（baseN/2 个 bf16 = baseN 字节）要求 32B 对齐，
        // 因此 baseN 必须为 32 的倍数。在搜索包络内收口（初值/上界/候选步长全部走 32 对齐），
        // 而不是搜索后强改单个 baseN（会绕过 L1/L0C/波次容量约束）。
        baseNAlignUnit = std::max(baseNAlignUnit, BASIC_BLOCK_SIZE_32);
    }

    // 计算候选解集的上界
    uint64_t maxBaseM = GetMaxBaseWithLimit(compileInfo, args, baseMNBufferLimit, baseMAlignUnit, false, isMemoryBound);
    uint64_t maxBaseN = GetMaxBaseWithLimit(compileInfo, args, baseMNBufferLimit, baseNAlignUnit, true, isMemoryBound);

    runInfo.baseM = std::max(BASIC_BLOCK_SIZE_16, std::min(maxBaseM, BASIC_BLOCK_SIZE_256));
    runInfo.baseN = std::max(
        BASIC_BLOCK_SIZE_16,
        std::min(maxBaseN, ops::FloorAlign(baseMNBufferLimit / DATA_SIZE_FP32 / runInfo.baseM, baseNAlignUnit)));
    runInfo.cubeBoundParam = (1.0 / runInfo.baseM) + (1.0 / runInfo.baseN);
    runInfo.cubeBoundEdge = runInfo.cubeBoundEdge * CUBE_BOUND_RATIO;
    double balanceRate = GetBalanceRateWithTail(args, runInfo.usedCoreNum, runInfo.baseM, runInfo.baseN);

    for (uint64_t curBaseM = maxBaseM; curBaseM >= 1 && curBaseM <= maxBaseM; curBaseM -= baseMAlignUnit) {
        if (args.preferNoMSplit && curBaseM < ops::CeilAlign(args.mValue, BASIC_BLOCK_SIZE_16)) {
            // 不拆 M：只保留覆盖整个 M 的 tile，让 N 拆分承担负载均衡（代价是少用几个核，
            // 换取 B 权重只搬运一遍）。FFN down matmul 实验开关 FFN_DOWN_NOMSPLIT 使用。
            continue;
        }
        uint64_t curMaxBaseN = std::min(maxBaseN,
                                        ops::FloorAlign(baseMNBufferLimit / DATA_SIZE_FP32 / curBaseM, baseNAlignUnit));
        for (uint64_t curBaseN = curMaxBaseN; curBaseN >= 1 && curBaseN <= curMaxBaseN; curBaseN -= baseNAlignUnit) {
            double curCubeBoundParam = (1.0 / curBaseM) + (1.0 / curBaseN);
            double curBalanceRate = GetBalanceRateWithTail(args, runInfo.usedCoreNum, curBaseM, curBaseN);
            // 当前最优解满足负载均衡阈值时，本轮解集无法在计算访存拿到收益时过滤本轮解集
            bool skipCond = balanceRate >= balanceRateEdge && curCubeBoundParam > runInfo.cubeBoundParam &&
                            curCubeBoundParam > runInfo.cubeBoundEdge && runInfo.cubeBoundEdge > 0;
            // FP32 cubeBound 场景且mCnt*nCnt大于核数时，baseM/baseN需大于64
            skipCond |= isFP32 && curCubeBoundParam < runInfo.cubeBoundEdge &&
                        ((curBaseM < FP32_MIN_BASE_BLOCK || curBaseN < FP32_MIN_BASE_BLOCK) &&
                         MathUtil::CeilDivision(args.mValue, curBaseM) * MathUtil::CeilDivision(args.nValue, curBaseN) >
                             compileInfo.aicNum);
            if (skipCond) {
                continue;
            }
            // 当前解满足cubebound并且负载均衡率更高
            bool cubeBoundCond = curCubeBoundParam <= runInfo.cubeBoundEdge && curBalanceRate > balanceRate;
            // 综合评选负载均衡和计算访存能力
            bool balanceCond = ((curCubeBoundParam / curBalanceRate) < (runInfo.cubeBoundParam / balanceRate)) ||
                               ((std::abs(curCubeBoundParam / curBalanceRate - runInfo.cubeBoundParam / balanceRate) <
                                 EPSILON) &&
                                curBalanceRate > balanceRate);
            bool updateCond = cubeBoundCond || balanceCond;
            if (updateCond) {
                if (cubeBoundCond) {
                    runInfo.cubeBoundEdge = curCubeBoundParam;
                }
                runInfo.baseM = curBaseM;
                runInfo.baseN = curBaseN;
                runInfo.cubeBoundParam = curCubeBoundParam;
                balanceRate = curBalanceRate;
            }
        }
    }
    runInfo.baseM = std::min(ops::CeilAlign(args.mValue, BASIC_BLOCK_SIZE_16), runInfo.baseM);
    runInfo.baseN = std::min(ops::CeilAlign(args.nValue, BASIC_BLOCK_SIZE_16), runInfo.baseN);
    if (args.preferL0cMSplitDB2 && runInfo.baseM > BASIC_BLOCK_SIZE_16 &&
        runInfo.baseM * runInfo.baseN * DATA_SIZE_FP32 * DB_SIZE > compileInfo.l0CSize) {
        // 沿 M 拆半（baseN 保持不变，保住 mmad N 宽度），把 tile 压到 L0C 一半，
        // 使 dbL0C=2：下一 tile 的 mmad 可与本 tile 的 L0C→UB fixpipe/epilogue 乒乓。
        runInfo.baseM = std::max(BASIC_BLOCK_SIZE_16, ops::FloorAlign(runInfo.baseM / NUM_TWO, BASIC_BLOCK_SIZE_16));
    }
    if (args.preferUbDB2 && runInfo.baseN > BASIC_BLOCK_SIZE_32 &&
        runInfo.baseM * runInfo.baseN * DATA_SIZE_FP32 > compileInfo.ubSize) {
        // 沿 N 拆半（baseM 保持不变，B 权重的 mCnt 重复搬运不变），把 tile 压入单 UB，
        // 使 ubDB=2：AIV epilogue 与下一 tile 的 L0C→UB fixpipe 乒乓（FFN up 融合相，
        // ubDB=1 时 epilogue 在相位边界串行暴露）。32 对齐兼容 16/32 及 fixpipe 256B 档。
        runInfo.baseN = std::max(BASIC_BLOCK_SIZE_32, ops::FloorAlign(runInfo.baseN / NUM_TWO, BASIC_BLOCK_SIZE_32));
    }
    GetBaseK(compileInfo, args, runInfo);
    uint64_t mCore = MathUtil::CeilDivision(args.mValue, runInfo.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args.nValue, runInfo.baseN);
    runInfo.usedCoreNum = std::min(batchNum * mCore * nCore, runInfo.usedCoreNum);
    runInfo.dbL0C = runInfo.baseM * runInfo.baseN * DATA_SIZE_FP32 * DB_SIZE <= compileInfo.l0CSize ? DB_SIZE : 1UL;
    runInfo.mixInfo.ubDB = runInfo.baseM * runInfo.baseN * DATA_SIZE_FP32 <= compileInfo.ubSize ? DB_SIZE : 1UL;
}

double MatMulV3TilingHelper::GetHbmBW(fe::PlatFormInfos* platformInfo)
{
    std::string coreCntStr = "32";
    std::string ddrRateStr = "31";
    platformInfo->GetPlatformRes("SoCInfo", "ai_core_cnt", coreCntStr);
    platformInfo->GetPlatformRes("AICoreMemoryRates", "ddr_rate", ddrRateStr);
    // 32:default coreCntStr; 31:default ddrRateStr;
    return GetCoreFreq(platformInfo) * StrToIntWithDefault(coreCntStr, 32) * StrToIntWithDefault(ddrRateStr, 31) /
           KB_SIZE;
}

double MatMulV3TilingHelper::GetL2BW(fe::PlatFormInfos* platformInfo)
{
    std::string coreCntStr = "32";
    std::string l2RateStr = "100";
    platformInfo->GetPlatformRes("SoCInfo", "ai_core_cnt", coreCntStr);
    platformInfo->GetPlatformRes("AICoreMemoryRates", "l2_rate", l2RateStr);
    // 32:default coreCntStr; 100:default coreCntStr;
    return GetCoreFreq(platformInfo) * StrToIntWithDefault(coreCntStr, 32) * StrToIntWithDefault(l2RateStr, 100) /
           KB_SIZE;
}

double MatMulV3TilingHelper::GetCoreFreq(fe::PlatFormInfos* platformInfo)
{
    std::string freqStr = "1650";
    platformInfo->GetPlatformRes("AICoreSpec", "cube_freq", freqStr);
    // 1650:default freqStr;
    return StrToIntWithDefault(freqStr, 1650) / static_cast<double>(THOUSAND_NUM);
}
} // namespace matmul_v3_advanced
} // namespace optiling
