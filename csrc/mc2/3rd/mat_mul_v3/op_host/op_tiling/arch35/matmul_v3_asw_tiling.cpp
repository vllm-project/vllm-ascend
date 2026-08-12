/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_asw_tiling.cc
 * \brief
 */
#include "matmul_v3_asw_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "./matmul_tiling_registry.h"
#include "common/op_host/math_util.h"

using Ops::Transformer::MathUtil;

namespace {
using namespace optiling;
using namespace mc2_matmul_v3_advanced;

constexpr uint64_t MIN_BASE_BLOCK = 112UL;
constexpr uint64_t MAX_BASE_BLOCK = 336UL;
constexpr double LOAD_BALANCING_THRESHOLD = 0.98;
// 负载均衡与重复搬运量归一化阈值
constexpr double BALANCE_REDUNDANT_THRESHOLD = 0.97;
constexpr double MIN_EQUALIZATION_COEFFICIENT = 1.02;
constexpr double LOAD_BALANCE_RATE_LIMIT = 0.88;
constexpr double MAX_SINGLE_CORE_ROUND = 10.0;
constexpr int MAX_LOOP_NUM = 20;
constexpr uint64_t NUM_NINE = 9UL;
constexpr uint64_t NUM_TEN = 10UL;

// 计算多核负载均衡率
static double CalcMultiCoreBalance(uint64_t M, uint64_t N, uint64_t coreNum, uint64_t baseM, uint64_t baseN)
{
    if (baseM == 0UL || baseN == 0UL) {
        return 0.0;
    }
    uint64_t mCnt = MathUtil::CeilDivision(M, baseM);
    uint64_t nCnt = MathUtil::CeilDivision(N, baseN);
    uint64_t mTail = M % baseM;
    uint64_t nTail = N % baseN;
    uint64_t mMainCnt = M / baseM;
    uint64_t nMainCnt = N / baseN;
    uint64_t totalTiles = mCnt * nCnt;
    uint64_t totalMainTiles = mMainCnt * nMainCnt;
    if (totalTiles == 0UL || coreNum == 0UL) {
        return 0.0;
    }
    // 计算每个核分配的块数分布
    double avgLoad = static_cast<double>(M * N) / coreNum;

    // 单核计算最大尾块
    uint64_t singleMaxTail = std::max(baseM * nTail, baseN * mTail);

    // 尾块数
    uint64_t coreTailNum = totalTiles % coreNum;
    // 尾块均分比例
    uint64_t tailRatio = coreTailNum != 0UL ? coreNum / coreTailNum : 1UL;

    auto getDiv = [](uint64_t base) -> uint64_t {
        if (base % BASIC_BLOCK_SIZE_256 == 0UL) {
            return BASIC_BLOCK_SIZE_16;
        } else if (base % BASIC_BLOCK_SIZE_128 == 0UL) {
            return BASIC_BLOCK_SIZE_16 / NUM_TWO;
        } else if (base % BASIC_BLOCK_SIZE_64 == 0UL) {
            return NUM_FOUR;
        } else if (base % BLOCK_BYTE_SIZE == 0UL) {
            return NUM_TWO;
        } else {
            return 1UL;
        }
    };
    uint64_t mDiv = getDiv(baseM);
    uint64_t nDiv = getDiv(baseN);

    tailRatio = std::min(tailRatio, mDiv * nDiv);
    uint64_t tailNum =
        MathUtil::CeilDivision(mCnt * nCnt, coreNum) - MathUtil::CeilDivision(mMainCnt * nMainCnt, coreNum);
    uint64_t noDivTailNum = tailNum >= NUM_TWO ? tailNum - 1UL : 0UL;

    // 单核实际计算量
    double maxLoad;
    if (tailNum == 0UL) {
        // 主块+无法切分主块的负载均衡
        maxLoad = static_cast<double>(totalMainTiles / coreNum) * baseM * baseN +
                  static_cast<double>(MathUtil::CeilDivision(totalMainTiles, coreNum) - totalMainTiles / coreNum) *
                      baseM * baseN / tailRatio;
    } else {
        // 主块向上取整+尾块的负载均衡
        maxLoad =
            static_cast<double>(MathUtil::CeilDivision(totalMainTiles, coreNum) * baseM * baseN +
                                noDivTailNum * singleMaxTail + (tailNum - noDivTailNum) * singleMaxTail / tailRatio);
    }
    return avgLoad / maxLoad; // 负载均衡率：1.0为完全均衡
}

// 计算重复搬运量
static uint64_t CalcRedundantDataMovement(uint64_t baseM, uint64_t baseN, uint64_t mValue, uint64_t nValue)
{
    uint64_t mBlocks = MathUtil::CeilDivision(mValue, baseM);
    uint64_t nBlocks = MathUtil::CeilDivision(nValue, baseN);
    uint64_t totalMovement = mBlocks * nValue + nBlocks * mValue;
    return totalMovement - (mValue + nValue);
}

// X1维度档位点快速查找索引表{x0, x1, x2} x0:待查找的base块， base块可用的列表个数，base块对应的TABLE索引
const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> X1_LOOKUP_TABLE = {
    {336, 6, 0},   {320, 5, 6},   {304, 6, 11},  {288, 7, 17},  {272, 8, 24},
    {256, 9, 32},  {240, 10, 41}, {224, 10, 51}, {208, 11, 61}, {192, 12, 72},
    {176, 12, 84}, {160, 11, 96}, {144, 9, 107}, {128, 7, 116}, {112, 1, 123}};

// 查找X1在表2中的位置信息
bool FindX1Info(uint64_t x1, uint64_t &startIndex, uint64_t &count)
{
    for (const auto &entry : X1_LOOKUP_TABLE) {
        if (std::get<0>(entry) == x1) {
            startIndex = std::get<NUM_TWO>(entry);
            count = std::get<1>(entry);
            return true;
        }
    }
    return false;
}
} // namespace

namespace optiling {
namespace mc2_matmul_v3_advanced {
using namespace strategy;

MC2_MM_REGISTER_TILING_TEMPLATE(Mc2MatMulV3, Mc2MatMulV3AswTiling, DAV_3510, BASE);
MC2_MM_REGISTER_TILING_TEMPLATE(Mc2MatMulV3, Mc2MatMulV3AswTiling, DAV_RESV, BASE); // supportMmadS8S4平台

void Mc2MatMulV3AswTiling::CalcTailBasicBlock()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    uint64_t mnCnt = mCnt * nCnt;
    uint64_t tailCnt = mnCnt <= compileInfo_.aicNum ? 0UL : mnCnt % compileInfo_.aicNum;
    runInfo_.tailInfo.mCnt = 1UL;
    runInfo_.tailInfo.nCnt = 1UL;
    if (tailCnt != 0UL) {
        while ((runInfo_.tailInfo.mCnt + 1UL) * runInfo_.tailInfo.nCnt * tailCnt <= compileInfo_.aicNum) {
            runInfo_.tailInfo.mCnt += 1UL;
            if (runInfo_.tailInfo.mCnt * (runInfo_.tailInfo.nCnt + 1UL) * tailCnt <= compileInfo_.aicNum) {
                runInfo_.tailInfo.nCnt += 1UL;
            }
        }
    }
}

void Mc2MatMulV3AswTiling::GetOuterAxisTailCnt(bool nLoadBalance, uint64_t &baseTailSplitCnt, uint64_t &tailMain)
{
    uint64_t aicNum = compileInfo_.aicNum;
    uint64_t x = args_.mValue;
    uint64_t y = args_.nValue;
    uint64_t baseX = runInfo_.baseM;
    uint64_t baseY = runInfo_.baseN;
    if (nLoadBalance) {
        x = args_.nValue;
        y = args_.mValue;
        baseX = runInfo_.baseN;
        baseY = runInfo_.baseM;
    }

    uint64_t xCnt = MathUtil::CeilDivision(x, baseX);
    uint64_t yCnt = MathUtil::CeilDivision(y, baseY);
    uint64_t xTail = x % baseX;

    uint64_t aswWindowLen = GetAswWindowLen();
    uint64_t totalWindows = MathUtil::CeilDivision(xCnt * yCnt, aicNum);
    uint64_t mainWindows = MathUtil::CeilDivision((xCnt - 1UL) * yCnt + yCnt % aicNum, aicNum);
    // 未做负载均衡的轴是核数的倍数且做负载均衡的轴是窗口的因子或轴的倍数，说明部分核只做主块
    if (yCnt % aicNum == 0UL && (xCnt % aswWindowLen == 0UL || aswWindowLen % xCnt == 0UL)) {
        mainWindows = totalWindows;
    }
    uint64_t tailWindows = totalWindows - mainWindows;
    uint64_t perfRes = mainWindows * baseX + tailWindows * xTail;

    uint64_t baseTailCntMax = 1UL;
    baseTailCntMax = std::min((baseX - xTail) / BASIC_BLOCK_SIZE_16, xCnt);

    for (uint64_t mergeLen = 1UL; mergeLen < baseTailCntMax; ++mergeLen) {
        uint64_t newTailMain =
            MathUtil::Align(MathUtil::CeilDivision((mergeLen * baseX + xTail), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
        uint64_t newTailLast = mergeLen * (baseX - newTailMain) + xTail;
        uint64_t newMainRound = 0UL;
        uint64_t newTailRound = 0UL;
        if (mergeLen < xCnt - 1UL) {
            // 按照最差的场景计算合并后的主轮，所以性能不会最优
            newMainRound =
                MathUtil::CeilDivision((xCnt - 1UL - mergeLen) * yCnt + (mergeLen + 1UL) * yCnt % aicNum, aicNum);
        }
        if (mergeLen > 0UL) {
            newTailRound =
                std::min(MathUtil::CeilDivision(mergeLen * yCnt + yCnt % aicNum, aicNum), totalWindows - newMainRound);
        }
        uint64_t curPerf = newMainRound * baseX + newTailRound * newTailMain +
                           (totalWindows - newMainRound - newTailRound) * newTailLast;
        // m轴尽量多分基本块，n轴少分基本块
        if (curPerf < perfRes || (!nLoadBalance && curPerf == perfRes)) {
            perfRes = curPerf;
            tailMain = newTailMain;
            baseTailSplitCnt = mergeLen + 1UL;
        }
    }
}

void Mc2MatMulV3AswTiling::OptimizeEdgeBasicBlock()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    if (mCore * nCore < compileInfo_.aicNum || mCore == 1UL || nCore == 1UL) {
        return;
    }
    uint64_t mBaseTail = args_.mValue % runInfo_.baseM;
    uint64_t nBaseTail = args_.nValue % runInfo_.baseN;

    // 判断均衡后是否会进入fixp模板
    bool balancePostFixp = args_.kValue <= BASIC_BLOCK_SIZE_256 && args_.nValue % BLOCK_BYTE_SIZE == 0UL;
    if (mBaseTail > 0UL && !args_.isATrans && (nBaseTail == 0UL || mBaseTail <= nBaseTail || balancePostFixp)) {
        GetOuterAxisTailCnt(false, runInfo_.mBaseTailSplitCnt, runInfo_.tailInfo.mTailMain);
    } else if (nBaseTail > 0UL && args_.isBTrans && !balancePostFixp) {
        GetOuterAxisTailCnt(true, runInfo_.nBaseTailSplitCnt, runInfo_.tailInfo.nTailMain);
    }
}

// 计算singleX=√(m * n)/coreNum
void Mc2MatMulV3AswTiling::CalcSingleX(uint64_t &higherSingleX, uint64_t &lowerSingleX)
{
    double data = static_cast<double>(args_.mValue * args_.nValue) / compileInfo_.aicNum;

    while (data > static_cast<double>(BASIC_BLOCK_SIZE_256 * BASIC_BLOCK_SIZE_256)) {
        data /= NUM_TWO;
    }

    double bestValue = std::sqrt(data);
    higherSingleX = ops::CeilAlign(static_cast<uint64_t>(std::ceil(bestValue)), BASIC_BLOCK_SIZE_16);
    lowerSingleX = ops::FloorAlign(static_cast<uint64_t>(std::floor(bestValue)), BASIC_BLOCK_SIZE_16);
}

void Mc2MatMulV3AswTiling::CalcBasicBlock()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    if (mCore == 0UL || nCore == 0UL) {
        return;
    }
    if (mCore <= nCore) {
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
        nCore = runInfo_.usedCoreNum / mCore;
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
    } else {
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
        nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
        mCore = runInfo_.usedCoreNum / nCore;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
    }

    while (runInfo_.baseN >= runInfo_.baseM * NUM_TWO && nCore < runInfo_.usedCoreNum / NUM_TWO) {
        nCore = nCore * NUM_TWO;
        mCore = runInfo_.usedCoreNum / nCore;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
        mCore = MathUtil::CeilDivision(args_.mValue, static_cast<uint64_t>(runInfo_.baseM));
        nCore = MathUtil::CeilDivision(args_.nValue, static_cast<uint64_t>(runInfo_.baseN));
    }

    while (runInfo_.baseM >= runInfo_.baseN * NUM_TWO && mCore < runInfo_.usedCoreNum / NUM_TWO) {
        mCore = mCore * NUM_TWO;
        nCore = runInfo_.usedCoreNum / mCore;
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, mCore), BASIC_BLOCK_SIZE_16);
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, nCore), BASIC_BLOCK_SIZE_16);
        mCore = MathUtil::CeilDivision(args_.mValue, static_cast<uint64_t>(runInfo_.baseM));
        nCore = MathUtil::CeilDivision(args_.nValue, static_cast<uint64_t>(runInfo_.baseN));
    }
}

void Mc2MatMulV3AswTiling::FormulateBasicBlock()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    if (mCore * nCore >= compileInfo_.aicNum) {
        runInfo_.baseM = std::min(ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16), runInfo_.baseM);
        runInfo_.baseN = std::min(ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16), runInfo_.baseN);
        return;
    }
    CalcBasicBlock();
    mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    runInfo_.usedCoreNum = mCore * nCore;
    uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(args_.kValue), BASIC_BLOCK_SIZE_16);
    uint64_t kValueMax = ops::FloorAlign(
        L0A_SIZE_2 / DB_SIZE / args_.aDtypeSize / std::max(runInfo_.baseM, runInfo_.baseN), BASIC_BLOCK_SIZE_16);
    runInfo_.baseK = std::min(kValueAlign, kValueMax);
}

void Mc2MatMulV3AswTiling::FormulateLoadBalanceBlock()
{
    runInfo_.baseM = std::min(ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16), runInfo_.baseM);
    runInfo_.baseN = std::min(ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16), runInfo_.baseN);

    // 计算默认负载均衡率
    runInfo_.defaultBalance =
        CalcMultiCoreBalance(args_.mValue, args_.nValue, compileInfo_.aicNum, runInfo_.baseM, runInfo_.baseN);
    // 重复搬运量
    runInfo_.redundantData = CalcRedundantDataMovement(runInfo_.baseM, runInfo_.baseN, args_.mValue, args_.nValue);

    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);

    double singleBlockNum = static_cast<double>(mCore * nCore / compileInfo_.aicNum);

    // 判断是否需要重选基本块
    bool needReselect = singleBlockNum >= 1.0 && singleBlockNum <= MAX_SINGLE_CORE_ROUND &&
                        runInfo_.defaultBalance < LOAD_BALANCE_RATE_LIMIT;

    // 如果不需要重选，直接使用默认值
    if (needReselect) {
        uint64_t higherSingleX;
        uint64_t lowerSingleX;
        CalcSingleX(higherSingleX, lowerSingleX);

        uint64_t minMN = ops::CeilAlign(std::min(args_.mValue, args_.nValue), BASIC_BLOCK_SIZE_16);
        uint64_t maxMN = ops::CeilAlign(std::max(args_.mValue, args_.nValue), BASIC_BLOCK_SIZE_16);
        bool isMLarger = (args_.mValue > args_.nValue);
        // 根据不同场景选择基本块
        if (lowerSingleX >= minMN) {
            HandleLargeSingleSide(minMN, maxMN, isMLarger);
        } else {
            HandleLargeBothSides(higherSingleX, lowerSingleX, minMN, isMLarger);
        }
    }

    // 处理mCore * nCore / coreNum < 1的情况
    if (singleBlockNum < 1.0) {
        CalcBasicBlock();
    }
    runInfo_.baseM = ops::CeilAlign(runInfo_.baseM, BASIC_BLOCK_SIZE_16);
    runInfo_.baseN = ops::CeilAlign(runInfo_.baseN, BASIC_BLOCK_SIZE_16);
    // l0c的dtype是fp32
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * sizeof(float) * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;

    mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    runInfo_.usedCoreNum = std::min(mCore * nCore, compileInfo_.aicNum);
    // 如果k是内轴，则可以对齐C0_size,否则以16个元素对齐
    uint64_t baseKAlignValue = !args_.isATrans && args_.isBTrans && args_.aType == ge::DT_FLOAT ?
                                   BLOCK_BYTE_SIZE / args_.aDtypeSize :
                                   BASIC_BLOCK_SIZE_16;
    uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(args_.kValue), baseKAlignValue);
    uint64_t kValueMax = ops::FloorAlign(
        L0A_SIZE_2 / DB_SIZE / args_.aDtypeSize / std::max(runInfo_.baseM, runInfo_.baseN), baseKAlignValue);
    runInfo_.baseK = std::min(kValueAlign, kValueMax);
}

// 单边大场景处理
uint64_t Mc2MatMulV3AswTiling::UpdateBaseBlock(uint64_t baseBlock, bool isMLarger)
{
    if (!isMLarger) {
        if (!args_.isBTrans || args_.kValue <= BASIC_BLOCK_SIZE_256) {
            // n较大, n是内轴或k<=256 n对齐128B,外轴对齐64B
            return ops::CeilAlign(baseBlock, BASIC_BLOCK_SIZE_128 / args_.bDtypeSize);
        } else {
            return ops::CeilAlign(baseBlock, BASIC_BLOCK_SIZE_64 / args_.bDtypeSize);
        }
    } else {
        return ops::CeilAlign(baseBlock, BASIC_BLOCK_SIZE_16);
    }
}

void Mc2MatMulV3AswTiling::CalcLargeSingleSide(uint64_t maxMN, uint64_t &targetBase, bool isMLarger)
{
    // ceil(核数*0.9)
    uint64_t minCoreNum = (compileInfo_.aicNum + 1UL) * NUM_NINE / NUM_TEN;
    for (uint64_t tmpCoreNum = compileInfo_.aicNum; tmpCoreNum >= minCoreNum; tmpCoreNum--) {
        int loop = 1;
        while (loop <= MAX_LOOP_NUM) { // loop/(loop+1)>=0.95 loop=19
            uint64_t baseBlock = MathUtil::CeilDivision(maxMN, tmpCoreNum * loop);
            baseBlock = UpdateBaseBlock(baseBlock, isMLarger);
            if (baseBlock >= MIN_BASE_BLOCK && baseBlock <= MAX_BASE_BLOCK) {
                targetBase = baseBlock;
                return;
            }
            loop++;
        }
    }
    return;
}

void Mc2MatMulV3AswTiling::HandleLargeSingleSide(uint64_t minMN, uint64_t maxMN, bool isMLarger)
{
    if (isMLarger) {
        runInfo_.baseN = minMN;
        runInfo_.baseM = compileInfo_.l0CSize / runInfo_.dbL0C / runInfo_.baseN;
        runInfo_.baseM = ops::FloorAlign(runInfo_.baseM, BASIC_BLOCK_SIZE_16);
        CalcLargeSingleSide(maxMN, runInfo_.baseM, isMLarger);
    } else {
        runInfo_.baseM = minMN;
        runInfo_.baseN = compileInfo_.l0CSize / runInfo_.dbL0C / runInfo_.baseM;
        runInfo_.baseN = ops::FloorAlign(runInfo_.baseN, BLOCK_BYTE_SIZE);
        CalcLargeSingleSide(maxMN, runInfo_.baseN, isMLarger);
    }
}

// 两边都比较大场景处理
bool Mc2MatMulV3AswTiling::UpdateBothBaseBlock(double balance, CalcParams &params, uint64_t currentBaseM,
                                               uint64_t currentBaseN, uint64_t baseK)
{
    // 调整base块后如果负载均衡率大于重复搬运率比值-1+0.03
    if (balance > LOAD_BALANCING_THRESHOLD) {
        // 如果调整base块后负载均衡率大于0.98则直接设置base块
        runInfo_.baseM = currentBaseM;
        runInfo_.baseN = currentBaseN;
        runInfo_.baseK = baseK;
        return true;
    } else if ((balance > params.bestBalance) && (balance > MIN_EQUALIZATION_COEFFICIENT * runInfo_.defaultBalance)) {
        params.bestBalance = balance;
        params.baseM = currentBaseM;
        params.baseN = currentBaseN;
        params.baseK = baseK;
    }
    return false;
}

bool Mc2MatMulV3AswTiling::CalcBestBalance(CalcParams &params, bool isMLarger)
{
    uint64_t startIndex;
    uint64_t count;
    uint64_t baseX = params.baseStart;
    bool condition = params.isNegativeSign ? baseX >= params.baseEnd : baseX <= params.baseEnd;
    while (condition) {
        bool isFindStartIndex = FindX1Info(baseX, startIndex, count);
        if (!isFindStartIndex) {
            baseX = params.isNegativeSign ? baseX - BASIC_BLOCK_SIZE_16 : baseX + BASIC_BLOCK_SIZE_16;
            condition = params.isNegativeSign ? baseX >= params.baseEnd : baseX <= params.baseEnd;
            continue;
        }

        for (uint64_t i = 0; i < count; i++) {
            if (startIndex + i >= BLOCK_TABLE.size()) {
                break;
            }
            auto [x1, x2, x3, x4, x5] = BLOCK_TABLE[startIndex + i];
            (void)x5;
            // 如果k是外轴，则跳过无法对齐16的baseK
            bool skipK = (args_.isATrans || !args_.isBTrans) && args_.aType == ge::DT_FLOAT &&
                         x3 / NUM_TWO % BASIC_BLOCK_SIZE_16 != 0;
            if (x3 > args_.kValue || skipK) {
                continue;
            }
            uint64_t currentBaseM = isMLarger ? x2 : x1;
            uint64_t currentBaseN = isMLarger ? x1 : x2;
            // n是内轴或k<=256 n对齐128B 否则对齐64B；m是内轴则对齐128B否则对齐64B
            bool nNotAligned = !args_.isBTrans || args_.kValue <= BASIC_BLOCK_SIZE_256 ?
                                   currentBaseN * args_.bDtypeSize % BASIC_BLOCK_SIZE_128 != 0UL :
                                   currentBaseN * args_.bDtypeSize % BASIC_BLOCK_SIZE_64 != 0UL;
            bool mNotAligned = args_.isATrans ? currentBaseM * args_.aDtypeSize % BASIC_BLOCK_SIZE_128 != 0UL :
                                                currentBaseM * args_.aDtypeSize % BASIC_BLOCK_SIZE_64 != 0UL;
            if (mNotAligned || nNotAligned) {
                continue;
            }
            double balance =
                CalcMultiCoreBalance(args_.mValue, args_.nValue, compileInfo_.aicNum, currentBaseM, currentBaseN) / x4;
            double removeRatio =
                static_cast<double>(CalcRedundantDataMovement(currentBaseM, currentBaseN, args_.mValue, args_.nValue) /
                                    runInfo_.redundantData);
            bool isUpdateBaseBlock = false;
            if (balance - runInfo_.defaultBalance > removeRatio - BALANCE_REDUNDANT_THRESHOLD) {
                isUpdateBaseBlock = UpdateBothBaseBlock(balance, params, currentBaseM, currentBaseN, x3);
            }
            if (isUpdateBaseBlock) {
                return true;
            }
        }
        baseX = params.isNegativeSign ? baseX - BASIC_BLOCK_SIZE_16 : baseX + BASIC_BLOCK_SIZE_16;
        condition = params.isNegativeSign ? baseX >= params.baseEnd : baseX <= params.baseEnd;
    }
    return false;
}

void Mc2MatMulV3AswTiling::HandleLargeBothSides(uint64_t higherSingleX, uint64_t lowerSingleX, uint64_t minMN,
                                                bool isMLarger)
{
    CalcParams params1 = {lowerSingleX,
                          MIN_BASE_BLOCK,
                          true,
                          runInfo_.defaultBalance,
                          BASIC_BLOCK_K_256_BYTE,
                          BASIC_BLOCK_K_256_BYTE,
                          BASIC_BLOCK_K_128_BYTE / args_.aDtypeSize};
    if (CalcBestBalance(params1, isMLarger)) {
        return;
    }

    CalcParams params2 = {higherSingleX,
                          std::min(MAX_BASE_BLOCK, minMN),
                          false,
                          runInfo_.defaultBalance,
                          BASIC_BLOCK_K_256_BYTE,
                          BASIC_BLOCK_K_256_BYTE,
                          BASIC_BLOCK_K_128_BYTE / args_.aDtypeSize};
    if (CalcBestBalance(params2, isMLarger)) {
        return;
    }

    runInfo_.baseM = params1.baseM;
    runInfo_.baseN = params1.baseN;
    runInfo_.baseK = params1.baseK;
    if (params1.bestBalance < params2.bestBalance) {
        runInfo_.baseM = params2.baseM;
        runInfo_.baseN = params2.baseN;
        runInfo_.baseK = params2.baseK;
    }
}

ge::graphStatus Mc2MatMulV3AswTiling::DoNormOpTiling()
{
    Mc2MatMulV3TilingHelper::ResetBase(context_, compileInfo_, args_, runInfo_);
    FormulateBasicBlock();
    OptimizeEdgeBasicBlock();
    CalcTailBasicBlock();
    Mc2MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    if (Mc2MatMulV3TilingHelper::CheckIfDoubleAswt(context_, args_, 1UL)) {
        aswtModel_ = Mc2MatMulV3Model::DOUBLE_ASWT;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2MatMulV3AswTiling::DoOpTiling()
{
    Mc2MatMulV3TilingHelper::ResetBase(context_, compileInfo_, args_, runInfo_);
    FormulateLoadBalanceBlock();
    if (runInfo_.baseM == BASIC_BLOCK_SIZE_256 && runInfo_.baseN == BASIC_BLOCK_SIZE_256) {
        OptimizeEdgeBasicBlock();
    }
    CalcTailBasicBlock();
    Mc2MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    if (Mc2MatMulV3TilingHelper::CheckIfDoubleAswt(context_, args_, 1UL)) {
        aswtModel_ = Mc2MatMulV3Model::DOUBLE_ASWT;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2MatMulV3AswTiling::GetTilingKey() const
{
    return Mc2MatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(aswtModel_)
        .SetL0C2Out(Mc2MatMulV3TilingHelper::GetL0C2Out(context_, compileInfo_, args_, runInfo_))
        .GetTilingKey();
}
} // namespace mc2_matmul_v3_advanced
} // namespace optiling
