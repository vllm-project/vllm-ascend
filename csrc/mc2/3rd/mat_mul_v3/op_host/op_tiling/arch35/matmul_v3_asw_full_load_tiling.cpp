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
 * \file matmul_v3_asw_full_load_tiling.cc
 * \brief
 */
#include "matmul_v3_asw_full_load_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "./matmul_tiling_registry.h"
#include "common/op_host/math_util.h"

using Ops::Transformer::MathUtil;
namespace {
using namespace optiling;
using namespace optiling::mc2_matmul_v3_advanced;

// ------------------------------ ABL1FullLoadExtraCond -------------------------------------------//
bool ABL1FullLoadExtraCondDefault(uint64_t /* al1SingleCoreSize */, uint64_t /* bl1SingleCoreSize */)
{
    return true;
}

bool ABL1FullLoadExtraCond91095(uint64_t al1SingleCoreSize, uint64_t bl1SingleCoreSize)
{
    // 单边矩阵小于64K，MMAD启动较快，AB全载更有优势
    constexpr uint64_t AB_L1_SINGLE_LOAD_THRE = 64 * 1024UL;
    // 泛化经验值
    constexpr uint64_t AB_L1_BOTH_LOAD_THRE = 272 * 1024UL;
    return (al1SingleCoreSize <= AB_L1_SINGLE_LOAD_THRE || bl1SingleCoreSize <= AB_L1_SINGLE_LOAD_THRE) &&
           (al1SingleCoreSize + bl1SingleCoreSize) <= AB_L1_BOTH_LOAD_THRE;
}

using ABL1FullLoadExtraCondFunc = bool (*)(uint64_t, uint64_t);

const static std::map<NpuArch, ABL1FullLoadExtraCondFunc> ABL1FullLoadExtraCondFuncMap = {
    {NpuArch::DAV_3510, ABL1FullLoadExtraCond91095},
};

using CheckBL1FullLoadFunc = bool (Mc2MatMulV3AswFullLoadTiling::*)(bool &, uint64_t, uint64_t);
const static std::map<NpuArch, CheckBL1FullLoadFunc> CheckBL1FullLoadMap = {
    {NpuArch::DAV_3510, &Mc2MatMulV3AswFullLoadTiling::CheckBL1FullLoad91095},
};

// ------------------------------ GetStepSmallK -------------------------------------------//
uint64_t GetStepSmallKDefault(const Mc2MatMulV3Args & /* args */, const Mc2MatMulV3RunInfo &runInfo, bool isBL1FullLoad)
{
    return isBL1FullLoad ? runInfo.stepKa : runInfo.stepKb;
}

uint64_t GetStepSmallK91095(const Mc2MatMulV3Args &args, const Mc2MatMulV3RunInfo &runInfo, bool isBL1FullLoad)
{
    uint64_t stepBigK = runInfo.stepKa;
    uint64_t stepSmallK = runInfo.stepKb;
    uint64_t dtypeSize = args.aDtypeSize;
    ge::DataType inputType = args.aType;
    ge::Format inputFormat = args.bFormat;
    bool isTrans = args.isBTrans;
    if (isBL1FullLoad) {
        stepBigK = runInfo.stepKb;
        stepSmallK = runInfo.stepKa;
        dtypeSize = args.bDtypeSize;
        inputType = args.bType;
        inputFormat = args.aFormat;
        isTrans = args.isATrans;
    }

    static const double SMALL_TAIL = 0.25;
    bool isSmallTail = static_cast<double>(stepBigK % stepSmallK) / stepSmallK <= SMALL_TAIL;
    isSmallTail = (isSmallTail && !isTrans) || runInfo.baseK * dtypeSize >= BASIC_BLOCK_SIZE_256;
    // A/B全载场景，要求单核K除以basek大于8，使stepKb/stepKa增大，搬运量变为64K，减少mte2耗时。
    if ((inputType == ge::DT_FLOAT && !args.isHf32) ||
        ((runInfo.singleCoreK / runInfo.baseK) < (BASIC_BLOCK_SIZE_16 / NUM_TWO) && !isBL1FullLoad)) {
        stepSmallK = 1UL;
    } else if (isSmallTail) {
        stepSmallK = 2UL;
    }
    return stepSmallK;
}

using GetStepSmallKFunc = uint64_t (*)(const Mc2MatMulV3Args &, const Mc2MatMulV3RunInfo &, bool);

const static std::map<NpuArch, GetStepSmallKFunc> GetStepSmallKFuncMap = {
    {NpuArch::DAV_3510, GetStepSmallK91095},
};

void ResetLoadBalance(Mc2MatMulV3RunInfo &runInfo)
{
    // 全载模板需重置负载均衡计算
    runInfo.mBaseTailSplitCnt = 1UL;
    runInfo.nBaseTailSplitCnt = 1UL;
    runInfo.tailInfo.mTailMain = 0UL;
    runInfo.tailInfo.nTailMain = 0UL;
}
} // namespace

namespace optiling {
namespace mc2_matmul_v3_advanced {
using namespace strategy;

MC2_MM_REGISTER_TILING_TEMPLATE(Mc2MatMulV3, Mc2MatMulV3AswFullLoadTiling, DAV_3510, FULL_LOAD_BASE);

void Mc2MatMulV3AswFullLoadTiling::FullLoadPre()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.singleCoreN);
    isSingleRound_ = mCnt * nCnt <= compileInfo_.aicNum;
    biasSize_ = args_.hasBias ? runInfo_.baseN * runInfo_.stepN * GetSizeByDataType(args_.biasType) : 0;
    return;
}

bool Mc2MatMulV3AswFullLoadTiling::ABL1FullLoadExtraCond(uint64_t al1SingleCoreSize, uint64_t bl1SingleCoreSize) const
{
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    auto iter = (ABL1FullLoadExtraCondFuncMap.find(npuArch) == ABL1FullLoadExtraCondFuncMap.end()) ?
                    ABL1FullLoadExtraCondDefault :
                    ABL1FullLoadExtraCondFuncMap.at(npuArch);
    return iter(al1SingleCoreSize, bl1SingleCoreSize);
}

bool Mc2MatMulV3AswFullLoadTiling::CheckABL1FullLoad() const
{
    uint64_t aBlockSize = BLOCK_BYTE_SIZE / args_.aDtypeSize;
    uint64_t bBlockSize = BLOCK_BYTE_SIZE / args_.bDtypeSize;
    uint64_t singleCoreMAlignedValue = args_.isATrans ? ops::CeilAlign(runInfo_.singleCoreM, aBlockSize) :
                                                        ops::CeilAlign(runInfo_.singleCoreM, BASIC_BLOCK_SIZE_16);
    uint64_t singleCoreKaAlignedValue = args_.isATrans ? ops::CeilAlign(runInfo_.singleCoreK, BASIC_BLOCK_SIZE_16) :
                                                         ops::CeilAlign(runInfo_.singleCoreK, aBlockSize);
    uint64_t singleCoreNAlignedValue = args_.isBTrans ? ops::CeilAlign(runInfo_.singleCoreN, BASIC_BLOCK_SIZE_16) :
                                                        ops::CeilAlign(runInfo_.singleCoreN, bBlockSize);
    uint64_t singleCoreKbAlignedValue = args_.isBTrans ? ops::CeilAlign(runInfo_.singleCoreK, bBlockSize) :
                                                         ops::CeilAlign(runInfo_.singleCoreK, BASIC_BLOCK_SIZE_16);
    uint64_t al1SingleCoreSize = singleCoreMAlignedValue * singleCoreKaAlignedValue * args_.aDtypeSize;
    uint64_t bl1SingleCoreSize = singleCoreKbAlignedValue * singleCoreNAlignedValue * args_.bDtypeSize;

    bool generalCond = isSingleRound_ && (al1SingleCoreSize + bl1SingleCoreSize + biasSize_) <= compileInfo_.l1Size;
    bool extraCond = ABL1FullLoadExtraCond(al1SingleCoreSize, bl1SingleCoreSize);
    return generalCond && extraCond;
}

void Mc2MatMulV3AswFullLoadTiling::DoABL1FullLoad()
{
    OP_LOGI(args_.opName, "Mc2MatMulV3 tiling enable state is DoABL1FullLoad.");
    ResetLoadBalance(runInfo_);
    runInfo_.singleCoreM = std::min(runInfo_.singleCoreM, args_.mValue);
    runInfo_.singleCoreN = std::min(runInfo_.singleCoreN, args_.nValue);
    return;
}

uint64_t Mc2MatMulV3AswFullLoadTiling::GetStepSmallK(bool isBL1FullLoad) const
{
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    auto iter = (GetStepSmallKFuncMap.find(npuArch) == GetStepSmallKFuncMap.end()) ? GetStepSmallKDefault :
                                                                                     GetStepSmallKFuncMap.at(npuArch);
    return iter(args_, runInfo_, isBL1FullLoad);
}

bool Mc2MatMulV3AswFullLoadTiling::CheckAL1FullLoad(bool &isKFullLoad) const
{
    if (args_.nValue < CACHELINE) {
        return false;
    }
    uint64_t mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BLOCK_BYTE_SIZE / args_.aDtypeSize);
    if (args_.isATrans) {
        // m为内轴时强制16对齐
        mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
        kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    }
    uint64_t al1Size = mAlignedValue * kAlignedValue * args_.aDtypeSize;
    if (al1Size > (compileInfo_.l1Size - biasSize_) / NUM_TWO) {
        return false;
    }
    bool isKaFullLoad = isSingleRound_ && args_.mValue < args_.nValue && runInfo_.baseM < args_.mValue;
    if (!isKaFullLoad && args_.mValue > BASIC_BLOCK_SIZE_256) {
        return false;
    }
    isKFullLoad = isKaFullLoad;
    return true;
}

void Mc2MatMulV3AswFullLoadTiling::CalcTailBasicBlockBL1Full()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t tailCnt = mCnt <= compileInfo_.aicNum ? 0UL : mCnt % compileInfo_.aicNum;
    runInfo_.tailInfo.mCnt = 1UL;
    runInfo_.tailInfo.nCnt = 1UL;
    if (tailCnt == 0UL) {
        return;
    }
    while ((runInfo_.tailInfo.mCnt + 1UL) * tailCnt <= compileInfo_.aicNum) {
        runInfo_.tailInfo.mCnt += 1UL;
    }
}

void Mc2MatMulV3AswFullLoadTiling::DoAL1FullLoad(bool isKFullLoad, uint64_t bBatchDimAll, uint64_t biasBatchDimAll)
{
    ResetLoadBalance(runInfo_);
    if (isKFullLoad) {
        OP_LOGD(args_.opName, "AL1 is full loaded with m splited in multi cores");
        runInfo_.singleCoreM = std::min(runInfo_.singleCoreM, args_.mValue);
        return;
    }
    uint64_t mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t al0Size = mAlignedValue * runInfo_.baseK * args_.aDtypeSize * DB_SIZE;
    runInfo_.baseM = al0Size <= compileInfo_.l0ASize ? mAlignedValue : std::min(mAlignedValue, runInfo_.baseM);
    runInfo_.stepM = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    runInfo_.stepKa = MathUtil::CeilDivision(args_.kValue, runInfo_.baseK);
    runInfo_.stepKb = GetStepSmallK(false);
    // take full use of cores
    if (bBatchDimAll * MathUtil::CeilDivision(args_.nValue, runInfo_.baseN) < compileInfo_.aicNum) {
        runInfo_.baseN = ops::CeilAlign(MathUtil::CeilDivision(bBatchDimAll * args_.nValue, compileInfo_.aicNum),
                                        BASIC_BLOCK_SIZE_16);
    }
    runInfo_.depthB1 = DB_SIZE * runInfo_.stepKb;
    runInfo_.depthA1 = runInfo_.stepM * runInfo_.stepKa;
    uint64_t loadSize = static_cast<uint64_t>(runInfo_.baseK) *
                        (runInfo_.depthA1 * runInfo_.baseM + runInfo_.depthB1 * runInfo_.baseN) * args_.aDtypeSize;
    uint64_t baseBiasSize = args_.hasBias ? runInfo_.baseN * GetSizeByDataType(args_.biasType) * biasBatchDimAll : 0;
    loadSize += baseBiasSize;
    // Check L1 load size
    while (loadSize > compileInfo_.l1Size) {
        loadSize -= runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK * args_.aDtypeSize;
        loadSize -= baseBiasSize;
        runInfo_.baseN = ops::CeilAlign(runInfo_.baseN >> 1, BASIC_BLOCK_SIZE_16);
        loadSize += runInfo_.depthB1 * runInfo_.baseN * runInfo_.baseK * args_.aDtypeSize;
        loadSize += baseBiasSize;
    }
    runInfo_.singleCoreN = runInfo_.baseN;
    runInfo_.singleCoreM = args_.mValue;
    uint64_t dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
    runInfo_.tailInfo.nCnt *= runInfo_.tailInfo.mCnt;
    runInfo_.tailInfo.mCnt = 1UL;
    return;
}

bool Mc2MatMulV3AswFullLoadTiling::CheckBL1FullLoadDefault(bool &isKFullLoad, uint64_t kAlignedValue,
                                                           uint64_t nAlignedValue) const
{
    // NZ以及其他平台走高级api, 若不随路就返回
    if (l0C2Out_ != Mc2MatMulV3L0C2Out::ON_THE_FLY) {
        return false;
    }
    uint64_t bl1Size = kAlignedValue * nAlignedValue * args_.bDtypeSize;
    bool isKbFullLoad = isSingleRound_ && args_.nValue < args_.mValue && runInfo_.baseN < args_.nValue;
    if (bl1Size > (compileInfo_.l1Size - biasSize_) / NUM_TWO ||
        (!isKbFullLoad && args_.nValue > BASIC_BLOCK_SIZE_256)) {
        return false;
    }
    isKFullLoad = isKbFullLoad;
    return true;
}

bool Mc2MatMulV3AswFullLoadTiling::CheckBL1FullLoad91095(bool &isKFullLoad, uint64_t kAlignedValue,
                                                         uint64_t nAlignedValue)
{
    if (l0C2Out_ != Mc2MatMulV3L0C2Out::ON_THE_FLY) {
        apiLevel_ = Mc2MatMulV3ApiLevel::BASIC_LEVEL;
        aswtModel_ = Mc2MatMulV3Model::BASIC;
    }
    uint64_t bl1Size = kAlignedValue * nAlignedValue * args_.bDtypeSize;
    // 单核上只有一轮，走basic api模板， 头开销较小，无需走全载模板
    if (isSingleRound_) {
        return false;
    }
    // L2命中率高， 不走全载模板
    uint64_t maxStepN =
        Mc2MatMulV3BaseTiling::GetAswWindowLen() > 1UL ? Mc2MatMulV3BaseTiling::GetAswWindowLen() - 1UL : 1;
    if (args_.nValue >= maxStepN * runInfo_.baseN) {
        return false;
    }
    uint64_t biasSize = args_.hasBias ? nAlignedValue * GetSizeByDataType(args_.biasType) : 0;
    // 3/4L1 = 384M
    if (bl1Size + biasSize > compileInfo_.l1Size * 3UL / 4UL) {
        return false;
    }
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.singleCoreN);
    uint64_t bL1FullMTE2Size = args_.nValue * compileInfo_.aicNum + args_.mValue * nCnt;
    uint64_t baseMTE2Size = args_.mValue * nCnt + args_.nValue * mCnt;
    // 1.2表示全载后至少减少了20%的总数据搬运量
    if (args_.nValue > BASIC_BLOCK_SIZE_256 && static_cast<float>(baseMTE2Size) < 1.2f * bL1FullMTE2Size) {
        return false;
    }
    isKFullLoad = false;
    apiLevel_ = Mc2MatMulV3ApiLevel::BASIC_LEVEL;
    aswtModel_ = Mc2MatMulV3Model::BASIC;
    return true;
}

bool Mc2MatMulV3AswFullLoadTiling::CheckBL1FullLoad(bool &isKFullLoad)
{
    if (args_.mValue < CACHELINE) {
        return false;
    }
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t nAlignedValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    if (args_.isBTrans) {
        kAlignedValue = ops::CeilAlign(args_.kValue, BLOCK_BYTE_SIZE / args_.bDtypeSize);
        nAlignedValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    }
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    auto iter =
        (CheckBL1FullLoadMap.find(npuArch) == CheckBL1FullLoadMap.end() || args_.bFormat == ge::FORMAT_FRACTAL_NZ) ?
            CheckBL1FullLoadDefault(isKFullLoad, kAlignedValue, nAlignedValue) :
            (this->*CheckBL1FullLoadMap.at(npuArch))(isKFullLoad, kAlignedValue, nAlignedValue);
    return iter;
}

void Mc2MatMulV3AswFullLoadTiling::AdjustTiling91095Basic(uint64_t biasBatchDimAll)
{
    if (args_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        AdjustTilingDefault(biasBatchDimAll);
        return;
    }
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t nAlignedValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t bL1Size = kAlignedValue * nAlignedValue * args_.bDtypeSize;
    uint64_t loadSize = static_cast<uint64_t>(runInfo_.baseK) * runInfo_.depthA1 * runInfo_.baseM * args_.aDtypeSize;
    uint64_t baseBiasSize = args_.hasBias ? nAlignedValue * GetSizeByDataType(args_.biasType) : 0;
    loadSize += baseBiasSize + bL1Size;
    // Check L1 load size
    while (loadSize > compileInfo_.l1Size) {
        // 第一轮有限调整stepK为2， 进一步的调整baseM
        runInfo_.baseM = (runInfo_.stepKa == std::min(runInfo_.stepKb, 2UL)) ? runInfo_.baseM >> 1 : runInfo_.baseM;
        runInfo_.stepKa = std::min(runInfo_.stepKb, 2UL); // 最小为2保证baseK * 2 * adtype = 256B
        runInfo_.depthA1 = DB_SIZE * runInfo_.stepKa;
        runInfo_.baseM = ops::CeilAlign(runInfo_.baseM, BASIC_BLOCK_SIZE_16);
        loadSize = runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * args_.aDtypeSize + baseBiasSize + bL1Size;
    }
    uint64_t dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    runInfo_.singleCoreN = args_.nValue;
    runInfo_.singleCoreM = runInfo_.baseM;
    // 内轴*dtype是256 * 2的整数倍表明tiling减半不影响MTE2搬运效率
    if (!args_.isATrans || (runInfo_.baseM * args_.aDtypeSize) % (BASIC_BLOCK_SIZE_256 * NUM_TWO) == 0) {
        runInfo_.baseM = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ?
                             runInfo_.baseM :
                             ops::CeilAlign(runInfo_.baseM >> 1, BASIC_BLOCK_SIZE_16);
    }
    uint64_t aL14Buffer = static_cast<uint64_t>(runInfo_.baseK) * runInfo_.stepKa * runInfo_.baseM * args_.aDtypeSize *
                          BASIC_L1_BUFFER_NUM;
    runInfo_.l1BufferNum = aL14Buffer + bL1Size + baseBiasSize > compileInfo_.l1Size ? DB_SIZE : BASIC_L1_BUFFER_NUM;
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
    runInfo_.mixInfo.ubDB = runInfo_.baseM * runInfo_.baseN * dtypeSize <= compileInfo_.ubSize ? DB_SIZE : 1UL;
}

void Mc2MatMulV3AswFullLoadTiling::AdjustTilingCommon(uint64_t aBatchDimAll)
{
    // fine tune tiling basen
    uint64_t nAlignedValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t bl0Size = nAlignedValue * runInfo_.baseK * args_.bDtypeSize * DB_SIZE;
    runInfo_.baseN = bl0Size <= compileInfo_.l0BSize ? nAlignedValue : std::min(nAlignedValue, runInfo_.baseN);
    runInfo_.stepN = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    runInfo_.stepKb = MathUtil::CeilDivision(args_.kValue, runInfo_.baseK);
    runInfo_.stepKa = GetStepSmallK(true);
    // take full use of cores
    if (aBatchDimAll * MathUtil::CeilDivision(args_.mValue, runInfo_.baseM) < compileInfo_.aicNum) {
        runInfo_.baseM = ops::CeilAlign(MathUtil::CeilDivision(aBatchDimAll * args_.mValue, compileInfo_.aicNum),
                                        BASIC_BLOCK_SIZE_16);
    }
    runInfo_.depthA1 = DB_SIZE * runInfo_.stepKa;
    runInfo_.depthB1 = runInfo_.stepN * runInfo_.stepKb;
}

void Mc2MatMulV3AswFullLoadTiling::AdjustTilingDefault(uint64_t biasBatchDimAll)
{
    uint64_t bL1Size = runInfo_.baseK * runInfo_.depthB1 * runInfo_.baseN * args_.bDtypeSize;
    uint64_t baseBiasSize = args_.hasBias ? runInfo_.baseN * GetSizeByDataType(args_.biasType) * biasBatchDimAll : 0;
    uint64_t aL1Size = runInfo_.baseK * runInfo_.depthA1 * runInfo_.baseM * args_.aDtypeSize;
    uint64_t loadSize = aL1Size + bL1Size + baseBiasSize;
    // Check L1 load size
    while (loadSize > compileInfo_.l1Size) {
        // 第一轮有限调整stepK为2， 进一步的调整baseM
        runInfo_.baseM = (runInfo_.stepKa == std::min(runInfo_.stepKb, 2UL)) ? runInfo_.baseM >> 1 : runInfo_.baseM;
        runInfo_.stepKa = std::min(runInfo_.stepKb, 2UL); // 最小为2保证baseK * 2 * adtype = 256B
        runInfo_.depthA1 = DB_SIZE * runInfo_.stepKa;
        runInfo_.baseM = ops::CeilAlign(runInfo_.baseM, BASIC_BLOCK_SIZE_16);
        loadSize = runInfo_.depthA1 * runInfo_.baseM * runInfo_.baseK * args_.aDtypeSize + baseBiasSize + bL1Size;
    }

    runInfo_.singleCoreN = args_.nValue;
    runInfo_.singleCoreM = runInfo_.baseM;
    uint64_t dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
    runInfo_.tailInfo.mCnt *= runInfo_.tailInfo.nCnt;
    runInfo_.tailInfo.nCnt = 1UL;
}

void Mc2MatMulV3AswFullLoadTiling::DoBL1FullLoad(bool isKFullLoad, uint64_t aBatchDimAll, uint64_t biasBatchDimAll)
{
    // 负载均衡屏蔽全载模板
    ResetLoadBalance(runInfo_);
    OP_LOGI(args_.opName, "Mc2MatMulV3 tiling enable state is DoBL1FullLoad.");
    if (isKFullLoad) {
        OP_LOGD(args_.opName, "BL1 is full loaded with n splited in multi cores.");
        runInfo_.singleCoreN = std::min(runInfo_.singleCoreN, args_.nValue);
        fullLoad_ = Mc2MatMulV3FullLoad::B_FULL_LOAD;
        return;
    }
    if (apiLevel_ == Mc2MatMulV3ApiLevel::BASIC_LEVEL) {
        AdjustTilingCommon(aBatchDimAll);
        AdjustTiling91095Basic(biasBatchDimAll);
        CalcTailBasicBlockBL1Full();
    } else {
        AdjustTilingCommon(aBatchDimAll);
        AdjustTilingDefault(biasBatchDimAll);
    }
    fullLoad_ = Mc2MatMulV3FullLoad::B_FULL_LOAD;
    return;
}

ge::graphStatus Mc2MatMulV3AswFullLoadTiling::DoOpTiling()
{
    Mc2MatMulV3AswTiling::DoOpTiling();
    l0C2Out_ = Mc2MatMulV3TilingHelper::GetL0C2Out(context_, compileInfo_, args_, runInfo_);
    FullLoadPre();
    bool isKFullLoad = false;
    if (l0C2Out_ == Mc2MatMulV3L0C2Out::ON_THE_FLY && CheckABL1FullLoad()) {
        DoABL1FullLoad();
        fullLoad_ = Mc2MatMulV3FullLoad::AB_FULL_LOAD;
        aswtModel_ = Mc2MatMulV3Model::BASIC;
    } else if (l0C2Out_ == Mc2MatMulV3L0C2Out::ON_THE_FLY && CheckAL1FullLoad(isKFullLoad)) {
        DoAL1FullLoad(isKFullLoad);
        fullLoad_ = Mc2MatMulV3FullLoad::A_FULL_LOAD;
    } else if (CheckBL1FullLoad(isKFullLoad)) {
        DoBL1FullLoad(isKFullLoad);
    }
    // 将非全载模板但fixpipe优化以及BL1全载的场景均设置高级api
    if (l0C2Out_ == Mc2MatMulV3L0C2Out::ON_THE_FLY && fullLoad_ == Mc2MatMulV3FullLoad::NONE_FULL_LOAD) {
        return Mc2MatMulV3AswTiling::DoOpTiling();
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2MatMulV3AswFullLoadTiling::GetTilingKey() const
{
    return Mc2MatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetFullLoad(fullLoad_)
        .SetModel(aswtModel_)
        .SetL0C2Out(l0C2Out_)
        .SetApiLevel(apiLevel_)
        .GetTilingKey();
}

} // namespace mc2_matmul_v3_advanced
} // namespace optiling
