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
 * \file matmul_v3_tiling_helper.cc
 * \brief
 */

#include "matmul_v3_tiling_helper.h"
#include "common/op_host/math_util.h"
#include "platform/soc_spec.h"

using Ops::Transformer::MathUtil;
namespace {
using namespace optiling;
using namespace optiling::mc2_matmul_v3_advanced;

// ------------------------------ ResetBase -------------------------------------------//
void ResetBaseDefault(const Mc2MatmulV3CompileInfo &compileInfo, const Mc2MatMulV3Args &args,
                      Mc2MatMulV3RunInfo &runInfo)
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

void ResetBase91095(const Mc2MatmulV3CompileInfo &compileInfo, const Mc2MatMulV3Args &args, Mc2MatMulV3RunInfo &runInfo)
{
    ResetBaseDefault(compileInfo, args, runInfo);
    runInfo.baseM = BASIC_BLOCK_SIZE_256;
    runInfo.singleCoreM = runInfo.baseM;
}

using ResetBaseFunc = void (*)(const Mc2MatmulV3CompileInfo &, const Mc2MatMulV3Args &, Mc2MatMulV3RunInfo &);

const static std::map<NpuArch, ResetBaseFunc> ResetBaseFuncMap = {
    {NpuArch::DAV_3510, ResetBase91095},
};

// ------------------------------ CalL1Tiling -------------------------------------------//
void CalL1TilingDefault(const Mc2MatmulV3CompileInfo &compileInfo, const Mc2MatMulV3Args &args,
                        Mc2MatMulV3RunInfo &runInfo)
{
    uint64_t totalL1Size = compileInfo.l1Size;
    uint64_t reserveBTSize = args.hasBias ? BIAS_TABLE_NUM * DATA_SIZE_FP32 : 0UL;
    runInfo.depthA1 = totalL1Size / NUM_TWO / runInfo.baseM / runInfo.baseK / args.aDtypeSize; // 2: half of l1
    runInfo.depthB1 = totalL1Size / NUM_TWO / runInfo.baseN / runInfo.baseK / args.bDtypeSize; // 2: half of l1

    uint64_t depthASize = runInfo.depthA1 * runInfo.baseM * runInfo.baseK * args.aDtypeSize;
    uint64_t depthBSize = runInfo.depthB1 * runInfo.baseN * runInfo.baseK * args.bDtypeSize;
    if (depthASize + depthBSize > totalL1Size - reserveBTSize) {
        if (runInfo.baseM <= runInfo.baseN) {
            runInfo.depthA1 = std::max(runInfo.depthA1 / NUM_TWO, 1UL); // 2: adjust deptch for l1 buffer
        } else {
            runInfo.depthB1 = std::max(runInfo.depthB1 / NUM_TWO, 1UL); // 2: adjust deptch for l1 buffer
        }
    }
    runInfo.stepKa = std::max(runInfo.depthA1 / DB_SIZE, 1UL);
    runInfo.stepKb = std::max(runInfo.depthB1 / DB_SIZE, 1UL);
    if (runInfo.baseM == BASIC_BLOCK_SIZE_256 && runInfo.baseN == BASIC_BLOCK_SIZE_256 &&
        args.mValue % BASIC_BLOCK_SIZE_16 == 0 && args.nValue % BASIC_BLOCK_SIZE_16 == 0 &&
        args.kValue % BASIC_BLOCK_SIZE_16 == 0 && runInfo.singleCoreK <= BASIC_BLOCK_SIZE_256) {
        runInfo.stepKa = std::min(runInfo.stepKa, 2UL);
        runInfo.stepKb = std::min(runInfo.stepKb, 2UL);
    }
    if (runInfo.stepKa >= runInfo.stepKb) {
        runInfo.stepKa = runInfo.stepKa / runInfo.stepKb * runInfo.stepKb;
    } else {
        runInfo.stepKb = runInfo.stepKb / runInfo.stepKa * runInfo.stepKa;
    }
    runInfo.depthA1 = runInfo.stepKa * DB_SIZE; // depth % (stepKa * stepM) == 0
    runInfo.depthB1 = runInfo.stepKb * DB_SIZE; // depth % (stepKb * stepN) == 0
    runInfo.singleCoreM = runInfo.baseM;
    runInfo.singleCoreN = runInfo.baseN;
    return;
}

void CalL1Tiling310P(const Mc2MatmulV3CompileInfo &compileInfo, const Mc2MatMulV3Args &args,
                     Mc2MatMulV3RunInfo &runInfo)
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

using CalL1TilingFunc = void (*)(const Mc2MatmulV3CompileInfo &, const Mc2MatMulV3Args &, Mc2MatMulV3RunInfo &);

const static std::map<platform_ascendc::SocVersion, CalL1TilingFunc> CalL1TilingFuncMap = {
    {platform_ascendc::SocVersion::ASCEND310P, CalL1Tiling310P},
};

// ------------------------------ GetL0C2Out -------------------------------------------//
Mc2MatMulV3L0C2Out GetL0C2OutDefault(const Mc2MatmulV3CompileInfo & /* compileInfo */,
                                     const Mc2MatMulV3Args & /* args */, const Mc2MatMulV3RunInfo & /* runInfo */)
{
    return Mc2MatMulV3L0C2Out::ON_THE_FLY;
}

Mc2MatMulV3L0C2Out GetL0C2Out91095(const Mc2MatmulV3CompileInfo &compileInfo, const Mc2MatMulV3Args &args,
                                   const Mc2MatMulV3RunInfo &runInfo)
{
    bool isValidMKN = args.kValue <= BASIC_BLOCK_SIZE_256 && args.mValue >= BASIC_BLOCK_SIZE_256;
    uint64_t mCnt = MathUtil::CeilDivision(args.mValue, runInfo.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(args.nValue, runInfo.singleCoreN);
    // make sure the fixpipe stream is large enough to be bound
    bool isMultiRound = mCnt * nCnt >= NUM_TWO * compileInfo.aicNum;
    uint64_t cDtypeSize = ge::GetSizeByDataType(args.cType);
    // 128: SMALL_SHAPE_LOWER_THRES
    bool isUnalignedN = args.nValue * cDtypeSize % 128UL != 0 && args.nValue * cDtypeSize > BASIC_BLOCK_SIZE_256;
    bool fixpipeBound = isValidMKN && isMultiRound && isUnalignedN;
    if (!fixpipeBound) {
        return Mc2MatMulV3L0C2Out::ON_THE_FLY;
    }
    if (args.aType == ge::DT_FLOAT16 || args.aType == ge::DT_BF16) {
        return Mc2MatMulV3L0C2Out::ND_FIXPIPE_1_1;
    }
    return Mc2MatMulV3L0C2Out::ND_FIXPIPE_1_2;
}

using GetL0C2OutFunc = Mc2MatMulV3L0C2Out (*)(const Mc2MatmulV3CompileInfo &, const Mc2MatMulV3Args &,
                                              const Mc2MatMulV3RunInfo &);

const static std::map<NpuArch, GetL0C2OutFunc> GetL0C2OutFuncMap = {
    {NpuArch::DAV_3510, GetL0C2Out91095},
};

// ------------------------------ CheckIfDoubleAswt -------------------------------------------//
bool CheckIfDoubleAswtDefault(const Mc2MatMulV3Args & /* args */, const uint64_t /* batchC */)
{
    return false;
}

bool CheckIfDoubleAswt91095(const Mc2MatMulV3Args &args, const uint64_t batchC)
{
    constexpr uint64_t halfL2Size = 64UL * 1024UL * 1024UL; // 64mb
    constexpr uint64_t cubeBoundRatio = 512UL;
    if (batchC * args.mValue * args.nValue * args.aDtypeSize < halfL2Size) { // check matC exceed half L2
        return false;
    }
    if ((args.mValue * args.nValue / (args.mValue + args.nValue)) > cubeBoundRatio) { // check if cube bound or streamk
        return false;
    }
    if (args.kValue > (args.mValue >> 1) ||
        args.kValue > (args.nValue >> 1)) { // check if matA or matb occupies most of L2
        return false;
    }
    return true;
}

using CheckIfDoubleAswtFunc = bool (*)(const Mc2MatMulV3Args &, const uint64_t);

const static std::map<NpuArch, CheckIfDoubleAswtFunc> CheckIfDoubleAswtFuncMap = {
    {NpuArch::DAV_3510, CheckIfDoubleAswt91095},
};
} // namespace

namespace optiling {
namespace mc2_matmul_v3_advanced {
void Mc2MatMulV3TilingHelper::ResetBase(const gert::TilingContext *context, const Mc2MatmulV3CompileInfo &compileInfo,
                                        const Mc2MatMulV3Args &args, Mc2MatMulV3RunInfo &runInfo)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    NpuArch npuArchHelper = ascendcPlatform.GetCurNpuArch();
    auto iter = (ResetBaseFuncMap.find(npuArchHelper) == ResetBaseFuncMap.end()) ? ResetBaseDefault :
                                                                                   ResetBaseFuncMap.at(npuArchHelper);
    iter(compileInfo, args, runInfo);
}

void Mc2MatMulV3TilingHelper::CalL1Tiling(const Mc2MatmulV3CompileInfo &compileInfo, const Mc2MatMulV3Args &args,
                                          Mc2MatMulV3RunInfo &runInfo)
{
    auto iter = (CalL1TilingFuncMap.find(compileInfo.socVersion) == CalL1TilingFuncMap.end()) ?
                    CalL1TilingDefault :
                    CalL1TilingFuncMap.at(compileInfo.socVersion);
    iter(compileInfo, args, runInfo);
}

Mc2MatMulV3L0C2Out Mc2MatMulV3TilingHelper::GetL0C2Out(const gert::TilingContext *context,
                                                       const Mc2MatmulV3CompileInfo &compileInfo,
                                                       const Mc2MatMulV3Args &args, const Mc2MatMulV3RunInfo &runInfo)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    NpuArch npuArchHelper = ascendcPlatform.GetCurNpuArch();
    auto iter = (GetL0C2OutFuncMap.find(npuArchHelper) == GetL0C2OutFuncMap.end()) ?
                    GetL0C2OutDefault :
                    GetL0C2OutFuncMap.at(npuArchHelper);
    return iter(compileInfo, args, runInfo);
}

bool Mc2MatMulV3TilingHelper::CheckIfDoubleAswt(const gert::TilingContext *context, const Mc2MatMulV3Args &args,
                                                const uint64_t batchC)
{
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    NpuArch npuArchHelper = ascendcPlatform.GetCurNpuArch();
    auto iter = (CheckIfDoubleAswtFuncMap.find(npuArchHelper) == CheckIfDoubleAswtFuncMap.end()) ?
                    CheckIfDoubleAswtDefault :
                    CheckIfDoubleAswtFuncMap.at(npuArchHelper);
    return iter(args, batchC);
}
} // namespace mc2_matmul_v3_advanced
} // namespace optiling
