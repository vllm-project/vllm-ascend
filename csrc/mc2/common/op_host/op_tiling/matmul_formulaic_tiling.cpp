/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_formulaic_tiling.cpp
 * \brief
 */
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "matmul_formulaic_tiling.h"

using namespace AscendC;
using namespace ge;

namespace mc2tiling {
constexpr uint32_t MIN_SUPPORT_L2CACHE_DIM = 2; // l2cache切分当前只支持2卡以上场景(不含2卡)
void MatmulFormulaicTiling::CalcBaseBlockTiling()
{
    uint32_t defaultBaseM = runInfo_.baseM;
    uint32_t defaultBaseN = runInfo_.baseN;
    if (args_.mValue >= defaultBaseM || socInfo_.socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        runInfo_.baseM = defaultBaseM;
        // calculate baseN based on aicCoreNum
        uint32_t mCore = MathCeil(args_.mValue, runInfo_.baseM);
        uint32_t nCore = MathFloor(args_.aicCoreNum, mCore);
        auto baseN = (args_.nValue >= nCore) ? MathCeil(args_.nValue, nCore) : args_.nValue;
        runInfo_.baseN = AlignUp(std::min(baseN, defaultBaseN), C0_SIZE);
    } else {
        runInfo_.baseM = AlignUp(args_.mValue, C0_SIZE);
        // calculate baseN based on L0C and L0B
        auto baseN1 = args_.nValue;
        if (args_.nValue >= args_.aicCoreNum) {
            baseN1 = MathFloor(args_.nValue, args_.aicCoreNum);
        }
        baseN1 = AlignUp(baseN1, C0_SIZE);
        uint32_t baseN2 = MathFloor(L0C_SIZE_DB_ON / FP32_SIZE, runInfo_.baseM);
        uint32_t maxBaseN1 = AlignDown(baseN2, C0_SIZE);
        uint32_t maxBaseN2 = L0_SIZE_DB_ON / args_.bDtypeSize / 16; // 16 is min kValue
        // 2 is half of BASE_BLOCK_M
        if (args_.mValue >= (defaultBaseM / 2)) {
            maxBaseN2 = defaultBaseN;
        }
        auto baseN = std::min(std::min(baseN1, maxBaseN2), maxBaseN1);
        baseN = args_.isBias ? std::min(MAX_BIAS_BASE_BLOCK_N, baseN) : baseN;
        // select best baseN with more tailCoreNum
        uint32_t tailCoreNum1 = MathCeil(args_.nValue, baseN) % args_.aicCoreNum;
        uint32_t tailCoreNum2 = 0;
        if (baseN > defaultBaseN) {
            tailCoreNum2 = MathCeil(args_.nValue, defaultBaseN) % args_.aicCoreNum;
        }
        if (tailCoreNum1 < tailCoreNum2) {
            baseN = defaultBaseN;
        }
        runInfo_.baseN = baseN;
    }
    // calculate baseK based on L0A/L0B Size
    if ((runInfo_.baseM != 0) && (runInfo_.baseN != 0) && (args_.aDtypeSize != 0) && (args_.bDtypeSize != 0)) {
        uint32_t baseKa = L0_SIZE_DB_ON / args_.aDtypeSize / runInfo_.baseM;
        uint32_t baseKb = L0_SIZE_DB_ON / args_.bDtypeSize / runInfo_.baseN;
        auto baseK = std::min(std::min(baseKa, baseKb), args_.kValue);
        if (baseK > BASE_K_ALIGN_SIZE) {
            baseK = AlignDown(baseK, BASE_K_ALIGN_SIZE);
        } else if (baseK > C0_SIZE) {
            baseK = AlignDown(baseK, C0_SIZE);
        } else {
            baseK = C0_SIZE;
        }
        runInfo_.baseK = baseK;
    }
}

void MatmulFormulaicTiling::UpdateDepth()
{
    auto depthA1 = runInfo_.depthA1;
    auto depthB1 = runInfo_.depthB1;
    while (depthA1 > MIN_DEPTH && depthB1 > MIN_DEPTH) {
        uint64_t depthASize = depthA1 * runInfo_.baseM * runInfo_.baseK * args_.aDtypeSize;
        uint64_t depthBSize = depthB1 * runInfo_.baseN * runInfo_.baseK * args_.bDtypeSize;
        if ((depthASize + depthBSize) <= L1_SIZE) {
            break;
        }
        depthA1 -= MIN_DEPTH;
        depthB1 -= MIN_DEPTH;
    }
    runInfo_.depthA1 = depthA1;
    runInfo_.depthB1 = depthB1;
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = runInfo_.baseN;
    OP_LOGD(opName_, "depthA1 is %u, depthB1 is %u", depthA1, depthB1);
}

bool MatmulFormulaicTiling::DoL2CacheTiling()
{
    auto rankTileM = args_.rankTileM;
    uint64_t sizeA = rankTileM * args_.kValue * args_.aDtypeSize;
    uint64_t sizeB = args_.kValue * args_.nValue * args_.bDtypeSize;
    uint64_t sizeC = rankTileM * args_.nValue * args_.cDtypeSize;
    uint64_t totalSize = sizeA + sizeB + sizeC;
    constexpr uint32_t limitSize = 128 * MB_SIZE; // 128 empiric value for l2 cache tile size
    if (totalSize < socInfo_.l2Size || (sizeA < limitSize && sizeB < limitSize && sizeC < limitSize)) {
        OP_LOGD(opName_, "data size is small L2CacheSize, so needn't split L2.");
        return false;
    }

    constexpr uint32_t length = 8192 * 3;
    uint64_t size = rankTileM + args_.nValue + args_.kValue;
    uint32_t tileSize = 45 * MB_SIZE;  // [15, 45] split L2 max tile size, 45 = 128/3
    uint32_t tileLimit = 15 * MB_SIZE; // [15, 45] split L2 min tile size
    // 20 is core numbers
    if (args_.aicCoreNum > 20 && size < length) {
        tileSize = 64 * MB_SIZE;  // [16, 64] split L2 max tile size, 64 = 128/2
        tileLimit = 16 * MB_SIZE; // [16, 64] split L2 min tile size
    }

    uint32_t mTileBlocks = MathCeil(tileSize / args_.kValue / args_.aDtypeSize, runInfo_.baseM);
    uint32_t nTileBlocks = MathCeil(tileSize / args_.kValue / args_.bDtypeSize, runInfo_.baseN);
    auto mTotalBlocks = MathCeil(args_.mValue, runInfo_.baseM) * args_.rankTileNum;
    auto nTotalBlocks = MathCeil(args_.nValue, runInfo_.baseN);
    auto mTileCnt = MathCeil(mTotalBlocks, mTileBlocks);
    auto nTileCnt = MathCeil(nTotalBlocks, nTileBlocks);

    // 不满足切分条件
    if (mTotalBlocks <= mTileBlocks || sizeA <= tileLimit) {
        mTileBlocks = mTotalBlocks;
        mTileCnt = 1;
    }
    if (nTotalBlocks <= nTileBlocks || sizeB <= tileLimit) {
        nTileBlocks = nTotalBlocks;
        nTileCnt = 1;
    }
    if ((mTileBlocks * nTileBlocks) < args_.aicCoreNum) {
        OP_LOGW(opName_, "L2cache tile fail, not fully use core num, enter splitk kernel.");
        return false;
    }

    if (mTileCnt > 1 || nTileCnt > 1) {
        runInfo_.l2Info.mL2TileCnt = mTileCnt;
        runInfo_.l2Info.mTileBlocks = mTileBlocks;
        runInfo_.l2Info.mTailBlocks = mTotalBlocks - mTileBlocks * (mTileCnt - 1);
        runInfo_.l2Info.nL2TileCnt = nTileCnt;
        runInfo_.l2Info.nTileBlocks = nTileBlocks;
        runInfo_.l2Info.nTailBlocks = nTotalBlocks - nTileBlocks * (nTileCnt - 1);
        OP_LOGD(opName_, "nTileCnt or mTileCnt bigger than 1, enable split L2cache.");
        return true;
    }

    OP_LOGD(opName_, "mValue and nValue not meet L2 Split Conditions, so not enable split "
                     "L2.");
    return false;
}

void MatmulFormulaicTiling::SetWeightFormat(const matmul_tiling::CubeFormat weightFormat)
{
    weightFormat_ = weightFormat;
}

ge::graphStatus MatmulFormulaicTiling::GetCubeTiling(TilingArgs &args, AscendC::tiling::TCubeTiling &cubeTiling,
                                                     Mc2Tiling::TileL2Tiling &tileL2Tiling)
{
    // 1.设置默认BaseM/N/K
    InitBaseBlockTiling();
    InitTilingArgs(args);
    // 2.小Shape的BaseM/N/K计算
    uint32_t usedCoreNum = MathCeil(args_.rankTileM, runInfo_.baseM) * MathCeil(args_.nValue, runInfo_.baseN);
    if (args_.nValue == 0) {
        usedCoreNum = MathCeil(args_.rankTileM, runInfo_.baseM) * MathCeil(args_.kValue, runInfo_.baseK);
    }
    if (usedCoreNum <= args_.aicCoreNum * FOMUL_AIC_NUM_THRESHOLD) {
        CalcBaseBlockTiling();
    }
    // 3.计算Depth&Step数据
    UpdateDepth();
    runInfo_.stepKa = runInfo_.depthA1 / DB_ON;
    runInfo_.stepKb = runInfo_.depthB1 / DB_ON;

    // 4.计算L2Cache切分的TilingData
    bool enableL2Tile = DoL2CacheTiling();

    // 5.设置TilingData
    if (args_.nValue == 0) {
        usedCoreNum = MathCeil(args_.rankTileM, runInfo_.baseM) * MathCeil(args_.kValue, runInfo_.baseK);
    } else {
        usedCoreNum = MathCeil(args_.rankTileM, runInfo_.baseM) * MathCeil(args_.nValue, runInfo_.baseN);
    }
    usedCoreNum = std::min(usedCoreNum, args_.aicCoreNum);
    OP_LOGD(opName_, "usedCoreNum is %u.", usedCoreNum);
    cubeTiling.usedCoreNum = usedCoreNum;
    cubeTiling.singleCoreM = runInfo_.baseM;
    cubeTiling.singleCoreN = runInfo_.baseN;
    cubeTiling.singleCoreK = args.kValue;
    cubeTiling.baseM = runInfo_.baseM;
    cubeTiling.baseN = runInfo_.baseN;
    cubeTiling.baseK = runInfo_.baseK;
    cubeTiling.depthA1 = runInfo_.depthA1;
    cubeTiling.depthB1 = runInfo_.depthB1;
    cubeTiling.stepM = 1;
    cubeTiling.stepN = 1;
    cubeTiling.stepKa = runInfo_.stepKa;
    cubeTiling.stepKb = runInfo_.stepKb;
    cubeTiling.dbL0C = 1;          // 这里是关闭L0C的double buffer，需要适配打开
    tileL2Tiling.enableL2Tile = 0; //
    if (enableL2Tile && args_.rankDim > MIN_SUPPORT_L2CACHE_DIM) {
        tileL2Tiling.mL2TileCnt = runInfo_.l2Info.mL2TileCnt;
        tileL2Tiling.nL2TileCnt = runInfo_.l2Info.nL2TileCnt;
        tileL2Tiling.mTileBlocks = runInfo_.l2Info.mTileBlocks;
        tileL2Tiling.nTileBlocks = runInfo_.l2Info.nTileBlocks;
        tileL2Tiling.mTailBlocks = runInfo_.l2Info.mTailBlocks;
        tileL2Tiling.nTailBlocks = runInfo_.l2Info.nTailBlocks;
        tileL2Tiling.rankTileNum = args.rankTileNum;
        tileL2Tiling.enableL2Tile = 1;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatmulFormulaicTiling::GetCubeTiling(TilingArgs &args, AscendC::tiling::TCubeTiling &cubeTiling)
{
    // 1.设置默认BaseM/N/K
    InitBaseBlockTiling();
    InitTilingArgs(args);
    // 2.小Shape的BaseM/N/K计算
    uint32_t usedCoreNum = MathCeil(args_.mValue, runInfo_.baseM) * MathCeil(args_.nValue, runInfo_.baseN);
    if (usedCoreNum <= args_.aicCoreNum * FOMUL_AIC_NUM_THRESHOLD) {
        CalcBaseBlockTiling();
    }
    usedCoreNum = MathCeil(args_.mValue, runInfo_.baseM) * MathCeil(args_.nValue, runInfo_.baseN);
    usedCoreNum = std::min(usedCoreNum, args_.aicCoreNum);
    runInfo_.usedCoreNum = usedCoreNum;

    // 3.计算Depth&Step数据
    UpdateDepth();
    runInfo_.stepKa = runInfo_.depthA1 / DB_ON;
    runInfo_.stepKb = runInfo_.depthB1 / DB_ON;
    args.usedCoreNum = runInfo_.usedCoreNum;
    // 5.设置TilingData
    OP_LOGD(opName_, "usedCoreNum is %u.", runInfo_.usedCoreNum);
    cubeTiling.usedCoreNum = runInfo_.usedCoreNum;
    cubeTiling.singleCoreM = runInfo_.singleCoreM;
    cubeTiling.singleCoreN = runInfo_.singleCoreN;
    cubeTiling.singleCoreK = args.kValue;
    cubeTiling.baseM = runInfo_.baseM;
    cubeTiling.baseN = runInfo_.baseN;
    cubeTiling.baseK = runInfo_.baseK;
    cubeTiling.depthA1 = runInfo_.depthA1;
    cubeTiling.depthB1 = runInfo_.depthB1;
    cubeTiling.stepM = 1;
    cubeTiling.stepN = 1;
    cubeTiling.stepKa = runInfo_.stepKa;
    cubeTiling.stepKb = runInfo_.stepKb;
    cubeTiling.dbL0C = 1; // 这里是关闭L0C的double buffer，需要适配打开
    return ge::GRAPH_SUCCESS;
}

uint32_t MatmulFormulaicTiling::GetRankSize(const char *group)
{
    uint32_t rankSize = 8;
    if (Mc2Hcom::MC2HcomTopology::CommGetInstSizeByGroup(group, &rankSize) != HCCL_SUCCESS) {
        OP_LOGE("", "Get rank size of group %s failed", group);
        return 0;
    }
    return rankSize;
}

void MatmulFormulaicTiling::InitBaseBlockTiling()
{
    runInfo_.baseM = BASE_BLOCK_M;
    runInfo_.baseN = BASE_BLOCK_N;
    runInfo_.baseK = BASE_BLOCK_K;
    runInfo_.depthA1 = DEPTH_A1;
    runInfo_.depthB1 = DEPTH_B1;
    if (socInfo_.socVersion == platform_ascendc::SocVersion::ASCEND310P) {
        runInfo_.baseM = BASE_BLOCK_M_L2CACHE;
        runInfo_.baseN = BASE_BLOCK_N_L2CACHE;
        runInfo_.baseK = BASE_BLOCK_K_L2CACHE;
        runInfo_.depthA1 = DEPTH_A1_L2CACHE;
        runInfo_.depthB1 = DEPTH_B1_L2CACHE;
        if (weightFormat_ == matmul_tiling::CubeFormat::NZ) {
            runInfo_.baseM = BASE_BLOCK_M_L2CACHE_NZ;
            runInfo_.baseN = BASE_BLOCK_N_L2CACHE_NZ;
            runInfo_.baseK = BASE_BLOCK_K_L2CACHE_NZ;
            runInfo_.depthA1 = DEPTH_A1_L2CACHE_NZ;
            runInfo_.depthB1 = DEPTH_B1_L2CACHE_NZ;
        }
    }
}

// 冗余函数，兼容all_reduce，待all_reduce重构后删除
void MatmulFormulaicTiling::GetBaseBlockParm(const platform_ascendc::SocVersion &version, uint64_t &blockBaseM,
                                             uint64_t &blockBaseN, uint64_t &blockBaseK, uint64_t &blockDepthA1,
                                             uint64_t &blockDepthB1)
{
    blockBaseM = BASE_BLOCK_M;
    blockBaseN = BASE_BLOCK_N;
    blockBaseK = BASE_BLOCK_K;
    blockDepthA1 = DEPTH_A1;
    blockDepthB1 = DEPTH_B1;
    if (version == platform_ascendc::SocVersion::ASCEND310P) {
        blockBaseM = BASE_BLOCK_M_L2CACHE;
        blockBaseN = BASE_BLOCK_N_L2CACHE;
        blockBaseK = BASE_BLOCK_K_L2CACHE;
        blockDepthA1 = DEPTH_A1_L2CACHE;
        blockDepthB1 = DEPTH_B1_L2CACHE;
    }
}

void MatmulFormulaicTiling::InitTilingArgs(TilingArgs &args)
{
    // Init Tiling Arguments
    args_.mValue = args.mValue;
    args_.nValue = args.nValue;
    args_.kValue = args.kValue;
    args_.isATrans = args.isATrans;
    args_.isBTrans = args.isBTrans;
    args_.isBias = args.isBias;
    args_.aDtypeSize = args.inputDtypeSize;
    args_.bDtypeSize = args.inputDtypeSize;
    args_.cDtypeSize = args.outputDtypeSize;
    args_.rankDim = args.rankDim;
    args_.rankM = args.mValue * args.rankDim;
    if (args.commAlg == optiling::COMM_ALG_DOUBLE_RING && !args.isLocal) {
        args_.rankM *= optiling::DOUBLE_RING_FACTOR;
    }
    OP_LOGD(opName_, " args_.rankM: %u.", args_.rankM);
    args_.rankTileM = args.rankTileNum * args.mValue;
    args_.rankTileNum = args.rankTileNum;
    args_.aicCoreNum = args.aicCoreNum;
}
} // namespace mc2tiling
