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
 * \file one_calc_two_comm_tiling.cpp
 * \brief
 */

#include <cmath>
#include "mc2_log.h"
#include "one_calc_two_comm_tiling.h"

void OneCalcTwoCommBase::ECutInit(CutResult &eCut)
{
    eCut.longTileLen = clusterInfo.batchSize;
    eCut.numLongTile = ONE;
    eCut.shortTileLen = 0;
    eCut.numShortTile = 0;
    eCut.shortTileAtBack = true;
    eCut.totalTileCnt = eCut.numLongTile;
}

void OneCalcTwoCommBase::EAxisCut(uint64_t tmpCnt, uint64_t eSize, CutResult &eCut) const
{
    tmpCnt = std::min(eSize, tmpCnt);
    tmpCnt = std::max(ONE, tmpCnt);
    while (tmpCnt > ONE && (eSize % tmpCnt != 0)) { // eSize代表E/Ep。约束E轴均匀切分，tmpCnt整除(E/Ep)
        tmpCnt--;
    }
    eCut.numLongTile = tmpCnt;
    eCut.longTileLen = eSize / tmpCnt;
    eCut.totalTileCnt = eCut.numLongTile;
}

void OneCalcTwoCommBase::MaxTileCntCheckForE()
{
    bool obeyMaxCnt = (localCutE.numLongTile + cutE.numLongTile * tilingC.cutRes.totalTileCnt) <= MAX_TILE_CNT_TWO_COMM;
    if (obeyMaxCnt) { // 已经满足约束
        return;
    }

    // 根据MAX_CNT调整E轴切分
    uint64_t tmpCnt = ONE;
    if (epDim <= TWO) { // 切分轮次均分给localCutE和cutE
        // if MAX_CNT not divisible by 2, give remainder cut to cutE
        tmpCnt = std::min(MAX_TILE_CNT_TWO_COMM / TWO, localCutE.numLongTile);
        EAxisCut(tmpCnt, clusterInfo.batchSize, localCutE);
        tmpCnt = std::min(MAX_TILE_CNT_TWO_COMM - tmpCnt, cutE.numLongTile);
        EAxisCut(tmpCnt, clusterInfo.batchSize, cutE);
    } else { // 优先切cutE
        tmpCnt = std::min(MAX_TILE_CNT_TWO_COMM - ONE,
                          cutE.numLongTile); // local本身要占用1个切分轮次
        EAxisCut(tmpCnt, clusterInfo.batchSize, cutE);

        if ((cutE.numLongTile + localCutE.numLongTile) > MAX_TILE_CNT_TWO_COMM) {
            tmpCnt = std::min(MAX_TILE_CNT_TWO_COMM - cutE.numLongTile, localCutE.numLongTile);
            EAxisCut(tmpCnt, clusterInfo.batchSize, localCutE);
        }
    }
}

void OneCalcTwoCommBase::GetTiling()
{
    OP_LOGD("MC2 Op Formulaic Tiling", "Shape info: Ep %lu, Tp %lu, E/Ep %lu, C %lu, H %lu, M %lu\n", epDim, tpDim,
            clusterInfo.batchSize, tilingC.totalLen, clusterInfo.kValue, clusterInfo.nValue);
    // Get threshold values
    // Local AG/RS: tile_e_local * C * (H/Tp) * Tp * dTypeSize
    // Non-local AG/RS: tile_e * ((Ep - 1) * tile_c) * (H/Tp) * Tp * dTypeSize
    // A2A: Ep * tile_c * (H/Tp) * dTypeSize
    // Non-local BMM: tile_e * ((Ep - 1) * tile_c) * H * (M / Tp), or tile_e *
    // ((Ep - 1) * Tp * tile_c) * H * (M / Tp)
    minProductLocalEC = tpCommPerf.GetLinearThresholdLenCoarse();
    minProductLocalEC = std::max(minProductLocalEC, bmmPerf.GetLinearThresholdLen(ONE));
    uint64_t nonLocalEpSize = epDim - ONE;
    minProductEC = std::max(minProductLocalEC / nonLocalEpSize, MIN_SPLIT_256);
    minC = epCommPerf.GetLinearThresholdLenCoarse(); // A2A cannot skip local
    OP_LOGD("MC2 Op Formulaic Tiling", "epDim %lu, nonLocalEpSize %lu\n", epDim, nonLocalEpSize);
    OP_LOGD("MC2 Op Formulaic Tiling", "Min: productLocalEC %lu, productEC %lu, minC %lu\n", minProductLocalEC,
            minProductEC, minC);

    // Cut local E
    if (cutLocal) {
        uint64_t tmpSize = std::max(minProductLocalEC / tilingC.totalLen,
                                    ONE); // use full C length
        uint64_t tmpCnt = std::max(clusterInfo.batchSize / tmpSize, ONE);
        EAxisCut(tmpCnt, clusterInfo.batchSize, localCutE);
    }

    // Cut E
    uint64_t tmpSize = 0;
    if (tilingC.totalLen < MIN_SPLIT_EC_LENGTH) {
        tmpSize = std::max(minProductEC / tilingC.totalLen, TWO);
    } else {
        tmpSize = std::max(minProductEC / tilingC.totalLen, ONE); // use full C length
    }
    uint64_t tmpCnt = std::max(clusterInfo.batchSize / tmpSize, ONE);
    EAxisCut(tmpCnt, clusterInfo.batchSize, cutE);

    OP_LOGD("MC2 Op Formulaic Tiling", "Initial E cut: Local E %lu %lu, E %lu %lu\n", localCutE.numLongTile,
            localCutE.longTileLen, cutE.numLongTile, cutE.longTileLen);

    // E max cnt check
    MaxTileCntCheckForE();

    // Cut c
    uint64_t minCutNum = TWO; // if cut c, c totalTileCnt >= 2
    bool underMaxCut = localCutE.totalTileCnt + minCutNum * cutE.totalTileCnt <= MAX_TILE_CNT_TWO_COMM;
    uint64_t maxCutNum = MAX_ALL_TO_ALL_NUM / clusterInfo.batchSize;
    underMaxCut = underMaxCut && (maxCutNum >= minCutNum);
    OP_LOGD("MC2 Op Formulaic Tiling", "maxCutNum: %lu, Enter cut C: %d\n", maxCutNum, underMaxCut);
    if (underMaxCut) {
        // Updates C threshold size
        OP_LOGD("MC2 Op Formulaic Tiling", "c threshold minC %lu, minProductEC %lu\n", minC, minProductEC);
        minC = std::max(minC, minProductEC / cutE.longTileLen);
        tilingC.SetMinLenByMax(minC);

        // Try cut C
        if (!tilingC.SetShortTileLen()) {
            tilingC.cutRes.longTileLen = tilingC.cutRes.shortTileLen;
            tilingC.GenerateInitialPartition();

            // C max cnt check
            maxCutNum = std::min(maxCutNum, (MAX_TILE_CNT_TWO_COMM - localCutE.totalTileCnt) / cutE.totalTileCnt);
            tilingC.SetMaxTileCnt(maxCutNum);
            bool kGreaterThanN = clusterInfo.kValue > clusterInfo.nValue;
            tilingC.FitTileLengthContinuous(kGreaterThanN);
            OP_LOGD("MC2 Op Formulaic Tiling", "C target:%d,%lu,%lu,%lu,%lu\n", tilingC.cutRes.shortTileAtBack,
                    tilingC.cutRes.numLongTile, tilingC.cutRes.longTileLen, tilingC.cutRes.numShortTile,
                    tilingC.cutRes.shortTileLen);

            // 长块短块一样长时归一
            if (tilingC.cutRes.shortTileLen == tilingC.cutRes.longTileLen) {
                tilingC.cutRes.shortTileLen = 0;
                tilingC.cutRes.numShortTile = 0;
                tilingC.cutRes.numLongTile++;
            }
        }
    }
}

void OneCalcTwoCommShardHBase::InitCutResult(CutResult &tmpCut, uint64_t totalLen) const
{
    tmpCut.longTileLen = totalLen;
    tmpCut.numLongTile = ONE;
    tmpCut.shortTileLen = 0;
    tmpCut.numShortTile = 0;
    tmpCut.shortTileAtBack = true;
    tmpCut.totalTileCnt = tmpCut.numLongTile;
}

void OneCalcTwoCommShardHBase::AsignMaxCutNumForBranches(double totalBMMTime, double totalCommTime)
{
    uint64_t maxTotalCnt = MAX_TILE_CNT_TWO_COMM;
    if (epDim > TWO) { // Do not cut local branch
        maxNonLocalCnt = maxTotalCnt - ONE;
        return;
    }
    double localMatmulTime = totalBMMTime / static_cast<double>(epDim);
    double nonlocalCommTime = totalCommTime - totalCommTime / static_cast<double>(epDim);
    if (totalCommTime < localMatmulTime) { // Do not cut local branch
        maxLocalCnt = maxTotalCnt - ONE;
    } else if (totalBMMTime < nonlocalCommTime) { // Do not cut non-local branch
        maxNonLocalCnt = maxTotalCnt - ONE;
    } else { // cut both branches
        maxNonLocalCnt = maxTotalCnt / TWO;
        maxLocalCnt = maxTotalCnt - maxNonLocalCnt;
    }
}

void OneCalcTwoCommShardHBase::CutAxisE(CutResult &cutRes, uint64_t maxCutNum, uint64_t minLen,
                                        double unbalanceRatio) const
{
    uint64_t totalLen = cutRes.longTileLen;
    minLen = std::max(minLen, ONE);
    OP_LOGD("MC2 Op Formulaic Tiling", "CutAxisE: minLen %lu, totalLen %lu\n", minLen, totalLen);
    if (TWO * minLen > totalLen) { // Total length too short, do not cut
        OP_LOGD("MC2 Op Formulaic Tiling", "CutAxisE: no cut branch\n");
        return;
    }

    if (maxCutNum == TWO) { // Cut by minByMaxRatio
        OP_LOGD("MC2 Op Formulaic Tiling", "CutAxisE: 2 cut branch\n");
        double targetLong = static_cast<double>(totalLen) / (ONE + unbalanceRatio);
        uint64_t tmpTileLen = std::min(totalLen / TWO, totalLen - static_cast<uint64_t>(targetLong));
        OP_LOGD("MC2 Op Formulaic Tiling", "targetShort %f, tmpTileLen %lu\n", targetLong, tmpTileLen);
        tmpTileLen = std::max(minLen, tmpTileLen);
        cutRes.longTileLen = (tmpTileLen * TWO >= totalLen) ? tmpTileLen : (totalLen - tmpTileLen);
    } else { // Uniform cut using minLen
        OP_LOGD("MC2 Op Formulaic Tiling", "CutAxisE: more cut branch\n");
        uint64_t targetLen = totalLen / maxCutNum;
        targetLen = (totalLen > targetLen * maxCutNum) ? (targetLen + 1) : targetLen;
        cutRes.longTileLen = std::max(targetLen, minLen);
    }

    // Generate final cut
    cutRes.numLongTile = totalLen / cutRes.longTileLen;
    if (totalLen > cutRes.numLongTile * cutRes.longTileLen) { // Need short tile
        cutRes.numShortTile = ONE;
        cutRes.shortTileLen = totalLen - cutRes.numLongTile * cutRes.longTileLen;
    }
    cutRes.totalTileCnt = cutRes.numLongTile + cutRes.numShortTile;
}

void OneCalcTwoCommShardHBase::CutAxisC(uint64_t maxCutNum, uint64_t minLen, double unbalanceRatio)
{
    uint64_t totalLen = tilingC.totalLen;
    OP_LOGD("MC2 Op Formulaic Tiling", "CutAxisC: minLen %lu, totalLen %lu\n", minLen, totalLen);
    tilingC.SetMinLenByMax(minLen);
    if (tilingC.SetShortTileLen()) { // Total length too short, do not cut
        OP_LOGD("MC2 Op Formulaic Tiling", "CutAxisC: no cut branch\n");
        return;
    }
    tilingC.SetMaxTileCnt(maxCutNum);

    if (maxCutNum == TWO) { // Cut by minByMaxRatio
        OP_LOGD("MC2 Op Formulaic Tiling", "CutAxis: 2 cut branch\n");
        double targetLong = static_cast<double>(totalLen) / (ONE + unbalanceRatio);
        uint64_t tmpTileLen = std::min(totalLen / TWO, totalLen - static_cast<uint64_t>(targetLong));
        OP_LOGD("MC2 Op Formulaic Tiling", "targetShort %f, tmpTileLen %lu\n", targetLong, tmpTileLen);
        tmpTileLen = std::max(minLen, tmpTileLen);
        tmpTileLen = (tmpTileLen + C_ALIGN_LENGTH - 1) / C_ALIGN_LENGTH * C_ALIGN_LENGTH;
        tilingC.cutRes.shortTileLen = (tmpTileLen * TWO < totalLen) ? tmpTileLen : (totalLen - tmpTileLen);
        tilingC.cutRes.longTileLen = totalLen - tilingC.cutRes.shortTileLen;
        tilingC.cutRes.numLongTile = ONE;
    } else { // Uniform cut using minLen
        OP_LOGD("MC2 Op Formulaic Tiling", "CutAxis: more cut branch\n");
        tilingC.cutRes.longTileLen = tilingC.cutRes.shortTileLen;
        tilingC.GenerateInitialPartition();
        bool kGreaterThanN = clusterInfo.kValue > clusterInfo.nValue;
        tilingC.FitTileLengthContinuous(kGreaterThanN);
        OP_LOGD("MC2 Op Formulaic Tiling", "C target: %d, %lu, %lu, %lu, %lu\n", tilingC.cutRes.shortTileAtBack,
                tilingC.cutRes.numLongTile, tilingC.cutRes.longTileLen, tilingC.cutRes.numShortTile,
                tilingC.cutRes.shortTileLen);
    }

    tilingC.cutRes.totalTileCnt = tilingC.cutRes.numLongTile + tilingC.cutRes.numShortTile;
}

void OneCalcTwoCommShardHBase::TrimCutResult(CutResult &tmpCut, bool setShortFlag) const
{
    // 长短块一样长时归一
    if (tmpCut.shortTileLen == tmpCut.longTileLen) {
        tmpCut.shortTileLen = 0;
        tmpCut.numShortTile = 0;
        tmpCut.numLongTile++;
    } else if (setShortFlag && tmpCut.numShortTile > 0) {
        tmpCut.shortTileAtBack = false;
    }
}

void OneCalcTwoCommShardHBase::GetTiling()
{
    // Local AG/RS: tile_e_local * C * (H/Tp) * Tp *dTypeSize
    // Non-local AG/RS: tile_e * ((Ep - 1) * tile_c) * (H/Tp) * Tp *dTypeSize
    // A2A: Ep * tile_c * (H/Tp) * dTypeSize
    // Non-local BMM: tile_e * ((Ep - 1) * tile_c) * H * (M / Tp)

    // 1. Estimate matmul, communication time
    bmmPerf.FindCubeUtil(clusterInfo.mValue, ONE, false);
    bmmPerf.GetMatmulGradient();
    uint64_t fullM = clusterInfo.batchSize * epDim * clusterInfo.mValue; // (E/ep) * (ep * C)
    double totalBMMTime = bmmPerf.MatmulTime(fullM, ONE);
    double epCommTime = epCommPerf.CommTime(fullM / tpDim);
    double tpCommTime = tpCommPerf.CommTime(fullM / TWO); // Double-ring comm, stepSize = tpDim, rankTile = 2
    double totalCommTime = epCommTime + tpCommTime;
    double unbalanceRatio = std::min(totalBMMTime, totalCommTime) / std::max(totalBMMTime, totalCommTime);
    unbalanceRatio = std::max(MIN_UNBALANCE_RATIO, unbalanceRatio);
    OP_LOGD("MC2 Op Formulaic Tiling",
            "MOE shard H: batch %lu, mValue %lu, kValue %lu, nValue %lu, "
            "totalBMMTime %f, epCommTime %f, tpCommTime %f, unbalanceRatio %f\n",
            clusterInfo.batchSize, clusterInfo.mValue, clusterInfo.kValue, clusterInfo.nValue, totalBMMTime, epCommTime,
            tpCommTime, unbalanceRatio);

    // 2. Get minimum cut length
    // Get threshold values
    uint64_t minProductLocalEC = tpCommPerf.GetLinearThresholdLenCoarse();
    minProductLocalEC = std::max(minProductLocalEC, bmmPerf.GetLinearThresholdLen(ONE));
    uint64_t nonLocalEpSize = epDim - ONE;
    uint64_t minProductEC = std::max(minProductLocalEC / nonLocalEpSize, ONE);
    uint64_t minC = epCommPerf.GetLinearThresholdLenCoarse(); // A2A cannot skip local
    minC = std::max(minC, bmmPerf.GetLinearThresholdLen(ONE)) / nonLocalEpSize;
    OP_LOGD("MC2 Op Formulaic Tiling", "epDim %lu, nonLocalEpSize %lu\n", epDim, nonLocalEpSize);
    OP_LOGD("MC2 Op Formulaic Tiling", "Min: productLocalEC %lu, productEC %lu, minC %lu\n", minProductLocalEC,
            minProductEC, minC);

    // 3. Asign maximum cut number for local and non-local branches
    AsignMaxCutNumForBranches(totalBMMTime, totalCommTime);

    // 4. Cut local branch
    if (maxLocalCnt > ONE) {
        uint64_t minSizeLocalE = std::max(minProductLocalEC / clusterInfo.mValue,
                                          ONE); // use full C length
        CutAxisE(localCutE, maxLocalCnt, minSizeLocalE, unbalanceRatio);
        if (maxLocalCnt > localCutE.totalTileCnt) { // Pass redundant counts to non-local
            maxNonLocalCnt += maxLocalCnt - localCutE.totalTileCnt;
        }
    }

    // 5. Cut non-local branch
    if (maxNonLocalCnt > ONE) {
        // Cut non-local E
        uint64_t minSizeE = std::max(minProductEC / clusterInfo.mValue, ONE); // use full C length
        CutAxisE(cutE, maxNonLocalCnt, minSizeE, unbalanceRatio);

        // Cut non-local C
        uint64_t eTileLen = std::max(cutE.longTileLen,
                                     cutE.shortTileLen); // Try cut c only if eTileLen == 1
        maxNonLocalCnt = maxNonLocalCnt / cutE.totalTileCnt;
        bool flagCutC = (maxNonLocalCnt > ONE) && (eTileLen == ONE);
        OP_LOGD("MC2 Op Formulaic Tiling", "Before cut C: maxNonLocalCnt %lu, eTileLen %lu, flagCutC %d\n",
                maxNonLocalCnt, eTileLen, flagCutC);
        if (flagCutC) {
            minC = std::max(minC, minProductEC / cutE.longTileLen);
            CutAxisC(maxNonLocalCnt, minC, unbalanceRatio);
        }
    }

    // 6. Set bool value isShortTileAtBack
    bool flagSetShort = SetShortTilePositionFlag(totalBMMTime, totalCommTime);
    TrimCutResult(localCutE, flagSetShort);
    TrimCutResult(cutE, flagSetShort);
    TrimCutResult(tilingC.cutRes, flagSetShort);
}
