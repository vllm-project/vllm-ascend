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
 * \file one_calc_two_comm_tiling.h
 * \brief
 */

#ifndef __ONE_CALC_TWO_COMM_BASE_H__
#define __ONE_CALC_TWO_COMM_BASE_H__

#pragma once
#include "hccl_performance.h"
#include "matmul_performance.h"
#include "hccl_formulaic_tiling.h"

constexpr uint64_t MAX_TILE_CNT_TWO_COMM = 5;
constexpr uint64_t MAX_ALL_TO_ALL_NUM = 32;
constexpr uint64_t MIN_SPLIT_256 = 256;
constexpr uint64_t MIN_SPLIT_EC_LENGTH = 512;
constexpr uint64_t C_ALIGN_LENGTH = 128;
constexpr uint64_t C_MIN_LENGTH = 128;
constexpr double MIN_UNBALANCE_RATIO = 0.2;

class OneCalcTwoCommBase {
public:
    MatmulPerformanceModel bmmPerf;
    HCCLPerformanceModel epCommPerf; // AllToAll
    HCCLPerformanceModel tpCommPerf; // AllGather or ReduceScatter
    FormPartition tilingC;           // Output parameters
    CutResult cutE;                  // Output parameters
    CutResult localCutE;             // Output parameters
    MatmulParameters clusterInfo;
    uint64_t tpDim = ONE;
    uint64_t epDim = ONE;
    bool cutLocal = true;
    uint64_t minProductLocalEC = ONE;
    uint64_t minC = ONE;
    uint64_t minProductEC = ONE;

    // Constructor
    explicit OneCalcTwoCommBase(const mc2tiling::TilingArgs &args, uint64_t inputEpDim, uint64_t inputTpDim,
                                uint64_t batchSize, SocVersion inputSocVersion = SocVersion::SOC910_93)
        : bmmPerf(args, inputSocVersion), epCommPerf(inputEpDim, KernelType::ALL_TO_ALL, inputSocVersion),
          tpCommPerf(inputTpDim, KernelType::ALL_GATHER, inputSocVersion), tilingC(args)
    {
        epDim = inputEpDim;
        tpDim = inputTpDim;
        bmmPerf.SetBatchSize(batchSize);
        clusterInfo = bmmPerf.mmShapeInfo_;

        ECutInit(cutE);
        ECutInit(localCutE);
        tilingC.cutRes.shortTileAtBack = true;
    };

    // TilingMethods
    void ECutInit(CutResult &eCut);
    void EAxisCut(uint64_t tmpCnt, uint64_t eSize, CutResult &eCut) const;
    void MaxTileCntCheckForE();

    // Wrapper function
    void GetTiling();

    virtual ~OneCalcTwoCommBase()
    {
    }
};

class OneCalcTwoCommShardHBase {
public:
    MatmulPerformanceModel bmmPerf;
    HCCLPerformanceModel epCommPerf; // AllToAll
    HCCLPerformanceModel tpCommPerf; // AllGather or ReduceScatter
    CutResult cutE;                  // Output parameters
    FormPartition tilingC;           // Output parameters
    CutResult localCutE;             // Output parameters
    MatmulParameters clusterInfo;
    uint64_t tpDim = ONE;
    uint64_t epDim = ONE;
    uint64_t maxLocalCnt = 1;    // Try cut local when maxLocalCut > 1
    uint64_t maxNonLocalCnt = 1; // Try cut non-local when maxNonLocalCut > 1

    // Constructor
    explicit OneCalcTwoCommShardHBase(const mc2tiling::TilingArgs &args, uint64_t inputEpDim, uint64_t inputTpDim,
                                      uint64_t batchSize, SocVersion inputSocVersion = SocVersion::SOC910_93)
        : bmmPerf(args, inputSocVersion), epCommPerf(inputEpDim, KernelType::ALL_TO_ALL, inputSocVersion),
          tpCommPerf(inputTpDim, KernelType::ALL_GATHER, inputSocVersion), tilingC(args)
    {
        epDim = inputEpDim;
        epCommPerf.SetFullMeshCommTimeFactor();

        tpDim = inputTpDim;
        bmmPerf.SetBatchSize(batchSize);
        clusterInfo = bmmPerf.mmShapeInfo_;

        InitCutResult(cutE, clusterInfo.batchSize);
        InitCutResult(localCutE, clusterInfo.batchSize);
        tilingC.cutRes.shortTileAtBack = true;
    };

    // TilingMethods
    void InitCutResult(CutResult &tmpCut, uint64_t totalLen) const;
    void CutAxisE(CutResult &cutRes, uint64_t maxCutNum, uint64_t minLen, double unbalanceRatio) const;
    void CutAxisC(uint64_t maxCutNum, uint64_t minLen, double unbalanceRatio);
    void AsignMaxCutNumForBranches(double totalBMMTime, double totalCommTime);
    virtual bool SetShortTilePositionFlag([[maybe_unused]] double totalBMMTime, [[maybe_unused]] double totalCommTime)
    {
        return false;
    };
    void TrimCutResult(CutResult &tmpCut, bool setShortFlag) const;

    // wrapper function
    void GetTiling();

    virtual ~OneCalcTwoCommShardHBase()
    {
    }
};

#endif // __ONE_CALC_TWO_COMM_BASE_H__
