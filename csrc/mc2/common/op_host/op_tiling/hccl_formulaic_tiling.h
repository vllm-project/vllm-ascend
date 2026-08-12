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
 * \file hccl_formulaic_tiling.h
 * \brief
 */
#ifndef __HCCL_FORMULAIC_TILING_H__
#define __HCCL_FORMULAIC_TILING_H__

#pragma once
#include "matmul_formulaic_tiling.h"
#include "hccl_performance.h"
#include "matmul_performance.h"

// FormPartition参数
constexpr uint64_t LARGE_NK_BAR_BASE = 32;
constexpr uint64_t SMALL_TILECNT_PAR = 2;
constexpr uint64_t OPT_TILE_CNT = 8;
constexpr uint64_t OPT_TILE_LEN_RATIO = 2;
constexpr uint64_t OPT_TILE_LEN_RATIO_SOC310P = 1;
constexpr double BACKTILE_ALIGN_RATIO = 0.75;
constexpr uint64_t LARGE_BACKTILE_COMPARE_PAR = 2;
constexpr uint64_t SOC310_BASE_M = 256;
constexpr uint64_t SMALL_SHORTTILE_BAR = 512;
constexpr uint64_t SHORT_ALIGN_DOWN_DIFF = 16;
constexpr double ALIGN_RATIO_CALC_BOUND = 0.5;
constexpr double ALIGN_RATIO_COMM_BOUND = 0.75;
constexpr uint64_t SMALL_SHAPE_BAR = 2048;
// OneCalcOneCommBase参数
constexpr uint64_t MAX_TILE_CNT = 16;
constexpr double commGrowRatio = 1.15;
constexpr double ALLGATHERMM_COMMTIME_FACTOR = 2;
constexpr double ALLREDUCE_COMMTIME_FACTOR = 2.0517;
constexpr double REDUCESCATTER_COMMTIME_FACTOR = 2.09136;

// 切分和对齐
class FormPartition {
public:
    CutResult cutRes;                            // 切分结果
    TileArguments tileArgs;                      // 切分参数
    uint64_t totalLen = ONE;                     // 待切分的长度
    double backTileRatio = BACKTILE_ALIGN_RATIO; // 计算长块个数使用的参数

    // Constructor
    explicit FormPartition(const mc2tiling::TilingArgs &args)
    {
        totalLen = args.mValue;
        tileArgs.mAlignLen = mc2tiling::BASE_BLOCK_M; // 切分长度照顾mm性能，对齐baseM
        tileArgs.minTileLen = tileArgs.mAlignLen;
        tileArgs.maxTileCnt = ONE;
        tileArgs.maxTileLen = totalLen;

        cutRes.shortTileAtBack = false; // 计算Bound时短块后置，通信bound时短块前置
        NoCutTiling();                  // 初始化为不切分
    };

    void SetBackTileRatio(double newRatio)
    {
        backTileRatio = (newRatio > 0) ? newRatio : backTileRatio;
    };
    void SetMinLenByMax(uint64_t newLen);
    void SetMinLenByMin(uint64_t newLen);
    uint64_t GetMinLen()
    {
        return tileArgs.minTileLen;
    };
    void SetAlignLength(uint64_t newLen)
    {
        tileArgs.mAlignLen = std::max(newLen, ONE);
    };
    uint64_t GetAlignLength()
    {
        return tileArgs.mAlignLen;
    };
    void SetMaxTileCnt(uint64_t newSize)
    {
        tileArgs.maxTileCnt = std::max(newSize, ONE);
    };

    // TilingMethods
    void GenerateInitialPartition(bool smallDimAlignUp = false);
    bool SetShortTileLen(bool noCutFlag = false);
    void NoCutTiling();
    void MaxTileCntUniform();

    // 离散切分对齐 (AllGatherMatmul和MatmulReduceScatter算子)
    void AlignTileSmall();
    void AlignTileLarge();
    void FitTileLengthDiscrete(bool smallDimAlignUp, bool goodLinearityShape, bool localAtFront = false);
    // 连续切分对齐 (MatmulAllReduce算子)
    void LargeBackTileCheck(bool soc310Flag, bool largeCalcCommRatio);
    void CalcShortTile();
    void CalcBackTile();
    void SmallFrontTileCheck(bool soc310Flag, bool kGreaterThanN);
    void RedistributeTile();
    void CommBoundShortAlign();
    void FitTileLengthContinuous(bool kGreaterThanN, bool largeCalcCommRatio = true, bool soc310Flag = false);

    virtual ~FormPartition()
    {
    }
};

// 单通信域算子通信轮次切分基类，每个算子各有一个派生类
class OneCalcOneCommBase {
public:
    MatmulPerformanceModel matmulPerf_; // 计算拟合
    HCCLPerformanceModel commPerf_;     // 通信拟合
    FormPartition tilingM_;             // 切分结果
    MatmulParameters clusterInfo_;
    uint64_t rankDim_ = 2; // Tp并行维度
    uint64_t rankTileNum_ = 1;
    double ratioCalcComm_ = 1.0;
    bool noCutFlag_ = false;
    bool inferFlag_ = false; // 部分逻辑只适配了推理或者训练算子，进行区分

    // Constructor
    explicit OneCalcOneCommBase(const mc2tiling::TilingArgs &args, uint32_t inputRankDim, KernelType inputKernelType,
                                SocVersion inputSocVersion = SocVersion::SOC910_B)
        : matmulPerf_(args, inputSocVersion), commPerf_(inputRankDim, inputKernelType, inputSocVersion), tilingM_(args)
    {
        rankDim_ = inputRankDim;
        clusterInfo_ = matmulPerf_.mmShapeInfo_;
        tilingM_.SetMaxTileCnt(MAX_TILE_CNT); // 受限ffts队列大小，最多切16轮
    };

    virtual void EstimateKernelTime() {};
    virtual void SelectTilingMethod() {};
    double EstimateTotalMatmulTime();
    double EstimateTotalCommTime();

    // 流水配平优化
    void ShortAtEndCalcBoundBalancing();
    void ShortAtFrontCommBoundBalancing();

    // 调用函数
    void GetTiling();

    virtual ~OneCalcOneCommBase()
    {
    }
};

#endif // __HCCL_FORMULAIC_TILING_H__
