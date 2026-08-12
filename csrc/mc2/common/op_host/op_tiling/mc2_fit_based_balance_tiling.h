/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_fit_based_balance_tiling.h
 * \brief
 */
#ifndef __MC2_FIT_BALANCE_TILING_H__
#define __MC2_FIT_BALANCE_TILING_H__

#pragma once
#include "matmul_formulaic_tiling.h"
#include "hccl_performance_arch35.h"
#include "matmul_performance_arch35.h"
#include "hccl_formulaic_tiling.h"

constexpr static double CALC_COMM_RATIO = 2.0;

class Mc2FitBasedBalanceTiling {
public:
    MatmulPerformanceArch35 matmulPerf_;
    HCCLPerformanceArch35 commPerf_;
    FormPartition tilingM_;
    MatmulParameters mmInfo_;
    uint64_t rankDim_ = 2;
    uint64_t rankTileNum_ = 1;
    uint64_t coreNum_ = 32;
    double ratioCalcComm_ = 1.0;

    explicit Mc2FitBasedBalanceTiling(const mc2tiling::TilingArgs &args, KernelType kernelType,
                                      TopoType topoType = TopoType::STANDARD_CARD,
                                      SocVersion socVersion = SocVersion::SOC950)
        : matmulPerf_(args, socVersion), commPerf_(args.rankDim, kernelType, socVersion, topoType), tilingM_(args)
    {
        rankDim_ = args.rankDim;
        mmInfo_ = matmulPerf_.mmShapeInfo_;
        tilingM_.SetMaxTileCnt(MAX_TILE_CNT); // Due to the limited FFTS queue size, a maximum of 16 rounds can be cut.
        rankTileNum_ = commPerf_.GetRankTileNum();
        coreNum_ = args.aicCoreNum;
    };

    virtual CutResult GetTiling();
    virtual void EstimateMMCommTime() {};
    virtual void SetShortTileLen() {};
    virtual void SetLongTileLen() {};
    virtual void AdjustLongShortTileLen() {};

    virtual ~Mc2FitBasedBalanceTiling()
    {
    }
};

#endif // __MC2_FIT_BALANCE_TILING_H__
