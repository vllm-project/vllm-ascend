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
 * \file matmul_performance_arch35.h
 * \brief
 */
#ifndef __MATMUL_PERFORMANCE_ARCH35_H__
#define __MATMUL_PERFORMANCE_ARCH35_H__
#pragma once
#include "matmul_performance.h"

class MatmulPerformanceArch35 : public MatmulPerformanceModel {
public:
    explicit MatmulPerformanceArch35(const mc2tiling::TilingArgs &args, SocVersion inputSocVersion = SocVersion::SOC950)
        : MatmulPerformanceModel(args, inputSocVersion)
    {
        TilingBestBaseBlock bestBaseBlock = GetBestBaseBlock(inputSocVersion);
        mmShapeInfo_.baseM = bestBaseBlock.baseM;
        mmShapeInfo_.baseN = bestBaseBlock.baseN;
        mmShapeInfo_.baseK = bestBaseBlock.baseK;
    }

    const std::map<SocVersion, TilingBestBaseBlock> TILING_BEST_BASE_MAP{
        {SocVersion::SOC950, TilingBestBaseBlock{256, 256, 128}},
    };

    TilingBestBaseBlock GetBestBaseBlock(SocVersion SocVersion)
    {
        if (TILING_BEST_BASE_MAP.find(SocVersion) != TILING_BEST_BASE_MAP.end()) {
            return TILING_BEST_BASE_MAP.at(SocVersion);
        }
        return TilingBestBaseBlock{mc2tiling::BASE_BLOCK_M, mc2tiling::BASE_BLOCK_N, mc2tiling::BASE_BLOCK_K};
    }

    void FindCubeUtil(uint64_t rankTileNum)
    {
        double mnharmonicMean = static_cast<double>(mmShapeInfo_.mValue * rankTileNum * mmShapeInfo_.nValue) /
                                static_cast<double>(mmShapeInfo_.mValue * rankTileNum + mmShapeInfo_.nValue);
        double kExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(static_cast<double>(mmShapeInfo_.kValue), MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::KVALUE_THRESHOLD, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double mnExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(mnharmonicMean, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::MNVALUE_THRESHOLD, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double result = MatmulPerformance::AVERAGE_CUBE_UTIL * kExponentiated * mnExponentiated;
        cubeUtil_ = std::min(MatmulPerformance::MAX_CUBE_UTIL, result);
    }
};

#endif // __MATMUL_PERFORMANCE_ARCH35_H__
