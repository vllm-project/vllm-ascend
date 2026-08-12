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
 * \file matmul_performance.cpp
 * \brief
 */
#include <cmath>
#include <map>
#include <string>
#include "mc2_log.h"
#include "matmul_performance.h"

const static std::map<std::string, double> CUBE_CALC_PER_CYCLE_MAP = {
    {MatmulPerformance::DEFAULT_KEY_FOR_PAR_MAP, MatmulPerformance::COMPUTES_PER_CYCLE},
    {"1_1_1_1_2", 8192},
    {"0_1_1_1_2", 8192},
    {"3_1_1_1_2", 8192},
    {"4_1_1_1_2", 8192},
    {"4_1_1_1_1", 8192}, // socType_calcType_matrixADtype_matrixBDtypeSize_matrixCDtype
};

const static std::map<std::string, L2CacheEstimateParameters> L2_PARAMETER_MAP = {
    {MatmulPerformance::DEFAULT_KEY_FOR_PAR_MAP, L2CacheEstimateParameters{128, 192, 0.85, 0.75, 0.65}},
    {"3_1_1_1_2", L2CacheEstimateParameters{96, 96, 0.4, 0.4, 0.35}},
    {"4_0_2_2_2", L2CacheEstimateParameters{80, 128, 0.95, 0.85, 0.75}},
    // 需要确认量化场景，如何定义其mac利用率，1）量化时的利用率，2）allgather、reducescatter和allreduce的差异
    {"4_1_1_1_2", L2CacheEstimateParameters{80, 128, 0.95, 0.85, 0.75}},
};

void MatmulPerformanceModel::GetMachineParameters()
{
    mmShapeInfo_.computesPerCycle = MatmulPerformance::COMPUTES_PER_CYCLE;
    auto key = GetCalcTypeString();
    if (CUBE_CALC_PER_CYCLE_MAP.find(key) != CUBE_CALC_PER_CYCLE_MAP.end()) {
        OP_LOGW("common", "cube calculation power mapping HIT\n");
        mmShapeInfo_.computesPerCycle = CUBE_CALC_PER_CYCLE_MAP.at(key);
        return;
    }
}

uint64_t MatmulPerformanceModel::GetLinearThresholdLen(uint64_t rankTileNum)
{
    uint64_t tmpN = std::max(mmShapeInfo_.baseM, mmShapeInfo_.nValue);
    uint64_t tmpK = std::max(mmShapeInfo_.baseM, mmShapeInfo_.kValue);
    bool bmmFlag = (mmShapeInfo_.batchSize > 1 ? true : false);
    if (bmmFlag) {
        uint64_t mmMinDataSize = MatmulPerformance::BMM_DATASIZE_LARGE_K;
        if (mmShapeInfo_.kValue < MatmulPerformance::BMM_DATASIZE_K_BAR) {
            mmMinDataSize = MatmulPerformance::BMM_DATASIZE_SMALL_K;
        }
        return std::max(mmMinDataSize / (tmpN * tmpK * rankTileNum), mmShapeInfo_.baseM);
    }

    bool discreteFlag = rankTileNum > MatmulPerformance::SMALL_RANKTILE;
    if (discreteFlag) {
        return std::max(mmMinDataSize_ / (tmpN * tmpK * rankTileNum), mmShapeInfo_.baseM);
    }

    uint64_t thresholdSize = 0;
    MinMatmulShapeParameters minShapePar;
    bool adaptCoreNumFlag = (mmShapeInfo_.socType != SocVersion::SOC310_P);
    if (adaptCoreNumFlag) {
        minShapePar.mmMinDataSize1 =
            minShapePar.mmMinDataSize1 / MatmulPerformance::MARK_CORE_NUM_SOC910B * mmShapeInfo_.coreNum;
        minShapePar.mmMinDataSize2 =
            minShapePar.mmMinDataSize2 / MatmulPerformance::MARK_CORE_NUM_SOC910B * mmShapeInfo_.coreNum;
        minShapePar.minMSize = MatmulPerformance::MIN_M_SIZE_FACTOR * mmShapeInfo_.baseM;
    }
    thresholdSize = std::max(minShapePar.mmMinDataSize1 / (tmpN * tmpK),
                             minShapePar.mmMinDataSize2 / (tmpN * tmpK / ONE_KBYTE + tmpN));
    if (rankTileNum > 1) {
        thresholdSize /= rankTileNum;
    }
    thresholdSize = std::max(minShapePar.minMSize, thresholdSize);

    return thresholdSize;
}

void MatmulPerformanceModel::CheckKvalueAlignVersion310P()
{
    if (mmShapeInfo_.kValue % MatmulPerformance::K_ALIGN_LEN != 0) {
        cubeUtil_ *= MatmulPerformance::K_UNALIGN_UTIL_RATIO_SOC310P;
    }
}

double MatmulPerformanceModel::FindCubeUtilByL2Usage(uint64_t mSize, uint64_t rankTileNum, uint64_t *maxTileLen)
{
    L2CacheEstimateParameters l2EstimatePar = L2_PARAMETER_MAP.at(MatmulPerformance::DEFAULT_KEY_FOR_PAR_MAP);
    auto key = GetCalcTypeString();
    if (L2_PARAMETER_MAP.find(key) != L2_PARAMETER_MAP.end()) {
        OP_LOGW("Common", "l2 cache parameter mapping HIT\n");
        l2EstimatePar = L2_PARAMETER_MAP.at(key);
    }
    uint64_t tmpBaseM = std::min(mmShapeInfo_.baseM, mSize);
    uint64_t tmpBaseN = mmShapeInfo_.baseN;
    uint64_t mBlockNum = (mSize + tmpBaseM - 1) / tmpBaseM * rankTileNum;
    uint64_t nBlockNum = (mmShapeInfo_.nValue + tmpBaseN - 1) / tmpBaseN;
    uint64_t memUsage =
        std::min(mBlockNum, mmShapeInfo_.coreNum) * tmpBaseM + std::min(nBlockNum, mmShapeInfo_.coreNum) * tmpBaseN;
    memUsage = memUsage * mmShapeInfo_.kValue * mmShapeInfo_.inMatrixADtypeSize / ONE_MBYTE;
    double tmpUtil;
    double diff = 0;
    if (memUsage <= l2EstimatePar.sizePartL2) {
        tmpUtil = l2EstimatePar.utilPartL2; // 0.85x cube utilization rate
        diff = l2EstimatePar.sizePartL2 - memUsage;
    } else if (memUsage <= l2EstimatePar.sizeFullL2) {
        tmpUtil = l2EstimatePar.utilFullL2; // 0.75x cube utilization rate
        diff = l2EstimatePar.sizeFullL2 - memUsage;
    } else {
        tmpUtil = l2EstimatePar.utilOutL2; // 0.65x cube utilization rate
    }
    tmpUtil *= static_cast<double>(std::min(mBlockNum * nBlockNum, mmShapeInfo_.coreNum) /
                                   static_cast<double>(mmShapeInfo_.coreNum));

    /*950
    cube利用率计算公式：两个输入为M、N的调和平均数和K。MN调和平均数的阈值为820，K阈值为1280，小于这个阈值
    认为提供负增益，大于这个阈值认为提供正增益，但是正增益最高不超过1.4倍。乘方函数用来缓和增长率，减慢变化速度*/
    if (mmShapeInfo_.socType == SocVersion::SOC950) {
        double mnharmonicMean = static_cast<double>(mSize * rankTileNum * mmShapeInfo_.nValue) /
                                static_cast<double>(mSize * rankTileNum + mmShapeInfo_.nValue);
        double kExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(static_cast<double>(mmShapeInfo_.kValue), MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::KVALUE_THRESHOLD, MatmulPerformance::CUBE_UTIL_EXPONENT_COEFFICIENT));
        double mnExponentiated = std::min(
            MatmulPerformance::MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL,
            std::pow(mnharmonicMean, MatmulPerformance::FRONT_UTIL_EXPONENT_COEFFICIENT) /
                std::pow(MatmulPerformance::MNVALUE_THRESHOLD, MatmulPerformance::FRONT_UTIL_EXPONENT_COEFFICIENT));
        double result = MatmulPerformance::AVERAGE_CUBE_UTIL * kExponentiated * mnExponentiated;
        tmpUtil = std::min(MatmulPerformance::MAX_CUBE_UTIL, result);
    }

    if (maxTileLen != nullptr) {
        *maxTileLen = mmShapeInfo_.mValue;
        uint64_t tmpSize =
            std::min(mBlockNum + rankTileNum, mmShapeInfo_.coreNum) - std::min(mBlockNum, mmShapeInfo_.coreNum);
        tmpSize = tmpSize * mmShapeInfo_.inMatrixADtypeSize * tmpBaseM * mmShapeInfo_.kValue / ONE_MBYTE;
        if (tmpSize > diff && memUsage <= l2EstimatePar.sizeFullL2 && nBlockNum > mmShapeInfo_.coreNum) {
            *maxTileLen = (mSize + mmShapeInfo_.baseM - 1) / mmShapeInfo_.baseM * mmShapeInfo_.baseM;
        }
    }
    return tmpUtil;
}

double MatmulPerformanceModel::FindCubeUtilQuantVersion310P() const
{
    return 1.0 / (MatmulPerformance::UTIL_BY_K_DENOMINATOR_OFFSET_SOC310P +
                  MatmulPerformance::UTIL_BY_K_DENOMINATOR_GRADIENT_SOC310P * static_cast<double>(ONE_KBYTE) /
                      static_cast<double>(mmShapeInfo_.kValue));
}

double MatmulPerformanceModel::FindCubeUtilVersion310P() const
{
    return 1.0 / (MatmulPerformance::UTIL_BY_K_DENOMINATOR_OFFSET_SOC310P_NO_QUANT +
                  MatmulPerformance::UTIL_BY_K_DENOMINATOR_GRADIENT_SOC310P_NO_QUANT * static_cast<double>(ONE_KBYTE) /
                      static_cast<double>(mmShapeInfo_.kValue));
}

void MatmulPerformanceModel::ChangeCubeUtilByKAlign()
{
    uint64_t kLength = mmShapeInfo_.kValue * mmShapeInfo_.inMatrixADtypeSize;
    if (mmShapeInfo_.socType == SocVersion::SOC910_B4) {
        if (kLength % MatmulPerformance::HALF_CACHE_LINE_LEN != 0) {
            cubeUtil_ *= MatmulPerformance::K_UNALIGN_UTIL_RATIO_910_B4;
        }
        return;
    }

    if (kLength % MatmulPerformance::HALF_CACHE_LINE_LEN != 0) {
        cubeUtil_ *= MatmulPerformance::HALF_K_UNALIGN_UTIL_RATIO;
    } else if (kLength % MatmulPerformance::CACHE_LINE_LEN != 0) {
        cubeUtil_ *= MatmulPerformance::K_UNALIGN_UTIL_RATIO;
    }
}

void MatmulPerformanceModel::FindCubeUtil(uint64_t mSize, uint64_t rankTileNum, bool flagAllReduce,
                                          uint64_t *maxTileLen)
{
    if (mmShapeInfo_.socType == SocVersion::SOC310_P) {
        if (calcType_ == MatmulCalcType::QUANT) {
            cubeUtil_ = FindCubeUtilQuantVersion310P();
        } else {
            cubeUtil_ = FindCubeUtilVersion310P();
        }
        CheckKvalueAlignVersion310P();
        return;
    }

    cubeUtil_ = FindCubeUtilByL2Usage(mSize, rankTileNum, maxTileLen);

    if (mmShapeInfo_.socType == SocVersion::SOC950) {
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

    if (flagAllReduce) {
        if (calcType_ == MatmulCalcType::QUANT) { // 全量化
            return ChangeCubeUtilByKAlign();
        }
        if (mmShapeInfo_.kValue % MatmulPerformance::K_ALIGN_LEN != 0) { // 非量化和伪量化
            cubeUtil_ *= MatmulPerformance::K_UNALIGN_UTIL_RATIO;
        }
    }
}

void MatmulPerformanceModel::GetMatmulGradient()
{
    uint64_t tmpN = std::max(mmShapeInfo_.baseN, mmShapeInfo_.nValue);
    uint64_t tmpK = std::max(mmShapeInfo_.baseK, mmShapeInfo_.kValue);
    matmulGradient_ = (tmpN * tmpK) / (mmShapeInfo_.computesPerCycle * mmShapeInfo_.cyclePerMicroSec *
                                       mmShapeInfo_.coreNum * cubeUtil_);
}

double MatmulPerformanceModel::MatmulTime(uint64_t tileSize, uint64_t rankTileNum)
{
    tileSize = std::max(mmShapeInfo_.baseM, tileSize);
    return static_cast<double>(tileSize) * static_cast<double>(rankTileNum) * matmulGradient_;
}

uint64_t MatmulPerformanceModel::InverseMatmulTime(double targetTime, uint64_t rankTileNum)
{
    uint64_t tileSize = static_cast<uint64_t>(targetTime / (static_cast<double>(rankTileNum) * matmulGradient_));
    tileSize = std::max(mmShapeInfo_.baseM, tileSize);
    return tileSize;
}
