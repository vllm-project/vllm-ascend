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
 * \file matmul_performance.h
 * \brief
 */
#ifndef __MATMUL_PERFORMANCE_H__
#define __MATMUL_PERFORMANCE_H__
#pragma once
#include "matmul_formulaic_tiling.h"
#include "formulaic_tiling_datatype.h"
namespace MatmulPerformance {
constexpr double COMPUTES_PER_CYCLE = 4096;
constexpr double K_UNALIGN_UTIL_RATIO_SOC310P = 0.8;
constexpr double K_UNALIGN_UTIL_RATIO = 0.7;
constexpr double K_UNALIGN_UTIL_RATIO_910_B4 = 0.8;
constexpr uint64_t CACHE_LINE_LEN = 512;
constexpr double HALF_K_UNALIGN_UTIL_RATIO = 0.5;
constexpr uint64_t HALF_CACHE_LINE_LEN = 256;
constexpr uint64_t K_ALIGN_LEN = 128;
constexpr uint64_t MIN_M_SIZE_FACTOR = 3;
constexpr uint64_t MARK_CORE_NUM_SOC910B = 20;
// start: cycle per micro second for different soc version
constexpr double CYCLE_PER_MICRO_SEC = 1.8 * ONE_KBYTE;
constexpr double CYCLE_PER_MICRO_SEC_NPUARCH_3510 = 1.65 * ONE_KBYTE;
constexpr double CYCLE_PER_MICRO_SEC_VERSION310_P = 1.08 * ONE_KBYTE;
// end: cycle per micro second for different soc version
constexpr auto DEFAULT_KEY_FOR_PAR_MAP = "0_0_2_2_2";
constexpr double UTIL_BY_K_DENOMINATOR_OFFSET_SOC310P = 1.5;
constexpr double UTIL_BY_K_DENOMINATOR_GRADIENT_SOC310P = 1.25;
constexpr double UTIL_BY_K_DENOMINATOR_OFFSET_SOC310P_NO_QUANT = 1.5;
constexpr double UTIL_BY_K_DENOMINATOR_GRADIENT_SOC310P_NO_QUANT = 0.5;
constexpr uint64_t MIN_MSIZE_QUANT_SOC310P = 1024;
constexpr uint64_t INT8_DTYPE_SIZE = 1;
constexpr uint64_t SMALL_RANKTILE = 2;
constexpr uint64_t BMM_DATASIZE_K_BAR = 2048;
constexpr uint64_t BMM_DATASIZE_SMALL_K = 16 * ONE_GBYTE;
constexpr uint64_t BMM_DATASIZE_LARGE_K = 24 * ONE_GBYTE;
constexpr uint64_t MM_MIN_DATASIZE_NPUARCH_3510 = 5 * ONE_GBYTE;
constexpr uint64_t MM_MIN_DATASIZE_OTHER_SOC = 10 * ONE_GBYTE; // M * N * K >= 10G
constexpr double MAX_PARTICAL_ENHANCEMENT_FACTOR_CUBE_UTIL = 1.4;
constexpr double MNVALUE_THRESHOLD = 820;
constexpr double KVALUE_THRESHOLD = 1280;
constexpr double FRONT_UTIL_EXPONENT_COEFFICIENT = 0.6;
constexpr double CUBE_UTIL_EXPONENT_COEFFICIENT = 0.7;
constexpr double MAX_CUBE_UTIL = 0.95;
constexpr double AVERAGE_CUBE_UTIL = 0.75;
} // namespace MatmulPerformance

class MatmulPerformanceModel {
public:
    MatmulParameters mmShapeInfo_;
    MatmulCalcType calcType_; // Distinguish the non-quantization, full-quantization and fake-quantization
    double cubeUtil_ = 0.8;
    double matmulGradient_ = 1.0;
    uint64_t mmMinDataSize_ = MatmulPerformance::MM_MIN_DATASIZE_OTHER_SOC;

    void SetCyclePerMicroSec(SocVersion inputSocVersion)
    {
        mmShapeInfo_.cyclePerMicroSec = MatmulPerformance::CYCLE_PER_MICRO_SEC;
        if (inputSocVersion == SocVersion::SOC310_P) {
            mmShapeInfo_.cyclePerMicroSec = MatmulPerformance::CYCLE_PER_MICRO_SEC_VERSION310_P;
        } else if (inputSocVersion == SocVersion::SOC950) {
            mmShapeInfo_.cyclePerMicroSec = MatmulPerformance::CYCLE_PER_MICRO_SEC_NPUARCH_3510;
            mmMinDataSize_ = MatmulPerformance::MM_MIN_DATASIZE_NPUARCH_3510;
        } else {
            mmShapeInfo_.cyclePerMicroSec = MatmulPerformance::CYCLE_PER_MICRO_SEC;
        }
    }
    void SetCalcType(const mc2tiling::TilingArgs &args)
    {
        if ((mmShapeInfo_.socType == SocVersion::SOC950) &&
            ((args.aType == matmul_tiling::DataType::DT_HIFLOAT8) ||
             (args.aType == matmul_tiling::DataType::DT_FLOAT8_E4M3FN) ||
             (args.aType == matmul_tiling::DataType::DT_FLOAT8_E5M2))) {
            calcType_ = MatmulCalcType::QUANT;
        } else if ((args.aType == matmul_tiling::DataType::DT_INT8) &&
                   (mmShapeInfo_.socType != SocVersion::SOC910_B)) { // A8W8
            calcType_ = MatmulCalcType::QUANT;
        } else if ((mmShapeInfo_.socType == SocVersion::SOC310_P) &&
                   (args.bType == matmul_tiling::DataType::DT_INT8)) { // A16W8
            mmShapeInfo_.inMatrixBDtypeSize = MatmulPerformance::INT8_DTYPE_SIZE;
            calcType_ = MatmulCalcType::ANTI_QUANT;
        }
    }
    // Constructor
    explicit MatmulPerformanceModel(const mc2tiling::TilingArgs &args,
                                    SocVersion inputSocVersion = SocVersion::SOC910_B)
        : calcType_(MatmulCalcType::FP16)
    {
        mmShapeInfo_.socType = inputSocVersion;
        mmShapeInfo_.coreNum = args.aicCoreNum; // 每die核数
        mmShapeInfo_.inMatrixADtypeSize = args.inputDtypeSize;
        mmShapeInfo_.inMatrixBDtypeSize = mmShapeInfo_.inMatrixADtypeSize;
        mmShapeInfo_.outMatrixCDtypeSize = args.outputDtypeSize;
        mmShapeInfo_.mValue = args.mValue;
        mmShapeInfo_.nValue = args.nValue;
        mmShapeInfo_.kValue = args.kValue;
        // 获取后续计算用的基本块
        mmShapeInfo_.baseM = mc2tiling::BASE_BLOCK_M;
        mmShapeInfo_.baseN = mc2tiling::BASE_BLOCK_N;
        mmShapeInfo_.baseK = mc2tiling::BASE_BLOCK_K;
        mmShapeInfo_.batchSize = 1UL; // 初始值
        SetCalcType(args);
        SetCyclePerMicroSec(inputSocVersion);
        GetMachineParameters();
    };

    std::string GetCalcTypeString()
    {
        return std::to_string(static_cast<int>(mmShapeInfo_.socType)) + "_" +
               std::to_string(static_cast<int>(calcType_)) + "_" + std::to_string(mmShapeInfo_.inMatrixADtypeSize) +
               "_" + std::to_string(mmShapeInfo_.inMatrixBDtypeSize) + "_" +
               std::to_string(mmShapeInfo_.outMatrixCDtypeSize);
    }
    // 根据输入shape，预测cube利用率（取值0 < cubeUtil_ < 1），再用利用率预测耗时
    // 耗时 = M * k * N / 频率 / 每cycle多少计算  /核数 / 利用率
    void GetMachineParameters();
    void CheckKvalueAlignVersion310P();
    void FindCubeUtil(uint64_t mSize, uint64_t rankTileNum, bool flagAllReduce, uint64_t *maxTileLen = nullptr);
    double FindCubeUtilByL2Usage(uint64_t mSize, uint64_t rankTileNum, uint64_t *maxTileLen = nullptr);
    double FindCubeUtilQuantVersion310P() const;
    double FindCubeUtilVersion310P() const;
    void ChangeCubeUtilByKAlign();
    void GetMatmulGradient();
    void SetBatchSize(uint64_t bSize)
    {
        mmShapeInfo_.batchSize = bSize;
    };
    uint32_t GetBaseM()
    {
        return mmShapeInfo_.baseM;
    };
    uint32_t GetBaseN()
    {
        return mmShapeInfo_.baseN;
    };
    // 返回允许切分的最小数据量
    uint64_t GetLinearThresholdLen(uint64_t rankTileNum);
    // 性能预测
    double MatmulTime(uint64_t tileSize, uint64_t rankTileNum);
    uint64_t InverseMatmulTime(double targetTime, uint64_t rankTileNum);
};

#endif // __MATMUL_PERFORMANCE_H__
