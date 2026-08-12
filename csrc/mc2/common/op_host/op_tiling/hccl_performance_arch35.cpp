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
 * \file hccl_performance_arch35.cpp
 * \brief
 */
#include "hccl_performance_arch35.h"
const static string DEFAULT_KEY_FOR_ARCH35 = "0_0_0_0";

const static map<string, HCCLFittingParameters> ARCH35_FITTING_PARAMETER_MAP = {
    {DEFAULT_KEY_FOR_ARCH35,
     HCCLFittingParameters{0.032, 1.371, 2.59568, 5.8887, 0, 2.86702, 5.608, 5.6997, 9.448, 0, 0, 0}},
    {"4_1_0_4", HCCLFittingParameters{0, 0, 5.841842, 8.590666, 0, 0, // 950 allGather standard_card 4p
                                      0, 0, 0, 0, 0, 0}},
    {"4_4_0_4", HCCLFittingParameters{0, 0, 5.8087, 5.6810, 0, 0, // 950 reduceScatter(alltoAll) standard_card 4p
                                      0, 0, 0, 0, 0, 0}},
    {"4_5_0_4", HCCLFittingParameters{0, 0, 13.3265, 36.1964, 0, 0, // 950 allreduce(two shot) standard_card 4p
                                      0, 0, 0, 0, 0, 0}},
    {"4_4_1_8", HCCLFittingParameters{0, 0, 2.642023, 7.700559, 0, 0, // 950 RS(A2A) POD/Server 8p
                                      0, 0, 0, 0, 0, 0}},
    {"4_3_0_4", HCCLFittingParameters{0, 0, 5.8416, 5.8148, 0, 0, // 950 alltoall standard_card 4p
                                      0, 0, 0, 0, 0, 0}},
    {"4_3_1_8", HCCLFittingParameters{0, 0, 2.5386, 7.6265, 0, 0, // 950 alltoall pod 8p
                                      0, 0, 0, 0, 0, 0}},
    {"4_5_1_8", HCCLFittingParameters{0, 0, 6.1588, 34.8712, 0, 0, // 950 allreduce(two shot) pod 8p
                                      0, 0, 0, 0, 0, 0}},
    {"4_0_1_8", HCCLFittingParameters{0, 0, 9.2419, 22, 0, 0, 0, 0, 0, 0, 0, 0}}, // 950 allreduce 8p
};

string HCCLPerformanceArch35::GetCommInfoString(SocVersion socVersion)
{
    return to_string(static_cast<int>(socVersion)) + "_" + to_string(static_cast<int>(commTypeInfo_.kernelType)) + "_" +
           to_string(static_cast<int>(commTypeInfo_.topoType)) + "_" +
           to_string(static_cast<int>(commTypeInfo_.rankDim));
};

void HCCLPerformanceArch35::GetCommArch35Parameters(SocVersion socVersion)
{
    keyToFittingMap_ = GetCommInfoString(socVersion);
    if (ARCH35_FITTING_PARAMETER_MAP.find(keyToFittingMap_) != ARCH35_FITTING_PARAMETER_MAP.end()) {
        commEstimatePar_ = ARCH35_FITTING_PARAMETER_MAP.at(keyToFittingMap_);
    } else {
        commEstimatePar_ = ARCH35_FITTING_PARAMETER_MAP.at(DEFAULT_KEY_FOR_ARCH35);
        ChangeCommTimeFactorByDivision(static_cast<double>(FITTING_RANK) / static_cast<double>(commTypeInfo_.rankDim));
    }
}

void HCCLPerformanceArch35::InitArch35Parameters()
{
    if ((commTypeInfo_.kernelType == KernelType::ALL_GATHER) ||
        (commTypeInfo_.kernelType == KernelType::REDUCE_SCATTER_VIA_ALL_TO_ALL)) {
        lookUpTileNum_ = commTypeInfo_.rankDim;
        commTimeFactor_ = 1;
    }
}

uint64_t HCCLPerformanceArch35::InverseCommTime(double targetTime) const
{
    double tmpSize = commEstimatePar_.sizeToTimeBoundary1;
    if (targetTime > commEstimatePar_.timeToSizeBoundary2) {
        tmpSize = (targetTime - commEstimatePar_.sizeToTimeLinearOffset) / commEstimatePar_.sizeToTimeLinearGradient;
    } else if (targetTime > commEstimatePar_.timeToSizeBoundary1) {
        if (commTypeInfo_.kernelType == KernelType::ALL_REDUCE && commEstimatePar_.timeToSizeParabolicPar2 != 0.0f) {
            tmpSize =
                (targetTime - commEstimatePar_.timeToSizeParabolicPar3) / commEstimatePar_.timeToSizeParabolicPar2;
        } else {
            tmpSize = (-sqrt(commEstimatePar_.timeToSizeParabolicPar1 *
                             (targetTime + commEstimatePar_.timeToSizeParabolicPar2)) +
                       commEstimatePar_.timeToSizeParabolicPar3);
            if (commEstimatePar_.timeToSizeParabolicPar3 < 0) {
                tmpSize = sqrt(commEstimatePar_.timeToSizeParabolicPar1 *
                               (targetTime + commEstimatePar_.timeToSizeParabolicPar2)) +
                          commEstimatePar_.timeToSizeParabolicPar3;
            }
        }
    }
    return static_cast<uint64_t>(tmpSize * ONE_MBYTE /
                                 (commTypeInfo_.commMatrixLen * lookUpTileNum_ * GetRealDtypeSizes()));
}
