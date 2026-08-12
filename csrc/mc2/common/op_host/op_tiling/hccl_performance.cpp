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
 * \file hccl_performance.cpp
 * \brief
 */
#include <cmath>
#include <map>
#include <string>

#include "mc2_log.h"
#include "hccl_performance.h"

// {"socType_commMethod",
// {sizeToTimeBoundary1, sizeToTimeBoundary2,
// sizeToTimeLinearGradient, sizeToTimeLinearOffset,
// sizeToTimeParabolicPar1, sizeToTimeParabolicPar2, sizeToTimeParabolicPar3,
// timeToSizeBoundary1, timeToSizeBoundary2,
// timeToSizeParabolicPar1, timeToSizeParabolicPar2, timeToSizeParabolicPar3}},
const static std::map<std::string, HCCLFittingParameters> FITTING_PARAMETER_MAP = {
    {DEFAULT_KEY_FOR_FITTING_MAP,
     HCCLFittingParameters{64.0 / ONE_KBYTE, 8, 13.58491263, 61.508333, -0.9698202, 27.0622573, 14.769, 18, 170,
                           -1.03111896, -203.558059, 13.9522034}},
    {"1_3",
     HCCLFittingParameters{64.0 / ONE_KBYTE, 64.0 / ONE_KBYTE, 72.04417474, 6.693209, 0, 0, 0, 10.0, 10.0, 0, 0, 0}},
    {"1_4", HCCLFittingParameters{32.0 / ONE_KBYTE, 1.8, 153.555534, -48.264442, -89.115824, 268.473025, 34.564670,
                                  42.8674248, 229.080845, -0.011221, -236.767157, 1.506315}},
    {"2_1", HCCLFittingParameters{2.0, 2.0, 11.38154641, 248.151561, 0.0, 0.0, 250.0, 250.0, 250.0, 0.0, 0.0, 2.0}},
    {"4_0", HCCLFittingParameters{0.032, 1.371, 2.59568, 5.8887, 0, 2.86702, 5.608, 5.6997, 9.448, 0, 0, 0}},
};

void HCCLPerformanceModel::SetCommMethodVersion310P()
{
    if (commTypeInfo_.rankDim == RANK_FOUR) {
        commTypeInfo_.commMethod = HCCLType::RING_RANK4_310P;
        return;
    }
    commTypeInfo_.commMethod = HCCLType::RING_RANK2_310P;
}

void HCCLPerformanceModel::ChangeCommTimeFactorByDivision(double factor)
{
    if (factor > 0) {
        commTimeFactor_ /= factor;
    }
}

void HCCLPerformanceModel::SetFullMeshCommTimeFactor()
{
    // 通信查表使用总数据量，实际时间 = 总数据量 / rankDim / 带宽
    // 拟合采用8 die通信
    commTimeFactor_ = FULL_MESH_TIME_FACTOR;
    ChangeCommTimeFactorByDivision(static_cast<double>(FITTING_RANK) / static_cast<double>(commTypeInfo_.rankDim));
}

void HCCLPerformanceModel::SetRingCommTimeFactor()
{
    // 通信查表使用总数据量，实际时间 = 总数据量 * 2 * (rankDim - 1) / rankDim /
    // 带宽 拟合采用8 die通信
    commTimeFactor_ = FULL_MESH_TIME_FACTOR;
    double fittingRatio = (static_cast<double>(FITTING_RANK) - 1.0) / static_cast<double>(FITTING_RANK);
    double currentRatio =
        (static_cast<double>(commTypeInfo_.rankDim) - 1.0) / static_cast<double>(commTypeInfo_.rankDim);
    ChangeCommTimeFactorByDivision(currentRatio / fittingRatio);
}

void HCCLPerformanceModel::InitParametersForFullMesh()
{
    commTypeInfo_.commMethod = HCCLType::FULL_MESH;
    if ((commTypeInfo_.kernelType == KernelType::ALL_GATHER) ||
        (commTypeInfo_.kernelType == KernelType::REDUCE_SCATTER)) {
        SetFullMeshCommTimeFactor();
        lookUpTileNum_ = commTypeInfo_.rankDim;
    }
}

void HCCLPerformanceModel::InitSOC91093()
{
    commTypeInfo_.commMethod = HCCLType::DOUBLE_RING;
    SetRingCommTimeFactor();
    lookUpTileNum_ = commTypeInfo_.rankDim;
}

void HCCLPerformanceModel::SetLocalReduceFactor()
{
    // 拟合采用8 die通信
    // die的数量越多，local累加的步骤就越多
    double changeFactor =
        1.0 + LOCAL_REDUCE_FACTOR * static_cast<double>(commTypeInfo_.rankDim) / static_cast<double>(FITTING_RANK);
    ChangeCommTimeFactorByDivision(changeFactor);
}

uint64_t HCCLPerformanceModel::GetFullMeshRankTileNum()
{
    if (commTypeInfo_.kernelType == KernelType::ALL_GATHER) {
        return commTypeInfo_.rankDim - 1;
    }
    return commTypeInfo_.rankDim;
}

uint64_t HCCLPerformanceModel::GetRankTileNum()
{
    if (commTypeInfo_.kernelType != KernelType::ALL_REDUCE) {
        return GetFullMeshRankTileNum();
    }
    return 1;
}

void HCCLPerformanceModel::SetMaxStepSize()
{
    if (commTypeInfo_.commMethod == HCCLType::DOUBLE_RING) {
        if (commTypeInfo_.rankDim == MIN_COMM_RANKDIM) { // rankDim = 2 uses full-mesh
            commTypeInfo_.maxStepSize = 1UL;
            return;
        }
        if (commTypeInfo_.kernelType == KernelType::REDUCE_SCATTER) {
            commTypeInfo_.maxStepSize = commTypeInfo_.rankDim;
            return;
        }
        if (commTypeInfo_.kernelType == KernelType::ALL_GATHER) {
            commTypeInfo_.maxStepSize = commTypeInfo_.rankDim - 1;
            return;
        }
    }
    commTypeInfo_.maxStepSize = 1;
}

uint64_t HCCLPerformanceModel::GetMaxStepSize()
{
    return commTypeInfo_.maxStepSize;
}

uint64_t HCCLPerformanceModel::GetLinearThresholdLen()
{
    uint64_t resultLen =
        static_cast<uint64_t>(HCCL_MIN_TILE_LEN / (commTypeInfo_.commMatrixLen * lookUpTileNum_ * GetRealDtypeSizes()));
    return resultLen;
}

uint64_t HCCLPerformanceModel::GetLinearThresholdLenCoarse()
{
    uint64_t resultLen = static_cast<uint64_t>(
        HCCL_MIN_TILE_LEN_COARSE / (commTypeInfo_.commMatrixLen * commTypeInfo_.rankDim * GetRealDtypeSizes()));
    return resultLen;
}

void HCCLPerformanceModel::GetCommEstimateParameters()
{
    commEstimatePar_ = FITTING_PARAMETER_MAP.at(DEFAULT_KEY_FOR_FITTING_MAP);
    if (FITTING_PARAMETER_MAP.find(keyToFittingMap_) != FITTING_PARAMETER_MAP.end()) {
        OP_LOGW("Common", "comm fitting parameter mapping HIT\n");
        commEstimatePar_ = FITTING_PARAMETER_MAP.at(keyToFittingMap_);
        return;
    }
}

double HCCLPerformanceModel::CommTime(uint64_t mSize) const
{
    double commDataSize = mSize * commTypeInfo_.commMatrixLen * lookUpTileNum_ * GetRealDtypeSizes();
    double tmpSize = commDataSize / ONE_MBYTE;
    double result = commEstimatePar_.timeToSizeBoundary1;
    if (tmpSize > commEstimatePar_.sizeToTimeBoundary2) {
        result = commEstimatePar_.sizeToTimeLinearGradient * tmpSize + commEstimatePar_.sizeToTimeLinearOffset;
    } else if (tmpSize > commEstimatePar_.sizeToTimeBoundary1) {
        result = commEstimatePar_.sizeToTimeParabolicPar1 * tmpSize * tmpSize +
                 commEstimatePar_.sizeToTimeParabolicPar2 * tmpSize + commEstimatePar_.sizeToTimeParabolicPar3;
    }

    result /= commTimeFactor_;
    return result;
}

uint64_t HCCLPerformanceModel::InverseCommTime(double targetTime) const
{
    targetTime *= commTimeFactor_;
    double tmpSize = commEstimatePar_.sizeToTimeBoundary1;
    if (targetTime > commEstimatePar_.timeToSizeBoundary2) {
        tmpSize = (targetTime - commEstimatePar_.sizeToTimeLinearOffset) / commEstimatePar_.sizeToTimeLinearGradient;
    } else if (targetTime > commEstimatePar_.timeToSizeBoundary1) {
        if (socVersion_ == SocVersion::SOC950) {
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
