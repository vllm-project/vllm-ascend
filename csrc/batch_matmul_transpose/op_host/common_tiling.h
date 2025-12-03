/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMMON_TILING_H
#define COMMMON_TILING_H

#include <iostream>
#include <cmath>
#include "common.h"
#include "tiling/platform/platform_ascendc.h"

namespace host_utils {

constexpr uint32_t FP16_SIZE = 2;
constexpr uint32_t FP32_SIZE = 4;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t BLOCK_SIZE_INT8_K = 32;
constexpr uint32_t BASE_BLOCK_STEP = 2;
constexpr uint32_t AXES_ALIGN_SIZE = 512;
constexpr uint32_t AXES_ALIGN_SIZE_INT8 = 256;
constexpr uint32_t ND_SHAPE_SIZE = 2;
constexpr uint32_t NZ_SHAPE_SIZE = 4;
constexpr uint32_t CUBE_BLOCK_SIZE = 256;
constexpr uint32_t CUBE_BLOCK_SIZE_INT8 = 512;
constexpr uint32_t L1AB_PINGPONG_BUFFER_LEN = 262144;
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN_INT8 = 131072 * 2;  // 256 KB
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN_FP16 = 131072;      // 128 KB
constexpr uint32_t L1AB_PINGPONG_BUFFER_LEN_INT8_SPARSE = 160 * 1024;
constexpr uint32_t UB_LIMIT_SIZE_910A = 128 * 1024;

enum class PlatformType { ASCEND_310P, ASCEND_910A, ASCEND_910B, ASCEND_910C, PLATFORM_INVALID };

struct PlatformInfo {
public:
    static const PlatformInfo &Instance()
    {
        static PlatformInfo platformInfo;
        return platformInfo;
    }

    PlatformType socType;
    uint32_t coreNum;
    uint32_t coreNumAic;
    uint32_t coreNumAiv;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l2Size;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint64_t l0cSize;

private:
    PlatformInfo()
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
        // TODO Hard coding set to 910_93xx, parse using aclrtGetSocName is better
        socType = PlatformType::ASCEND_910C;
        coreNum = ascendcPlatform->GetCoreNum();
        coreNumAic = ascendcPlatform->GetCoreNumAic();
        coreNumAiv = ascendcPlatform->GetCoreNumAiv();
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0aSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize);
        ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    }

    PlatformInfo(const PlatformInfo &) = delete;
    PlatformInfo &operator=(const PlatformInfo &) = delete;
    PlatformInfo(PlatformInfo &&) = delete;
    PlatformInfo &operator=(PlatformInfo &&) = delete;
};

inline __attribute__((always_inline)) uint32_t GetN0TilingLimit(bool compressFlag, uint32_t tilingN,
                                                                const PlatformType &platformType)
{
    if (compressFlag) {
        return std::min(tilingN * BLOCK_SIZE, AXES_ALIGN_SIZE_INT8);
    } else {
        return (platformType == PlatformType::ASCEND_310P || platformType == PlatformType::ASCEND_910A)
                   ? AXES_ALIGN_SIZE
                   : AXES_ALIGN_SIZE_INT8;
    }
}

template <typename OpShareType>
inline __attribute__((always_inline)) uint32_t GetN0TilingInit(const OpShareType &opShape, bool compressFlag,
                                                               uint32_t tilingN)
{
    const uint32_t rnd = 16;
    return compressFlag
               ? ((tilingN * BLOCK_SIZE > opShape.n) ? RoundUp<uint32_t>(opShape.n, rnd) : tilingN * BLOCK_SIZE)
               : BLOCK_SIZE;
}

template <bool PRI_FLAG>
inline __attribute__((always_inline)) bool IsExceedTilingLimit(uint32_t axes0, uint32_t priAxes0,
                                                               uint32_t n0TilingLimit, PlatformType platformType,
                                                               uint32_t basicBlockSize)
{
    return (PRI_FLAG && axes0 > n0TilingLimit) || (!PRI_FLAG && priAxes0 > n0TilingLimit) ||
           (platformType == PlatformType::ASCEND_910A && basicBlockSize > UB_LIMIT_SIZE_910A);
}

template <bool PRI_FLAG, typename OpShareType>
inline __attribute__((always_inline)) void SetOpShapeAxesInfo(OpShareType &opShape, uint32_t priAxes0, uint32_t axes0)
{
    opShape.m0 = PRI_FLAG ? priAxes0 : axes0;
    opShape.n0 = PRI_FLAG ? axes0 : priAxes0;
}

template <typename HardwareType, typename OpShapeType>
inline __attribute__((always_inline)) float CostFunc(const HardwareType &hwInfor, OpShapeType &shape)
{
    float aCoef = 1;
    float bCoef = 1;
    float bwCoef = static_cast<float>(hwInfor.l2BandWidth) / static_cast<float>(hwInfor.hbmBandWidth);
    uint32_t mLoop = CeilDiv(shape.m, shape.m0);
    uint32_t nLoop = CeilDiv(shape.n, shape.n0);
    if (mLoop == 0 || nLoop == 0) {
        return 1;
    }
    uint32_t coreNeed = shape.batchSize * mLoop * nLoop;
    uint32_t blockDim = std::min(coreNeed, hwInfor.coreNum);
    uint32_t mOnce = blockDim < nLoop ? shape.m0 : blockDim / nLoop * shape.m0;
    uint32_t nOnce = blockDim < nLoop ? hwInfor.coreNum * shape.n0 : shape.n;
    if (mOnce * shape.k * FP16_SIZE > hwInfor.l2Size) {
        aCoef = bwCoef;
    }
    if (nOnce * shape.k * FP16_SIZE > hwInfor.l2Size) {
        bCoef = bwCoef;
    }
    return 1 / (aCoef * static_cast<float>(shape.n0)) + 1 / (bCoef * static_cast<float>(shape.m0));
}

template <bool PRI_FLAG, typename OpShareType, typename TilingType, typename HardwareType, typename MatMulInfoType>
void TilingFunc(OpShareType &opShape, TilingType &tilingParam, const HardwareType &hwInfor,
                const MatMulInfoType &mmInfo, bool compressFlag = false, const uint32_t tilingN = 1)
{
    float costMin = 1;
    const float CONST_2 = 2.0;
    const uint32_t ROUND_CONST_16 = 16;
    uint32_t roundBase = static_cast<uint32_t>(
        pow(2, ceil(log(CeilDiv(PRI_FLAG ? opShape.n : opShape.m, ROUND_CONST_16)))) * ROUND_CONST_16);
    uint32_t priAxes = RoundUp<uint32_t>(PRI_FLAG ? opShape.m : opShape.n, ROUND_CONST_16);
    uint32_t axes = RoundUp<uint32_t>(PRI_FLAG ? opShape.n : opShape.m, roundBase);
    float axes0Max = static_cast<float>(AXES_ALIGN_SIZE) / mmInfo.inDtype;
    auto platformType = PlatformInfo::Instance().socType;
    if (mmInfo.isInt8 && (platformType == PlatformType::ASCEND_310P || platformType == PlatformType::ASCEND_910A)) {
        axes0Max /= CONST_2;
    }

    uint32_t n0TilingInit = GetN0TilingInit(opShape, compressFlag, tilingN);
    uint32_t n0TilingLimit = GetN0TilingLimit(compressFlag, tilingN, platformType);
    uint32_t priAxes0Init = PRI_FLAG ? BLOCK_SIZE : n0TilingInit;
    uint32_t axes0Init = PRI_FLAG ? n0TilingInit : BLOCK_SIZE;
    for (uint32_t priAxes0 = priAxes0Init; priAxes0 <= priAxes && priAxes0 <= axes0Max; priAxes0 *= BASE_BLOCK_STEP) {
        for (uint32_t axes0 = axes0Init; axes0 <= axes && axes0 <= axes0Max; axes0 *= BASE_BLOCK_STEP) {
            uint32_t basicBlockSize = priAxes0 * axes0 * FP32_SIZE;
            if (basicBlockSize > hwInfor.l0cSize) {
                continue;
            }
            if (mmInfo.isInt8 &&
                IsExceedTilingLimit<PRI_FLAG>(axes0, priAxes0, n0TilingLimit, platformType, basicBlockSize)) {
                continue;
            }
            SetOpShapeAxesInfo<PRI_FLAG>(opShape, priAxes0, axes0);
            float cost = CostFunc<HardwareType, OpShareType>(hwInfor, opShape);
            if (cost >= costMin) {
                continue;
            }
            costMin = cost;
            if constexpr (std::is_same<TilingType, pp_matmul::PpMatmulTilingData>::value) {
                tilingParam.SetBaseOp(hwInfor.coreNum, opShape.m0, opShape.n0, mmInfo);
            } else {
                tilingParam.SetBaseOp(hwInfor.coreNum, opShape.m0, opShape.n0);
            }
        }
    }
}

template <typename PpTilingDataType>
uint32_t Swizzl(PpTilingDataType &tilingData)
{
    uint32_t swizzlDirect = 0;
    uint32_t swizzlCount = 1;
    float m0 = tilingData.opShape.m0;
    float n0 = tilingData.opShape.n0;
    float m = tilingData.opShape.m;
    float k = tilingData.opShape.k;
    float n = tilingData.opShape.n;
    float mincost = m * k + k * n;

    for (uint32_t i = 1; i <= tilingData.blockDim; ++i) {
        int c = static_cast<int32_t>((tilingData.blockDim + i - 1) / i);
        float cost;
        // B0 + A < A0 + B
        if (i * n0 + m < m0 * c + n) {
            swizzlDirect = 1;  // Nz
            cost = n0 * i + m0 * c;
            if (cost <= mincost) {
                mincost = cost;
                swizzlCount = i;
            }
        } else {
            swizzlDirect = 0;  // Zn
            cost = m0 * i + n0 * c;
            if (cost < mincost) {
                mincost = cost;
                swizzlCount = i;
            }
        }
    }
    tilingData.swizzlDirect = swizzlDirect;
    tilingData.swizzlCount = swizzlCount;
    return swizzlDirect;
}

template <typename PpTilingDataType>
inline __attribute__((always_inline)) void PpMatmulTilingCheck(const PpTilingDataType &tilingData)
{
    TORCH_CHECK(tilingData.opShape.m0 > 0, "m0 is invalid");
    TORCH_CHECK(tilingData.opShape.k0 > 0, "k0 is invalid");
    TORCH_CHECK(tilingData.opShape.n0 > 0, "n0 is invalid");
    TORCH_CHECK(tilingData.mLoop > 0, "mLoop is invalid");
    TORCH_CHECK(tilingData.kLoop > 0, "kLoop is invalid");
    TORCH_CHECK(tilingData.nLoop > 0, "nLoop is invalid");
    TORCH_CHECK(tilingData.blockDim > 0, "nLoop is invalid");
}
}  // namespace host_utils
#endif
