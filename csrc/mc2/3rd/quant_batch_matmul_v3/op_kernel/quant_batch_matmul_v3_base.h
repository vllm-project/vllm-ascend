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
 * \file quant_batch_matmul_v3_base.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_V3_BASE_H
#define MC2_QUANT_BATCH_MATMUL_V3_BASE_H

#include <cstdint>
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/quantization/ascend_dequant.h"
#include "kernel_type.h"
#include "lib/matmul_intf.h"
#include "quant_batch_matmul_v3_tiling_data.h"

#define TemplateBasicType                                                                                              \
    typename x1Type, typename x2Type, typename scaleType, typename yType, int x1Format, int x2Format, bool aTrans,     \
        bool bTrans, class UPDATE_TYPE
#define TemplateBasicValue x1Type, x2Type, scaleType, yType, x1Format, x2Format, aTrans, bTrans, UPDATE_TYPE

constexpr uint32_t BMM_BLOCK_NUM = 16;
constexpr uint32_t K0_INT8 = 32;
constexpr uint32_t k0_FLOAT16 = 16;
constexpr uint32_t k0_FLOAT32 = 8;
constexpr int FORMAT_FRACTAL_NZ_INT = 29;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t M_N_TWO_DIMS = 2;

const uint32_t ROW_FIRST = 1;
const uint32_t COL_FIRST = 2;

constexpr uint16_t C2V_PING_FLAG = 0x4;
constexpr uint16_t C2V_PONG_FLAG = 0x5;
constexpr uint16_t V2C_PING_FLAG = 0x6;
// asw const
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 8;
constexpr uint8_t AIC_SYNC_AIV_MODE = 4;
constexpr uint8_t CV_RATIO = 2;
constexpr int32_t MXFP_GROUP_SIZE = 32;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int32_t ONE_BLOCK_SIZE = 32;
constexpr uint16_t DATA_BLOCK = 32;
const uint32_t FP32_OUTPUT_TIMES = 4;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
constexpr MatmulConfig MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG =
    GetMDLConfig(false, false, 0, false, false, false, true, true, false, false, false, true);
constexpr MatmulConfig MM_DEFAULT_MDL_CFG =
    GetMDLConfig(false, false, 0, false, false, false, false, true, true, false, false, true);
#else
constexpr MatmulConfig MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG =
    GetMDLConfig(false, false, 0, false, false, false, true, true, false, false, false);
constexpr MatmulConfig MM_DEFAULT_MDL_CFG = GetMDLConfig();
#endif

struct L2CacheParam {
    uint32_t l2MCnt;
    uint32_t l2NCnt;
    uint32_t l2MCntTail;
    uint32_t l2NCntTail;
    uint32_t l2TotalTileCnt;
    uint32_t l2MCntUse;
    uint32_t l2NCntUse;
};

// 量化mm和mc2都用的输入输出地址结构体，不可随意修改
struct QBmmBlockOffset {
    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetScale = 0;
    uint64_t offsetBias = 0;
    uint64_t offsetPertoken = 0;
    uint64_t offsetC = 0;
};

// 量化mm和mc2都用的block输入信息结构体，不可随意修改
struct QBmmBaseBlockArgs {
    uint64_t index;
    uint64_t totalTileCnt;
    uint64_t singleCoreM; // 当前基本块计算大小
    uint64_t singleCoreN;
    uint64_t mTileCntL2;
    uint64_t nTileCntL2;
    uint64_t mTotalCnt;
    uint64_t nTotalCnt;
    uint64_t mCntUse;
    uint64_t nCntUse;
    uint64_t mTileAddrOffset;
    uint64_t nTileAddrOffset;
};

enum class BasicQuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
};

namespace DequantBmm {
template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b = 16)
{
    return (a + b - 1) / b * b;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t Align16(uint64_t num)
{
    return (num + 15UL) & 0xFFFFFFFFFFFFFFF0UL;
}

__aicore__ inline uint64_t Align32(uint64_t num)
{
    return (num + 31UL) & 0xFFFFFFFFFFFFFFE0UL;
}

__aicore__ inline uint64_t Align64(uint64_t num)
{
    return (num + 31UL) & 0xFFFFFFFFFFFFFFC0UL;
}

__aicore__ inline constexpr CubeFormat GetFormat(int format)
{
    if (format == FORMAT_FRACTAL_NZ_INT) {
        return CubeFormat::NZ;
    }
    return CubeFormat::ND;
}

template <typename T>
__aicore__ inline constexpr uint32_t GetC0Size()
{
    if constexpr (sizeof(T) == sizeof(float)) {
        return k0_FLOAT32;
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        return K0_INT8;
    } else {
        return k0_FLOAT16;
    }
}

template <typename T, bool trans>
__aicore__ inline uint64_t CalcAL1Size(uint64_t mL1, uint64_t kL1)
{
    uint64_t innerAlignedBlock = ONE_BLOCK_SIZE / sizeof(T);
    uint64_t mAligned = 0;
    uint64_t kAligned = 0;
    if constexpr (trans) {
        mAligned = Align(mL1, innerAlignedBlock);
        kAligned = Align(kL1, BMM_BLOCK_NUM);
    } else {
        mAligned = Align(mL1, BMM_BLOCK_NUM);
        kAligned = Align(kL1, innerAlignedBlock);
    }
    return mAligned * kAligned * sizeof(T);
}

/**
 * Get the aiv corenum in different platforms
 */
__aicore__ inline constexpr uint32_t GetTaskRation()
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    return 2U; // aiv corenum is 2 in C310 platform
#else
    return 1U;
#endif
}


#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
template <typename T>
__aicore__ inline constexpr bool IsMxType()
{
    return AscendC::IsSameType<T, AscendC::fp8_e8m0_t>::value;
}

template <typename T>
__aicore__ inline constexpr bool IsFp4()
{
    return (AscendC::IsSameType<T, fp4x2_e2m1_t>::value || AscendC::IsSameType<T, fp4x2_e1m2_t>::value);
}
#endif

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
__aicore__ inline void CalcDequantParams(uint32_t curAivM, uint32_t curAivN, AscendC::DequantParams &dequantParams,
                                         bool needUpdate = true)
{
    if (!needUpdate) {
        return;
    }
    uint32_t computedAivN = Align(curAivN, 8U); // 8: 32B aligned for int32_t
    uint32_t ubResAlignedN = Align(curAivN);    // 16: sizeof(yType) is 2, 32B / 2
    if (computedAivN == ubResAlignedN) {
        // choose ddequat high performance
        dequantParams.m = 1;
        dequantParams.n = curAivM * computedAivN;
        dequantParams.calCount = computedAivN;
    } else {
        // general
        dequantParams.m = curAivM;
        dequantParams.n = computedAivN;
        dequantParams.calCount = curAivN;
    }
}

__aicore__ inline void SetGm2UbParams(AscendC::DataCopyParams &gm2UbParams, uint32_t curAivM, uint32_t curAivN)
{
    gm2UbParams.blockLen = curAivN * sizeof(int32_t);
    gm2UbParams.blockCount = curAivM;
    gm2UbParams.srcStride = 0;
}

template <typename yType>
__aicore__ inline void SetUb2GmParams(AscendC::DataCopyExtParams &ub2GmParams, uint32_t curAivM, uint32_t curAivN,
                                      uint32_t n)
{
    ub2GmParams.blockLen = curAivN * sizeof(yType);
    ub2GmParams.blockCount = curAivM;
    ub2GmParams.dstStride = (n - curAivN) * sizeof(yType);
}

__aicore__ inline void CopyMmOutToLocal(AscendC::LocalTensor<int32_t> &srcLocal,
                                        AscendC::GlobalTensor<int32_t> &curMmOutGm,
                                        AscendC::DataCopyParams &gm2UbParams, AscendC::DataCopyPadParams &padParams,
                                        uint32_t curAicAivOffset)
{
    DataCopyPad(srcLocal, curMmOutGm[curAicAivOffset], gm2UbParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
}

template <typename yType>
__aicore__ inline void CopyUbToGm(uint64_t yGmOffset, AscendC::DataCopyExtParams &ub2GmParams,
                                  AscendC::LocalTensor<yType> &dstLocal, AscendC::GlobalTensor<yType> &yGm,
                                  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> &vecQueOut)
{
    DataCopyPad(yGm[yGmOffset], dstLocal, ub2GmParams);
    vecQueOut.FreeTensor(dstLocal);
}

template <typename scaleType>
__aicore__ inline void Bf16ScaleGm2Ub(AscendC::LocalTensor<scaleType> &scaleLocal,
                                      AscendC::GlobalTensor<scaleType> &scaleGm, AscendC::DataCopyPadParams &padParams,
                                      uint32_t curAivN, uint64_t offsetScale)
{
    AscendC::DataCopyParams scale2UbParams{1, 0, 0, 0};
    scale2UbParams.blockLen = curAivN * sizeof(scaleType);
    DataCopyPad(scaleLocal, scaleGm[offsetScale], scale2UbParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
}
#endif
} // namespace DequantBmm

#endif // QUANT_BATCH_MATMUL_V3_BASE_H
