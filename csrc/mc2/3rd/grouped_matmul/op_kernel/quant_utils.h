/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file quant_utils.h
 * \brief
 */
#ifndef MC2_QUANT_UTILS_H
#define MC2_QUANT_UTILS_H

#include "kernel_basic_intf.h"
#include "grouped_matmul_utils.h"
#include "lib/matmul_intf.h"

#define LOCAL_TEMPLATE_CLASS_PARAMS                                                                                    \
    template <class xType, class wType, class biasType, class scaleType, class yType, CubeFormat wFormat, bool aTrans, \
              bool bTrans>
#define LOCAL_TEMPLATE_FUNC_PARAMS xType, wType, biasType, scaleType, yType, wFormat, aTrans, bTrans

namespace Mc2QuantUtils {
constexpr uint32_t PER_BLOCK_SIZE = 128;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr uint32_t MAX_REPEAT_TIMES = 255;
constexpr uint32_t UB_ALIGN_SIZE = 32;
const uint32_t FP32_OUTPUT_TIMES = 4;
constexpr uint8_t BUFFER_SWITCH = 2;
constexpr uint8_t SPLIT_M = 0;
constexpr uint8_t SPLIT_K = 2;
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t INNER_AXIS_MIN_SPLIT_VAL = 128; // ND2NZ cacheline 128

constexpr uint32_t WEIGHTNZ_K0_16 = 16;
constexpr uint32_t WEIGHTNZ_N0_16 = 16;
constexpr uint32_t WEIGHTNZ_K0_32 = 32;
constexpr uint32_t WEIGHTNZ_N0_32 = 32;
constexpr uint32_t WEIGHTNZ_N0_K0 = 512;

constexpr float NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
constexpr float TANH_APPROX_FACTOR = 1 / 0.044715;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, true};
#endif

constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr MatmulConfig MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG = GetMDLConfig(false, false, 0, false, false, false, true);

enum class QuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERGROUP_MODE = 0x1U << 4,
    PERBLOCK_MODE = 0x1U << 5,
};

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

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b)
{
    return CeilDiv(a, b) * b;
}

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

template <typename aType, typename biasType>
__aicore__ inline constexpr bool IsBiasEpilogue()
{
    return AscendC::IsSameType<aType, int8_t>::value &&
           (AscendC::IsSameType<biasType, bfloat16_t>::value || AscendC::IsSameType<biasType, half>::value ||
            AscendC::IsSameType<biasType, float>::value);
}

/* *
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

/* *
 * Get the aiv corenum in different platforms
 */
__aicore__ inline constexpr uint32_t GetTaskRation()
{
#if __CCE_AICORE__ == 310
    return 2U; // aiv corenum is 2 in C310 platform
#else
    return 1U;
#endif
}

__aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, int32_t &preOffset, int32_t groupType,
                                                     uint32_t groupListType,
                                                     const AscendC::GlobalTensor<int64_t> &groupListGm)
{
    int32_t splitValue = 0;
    if (likely(groupType != -1)) { // -1: no  need to split
        if (groupListType == 0) {
            int32_t offset = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
            splitValue = offset - preOffset;
            preOffset = offset;
        } else {
            splitValue = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
        }
    }
    return splitValue;
}

#if defined(__CCE_AICORE__) && __CCE_AICORE__ >= 310
template <typename T>
__aicore__ inline void InitOutputWithZero(AscendC::GlobalTensor<T> yInitGlobal, AscendC::LocalTensor<T> &initLocal,
                                          uint64_t ySize, int32_t usedCoreNum, bool &isKZeroInit)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (AscendC::GetSubBlockIdx() >= 1) { // 搬运共享带宽，只需要把aiv0的0搬到GM即可，也兼容aic:aiv=1:1的场景
        return;
    }
    uint32_t blockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
    // 仿照InitOutput接口取值
    uint64_t initSize =
        (Mc2QuantUtils::MAX_REPEAT_TIMES * AscendC::ONE_BLK_SIZE) / sizeof(T); // 能存放输出dtype的多少个元素
    uint64_t perCoreSize = Mc2QuantUtils::CeilDiv(ySize, usedCoreNum);
    perCoreSize = MC2_GROUPED_MATMUL::AlignUp<Mc2QuantUtils::UB_ALIGN_SIZE>(perCoreSize * sizeof(T)) / sizeof(T);
    initSize = Mc2QuantUtils::Min(initSize, perCoreSize);
    uint64_t realCoreNum =
        Mc2QuantUtils::Min(Mc2QuantUtils::CeilDiv(ySize, initSize), static_cast<uint64_t>(usedCoreNum));
    if (blockIdx >= realCoreNum) { // 多余核数返回，每个核上最少32B
        return;
    }
    if (!isKZeroInit) { // 第一次k==0时，需要初始化ub中buffer全0
        AscendC::Duplicate<T>(initLocal, 0, initSize);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIdVToMte3);
        isKZeroInit = true;
    }
    uint64_t yOffset = perCoreSize * blockIdx;
    uint64_t outCurSize = (blockIdx == realCoreNum - 1) ? (ySize - yOffset) : perCoreSize;
    uint64_t movRound = outCurSize / initSize;
    uint64_t movTail = outCurSize - movRound * initSize;

    AscendC::DataCopyExtParams ub2GmParams{1, static_cast<uint32_t>(initSize * sizeof(T)), 0, 0, 0};
    for (uint64_t i = 0; i < movRound; ++i) {
        AscendC::DataCopyPad(yInitGlobal[yOffset], initLocal, ub2GmParams);
        yOffset += initSize;
    }
    if (movTail != 0) { // mov tail zero data
        ub2GmParams.blockLen = static_cast<uint32_t>(movTail * sizeof(T));
        AscendC::DataCopyPad(yInitGlobal[yOffset], initLocal, ub2GmParams);
    }
}
#endif
} // namespace Mc2QuantUtils

#endif
