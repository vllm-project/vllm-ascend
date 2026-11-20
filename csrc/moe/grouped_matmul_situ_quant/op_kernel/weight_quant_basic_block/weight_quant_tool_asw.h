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
 * \file weight_quant_tool_asw.h
 * \brief
 */

#pragma once
// 本文件从 ops-transformer 的 gmm/common/op_kernel 同步而来。上游那一行
// `#include "../cgmct/utils/cgmct_common_utils.h"` 在本仓库没有对应目录，
// 且本文件与 weight_quant_basic_block 下所有文件都不引用 Cgmct:: 命名空间，
// 属于上游遗留的无效包含，此处不再引入。

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "op_kernel/math_util.h"

namespace WeightQuantTools {
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::Dn2NzParams;
using AscendC::fp8_e8m0_t;
using AscendC::GlobalTensor;
using AscendC::hif4_scale;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::IsSupportB4;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;

constexpr uint64_t MX_GROUP_SIZE_32 = 32UL;
constexpr uint64_t MX_SCALE_LAST_DIM_2 = 2UL;
constexpr uint64_t MX_GROUP_SIZE_64 = 64UL;
constexpr uint64_t MX_SCALE_DIVISOR_SIZE = 64UL;
constexpr uint64_t B8_IN_B16_NUM = 2UL;
constexpr uint64_t B32_IN_B16_NUM = 2UL;
constexpr uint64_t SWIGLU_N_HALF_FACTOR = 2UL;

constexpr uint8_t SPLIT_M = 0;
constexpr uint64_t INNER_AXIS_MIN_SPLIT_VAL = 128UL; // ND2NZ cacheline 128

static constexpr uint32_t L1_BUFFER_HALF_SIZE_KB = 256U;
static constexpr uint32_t L0A_BUFFER_SIZE_KB = 64U;
static constexpr uint32_t L0B_BUFFER_SIZE_KB = 64U;
static constexpr uint32_t BIAS_TABLE_SIZE_KB = 4U;
static constexpr uint32_t KB_UNIT = 1024U;

static constexpr uint8_t DOUBLE_BUFFER = 2;
static constexpr uint8_t MTE1_MTE2_EVENT_ID = 2;
static constexpr uint8_t MTE1_MTE2_SCALE_EVENT_ID = 4;
static constexpr uint8_t MTE2_MTE1_EVENT_ID = 2;
static constexpr uint8_t M_MTE1_EVENT_ID = 3;
static constexpr uint8_t MTE1_M_EVENT_ID = 6;

template <typename T>
struct C0 {
    static constexpr uint64_t value = AscendC::IsSupportB4<T>() ? 64UL : (32UL / sizeof(T));
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

template <typename xType, typename wType, typename antiquantScaleType>
__aicore__ inline constexpr bool IsMxA8W4()
{
    return IsSameType<xType, fp8_e4m3fn_t>::value && IsSameType<wType, fp4x2_e2m1_t>::value &&
           IsSameType<antiquantScaleType, fp8_e8m0_t>::value;
}

template <typename xType, typename wType, typename antiquantScaleType>
__aicore__ inline constexpr bool IsMxA8HiF4()
{
    return IsSameType<xType, fp8_e4m3fn_t>::value && IsSameType<wType, hifloat4x2_t>::value &&
           IsSameType<antiquantScaleType, hif4_scale>::value;
}

template <typename xType, typename wType, typename antiquantScaleType>
__aicore__ inline constexpr bool IsA16MxFp4()
{
    return (IsSameType<xType, half>::value || IsSameType<xType, bfloat16_t>::value) &&
           IsSameType<wType, fp4x2_e2m1_t>::value && IsSameType<antiquantScaleType, fp8_e8m0_t>::value;
}

template <typename xType, typename wType, typename antiquantScaleType>
__aicore__ inline constexpr bool IsA16MxHiF4()
{
    return (IsSameType<xType, half>::value || IsSameType<xType, bfloat16_t>::value) &&
           IsSameType<wType, hifloat4x2_t>::value && IsSameType<antiquantScaleType, hif4_scale>::value;
}

template <typename T>
__aicore__ constexpr uint32_t GetKBUnit()
{
    return AscendC::IsSupportB4<T>() ? KB_UNIT << 1 : KB_UNIT / sizeof(T);
}

__aicore__ inline uint64_t GetSplitValueFromGroupList(uint32_t groupIdx, uint64_t& preOffset, int32_t groupType,
                                                      uint32_t groupListType,
                                                      const AscendC::GlobalTensor<int64_t>& groupListGm)
{
    uint64_t splitValue = 0;
    if (likely(groupType != -1)) { // -1: no need to split
        if (groupListType == 0) {
            uint64_t offset = static_cast<uint64_t>(groupListGm.GetValue(groupIdx));
            splitValue = offset - preOffset;
            preOffset = offset;
        } else {
            splitValue = static_cast<uint64_t>(groupListGm.GetValue(groupIdx));
        }
    }
    return splitValue;
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t blockCount,
                                     uint32_t blockLen, uint32_t dstInnerLength, uint32_t srcInnerLength,
                                     bool enablePreAlloc)
{
    DataCopyExtParams params;
    params.blockCount = blockCount;
    params.blockLen = blockLen * sizeof(T);
    params.srcStride = (srcInnerLength - blockLen) * sizeof(T);
    params.dstStride = (dstInnerLength - blockLen) * sizeof(T) / ONE_BLK_SIZE;
    DataCopyPadExtParams<T> padParams;
    if (blockLen % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = Ops::Base::CeilAlign(blockLen, static_cast<uint32_t>(32 / sizeof(T))) - blockLen;
        padParams.paddingValue = 0;
    }

    if constexpr (IsSameType<T, int4b_t>::value || IsSameType<T, fp4x2_e2m1_t>::value ||
                  IsSameType<T, fp4x2_e1m2_t>::value) {
        // 4bit场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
        padParams.rightPadding = padParams.rightPadding >> 1;
    }
    DataCopyPad(dst, src, params, padParams, enablePreAlloc);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t dim1,
                                     uint32_t dim0, uint32_t srcFullDim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = Ops::Base::CeilDiv((srcFullDim0 - dim0) * sizeof(T), static_cast<uint64_t>(ONE_BLK_SIZE));
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

__aicore__ inline void ConfigE8M0ScaleDn2NzParams(uint64_t rowNum, uint64_t scaleKGmGroupNum, uint64_t scaleL1GroupNum,
                                                  Dn2NzParams& dn2NzParams)
{
    dn2NzParams.dnNum = 1;
    dn2NzParams.dValue = rowNum; // 矩阵的行数，即待搬运的mxScaleA的m或mxScaleB的n
    dn2NzParams.nValue = Ops::Base::CeilDiv(scaleL1GroupNum, B8_IN_B16_NUM); // 矩阵的列数，使用B16搬B8需要除以2
    dn2NzParams.srcDnMatrixStride = 0;
    dn2NzParams.srcDValue = Ops::Base::CeilDiv(scaleKGmGroupNum, B8_IN_B16_NUM); // 源矩阵一行所含B16元素个数
    // 目标矩阵行方向两个相邻分形起始地址之间的间隔，单位32B(一个16 * 2分形)
    dn2NzParams.dstNzC0Stride = Ops::Base::CeilDiv(scaleL1GroupNum, B8_IN_B16_NUM);
    // 目标矩阵列方向两个相邻分形起始地址之间的间隔，单位32B(一个16 * 2分形)
    dn2NzParams.dstNzNStride = 1;
    dn2NzParams.dstNzMatrixStride = 0;
}

__aicore__ inline void ConfigHif4ScaleDn2NzParams(uint64_t rowNum, uint64_t scaleKGmGroupNum, uint64_t scaleL1GroupNum,
                                                  Dn2NzParams& dn2NzParams)
{
    dn2NzParams.dnNum = 1;
    dn2NzParams.dValue = rowNum; // 矩阵的行数，即待搬运的mxScaleA的m或mxScaleB的n
    dn2NzParams.nValue = scaleL1GroupNum * B32_IN_B16_NUM; // 矩阵的列数，使用B16搬B32需要乘以2
    dn2NzParams.srcDnMatrixStride = 0;
    dn2NzParams.srcDValue = scaleKGmGroupNum * B32_IN_B16_NUM; // 源矩阵一行所含B16元素个数
    // 目标矩阵行方向两个相邻分形起始地址之间的间隔，单位32B(一个16 * 2分形)
    dn2NzParams.dstNzC0Stride = scaleL1GroupNum * B32_IN_B16_NUM;
    // 目标矩阵列方向两个相邻分形起始地址之间的间隔，单位32B(一个16 * 2分形)
    dn2NzParams.dstNzNStride = 1;
    dn2NzParams.dstNzMatrixStride = 0;
}

} // namespace WeightQuantTools
