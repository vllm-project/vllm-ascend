/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KV_COMPRESS_EPILOG_V2_COMMON_H
#define KV_COMPRESS_EPILOG_V2_COMMON_H

#include "kernel_operator.h"

namespace KvCompressEpilogV2Ops {
using namespace AscendC;
// MicroAPI using 声明供 include 本文件的 quant_*/layout_* 头文件共享使用
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UnalignReg;

constexpr int32_t KCEV2_BLOCK_BYTES = 32;
constexpr int32_t KCEV2_GROUP_ELEMS = 32;
constexpr int32_t KCEV2_VL_FP32 = 64;
constexpr uint16_t KCEV2_BF16_EXP_MASK = 0x7F80U;
constexpr uint16_t KCEV2_BF16_NAN = 0x7F81U;
constexpr uint16_t KCEV2_BF16_INV_BIAS = 0x7F00U;
constexpr uint16_t KCEV2_FP4_E2M1_MAX_EXP = 0x0100U;
constexpr uint16_t KCEV2_FP4_SPECIAL_INV = 0x0040U;

constexpr int32_t KCEV2_PERF_STAT_ELEMS = 256;
constexpr int32_t KCEV2_PERF_MAX_EXP_PER_BEAT = 8;
constexpr int32_t KCEV2_QUEUE_DEPTH = 2;

#define FLOAT_OVERFLOW_MODE_CTRL 60

__aicore__ inline int32_t CeilDiv(int32_t value, int32_t divisor)
{
    return divisor == 0 ? value : (value + divisor - 1) / divisor;
}

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t count)
{
    const int32_t blockElems = KCEV2_BLOCK_BYTES / sizeof(T);
    return CeilDiv(count, blockElems) * blockElems;
}

template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T> &src, const LocalTensor<T> &dst, uint32_t count)
{
    DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}

__aicore__ inline void CopyOutBytes(const LocalTensor<uint8_t> &src, const GlobalTensor<uint8_t> &dst, uint32_t count)
{
    DataCopyExtParams copyParams{1, count, 0, 0, 0};
    DataCopyPad(dst, src, copyParams);
}

}  // namespace KvCompressEpilogV2Ops

#endif
