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
 * \file compress_norm_rope_comm.h
 * \brief compress_norm_rope 公共定义：常量、对齐工具、DataCopy 帮助函数
 *
 * 算子语义：
 *   输入 mm_kv/mm_score = 外部 GEMM 输出 [T, coff*headDim]（raw，不含 ape）
 *   每个压缩组（cmpRatio 个 token）产出 1 行压缩 kv：
 *     C4  (r=4,   coff=2): 窗口 8 行 = [前一组 4 行 coff0 | 当前组 4 行 coff1]
 *     C128(r=128, coff=1): 窗口 128 行 = 当前组
 *   score += ape（按组内位置）→ 逐列 ColumnSoftMax → p·kv 逐列求和 → cmp_kv
 *   全部 token 的 [kv(raw) | score(+ape)] fp32 写回分页 state_cache（in-place）
 */

#ifndef COMPRESS_NORM_ROPE_COMM_H
#define COMPRESS_NORM_ROPE_COMM_H

#include "kernel_operator.h"

using namespace AscendC;

namespace CompressNormRope {

template <typename T>
__aicore__ inline T CeilDivT(T num1, T num2)
{
    if (num2 == 0) {
        return static_cast<T>(0);
    }
    return (num1 + num2 - 1) / num2;
}

// BUFFER 字节数
inline constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
inline constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
inline constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
inline constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
inline constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
inline constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
inline constexpr uint32_t BUFFER_SIZE_BYTE_64K = 65536;

inline constexpr uint32_t BYTE_BLOCK = 32UL;
inline constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);      // 8
inline constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
inline constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float); // 64
inline constexpr uint32_t REPEAT_STRIDE_NUM = REPEAT_BLOCK_BYTE / BYTE_BLOCK;          // 8
inline constexpr uint32_t REPEAT_MAX_NUM = 255;
inline constexpr uint32_t MAX_R = 256;

inline constexpr float SOFTMAX_MIN_VALUE = -2e38f;
inline constexpr float FLOAT_ZERO = 0.0f;
inline constexpr uint32_t BRCB_NUM = 8; // Brcb 单指令广播元素数

// rotary_mode 属性值
enum class ROTARY_MODE : uint8_t {
    HALF = 1,       // 半旋转：dst[i]=src[i]c[i]-src[i+h]s[i]
    INTERLEAVE = 2, // 交错：dst[2k]=src[2k]c[2k]-src[2k+1]s[2k]
};

template <typename C>
__aicore__ inline constexpr uint32_t BlockElementNum()
{
    return static_cast<uint32_t>(BYTE_BLOCK / sizeof(C));
}

// 二维 strided DataCopy 帮助函数（blockLen/gap 以 32B block 为单位）
// 契约（调用侧保证）：copyRowCount >= 1；copyColCount 为 BlockElementNum<O>() 的整数倍且
// 不超过 src/dst 行宽。
template <typename O>
__aicore__ inline void DataCopyAlignGmToUb(const LocalTensor<O> &dstLocal, const GlobalTensor<O> &srcGm,
                                           uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                           uint32_t dstSingleRowCount)
{
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / BlockElementNum<O>();
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / BlockElementNum<O>();
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / BlockElementNum<O>();
    DataCopy(dstLocal, srcGm, intriParams);
}

template <typename O>
__aicore__ inline void DataCopyAlignUbToGm(const GlobalTensor<O> &dstGm, const LocalTensor<O> &srcLocal,
                                           uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                           uint32_t dstSingleRowCount)
{
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / BlockElementNum<O>();
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / BlockElementNum<O>();
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / BlockElementNum<O>();
    DataCopy(dstGm, srcLocal, intriParams);
}

template <typename O>
__aicore__ inline void DataCopyAlignUbToUb(const LocalTensor<O> &dstLocal, const LocalTensor<O> &srcLocal,
                                           uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount,
                                           uint32_t dstSingleRowCount)
{
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / BlockElementNum<O>();
    intriParams.dstGap = (dstSingleRowCount - copyColCount) / BlockElementNum<O>();
    intriParams.srcGap = (srcSingleRowCount - copyColCount) / BlockElementNum<O>();
    DataCopy(dstLocal, srcLocal, intriParams);
}

} // namespace CompressNormRope

#endif // COMPRESS_NORM_ROPE_COMM_H
