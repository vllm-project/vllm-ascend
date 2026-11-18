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
 * \file reduce_common.h
 * \brief Half-interval reduce; vecMeta helpers — **禁止** LocalTensor::GetValue/SetValue。
 */
#ifndef REDUCE_COMMON_H_ATTN_RES_FWD
#define REDUCE_COMMON_H_ATTN_RES_FWD
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
using namespace AscendC;

constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr uint32_t ELEM_PER_BLK_FP32 = 8;
constexpr uint32_t SCALAR_LOCAL_ELEMS = ELEM_PER_BLK_FP32; // Brcb dup + workScalar[0..1]
constexpr float SOFTMAX_PAD = -3.402823466e38f;
constexpr int32_t HALf_INTERVAL = 2;
constexpr int32_t INDEX_TWO = 2;
constexpr int32_t INDEX_FOUR = 4;
constexpr int32_t INDEX_EIGHT = 8;
constexpr int32_t INDEX_SIXTEEN = 16;
constexpr uint32_t MOV_8 = 8;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

__aicore__ inline uint32_t RoundUpFp32(uint32_t num)
{
    return CeilDivU32(num, ELEM_PER_BLK_FP32) * ELEM_PER_BLK_FP32;
}

__aicore__ inline int32_t findPowerTwo(int32_t n)
{
    n |= n >> 1;
    n |= n >> INDEX_TWO;
    n |= n >> INDEX_FOUR;
    n |= n >> INDEX_EIGHT;
    n |= n >> INDEX_SIXTEEN;
    return (n + 1) >> 1;
}

/*!
 * Half-interval fold of src, then WholeReduceMax into dst.
 * NOTE: src is destroyed in-place. No GetValue/SetValue.
 * 支持 count > ELEM_PER_REP_FP32（64）。
 */
__aicore__ inline void ReduceMaxHalfInterval(const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local,
                                             int32_t count)
{
    if (likely(count > static_cast<int32_t>(ELEM_PER_REP_FP32))) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            // Level2 Max 计数需 8 对齐；尾区外须已 pad（如 SOFTMAX_PAD）
            Max(src_local, src_local, src_local[bodyCount],
                static_cast<int32_t>(RoundUpFp32(static_cast<uint32_t>(tailCount))));
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > static_cast<int32_t>(ELEM_PER_REP_FP32)) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Max(src_local, src_local, src_local[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }

        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        WholeReduceMax<float, false>(dst_local, src_local, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#else
    WholeReduceMax<float, false>(dst_local, src_local, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();
}

/*!
 * Half-interval fold of src, then WholeReduceSum into dst (e.g. vecMeta[n]).
 * NOTE: src is destroyed in-place. No GetValue/SetValue.
 * 支持 count > ELEM_PER_REP_FP32（64）。
 */
__aicore__ inline void ReduceSumHalfInterval(const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local,
                                             int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local, src_local, src_local[bodyCount],
                static_cast<int32_t>(RoundUpFp32(static_cast<uint32_t>(tailCount))));
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(src_local, src_local, src_local[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }

        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        WholeReduceSum<float, false>(dst_local, src_local, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#else
    WholeReduceSum<float, false>(dst_local, src_local, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();
}

/*! 从 meta 标量槽拷 1 个 float 到 dst；用 Vector Copy（PIPE_V），便于 EnQue V_MTE3 同步。
 *  禁止 GetValue/SetValue。dst/src 建议 32B 对齐（如 scalarLocal_ / invQue_ block）。
 */
__aicore__ inline void CopyMetaScalarToLocal(const LocalTensor<float>& dst, const LocalTensor<float>& metaSrc)
{
    // mask=1, repeat=1：仅拷 1 个 float；stride 参数同官方 Copy 示例
    Copy(dst, metaSrc, static_cast<uint64_t>(1), 1, {1, 1, 8, 8});
}

/*!
 * UB→UB 紧凑 float 拷贝。dst/src 基址须 32B 对齐；计数 RoundUp 到 8，
 * 调用方保证容量 ≥ RoundUpFp32(elemCount)（如 metaAlign）。
 * （实测 DataCopy(非 8 倍 count) 在跨 VL rem 上可能漏写。）
 */
__aicore__ inline void CopyCompactFloatsUb(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                         uint32_t elemCount)
{
    if (elemCount == 0) {
        return;
    }
    DataCopy(dst, src, RoundUpFp32(elemCount));
    PipeBarrier<PIPE_V>();
}

/*! Brcb 标量到 1 block（8 float）。 */
__aicore__ inline void BrcbScalarRow1(const LocalTensor<float>& tmpBuffer, const LocalTensor<float>& scalarRow)
{
    Brcb(tmpBuffer, scalarRow, 1, {1, MOV_8});
    PipeBarrier<PIPE_V>();
}

/*! curRowNum=1 末轴 Sub；tmpBuffer 已由 BrcbScalarRow1 填好 broadcast 值。支持 curColNum>64。 */
__aicore__ inline void SubLastDimRow1NoBrc(const LocalTensor<float>& output, const LocalTensor<float>& input0,
                                          const LocalTensor<float>& tmpBuffer, int32_t curColNum)
{
    const uint32_t curColNumAlign = RoundUpFp32(static_cast<uint32_t>(curColNum));
    if (curColNum <= static_cast<int32_t>(ELEM_PER_BLK_FP32)) {
        Sub(output, input0, tmpBuffer, curColNumAlign);
        PipeBarrier<PIPE_V>();
        return;
    }
    const int32_t numRepeatPerLine = curColNum / static_cast<int32_t>(ELEM_PER_REP_FP32);
    const int32_t numRemainPerLine = curColNum % static_cast<int32_t>(ELEM_PER_REP_FP32);
    // 紧凑 1 行：每 VL 前进 8 个 block；src1 为 Brcb 块，repStride=0
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 1;
    instrParams.src1BlkStride = 0;
    instrParams.dstRepStride = static_cast<uint8_t>(ELEM_PER_REP_FP32 / ELEM_PER_BLK_FP32);
    instrParams.src0RepStride = static_cast<uint8_t>(ELEM_PER_REP_FP32 / ELEM_PER_BLK_FP32);
    instrParams.src1RepStride = 0;
    if (numRepeatPerLine > 0) {
        Sub(output, input0, tmpBuffer, ELEM_PER_REP_FP32, numRepeatPerLine, instrParams);
        PipeBarrier<PIPE_V>();
    }
    if (numRemainPerLine > 0) {
        Sub(output[numRepeatPerLine * static_cast<int32_t>(ELEM_PER_REP_FP32)],
            input0[numRepeatPerLine * static_cast<int32_t>(ELEM_PER_REP_FP32)], tmpBuffer,
            static_cast<uint32_t>(numRemainPerLine), 1, instrParams);
        PipeBarrier<PIPE_V>();
    }
}

/*! curRowNum=1 末轴 Mul；支持 curColNum>64。 */
__aicore__ inline void MulLastDimRow1NoBrc(const LocalTensor<float>& output, const LocalTensor<float>& input0,
                                           const LocalTensor<float>& tmpBuffer, int32_t curColNum)
{
    const uint32_t curColNumAlign = RoundUpFp32(static_cast<uint32_t>(curColNum));
    if (curColNum <= static_cast<int32_t>(ELEM_PER_BLK_FP32)) {
        Mul(output, input0, tmpBuffer, curColNumAlign);
        PipeBarrier<PIPE_V>();
        return;
    }
    const int32_t numRepeatPerLine = curColNum / static_cast<int32_t>(ELEM_PER_REP_FP32);
    const int32_t numRemainPerLine = curColNum % static_cast<int32_t>(ELEM_PER_REP_FP32);
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 1;
    instrParams.src1BlkStride = 0;
    instrParams.dstRepStride = static_cast<uint8_t>(ELEM_PER_REP_FP32 / ELEM_PER_BLK_FP32);
    instrParams.src0RepStride = static_cast<uint8_t>(ELEM_PER_REP_FP32 / ELEM_PER_BLK_FP32);
    instrParams.src1RepStride = 0;
    if (numRepeatPerLine > 0) {
        Mul(output, input0, tmpBuffer, ELEM_PER_REP_FP32, numRepeatPerLine, instrParams);
        PipeBarrier<PIPE_V>();
    }
    if (numRemainPerLine > 0) {
        const uint32_t remAlign = RoundUpFp32(static_cast<uint32_t>(numRemainPerLine));
        Mul(output[numRepeatPerLine * static_cast<int32_t>(ELEM_PER_REP_FP32)],
            input0[numRepeatPerLine * static_cast<int32_t>(ELEM_PER_REP_FP32)], tmpBuffer, remAlign, 1,
            instrParams);
        PipeBarrier<PIPE_V>();
    }
}

#endif // REDUCE_COMMON_H_ATTN_RES_FWD
