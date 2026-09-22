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

constexpr uint32_t MAX_REP_NUM = 255;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr uint32_t ELEM_PER_BLK_FP32 = 8;
constexpr uint32_t SCALAR_LOCAL_ELEMS = ELEM_PER_BLK_FP32; // Brcb dup + workScalar[0..1]
constexpr float ZERO = 0;
constexpr float SOFTMAX_PAD = -1e20f;
constexpr int32_t HALf_INTERVAL = 2;
constexpr int32_t INDEX_TWO = 2;
constexpr int32_t INDEX_FOUR = 4;
constexpr int32_t INDEX_EIGHT = 8;
constexpr int32_t INDEX_SIXTEEN = 16;
constexpr uint32_t MOV_8 = 8;
constexpr uint32_t MAX_REPEAT_STRIDE = 255U;
constexpr uint32_t MUL_BRC_REP_STRIDE = 8U; // 64 fp32 / repeat，与 mhc DEFAULT_REPEAT_STRIDE 一致

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

__aicore__ inline uint32_t RoundUpFp32(uint32_t num)
{
    return CeilDivU32(num, ELEM_PER_BLK_FP32) * ELEM_PER_BLK_FP32;
}

/*! float 索引向下对齐到 32B block */
__aicore__ inline uint32_t AlignDownFloatOffset(uint32_t idx)
{
    return (idx / ELEM_PER_BLK_FP32) * ELEM_PER_BLK_FP32;
}

__aicore__ inline void ReduceSumForSmallReduceDimPreRepeat(
    const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal, const LocalTensor<float>& tmpLocal,
    const uint32_t elemNum, const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat,
    const uint8_t repStride)
{
    uint32_t elemIndex = 0;
    for (; elemIndex + ELEM_PER_REP_FP32 <= numLastDim; elemIndex += ELEM_PER_REP_FP32) {
        Add(tmpLocal, srcLocal[elemIndex], tmpLocal, elemNum, repeat,
            {1, 1, 1, ELEM_PER_BLK_FP32, repStride, ELEM_PER_BLK_FP32});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(tailCount != 0)) {
        Add(tmpLocal, srcLocal[elemIndex], tmpLocal, tailCount, repeat,
            {1, 1, 1, ELEM_PER_BLK_FP32, repStride, ELEM_PER_BLK_FP32});
    }
    PipeBarrier<PIPE_V>();
    AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        WholeReduceSum<float, false>(dstLocal, tmpLocal, MASK_PLACEHOLDER, repeat, 1, 1, ELEM_PER_BLK_FP32);
    }
#else
    WholeReduceSum<float, false>(dstLocal, tmpLocal, MASK_PLACEHOLDER, repeat, 1, 1, ELEM_PER_BLK_FP32);
#endif
}

__aicore__ inline void ReduceSumForSmallReduceDim(
    const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal, const LocalTensor<float>& tmpLocal,
    const uint32_t numLastDimAligned, const uint32_t numLastDim, const uint32_t tailCount, const uint32_t repeat,
    const uint8_t repStride)
{
    uint32_t repeatTimes = repeat / MAX_REP_NUM;
    if (repeatTimes == 0) {
        ReduceSumForSmallReduceDimPreRepeat(
            dstLocal, srcLocal, tmpLocal, ELEM_PER_REP_FP32, numLastDim, tailCount, repeat, repStride);
    } else {
        uint32_t repTailNum = repeat % MAX_REP_NUM;
        uint32_t repIndex = 0;
        for (; repIndex + MAX_REP_NUM <= repeat; repIndex += MAX_REP_NUM) {
            ReduceSumForSmallReduceDimPreRepeat(
                dstLocal[repIndex], srcLocal[repIndex * numLastDimAligned], tmpLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32, numLastDim, tailCount, MAX_REP_NUM, repStride);
        }
        if (repTailNum != 0) {
            ReduceSumForSmallReduceDimPreRepeat(
                dstLocal[repIndex], srcLocal[repIndex * numLastDimAligned], tmpLocal[repIndex * ELEM_PER_REP_FP32],
                ELEM_PER_REP_FP32, numLastDim, tailCount, repTailNum, repStride);
        }
    }
}

__aicore__ inline void ReduceSumMultiN(
    const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal, const LocalTensor<float>& tmpLocal,
    const uint32_t numRow, const uint32_t numCol, const uint32_t numColAlign)
{
    const uint32_t tailCount = numCol % ELEM_PER_REP_FP32;
    const uint32_t repeat = numRow;
    const uint8_t repStride = numColAlign / ELEM_PER_BLK_FP32;
    Duplicate(tmpLocal, ZERO, numRow * ELEM_PER_REP_FP32);
    PipeBarrier<PIPE_V>();
    ReduceSumForSmallReduceDim(dstLocal, srcLocal, tmpLocal, numColAlign, numCol, tailCount, repeat, repStride);
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
 * Half-interval fold of src, then WholeReduceSum into dst (e.g. vecMeta[n]).
 * NOTE: src is destroyed in-place. No GetValue/SetValue.
 */
__aicore__ inline void ReduceSumHalfInterval(const LocalTensor<float> &dst_local, const LocalTensor<float> &src_local,
                                             int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local, src_local, src_local[bodyCount], tailCount);
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

/*!
 * sumSq → invRms in-place（1 元素）。
 * 对齐 ops-nn RMSNorm：Sqrt + Duplicate(1.0) + Div（避免 Rsqrt / Reciprocal 融合近似）。
 * scratch 需 ≥1 个 32B 对齐块，用于存放 1.0。
 */
__aicore__ inline void InvRmsInPlace(const LocalTensor<float> &dst, float invHiddenSize, float normEps,
                                     const LocalTensor<float> &scratch)
{
    Muls(dst, dst, invHiddenSize, 1);
    PipeBarrier<PIPE_V>();
    Adds(dst, dst, normEps, 1);
    PipeBarrier<PIPE_V>();
    Sqrt(dst, dst, 1);
    PipeBarrier<PIPE_V>();
    Duplicate(scratch, 1.0f, ELEM_PER_BLK_FP32);
    PipeBarrier<PIPE_V>();
    Div(dst, scratch, dst, 1); // 1 / sqrt(meanSq + eps)
    PipeBarrier<PIPE_V>();
}

/*! 从 meta 标量槽拷 1 个 float 到 dst；Vector Copy（PIPE_V），便于 EnQue V_MTE3 同步。
 *  dst/src 起始须 32B 对齐（如 scalarLocal_ / invQue_ block）。
 */
__aicore__ inline void CopyMetaScalarToLocal(const LocalTensor<float>& dst, const LocalTensor<float>& metaSrc)
{
    // mask=1, repeat=1：仅拷 1 个 float；stride 同官方 Copy 示例
    Copy(dst, metaSrc, static_cast<uint64_t>(1), 1, {1, 1, 8, 8});
}

/*!
 * UB→UB 紧凑 float 拷贝：Vector Copy（PIPE_V）。
 * float32 单次 mask∈[1,64]；对任意 elemCount 按 64 切段（repeat≤255），支持 >32/>64。
 * dst/src 基址须 32B 对齐。
 */
__aicore__ inline void CopyCompactFloatsUb(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                         uint32_t elemCount)
{
    if (elemCount == 0) {
        return;
    }
    uint32_t offset = 0;
    uint32_t remain = elemCount;
    while (remain >= ELEM_PER_REP_FP32) {
        const uint32_t reps = (remain / ELEM_PER_REP_FP32 > MAX_REP_NUM) ? MAX_REP_NUM : (remain / ELEM_PER_REP_FP32);
        Copy(dst[offset], src[offset], static_cast<uint64_t>(ELEM_PER_REP_FP32), static_cast<uint8_t>(reps),
             {1, 1, 8, 8});
        offset += reps * ELEM_PER_REP_FP32;
        remain -= reps * ELEM_PER_REP_FP32;
    }
    if (remain > 0) {
        Copy(dst[offset], src[offset], static_cast<uint64_t>(remain), 1, {1, 1, 8, 8});
    }
    PipeBarrier<PIPE_V>();
}

/*! stride=8：紧凑 meta[n] → vecMeta[n*metaStride]（Softmax 后 scatter）。 */
__aicore__ inline void ScatterCompactMetaToStrided(const LocalTensor<float>& vecMetaStrided,
                                                   const LocalTensor<float>& compact, uint32_t blockCount,
                                                   uint32_t metaStride)
{
    for (uint32_t n = 0; n < blockCount; ++n) {
        CopyMetaScalarToLocal(vecMetaStrided[n * metaStride], compact[n]);
    }
}

/*! stride=8：vecMeta[n*metaStride] → 紧凑 meta[n]（Softmax 前 gather）。 */
__aicore__ inline void GatherStridedMetaToCompact(const LocalTensor<float>& compact,
                                                  const LocalTensor<float>& vecMetaStrided, uint32_t blockCount,
                                                  uint32_t metaStride)
{
    for (uint32_t n = 0; n < blockCount; ++n) {
        CopyMetaScalarToLocal(compact[n], vecMetaStrided[n * metaStride]);
    }
}

/*!
 * dst = src * brcOneBlock（1 block 经 Brcb 扩出）。
 * Counter 单次 Mul：mask=hiddenSize，repeat=1；src1Blk/RepStride=0 复用同一 broadcast block。
 * repStride=8（MUL_BRC_REP_STRIDE）与文档 Counter 示例 {1,1,1,8,8,8} 一致（block 单位步长）。
 */
__aicore__ inline void MulRowByBrcBlock(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                        const LocalTensor<float>& brcOneBlock, uint32_t hiddenSize,
                                        uint32_t hiddenSizeAlignFp32)
{
    (void)hiddenSizeAlignFp32;
    // blkStride=1：dst/src 沿 H 连续；src1Blk/RepStride=0：复用同一 brcOneBlock（broadcast）
    // repStride=8：与文档 Counter 示例 {1,1,1,8,8,8} 一致（block 单位步长）
    const BinaryRepeatParams repeatParams{1, 1, 0, static_cast<uint8_t>(MUL_BRC_REP_STRIDE),
                                          static_cast<uint8_t>(MUL_BRC_REP_STRIDE), 0};
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(hiddenSize);
    Mul<float, false>(dst, src, brcOneBlock, MASK_PLACEHOLDER, 1, repeatParams);
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
}

/*!
 * dst += src * brcOneBlock（1 block 经 Brcb 扩出）。
 * Counter 外置：isSetMask=false；src1Blk/RepStride=0 广播同一 block。
 * manageMask=false：调用方已 SetMaskCount + SetVectorMask。
 */
__aicore__ inline void MulAddRowByBrcBlock(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                           const LocalTensor<float>& brcOneBlock, uint32_t hiddenSize,
                                           uint32_t hiddenSizeAlignFp32, bool manageMask = true)
{
    (void)hiddenSizeAlignFp32;
    const BinaryRepeatParams repeatParams{1, 1, 0, static_cast<uint8_t>(MUL_BRC_REP_STRIDE),
                                          static_cast<uint8_t>(MUL_BRC_REP_STRIDE), 0};
    if (manageMask) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(hiddenSize);
    }
    MulAddDst<float, float, false>(dst, src, brcOneBlock, MASK_PLACEHOLDER, 1, repeatParams);
    PipeBarrier<PIPE_V>();
    if (manageMask) {
        SetMaskNorm();
        ResetMask();
    }
}

/*!
 * BF16→FP32：Counter 外置 + Level0 连续 stride（等价 Adds 前n 的 isSetMask=false）。
 * Cast Level2 无 isSetMask，故用 Level0 连续：{dstBlk,srcBlk,dstRep,srcRep}={1,1,8,4}。
 * 调用前须已 SetMaskCount + SetVectorMask<float, COUNTER>(elemCount)。
 */
template <typename SrcT>
__aicore__ inline void CastRowToFp32CounterNoSetMask(const LocalTensor<float>& dst, const LocalTensor<SrcT>& src)
{
    // 与 Level2 Cast 内部一致：fp32 dstRep=8，bf16/half srcRep=4
    const UnaryRepeatParams castParams{1, 1, static_cast<uint8_t>(MUL_BRC_REP_STRIDE),
                                       static_cast<uint8_t>(MUL_BRC_REP_STRIDE / 2U)};
    Cast<float, SrcT, false>(dst, src, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, castParams);
}

/*! 小 B 末轴归约 max/sum（等价 mhc LastDimReduceMax/SumPerf，curRowNum=1）。 */
__aicore__ inline void ReduceMaxSmallB(const LocalTensor<float>& workScalar, const LocalTensor<float>& src,
                                       uint32_t blockCount)
{
    const uint32_t srcRepStride = (blockCount + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        WholeReduceMax<float, true>(workScalar, src, static_cast<int32_t>(blockCount), 1, 1, 1, srcRepStride,
                                    ReduceOrder::ORDER_ONLY_VALUE);
    }
#else
    WholeReduceMax<float, true>(workScalar, src, static_cast<int32_t>(blockCount), 1, 1, 1, DEFAULT_REPEAT_STRIDE,
                                ReduceOrder::ORDER_ONLY_VALUE);
#endif
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ReduceSumSmallB(const LocalTensor<float>& workScalar, const LocalTensor<float>& src,
                                       uint32_t blockCount)
{
    const uint32_t srcRepStride = (blockCount + ELEM_PER_BLK_FP32 - 1U) / ELEM_PER_BLK_FP32;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        WholeReduceSum<float, true>(workScalar, src, static_cast<int32_t>(blockCount), 1, 1, 1, srcRepStride);
    }
#else
    WholeReduceSum<float, true>(workScalar, src, static_cast<int32_t>(blockCount), 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
}

/*! Brcb 标量到 1 block（8 float）。 */
__aicore__ inline void BrcbScalarRow1(const LocalTensor<float>& tmpBuffer, const LocalTensor<float>& scalarRow)
{
    Brcb(tmpBuffer, scalarRow, 1, {1, MOV_8});
    PipeBarrier<PIPE_V>();
}

/*! curRowNum=1 末轴 Sub；tmpBuffer 已由 BrcbScalarRow1 填好 broadcast 值。 */
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
    const int32_t dstRepStridePerLine =
        static_cast<int32_t>(CeilDivU32(static_cast<uint32_t>(curColNum), ELEM_PER_BLK_FP32));
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 1;
    instrParams.src1BlkStride = 0;
    instrParams.dstRepStride = static_cast<uint8_t>(dstRepStridePerLine);
    instrParams.src0RepStride = static_cast<uint8_t>(dstRepStridePerLine);
    instrParams.src1RepStride = 0;
    if (numRemainPerLine > 0) {
        Sub(output[numRepeatPerLine * static_cast<int32_t>(ELEM_PER_REP_FP32)],
            input0[numRepeatPerLine * static_cast<int32_t>(ELEM_PER_REP_FP32)], tmpBuffer,
            static_cast<uint32_t>(numRemainPerLine), 1, instrParams);
    }
    PipeBarrier<PIPE_V>();
}

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
    const int32_t dstRepStridePerLine =
        static_cast<int32_t>(CeilDivU32(static_cast<uint32_t>(curColNum), ELEM_PER_BLK_FP32));
    BinaryRepeatParams instrParams;
    instrParams.dstBlkStride = 1;
    instrParams.src0BlkStride = 1;
    instrParams.src1BlkStride = 0;
    instrParams.dstRepStride = static_cast<uint8_t>(dstRepStridePerLine);
    instrParams.src0RepStride = static_cast<uint8_t>(dstRepStridePerLine);
    instrParams.src1RepStride = 0;
    if (numRemainPerLine > 0) {
        Mul(output[numRepeatPerLine * static_cast<int32_t>(ELEM_PER_REP_FP32)],
            input0[numRepeatPerLine * static_cast<int32_t>(ELEM_PER_REP_FP32)], tmpBuffer,
            static_cast<uint32_t>(numRemainPerLine), 1, instrParams);
    }
    PipeBarrier<PIPE_V>();
}

/*! curRowNum=1 末轴 Sub broadcast（含 Brcb）。 */
__aicore__ inline void SubLastDimBrcRow1(const LocalTensor<float>& output, const LocalTensor<float>& input0,
                                         const LocalTensor<float>& scalarRow, const LocalTensor<float>& tmpBuffer,
                                         int32_t curColNum)
{
    BrcbScalarRow1(tmpBuffer, scalarRow);
    SubLastDimRow1NoBrc(output, input0, tmpBuffer, curColNum);
}

/*!
 * dst = src * broadcast(scalarSrc[0])；1 标量 Brcb 成 1 block，再 MulRowByBrcBlock 沿 H 复用。
 * brcScratch 前 1 block 作 Brcb 输出；dupLocal 为 Brcb 源 block（8 float）。dupLocal 不可与 scalarSrc 同址。
 */
__aicore__ inline void BroadcastScalarMulTensor(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                const LocalTensor<float>& scalarSrc,
                                                const LocalTensor<float>& brcScratch,
                                                const LocalTensor<float>& dupLocal, uint32_t hiddenSize,
                                                uint32_t hiddenSizeAlignFp32)
{
    Duplicate(dupLocal, 0.0f, ELEM_PER_BLK_FP32);
    PipeBarrier<PIPE_V>();
    CopyMetaScalarToLocal(dupLocal, scalarSrc);
    PipeBarrier<PIPE_V>();
    Brcb(brcScratch, dupLocal, 1, {1, MOV_8});
    PipeBarrier<PIPE_V>();
    MulRowByBrcBlock(dst, src, brcScratch, hiddenSize, hiddenSizeAlignFp32);
}

/*!
 * 小 B Softmax（向量路径：WholeReduceMax/Sum + Brcb + Exp + Div）。
 */
__aicore__ inline void SoftmaxSmallVec(const LocalTensor<float>& vecMeta, uint32_t blockCount, uint32_t metaAlign,
                                       const LocalTensor<float>& workScalar, const LocalTensor<float>& brcMeta,
                                       const LocalTensor<float>& brcPack)
{
    const LocalTensor<float> brcScratch = brcPack;
    const int32_t curColNum = static_cast<int32_t>(blockCount);

    Duplicate(brcMeta, SOFTMAX_PAD, metaAlign);
    PipeBarrier<PIPE_V>();
    CopyCompactFloatsUb(brcMeta, vecMeta, blockCount);

    AscendCUtils::SetMask<float>(blockCount);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        WholeReduceMax<float, false>(workScalar, brcMeta, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#else
    WholeReduceMax<float, false>(workScalar, brcMeta, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();

    BrcbScalarRow1(brcScratch, workScalar);
    SubLastDimRow1NoBrc(brcMeta, brcMeta, brcScratch, curColNum);

    Exp(brcMeta, brcMeta, metaAlign);
    PipeBarrier<PIPE_V>();

    AscendCUtils::SetMask<float>(blockCount);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
        WholeReduceSum<float, false>(workScalar, brcMeta, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#else
    WholeReduceSum<float, false>(workScalar, brcMeta, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
#endif
    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    PipeBarrier<PIPE_V>();

    Duplicate(brcScratch, 1.0f, 1);
    PipeBarrier<PIPE_V>();
    Div(workScalar, brcScratch, workScalar, 1);
    PipeBarrier<PIPE_V>();

    BrcbScalarRow1(brcScratch, workScalar);
    MulLastDimRow1NoBrc(brcMeta, brcMeta, brcScratch, curColNum);

    CopyCompactFloatsUb(vecMeta, brcMeta, blockCount);
    PipeBarrier<PIPE_V>();
}

#endif // REDUCE_COMMON_H_ATTN_RES_FWD
