/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attn_sink_gs1.h
 * \brief gS1 合轴场景 sink 特性级 common: CopyIn(GM->UB 值区布局) + 注入链寄存器直写。
 * sinks 值类型为 float32(与 q dtype 解耦)。
 *
 * 数学目标(Scheme B, prologue 注入): maxUb[r] = inUb[idx(r)], r < m(<= 64)。
 *
 * idx(r) 依布局(值区布局由 SinkCopyInGS1/SinkCopyInS1G 决定, 直写函数只读不写):
 *   GS1(BNSD):       idx = (s0 + r) / actS1Size           —— SinkGs1DirectVF;
 *   S1G gSize==1:    恒同值, BRC 单值广播                  —— SinkBrcVF;
 *   S1G case1:       idx = r(值区连续)                     —— SinkContigDirectVF;
 *   S1G case2.1:     前 headLen 行 idx = headOff + r,
 *                    其余行 idx = r - headLen(gap 双段)    —— SinkGapDirectVF;
 *   S1G case2.2/3:   idx = (gStart + r) mod gSize(回绕)    —— SinkRotDirectVF。
 *
 * 数据流: 值区(float) DIST_NORM 直读进寄存器 -> (Gather/Select 重排) ->
 * 整块 StoreAlign 单次写 maxUb(唯一 REG->UB 落点, 256B, 带 mask)。
 * 值区跨寄存器时统一宽模式: A/B 双 load(基址 0/64, 后者 256B 天然 32B 对齐) + 双 Gather
 * (B 路索引 = idx - 64) + Compares(idx < 64) Select 拼合 —— 值域 <= 64 时 B 路恒被掩掉。
 * 索引取模/整除用寄存器 Div 实现(无 Mod 指令): k = a - (a / d) * d。
 * 越界语义: Gather 索引越界按 mod (VL/sizeof(T)) 取值, 负数经 uint32 重解释变 huge,
 * 其结果恒被 Select/store mask 掩掉, 不产生有效输出。
 */
#ifndef ATTN_SINK_GS1_H_
#define ATTN_SINK_GS1_H_

#include "kernel_operator.h"
#include "../../../a5_mla_common/op_kernel/const_def.h"

using namespace AscendC;

namespace AttentionCommon {
using namespace Reg;

template <typename INPUT_T>
__aicore__ inline void SinkCopyInGS1(LocalTensor<INPUT_T> &sinkTmpUb, GlobalTensor<INPUT_T> &sinkGm, uint32_t gs1Start,
                                     uint32_t actVecMSize, uint32_t actS1Size, uint32_t n2Idx, uint32_t gSize)
{
    uint32_t gStart = gs1Start / actS1Size;
    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End / actS1Size;

    uint32_t gCount = gEnd - gStart + 1;
    DataCopyExtParams sinkCopyParams;
    sinkCopyParams.blockCount = 1;
    sinkCopyParams.blockLen = gCount * sizeof(INPUT_T);
    sinkCopyParams.srcStride = 0;
    sinkCopyParams.dstStride = 0;
    DataCopyPadExtParams<INPUT_T> sinkPadParams;
    sinkPadParams.isPad = true;
    sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
    DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);
}

template <typename INPUT_T>
__aicore__ inline void SinkCopyInS1G(LocalTensor<INPUT_T> &sinkTmpUb, GlobalTensor<INPUT_T> &sinkGm, uint32_t gs1Start,
                                     uint32_t actVecMSize, uint32_t actS1Size, uint32_t n2Idx, uint32_t gSize)
{
    uint32_t gStart = gs1Start % gSize;
    uint32_t s1Start = gs1Start / gSize;

    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End % gSize;
    uint32_t s1End = gs1End / gSize;

    if (s1Start == s1End) {
        // 情况1: 只有一个头块, 此时[gStart, gEnd]之间的行就是实际处理的M轴
        uint32_t gCount = gEnd - gStart + 1;
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;
        sinkCopyParams.blockLen = gCount * sizeof(INPUT_T);
        sinkCopyParams.srcStride = 0;
        sinkCopyParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> sinkPadParams;
        sinkPadParams.isPad = true;
        sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
        DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);
    } else if (s1Start + 1 == s1End) {
        if (gStart > gEnd) {
            // 情况2.1: 头块 + 尾块, 此时头块的[gStart, gSize)和尾块的[0, gEnd]为实际处理的M轴.
            // 并且gStart > gEnd, 此时它们不重叠.
            // 由于头尾块的长度都可能不对齐, 并且gSize有可能很大, 而UB要求32B对齐,
            // 约定先拷贝尾块的GM[0, gEnd]->UB[0, Align(gEnd + 1, 8) - 1],
            // 再拷贝头块的GM[gStart, gSize)->UB[Align(gEnd + 1, 8), Align(gEnd + 1, 8) + Align(gSize - gStart,
            // 8))
            DataCopyExtParams sinkCopyParams;
            sinkCopyParams.blockCount = 1;
            sinkCopyParams.blockLen = (gEnd + 1) * sizeof(INPUT_T);
            sinkCopyParams.srcStride = 0;
            sinkCopyParams.dstStride = 0;
            DataCopyPadExtParams<INPUT_T> sinkPadParams;
            sinkPadParams.isPad = true;
            sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
            DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);

            uint32_t ubOffset = AttentionCommon::Align(gEnd + 1, 8U);
            sinkCopyParams.blockLen = (gSize - gStart) * sizeof(INPUT_T);
            DataCopyPad(sinkTmpUb[ubOffset], sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);

        } else {
            // 情况2.2: 头块 + 尾块, 此时头块的[gStart, gSize)和尾块的[0, gEnd]为实际处理的M轴.
            // 并且gStart <= gEnd, 此时它们重叠, 合并后的覆盖范围为[0, gSize)
            DataCopyExtParams sinkCopyParams;
            sinkCopyParams.blockCount = 1;
            sinkCopyParams.blockLen = gSize * sizeof(INPUT_T);
            sinkCopyParams.srcStride = 0;
            sinkCopyParams.dstStride = 0;
            DataCopyPadExtParams<INPUT_T> sinkPadParams;
            sinkPadParams.isPad = true;
            sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
            DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);
        }
    } else {
        // 情况3: 头块 + 中间块 + 尾块, 合并后的覆盖范围为[0, gSize)
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;
        sinkCopyParams.blockLen = gSize * sizeof(INPUT_T);
        sinkCopyParams.srcStride = 0;
        sinkCopyParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> sinkPadParams;
        sinkPadParams.isPad = true;
        sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
        DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);
    }
}

/*!
 * \brief gSize==1: 单值广播直写。
 * 前置: CopyIn 已将唯一 sink 值(float)放入 inUb[0]。BRC_B32 将其广播到 64 个 lane。
 */
template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkBrcVF(__ubuf__ T *maxUb, __ubuf__ INPUT_T *inUb, uint32_t m)
{
    RegTensor<T> valReg;
    LoadAlign<T, Reg::LoadDist::DIST_BRC_B32>(valReg, inUb);
    uint32_t sreg = m;
    MaskReg pregF = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, valReg, pregF);
}

/*!
 * \brief 连续值段直写: maxUb[r] = inUb[r], r < m。
 * 前置: inUb[0..m) 为本块行值连续段(float, 基址 32B 对齐)。
 */
template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkContigDirectVF(__ubuf__ T *maxUb, __ubuf__ INPUT_T *inUb, uint32_t m)
{
    RegTensor<T> fReg;
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fReg, inUb);
    uint32_t sreg = m;
    MaskReg pregF = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, fReg, pregF);
}

/*!
 * \brief GS1(BNSD) 直写: maxUb[r] = inUb[(s0 + r) / actS1Size], r < m。
 * 前置: inUb[0..gCount) 为本块 g 值连续区(float, GS1 CopyIn 布局)。
 */
template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkGs1DirectVF(__ubuf__ T *maxUb, __ubuf__ INPUT_T *inUb, uint32_t s0, uint32_t actS1Size,
                                        uint32_t m)
{
    RegTensor<T> fRegA;
    RegTensor<T> fRegB;
    RegTensor<T> gRegA;
    RegTensor<T> gRegB;
    RegTensor<T> oReg;
    RegTensor<int32_t> aReg;
    RegTensor<int32_t> qReg;
    RegTensor<int32_t> qSub;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fRegA, inUb);
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fRegB, inUb + 64);
    // 行索引 idx = (s0 + r) / actS1Size(寄存器 Div 整除)
    Reg::Arange<int32_t>(aReg, static_cast<int32_t>(s0));
    RegTensor<int32_t> aszReg;
    Reg::Duplicate<int32_t>(aszReg, static_cast<int32_t>(actS1Size), pregAll);
    Reg::Div<int32_t>(qReg, aReg, aszReg, pregAll);
    Reg::Adds<int32_t>(qSub, qReg, -64, pregAll); // B 段 lane = idx - 64
    // 宽模式拼合: idx < 64 取 A 段, 否则取 B 段(负值 huge 越界, 被 Select 掩掉)
    RegTensor<uint32_t> &qU32 = (RegTensor<uint32_t> &)qReg;
    RegTensor<uint32_t> &qSubU32 = (RegTensor<uint32_t> &)qSub;
    Reg::Gather<T>(gRegA, fRegA, qU32);
    Reg::Gather<T>(gRegB, fRegB, qSubU32);
    MaskReg pregA = CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Compares<int32_t, CMPMODE::LT>(pregA, qReg, 64, pregAll);
    Reg::Select<T>(oReg, gRegA, gRegB, pregA);
    uint32_t sreg = m;
    MaskReg pregF = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregF);
}

/*!
 * \brief 旋转(S1G case2.2/3) 直写: maxUb[r] = inUb[(gStart + r) mod gSize], r < m。
 * 前置: inUb[0..gSize) 为本 n2 完整值区(float, case2.2/3 CopyIn 布局)。
 * case2.2(gStart <= gEnd)序列不回绕, 索引恒等重排, 同样正确。
 */
template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkRotDirectVF(__ubuf__ T *maxUb, __ubuf__ INPUT_T *inUb, uint32_t gStart, uint32_t gSize,
                                        uint32_t m)
{
    RegTensor<T> fRegA;
    RegTensor<T> fRegB;
    RegTensor<T> gRegA;
    RegTensor<T> gRegB;
    RegTensor<T> oReg;
    RegTensor<int32_t> aReg;
    RegTensor<int32_t> qReg;
    RegTensor<int32_t> kReg;
    RegTensor<int32_t> kSub;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fRegA, inUb);
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fRegB, inUb + 64);
    // 行索引 k = (gStart + r) mod gSize(Div 取模: k = a - (a/gSize)*gSize)
    Reg::Arange<int32_t>(aReg, static_cast<int32_t>(gStart));
    RegTensor<int32_t> gszReg;
    Reg::Duplicate<int32_t>(gszReg, static_cast<int32_t>(gSize), pregAll);
    Reg::Div<int32_t>(qReg, aReg, gszReg, pregAll);
    Reg::Muls<int32_t>(kReg, qReg, static_cast<int32_t>(gSize), pregAll);
    Reg::Sub<int32_t>(kReg, aReg, kReg, pregAll);
    Reg::Adds<int32_t>(kSub, kReg, -64, pregAll); // B 段 lane = k - 64
    // 宽模式拼合: k < 64 取 A 段, 否则取 B 段(负值 huge 越界, 被 Select 掩掉)
    RegTensor<uint32_t> &kU32 = (RegTensor<uint32_t> &)kReg;
    RegTensor<uint32_t> &kSubU32 = (RegTensor<uint32_t> &)kSub;
    Reg::Gather<T>(gRegA, fRegA, kU32);
    Reg::Gather<T>(gRegB, fRegB, kSubU32);
    MaskReg pregA = CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Compares<int32_t, CMPMODE::LT>(pregA, kReg, 64, pregAll);
    Reg::Select<T>(oReg, gRegA, gRegB, pregA);
    uint32_t sreg = m;
    MaskReg pregF = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregF);
}

/*!
 * \brief gap 双段(S1G case2.1) 直写。
 * 前置: 尾段值在 inUb[0..gEnd], 头段值在 inUb[headOff..headOff+headLen)
 * (headOff 8 元素对齐 = 32B, 由 CopyIn gap 布局约定; 最坏 span = headOff+headLen <= 128)。
 * 行序列: r < headLen 取 inUb[headOff + r], 其余行取 inUb[r - headLen]。
 */
template <typename T, typename INPUT_T>
__simd_vf__ inline void SinkGapDirectVF(__ubuf__ T *maxUb, __ubuf__ INPUT_T *inUb, uint32_t headOff, uint32_t headLen,
                                        uint32_t m)
{
    RegTensor<T> fHeadA;
    RegTensor<T> fHeadB;
    RegTensor<T> fTail;
    RegTensor<T> gHeadA;
    RegTensor<T> gHeadB;
    RegTensor<T> gHead;
    RegTensor<T> gTail;
    RegTensor<T> oReg;
    RegTensor<int32_t> rReg;
    RegTensor<int32_t> rSub;
    RegTensor<int32_t> idxTail;
    MaskReg pregAll = CreateMask<T, Reg::MaskPattern::ALL>();
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fHeadA, inUb + headOff);
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fHeadB, inUb + headOff + 64);
    LoadAlign<T, Reg::LoadDist::DIST_NORM>(fTail, inUb);
    // 行号与两段的 lane 索引: 头段 lane = r(宽模式 r / r-64), 尾段 lane = r - headLen
    Reg::Arange<int32_t>(rReg, 0);
    Reg::Adds<int32_t>(rSub, rReg, -64, pregAll);
    Reg::Adds<int32_t>(idxTail, rReg, -static_cast<int32_t>(headLen), pregAll);
    // 头段宽模式: r < 64 取 A 段, 否则取 B 段
    RegTensor<uint32_t> &rU32 = (RegTensor<uint32_t> &)rReg;
    RegTensor<uint32_t> &rSubU32 = (RegTensor<uint32_t> &)rSub;
    Reg::Gather<T>(gHeadA, fHeadA, rU32);
    Reg::Gather<T>(gHeadB, fHeadB, rSubU32);
    MaskReg pregHA = CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Compares<int32_t, CMPMODE::LT>(pregHA, rReg, 64, pregAll);
    Reg::Select<T>(gHead, gHeadA, gHeadB, pregHA);
    // 头尾拼合: r < headLen 取头段, 否则取尾段(尾索引负值 huge 越界, 被 Select 掩掉)
    RegTensor<uint32_t> &idxTailU32 = (RegTensor<uint32_t> &)idxTail;
    Reg::Gather<T>(gTail, fTail, idxTailU32);
    MaskReg pregHead = CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Compares<int32_t, CMPMODE::LT>(pregHead, rReg, static_cast<int32_t>(headLen), pregAll);
    Reg::Select<T>(oReg, gHead, gTail, pregHead);
    uint32_t sreg = m;
    MaskReg pregF = UpdateMask<T>(sreg);
    StoreAlign<T, Reg::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregF);
}

/*!
 * \brief sink 注入 dispatcher(V 锁内调用), 全场景直写后即返回。
 * \param maxUb       [out] softmaxMax 槽位(64 float, 唯一 REG->UB 落点)
 * \param sinkTmpUb   [in]  float 值区(CopyIn 布局: GS1 紧凑 gCount 值 / S1G 4 情况)
 * \param gs1Start    [in]  本 block 的 GS1 轴起始(相位换算基准)
 * \param actVecMSize [in]  有效行数 m(<= 64)
 * \param actS1Size   [in]  GS1 布局的 s1 维块大小
 * \param gSize       [in]  GQA 组大小
 */
template <typename T, typename INPUT_T, bool IS_GS1_LAYOUT>
__aicore__ inline void SinkExpandMaxVf(const LocalTensor<T> &maxUb, const LocalTensor<INPUT_T> &sinkTmpUb,
                                       uint32_t gs1Start, uint32_t actVecMSize, uint32_t actS1Size, uint32_t gSize)
{
    if (actVecMSize == 0) {
        return;
    }
    __ubuf__ T *maxUbPtr = (__ubuf__ T *)maxUb.GetPhyAddr();
    __ubuf__ INPUT_T *inUb = (__ubuf__ INPUT_T *)sinkTmpUb.GetPhyAddr();

    if constexpr (IS_GS1_LAYOUT) {
        uint32_t s0 = gs1Start % actS1Size;
        SinkGs1DirectVF<T, INPUT_T>(maxUbPtr, inUb, s0, actS1Size, actVecMSize);
        return;
    }
    if (gSize == 1U) {
        SinkBrcVF<T, INPUT_T>(maxUbPtr, inUb, actVecMSize);
        return;
    }
    uint32_t gStart = gs1Start % gSize;
    uint32_t s1Start = gs1Start / gSize;
    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End % gSize;
    uint32_t s1End = gs1End / gSize;
    if (s1Start == s1End) {
        // case1: 单 s1 行, 值区连续
        SinkContigDirectVF<T, INPUT_T>(maxUbPtr, inUb, actVecMSize);
        return;
    }
    if (s1Start + 1 == s1End && gStart > gEnd) {
        // case2.1: 头段 [gStart, gSize) 与尾段 [0, gEnd] 不重叠, CopyIn 按 gap 布局分段存放
        uint32_t headOff = (gEnd + 1 + 7U) & ~7U;
        uint32_t headLen = gSize - gStart;
        SinkGapDirectVF<T, INPUT_T>(maxUbPtr, inUb, headOff, headLen, actVecMSize);
        return;
    }
    // case2.2/3: 值区回绕(m 跨多个 g 周期)
    SinkRotDirectVF<T, INPUT_T>(maxUbPtr, inUb, gStart, gSize, actVecMSize);
}
} // namespace AttentionCommon

#endif // ATTN_SINK_GS1_H_
