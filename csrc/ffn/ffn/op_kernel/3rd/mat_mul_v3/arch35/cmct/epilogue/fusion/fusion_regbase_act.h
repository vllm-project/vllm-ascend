/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software and you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file fusion_regbase_act.h
 * \brief FFN epilogue 激活的 RegBase实现
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Cmct {
namespace Gemm {
namespace Block {
namespace RegBaseAct {
using namespace AscendC::Reg;

// ---- gelu(erf)：y = 0.5*x*(1+erf(x/√2)) ----
template <typename DstT>
__simd_callee__ inline void GeluErfChunkB16(__ubuf__ DstT *dstUb, __ubuf__ float *srcUb, uint32_t off,
                                            MaskReg mask)
{
    static constexpr CastTrait castB32ToB16 = {
        RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    // erf Pade 系数
    constexpr float P0 = 0.29639384698e5f, P1 = 0.50637915060e4f, P2 = 0.13938061484e4f;
    constexpr float P3 = 0.10162808918e3f, P4 = 0.75517016694e1f, P5 = 0.053443748819f;
    constexpr float Q0 = 0.26267224157e5f, Q1 = 0.13243365831e5f, Q2 = 0.30231248150e4f;
    constexpr float Q3 = 0.39856963806e3f, Q4 = 0.31212858877e2f;

    RegTensor<float> srcVreg;
    RegTensor<float> clipReg;
    RegTensor<float> x2Reg;
    RegTensor<float> pqReg;
    RegTensor<float> yReg;
    RegTensor<DstT> dstVreg;
    LoadAlign<float>(srcVreg, srcUb + off);
        Muls(yReg, srcVreg, 0.70710678118654752f, mask); // t = x/√2
        Mins(clipReg, yReg, 3.92f, mask);
        Maxs(clipReg, clipReg, -3.92f, mask);
        Mul(x2Reg, clipReg, clipReg, mask);
        Muls(pqReg, x2Reg, P5, mask);
        Adds(pqReg, pqReg, P4, mask);
        Mul(pqReg, pqReg, x2Reg, mask);
        Adds(pqReg, pqReg, P3, mask);
        Mul(pqReg, pqReg, x2Reg, mask);
        Adds(pqReg, pqReg, P2, mask);
        Mul(pqReg, pqReg, x2Reg, mask);
        Adds(pqReg, pqReg, P1, mask);
        Mul(pqReg, pqReg, x2Reg, mask);
        Adds(pqReg, pqReg, P0, mask);
        Mul(pqReg, pqReg, clipReg, mask); // P(x)
        Adds(yReg, x2Reg, Q4, mask); // Q(x2) Horner 起点
        Mul(yReg, yReg, x2Reg, mask);
        Adds(yReg, yReg, Q3, mask);
        Mul(yReg, yReg, x2Reg, mask);
        Adds(yReg, yReg, Q2, mask);
        Mul(yReg, yReg, x2Reg, mask);
        Adds(yReg, yReg, Q1, mask);
        Mul(yReg, yReg, x2Reg, mask);
        Adds(yReg, yReg, Q0, mask); // Q(x2)
        Div(yReg, pqReg, yReg, mask); // erf(t)
        Adds(yReg, yReg, 1.0f, mask);
        Muls(yReg, yReg, 0.5f, mask);
        Mul(yReg, yReg, srcVreg, mask); // y = 0.5*x*(1+erf)
        Cast<DstT, float, castB32ToB16>(dstVreg, yReg, mask);
    StoreAlign<DstT, StoreDist::DIST_PACK_B32>(dstUb + off, dstVreg, mask);
}

// 双 chunk 体：两套独立寄存器组、手工交错发射。单 chunk 内 P/Q 两条 Horner 链本已并行，
// 双 chunk 后共 4 条无依赖链，掩盖 VF 乘加延迟（串行链延迟是 erf-Pade 的 ALU 瓶颈主因）。
template <typename DstT>
__simd_callee__ inline void GeluErfChunk2B16(__ubuf__ DstT *dstUb, __ubuf__ float *srcUb, uint32_t off,
                                             MaskReg mask)
{
    static constexpr CastTrait castB32ToB16 = {
        RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr uint32_t E = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
    constexpr float P0 = 0.29639384698e5f, P1 = 0.50637915060e4f, P2 = 0.13938061484e4f;
    constexpr float P3 = 0.10162808918e3f, P4 = 0.75517016694e1f, P5 = 0.053443748819f;
    constexpr float Q0 = 0.26267224157e5f, Q1 = 0.13243365831e5f, Q2 = 0.30231248150e4f;
    constexpr float Q3 = 0.39856963806e3f, Q4 = 0.31212858877e2f;

    RegTensor<float> srcV0;
    RegTensor<float> srcV1;
    RegTensor<float> clip0;
    RegTensor<float> clip1;
    RegTensor<float> x20;
    RegTensor<float> x21;
    RegTensor<float> pq0;
    RegTensor<float> pq1;
    RegTensor<float> y0;
    RegTensor<float> y1;
    RegTensor<DstT> dst0;
    RegTensor<DstT> dst1;
    const uint32_t off1 = off + E;
    LoadAlign<float>(srcV0, srcUb + off);
    LoadAlign<float>(srcV1, srcUb + off1);
    Muls(y0, srcV0, 0.70710678118654752f, mask);
    Muls(y1, srcV1, 0.70710678118654752f, mask);
    Mins(clip0, y0, 3.92f, mask);
    Mins(clip1, y1, 3.92f, mask);
    Maxs(clip0, clip0, -3.92f, mask);
    Maxs(clip1, clip1, -3.92f, mask);
    Mul(x20, clip0, clip0, mask);
    Mul(x21, clip1, clip1, mask);
    Muls(pq0, x20, P5, mask);
    Muls(pq1, x21, P5, mask);
    Adds(pq0, pq0, P4, mask);
    Adds(pq1, pq1, P4, mask);
    Mul(pq0, pq0, x20, mask);
    Mul(pq1, pq1, x21, mask);
    Adds(pq0, pq0, P3, mask);
    Adds(pq1, pq1, P3, mask);
    Mul(pq0, pq0, x20, mask);
    Mul(pq1, pq1, x21, mask);
    Adds(pq0, pq0, P2, mask);
    Adds(pq1, pq1, P2, mask);
    Mul(pq0, pq0, x20, mask);
    Mul(pq1, pq1, x21, mask);
    Adds(pq0, pq0, P1, mask);
    Adds(pq1, pq1, P1, mask);
    Mul(pq0, pq0, x20, mask);
    Mul(pq1, pq1, x21, mask);
    Adds(pq0, pq0, P0, mask);
    Adds(pq1, pq1, P0, mask);
    Mul(pq0, pq0, clip0, mask); // P(x)
    Mul(pq1, pq1, clip1, mask);
    Adds(y0, x20, Q4, mask); // Q(x2) Horner 起点
    Adds(y1, x21, Q4, mask);
    Mul(y0, y0, x20, mask);
    Mul(y1, y1, x21, mask);
    Adds(y0, y0, Q3, mask);
    Adds(y1, y1, Q3, mask);
    Mul(y0, y0, x20, mask);
    Mul(y1, y1, x21, mask);
    Adds(y0, y0, Q2, mask);
    Adds(y1, y1, Q2, mask);
    Mul(y0, y0, x20, mask);
    Mul(y1, y1, x21, mask);
    Adds(y0, y0, Q1, mask);
    Adds(y1, y1, Q1, mask);
    Mul(y0, y0, x20, mask);
    Mul(y1, y1, x21, mask);
    Adds(y0, y0, Q0, mask); // Q(x2)
    Adds(y1, y1, Q0, mask);
    Div(y0, pq0, y0, mask); // erf(t)
    Div(y1, pq1, y1, mask);
    Adds(y0, y0, 1.0f, mask);
    Adds(y1, y1, 1.0f, mask);
    Muls(y0, y0, 0.5f, mask);
    Muls(y1, y1, 0.5f, mask);
    Mul(y0, y0, srcV0, mask); // y = 0.5*x*(1+erf)
    Mul(y1, y1, srcV1, mask);
    Cast<DstT, float, castB32ToB16>(dst0, y0, mask);
    Cast<DstT, float, castB32ToB16>(dst1, y1, mask);
    StoreAlign<DstT, StoreDist::DIST_PACK_B32>(dstUb + off, dst0, mask);
    StoreAlign<DstT, StoreDist::DIST_PACK_B32>(dstUb + off1, dst1, mask);
}

template <typename DstT>
__simd_vf__ inline void RegGeluErfB16(__ubuf__ DstT *dstUb, __ubuf__ float *srcUb, uint32_t count)
{
    static constexpr uint32_t E = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
    const uint32_t fullLoops = count / E;
    uint32_t tailRemain = count % E;
    // mask 提升：满块复用循环外算好的全 1 掩码，仅尾块按余量精确截断
    // （UpdateMask 形参为非 const 引用，须用可变左值——bisheng 陷阱）
    uint32_t fullCnt = E;
    MaskReg fullMask = UpdateMask<float>(fullCnt);
    const uint32_t pairs = fullLoops / 2;
    for (uint32_t i = 0; i < pairs; ++i) {
        GeluErfChunk2B16<DstT>(dstUb, srcUb, 2 * i * E, fullMask);
    }
    if ((fullLoops & 1U) != 0U) {
        GeluErfChunkB16<DstT>(dstUb, srcUb, (fullLoops - 1) * E, fullMask);
    }
    if (tailRemain != 0) {
        MaskReg tailMask = UpdateMask<float>(tailRemain);
        GeluErfChunkB16<DstT>(dstUb, srcUb, fullLoops * E, tailMask);
    }
}

// ---- silu：y = x/(1+e^-x) ----
template <typename DstT>
__simd_vf__ inline void RegSiluB16(__ubuf__ DstT *dstUb, __ubuf__ float *srcUb, uint32_t count)
{
    static constexpr CastTrait castB32ToB16 = {
        RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr uint32_t E = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
    RegTensor<float> srcVreg;
    RegTensor<float> tReg;
    RegTensor<DstT> dstVreg;
    const uint32_t loops = AscendC::CeilDivision(count, E);
    for (uint32_t i = 0; i < loops; ++i) {
        const uint32_t off = i * E;
        const uint32_t remain = (count - off > E) ? E : (count - off);
        uint32_t tailRemain = remain;
        MaskReg mask = UpdateMask<float>(tailRemain); // 尾块精确截断，StoreAlign 不越界
        LoadAlign<float>(srcVreg, srcUb + off);
        Muls(tReg, srcVreg, -1.0f, mask);
        Exp(tReg, tReg, mask);
        Adds(tReg, tReg, 1.0f, mask);
        Div(tReg, srcVreg, tReg, mask);
        Cast<DstT, float, castB32ToB16>(dstVreg, tReg, mask);
        StoreAlign<DstT, StoreDist::DIST_PACK_B32>(dstUb + off, dstVreg, mask);
    }
}

// ---- swiglu 单 matmul：行布局 [g(cols/2) | u(cols/2)]，y = silu(g)*u，输出半宽 ----
template <typename DstT>
__simd_vf__ inline void RegSwigluSingleB16(__ubuf__ DstT *dstUb, __ubuf__ float *srcUb, uint32_t rows,
                                           uint32_t colsPadded)
{
    static constexpr CastTrait castB32ToB16 = {
        RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr uint32_t E = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
    const uint32_t halfW = colsPadded / 2;
    const uint32_t loops = AscendC::CeilDivision(halfW, E);
    RegTensor<float> gVreg;
    RegTensor<float> uVreg;
    RegTensor<float> tReg;
    RegTensor<DstT> dstVreg;
    for (uint32_t r = 0; r < rows; ++r) {
        __ubuf__ float *srcRow = srcUb + static_cast<size_t>(r) * colsPadded;
        __ubuf__ DstT *dstRow = dstUb + static_cast<size_t>(r) * halfW;
        for (uint32_t i = 0; i < loops; ++i) {
            const uint32_t off = i * E;
            const uint32_t remain = (halfW - off > E) ? E : (halfW - off);
            uint32_t tailRemain = remain;
        MaskReg mask = UpdateMask<float>(tailRemain); // 尾块精确截断，StoreAlign 不越界
            LoadAlign<float>(gVreg, srcRow + off);             // gate 前半
            LoadAlign<float>(uVreg, srcRow + halfW + off);     // up 后半
            Muls(tReg, gVreg, -1.0f, mask);
            Exp(tReg, tReg, mask);
            Adds(tReg, tReg, 1.0f, mask);
            Div(tReg, gVreg, tReg, mask); // silu(g)
            Mul(tReg, tReg, uVreg, mask); // * u
            Cast<DstT, float, castB32ToB16>(dstVreg, tReg, mask);
            StoreAlign<DstT, StoreDist::DIST_PACK_B32>(dstRow + off, dstVreg, mask);
        }
    }
}
} // namespace RegBaseAct
} // namespace Block
} // namespace Gemm
} // namespace Cmct
