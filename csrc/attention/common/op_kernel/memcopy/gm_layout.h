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
 * \file gm_layout.h
 * \brief
 */
#ifndef GM_LAYOUT_H
#define GM_LAYOUT_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

// ----------------------------------------------GmLayout--------------------------------
enum class GmFormat {
    BSNGD = 0,
    BNGSD = 1,
    NGBSD = 2,
    TNGD = 3,
    NGTD = 4,
    BSND = 5,
    BNSD = 6,
    TND = 7,
    NTD = 8,
    PA_BnBsND = 9,
    PA_BnNBsD = 10,
    PA_NZ = 11,
    NGD = 12, // post_quant
    ND = 13, //antiquant no PA
    BS2 = 14,
    BNS2 = 15,
    PA_BnBs = 16, //antiquant PA
    PA_BnNBs = 17,
    BN2GS1S2 = 18, //PSE_GmFormat
    SBNGD = 19,
    SBND = 20,
    NTGD = 21,
    PA_NZ_K_SCALE = 22,
};

template <GmFormat FORMAT>
struct GmLayout {
};

template <>
struct GmLayout<GmFormat::BSNGD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t g, uint32_t s, uint32_t d) {
        shape = AscendC::MakeShape(b, n, g, s, d);
        uint64_t dStride = 1;
        uint64_t gStride = dStride * d;
        uint64_t nStride = gStride * g;
        uint64_t sStride = nStride * n;
        uint64_t bStride = sStride * s;
        stride = AscendC::MakeStride(bStride, nStride, gStride, sStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::BNGSD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t g, uint32_t s, uint32_t d) {
        shape = AscendC::MakeShape(b, n, g, s, d);
        uint64_t dStride = 1;
        uint64_t sStride = dStride * d;
        uint64_t gStride = sStride * s;
        uint64_t nStride = gStride * g;
        uint64_t bStride = nStride * n;
        stride = AscendC::MakeStride(bStride, nStride, gStride, sStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::NGBSD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t g, uint32_t s, uint32_t d) {
        shape = AscendC::MakeShape(b, n, g, s, d);
        uint64_t dStride = 1;
        uint64_t sStride = dStride * d;
        uint64_t bStride = sStride * s;
        uint64_t gStride = bStride * b;
        uint64_t nStride = gStride * g;
        stride = AscendC::MakeStride(bStride, nStride, gStride, sStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::TNGD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t g, uint32_t d) {
        shape = AscendC::MakeShape(t, n, g, d);
        uint64_t dStride = 1;
        uint64_t gStride = dStride * d;
        uint64_t nStride = gStride * g;
        uint64_t tStride = nStride * n;
        stride = AscendC::MakeStride(tStride, nStride, gStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::NGTD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t g, uint32_t d) {
        shape = AscendC::MakeShape(t, n, g, d);
        uint64_t dStride = 1;
        uint64_t tStride = dStride * d;
        uint64_t gStride = tStride * t;
        uint64_t nStride = gStride * g;
        stride = AscendC::MakeStride(tStride, nStride, gStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::NTGD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t g, uint32_t d) {
        shape = AscendC::MakeShape(t, n, g, d);
        uint64_t dStride = 1;
        uint64_t gStride = dStride * d;
        uint64_t tStride = gStride * g;
        uint64_t nStride = tStride * t;
        stride = AscendC::MakeStride(tStride, nStride, gStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::BSND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t s, uint32_t d) {
        shape = AscendC::MakeShape(b, n, s, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t sStride = nStride * n;
        uint64_t bStride = sStride * s;
        stride = AscendC::MakeStride(bStride, nStride, sStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::BNSD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t s, uint32_t d) {
        shape = AscendC::MakeShape(b, n, s, d);
        uint64_t dStride = 1;
        uint64_t sStride = dStride * d;
        uint64_t nStride = sStride * s;
        uint64_t bStride = nStride * n;
        stride = AscendC::MakeStride(bStride, nStride, sStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::TND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t d) {
        shape = AscendC::MakeShape(t, n, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t tStride = nStride * n;
        stride = AscendC::MakeStride(tStride, nStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::NTD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t d) {
        shape = AscendC::MakeShape(t, n, d);
        uint64_t dStride = 1;
        uint64_t tStride = dStride * d;
        uint64_t nStride = tStride * t;
        stride = AscendC::MakeStride(tStride, nStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::PA_BnBsND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize, uint32_t d) {
        shape = AscendC::MakeShape(n, blockSize, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t bsStride = nStride * n;
        uint64_t bnStride = bsStride * blockSize;
        stride = AscendC::MakeStride(bnStride, nStride, bsStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::PA_BnNBsD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize, uint32_t d) {
        shape = AscendC::MakeShape(n, blockSize, d);
        uint64_t dStride = 1;
        uint64_t bsStride = dStride * d;
        uint64_t nStride = bsStride * blockSize;
        uint64_t bnStride = nStride * n;
        stride = AscendC::MakeStride(bnStride, nStride, bsStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::PA_NZ> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize, uint32_t d1, uint32_t d0) {
        shape = AscendC::MakeShape(n, d1, blockSize, d0);
        uint64_t d0Stride = 1;
        uint64_t bsStride = d0Stride * d0;
        uint64_t d1Stride = bsStride * blockSize;
        uint64_t nStride = d1Stride * d1;
        uint64_t bnStride = nStride * n;
        stride = AscendC::MakeStride(bnStride, nStride, d1Stride, bsStride, d0Stride);
    }
};

template <>
struct GmLayout<GmFormat::PA_NZ_K_SCALE> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize1, uint32_t d, uint32_t blockSize0) {
        shape = AscendC::MakeShape(n, blockSize1, d, blockSize0);
        uint64_t bs0Stride = 1;
        uint64_t dStride = bs0Stride * blockSize0;
        uint64_t bs1Stride = dStride * d;
        uint64_t nStride = bs1Stride * blockSize1;
        uint64_t bnStride = nStride * n;
        stride = AscendC::MakeStride(bnStride, nStride, bs1Stride, dStride, bs0Stride);
    }
};

// post_quant
template <>
struct GmLayout<GmFormat::NGD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t g, uint32_t d) {
        shape = AscendC::MakeShape(n, g, d);
        uint64_t dStride = 1;
        uint64_t gStride = dStride * d;
        uint64_t nStride = gStride * g;
        stride = AscendC::MakeStride(nStride, gStride, dStride);
    }
};

//antiquant
template <>
struct GmLayout<GmFormat::ND> {
    AscendC::Shape<uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t d) {
        shape = AscendC::MakeShape(n, d);

        uint64_t dStride = 1;
        uint64_t nStride = dStride * d; //headDim
        stride = AscendC::MakeStride(nStride, dStride);
    }
};
template <>
struct GmLayout<GmFormat::BS2> {
    AscendC::Shape<uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t s) {
        shape = AscendC::MakeShape(b, s);

        uint64_t sStride = 1;
        uint64_t bStride = sStride * s;

        stride = AscendC::MakeStride(bStride, sStride);
    }
};
template <>
struct GmLayout<GmFormat::BNS2> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t s) {
        shape = AscendC::MakeShape(b, n, s);

        uint64_t sStride = 1;
        uint64_t nStride = sStride * s;
        uint64_t bStride = nStride * n;

        stride = AscendC::MakeStride(bStride, nStride, sStride);
    }
};
template <>
struct GmLayout<GmFormat::PA_BnBs> {
    AscendC::Shape<uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t blockSize) {
        shape = AscendC::MakeShape(blockSize);

        uint64_t bsStride = 1;
        uint64_t bnStride = bsStride * blockSize;
        stride = AscendC::MakeStride(bnStride, bsStride);
    }
};
template <>
struct GmLayout<GmFormat::PA_BnNBs> {
    AscendC::Shape<uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize) {
        shape = AscendC::MakeShape(n, blockSize);

        uint64_t bsStride = 1;
        uint64_t nStride = bsStride * blockSize;
        uint64_t bnStride = nStride * n; //blockSize * kvHeadNum
        stride = AscendC::MakeStride(bnStride, nStride, bsStride);
    }
};

//PSE_GmLayout
template <>
struct GmLayout<GmFormat::BN2GS1S2> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t g, uint32_t s1, uint32_t s2)
    {
        shape = AscendC::MakeShape(b, n, g, s1, s2);
        uint64_t s2Stride = 1;
        uint64_t s1Stride = s2Stride * s2;
        uint64_t gStride = s1Stride * s1;
        uint64_t nStride = gStride * g;
        uint64_t bStride = nStride * n;
        stride = AscendC::MakeStride(bStride, nStride, gStride, s1Stride, s2Stride);
    }
};

template <>
struct GmLayout<GmFormat::SBNGD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t g, uint32_t s, uint32_t d) {
        shape = AscendC::MakeShape(b, n, g, s, d);
        uint64_t dStride = 1;
        uint64_t gStride = dStride * d;
        uint64_t nStride = gStride * g;
        uint64_t bStride = nStride * n;
        uint64_t sStride = bStride * b;
        stride = AscendC::MakeStride(bStride, nStride, gStride, sStride, dStride);
    }
};

template <>
struct GmLayout<GmFormat::SBND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t s, uint32_t d) {
        shape = AscendC::MakeShape(b, n, s, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t bStride = nStride * n;
        uint64_t sStride = bStride * b;
        stride = AscendC::MakeStride(bStride, nStride, sStride, dStride);
    }
};

#endif