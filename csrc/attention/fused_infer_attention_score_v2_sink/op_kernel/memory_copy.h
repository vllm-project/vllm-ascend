/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file memory_copy.h
   GM->L1
   PA
   PARope
 * \brief
 */
#ifndef MEMMORY_COPY_H
#define MEMMORY_COPY_H
#include "fia_public_define.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
using namespace matmul;

constexpr uint32_t HALF_SIZE_DIVISOR = 2;
constexpr uint32_t ND_MATRIX_STRIDE_LIMIT =
    65536; // Mutil ND2NZ搬运时，Nd2NzParams支持的srcNdMatrixStride的取值范围为[0, 65536]，单位为元素
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
    ND = 13, // antiquant no PA
    BS2 = 14,
    BNS2 = 15,
    PA_BnBs = 16, // antiquant PA
    PA_BnNBs = 17,
    BN2GS1S2 = 18 // PSE_GmFormat
};

template <GmFormat FORMAT> struct GmLayout {
};

template <> struct GmLayout<GmFormat::BSNGD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t g, uint32_t s, uint32_t d)
    {
        shape = AscendC::MakeShape(b, n, g, s, d);
        uint64_t dStride = 1;
        uint64_t gStride = dStride * d;
        uint64_t nStride = gStride * g;
        uint64_t sStride = nStride * n;
        uint64_t bStride = sStride * s;
        stride = AscendC::MakeStride(bStride, nStride, gStride, sStride, dStride);
    }
};

template <> struct GmLayout<GmFormat::BNGSD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t g, uint32_t s, uint32_t d)
    {
        shape = AscendC::MakeShape(b, n, g, s, d);
        uint64_t dStride = 1;
        uint64_t sStride = dStride * d;
        uint64_t gStride = sStride * s;
        uint64_t nStride = gStride * g;
        uint64_t bStride = nStride * n;
        stride = AscendC::MakeStride(bStride, nStride, gStride, sStride, dStride);
    }
};

template <> struct GmLayout<GmFormat::NGBSD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t g, uint32_t s, uint32_t d)
    {
        shape = AscendC::MakeShape(b, n, g, s, d);
        uint64_t dStride = 1;
        uint64_t sStride = dStride * d;
        uint64_t bStride = sStride * s;
        uint64_t gStride = bStride * b;
        uint64_t nStride = gStride * g;
        stride = AscendC::MakeStride(bStride, nStride, gStride, sStride, dStride);
    }
};

template <> struct GmLayout<GmFormat::TNGD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t g, uint32_t d)
    {
        shape = AscendC::MakeShape(t, n, g, d);
        uint64_t dStride = 1;
        uint64_t gStride = dStride * d;
        uint64_t nStride = gStride * g;
        uint64_t tStride = nStride * n;
        stride = AscendC::MakeStride(tStride, nStride, gStride, dStride);
    }
};

template <> struct GmLayout<GmFormat::NGTD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t g, uint32_t d)
    {
        shape = AscendC::MakeShape(t, n, g, d);
        uint64_t dStride = 1;
        uint64_t tStride = dStride * d;
        uint64_t gStride = tStride * t;
        uint64_t nStride = gStride * g;
        stride = AscendC::MakeStride(tStride, nStride, gStride, dStride);
    }
};

template <> struct GmLayout<GmFormat::BSND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t s, uint32_t d)
    {
        shape = AscendC::MakeShape(b, n, s, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t sStride = nStride * n;
        uint64_t bStride = sStride * s;
        stride = AscendC::MakeStride(bStride, nStride, sStride, dStride);
    }
};

template <> struct GmLayout<GmFormat::BNSD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t s, uint32_t d)
    {
        shape = AscendC::MakeShape(b, n, s, d);
        uint64_t dStride = 1;
        uint64_t sStride = dStride * d;
        uint64_t nStride = sStride * s;
        uint64_t bStride = nStride * n;
        stride = AscendC::MakeStride(bStride, nStride, sStride, dStride);
    }
};

template <> struct GmLayout<GmFormat::TND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t d)
    {
        shape = AscendC::MakeShape(t, n, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t tStride = nStride * n;
        stride = AscendC::MakeStride(tStride, nStride, dStride);
    }
};

template <> struct GmLayout<GmFormat::NTD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t t, uint32_t n, uint32_t d)
    {
        shape = AscendC::MakeShape(t, n, d);
        uint64_t dStride = 1;
        uint64_t tStride = dStride * d;
        uint64_t nStride = tStride * t;
        stride = AscendC::MakeStride(tStride, nStride, dStride);
    }
};

template <> struct GmLayout<GmFormat::PA_BnBsND> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize, uint32_t d)
    {
        shape = AscendC::MakeShape(n, blockSize, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t bsStride = nStride * n;
        uint64_t bnStride = bsStride * blockSize;
        stride = AscendC::MakeStride(bnStride, nStride, bsStride, dStride);
    }
    // 使用传入的 stride 数组构造非连续布局
    // stride[0-2] = {bnStride, blockSizeStride, HStride}
    __aicore__ inline void MakeLayoutUnContiguous(uint32_t n, uint32_t blockSize, uint32_t d, const uint64_t *strideTmp)
    {
        shape = AscendC::MakeShape(n, blockSize, d);
        uint64_t dStride = 1;
        uint64_t nStride = dStride * d;
        uint64_t bsStride = strideTmp[1];
        uint64_t bnStride = strideTmp[0];
        this->stride = AscendC::MakeStride(bnStride, nStride, bsStride, dStride);
    }
};

template <> struct GmLayout<GmFormat::PA_BnNBsD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize, uint32_t d)
    {
        shape = AscendC::MakeShape(n, blockSize, d);
        uint64_t dStride = 1;
        uint64_t bsStride = dStride * d;
        uint64_t nStride = bsStride * blockSize;
        uint64_t bnStride = nStride * n;
        stride = AscendC::MakeStride(bnStride, nStride, bsStride, dStride);
    }
    // 使用传入的 stride 数组构造非连续布局
    // stride[0-3] = {bnStride, nStride, blockSizeStride, dStride}
    __aicore__ inline void MakeLayoutUnContiguous(uint32_t n, uint32_t blockSize, uint32_t d, const uint64_t *strideTmp)
    {
        shape = AscendC::MakeShape(n, blockSize, d);
        uint64_t dStride = strideTmp[3];
        uint64_t bsStride = strideTmp[2];
        uint64_t nStride = strideTmp[1];
        uint64_t bnStride = strideTmp[0];
        stride = AscendC::MakeStride(bnStride, nStride, bsStride, dStride);
    }
};

template <> struct GmLayout<GmFormat::PA_NZ> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize, uint32_t d1, uint32_t d0)
    {
        shape = AscendC::MakeShape(n, d1, blockSize, d0);
        uint64_t d0Stride = 1;
        uint64_t bsStride = d0Stride * d0;
        uint64_t d1Stride = bsStride * blockSize;
        uint64_t nStride = d1Stride * d1;
        uint64_t bnStride = nStride * n;
        stride = AscendC::MakeStride(bnStride, nStride, d1Stride, bsStride, d0Stride);
    }
    // 使用传入的 stride 数组构造非连续布局
    // stride[0-4] = {bnStride, nStride, d1Stride, blockSizeStride, d0Stride}
    __aicore__ inline void
    MakeLayoutUnContiguous(uint32_t n, uint32_t blockSize, uint32_t d1, uint32_t d0, const uint64_t *strideTmp)
    {
        shape = AscendC::MakeShape(n, d1, blockSize, d0);
        uint64_t d0Stride = strideTmp[4];
        uint64_t bsStride = strideTmp[3];
        uint64_t d1Stride = strideTmp[2];
        uint64_t nStride = strideTmp[1];
        uint64_t bnStride = strideTmp[0];
        stride = AscendC::MakeStride(bnStride, nStride, d1Stride, bsStride, d0Stride);
    }
};

// post_quant
template <> struct GmLayout<GmFormat::NGD> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t g, uint32_t d)
    {
        shape = AscendC::MakeShape(n, g, d);
        uint64_t dStride = 1;
        uint64_t gStride = dStride * d;
        uint64_t nStride = gStride * g;
        stride = AscendC::MakeStride(nStride, gStride, dStride);
    }
};

// antiquant
template <> struct GmLayout<GmFormat::ND> {
    AscendC::Shape<uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t d)
    {
        shape = AscendC::MakeShape(n, d);

        uint64_t dStride = 1;
        uint64_t nStride = dStride * d; // headDim
        stride = AscendC::MakeStride(nStride, dStride);
    }
};
template <> struct GmLayout<GmFormat::BS2> {
    AscendC::Shape<uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t s)
    {
        shape = AscendC::MakeShape(b, s);

        uint64_t sStride = 1;
        uint64_t bStride = sStride * s;

        stride = AscendC::MakeStride(bStride, sStride);
    }
};
template <> struct GmLayout<GmFormat::BNS2> {
    AscendC::Shape<uint32_t, uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t b, uint32_t n, uint32_t s)
    {
        shape = AscendC::MakeShape(b, n, s);

        uint64_t sStride = 1;
        uint64_t nStride = sStride * s;
        uint64_t bStride = nStride * n;

        stride = AscendC::MakeStride(bStride, nStride, sStride);
    }
};
template <> struct GmLayout<GmFormat::PA_BnBs> {
    AscendC::Shape<uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t blockSize)
    {
        shape = AscendC::MakeShape(blockSize);

        uint64_t bsStride = 1;
        uint64_t bnStride = bsStride * blockSize;
        stride = AscendC::MakeStride(bnStride, bsStride);
    }
};
template <> struct GmLayout<GmFormat::PA_BnNBs> {
    AscendC::Shape<uint32_t, uint32_t> shape;
    AscendC::Stride<uint64_t, uint64_t, uint64_t> stride;

    __aicore__ inline GmLayout() = default;
    __aicore__ inline void MakeLayout(uint32_t n, uint32_t blockSize)
    {
        shape = AscendC::MakeShape(n, blockSize);

        uint64_t bsStride = 1;
        uint64_t nStride = bsStride * blockSize;
        uint64_t bnStride = nStride * n; // blockSize * kvHeadNum
        stride = AscendC::MakeStride(bnStride, nStride, bsStride);
    }
};

// PSE_GmLayout
template <> struct GmLayout<GmFormat::BN2GS1S2> {
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

// ----------------------------------------------ActualSeqLensParser--------------------------------
enum class ActualSeqLensMode {
    BY_BATCH = 0,
    ACCUM = 1,
};

template <FIA_LAYOUT LAYOUT_T> __aicore__ inline constexpr ActualSeqLensMode GetQActSeqMode()
{
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}
template <FIA_LAYOUT LAYOUT_T, const bool PAGE_ATTENTION>
__aicore__ inline constexpr ActualSeqLensMode GetKvActSeqMode()
{
    if constexpr (PAGE_ATTENTION) {
        return ActualSeqLensMode::BY_BATCH;
    }
    if constexpr (LAYOUT_T == FIA_LAYOUT::TND || LAYOUT_T == FIA_LAYOUT::NTD) {
        return ActualSeqLensMode::ACCUM;
    } else {
        return ActualSeqLensMode::BY_BATCH;
    }
}

template <ActualSeqLensMode MODE> class ActualSeqLensParser {
};

template <> class ActualSeqLensParser<ActualSeqLensMode::ACCUM> {
  public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void
    Init(GlobalTensor<uint64_t> actualSeqLengthsGm, uint32_t actualLenDims, uint64_t defaultVal = 0)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
    }

    __aicore__ inline uint64_t GetTBase(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return 0;
        }
        return actualSeqLengthsGm.GetValue(bIdx - 1);
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (bIdx == 0) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return (actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1));
    }

    __aicore__ inline uint64_t GetTSize() const
    {
        return actualSeqLengthsGm.GetValue(actualLenDims - 1);
    }

  private:
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    uint32_t actualLenDims;
};

template <> class ActualSeqLensParser<ActualSeqLensMode::BY_BATCH> {
  public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<uint64_t> actualSeqLengthsGm, uint32_t actualLenDims, uint64_t defaultVal)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
        this->defaultVal = defaultVal;
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (actualLenDims == 0) {
            return defaultVal;
        }
        if (actualLenDims == 1) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return actualSeqLengthsGm.GetValue(bIdx);
    }

    __aicore__ inline uint32_t GetActualLenDims() const
    {
        return actualLenDims;
    }

  private:
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    uint32_t actualLenDims;
    uint64_t defaultVal;
};

// ----------------------------------------------BlockTableParser--------------------------------
class BlockTableParser {
  public:
    __aicore__ inline BlockTableParser() = default;

    __aicore__ inline void Init(GlobalTensor<int32_t> blockTableGm, uint32_t maxblockNumPerBatch)
    {
        this->blockTableGm = blockTableGm;
        this->maxblockNumPerBatch = maxblockNumPerBatch;
    }

    __aicore__ inline int32_t GetBlockIdx(uint32_t bIdx, uint32_t blockIdxInBatch) const
    {
        return blockTableGm.GetValue(bIdx * maxblockNumPerBatch + blockIdxInBatch);
    }

  private:
    GlobalTensor<int32_t> blockTableGm;
    uint32_t maxblockNumPerBatch;
};

// ----------------------------------------------GmLayoutParams--------------------------------
enum class FormatCategory {
    GM_Q_OUT_BNGSD = 0,
    GM_Q_OUT_TND = 1,
    GM_KV_BNSD = 2,
    GM_KV_TND = 3,
    GM_KV_PA_BNBD = 4,
    GM_KV_PA_NZ = 5,
    GM_POST_QUANT_NGD = 6, // post_quant
    GM_ANTIQ_ND = 7, // antiquant no PA
    GM_ANTIQ_BS = 8,
    GM_ANTIQ_BNS = 9,
    GM_ANTIQ_BnBs = 10, // antiquant PA
    GM_ANTIQ_BnNBs = 11,
    GM_PSE_BN2GS1S2 = 12 // PSE
};

template <GmFormat FORMAT> struct GmLayoutParams {
};

template <> struct GmLayoutParams<GmFormat::BSNGD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_BNGSD;
};

template <> struct GmLayoutParams<GmFormat::BNGSD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_BNGSD;
};

template <> struct GmLayoutParams<GmFormat::NGBSD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_BNGSD;
};

template <> struct GmLayoutParams<GmFormat::TNGD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_TND;
};

template <> struct GmLayoutParams<GmFormat::NGTD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_TND;
};

template <> struct GmLayoutParams<GmFormat::BSND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_BNSD;
};

template <> struct GmLayoutParams<GmFormat::BNSD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_BNSD;
};

template <> struct GmLayoutParams<GmFormat::TND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_TND;
};

template <> struct GmLayoutParams<GmFormat::NTD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_TND;
};

template <> struct GmLayoutParams<GmFormat::PA_BnBsND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_BNBD;
};

template <> struct GmLayoutParams<GmFormat::PA_BnNBsD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_BNBD;
};

template <> struct GmLayoutParams<GmFormat::PA_NZ> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_NZ;
};

// post_quant
template <> struct GmLayoutParams<GmFormat::NGD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_POST_QUANT_NGD;
};

// antiquant
template <> struct GmLayoutParams<GmFormat::ND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_ND;
};
template <> struct GmLayoutParams<GmFormat::BS2> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BS;
};
template <> struct GmLayoutParams<GmFormat::BNS2> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BNS;
};
template <> struct GmLayoutParams<GmFormat::PA_BnBs> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BnBs;
};
template <> struct GmLayoutParams<GmFormat::PA_BnNBs> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BnNBs;
};

// pse
template <> struct GmLayoutParams<GmFormat::BN2GS1S2> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_PSE_BN2GS1S2;
};

// ----------------------------------------------OffsetCalculator--------------------------------
template <GmFormat FORMAT, FormatCategory CATEGORY> struct OffsetCalculatorImpl {
};

template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_Q_OUT_BNGSD> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH> actualSeqLensQParser;
    bool isQPaddingFlag = false;
    uint64_t qPaddingSize = 0;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b,
                                uint32_t n2,
                                uint32_t g,
                                uint32_t s1,
                                uint32_t d,
                                GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                uint32_t actualLenQDims,
                                bool isQPaddingFlag = false,
                                uint64_t qPaddingSize = 0)
    {
        this->isQPaddingFlag = isQPaddingFlag;
        this->qPaddingSize = qPaddingSize;
        if (actualLenQDims != 0) {
            actualSeqLensQParser.Init(actualSeqLengthsGmQ, actualLenQDims, 0);
        }
        gmLayout.MakeLayout(b, n2, g, s1, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t gIdx, uint32_t s1Idx, uint32_t dIdx)
    {
        if (isQPaddingFlag) {
            s1Idx += GetDimS1() - qPaddingSize - actualSeqLensQParser.GetActualSeqLength(bIdx);
        }
        uint64_t offset = bIdx * GetStrideB() + n2Idx * GetStrideN2() + gIdx * GetStrideG() + s1Idx * GetStrideS1() +
                          dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideG()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideS1()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<4>(gmLayout.stride); // 4:代表第5个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimB()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimG()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimS1()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<4>(gmLayout.shape); // 4:代表第5个维度，索引从0开始
    }
};

template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_Q_OUT_TND> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::ACCUM> actualSeqLensQParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void
    Init(uint32_t n2, uint32_t g, uint32_t d, GlobalTensor<uint64_t> actualSeqLengthsGmQ, uint32_t actualLenQDims)
    {
        actualSeqLensQParser.Init(actualSeqLengthsGmQ, actualLenQDims);
        gmLayout.MakeLayout(actualSeqLensQParser.GetTSize(), n2, g, d);
    }

    // qTSize overload: used by NGTD (NTD output) when qT > actSeqLen[-1].
    // MakeLayout must use the real query T dimension (qTSize) so that nStride = g * qTSize * d,
    // not g * actSeqLen[-1] * d, which would produce wrong offsets when n2Idx >= 1.
    __aicore__ inline void Init(uint32_t n2,
                                uint32_t g,
                                uint32_t d,
                                GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                uint32_t actualLenQDims,
                                uint32_t qTSize)
    {
        actualSeqLensQParser.Init(actualSeqLengthsGmQ, actualLenQDims);
        gmLayout.MakeLayout(qTSize, n2, g, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t gIdx, uint32_t s1Idx, uint32_t dIdx)
    {
        uint64_t tIdx = actualSeqLensQParser.GetTBase(bIdx) + s1Idx;
        uint64_t offset = tIdx * GetStrideT() + n2Idx * GetStrideN2() + gIdx * GetStrideG() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideT()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideG()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideS1()
    {
        return GetStrideT();
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimT()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimG()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }
};

template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_BNSD> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH> actualSeqLensKVParser;
    bool isKvPaddingFlag = false;
    uint64_t kvPaddingSize = 0;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d)
    {
        gmLayout.MakeLayout(b, n2, s2, d);
    }

    __aicore__ inline void Init(uint32_t b,
                                uint32_t n2,
                                uint32_t s2,
                                uint32_t d,
                                GlobalTensor<uint64_t> actualSeqLengthsGm,
                                uint32_t actualLenKvDims,
                                bool isKvPaddingFlag = false,
                                uint64_t kvPaddingSize = 0)
    {
        this->isKvPaddingFlag = isKvPaddingFlag;
        this->kvPaddingSize = kvPaddingSize;
        if (actualLenKvDims != 0) {
            actualSeqLensKVParser.Init(actualSeqLengthsGm, actualLenKvDims, 0);
        }
        gmLayout.MakeLayout(b, n2, s2, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        if (isKvPaddingFlag) {
            s2Idx += GetDimS2() - kvPaddingSize - actualSeqLensKVParser.GetActualSeqLength(bIdx);
        }

        uint64_t offset = bIdx * GetStrideB() + n2Idx * GetStrideN2() + s2Idx * GetStrideS2() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimB()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimS2()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }
};

template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_TND> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::ACCUM> actualSeqLensKVParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void
    Init(uint32_t n2, uint32_t d, GlobalTensor<uint64_t> actualSeqLengthsGmKV, uint32_t actualLenKVDims)
    {
        actualSeqLensKVParser.Init(actualSeqLengthsGmKV, actualLenKVDims);
        gmLayout.MakeLayout(actualSeqLensKVParser.GetTSize(), n2, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t tIdx = actualSeqLensKVParser.GetTBase(bIdx) + s2Idx;
        uint64_t offset = tIdx * GetStrideT() + n2Idx * GetStrideN2() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideT()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return GetStrideT();
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimT()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }
};

template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_BNBD> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2,
                                uint32_t blockSize,
                                uint32_t d,
                                GlobalTensor<int32_t> blockTableGm,
                                uint32_t maxblockNumPerBatch,
                                const uint64_t *stride = nullptr)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        if (stride != nullptr && stride[0] > 0) {
            gmLayout.MakeLayoutUnContiguous(n2, blockSize, d, stride);
        } else {
            gmLayout.MakeLayout(n2, blockSize, d);
        }
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t blockIdxInBatch = s2Idx / GetBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = s2Idx % GetBlockSize(); // 获取在单个块上超出的行数
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
        uint64_t offset =
            blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() + bsIdx * GetStrideBlockSize() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetN2()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBlockSize()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }
};

template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_NZ> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2,
                                uint32_t blockSize,
                                uint32_t d1,
                                uint32_t d0,
                                GlobalTensor<int32_t> blockTableGm,
                                uint32_t maxblockNumPerBatch,
                                const uint64_t *stride = nullptr)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        if (stride != nullptr && stride[0] > 0) {
            gmLayout.MakeLayoutUnContiguous(n2, blockSize, d1, d0, stride);
        } else {
            gmLayout.MakeLayout(n2, blockSize, d1, d0);
        }
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t blockIdxInBatch = s2Idx / GetBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = s2Idx % GetBlockSize(); // 获取在单个块上超出的行数
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);

        uint32_t d1Idx = dIdx / GetD0();
        uint32_t d0Idx = dIdx % GetD0();
        uint64_t offset = blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() + d1Idx * GetStrideD1() +
                          bsIdx * GetStrideBlockSize() + d0Idx * GetStrideD0();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD1()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD0()
    {
        return AscendC::Std::get<4>(gmLayout.stride); // 4:代表第5个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetN2()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD1()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBlockSize()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetD0()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }
};

// post_quant
template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_POST_QUANT_NGD> {
    GmLayout<FORMAT> gmLayout;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t g, uint32_t d)
    {
        gmLayout.MakeLayout(n2, g, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t n2Idx, uint32_t gIdx, uint32_t dIdx)
    {
        uint64_t offset = n2Idx * GetStrideN2() + gIdx * GetStrideG() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideG()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint32_t GetDimN2()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimG()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimD()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }
};

// antiquant
template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_ND> {
    GmLayout<FORMAT> gmLayout;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t d)
    {
        gmLayout.MakeLayout(n2, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t offset = n2Idx * GetStrideN2() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    // Get Dim
    __aicore__ inline uint32_t GetDimN2()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimD()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }
};

template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BS> {
    GmLayout<FORMAT> gmLayout;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t s2)
    {
        gmLayout.MakeLayout(b, s2);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t offset = bIdx * GetStrideB() + s2Idx * GetStrideS2();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    // Get Dim
    __aicore__ inline uint32_t GetDimB()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimS2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }
};

template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BNS> {
    GmLayout<FORMAT> gmLayout;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t s2)
    {
        gmLayout.MakeLayout(b, n2, s2);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t offset = bIdx * GetStrideB() + n2Idx * GetStrideN2() + s2Idx * GetStrideS2();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint32_t GetDimB()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimS2()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }
};

template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BnBs> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t blockSize, GlobalTensor<int32_t> blockTableGm, uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(blockSize);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t nIdx, uint32_t sIdx)
    {
        uint64_t blockIdxInBatch = sIdx / GetStrideBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = sIdx % GetStrideBlockSize(); // 获取在单个块上超出的行数
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
        uint64_t offset = blockIdx * GetStrideBlockNum() + bsIdx * GetStrideBlockSize();

        return offset;
    }

    // Get Stride

    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    // Get Dim
    __aicore__ inline uint32_t GetDimBlockSize()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }
};

template <GmFormat FORMAT> struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BnNBs> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void
    Init(uint32_t n, uint32_t blockSize, GlobalTensor<int32_t> blockTableGm, uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n, blockSize);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t nIdx, uint32_t sIdx)
    {
        uint64_t blockIdxInBatch = sIdx / GetStrideBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = sIdx % GetStrideBlockSize(); // 获取在单个块上超出的行数
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
        uint64_t offset = blockIdx * GetStrideBlockNum() + nIdx * GetStrideN() + bsIdx * GetStrideBlockSize();

        return offset;
    }

    // Get Stride

    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint32_t GetDimN()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimBlockSize()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }
};

// PSE
template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_PSE_BN2GS1S2> {
    GmLayout<FORMAT> gmLayout;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t s2)
    {
        gmLayout.MakeLayout(b, n2, g, s1, s2);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t gIdx, uint32_t s1Idx, uint32_t s2Idx)
    {
        uint64_t offset = bIdx * GetStrideB() + n2Idx * GetStrideN2() + gIdx * GetStrideG() + s1Idx * GetStrideS1() +
                          s2Idx * GetStrideS2();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideG()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideS1()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return AscendC::Std::get<4>(gmLayout.stride); // 4:代表第5个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint32_t GetDimB()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimN2()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint32_t GetDimG()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint32_t GetDimS1()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint32_t GetDimS2()
    {
        return AscendC::Std::get<4>(gmLayout.shape); // 4:代表第5个维度，索引从0开始
    }
};

template <GmFormat FORMAT>
struct OffsetCalculator : public OffsetCalculatorImpl<FORMAT, GmLayoutParams<FORMAT>::CATEGORY> {
};

// ----------------------------------------------CopyQueryGmToL1--------------------------------
template <typename Q_T, GmFormat FORMAT> struct FaGmTensor {
    GlobalTensor<Q_T> gmTensor;
    OffsetCalculator<FORMAT> offsetCalculator;
};

enum class L1Format {
    NZ = 0
};

template <typename Q_T, L1Format FORMAT> struct FaL1Tensor {
    LocalTensor<Q_T> tensor;
    uint32_t rowCount;
};

struct GmCoord {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t dIdx;
    uint32_t gS1DealSize;
    uint32_t dDealSize;
};

template <typename T>
__aicore__ inline void CopySingleMatrixNDToNZ(LocalTensor<T> l1Tensor,
                                              const GlobalTensor<T> gmTensor,
                                              uint32_t nValue,
                                              uint32_t dValue,
                                              uint32_t srcDValue,
                                              uint32_t dstNzC0Stride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = nValue; // nd矩阵的行数
    if constexpr (IsSameType<T, int4b_t>::value) {
        nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
        nd2nzPara.srcDValue = srcDValue / HALF_SIZE_DIVISOR;
    } else {
        nd2nzPara.dValue = dValue; // nd矩阵的列数
        nd2nzPara.srcDValue = srcDValue; // 同一nd矩阵相邻行起始地址间的偏移
    }
    nd2nzPara.dstNzC0Stride = dstNzC0Stride;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

template <typename T>
__aicore__ inline void CopyMultiMatrixNDToNZ(LocalTensor<T> l1Tensor,
                                             const GlobalTensor<T> gmTensor,
                                             uint32_t srcNdMatrixNum,
                                             uint32_t srcNdMatrixStride,
                                             uint32_t dstNzMatrixStride,
                                             uint32_t nValue,
                                             uint32_t dValue,
                                             uint32_t srcDValue,
                                             uint32_t dstNzC0Stride)
{
    if (unlikely(srcNdMatrixStride > ND_MATRIX_STRIDE_LIMIT)) {
        uint64_t l1Offset = 0;
        uint64_t gmOffset = 0;
        for (uint32_t i = 0; i < srcNdMatrixNum; i++) {
            CopySingleMatrixNDToNZ(l1Tensor[l1Offset], gmTensor[gmOffset], nValue, dValue, srcDValue, dstNzC0Stride);
            gmOffset += srcNdMatrixStride;
            l1Offset += dstNzMatrixStride;
        }
    } else {
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = srcNdMatrixNum;
        nd2nzPara.nValue = nValue; // nd矩阵的行数
        if constexpr (IsSameType<T, int4b_t>::value) {
            nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
            nd2nzPara.srcDValue = srcDValue / HALF_SIZE_DIVISOR;
        } else {
            nd2nzPara.dValue = dValue; // nd矩阵的列数
            nd2nzPara.srcDValue = srcDValue; // 同一nd矩阵相邻行起始地址间的偏移
        }
        nd2nzPara.dstNzC0Stride = dstNzC0Stride;
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzPara.dstNzMatrixStride = dstNzMatrixStride;
        DataCopy(l1Tensor, gmTensor, nd2nzPara);
    }
}

template <typename Q_T, GmFormat GM_FORMAT, L1Format L1_FORMAT = L1Format::NZ> class CopyQueryGmToL1 {
  public:
    __aicore__ inline void
    operator()(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        if constexpr ((GM_FORMAT == GmFormat::BSNGD) || (GM_FORMAT == GmFormat::TNGD)) {
            ProcessS1G(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::BNGSD) {
            OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                ProcessGS1(dstTensor, srcTensor, gmCoord);
            } else {
                ProcessContinuous(dstTensor, srcTensor, gmCoord);
            }
        } else if constexpr (GM_FORMAT == GmFormat::NGTD) {
            ProcessGS1(dstTensor, srcTensor, gmCoord);
        }
    }

  private:
    __aicore__ inline void
    ProcessS1G(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
        uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, gmCoord.dIdx);

        if (offsetCalculator.GetDimG() == 1) {
            CopySingleMatrixNDToNZ(dstTensor.tensor,
                                   srcTensor.gmTensor[queryGmbaseOffset],
                                   s1IdxEnd - s1IdxStart,
                                   gmCoord.dDealSize,
                                   offsetCalculator.GetStrideS1(),
                                   dstTensor.rowCount);
            return;
        }

        // 处理第一个S
        uint32_t headSize = 0;
        if (s1IdxStart == s1IdxEnd) {
            headSize = gIdxEnd - gIdxStart;
        } else {
            headSize = offsetCalculator.GetDimG() - gIdxStart;
        }

        uint64_t offset = queryGmbaseOffset + gIdxStart * offsetCalculator.GetDimD();
        CopySingleMatrixNDToNZ(dstTensor.tensor,
                               srcTensor.gmTensor[offset],
                               headSize,
                               gmCoord.dDealSize,
                               offsetCalculator.GetStrideG(),
                               dstTensor.rowCount);

        if (s1IdxEnd - s1IdxStart >= 1) {
            // 处理中间块
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideS1();
            uint64_t l1Offset = headSize * 16U;
            if (s1IdxEnd - s1IdxStart > 1) {
                CopyMultiMatrixNDToNZ(dstTensor.tensor[l1Offset],
                                      srcTensor.gmTensor[gmOffset],
                                      s1IdxEnd - s1IdxStart - 1,
                                      offsetCalculator.GetStrideS1(),
                                      offsetCalculator.GetDimG() * 16U,
                                      offsetCalculator.GetDimG(),
                                      gmCoord.dDealSize,
                                      offsetCalculator.GetStrideG(),
                                      dstTensor.rowCount);
                gmOffset += (s1IdxEnd - s1IdxStart - 1) * offsetCalculator.GetStrideS1();
                l1Offset += (s1IdxEnd - s1IdxStart - 1) * offsetCalculator.GetDimG() * 16U;
            }

            // 处理尾块
            if (gIdxEnd > 0) {
                CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset],
                                       srcTensor.gmTensor[gmOffset],
                                       gIdxEnd,
                                       gmCoord.dDealSize,
                                       offsetCalculator.GetStrideG(),
                                       dstTensor.rowCount);
            }
        }
    }

    __aicore__ inline void
    ProcessContinuous(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        // B*N2*GS1*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t gIdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimS1();
        uint32_t s1IdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimS1();

        uint64_t offset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, s1IdxStart, gmCoord.dIdx);
        CopySingleMatrixNDToNZ(dstTensor.tensor,
                               srcTensor.gmTensor[offset],
                               gmCoord.gS1DealSize,
                               gmCoord.dDealSize,
                               offsetCalculator.GetDimD(),
                               dstTensor.rowCount);
    }

    __aicore__ inline void
    ProcessGS1(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensor<Q_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        // N2*G*T(BS1)*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t s1Size = 0;
        if constexpr (GmLayoutParams<GM_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        } else {
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
            } else {
                s1Size = offsetCalculator.GetDimS1();
            }
        }

        uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
        uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / s1Size;
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % s1Size;

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0, gmCoord.dIdx);

        // 处理第一个S
        uint32_t headSize = 0;
        if (gIdxStart == gIdxEnd) {
            headSize = s1IdxEnd - s1IdxStart;
        } else {
            headSize = s1Size - s1IdxStart;
        }

        uint64_t offset = queryGmbaseOffset + s1IdxStart * offsetCalculator.GetDimD();
        CopySingleMatrixNDToNZ(dstTensor.tensor,
                               srcTensor.gmTensor[offset],
                               headSize,
                               gmCoord.dDealSize,
                               offsetCalculator.GetStrideS1(),
                               dstTensor.rowCount);

        if (gIdxEnd - gIdxStart >= 1) {
            // 处理中间块
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideG();
            uint64_t l1Offset = headSize * 16U;

            if (gIdxEnd - gIdxStart > 1) {
                CopyMultiMatrixNDToNZ(dstTensor.tensor[l1Offset],
                                      srcTensor.gmTensor[gmOffset],
                                      gIdxEnd - gIdxStart - 1,
                                      offsetCalculator.GetStrideG(),
                                      s1Size * 16U,
                                      s1Size,
                                      gmCoord.dDealSize,
                                      offsetCalculator.GetStrideS1(),
                                      dstTensor.rowCount);
                gmOffset += (gIdxEnd - gIdxStart - 1) * offsetCalculator.GetStrideG();
                l1Offset += (gIdxEnd - gIdxStart - 1) * s1Size * 16U;
            }

            // 处理尾块
            if (s1IdxEnd > 0) {
                CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset],
                                       srcTensor.gmTensor[gmOffset],
                                       s1IdxEnd,
                                       gmCoord.dDealSize,
                                       offsetCalculator.GetStrideS1(),
                                       dstTensor.rowCount);
            }
        }
    }
};

// ----------------------------------------------CopyAttenOutUbToGm--------------------------------
enum class UbFormat {
    GS1 = 0,
    S1G = 1
};

template <typename OUT_T> struct FaUbTensor {
    LocalTensor<OUT_T> tensor;
    uint32_t rowCount;
    uint32_t colCount;
};

template <typename OUT_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT> class CopyAttenOutUbToGm {
  public:
    __aicore__ inline void SafeStrideCopy(GlobalTensor<OUT_T> gmTensor,
                                          const LocalTensor<OUT_T> ubTensor,
                                          uint32_t blockCount,
                                          uint32_t blockLen,
                                          uint32_t srcStride,
                                          uint64_t dstStride)
    {
        DataCopyExtParams dataCopyParams;
        // B*S过大时，跳写参数dataCopyParams.dstStride(uint32_t)计算结果将溢出，使用for循环拷贝代替
        if (dstStride > UINT32_MAX) {
            uint64_t gmSingleStride = (dstStride + blockLen) / sizeof(OUT_T);
            uint64_t ubSingleStride =
                (srcStride * AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK + blockLen) / sizeof(OUT_T);
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = blockLen;
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0; // 单位为Byte
            for (uint32_t i = 0; i < blockCount; i++) {
                DataCopyPad(gmTensor[i * gmSingleStride], ubTensor[i * ubSingleStride], dataCopyParams);
            }
        } else {
            // dataCopyParams.dstStride(uint32_t)没有溢出时，进行跳写
            dataCopyParams.blockCount = blockCount;
            dataCopyParams.blockLen = blockLen;
            dataCopyParams.srcStride = srcStride;
            dataCopyParams.dstStride = dstStride; // 单位为Byte
            DataCopyPad(gmTensor, ubTensor, dataCopyParams);
        }
    }
    __aicore__ inline void
    operator()(FaGmTensor<OUT_T, GM_FORMAT> &dstTensor, FaUbTensor<OUT_T> &srcTensor, GmCoord &gmCoord)
    {
        if constexpr (UB_FORMAT == UbFormat::GS1) {
            OffsetCalculator<GM_FORMAT> &offsetCalculator = dstTensor.offsetCalculator;
            uint32_t s1Size = 0;
            if constexpr (GmLayoutParams<GM_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
                s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
            } else {
                if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                    s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
                } else {
                    s1Size = offsetCalculator.GetDimS1();
                }
            }
            uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
            uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;
            uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / s1Size;
            uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % s1Size;

            uint64_t attenOutGmbaseOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0, 0);

            // 处理第一个S
            uint32_t headS1 = 0;
            if (gIdxStart == gIdxEnd) {
                headS1 = s1IdxEnd - s1IdxStart;
            } else {
                headS1 = s1Size - s1IdxStart;
            }
            uint64_t gmOffset = attenOutGmbaseOffset + s1IdxStart * offsetCalculator.GetStrideS1();
            uint64_t ubOffset = 0;
            uint32_t blockCount = headS1;
            uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
            uint32_t srcStride =
                (srcTensor.colCount - gmCoord.dDealSize) /
                (AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(OUT_T));
            uint64_t dstStride = (offsetCalculator.GetStrideS1() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
            SafeStrideCopy(
                dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride, dstStride);

            if (gIdxEnd - gIdxStart >= 1) {
                // 处理中间块
                gmOffset = attenOutGmbaseOffset + offsetCalculator.GetStrideG();
                ubOffset = headS1 * srcTensor.colCount;
                for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
                    blockCount = s1Size;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset],
                                   srcTensor.tensor[ubOffset],
                                   blockCount,
                                   blockLen,
                                   srcStride,
                                   dstStride);
                    gmOffset += offsetCalculator.GetStrideG();
                    ubOffset += s1Size * srcTensor.colCount;
                }

                // 处理尾块
                if (s1IdxEnd > 0) {
                    blockCount = s1IdxEnd;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset],
                                   srcTensor.tensor[ubOffset],
                                   blockCount,
                                   blockLen,
                                   srcStride,
                                   dstStride);
                }
            }
        } else if constexpr (UB_FORMAT == UbFormat::S1G) {
            OffsetCalculator<GM_FORMAT> &offsetCalculator = dstTensor.offsetCalculator;
            uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
            uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
            uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
            uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

            uint64_t attenOutGmbaseOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, 0);

            // 处理第一个S
            uint32_t headSize = 0;
            if (s1IdxStart == s1IdxEnd) {
                headSize = gIdxEnd - gIdxStart;
            } else {
                headSize = offsetCalculator.GetDimG() - gIdxStart;
            }
            uint64_t gmOffset = attenOutGmbaseOffset + gIdxStart * offsetCalculator.GetStrideG();
            uint64_t ubOffset = 0;
            uint32_t blockCount = headSize;
            uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
            uint32_t srcStride =
                (srcTensor.colCount - gmCoord.dDealSize) /
                (AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(OUT_T));
            uint64_t dstStride = (offsetCalculator.GetStrideG() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
            SafeStrideCopy(
                dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride, dstStride);

            if (s1IdxEnd - s1IdxStart >= 1) {
                // 处理中间块
                gmOffset = attenOutGmbaseOffset + offsetCalculator.GetStrideS1();
                ubOffset = ((uint64_t)headSize) * ((uint64_t)srcTensor.colCount);
                for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                    blockCount = offsetCalculator.GetDimG();
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset],
                                   srcTensor.tensor[ubOffset],
                                   blockCount,
                                   blockLen,
                                   srcStride,
                                   dstStride);
                    gmOffset += offsetCalculator.GetStrideS1();
                    ubOffset += offsetCalculator.GetDimG() * srcTensor.colCount;
                }

                // 处理尾块
                if (gIdxEnd > 0) {
                    blockCount = gIdxEnd;
                    SafeStrideCopy(dstTensor.gmTensor[gmOffset],
                                   srcTensor.tensor[ubOffset],
                                   blockCount,
                                   blockLen,
                                   srcStride,
                                   dstStride);
                }
            }
        }
    }
};

// ----------------------------------------------CopyKvGmToL1--------------------------------
struct GmKvCoord {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t s2Idx;
    uint32_t dIdx;
    uint32_t s2DealSize;
    uint32_t dDealSize;
};

template <typename KV_T, GmFormat GM_FORMAT, L1Format L1_FORMAT = L1Format::NZ> class CopyKvGmToL1 {
  public:
    __aicore__ inline void
    operator()(FaL1Tensor<KV_T, L1_FORMAT> &dstTensor, FaGmTensor<KV_T, GM_FORMAT> &srcTensor, GmKvCoord &gmCoord)
    {
        if constexpr (GM_FORMAT == GmFormat::BNSD || GM_FORMAT == GmFormat::BSND || GM_FORMAT == GmFormat::NTD ||
                      GM_FORMAT == GmFormat::TND) {
            ProcessContinuousOrTensorlist(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::PA_BnBsND || GM_FORMAT == GmFormat::PA_BnNBsD ||
                             GM_FORMAT == GmFormat::PA_NZ) {
            ProcessPageAttention(dstTensor, srcTensor, gmCoord);
        }
    }

  private:
    __aicore__ inline void ProcessContinuousOrTensorlist(FaL1Tensor<KV_T, L1_FORMAT> &dstTensor,
                                                         FaGmTensor<KV_T, GM_FORMAT> &srcTensor,
                                                         GmKvCoord &gmCoord)
    {
        // B*N2*GS1*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t offset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gmCoord.s2Idx, gmCoord.dIdx);
        CopySingleMatrixNDToNZ(dstTensor.tensor,
                               srcTensor.gmTensor[offset],
                               gmCoord.s2DealSize,
                               gmCoord.dDealSize,
                               offsetCalculator.GetStrideS2(),
                               dstTensor.rowCount);
    }

    __aicore__ inline void ProcessPageAttention(FaL1Tensor<KV_T, L1_FORMAT> &dstTensor,
                                                FaGmTensor<KV_T, GM_FORMAT> &srcTensor,
                                                GmKvCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t curS2Idx = gmCoord.s2Idx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (IsSameType<KV_T, int4b_t>::value) {
            blockElementCnt = 64; // int4b时32B可以存64个元素
        }
        while (copyFinishRowCnt < gmCoord.s2DealSize) {
            // 获取需要拷贝的行数
            uint32_t copyRowCnt = offsetCalculator.GetBlockSize() - curS2Idx % offsetCalculator.GetBlockSize();
            if (copyFinishRowCnt + copyRowCnt > gmCoord.s2DealSize) {
                copyRowCnt = gmCoord.s2DealSize - copyFinishRowCnt;  // 一个block未拷满
            }

            // 计算offset
            uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, curS2Idx, gmCoord.dIdx);
            uint64_t l1Offset = copyFinishRowCnt * blockElementCnt;

            // 拷贝数据
            if constexpr (GM_FORMAT == GmFormat::PA_NZ) {
                DataCopyParams intriParams;
                intriParams.blockCount = gmCoord.dDealSize / blockElementCnt;
                intriParams.blockLen = copyRowCnt;
                intriParams.dstStride = dstTensor.rowCount - copyRowCnt;
                intriParams.srcStride = offsetCalculator.GetBlockSize() - copyRowCnt;
                DataCopy(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset], intriParams);
            } else {
                CopySingleMatrixNDToNZ(dstTensor.tensor[l1Offset],
                                       srcTensor.gmTensor[gmOffset],
                                       copyRowCnt,
                                       gmCoord.dDealSize,
                                       offsetCalculator.GetStrideBlockSize(),
                                       dstTensor.rowCount);
            }

            // 更新完成拷贝的行数和s2Idx
            copyFinishRowCnt += copyRowCnt;
            curS2Idx += copyRowCnt;
        }
    }
};

template <typename KV_T, GmFormat GM_FORMAT, L1Format L1_FORMAT = L1Format::NZ> class CopyKKropePAGmToL1 {
  public:
    __aicore__ inline void operator()(FaL1Tensor<KV_T, L1_FORMAT> &dstTensorK,
                                      FaL1Tensor<KV_T, L1_FORMAT> &dstTensorKrope,
                                      FaGmTensor<KV_T, GM_FORMAT> &srcTensorK,
                                      FaGmTensor<KV_T, GM_FORMAT> &srcTensorKrope,
                                      GmKvCoord &gmCoordK,
                                      GmKvCoord &gmCoordKrope)
    {
        if constexpr (GM_FORMAT == GmFormat::PA_NZ || GM_FORMAT == GmFormat::PA_BnNBsD ||
                      GM_FORMAT == GmFormat::PA_BnBsND) {
            ProcessPageAttention(dstTensorK, dstTensorKrope, srcTensorK, srcTensorKrope, gmCoordK, gmCoordKrope);
        }
    }

  private:
    __aicore__ inline uint64_t
    GetBlockIdx(FaGmTensor<KV_T, GM_FORMAT> &srcKTensor, uint64_t blockIdxInBatch, uint32_t bIdx)
    {
        return srcKTensor.offsetCalculator.blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
    }

    __aicore__ inline uint64_t
    GetOffset(FaGmTensor<KV_T, GM_FORMAT> &srcTensor, int32_t blockIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t bsIdx = s2Idx % offsetCalculator.GetBlockSize();
        uint64_t offset = 0;
        if constexpr (GM_FORMAT == GmFormat::PA_NZ) {
            uint32_t d1Idx = dIdx / offsetCalculator.GetD0();
            uint32_t d0Idx = dIdx % offsetCalculator.GetD0();
            offset = blockIdx * offsetCalculator.GetStrideBlockNum() + n2Idx * offsetCalculator.GetStrideN2() +
                     d1Idx * offsetCalculator.GetStrideD1() + bsIdx * offsetCalculator.GetStrideBlockSize() +
                     d0Idx * offsetCalculator.GetStrideD0();
        } else {
            offset = blockIdx * offsetCalculator.GetStrideBlockNum() + n2Idx * offsetCalculator.GetStrideN2() +
                     bsIdx * offsetCalculator.GetStrideBlockSize() + dIdx * offsetCalculator.GetStrideD();
        }

        return offset;
    }

    __aicore__ inline void ProcessPageAttention(FaL1Tensor<KV_T, L1_FORMAT> &dstTensorK,
                                                FaL1Tensor<KV_T, L1_FORMAT> &dstTensorKrope,
                                                FaGmTensor<KV_T, GM_FORMAT> &srcTensorK,
                                                FaGmTensor<KV_T, GM_FORMAT> &srcTensorKrope,
                                                GmKvCoord &gmCoordK,
                                                GmKvCoord &gmCoordKrope)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculatorK = srcTensorK.offsetCalculator;
        OffsetCalculator<GM_FORMAT> &offsetCalculatorKrope = srcTensorKrope.offsetCalculator;
        uint32_t curS2Idx = gmCoordK.s2Idx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        if constexpr (IsSameType<KV_T, int4b_t>::value) {
            blockElementCnt = 64; // int4b时32B可以存64个元素
        }

        while (copyFinishRowCnt < gmCoordK.s2DealSize) {
            // 获取需要拷贝的行数
            uint32_t copyRowCnt = offsetCalculatorK.GetBlockSize() - curS2Idx % offsetCalculatorK.GetBlockSize();
            if (copyFinishRowCnt + copyRowCnt > gmCoordK.s2DealSize) {
                copyRowCnt = gmCoordK.s2DealSize - copyFinishRowCnt;  // 一个block未拷满
            }

            // 计算offset
            uint64_t blockIdxInBatch = curS2Idx / offsetCalculatorK.GetBlockSize(); // 获取block table上的索引
            uint32_t blockIdx = GetBlockIdx(srcTensorK, blockIdxInBatch, gmCoordK.bIdx);
            uint64_t gmOffsetK = GetOffset(srcTensorK, blockIdx, gmCoordK.n2Idx, curS2Idx, gmCoordK.dIdx);
            uint64_t gmOffsetKrope =
                GetOffset(srcTensorKrope, blockIdx, gmCoordKrope.n2Idx, curS2Idx, gmCoordKrope.dIdx);
            uint64_t l1Offset = copyFinishRowCnt * blockElementCnt;

            // 拷贝数据
            if constexpr (GM_FORMAT == GmFormat::PA_NZ) {
                DataCopyParams intriParamsK;
                intriParamsK.blockCount = gmCoordK.dDealSize / blockElementCnt;
                intriParamsK.blockLen = copyRowCnt;
                intriParamsK.dstStride = dstTensorK.rowCount - copyRowCnt;
                intriParamsK.srcStride = offsetCalculatorK.GetBlockSize() - copyRowCnt;
                DataCopy(dstTensorK.tensor[l1Offset], srcTensorK.gmTensor[gmOffsetK], intriParamsK);

                DataCopyParams intriParamsKrope;
                intriParamsKrope.blockCount = gmCoordKrope.dDealSize / blockElementCnt;
                intriParamsKrope.blockLen = copyRowCnt;
                intriParamsKrope.dstStride = dstTensorKrope.rowCount - copyRowCnt;
                intriParamsKrope.srcStride = offsetCalculatorKrope.GetBlockSize() - copyRowCnt;
                DataCopy(dstTensorKrope.tensor[l1Offset], srcTensorKrope.gmTensor[gmOffsetKrope], intriParamsKrope);
            } else {
                CopySingleMatrixNDToNZ(dstTensorK.tensor[l1Offset],
                                       srcTensorK.gmTensor[gmOffsetK],
                                       copyRowCnt,
                                       gmCoordK.dDealSize,
                                       offsetCalculatorK.GetStrideBlockSize(),
                                       dstTensorK.rowCount);
                CopySingleMatrixNDToNZ(dstTensorKrope.tensor[l1Offset],
                                       srcTensorKrope.gmTensor[gmOffsetKrope],
                                       copyRowCnt,
                                       gmCoordKrope.dDealSize,
                                       offsetCalculatorKrope.GetStrideBlockSize(),
                                       dstTensorKrope.rowCount);
            }

            // 更新完成拷贝的行数和s2Idx
            copyFinishRowCnt += copyRowCnt;
            curS2Idx += copyRowCnt;
        }
    }
};

template <FIA_LAYOUT LAYOUT_T> __aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) || (LAYOUT_T == FIA_LAYOUT::BNSD) || (LAYOUT_T == FIA_LAYOUT::TND) ||
                      (LAYOUT_T == FIA_LAYOUT::NTD),
                  "Get Query GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::TND) {
        return GmFormat::TNGD;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::NTD) {
        return GmFormat::NGTD;
    }
}

template <FIA_LAYOUT KV_LAYOUT_T, const bool PAGE_ATTENTION> __aicore__ inline constexpr GmFormat GetKVFormat()
{
    if constexpr (PAGE_ATTENTION) {
        static_assert((KV_LAYOUT_T == FIA_LAYOUT::BSH) || (KV_LAYOUT_T == FIA_LAYOUT::BNSD) ||
                          (KV_LAYOUT_T == FIA_LAYOUT::NZ),
                      "Get Key or Value GmFormat fail, KV_LAYOUT_T is incorrect when PageAttention");
        if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BSH) {
            return GmFormat::PA_BnBsND;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BNSD) {
            return GmFormat::PA_BnNBsD;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::NZ) {
            return GmFormat::PA_NZ;
        }
    } else {
        static_assert((KV_LAYOUT_T == FIA_LAYOUT::BSH) || (KV_LAYOUT_T == FIA_LAYOUT::BNSD) ||
                          (KV_LAYOUT_T == FIA_LAYOUT::TND) || (KV_LAYOUT_T == FIA_LAYOUT::NTD),
                      "Get Key or Value GmFormat fail, KV_LAYOUT_T is incorrect when KV Continuous or TensorList");
        if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BSH) {
            return GmFormat::BSND;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::TND) {
            return GmFormat::TND;
        } else if constexpr (KV_LAYOUT_T == FIA_LAYOUT::NTD) {
            return GmFormat::NTD;
        }
    }
}

template <FIA_LAYOUT OUT_LAYOUT_T> __aicore__ inline constexpr GmFormat GetOutGmFormat()
{
    static_assert((OUT_LAYOUT_T == FIA_LAYOUT::BSH) || (OUT_LAYOUT_T == FIA_LAYOUT::BNSD) ||
                      (OUT_LAYOUT_T == FIA_LAYOUT::TND) || (OUT_LAYOUT_T == FIA_LAYOUT::NTD) ||
                      (OUT_LAYOUT_T == FIA_LAYOUT::NBSD),
                  "Get OutAttention GmFormat fail, OUT_LAYOUT_T is incorrect");
    if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::TND) {
        return GmFormat::TNGD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::NTD) {
        return GmFormat::NGTD;
    } else if constexpr (OUT_LAYOUT_T == FIA_LAYOUT::NBSD) {
        return GmFormat::NGBSD;
    }
}

template <FIA_LAYOUT LAYOUT_T> __aicore__ inline constexpr UbFormat GetOutUbFormat()
{
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) || (LAYOUT_T == FIA_LAYOUT::BNSD) || (LAYOUT_T == FIA_LAYOUT::TND) ||
                      (LAYOUT_T == FIA_LAYOUT::NTD),
                  "Get OutAttention UB GmFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BSH || LAYOUT_T == FIA_LAYOUT::TND) {
        return UbFormat::S1G;
    } else if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::NTD) {
        return UbFormat::GS1;
    }
}

// post_quant
// ----------------------------------------------CopyPostQuantGmToUB--------------------------------
struct PostQuantCoord {
    uint32_t n2Idx;
    uint32_t gIdx;
    uint32_t dIdx;
    uint32_t gDealSize;
};

// GM->UB
template <typename T>
__aicore__ inline void CopySingleMatrixNDToND(LocalTensor<T> ubTensor,
                                              const GlobalTensor<T> gmTensor,
                                              uint32_t blockCount,
                                              uint32_t blockLen,
                                              uint32_t srcStride,
                                              uint32_t dstStride,
                                              uint32_t rightPadding)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = static_cast<uint16_t>(blockCount); // 外部传入
    dataCopyParams.blockLen = blockLen;
    dataCopyParams.srcStride = srcStride;
    dataCopyParams.dstStride = dstStride; // 外部传入

    DataCopyPadExtParams<T> dataCopyPadParams;
    dataCopyPadParams.isPad = true;
    dataCopyPadParams.leftPadding = 0;
    dataCopyPadParams.rightPadding = rightPadding;
    dataCopyPadParams.paddingValue = 0;
    DataCopyPad(ubTensor, gmTensor, dataCopyParams, dataCopyPadParams);
}

template <typename POST_QUANT_T, GmFormat GM_FORMAT> class CopyPostQuantGmToUB {
  public:
    __aicore__ inline void operator()(FaUbTensor<POST_QUANT_T> &dstTensor,
                                      FaGmTensor<POST_QUANT_T, GM_FORMAT> &srcTensor,
                                      PostQuantCoord &gmCoord)
    {
        ProcessPerCh(dstTensor, srcTensor, gmCoord);
    }

  private:
    __aicore__ inline void ProcessPerCh(FaUbTensor<POST_QUANT_T> &dstTensor,
                                        FaGmTensor<POST_QUANT_T, GM_FORMAT> &srcTensor,
                                        PostQuantCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t offset = offsetCalculator.GetOffset(gmCoord.n2Idx, gmCoord.gIdx, gmCoord.dIdx);
        uint64_t actualColumnCount = offsetCalculator.GetDimD();
        uint32_t elementNum = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(POST_QUANT_T);

        uint32_t blockCount = gmCoord.gDealSize;
        uint32_t blockLen = actualColumnCount * sizeof(POST_QUANT_T);
        uint32_t srcStride = 0;
        uint32_t dstStride = (dstTensor.colCount - actualColumnCount) / elementNum;
        uint32_t rightPadding = (dstTensor.colCount - actualColumnCount) % elementNum;
        CopySingleMatrixNDToND(
            dstTensor.tensor, srcTensor.gmTensor[offset], blockCount, blockLen, srcStride, dstStride, rightPadding);
    }
};

// antiquant
// ----------------------------------------------CopyAntiquantGmToUb--------------------------------
struct AntiqGmCoord {
    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t s2Idx = 0;

    uint32_t s2DealSize = 0; // actualSingleProcessSInnerSize //实际s2长度
};

template <typename T, GmFormat GM_FORMAT> class CopyAntiquantGmToUb {
  public:
    __aicore__ inline void
    operator()(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor, AntiqGmCoord &antiqGmCoord)
    {
        // per tensor场景在接口外部直接getvalue
        // per channel / per token
        if constexpr ((GM_FORMAT == GmFormat::ND) || (GM_FORMAT == GmFormat::BS2) || (GM_FORMAT == GmFormat::BNS2)) {
            ProcessAntiqPerChannelOrPerToken(dstTensor, srcTensor, antiqGmCoord);
        } else if constexpr ((GM_FORMAT == GmFormat::PA_BnBs) || (GM_FORMAT == GmFormat::PA_BnNBs)) {
            // per token + PA
            ProcessAntiqPA(dstTensor, srcTensor, antiqGmCoord);
        }
    }

  private:
    __aicore__ inline void ProcessAntiqPerChannelOrPerToken(FaUbTensor<T> &dstTensor,
                                                            FaGmTensor<T, GM_FORMAT> &srcTensor,
                                                            AntiqGmCoord &antiqGmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t offset = offsetCalculator.GetOffset(antiqGmCoord.bIdx, antiqGmCoord.n2Idx, antiqGmCoord.s2Idx, 0);
        uint32_t blockLen = 0;
        uint32_t dstStride = 0;
        uint32_t rightPadding = 0;
        uint32_t elementNum = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(T);
        if constexpr (GM_FORMAT == GmFormat::ND) {
            blockLen = offsetCalculator.GetDimD() * sizeof(T);
            dstStride = (dstTensor.colCount - offsetCalculator.GetDimD()) / elementNum;
            rightPadding = (dstTensor.colCount - offsetCalculator.GetDimD()) % elementNum;
        } else if constexpr (GM_FORMAT == GmFormat::BS2 || GM_FORMAT == GmFormat::BNS2) {
            blockLen = antiqGmCoord.s2DealSize * sizeof(T);
            dstStride = (dstTensor.colCount - antiqGmCoord.s2DealSize) / elementNum;
            rightPadding = (dstTensor.colCount - antiqGmCoord.s2DealSize) % elementNum;
        }
        CopySingleMatrixNDToND(dstTensor.tensor, srcTensor.gmTensor[offset], 1, blockLen, 0, dstStride, rightPadding);
    }

    __aicore__ inline void
    ProcessAntiqPA(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor, AntiqGmCoord &antiqGmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;

        uint64_t dstOffset = 0;
        uint32_t copyFinishElmeCnt = 0;
        uint32_t curS2Idx = antiqGmCoord.s2Idx;
        uint32_t elementNum = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(T); // 每个块的元素数量

        while (copyFinishElmeCnt < antiqGmCoord.s2DealSize) {
            uint32_t copyElemCnt = offsetCalculator.GetDimBlockSize() - curS2Idx % offsetCalculator.GetDimBlockSize();
            if (copyFinishElmeCnt + copyElemCnt > antiqGmCoord.s2DealSize) {
                copyElemCnt = antiqGmCoord.s2DealSize - copyFinishElmeCnt; // 一个block未拷满
            }

            uint32_t rightPadding = elementNum - copyElemCnt % elementNum;

            uint64_t srcOffset = offsetCalculator.GetOffset(antiqGmCoord.bIdx, antiqGmCoord.n2Idx, curS2Idx);
            CopySingleMatrixNDToND(dstTensor.tensor[dstOffset],
                                   srcTensor.gmTensor[srcOffset],
                                   1,
                                   copyElemCnt * sizeof(T),
                                   0,
                                   0,
                                   rightPadding);

            dstOffset += copyElemCnt;
            copyFinishElmeCnt += copyElemCnt;
            curS2Idx += copyElemCnt;
        }
    }
};
// ----------------------------------------------CopyQueryGmToUb--------------------------------

template <typename T, GmFormat GM_FORMAT> class CopyQueryGmToUb {
  public:
    __aicore__ inline void operator()(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        if constexpr ((GM_FORMAT == GmFormat::BSNGD) || (GM_FORMAT == GmFormat::TNGD)) {
            ProcessS1G(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::BNGSD) {
            OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                ProcessGS1(dstTensor, srcTensor, gmCoord);
            } else {
                ProcessContinuous(dstTensor, srcTensor, gmCoord);
            }
        } else if constexpr (GM_FORMAT == GmFormat::NGTD) {
            ProcessGS1(dstTensor, srcTensor, gmCoord);
        }
    }

  private:
    __aicore__ inline void ProcessGS1(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t s1Size = 0;
        if constexpr (GmLayoutParams<GM_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        } else {
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
            } else {
                s1Size = offsetCalculator.GetDimS1();
            }
        }
        uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
        uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / s1Size;
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % s1Size;

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0, gmCoord.dIdx);

        // 处理 首行
        uint32_t headS1 = 0;
        if (gIdxStart == gIdxEnd) {
            headS1 = s1IdxEnd - s1IdxStart;
        } else {
            headS1 = s1Size - s1IdxStart;
        }

        uint32_t elementNum = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(T); // 每个块的元素数量
        uint32_t blockLen = gmCoord.dDealSize * sizeof(T);
        uint32_t srcStride = (offsetCalculator.GetStrideS1() - gmCoord.dDealSize) * sizeof(T);
        uint32_t dstStride = (dstTensor.colCount - gmCoord.dDealSize) / elementNum;
        uint32_t rightPadding = (dstTensor.colCount - gmCoord.dDealSize) % elementNum;

        CopySingleMatrixNDToND(dstTensor.tensor,
                               srcTensor.gmTensor[queryGmbaseOffset + s1IdxStart * offsetCalculator.GetDimD()],
                               headS1,
                               blockLen,
                               srcStride,
                               dstStride,
                               rightPadding);

        if (gIdxEnd - gIdxStart >= 1) {
            // 处理中间块
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideG();
            uint32_t ubOffset = headS1 * dstTensor.colCount;
            //
            for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
                CopySingleMatrixNDToND(dstTensor.tensor[ubOffset],
                                       srcTensor.gmTensor[gmOffset],
                                       s1Size,
                                       blockLen,
                                       srcStride,
                                       dstStride,
                                       rightPadding);
                gmOffset += offsetCalculator.GetStrideG();
                ubOffset += s1Size * dstTensor.colCount;
            }

            // 处理尾块
            if (s1IdxEnd > 0) {
                CopySingleMatrixNDToND(dstTensor.tensor[ubOffset],
                                       srcTensor.gmTensor[gmOffset],
                                       s1IdxEnd,
                                       blockLen,
                                       srcStride,
                                       dstStride,
                                       rightPadding);
            }
        }
    }

    __aicore__ inline void
    ProcessContinuous(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        // B*N2*GS1*D
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t gIdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimS1();
        uint32_t s1IdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimS1();

        uint64_t offset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, s1IdxStart, gmCoord.dIdx);
        uint32_t elementNum = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(T); // 每个块的元素数量
        uint32_t blockCount = gmCoord.gS1DealSize;
        uint32_t blockLen = gmCoord.dDealSize * sizeof(T);
        uint32_t srcStride = (offsetCalculator.GetStrideS1() - gmCoord.dDealSize) * sizeof(T);
        uint32_t dstStride = (dstTensor.colCount - gmCoord.dDealSize) / elementNum;
        uint32_t rightPadding = (dstTensor.colCount - gmCoord.dDealSize) % elementNum;
        CopySingleMatrixNDToND(
            dstTensor.tensor, srcTensor.gmTensor[offset], blockCount, blockLen, srcStride, dstStride, rightPadding);
    }

    __aicore__ inline void ProcessS1G(FaUbTensor<T> &dstTensor, FaGmTensor<T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = dstTensor.offsetCalculator;
        uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
        uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

        uint64_t queryGmbaseOffset =
            offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, gmCoord.dIdx);
        // 处理第一个g
        uint32_t headSize = 0;
        if (s1IdxStart == s1IdxEnd) {
            headSize = gIdxEnd - gIdxStart;
        } else {
            headSize = offsetCalculator.GetDimG() - gIdxStart;
        }
        uint32_t elementNum = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(T); // 每个块的元素数量
        uint32_t blockLen = gmCoord.dDealSize * sizeof(T);
        uint32_t srcStride = (offsetCalculator.GetStrideG() - gmCoord.dDealSize) * sizeof(T);
        uint32_t dstStride = (dstTensor.colCount - gmCoord.dDealSize) / elementNum;
        uint32_t rightPadding = (dstTensor.colCount - gmCoord.dDealSize) % elementNum;

        CopySingleMatrixNDToND(dstTensor.tensor,
                               srcTensor.gmTensor[queryGmbaseOffset + gIdxStart * offsetCalculator.GetDimD()],
                               headSize,
                               blockLen,
                               srcStride,
                               dstStride,
                               rightPadding);

        if (s1IdxEnd - s1IdxStart >= 1) {
            uint64_t gmOffset = queryGmbaseOffset + offsetCalculator.GetStrideS1();
            uint32_t ubOffset = headSize * dstTensor.colCount;
            // 处理中间块
            for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                uint32_t blockCount = offsetCalculator.GetDimG();
                CopySingleMatrixNDToND(dstTensor.tensor[ubOffset],
                                       srcTensor.gmTensor[gmOffset],
                                       blockCount,
                                       blockLen,
                                       srcStride,
                                       dstStride,
                                       rightPadding);
                gmOffset += offsetCalculator.GetStrideS1();
                ubOffset += offsetCalculator.GetDimG() * dstTensor.colCount;
            }

            // 处理尾块
            if (gIdxEnd > 0) {
                CopySingleMatrixNDToND(dstTensor.tensor[ubOffset],
                                       srcTensor.gmTensor[gmOffset],
                                       gIdxEnd,
                                       blockLen,
                                       srcStride,
                                       dstStride,
                                       rightPadding);
            }
        }
    }
};

// ----------------------------------------------CopyKvGmToUb--------------------------------
template <typename KV_T, GmFormat GM_FORMAT> class CopyKvGmToUb {
  public:
    __aicore__ inline void
    operator()(FaUbTensor<KV_T> &dstTensor, FaGmTensor<KV_T, GM_FORMAT> &srcTensor, GmCoord &gmCoord)
    {
        if constexpr (GM_FORMAT == GmFormat::BNSD || GM_FORMAT == GmFormat::BSND || GM_FORMAT == GmFormat::NTD ||
                      GM_FORMAT == GmFormat::TND) {
            ProcessContinuousOrTensorlist(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::PA_BnBsND || GM_FORMAT == GmFormat::PA_BnNBsD ||
                             GM_FORMAT == GmFormat::PA_NZ) {
            ProcessPageAttention(dstTensor, srcTensor, gmCoord);
        }
    }

  private:
    __aicore__ inline void ProcessContinuousOrTensorlist(FaUbTensor<KV_T> &dstTensor,
                                                         FaGmTensor<KV_T, GM_FORMAT> &srcTensor,
                                                         GmKvCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint64_t offset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gmCoord.s2Idx, gmCoord.dIdx);
        uint32_t elementNum = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK;
        uint32_t blockLen = gmCoord.dDealSize;
        uint32_t srcStride = (offsetCalculator.GetStrideS2() - gmCoord.dDealSize);
        if constexpr (IsSameType<KV_T, int4b_t>::value) {
            elementNum = elementNum * HALF_SIZE_DIVISOR;
            blockLen = blockLen / HALF_SIZE_DIVISOR;
            srcStride = srcStride / HALF_SIZE_DIVISOR;
        } else {
            elementNum = elementNum / sizeof(KV_T);
            blockLen = blockLen * sizeof(KV_T); // 列数，单位byte
            srcStride = srcStride * sizeof(KV_T);
        }

        uint32_t blockCount = gmCoord.s2DealSize; // 行数
        uint32_t dstStride = (dstTensor.colCount - gmCoord.dDealSize) / elementNum;
        uint32_t rightPadding = (dstTensor.colCount - gmCoord.dDealSize) % elementNum;

        CopySingleMatrixNDToND(
            dstTensor.tensor, srcTensor.gmTensor[offset], blockCount, blockLen, srcStride, dstStride, rightPadding);
    }

    __aicore__ inline void
    ProcessPageAttention(FaUbTensor<KV_T> &dstTensor, FaGmTensor<KV_T, GM_FORMAT> &srcTensor, GmKvCoord &gmCoord)
    {
        OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
        uint32_t curS2Idx = gmCoord.s2Idx;
        uint32_t copyFinishRowCnt = 0;
        uint32_t blockElementCnt = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(KV_T);
        if constexpr (IsSameType<KV_T, int4b_t>::value) { // FP4（E2M1\E1M2）
            blockElementCnt = 64; // fp4时32B可以存64个元素
        }

        if constexpr (GM_FORMAT == GmFormat::PA_NZ) {
            while (copyFinishRowCnt < gmCoord.s2DealSize) {
                // 获取需要拷贝的行数
                uint32_t copyRowCnt = offsetCalculator.GetBlockSize() - curS2Idx % offsetCalculator.GetBlockSize();
                if (copyFinishRowCnt + copyRowCnt > gmCoord.s2DealSize) {
                    copyRowCnt =
                        gmCoord.s2DealSize - copyFinishRowCnt; // block table中当前batch表项的尾块，一个block未拷满
                }

                // 计算offset
                uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, curS2Idx, gmCoord.dIdx);
                uint64_t l1Offset = copyFinishRowCnt * blockElementCnt;

                // 拷贝数据
                // DataCopy
                DataCopyParams repeatParams;
                repeatParams.blockCount = gmCoord.dDealSize / blockElementCnt; // D可切出多少个32B
                repeatParams.blockLen = copyRowCnt; // 单位32B
                repeatParams.srcStride = offsetCalculator.GetBlockSize() - copyRowCnt; // 单位32B
                repeatParams.dstStride = dstTensor.rowCount - copyRowCnt; // 单位32B
                DataCopy(dstTensor.tensor[l1Offset], srcTensor.gmTensor[gmOffset], repeatParams);

                // 更新完成拷贝的行数和s2Idx
                copyFinishRowCnt += copyRowCnt;
                curS2Idx += copyRowCnt;
            }
        } else { // BBH BNBD
            uint32_t blockLen = gmCoord.dDealSize * sizeof(KV_T); // 单位B
            uint32_t srcStride = (offsetCalculator.GetStrideBlockSize() - gmCoord.dDealSize) * sizeof(KV_T); // 单位B

            if constexpr (IsSameType<KV_T, int4b_t>::value) { // FP4（E2M1\E1M2）
                blockLen = gmCoord.dDealSize / HALF_SIZE_DIVISOR;
                srcStride = (offsetCalculator.GetStrideBlockSize() - gmCoord.dDealSize) / HALF_SIZE_DIVISOR;
            }

            while (copyFinishRowCnt < gmCoord.s2DealSize) {
                // 获取需要拷贝的行数
                uint32_t copyRowCnt = offsetCalculator.GetBlockSize() - curS2Idx % offsetCalculator.GetBlockSize();
                if (copyFinishRowCnt + copyRowCnt > gmCoord.s2DealSize) {
                    copyRowCnt =
                        gmCoord.s2DealSize - copyFinishRowCnt; // block table中当前batch表项的尾块，一个block未拷满
                }

                // 计算offset
                uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, curS2Idx, gmCoord.dIdx);
                uint64_t l1Offset = copyFinishRowCnt * blockElementCnt;

                uint32_t blockCount = copyRowCnt;

                uint32_t dstStride = (dstTensor.rowCount - gmCoord.dDealSize) / blockElementCnt; // 单位32B
                uint32_t rightPadding = blockElementCnt - gmCoord.dDealSize % blockElementCnt;

                // DataCopyPad
                CopySingleMatrixNDToND(dstTensor.tensor[l1Offset],
                                       srcTensor.gmTensor[gmOffset],
                                       blockCount,
                                       blockLen,
                                       srcStride,
                                       dstStride,
                                       rightPadding);

                // 更新完成拷贝的行数和s2Idx
                copyFinishRowCnt += copyRowCnt;
                curS2Idx += copyRowCnt;
            }
        }
    }
};

// ---------------------------------------------CopyPSEGmToUb--------------------------------------
struct GmPseCoord {
    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx = 0;
    uint32_t s2Idx = 0;
    uint32_t gS1DealSize = 0;
    uint32_t s2DealSize = 0;
    uint64_t s1LeftPaddingSize = 0;
    uint64_t s2LeftPaddingSize = 0;
};

// 对齐暂不考虑TND
template <typename PSE_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT> class CopyPSEGmToUb {
  public:
    __aicore__ inline void
    operator()(FaUbTensor<PSE_T> &dstTensor, FaGmTensor<PSE_T, GM_FORMAT> &srcTensor, GmPseCoord &gmPseCoord)
    {
        uint32_t elementNum = AiInfraInferenceCommonFaBaseVector::BYTE_BLOCK / sizeof(PSE_T);
        uint32_t blockLen = gmPseCoord.s2DealSize * sizeof(PSE_T);
        uint32_t dstStride = (dstTensor.colCount - gmPseCoord.s2DealSize) / elementNum;
        uint32_t rightPadding = (dstTensor.colCount - gmPseCoord.s2DealSize) % elementNum;
        if constexpr (UB_FORMAT == UbFormat::GS1) {
            // 连续，单次拷贝
            OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
            uint32_t gIdxStart = gmPseCoord.gS1Idx / offsetCalculator.GetDimS1();
            uint32_t s1IdxStart = gmPseCoord.gS1Idx % offsetCalculator.GetDimS1();
            uint64_t offset = offsetCalculator.GetOffset(gmPseCoord.bIdx,
                                                         gmPseCoord.n2Idx,
                                                         gIdxStart,
                                                         gmPseCoord.s1LeftPaddingSize + s1IdxStart,
                                                         gmPseCoord.s2LeftPaddingSize + gmPseCoord.s2Idx);
            // 统一的接口
            uint32_t blockCount = gmPseCoord.gS1DealSize;
            uint32_t srcStride = (offsetCalculator.GetStrideS1() - gmPseCoord.s2DealSize) * sizeof(PSE_T);
            CopySingleMatrixNDToND(
                dstTensor.tensor, srcTensor.gmTensor[offset], blockCount, blockLen, srcStride, dstStride, rightPadding);
        } else if constexpr (UB_FORMAT == UbFormat::S1G) {
            // 不连续，需要分3次拷贝
            OffsetCalculator<GM_FORMAT> &offsetCalculator = srcTensor.offsetCalculator;
            uint32_t s1IdxStart = gmPseCoord.gS1Idx / offsetCalculator.GetDimG();
            uint32_t gIdxStart = gmPseCoord.gS1Idx % offsetCalculator.GetDimG();
            uint32_t s1IdxEnd = (gmPseCoord.gS1Idx + gmPseCoord.gS1DealSize) / offsetCalculator.GetDimG();
            uint32_t gIdxEnd = (gmPseCoord.gS1Idx + gmPseCoord.gS1DealSize) % offsetCalculator.GetDimG();
            uint64_t gmOffset =
                offsetCalculator.GetOffset(gmPseCoord.bIdx,
                                           gmPseCoord.n2Idx,
                                           gIdxStart,
                                           gmPseCoord.s1LeftPaddingSize + s1IdxStart,
                                           gmPseCoord.s2LeftPaddingSize + gmPseCoord.s2Idx); // GM上为GS1

            // 处理第一个S
            uint32_t headSize = 0;
            if (s1IdxStart == s1IdxEnd) {
                headSize = gIdxEnd - gIdxStart;
            } else {
                headSize = offsetCalculator.GetDimG() - gIdxStart;
            }

            uint32_t srcStride = (offsetCalculator.GetStrideG() - gmPseCoord.s2DealSize) * sizeof(PSE_T);
            CopySingleMatrixNDToND(
                dstTensor.tensor, srcTensor.gmTensor[gmOffset], headSize, blockLen, srcStride, dstStride, rightPadding);
            if (s1IdxEnd - s1IdxStart >= 1) {
                uint64_t ubOffset = ((uint64_t)headSize) * ((uint64_t)dstTensor.colCount);
                // 处理中间块
                gmOffset = offsetCalculator.GetOffset(gmPseCoord.bIdx,
                                                      gmPseCoord.n2Idx,
                                                      0,
                                                      gmPseCoord.s1LeftPaddingSize + s1IdxStart + 1,
                                                      gmPseCoord.s2LeftPaddingSize + gmPseCoord.s2Idx); // GM上为GS1
                // 处理中间块
                for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                    CopySingleMatrixNDToND(dstTensor.tensor[ubOffset],
                                           srcTensor.gmTensor[gmOffset],
                                           offsetCalculator.GetDimG(),
                                           blockLen,
                                           srcStride,
                                           dstStride,
                                           rightPadding);
                    ubOffset += offsetCalculator.GetDimG() * dstTensor.colCount;
                    gmOffset += offsetCalculator.GetStrideS1();
                }

                // 处理尾块
                if (gIdxEnd > 0) {
                    CopySingleMatrixNDToND(dstTensor.tensor[ubOffset],
                                           srcTensor.gmTensor[gmOffset],
                                           gIdxEnd,
                                           blockLen,
                                           srcStride,
                                           dstStride,
                                           rightPadding);
                }
            }
        }
    }
};

template <FIA_LAYOUT LAYOUT_T> __aicore__ inline constexpr bool IsSupportPse()
{
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::BSH) {
        return true;
    } else {
        return false;
    }
}

template <FIA_LAYOUT LAYOUT_T> __aicore__ inline constexpr UbFormat GetPseUbFormat()
{
    static_assert((LAYOUT_T == FIA_LAYOUT::BSH) || (LAYOUT_T == FIA_LAYOUT::BNSD) || (LAYOUT_T == FIA_LAYOUT::TND) ||
                      (LAYOUT_T == FIA_LAYOUT::NTD),
                  "Get PSE UbFormat fail, LAYOUT_T is incorrect");
    if constexpr (LAYOUT_T == FIA_LAYOUT::BNSD || LAYOUT_T == FIA_LAYOUT::NTD) {
        return UbFormat::GS1;
    } else {
        return UbFormat::S1G;
    }
}
// --------------CopyAttentionMask----------------------------------------------------------------
enum SparseMode : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
};

struct MaskCopyInfo {
    uint32_t gs1dealNum;
    uint32_t s1Size;
    uint32_t s2StartIdx;
    uint32_t s2dealNum;
    uint32_t s2Size;
    int64_t preToken = 0;
    int64_t nextToken = 0;
    uint32_t batchIdx;
    uint32_t batchOffset;
    uint32_t attenMaskStride;
    SparseMode sparseMode;
    uint32_t s1StartIdx;
    uint32_t s1EndIdx;
    bool isPre = false;
};

__aicore__ inline uint64_t ComputeAttenMaskOffsetNoCompress(MaskCopyInfo &info)
{
    uint64_t bOffset = info.batchIdx * info.batchOffset;
    uint64_t s1Offset = info.s1StartIdx % info.s1Size * info.attenMaskStride;
    uint64_t s2Offset = info.s2StartIdx;
    return bOffset + s1Offset + s2Offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompress(MaskCopyInfo &info)
{
    int64_t nextToken = 0; // sparse2 本身原点就是左上角
    if (info.sparseMode == RIGHT_DOWN_CAUSAL) {
        nextToken =
            static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size); // 统一以左上角为原点计算token
    } else if (info.sparseMode == BAND) { // 4
        nextToken = info.nextToken + static_cast<int64_t>(info.s2Size) - static_cast<int64_t>(info.s1Size);
    }

    uint64_t offset = 0;
    int64_t delta = nextToken + info.s1StartIdx - info.s2StartIdx;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                 info.attenMaskStride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffsetCompressPre(MaskCopyInfo &info)
{
    int64_t preToken = info.preToken + static_cast<int64_t>(info.s1Size) -
                       static_cast<int64_t>(info.s2Size); // 统一以左上角为原点计算token
    int64_t delta = -preToken + static_cast<int64_t>(info.s1StartIdx) - static_cast<int64_t>(info.s2StartIdx) - 1;
    uint64_t offset = 0;
    uint32_t attenMaskSizeAlign = Align(info.s2dealNum, 32U);
    if (delta < 0) {
        offset = (-delta) < static_cast<int64_t>(info.gs1dealNum) ? (-delta) : info.gs1dealNum; // min (-delta, s1Size)
    } else {
        offset = (delta < static_cast<int64_t>(attenMaskSizeAlign) ? delta : attenMaskSizeAlign) *
                 info.attenMaskStride; // min(delta, s2inner)
    }
    return offset;
}

__aicore__ inline uint64_t ComputeAttenMaskOffset(MaskCopyInfo &info)
{
    if (info.isPre) {
        return ComputeAttenMaskOffsetCompressPre(info);
    } else {
        if (info.sparseMode == DEFAULT_MASK || info.sparseMode == ALL_MASK) {
            return ComputeAttenMaskOffsetNoCompress(info);
        } else {
            return ComputeAttenMaskOffsetCompress(info);
        }
    }
}

template <typename T>
__aicore__ inline void CopyAttentionMask(FaUbTensor<T> &attenMaskUb, GlobalTensor<T> &srcGmAddr, MaskCopyInfo &info)
{
    uint64_t maskOffset = ComputeAttenMaskOffset(info);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = info.s1EndIdx - info.s1StartIdx;
    dataCopyParams.blockLen = info.s2dealNum;
    dataCopyParams.srcStride = info.attenMaskStride - info.s2dealNum;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<bool> padParams{true, 0, static_cast<uint8_t>(attenMaskUb.colCount - info.s2dealNum), 0};

    DataCopyPad(attenMaskUb.tensor, srcGmAddr[maskOffset], dataCopyParams, padParams);
}

// ----------------------------------------------Copy LSE UB To Gm--------------------------------
template <typename T, ActualSeqLensMode Q_MODE>
__aicore__ inline void DataCopySoftmaxLseBSND(GlobalTensor<float> softmaxLseGm,
                                              LocalTensor<T> lseSrc,
                                              uint64_t bN2Offset,
                                              uint32_t mOffset,
                                              uint32_t dealCount,
                                              const ConstInfo &constInfo,
                                              ActualSeqLensParser<Q_MODE> qActSeqLensParser,
                                              uint64_t bIdx)
{
    uint32_t startS1Idx = mOffset / constInfo.gSize;
    uint32_t startGIdx = mOffset % constInfo.gSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.gSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;
    uint64_t s1LeftPaddingSize = 0;
    if (constInfo.isQHasLeftPadding) {
        s1LeftPaddingSize =
            constInfo.qSeqSize - constInfo.qLeftPaddingSize - qActSeqLensParser.GetActualSeqLength(bIdx);
    }

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + startGIdx * constInfo.qSeqSize + s1Idx + s1LeftPaddingSize;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.gSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.qSeqSize - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    }
}

template <typename T, ActualSeqLensMode Q_MODE>
__aicore__ inline void DataCopySoftmaxLseBNSD(GlobalTensor<float> softmaxLseGm,
                                              LocalTensor<T> lseSrc,
                                              uint64_t bN2Offset,
                                              uint32_t mOffset,
                                              uint32_t dealCount,
                                              const ConstInfo &constInfo,
                                              ActualSeqLensParser<Q_MODE> qActSeqLensParser,
                                              uint64_t bIdx)
{
    uint64_t gOffset = mOffset / qActSeqLensParser.GetActualSeqLength(bIdx) * constInfo.qSeqSize;
    uint64_t seqOffset = mOffset % qActSeqLensParser.GetActualSeqLength(bIdx);
    uint64_t s1LeftPaddingSize = 0;
    if (constInfo.isQHasLeftPadding) {
        s1LeftPaddingSize =
            constInfo.qSeqSize - constInfo.qLeftPaddingSize - qActSeqLensParser.GetActualSeqLength(bIdx);
    }
    uint64_t outOffset = bN2Offset + gOffset + seqOffset + s1LeftPaddingSize;
    uint64_t ubOffset = 0;
    // dealCount ≤ 当前actQs剩余部分，则直接搬运全部dealCount
    if ((qActSeqLensParser.GetActualSeqLength(bIdx) - seqOffset) >= dealCount) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        return;
    }
    // dealCount > 当前actQs剩余部分，分块搬运dealCount
    // dealCount首块
    uint64_t headActSeq = qActSeqLensParser.GetActualSeqLength(bIdx) - seqOffset;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = headActSeq;
    dataCopyParams.blockLen = sizeof(float);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
    outOffset += constInfo.qSeqSize - qActSeqLensParser.GetActualSeqLength(bIdx) + headActSeq;
    ubOffset += headActSeq * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    // dealCount中间块
    uint64_t pendingCount = dealCount - headActSeq;
    while (pendingCount > qActSeqLensParser.GetActualSeqLength(bIdx)) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = qActSeqLensParser.GetActualSeqLength(bIdx);
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        outOffset += constInfo.qSeqSize;
        ubOffset +=
            qActSeqLensParser.GetActualSeqLength(bIdx) * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
        pendingCount -= qActSeqLensParser.GetActualSeqLength(bIdx);
    }
    // dealCount尾块
    if (pendingCount > 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = pendingCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
    }
}

template <typename T>
__aicore__ inline void DataCopySoftmaxLseTND(GlobalTensor<float> softmaxLseGm,
                                             LocalTensor<T> lseSrc,
                                             uint64_t bN2Offset,
                                             uint32_t mOffset,
                                             uint32_t dealCount,
                                             const ConstInfo &constInfo)
{
    uint32_t startS1Idx = mOffset / constInfo.gSize;
    uint32_t startGIdx = mOffset % constInfo.gSize;
    uint32_t endS1Idx = (mOffset + dealCount - 1) / constInfo.gSize;
    uint32_t endGIdx = (mOffset + dealCount - 1) % constInfo.gSize;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t s1Idx = startS1Idx; s1Idx <= endS1Idx; s1Idx++) {
        outOffset = bN2Offset + s1Idx * constInfo.kvHeadNum * constInfo.gSize + startGIdx;
        if (s1Idx != endS1Idx) {
            curDealRowCount = constInfo.gSize - startGIdx;
        } else {
            curDealRowCount = endGIdx + 1 - startGIdx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startGIdx = 0;
        ubOffset += curDealRowCount * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    }
}

template <typename T>
__aicore__ inline void DataCopySoftmaxLseNTD(GlobalTensor<float> softmaxLseGm,
                                             LocalTensor<T> lseSrc,
                                             uint64_t bN2Offset,
                                             uint32_t mOffset,
                                             uint32_t dealCount,
                                             const ConstInfo &constInfo,
                                             uint32_t s1Size)
{
    uint32_t startS1Idx = mOffset % s1Size;
    uint32_t startGIdx = mOffset / s1Size;
    uint32_t endS1Idx = (mOffset + dealCount - 1) % s1Size;
    uint32_t endGIdx = (mOffset + dealCount - 1) / s1Size;
    uint64_t outOffset = 0;
    uint64_t ubOffset = 0;
    uint32_t curDealRowCount = 0;

    for (uint32_t gIdx = startGIdx; gIdx <= endGIdx; gIdx++) {
        outOffset = bN2Offset + startS1Idx * constInfo.kvHeadNum * constInfo.gSize + gIdx;
        if (gIdx != endGIdx) {
            curDealRowCount = s1Size - startS1Idx;
        } else {
            curDealRowCount = endS1Idx + 1 - startS1Idx;
        }
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.gSize * constInfo.kvHeadNum - 1) * sizeof(float);
        DataCopyPad(softmaxLseGm[outOffset], lseSrc[ubOffset], dataCopyParams);
        startS1Idx = 0;
        ubOffset += curDealRowCount * AiInfraInferenceCommonFaBaseVector::FP32_BLOCK_ELEMENT_NUM;
    }
}

// ---------------------------------------Set attention Gm To Zero--------------------------------
template <GmFormat FORMAT, typename OUT_T>
__aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx,
                                           uint32_t n2Idx,
                                           OffsetCalculator<FORMAT> &offsetCalculator,
                                           GlobalTensor<OUT_T> &attentionOutGm)
{
    if constexpr (FORMAT == GmFormat::TNGD) {
        uint32_t s1Count = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(bIdx);
        for (int s1Idx = 0; s1Idx < s1Count; s1Idx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, s1Idx, 0);
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0);
        }
    } else if constexpr (FORMAT == GmFormat::NGTD) {
        uint32_t s1Count = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(bIdx);
        uint32_t gSize = offsetCalculator.GetDimG();
        for (int gIdx = 0; gIdx < gSize; gIdx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, gIdx, 0, 0);
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], s1Count * offsetCalculator.GetDimD(), 0);
        }
    } else if constexpr (FORMAT == GmFormat::BNGSD) {
        uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, 0, 0);
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0);
    } else if constexpr (FORMAT == GmFormat::BSNGD) {
        uint32_t s1Size = offsetCalculator.GetDimS1();
        for (int s1Idx = 0; s1Idx < s1Size; s1Idx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, 0, s1Idx, 0);
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideN2(), 0);
        }
    } else if constexpr (FORMAT == GmFormat::NGBSD) {
        uint32_t gSize = offsetCalculator.GetDimG();
        for (int gIdx = 0; gIdx < gSize; gIdx++) {
            uint64_t attenOutOffset = offsetCalculator.GetOffset(bIdx, n2Idx, gIdx, 0, 0);
            matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], offsetCalculator.GetStrideB(), 0);
        }
    }
}
#endif
