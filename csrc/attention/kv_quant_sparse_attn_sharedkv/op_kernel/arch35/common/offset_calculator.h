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
 * \file offset_calculator.h
 * \brief
 */
#ifndef OFFSET_CALCULATOR_H
#define OFFSET_CALCULATOR_H

#include "kernel_operator.h"

using namespace AscendC;
using AscendC::GlobalTensor;

static constexpr uint32_t SHAPE_AXIS_DIM_0 = 0U;
static constexpr uint32_t SHAPE_AXIS_DIM_1 = 1U;
static constexpr uint32_t SHAPE_AXIS_DIM_2 = 2U;
static constexpr uint32_t SHAPE_AXIS_DIM_3 = 3U;
static constexpr uint32_t SHAPE_AXIS_DIM_4 = 4U;

// ----------------------------------------------GmLayout--------------------------------
enum class GmFormat
{
    BSNGD = 0,
    BNGSD = 1,
    NGBSD = 2,
    TNGD = 3,
    NGTD = 4,
    BSND = 5,
    BNSD = 6,
    TND = 7,
    NTD = 8,
    PA_BNBSND = 9,
    PA_BNNBSD = 10,
    PA_NZ = 11,
    SBNGD = 12,
    SBND = 13
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
struct GmLayout<GmFormat::PA_BNBSND> {
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
struct GmLayout<GmFormat::PA_BNNBSD> {
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

// ----------------------------------------------ActualSeqLensParser--------------------------------
enum class ActualSeqLensMode
{
    BY_BATCH = 0,
    ACCUM = 1,
};

template <ActualSeqLensMode MODE>
class ActualSeqLensParser {
};

template <>
class ActualSeqLensParser<ActualSeqLensMode::ACCUM> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<int32_t> actualSeqLengthsGm, uint32_t actualLenDims)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
    }

    __aicore__ inline int64_t GetTBase(uint32_t bIdx) const
    {
        return actualSeqLengthsGm.GetValue(bIdx);
    }

    __aicore__ inline int64_t GetActualSeqLength(uint32_t bIdx) const
    {
        return (actualSeqLengthsGm.GetValue(bIdx + 1) - actualSeqLengthsGm.GetValue(bIdx));
    }

    __aicore__ inline int64_t GetTSize() const
    {
        return actualSeqLengthsGm.GetValue(actualLenDims - 1);
    }
private:
    GlobalTensor<int32_t> actualSeqLengthsGm;
    uint32_t actualLenDims;
};

template <>
class ActualSeqLensParser<ActualSeqLensMode::BY_BATCH> {
public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<int32_t> actualSeqLengthsGm, uint32_t actualLenDims, int64_t defaultVal)
    {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
        this->defaultVal = defaultVal;
    }

    __aicore__ inline int64_t GetActualSeqLength(uint32_t bIdx) const
    {
        if (actualLenDims == 0) {
            return defaultVal;
        }
        if (actualLenDims == 1) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return actualSeqLengthsGm.GetValue(bIdx);
    }
private:
    GlobalTensor<int32_t> actualSeqLengthsGm;
    uint32_t actualLenDims;
    int64_t defaultVal;
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
enum class FormatCategory
{
    GM_Q_OUT_BNGSD = 0,
    GM_Q_OUT_TND = 1,
    GM_KV_BNSD = 2,
    GM_KV_TND = 3,
    GM_KV_PA_BNBD = 4,
    GM_KV_PA_NZ = 5,
};

template <GmFormat FORMAT>
struct GmLayoutParams {};

template <>
struct GmLayoutParams<GmFormat::BSNGD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_BNGSD;
};

template <>
struct GmLayoutParams<GmFormat::BNGSD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_BNGSD;
};

template <>
struct GmLayoutParams<GmFormat::NGBSD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_BNGSD;
};

template <>
struct GmLayoutParams<GmFormat::TNGD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_TND;
};

template <>
struct GmLayoutParams<GmFormat::NGTD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_TND;
};

template <>
struct GmLayoutParams<GmFormat::BSND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_BNSD;
};

template <>
struct GmLayoutParams<GmFormat::BNSD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_BNSD;
};

template <>
struct GmLayoutParams<GmFormat::TND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_TND;
};

template <>
struct GmLayoutParams<GmFormat::NTD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_TND;
};

template <>
struct GmLayoutParams<GmFormat::PA_BNBSND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_BNBD;
};

template <>
struct GmLayoutParams<GmFormat::PA_BNNBSD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_BNBD;
};

template <>
struct GmLayoutParams<GmFormat::PA_NZ> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_NZ;
};

template <>
struct GmLayoutParams<GmFormat::SBNGD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_BNGSD;
};

template <>
struct GmLayoutParams<GmFormat::SBND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_BNSD;
};

// ----------------------------------------------OffsetCalculator--------------------------------
template <GmFormat FORMAT, FormatCategory CATEGORY>
struct OffsetCalculatorImpl {};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_Q_OUT_BNGSD> {
    GmLayout<FORMAT> gmLayout;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t d)
    {
        gmLayout.MakeLayout(b, n2, g, s1, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t gIdx, uint32_t s1Idx, uint32_t dIdx)
    {
        uint64_t offset = bIdx * GetStrideB() + n2Idx * GetStrideN2() + gIdx * GetStrideG() + s1Idx * GetStrideS1() +
                          dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideG()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideS1()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_3>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_4>(gmLayout.stride);
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimB()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimG()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimS1()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_3>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_4>(gmLayout.shape);
    }
};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_Q_OUT_TND> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::ACCUM> actualSeqLensQParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t g, uint32_t d, GlobalTensor<int32_t> actualSeqLengthsGmQ,
                                uint32_t actualLenQDims)
    {
        actualSeqLensQParser.Init(actualSeqLengthsGmQ, actualLenQDims);
        gmLayout.MakeLayout(actualSeqLensQParser.GetTSize(), n2, g, d);
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
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideG()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_3>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideS1()
    {
        return GetStrideT();
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimT()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimG()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_3>(gmLayout.shape);
    }
};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_BNSD> {
    GmLayout<FORMAT> gmLayout;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d)
    {
        gmLayout.MakeLayout(b, n2, s2, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t offset = bIdx * GetStrideB() + n2Idx * GetStrideN2() + s2Idx * GetStrideS2() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideB()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_3>(gmLayout.stride);
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimB()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimS2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_3>(gmLayout.shape);
    }
};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_TND> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::ACCUM> actualSeqLensKVParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t d, GlobalTensor<int32_t> actualSeqLengthsGmKV,
                                uint32_t actualLenKVDims)
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
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideS2()
    {
        return GetStrideT();
    }

    // Get Dim
    __aicore__ inline uint64_t GetDimT()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetDimD()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.shape);
    }
};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_BNBD> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t blockSize, uint32_t d, GlobalTensor<int32_t> blockTableGm,
                                uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n2, blockSize, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t blockIdxInBatch = s2Idx / GetBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = s2Idx % GetBlockSize();           // 获取在单个块上超出的行数
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);
        uint64_t offset =
            blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() + bsIdx * GetStrideBlockSize() + dIdx * GetStrideD();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_3>(gmLayout.stride);
    }

    // Get Dim
    __aicore__ inline uint64_t GetN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBlockSize()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.shape);
    }
};

template <GmFormat FORMAT>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_NZ> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t blockSize, uint32_t d1, uint32_t d0,
                                GlobalTensor<int32_t> blockTableGm, uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n2, blockSize, d1, d0);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t blockIdxInBatch = s2Idx / GetBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = s2Idx % GetBlockSize();           // 获取在单个块上超出的行数
        int32_t blockIdx = blockTableParser.GetBlockIdx(bIdx, blockIdxInBatch);

        uint32_t d1Idx = dIdx / GetD0();
        uint32_t d0Idx = dIdx % GetD0();
        uint64_t offset = blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() +
                          d1Idx * GetStrideD1() + bsIdx * GetStrideBlockSize() + d0Idx * GetStrideD0();
        return offset;
    }

    // Get Stride
    __aicore__ inline uint64_t GetStrideBlockNum()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD1()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideBlockSize()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_3>(gmLayout.stride);
    }

    __aicore__ inline uint64_t GetStrideD0()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_4>(gmLayout.stride);
    }

    // Get Dim
    __aicore__ inline uint64_t GetN2()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD1()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBlockSize()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_2>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD0()
    {
        return AscendC::Std::get<SHAPE_AXIS_DIM_3>(gmLayout.shape);
    }
};

template <GmFormat FORMAT>
struct OffsetCalculator : public OffsetCalculatorImpl<FORMAT, GmLayoutParams<FORMAT>::CATEGORY> {
};

// ----------------------------------------------CopyQueryGmToL1--------------------------------
template <typename Q_T, GmFormat FORMAT>
struct FaGmTensor {
    GlobalTensor<Q_T> gmTensor;
    OffsetCalculator<FORMAT> offsetCalculator;
};

enum class L1Format
{
    NZ = 0
};

template <typename Q_T, L1Format FORMAT>
struct FaL1Tensor {
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
#endif
