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
 * \file offset_calculator_v2.h
 * \brief
 */
#ifndef OFFSET_CALCULATOR_V2_H
#define OFFSET_CALCULATOR_V2_H

#include "gm_layout.h"
#include "parser.h"

using AscendC::GlobalTensor;

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
    GM_PSE_BN2GS1S2 = 12, // PSE
    GM_V_SCALE_TND = 13,
    GM_K_SCALE_PA_NZ = 14,
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
struct GmLayoutParams<GmFormat::NTGD> {
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
struct GmLayoutParams<GmFormat::PA_BnBsND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_BNBD;
};


template <>
struct GmLayoutParams<GmFormat::PA_BnNBsD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_BNBD;
};

template <>
struct GmLayoutParams<GmFormat::PA_NZ> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_PA_NZ;
};

template <>
struct GmLayoutParams<GmFormat::PA_NZ_K_SCALE> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_K_SCALE_PA_NZ;
};

template <>
struct GmLayoutParams<GmFormat::SBNGD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_Q_OUT_BNGSD;
};

template <>
struct GmLayoutParams<GmFormat::SBND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_KV_BNSD;
};

// post_quant
template <>
struct GmLayoutParams<GmFormat::NGD> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_POST_QUANT_NGD;
};

// antiquant
template <>
struct GmLayoutParams<GmFormat::ND> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_ND;
};
template <>
struct GmLayoutParams<GmFormat::BS2> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BS;
};
template <>
struct GmLayoutParams<GmFormat::BNS2> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BNS;
};
template <>
struct GmLayoutParams<GmFormat::PA_BnBs> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BnBs;
};
template <>
struct GmLayoutParams<GmFormat::PA_BnNBs> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_ANTIQ_BnNBs;
};

// pse
template <>
struct GmLayoutParams<GmFormat::BN2GS1S2> {
    static constexpr FormatCategory CATEGORY = FormatCategory::GM_PSE_BN2GS1S2;
};

// ----------------------------------------------OffsetCalculator--------------------------------
template <GmFormat FORMAT, FormatCategory CATEGORY, typename ACTLEN_T>
struct OffsetCalculatorImpl {};

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_Q_OUT_BNGSD, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, ACTLEN_T> actualSeqLensQParser;
    bool isQPaddingFlag = false;
    uint64_t qPaddingSize = 0;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t d)
    {
        gmLayout.MakeLayout(b, n2, g, s1, d);
    }

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t d,
                                GlobalTensor<ACTLEN_T> actualSeqLengthsGmQ, uint32_t actualLenQDims,
                                bool isQPaddingFlag = false, uint64_t qPaddingSize = 0)
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

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_Q_OUT_TND, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T> actualSeqLensQParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t g, uint32_t d, GlobalTensor<ACTLEN_T> actualSeqLengthsGmQ,
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

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_BNSD, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, ACTLEN_T> actualSeqLensKVParser;
    bool isKvPaddingFlag = false;
    uint64_t kvPaddingSize = 0;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d)
    {
        gmLayout.MakeLayout(b, n2, s2, d);
    }

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t s2, uint32_t d,
                                GlobalTensor<ACTLEN_T> actualSeqLengthsGm, uint32_t actualLenKvDims,
                                bool isKvPaddingFlag = false, uint64_t kvPaddingSize = 0)
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

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_TND, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T> actualSeqLensKVParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t d, GlobalTensor<ACTLEN_T> actualSeqLengthsGmKV,
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

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_V_SCALE_TND, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T> actualSeqLensKVParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n2, uint32_t d, GlobalTensor<ACTLEN_T> actualSeqLengthsGmKV,
                                uint32_t actualLenKVDims)
    {
        actualSeqLensKVParser.Init(actualSeqLengthsGmKV, actualLenKVDims);
        gmLayout.MakeLayout(actualSeqLensKVParser.GetTSize(), n2, d);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t s2Idx, uint32_t dIdx)
    {
        uint64_t tIdx = actualSeqLensKVParser.GetMxVscaleTBase(bIdx) + s2Idx;
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

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_BNBD, ACTLEN_T> {
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

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_NZ, ACTLEN_T> {
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

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_K_SCALE_PA_NZ, ACTLEN_T> {
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

        constexpr uint32_t bs0 = 16;
        uint32_t bs1Idx = bsIdx / bs0;
        uint32_t bs0Idx = bsIdx % bs0;

        uint64_t offset = blockIdx * GetStrideBlockNum() + n2Idx * GetStrideN2() + bs1Idx * GetStrideBlockSize1() +
                          dIdx * GetStrideD() + bs0Idx * GetStrideBlockSize0();
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

    __aicore__ inline uint64_t GetStrideBlockSize1()
    {
        return AscendC::Std::get<2>(gmLayout.stride); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideD()
    {
        return AscendC::Std::get<3>(gmLayout.stride); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetStrideBlockSize0()
    {
        return AscendC::Std::get<4>(gmLayout.stride); // 4:代表第5个维度，索引从0开始
    }

    // Get Dim
    __aicore__ inline uint64_t GetN2()
    {
        return AscendC::Std::get<0>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetBlockSize1()
    {
        return AscendC::Std::get<1>(gmLayout.shape);
    }

    __aicore__ inline uint64_t GetD()
    {
        return AscendC::Std::get<2>(gmLayout.shape); // 2:代表第3个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetBlockSize0()
    {
        return AscendC::Std::get<3>(gmLayout.shape); // 3:代表第4个维度，索引从0开始
    }

    __aicore__ inline uint64_t GetBlockSize()
    {
        return GetBlockSize1() * 16;
    }
};

// post_quant
template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_POST_QUANT_NGD, ACTLEN_T> {
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
template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_ND, ACTLEN_T> {
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

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BS, ACTLEN_T> {
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

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BNS, ACTLEN_T> {
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

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BnBs, ACTLEN_T> {
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
        uint64_t bsIdx = sIdx % GetStrideBlockSize();           // 获取在单个块上超出的行数
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

template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_ANTIQ_BnNBs, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    BlockTableParser blockTableParser;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t n, uint32_t blockSize, GlobalTensor<int32_t> blockTableGm,
                                uint32_t maxblockNumPerBatch)
    {
        blockTableParser.Init(blockTableGm, maxblockNumPerBatch);
        gmLayout.MakeLayout(n, blockSize);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t nIdx, uint32_t sIdx)
    {
        uint64_t blockIdxInBatch = sIdx / GetStrideBlockSize(); // 获取block table上的索引
        uint64_t bsIdx = sIdx % GetStrideBlockSize();           // 获取在单个块上超出的行数
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
template <GmFormat FORMAT, typename ACTLEN_T>
struct OffsetCalculatorImpl<FORMAT, FormatCategory::GM_PSE_BN2GS1S2, ACTLEN_T> {
    GmLayout<FORMAT> gmLayout;
    ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, ACTLEN_T> actualSeqLensQParser;
    bool isQPaddingFlag = false;
    uint64_t qPaddingSize = 0;

    __aicore__ inline OffsetCalculatorImpl() = default;

    __aicore__ inline void Init(uint32_t b, uint32_t n2, uint32_t g, uint32_t s1, uint32_t s2,
                                GlobalTensor<ACTLEN_T> actualSeqLengthsGmQ, uint32_t actualLenQDims,
                                bool isQPaddingFlag = false, uint64_t qPaddingSize = 0)
    {
        this->isQPaddingFlag = isQPaddingFlag;
        this->qPaddingSize = qPaddingSize;
        if (actualLenQDims != 0) {
            actualSeqLensQParser.Init(actualSeqLengthsGmQ, actualLenQDims, 0);
        }
        gmLayout.MakeLayout(b, n2, g, s1, s2);
    }

    __aicore__ inline uint64_t GetOffset(uint32_t bIdx, uint32_t n2Idx, uint32_t gIdx, uint32_t s1Idx, uint32_t s2Idx)
    {
        if (isQPaddingFlag) {
            s1Idx += GetDimS1() - qPaddingSize - actualSeqLensQParser.GetActualSeqLength(bIdx);
        }
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

template <GmFormat FORMAT, typename ACTLEN_T = uint64_t>
struct OffsetCalculator : public OffsetCalculatorImpl<FORMAT, GmLayoutParams<FORMAT>::CATEGORY, ACTLEN_T> {};

#endif