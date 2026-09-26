// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
#ifndef FLASH_MLA_C8_DCP_OUTPUT_H
#define FLASH_MLA_C8_DCP_OUTPUT_H

#include "flash_mla_c8_public.h"
#include "flash_mla_c8_memory.h"
#include <limits>

namespace BaseApi {

// Only final GM storage is wider. The attention arithmetic, cast source and
// split-K workspace keep their original 512-column shape.
constexpr uint32_t C8_DCP_VALUE_DIM = 512;
constexpr uint32_t C8_DCP_ROW_BF16 = 544;
constexpr uint32_t C8_DCP_ROW_WORDS = 272;
constexpr uint32_t C8_DCP_LSE_WORD = 256;

template <typename OUTPUT_T, uint32_t EVENT_ID, typename CONST_INFO_T>
__aicore__ inline void InitDcpOutput(GlobalTensor<OUTPUT_T> output,
                                    GlobalTensor<float> wire, GlobalTensor<float> lse,
                                    const CONST_INFO_T &info)
{
    static_assert(IsSameType<OUTPUT_T, bfloat16_t>::value);
    const uint64_t totalRows = static_cast<uint64_t>(info.t1Size) * info.realN2Size * info.realGSize;
    const uint64_t rowsPerCore = (totalRows + 2 * info.coreNum - 1) / (2 * info.coreNum);
    const uint64_t firstRow = info.aivIdx * rowsPerCore;
    if (firstRow >= totalRows) {
        return;
    }
    const uint64_t rowCount = totalRows - firstRow < rowsPerCore ? totalRows - firstRow : rowsPerCore;
    SetFlag<HardEvent::MTE3_V>(EVENT_ID);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID);
    matmul::InitOutput<OUTPUT_T>(output[firstRow * C8_DCP_ROW_BF16], rowCount * C8_DCP_ROW_BF16, 0);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID);
    matmul::InitOutput<float>(lse[firstRow], rowCount, -std::numeric_limits<float>::infinity());
    SetFlag<HardEvent::MTE3_V>(EVENT_ID);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID);

    // ClearOutput runs before the ordinary TPipe buffers are allocated. As in
    // InitOutput, this scratch may use UB address zero and must drain MTE3 before
    // returning. Each 4-byte transfer consumes one 32-byte source block.
    constexpr uint32_t initRows = 128;
    constexpr uint32_t floatsPerBlock = 8;
    LocalTensor<float> initLse = LocalTensor<uint8_t>(TPosition::VECIN, 0, initRows * 32)
                                   .template ReinterpretCast<float>();
    Duplicate(initLse, -std::numeric_limits<float>::infinity(), initRows * floatsPerBlock);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID);
    for (uint64_t done = 0; done < rowCount; done += initRows) {
        const uint32_t count = rowCount - done < initRows ? rowCount - done : initRows;
        DataCopyExtParams params;
        params.blockCount = count;
        params.blockLen = sizeof(float);
        params.srcStride = 0;
        params.dstStride = (C8_DCP_ROW_WORDS - 1) * sizeof(float);
        DataCopyPad(wire[(firstRow + done) * C8_DCP_ROW_WORDS + C8_DCP_LSE_WORD], initLse, params);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID);
}

template <typename OUTPUT_T, typename CONST_INFO_T>
__aicore__ inline void CopyDcpAttentionOut(GlobalTensor<OUTPUT_T> output, FaUbTensor<OUTPUT_T> &src,
                                          const GmCoordGs1Merge &coord, uint32_t prefixT,
                                          const CONST_INFO_T &info)
{
    static_assert(IsSameType<OUTPUT_T, bfloat16_t>::value);
    // TND input has S1G rows in UB; split a tile only when it crosses a token.
    uint32_t done = 0;
    const uint64_t headStride = static_cast<uint64_t>(info.t1Size) * C8_DCP_ROW_BF16;
    while (done < coord.gS1DealSize) {
        const uint32_t m = coord.gS1Idx + done;
        const uint32_t g = m % info.realGSize;
        const uint32_t t = prefixT + m / info.realGSize;
        const uint32_t left = coord.gS1DealSize - done;
        const uint32_t count = info.realGSize - g < left ? info.realGSize - g : left;
        const uint64_t row = static_cast<uint64_t>(coord.n2Idx * info.realGSize + g) * info.t1Size + t;
        const uint32_t bytes = coord.dDealSize * sizeof(OUTPUT_T);
        const uint32_t srcStride = (src.colCount - coord.dDealSize) * sizeof(OUTPUT_T) / 32;
        SafeStrideCopy<OUTPUT_T>(output[row * C8_DCP_ROW_BF16 + coord.dIdx], src.tensor[done * src.colCount],
                                count, bytes, srcStride, headStride * sizeof(OUTPUT_T) - bytes);
        done += count;
    }
}

template <typename T, typename CONST_INFO_T>
__aicore__ inline void CopyDcpLse(GlobalTensor<float> wire, LocalTensor<T> src,
                                 uint32_t n2Idx, uint32_t mOffset, uint32_t dealCount,
                                 uint32_t prefixT, const CONST_INFO_T &info)
{
    static_assert(IsSameType<T, float>::value);
    uint32_t done = 0;
    const uint64_t headStrideBytes = static_cast<uint64_t>(info.t1Size) * C8_DCP_ROW_WORDS * sizeof(float);
    while (done < dealCount) {
        const uint32_t m = mOffset + done;
        const uint32_t g = m % info.realGSize;
        const uint32_t t = prefixT + m / info.realGSize;
        const uint32_t left = dealCount - done;
        const uint32_t count = info.realGSize - g < left ? info.realGSize - g : left;
        const uint64_t row = static_cast<uint64_t>(n2Idx * info.realGSize + g) * info.t1Size + t;
        // Native LSE already repeats each FP32 value in an aligned 32B block.
        // Copy 4B without a cast, preserving the same bits as independent LSE.
        SafeStrideCopy<float>(wire[row * C8_DCP_ROW_WORDS + C8_DCP_LSE_WORD], src[done * 8],
                              count, sizeof(float), 0, headStrideBytes - sizeof(float));
        done += count;
    }
}

} // namespace BaseApi
#endif
