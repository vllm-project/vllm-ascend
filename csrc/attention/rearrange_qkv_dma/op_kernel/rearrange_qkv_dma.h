// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vllm-ascend project

#pragma once

#include "kernel_operator.h"

using namespace AscendC;

namespace rearrange_qkv_impl {
constexpr uint32_t BYTES_PER_ELEMENT = sizeof(uint16_t);
constexpr uint32_t DATABLOCK_BYTES = 32;
constexpr uint32_t ELEMENTS_PER_DATABLOCK = DATABLOCK_BYTES / BYTES_PER_ELEMENT;
constexpr uint32_t TILE_BYTES = 160 * 1024;
constexpr uint32_t TILE_ELEMENTS = TILE_BYTES / BYTES_PER_ELEMENT;
constexpr uint32_t LOAD_TO_STORE_EVENT_ID = 0;
constexpr uint32_t STORE_TO_LOAD_EVENT_ID = 1;

class RearrangeQkvKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RearrangeQkvDmaTilingData& tiling, TPipe* pipe)
    {
        tokens_ = tiling.tokens;
        qDim_ = tiling.qDim;
        kDim_ = tiling.kDim;
        vDim_ = tiling.vDim;
        rowDim_ = tiling.rowDim;
        tileRows_ = tiling.tileRows;
        usedCoreNum_ = tiling.usedCoreNum;
        // This kernel performs no arithmetic. bfloat16_t is only a 16-bit DMA
        // storage type, so BF16 and FP16 payloads are copied bit-for-bit.
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
        x_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        y_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
        pipe->InitBuffer(copyBuffer_, TILE_BYTES);
    }

    __aicore__ inline void Process()
    {
        const uint64_t coreId = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t baseRows = tokens_ / usedCoreNum_;
        const uint64_t extraRows = tokens_ % usedCoreNum_;
        const uint64_t coreRows = baseRows + (coreId < extraRows ? 1 : 0);
        const uint64_t rowStart = coreId * baseRows + (coreId < extraRows ? coreId : extraRows);
        LocalTensor<bfloat16_t> local = copyBuffer_.Get<bfloat16_t>();

        uint64_t row = rowStart;
        const uint64_t rowEnd = rowStart + coreRows;
        while (row < rowEnd) {
            const uint64_t qDst = row * qDim_;
            const uint64_t kDst = tokens_ * qDim_ + row * kDim_;
            const uint64_t vDst = tokens_ * (qDim_ + kDim_) + row * vDim_;
            if (tileRows_ == 0) {
                const uint64_t src = row * rowDim_;
                CopySegment(local, src, qDst, qDim_);
                CopySegment(local, src + qDim_, kDst, kDim_);
                CopySegment(local, src + qDim_ + kDim_, vDst, vDim_);
                ++row;
                continue;
            }

            uint64_t rows = rowEnd - row;
            if (rows > tileRows_) {
                rows = tileRows_;
            }
            DataCopy(local, x_[row * rowDim_], static_cast<uint32_t>(rows * rowDim_));
            WaitLoadBeforeStore();

            const DataCopyParams qParams{
                static_cast<uint16_t>(rows), static_cast<uint16_t>(qDim_ / ELEMENTS_PER_DATABLOCK),
                static_cast<uint16_t>((rowDim_ - qDim_) / ELEMENTS_PER_DATABLOCK), 0};
            const DataCopyParams kParams{
                static_cast<uint16_t>(rows), static_cast<uint16_t>(kDim_ / ELEMENTS_PER_DATABLOCK),
                static_cast<uint16_t>((rowDim_ - kDim_) / ELEMENTS_PER_DATABLOCK), 0};
            const DataCopyParams vParams{
                static_cast<uint16_t>(rows), static_cast<uint16_t>(vDim_ / ELEMENTS_PER_DATABLOCK),
                static_cast<uint16_t>((rowDim_ - vDim_) / ELEMENTS_PER_DATABLOCK), 0};
            DataCopy(y_[qDst], local, qParams);
            DataCopy(y_[kDst], local[qDim_], kParams);
            DataCopy(y_[vDst], local[qDim_ + kDim_], vParams);
            WaitStoreBeforeLoad();
            row += rows;
        }
    }

private:
    __aicore__ inline void WaitLoadBeforeStore()
    {
        SetFlag<HardEvent::MTE2_MTE3>(LOAD_TO_STORE_EVENT_ID);
        WaitFlag<HardEvent::MTE2_MTE3>(LOAD_TO_STORE_EVENT_ID);
    }

    __aicore__ inline void WaitStoreBeforeLoad()
    {
        SetFlag<HardEvent::MTE3_MTE2>(STORE_TO_LOAD_EVENT_ID);
        WaitFlag<HardEvent::MTE3_MTE2>(STORE_TO_LOAD_EVENT_ID);
    }

    __aicore__ inline void CopySegment(LocalTensor<bfloat16_t> local, uint64_t src, uint64_t dst, uint64_t width)
    {
        for (uint64_t offset = 0; offset < width; offset += TILE_ELEMENTS) {
            uint64_t count = width - offset;
            if (count > TILE_ELEMENTS) {
                count = TILE_ELEMENTS;
            }
            DataCopy(local, x_[src + offset], static_cast<uint32_t>(count));
            WaitLoadBeforeStore();
            const DataCopyParams params{
                1, static_cast<uint16_t>(count / ELEMENTS_PER_DATABLOCK), 0, 0};
            DataCopy(y_[dst + offset], local, params);
            WaitStoreBeforeLoad();
        }
    }

    GlobalTensor<bfloat16_t> x_;
    GlobalTensor<bfloat16_t> y_;
    TBuf<TPosition::A1> copyBuffer_;
    uint64_t tokens_ = 0;
    uint64_t qDim_ = 0;
    uint64_t kDim_ = 0;
    uint64_t vDim_ = 0;
    uint64_t rowDim_ = 0;
    uint32_t tileRows_ = 0;
    uint32_t usedCoreNum_ = 1;
};
}  // namespace rearrange_qkv_impl
