/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 2d 布局（TilingKey 2000-2002）：cache 为连续 [cacheRows, kvCacheCol] 行存，slot 即行号。
// 承载搬运骨架（LoadRows/CopyOutRows）+ 三个场景薄壳，量化计算全部调用 quant_* 文件的核心。

#ifndef KV_COMPRESS_EPILOG_V2_LAYOUT_2D_H
#define KV_COMPRESS_EPILOG_V2_LAYOUT_2D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kv_compress_epilog_v2_common.h"
#include "kv_compress_epilog_v2_quant_fp8.h"
#include "kv_compress_epilog_v2_quant_fp4_g32.h"
#include "kv_compress_epilog_v2_quant_fp4_g16.h"

namespace KvCompressEpilogV2Ops {
using namespace AscendC;

// 2d 搬运骨架：InitCommon 分配队列/缓存，LoadRows 过滤 slot 并聚拢搬入，CopyOutRows 整行融合落盘。
template <typename TX, typename TSlot>
class KvCompressEpilogV2Layout2dCommon {
public:
    __aicore__ inline explicit KvCompressEpilogV2Layout2dCommon(TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void InitCommon(
        GM_ADDR cache, GM_ADDR x, GM_ADDR slotMapping,
        const KvCompressEpilogV2TilingData *tilingData, uint32_t scratchBytes)
    {
        tilingData_ = tilingData;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TX *>(x));
        slotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TSlot *>(slotMapping));
        cacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(cache));
        pipe_->InitBuffer(xQueue_, KCEV2_QUEUE_DEPTH,
                          tilingData_->rowFactor * RoundUp<TX>(tilingData_->d) * sizeof(TX));
        pipe_->InitBuffer(outputQueue_, KCEV2_QUEUE_DEPTH,
                          tilingData_->rowFactor * RoundUp<uint8_t>(tilingData_->kvCacheCol));
        pipe_->InitBuffer(scratchBuffer_, scratchBytes);
        pipe_->InitBuffer(paddingBuffer_, KCEV2_BLOCK_BYTES);
        pipe_->InitBuffer(indexBuffer_, RoundUp<TSlot>(tilingData_->rowFactor) * sizeof(TSlot));
        indexLocal_ = indexBuffer_.Get<TSlot>();
        paddingLocal_ = paddingBuffer_.Get<uint8_t>();
        Duplicate(paddingLocal_, static_cast<uint8_t>(0), KCEV2_BLOCK_BYTES);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    }

protected:
    __aicore__ inline int64_t LoopCount() const
    {
        return GetBlockIdx() == GetBlockNum() - 1 ? tilingData_->rowLoopOfTailBlock :
                                                    tilingData_->rowLoopOfFormerBlock;
    }

    __aicore__ inline int64_t RowsInLoop(int64_t loop, int64_t loops) const
    {
        if (loop != loops - 1) {
            return tilingData_->rowFactor;
        }
        return GetBlockIdx() == GetBlockNum() - 1 ? tilingData_->tailRowFactorOfTailBlock :
                                                    tilingData_->tailRowFactorOfFormerBlock;
    }

    __aicore__ inline LocalTensor<TX> LoadRows(int64_t loop, int64_t rows, int64_t &validRows)
    {
        LocalTensor<TX> xLocal = xQueue_.template AllocTensor<TX>();
        const int64_t baseRow = GetBlockIdx() * tilingData_->rowOfFormerBlock;
        validRows = 0;
        for (int64_t row = 0; row < rows; ++row) {
            const int64_t inputRow = baseRow + loop * tilingData_->rowFactor + row;
            const int64_t slot = static_cast<int64_t>(slotGm_.GetValue(inputRow));
            if (slot < 0 || slot >= tilingData_->cacheRows) {
                continue;
            }
            CopyIn(xGm_[inputRow * tilingData_->d],
                   xLocal[validRows * RoundUp<TX>(tilingData_->d)], tilingData_->d);
            indexLocal_.SetValue(validRows++, static_cast<TSlot>(slot));
        }
        xQueue_.template EnQue<TX>(xLocal);
        return xQueue_.template DeQue<TX>();
    }

    __aicore__ inline void CopyOutRows(const LocalTensor<uint8_t> &output, int64_t validRows)
    {
        for (int64_t row = 0; row < validRows; ++row) {
            const int64_t slot = static_cast<int64_t>(indexLocal_.GetValue(row));
            const int64_t cacheOffset = slot * tilingData_->cacheRowStride;
            CopyOutBytes(output[row * RoundUp<uint8_t>(tilingData_->kvCacheCol)],
                         cacheGm_[cacheOffset], tilingData_->kvCacheCol);
        }
    }

    TPipe *pipe_ = nullptr;
    const KvCompressEpilogV2TilingData *tilingData_ = nullptr;
    GlobalTensor<TX> xGm_;
    GlobalTensor<TSlot> slotGm_;
    GlobalTensor<uint8_t> cacheGm_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> scratchBuffer_;
    TBuf<QuePosition::VECCALC> paddingBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    LocalTensor<TSlot> indexLocal_;
    LocalTensor<uint8_t> paddingLocal_;
};

// TilingKey 2000：FP8-g32，rs=1 走 FullRow 流水，rs=0 走数值基线。
template <typename TX, typename TSlot, typename TCache>
class KvCompressEpilogV2Layout2dFp8Kernel : public KvCompressEpilogV2Layout2dCommon<TX, TSlot> {
    using Base = KvCompressEpilogV2Layout2dCommon<TX, TSlot>;
public:
    __aicore__ inline explicit KvCompressEpilogV2Layout2dFp8Kernel(TPipe *pipe) : Base(pipe) {}

    __aicore__ inline void Init(GM_ADDR cache, GM_ADDR x, GM_ADDR slotMapping,
                                const KvCompressEpilogV2TilingData *tilingData)
    {
        const uint32_t groups = static_cast<uint32_t>(tilingData->d / tilingData->perGroupSize);
        Base::InitCommon(cache, x, slotMapping, tilingData,
                         tilingData->rowFactor * RoundUp<float>(groups) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t loops = Base::LoopCount();
        for (int64_t loop = 0; loop < loops; ++loop) {
            int64_t validRows = 0;
            LocalTensor<TX> xLocal = Base::LoadRows(loop, Base::RowsInLoop(loop, loops), validRows);
            if (validRows == 0) {
                Base::xQueue_.FreeTensor(xLocal);
                continue;
            }
            LocalTensor<uint8_t> output = Base::outputQueue_.template AllocTensor<uint8_t>();
            if (Base::tilingData_->roundScale == 1) {
                LocalTensor<float> scratchF32Local = Base::scratchBuffer_.template Get<float>();
                VFProcessMxFp8FullRowPerf<TCache>(output, xLocal, scratchF32Local,
                    static_cast<uint16_t>(validRows), Base::tilingData_->d, Base::tilingData_->dataCol,
                    Base::tilingData_->concatCol, Base::tilingData_->kvCacheCol);
            } else {
                VFProcessMxFp8<TCache, false>(output, xLocal, static_cast<uint16_t>(validRows),
                    Base::tilingData_->d, Base::tilingData_->dataCol,
                    Base::tilingData_->concatCol, Base::tilingData_->kvCacheCol);
            }
            Base::xQueue_.FreeTensor(xLocal);
            Base::outputQueue_.template EnQue<uint8_t>(output);
            output = Base::outputQueue_.template DeQue<uint8_t>();
            Base::CopyOutRows(output, validRows);
            Base::outputQueue_.FreeTensor(output);
        }
    }
};

// TilingKey 2001：FP4-g32（非主场景，可能日落）。
template <typename TX, typename TSlot>
class KvCompressEpilogV2Layout2dFp4G32Kernel : public KvCompressEpilogV2Layout2dCommon<TX, TSlot> {
    using Base = KvCompressEpilogV2Layout2dCommon<TX, TSlot>;
public:
    __aicore__ inline explicit KvCompressEpilogV2Layout2dFp4G32Kernel(TPipe *pipe) : Base(pipe) {}

    __aicore__ inline void Init(GM_ADDR cache, GM_ADDR x, GM_ADDR slotMapping,
                                const KvCompressEpilogV2TilingData *tilingData)
    {
        const uint32_t groups = static_cast<uint32_t>(tilingData->d / tilingData->perGroupSize);
        Base::InitCommon(cache, x, slotMapping, tilingData,
                         tilingData->rowFactor * RoundUp<uint16_t>(groups) * sizeof(uint16_t));
    }

    __aicore__ inline void Process()
    {
        const int64_t loops = Base::LoopCount();
        for (int64_t loop = 0; loop < loops; ++loop) {
            int64_t validRows = 0;
            LocalTensor<TX> xLocal = Base::LoadRows(loop, Base::RowsInLoop(loop, loops), validRows);
            if (validRows == 0) {
                Base::xQueue_.FreeTensor(xLocal);
                continue;
            }
            LocalTensor<uint8_t> output = Base::outputQueue_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> scratchLocal = Base::scratchBuffer_.template Get<uint16_t>();
            VFProcessMxFp4RowBatchPerf<bfloat16_t>(output, xLocal, scratchLocal,
                static_cast<uint16_t>(validRows), Base::tilingData_->d, Base::tilingData_->dataCol,
                Base::tilingData_->concatCol, Base::tilingData_->kvCacheCol);
            Base::xQueue_.FreeTensor(xLocal);
            Base::outputQueue_.template EnQue<uint8_t>(output);
            output = Base::outputQueue_.template DeQue<uint8_t>();
            Base::CopyOutRows(output, validRows);
            Base::outputQueue_.FreeTensor(output);
        }
    }
};

// TilingKey 2002：FP4-g16（主场景）。
template <typename TX, typename TSlot>
class KvCompressEpilogV2Layout2dFp4G16Kernel : public KvCompressEpilogV2Layout2dCommon<TX, TSlot> {
    using Base = KvCompressEpilogV2Layout2dCommon<TX, TSlot>;
public:
    __aicore__ inline explicit KvCompressEpilogV2Layout2dFp4G16Kernel(TPipe *pipe) : Base(pipe) {}

    __aicore__ inline void Init(GM_ADDR cache, GM_ADDR x, GM_ADDR slotMapping,
                                const KvCompressEpilogV2TilingData *tilingData)
    {
        const uint32_t groups = static_cast<uint32_t>(tilingData->d / tilingData->perGroupSize);
        Base::InitCommon(cache, x, slotMapping, tilingData,
                         tilingData->rowFactor * RoundUp<uint16_t>(groups) * sizeof(uint16_t));
    }

    __aicore__ inline void Process()
    {
        const int64_t loops = Base::LoopCount();
        for (int64_t loop = 0; loop < loops; ++loop) {
            int64_t validRows = 0;
            LocalTensor<TX> xLocal = Base::LoadRows(loop, Base::RowsInLoop(loop, loops), validRows);
            if (validRows == 0) {
                Base::xQueue_.FreeTensor(xLocal);
                continue;
            }
            LocalTensor<uint8_t> output = Base::outputQueue_.template AllocTensor<uint8_t>();
            LocalTensor<uint16_t> scratchLocal = Base::scratchBuffer_.template Get<uint16_t>();
            VFProcessMxFp4Group16RowBatchPerf(output, xLocal, scratchLocal,
                static_cast<uint16_t>(validRows), Base::tilingData_->d, Base::tilingData_->dataCol,
                Base::tilingData_->concatCol, Base::tilingData_->kvCacheCol);
            Base::xQueue_.FreeTensor(xLocal);
            Base::outputQueue_.template EnQue<uint8_t>(output);
            output = Base::outputQueue_.template DeQue<uint8_t>();
            Base::CopyOutRows(output, validRows);
            Base::outputQueue_.FreeTensor(output);
        }
    }
};

}  // namespace KvCompressEpilogV2Ops

#endif
