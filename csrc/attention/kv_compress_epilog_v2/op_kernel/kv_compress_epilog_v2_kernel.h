/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 退役基线壳（TilingKey 2000/2001 原始实现）：入口已不实例化，仅作回滚参考保留。
// 其独占依赖（VFProcessMxFp4Verified 等）在 quant_fp4_g32.h 尾部"基线参考"段。

#ifndef KV_COMPRESS_EPILOG_V2_KERNEL_H
#define KV_COMPRESS_EPILOG_V2_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kv_compress_epilog_v2_common.h"
#include "kv_compress_epilog_v2_quant_fp8.h"
#include "kv_compress_epilog_v2_quant_fp4_g32.h"

namespace KvCompressEpilogV2Ops {
using namespace AscendC;

template <typename TX, typename TSlot, typename TCache, bool isMxFp4>
class KvCompressEpilogV2Kernel {
public:
    __aicore__ inline explicit KvCompressEpilogV2Kernel(TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR cache, GM_ADDR x, GM_ADDR slotMapping, const KvCompressEpilogV2TilingData *tilingData)
    {
        tilingData_ = tilingData;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TX *>(x));
        slotGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TSlot *>(slotMapping));
        cacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(cache));
        pipe_->InitBuffer(xQueue_, 2,
                          tilingData_->rowFactor * RoundUp<TX>(tilingData_->d) * sizeof(TX));
        pipe_->InitBuffer(outputQueue_, 2,
                          tilingData_->rowFactor * RoundUp<uint8_t>(tilingData_->kvCacheCol));
        pipe_->InitBuffer(scaleQueue_, 2,
                          tilingData_->rowFactor * ScaleRowBytesMxFp4(tilingData_->scaleCol));
        pipe_->InitBuffer(maxExpBuffer_, RoundUp<uint16_t>(tilingData_->scaleCol) * sizeof(uint16_t));
        pipe_->InitBuffer(halfScaleBuffer_, RoundUp<uint16_t>(tilingData_->scaleCol) * sizeof(uint16_t));
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

    __aicore__ inline void Process()
    {
        const int64_t block = GetBlockIdx();
        const int64_t rowLoops = block == GetBlockNum() - 1 ?
            tilingData_->rowLoopOfTailBlock : tilingData_->rowLoopOfFormerBlock;
        const int64_t tailRows = block == GetBlockNum() - 1 ?
            tilingData_->tailRowFactorOfTailBlock : tilingData_->tailRowFactorOfFormerBlock;
        const int64_t blockRowBase = block * tilingData_->rowOfFormerBlock;

        for (int64_t loop = 0; loop < rowLoops; ++loop) {
            const int64_t currentRows = loop == rowLoops - 1 ? tailRows : tilingData_->rowFactor;
            LocalTensor<TX> xLocal = xQueue_.template AllocTensor<TX>();
            int64_t validRows = 0;
            for (int64_t row = 0; row < currentRows; ++row) {
                const int64_t inputRow = blockRowBase + loop * tilingData_->rowFactor + row;
                const int64_t slot = static_cast<int64_t>(slotGm_.GetValue(inputRow));
                if (slot < 0 || slot >= tilingData_->cacheRows) {
                    continue;
                }
                CopyIn(xGm_[inputRow * tilingData_->d],
                       xLocal[validRows * RoundUp<TX>(tilingData_->d)], tilingData_->d);
                indexLocal_.SetValue(validRows, static_cast<TSlot>(slot));
                ++validRows;
            }
            xQueue_.template EnQue<TX>(xLocal);
            xLocal = xQueue_.template DeQue<TX>();
            if (validRows == 0) {
                xQueue_.FreeTensor(xLocal);
                continue;
            }

            LocalTensor<uint8_t> outputLocal = outputQueue_.template AllocTensor<uint8_t>();
            if constexpr (isMxFp4) {
                LocalTensor<bfloat16_t> scaleLocal = scaleQueue_.template AllocTensor<bfloat16_t>();
                LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
                LocalTensor<uint16_t> halfScaleLocal = halfScaleBuffer_.Get<uint16_t>();
                VFProcessMxFp4Verified(outputLocal.ReinterpretCast<int8_t>(), scaleLocal, xLocal,
                                       maxExpLocal, halfScaleLocal, static_cast<uint16_t>(validRows), tilingData_->d);
                scaleQueue_.template EnQue<bfloat16_t>(scaleLocal);
            } else if (tilingData_->roundScale == 1) {
                VFProcessMxFp8<TCache, true>(outputLocal, xLocal, static_cast<uint16_t>(validRows),
                                             tilingData_->d, tilingData_->dataCol,
                                             tilingData_->concatCol, tilingData_->kvCacheCol);
            } else {
                VFProcessMxFp8<TCache, false>(outputLocal, xLocal, static_cast<uint16_t>(validRows),
                                              tilingData_->d, tilingData_->dataCol,
                                              tilingData_->concatCol, tilingData_->kvCacheCol);
            }
            xQueue_.FreeTensor(xLocal);
            outputQueue_.template EnQue<uint8_t>(outputLocal);
            outputLocal = outputQueue_.template DeQue<uint8_t>();
            LocalTensor<bfloat16_t> scaleLocal;
            if constexpr (isMxFp4) {
                scaleLocal = scaleQueue_.template DeQue<bfloat16_t>();
            }
            for (int64_t row = 0; row < validRows; ++row) {
                const int64_t slot = static_cast<int64_t>(indexLocal_.GetValue(row));
                const int64_t cacheOffset = slot * tilingData_->cacheRowStride;
                if constexpr (isMxFp4) {
                    CopyOutBytes(outputLocal[row * RoundUp<int8_t>(tilingData_->dataCol)],
                                 cacheGm_[cacheOffset], tilingData_->dataCol);
                    CopyOutBytes(
                        scaleLocal.ReinterpretCast<uint8_t>()[row * ScaleRowBytesMxFp4(tilingData_->scaleCol)],
                        cacheGm_[cacheOffset + tilingData_->dataCol], tilingData_->scaleCol * sizeof(bfloat16_t));
                    if (tilingData_->padCol > 0) {
                        CopyOutBytes(
                            paddingLocal_, cacheGm_[cacheOffset + tilingData_->concatCol], tilingData_->padCol);
                    }
                } else {
                    CopyOutBytes(outputLocal[row * RoundUp<uint8_t>(tilingData_->kvCacheCol)],
                                 cacheGm_[cacheOffset], tilingData_->kvCacheCol);
                }
            }
            outputQueue_.FreeTensor(outputLocal);
            if constexpr (isMxFp4) {
                scaleQueue_.FreeTensor(scaleLocal);
            }
        }
    }

private:
    TPipe *pipe_ = nullptr;
    const KvCompressEpilogV2TilingData *tilingData_ = nullptr;
    GlobalTensor<TX> xGm_;
    GlobalTensor<TSlot> slotGm_;
    GlobalTensor<uint8_t> cacheGm_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TQue<QuePosition::VECOUT, 1> scaleQueue_;
    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> halfScaleBuffer_;
    TBuf<QuePosition::VECCALC> paddingBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    LocalTensor<TSlot> indexLocal_;
    LocalTensor<uint8_t> paddingLocal_;
};

}  // namespace KvCompressEpilogV2Ops

#endif
