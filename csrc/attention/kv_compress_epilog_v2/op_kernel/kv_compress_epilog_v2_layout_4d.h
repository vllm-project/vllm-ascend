/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 4d 布局（TilingKey 2100-2102）：cache 为分页块结构，token 按 tokenStride 排布（data+scale 连续）。
// slot 拆 (block,pos) 寻址；对齐时整 token 单拷贝，未对齐时 data/scale 两次拷贝。

#ifndef KV_COMPRESS_EPILOG_V2_LAYOUT_4D_H
#define KV_COMPRESS_EPILOG_V2_LAYOUT_4D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kv_compress_epilog_v2_common.h"
#include "kv_compress_epilog_v2_quant_fp8.h"
#include "kv_compress_epilog_v2_quant_fp4_g32.h"
#include "kv_compress_epilog_v2_quant_fp4_g16.h"

namespace KvCompressEpilogV2Ops {
using namespace AscendC;

// 4d 搬运骨架：InitCommon 分配队列/缓存，LoadRows 过滤 slot 并聚拢搬入，CopyOutRows 分页落盘。
template <typename TX, typename TSlot>
class KvCompressEpilogV2Layout4dCommon {
public:
    __aicore__ inline explicit KvCompressEpilogV2Layout4dCommon(TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void InitCommon(GM_ADDR cache, GM_ADDR x, GM_ADDR slotMapping,
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
        pipe_->InitBuffer(indexBuffer_, RoundUp<TSlot>(tilingData_->rowFactor) * sizeof(TSlot));
        indexLocal_ = indexBuffer_.Get<TSlot>();
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
        const int64_t scaleBytes = tilingData_->scaleBytes;
        for (int64_t row = 0; row < validRows; ++row) {
            const int64_t slot = static_cast<int64_t>(indexLocal_.GetValue(row));
            const int64_t block = slot / tilingData_->blockSize;
            const int64_t pos = slot % tilingData_->blockSize;
            const int64_t blockBase = block * tilingData_->blockStride;
            const int64_t outBase = row * RoundUp<uint8_t>(tilingData_->kvCacheCol);
            const int64_t tokenBase = blockBase + pos * tilingData_->tokenStride;
            const int64_t outputScaleOffset = RoundUp<uint8_t>(tilingData_->dataCol);
            if (outputScaleOffset == tilingData_->dataCol) {
                CopyOutBytes(output[outBase], cacheGm_[tokenBase],
                             static_cast<uint32_t>(tilingData_->tokenStride));
                continue;
            }
            CopyOutBytes(output[outBase], cacheGm_[tokenBase],
                         static_cast<uint32_t>(tilingData_->dataCol));
            CopyOutBytes(output[outBase + outputScaleOffset], cacheGm_[tokenBase + tilingData_->dataCol],
                         static_cast<uint32_t>(scaleBytes));
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
    TBuf<QuePosition::VECCALC> indexBuffer_;
    LocalTensor<TSlot> indexLocal_;
};

// TilingKey 2100：FP8-g32，rs=1 走 FullRow 流水，rs=0 走数值基线；输出行按 4d 分页寻址。
template <typename TX, typename TSlot, typename TCache>
class KvCompressEpilogV2Layout4dFp8Kernel : public KvCompressEpilogV2Layout4dCommon<TX, TSlot> {
    using Base = KvCompressEpilogV2Layout4dCommon<TX, TSlot>;
public:
    __aicore__ inline explicit KvCompressEpilogV2Layout4dFp8Kernel(TPipe *pipe) : Base(pipe) {}
    __aicore__ inline void Init(GM_ADDR cache, GM_ADDR x, GM_ADDR slotMapping,
                                const KvCompressEpilogV2TilingData *tilingData)
    {
        const uint32_t groups = static_cast<uint32_t>(tilingData->d / tilingData->perGroupSize);
        Base::InitCommon(cache, x, slotMapping, tilingData,
                         tilingData->rowFactor * RoundUp<float>(groups) * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        const uint32_t outputDataCol = RoundUp<uint8_t>(Base::tilingData_->dataCol);
        const uint32_t outputConcatCol = outputDataCol + Base::tilingData_->scaleCol * sizeof(uint16_t);
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
                VFProcessMxFp8FullRowPerf<TCache>(output, xLocal, Base::scratchBuffer_.template Get<float>(),
                    static_cast<uint16_t>(validRows), Base::tilingData_->d, outputDataCol,
                    outputConcatCol, Base::tilingData_->kvCacheCol);
            } else {
                VFProcessMxFp8<TCache, false>(output, xLocal, static_cast<uint16_t>(validRows),
                    Base::tilingData_->d, outputDataCol,
                    outputConcatCol, Base::tilingData_->kvCacheCol);
            }
            Base::xQueue_.FreeTensor(xLocal);
            Base::outputQueue_.template EnQue<uint8_t>(output);
            output = Base::outputQueue_.template DeQue<uint8_t>();
            Base::CopyOutRows(output, validRows);
            Base::outputQueue_.FreeTensor(output);
        }
    }
};

// TilingKey 2101：FP4-g32（非主场景，可能日落）。
template <typename TX, typename TSlot>
class KvCompressEpilogV2Layout4dFp4G32Kernel : public KvCompressEpilogV2Layout4dCommon<TX, TSlot> {
    using Base = KvCompressEpilogV2Layout4dCommon<TX, TSlot>;
public:
    __aicore__ inline explicit KvCompressEpilogV2Layout4dFp4G32Kernel(TPipe *pipe) : Base(pipe) {}
    __aicore__ inline void Init(GM_ADDR cache, GM_ADDR x, GM_ADDR slotMapping,
                                const KvCompressEpilogV2TilingData *tilingData)
    {
        const uint32_t groups = static_cast<uint32_t>(tilingData->d / tilingData->perGroupSize);
        Base::InitCommon(cache, x, slotMapping, tilingData,
                         tilingData->rowFactor * RoundUp<uint16_t>(groups) * sizeof(uint16_t));
    }
    __aicore__ inline void Process()
    {
        const uint32_t outputDataCol = RoundUp<uint8_t>(Base::tilingData_->dataCol);
        const uint32_t outputConcatCol = outputDataCol + Base::tilingData_->scaleCol * sizeof(uint16_t);
        const int64_t loops = Base::LoopCount();
        for (int64_t loop = 0; loop < loops; ++loop) {
            int64_t validRows = 0;
            LocalTensor<TX> xLocal = Base::LoadRows(loop, Base::RowsInLoop(loop, loops), validRows);
            if (validRows == 0) {
                Base::xQueue_.FreeTensor(xLocal);
                continue;
            }
            LocalTensor<uint8_t> output = Base::outputQueue_.template AllocTensor<uint8_t>();
            VFProcessMxFp4RowBatchPerf<bfloat16_t>(output, xLocal, Base::scratchBuffer_.template Get<uint16_t>(),
                static_cast<uint16_t>(validRows), Base::tilingData_->d, outputDataCol,
                outputConcatCol, Base::tilingData_->kvCacheCol);
            Base::xQueue_.FreeTensor(xLocal);
            Base::outputQueue_.template EnQue<uint8_t>(output);
            output = Base::outputQueue_.template DeQue<uint8_t>();
            Base::CopyOutRows(output, validRows);
            Base::outputQueue_.FreeTensor(output);
        }
    }
};

// TilingKey 2102：FP4-g16（主场景）。
template <typename TX, typename TSlot>
class KvCompressEpilogV2Layout4dFp4G16Kernel : public KvCompressEpilogV2Layout4dCommon<TX, TSlot> {
    using Base = KvCompressEpilogV2Layout4dCommon<TX, TSlot>;
public:
    __aicore__ inline explicit KvCompressEpilogV2Layout4dFp4G16Kernel(TPipe *pipe) : Base(pipe) {}
    __aicore__ inline void Init(GM_ADDR cache, GM_ADDR x, GM_ADDR slotMapping,
                                const KvCompressEpilogV2TilingData *tilingData)
    {
        const uint32_t groups = static_cast<uint32_t>(tilingData->d / tilingData->perGroupSize);
        Base::InitCommon(cache, x, slotMapping, tilingData,
                         tilingData->rowFactor * RoundUp<uint16_t>(groups) * sizeof(uint16_t));
    }
    __aicore__ inline void Process()
    {
        const uint32_t outputDataCol = RoundUp<uint8_t>(Base::tilingData_->dataCol);
        const uint32_t outputConcatCol = outputDataCol + Base::tilingData_->scaleCol * sizeof(uint16_t);
        const int64_t loops = Base::LoopCount();
        for (int64_t loop = 0; loop < loops; ++loop) {
            int64_t validRows = 0;
            LocalTensor<TX> xLocal = Base::LoadRows(loop, Base::RowsInLoop(loop, loops), validRows);
            if (validRows == 0) {
                Base::xQueue_.FreeTensor(xLocal);
                continue;
            }
            LocalTensor<uint8_t> output = Base::outputQueue_.template AllocTensor<uint8_t>();
            VFProcessMxFp4Group16RowBatchPerf(output, xLocal, Base::scratchBuffer_.template Get<uint16_t>(),
                static_cast<uint16_t>(validRows), Base::tilingData_->d, outputDataCol,
                outputConcatCol, Base::tilingData_->kvCacheCol);
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
