/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_ATTENTION_SCORE_FD_COMBINE_ARCH22_H
#define SPARSE_ATTENTION_SCORE_FD_COMBINE_ARCH22_H

#include <limits>
#include "kernel_operator.h"
#include "adv_api/pad/broadcast.h"
#include "adv_api/reduce/reduce.h"

namespace SasaKernelArch22 {

template <class ElementO, class Resource>
class SparseAttentionScoreFdCombineArch22 {
public:
    static constexpr uint32_t MAX_SPLIT_NUM = 16;
    static constexpr uint32_t MAX_ROW_TILE = 8;
    static constexpr uint32_t FLOATS_PER_BLOCK = 8;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t MAX_LSE_ELEMS = MAX_SPLIT_NUM * MAX_ROW_TILE;
    static constexpr uint32_t MAX_O_TILE_ELEMS = MAX_ROW_TILE * HEAD_DIM;

    __aicore__ inline
    SparseAttentionScoreFdCombineArch22(Resource &resource)
    {
        constexpr uint32_t LSE_OFFSET = 0;
        constexpr uint32_t LSE_BROADCAST_OFFSET = LSE_OFFSET + MAX_LSE_ELEMS * sizeof(float);
        constexpr uint32_t WEIGHT_OFFSET = LSE_BROADCAST_OFFSET + MAX_LSE_ELEMS * sizeof(float);
        constexpr uint32_t LSE_MAX_OFFSET = WEIGHT_OFFSET + MAX_LSE_ELEMS * sizeof(float);
        constexpr uint32_t LSE_SUM_OFFSET = LSE_MAX_OFFSET + MAX_ROW_TILE * sizeof(float);
        constexpr uint32_t GLOBAL_LSE_OFFSET = LSE_SUM_OFFSET + MAX_ROW_TILE * sizeof(float);
        constexpr uint32_t O_TMP_OFFSET = GLOBAL_LSE_OFFSET + MAX_ROW_TILE * sizeof(float);
        constexpr uint32_t O_ACC_OFFSET = O_TMP_OFFSET + MAX_O_TILE_ELEMS * sizeof(float);
        constexpr uint32_t REDUCE_TMP_OFFSET = O_ACC_OFFSET + MAX_O_TILE_ELEMS * sizeof(float);

        lseUb_ = resource.ubBuf.template GetBufferByByte<float>(LSE_OFFSET);
        broadcastUb_ = resource.ubBuf.template GetBufferByByte<float>(LSE_BROADCAST_OFFSET);
        weightUb_ = resource.ubBuf.template GetBufferByByte<float>(WEIGHT_OFFSET);
        lseMaxUb_ = resource.ubBuf.template GetBufferByByte<float>(LSE_MAX_OFFSET);
        lseSumUb_ = resource.ubBuf.template GetBufferByByte<float>(LSE_SUM_OFFSET);
        globalLseUb_ = resource.ubBuf.template GetBufferByByte<float>(GLOBAL_LSE_OFFSET);
        oTmpUb_ = resource.ubBuf.template GetBufferByByte<float>(O_TMP_OFFSET);
        oAccUb_ = resource.ubBuf.template GetBufferByByte<float>(O_ACC_OFFSET);
        // The partial O tile is dead after accumulation, so its storage can be
        // reused for the cast output without overlapping any live data.
        oOutUb_ = resource.ubBuf.template GetBufferByByte<ElementO>(O_TMP_OFFSET);
        reduceTmpUb_ = resource.ubBuf.template GetBufferByByte<uint8_t>(REDUCE_TMP_OFFSET);
    }

    __aicore__ inline
    void operator()(__gm__ SparseAttn::SparseAttentionScoreTilingData *tilingData,
                    AscendC::GlobalTensor<float> &gPartialLse,
                    AscendC::GlobalTensor<float> &gPartialO,
                    AscendC::GlobalTensor<ElementO> &gO)
    {
        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));

        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t combineTaskNum = tilingData->fdCombineTaskNum;
        const uint32_t groupSize = tilingData->groupSize;
        if (subBlockNum == 0 || blockNum == 0 || combineTaskNum == 0 || groupSize == 0) {
            return;
        }

        // On C220, GetBlockIdx() is the global AIV lane id in the vector
        // branch, while GetBlockNum() is the launched AIC count.
        const uint32_t aivIdx = AscendC::GetBlockIdx();
        const uint32_t aivNum = blockNum * subBlockNum;
        // Keep source row ownership identical to the Arch22 partial writers.
        // For the single-row case, source sub-block 0 is empty and source
        // sub-block 1 owns the only row.
        const uint32_t rowSplit = groupSize == 1 ? 0 : groupSize / subBlockNum;
        const uint32_t sub0Rows = rowSplit;
        const uint32_t sub1Rows = groupSize - rowSplit;
        const uint32_t totalRows = combineTaskNum * groupSize;
        const uint32_t rowTile = Min(MAX_ROW_TILE, Max(1U, CeilDiv(totalRows, aivNum)));
        const uint32_t sub0Tiles = CeilDiv(sub0Rows, rowTile);
        const uint32_t sub1Tiles = CeilDiv(sub1Rows, rowTile);
        const uint32_t tilesPerTask = sub0Tiles + sub1Tiles;
        const uint32_t workCount = combineTaskNum * tilesPerTask;
        const uint32_t lseSubStride = tilingData->fdLseSubStride;
        const uint32_t lsePartialStride = 2 * lseSubStride;

        // Flatten (combine task, source sub-block, row tile) and distribute the
        // work globally across all AIVs. A tile never crosses the partial
        // writer's sub-block boundary, so its LSE rows remain contiguous in GM.
        for (uint32_t work = aivIdx; work < workCount; work += aivNum) {
            const uint32_t combineTask = work / tilesPerTask;
            const uint32_t tileInTask = work % tilesPerTask;
            const uint32_t sourceSubBlock = tileInTask < sub0Tiles ? 0U : 1U;
            const uint32_t tileInSubBlock = sourceSubBlock == 0 ? tileInTask : tileInTask - sub0Tiles;
            const uint32_t sourceRows = sourceSubBlock == 0 ? sub0Rows : sub1Rows;
            const uint32_t sourceLocalRow = tileInSubBlock * rowTile;
            const uint32_t rowCount = Min(rowTile, sourceRows - sourceLocalRow);
            const uint32_t groupRowStart = (sourceSubBlock == 0 ? 0U : rowSplit) + sourceLocalRow;
            const uint32_t rowCountAlign = RoundUp(rowCount, FLOATS_PER_BLOCK);

            const uint32_t baseTask = tilingData->fdCombineBaseTask[combineTask];
            const uint32_t splitCount = tilingData->fdPartialCountByBase[baseTask];
            const uint32_t splitCountAlign = RoundUp(splitCount, FLOATS_PER_BLOCK);
            const uint32_t firstPartialTask = tilingData->fdPartialStartByBase[baseTask];

            PrepareWeights(gPartialLse, firstPartialTask, splitCount, splitCountAlign,
                sourceSubBlock, sourceLocalRow, rowCount, rowCountAlign, lseSubStride, lsePartialStride);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

            const uint32_t qToken = baseTask / tilingData->kvHeads;
            const uint32_t kvHead = baseTask % tilingData->kvHeads;
            const uint64_t baseOutputOffset =
                (static_cast<uint64_t>(qToken) * tilingData->numHeads + kvHead * groupSize + groupRowStart) *
                HEAD_DIM;
            const uint32_t tileElems = rowCount * HEAD_DIM;
            AscendC::Duplicate(oAccUb_, 0.0f, tileElems);
            AscendC::PipeBarrier<PIPE_V>();

            for (uint32_t split = 0; split < splitCount; ++split) {
                const uint32_t partialTask = firstPartialTask + split;
                const uint64_t partialOOffset =
                    (static_cast<uint64_t>(partialTask) * groupSize + groupRowStart) * HEAD_DIM;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::DataCopyPad(oTmpUb_, gPartialO[partialOOffset],
                    AscendC::DataCopyExtParams(rowCount, HEAD_DIM * sizeof(float), 0, 0, 0),
                    AscendC::DataCopyPadExtParams<float>(false, 0, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

                for (uint32_t row = 0; row < rowCount; ++row) {
                    const float weight = weightUb_.GetValue(split * rowCountAlign + row);
                    AscendC::Muls(oTmpUb_[row * HEAD_DIM], oTmpUb_[row * HEAD_DIM], weight, HEAD_DIM);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(oAccUb_, oAccUb_, oTmpUb_, tileElems);
                AscendC::PipeBarrier<PIPE_V>();
            }

            if (std::is_same<ElementO, bfloat16_t>::value) {
                AscendC::Cast(oOutUb_, oAccUb_, AscendC::RoundMode::CAST_RINT, tileElems);
            } else {
                AscendC::Cast(oOutUb_, oAccUb_, AscendC::RoundMode::CAST_NONE, tileElems);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(gO[baseOutputOffset], oOutUb_,
                AscendC::DataCopyExtParams(rowCount, HEAD_DIM * sizeof(ElementO), 0, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline
    void PrepareWeights(AscendC::GlobalTensor<float> &gPartialLse,
                        uint32_t firstPartialTask,
                        uint32_t splitCount,
                        uint32_t splitCountAlign,
                        uint32_t sourceSubBlock,
                        uint32_t sourceLocalRow,
                        uint32_t rowCount,
                        uint32_t rowCountAlign,
                        uint32_t lseSubStride,
                        uint32_t lsePartialStride)
    {
        const uint32_t calcElems = splitCountAlign * rowCountAlign;
        AscendC::Duplicate(lseUb_, std::numeric_limits<float>::lowest(), calcElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        const uint64_t lseOffset = static_cast<uint64_t>(firstPartialTask) * lsePartialStride +
            sourceSubBlock * lseSubStride + sourceLocalRow;
        AscendC::DataCopyPad(lseUb_, gPartialLse[lseOffset],
            AscendC::DataCopyExtParams(splitCount, rowCount * sizeof(float),
                (lsePartialStride - rowCount) * sizeof(float), 0, 0),
            AscendC::DataCopyPadExtParams<float>(true, 0, rowCountAlign - rowCount,
                std::numeric_limits<float>::lowest()));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        uint32_t reduceShape[] = {splitCountAlign, rowCountAlign};
        AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, false>(
            lseMaxUb_, lseUb_, reduceTmpUb_, reduceShape, true);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastRows(lseMaxUb_, rowCountAlign, splitCountAlign);
        AscendC::Sub(weightUb_, lseUb_, broadcastUb_, calcElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(weightUb_, weightUb_, calcElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, false>(
            lseSumUb_, weightUb_, reduceTmpUb_, reduceShape, true);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Ln(globalLseUb_, lseSumUb_, rowCountAlign);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(globalLseUb_, globalLseUb_, lseMaxUb_, rowCountAlign);
        AscendC::PipeBarrier<PIPE_V>();
        BroadcastRows(globalLseUb_, rowCountAlign, splitCountAlign);
        AscendC::Sub(weightUb_, lseUb_, broadcastUb_, calcElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(weightUb_, weightUb_, calcElems);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline
    void BroadcastRows(const AscendC::LocalTensor<float> &src,
                       uint32_t rowCountAlign,
                       uint32_t splitCountAlign)
    {
        uint32_t dstShape[] = {splitCountAlign, rowCountAlign};
        uint32_t srcShape[] = {1, rowCountAlign};
        AscendC::BroadCast<float, 2, 0>(broadcastUb_, src, dstShape, srcShape, reduceTmpUb_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline static uint32_t RoundUp(uint32_t value, uint32_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    __aicore__ inline static uint32_t CeilDiv(uint32_t value, uint32_t divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    __aicore__ inline static uint32_t Min(uint32_t lhs, uint32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline static uint32_t Max(uint32_t lhs, uint32_t rhs)
    {
        return lhs > rhs ? lhs : rhs;
    }

    AscendC::LocalTensor<float> lseUb_;
    AscendC::LocalTensor<float> broadcastUb_;
    AscendC::LocalTensor<float> weightUb_;
    AscendC::LocalTensor<float> lseMaxUb_;
    AscendC::LocalTensor<float> lseSumUb_;
    AscendC::LocalTensor<float> globalLseUb_;
    AscendC::LocalTensor<float> oTmpUb_;
    AscendC::LocalTensor<float> oAccUb_;
    AscendC::LocalTensor<ElementO> oOutUb_;
    AscendC::LocalTensor<uint8_t> reduceTmpUb_;
};

}  // namespace SasaKernelArch22

#endif  // SPARSE_ATTENTION_SCORE_FD_COMBINE_ARCH22_H
