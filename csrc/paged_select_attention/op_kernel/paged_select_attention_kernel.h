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
 * \file paged_select_attention_kernel.h
 * \brief
 */
#ifndef FLASH_ATTENTION_REGULAR_H
#define FLASH_ATTENTION_REGULAR_H

#include "kernel_common.hpp"

using namespace NpuArch;
using namespace KernelCommon;

namespace SplitFuse {
    template <
        class BlockMmadQK,
        class BlockMmadPV,
        class EpilogueOnlineSoftmax,
        class EpilogueRescaleO,
        class EpilogueInitOut,
        bool PAGED_CACHE_FLAG,
        FaiKernel::MaskType MASK_TYPE = FaiKernel::MaskType::NO_MASK,
        FaiKernel::inputLayout INPUT_LAYOUT = FaiKernel::inputLayout::BSND>
    class FAInferKernel {
    public:
        using ArchTag = typename BlockMmadQK::ArchTag;
        using L1TileShape = typename BlockMmadQK::L1TileShape;
        using ElementQ = typename BlockMmadQK::ElementA;
        using LayoutQ = typename BlockMmadQK::LayoutA;
        using ElementK = typename BlockMmadQK::ElementB;
        using LayoutK = typename BlockMmadQK::LayoutB;
        using ElementS = typename BlockMmadQK::ElementC;
        using LayoutS = typename BlockMmadQK::LayoutC;

        using ElementP = typename BlockMmadPV::ElementA;
        using LayoutP = typename BlockMmadPV::LayoutA;
        using ElementV = typename BlockMmadPV::ElementB;
        using LayoutV = typename BlockMmadPV::LayoutB;

        using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
        using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;

        using ElementO = typename EpilogueRescaleO::ElementOutput;
        using LayoutO = typename EpilogueRescaleO::LayoutOutput;

        using ElementOTmp = typename EpilogueRescaleO::ElementInput;
        using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

        using ElementLse = typename EpilogueRescaleO::ElementLse;
        using LayoutLse = typename EpilogueRescaleO::LayoutLse;

        using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
        using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

        static constexpr Epilogue::LseMode LSE_MODE = EpilogueRescaleO::LSE_MODE;
        static constexpr Epilogue::SinkMode SINK_MODE = EpilogueOnlineSoftmax::SINK_MODE;

        // Selected-kv sparse paged decode is the in-place path being kept. Dense
        // execution still exists elsewhere, but selected_kv_indices now targets this
        // narrowed decode-only contract only.
        struct SelectedKvSparseRuntimeState {
            bool hasSelectedKvIndices = false;
            bool sparseLaunchEnabled = false;
            uint32_t sparseKMax = 0;
            uint32_t pagedBlockSize = 0;
        };

        struct SparseBatchDecodeBounds {
            uint32_t effectiveBlockCount = 0;
            uint32_t validLogicalBlockCount = 0;
            uint32_t tailLogicalBlock = 0;
            uint32_t tailToken = 0;
        };

        struct SparseHeadDecodeState {
            uint64_t selectedRowBase = 0;
            uint32_t selectedBlockCount = 0;
            uint32_t validLogicalBlockCount = 0;
            bool useSparseDecode = false;
            uint32_t taskMaskType = 0;
            int64_t noSkipKvS = 0;
        };

        // Methods
        __aicore__ inline
        FAInferKernel() {}

        __aicore__ inline
        static void InitSelectedKvSparseRuntimeState(
            const FAIKernelParams &params,
            uint32_t sparseLaunchEnabled,
            uint32_t sparseKMax,
            uint32_t pagedBlockSize,
            SelectedKvSparseRuntimeState &selectedKvSparseState)
        {
            selectedKvSparseState.hasSelectedKvIndices = (params.selectedKvIndices != nullptr);
            selectedKvSparseState.sparseLaunchEnabled = (sparseLaunchEnabled != 0U);
            selectedKvSparseState.sparseKMax = sparseKMax;
            selectedKvSparseState.pagedBlockSize = pagedBlockSize;
        }

        __aicore__ inline
        static bool ShouldSerialiseSelectedKvSparseHeads(
            const SelectedKvSparseRuntimeState &selectedKvSparseState,
            uint32_t qSeqlen,
            uint32_t qSBlockSize)
        {
            if (selectedKvSparseState.sparseLaunchEnabled &&
                selectedKvSparseState.hasSelectedKvIndices &&
                (selectedKvSparseState.sparseKMax > 0U)) {
                FIA_HARD_FAIL_IF((qSeqlen != 1U) || (qSBlockSize != 1U),
                    "selected_kv_indices sparse contract requires pure decode task shape qSeqlen=%u qSBlockSize=%u",
                    qSeqlen, qSBlockSize);
                return true;
            }
            return false;
        }

        __aicore__ inline
        static void InitSparseBatchDecodeBounds(
            uint32_t kvSeqlen,
            uint32_t sparseKMax,
            uint32_t pagedBlockSize,
            SparseBatchDecodeBounds &sparseBatchBounds)
        {
            FIA_HARD_FAIL_IF((sparseKMax == 0U) || (pagedBlockSize == 0U),
                "invalid selected_kv_indices sparse contract sparseKMax=%u pagedBlockSize=%u",
                sparseKMax, pagedBlockSize);
            sparseBatchBounds.validLogicalBlockCount =
                (kvSeqlen == 0U) ? 0U : (kvSeqlen + pagedBlockSize - 1U) / pagedBlockSize;
            sparseBatchBounds.effectiveBlockCount =
                AscendC::Std::min(sparseKMax, sparseBatchBounds.validLogicalBlockCount);
            sparseBatchBounds.tailLogicalBlock = 0U;
            sparseBatchBounds.tailToken = 0U;
            if (sparseBatchBounds.validLogicalBlockCount == 0U) {
                return;
            }
            sparseBatchBounds.tailLogicalBlock = sparseBatchBounds.validLogicalBlockCount - 1U;
            sparseBatchBounds.tailToken = kvSeqlen % pagedBlockSize;
            if (sparseBatchBounds.tailToken == 0U) {
                sparseBatchBounds.tailToken = pagedBlockSize;
            }
        }

        __aicore__ inline
        static uint32_t CountSparseSelectedBlocks(
            AscendC::GlobalTensor<int32_t> &gSelectedKvIndices,
            uint64_t selectedRowBase,
            uint32_t selectedBlockLimit,
            uint32_t validLogicalBlockCount,
            uint32_t tailLogicalBlock,
            uint32_t tailToken,
            uint32_t pagedBlockSize,
            int64_t &sparseKvS)
        {
            sparseKvS = 0;
            // Step-3 narrowed contract: the third dimension of selected_kv_indices is
            // the full active row width. The kernel no longer discovers a shorter row
            // via negative sentinels in the hot path.
            for (uint32_t i = 0; i < selectedBlockLimit; ++i) {
                int32_t blockId = gSelectedKvIndices.GetValue(selectedRowBase + i);
                FIA_HARD_FAIL_IF((blockId < 0) ||
                    (static_cast<uint32_t>(blockId) >= validLogicalBlockCount),
                    "selected_kv_indices sparse contract violation: logical page id rowBase=%llu idx=%u value=%d validBlocks=%u",
                    static_cast<unsigned long long>(selectedRowBase), i, blockId, validLogicalBlockCount);
                sparseKvS += (static_cast<uint32_t>(blockId) == tailLogicalBlock) ? tailToken : pagedBlockSize;
            }
            return selectedBlockLimit;
        }

        __aicore__ inline
        static void FillSparseWindowPhysicalIds(
            AscendC::GlobalTensor<int32_t> &gSelectedKvIndices,
            AscendC::GlobalTensor<int32_t> &gBlockTable,
            uint64_t selectedRowBase,
            uint64_t blockBOffset,
            uint32_t selectedBlockLimit,
            uint32_t validLogicalBlockCount,
            uint32_t sparseBlockBase,
            uint32_t *windowPhysicalIds,
            uint32_t &windowPhysicalCount)
        {
            windowPhysicalCount = 0;
            uint32_t skipCount = sparseBlockBase;
            // selected_kv_indices provides the full logical-page row; block_table
            // performs the logical-to-physical remap for the narrowed sparse path.
            for (uint32_t i = 0; i < selectedBlockLimit; ++i) {
                int32_t blockId = gSelectedKvIndices.GetValue(selectedRowBase + i);
                FIA_HARD_FAIL_IF((blockId < 0) ||
                    (static_cast<uint32_t>(blockId) >= validLogicalBlockCount),
                    "selected_kv_indices sparse contract violation: logical page id rowBase=%llu idx=%u value=%d validBlocks=%u",
                    static_cast<unsigned long long>(selectedRowBase), i, blockId, validLogicalBlockCount);
                if (skipCount > 0U) {
                    --skipCount;
                    continue;
                }
                if (windowPhysicalCount >= KernelCommon::MAX_SELECTED_KV_STACK_BLOCKS) {
                    break;
                }
                int32_t physicalBlockIdRaw = gBlockTable.GetValue(blockBOffset + static_cast<uint32_t>(blockId));
                windowPhysicalIds[windowPhysicalCount++] = static_cast<uint32_t>(physicalBlockIdRaw);
            }
        }

        __aicore__ inline
        static void PrepareSelectedKvSparseHeadState(
            bool serialSparseHeads,
            const SelectedKvSparseRuntimeState &selectedKvSparseState,
            AscendC::GlobalTensor<int32_t> &gSelectedKvIndices,
            uint32_t curBatch,
            uint32_t qHeads,
            uint32_t headQNStartIdx,
            uint32_t kvSeqlen,
            const SparseBatchDecodeBounds &sparseBatchBounds,
            uint32_t qSeqlen,
            uint32_t qSBlockSize,
            uint32_t headQNBlockSize,
            SparseHeadDecodeState &sparseHeadState)
        {
            if (!serialSparseHeads) {
                return;
            }
            UpdateSparseHeadDecodeState(
                gSelectedKvIndices,
                curBatch,
                qHeads,
                headQNStartIdx,
                kvSeqlen,
                sparseBatchBounds,
                qSeqlen,
                qSBlockSize,
                headQNBlockSize,
                selectedKvSparseState.sparseKMax,
                selectedKvSparseState.pagedBlockSize,
                sparseHeadState);
        }

        __aicore__ inline
        static int64_t ResolveHeadTaskKvSpan(
            uint32_t taskMaskType,
            int64_t noSkipKvS,
            uint32_t kvSeqlen,
            uint32_t qSeqlen,
            uint32_t qSBlockIdx,
            uint32_t curQSBlockTile)
        {
            if (taskMaskType == 0U) {
                return noSkipKvS;
            }
            int64_t diffS = kvSeqlen - qSeqlen;
            diffS = (diffS < 0) ? 0 : diffS;
            noSkipKvS = (qSBlockIdx + 1U) * curQSBlockTile + diffS;
            return AscendC::Std::min(static_cast<int64_t>(kvSeqlen), noSkipKvS);
        }

        __aicore__ inline
        static void InitSparseHeadDecodeState(
            uint32_t kvSeqlen,
            uint32_t maskType,
            SparseHeadDecodeState &sparseHeadState)
        {
            sparseHeadState.selectedRowBase = 0;
            sparseHeadState.selectedBlockCount = 0;
            sparseHeadState.validLogicalBlockCount = 0;
            sparseHeadState.useSparseDecode = false;
            sparseHeadState.taskMaskType = maskType;
            sparseHeadState.noSkipKvS = static_cast<int64_t>(kvSeqlen);
        }

        __aicore__ inline
        static uint64_t GetSparseSelectedRowBase(
            uint32_t curBatch,
            uint32_t qHeads,
            uint32_t headQNStartIdx,
            uint32_t sparseKMax)
        {
            return (static_cast<uint64_t>(curBatch) * qHeads + headQNStartIdx) * sparseKMax;
        }

        __aicore__ inline
        static void UpdateSparseHeadDecodeState(
            AscendC::GlobalTensor<int32_t> &gSelectedKvIndices,
            uint32_t curBatch,
            uint32_t qHeads,
            uint32_t headQNStartIdx,
            uint32_t kvSeqlen,
            const SparseBatchDecodeBounds &sparseBatchBounds,
            uint32_t qSeqlen,
            uint32_t qSBlockSize,
            uint32_t headQNBlockSize,
            uint32_t sparseKMax,
            uint32_t pagedBlockSize,
            SparseHeadDecodeState &sparseHeadState)
        {
            FIA_HARD_FAIL_IF((sparseKMax == 0U) || (pagedBlockSize == 0U),
                "invalid selected_kv_indices sparse contract sparseKMax=%u pagedBlockSize=%u",
                sparseKMax, pagedBlockSize);
            sparseHeadState.validLogicalBlockCount = sparseBatchBounds.validLogicalBlockCount;
            sparseHeadState.selectedRowBase =
                GetSparseSelectedRowBase(curBatch, qHeads, headQNStartIdx, sparseKMax);
            int64_t sparseKvS = 0;
            sparseHeadState.selectedBlockCount = CountSparseSelectedBlocks(
                gSelectedKvIndices,
                sparseHeadState.selectedRowBase,
                sparseBatchBounds.effectiveBlockCount,
                sparseHeadState.validLogicalBlockCount,
                sparseBatchBounds.tailLogicalBlock,
                sparseBatchBounds.tailToken,
                pagedBlockSize,
                sparseKvS);
            FIA_HARD_FAIL_IF((qSeqlen != 1U) || (qSBlockSize != 1U) || (headQNBlockSize != 1U),
                "sparse decode task-shape invariant violated qSeqlen=%u qSBlockSize=%u qN=%u",
                qSeqlen, qSBlockSize, headQNBlockSize);
            sparseHeadState.useSparseDecode = true;
            sparseHeadState.noSkipKvS = AscendC::Std::min(static_cast<int64_t>(kvSeqlen), sparseKvS);
            // The narrowed sparse path treats selected rows as the complete visibility
            // set, so it bypasses causal masking here.
            sparseHeadState.taskMaskType = 0U;
        }

        __aicore__ inline
        static int32_t GetSparseFirstSelectedBlockForDebug(
            AscendC::GlobalTensor<int32_t> &gSelectedKvIndices,
            const SparseHeadDecodeState &sparseHeadState)
        {
            if (!sparseHeadState.useSparseDecode || (sparseHeadState.selectedBlockCount == 0U)) {
                return -1;
            }
            return gSelectedKvIndices.GetValue(sparseHeadState.selectedRowBase);
        }

        __aicore__ inline
        static void PrepareSparseWindowPhysicalIds(
            AscendC::GlobalTensor<int32_t> &gSelectedKvIndices,
            AscendC::GlobalTensor<int32_t> &gBlockTable,
            uint64_t blockBOffset,
            uint32_t pagedBlockSize,
            uint32_t kvSIdx,
            const SparseHeadDecodeState &sparseHeadState,
            uint32_t *windowPhysicalIds,
            uint32_t &windowPhysicalCount)
        {
            uint32_t sparseBlockBase = (kvSIdx * MAX_KV_STACK_LEN) / pagedBlockSize;
            FillSparseWindowPhysicalIds(
                gSelectedKvIndices,
                gBlockTable,
                sparseHeadState.selectedRowBase,
                blockBOffset,
                sparseHeadState.selectedBlockCount,
                sparseHeadState.validLogicalBlockCount,
                sparseBlockBase,
                windowPhysicalIds,
                windowPhysicalCount);
        }

        __aicore__ inline
        static void PrepareSelectedKvSparseWindowPhysicalIds(
            const SelectedKvSparseRuntimeState &selectedKvSparseState,
            AscendC::GlobalTensor<int32_t> &gSelectedKvIndices,
            AscendC::GlobalTensor<int32_t> &gBlockTable,
            uint64_t blockBOffset,
            uint32_t kvSIdx,
            const SparseHeadDecodeState &sparseHeadState,
            uint32_t *windowPhysicalIds,
            uint32_t &windowPhysicalCount)
        {
            PrepareSparseWindowPhysicalIds(
                gSelectedKvIndices,
                gBlockTable,
                blockBOffset,
                selectedKvSparseState.pagedBlockSize,
                kvSIdx,
                sparseHeadState,
                windowPhysicalIds,
                windowPhysicalCount);
        }

        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            __gm__ PagedSelectAttentionTilingData *fATilingData =
                reinterpret_cast<__gm__ PagedSelectAttentionTilingData *>(params.tiling);
            uint64_t mm1OutSize = fATilingData->mm1OutSize;
            uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize;
            uint64_t mm2OutSize = fATilingData->mm2OutSize;
            uint32_t batch = fATilingData->batch;
            uint32_t qHeads = fATilingData->numHeads;
            uint32_t kvHeads = fATilingData->kvHeads;
            uint32_t embed = fATilingData->embeddingSize;
            uint32_t embedV = fATilingData->embeddingSizeV;
            uint32_t numBlocks = fATilingData->numBlocks;
            uint32_t pagedBlockSize = fATilingData->blockSize;
            uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
            uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;
            uint32_t totalTaskNum = fATilingData->totalTaskNum;
            uint32_t blockSize = fATilingData->blockSize;
            uint32_t maskType = fATilingData->maskType;
            uint32_t sparseLaunchEnabled = fATilingData->sparseLaunchEnabled;
            uint32_t sparseKMax = fATilingData->sparseKMax;
            float scaleValue = fATilingData->scaleValue;
            (void)numBlocks;
            (void)sparseLaunchEnabled;
            (void)sparseKMax;
            FIA_HARD_FAIL_IF(batch == 0U, "batch is 0");
            FIA_HARD_FAIL_IF((qHeads == 0U) || (kvHeads == 0U),
                "invalid heads. qHeads=%u kvHeads=%u", qHeads, kvHeads);
            FIA_HARD_FAIL_IF((qHeads % kvHeads) != 0U,
                "qHeads(%u) must be divisible by kvHeads(%u)", qHeads, kvHeads);
            FIA_HARD_FAIL_IF(params.actualQseqlen == nullptr, "actualQseqlen is null");
            FIA_HARD_FAIL_IF(params.actualKvseqlen == nullptr, "actualKvseqlen is null");
            if constexpr (PAGED_CACHE_FLAG) {
                FIA_HARD_FAIL_IF(params.blockTables == nullptr, "blockTables is null in paged path");
                FIA_HARD_FAIL_IF((pagedBlockSize == 0U) || (maxNumBlocksPerBatch == 0U) || (numBlocks == 0U),
                    "invalid paged config blockSize=%u maxBlocksPerBatch=%u numBlocks=%u",
                    pagedBlockSize, maxNumBlocksPerBatch, numBlocks);
                FIA_HARD_FAIL_IF((sparseLaunchEnabled != 0U) && (params.selectedKvIndices == nullptr),
                    "sparseLaunchEnabled is set but selectedKvIndices is null");
            }

            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            __gm__ uint8_t* currentKey = (__gm__ uint8_t*)params.k;
            __gm__ uint8_t* currentValue = (__gm__ uint8_t*)params.v;
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)currentKey);
            AscendC::GlobalTensor<ElementK> gV;
            gV.SetGlobalBuffer((__gm__ ElementK *)currentValue);
            AscendC::GlobalTensor<ElementMask> gMask;
            gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
            AscendC::GlobalTensor<int32_t> gBlockTable;
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
            AscendC::GlobalTensor<int32_t> gSelectedKvIndices;
            if (params.selectedKvIndices != nullptr) {
                gSelectedKvIndices.SetGlobalBuffer((__gm__ int32_t *)(params.selectedKvIndices));
            }
            SelectedKvSparseRuntimeState selectedKvSparseState;
            InitSelectedKvSparseRuntimeState(params, sparseLaunchEnabled, sparseKMax, pagedBlockSize, selectedKvSparseState);
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            AscendC::GlobalTensor<ElementO> gO;
            gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
            AscendC::GlobalTensor<ElementLse> gLse;
            gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);
            AscendC::GlobalTensor<ElementS> gS;
            gS.SetGlobalBuffer((__gm__ ElementS *)(params.workSpace));
            AscendC::GlobalTensor<ElementP> gP;
            gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + mm1OutSize));
            AscendC::GlobalTensor<ElementOTmp> gOTmp;
            gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + mm1OutSize + smOnlineOutSize));
            AscendC::GlobalTensor<ElementOTmp> gOUpdate;
            gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace +
                mm1OutSize + smOnlineOutSize + mm2OutSize));
            AscendC::GlobalTensor<bfloat16_t> gSink;
            gSink.SetGlobalBuffer((__gm__ bfloat16_t *)(params.sink));

            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
#ifdef __DAV_C220_CUBE__
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

            uint32_t kDynNum = NpuArch::Detail::Alignment::RoundUp(embed, NUM_128);
            kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;
            uint32_t maxQKPL1Size = L1_MAX_SIZE - embedV * MAX_KV_STACK_LEN * sizeof(ElementV);
            uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);
            uint32_t maxNDynNum =
                ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementV) / DOUBLE_BUFFER) / NUM_32 * NUM_32;

            uint32_t nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ?
                NpuArch::Detail::Alignment::RoundDown((nDynNum - 1), NUM_32) : nDynNum;

            uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * kDynNum * sizeof(ElementQ);
            BlockMmadQK blockMmadQK(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN);
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            BlockMmadPV blockMmadPV(resource, nDynNum, kPVDynNum, MAX_KV_STACK_LEN, L1_QK_SIZE);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

            EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
            EpilogueRescaleO epilogueRescaleO(resource);
            EpilogueInitOut epilogueInitOut(resource);

            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
            uint64_t strideQ = static_cast<uint64_t>(qHeads * embed);
            uint64_t strideO = static_cast<uint64_t>(qHeads * embedV);
            uint64_t strideK = static_cast<uint64_t>(kvHeads * embed);
            uint64_t strideV = static_cast<uint64_t>(kvHeads * embedV);
            uint32_t embedRound = NpuArch::Detail::Alignment::RoundUp(embed, FaiKernel::BLOCK_SIZE);
            uint32_t embedRoundV = NpuArch::Detail::Alignment::RoundUp(embedV, FaiKernel::BLOCK_SIZE);
            uint32_t groupSize = qHeads / kvHeads;

            uint64_t qBOffset = 0;
            uint64_t kBOffset = 0;
            uint64_t vBOffset = 0;
            uint64_t oBOffset = 0;
            uint64_t lseBOffset = 0;
            uint64_t blockBOffset = 0;

            uint32_t preTotalTaskNum = 0;
            uint32_t curBatch = 0;
            constexpr int64_t U32_MAX_I64 = 0xFFFFFFFFLL;
            int64_t totalQTokensRaw = gActualQseqlen.GetValue(batch - 1);
            FIA_HARD_FAIL_IF((totalQTokensRaw < 0) || (totalQTokensRaw > U32_MAX_I64),
                "invalid totalQTokens=%ld", totalQTokensRaw);
            uint32_t totalQTokens = static_cast<uint32_t>(totalQTokensRaw);
            int64_t qSeqRaw = gActualQseqlen.GetValue(curBatch);
            int64_t kvSeqRaw = gActualKvseqlen.GetValue(curBatch);
            FIA_HARD_FAIL_IF((qSeqRaw < 0) || (qSeqRaw > U32_MAX_I64),
                "invalid q cumulative seqlen batch=%u value=%ld", curBatch, qSeqRaw);
            FIA_HARD_FAIL_IF((kvSeqRaw < 0) || (kvSeqRaw > U32_MAX_I64),
                "invalid kv seqlen batch=%u value=%ld", curBatch, kvSeqRaw);
            uint32_t qSeqlen = static_cast<uint32_t>(qSeqRaw);
            uint32_t kvSeqlen = static_cast<uint32_t>(kvSeqRaw);
            if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                int64_t prevQRaw = (curBatch == 0) ? 0 : gActualQseqlen.GetValue(curBatch - 1);
                FIA_HARD_FAIL_IF((prevQRaw < 0) || (prevQRaw > U32_MAX_I64),
                    "invalid prev q cumulative batch=%u value=%ld", curBatch, prevQRaw);
                FIA_HARD_FAIL_IF(qSeqRaw < prevQRaw,
                    "non-monotonic q cumulative seqlen batch=%u prev=%ld curr=%ld",
                    curBatch, prevQRaw, qSeqRaw);
                uint32_t prevQSeqlenSum = static_cast<uint32_t>(prevQRaw);
                qSeqlen = qSeqlen - prevQSeqlenSum;
                if constexpr (!PAGED_CACHE_FLAG) {
                    int64_t prevKvRaw = (curBatch == 0) ? 0 : gActualKvseqlen.GetValue(curBatch - 1);
                    FIA_HARD_FAIL_IF((prevKvRaw < 0) || (prevKvRaw > U32_MAX_I64),
                        "invalid prev kv cumulative batch=%u value=%ld", curBatch, prevKvRaw);
                    FIA_HARD_FAIL_IF(kvSeqRaw < prevKvRaw,
                        "non-monotonic kv cumulative seqlen batch=%u prev=%ld curr=%ld",
                        curBatch, prevKvRaw, kvSeqRaw);
                    uint32_t prevKvSeqlenSum = static_cast<uint32_t>(prevKvRaw);
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
            }
            if constexpr (PAGED_CACHE_FLAG) {
                uint64_t kvCapacity = static_cast<uint64_t>(maxNumBlocksPerBatch) * pagedBlockSize;
                FIA_HARD_FAIL_IF(static_cast<uint64_t>(kvSeqlen) > kvCapacity,
                    "kvSeqlen(%u) exceeds paged capacity(%llu) batch=%u",
                    kvSeqlen, static_cast<unsigned long long>(kvCapacity), curBatch);
            }
            SparseBatchDecodeBounds sparseBatchBounds;
            if constexpr (PAGED_CACHE_FLAG) {
                InitSparseBatchDecodeBounds(kvSeqlen, sparseKMax, pagedBlockSize, sparseBatchBounds);
            }
            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
            uint32_t qNBlockNumPerGroup = NpuArch::Detail::Alignment::CeilDiv(groupSize, curQNBlockTile);
            uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
            uint32_t curQSBlockNum = NpuArch::Detail::Alignment::CeilDiv(qSeqlen, curQSBlockTile);
            uint32_t curTotalTaskNum = firstBatchTaskNum;

            //  prepare for addding sink
            bool isLastStackTile = false;
            // Go through each task.
            for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
                isLastStackTile = false;
                // Get the offset of each core on the GM.
                while (taskIdx >= curTotalTaskNum) {
                    ++curBatch;
                    FIA_HARD_FAIL_IF(curBatch >= batch,
                        "task split overflow taskIdx=%u curTotalTaskNum=%u batch=%u",
                        taskIdx, curTotalTaskNum, batch);
                    preTotalTaskNum = curTotalTaskNum;
                    qBOffset += qSeqlen * strideQ;
                    if constexpr (!PAGED_CACHE_FLAG) {
                        kBOffset += static_cast<uint64_t>(kvSeqlen * strideK);
                        vBOffset += static_cast<uint64_t>(kvSeqlen * strideV);
                    } else {
                        blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch);
                    }
                    oBOffset += static_cast<uint64_t>(qSeqlen * strideO);
                    lseBOffset += static_cast<uint64_t>(qSeqlen * qHeads);

                    qSeqRaw = gActualQseqlen.GetValue(curBatch);
                    kvSeqRaw = gActualKvseqlen.GetValue(curBatch);
                    FIA_HARD_FAIL_IF((qSeqRaw < 0) || (qSeqRaw > U32_MAX_I64),
                        "invalid q cumulative seqlen batch=%u value=%ld", curBatch, qSeqRaw);
                    FIA_HARD_FAIL_IF((kvSeqRaw < 0) || (kvSeqRaw > U32_MAX_I64),
                        "invalid kv seqlen batch=%u value=%ld", curBatch, kvSeqRaw);
                    qSeqlen = static_cast<uint32_t>(qSeqRaw);
                    kvSeqlen = static_cast<uint32_t>(kvSeqRaw);
                    if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                        int64_t prevQRaw = (curBatch == 0) ? 0 : gActualQseqlen.GetValue(curBatch - 1);
                        FIA_HARD_FAIL_IF((prevQRaw < 0) || (prevQRaw > U32_MAX_I64),
                            "invalid prev q cumulative batch=%u value=%ld", curBatch, prevQRaw);
                        FIA_HARD_FAIL_IF(qSeqRaw < prevQRaw,
                            "non-monotonic q cumulative seqlen batch=%u prev=%ld curr=%ld",
                            curBatch, prevQRaw, qSeqRaw);
                        uint32_t prevQSeqlenSum = static_cast<uint32_t>(prevQRaw);
                        qSeqlen = qSeqlen - prevQSeqlenSum;
                        if constexpr (!PAGED_CACHE_FLAG) {
                            int64_t prevKvRaw = (curBatch == 0) ? 0 : gActualKvseqlen.GetValue(curBatch - 1);
                            FIA_HARD_FAIL_IF((prevKvRaw < 0) || (prevKvRaw > U32_MAX_I64),
                                "invalid prev kv cumulative batch=%u value=%ld", curBatch, prevKvRaw);
                            FIA_HARD_FAIL_IF(kvSeqRaw < prevKvRaw,
                                "non-monotonic kv cumulative seqlen batch=%u prev=%ld curr=%ld",
                                curBatch, prevKvRaw, kvSeqRaw);
                            uint32_t prevKvSeqlenSum = static_cast<uint32_t>(prevKvRaw);
                            kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                        }
                    }
                    if constexpr (PAGED_CACHE_FLAG) {
                        uint64_t kvCapacity = static_cast<uint64_t>(maxNumBlocksPerBatch) * pagedBlockSize;
                        FIA_HARD_FAIL_IF(static_cast<uint64_t>(kvSeqlen) > kvCapacity,
                            "kvSeqlen(%u) exceeds paged capacity(%llu) batch=%u",
                            kvSeqlen, static_cast<unsigned long long>(kvCapacity), curBatch);
                    }
                    if constexpr (PAGED_CACHE_FLAG) {
                        InitSparseBatchDecodeBounds(kvSeqlen, sparseKMax, pagedBlockSize, sparseBatchBounds);
                    }
                    curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                    qNBlockNumPerGroup = NpuArch::Detail::Alignment::CeilDiv(groupSize, curQNBlockTile);
                    curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                    curQSBlockTile = GetQSBlockTile(kvSeqlen);
                    curQSBlockNum = NpuArch::Detail::Alignment::CeilDiv(qSeqlen, curQSBlockTile);
                    curTotalTaskNum += curQNBlockNum * curQSBlockNum;
                }
                uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
                uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
                uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
                uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;

                uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
                FIA_HARD_FAIL_IF(kvNIdx >= kvHeads,
                    "kvNIdx(%u) out of range kvHeads=%u", kvNIdx, kvHeads);
                uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
                uint32_t lseTokenOffset = qSBlockIdx * curQSBlockTile * qHeads;

                uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
                    (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
                uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U)) ?
                    (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;
                FIA_HARD_FAIL_IF((qNStartIdx >= qHeads) || ((qNStartIdx + qNBlockSize) > qHeads),
                    "q-head tile overflow qNStartIdx=%u qNBlockSize=%u qHeads=%u",
                    qNStartIdx, qNBlockSize, qHeads);
                bool serialSparseHeads = false;
                if constexpr (PAGED_CACHE_FLAG) {
                    serialSparseHeads =
                        ShouldSerialiseSelectedKvSparseHeads(selectedKvSparseState, qSeqlen, qSBlockSize);
                }
                uint32_t headTaskNum = serialSparseHeads ? qNBlockSize : 1U;
                for (uint32_t headTaskIdx = 0; headTaskIdx < headTaskNum; ++headTaskIdx) {
                    uint32_t headQNStartIdx = qNStartIdx + (serialSparseHeads ? headTaskIdx : 0U);
                    uint32_t headQNBlockSize = serialSparseHeads ? 1U : qNBlockSize;
                    uint32_t rowNum = qSBlockSize * headQNBlockSize;
                    uint64_t gmOffsetSink = headQNStartIdx;
                    uint64_t gmOffsetQ = qBOffset +
                        static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ +
                        static_cast<uint64_t>(headQNStartIdx * embed);
                    uint64_t gmOffsetK = kBOffset + static_cast<uint64_t>(kvNIdx * embed);
                    uint64_t gmOffsetV = vBOffset + static_cast<uint64_t>(kvNIdx * embedV);
                    uint64_t gmOffsetO = oBOffset +
                        static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO +
                        static_cast<uint64_t>(headQNStartIdx * embedV);
                    uint64_t gmOffsetLse = lseBOffset +
                        static_cast<uint64_t>(lseTokenOffset + headQNStartIdx);

                    SparseHeadDecodeState sparseHeadState;
                    InitSparseHeadDecodeState(kvSeqlen, maskType, sparseHeadState);
                    if constexpr (PAGED_CACHE_FLAG) {
                        PrepareSelectedKvSparseHeadState(
                            serialSparseHeads,
                            selectedKvSparseState,
                            gSelectedKvIndices,
                            curBatch,
                            qHeads,
                            headQNStartIdx,
                            kvSeqlen,
                            sparseBatchBounds,
                            qSeqlen,
                            qSBlockSize,
                            headQNBlockSize,
                            sparseHeadState);
                    }
                    uint32_t taskMaskType = sparseHeadState.taskMaskType;
                    int64_t noSkipKvS = ResolveHeadTaskKvSpan(
                        taskMaskType, sparseHeadState.noSkipKvS, kvSeqlen, qSeqlen, qSBlockIdx, curQSBlockTile);
                    if ((coreIdx == 0U) && (taskIdx == 0U) && (headTaskIdx == 0U)) {
                        int32_t firstSelectedBlock = GetSparseFirstSelectedBlockForDebug(gSelectedKvIndices, sparseHeadState);
                        FIA_DEBUG_PRINTF(
                            "[SplitFuse] b=%u qS=%u kvS=%u qNTile=%u qNBlock=%u useSparse=%u selCnt=%u firstSel=%d sparseLaunch=%u kmax=%u\n",
                            curBatch, qSeqlen, kvSeqlen, curQNBlockTile, headQNBlockSize,
                            static_cast<uint32_t>(sparseHeadState.useSparseDecode),
                            sparseHeadState.selectedBlockCount,
                            firstSelectedBlock,
                            sparseLaunchEnabled, sparseKMax);
                    }

                    uint32_t kvSLoopNumTotal = NpuArch::Detail::Alignment::CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
                    uint32_t blockStackNum = (MAX_KV_STACK_LEN - 1 + pagedBlockSize) / pagedBlockSize;
                    uint32_t stackSeqTile = MAX_KV_STACK_LEN;
                    uint32_t stackSeqTilePad = MAX_KV_STACK_LEN;
                    uint32_t preKVNum = PRE_LAUNCH;
                    int32_t stackSeqCount = 0;
                    uint32_t stackSelectedPhysicalIds[PRE_LAUNCH + 1U][KernelCommon::MAX_SELECTED_KV_STACK_BLOCKS] = {};
                    uint32_t stackSelectedPhysicalCount[PRE_LAUNCH + 1U] = {};
#ifdef __DAV_C220_VEC__
                    if (kvSLoopNumTotal <= 0) {
                        LayoutO layoutO(qSeqlen, embed * qHeads);
                        LayoutLse layoutLse(totalQTokens, qHeads);
                        epilogueInitOut(gO[gmOffsetO], gLse[gmOffsetLse], layoutO, layoutLse, qSBlockSize, headQNBlockSize);
                    }
#endif
#ifdef __DAV_C220_CUBE__
                    LayoutQ layoutQTemp(rowNum, embed);
                    LayoutK layoutKTemp(strideK, stackSeqTile);
                    LayoutV layoutVTemp(stackSeqTile, strideV);
                    blockMmadQK.resetBlockStart();
                    blockMmadPV.resetBlockStart();
                    blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, headQNBlockSize, qHeads);
#endif
                    for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx ++) {
                        if (kvSIdx < kvSLoopNumTotal) {
                            if (kvSIdx + 1 > kvSLoopNumTotal - 1U) {
                                stackSeqTile = noSkipKvS - kvSIdx * MAX_KV_STACK_LEN;
                            } else {
                                stackSeqTile = MAX_KV_STACK_LEN;
                            }
                            isLastStackTile = (kvSIdx + 1) >= kvSLoopNumTotal;
                            uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1U);
                            if (sparseHeadState.useSparseDecode) {
                                PrepareSelectedKvSparseWindowPhysicalIds(
                                    selectedKvSparseState,
                                    gSelectedKvIndices,
                                    gBlockTable,
                                    blockBOffset,
                                    kvSIdx,
                                    sparseHeadState,
                                    stackSelectedPhysicalIds[curStackTileMod],
                                    stackSelectedPhysicalCount[curStackTileMod]);
                            }
                            uint64_t gmOffsetS =
                                static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                            GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                            LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                            if constexpr (PAGED_CACHE_FLAG) {
                                blockMmadQK(
                                    gQ[gmOffsetQ],
                                    gK[gmOffsetK],
                                    gS[gmOffsetS],
                                    gBlockTable[blockBOffset],
                                    layoutQTemp,
                                    layoutKTemp,
                                    layOutS,
                                    actualBlockShapeQK,
                                    kvSIdx,
                                    kvSLoopNumTotal,
                                    pagedBlockSize,
                                    strideK,
                                    stackSelectedPhysicalIds[curStackTileMod],
                                    stackSelectedPhysicalCount[curStackTileMod],
                                    sparseHeadState.useSparseDecode);
                            } else {
                                blockMmadQK(
                                    gQ[gmOffsetQ],
                                    gK[gmOffsetK],
                                    gS[gmOffsetS],
                                    gBlockTable,
                                    layoutQTemp,
                                    layoutKTemp,
                                    layOutS,
                                    actualBlockShapeQK,
                                    kvSIdx,
                                    kvSLoopNumTotal,
                                    pagedBlockSize,
                                    strideK,
                                    stackSelectedPhysicalIds[curStackTileMod],
                                    stackSelectedPhysicalCount[curStackTileMod],
                                    sparseHeadState.useSparseDecode);
                            }
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                            LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                            LayoutMask layOutMask(COMP_TRIU_MASK_DIM_LEN, COMP_TRIU_MASK_DIM_LEN);
                            uint64_t gmOffsetP = gmOffsetS;
                            uint32_t triUp = noSkipKvS - qSBlockSize;
                            uint32_t triDown = noSkipKvS;
                            uint32_t kvSStartIdx = kvSIdx * MAX_KV_STACK_LEN;
                            uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
                            bool doTriUMask = triUp < kvSEndIdx - 1;
                            if constexpr (MASK_TYPE == FaiKernel::MaskType::MASK_CAUSAL) {
                                if (doTriUMask) {
                                    epilogueOnlineSoftmax(
                                        gP[gmOffsetP],
                                        gS[gmOffsetS],
                                        gSink[gmOffsetSink],
                                        gMask,
                                        layOutP,
                                        layOutS,
                                        layOutMask,
                                        actualBlockShapeQK,
                                        (stackSeqCount == 0),
                                        qSBlockSize,
                                        headQNBlockSize,
                                        curStackTileMod,
                                        qkReady,
                                        triUp,
                                        triDown,
                                        kvSStartIdx,
                                        kvSEndIdx,
                                        isLastStackTile);
                                } else {
                                    uint32_t noMaskStackSeqNum = (triUp + 1) / MAX_KV_STACK_LEN;
                                    Arch::CrossCoreWaitFlag(qkReady);
                                    epilogueOnlineSoftmax(
                                        gP[gmOffsetP],
                                        gS[gmOffsetS],
                                        gSink[gmOffsetSink],
                                        layOutP,
                                        layOutS,
                                        actualBlockShapeQK,
                                        (stackSeqCount == 0),
                                        (stackSeqCount == noMaskStackSeqNum - 1),
                                        qSBlockSize,
                                        headQNBlockSize,
                                        curStackTileMod,
                                        isLastStackTile);
                                }
                            } else {
                                Arch::CrossCoreWaitFlag(qkReady);
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    gSink[gmOffsetSink],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    0,
                                    qSBlockSize,
                                    headQNBlockSize,
                                    curStackTileMod,
                                    isLastStackTile);
                            }
                            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                        }
                        if (kvSIdx >= preKVNum) {
                            uint32_t nowkvSIdx = kvSIdx - preKVNum;
                            if (nowkvSIdx + 1 > kvSLoopNumTotal - 1U) {
                                stackSeqTile = noSkipKvS - nowkvSIdx * MAX_KV_STACK_LEN;
                            } else {
                                stackSeqTile = MAX_KV_STACK_LEN;
                            }
                            uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1U);
                            uint64_t gmOffsetOTmp =
                                static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                            GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};
                            LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__
                            LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
                            uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;
                            if constexpr (PAGED_CACHE_FLAG) {
                                blockMmadPV(
                                    gP[gmOffsetP],
                                    gV[gmOffsetV],
                                    gOTmp[gmOffsetOTmp],
                                    gBlockTable[blockBOffset],
                                    layoutPTemp,
                                    layoutVTemp,
                                    layoutOTmp,
                                    actualBlockShapePV,
                                    nowkvSIdx,
                                    kvSLoopNumTotal,
                                    pagedBlockSize,
                                    noSkipKvS,
                                    strideV,
                                    blockStackNum,
                                    softmaxReady,
                                    stackSelectedPhysicalIds[curStackTileMod],
                                    stackSelectedPhysicalCount[curStackTileMod],
                                    sparseHeadState.useSparseDecode);
                            } else {
                                blockMmadPV(
                                    gP[gmOffsetP],
                                    gV[gmOffsetV],
                                    gOTmp[gmOffsetOTmp],
                                    gBlockTable,
                                    layoutPTemp,
                                    layoutVTemp,
                                    layoutOTmp,
                                    actualBlockShapePV,
                                    nowkvSIdx,
                                    kvSLoopNumTotal,
                                    pagedBlockSize,
                                    noSkipKvS,
                                    strideV,
                                    blockStackNum,
                                    softmaxReady,
                                    stackSelectedPhysicalIds[curStackTileMod],
                                    stackSelectedPhysicalCount[curStackTileMod],
                                    sparseHeadState.useSparseDecode);
                            }
                            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                            LayoutO layoutO(qSeqlen, embed * qHeads);
                            LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                            LayoutLse layoutLse(totalQTokens, qHeads);
                            uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                            Arch::CrossCoreWaitFlag(pvReady);
                            epilogueRescaleO(
                                gO[gmOffsetO],
                                gOTmp[gmOffsetOTmp],
                                gOUpdate[gmOffsetUpdate],
                                gLse[gmOffsetLse],
                                layoutO,
                                layoutOTmp,
                                layoutUpdate,
                                layoutLse,
                                actualBlockShapePV,
                                qSBlockSize,
                                headQNBlockSize,
                                (stackSeqCount - PRE_LAUNCH == 0),
                                nowkvSIdx + 1 >= kvSLoopNumTotal,
                                curStackTileMod);
#endif
                        }
                        stackSeqCount++;
                    }
                }
            }
#ifdef __DAV_C220_CUBE__
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
            AscendC::PipeBarrier<PIPE_ALL>();
        }

    private:
        Arch::Resource<ArchTag> resource;
        Arch::CrossCoreFlag qkReady{QK_READY_ID};
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        Arch::CrossCoreFlag pvReady{PV_READY_ID};
    };
}
#endif
