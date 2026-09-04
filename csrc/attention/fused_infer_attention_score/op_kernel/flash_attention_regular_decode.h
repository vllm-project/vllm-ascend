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
 * \file flash_attention_regular_decode.h
 * \brief
 */
#ifndef FLASH_ATTENTION_REGULAR_DECODE_H
#define FLASH_ATTENTION_REGULAR_DECODE_H

#include "fia_kernel_common.hpp"

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
    class FAInferKernelDecoding {
    public:
        using ArchTag = typename BlockMmadQK::ArchTag;
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
        using L1TileShape = typename BlockMmadQK::L1TileShape;

        using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
        using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;

        using ElementLse = typename EpilogueRescaleO::ElementLse;
        using LayoutLse = typename EpilogueRescaleO::LayoutLse;

        using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
        using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

        using ElementOTmp = typename EpilogueRescaleO::ElementInput;
        using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

        using ElementO = typename EpilogueRescaleO::ElementOutput;
        using LayoutO = typename EpilogueRescaleO::LayoutOutput;

        static constexpr Epilogue::LseMode LSE_MODE = EpilogueRescaleO::LSE_MODE;
        static constexpr Epilogue::SinkMode SINK_MODE = EpilogueOnlineSoftmax::SINK_MODE;

        // Methods
        __aicore__ inline
        FAInferKernelDecoding() {}

        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            __gm__ FAInferTilingData *fATilingData = reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
            uint64_t mm1OutSize = fATilingData->mm1OutSize;
            uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize;
            uint64_t mm2OutSize = fATilingData->mm2OutSize;
            uint32_t batch = fATilingData->batch;
            uint32_t qHeads = fATilingData->numHeads;
            uint32_t kvHeads = fATilingData->kvHeads;
            uint32_t embed = fATilingData->embeddingSize;
            uint32_t embedV = fATilingData->embeddingSizeV;
            uint32_t pagedBlockSize = fATilingData->blockSize;
            uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
            uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;
            uint32_t totalTaskNum = fATilingData->totalTaskNum;
            uint32_t mainLoopTaskNum = fATilingData->mainLoopTaskNum;
            uint32_t tailLoopTaskNum = fATilingData->tailLoopTaskNum;
            uint32_t tailStartBatch = fATilingData->tailStartBatch;
            uint32_t tailStartN2 = fATilingData->tailStartN2;
            uint32_t tailKvNBlockTile = fATilingData->tailKvNBlockTile;
            uint32_t blockSize = fATilingData->blockSize;
            uint32_t maskType = fATilingData->maskType;
            float scaleValue = fATilingData->scaleValue;
            uint32_t sparseMode = fATilingData->sparseMode;
            int64_t preToken = fATilingData->preToken;
            int64_t nextToken = fATilingData->nextToken;
            uint64_t keyBnStride = fATilingData->keyStrides.bnStride;
            uint64_t valueBnStride = fATilingData->valueStrides.bnStride;
            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            AscendC::ListTensorDesc keyListTensorDescInit((__gm__ void*)params.k);
            AscendC::ListTensorDesc valueListTensorDescInit((__gm__ void*)params.v);
            __gm__ uint8_t* currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            __gm__ uint8_t* currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)currentKey);
            AscendC::GlobalTensor<ElementK> gV;
            gV.SetGlobalBuffer((__gm__ ElementK *)currentValue);
            AscendC::GlobalTensor<ElementMask> gMask;
            gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
            AscendC::GlobalTensor<int32_t> gBlockTable;
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
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
            blockMmadQK.init(resource, nDynNum, kDynNum);
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            blockMmadPV.init(resource, nDynNum, kPVDynNum, L1_QK_SIZE);
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

            epilogueOnlineSoftmax.init(resource, scaleValue);
            epilogueRescaleO.init(resource);
            epilogueInitOut.init(resource);

            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
            uint64_t strideQ = static_cast<uint64_t>(qHeads * embed);
            uint64_t strideO = static_cast<uint64_t>(qHeads * embedV);
            uint64_t strideK = static_cast<uint64_t>(kvHeads * embed);
            uint64_t strideV = static_cast<uint64_t>(kvHeads * embedV);
            uint32_t embedRound = NpuArch::Detail::Alignment::RoundUp(embed, FaiKernel::BLOCK_SIZE);
            uint32_t embedRoundV = NpuArch::Detail::Alignment::RoundUp(embedV, FaiKernel::BLOCK_SIZE);
            uint32_t groupSize = qHeads / kvHeads;

            uint32_t totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch - 1));
            uint32_t curKvNBlockTile = GetKvNBlockTile(groupSize, kvHeads);
            uint32_t curKvNBlockNum = firstBatchTaskNum;
            uint32_t rowNumConst = groupSize;
            uint32_t rowNumRoundConst = NpuArch::Detail::Alignment::RoundUp(rowNumConst, FaiKernel::BLOCK_SIZE);
            uint32_t useMainLoopTaskNum = (tailKvNBlockTile > 0) ? mainLoopTaskNum : totalTaskNum;

            for (uint32_t taskIdx = coreIdx; taskIdx < useMainLoopTaskNum; taskIdx += uint32_t(coreNum)) {
                uint32_t curBatch = taskIdx / firstBatchTaskNum;
                uint32_t kvNBlockIdx = taskIdx - curBatch * firstBatchTaskNum;

                uint32_t qSeqlen = 1;
                uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                    if constexpr (!PAGED_CACHE_FLAG) {
                        uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                            0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch - 1));
                        kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                    }
                }

                uint64_t qBOffset = static_cast<uint64_t>(curBatch) * strideQ;
                uint64_t oBOffset = static_cast<uint64_t>(curBatch) * strideO;
                uint64_t lseBOffset = static_cast<uint64_t>(curBatch) * qHeads;
                uint64_t kBOffset = 0;
                uint64_t vBOffset = 0;
                uint64_t blockBOffset = 0;
                if constexpr (PAGED_CACHE_FLAG) {
                    blockBOffset = static_cast<uint64_t>(curBatch) * maxNumBlocksPerBatch;
                } else {
                    uint64_t cumKvSeqlen = (curBatch > 0) ?
                        static_cast<uint64_t>(gActualKvseqlen.GetValue(curBatch - 1)) : 0;
                    kBOffset = cumKvSeqlen * strideK;
                    vBOffset = cumKvSeqlen * strideV;
                }

                uint32_t kvNStartIdx = kvNBlockIdx * curKvNBlockTile;
                uint32_t qNStartIdx = kvNStartIdx * groupSize;

                uint32_t qSBlockSize = 1;
                uint32_t gBlockSize = groupSize;
                uint32_t kvNBlockSize = (kvNBlockIdx == (curKvNBlockNum - 1U)) ?
                    (kvHeads - kvNBlockIdx * curKvNBlockTile) : curKvNBlockTile;
                uint32_t rowNum = rowNumConst;
                uint32_t rowNumRound = rowNumRoundConst;

                uint64_t qSOffset = 0;
                uint64_t qNStartOffset = static_cast<uint64_t>(qNStartIdx) * embed;
                uint64_t kNStartOffset = static_cast<uint64_t>(kvNStartIdx) * embed;
                uint64_t vNStartOffset = static_cast<uint64_t>(kvNStartIdx) * embedV;
                uint64_t oSOffset = 0;
                uint64_t oNStartOffset = static_cast<uint64_t>(qNStartIdx) * embedV;
                uint64_t lseTokenOffset = 0;

                uint32_t noSkipKvS = kvSeqlen;
                uint32_t kvSLoopNumTotal = NpuArch::Detail::Alignment::CeilDiv(noSkipKvS, pagedBlockSize);

                uint32_t blockStackNum = MAX_KV_STACK_LEN / pagedBlockSize;
                uint32_t stackSeqTile;
                uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
                uint32_t preKVNum = PRE_LAUNCH * blockStackNum;
                int32_t stackSeqCount = 0;
#ifdef __DAV_C220_CUBE__
                LayoutQ layoutQTemp(rowNum, embed);
                LayoutK layoutKTemp(strideK, blockStackNum * pagedBlockSize);
                LayoutV layoutVTemp(blockStackNum * pagedBlockSize, strideV);
#endif
                for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx += blockStackNum) {
                    if (kvSIdx < kvSLoopNumTotal) {
                        if (kvSIdx + blockStackNum > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - kvSIdx * pagedBlockSize;
                        } else {
                            stackSeqTile = pagedBlockSize * blockStackNum;
                        }
                        uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1U);
#ifdef __DAV_C220_CUBE__

                        uint64_t gmOffsetQGmtoL1 = qBOffset + qSOffset + qNStartOffset;
                        if (kvSIdx == 0) {
                            uint32_t taskRowNum = rowNum * kvNBlockSize;
                            LayoutQ layoutQL1(taskRowNum, embed);
                            uint32_t taskColNum = gBlockSize * kvNBlockSize;
                            blockMmadQK.loadQGM(gQ[gmOffsetQGmtoL1], layoutQL1, taskRowNum,
                                taskColNum, qHeads, kvNBlockSize);
                        }
#endif

                        for (uint32_t kvNIncreIdx = 0; kvNIncreIdx < kvNBlockSize; kvNIncreIdx++) {
                            uint64_t gmOffsetQ = qBOffset + qSOffset + qNStartOffset +
                                static_cast<uint64_t>(kvNIncreIdx * groupSize * embed);
                            uint64_t gmOffsetK = kBOffset + kNStartOffset +
                                static_cast<uint64_t>(kvNIncreIdx * embed);
                            uint64_t sWorkspaceIncreOffset =
                                static_cast<uint64_t>(kvNIncreIdx) * rowNum * MAX_KV_STACK_LEN;
                            uint64_t gmOffsetS =
                                static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB +
                                sWorkspaceIncreOffset;
                            GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                            LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
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
                                kvNIncreIdx,
                                keyBnStride);
                            if (kvNIncreIdx == kvNBlockSize - 1) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                            }
#endif
                        }
#ifdef __DAV_C220_VEC__
                        LayoutP layOutP(rowNum * kvNBlockSize, stackSeqTile, stackSeqTilePad);

                        uint64_t gmOffsetSBase =
                            static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB;
                        uint64_t gmOffsetPBase = gmOffsetSBase;
                        LayoutS layOutS(rowNum * kvNBlockSize, stackSeqTile, stackSeqTilePad);

                        GemmCoord actualBlockShapeQK{rowNum * kvNBlockSize, stackSeqTile, embed};

                        epilogueOnlineSoftmax(
                            gP[gmOffsetPBase],
                            gS[gmOffsetSBase],
                            layOutP,
                            layOutS,
                            actualBlockShapeQK,
                            (stackSeqCount == 0),
                            0,
                            qSBlockSize,
                            gBlockSize,
                            curStackTileMod,
                            kvNBlockSize,
                            gmOffsetSBase,
                            gmOffsetPBase,
                            qkReady,
                            softmaxReady);
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                    }
                    if (kvSIdx >= preKVNum) {
                        uint32_t nowkvSIdx = kvSIdx - preKVNum;
                        if (nowkvSIdx + blockStackNum > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - nowkvSIdx * pagedBlockSize;
                        } else {
                            stackSeqTile = pagedBlockSize * blockStackNum;
                        }
                        uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1U);

                        for (uint32_t kvNIncreIdx = 0; kvNIncreIdx < kvNBlockSize; kvNIncreIdx++) {
                            uint64_t gmOffsetV = vBOffset + vNStartOffset +
                                static_cast<uint64_t>(kvNIncreIdx * embedV);
                            uint64_t gmOffsetO = oBOffset + oSOffset + oNStartOffset +
                                static_cast<uint64_t>(kvNIncreIdx * groupSize * embed);
                            uint64_t gmOffsetLse = lseBOffset + lseTokenOffset + qNStartIdx +
                                static_cast<uint64_t>(kvNIncreIdx * groupSize);

                            uint64_t oWorkspaceIncreOffset = static_cast<uint64_t>(kvNIncreIdx) * rowNum * embedRoundV;
                            uint64_t gmOffsetOTmp =
                                static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB +
                                oWorkspaceIncreOffset;
                            GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};
                            LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__

                            uint64_t pWorkspaceIncreOffset =
                                static_cast<uint64_t>(kvNIncreIdx) * rowNum * stackSeqTilePad;
                            uint64_t gmOffsetP =
                                static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB +
                                pWorkspaceIncreOffset;
                            LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
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
                                (kvNIncreIdx == 0),
                                valueBnStride);
                            if (kvNIncreIdx == kvNBlockSize - 1) {
                                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                            }
#endif
                        }
#ifdef __DAV_C220_VEC__

                        LayoutO layoutO(qSeqlen, embed * qHeads);
                        LayoutOTmp layoutUpdate(rowNum * kvNBlockSize, embed, embedRound);
                        LayoutLse layoutLse(totalQTokens, qHeads);
                        LayoutOTmp layoutOTmp(rowNum * kvNBlockSize, embedV, embedRoundV);
                        GemmCoord actualBlockShapePV{rowNum * kvNBlockSize, embedV, stackSeqTile};

                        Arch::CrossCoreWaitFlag(pvReady);

                        uint64_t gmOffsetO = oBOffset + oSOffset + oNStartOffset;
                        uint64_t gmOffsetUpdate = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB;
                        uint64_t gmOffsetOTmp =
                                static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB;
                        uint64_t gmOffsetLse = lseBOffset + lseTokenOffset + qNStartIdx;

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
                            gBlockSize,
                            kvNBlockSize,
                            (stackSeqCount - PRE_LAUNCH == 0),
                            nowkvSIdx + blockStackNum >= kvSLoopNumTotal,
                            curStackTileMod,
                            1U);
#endif
                    }
                    stackSeqCount++;
                }
            }
            if (tailKvNBlockTile > 0 && tailLoopTaskNum > 0) {
                uint32_t tailFirstBatchTasks = kvHeads - tailStartN2;

                for (uint32_t tailTaskIdx = coreIdx; tailTaskIdx < tailLoopTaskNum;
                     tailTaskIdx += uint32_t(coreNum)) {
                    uint32_t tailCurBatch;
                    uint32_t tailKvNBlockIdx;
                    uint32_t tailCurKvNStartOffset;
                    if (tailTaskIdx < tailFirstBatchTasks) {
                        tailCurBatch = tailStartBatch;
                        tailKvNBlockIdx = tailTaskIdx;
                        tailCurKvNStartOffset = tailStartN2;
                    } else {
                        uint32_t adjustedIdx = tailTaskIdx - tailFirstBatchTasks;
                        uint32_t batchOffset = adjustedIdx / kvHeads;
                        tailCurBatch = tailStartBatch + 1 + batchOffset;
                        tailKvNBlockIdx = adjustedIdx - batchOffset * kvHeads;
                        tailCurKvNStartOffset = 0;
                    }

                    uint32_t tailQSeqlen = 1;
                    uint32_t tailKvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(tailCurBatch));
                    if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                        if constexpr (!PAGED_CACHE_FLAG) {
                            uint32_t prevKvSum = (tailCurBatch == 0) ?
                                0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(tailCurBatch - 1));
                            tailKvSeqlen = tailKvSeqlen - prevKvSum;
                        }
                    }

                    uint64_t tailQBOffset = static_cast<uint64_t>(tailCurBatch) * strideQ;
                    uint64_t tailOBOffset = static_cast<uint64_t>(tailCurBatch) * strideO;
                    uint64_t tailLseBOffset = static_cast<uint64_t>(tailCurBatch) * qHeads;
                    uint64_t tailKBOffset = 0;
                    uint64_t tailVBOffset = 0;
                    uint64_t tailBlockBOffset = 0;
                    if constexpr (PAGED_CACHE_FLAG) {
                        tailBlockBOffset = static_cast<uint64_t>(tailCurBatch) * maxNumBlocksPerBatch;
                    } else {
                        uint64_t cumKvSeqlen = (tailCurBatch > 0) ?
                            static_cast<uint64_t>(gActualKvseqlen.GetValue(tailCurBatch - 1)) : 0;
                        tailKBOffset = cumKvSeqlen * strideK;
                        tailVBOffset = cumKvSeqlen * strideV;
                    }

                    uint32_t tailKvNStartIdx = tailCurKvNStartOffset + tailKvNBlockIdx;
                    uint32_t tailQNStartIdx = tailKvNStartIdx * groupSize;

                    uint32_t tailQSBlockSize = 1;
                    uint32_t tailGBlockSize = groupSize;
                    uint32_t tailKvNBlockSize = 1;
                    uint32_t rowNum = rowNumConst;
                    uint32_t rowNumRound = rowNumRoundConst;

                    uint64_t qSOffset = 0;
                    uint64_t qNStartOffset = static_cast<uint64_t>(tailQNStartIdx) * embed;
                    uint64_t kNStartOffset = static_cast<uint64_t>(tailKvNStartIdx) * embed;
                    uint64_t vNStartOffset = static_cast<uint64_t>(tailKvNStartIdx) * embedV;
                    uint64_t oSOffset = 0;
                    uint64_t oNStartOffset = static_cast<uint64_t>(tailQNStartIdx) * embedV;
                    uint64_t lseTokenOffset = 0;

                    uint32_t noSkipKvS = tailKvSeqlen;
                    uint32_t kvSLoopNumTotal = NpuArch::Detail::Alignment::CeilDiv(noSkipKvS, pagedBlockSize);

                    uint32_t blockStackNum = MAX_KV_STACK_LEN / pagedBlockSize;
                    uint32_t stackSeqTilePad = blockStackNum * pagedBlockSize;
                    uint32_t preKVNum = PRE_LAUNCH * blockStackNum;
                    uint32_t stackSeqTileTemp;
                    int32_t stackSeqCount = 0;

#ifdef __DAV_C220_CUBE__
                    LayoutQ layoutQTemp(rowNum, embed);
                    LayoutK layoutKTemp(strideK, blockStackNum * pagedBlockSize);
                    LayoutV layoutVTemp(blockStackNum * pagedBlockSize, strideV);
#endif
                    for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx += blockStackNum) {
                        if (kvSIdx < kvSLoopNumTotal) {
                            if (kvSIdx + blockStackNum > kvSLoopNumTotal - 1U) {
                                stackSeqTileTemp = noSkipKvS - kvSIdx * pagedBlockSize;
                            } else {
                                stackSeqTileTemp = pagedBlockSize * blockStackNum;
                            }
                            uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1U);
#ifdef __DAV_C220_CUBE__
                            uint64_t gmOffsetQGmtoL1 = tailQBOffset + qSOffset + qNStartOffset;
                            if (kvSIdx == 0) {
                                uint32_t taskRowNum = rowNum * tailKvNBlockSize;
                                LayoutQ layoutQL1(taskRowNum, embed);
                                uint32_t taskColNum = tailGBlockSize * tailKvNBlockSize;
                                blockMmadQK.loadQGM(gQ[gmOffsetQGmtoL1], layoutQL1, taskRowNum,
                                    taskColNum, qHeads, tailKvNBlockSize);
                            }
#endif

                            for (uint32_t kvNIncreIdx = 0; kvNIncreIdx < tailKvNBlockSize; kvNIncreIdx++) {
                                uint64_t gmOffsetQ = tailQBOffset + qSOffset + qNStartOffset +
                                    static_cast<uint64_t>(kvNIncreIdx * groupSize * embed);
                                uint64_t gmOffsetK = tailKBOffset + kNStartOffset +
                                    static_cast<uint64_t>(kvNIncreIdx * embed);
                                uint64_t sWorkspaceIncreOffset =
                                    static_cast<uint64_t>(kvNIncreIdx) * rowNum * MAX_KV_STACK_LEN;
                                uint64_t gmOffsetS =
                                    static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                    static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB +
                                    sWorkspaceIncreOffset;
                                GemmCoord actualBlockShapeQK{rowNum, stackSeqTileTemp, embed};
                                LayoutS layOutS(rowNum, stackSeqTileTemp, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                                if constexpr (PAGED_CACHE_FLAG) {
                                    blockMmadQK(
                                        gQ[gmOffsetQ],
                                        gK[gmOffsetK],
                                        gS[gmOffsetS],
                                        gBlockTable[tailBlockBOffset],
                                        layoutQTemp,
                                        layoutKTemp,
                                        layOutS,
                                        actualBlockShapeQK,
                                        kvSIdx,
                                        kvSLoopNumTotal,
                                        pagedBlockSize,
                                        strideK,
                                        kvNIncreIdx,
                                        keyBnStride);
                                }

                                if (kvNIncreIdx == tailKvNBlockSize - 1) {
                                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                                }
#endif
                            }
#ifdef __DAV_C220_VEC__
                            LayoutP layOutP(rowNum * tailKvNBlockSize, stackSeqTileTemp, stackSeqTilePad);
                            uint64_t gmOffsetSBase =
                                static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB;
                            uint64_t gmOffsetPBase = gmOffsetSBase;
                            LayoutS layOutS(rowNum * tailKvNBlockSize, stackSeqTileTemp, stackSeqTilePad);
                            GemmCoord actualBlockShapeQK{rowNum * tailKvNBlockSize, stackSeqTileTemp, embed};

                            epilogueOnlineSoftmax(
                                gP[gmOffsetPBase],
                                gS[gmOffsetSBase],
                                layOutP,
                                layOutS,
                                actualBlockShapeQK,
                                (stackSeqCount == 0),
                                0,
                                tailQSBlockSize,
                                tailGBlockSize,
                                curStackTileMod,
                                tailKvNBlockSize,
                                gmOffsetSBase,
                                gmOffsetPBase,
                                qkReady,
                                softmaxReady);
                            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                        }
                        if (kvSIdx >= preKVNum) {
                            uint32_t nowkvSIdx = kvSIdx - preKVNum;
                            if (nowkvSIdx + blockStackNum > kvSLoopNumTotal - 1U) {
                                stackSeqTileTemp = noSkipKvS - nowkvSIdx * pagedBlockSize;
                            } else {
                                stackSeqTileTemp = pagedBlockSize * blockStackNum;
                            }
                            uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1U);

                            for (uint32_t kvNIncreIdx = 0; kvNIncreIdx < tailKvNBlockSize; kvNIncreIdx++) {
                                uint64_t gmOffsetV = tailVBOffset + vNStartOffset +
                                    static_cast<uint64_t>(kvNIncreIdx * embedV);
                                uint64_t gmOffsetO = tailOBOffset + oSOffset + oNStartOffset +
                                    static_cast<uint64_t>(kvNIncreIdx * groupSize * embed);
                                uint64_t gmOffsetLse = tailLseBOffset + lseTokenOffset + tailQNStartIdx +
                                    static_cast<uint64_t>(kvNIncreIdx * groupSize);
                                uint64_t oWorkspaceIncreOffset =
                                    static_cast<uint64_t>(kvNIncreIdx) * rowNum * embedRoundV;
                                uint64_t gmOffsetOTmp =
                                    static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                    static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB +
                                    oWorkspaceIncreOffset;
                                GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTileTemp};
                                LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__
                                uint64_t pWorkspaceIncreOffset =
                                    static_cast<uint64_t>(kvNIncreIdx) * rowNum * stackSeqTilePad;
                                uint64_t gmOffsetP =
                                    static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                                    static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB +
                                    pWorkspaceIncreOffset;
                                LayoutP layoutPTemp(rowNum, stackSeqTileTemp, stackSeqTilePad);

                                blockMmadPV(
                                    gP[gmOffsetP],
                                    gV[gmOffsetV],
                                    gOTmp[gmOffsetOTmp],
                                    gBlockTable[tailBlockBOffset],
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
                                    (kvNIncreIdx == 0),
                                    valueBnStride);
                                if (kvNIncreIdx == tailKvNBlockSize - 1) {
                                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                                }
#endif
                            }
#ifdef __DAV_C220_VEC__
                            LayoutO layoutO(tailQSeqlen, embed * qHeads);
                            LayoutOTmp layoutUpdate(rowNum * tailKvNBlockSize, embed, embedRound);
                            LayoutLse layoutLse(totalQTokens, qHeads);
                            LayoutOTmp layoutOTmp(rowNum * tailKvNBlockSize, embedV, embedRoundV);
                            GemmCoord actualBlockShapePV{rowNum * tailKvNBlockSize, embedV, stackSeqTileTemp};

                            Arch::CrossCoreWaitFlag(pvReady);

                            uint64_t gmOffsetO = tailOBOffset + oSOffset + oNStartOffset;
                            uint64_t gmOffsetUpdate = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB;
                            uint64_t gmOffsetOTmp =
                                static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                                static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB;
                            uint64_t gmOffsetLse = tailLseBOffset + lseTokenOffset + tailQNStartIdx;

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
                                tailQSBlockSize,
                                tailGBlockSize,
                                tailKvNBlockSize,
                                (stackSeqCount - PRE_LAUNCH == 0),
                                nowkvSIdx + blockStackNum >= kvSLoopNumTotal,
                                curStackTileMod,
                                1U);
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

        BlockMmadQK blockMmadQK;
        BlockMmadPV blockMmadPV;
        EpilogueRescaleO epilogueRescaleO;
        EpilogueOnlineSoftmax epilogueOnlineSoftmax;
        EpilogueInitOut epilogueInitOut;
    };
}
#endif
