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
* \file flash_attention_regular.h
* \brief
*/
#ifndef FLASH_ATTENTION_REGULAR_H
#define FLASH_ATTENTION_REGULAR_H

#include "fia_kernel_common.hpp"

using namespace KernelCommon;
using namespace NpuArch;

namespace SplitFuse {
    template <
        class BlockMmadQK,
        class BlockMmadPV,
        class EpilogueOnlineSoftmax,
        class EpilogueRescaleO,
        class EpilogueInitOut,
        bool PAGED_CACHE_FLAG,
        FaiKernel::MaskType MASK_TYPE = FaiKernel::MaskType::NO_MASK,
        FaiKernel::inputLayout INPUT_LAYOUT = FaiKernel::inputLayout::BSND,
        class CombineScale = void,
        bool IS_FD = false>
    class FAInferKernel {
    public:
        using ArchTag = typename BlockMmadQK::ArchTag;
        using ElementQ = typename BlockMmadQK::ElementA;
        using LayoutQ = typename BlockMmadQK::LayoutA;
        using ElementK = typename BlockMmadQK::ElementB;
        using LayoutK = typename BlockMmadQK::LayoutB;
        using ElementS = typename BlockMmadQK::ElementC;
        using LayoutS = typename BlockMmadQK::LayoutC;
        using L1TileShape = typename BlockMmadQK::L1TileShape;

        using ElementP = typename BlockMmadPV::ElementA;
        using LayoutP = typename BlockMmadPV::LayoutA;
        using ElementV = typename BlockMmadPV::ElementB;
        using LayoutV = typename BlockMmadPV::LayoutB;

        using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
        using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;
        using ElementSink = typename EpilogueOnlineSoftmax::ElementSink;

        using ElementOTmp = typename EpilogueRescaleO::ElementInput;
        using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

        using ElementLse = typename EpilogueRescaleO::ElementLse;
        using LayoutLse = typename EpilogueRescaleO::LayoutLse;

        using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
        using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

        using ElementO = typename EpilogueRescaleO::ElementOutput;
        using LayoutO = typename EpilogueRescaleO::LayoutOutput;

        static constexpr Epilogue::LseMode LSE_MODE = EpilogueRescaleO::LSE_MODE;
        static constexpr Epilogue::SinkMode SINK_MODE = EpilogueOnlineSoftmax::SINK_MODE;

        struct GlobalTensorBundle {
            AscendC::GlobalTensor<ElementQ>& gQ;
            AscendC::GlobalTensor<ElementK>& gK;
            AscendC::GlobalTensor<ElementK>& gV;
            AscendC::GlobalTensor<ElementQ>& gPseShift;
            AscendC::GlobalTensor<ElementMask>& gMask;
            AscendC::GlobalTensor<int32_t>& gBlockTable;
            AscendC::GlobalTensor<int64_t>& gActualQseqlen;
            AscendC::GlobalTensor<int64_t>& gActualKvseqlen;
            AscendC::GlobalTensor<ElementO>& gO;
            AscendC::GlobalTensor<ElementLse>& gLse;
            AscendC::GlobalTensor<ElementLse>& gLseFD;
            AscendC::GlobalTensor<ElementLse>& gOFD;
            AscendC::GlobalTensor<ElementS>& gS;
            AscendC::GlobalTensor<ElementP>& gP;
            AscendC::GlobalTensor<ElementOTmp>& gOTmp;
            AscendC::GlobalTensor<ElementOTmp>& gOUpdate;
            AscendC::GlobalTensor<ElementSink>& gSink;
        };

        __aicore__ inline
        FAInferKernel() {}

        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            __gm__ FAInferTilingData *fATilingData = reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
            mm1OutSize = fATilingData->mm1OutSize;
            smOnlineOutSize = fATilingData->smOnlineOutSize;
            mm2OutSize = fATilingData->mm2OutSize;
            batch = fATilingData->batch;
            qHeads = fATilingData->numHeads;
            kvHeads = fATilingData->kvHeads;
            embed = fATilingData->embeddingSize;
            embedV = fATilingData->embeddingSizeV;
            pagedBlockSize = fATilingData->blockSize;
            maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
            firstBatchTaskNum = fATilingData->firstBatchTaskNum;
            totalTaskNum = fATilingData->totalTaskNum;
            blockSize = fATilingData->blockSize;
            maskType = fATilingData->maskType;
            scaleValue = fATilingData->scaleValue;
            sparseMode = fATilingData->sparseMode;
            preToken = fATilingData->preToken;
            nextToken = fATilingData->nextToken;
            pseQ = fATilingData->pseQ;
            pseKv = fATilingData->pseKv;
            keyBnStride = fATilingData->keyStrides.bnStride;
            valueBnStride = fATilingData->valueStrides.bnStride;
            uint64_t Lsesize = 0;
            uint64_t Losize = 0;
            if constexpr (IS_FD) {
                Lsesize = fATilingData->splitLseTotalSize;
                Losize = fATilingData->splitOTotalSize;
            }

            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            AscendC::ListTensorDesc keyListTensorDescInit((__gm__ void*)params.k);
            __gm__ uint8_t* currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            AscendC::ListTensorDesc valueListTensorDescInit((__gm__ void*)params.v);
            __gm__ uint8_t* currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)currentKey);
            AscendC::GlobalTensor<ElementK> gV;
            gV.SetGlobalBuffer((__gm__ ElementK *)currentValue);
            AscendC::GlobalTensor<ElementQ> gPseShift;
            gPseShift.SetGlobalBuffer((__gm__ ElementQ *)params.pseShift);
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            AscendC::GlobalTensor<ElementMask> gMask;
            gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
            AscendC::GlobalTensor<int32_t> gBlockTable;
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            AscendC::GlobalTensor<ElementO> gO;
            gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
            AscendC::GlobalTensor<ElementLse> gLse;
            gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);
            AscendC::GlobalTensor<ElementLse> gLseFD;
            AscendC::GlobalTensor<ElementLse> gOFD;
            if constexpr (IS_FD) {
                gLseFD.SetGlobalBuffer((__gm__ ElementLse *)(params.workSpace));
                gOFD.SetGlobalBuffer((__gm__ ElementLse *)(params.workSpace + Lsesize));
            }
            AscendC::GlobalTensor<ElementS> gS;
            gS.SetGlobalBuffer((__gm__ ElementS *)(params.workSpace + Lsesize + Losize));
            AscendC::GlobalTensor<ElementP> gP;
            gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + Lsesize + Losize + mm1OutSize));
            AscendC::GlobalTensor<ElementOTmp> gOTmp;
            gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + Lsesize + Losize +
                mm1OutSize + smOnlineOutSize));
            AscendC::GlobalTensor<ElementOTmp> gOUpdate;
            gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + Lsesize + Losize +
                mm1OutSize + smOnlineOutSize + mm2OutSize));
            AscendC::GlobalTensor<ElementSink> gSink;
            gSink.SetGlobalBuffer((__gm__ ElementSink *)(params.sink));

            GlobalTensorBundle globalTensors{
                gQ, gK, gV, gPseShift, gMask, gBlockTable,
                gActualQseqlen, gActualKvseqlen,
                gO, gLse, gLseFD, gOFD,
                gS, gP, gOTmp, gOUpdate, gSink
            };

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
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

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
            blockMmadQK.init(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN);
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            blockMmadPV.init(resource, nDynNum, kPVDynNum, MAX_KV_STACK_LEN, L1_QK_SIZE);
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
            strideQ = static_cast<uint64_t>(qHeads * embed);
            strideO = static_cast<uint64_t>(qHeads * embedV);
            strideK = static_cast<uint64_t>(kvHeads * embed);
            strideV = static_cast<uint64_t>(kvHeads * embedV);
            embedRound = NpuArch::Detail::Alignment::RoundUp(embed, FaiKernel::BLOCK_SIZE);
            embedRoundV = NpuArch::Detail::Alignment::RoundUp(embedV, FaiKernel::BLOCK_SIZE);
            groupSize = qHeads / kvHeads;

            totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch - 1));

            if constexpr (IS_FD) {
                uint32_t startBIdx = fATilingData->coreInfo.startBIdx[coreIdx];
                uint32_t startN1Idx = fATilingData->coreInfo.startN1Idx[coreIdx];
                uint32_t startS1Idx = fATilingData->coreInfo.startS1Idx[coreIdx];
                uint32_t startS2Idx = fATilingData->coreInfo.startS2Idx[coreIdx];
                uint32_t endBIdx = fATilingData->coreInfo.endBIdx[coreIdx];
                uint32_t endN1Idx = fATilingData->coreInfo.endN1Idx[coreIdx];
                uint32_t endS1Idx = fATilingData->coreInfo.endS1Idx[coreIdx];
                uint32_t endS2Idx = fATilingData->coreInfo.endS2Idx[coreIdx];
                uint64_t gmOffsetLseFD = fATilingData->coreInfo.firstSplitKVTaskLseOffset[coreIdx];
                uint64_t gmOffsetOFD = fATilingData->coreInfo.firstSplitKVTaskOOffset[coreIdx];

                for (uint32_t BIdx = startBIdx; BIdx <= endBIdx; BIdx++) {
                    uint32_t qSeqlenCur = static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx));
                    uint32_t kvSeqlenCur = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx));
                    if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                        uint32_t prevQSeqlenSum = (BIdx == 0) ?
                            0 : static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx - 1));
                        qSeqlenCur = qSeqlenCur - prevQSeqlenSum;
                        if constexpr (!PAGED_CACHE_FLAG) {
                            uint32_t prevKvSeqlenSum = (BIdx == 0) ?
                                0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx - 1));
                            kvSeqlenCur = kvSeqlenCur - prevKvSeqlenSum;
                        }
                    }

                    uint32_t curQNBlockTileTmp = GetQNBlockTile(qSeqlenCur, groupSize);
                    uint32_t qNBlockNumPerGroupTmp = NpuArch::Detail::Alignment::CeilDiv(groupSize, curQNBlockTileTmp);
                    uint32_t curQNBlockNumTmp = qNBlockNumPerGroupTmp * kvHeads;
                    uint32_t curQSBlockTileTmp = GetQSBlockTile(kvSeqlenCur);
                    uint32_t curQSBlockNumTmp = NpuArch::Detail::Alignment::CeilDiv(qSeqlenCur, curQSBlockTileTmp);
                    uint32_t curKSBlockNumTmp = NpuArch::Detail::Alignment::CeilDiv(kvSeqlenCur,
                        GetKSBlockTile(kvSeqlenCur));

                    int32_t stN1IdxNow = (BIdx == startBIdx) ? startN1Idx : 0;
                    int32_t enN1IdxNow = (BIdx == endBIdx) ? endN1Idx : curQNBlockNumTmp - 1;

                    for (int32_t n1Idx = stN1IdxNow; n1Idx <= enN1IdxNow; n1Idx++) {
                        int32_t stS1IdxNow = (BIdx == startBIdx && n1Idx == stN1IdxNow) ? startS1Idx : 0;
                        int32_t enS1IdxNow = (BIdx == endBIdx && n1Idx == enN1IdxNow) ? endS1Idx : curQSBlockNumTmp - 1;

                        for (int32_t s1Idx = stS1IdxNow; s1Idx <= enS1IdxNow; s1Idx++) {
                            int32_t stS2IdxNow = (BIdx == startBIdx && n1Idx == stN1IdxNow && s1Idx == stS1IdxNow) ?
                                startS2Idx : 0;
                            int32_t enS2IdxNow = (BIdx == endBIdx && n1Idx == enN1IdxNow && s1Idx == enS1IdxNow) ?
                                endS2Idx :
                                curKSBlockNumTmp;

                            bool isSplitKV = (enS2IdxNow - stS2IdxNow) > 0 &&
                                (enS2IdxNow - stS2IdxNow) < static_cast<int32_t>(curKSBlockNumTmp);

                            runMainLoop(
                                coreIdx, BIdx, n1Idx, s1Idx,
                                isSplitKV, stS2IdxNow, enS2IdxNow,
                                gmOffsetLseFD, gmOffsetOFD,
                                globalTensors, pseQ, pseKv);

                            if (isSplitKV) {
                                uint32_t qSBlockSizeTmp = (s1Idx == static_cast<int32_t>(curQSBlockNumTmp - 1U)) ?
                                    (qSeqlenCur - s1Idx * curQSBlockTileTmp) : curQSBlockTileTmp;
                                uint32_t qNBlockIdxCurGroupTmp = n1Idx % qNBlockNumPerGroupTmp;
                                uint32_t qNBlockSizeTmp = (qNBlockIdxCurGroupTmp == (qNBlockNumPerGroupTmp - 1U)) ?
                                    (groupSize - qNBlockIdxCurGroupTmp * curQNBlockTileTmp) : curQNBlockTileTmp;
                                gmOffsetLseFD += qSBlockSizeTmp * qNBlockSizeTmp;
                                gmOffsetOFD += qSBlockSizeTmp * qNBlockSizeTmp * embedV;
                            }
                        }
                    }
                }
            } else {
                for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
                    uint32_t curBatchTmp = 0;
                    uint32_t preTotalTaskNumTmp = 0;
                    uint32_t curTotalTaskNumTmp = firstBatchTaskNum;

                    while (taskIdx >= curTotalTaskNumTmp && curBatchTmp < batch - 1) {
                        ++curBatchTmp;
                        preTotalTaskNumTmp = curTotalTaskNumTmp;

                        uint32_t qSeqlenTmp = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatchTmp));
                        uint32_t kvSeqlenTmp = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatchTmp));
                        if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                            uint32_t prevQSeqlenSumTmp = (curBatchTmp == 0) ?
                                0 : static_cast<uint32_t>(gActualQseqlen.GetValue(curBatchTmp - 1));
                            qSeqlenTmp = qSeqlenTmp - prevQSeqlenSumTmp;
                            if constexpr (!PAGED_CACHE_FLAG) {
                                uint32_t prevKvSeqlenSumTmp = (curBatchTmp == 0) ?
                                    0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatchTmp - 1));
                                kvSeqlenTmp = kvSeqlenTmp - prevKvSeqlenSumTmp;
                            }
                        }

                        uint32_t curQNBlockTileTmp = GetQNBlockTile(qSeqlenTmp, groupSize);
                        uint32_t qNBlockNumPerGroupTmp = NpuArch::Detail::Alignment::CeilDiv(groupSize,
                            curQNBlockTileTmp);
                        uint32_t curQNBlockNumTmp = qNBlockNumPerGroupTmp * kvHeads;
                        uint32_t curQSBlockTileTmp = GetQSBlockTile(kvSeqlenTmp);
                        uint32_t curQSBlockNumTmp = NpuArch::Detail::Alignment::CeilDiv(qSeqlenTmp, curQSBlockTileTmp);
                        curTotalTaskNumTmp += curQNBlockNumTmp * curQSBlockNumTmp;
                    }

                    uint32_t qSeqlenCur = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatchTmp));
                    uint32_t kvSeqlenCur = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatchTmp));
                    if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                        uint32_t prevQSeqlenSumCur = (curBatchTmp == 0) ?
                            0 : static_cast<uint32_t>(gActualQseqlen.GetValue(curBatchTmp - 1));
                        qSeqlenCur = qSeqlenCur - prevQSeqlenSumCur;
                        if constexpr (!PAGED_CACHE_FLAG) {
                            uint32_t prevKvSeqlenSumCur = (curBatchTmp == 0) ?
                                0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatchTmp - 1));
                            kvSeqlenCur = kvSeqlenCur - prevKvSeqlenSumCur;
                        }
                    }

                    uint32_t curQNBlockTileCur = GetQNBlockTile(qSeqlenCur, groupSize);
                    uint32_t qNBlockNumPerGroupCur = NpuArch::Detail::Alignment::CeilDiv(groupSize, curQNBlockTileCur);
                    uint32_t curQNBlockNumCur = qNBlockNumPerGroupCur * kvHeads;

                    uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNumTmp;
                    uint32_t qSBlockIdxCur = taskIdxCurBatch / curQNBlockNumCur;
                    uint32_t qNBlockIdxCur = taskIdxCurBatch - qSBlockIdxCur * curQNBlockNumCur;

                    runMainLoop(
                        coreIdx, curBatchTmp, qNBlockIdxCur, qSBlockIdxCur,
                        false, 0, 0,
                        0, 0,
                        globalTensors, pseQ, pseKv);
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

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
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

            if constexpr (IS_FD) {
            AscendC::SyncAll();
#ifdef __DAV_C220_VEC__
                CombineScale combineScale;
                combineScale.init(resource);
                combineScale(
                    qHeads,
                    fATilingData->totalSplitNodeNum,
                    embedV,
                    &fATilingData->splitInfo,
                    gLseFD,
                    gOFD,
                    gO,
                    gActualQseqlen,
                    true,
                    (LSE_MODE == Epilogue::LseMode::OUT_ONLY),
                    gLse);
#endif
            }
        }

        __aicore__ inline void runMainLoop(
            uint32_t coreIdx,
            uint32_t BIdx,
            uint32_t qNBlockIdx,
            uint32_t qSBlockIdx,
            bool isSplitKV,
            int32_t stS2IdxNow,
            int32_t enS2IdxNow,
            uint64_t gmOffsetLseFD,
            uint64_t gmOffsetOFD,
            GlobalTensorBundle& globalTensors,
            int64_t pseQ,
            int64_t pseKv)
        {
            auto& gQ = globalTensors.gQ;
            auto& gK = globalTensors.gK;
            auto& gV = globalTensors.gV;
            auto& gPseShift = globalTensors.gPseShift;
            auto& gMask = globalTensors.gMask;
            auto& gBlockTable = globalTensors.gBlockTable;
            auto& gActualQseqlen = globalTensors.gActualQseqlen;
            auto& gActualKvseqlen = globalTensors.gActualKvseqlen;
            auto& gO = globalTensors.gO;
            auto& gLse = globalTensors.gLse;
            auto& gLseFD = globalTensors.gLseFD;
            auto& gOFD = globalTensors.gOFD;
            auto& gS = globalTensors.gS;
            auto& gP = globalTensors.gP;
            auto& gOTmp = globalTensors.gOTmp;
            auto& gOUpdate = globalTensors.gOUpdate;
            auto& gSink = globalTensors.gSink;

            uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx));
            uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx));
            uint32_t prevQSeqlenSum = 0;
            uint32_t prevKvSeqlenSum = 0;

            if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                prevQSeqlenSum = (BIdx == 0) ?
                    0 : static_cast<uint32_t>(gActualQseqlen.GetValue(BIdx - 1));
                qSeqlen = qSeqlen - prevQSeqlenSum;
                if constexpr (!PAGED_CACHE_FLAG) {
                    prevKvSeqlenSum = (BIdx == 0) ?
                        0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(BIdx - 1));
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
            }

            uint64_t qBOffset = static_cast<uint64_t>(prevQSeqlenSum) * strideQ;
            uint64_t kBOffset = 0;
            uint64_t vBOffset = 0;
            uint64_t blockBOffset = 0;
            if constexpr (!PAGED_CACHE_FLAG) {
                kBOffset = static_cast<uint64_t>(prevKvSeqlenSum) * strideK;
                vBOffset = static_cast<uint64_t>(prevKvSeqlenSum) * strideV;
            } else {
                blockBOffset = BIdx * static_cast<uint64_t>(maxNumBlocksPerBatch);
            }
            uint64_t oBOffset = static_cast<uint64_t>(prevQSeqlenSum) * strideO;
            uint64_t lseBOffset = static_cast<uint64_t>(prevQSeqlenSum) * qHeads;

            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
            uint32_t qNBlockNumPerGroup = NpuArch::Detail::Alignment::CeilDiv(groupSize, curQNBlockTile);
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
            uint32_t curQSBlockNum = NpuArch::Detail::Alignment::CeilDiv(qSeqlen, curQSBlockTile);
            uint32_t curKSBlockNum = NpuArch::Detail::Alignment::CeilDiv(kvSeqlen, GetKSBlockTile(kvSeqlen));

            uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
            uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
            uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
            uint64_t lseTokenOffset = static_cast<uint64_t>(qSBlockIdx) * curQSBlockTile * qHeads;

            uint64_t gmOffsetQ = qBOffset +
                static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ +
                static_cast<uint64_t>(qNStartIdx * embed);
            uint64_t gmOffsetK = kBOffset + static_cast<uint64_t>(kvNIdx * embed);
            uint64_t gmOffsetV = vBOffset + static_cast<uint64_t>(kvNIdx * embedV);
            uint64_t gmOffsetO = oBOffset +
                static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO +
                static_cast<uint64_t>(qNStartIdx * embedV);
            if constexpr (PAGED_CACHE_FLAG && std::is_same_v<LayoutK, layout::nZ>) {
                gmOffsetK = kBOffset + kvNIdx * embed * pagedBlockSize;
            }
            if constexpr (PAGED_CACHE_FLAG && std::is_same_v<LayoutV, layout::zN>) {
                gmOffsetV = vBOffset + kvNIdx * embedV * pagedBlockSize;
            }
            uint64_t gmOffsetLse = lseBOffset +
                static_cast<uint64_t>(lseTokenOffset + qNStartIdx);
            uint64_t gmOffsetSink = qNStartIdx;

            uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
                (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
            uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U)) ?
                (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;

            int64_t noSkipKvS = static_cast<int64_t>(kvSeqlen);
            uint32_t kvSLoopNumTotal = 0;
            uint32_t startIdx = 0;
            int32_t preTokenStartLen = 0;
            int32_t preTokenEndLen = 0;
            int32_t nextTokenStartLen = 0;
            int32_t nextTokenEndLen = 0;
            bool notPreMask = true;
            bool notNextMask = true;
            int32_t delStartRow = 0;
            int32_t delEndRow = qSeqlen;
            bool startsWithMaskTile = false;
            bool startsWithMaskThenNomaskFlag = false;
            if constexpr (IS_FD) {
                noSkipKvS = kvSeqlen;
                if (maskType != 0U) {
                    int64_t diffS = kvSeqlen - qSeqlen;
                    diffS = (diffS < 0) ? 0 : diffS;
                    noSkipKvS = (qSBlockIdx + 1U) * curQSBlockTile + diffS;
                    noSkipKvS = AscendC::Std::min(static_cast<int64_t>(kvSeqlen), noSkipKvS);
                }
                kvSLoopNumTotal = NpuArch::Detail::Alignment::CeilDiv(static_cast<uint32_t>(noSkipKvS),
                    MAX_KV_STACK_LEN);
            } else {
                if (maskType != 0U && sparseMode != 4U && maskType != 4U) {
                    int64_t diffS = kvSeqlen - qSeqlen;
                    diffS = (diffS < 0) ? 0 : diffS;
                    noSkipKvS = (qSBlockIdx + 1U) * curQSBlockTile + diffS;
                    noSkipKvS = AscendC::Std::min(static_cast<int64_t>(kvSeqlen), noSkipKvS);
                    kvSLoopNumTotal = NpuArch::Detail::Alignment::CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
                } else if (maskType != 0U && sparseMode == 4U) {
                    int32_t leftPointPreToken = kvSeqlen;
                    int32_t leftPointNextToken = 0;
                    if (preToken < 0 && preToken * (-1) >= qSeqlen) {
                        startIdx = kvSeqlen / MAX_KV_STACK_LEN + 1;
                    } else if (preToken != SPARSE_MODE_INT_MAX) {
                        leftPointPreToken = kvSeqlen - qSeqlen - preToken;
                        preTokenStartLen = qSBlockIdx * curQSBlockTile + leftPointPreToken;
                        preTokenEndLen = qSBlockIdx * curQSBlockTile + qSBlockSize + leftPointPreToken;
                        startIdx = AscendC::Std::max(static_cast<int32_t>(0), preTokenStartLen) /
                            static_cast<int32_t>(MAX_KV_STACK_LEN);
                        notPreMask = false;
                    } else {
                        startIdx = 0;
                    }
                    if (nextToken < 0 && nextToken * (-1) >= kvSeqlen) {
                        kvSLoopNumTotal = 0;
                    } else if (nextToken != SPARSE_MODE_INT_MAX) {
                        leftPointNextToken = kvSeqlen - qSeqlen + nextToken;
                        nextTokenStartLen = qSBlockIdx * curQSBlockTile + leftPointNextToken;
                        nextTokenEndLen = qSBlockIdx * curQSBlockTile + qSBlockSize + leftPointNextToken;
                        noSkipKvS = AscendC::Std::min(static_cast<int32_t>(kvSeqlen),
                            NpuArch::Detail::Alignment::RoundUp(nextTokenEndLen,
                                static_cast<int32_t>(MAX_KV_STACK_LEN)));
                        noSkipKvS = noSkipKvS <= 0 ? kvSeqlen : noSkipKvS;
                        kvSLoopNumTotal = NpuArch::Detail::Alignment::CeilDiv(static_cast<uint32_t>(noSkipKvS),
                            MAX_KV_STACK_LEN);
                        notNextMask = false;
                    } else {
                        noSkipKvS = kvSeqlen;
                        kvSLoopNumTotal = NpuArch::Detail::Alignment::CeilDiv(static_cast<uint32_t>(noSkipKvS),
                            MAX_KV_STACK_LEN);
                    }
                    if (preTokenEndLen > static_cast<int32_t>(kvSeqlen) && preToken != SPARSE_MODE_INT_MAX) {
                        delStartRow = kvSeqlen - leftPointPreToken;
                    } else if (nextTokenStartLen < 0 && nextToken != SPARSE_MODE_INT_MAX) {
                        delEndRow = -leftPointNextToken;
                    }
                } else {
                    kvSLoopNumTotal = NpuArch::Detail::Alignment::CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);
                }
            }

            uint32_t kvStart = 0;
            uint32_t kvEnd = kvSLoopNumTotal;
            if constexpr (IS_FD) {
                kvStart = stS2IdxNow;
                kvEnd = (enS2IdxNow == static_cast<int32_t>(curKSBlockNum)) ? kvSLoopNumTotal : enS2IdxNow;
            } else {
                kvStart = startIdx;
                kvEnd = kvSLoopNumTotal;
            }

            uint32_t rowNum = qSBlockSize * qNBlockSize;
            int32_t stackSeqCount = 0;
            uint32_t preKVNum = PRE_LAUNCH;
            uint32_t blockStackNum = (MAX_KV_STACK_LEN - 1 + pagedBlockSize) / pagedBlockSize;
            uint32_t stackSeqTile = MAX_KV_STACK_LEN;
            uint32_t stackSeqTilePad = MAX_KV_STACK_LEN;
            bool isLastStackTile = false;

#ifdef __DAV_C220_VEC__
                if (kvSLoopNumTotal <= 0 || startIdx >= kvSLoopNumTotal) {
                    LayoutO layoutO(qSeqlen, embed * qHeads);
                    LayoutLse layoutLse(totalQTokens, qHeads);
                    epilogueInitOut(gO[gmOffsetO], gLse[gmOffsetLse], layoutO, layoutLse, qSBlockSize, qNBlockSize);
                }
#endif
#ifdef __DAV_C220_CUBE__
                LayoutQ layoutQTemp(rowNum, embed);
                LayoutK layoutKTemp(strideK, stackSeqTile);
                LayoutV layoutVTemp(stackSeqTile, strideV);
                uint32_t kRow = strideK;
                uint32_t kCol = stackSeqTile;
                uint32_t vRow = stackSeqTile;
                uint32_t vCol = strideV;
                if constexpr (PAGED_CACHE_FLAG && std::is_same_v<LayoutK, layout::nZ>) {
                    kRow = blockStackNum * strideK;
                    kCol = pagedBlockSize;
                    layoutKTemp = LayoutK::template MakeLayout<ElementK>(kRow, kCol);
                }
                if constexpr (PAGED_CACHE_FLAG && std::is_same_v<LayoutV, layout::zN>) {
                    vRow = pagedBlockSize;
                    vCol = blockStackNum * strideV;
                    layoutVTemp = LayoutV::template MakeLayout<ElementV>(vRow, vCol);
                }
                blockMmadQK.resetBlockStart(kvStart, pagedBlockSize);
                blockMmadPV.resetBlockStart(kvStart, pagedBlockSize);
                blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);
#endif
                for (uint32_t kvSIdx = kvStart; kvSIdx < kvEnd + preKVNum; kvSIdx++) {
                    if (kvSIdx < kvEnd) {
                        if (kvSIdx + 1 > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - kvSIdx * MAX_KV_STACK_LEN;
                        } else {
                            stackSeqTile = MAX_KV_STACK_LEN;
                        }
                        isLastStackTile = (kvSIdx + 1) >= kvSLoopNumTotal;
                        uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1U);
                        uint64_t gmOffsetS =
                            static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB;
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
                                keyBnStride);
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
                                keyBnStride);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                        LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                        LayoutMask layOutMask(COMP_TRIU_MASK_DIM_LEN, COMP_TRIU_MASK_DIM_LEN);
                        LayoutQ layOutFullMask(pseQ, pseKv);
                        uint64_t gmOffsetP = gmOffsetS;
                        uint32_t kvSStartIdx = kvSIdx * MAX_KV_STACK_LEN;
                        uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
                        if constexpr (MASK_TYPE == FaiKernel::MaskType::MASK_CAUSAL) {
                            uint32_t triUp = noSkipKvS - qSBlockSize;
                            uint32_t triDown = noSkipKvS;
                            bool doTriUMask = triUp < kvSEndIdx - 1;
                            if (doTriUMask) {
                                if constexpr (IS_FD) {
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
                                        qNBlockSize,
                                        curStackTileMod,
                                        qkReady,
                                        triUp,
                                        triDown,
                                        kvSStartIdx,
                                        kvSEndIdx,
                                        isLastStackTile,
                                        isSplitKV);
                                } else {
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
                                        qNBlockSize,
                                        curStackTileMod,
                                        qkReady,
                                        triUp,
                                        triDown,
                                        kvSStartIdx,
                                        kvSEndIdx,
                                        isLastStackTile,
                                        false);
                                }
                            } else {
                                uint32_t noMaskStackSeqNum = (triUp + 1) / MAX_KV_STACK_LEN;
                                Arch::CrossCoreWaitFlag(qkReady);
                                if constexpr (IS_FD) {
                                    int32_t localLastNoMaskStackId = (int32_t)(noMaskStackSeqNum - 1) - kvStart;
                                    epilogueOnlineSoftmax(
                                        gP[gmOffsetP],
                                        gS[gmOffsetS],
                                        gSink[gmOffsetSink],
                                        layOutP,
                                        layOutS,
                                        actualBlockShapeQK,
                                        (stackSeqCount == 0),
                                        (stackSeqCount == localLastNoMaskStackId),
                                        qSBlockSize,
                                        qNBlockSize,
                                        curStackTileMod,
                                        isLastStackTile,
                                        isSplitKV,
                                        false,
                                        startsWithMaskThenNomaskFlag);
                                } else {
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
                                        qNBlockSize,
                                        curStackTileMod,
                                        isLastStackTile,
                                        false,
                                        false,
                                        startsWithMaskThenNomaskFlag);
                                }
                            }
                        } else if constexpr (MASK_TYPE == FaiKernel::MaskType::MASK_SWA) {
                            if constexpr (!IS_FD) {
                                bool doTriUPreMask = (sparseMode != 4 || notPreMask) ? false :
                                (preTokenStartLen >= kvSStartIdx && preTokenStartLen < kvSEndIdx) ||
                                (preTokenEndLen > kvSStartIdx && preTokenEndLen <= kvSEndIdx) ||
                                (preTokenStartLen <= kvSStartIdx && preTokenEndLen >= kvSEndIdx);
                                bool doTriUNextMask = (sparseMode != 4 || notNextMask) ? false :
                                    (nextTokenStartLen >= kvSStartIdx && nextTokenStartLen < kvSEndIdx) ||
                                    (nextTokenEndLen > kvSStartIdx && nextTokenEndLen <= kvSEndIdx) ||
                                    (nextTokenStartLen <= kvSStartIdx && nextTokenEndLen >= kvSEndIdx);
                                bool doTriUMask = (doTriUPreMask || doTriUNextMask);
                                if (doTriUMask) {
                                    startsWithMaskTile = true;
                                    startsWithMaskThenNomaskFlag = true;
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
                                        qNBlockSize,
                                        curStackTileMod,
                                        qkReady,
                                        kvSStartIdx,
                                        doTriUPreMask,
                                        doTriUNextMask,
                                        preTokenStartLen,
                                        preTokenEndLen,
                                        nextTokenStartLen,
                                        nextTokenEndLen,
                                        isLastStackTile);
                                } else {
                                    bool isLastNoMaskStackTile = (nextTokenStartLen >= kvSeqlen) ||
                                        (nextTokenStartLen < 0);
                                    uint32_t kvSeqlenLimit = isLastNoMaskStackTile ? kvSeqlen : nextTokenStartLen;
                                    uint32_t alignedKvSeqlenLimit = isLastNoMaskStackTile ?
                                        NpuArch::Detail::Alignment::RoundUp(kvSeqlenLimit, MAX_KV_STACK_LEN) :
                                        NpuArch::Detail::Alignment::RoundDown(kvSeqlenLimit, MAX_KV_STACK_LEN);
                                    uint32_t noMaskStackSeqNum = (alignedKvSeqlenLimit - kvStart * MAX_KV_STACK_LEN) /
                                        MAX_KV_STACK_LEN;
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
                                        qNBlockSize,
                                        curStackTileMod,
                                        isLastStackTile,
                                        false,
                                        startsWithMaskTile,
                                        startsWithMaskThenNomaskFlag);
                                        startsWithMaskTile = false;
                                }
                            }
                        } else if constexpr (MASK_TYPE == FaiKernel::MaskType::FULL_MASK) {
                            uint32_t rowOffer = qSBlockIdx * curQSBlockTile;
                            if constexpr (!IS_FD) {
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    gSink[gmOffsetSink],
                                    gPseShift,
                                    layOutP,
                                    layOutS,
                                    layOutFullMask,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    qkReady,
                                    rowOffer,
                                    kvSStartIdx,
                                    kvSEndIdx,
                                    qNStartIdx,
                                    BIdx,
                                    qHeads,
                                    pseQ,
                                    pseKv,
                                    isLastStackTile);
                            }
                        } else {
                            Arch::CrossCoreWaitFlag(qkReady);
                            if constexpr (IS_FD) {
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
                                    qNBlockSize,
                                    curStackTileMod,
                                    isLastStackTile,
                                    isSplitKV,
                                    false);
                            } else {
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
                                    qNBlockSize,
                                    curStackTileMod,
                                    isLastStackTile,
                                    false,
                                    false,
                                    startsWithMaskThenNomaskFlag);
                            }
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                    }
                    if (kvSIdx >= kvStart + preKVNum) {
                        uint32_t nowkvSIdx = kvSIdx - preKVNum;
                        if (nowkvSIdx + 1 > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - nowkvSIdx * MAX_KV_STACK_LEN;
                        } else {
                            stackSeqTile = MAX_KV_STACK_LEN;
                        }
                        uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1U);
                        uint64_t gmOffsetOTmp =
                            static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB;
                        GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};
                        LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__
                        LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
                        uint64_t gmOffsetP =
                            static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            static_cast<uint64_t>(curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB;
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
                                valueBnStride);
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
                                valueBnStride);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                        LayoutO layoutO(qSeqlen, embed * qHeads);
                        LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                        LayoutLse layoutLse(totalQTokens, qHeads);
                        uint64_t gmOffsetUpdate = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB;
                        Arch::CrossCoreWaitFlag(pvReady);

                        if constexpr (IS_FD) {
                            LayoutLse layoutgmLse(qSBlockSize, qNBlockSize);
                            LayoutLse layoutgmLo(qSBlockSize, embed * qNBlockSize);
                            typename EpilogueRescaleO::SplitKVParams splitParams;
                            splitParams.isSplitkv = isSplitKV;
                            splitParams.gCombineLse = gLseFD[gmOffsetLseFD];
                            splitParams.gCombineo = gOFD[gmOffsetOFD];
                            splitParams.layoutgmLse = &layoutgmLse;
                            splitParams.layoutgmLo = &layoutgmLo;

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
                                qNBlockSize,
                                (stackSeqCount - PRE_LAUNCH == 0),
                                nowkvSIdx + 1 >= kvEnd,
                                curStackTileMod,
                                delStartRow,
                                delEndRow,
                                qSeqlen,
                                qSBlockIdx,
                                curQNBlockTile,
                                splitParams);
                        } else {
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
                                qNBlockSize,
                                (stackSeqCount - PRE_LAUNCH == 0),
                                nowkvSIdx + 1 >= kvSLoopNumTotal,
                                curStackTileMod,
                                delStartRow,
                                delEndRow,
                                qSeqlen,
                                qSBlockIdx,
                                curQNBlockTile);
                        }
#endif
                    }
                    stackSeqCount++;
                }
        }

    private:
        uint64_t mm1OutSize;
        uint64_t smOnlineOutSize;
        uint64_t mm2OutSize;
        uint32_t batch;
        uint32_t qHeads;
        uint32_t kvHeads;
        uint32_t embed;
        uint32_t embedV;
        uint32_t pagedBlockSize;
        uint32_t maxNumBlocksPerBatch;
        uint32_t firstBatchTaskNum;
        uint32_t totalTaskNum;
        uint32_t blockSize;
        uint32_t maskType;
        float scaleValue;
        uint32_t sparseMode;
        int64_t preToken;
        int64_t nextToken;
        int64_t pseQ;
        int64_t pseKv;
        uint32_t totalQTokens;

        uint64_t strideQ;
        uint64_t strideO;
        uint64_t strideK;
        uint64_t strideV;
        uint64_t keyBnStride;
        uint64_t valueBnStride;
        uint32_t embedRound;
        uint32_t embedRoundV;
        uint32_t groupSize;

        Arch::Resource<ArchTag> resource;
        Arch::CrossCoreFlag qkReady{QK_READY_ID};
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        Arch::CrossCoreFlag pvReady{PV_READY_ID};

        BlockMmadQK blockMmadQK;
        BlockMmadPV blockMmadPV;
        EpilogueRescaleO epilogueRescaleO;
        EpilogueInitOut epilogueInitOut;
        EpilogueOnlineSoftmax epilogueOnlineSoftmax;
    };
}
#endif
