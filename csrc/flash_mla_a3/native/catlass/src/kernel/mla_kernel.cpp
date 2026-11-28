/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

// imported locally to reflect changes in epilogue
#include "epilogue/block/block_epilogue.hpp"

#include "kernel_common.hpp"

using namespace Catlass;

/*
This example demonstrates how to compute mla.
*/
template <
    class BlockMmadQK,
    class BlockMmadPV,
    class EpilogueMLASoftmax,
    class EpilogueMLARescaleO,
    class EpilogueMLAFDRescaleO,
    bool GraphDecode = false, bool Balanced = false>
class MLAKernel {
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

    using ElementMask = half;

    using ElementO = typename EpilogueMLARescaleO::ElementOutput;
    using LayoutO = typename EpilogueMLARescaleO::LayoutOutput;

    using ElementOTmp = typename EpilogueMLARescaleO::ElementInput;
    using LayoutOTmp = typename EpilogueMLARescaleO::LayoutInput;

    using ElementUpdate = typename EpilogueMLARescaleO::ElementUpdate;
    using LayoutUpdate = typename EpilogueMLARescaleO::LayoutUpdate;

    static constexpr uint32_t KV_SPLIT_MAX = EpilogueMLAFDRescaleO::KV_SPLIT_MAX;
    static constexpr uint32_t HEADS_PROCESS_MAX = EpilogueMLAFDRescaleO::HEADS_PROCESS_MAX;
    static constexpr uint32_t COMPUTE_ELE_NUM = EpilogueMLAFDRescaleO::COMPUTE_ELE_NUM;

    static constexpr uint32_t TILING_OFFSET_KVSEQLEN = 1;
    static constexpr uint32_t TILING_OFFSET_QADDRHIGH = 4;
    static constexpr uint32_t TILING_OFFSET_QADDRLOW = 5;
    static constexpr uint32_t TILING_OFFSET_QROPEADDRHIGH = 6;
    static constexpr uint32_t TILING_OFFSET_QROPEADDRLOW = 7;
    static constexpr uint32_t TILING_OFFSET_OADDRHIGH = 4;
    static constexpr uint32_t TILING_OFFSET_OADDRLOW = 5;
    static constexpr uint32_t TILING_OFFSET_LADDRHIGH = 11;
    static constexpr uint32_t TILING_OFFSET_LADDRLOW = 12;
    static constexpr uint32_t TILING_OFFSET_OFDADDRHIGH = 13;
    static constexpr uint32_t TILING_OFFSET_OFDADDRLOW = 14;

    static constexpr uint32_t ADDR_HALF_WIDTH = 32;
    static constexpr uint32_t MOD_2 = 2;

    CATLASS_DEVICE static uint32_t ActiveSplits(uint32_t kvLength, uint32_t splitSpan, bool causalMtp)
    {
        uint32_t count = CeilDiv(kvLength, splitSpan);
        // A causal row must have an eligible key in the first tile of any
        // active split. Keep a 1-3 key suffix in the preceding split.
        if (causalMtp && count > 1 && kvLength - (count - 1) * splitSpan < 4) {
            --count;
        }
        return count;
    }


    struct CorePages {
        uint32_t begin, end, firstBatch;
    };

    // One contiguous interval per Cube core. The scheduler counts a short
    // masked Q4 tail together with its preceding page as one work unit.
    CATLASS_DEVICE static CorePages CoreInterval(AscendC::GlobalTensor<uint32_t> &records,
                                                 uint32_t core, uint32_t batches)
    {
        const uint32_t total = records.GetValue(0);
        const uint32_t active = records.GetValue(1);
        if (core >= active) return {0, 0, batches};
        const uint32_t begin = static_cast<uint64_t>(core) * total / active;
        const uint32_t end = static_cast<uint64_t>(core + 1) * total / active;
        uint32_t lo = 0, hi = batches;
        while (lo < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;
            const uint32_t entry = 4 * (1 + mid);
            if (records.GetValue(entry) + records.GetValue(entry + 1) <= begin) lo = mid + 1;
            else hi = mid;
        }
        return {begin, end, lo};
    }

    /// Parameters structure
    struct Params {
        // Data members
        GM_ADDR q;
        GM_ADDR qRope;
        GM_ADDR k;
        GM_ADDR kRope;
        GM_ADDR blockTables;
        GM_ADDR o;
        GM_ADDR s;
        GM_ADDR p;
        GM_ADDR oTmp;
        GM_ADDR oUpdate;
        GM_ADDR oCoreTmp;
        GM_ADDR l;
        GM_ADDR tiling;
        GM_ADDR lengths;
        GM_ADDR schedule;

        // Methods
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(GM_ADDR q_, GM_ADDR qRope_, GM_ADDR k_, GM_ADDR kRope_, GM_ADDR blockTables_,
               GM_ADDR o_, GM_ADDR s_, GM_ADDR p_, GM_ADDR oTmp_, GM_ADDR oUpdate_,
               GM_ADDR oCoreTmp_, GM_ADDR l_, GM_ADDR tiling_, GM_ADDR lengths_ = nullptr, GM_ADDR schedule_ = nullptr)
            : q(q_), qRope(qRope_), k(k_), kRope(kRope_), blockTables(blockTables_), o(o_),
              s(s_), p(p_), oTmp(oTmp_), oUpdate(oUpdate_), oCoreTmp(oCoreTmp_), l(l_), tiling(tiling_), lengths(lengths_), schedule(schedule_) {}
    };

    // Methods
    CATLASS_DEVICE
    MLAKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params)
    {
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
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);

        // Get the memory offset address of the input on Global Memory
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);

        AscendC::GlobalTensor<ElementQ> gQRope;
        gQRope.SetGlobalBuffer((__gm__ ElementQ *)params.qRope);

        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);

        AscendC::GlobalTensor<ElementK> gKRope;
        gKRope.SetGlobalBuffer((__gm__ ElementK *)params.kRope);

        AscendC::GlobalTensor<int32_t> gblockTable;
        gblockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));

        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTmp);
        AscendC::GlobalTensor<uint32_t> gTiling;
        gTiling.SetGlobalBuffer((__gm__ uint32_t *)params.tiling);

        AscendC::GlobalTensor<int32_t> gLengths;
        if constexpr (GraphDecode) {
            gLengths.SetGlobalBuffer((__gm__ int32_t*)params.lengths);
        }
        // Get tiling parameters
        uint32_t batch = gTiling.GetValue(TILING_BATCH);
        uint32_t qHeads = gTiling.GetValue(TILING_NUMHEADS);
        uint32_t embed = gTiling.GetValue(TILING_HEADDIM);
        uint32_t embedRope = gTiling.GetValue(TILING_HEADDIM_ROPE);
        uint32_t blockSize = gTiling.GetValue(TILING_BLOCKSIZE);
        uint32_t maxNumBlocksPerQuery = gTiling.GetValue(TILING_MAXBLOCKS);
        uint32_t kvHeads = gTiling.GetValue(TILING_KVHEADS);
        uint32_t tilingHeadSize = gTiling.GetValue(TILING_HEADSIZE);
        uint32_t tilingParaSize = gTiling.GetValue(TILING_PARASIZE);
        const bool directRows = tilingHeadSize == 32 && gTiling.GetValue(23) == 1;
        uint32_t curQheadSplitSize = gTiling.GetValue(TILING_HEAD_SPLIT_SIZE);
        uint32_t curQheadSplitNum = gTiling.GetValue(TILING_HEAD_SPLIT_NUM);
        const uint32_t configuredSpan = gTiling.GetValue(TILING_KVSPLIT);
        uint32_t kvSplitCoreNum = gTiling.GetValue(TILING_KVCORENUM);

        bool causalMtp = gTiling.GetValue(TILING_MASKTYPE) == 3 && (qHeads == 16 || qHeads == 32);
        uint32_t packedStride = gTiling.GetValue(TILING_INPUT_STRIDE);
        // Reserve the extra L1 slot once. A short split can use the ordinary two-slot
        // pipeline even when another request in the same graph benefits from prefetch.
        const bool reservePrefetch = GraphDecode || (packedStride == 576 && qHeads <= 16 &&
            gTiling.GetValue(TILING_MAX_QSEQLEN) == 1 && gTiling.GetValue(TILING_MAX_KVSEQLEN) >= 4096);
        const uint32_t queryRows = qHeads > 16 ? 32 : 16;
        BlockMmadQK blockMmadQK(resource, reservePrefetch, queryRows);
        BlockMmadPV blockMmadPV(resource, reservePrefetch, queryRows);
        uint32_t queryStride = packedStride == 0 ? embed : packedStride;
        uint32_t queryRopeStride = packedStride == 0 ? embedRope : packedStride;
        uint32_t strideKV = kvHeads * queryStride;
        uint32_t strideKVRope = kvHeads * queryRopeStride;
        // These optional fields belong to the prepared 32-word split header.
        // The legacy 24-word and fused headers keep their dense page layout.
        const bool splitHeader = packedStride == 0 && tilingHeadSize == 32;
        const uint32_t latentPageStride = splitHeader ? gTiling.GetValue(24) : 0;
        const uint32_t ropePageStride = splitHeader ? gTiling.GetValue(25) : 0;
        const uint32_t pageStrideKV = latentPageStride == 0 ? blockSize * strideKV : latentPageStride;
        const uint32_t pageStrideKVRope = ropePageStride == 0 ? blockSize * strideKVRope : ropePageStride;
        uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        AscendC::GlobalTensor<uint32_t> schedule;
        CorePages corePages{};
        if constexpr (Balanced) {
            schedule.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(params.schedule));
            corePages = CoreInterval(schedule, coreIdx, batch);
        }
        uint32_t processNum = Balanced ? batch : batch * curQheadSplitNum * kvSplitCoreNum;

        // Go through each task
        for (uint32_t process = Balanced ? corePages.firstBatch : coreIdx;
             process < processNum; process += Balanced ? 1 : coreNum) {
            // Get the offset of each core on the GM
            uint32_t curBatch = Balanced ? process : process / (curQheadSplitNum * kvSplitCoreNum);
            const uint32_t rec = (1 + curBatch) * 4;
            if constexpr (Balanced) {
                if (schedule.GetValue(rec) >= corePages.end) break;
                if (schedule.GetValue(rec + 1) == 0) continue;
            }
            uint32_t offsetTiling = tilingHeadSize + tilingParaSize * curBatch;
            uint32_t qSeqlen = gTiling.GetValue(offsetTiling);
            uint32_t kvSeqlen = GraphDecode ? static_cast<uint32_t>(max(0, gLengths.GetValue(curBatch)))
                                           : gTiling.GetValue(offsetTiling + TILING_OFFSET_KVSEQLEN);
            if constexpr (GraphDecode) {
                kvSeqlen = min(kvSeqlen, gTiling.GetValue(TILING_MAX_KVSEQLEN));
            }
            const uint32_t kvSplitPerCore = GraphDecode ? max(1u, CeilDiv(CeilDiv(kvSeqlen, blockSize), kvSplitCoreNum)) * blockSize : configuredSpan;
            uint32_t qAddrHigh = gTiling.GetValue(offsetTiling + TILING_OFFSET_QADDRHIGH);
            uint32_t qAddrLow = gTiling.GetValue(offsetTiling + TILING_OFFSET_QADDRLOW);
            uint64_t qAddr = (uint64_t)(((uint64_t)qAddrHigh) << ADDR_HALF_WIDTH | qAddrLow);
            uint32_t qRopeAddrHigh = gTiling.GetValue(offsetTiling + TILING_OFFSET_QROPEADDRHIGH);
            uint32_t qRopeAddrLow = gTiling.GetValue(offsetTiling + TILING_OFFSET_QROPEADDRLOW);
            uint64_t qRopeAddr = (uint64_t)(((uint64_t)qRopeAddrHigh) << ADDR_HALF_WIDTH | qRopeAddrLow);

            if (kvSeqlen == 0) {
                continue;
            }
            uint32_t tableBatch = directRows ? gTiling.GetValue(offsetTiling + 17) : curBatch;
            uint32_t qHeadSplitIdx = Balanced ? 0 : (process % (curQheadSplitNum * kvSplitCoreNum)) / kvSplitCoreNum;
            uint32_t qHeadSplitSizeActual = (qHeadSplitIdx ==
                                             (curQheadSplitNum - 1))
                                                ? (qHeads - qHeadSplitIdx * curQheadSplitSize)
                                                : curQheadSplitSize;
            uint32_t curStartHeadIdx = qHeadSplitIdx * curQheadSplitSize;
            uint64_t gQOffset = (qAddr / embed + curStartHeadIdx) * queryStride;
            uint64_t gQRopeOffset = (qRopeAddr / embedRope + curStartHeadIdx) * queryRopeStride;

            uint32_t kvSeqlenAlign = RoundUp(kvSeqlen, blockSize);
            uint32_t curNIdx, startKV, curKVSeqlen, splitStride;
            if constexpr (Balanced) {
                const uint32_t first = schedule.GetValue(rec);
                const uint32_t units = schedule.GetValue(rec + 1);
                const uint32_t localBegin = max(first, corePages.begin) - first;
                const uint32_t localEnd = min(first + units, corePages.end) - first;
                startKV = localBegin * blockSize;
                curKVSeqlen = (localEnd == units ? kvSeqlen : localEnd * blockSize) - startKV;
                curNIdx = coreIdx - schedule.GetValue(rec + 2);
                splitStride = schedule.GetValue(rec + 3);
            } else {
                curNIdx = process % kvSplitCoreNum;
                const uint32_t kvLoop = ActiveSplits(kvSeqlen, kvSplitPerCore, causalMtp);
                if (curNIdx >= kvLoop) continue;
                startKV = curNIdx * kvSplitPerCore;
                curKVSeqlen = curNIdx + 1 == kvLoop ? kvSeqlen - startKV : kvSplitPerCore;
                splitStride = kvSplitCoreNum;
            }

            uint32_t tokenNumPerHead = qSeqlen;
            uint32_t seqTile = blockSize;
            uint32_t nLoop = (curKVSeqlen + seqTile - 1) / seqTile;
            // Wider tiles spend more time on Cube math before the next page is needed.
            const bool prefetchKV = reservePrefetch && (!GraphDecode || nLoop >= max(3u, qHeads / 2));
            uint32_t kSeqTile = seqTile;
            uint32_t kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
            uint32_t vSeqTile = seqTile;
            uint32_t vSeqTileRound = RoundUp<BLOCK_SIZE>(vSeqTile);

            uint32_t rowNum = qHeadSplitSizeActual * tokenNumPerHead;
            uint32_t rowNumRound = RoundUp<BLOCK_SIZE>(rowNum);

            uint32_t pageTableStart = tableBatch * maxNumBlocksPerQuery + startKV / blockSize;
            if (prefetchKV) {
                uint32_t page = gblockTable.GetValue(pageTableStart);
                blockMmadQK.PrefetchKV(gK[static_cast<uint64_t>(page) * pageStrideKV], gKRope[static_cast<uint64_t>(page) * pageStrideKVRope], packedStride == 0, 0,
                                       min(curKVSeqlen, seqTile));
            }
            for (uint32_t nIdx = 0; nIdx < nLoop + 1; ++nIdx) {
                if (nIdx != nLoop) {
                    if (prefetchKV && nIdx + 1 < nLoop) {
                        uint32_t page = gblockTable.GetValue(pageTableStart + nIdx + 1);
                        uint32_t rows = min(curKVSeqlen - (nIdx + 1) * seqTile, seqTile);
                        blockMmadQK.PrefetchKV(gK[static_cast<uint64_t>(page) * pageStrideKV], gKRope[static_cast<uint64_t>(page) * pageStrideKVRope], packedStride == 0, nIdx + 1, rows);
                    }
                    if (nIdx == (nLoop - 1)) {
                        kSeqTile = (curKVSeqlen - nIdx * seqTile);
                        kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
                    }
                    LayoutQ layoutQ(rowNum, embed, queryStride);
                    LayoutQ layoutQRope(rowNum, embedRope, queryRopeStride);
                    LayoutK layoutK(embed, kSeqTile, strideKV);
                    LayoutK layoutKRope(embedRope, kSeqTile, strideKVRope);
                    LayoutS layoutS(rowNumRound, kSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embed + embedRope};
                    MatrixCoord qShapeSingleNd{qHeadSplitSizeActual, embed};
                    uint32_t qkPingPongFlag = nIdx % MOD_2;
                    uint64_t kvOffset = 0;
                    uint64_t kvOffsetRope = 0;
                    if (!prefetchKV) {
                        uint32_t page = gblockTable.GetValue(pageTableStart + nIdx);
                        kvOffset = static_cast<uint64_t>(page) * pageStrideKV;
                        kvOffsetRope = static_cast<uint64_t>(page) * pageStrideKVRope;
                    }
                    uint64_t gSOffset =
                        (uint64_t)coreIdx * TMP_SIZE_DECODER + (uint64_t)qkPingPongFlag * TMP_SIZE_DECODER / 2;
                    // Calculate a Q * K^T in advance
                    blockMmadQK(
                        gQ[gQOffset],
                        gQRope[gQRopeOffset],
                        gK[kvOffset],
                        gKRope[kvOffsetRope],
                        gS[gSOffset],
                        layoutQ, layoutQRope, layoutK, layoutKRope, layoutS,
                        actualBlockShapeQK, qShapeSingleNd,
                        qHeads, nIdx, packedStride == 576, prefetchKV);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                }
                // Because a Q * K^T is calculated in advance, the first round is skipped.
                if (nIdx != 0) {
                    if (nIdx == nLoop) {
                        vSeqTile = (curKVSeqlen - (nIdx - 1) * seqTile);
                        vSeqTileRound = RoundUp<BLOCK_SIZE>(vSeqTile);
                    }
                    LayoutP layoutP(rowNum, vSeqTile, vSeqTileRound);
                    LayoutV layoutV(embed, vSeqTile);
                    LayoutOTmp layoutOTmp(rowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed, vSeqTile};
                    uint32_t pvPingPongFlag = (nIdx - 1) % MOD_2;
                    uint64_t gPOffset = (uint64_t)coreIdx * TMP_SIZE + (uint64_t)pvPingPongFlag * TMP_SIZE / 2;
                    uint64_t gOTmpOffset = (uint64_t)coreIdx * TMP_SIZE * 2 + (uint64_t)pvPingPongFlag * TMP_SIZE;
                    // Calculate P * V
                    blockMmadPV(
                        gP[gPOffset],
                        gOTmp[gOTmpOffset],
                        layoutP, layoutV, layoutOTmp,
                        actualBlockShapePV, nIdx, softmaxReady, prefetchKV);
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                }
            }
        }

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

        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID5);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);;

        // Get the memory offset address of the input on Global Memory
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.s);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)params.p);
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oTmp);
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)params.oUpdate);
        AscendC::GlobalTensor<ElementOTmp> gOCoreTmp;
        gOCoreTmp.SetGlobalBuffer((__gm__ ElementOTmp *)params.oCoreTmp);
        AscendC::GlobalTensor<ElementOTmp> gl;
        gl.SetGlobalBuffer((__gm__ ElementOTmp *)params.l);
        AscendC::GlobalTensor<uint32_t> gTiling;
        gTiling.SetGlobalBuffer((__gm__ uint32_t *)params.tiling);
        AscendC::GlobalTensor<float> gTilingFp64;
        gTilingFp64.SetGlobalBuffer((__gm__ float *)params.tiling);

        AscendC::GlobalTensor<int32_t> gLengths;
        if constexpr (GraphDecode) {
            gLengths.SetGlobalBuffer((__gm__ int32_t*)params.lengths);
        }
        // Get tiling parameters
        uint32_t batch = gTiling.GetValue(TILING_BATCH);
        uint32_t qHeads = gTiling.GetValue(TILING_NUMHEADS);
        uint32_t embed = gTiling.GetValue(TILING_HEADDIM);
        uint32_t blockSize = gTiling.GetValue(TILING_BLOCKSIZE);
        float tor = gTilingFp64.GetValue(TILING_TOR);
        uint32_t kvHeads = gTiling.GetValue(TILING_KVHEADS);
        uint32_t tilingHeadSize = gTiling.GetValue(TILING_HEADSIZE);
        uint32_t tilingParaSize = gTiling.GetValue(TILING_PARASIZE);
        const bool directRows = tilingHeadSize == 32 && gTiling.GetValue(23) == 1;
        uint32_t curQheadSplitSize = gTiling.GetValue(TILING_HEAD_SPLIT_SIZE);
        uint32_t curQheadSplitNum = gTiling.GetValue(TILING_HEAD_SPLIT_NUM);
        const uint32_t configuredSpan = gTiling.GetValue(TILING_KVSPLIT);
        uint32_t kvSplitCoreNum = gTiling.GetValue(TILING_KVCORENUM);

        uint32_t strideQO = qHeads * embed;
        uint32_t embedRound = RoundUp<BLOCK_SIZE>(embed);
        uint32_t glFlag = 1;

        EpilogueMLASoftmax epilogueMLASoftmax(resource, tor, kvSplitCoreNum);
        // Ling packs one or four query tokens with four or eight local heads.
        const uint32_t headsPerToken = (qHeads == 8 || qHeads == 32) ? 8 : 4;
        bool causalMtp = gTiling.GetValue(TILING_MASKTYPE) == 3 && qHeads == 4 * headsPerToken;
        EpilogueMLARescaleO epilogueMLARescaleO(resource, kvSplitCoreNum);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();
        AscendC::GlobalTensor<uint32_t> schedule;
        CorePages corePages{};
        if constexpr (Balanced) {
            schedule.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(params.schedule));
            corePages = CoreInterval(schedule, coreIdx, batch);
        }
        uint32_t processNum = Balanced ? batch : batch * curQheadSplitNum * kvSplitCoreNum;
        // Go through each task.
        for (uint32_t process = Balanced ? corePages.firstBatch : coreIdx;
             process < processNum; process += Balanced ? 1 : coreNum) {
            // Get the offset of each core on the GM.
            uint32_t curBatch = Balanced ? process : process / (curQheadSplitNum * kvSplitCoreNum);
            const uint32_t rec = (1 + curBatch) * 4;
            if constexpr (Balanced) {
                if (schedule.GetValue(rec) >= corePages.end) break;
                if (schedule.GetValue(rec + 1) == 0) continue;
            }
            uint32_t offsetTiling = tilingHeadSize + tilingParaSize * curBatch;
            uint32_t qSeqlen = gTiling.GetValue(offsetTiling);
            uint32_t kvSeqlen = GraphDecode ? static_cast<uint32_t>(max(0, gLengths.GetValue(curBatch)))
                                           : gTiling.GetValue(offsetTiling + TILING_OFFSET_KVSEQLEN);
            if constexpr (GraphDecode) {
                kvSeqlen = min(kvSeqlen, gTiling.GetValue(TILING_MAX_KVSEQLEN));
            }
            const uint32_t kvSplitPerCore = GraphDecode ? max(1u, CeilDiv(CeilDiv(kvSeqlen, blockSize), kvSplitCoreNum)) * blockSize : configuredSpan;
            uint32_t oAddrHigh32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_OADDRHIGH);
            uint32_t oAddrLow32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_OADDRLOW);
            uint64_t oAddr = (uint64_t)(((uint64_t)oAddrHigh32) << ADDR_HALF_WIDTH | oAddrLow32);
            if (kvSeqlen == 0) {
                continue;
            }
            uint32_t qHeadSplitIdx = Balanced ? 0 : (process % (curQheadSplitNum * kvSplitCoreNum)) / kvSplitCoreNum;
            uint32_t qHeadSplitSizeActual = (qHeadSplitIdx == (curQheadSplitNum - 1))
                                                ? (qHeads - qHeadSplitIdx * curQheadSplitSize)
                                                : curQheadSplitSize;
            uint32_t curStartHeadIdx = qHeadSplitIdx * curQheadSplitSize;
            uint64_t gmOffsetO = oAddr + curStartHeadIdx * embed;

            uint32_t kvSeqlenAlign = RoundUp(kvSeqlen, blockSize);
            uint32_t curNIdx, startKV, curKVSeqlen, splitStride;
            if constexpr (Balanced) {
                const uint32_t first = schedule.GetValue(rec);
                const uint32_t units = schedule.GetValue(rec + 1);
                const uint32_t localBegin = max(first, corePages.begin) - first;
                const uint32_t localEnd = min(first + units, corePages.end) - first;
                startKV = localBegin * blockSize;
                curKVSeqlen = (localEnd == units ? kvSeqlen : localEnd * blockSize) - startKV;
                curNIdx = coreIdx - schedule.GetValue(rec + 2);
                splitStride = schedule.GetValue(rec + 3);
            } else {
                curNIdx = process % kvSplitCoreNum;
                const uint32_t kvLoop = ActiveSplits(kvSeqlen, kvSplitPerCore, causalMtp);
                if (curNIdx >= kvLoop) continue;
                startKV = curNIdx * kvSplitPerCore;
                curKVSeqlen = curNIdx + 1 == kvLoop ? kvSeqlen - startKV : kvSplitPerCore;
                splitStride = kvSplitCoreNum;
            }

            uint32_t seqTile = blockSize;
            uint32_t tokenNumPerHead = qSeqlen;
            uint32_t nLoop = (curKVSeqlen + seqTile - 1) / seqTile;

            uint32_t kSeqTile = seqTile;
            uint32_t kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
            uint32_t vSeqTile = seqTile;
            uint32_t vSeqTileRound = RoundUp<BLOCK_SIZE>(vSeqTile);

            uint32_t rowNum = tokenNumPerHead * qHeadSplitSizeActual;
            uint32_t rowNumRound = RoundUp<BLOCK_SIZE>(rowNum);

            uint32_t oFdOffset = 0;
            uint32_t lOffset = 0;
            uint32_t lHeadOffset = (qHeads == 64) ? qHeads * process : 0;

            uint32_t lAddrHigh32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_LADDRHIGH);
            uint32_t lAddrLow32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_LADDRLOW);
            uint64_t lAddr = (uint64_t)(((uint64_t)lAddrHigh32) << ADDR_HALF_WIDTH | lAddrLow32);
            uint32_t oFdAddrHigh32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_OFDADDRHIGH);
            uint32_t oFdAddrLow32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_OFDADDRLOW);
            uint64_t FdAddr = (uint64_t)(((uint64_t)oFdAddrHigh32) << ADDR_HALF_WIDTH | oFdAddrLow32);
            uint32_t headIdx = curStartHeadIdx + AscendC::GetSubBlockIdx() * qHeadSplitSizeActual / 2;
            if constexpr (Balanced) {
                const uint32_t slot = curBatch + schedule.GetValue(rec + 2);
                const uint32_t partialRow = slot * qHeads + headIdx * splitStride + curNIdx;
                oFdOffset = partialRow * embed;
                lOffset = partialRow;
                const bool finishDirect = schedule.GetValue(3) == 1 && splitStride == 1 && (!causalMtp || kvSeqlen >= 4);
                if (finishDirect) gmOffsetO = 0;
                epilogueMLARescaleO.SetBalancedOutput(splitStride, finishDirect, curBatch * qHeads,
                                                      gTiling.GetValue(22), headsPerToken);
            } else {
                oFdOffset = FdAddr * kvSplitCoreNum + headIdx * embed * kvSplitCoreNum + curNIdx * embed;
                lOffset = lAddr + headIdx * kvSplitCoreNum + curNIdx + lHeadOffset;
            }

            uint64_t gmOffsetP = 0;
            uint64_t gmOffsetS = 0;
            uint64_t gmOffsetMask = 0;
            int32_t firstBlockedInSequence = static_cast<int32_t>(kvSeqlen) - 3;
            if (directRows) {
                firstBlockedInSequence += static_cast<int32_t>(gTiling.GetValue(offsetTiling + 18));
            }
            // Split k seqlen
            for (uint32_t nIdx = 0; nIdx < nLoop + 1; nIdx++) {
                if (nIdx != nLoop) {
                    if (nIdx == (nLoop - 1)) {
                        // Calculate the size of the tail block
                        kSeqTile = (curKVSeqlen - nIdx * seqTile);
                        kSeqTileRound = RoundUp<BLOCK_SIZE>(kSeqTile);
                    }
                    // Wait for Q * K calculation to complete
                    Arch::CrossCoreWaitFlag(qkReady);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);

                    LayoutP layoutP(rowNum, kSeqTile, kSeqTileRound);
                    LayoutS layoutS(rowNumRound, kSeqTile, kSeqTileRound);
                    GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embedRound};
                    uint32_t softmaxPingPongFlag = nIdx % MOD_2;
                    uint64_t gmOffsetP = (uint64_t)coreIdx * TMP_SIZE + softmaxPingPongFlag * TMP_SIZE / 2;
                    uint64_t gmOffsetS =
                        (uint64_t)coreIdx * TMP_SIZE_DECODER + softmaxPingPongFlag * TMP_SIZE_DECODER / 2;
                    // Softmax one-stage calculation
                    // In this standalone probe, 16 flattened Q rows represent
                    // four query positions with four local heads each.
                    int32_t firstBlocked = firstBlockedInSequence -
                        static_cast<int32_t>(startKV + nIdx * seqTile);
                    epilogueMLASoftmax(
                        gP[gmOffsetP], gS[gmOffsetS],
                        layoutP, layoutS,
                        actualBlockShapeQK,
                        nIdx, qHeadSplitSizeActual, softmaxPingPongFlag, glFlag,
                        causalMtp ? headsPerToken : 0, firstBlocked, curStartHeadIdx);
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
                }

                if (nIdx != 0) {
                    // Calculate the size of the tail block
                    if (nIdx == nLoop) {
                        vSeqTile = (curKVSeqlen - (nIdx - 1) * seqTile);
                        vSeqTileRound = RoundUp<BLOCK_SIZE>(vSeqTile);
                    }
                    // Wait for P * V calculation to complete
                    Arch::CrossCoreWaitFlag(pvReady);

                    LayoutO layoutO(tokenNumPerHead, strideQO);
                    LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                    LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed, vSeqTile};
                    uint32_t rescaleOPingPongFlag = (nIdx - 1) % MOD_2;
                    uint64_t gmOffsetOTmp = (uint64_t)(coreIdx * TMP_SIZE * 2 + rescaleOPingPongFlag * TMP_SIZE);
                    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * TMP_SIZE);
                    uint32_t isLastNTile = (nIdx == nLoop) ? 1 : 0;
                    // Softmax two-stage update
                    epilogueMLARescaleO(
                        gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gO[gmOffsetO],
                        gOCoreTmp[oFdOffset], gl[lOffset],
                        layoutOTmp, layoutO, layoutUpdate,
                        actualBlockShapePV,
                        nIdx, isLastNTile, qHeadSplitSizeActual, rescaleOPingPongFlag, glFlag);
                }
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

        // flash decoding
        if constexpr (Balanced) {
            if (schedule.GetValue(3) == 1 && schedule.GetValue(2) == 0) return;
        }
        if (kvSplitCoreNum != 1) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();

            AscendC::SetAtomicNone();
            AscendC::SetMaskNorm();
            AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

            EpilogueMLAFDRescaleO epilogueMLAFDRescaleO(resource, kvSplitCoreNum);

            uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
            uint32_t aivId = AscendC::GetBlockIdx();

            uint32_t headsProcess = (COMPUTE_ELE_NUM / embed) > HEADS_PROCESS_MAX
                                    ? HEADS_PROCESS_MAX
                                    : (COMPUTE_ELE_NUM / embed);
            uint32_t loopsPerBatch = (qHeads + headsProcess - 1) / headsProcess;
            uint32_t loopsTotal = batch * loopsPerBatch;

            for (uint32_t loopIdx = aivId; loopIdx < loopsTotal; loopIdx += aivNum) {
                uint32_t batchIdx = loopIdx / loopsPerBatch;
                uint32_t loopIdxInBatch = loopIdx % loopsPerBatch;

                uint32_t offsetTiling = tilingHeadSize + tilingParaSize * batchIdx;
                uint32_t kvSeqlen = GraphDecode ? static_cast<uint32_t>(max(0, gLengths.GetValue(batchIdx)))
                                               : gTiling.GetValue(offsetTiling + 1);
                if constexpr (GraphDecode) {
                    kvSeqlen = min(kvSeqlen, gTiling.GetValue(TILING_MAX_KVSEQLEN));
                }
                const uint32_t kvSplitPerCore = GraphDecode ? max(1u, CeilDiv(CeilDiv(kvSeqlen, blockSize), kvSplitCoreNum)) * blockSize : configuredSpan;

                if (!GraphDecode && kvSeqlen == 0) {
                    continue;
                }

                uint32_t oAddrHigh32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_QADDRHIGH);
                uint32_t oAddrLow32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_QADDRLOW);
                uint64_t oAddr = (uint64_t)(((uint64_t)oAddrHigh32) << ADDR_HALF_WIDTH | oAddrLow32);
                uint32_t lAddrHigh32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_LADDRHIGH);
                uint32_t lAddrLow32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_LADDRLOW);
                uint64_t lOffset = (uint64_t)(((uint64_t)lAddrHigh32) << ADDR_HALF_WIDTH | lAddrLow32);
                uint32_t oFdAddrHigh32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_OFDADDRHIGH);
                uint32_t oFdAddrLow32 = gTiling.GetValue(offsetTiling + TILING_OFFSET_OFDADDRLOW);
                uint64_t oFdOffset = (uint64_t)(((uint64_t)oFdAddrHigh32) << ADDR_HALF_WIDTH | oFdAddrLow32);

                uint32_t actualHeads = headsProcess;
                if (loopIdxInBatch == loopsPerBatch - 1) {
                    actualHeads = qHeads - loopIdxInBatch * headsProcess;
                }

                // Only the padded scheduler plan can request direct NTD output.
                const uint32_t outputTokens = tilingHeadSize == 32 ? gTiling.GetValue(22) : 0;
                const uint32_t headBase = loopIdxInBatch * headsProcess;
                const uint32_t firstRow = directRows ? static_cast<uint32_t>(oAddr / embed) : batchIdx * qHeads;
                const uint32_t validRowBegin = directRows ? gTiling.GetValue(offsetTiling + 19) : 0;
                const uint32_t validRowEnd = directRows ? gTiling.GetValue(offsetTiling + 20) : qHeads;
                uint32_t mergeSplitCount;
                uint32_t mergeStride;
                uint64_t partialBase;
                if constexpr (Balanced) {
                    const uint32_t rec = 4 * (1 + batchIdx);
                    mergeSplitCount = schedule.GetValue(rec + 3);
                    mergeStride = mergeSplitCount;
                    partialBase = static_cast<uint64_t>(batchIdx + schedule.GetValue(rec + 2)) * qHeads;
                    epilogueMLAFDRescaleO.SetPartialStride(mergeStride);
                    if (mergeSplitCount == 0) kvSeqlen = 0;
                    if (schedule.GetValue(3) == 1 && mergeSplitCount == 1 && (!causalMtp || kvSeqlen >= 4)) continue;
                } else {
                    mergeSplitCount = ActiveSplits(kvSeqlen, kvSplitPerCore, causalMtp);
                    mergeStride = kvSplitCoreNum;
                    partialBase = oFdOffset * kvSplitCoreNum / embed;
                }
                if (kvSeqlen != 0) {
                    epilogueMLAFDRescaleO(
                    gO[(outputTokens || directRows) ? 0 : oAddr + headBase * embed],
                    gOCoreTmp[(partialBase + headBase * mergeStride) * embed],
                    gl[(Balanced ? partialBase : lOffset) + headBase * mergeStride],
                    actualHeads, headsProcess, embed, mergeSplitCount,
                    outputTokens, firstRow + headBase, directRows,
                    validRowBegin > headBase ? validRowBegin - headBase : 0,
                    validRowEnd > headBase ? validRowEnd - headBase : 0, headsPerToken);
                }
                if constexpr (GraphDecode) {
                    const uint32_t invalid = kvSeqlen == 0 ? qHeads :
                        (causalMtp && kvSeqlen < 4 ? (4 - kvSeqlen) * headsPerToken : 0);
                    if (headBase < invalid) {
                        epilogueMLAFDRescaleO.ZeroRows(gO, firstRow + headBase, min(actualHeads, invalid - headBase),
                                                      headsPerToken, outputTokens, embed);
                    }
                }
            }
        }
    }

private:
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
};


static constexpr uint64_t L1_TILE_DIM_0 = 128;
static constexpr uint64_t L1_TILE_DIM_1 = 128;
static constexpr uint64_t L1_TILE_DIM_2 = 576;
constexpr uint32_t ComputeEleNum = 6144;


CATLASS_GLOBAL void MLAFp16(uint64_t fftsAddr,
                        GM_ADDR q, GM_ADDR qRope, GM_ADDR k, GM_ADDR kRope,
                        GM_ADDR blockTables, GM_ADDR o, GM_ADDR s, GM_ADDR p,
                        GM_ADDR oTmp, GM_ADDR oUpdate,GM_ADDR oCoreTmp, GM_ADDR l,
                        GM_ADDR tiling)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);;

    using ElementQ = half;
    using LayoutQ = layout::RowMajor;
    using ElementK = half;
    using LayoutK = layout::ColumnMajor;
    using ElementV = half;
    using LayoutV = layout::RowMajor;;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = half;
    using LayoutP = layout::RowMajor;
    using ElementO = half;
    using LayoutO = layout::RowMajor;
    using ElementMask = half;
    using LayoutMask = layout::RowMajor;;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;

    // L1TileShape::K must be embedding
    using L1TileShape = GemmShape<L1_TILE_DIM_0, L1_TILE_DIM_1, L1_TILE_DIM_2>;
    using L0TileShape = L1TileShape;;

    // GEMM Block模块，实现Flash MLA的Q * K^T
    using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;;

    // Epilogue Block模块，实现Flash MLA中当前S基块的softmax
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueMLASoftmax =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLASoftmax, PType, SType, MaskType>;

    // GEMM Block模块，实现Flash MLA的P * V
    using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中当前O基块的更新
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;;
    using EpilogueMLARescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLARescaleO, OType, OUpdateType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中flash decoding
    using lType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;;
    using EpilogueMLAFDRescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;

    // Kernel level
    using MLAKernel = MLAKernel<BlockMmadQK, BlockMmadPV, EpilogueMLASoftmax,
                                EpilogueMLARescaleO, EpilogueMLAFDRescaleO>;
    typename MLAKernel::Params params{q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling};;

    // call kernel
    MLAKernel mla;
    mla(params);
}


template<bool GraphDecode, bool Balanced = false>
CATLASS_DEVICE void RunMLABf16(uint64_t fftsAddr,
                        GM_ADDR q, GM_ADDR qRope, GM_ADDR k, GM_ADDR kRope,
                        GM_ADDR blockTables, GM_ADDR o, GM_ADDR s, GM_ADDR p,
                        GM_ADDR oTmp, GM_ADDR oUpdate, GM_ADDR oCoreTmp, GM_ADDR l,
                        GM_ADDR tiling, GM_ADDR lengths, GM_ADDR schedule = nullptr)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ElementQ = __bf16;
    using LayoutQ = layout::RowMajor;
    using ElementK = __bf16;
    using LayoutK = layout::ColumnMajor;
    using ElementV = __bf16;
    using LayoutV = layout::RowMajor;
    using ElementS = float;;
    using LayoutS = layout::RowMajor;
    using ElementP = __bf16;
    using LayoutP = layout::RowMajor;
    using ElementO = __bf16;
    using LayoutO = layout::RowMajor;
    using ElementMask = __bf16;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;

    // L1TileShape::K must be embedding
    using L1TileShape = GemmShape<L1_TILE_DIM_0, L1_TILE_DIM_1, L1_TILE_DIM_2>;
    using L0TileShape = L1TileShape;

    // GEMM Block模块，实现Flash MLA的Q * K^T
    using DispatchPolicyQK = Gemm::MmadAtlasA2MLAQK;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

    // Epilogue Block模块，实现Flash MLA中当前S基块的softmax
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueMLASoftmax =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLASoftmax, PType, SType, MaskType>;

    // GEMM Block模块，实现Flash MLA的P * V
    using DispatchPolicyPV = Gemm::MmadAtlasA2MLAPV;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中当前O基块的更新
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using EpilogueMLARescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLARescaleO, OType, OUpdateType, OTmpType>;

    // Epilogue Block模块，实现Flash MLA中flash decoding
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using lType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;

    using EpilogueMLAFDRescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2MLAFDRescaleO<ComputeEleNum>, OType, lType>;

    // Kernel level
    using MLAKernel = MLAKernel<BlockMmadQK, BlockMmadPV, EpilogueMLASoftmax,
                                EpilogueMLARescaleO, EpilogueMLAFDRescaleO, GraphDecode, Balanced>;
    typename MLAKernel::Params params{q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling, lengths, schedule};

    // call kernel
    MLAKernel mla;
    mla(params);
}

CATLASS_GLOBAL void MLABf16(uint64_t fftsAddr,
                        GM_ADDR q, GM_ADDR qRope, GM_ADDR k, GM_ADDR kRope,
                        GM_ADDR blockTables, GM_ADDR o, GM_ADDR s, GM_ADDR p,
                        GM_ADDR oTmp, GM_ADDR oUpdate, GM_ADDR oCoreTmp, GM_ADDR l,
                        GM_ADDR tiling)
{
    RunMLABf16<false>(fftsAddr, q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling, nullptr);
}

CATLASS_GLOBAL void MLAGraphBf16(uint64_t fftsAddr,
                        GM_ADDR q, GM_ADDR qRope, GM_ADDR k, GM_ADDR kRope,
                        GM_ADDR blockTables, GM_ADDR o, GM_ADDR s, GM_ADDR p,
                        GM_ADDR oTmp, GM_ADDR oUpdate, GM_ADDR oCoreTmp, GM_ADDR l,
                        GM_ADDR tiling, GM_ADDR lengths)
{
    RunMLABf16<true>(fftsAddr, q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling, lengths);
}

CATLASS_GLOBAL void MLABalancedGraphBf16(uint64_t fftsAddr,
                        GM_ADDR q, GM_ADDR qRope, GM_ADDR k, GM_ADDR kRope,
                        GM_ADDR blockTables, GM_ADDR o, GM_ADDR s, GM_ADDR p,
                        GM_ADDR oTmp, GM_ADDR oUpdate, GM_ADDR oCoreTmp, GM_ADDR l,
                        GM_ADDR tiling, GM_ADDR lengths, GM_ADDR schedule)
{
    RunMLABf16<true, true>(fftsAddr, q, qRope, k, kRope, blockTables, o, s, p, oTmp, oUpdate, oCoreTmp, l, tiling, lengths, schedule);
}
