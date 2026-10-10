/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file blasst_attention_score_tiling.cpp
 * \brief
 */

#include "blasst_attention_score_tiling.h"
#include "error/ops_error.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "exe_graph/runtime/tensor_data.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
// Dummy base tiling-data class for the op_type entry; SplitFuse kernels use
// FAInferTilingData registered per tiling key below.
BEGIN_TILING_DATA_DEF(VllmBlasstAttentionScoreTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, reserved)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore, VllmBlasstAttentionScoreTilingData)

// Register the single FAInferTilingData class for the supported tiling keys.
// Non-paged TND regular keys.
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000000200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000000201100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000000200103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000000201103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000000200200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000000201200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000000200203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000000201203, FAInferTilingData)
// Paged-cache regular keys (paged + TND, no flash-decode).
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000010200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000010201100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000010200103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000010201103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000010200200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000010201200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000010200203, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5000000000010201203, FAInferTilingData)
// Regular-kernel flash-decode keys (split-KV + cross-core LSE merge):
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5100000000010200100, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5100000000010200103, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5100000000010200200, FAInferTilingData)
REGISTER_TILING_DATA_CLASS(VllmBlasstAttentionScore_5100000000010200203, FAInferTilingData)

struct VllmBlasstAttentionScoreCompileInfo {};

uint32_t FAInferTiling::GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qRowNumCeil = Q_TILE_CEIL;
    uint32_t qNBlockTile = (qSeqlen != 0) ?
        (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
    qNBlockTile = std::min(qNBlockTile, groupSize);
    qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
    // GQA prefill head packing: mirror of the kernel-side GetQNBlockTile.
    // Must stay in lockstep with blasst_attention_score_common.h so the
    // host-side task decomposition matches the kernel's.
    if (qSeqlen >= Q_TILE_CEIL) {
        uint32_t headPack = Q_TILE_CEIL / 16;
        qNBlockTile = std::min(headPack, groupSize);
    }
    return qNBlockTile;
}

uint32_t FAInferTiling::GetQSBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    // Complement of the qN head pack (see kernel-side GetQSBlockTile).
    uint32_t qSBlockTile = std::max(Q_TILE_CEIL / GetQNBlockTile(qSeqlen, groupSize),
                                    static_cast<uint32_t>(1));
    return qSBlockTile;
}

uint32_t FAInferTiling::GetKvNBlockTile(uint32_t rowNumPerQSGTile, uint32_t kvHead)
{
    uint32_t rowNumCeilPerQSGKvNTile = Q_TILE_CEIL;
    uint32_t kvNBlockTile = rowNumCeilPerQSGKvNTile / rowNumPerQSGTile;
    kvNBlockTile = std::min(kvNBlockTile, kvHead);
    kvNBlockTile = std::max(kvNBlockTile, static_cast<uint32_t>(1));
    return kvNBlockTile;
}
uint32_t FAInferTiling::GetKSBlockTile()
{
    uint32_t kSBlockTile = MAX_KV_STACK_LEN;
    return kSBlockTile;
}

void FAInferTiling::FillBasicTilingData(FAInferTilingData &faTilingData)
{
    faTilingData.set_batch(static_cast<uint32_t>(faInfo_.batch));
    faTilingData.set_numHeads(static_cast<uint32_t>(faInfo_.numHeads));
    faTilingData.set_kvHeads(static_cast<uint32_t>(faInfo_.kvHeads));

    faTilingData.set_embeddingSize(static_cast<uint32_t>(faInfo_.embeddingSize));
    faTilingData.set_embeddingSizeV(static_cast<uint32_t>(faInfo_.embeddingSizeV));
    faTilingData.set_numBlocks(static_cast<uint32_t>(faInfo_.numBlocks));
    if (faInfo_.pagedCacheFlag) {
        faTilingData.set_blockSize(static_cast<uint32_t>(faInfo_.blockSize));
    } else {
        faTilingData.set_blockSize(BASE_KV_SIZE);
    }
    faTilingData.set_maxQSeqlen(faInfo_.maxQSeqlen);
    faTilingData.set_maxKvSeqlen(faInfo_.maxKvSeqlen);
    faTilingData.set_maxNumBlocksPerBatch(faInfo_.maxNumBlocksPerBatch);
    faTilingData.set_maskType(static_cast<uint32_t>(faInfo_.maskType));
    faTilingData.set_scaleValue(faInfo_.scaleValue);
    faTilingData.set_sparseLamda(faInfo_.sparseLamda);
    faTilingData.set_sparseMode(faInfo_.sparseMode);
    faTilingData.set_preToken(static_cast<int64_t>(faInfo_.preToken));
    faTilingData.set_nextToken(static_cast<int64_t>(faInfo_.nextToken));
}

uint64_t FAInferTiling::GetTilingKey()
{
    constexpr uint64_t SPLIT_FUSE_BASE_KEY = 5000000000000000000;
    constexpr uint64_t PAGED_CACHE_KEY = 10000000;
    constexpr uint64_t COMP_CAUSAL_MASK_KEY = 3;
    constexpr uint64_t LAYOUTQ_TND_KEY = 200000;
    constexpr uint64_t DTYPE_FP16_KEY = 100;
    constexpr uint64_t DTYPE_BF16_KEY = 200;
    constexpr uint64_t LSE_OUT_ONLY_KEY = 1000;
    constexpr uint64_t FLASH_DECODE_KEY = 100000000000000000;
    uint64_t tilingKey = SPLIT_FUSE_BASE_KEY;
    if (faInfo_.pagedCacheFlag) {
        tilingKey += static_cast<uint64_t>(PAGED_CACHE_KEY);
    }
    if (faInfo_.maskType == MaskType::MASK_SPEC) {
        tilingKey += static_cast<uint64_t>(COMP_CAUSAL_MASK_KEY);
    }
    tilingKey += static_cast<uint64_t>(LAYOUTQ_TND_KEY);
    if (faInfo_.dataType == DataType::FP16) {
        tilingKey += static_cast<uint64_t>(DTYPE_FP16_KEY);
    } else if (faInfo_.dataType == DataType::BF16) {
        tilingKey += static_cast<uint64_t>(DTYPE_BF16_KEY);
    }
    if (faInfo_.lseFlag) {
        tilingKey += static_cast<uint64_t>(LSE_OUT_ONLY_KEY);
    }
    if (faInfo_.flashDecodeFlag) {
        tilingKey += static_cast<uint64_t>(FLASH_DECODE_KEY);
    }
    return tilingKey;
}

void FAInferTiling::FillWorkSpaceTilingData(FAInferTilingData &faTilingData)
{
    // S workspace region is fp32 (ElementS=float in the kernel): S lands
    // directly in the fp32 softmax buffer with no upcast pass. The fp16
    // variant halved L2 bytes but its per-tile fp16 Muls + fp16->fp32 Cast
    // stalled the cube/vec ring and dominated the occ0 prefill gap.
    uint64_t mm1OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
        SIZE_OF_32BIT * PRELANCH_NUM;
    uint64_t smOnlineOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
        SIZE_OF_16BIT * PRELANCH_NUM;
    // OTmp region is fp16 (ElementOTmp=half in the kernel): matches the
    // fp32->fp16 FIX conversion on the PV cube side and halves the per-stack-
    // tile OTmp round-trip bytes through L2. The host workspace request in
    // TilingVllmBlasstAttentionScore deliberately stays at the fp32-sized
    // upper bound, so it needs no matching change.
    uint64_t mm2OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
        SIZE_OF_16BIT * PRELANCH_NUM;
    uint64_t UpdateSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
        SIZE_OF_32BIT * PRELANCH_NUM;
    uint64_t spFlagSize = static_cast<uint64_t>(blockNum_) * 2 * 32 * (PRELANCH_NUM + 1);
    uint64_t sparseStatsSize = static_cast<uint64_t>(blockNum_) * 16 * SIZE_OF_32BIT;
    sparseStatsSize = (sparseStatsSize + 511) / 512 * 512;
    uint64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize + spFlagSize +
        sparseStatsSize;
    workSpaceSize += faTilingData.get_splitLseTotalSize() + faTilingData.get_splitOTotalSize();
    faTilingData.set_mm1OutSize(mm1OutSize);
    faTilingData.set_smOnlineOutSize(smOnlineOutSize);
    faTilingData.set_mm2OutSize(mm2OutSize);
    faTilingData.set_UpdateSize(UpdateSize);
    faTilingData.set_sparseStatsSize(sparseStatsSize);
    faTilingData.set_workSpaceSize(workSpaceSize);
}

void FAInferTiling::FillSplitCoreTilingData(FAInferTilingData &faTilingData)
{
    uint32_t totalTaskNum = 0;
    uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
    for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
        uint32_t qSeqlen = *(faInfo_.qSeqlenList + batchIdx);
        uint32_t kvSeqlen = *(faInfo_.kvSeqlenList + batchIdx);
        if (batchIdx > 0) {
            uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + batchIdx - 1);
            qSeqlen = qSeqlen - prevQSeqlenSum;
            if (!faInfo_.pagedCacheFlag) {
                uint64_t prevKvSeqlenSum = *(faInfo_.kvSeqlenList + batchIdx - 1);
                kvSeqlen = kvSeqlen - prevKvSeqlenSum;
            }
        }
        uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
        uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
        uint32_t curQNBlockNum = qNBlockNumPerGroup * faInfo_.kvHeads;
        uint32_t curQSBlockTile = GetQSBlockTile(qSeqlen, groupSize);
        uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
        uint32_t curTaskNum = curQNBlockNum * curQSBlockNum;
        if (batchIdx == 0) {
            faTilingData.set_firstBatchTaskNum(curTaskNum);
        }
        totalTaskNum += curTaskNum;
    }
    faTilingData.set_totalTaskNum(totalTaskNum);
}

FAInferTiling::BatchParams FAInferTiling::getBatchParams(uint32_t bIdx, uint32_t groupSize)
{
    BatchParams p;
    p.qSeqlen = *(faInfo_.qSeqlenList + bIdx);
    p.kvSeqlen = *(faInfo_.kvSeqlenList + bIdx);
    if (bIdx > 0) {
        uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + bIdx - 1);
        p.qSeqlen = p.qSeqlen - prevQSeqlenSum;
        if (!faInfo_.pagedCacheFlag) {
            uint64_t prevKvSeqlenSum = *(faInfo_.kvSeqlenList + bIdx - 1);
            p.kvSeqlen = p.kvSeqlen - prevKvSeqlenSum;
        }
    }
    p.curQNBlockTile = GetQNBlockTile(p.qSeqlen, groupSize);
    p.qNBlockNumPerGroup = (groupSize + p.curQNBlockTile - 1) / p.curQNBlockTile;
    p.curQNBlockNum = p.qNBlockNumPerGroup * faInfo_.kvHeads;
    p.curQSBlockTile = GetQSBlockTile(p.qSeqlen, groupSize);
    p.curQSBlockNum = (p.qSeqlen + p.curQSBlockTile - 1) / p.curQSBlockTile;
    p.curKSBlockTile = GetKSBlockTile();
    p.curKSBlockNum = (p.kvSeqlen + p.curKSBlockTile - 1) / p.curKSBlockTile;
    return p;
}

uint32_t FAInferTiling::NoSkipKvOf(const BatchParams &p, uint32_t s1Idx) const
{
    // Host mirror of runMainLoop's noSkipKvS: with a causal mask a qS block
    // only iterates KV tiles up to its own diagonal. Decode (single qS block
    // per task) degenerates to the full kvSeqlen, so flash-decode accounting
    // is unchanged there.
    uint64_t noSkip = p.kvSeqlen;
    if (faInfo_.maskType == MaskType::MASK_SPEC) {
        uint64_t diffS = (p.kvSeqlen > p.qSeqlen) ? (p.kvSeqlen - p.qSeqlen) : 0;
        noSkip = static_cast<uint64_t>(s1Idx + 1) * p.curQSBlockTile + diffS;
        noSkip = std::min<uint64_t>(noSkip, static_cast<uint64_t>(p.kvSeqlen));
    }
    return static_cast<uint32_t>(noSkip);
}

void FAInferTiling::FillTaskAssignTilingData(FAInferTilingData &faTilingData)
{
    faTilingData.set_taskAssignFlag(0U);
    if (faInfo_.flashDecodeFlag || blockNum_ <= 1 ||
        blockNum_ > static_cast<uint32_t>(MAX_CORE_NUM_FD)) {
        return;
    }
    uint32_t totalTaskNum = faTilingData.get_totalTaskNum();
    // Below one task per core the round-robin stride is already idle-bound;
    // above the perm bound taskPerm cannot hold the schedule.
    if (totalTaskNum < blockNum_ || totalTaskNum > static_cast<uint32_t>(MAX_PREFILL_ASSIGN_TASKS)) {
        return;
    }

    uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
    // Per-task cost in KV-column units plus a per-tile latency allowance
    // (each stack tile costs a roughly fixed dependency-bound latency on top
    // of its column-proportional work, so a near-empty causal tail tile is
    // weighted ~half a full tile). Task order mirrors the kernel's decode:
    // batch-major, then s1-major with n1 fastest.
    std::vector<uint64_t> cost(totalTaskNum, 0);
    uint32_t taskIdx = 0;
    for (int32_t bIdx = 0; bIdx < faInfo_.batch && taskIdx < totalTaskNum; bIdx++) {
        BatchParams p = getBatchParams(bIdx, groupSize);
        for (uint32_t s1Idx = 0; s1Idx < p.curQSBlockNum && taskIdx < totalTaskNum; s1Idx++) {
            uint32_t noSkip = NoSkipKvOf(p, s1Idx);
            uint32_t tiles = (noSkip + p.curKSBlockTile - 1) / p.curKSBlockTile;
            uint64_t taskCost = static_cast<uint64_t>(noSkip) +
                static_cast<uint64_t>(tiles) * p.curKSBlockTile;
            for (uint32_t n1Idx = 0; n1Idx < p.curQNBlockNum && taskIdx < totalTaskNum; n1Idx++) {
                cost[taskIdx++] = taskCost;
            }
        }
    }

    // Longest-processing-time-first packing, deterministic (ties break to the
    // lowest core index). With near-uniform task costs (GQA8 prefill packing:
    // every task holds 5 causal KV tiles) plain round-robin leaves cores at
    // ceil(T/C) vs floor(T/C) tasks -- a +12% makespan spread at 128/24 that
    // LPT flattens to the cost floor.
    std::vector<uint32_t> order(totalTaskNum);
    for (uint32_t i = 0; i < totalTaskNum; i++) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&cost](uint32_t a, uint32_t b) { return cost[a] > cost[b]; });

    std::vector<uint64_t> load(blockNum_, 0);
    std::vector<std::vector<uint32_t>> coreTasks(blockNum_);
    for (uint32_t i = 0; i < totalTaskNum; i++) {
        uint32_t best = 0;
        for (uint32_t c = 1; c < blockNum_; c++) {
            if (load[c] < load[best]) {
                best = c;
            }
        }
        coreTasks[best].push_back(order[i]);
        load[best] += cost[order[i]];
    }

    // Swap refinement: plain LPT leaves 6-task cores holding "5 average + the
    // single lightest" task (measured [32560,37168] vs the 34133 average-load
    // floor on occ0's 128 tasks / 24 cores). Swapping a pair between the
    // max/min cores with cost delta closest to half their load gap closes it
    // in a few passes; task costs are near-continuous (noSkip steps by the
    // 16-row qS tile), so the residual gap drops under one cost step.
    for (uint32_t pass = 0; pass < 64; pass++) {
        uint32_t maxCore = 0;
        uint32_t minCore = 0;
        for (uint32_t c = 1; c < blockNum_; c++) {
            if (load[c] > load[maxCore]) {
                maxCore = c;
            }
            if (load[c] < load[minCore]) {
                minCore = c;
            }
        }
        uint64_t gap = load[maxCore] - load[minCore];
        if (gap <= MAX_KV_STACK_LEN / 2) {
            break;
        }
        // best swap pair: cost(a) - cost(b) closest to gap/2, a from maxCore,
        // b from minCore, positive delta only
        uint64_t halfGap = gap / 2;
        int32_t bestA = -1;
        int32_t bestB = -1;
        uint64_t bestDelta = 0;
        uint64_t bestErr = UINT64_MAX;
        for (uint32_t ia = 0; ia < coreTasks[maxCore].size(); ia++) {
            for (uint32_t ib = 0; ib < coreTasks[minCore].size(); ib++) {
                uint64_t delta = cost[coreTasks[maxCore][ia]] - cost[coreTasks[minCore][ib]];
                if (delta == 0 || delta >= gap) {
                    continue;
                }
                uint64_t err = (delta > halfGap) ? (delta - halfGap) : (halfGap - delta);
                if (err < bestErr) {
                    bestErr = err;
                    bestDelta = delta;
                    bestA = static_cast<int32_t>(ia);
                    bestB = static_cast<int32_t>(ib);
                }
            }
        }
        if (bestA < 0) {
            break;
        }
        uint32_t taskA = coreTasks[maxCore][bestA];
        uint32_t taskB = coreTasks[minCore][bestB];
        coreTasks[maxCore][bestA] = taskB;
        coreTasks[minCore][bestB] = taskA;
        load[maxCore] -= bestDelta;
        load[minCore] += bestDelta;
    }

    uint32_t pos = 0;
    for (uint32_t c = 0; c < blockNum_; c++) {
        faTilingData.get_coreTaskStart()[c] = pos;
        for (uint32_t t : coreTasks[c]) {
            faTilingData.get_taskPerm()[pos++] = t;
        }
        faTilingData.get_coreTaskEnd()[c] = pos;
    }
    faTilingData.set_taskAssignFlag(1U);
}

void FAInferTiling::fillCoreInfoForPrefillFlashDecode(FAInferTilingData &faTilingData,
                                                      uint32_t groupSize)
{
    // Causal-weighted contiguous assignment of (B, N1, S1, S2-tile) work units
    // over cores, cut at proportional boundaries of the TOTAL causal work.
    // Round-robin over equal-size qS tiles concentrates the causal tail on a
    // few cores (q2048/kv4096/GQA8 packs 8 heads x 16 tokens per task, so
    // 128 tasks with 5..8 kv tiles each spread 39/33 tiles across cores);
    // cutting on real work evens cores out at ~34/35 tiles. A cut may land
    // inside a task's KV range: that task becomes a split-KV node handled by
    // the existing flash-decode combine machinery.
    for (uint32_t coreIdx = 0; coreIdx < blockNum_; coreIdx++) {
        faTilingData.coreInfo.get_startBIdx()[coreIdx] = 0;
        faTilingData.coreInfo.get_startN1Idx()[coreIdx] = 0;
        faTilingData.coreInfo.get_startS1Idx()[coreIdx] = 0;
        faTilingData.coreInfo.get_startS2Idx()[coreIdx] = 0;
        faTilingData.coreInfo.get_endBIdx()[coreIdx] = 0;
        faTilingData.coreInfo.get_endN1Idx()[coreIdx] = 0;
        faTilingData.coreInfo.get_endS1Idx()[coreIdx] = 0;
        faTilingData.coreInfo.get_endS2Idx()[coreIdx] = 0;
        faTilingData.coreInfo.get_firstSplitKVTaskLseOffset()[coreIdx] = 0;
        faTilingData.coreInfo.get_firstSplitKVTaskOOffset()[coreIdx] = 0;
    }

    // Unit weight: one S2 tile-iteration costs a roughly fixed latency
    // (dependency-bound chain) plus work proportional to its kv width, so a
    // full tile is weighted ~3x a near-empty causal tail tile. Pure element
    // accounting overloads the cores that absorb the many light tail tiles;
    // pure tile counting (weight 1 per tile) is the other extreme. The
    // dependency-bound regime measured on occ0 (per-tile ~fixed) puts the
    // sweet spot at kv + 2*tile: max/min tiles 36/33 vs 39/33 round-robin
    // (832 causal tiles over 24 cores floor at 35).
    uint64_t totalWork = 0;
    for (int32_t bIdx = 0; bIdx < faInfo_.batch; bIdx++) {
        BatchParams p = getBatchParams(bIdx, groupSize);
        for (uint32_t n1Idx = 0; n1Idx < p.curQNBlockNum; n1Idx++) {
            uint32_t qNBlockSize = (n1Idx % p.qNBlockNumPerGroup == p.qNBlockNumPerGroup - 1) ?
                (groupSize - (n1Idx % p.qNBlockNumPerGroup) * p.curQNBlockTile) : p.curQNBlockTile;
            for (uint32_t s1Idx = 0; s1Idx < p.curQSBlockNum; s1Idx++) {
                uint32_t qSBlockSize = (s1Idx == p.curQSBlockNum - 1) ?
                    (p.qSeqlen - s1Idx * p.curQSBlockTile) : p.curQSBlockTile;
                uint32_t noSkip = NoSkipKvOf(p, s1Idx);
                uint32_t causalKSBlockNum = (noSkip + p.curKSBlockTile - 1) / p.curKSBlockTile;
                for (uint32_t s2Idx = 0; s2Idx < causalKSBlockNum; s2Idx++) {
                    uint32_t kvThisTile = std::min(noSkip - s2Idx * p.curKSBlockTile,
                                                   p.curKSBlockTile);
                    totalWork += static_cast<uint64_t>(qSBlockSize) * qNBlockSize *
                        (kvThisTile + 2 * p.curKSBlockTile);
                }
            }
        }
    }

    uint32_t nowBIdx = 0;
    uint32_t nowN1Idx = 0;
    uint32_t nowS1Idx = 0;
    uint32_t nowS2Idx = 0;
    uint64_t assignedWork = 0;
    bool workLeft = (faInfo_.batch > 0);
    for (uint32_t coreIdx = 0; coreIdx < blockNum_; coreIdx++) {
        if (!workLeft) {
            // idle core: empty span
            faTilingData.coreInfo.get_startBIdx()[coreIdx] =
                static_cast<uint32_t>(faInfo_.batch);
            faTilingData.coreInfo.get_startN1Idx()[coreIdx] = 0;
            faTilingData.coreInfo.get_startS1Idx()[coreIdx] = 0;
            faTilingData.coreInfo.get_startS2Idx()[coreIdx] = 0;
            faTilingData.coreInfo.get_endBIdx()[coreIdx] = 0;
            faTilingData.coreInfo.get_endN1Idx()[coreIdx] = 0;
            faTilingData.coreInfo.get_endS1Idx()[coreIdx] = 0;
            faTilingData.coreInfo.get_endS2Idx()[coreIdx] = 0;
            continue;
        }
        faTilingData.coreInfo.get_startBIdx()[coreIdx] = nowBIdx;
        faTilingData.coreInfo.get_startN1Idx()[coreIdx] = nowN1Idx;
        faTilingData.coreInfo.get_startS1Idx()[coreIdx] = nowS1Idx;
        faTilingData.coreInfo.get_startS2Idx()[coreIdx] = nowS2Idx;

        // this core's share ends at (coreIdx + 1) / blockNum_ of the total
        uint64_t boundary = (totalWork * (coreIdx + 1)) / blockNum_;

        // position of the last unit this core consumed (endS2 exclusive)
        uint32_t lastBIdx = nowBIdx;
        uint32_t lastN1Idx = nowN1Idx;
        uint32_t lastS1Idx = nowS1Idx;
        uint32_t lastS2End = nowS2Idx;
        BatchParams p = getBatchParams(nowBIdx, groupSize);
        while (nowBIdx < static_cast<uint32_t>(faInfo_.batch)) {
            if (nowN1Idx >= p.curQNBlockNum || nowS1Idx >= p.curQSBlockNum ||
                nowS2Idx >= p.curKSBlockNum) {
                nowS2Idx = 0;
                nowS1Idx++;
                if (nowS1Idx >= p.curQSBlockNum) {
                    nowS1Idx = 0;
                    nowN1Idx++;
                }
                if (nowN1Idx >= p.curQNBlockNum) {
                    nowN1Idx = 0;
                    nowBIdx++;
                }
                if (nowBIdx >= static_cast<uint32_t>(faInfo_.batch)) {
                    break;
                }
                p = getBatchParams(nowBIdx, groupSize);
                continue;
            }

            uint32_t causalKSBlockNum =
                (NoSkipKvOf(p, nowS1Idx) + p.curKSBlockTile - 1) / p.curKSBlockTile;
            if (nowS2Idx >= causalKSBlockNum) {
                // kv tiles past the causal diagonal carry no work: skip them
                nowS2Idx = p.curKSBlockNum;
                continue;
            }

            uint32_t qNBlockSize = (nowN1Idx % p.qNBlockNumPerGroup == p.qNBlockNumPerGroup - 1) ?
                (groupSize - (nowN1Idx % p.qNBlockNumPerGroup) * p.curQNBlockTile) : p.curQNBlockTile;
            uint32_t qSBlockSize = (nowS1Idx == p.curQSBlockNum - 1) ?
                (p.qSeqlen - nowS1Idx * p.curQSBlockTile) : p.curQSBlockTile;
            uint32_t kvRemain = NoSkipKvOf(p, nowS1Idx) - nowS2Idx * p.curKSBlockTile;
            uint32_t kvThisTile = std::min(kvRemain, p.curKSBlockTile);
            assignedWork += static_cast<uint64_t>(qSBlockSize) * qNBlockSize *
                (kvThisTile + 2 * p.curKSBlockTile);
            lastBIdx = nowBIdx;
            lastN1Idx = nowN1Idx;
            lastS1Idx = nowS1Idx;
            lastS2End = nowS2Idx + 1;
            nowS2Idx++;

            if (assignedWork >= boundary) {
                break;
            }
        }
        workLeft = (nowBIdx < static_cast<uint32_t>(faInfo_.batch));

        // Close the core's span at the last consumed unit. endS2Idx is
        // exclusive; a task whose causal range is fully covered from its
        // span start simply has coveredS2 == causalKSBlockNum, which both
        // the kernel and fillSplitInfoForFlashDecode already classify as
        // unsplit -- no full-KV sentinel needed.
        BatchParams pEnd = getBatchParams(lastBIdx, groupSize);
        uint32_t causalKSBlockNum =
            (NoSkipKvOf(pEnd, lastS1Idx) + pEnd.curKSBlockTile - 1) / pEnd.curKSBlockTile;
        if (lastS2End == causalKSBlockNum - 1) {
            // A cut that leaves the continuation a single KV tile makes the
            // part-start tile the masked causal diagonal: the masked softmax
            // overload's EVENT_ID4 part-start Wait then lives inside the one
            // runMainLoop that must also issue the matching Set, and the
            // part deadlocks the MTE3_V relay (507015 sync timeout; proven
            // by bisect -- suppressing only that Wait runs clean). Absorb
            // the last tile into this span so the task is not split at all.
            uint32_t qNBlockSizeEnd = (lastN1Idx % pEnd.qNBlockNumPerGroup == pEnd.qNBlockNumPerGroup - 1) ?
                (groupSize - (lastN1Idx % pEnd.qNBlockNumPerGroup) * pEnd.curQNBlockTile) : pEnd.curQNBlockTile;
            uint32_t qSBlockSizeEnd = (lastS1Idx == pEnd.curQSBlockNum - 1) ?
                (pEnd.qSeqlen - lastS1Idx * pEnd.curQSBlockTile) : pEnd.curQSBlockTile;
            uint32_t kvThisTile = NoSkipKvOf(pEnd, lastS1Idx) - lastS2End * pEnd.curKSBlockTile;
            assignedWork += static_cast<uint64_t>(qSBlockSizeEnd) * qNBlockSizeEnd *
                (kvThisTile + 2 * pEnd.curKSBlockTile);
            lastS2End = causalKSBlockNum;
        }
        faTilingData.coreInfo.get_endBIdx()[coreIdx] = lastBIdx;
        faTilingData.coreInfo.get_endN1Idx()[coreIdx] = lastN1Idx;
        faTilingData.coreInfo.get_endS1Idx()[coreIdx] = lastS1Idx;
        faTilingData.coreInfo.get_endS2Idx()[coreIdx] = lastS2End;
        // When the closed task's causal range is fully covered, advance the
        // walk cursor past the task so the next core starts on a task
        // boundary; leaving it mid-task would hand the next core an empty
        // continuation span that re-initializes (zeroes) the task's output.
        nowBIdx = lastBIdx;
        nowN1Idx = lastN1Idx;
        nowS1Idx = lastS1Idx;
        nowS2Idx = (lastS2End >= causalKSBlockNum) ? pEnd.curKSBlockNum : lastS2End;
    }
}

void FAInferTiling::fillCoreInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize,
                                               uint64_t perCoreTaskNum)
{
    uint32_t nowBIdx = 0;
    uint32_t nowN1Idx = 0;
    uint32_t nowS1Idx = 0;
    uint32_t nowS2Idx = 0;

    for (uint32_t coreIdx = 0; coreIdx < blockNum_; coreIdx++) {
        faTilingData.coreInfo.get_startBIdx()[coreIdx] = 0;
        faTilingData.coreInfo.get_startN1Idx()[coreIdx] = 0;
        faTilingData.coreInfo.get_startS1Idx()[coreIdx] = 0;
        faTilingData.coreInfo.get_startS2Idx()[coreIdx] = 0;
        faTilingData.coreInfo.get_endBIdx()[coreIdx] = 0;
        faTilingData.coreInfo.get_endN1Idx()[coreIdx] = 0;
        faTilingData.coreInfo.get_endS1Idx()[coreIdx] = 0;
        faTilingData.coreInfo.get_endS2Idx()[coreIdx] = 0;
    }

    auto finishBatch = [&](uint32_t coreIdx) {
        BatchParams p = getBatchParams(faInfo_.batch - 1, groupSize);
        faTilingData.coreInfo.get_endBIdx()[coreIdx] = faInfo_.batch - 1;
        faTilingData.coreInfo.get_endN1Idx()[coreIdx] = p.curQNBlockNum - 1;
        faTilingData.coreInfo.get_endS1Idx()[coreIdx] = p.curQSBlockNum - 1;
        faTilingData.coreInfo.get_endS2Idx()[coreIdx] = p.curKSBlockNum;
    };

    for (uint32_t coreIdx = 0; coreIdx < blockNum_; coreIdx++) {
        int64_t resTaskNum = perCoreTaskNum;
        faTilingData.coreInfo.get_startBIdx()[coreIdx] = nowBIdx;
        faTilingData.coreInfo.get_startN1Idx()[coreIdx] = nowN1Idx;
        faTilingData.coreInfo.get_startS1Idx()[coreIdx] = nowS1Idx;
        faTilingData.coreInfo.get_startS2Idx()[coreIdx] = nowS2Idx;

        BatchParams p = getBatchParams(nowBIdx, groupSize);

        auto advanceCounters = [&]() {
            if (nowS2Idx == p.curKSBlockNum) { nowS1Idx++; nowS2Idx = 0; }
            if (nowS1Idx == p.curQSBlockNum) { nowN1Idx++; nowS1Idx = 0; nowS2Idx = 0; }
            if (nowN1Idx == p.curQNBlockNum) { nowBIdx++; nowN1Idx = 0; nowS1Idx = 0; nowS2Idx = 0; }
        };

        while (nowS2Idx < p.curKSBlockNum && resTaskNum > 0) {
            p = getBatchParams(nowBIdx, groupSize);
            uint32_t remainingQ = (nowS1Idx < p.curQSBlockNum - 1) ? p.curQSBlockTile :
                (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint32_t remainingKV = (nowS2Idx < p.curKSBlockNum - 1) ? p.curKSBlockTile :
                (p.kvSeqlen - nowS2Idx * p.curKSBlockTile);
            uint64_t singleS2Task = remainingQ * remainingKV;
            resTaskNum -= singleS2Task;
            nowS2Idx += 1;
        }

        if (resTaskNum <= 0) {
            faTilingData.coreInfo.get_endBIdx()[coreIdx] = nowBIdx;
            faTilingData.coreInfo.get_endN1Idx()[coreIdx] = nowN1Idx;
            faTilingData.coreInfo.get_endS1Idx()[coreIdx] = nowS1Idx;
            faTilingData.coreInfo.get_endS2Idx()[coreIdx] = nowS2Idx;
        }

        advanceCounters();
        if (nowBIdx < static_cast<uint32_t>(faInfo_.batch) && resTaskNum <= 0) continue;
        if (nowBIdx == static_cast<uint32_t>(faInfo_.batch)) { finishBatch(coreIdx); break; }

        while (nowBIdx < static_cast<uint32_t>(faInfo_.batch) && resTaskNum > 0) {
            p = getBatchParams(nowBIdx, groupSize);
            uint32_t remainingQ = p.qSeqlen * (faInfo_.numHeads - p.curQNBlockTile * nowN1Idx) -
                nowS1Idx * p.curQSBlockTile;
            uint32_t remainingKV = p.kvSeqlen;
            uint32_t remainingInBatch = remainingQ * remainingKV;

            if (resTaskNum >= static_cast<int64_t>(remainingInBatch)) {
                resTaskNum -= remainingInBatch;
                nowBIdx++; nowN1Idx = 0; nowS1Idx = 0; nowS2Idx = 0;
            } else {
                break;
            }
        }

        if (nowBIdx == static_cast<uint32_t>(faInfo_.batch)) { finishBatch(coreIdx); break; }
        p = getBatchParams(nowBIdx, groupSize);

        while (nowN1Idx < p.curQNBlockNum && resTaskNum > 0) {
            uint32_t remainingQ = p.qSeqlen * p.curQNBlockTile - nowS1Idx * p.curQSBlockTile;
            uint32_t remainingInN1 = remainingQ * p.kvSeqlen;
            if (resTaskNum >= static_cast<int64_t>(remainingInN1)) {
                resTaskNum -= remainingInN1;
                nowN1Idx++; nowS1Idx = 0; nowS2Idx = 0;
            } else {
                break;
            }
        }

        advanceCounters();
        if (nowBIdx == static_cast<uint32_t>(faInfo_.batch)) { finishBatch(coreIdx); break; }
        p = getBatchParams(nowBIdx, groupSize);

        while (nowS1Idx < p.curQSBlockNum && resTaskNum > 0) {
            uint32_t remainingQ = (nowS1Idx < p.curQSBlockNum - 1) ? p.curQSBlockTile :
                (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint64_t remainingInS1 = remainingQ * p.kvSeqlen;
            if (resTaskNum >= static_cast<int64_t>(remainingInS1)) {
                resTaskNum -= remainingInS1;
                nowS1Idx++; nowS2Idx = 0;
            } else {
                break;
            }
        }

        advanceCounters();
        if (nowBIdx == static_cast<uint32_t>(faInfo_.batch)) { finishBatch(coreIdx); break; }
        p = getBatchParams(nowBIdx, groupSize);

        while (nowS2Idx < p.curKSBlockNum && resTaskNum > 0) {
            uint32_t remainingQ = (nowS1Idx < p.curQSBlockNum - 1) ? p.curQSBlockTile :
                (p.qSeqlen - nowS1Idx * p.curQSBlockTile) * p.curQNBlockTile;
            uint32_t remainingKV = (nowS2Idx < p.curKSBlockNum - 1) ? p.curKSBlockTile :
                (p.kvSeqlen - nowS2Idx * p.curKSBlockTile);
            uint64_t singleS2Task = remainingQ * remainingKV;
            resTaskNum -= singleS2Task;
            nowS2Idx += 1;
        }

        if (nowBIdx == static_cast<uint32_t>(faInfo_.batch)) { finishBatch(coreIdx); break; }

        faTilingData.coreInfo.get_endBIdx()[coreIdx] = nowBIdx;
        faTilingData.coreInfo.get_endN1Idx()[coreIdx] = nowN1Idx;
        faTilingData.coreInfo.get_endS1Idx()[coreIdx] = nowS1Idx;
        faTilingData.coreInfo.get_endS2Idx()[coreIdx] = nowS2Idx;

        advanceCounters();
    }
}

void FAInferTiling::fillSplitInfoForFlashDecode(FAInferTilingData &faTilingData, uint32_t groupSize)
{
    // splitInfo arrays hold MAX_CORE_NUM_FD entries; the kernel never reads
    // entry [blockNum_].
    for (uint32_t splitIdx = 0; splitIdx < blockNum_; splitIdx++) {
        faTilingData.splitInfo.get_batchIdx()[splitIdx] = 0;
        faTilingData.splitInfo.get_headStartIdx()[splitIdx] = 0;
        faTilingData.splitInfo.get_headEndIdx()[splitIdx] = 0;
        faTilingData.splitInfo.get_qStartIdx()[splitIdx] = 0;
        faTilingData.splitInfo.get_qEndIdx()[splitIdx] = 0;
        faTilingData.splitInfo.get_splitNum()[splitIdx] = 0;
        faTilingData.splitInfo.get_lseTaskOffset()[splitIdx] = 0;
        faTilingData.splitInfo.get_oTaskOffset()[splitIdx] = 0;
    }

    int64_t currentLseTaskOffset = 0;
    int64_t currentOTaskOffset = 0;
    int32_t splitIdx = -1;
    int32_t prevBIdx = -1;
    int32_t prevN1Idx = -1;
    int32_t prevS1Idx = -1;

    for (uint32_t coreIdx = 0; coreIdx < blockNum_; coreIdx++) {
        int32_t startBIdx = faTilingData.coreInfo.get_startBIdx()[coreIdx];
        int32_t startN1Idx = faTilingData.coreInfo.get_startN1Idx()[coreIdx];
        int32_t startS1Idx = faTilingData.coreInfo.get_startS1Idx()[coreIdx];
        int32_t startS2Idx = faTilingData.coreInfo.get_startS2Idx()[coreIdx];
        int32_t endBIdx = faTilingData.coreInfo.get_endBIdx()[coreIdx];
        int32_t endN1Idx = faTilingData.coreInfo.get_endN1Idx()[coreIdx];
        int32_t endS1Idx = faTilingData.coreInfo.get_endS1Idx()[coreIdx];
        int32_t endS2Idx = faTilingData.coreInfo.get_endS2Idx()[coreIdx];

        faTilingData.coreInfo.get_firstSplitKVTaskLseOffset()[coreIdx] = 0;
        faTilingData.coreInfo.get_firstSplitKVTaskOOffset()[coreIdx] = 0;

        bool foundFirstSplitKV = false;

        for (int32_t BIdx = startBIdx; BIdx <= endBIdx; BIdx++) {
            BatchParams p = getBatchParams(BIdx, groupSize);

            int32_t curStartN1 = (BIdx == startBIdx) ? startN1Idx : 0;
            int32_t curEndN1 = (BIdx == endBIdx) ? endN1Idx : p.curQNBlockNum - 1;

            for (int32_t N1Idx = curStartN1; N1Idx <= curEndN1; N1Idx++) {
                int32_t curStartS1 = (BIdx == startBIdx && N1Idx == startN1Idx) ? startS1Idx : 0;
                int32_t curEndS1 = (BIdx == endBIdx && N1Idx == endN1Idx) ? endS1Idx : p.curQSBlockNum - 1;

                for (int32_t S1Idx = curStartS1; S1Idx <= curEndS1; S1Idx++) {
                    int32_t curStartS2 = (BIdx == startBIdx && N1Idx == startN1Idx &&
                        S1Idx == startS1Idx) ? startS2Idx : 0;
                    // A prefill task only ever covers its causal KV range, so
                    // both the split threshold and the non-end-task span end
                    // must use the causal tile count, mirroring the kernel's
                    // causalKSBlockNumTmp exactly. Using the full-KV count for
                    // a continuation part ([st,causal) of a cut task) hid the
                    // split from the host while the kernel still wrote FD
                    // parts -- unaccounted offset advances past Losize.
                    uint32_t causalKSBlockNum = (NoSkipKvOf(p, S1Idx) + p.curKSBlockTile - 1) /
                        p.curKSBlockTile;
                    int32_t curEndS2 = (BIdx == endBIdx && N1Idx == endN1Idx &&
                        S1Idx == endS1Idx) ? endS2Idx : static_cast<int32_t>(causalKSBlockNum);
                    uint32_t coveredS2 = curEndS2 - curStartS2;
                    bool isSplitKV = (coveredS2 > 0 && coveredS2 < causalKSBlockNum);

                    int64_t tmpLseOffset = currentLseTaskOffset;
                    int64_t tmpOOffset = currentOTaskOffset;

                    uint32_t N1IdxPerGroup = N1Idx % p.qNBlockNumPerGroup;
                    uint32_t kvHeadIdx = N1Idx / p.qNBlockNumPerGroup;
                    uint32_t currentHeadStart = kvHeadIdx * groupSize + N1IdxPerGroup * p.curQNBlockTile;
                    uint32_t currentHeadEnd = std::min(currentHeadStart + p.curQNBlockTile,
                                                       (kvHeadIdx + 1) * groupSize);

                    uint32_t currentQStart = S1Idx * p.curQSBlockTile;
                    uint32_t currentQEnd = std::min(currentQStart + p.curQSBlockTile, p.qSeqlen);

                    uint32_t headLen = currentHeadEnd - currentHeadStart;
                    uint32_t qLen = currentQEnd - currentQStart;

                    if (isSplitKV) {
                        if (BIdx != prevBIdx || N1Idx != prevN1Idx || S1Idx != prevS1Idx) {
                            splitIdx++;
                            if (splitIdx >= 0 && splitIdx < (int32_t)blockNum_) {
                                faTilingData.splitInfo.get_batchIdx()[splitIdx] = BIdx;
                                faTilingData.splitInfo.get_splitNum()[splitIdx] = 0;
                                faTilingData.splitInfo.get_headStartIdx()[splitIdx] = currentHeadStart;
                                faTilingData.splitInfo.get_headEndIdx()[splitIdx] = currentHeadEnd;
                                faTilingData.splitInfo.get_qStartIdx()[splitIdx] = currentQStart;
                                faTilingData.splitInfo.get_qEndIdx()[splitIdx] = currentQEnd;
                                faTilingData.splitInfo.get_lseTaskOffset()[splitIdx] = currentLseTaskOffset;
                                faTilingData.splitInfo.get_oTaskOffset()[splitIdx] = currentOTaskOffset;
                            }
                            prevBIdx = BIdx;
                            prevN1Idx = N1Idx;
                            prevS1Idx = S1Idx;
                        }
                        if (splitIdx >= 0 && splitIdx < (int32_t)blockNum_) {
                            faTilingData.splitInfo.get_splitNum()[splitIdx]++;
                            currentLseTaskOffset += (int64_t)headLen * qLen;
                            currentOTaskOffset += (int64_t)headLen * qLen * faInfo_.embeddingSizeV;
                        }

                        if (!foundFirstSplitKV) {
                            foundFirstSplitKV = true;
                            faTilingData.coreInfo.get_firstSplitKVTaskLseOffset()[coreIdx] = tmpLseOffset;
                            faTilingData.coreInfo.get_firstSplitKVTaskOOffset()[coreIdx] = tmpOOffset;
                        }
                    }
                }
            }
        }
    }

    uint32_t actualSplitNum = (splitIdx + 1 > (int32_t)blockNum_) ? blockNum_ : (splitIdx + 1);
    faTilingData.set_totalSplitNodeNum(actualSplitNum);
    faTilingData.set_splitLseTotalSize(currentLseTaskOffset * SIZE_OF_32BIT);
    faTilingData.set_splitOTotalSize(currentOTaskOffset * SIZE_OF_32BIT);
}

void FAInferTiling::splitBN2S1GS2(FAInferTilingData &faTilingData)
{
    // The FD tiling arrays are sized MAX_CORE_NUM_FD (26): clamp the core
    // count BEFORE dividing tasks across cores, or fillCoreInfoForFlashDecode
    // / fillSplitInfoForFlashDecode would write past the arrays on SoCs with
    // more AI cores (the current gate bounds numTasks, not the core count).
    blockNum_ = std::min(blockNum_, static_cast<uint32_t>(MAX_CORE_NUM_FD));
    uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;

    if (faInfo_.maxQSeqlen > 1) {
        // prefill/chunked-prefill: causal-weighted assignment (see
        // fillCoreInfoForPrefillFlashDecode)
        fillCoreInfoForPrefillFlashDecode(faTilingData, groupSize);
    } else {
        uint64_t totalTaskNum = 0;
        for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
            BatchParams p = getBatchParams(batchIdx, groupSize);
            totalTaskNum += faInfo_.numHeads * p.qSeqlen * p.kvSeqlen;
        }
        uint64_t perCoreTaskNum = (totalTaskNum + blockNum_ - 1) / blockNum_;
        fillCoreInfoForFlashDecode(faTilingData, groupSize, perCoreTaskNum);
    }
    fillSplitInfoForFlashDecode(faTilingData, groupSize);
}

ge::graphStatus FAInferTiling::DoTiling(FAInferTilingData &tilingdata)
{
    tilingdata.set_totalSplitNodeNum(0);
    tilingdata.set_splitLseTotalSize(0);
    tilingdata.set_splitOTotalSize(0);
    FillBasicTilingData(tilingdata);
    FillSplitCoreTilingData(tilingdata);
    FillTaskAssignTilingData(tilingdata);
    if (faInfo_.flashDecodeFlag) {
        splitBN2S1GS2(tilingdata);
    }
    tilingdata.set_sparseStatsFlag(faInfo_.sparseStatsFlag ? 1U : 0U);
    FillWorkSpaceTilingData(tilingdata);
    return ge::GRAPH_SUCCESS;
}

static bool TryGetSeqLengthsFromAttr(const gert::TilingContext *context, uint32_t attrIndex,
                                     std::vector<int64_t> &hostData)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    const gert::ContinuousVector *vec = attrs->GetAttrPointer<gert::ContinuousVector>(attrIndex);
    if (vec == nullptr || vec->GetSize() == 0U) {
        return false;
    }
    const int64_t *data = reinterpret_cast<const int64_t *>(vec->GetData());
    hostData.assign(data, data + vec->GetSize());
    return true;
}

static ge::graphStatus TilingPrepareForVllmBlasstAttentionScore(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertContextToFAInferContext(gert::TilingContext *context, FAInferContext &faInfo,
                                                      const std::vector<int64_t> &hostActualQSeq,
                                                      const std::vector<int64_t> &hostActualKvSeq,
                                                      uint32_t aicoreNum)
{
    auto qDesc = context->GetInputDesc(BLASST_QUERY_INPUT_INDEX);
    OPS_ERR_IF(qDesc == nullptr, OPS_LOG_E("VllmBlasstAttentionScore", "query desc is nullptr"),
               return ge::GRAPH_FAILED);
    ge::DataType qDataType = qDesc->GetDataType();
    OPS_ERR_IF(qDataType != ge::DT_FLOAT16 && qDataType != ge::DT_BF16,
               OPS_LOG_E("VllmBlasstAttentionScore", "query dtype must be FP16 or BF16"),
               return ge::GRAPH_FAILED);

    auto kDesc = context->GetInputDesc(BLASST_KEY_INPUT_INDEX);
    auto vDesc = context->GetInputDesc(BLASST_VALUE_INPUT_INDEX);
    OPS_ERR_IF(kDesc == nullptr || vDesc == nullptr,
               OPS_LOG_E("VllmBlasstAttentionScore", "key/value desc is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kDesc->GetDataType() != qDataType || vDesc->GetDataType() != qDataType,
               OPS_LOG_E("VllmBlasstAttentionScore", "key/value dtype must match query dtype"),
               return ge::GRAPH_FAILED);

    auto qShape = context->GetInputShape(BLASST_QUERY_INPUT_INDEX);
    auto kShape = context->GetInputShape(BLASST_KEY_INPUT_INDEX);
    auto vShape = context->GetInputShape(BLASST_VALUE_INPUT_INDEX);
    OPS_ERR_IF(qShape == nullptr || kShape == nullptr || vShape == nullptr,
               OPS_LOG_E("VllmBlasstAttentionScore", "input shape is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShape->GetStorageShape().GetDimNum() != 3,
               OPS_LOG_E("VllmBlasstAttentionScore", "query must be 3D TND layout"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(kShape->GetStorageShape().GetDimNum() != 3 || vShape->GetStorageShape().GetDimNum() != 3,
               OPS_LOG_E("VllmBlasstAttentionScore", "key/value must be 3D TND layout"),
               return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E("VllmBlasstAttentionScore", "attrs is nullptr"),
               return ge::GRAPH_FAILED);

    const char *layoutPtr = attrs->GetAttrPointer<char>(BLASST_INPUT_LAYOUT_ATTR_INDEX);
    OPS_ERR_IF(layoutPtr == nullptr, OPS_LOG_E("VllmBlasstAttentionScore", "input_layout attr is nullptr"),
               return ge::GRAPH_FAILED);
    std::string layoutStr(layoutPtr);
    OPS_ERR_IF(layoutStr != "TND",
               OPS_LOG_E("VllmBlasstAttentionScore", "only TND layout is supported"),
               return ge::GRAPH_FAILED);

    const int64_t *numHeadsPtr = attrs->GetAttrPointer<int64_t>(BLASST_NUM_HEADS_ATTR_INDEX);
    const int64_t *numKvHeadsPtr = attrs->GetAttrPointer<int64_t>(BLASST_NUM_KEY_VALUE_HEADS_ATTR_INDEX);
    OPS_ERR_IF(numHeadsPtr == nullptr || numKvHeadsPtr == nullptr,
               OPS_LOG_E("VllmBlasstAttentionScore", "num_heads/num_key_value_heads attr is nullptr"),
               return ge::GRAPH_FAILED);
    int64_t numHeads = *numHeadsPtr;
    int64_t numKvHeads = (*numKvHeadsPtr == 0) ? numHeads : *numKvHeadsPtr;
    OPS_ERR_IF(numHeads == 0 || numKvHeads == 0 || numHeads % numKvHeads != 0,
               OPS_LOG_E("VllmBlasstAttentionScore", "invalid num_heads/num_key_value_heads"),
               return ge::GRAPH_FAILED);

    const int64_t *sparseModePtr = attrs->GetAttrPointer<int64_t>(BLASST_SPARSE_MODE_ATTR_INDEX);
    OPS_ERR_IF(sparseModePtr == nullptr,
               OPS_LOG_E("VllmBlasstAttentionScore", "sparse_mode attr is nullptr"),
               return ge::GRAPH_FAILED);
    int64_t sparseMode = *sparseModePtr;
    OPS_ERR_IF(sparseMode != 0 && sparseMode != 3,
               OPS_LOG_E("VllmBlasstAttentionScore", "only sparse_mode 0 or 3 is supported"),
               return ge::GRAPH_FAILED);

    // pse_shift is not supported in this minimal migration
    OPS_ERR_IF(context->GetOptionalInputShape(BLASST_PSE_SHIFT_INPUT_INDEX) != nullptr,
               OPS_LOG_E("VllmBlasstAttentionScore", "pse_shift is not supported"),
               return ge::GRAPH_FAILED);

    const int64_t *antiquantModePtr = attrs->GetAttrPointer<int64_t>(BLASST_ANTIQUANT_MODE_ATTR_INDEX);
    OPS_ERR_IF(antiquantModePtr == nullptr,
               OPS_LOG_E("VllmBlasstAttentionScore", "antiquant_mode attr is nullptr"),
               return ge::GRAPH_FAILED);
    int64_t antiquantMode = *antiquantModePtr;
    OPS_ERR_IF(antiquantMode != 0,
               OPS_LOG_E("VllmBlasstAttentionScore", "only antiquant_mode 0 is supported in this migration"),
               return ge::GRAPH_FAILED);

    const float *sparseLambdaPtr = attrs->GetAttrPointer<float>(BLASST_SPARSE_LAMBDA_ATTR_INDEX);
    OPS_ERR_IF(sparseLambdaPtr == nullptr,
               OPS_LOG_E("VllmBlasstAttentionScore", "sparse_lambda attr is nullptr"),
               return ge::GRAPH_FAILED);
    float sparseLamda = *sparseLambdaPtr;

    const int64_t *preTokenPtr = attrs->GetAttrPointer<int64_t>(BLASST_PRE_TOKENS_ATTR_INDEX);
    const int64_t *nextTokenPtr = attrs->GetAttrPointer<int64_t>(BLASST_NEXT_TOKENS_ATTR_INDEX);
    OPS_ERR_IF(preTokenPtr == nullptr || nextTokenPtr == nullptr,
               OPS_LOG_E("VllmBlasstAttentionScore", "pre_tokens/next_tokens attr is nullptr"),
               return ge::GRAPH_FAILED);
    int64_t preToken = *preTokenPtr;
    int64_t nextToken = *nextTokenPtr;
    if (preToken > SPARSE_MODE_INT_MAX) {
        preToken = SPARSE_MODE_INT_MAX;
    } else if (preToken < -SPARSE_MODE_INT_MAX) {
        preToken = -SPARSE_MODE_INT_MAX;
    }
    if (nextToken > SPARSE_MODE_INT_MAX) {
        nextToken = SPARSE_MODE_INT_MAX;
    } else if (nextToken < -SPARSE_MODE_INT_MAX) {
        nextToken = -SPARSE_MODE_INT_MAX;
    }

    const float *scalePtr = attrs->GetAttrPointer<float>(BLASST_SCALE_ATTR_INDEX);
    OPS_ERR_IF(scalePtr == nullptr, OPS_LOG_E("VllmBlasstAttentionScore", "scale attr is nullptr"),
               return ge::GRAPH_FAILED);

    const int64_t *blockSizePtr = attrs->GetAttrPointer<int64_t>(BLASST_BLOCK_SIZE_ATTR_INDEX);
    const int64_t *innerPrecisePtr = attrs->GetAttrPointer<int64_t>(BLASST_INNER_PRECISE_ATTR_INDEX);
    const bool *lseFlagPtr = attrs->GetAttrPointer<bool>(BLASST_SOFTMAX_LSE_FLAG_ATTR_INDEX);
    OPS_ERR_IF(blockSizePtr == nullptr || innerPrecisePtr == nullptr || lseFlagPtr == nullptr,
               OPS_LOG_E("VllmBlasstAttentionScore", "block_size/inner_precise/softmax_lse_flag attr is nullptr"),
               return ge::GRAPH_FAILED);
    int64_t blockSize = *blockSizePtr;
    int64_t innerPrecise = *innerPrecisePtr;
    bool lseFlag = *lseFlagPtr;

    const bool *sparseStatsFlagPtr = attrs->GetAttrPointer<bool>(BLASST_SPARSE_STATS_FLAG_ATTR_INDEX);
    const bool sparseStatsFlag = (sparseStatsFlagPtr != nullptr) && (*sparseStatsFlagPtr);

    OPS_ERR_IF(innerPrecise != 0,
               OPS_LOG_E("VllmBlasstAttentionScore", "only inner_precise 0 is supported"),
               return ge::GRAPH_FAILED);

    auto actualQSeq = context->GetOptionalInputTensor(BLASST_ACTUAL_SEQ_LENGTHS_INPUT_INDEX);
    auto actualKvSeq = context->GetOptionalInputTensor(BLASST_ACTUAL_SEQ_LENGTHS_KV_INPUT_INDEX);
    // host-list-only: the device seq inputs are absent (adapter passes empty
    // optionals); the host IntArray attrs are the single source of seq lens.
    OPS_ERR_IF((actualQSeq != nullptr || actualKvSeq != nullptr) &&
               (actualQSeq == nullptr || actualKvSeq == nullptr),
               OPS_LOG_E("VllmBlasstAttentionScore", "seq inputs must be passed as a pair"),
               return ge::GRAPH_FAILED);
    const bool hostSeqAvail = !hostActualQSeq.empty() && !hostActualKvSeq.empty();
    OPS_ERR_IF(!hostSeqAvail,
               OPS_LOG_E("VllmBlasstAttentionScore",
                         "host seq attrs are required (host-list-only mode)"),
               return ge::GRAPH_FAILED);
    const int64_t *actualSeqQ = hostActualQSeq.data();
    const int64_t *actualSeqKv = hostActualKvSeq.data();
    int32_t batch = static_cast<int32_t>(hostActualQSeq.size());
    OPS_ERR_IF(batch <= 0,
               OPS_LOG_E("VllmBlasstAttentionScore", "invalid actual_seq_lengths size"),
               return ge::GRAPH_FAILED);
    // Length equality must be checked HERE: ConvertContextToFAInferContext
    // indexes actualSeqKv[b] for b < batch, so a shorter kv list would be a
    // host heap out-of-bounds read before any later guard runs.
    OPS_ERR_IF(hostActualQSeq.size() != hostActualKvSeq.size(),
               OPS_LOG_E("VllmBlasstAttentionScore", "q/kv seq list size mismatch"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(batch > BLASST_MAX_HOST_SEQ_LIST,
               OPS_LOG_E("VllmBlasstAttentionScore",
                         "batch ", batch, " exceeds BLASST_MAX_HOST_SEQ_LIST ",
                         BLASST_MAX_HOST_SEQ_LIST, " (tiling-embedded seq)"),
               return ge::GRAPH_FAILED);

    auto blockTableShape = context->GetOptionalInputShape(BLASST_BLOCK_TABLE_INPUT_INDEX);
    bool pagedCacheFlag = (blockTableShape != nullptr);
    if (pagedCacheFlag) {
        OPS_ERR_IF(blockTableShape->GetStorageShape().GetDimNum() != 2,
                   OPS_LOG_E("VllmBlasstAttentionScore", "block_table must be 2D"),
                   return ge::GRAPH_FAILED);
        auto blockTableDesc = context->GetOptionalInputDesc(BLASST_BLOCK_TABLE_INPUT_INDEX);
        OPS_ERR_IF(blockTableDesc != nullptr && blockTableDesc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E("VllmBlasstAttentionScore", "block_table must be INT32"),
                   return ge::GRAPH_FAILED);
    }

    faInfo.pagedCacheFlag = pagedCacheFlag;
    faInfo.numHeads = static_cast<int32_t>(numHeads);
    faInfo.kvHeads = static_cast<int32_t>(numKvHeads);
    faInfo.numBlocks = static_cast<int32_t>(kShape->GetStorageShape().GetDim(0));
    faInfo.blockSize = static_cast<int32_t>(blockSize);
    faInfo.embeddingSize = static_cast<int32_t>(qShape->GetStorageShape().GetDim(2));
    faInfo.embeddingSizeV = faInfo.embeddingSize;
    faInfo.scaleValue = *scalePtr;
    faInfo.sparseLamda = sparseLamda;
    faInfo.sparseMode = static_cast<int32_t>(sparseMode);
    faInfo.preToken = preToken;
    faInfo.nextToken = nextToken;
    faInfo.layout = layoutStr;
    faInfo.lseFlag = lseFlag;
    faInfo.sparseStatsFlag = sparseStatsFlag;
    faInfo.innerPrecise = static_cast<int32_t>(innerPrecise);
    faInfo.dataType = (qDataType == ge::DT_BF16) ? DataType::BF16 : DataType::FP16;
    faInfo.batch = batch;
    faInfo.qSeqlenList = actualSeqQ;
    faInfo.kvSeqlenList = actualSeqKv;
    faInfo.maskType = (sparseMode == 3) ? MaskType::MASK_SPEC : MaskType::NO_MASK;
    if (pagedCacheFlag) {
        faInfo.maxNumBlocksPerBatch = static_cast<uint32_t>(blockTableShape->GetStorageShape().GetDim(1));
    }

    int64_t maxQSeqlen = 0;
    int64_t minQSeqlen = INT64_MAX;
    int64_t maxKvSeqlen = 0;
    int64_t minKvSeqlen = INT64_MAX;
    if (hostSeqAvail) {
        for (int32_t b = 0; b < batch; ++b) {
            int64_t qSeqlen = actualSeqQ[b];
            int64_t kvSeqlen = actualSeqKv[b];
            if (b > 0) {
                qSeqlen -= actualSeqQ[b - 1];
                if (!pagedCacheFlag) {
                    kvSeqlen -= actualSeqKv[b - 1];
                }
            }
            maxQSeqlen = std::max(maxQSeqlen, qSeqlen);
            minQSeqlen = std::min(minQSeqlen, qSeqlen);
            maxKvSeqlen = std::max(maxKvSeqlen, kvSeqlen);
            minKvSeqlen = std::min(minKvSeqlen, kvSeqlen);
        }
    }
    faInfo.maxQSeqlen = static_cast<int64_t>(maxQSeqlen);
    faInfo.maxKvSeqlen = static_cast<int64_t>(maxKvSeqlen);

    faInfo.flashDecodeFlag = false;
    uint32_t groupSize = static_cast<uint32_t>(faInfo.numHeads / faInfo.kvHeads);
    uint32_t kvNBlockTile = std::max(1U, std::min(Q_TILE_CEIL / groupSize,
                                                  static_cast<uint32_t>(faInfo.kvHeads)));
    uint32_t kvNBlockNum = (static_cast<uint32_t>(faInfo.kvHeads) + kvNBlockTile - 1) / kvNBlockTile;
    uint32_t numTasks = static_cast<uint32_t>(batch) * kvNBlockNum;
    const int64_t fdMinKv = (numTasks <= 8) ? 4096 : 16384;
    if (hostSeqAvail && pagedCacheFlag && maxQSeqlen == 1 && minQSeqlen == 1 &&
        (faInfo.blockSize == 128) &&
        !lseFlag && (faInfo.innerPrecise == 0) &&
        (numTasks * 5 <= aicoreNum * 4) && (numTasks <= static_cast<uint32_t>(MAX_CORE_NUM_FD)) &&
        (minKvSeqlen >= fdMinKv)) {
        faInfo.flashDecodeFlag = true;
    }

    // Prefill stays on the regular kernel. Re-routing prefill through the
    // flash-decode span dispatch (coreInfo cuts on causal work + split-KV
    // combine) was tried and measured a NET LOSS on occ0 (294.9us vs 256.4us
    // round-robin, FIA 228.3): the split machinery (per-node Q reload,
    // pipeline restart, fp32 O/lse transport, CombineScale) adds ~40us aic /
    // ~54us aiv of genuine busy time -- more than the load-imbalance it
    // removes. The imbalance is instead fixed inside the regular kernel by
    // the host LPT task assignment (FillTaskAssignTilingData).

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingVllmBlasstAttentionScore(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("VllmBlasstAttentionScore", "TilingContext is nullptr"),
               return ge::GRAPH_FAILED);

    auto platformInfoPtr = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfoPtr == nullptr,
               OPS_LOG_E("VllmBlasstAttentionScore", "PlatformInfo is nullptr"),
               return ge::GRAPH_FAILED);

    // Use batch mode to ensure all cores start together.
    context->SetScheduleMode(1);

    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfoPtr);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAic();

    FAInferContext faInfo;
    std::vector<int64_t> hostActualQSeq;
    std::vector<int64_t> hostActualKvSeq;
    bool hostSeqFromAttr =
        TryGetSeqLengthsFromAttr(context, BLASST_ACTUAL_SEQ_LENGTHS_Q_HOST_ATTR_INDEX, hostActualQSeq) &&
        TryGetSeqLengthsFromAttr(context, BLASST_ACTUAL_SEQ_LENGTHS_KV_HOST_ATTR_INDEX, hostActualKvSeq);
    // host-list-only: host IntArray attrs are the single source; the adapter
    // no longer uploads device seq tensors (no D2H fallback path either).
    OPS_ERR_IF(!hostSeqFromAttr,
               OPS_LOG_E("VllmBlasstAttentionScore",
                         "host seq attrs are required (host-list-only mode)"),
               return ge::GRAPH_FAILED);
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    ret = ConvertContextToFAInferContext(context, faInfo, hostActualQSeq, hostActualKvSeq, coreNum);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    FAInferTiling faTiling(faInfo);
    faTiling.SetCoreNum(coreNum);

    FAInferTilingData faTilingData;
    ret = faTiling.DoTiling(faTilingData);
    OPS_ERR_IF(ret != ge::GRAPH_SUCCESS,
               OPS_LOG_E("VllmBlasstAttentionScore", "FAInferTiling DoTiling failed"),
               return ge::GRAPH_FAILED);

    OPS_LOG_D(context->GetNodeName(),
              "FIA debug: key=%lu paged=%d fd=%d mask=%d lse=%d "
              "maxQ=%ld maxKv=%ld layout=%s dtype=%d sparseLambda=%.2f",
              faTiling.GetTilingKey(), faInfo.pagedCacheFlag,
              faInfo.flashDecodeFlag, static_cast<int>(faInfo.maskType), faInfo.lseFlag,
              faInfo.maxQSeqlen, faInfo.maxKvSeqlen,
              faInfo.layout.c_str(), static_cast<int>(faInfo.dataType), faInfo.sparseLamda);

    // host-list-only: embed the seq lists into the tiling data so the kernel
    // reads them from the framework-managed tiling buffer (task-update safe),
    // replacing the adapter-uploaded device seq inputs.
    for (size_t i = 0; i < hostActualQSeq.size(); ++i) {
        faTilingData.get_actualQSeq()[i] = hostActualQSeq[i];
        faTilingData.get_actualKvSeq()[i] = hostActualKvSeq[i];
    }

    faTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(faTilingData.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr,
               OPS_LOG_E("VllmBlasstAttentionScore", "workspace sizes array is nullptr"),
               return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(16U * 1024U * 1024U) +
                    static_cast<size_t>(faTiling.GetCoreNum()) * WORKSPACE_BLOCK_SIZE_DB * 4U * 3U * 4U +
                    static_cast<size_t>(faTiling.GetCoreNum()) * 2U * 32U * (PRELANCH_NUM + 1U) +
                    static_cast<size_t>(faTilingData.get_sparseStatsSize()) +
                    static_cast<size_t>(faTilingData.get_splitLseTotalSize()) +
                    static_cast<size_t>(faTilingData.get_splitOTotalSize());

    context->SetBlockDim(faTiling.GetCoreNum());
    context->SetTilingKey(faTiling.GetTilingKey());

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(VllmBlasstAttentionScore)
    .TilingInputsDataDependency({BLASST_ACTUAL_SEQ_LENGTHS_INPUT_INDEX, BLASST_ACTUAL_SEQ_LENGTHS_KV_INPUT_INDEX},
                                {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_HOST})
    .Tiling(TilingVllmBlasstAttentionScore)
    .TilingParse<VllmBlasstAttentionScoreCompileInfo>(TilingPrepareForVllmBlasstAttentionScore);
} // namespace optiling
