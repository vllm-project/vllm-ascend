/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_grad_metadata_split.h
 * \brief FAG (Flash Attn Grad) metadata splitting logic, migrated from
 *        flash_attention_score_grad tiling DoSparse() and sub-functions.
 *        Covers 3 scenarios: Dense, TND, Sparse (causal/band).
 *        BN2 optimization is not included (to be added later).
 *
 * Migration source mapping:
 *   InitFagParams        <- DoSplit() + SetSparseParams() + ProcessTokensInfo() simplified
 *   DoFagDenseSplit      <- normal_regbase.cpp:701-723 (else branch)
 *   DoFagTndSplit        <- varlen_regbase.cpp:953-1044 (GetSparseUnpadBlockInfo)
 *   DoFagSparseBlockInfo <- normal_regbase.cpp:1553-1622 (GetSparseBlockInfo)
 *   GetFagParseS1S2OuterInfo <- normal_regbase.cpp:1509-1551
 *   FagCheckSparseLeftAndRight <- normal_regbase.cpp:734-771
 *   FagFillBlockInfoLoadBalance <- normal_regbase.cpp:1685-1750
 */
#ifndef FLASH_ATTN_GRAD_METADATA_SPLIT_H
#define FLASH_ATTN_GRAD_METADATA_SPLIT_H

#include "flash_attn_metadata_aicpu.h"
#include <algorithm>
#include <cmath>
#include <climits>
#include <vector>

using namespace optiling;

namespace aicpu {

// ===== FAG metadata start =====

// Helper: align value up to alignment
static inline int64_t FagAlignTo(int64_t value, int64_t alignment)
{
    if (alignment <= 0) {
        return value;
    }
    return (value + alignment - 1) / alignment * alignment;
}

// Helper: calcle actual token for TND/unpad sparse modes
// Source: common_regbase.cpp:914-936 CalcleActualToken
static inline void FagCalcleActualToken(uint32_t sparseMode, int64_t s1Token, int64_t s2Token, int64_t actualS1Len,
                                        int64_t actualS2Len, int64_t &actualCalcS1Token, int64_t &actualCalcS2Token)
{
    actualCalcS1Token = s1Token;
    actualCalcS2Token = s2Token;
    // RIGHT_DOWN_CASUAL_BAND(7) / BAND_LEFT_UP_CASUAL(8) with non-bandIdx batch: full mask
    // Not applicable for FAG (mask_mode only 0/3/4), but kept for completeness
    if (sparseMode == FAG_SPARSE_RIGHT_DOWN_CAUSAL || sparseMode == FAG_SPARSE_BAND) {
        actualCalcS1Token = actualCalcS1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = actualCalcS2Token - actualS1Len + actualS2Len;
    }
}

// ===== InitFagParams: initialize split parameters =====
// Source: normal_regbase.cpp:478-509 DoSplit + common_regbase.cpp:1256-1281 ProcessTokensInfo
void FlashAttnMetadataCpuKernel::InitFagParams()
{
    InitFagBaseDims();
    InitFagLayoutAndSparse();
    InitFagActualSeqlen();
    fagSplitAxis_ = FAG_SPLIT_AXIS_BN2GS1S2;
}

// Compute s1/s2 inner, cvInner, outer dims and group size.
void FlashAttnMetadataCpuKernel::InitFagBaseDims()
{
    fagS1Inner_ = 64;
    fagS2Inner_ = 128;

    int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);
    int64_t s2 = static_cast<int64_t>(maxSeqlenKv_);
    fagS1CvInner_ =
        (s1 == 0) ? 1 : ((fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT > s1) ? s1 : fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT);
    fagCvS2Inner_ =
        (s2 == 0) ? 1 : ((fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT > s2) ? s2 : fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT);
    fagS1Outer_ = (s1 == 0) ? 0 : (s1 + fagS1CvInner_ - 1) / fagS1CvInner_;
    fagS2Outer_ = (s2 == 0) ? 0 : (s2 + fagCvS2Inner_ - 1) / fagCvS2Inner_;
    fagG_ = static_cast<int64_t>(numHeadsQ_) / static_cast<int64_t>(numHeadsKv_);
}

// Map layout string to FAG layout type; set sparse mode and tokens.
void FlashAttnMetadataCpuKernel::InitFagLayoutAndSparse()
{
    if (layoutQ_ == "TND") {
        fagLayoutType_ = FAG_INPUT_FORMAT_TND;
    } else if (layoutQ_ == "BNSD") {
        fagLayoutType_ = FAG_INPUT_FORMAT_BN2GS2D;
    } else {
        fagLayoutType_ = FAG_INPUT_FORMAT_BS2N2GD;
    }

    // seqused has highest priority: if provided in non-TND layout, use TND-style
    // splitting (per-batch variable lengths) without changing the actual layout type.
    if (fagLayoutType_ == FAG_INPUT_FORMAT_TND) {
        fagUseTndSplit_ = true;
    } else {
        bool hasSequsedQ = !actualSeqlenQ_.empty() && !isActualSeqlenQAccum_;
        bool hasSequsedK = !actualSeqlenKv_.empty() && !isActualSeqlenKvAccum_;
        if (hasSequsedQ || hasSequsedK) {
            fagUseTndSplit_ = true;
        }
    }

    fagSparseMode_ = FAG_SPARSE_NO_MASK;
    fagIsSparse_ = false;
    fagS1Token_ = (winLeft_ == -1) ? INT32_MAX : winLeft_;
    fagS2Token_ = (winRight_ == -1) ? INT32_MAX : winRight_;

    int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);
    int64_t s2 = static_cast<int64_t>(maxSeqlenKv_);

    if (maskMode_ == 0) {
        fagIsSparse_ = false;
    } else if (maskMode_ == 3) {
        fagSparseMode_ = FAG_SPARSE_RIGHT_DOWN_CAUSAL;
        fagIsSparse_ = true;
        fagS1Token_ = INT32_MAX;
        fagS2Token_ = 0;
        if (!fagUseTndSplit_) {
            fagS1Token_ = fagS1Token_ + s1 - s2;
            fagS2Token_ = fagS2Token_ - s1 + s2;
        }
    } else if (maskMode_ == 4) {
        fagSparseMode_ = FAG_SPARSE_BAND;
        if (s1 > fagS1Token_ || s2 > fagS2Token_) {
            fagIsSparse_ = true;
        }
        if (!fagUseTndSplit_) {
            fagS1Token_ = fagS1Token_ + s1 - s2;
            fagS2Token_ = fagS2Token_ - s1 + s2;
        }
    }
}

// Ensure actualSeqlenQ_/Kv_ hold per-batch lengths (not prefix sums) for FAG split.
void FlashAttnMetadataCpuKernel::InitFagActualSeqlen()
{
    if (!fagUseTndSplit_) {
        return;
    }

    int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);
    int64_t s2 = static_cast<int64_t>(maxSeqlenKv_);
    int64_t b = static_cast<int64_t>(batchSize_);

    // If no actual seq lens available, use maxSeqlen as single-batch
    if (actualSeqlenQ_.empty() && actualSeqlenKv_.empty()) {
        actualSeqlenQ_.push_back(s1);
        actualSeqlenKv_.push_back(s2);
        return;
    }

    // If only one side has seqused, fill the other with maxSeqlen for all batches
    if (actualSeqlenQ_.empty()) {
        actualSeqlenQ_.assign(b, s1);
    }
    if (actualSeqlenKv_.empty()) {
        actualSeqlenKv_.assign(b, s2);
    }

    // When actualSeqlenQ_ comes from cuSeqlensQ (accumulated prefix sums),
    // convert to per-batch actual lengths. FA side uses isActualSeqlenQAccum_=true
    // to keep prefix-sum semantics; FAG split treats actualSeqlenQ_[i] as batch i length.
    if (isActualSeqlenQAccum_) {
        std::vector<int64_t> perBatch(actualSeqlenQ_.size());
        int64_t prev = 0;
        for (size_t i = 0; i < actualSeqlenQ_.size(); i++) {
            perBatch[i] = actualSeqlenQ_[i] - prev;
            prev = actualSeqlenQ_[i];
        }
        actualSeqlenQ_.swap(perBatch);
    }
    if (isActualSeqlenKvAccum_) {
        std::vector<int64_t> perBatch(actualSeqlenKv_.size());
        int64_t prev = 0;
        for (size_t i = 0; i < actualSeqlenKv_.size(); i++) {
            perBatch[i] = actualSeqlenKv_[i] - prev;
            prev = actualSeqlenKv_[i];
        }
        actualSeqlenKv_.swap(perBatch);
    }

    // Detect zero-length sequences: TND with s1=0 or s2=0 in any batch
    // Source: normal_regbase.cpp:250-266 isSeqExistZero / sValueZeroUnderTND
    for (size_t i = 0; i < actualSeqlenQ_.size(); i++) {
        if (actualSeqlenQ_[i] == 0 || (i < actualSeqlenKv_.size() && actualSeqlenKv_[i] == 0)) {
            fagIsSeqExistZero_ = true;
            break;
        }
    }
}

void FlashAttnMetadataCpuKernel::SupportTransBSND()
{
    bool isTND = (fagLayoutType_ == FAG_INPUT_FORMAT_TND);
    if (!actualSeqlenQ_.empty() && isTND) {
        fagIsAllSame_ = true;
        int64_t s1First = actualSeqlenQ_[0];
        int64_t s2First = actualSeqlenKv_[0];
        for (size_t i = 1; i < actualSeqlenQ_.size(); i++) {
            if (actualSeqlenQ_[i] != s1First || actualSeqlenKv_[i] != s2First) {
                fagIsAllSame_ = false;
                break;
            }
        }
    }
    if (fagIsAllSame_ && isTND) {
        fagLayoutType_ = FAG_INPUT_FORMAT_BS2N2GD;
        fagUseTndSplit_ = false;
        if (fagSparseMode_ == FAG_SPARSE_RIGHT_DOWN_CAUSAL || fagSparseMode_ == FAG_SPARSE_BAND) {
            int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);
            int64_t s2 = static_cast<int64_t>(maxSeqlenKv_);
            fagS1Token_ = fagS1Token_ + s1 - s2;
            fagS2Token_ = fagS2Token_ - s1 + s2;
        }
    }
}

// ===== SetFagSplitAxis: determine BN2/BN2S2/BN2GS1S2 =====
// Source: common_regbase.cpp:1581-1649 SetSplitAxis (simplified: no dtype/rope/dropout conditions)
void FlashAttnMetadataCpuKernel::SetFagSplitAxis()
{
    int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);
    int64_t s2 = static_cast<int64_t>(maxSeqlenKv_);
    int64_t n1 = static_cast<int64_t>(numHeadsQ_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t d = static_cast<int64_t>(headDim_);
    int64_t b = static_cast<int64_t>(batchSize_);

    // isAllSame: all batches have same seq lengths (for TND)
    SupportTransBSND();

    // isBn2: S<=128, N1==N2, D<=512, no zero-length sequences in TND
    // Source: common_regbase.cpp:1584-1589 (tailZeroCount==0 && !isSeqExistZero)
    fagIsBn2_ = (s1 <= 128 && s2 <= 128) && (n1 == n2) && (d <= 512) && !fagIsSeqExistZero_;

    // bnLimit: BN >= 256, or BN >= 128 with S1/S2 aligned to 128
    bool bnLimit = ((b * n1) >= 256) || ((b * n1) >= 128 && (s1 % 128 == 0) && (s2 % 128 == 0));
    bool bnSparseLimit = bnLimit && !fagUseTndSplit_;

    // isBn2MultiBlk: BN limit + S>128 + S<=640 + N1==N2 + D<=512
    fagIsBn2MultiBlk_ = bnSparseLimit && (s1 > 128 || s2 > 128) && (s1 <= 640 && s2 <= 640) && (n1 == n2) && (d <= 512);
    fagIsBn2_ = fagIsBn2MultiBlk_ ? true : fagIsBn2_;

    // Source: common_regbase.cpp:1608-1614
    // isBn2 && !isBn2MultiBlk: TND+D>128 时关闭 isBn2 (dropMaskOuter 不涉及)
    if (fagIsBn2_ && !fagIsBn2MultiBlk_) {
        if (fagUseTndSplit_ && d > 128) {
            fagIsBn2_ = false;
        }
    }

    // BN2S2 route: Source common_regbase.cpp:1631-1639
    // bn2S2RouteLimit = !hasRope(removed) && d<=512 &&
    //   (isTND || (isAllSame && !isDeterministic(removed)) || bn2S2NotTndLimit) &&
    //   (keepProb>=1(removed) || ...) && (n1==n2) && (queryType checks removed)
    // Simplified: d<=512 && (isTND || isAllSame || bn2S2NotTndLimit) && (n1==n2)
    bool bn2S2NotTndLimit = (s1 < s2) && (s2 <= 1024) && (s2 - s1 >= 128) && (d <= 128) && !fagIsSparse_;
    bool bn2S2RouteLimit = (d <= 512) && (fagUseTndSplit_ || fagIsAllSame_ || bn2S2NotTndLimit) && (n1 == n2);

    if (fagIsBn2_) {
        fagSplitAxis_ = FAG_SPLIT_AXIS_BN2;
    } else if (bn2S2RouteLimit) {
        fagSplitAxis_ = FAG_SPLIT_AXIS_BN2S2;
        if (fagIsAllSame_) {
            fagUseTndSplit_ = true;
        }
    } else {
        fagSplitAxis_ = FAG_SPLIT_AXIS_BN2GS1S2;
    }
}

// ===== DoFagSparse: dispatch to 5 scenarios =====
// Source: normal_regbase.cpp:660-731 DoSparse
void FlashAttnMetadataCpuKernel::DoFagSparse()
{
    SetFagSplitAxis();

    // Scenario 1: BN2S2
    if (fagSplitAxis_ == FAG_SPLIT_AXIS_BN2S2) {
        TryBn2s2Sparse();
        if (fagBlockOuter_ >= static_cast<uint32_t>(aicCoreNum_)) {
            return;
        }
        // BN2S2 didn't produce enough cores, fall through to try BN2
        // Match old FAG: if is_all_same degraded TND->BSND in SupportTransBSND,
        // SetFagSplitAxis set fagUseTndSplit_=true for BN2S2 TND attempt.
        // Now that BN2S2 failed, restore fagUseTndSplit_=false and clear stale
        // tnd_start_bidx from the failed TND attempt, so Layer 3 uses the
        // correct non-TND path (DoFagSparseBlockInfo / DoFagDenseSplit).
        if (fagIsAllSame_ && fagLayoutType_ != FAG_INPUT_FORMAT_TND) {
            fagUseTndSplit_ = false;
            for (uint32_t c = 0; c < FAG_CORE_LIST_NUM; c++) {
                fagTndStartBIdx_[c] = 0;
            }
        }
    }

    // Scenario 2: BN2 multi-block
    if (fagIsBn2_ && fagIsBn2MultiBlk_) {
        bool success = TryBn2MultiBlkSparse();
        if (success && !fagIsInvalidCol_ && !fagIsInvalidRow_) {
            return;
        }
        // Fallback: degrade to BN2GS1S2
        fagIsBn2_ = false;
        fagIsBn2MultiBlk_ = false;
    }

    // Scenario 3-5: BN2GS1S2 (existing)
    fagSplitAxis_ = fagIsBn2_ ? FAG_SPLIT_AXIS_BN2 : FAG_SPLIT_AXIS_BN2GS1S2;
    if (fagUseTndSplit_) {
        DoFagTndSplit();
    } else if (fagIsSparse_) {
        DoFagSparseBlockInfo();
    } else {
        DoFagDenseSplit();
    }
}

// ===== DoFagDenseSplit: full compute, linear uniform split =====
// Source: normal_regbase.cpp:701-723
void FlashAttnMetadataCpuKernel::DoFagDenseSplit()
{
    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    int64_t fusedOuter = b * n2 * g * fagS1Outer_ * fagS2Outer_;
    int64_t aicNum = static_cast<int64_t>(aicCoreNum_);
    if (fusedOuter == 0) {
        fagBlockOuter_ = 0;
        fagBlockFactor_ = 0;
        return;
    }
    int64_t blockFactor = (fusedOuter + aicNum - 1) / aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

    fagBlockOuter_ = static_cast<uint32_t>(blockOuter);
    fagBlockFactor_ = blockFactor;

    for (int64_t i = 0; i < blockOuter; i++) {
        fagBlockStarts_[i] = blockFactor * i;
        fagBlockEnds_[i] = std::min(blockFactor * (i + 1), fusedOuter);
    }
    for (uint32_t i = static_cast<uint32_t>(blockOuter); i < FAG_CORE_LIST_NUM; i++) {
        fagBlockStarts_[i] = 0;
        fagBlockEnds_[i] = 0;
    }
}

// ===== DoFagSparseBlockInfo: non-TND sparse split (causal/band) =====
// Source: normal_regbase.cpp:1553-1622 GetSparseBlockInfo
void FlashAttnMetadataCpuKernel::DoFagSparseBlockInfo()
{
    if (fagS2Outer_ == 0) {
        fagBlockOuter_ = 0;
        fagBlockFactor_ = 0;
        return;
    }
    std::vector<std::vector<int64_t>> parseInfo(fagS2Outer_, std::vector<int64_t>(FAG_ARRAY_LENGTH, 0));
    GetFagParseS1S2OuterInfo(parseInfo);

    int64_t s1s2oCount = parseInfo[fagS2Outer_ - 1][2]; // LENGTH_IDX = 2

    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    int64_t aicNum = static_cast<int64_t>(aicCoreNum_);

    int64_t fusedOuter = b * n2 * g * s1s2oCount;
    int64_t blockFactor = (fusedOuter + aicNum - 1) / aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

    fagBlockOuter_ = static_cast<uint32_t>(blockOuter);
    fagBlockFactor_ = blockFactor;

    int64_t n2gs1s2o = n2 * g * s1s2oCount;
    int64_t gs1s2o = g * s1s2oCount;

    fagBlockStarts_[0] = 0;
    fagBlockEnds_[blockOuter - 1] = b * n2 * g * fagS1Outer_ * fagS2Outer_;

    for (int64_t c = 1; c < blockOuter; c++) {
        int64_t currentIdx = std::min(c * blockFactor, fusedOuter);
        int64_t bIdx = currentIdx / n2gs1s2o;
        int64_t bTail = currentIdx % n2gs1s2o;
        int64_t n2Idx = bTail / gs1s2o;
        int64_t n2Tail = bTail % gs1s2o;
        int64_t gIdx = n2Tail / s1s2oCount;
        int64_t gTail = n2Tail % s1s2oCount;

        // Reverse map: linear valid-block index -> (s1oIdx, s2oIdx)
        // Source: common_regbase.cpp:894-912 GetCommonS1S2OuterIndex
        int64_t s1oIdx = 0;
        int64_t s2oIdx = 0;
        int64_t preSize = 0;
        for (int64_t i = 0; i < fagS2Outer_; i++) {
            int64_t nextSize = parseInfo[i][2]; // LENGTH_IDX
            if (gTail >= preSize && gTail < nextSize) {
                s2oIdx = i;
                s1oIdx = parseInfo[i][0] + gTail - preSize - 1; // BEGIN_IDX=0
                break;
            }
            preSize = nextSize;
        }

        // Encode as flat BNGS1S2 index
        // Source: normal_regbase.cpp:1606-1608
        fagBlockStarts_[c] = (((bIdx * n2 + n2Idx) * g + gIdx) * fagS2Outer_ + s2oIdx) * fagS1Outer_ + s1oIdx + 1;
        fagBlockEnds_[c - 1] = fagBlockStarts_[c];
    }
    for (uint32_t c = static_cast<uint32_t>(blockOuter); c < FAG_CORE_LIST_NUM; c++) {
        fagBlockStarts_[c] = 0;
        fagBlockEnds_[c] = 0;
    }
}

// ===== GetFagParseS1S2OuterInfo: build sparse valid S1xS2 mapping table =====
// Source: normal_regbase.cpp:1509-1551
void FlashAttnMetadataCpuKernel::GetFagParseS1S2OuterInfo(std::vector<std::vector<int64_t>> &parseInfo)
{
    int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);
    int64_t s2 = static_cast<int64_t>(maxSeqlenKv_);
    fagIsInvalidCol_ = false;
    fagIsInvalidRow_ = false;
    std::vector<bool> invalidS1Array(fagS1Outer_, false);
    for (int64_t i = 0; i < fagS2Outer_; i++) {
        int64_t leftIntersectionPoint = std::max(static_cast<int64_t>(0), fagCvS2Inner_ * i - fagS2Token_);
        if (leftIntersectionPoint > s1) {
            parseInfo[i][0] = (s1 + fagS1CvInner_ - 1) / fagS1CvInner_; // BEGIN_IDX=0
        } else {
            parseInfo[i][0] = leftIntersectionPoint / fagS1CvInner_;
        }
        // Source: normal_regbase.cpp:1519: cvBlockTail = s2CvTail for last block, cvS2Inner otherwise
        int64_t cvBlockTail =
            (i == fagS2Outer_ - 1) ? (s2 % fagCvS2Inner_ == 0 ? fagCvS2Inner_ : s2 % fagCvS2Inner_) : fagCvS2Inner_;
        int64_t rightPoint =
            std::min(std::max(static_cast<int64_t>(0), fagCvS2Inner_ * i + cvBlockTail + fagS1Token_), s1);
        parseInfo[i][1] = (rightPoint + fagS1CvInner_ - 1) / fagS1CvInner_; // END_IDX=1

        int64_t tmpSize = (parseInfo[i][1] > parseInfo[i][0]) ? parseInfo[i][1] - parseInfo[i][0] : 0;
        if (i == 0) {
            parseInfo[i][2] = tmpSize; // LENGTH_IDX=2
        } else {
            parseInfo[i][2] = parseInfo[i - 1][2] + tmpSize;
        }
        // Source: normal_regbase.cpp:1533-1550: invalid col/row detection
        if (parseInfo[i][0] >= parseInfo[i][1]) {
            fagIsInvalidCol_ = true;
        }
        for (int64_t j = 0; j < static_cast<int64_t>(invalidS1Array.size()); j++) {
            if (j >= parseInfo[i][0] && j < parseInfo[i][1]) {
                invalidS1Array[j] = true;
            }
        }
    }
    for (size_t j = 0; j < invalidS1Array.size(); j++) {
        if (!invalidS1Array[j]) {
            fagIsInvalidRow_ = true;
            break;
        }
    }
}

// ===== FagCheckSparseLeftAndRight: check sparse validity for non-TND =====
// Source: normal_regbase.cpp:734-771
bool FlashAttnMetadataCpuKernel::FagCheckSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft, int64_t s2IdxRight,
                                                            int64_t bIdx)
{
    int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);
    int64_t s2 = static_cast<int64_t>(maxSeqlenKv_);

    if (fagSparseMode_ == FAG_SPARSE_RIGHT_DOWN_CAUSAL) {
        int64_t s2IgnoredEndLen = s1 - fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT * (s1oDimIdx + 1);
        int64_t s2EndLen = (s2 > s2IgnoredEndLen) ? (s2 - s2IgnoredEndLen) : 0;
        return s2IdxLeft < s2EndLen;
    } else {
        // BAND mode
        int64_t s2SparseLeft =
            std::max(fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT * s1oDimIdx - fagS1Token_, static_cast<int64_t>(0));
        s2SparseLeft = FagAlignTo(s2SparseLeft, FAG_ALIGN64);
        int64_t s2SparseRight =
            FagAlignTo(std::min(fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT * (s1oDimIdx + 1), s1) + fagS2Token_, FAG_ALIGN64);
        s2SparseRight = std::min(s2SparseRight, s2);
        return s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
    }
}

// ===== FagIsValid: check block validity for non-TND =====
// Source: normal_regbase.cpp:773-788
bool FlashAttnMetadataCpuKernel::FagIsValid(int64_t blockIdx)
{
    int64_t gDimTail = blockIdx % (fagS1Outer_ * fagS2Outer_);
    int64_t s2oDimIdx = gDimTail / fagS1Outer_;
    int64_t s1oDimIdx = gDimTail % fagS1Outer_;
    int64_t s2IdxLeft = s2oDimIdx * fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT;
    int64_t s2IdxRight =
        std::min((s2oDimIdx + 1) * fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT, static_cast<int64_t>(maxSeqlenKv_));
    return FagCheckSparseLeftAndRight(s1oDimIdx, s2IdxLeft, s2IdxRight, 0);
}

// ===== DoFagTndSplit: TND variable-length split =====
// Source: varlen_regbase.cpp:953-1044 GetSparseUnpadBlockInfo
void FlashAttnMetadataCpuKernel::DoFagTndSplit()
{
    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    int64_t aicNum = static_cast<int64_t>(aicCoreNum_);

    if (fagS2Outer_ == 0 || b == 0) {
        fagBlockOuter_ = 0;
        fagBlockFactor_ = 0;
        return;
    }

    // calculatedBlockInfo[b][s2Outer][4]: BEGIN, END, SUM_S1S2, SUM_ALL
    // totalBlockInfo[b][2]: blockCount, cumulativePrefixSum
    std::vector<std::vector<std::vector<int64_t>>> calculatedBlockInfo(
        b, std::vector<std::vector<int64_t>>(fagS2Outer_, std::vector<int64_t>(4, 0)));
    std::vector<std::vector<int64_t>> totalBlockInfo(b, std::vector<int64_t>(2, 0));

    for (int64_t i = 0; i < b; i++) {
        int64_t actualS1Len = actualSeqlenQ_[i];
        int64_t actualS2Len = actualSeqlenKv_[i];
        auto actualS1Outer = (actualS1Len + fagS1CvInner_ - 1) / fagS1CvInner_;
        auto actualS2Outer = (actualS2Len + fagCvS2Inner_ - 1) / fagCvS2Inner_;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;

        // CalValidUnpadBlockInfo
        // Source: varlen_regbase.cpp:1077-1125
        int64_t actualCalcS1Token, actualCalcS2Token;
        FagCalcleActualToken(fagSparseMode_, fagS1Token_, fagS2Token_, actualS1Len, actualS2Len, actualCalcS1Token,
                             actualCalcS2Token);

        for (int64_t j = 0; j < fagS2Outer_; j++) {
            if (fagCvS2Inner_ * j >= actualS2Len) {
                calculatedBlockInfo[i][j][0] = 0; // BEGIN
                calculatedBlockInfo[i][j][1] = 0; // END
            } else {
                int64_t leftIntersectionPoint =
                    std::max(fagCvS2Inner_ * j - actualCalcS2Token, static_cast<int64_t>(0));
                if (leftIntersectionPoint > actualS1Len) {
                    calculatedBlockInfo[i][j][0] = (actualS1Len + fagS1CvInner_ - 1) / fagS1CvInner_;
                } else {
                    calculatedBlockInfo[i][j][0] = leftIntersectionPoint / fagS1CvInner_;
                }
                int64_t cvBlockTail =
                    (fagCvS2Inner_ * (j + 1) > actualS2Len) ? actualS2Len - fagCvS2Inner_ * j : fagCvS2Inner_;
                calculatedBlockInfo[i][j][1] =
                    (std::min(actualS1Len,
                              std::max(fagCvS2Inner_ * j + cvBlockTail + actualCalcS1Token, static_cast<int64_t>(0))) +
                     fagS1CvInner_ - 1) /
                    fagS1CvInner_;
            }
            int64_t tmpLength = (calculatedBlockInfo[i][j][1] > calculatedBlockInfo[i][j][0]) ?
                                    calculatedBlockInfo[i][j][1] - calculatedBlockInfo[i][j][0] :
                                    0;
            if (j == 0) {
                calculatedBlockInfo[i][j][2] = tmpLength; // SUM_S1S2
            } else {
                calculatedBlockInfo[i][j][2] = calculatedBlockInfo[i][j - 1][2] + tmpLength;
            }
            calculatedBlockInfo[i][j][3] = 0; // SUM_ALL init
        }

        if (i == 0) {
            calculatedBlockInfo[0][0][3] = n2 * g * calculatedBlockInfo[0][fagS2Outer_ - 1][2];
            totalBlockInfo[0][1] = n2 * g * totalBlockInfo[0][0];
        } else {
            calculatedBlockInfo[i][0][3] =
                n2 * g * calculatedBlockInfo[i][fagS2Outer_ - 1][2] + calculatedBlockInfo[i - 1][0][3];
            totalBlockInfo[i][1] = n2 * g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
    }

    // Block split
    // Source: varlen_regbase.cpp:961-964
    int64_t fusedOuter = calculatedBlockInfo[b - 1][0][3]; // SUM_ALL
    int64_t blockFactor = (fusedOuter + aicNum - 1) / aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

    fagBlockOuter_ = static_cast<uint32_t>(blockOuter);
    fagBlockFactor_ = blockFactor;

    fagBlockStarts_[0] = 0;
    fagBlockEnds_[blockOuter - 1] = totalBlockInfo[b - 1][1];

    // Source: varlen_regbase.cpp:985-1029
    for (int64_t c = 1; c < blockOuter; c++) {
        int64_t currentIdx = std::min(c * blockFactor, fusedOuter);
        uint64_t tndS1S2PrefixSumTmp = 0;
        uint64_t tndPrefixSumTmp = 0;
        int64_t bIdx = 0, bTail = 0, n2Idx = 0, n2Tail = 0, gIdx = 0, gTail = 0;
        int64_t s1oIdx = 0, s2oIdx = 0;
        int64_t s1OuterTmp = 0;

        for (int64_t bb = 0; bb < b; bb++) {
            if (calculatedBlockInfo[bb][0][3] > currentIdx) {
                bIdx = bb;
                auto s1os2o = calculatedBlockInfo[bb][fagS2Outer_ - 1][2];
                auto gs1os2o = s1os2o * g;
                bTail = (bb == 0) ? currentIdx : currentIdx - calculatedBlockInfo[bb - 1][0][3];
                n2Idx = bTail / gs1os2o;
                n2Tail = bTail % gs1os2o;
                gIdx = n2Tail / s1os2o;
                gTail = n2Tail % s1os2o;

                // GetUnpadS1S2OuterIndex
                // Source: varlen_regbase.cpp:1127-1139
                for (int64_t i = 0; i < fagS2Outer_; i++) {
                    if (calculatedBlockInfo[bb][i][2] > gTail) {
                        s2oIdx = i;
                        int64_t s1oTail = (i == 0) ? gTail : gTail - calculatedBlockInfo[bb][i - 1][2];
                        s1oIdx = calculatedBlockInfo[bb][i][0] + s1oTail;
                        break;
                    }
                }
                s1OuterTmp = (actualSeqlenQ_[bb] + fagS1CvInner_ - 1) / fagS1CvInner_;

                fagTndStartBIdx_[c] = static_cast<int64_t>(bb);
                break;
            } else {
                tndS1S2PrefixSumTmp += static_cast<uint64_t>(actualSeqlenQ_[bb] * actualSeqlenKv_[bb]);
                int64_t s1Ot = (actualSeqlenQ_[bb] + fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT - 1) /
                               (fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT);
                int64_t s2Ot = (actualSeqlenKv_[bb] + fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT - 1) /
                               (fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT);
                tndPrefixSumTmp += static_cast<uint64_t>(s1Ot * s2Ot);
            }
        }

        if (bIdx == 0) {
            fagBlockStarts_[c] = (n2Idx * g + gIdx) * totalBlockInfo[bIdx][0] + s2oIdx * s1OuterTmp + s1oIdx;
        } else {
            fagBlockStarts_[c] = totalBlockInfo[bIdx - 1][1] + (n2Idx * g + gIdx) * totalBlockInfo[bIdx][0] +
                                 s2oIdx * s1OuterTmp + s1oIdx;
        }
        fagBlockEnds_[c - 1] = fagBlockStarts_[c];
    }

    for (uint32_t c = static_cast<uint32_t>(blockOuter); c < FAG_CORE_LIST_NUM; c++) {
        fagBlockStarts_[c] = 0;
        fagBlockEnds_[c] = 0;
        fagTndStartBIdx_[c] = 0;
    }
}

// ===== FagIsValidUnpad: check block validity for TND =====
// Source: varlen_regbase.cpp:1141-1166
bool FlashAttnMetadataCpuKernel::FagIsValidUnpad(int64_t blockIdx)
{
    int64_t resbaseIdx = blockIdx;
    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    for (int64_t bIdx = 0; bIdx < b; bIdx++) {
        int64_t actualS1Len = actualSeqlenQ_[bIdx];
        int64_t actualS2Len = actualSeqlenKv_[bIdx];
        int64_t s1OuterTmp =
            (actualS1Len + fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT - 1) / (fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT);
        int64_t s2OuterTmp =
            (actualS2Len + fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT - 1) / (fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT);
        int64_t totalBaseIdx = n2 * g * s1OuterTmp * s2OuterTmp;
        if (resbaseIdx < totalBaseIdx) {
            int64_t gDimTail = resbaseIdx % (s1OuterTmp * s2OuterTmp);
            int64_t s2oDimIdx = gDimTail / s1OuterTmp;
            int64_t s1oDimIdx = gDimTail % s1OuterTmp;
            int64_t s2IdxLeft = s2oDimIdx * fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT;
            int64_t s2IdxRight = std::min((s2oDimIdx + 1) * fagS2Inner_ * FAG_S2CV_RATIO_DEFAULT, actualS2Len);
            if (fagIsSparse_) {
                return FagCheckUnpadSparseLeftAndRight(s1oDimIdx, s2IdxLeft, s2IdxRight, bIdx);
            }
            return true;
        }
        resbaseIdx -= totalBaseIdx;
    }
    return false;
}

// ===== FagCheckUnpadSparseLeftAndRight: check sparse validity for TND =====
// Source: varlen_regbase.cpp:1168-1214
bool FlashAttnMetadataCpuKernel::FagCheckUnpadSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft,
                                                                 int64_t s2IdxRight, int64_t bIdx)
{
    int64_t actualS1Len = actualSeqlenQ_[bIdx];
    int64_t actualS2Len = actualSeqlenKv_[bIdx];
    int64_t s1 = static_cast<int64_t>(maxSeqlenQ_);

    int64_t actualCalcS1Token = fagS1Token_;
    int64_t actualCalcS2Token = fagS2Token_;
    // Token correction for RIGHT_DOWN_CAUSAL / BAND
    if (fagSparseMode_ == FAG_SPARSE_RIGHT_DOWN_CAUSAL || fagSparseMode_ == FAG_SPARSE_BAND) {
        actualCalcS1Token = fagS1Token_ + actualS1Len - actualS2Len;
        actualCalcS2Token = fagS2Token_ - actualS1Len + actualS2Len;
    }

    int64_t s2SparseLeft =
        std::max(fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT * s1oDimIdx - actualCalcS1Token, static_cast<int64_t>(0));
    s2SparseLeft = FagAlignTo(s2SparseLeft, FAG_ALIGN64);
    int64_t s2SparseRight = FagAlignTo(
        std::min(fagS1Inner_ * FAG_S1CV_RATIO_DEFAULT * (s1oDimIdx + 1), s1) + actualCalcS2Token, FAG_ALIGN64);
    s2SparseRight = std::min(s2SparseRight, actualS2Len);
    return s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft;
}

// ===== FagFillBlockInfoLoadBalance: build load balance table for TND =====
// Source: normal_regbase.cpp:1685-1750
void FlashAttnMetadataCpuKernel::FagFillBlockInfoLoadBalance(std::vector<std::vector<int64_t>> &totalBlockInfo,
                                                             std::vector<std::vector<float>> &acturalBlockInfo)
{
    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;

    acturalBlockInfo[b][0] = 0;
    acturalBlockInfo[b + 1][0] = 0;

    for (int64_t i = 0; i < b; i++) {
        int64_t actualS1Len = actualSeqlenQ_[i];
        int64_t actualS2Len = actualSeqlenKv_[i];
        auto actualS1Outer = (actualS1Len + fagS1CvInner_ - 1) / fagS1CvInner_;
        auto actualS2Outer = (actualS2Len + fagCvS2Inner_ - 1) / fagCvS2Inner_;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;

        int64_t actualCalcS1Token, actualCalcS2Token;
        FagCalcleActualToken(fagSparseMode_, fagS1Token_, fagS2Token_, actualS1Len, actualS2Len, actualCalcS1Token,
                             actualCalcS2Token);

        for (int64_t j = 0; j < fagS2Outer_; j++) {
            if (fagCvS2Inner_ * j >= actualS2Len) {
                acturalBlockInfo[i][j] = 0;
            } else {
                int64_t leftIntersectionPoint =
                    std::max(fagCvS2Inner_ * j - actualCalcS2Token, static_cast<int64_t>(0));
                int64_t cvBlockTail =
                    (fagCvS2Inner_ * (j + 1) > actualS2Len) ? actualS2Len - fagCvS2Inner_ * j : fagCvS2Inner_;
                float acturalS1Begin =
                    static_cast<float>(leftIntersectionPoint > actualS1Len ? actualS1Len : leftIntersectionPoint);
                float acturalS1End = static_cast<float>(
                    std::min(actualS1Len,
                             std::max(fagCvS2Inner_ * j + cvBlockTail + actualCalcS1Token, static_cast<int64_t>(0))));
                float acturalS1Num = acturalS1Begin > acturalS1End ? 0 : acturalS1End - acturalS1Begin;
                float acturalS2Num = static_cast<float>(cvBlockTail);
                acturalBlockInfo[i][j] =
                    acturalS1Num / static_cast<float>(fagS1CvInner_) + acturalS2Num / static_cast<float>(fagCvS2Inner_);
                acturalBlockInfo[b][0] += acturalBlockInfo[i][j] * n2 * g;
                acturalBlockInfo[b + 1][0] = (acturalBlockInfo[b + 1][0] < acturalBlockInfo[i][j]) ?
                                                 acturalBlockInfo[i][j] :
                                                 acturalBlockInfo[b + 1][0];
            }
        }
        if (i == 0) {
            totalBlockInfo[0][1] = n2 * g * totalBlockInfo[0][0];
        } else {
            totalBlockInfo[i][1] = n2 * g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
    }
}

// ===== BN2S2 split: TryBn2s2Sparse =====
// Source: normal_regbase.cpp:511-547 DoBn2s2Sparse
void FlashAttnMetadataCpuKernel::TryBn2s2Sparse()
{
    fagSplitAxis_ = FAG_SPLIT_AXIS_BN2S2;
    if (fagUseTndSplit_) {
        DoFagBn2s2SparseTndSplit();
    } else {
        DoFagBn2s2DenseSplit();
    }
}

// ===== BN2S2 dense: BNS2 合轴均匀分核 =====
// Source: normal_regbase.cpp:519-544
void FlashAttnMetadataCpuKernel::DoFagBn2s2DenseSplit()
{
    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    int64_t aicNum = static_cast<int64_t>(aicCoreNum_);

    int64_t fusedOuter = b * n2 * g * fagS2Outer_;
    if (fusedOuter == 0 || fagS1Outer_ == 0) {
        fagBlockOuter_ = 0;
        fagBlockFactor_ = 0;
        return;
    }
    int64_t bns2Factor = (fusedOuter + aicNum - 1) / aicNum;
    int64_t blockOuter = (fusedOuter + bns2Factor - 1) / bns2Factor;
    int64_t totalBlock = fusedOuter * fagS1Outer_;
    int64_t blockFactor = bns2Factor * fagS1Outer_;

    fagBlockOuter_ = static_cast<uint32_t>(blockOuter);
    fagBlockFactor_ = blockFactor;

    for (int64_t i = 0; i < blockOuter; i++) {
        fagBlockStarts_[i] = blockFactor * i;
        fagBlockEnds_[i] = std::min(blockFactor * (i + 1), totalBlock);
    }
    for (uint32_t i = static_cast<uint32_t>(blockOuter); i < FAG_CORE_LIST_NUM; i++) {
        fagBlockStarts_[i] = 0;
        fagBlockEnds_[i] = 0;
    }
}

// ===== BN2S2 sparse/TND: 二分搜索负载均衡分核 =====
// Source: varlen_regbase.cpp:1216-1314 GetBlockInfoOfBNS4TND
void FlashAttnMetadataCpuKernel::DoFagBn2s2SparseTndSplit()
{
    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    int64_t aicNum = static_cast<int64_t>(aicCoreNum_);

    if (fagS2Outer_ == 0 || b == 0) {
        fagBlockOuter_ = 0;
        fagBlockFactor_ = 0;
        return;
    }

    std::vector<std::vector<int64_t>> totalBlockInfo(b, std::vector<int64_t>(2, 0));
    std::vector<std::vector<float>> acturalBlockInfo(b + 2, std::vector<float>(fagS2Outer_ + 2, 0.0f));

    FagFillBlockInfoLoadBalance(totalBlockInfo, acturalBlockInfo);

    float maxBlockNumPerCore = Bn2s2BinarySearchMaxBlockNumPerCore(b, n2, g, aicNum, totalBlockInfo, acturalBlockInfo);

    bool success = Bn2s2CaclePerCoreBlockInfo(b, n2, g, aicNum, totalBlockInfo, acturalBlockInfo, maxBlockNumPerCore);

    if (!success) {
        fagBlockOuter_ = 0;
    } else {
        fagBlockFactor_ = static_cast<int64_t>(std::ceil(maxBlockNumPerCore));
    }
}

// Source: varlen_regbase.cpp:1247-1262 BinarySearchMaxBlockNumPerCore
float FlashAttnMetadataCpuKernel::Bn2s2BinarySearchMaxBlockNumPerCore(int64_t b, int64_t n2, int64_t g, int64_t aicNum,
                                                                      std::vector<std::vector<int64_t>> &totalBlockInfo,
                                                                      std::vector<std::vector<float>> &acturalBlockInfo)
{
    float left = acturalBlockInfo[b + 1][0];
    float right = acturalBlockInfo[b][0];
    float mid = 0;
    while (left < right - 1) {
        mid = (left + right) / 2;
        if (Bn2s2IsPossible(b, n2, g, aicNum, mid, acturalBlockInfo)) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return right;
}

// Source: varlen_regbase.cpp:1373-1412 IsPossible
bool FlashAttnMetadataCpuKernel::Bn2s2IsPossible(int64_t b, int64_t n2, int64_t g, int64_t aicNum, float possibleMax,
                                                 std::vector<std::vector<float>> &acturalBlockInfo)
{
    float currentSum = 0;
    uint64_t needCoreNum = 1;
    int64_t n2g = n2 * g;
    int64_t bn2g = b * n2g;
    for (int64_t i = 0; i < bn2g; i++) {
        int64_t bi = i / n2g;
        for (int64_t j = 0; j < fagS2Outer_; j++) {
            float num = acturalBlockInfo[bi][j];
            if (currentSum + num > possibleMax) {
                needCoreNum += 1;
                currentSum = num;
            } else {
                currentSum += num;
            }
            if (needCoreNum > static_cast<uint64_t>(aicNum)) {
                return false;
            }
        }
    }
    return true;
}

// Source: varlen_regbase.cpp:1264-1314 CaclePerCoreBlockInfo
bool FlashAttnMetadataCpuKernel::Bn2s2CaclePerCoreBlockInfo(int64_t b, int64_t n2, int64_t g, int64_t aicNum,
                                                            std::vector<std::vector<int64_t>> &totalBlockInfo,
                                                            std::vector<std::vector<float>> &acturalBlockInfo,
                                                            float maxBlockNumPerCore)
{
    float currentSum = 0;
    int64_t coreIdx = 0;
    for (int64_t bb = 0; bb < b; bb++) {
        for (int64_t n = 0; n < n2 * g; n++) {
            int64_t actualS1Outer = (actualSeqlenQ_[bb] + fagS1CvInner_ - 1) / fagS1CvInner_;
            for (int64_t j = 0; j < fagS2Outer_; j++) {
                float num = acturalBlockInfo[bb][j];
                if (coreIdx >= static_cast<int64_t>(FAG_CORE_LIST_NUM)) {
                    return false;
                } else if (currentSum + num > maxBlockNumPerCore) {
                    int64_t preBatchBlockNum = (bb == 0) ? 0 : totalBlockInfo[bb - 1][1];
                    int64_t preNGBlockNum = n * totalBlockInfo[bb][0];
                    int64_t preS2BlockNum = j * actualS1Outer;
                    fagBlockEnds_[coreIdx] = preBatchBlockNum + preNGBlockNum + preS2BlockNum;
                    fagBlockStarts_[coreIdx + 1] = fagBlockEnds_[coreIdx];
                    coreIdx += 1;
                    currentSum = num;
                    fagTndStartBIdx_[coreIdx] = static_cast<int64_t>(bb);
                } else {
                    currentSum += num;
                }
            }
        }
    }
    fagBlockStarts_[0] = 0;
    fagBlockEnds_[coreIdx] = totalBlockInfo[b - 1][1];
    fagBlockOuter_ = static_cast<uint32_t>(coreIdx + 1);

    for (uint32_t c = fagBlockOuter_; c < FAG_CORE_LIST_NUM; c++) {
        fagBlockStarts_[c] = 0;
        fagBlockEnds_[c] = 0;
    }
    return true;
}

// ===== BN2 split: TryBn2MultiBlkSparse =====
// Source: normal_regbase.cpp:622-658 DoBn2MultiBlkSparse (TND path skipped)
bool FlashAttnMetadataCpuKernel::TryBn2MultiBlkSparse()
{
    fagSplitAxis_ = FAG_SPLIT_AXIS_BN2;
    // TND-style split (genuine TND or seqused-induced) does not support BN2
    if (fagUseTndSplit_) {
        return false;
    }
    if (fagIsSparse_) {
        return DoFagBn2SparseBlockInfo();
    } else {
        DoFagBn2DenseSplit();
        return true;
    }
}

// ===== BN2 dense: BN 合轴多块均匀分核 =====
// Source: normal_regbase.cpp:629-655
void FlashAttnMetadataCpuKernel::DoFagBn2DenseSplit()
{
    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    int64_t aicNum = static_cast<int64_t>(aicCoreNum_);

    int64_t fusedOuter = b * n2 * g;
    if (fusedOuter == 0 || fagS1Outer_ == 0 || fagS2Outer_ == 0) {
        fagBlockOuter_ = 0;
        fagBlockFactor_ = 0;
        return;
    }
    int64_t blockFactor = (fusedOuter + aicNum - 1) / aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;
    blockFactor *= (fagS1Outer_ * fagS2Outer_);
    fusedOuter *= (fagS1Outer_ * fagS2Outer_);

    fagBlockOuter_ = static_cast<uint32_t>(blockOuter);
    fagBlockFactor_ = blockFactor;

    for (int64_t i = 0; i < blockOuter; i++) {
        fagBlockStarts_[i] = blockFactor * i;
        fagBlockEnds_[i] = std::min(blockFactor * (i + 1), fusedOuter);
    }
    for (uint32_t i = static_cast<uint32_t>(blockOuter); i < FAG_CORE_LIST_NUM; i++) {
        fagBlockStarts_[i] = 0;
        fagBlockEnds_[i] = 0;
    }
}

// ===== BN2 sparse: BN 合轴 sparse 分核 (含 fallback 判定) =====
// Source: normal_regbase.cpp:549-619 GetSparseBlockInfoBn2
bool FlashAttnMetadataCpuKernel::DoFagBn2SparseBlockInfo()
{
    if (fagS2Outer_ == 0) {
        fagBlockOuter_ = 0;
        fagBlockFactor_ = 0;
        return false;
    }
    std::vector<std::vector<int64_t>> parseInfo(fagS2Outer_, std::vector<int64_t>(FAG_ARRAY_LENGTH, 0));
    GetFagParseS1S2OuterInfo(parseInfo);

    int64_t s1s2oCount = parseInfo[fagS2Outer_ - 1][2];

    int64_t b = static_cast<int64_t>(batchSize_);
    int64_t n2 = static_cast<int64_t>(numHeadsKv_);
    int64_t g = fagG_;
    int64_t aicNum = static_cast<int64_t>(aicCoreNum_);

    int64_t fusedOuter = b * n2 * g;
    int64_t blockFactor = (fusedOuter + aicNum - 1) / aicNum;
    int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

    fusedOuter *= s1s2oCount;
    blockFactor *= s1s2oCount;

    fagBlockOuter_ = static_cast<uint32_t>(blockOuter);
    fagBlockFactor_ = blockFactor;

    int64_t n2gs1s2o = n2 * g * s1s2oCount;
    int64_t gs1s2o = g * s1s2oCount;

    fagBlockStarts_[0] = 0;
    fagBlockEnds_[blockOuter - 1] = b * n2 * g * fagS1Outer_ * fagS2Outer_;

    for (int64_t c = 1; c < blockOuter; c++) {
        int64_t currentIdx = std::min(c * blockFactor, fusedOuter);
        int64_t bIdx = currentIdx / n2gs1s2o;
        int64_t bTail = currentIdx % n2gs1s2o;
        int64_t n2Idx = bTail / gs1s2o;
        int64_t n2Tail = bTail % gs1s2o;
        int64_t gIdx = n2Tail / s1s2oCount;
        int64_t gTail = n2Tail % s1s2oCount;

        // Reverse map: linear valid-block index -> (s1oIdx, s2oIdx)
        int64_t s1oIdx = 0;
        int64_t s2oIdx = 0;
        int64_t preSize = 0;
        for (int64_t i = 0; i < fagS2Outer_; i++) {
            int64_t nextSize = parseInfo[i][2];
            if (gTail >= preSize && gTail < nextSize) {
                s2oIdx = i;
                s1oIdx = parseInfo[i][0] + gTail - preSize - 1;
                break;
            }
            preSize = nextSize;
        }

        fagBlockStarts_[c] = (((bIdx * n2 + n2Idx) * g + gIdx) * fagS2Outer_ + s2oIdx) * fagS1Outer_ + s1oIdx + 1;
        fagBlockEnds_[c - 1] = fagBlockStarts_[c];
    }
    for (uint32_t c = static_cast<uint32_t>(blockOuter); c < FAG_CORE_LIST_NUM; c++) {
        fagBlockStarts_[c] = 0;
        fagBlockEnds_[c] = 0;
    }

    // Fallback check: if isInvalidCol or isInvalidRow, return false to trigger BN2GS1S2 fallback
    // isInvalidCol/isInvalidRow are set by GetFagParseS1S2OuterInfo
    if (fagIsInvalidCol_ || fagIsInvalidRow_) {
        return false;
    }
    return true;
}

// ===== GenFagMetadata: serialize FAG split results to metadata tensor =====
// FAG data is appended after FA/FD data, core counts are dynamic (read from attrs)
void FlashAttnMetadataCpuKernel::GenFagMetadata(uint32_t sectionNum)
{
    auto *basePtr = static_cast<FA_METADATA_T *>(metadata_->GetData());
    // FAG start offset = Head(16) + FA(sectionNum*aicNum*16) + FD(sectionNum*aivNum*16)
    uint32_t fagOffset = METADATA_STRIDE + sectionNum * (static_cast<uint32_t>(aicCoreNum_) * METADATA_STRIDE +
                                                         static_cast<uint32_t>(aivCoreNum_) * METADATA_STRIDE);

    // Record FAG start offset in head[7]
    basePtr[HEAD_FAG_START_OFFSET_INDEX] = fagOffset;

    auto *fag = basePtr + fagOffset;
    // Zero FAG region
    std::fill_n(fag, FAG_METADATA_SIZE, static_cast<FA_METADATA_T>(0));

    // Write header
    fag[FAG_SPLIT_AXIS_INDEX] = fagSplitAxis_;
    fag[FAG_BLOCK_OUTER_INDEX] = fagBlockOuter_;
    fag[FAG_BLOCK_FACTOR_INDEX] = static_cast<int32_t>(fagBlockFactor_);
    fag[FAG_S1_OUTER_INDEX] = static_cast<int32_t>(fagS1Outer_);
    fag[FAG_S2_OUTER_INDEX] = static_cast<int32_t>(fagS2Outer_);
    fag[FAG_LAYOUT_TYPE_INDEX] = fagLayoutType_;
    fag[FAG_IS_SPARSE_INDEX] = fagIsSparse_ ? 1 : 0;
    fag[FAG_SPARSE_MODE_INDEX] = fagSparseMode_;

    // Write block arrays
    for (uint32_t i = 0; i < FAG_CORE_LIST_NUM; i++) {
        fag[FAG_BLOCK_STARTS_OFFSET + i] = static_cast<int32_t>(fagBlockStarts_[i]);
        fag[FAG_BLOCK_ENDS_OFFSET + i] = static_cast<int32_t>(fagBlockEnds_[i]);
        fag[FAG_TND_START_BIDX_OFFSET + i] = static_cast<int32_t>(fagTndStartBIdx_[i]);
    }

    // Write scalar params (passed via metadata instead of tilingData)
    fag[FAG_MASK_MODE_INDEX] = static_cast<FA_METADATA_T>(maskMode_);
    fag[FAG_WIN_LEFT_INDEX] = static_cast<FA_METADATA_T>(winLeft_);
    fag[FAG_WIN_RIGHT_INDEX] = static_cast<FA_METADATA_T>(winRight_);
    fag[FAG_MAX_SEQLEN_Q_INDEX] = static_cast<FA_METADATA_T>(maxSeqlenQ_);
    fag[FAG_MAX_SEQLEN_KV_INDEX] = static_cast<FA_METADATA_T>(maxSeqlenKv_);
}

// ===== FAG metadata end =====

} // namespace aicpu

#endif // FLASH_ATTN_GRAD_METADATA_SPLIT_H
