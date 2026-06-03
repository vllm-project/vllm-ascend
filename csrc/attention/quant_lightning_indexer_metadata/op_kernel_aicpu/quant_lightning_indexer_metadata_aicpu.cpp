/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cmath>
#include "../../quant_lightning_indexer/op_kernel/quant_lightning_indexer_metadata.h"
#include "../../common/aicpu/cpu_context_util.h"
#include "quant_lightning_indexer_metadata_aicpu.h"

using namespace optiling;

namespace aicpu {
uint32_t
QuantLightningIndexerMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes {aicCoreNum_, aivCoreNum_};
    success = BalanceSchedule(splitRes) && GenMetaData(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool QuantLightningIndexerMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    actSeqLenQ_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenQ));
    actSeqLenKey_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenKV));
    // output
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs = GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
                         GetAttrValue(ctx, "aiv_core_num", aivCoreNum_) &&
                         GetAttrValue(ctx, "soc_version", socVersion_) &&
                         GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                         GetAttrValue(ctx, "num_heads_k", numHeadsK_) &&
                         GetAttrValue(ctx, "head_dim", headDim_) &&
                         GetAttrValue(ctx, "query_quant_mode", queryQuantMode_) &&
                         GetAttrValue(ctx, "key_quant_mode", keyQuantMode_);
    if (!requiredAttrs) {
        return false;
    }

    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_k", maxSeqlenK_);
    GetAttrValueOpt(ctx, "layout_query", layoutQuery_);
    GetAttrValueOpt(ctx, "layout_key", layoutKey_);
    GetAttrValueOpt(ctx, "sparse_count", sparseCount_);
    GetAttrValueOpt(ctx, "sparse_mode", sparseMode_);
    GetAttrValueOpt(ctx, "pre_tokens", preToken_);
    GetAttrValueOpt(ctx, "next_tokens", nextToken_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);

    return (ParamsCheck() && ParamsInit());
}

bool QuantLightningIndexerMetadataCpuKernel::ParamsCheck()
{
    return (CheckSingleParam() && CheckExistence() && CheckConsistency() && CheckFeature());
}

bool QuantLightningIndexerMetadataCpuKernel::CheckSingleParam()
{
    // 基础输出校验
    KERNEL_CHECK_NULLPTR(metaData_, false, "metadata is null");
    auto metaShape = metaData_->GetTensorShape();
    KERNEL_CHECK_NULLPTR(metaShape, false, "shape of metadata is null");
    KERNEL_CHECK_NULLPTR(metaData_->GetData(), false, "data of metadata is null");
    // 核心数校验
    if (aicCoreNum_ == 0 || aivCoreNum_ == 0 || (aivCoreNum_ % aicCoreNum_ != 0)) {
        KERNEL_LOG_ERROR("Core num invalid: aic:%u, aiv:%u", aicCoreNum_, aivCoreNum_);
        return false;
    }
    // batch_size 非负校验
    if (batchSize_ < 0) {
        KERNEL_LOG_ERROR("batch_size should not be negative, but got %d", batchSize_);
        return false;
    }
    // max_seqlen_q 非负校验
    if (maxSeqlenQ_ < 0) {
        KERNEL_LOG_ERROR("max_seqlen_q should not be negative, but got %d", maxSeqlenQ_);
        return false;
    }
    // max_seqlen_k 非负校验
    if (maxSeqlenK_ < 0) {
        KERNEL_LOG_ERROR("max_seqlen_k should not be negative, but got %d", maxSeqlenK_);
        return false;
    }
    // num_heads_q 校验
    if (numHeadsQ_ != 64) {
        KERNEL_LOG_ERROR("num_heads_q should only be 64, but got %d", numHeadsQ_);
        return false;
    }
    // num_heads_k 校验
    if (numHeadsK_ != 1) {
        KERNEL_LOG_ERROR("num_heads_k should only be 1, but got %d", numHeadsK_);
        return false;
    }
    // layout_query 校验
    if (layoutQuery_ != "TND" && layoutQuery_ != "BSND") {
        KERNEL_LOG_ERROR("For layout_query, layout must be TND or BSND!");
        return false;
    }
    // layout_key 校验
    if (layoutKey_ != "PA_BSND" && layoutKey_ != "TND" && layoutKey_ != "BSND") {
	KERNEL_LOG_ERROR("For layout_key, layout must be PA_BSND/TND/BSND!");
        return false;
    }
    if (layoutQuery_ == "TND" && layoutKey_ == "BSND") {
	KERNEL_LOG_ERROR("For layout_query TND, layout_key should be PA_BSND/TND!");
	return false;
    }
    if (layoutQuery_ == "BSND" && layoutKey_ == "TND") {
	KERNEL_LOG_ERROR("For layout_query BSND, layout_key should be PA_BSND/BSND!");
        return false;
    }
    // sparse_mode 校验
    if (sparseMode_ != static_cast<uint32_t>(SparseMode::DEFAULT_MASK) &&
        sparseMode_ != static_cast<uint32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        KERNEL_LOG_ERROR("sparse_mode should be 0/3, but got %d", sparseMode_);
        return false;
    }
    // pre_tokens 校验
    if (preToken_ != INT64_MAX) {
        KERNEL_LOG_ERROR("pre_tokens should only be 2^63-1, but got %ld", preToken_);
        return false;
    }
    // next_tokens 校验
    if (nextToken_ != INT64_MAX) {
        KERNEL_LOG_ERROR("next_tokens should only be 2^63-1, but got %ld", nextToken_);
        return false;
    }
    return true;
}

bool QuantLightningIndexerMetadataCpuKernel::CheckExistence()
{
    auto isInvalid = [](Tensor* t) { return t == nullptr || t->GetData() == nullptr; };
    // Query 存在性逻辑
    if (layoutQuery_ == "TND") {
        if (isInvalid(actSeqLenQ_)) {
            KERNEL_LOG_ERROR("For layout_query TND, actual_seq_lengths_query must be provided!");
            return false;
        }
    }
    // KV 存在性逻辑
    if (layoutKey_ == "PA_BSND") {
        if (isInvalid(actSeqLenKey_)) {
            KERNEL_LOG_ERROR("For layout_key PA_BSND, actual_seq_lengths_key must be provided!");
            return false;
        }
    }
    if (layoutKey_ == "TND") {
        if (isInvalid(actSeqLenKey_)) {
            KERNEL_LOG_ERROR("For layout_key TND, actual_seq_lengths_key must be provided!");
            return false;
        }
    }
    return true;
}

int32_t QuantLightningIndexerMetadataCpuKernel::GetQueryBatchSize()
{
    if (actSeqLenQ_ != nullptr && actSeqLenQ_->GetData() != nullptr) {
        if (actSeqLenQ_->GetTensorShape() != nullptr) {
            return actSeqLenQ_->GetTensorShape()->GetDimSize(0);
        }
    }
    return batchSize_;
}

int32_t QuantLightningIndexerMetadataCpuKernel::GetKvBatchSize()
{
    if (actSeqLenKey_ != nullptr && actSeqLenKey_->GetData() != nullptr) {
        if (actSeqLenKey_->GetTensorShape() != nullptr) {
            return actSeqLenKey_->GetTensorShape()->GetDimSize(0);
        }
    }
    return batchSize_;
}

bool QuantLightningIndexerMetadataCpuKernel::CheckConsistency()
{
    int32_t queryBatchSize = GetQueryBatchSize();
    int32_t kvBatchSize = GetKvBatchSize();
    if ((layoutQuery_ == "BSND" || (layoutQuery_ == "TND" && layoutKey_ == "TND")) && queryBatchSize != kvBatchSize) {
	KERNEL_LOG_ERROR("For the layout_query is BSND or both layout_query and layout_key are TND, the dim of actual_seq_lengths_query and the dim of actual_seq_lengths_key should be equal.");
        return false;
    }
    if (std::abs(queryBatchSize - kvBatchSize) > 1) {
	KERNEL_LOG_ERROR("The difference between the dim of actual_seq_lengths_query and the dim of actual_seq_lengths_key should not be greater than 1.");
        return false;
    }
    return true;
}

bool QuantLightningIndexerMetadataCpuKernel::CheckFeature()
{
    // 压缩率校验
    if (cmpRatio_ < 1 || cmpRatio_ > 128) {
        KERNEL_LOG_ERROR("cmp_ratio should be [1, 128], but got %d", cmpRatio_);
        return false;
    }
    validSocVersion_ = ProcessSocVersion();
    if (validSocVersion_ == ValidSocVersion::ASCEND910B) {
        // 校验 2 的幂次方: 1, 2, 4, ..., 128
        if ((cmpRatio_ & (cmpRatio_ - 1)) != 0) {
            KERNEL_LOG_ERROR("For Atlas A3, cmp_ratio should be 1/2/4/8/16/32/64/128, but got %d", cmpRatio_);
            return false;
        }
    } else {
        if (cmpRatio_ != 1 && cmpRatio_ != 4 && cmpRatio_ != 128) {
            KERNEL_LOG_ERROR("For Ascend950, cmp_ratio should be 1/4/128, but got %d", cmpRatio_);
            return false;
        }
    }
    return true;
}

ValidSocVersion QuantLightningIndexerMetadataCpuKernel::ProcessSocVersion()
{
    const std::string ascend950 = "Ascend950";
    if (socVersion_.find(ascend950) != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    } else {
        return ValidSocVersion::ASCEND910B;
    }

    return ValidSocVersion::RESERVED_VERSION;
}

bool QuantLightningIndexerMetadataCpuKernel::ParamsInit()
{
    int32_t qBatchSize_ = GetQueryBatchSize();
    int32_t kvBatchSize_ = GetKvBatchSize();
    isActQBatchPlus = (qBatchSize_ > kvBatchSize_);
    auto mode = static_cast<SparseMode>(sparseMode_);
    if (mode == SparseMode::RIGHT_DOWN_CAUSAL) {
        attentionMode_ = 1;
        preToken_ = INT64_MAX;
    } else if (mode == SparseMode::DEFAULT_MASK) {
        attentionMode_ = 0;
    } else if (mode == SparseMode::BAND) {
        attentionMode_ = 1;
    }
    groupSize_ = numHeadsQ_ / numHeadsK_;
    batchSize_ = std::min(qBatchSize_, kvBatchSize_);
    validSocVersion_ = ProcessSocVersion();
    if (validSocVersion_ == ValidSocVersion::ASCEND910B){
        s2BaseSize_ = 2048U; // 仅用于A3
    } else if (validSocVersion_ == ValidSocVersion::ASCEND950){
        s2BaseSize_ = 128U; // 仅用于A5
    } else {
        s2BaseSize_ = 128U; // 其他情况
    }
    return true;
}

uint32_t QuantLightningIndexerMetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    if (actSeqLenQ_ == nullptr || actSeqLenQ_->GetData() == nullptr) {
        return static_cast<uint32_t>(maxSeqlenQ_);
    }
    const int32_t *s1Ptr = (int32_t*)actSeqLenQ_->GetData();
    if (layoutQuery_ == "TND") {
        if (isActQBatchPlus) {
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
        return (bIdx == 0) ? static_cast<uint32_t>(s1Ptr[bIdx]) :
            static_cast<uint32_t>(s1Ptr[bIdx] - s1Ptr[bIdx - 1U]);
    } else {
        return static_cast<uint32_t>(s1Ptr[bIdx]);
    }
}

uint32_t QuantLightningIndexerMetadataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    if (actSeqLenKey_ == nullptr || actSeqLenKey_->GetData() == nullptr) {
        return static_cast<uint32_t>(maxSeqlenK_ * cmpRatio_);
    }
    const int32_t *s2Ptr = (int32_t*)actSeqLenKey_->GetData();
    if (layoutKey_ == "TND") {
        return (bIdx == 0) ? static_cast<uint32_t>(s2Ptr[bIdx]) :
            static_cast<uint32_t>(s2Ptr[bIdx] - s2Ptr[bIdx - 1U]);
    } else {
        return static_cast<uint32_t>(s2Ptr[bIdx]);
    }
}

void QuantLightningIndexerMetadataCpuKernel::CalcSplitInfo(SplitContext &splitContext)
{
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        uint32_t s2Size = GetS2SeqSize(bIdx) / cmpRatio_;
        splitInfo.s1GBaseNum[bIdx] = (s1Size * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (s1Size * groupSize_) % mBaseSize_;
        splitInfo.s2BaseNum[bIdx] = (s2Size + s2BaseSize_ - 1U) / s2BaseSize_;
        splitInfo.s2TailSize[bIdx] = s2Size % s2BaseSize_;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
    return;
}

int64_t QuantLightningIndexerMetadataCpuKernel::CalcPreTokenLeftUp(
    uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(sparseMode_);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

int64_t QuantLightningIndexerMetadataCpuKernel::CalcNextTokenLeftUp(
    uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(sparseMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return nextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + nextToken_;
        default:
            return nextToken_;
    }
}

int64_t QuantLightningIndexerMetadataCpuKernel::CalcCost(
    uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignBasicM = basicM / groupSize_;
    uint32_t alignBasicS2 = 1;
    if (validSocVersion_ == ValidSocVersion::ASCEND910B) {
        if (basicS2 <= 1024) {
            alignBasicS2 = (1536 + basicS2) / 2;
        } else {
            alignBasicS2 = (2560 + basicS2) / 2;
        }
        return static_cast<int64_t>(alignBasicM * alignBasicS2);
    } else {
        uint32_t alignCoefM = 16U;
        uint32_t alignCoefS2 = 64U;
        alignBasicM = (basicM + alignCoefM - 1U) >> 4U;      // 按alignCoefM对齐，向上取整，4：移位操作实现除16
        alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U;   // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
        return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2);
    }
}

BlockCost<int64_t> QuantLightningIndexerMetadataCpuKernel::CalcCostTable(uint32_t s1NormalSize,
    uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost {};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

Range<int64_t> QuantLightningIndexerMetadataCpuKernel::CalcS2TokenRange(
        uint32_t s1GIdx, const BatchCache &batchCache) {

    // no mask
    if (!attentionMode_) { //attentionMaskFlag ?
        return std::make_pair(0, static_cast<int64_t>(batchCache.s2Size));
    }
    // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
        static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(groupSize_)) - 1;
    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (isS1G_) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(groupSize_);
        s1LastToken = s1GLastToken / static_cast<int64_t>(groupSize_);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            // start and end locate in tow or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    return std::make_pair(s2FirstToken, s2LastToken);
}

void QuantLightningIndexerMetadataCpuKernel::CalcBatchCache(
    uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.s2Size = GetS2SeqSize(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.typeCost = CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[bIdx],
        splitInfo.s2TailSize[bIdx]);
}

void QuantLightningIndexerMetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx,
    const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0 || splitInfo.s2BaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    auto s2TokenRange = CalcS2TokenRange(s1GIdx, batchCache);
    int64_t s2FirstToken = s2TokenRange.first;
    int64_t s2LastToken = s2TokenRange.second;
    // get valid range
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));

    // get block start & end
    uint32_t s2CmpLength = (s2LastToken - s2FirstToken + 1) / cmpRatio_;
    if (s2CmpLength == 0) {
        s1GCache.s2Start = 0U;
        s1GCache.s2End = 0U;
    } else {
        s1GCache.s2Start = 0U;
        s1GCache.s2End = (s2CmpLength + s2BaseSize_ - 1) / s2BaseSize_; // end of block index, Right-open interval
    }

    if (s1GCache.s2Start >= s1GCache.s2End) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    uint32_t s2TailSize = s2CmpLength % s2BaseSize_;

    // 计算S2方向满块、尾块数量
    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num = s2TailSize != 0 ? 1U : 0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;

    BlockCost<int64_t> typeCost = CalcCostTable(mBaseSize_, s2BaseSize_,
                                                splitInfo.s1GTailSize[batchCache.bIdx], s2TailSize);

    if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) &&
               splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost = typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                           typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? typeCost[TAIL_BLOCK][TAIL_BLOCK] :
                                    typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost = typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
                           typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? typeCost[NORMAL_BLOCK][TAIL_BLOCK] :
                                    typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }
}

void QuantLightningIndexerMetadataCpuKernel::CalcBatchCost(
    uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (GetS1SeqSize(bIdx) == 0U || GetS2SeqSize(bIdx) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;
        // 更新最大S1G行开销
        if (s1GCache.s1GCost > costInfo.maxS1GCost) {
            costInfo.maxS1GCost = s1GCache.s1GCost;
        }

        if(s1GCache.s1GBlock > 0){
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void QuantLightningIndexerMetadataCpuKernel::CalcCostInfo(SplitContext &splitContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    // 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * numHeadsK_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * numHeadsK_;
    }
}

void QuantLightningIndexerMetadataCpuKernel::UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) {    // 边界assignInfo.s2End是取不到的开区间
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }

    // Update S1G
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }

    // Update Batch
    if (assignContext.curBN2Idx == batchSize_ * numHeadsK_) {  // 所有负载全部分配完，设置最后一个核的右开区间，返回
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / numHeadsK_ != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / numHeadsK_;
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }

    // Update Cache
    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void QuantLightningIndexerMetadataCpuKernel::AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo &costInfo = splitContext.costInfo;
    while (assignContext.bN2Cost == 0 || IsWithinTolerance(assignContext.coreCache.costLimit,
        costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        // to the end
        if (assignContext.curBN2Idx == batchSize_ * numHeadsK_) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / numHeadsK_ != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / numHeadsK_;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void QuantLightningIndexerMetadataCpuKernel::AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
        assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                assignContext.bN2Cost - assignContext.s1GCache.s1GCost : 0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                 assignContext.bN2Block - assignContext.s1GCache.s1GBlock : 0U;
        // 计算新一行的信息
        do{
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        }while(assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void QuantLightningIndexerMetadataCpuKernel::AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
            assignContext.coreCache.cost + curCost)) { // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
    }
}

void QuantLightningIndexerMetadataCpuKernel::ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    // 当前batch被分配一块出去，更新剩余负载
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    // 当前行被分配一块出去，更新剩余负载
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext);
}

bool QuantLightningIndexerMetadataCpuKernel::IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes)
{
    // 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    // 核0无需处理
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 无跨核行，无需处理
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    // 需要归约的行还未处理完
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void QuantLightningIndexerMetadataCpuKernel::RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / numHeadsK_;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
                            (s1Size * groupSize_ - splitS1GIdx * mBaseSize_) : mBaseSize_;
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.fdRes.fdBN2Idx[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.fdMIdx[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
        result.fdRes.fdWorkspaceIdx[result.numOfFdHead] = assignContext.preFdDataNum;
    result.fdRes.fdS2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.fdMSize[result.numOfFdHead] = curFdS1gSize / groupSize_;
    result.numOfFdHead++;
}

void QuantLightningIndexerMetadataCpuKernel::AssignBlockToCore(uint32_t coreNum, const SplitContext &splitContext,
                                                               AssignContext &assignContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
        result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] = assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;
    assignContext.coreCache = {};
    assignContext.coreCache.costLimit = assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);
    if (!supportFd_){
        assignContext.coreCache.costLimit = costInfo.maxS1GCost > assignContext.coreCache.costLimit ? costInfo.maxS1GCost :assignContext.coreCache.costLimit;
    }
    // 1、按整batch分配
    AssignByBatch(splitContext, assignContext);
    // 2、按行分配
    AssignByRow(splitContext, assignContext);
    // 3、按块分配
    if (supportFd_){
    AssignByBlock(splitContext, assignContext);
        // 4、强制分配
        if (assignContext.coreCache.block == 0) {
            ForceAssign(splitContext, assignContext);
        }
    }
    result.bN2End[assignContext.curCoreIdx] = assignContext.curBN2Idx;
    result.gS1End[assignContext.curCoreIdx] = assignContext.curS1GIdx;
    result.s2End[assignContext.curCoreIdx] = assignContext.curS2Idx;
    result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
    assignContext.unassignedCost -= assignContext.coreCache.cost;
    // 对之前的归约信息进行记录并清理
    if (supportFd_ && IsNeedRecordFDInfo(assignContext, result)) {
        RecordFDInfo(splitContext, assignContext, result);
            assignContext.preFdDataNum += assignContext.curKvSplitPart;
        assignContext.curKvSplitPart = 1U;
    }
    // 更新S2切分信息
    if (supportFd_ && assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
        assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
        assignContext.curKvSplitPart++;
    }
}

void QuantLightningIndexerMetadataCpuKernel::CalcSplitPlan(uint32_t coreNum,
    int64_t costLimit, const SplitContext &splitContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    if (coreNum == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;
    AssignContext assignContext {};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    for (uint32_t i = 0; i < coreNum; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }
        assignContext.curCoreIdx = i;
        AssignBlockToCore(coreNum, splitContext, assignContext, result);
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void QuantLightningIndexerMetadataCpuKernel::SplitFD(SplitResult &splitRes)
{
    // 计算FD的总数据量
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    // 计算每个核处理的load
    uint64_t averageLoad = (totalFDLoad + aivCoreNum_ - 1U) / aivCoreNum_; //向上取整，避免核负载为0
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i] / averageLoad; // 计算当前归约任务所用核数，向下取整，避免使用核数超出总核数
        uint32_t curAveMSize = (splitRes.fdRes.fdMSize[i] + curFDVectorNum - 1U) / curFDVectorNum; // 计算当前归约任务每个核的行数，向上取整，避免行数为0
        curFDVectorNum = (splitRes.fdRes.fdMSize[i] + curAveMSize -1U)/ curAveMSize;
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = vid * curAveMSize;
            splitRes.fdRes.fdMNum[curCoreIndex] =
                (vid < curFDVectorNum - 1) ? curAveMSize : (splitRes.fdRes.fdMSize[i] - vid * curAveMSize);
            curCoreIndex++;
        }
    }
    splitRes.fdRes.fdUsedVecNum = curCoreIndex;
}

bool QuantLightningIndexerMetadataCpuKernel::BalanceSchedule(SplitResult &splitRes)
{
    SplitContext splitContext(batchSize_);
    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = batchSize_ * numHeadsK_;
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        return true;
    }
    CalcCostInfo(splitContext);

    splitRes.maxCost = INT64_MAX;
    splitRes.usedCoreNum = 1U;
    CalcSplitPlan(aicCoreNum_, splitRes.maxCost, splitContext, splitRes);
    // 3、存在FD任务，对FD进行负载均衡分配
    if (supportFd_ && splitRes.numOfFdHead > 0U) {
        SplitFD(splitRes);
    }
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U);  // 至少使用1个core
    return true;
}

bool QuantLightningIndexerMetadataCpuKernel::GenMetaData(SplitResult &splitRes)
{
    optiling::detail::QliMetaData* metaDataPtr = (optiling::detail::QliMetaData*)metaData_->GetData();
    // LI Metadata Generate
    for (size_t i = 0; i < aicCoreNum_; ++i) {
        if (i >= splitRes.usedCoreNum) {
            metaDataPtr->LIMetadata[i][LI_CORE_ENABLE_INDEX] = 0; // AIC disenable
            continue;
        }
        metaDataPtr->LIMetadata[i][LI_CORE_ENABLE_INDEX] = 1; // AIC enable
        // FA START
        metaDataPtr->LIMetadata[i][LI_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i-1];
        metaDataPtr->LIMetadata[i][LI_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i-1];
        metaDataPtr->LIMetadata[i][LI_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i-1];
        // FA END
        metaDataPtr->LIMetadata[i][LI_BN2_END_INDEX] = splitRes.bN2End[i];
        metaDataPtr->LIMetadata[i][LI_M_END_INDEX] = splitRes.gS1End[i];
        metaDataPtr->LIMetadata[i][LI_S2_END_INDEX] = splitRes.s2End[i];
        //
        metaDataPtr->LIMetadata[i][LI_FIRST_LD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
    }

    // LD Metadata Generate
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            metaDataPtr->LDMetadata[i][LD_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        metaDataPtr->LDMetadata[i][LD_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        metaDataPtr->LDMetadata[i][LD_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        metaDataPtr->LDMetadata[i][LD_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        metaDataPtr->LDMetadata[i][LD_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        metaDataPtr->LDMetadata[i][LD_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        metaDataPtr->LDMetadata[i][LD_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        metaDataPtr->LDMetadata[i][LD_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    return true;
}

namespace {
    static const char *kernelType = "QuantLightningIndexerMetadata";
    REGISTER_CPU_KERNEL(kernelType, QuantLightningIndexerMetadataCpuKernel);
}
}; // namespace aicpu
