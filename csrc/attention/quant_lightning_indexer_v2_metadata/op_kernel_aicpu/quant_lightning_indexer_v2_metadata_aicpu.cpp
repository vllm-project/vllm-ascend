/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quant_lightning_indexer_v2_metadata_aicpu.h"
#include <cstdio>
#include <cmath>
#include "status.h"

using namespace optiling;

namespace aicpu {
uint32_t QuantLightningIndexerV2MetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes{aicCoreNum_, aivCoreNum_};
    success = BalanceSchedule(splitRes) && GenMetadata(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool QuantLightningIndexerV2MetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenQ));
    cuSeqlensK_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenK));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedQ));
    sequsedK_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedK));
    cmpResidualK_ = ctx.Input(static_cast<uint32_t>(ParamId::cmpResidualK));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metadata));

    bool requiredAttrs = GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
                         GetAttrValue(ctx, "aiv_core_num", aivCoreNum_) &&
                         GetAttrValue(ctx, "soc_version", socVersion_) &&
                         GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_k", numHeadsK_) &&
                         GetAttrValue(ctx, "head_dim", headDim_) && GetAttrValue(ctx, "topk", topk_);
    if (!requiredAttrs) {
        return false;
    }

    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_k", maxSeqlenK_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_k", layoutK_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);

    return (ParamsCheck() && ParamsInit());
}

bool QuantLightningIndexerV2MetadataCpuKernel::ParamsCheck()
{
    // 校验输出 metadata 是否为空
    if (metadata_ == nullptr) {
        KERNEL_LOG_ERROR("Output metadata is nullptr");
        return false;
    } else if (metadata_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("Output metadata data is nullptr");
        return false;
    }
    int32_t batchSize = GetQueryBatchSize();
    // 校验 cu_seqlens_q 元素
    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t *cuSeqlensQPtr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
            // 校验 cu_seqlens_q 首元素为 0
            if (cuSeqlensQPtr[0] != 0) {
                KERNEL_LOG_ERROR("The first element of cu_seqlens_q should be 0, but got %d", cuSeqlensQPtr[0]);
                return false;
            }
            for (int i = 0; i < batchSize + 1; i++) {
                // 校验 cu_seqlens_q 元素递增
                if (i > 0 && cuSeqlensQPtr[i - 1] > cuSeqlensQPtr[i]) {
                    KERNEL_LOG_ERROR("The elements in cu_seqlens_q must be in ascending order, "
                                     "but got cu_seqlens_q[%d] = %d, cu_seqlens_q[%d] = %d",
                                     i - 1, cuSeqlensQPtr[i - 1], i, cuSeqlensQPtr[i]);
                    return false;
                }
            }
        }
    }
    // 校验 cu_seqlens_k 元素
    if (layoutK_ == "TND") {
        if (cuSeqlensK_ != nullptr && cuSeqlensK_->GetData() != nullptr) {
            const int32_t *cuSeqlensKPtr = static_cast<const int32_t *>(cuSeqlensK_->GetData());
            // 校验 cu_seqlens_k 首元素为 0
            if (cuSeqlensKPtr[0] != 0) {
                KERNEL_LOG_ERROR("The first element of cu_seqlens_k should be 0, but got %d", cuSeqlensKPtr[0]);
                return false;
            }
            for (int i = 0; i < batchSize + 1; i++) {
                // 校验 cu_seqlens_k 元素递增
                if (i > 0 && cuSeqlensKPtr[i - 1] > cuSeqlensKPtr[i]) {
                    KERNEL_LOG_ERROR("The elements in cu_seqlens_k must be in ascending order, "
                                     "but got cu_seqlens_k[%d] = %d, cu_seqlens_k[%d] = %d",
                                     i - 1, cuSeqlensKPtr[i - 1], i, cuSeqlensKPtr[i]);
                    return false;
                }
            }
        }
    }
    // 校验 seqused_q 元素非负
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t *sequsedQPtr = static_cast<const int32_t *>(sequsedQ_->GetData());
        const int32_t *cuSeqlensQPtr = (layoutQ_ == "TND" && cuSeqlensQ_ != nullptr &&
                                        cuSeqlensQ_->GetData() != nullptr) ?
                                           static_cast<const int32_t *>(cuSeqlensQ_->GetData()) : nullptr;
        for (int i = 0; i < batchSize; i++) {
            if (sequsedQPtr[i] < 0) {
                KERNEL_LOG_ERROR("The elements in seqused_q should be >= 0, but got seqused_q[%d] = %d", i,
                                 sequsedQPtr[i]);
                return false;
            }
            // 校验 seqused_q 元素不大于 max_seqlen_q (BSND) 或 cu_seqlens_q 序列长度 (TND)
            if (layoutQ_ == "BSND" && sequsedQPtr[i] > maxSeqlenQ_) {
                KERNEL_LOG_ERROR("The elements in seqused_q should not be greater than max_seqlen_q %d, "
                                 "but got seqused_q[%d] = %d", maxSeqlenQ_, i, sequsedQPtr[i]);
                return false;
            }
            if (cuSeqlensQPtr != nullptr) {
                int32_t seqLen = cuSeqlensQPtr[i + 1] - cuSeqlensQPtr[i];
                if (sequsedQPtr[i] > seqLen) {
                    KERNEL_LOG_ERROR("The elements in seqused_q should not be greater than the sequence length "
                                     "from cu_seqlens_q %d, but got seqused_q[%d] = %d", seqLen, i, sequsedQPtr[i]);
                    return false;
                }
            }
        }
    }
    // 校验 seqused_k 元素非负
    if (sequsedK_ != nullptr && sequsedK_->GetData() != nullptr) {
        const int32_t *sequsedKPtr = static_cast<const int32_t *>(sequsedK_->GetData());
        const int32_t *cuSeqlensKPtr = (layoutK_ == "TND" && cuSeqlensK_ != nullptr &&
                                        cuSeqlensK_->GetData() != nullptr) ?
                                           static_cast<const int32_t *>(cuSeqlensK_->GetData()) : nullptr;
        for (int i = 0; i < batchSize; i++) {
            if (sequsedKPtr[i] < 0) {
                KERNEL_LOG_ERROR("The elements in seqused_k should be >= 0, but got seqused_k[%d] = %d", i,
                                 sequsedKPtr[i]);
                return false;
            }
            // 校验 seqused_k 元素不大于 max_seqlen_k (BSND) 或 cu_seqlens_k 序列长度 (TND)
            if (layoutK_ == "BSND" && sequsedKPtr[i] > maxSeqlenK_) {
                KERNEL_LOG_ERROR("The elements in seqused_k should not be greater than max_seqlen_k %d, "
                                 "but got seqused_k[%d] = %d", maxSeqlenK_, i, sequsedKPtr[i]);
                return false;
            }
            if (cuSeqlensKPtr != nullptr) {
                int32_t seqLen = cuSeqlensKPtr[i + 1] - cuSeqlensKPtr[i];
                if (sequsedKPtr[i] > seqLen) {
                    KERNEL_LOG_ERROR("The elements in seqused_k should not be greater than the sequence length "
                                     "from cu_seqlens_k %d, but got seqused_k[%d] = %d", seqLen, i, sequsedKPtr[i]);
                    return false;
                }
            }
        }
    }
    // 校验 cmp_residual_k 元素
    if (cmpResidualK_ != nullptr && cmpResidualK_->GetData() != nullptr) {
        const int32_t *cmpResidualKPtr = static_cast<const int32_t *>(cmpResidualK_->GetData());
        for (int i = 0; i < batchSize; i++) {
            if (cmpResidualKPtr[i] < 0 || cmpResidualKPtr[i] >= cmpRatio_) {
                KERNEL_LOG_ERROR("The elements in cmp_residual_k should be in [0, cmpRatio_(%d)), but got "
                                 "cmp_residual_k[%d] = %d", cmpRatio_,
                                 i, cmpResidualKPtr[i]);
                return false;
            }
        }
    }
    return true;
}

int32_t QuantLightningIndexerV2MetadataCpuKernel::GetQueryBatchSize()
{
    // 1. 如果sequsedQ_传了，使用sequsedQ_获取BatchSize
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        if (sequsedQ_->GetTensorShape() != nullptr) {
            return sequsedQ_->GetTensorShape()->GetDimSize(0);
        }
    }
    // 2. sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_获取BatchSize
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            if (cuSeqlensQ_->GetTensorShape() != nullptr) {
                return cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1;
            }
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensQ_ 为空，使用batchSize_
    return batchSize_;
}

ValidSocVersion QuantLightningIndexerV2MetadataCpuKernel::ProcessSocVersion()
{
    const std::string ascend950 = "Ascend950";
    if (socVersion_.find(ascend950) != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    } else {
        return ValidSocVersion::ASCEND910B;
    }
}

bool QuantLightningIndexerV2MetadataCpuKernel::ParamsInit()
{
    batchSize_ = GetQueryBatchSize();
    auto mode = static_cast<SparseMode>(maskMode_);
    if (mode == SparseMode::RIGHT_DOWN_CAUSAL) {
        attentionMode_ = 1;
        preToken_ = INT64_MAX;
    } else if (mode == SparseMode::DEFAULT_MASK) {
        attentionMode_ = 0;
    } else if (mode == SparseMode::BAND) {
        attentionMode_ = 1;
    }
    groupSize_ = numHeadsQ_ / numHeadsK_;
    ValidSocVersion validSocVersion = ProcessSocVersion();
    if (validSocVersion == ValidSocVersion::ASCEND910B) {
        mBaseSize_ = s1BaseSize_ * groupSize_;
        s2BaseSize_ = 2048U;
    } else if (validSocVersion == ValidSocVersion::ASCEND950) {
        if (topk_ > TOPK_6K) {
            s1BaseSize_ = S1_BASE_SIZE_SMALL;
        }
        mBaseSize_ = s1BaseSize_ * groupSize_;
        s2BaseSize_ = 128U;
    } else {
        mBaseSize_ = s1BaseSize_ * groupSize_;
        s2BaseSize_ = 128U;
    }
    return true;
}

uint32_t QuantLightningIndexerV2MetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    // 1. 如果 sequsedQ_ 传了，直接使用
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // 2. sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t *s1Ptr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    // 3. 如果不是 TND，或者 cuSeqlensQ_ 为空，使用 querySeqSize_
    return static_cast<uint32_t>(maxSeqlenQ_);
}

uint32_t QuantLightningIndexerV2MetadataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    // 如果 seqUsedKv_ 传了，直接使用
    if (sequsedK_ != nullptr && sequsedK_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedK_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // seqUsedKv_ 没传，判断 Layout
    if (layoutK_ == "TND") {
        // 如果是 TND，尝试使用 actSeqLenOriKv_
        if (cuSeqlensK_ != nullptr && cuSeqlensK_->GetData() != nullptr) {
            const int32_t *s2Ptr = static_cast<const int32_t *>(cuSeqlensK_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 使用 max_seqlen_k
    return static_cast<uint32_t>(maxSeqlenK_);
}

uint64_t QuantLightningIndexerV2MetadataCpuKernel::GetRevertS2Size(uint32_t bIdx)
{
    uint32_t cmpS2Size = GetS2SeqSize(bIdx);
    if (cmpResidualK_ != nullptr && cmpResidualK_->GetData() != nullptr) {
        const int32_t *residualPtr = static_cast<const int32_t *>(cmpResidualK_->GetData());
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_) + residualPtr[bIdx];
    } else {
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_);
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::CalcSplitInfo(SplitContext &splitContext)
{
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        uint32_t s2Size = GetS2SeqSize(bIdx);
        maxS2Size_ = std::max(maxS2Size_, s2Size);
        splitInfo.s1GBaseNum[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_) % mBaseSize_;
        splitInfo.s2BaseNum[bIdx] = (s2Size + s2BaseSize_ - 1U) / s2BaseSize_;
        splitInfo.s2TailSize[bIdx] = s2Size % s2BaseSize_;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
    ValidSocVersion validSocVersion = ProcessSocVersion();
    if (validSocVersion == ValidSocVersion::ASCEND950) {
        if (maxS2Size_ < topk_) {
            supportFd_ = false;
            return;
        }
        if (maxS2Size_ > fdToleranceRatio * s2BaseSize_) {
            supportFd_ = true;
            return;
        }
    }
}

int64_t QuantLightningIndexerV2MetadataCpuKernel::CalcPreTokenLeftUp(uint32_t s1Size, uint64_t s2Size)
{
    auto mode = static_cast<SparseMode>(maskMode_);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

int64_t QuantLightningIndexerV2MetadataCpuKernel::CalcNextTokenLeftUp(uint32_t s1Size, uint64_t s2Size)
{
    auto mode = static_cast<SparseMode>(maskMode_);
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

int64_t QuantLightningIndexerV2MetadataCpuKernel::CalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U; // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U; // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(COST_WEIGHT_M * alignBasicM + COST_WEIGHT_S2 * alignBasicS2);
}

BlockCost<int64_t> QuantLightningIndexerV2MetadataCpuKernel::CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize,
                                                                           uint32_t s1GTailSize, uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost{};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

Range<int64_t> QuantLightningIndexerV2MetadataCpuKernel::CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache)
{
    // no mask
    if (!attentionMode_) {
        return std::make_pair(0, static_cast<int64_t>(batchCache.revertS2Size));
    }

    // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
                                    static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(groupSize_)) -
                           1;
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
            // start and end locate in two or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    return std::make_pair(s2FirstToken, s2LastToken);
}

void QuantLightningIndexerV2MetadataCpuKernel::CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext,
                                                              BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.revertS2Size = GetRevertS2Size(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.revertS2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.revertS2Size);
}

void QuantLightningIndexerV2MetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext,
                                                            const BatchCache &batchCache, S1GCache &s1GCache)
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
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.revertS2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.revertS2Size - 1U));

    // get block start & end
    int64_t s2CmpLength = (s2LastToken - s2FirstToken + 1) / cmpRatio_;
    if (s2CmpLength <= 0) {
        s1GCache.s2Start = 0U;
        s1GCache.s2End = 0U;
    } else {
        s1GCache.s2Start = 0U;
        // end of block index, Right-open interval
        s1GCache.s2End = (static_cast<uint32_t>(s2CmpLength) + s2BaseSize_ - 1) / s2BaseSize_;
    }

    if (s1GCache.s2Start >= s1GCache.s2End) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    uint32_t s2TailSize = static_cast<uint32_t>(s2CmpLength) % s2BaseSize_;

    // 计算S2方向满块、尾块数量
    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num = s2TailSize != 0 ? 1U : 0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;

    BlockCost<int64_t> typeCost =
        CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[batchCache.bIdx], s2TailSize);

    if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost =
            typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num + typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost =
            curTailS2Num > 0U ? typeCost[TAIL_BLOCK][TAIL_BLOCK] : typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost =
            typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num + typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost =
            curTailS2Num > 0U ? typeCost[NORMAL_BLOCK][TAIL_BLOCK] : typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext,
                                                             CostInfo &costInfo)
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

        if (s1GCache.s1GBlock > 0) {
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::CalcCostInfo(SplitContext &splitContext)
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

void QuantLightningIndexerV2MetadataCpuKernel::UpdateCursor(const SplitContext &splitContext,
                                                            AssignContext &assignContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) { // 边界assignInfo.s2End是取不到的开区间
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
    if (assignContext.curBN2Idx == batchSize_ * numHeadsK_) { // 所有负载全部分配完，设置最后一个核的右开区间，返回
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

void QuantLightningIndexerV2MetadataCpuKernel::AssignByBatch(const SplitContext &splitContext,
                                                             AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo &costInfo = splitContext.costInfo;
    while (assignContext.bN2Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
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

void QuantLightningIndexerV2MetadataCpuKernel::AssignByRow(const SplitContext &splitContext,
                                                           AssignContext &assignContext)
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
                                    assignContext.bN2Cost - assignContext.s1GCache.s1GCost :
                                    0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                     assignContext.bN2Block - assignContext.s1GCache.s1GBlock :
                                     0U;
        // 计算新一行的信息
        do {
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        } while (assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void QuantLightningIndexerV2MetadataCpuKernel::AssignByBlock(const SplitContext &splitContext,
                                                             AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + curCost)) {
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

void QuantLightningIndexerV2MetadataCpuKernel::ForceAssign(const SplitContext &splitContext,
                                                           AssignContext &assignContext)
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

bool QuantLightningIndexerV2MetadataCpuKernel::IsNeedRecordFDInfo(const AssignContext &assignContext,
                                                                  const SplitResult &splitRes)
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

void QuantLightningIndexerV2MetadataCpuKernel::RecordFDInfo(const SplitContext &splitContext,
                                                            const AssignContext &assignContext, SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / numHeadsK_;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize =
        (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
            (static_cast<uint64_t>(s1Size) * groupSize_ - static_cast<uint64_t>(splitS1GIdx) * mBaseSize_) :
            mBaseSize_;
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

void QuantLightningIndexerV2MetadataCpuKernel::AssignBlockToCore(const SplitContext &splitContext,
                                                                 AssignContext &assignContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] =
        assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;
    assignContext.coreCache = {};
    assignContext.coreCache.costLimit = assignContext.unassignedCost / (aicCoreNum_ - assignContext.curCoreIdx);
    if (!supportFd_) {
        assignContext.coreCache.costLimit = costInfo.maxS1GCost > assignContext.coreCache.costLimit ?
                                                costInfo.maxS1GCost :
                                                assignContext.coreCache.costLimit;
    }
    // 1、按整batch分配
    AssignByBatch(splitContext, assignContext);
    // 2、按行分配
    AssignByRow(splitContext, assignContext);
    // 3、按块分配
    if (supportFd_) {
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

void QuantLightningIndexerV2MetadataCpuKernel::CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext,
                                                             SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    if (aicCoreNum_ == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;
    AssignContext assignContext{};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    for (uint32_t i = 0; i < aicCoreNum_; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }
        assignContext.curCoreIdx = i;
        AssignBlockToCore(splitContext, assignContext, result);
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void QuantLightningIndexerV2MetadataCpuKernel::SplitFD(SplitResult &splitRes)
{
    // 计算FD的总数据量
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    // 计算当前最大冗余vec核数
    uint32_t emptyVectorNum = aivCoreNum_ - splitRes.numOfFdHead;
    // 计算每个核处理的load
    // 向上取整，避免核负载为0
    uint64_t averageLoad = (totalFDLoad + aivCoreNum_ - 1U) / aivCoreNum_;
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        // 冗余vec核数为0，此时规约任务无法进行更小的切分，只能1个vec核计算1个规约任务
        if (emptyVectorNum == 0U) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = 0U;
            splitRes.fdRes.fdMNum[curCoreIndex] = splitRes.fdRes.fdMSize[i];
            curCoreIndex++;
            continue;
        }
        // 计算当前归约任务所用核数，向下取整，避免使用核数超出总核数
        uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i] / averageLoad;
        curFDVectorNum = std::max(1U, curFDVectorNum);
        // 计算当前归约任务每个核的行数，向上取整，避免行数为0
        uint32_t curAveMSize = (splitRes.fdRes.fdMSize[i] + curFDVectorNum - 1U) / curFDVectorNum;
        curFDVectorNum = (splitRes.fdRes.fdMSize[i] + curAveMSize - 1U) / curAveMSize;
        // 需要使用的vec核数与当前剩余可用vec核数取最小
        curFDVectorNum = std::min(curFDVectorNum, emptyVectorNum + 1U); // 1: Fd任务自身带一个核
        // FD负载分配
        for (uint32_t vid = 0; vid < curFDVectorNum; vid++) {
            splitRes.fdRes.fdIdx[curCoreIndex] = i;
            splitRes.fdRes.fdMStart[curCoreIndex] = vid * curAveMSize;
            splitRes.fdRes.fdMNum[curCoreIndex] =
                (vid < curFDVectorNum - 1) ? curAveMSize : (splitRes.fdRes.fdMSize[i] - vid * curAveMSize);
            curCoreIndex++;
        }
        // 更新冗余vec核数
        emptyVectorNum -= (curFDVectorNum - 1U); // 1: 空余核不包含FD自身的核，要-1
    }
    splitRes.fdRes.fdUsedVecNum = curCoreIndex;
}

bool QuantLightningIndexerV2MetadataCpuKernel::BalanceSchedule(SplitResult &splitRes)
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
    CalcSplitPlan(splitRes.maxCost, splitContext, splitRes);
    // 3、存在FD任务，对FD进行负载均衡分配
    if (supportFd_ && splitRes.numOfFdHead > 0U) {
        SplitFD(splitRes);
    }
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U); // 至少使用1个core
    return true;
}

bool QuantLightningIndexerV2MetadataCpuKernel::GenMetadata(SplitResult &splitRes)
{
    optiling::detail::QliV2Metadata *metadataPtr = static_cast<optiling::detail::QliV2Metadata *>(metadata_->GetData());
    *metadataPtr = {};
    // LI Metadata Generate
    for (size_t i = 0; i < aicCoreNum_; ++i) {
        if (i >= splitRes.usedCoreNum) {
            metadataPtr->qliV2Metadata[i][QLI_V2_CORE_ENABLE_INDEX] = 0; // AIC disenable
            continue;
        }
        metadataPtr->qliV2Metadata[i][QLI_V2_CORE_ENABLE_INDEX] = 1; // AIC enable
        // FA START
        metadataPtr->qliV2Metadata[i][QLI_V2_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
        metadataPtr->qliV2Metadata[i][QLI_V2_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
        metadataPtr->qliV2Metadata[i][QLI_V2_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
        // FA END
        metadataPtr->qliV2Metadata[i][QLI_V2_BN2_END_INDEX] = splitRes.bN2End[i];
        metadataPtr->qliV2Metadata[i][QLI_V2_M_END_INDEX] = splitRes.gS1End[i];
        metadataPtr->qliV2Metadata[i][QLI_V2_S2_END_INDEX] = splitRes.s2End[i];
        metadataPtr->qliV2Metadata[i][QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX] =
            splitRes.firstFdDataWorkspaceIdx[i];
    }

    // LD Metadata Generate
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            metadataPtr->qldV2Metadata[i][QLD_V2_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        metadataPtr->qldV2Metadata[i][QLD_V2_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        metadataPtr->qldV2Metadata[i][QLD_V2_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        metadataPtr->qldV2Metadata[i][QLD_V2_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        metadataPtr->qldV2Metadata[i][QLD_V2_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        metadataPtr->qldV2Metadata[i][QLD_V2_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        metadataPtr->qldV2Metadata[i][QLD_V2_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        metadataPtr->qldV2Metadata[i][QLD_V2_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    return true;
}

namespace {
static const char *kernelType = "QuantLightningIndexerV2Metadata";
REGISTER_CPU_KERNEL(kernelType, QuantLightningIndexerV2MetadataCpuKernel);
} // namespace
}; // namespace aicpu
