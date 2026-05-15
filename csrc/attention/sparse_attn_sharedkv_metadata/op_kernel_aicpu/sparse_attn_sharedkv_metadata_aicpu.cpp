/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file sparse_attn_sharedkv_metadata_aicpu.cpp
 * \brief
 */

#include <cstdio>
#include <cmath>
#include "../../sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_metadata.h"
#include "../../common/aicpu/cpu_context_util.h"
#include "sparse_attn_sharedkv_metadata_aicpu.h"

using namespace optiling;

namespace aicpu {
uint32_t
SparseAttnSharedkvMetadataCpuKernel::Compute(CpuKernelContext &ctx) {
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes {aicCoreNum_, aivCoreNum_};
    success = BalanceSchedule(splitRes) && GenMetaData(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool SparseAttnSharedkvMetadataCpuKernel::Prepare(
    CpuKernelContext &ctx) {
    // input
    actSeqLenQ_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenQ));
    actSeqLenOriKv_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenOriKv));
    actSeqLenCmpKv_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenCmpKv));
    seqUsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedQ));
    seqUsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::seqUsedKv));
    // output
    metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", queryHeadNum_) &&
                        GetAttrValue(ctx, "num_heads_kv", kvHeadNum_) &&
                        GetAttrValue(ctx, "head_dim", headDim_);
                        GetAttrValueOpt(ctx, "soc_version", socVersion_);
                        GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);
                        GetAttrValueOpt(ctx, "aiv_core_num", aivCoreNum_);
    if (!requiredAttrs) {
        return false;
    }

    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", querySeqSize_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", kvSeqSize_);
    GetAttrValueOpt(ctx, "ori_topk", oriTopK_);
    GetAttrValueOpt(ctx, "cmp_topk", cmpTopK_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);
    GetAttrValueOpt(ctx, "ori_mask_mode", oriMaskMode_);
    GetAttrValueOpt(ctx, "cmp_mask_mode", cmpMaskMode_);
    GetAttrValueOpt(ctx, "ori_win_left", winLeft_);
    GetAttrValueOpt(ctx, "ori_win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQuery_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "has_ori_kv", hasOriKv_);
    GetAttrValueOpt(ctx, "has_cmp_kv", hasCmpKv_);

    coreNum_ = aicCoreNum_;
    sparseMode_ = oriMaskMode_;
    preToken_ = (winLeft_ > -1) ? winLeft_ : INT64_MAX;
    nextToken_ = 0;
    attentionMode_ = 1;
    isS1G_ = (layoutQuery_ == "BSND" || layoutQuery_ == "BSH" || layoutQuery_ == "TND");

  return (ParamsCheck() && ParamsInit());
}

bool SparseAttnSharedkvMetadataCpuKernel::ParamsCheck() {
    return true;
}

ValidSocVersion SparseAttnSharedkvMetadataCpuKernel::ProcessSocVersion() {
    const std::string ascend910D = "Ascend910_95";
    if (socVersion_.find(ascend910D) != std::string::npos) {
        return ValidSocVersion::ASCEND910D;
    } else {
        return ValidSocVersion::ASCEND910B;
    }
    return ValidSocVersion::RESERVED_VERSION;
}

bool SparseAttnSharedkvMetadataCpuKernel::ParamsInit() {
    groupSize_ = queryHeadNum_ / kvHeadNum_;
    if (cmpRatio_ > 1) {
        if (cmpTopK_ > 0) {
            isSCFA = true;
        } else {
            isCFA = true;
        }
    }
    ValidSocVersion validSocVersion = ProcessSocVersion();
    if (validSocVersion == ValidSocVersion::ASCEND910B) {
        uint32_t MBaseBlockLen = 128U;
        uint32_t s1BlockLen = MBaseBlockLen / groupSize_;
        if (isSCFA) {
            s1BlockLen = 1U;
        }
        mBaseSize_ = groupSize_ * s1BlockLen;
        s2BaseSize_ = 512U;
        gS1BaseSizeOfFd_ = 8U;
    } else if (validSocVersion == ValidSocVersion::ASCEND910D){
        mBaseSize_ = 64U;
        s2BaseSize_ = 128U;
        gS1BaseSizeOfFd_ = 8U;
    }
    return true;
}

uint32_t SparseAttnSharedkvMetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    // 1. 如果 seqUsedQ_ 传了，直接使用
    if (seqUsedQ_ != nullptr && seqUsedQ_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t*>(seqUsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // 2. seqUsedQ_ 没传，判断 Layout
    if (layoutQuery_ == "TND") {
        // 如果是 TND，尝试使用 actSeqLenQ_
        if (actSeqLenQ_ != nullptr && actSeqLenQ_->GetData() != nullptr) {
            const int32_t *s1Ptr =static_cast<const int32_t*>(actSeqLenQ_->GetData());
            return static_cast<uint32_t>(s1Ptr[bIdx + 1U] - s1Ptr[bIdx]);
        }
    }
    // 3. 如果不是 TND，或者 actSeqLenQ_ 为空，使用 querySeqSize_
    return querySeqSize_;
}

uint32_t SparseAttnSharedkvMetadataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    // 1. 如果 seqUsedKv_ 传了，直接使用
    if (seqUsedKv_ != nullptr && seqUsedKv_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t*>(seqUsedKv_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // 2. seqUsedKv_ 没传，判断 Layout
    if (layoutKv_ == "TND") {
        // 如果是 TND，尝试使用 actSeqLenOriKv_
        if (actSeqLenOriKv_ != nullptr && actSeqLenOriKv_->GetData() != nullptr) {
            const int32_t *s2Ptr = static_cast<const int32_t*>(actSeqLenOriKv_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 3. 如果不是 TND，或者 actSeqLenOriKv_ 为空，使用 kvSeqSize_
    return kvSeqSize_;
}

void SparseAttnSharedkvMetadataCpuKernel::CalcSplitInfo(SplitContext &splitContext)
{
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        uint32_t s2Size = GetS2SeqSize(bIdx);
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

int64_t SparseAttnSharedkvMetadataCpuKernel::CalcPreTokenLeftUp(
    uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(sparseMode_);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

int64_t SparseAttnSharedkvMetadataCpuKernel::CalcNextTokenLeftUp(
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

int64_t SparseAttnSharedkvMetadataCpuKernel::WinCalcCost(
    uint32_t basicM, uint32_t basicS2)
{
    uint32_t winAlignCoefM = 16U;
    uint32_t winAlignCoefS2 = 64U;
    uint32_t winAlignBasicM = (basicM + winAlignCoefM - 1U) >> 4U;      // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t winAlignBasicS2 = (basicS2 + winAlignCoefS2 - 1U) >> 6U;   // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * winAlignBasicM + 10U * winAlignBasicS2);                 // 6：M轴系数，10：S2轴系数
}

int64_t SparseAttnSharedkvMetadataCpuKernel::CmpCalcCost(
    uint32_t basicM, uint32_t basicS2)
{
    uint32_t cmpAlignCoefM = 16U;
    uint32_t cmpAlignCoefS2 = 64U;
    uint32_t cmpAlignBasicM = (basicM + cmpAlignCoefM - 1U) >> 4U;      // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t cmpAlignBasicS2 = (basicS2 + cmpAlignCoefS2 - 1U) >> 6U;   // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * cmpAlignBasicM + 10U * cmpAlignBasicS2);                 // 6：M轴系数，10：S2轴系数
}

void SparseAttnSharedkvMetadataCpuKernel::CalcCostTable(uint32_t s1NormalSize, 
    uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t winS2TailSize, uint32_t cmpS2TailSize)
{
    // win部分cost
    typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK] = WinCalcCost(s1NormalSize, s2NormalSize);
    typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : WinCalcCost(s1GTailSize, s2NormalSize);
    typeCost_[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] = (winS2TailSize == 0U) ? 0U : WinCalcCost(s1NormalSize, winS2TailSize);
    typeCost_[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] = (s1GTailSize == 0U || winS2TailSize == 0U) ? 0U : WinCalcCost(s1GTailSize, winS2TailSize);
    // cmp部分cost
    if (isCFA || isSCFA) {
        typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] = CmpCalcCost(s1NormalSize, s2NormalSize);
        typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CmpCalcCost(s1GTailSize, s2NormalSize);
        typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] = (cmpS2TailSize == 0U) ? 0U : CmpCalcCost(s1NormalSize, cmpS2TailSize);
        typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] = (s1GTailSize == 0U || cmpS2TailSize == 0U) ? 0U : CmpCalcCost(s1GTailSize, cmpS2TailSize);
    }
}

Range<int64_t> SparseAttnSharedkvMetadataCpuKernel::CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache)
{
    // actual seq == 0
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(0, 0);
    }

    // no mask
    if (!attentionMode_) { //attentionMaskFlag ?
        return std::make_pair(0, static_cast<int64_t>(batchCache.s2Size) - 1);
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

void SparseAttnSharedkvMetadataCpuKernel::CalcBatchCache(
    uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.s2Size = GetS2SeqSize(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
}

void SparseAttnSharedkvMetadataCpuKernel::CalcWinS1GCache(S1GCache &s1GCache, 
                                                            const SplitInfo &splitInfo)
{
    // 处理win部分block信息
    if (s1GCache.winS2Start >= s1GCache.winS2End) {
        // win范围无效, 则整个s1g行等效为空行
        s1GCache.winS1GBlock = 0;
        s1GCache.winS1GCost = 0;
        s1GCache.winS1GLastBlockCost = 0;
        s1GCache.winS1GNormalBlockCost = 0;
    } else {
        //计算 Win 方向 Block 数量及 Cost
        s1GCache.winS1GBlock = s1GCache.winS2End - s1GCache.winS2Start;
        // 判断 Win S2 方向是否包含尾块
        uint32_t curWinTailS2Num = (s1GCache.winS2TailSize != 0U) ? 1U : 0U;// Updated check using local var
        uint32_t curWinNormalS2Num = s1GCache.winS1GBlock - curWinTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) && splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
            s1GCache.winS1GCost = typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK] * curWinNormalS2Num +
                typeCost_[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] * curWinTailS2Num;
            s1GCache.winS1GLastBlockCost = curWinTailS2Num > 0U ? typeCost_[WIN_TAIL_BLOCK][WIN_TAIL_BLOCK] :
                                            typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK];
            s1GCache.winS1GNormalBlockCost = typeCost_[WIN_TAIL_BLOCK][WIN_NORMAL_BLOCK];
        }
        else {
            s1GCache.winS1GCost = typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK] * curWinNormalS2Num +
                typeCost_[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] * curWinTailS2Num;
            s1GCache.winS1GLastBlockCost = curWinTailS2Num > 0U ? typeCost_[WIN_NORMAL_BLOCK][WIN_TAIL_BLOCK] :
                                            typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK];
            s1GCache.winS1GNormalBlockCost = typeCost_[WIN_NORMAL_BLOCK][WIN_NORMAL_BLOCK];
        }
    }
}

void SparseAttnSharedkvMetadataCpuKernel::CalcCmpS1GCache(S1GCache &s1GCache, 
                                                            const SplitInfo &splitInfo)
{
    // 处理cmp部分block信息
    if (s1GCache.cmpS2Start >= s1GCache.cmpS2End) {
        // Cmp范围无效, Cost保持为 0
        s1GCache.cmpS1GBlock = 0;
        s1GCache.cmpS1GCost = 0;
        s1GCache.cmpS1GLastBlockCost = 0;
        s1GCache.cmpS1GNormalBlockCost = 0;
    } else {
        //计算 cmp 方向 Block 数量及 Cost
        s1GCache.cmpS1GBlock = s1GCache.cmpS2End - s1GCache.cmpS2Start;
        // 判断 Cmp S2 方向是否包含尾块
        uint32_t curCmpTailS2Num = (s1GCache.cmpS2TailSize != 0U) ? 1U : 0U;// Updated check using local var
        uint32_t curCmpNormalS2Num = s1GCache.cmpS1GBlock - curCmpTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) && splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
            s1GCache.cmpS1GCost = typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
                typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
            s1GCache.cmpS1GLastBlockCost = curCmpTailS2Num > 0U ? typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] :
                                                typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
            s1GCache.cmpS1GNormalBlockCost = typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK];
        } else {
            s1GCache.cmpS1GCost = typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] * curCmpNormalS2Num +
                typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] * curCmpTailS2Num;
            s1GCache.cmpS1GLastBlockCost = curCmpTailS2Num > 0U ? typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] :
                                             typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
            s1GCache.cmpS1GNormalBlockCost = typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK];
        }
    }
}

void SparseAttnSharedkvMetadataCpuKernel::CalcBlockRangeAndTailSize(Range<int64_t> &oriS2TokenRange, 
                                                                            const BatchCache &batchCache, 
                                                                            S1GCache &s1GCache)
{
    int64_t oriS2FirstToken = oriS2TokenRange.first;
    int64_t oriS2LastToken = oriS2TokenRange.second;
    // win部分s2起止和tailSize
    if (oriS2FirstToken >= static_cast<int64_t>(batchCache.s2Size) || oriS2LastToken < 0 || 
            oriS2LastToken < oriS2FirstToken) {
        oriS2FirstToken = 0;
        oriS2LastToken = 0;
        s1GCache.winS2Start = 0;
        s1GCache.winS2End = 0;
        s1GCache.winS2TailSize = 0;
    } else {
        oriS2FirstToken = Clip(oriS2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
        oriS2LastToken = Clip(oriS2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
        s1GCache.winS2Start = 0;
        s1GCache.winS2End = (oriS2LastToken - oriS2FirstToken) / s2BaseSize_ + 1U;
        s1GCache.winS2TailSize = (oriS2LastToken - oriS2FirstToken + 1) % s2BaseSize_;
    }
    // cmp部分s2起止和tailSize
    s1GCache.cmpS2Start = s1GCache.winS2End;
    // 计算CmpS2LastToken的长度
    uint32_t cmpS2LastTokenSize = (cmpRatio_ > 1) ? (oriS2LastToken + 1) / cmpRatio_ : 0;
    uint32_t actCmpS2LastTokenSize = 0;
    if (isCFA) {
        actCmpS2LastTokenSize = cmpS2LastTokenSize;
    } else if (isSCFA) {
        // CmpS2LastToken与topk取最小
        actCmpS2LastTokenSize = std::min(cmpS2LastTokenSize, cmpTopK_);
    }
    // 将token长度转化为token索引，然后由token索引计算s2索引
    s1GCache.cmpS2End = (actCmpS2LastTokenSize == 0) ? s1GCache.cmpS2Start : s1GCache.cmpS2Start + 
                        (actCmpS2LastTokenSize - 1) / s2BaseSize_ + 1U;
    // 由token长度计算cmpS2TailSize
    s1GCache.cmpS2TailSize = actCmpS2LastTokenSize % s2BaseSize_;
}

void SparseAttnSharedkvMetadataCpuKernel::GatherWinAndCmpCache(S1GCache &s1GCache)
{
    s1GCache.s2Start = (s1GCache.winS1GBlock > 0) ? s1GCache.winS2Start : s1GCache.cmpS2Start;
    if (s1GCache.cmpS1GBlock > 0) {
        s1GCache.s1GLastBlockCost = s1GCache.cmpS1GLastBlockCost;
        s1GCache.s2End = s1GCache.cmpS2End;
    } else {
        s1GCache.s1GLastBlockCost = s1GCache.winS1GLastBlockCost;
        s1GCache.s2End = s1GCache.winS2End;
    }
    s1GCache.s1GBlock = s1GCache.winS1GBlock + s1GCache.cmpS1GBlock;
    s1GCache.s1GCost = s1GCache.winS1GCost + s1GCache.cmpS1GCost;
}

void SparseAttnSharedkvMetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx,
    const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 如果s1G是空行，则直接返回
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.winS1GNormalBlockCost = 0;
        s1GCache.winS1GLastBlockCost = 0;
        s1GCache.cmpS1GNormalBlockCost = 0;
        s1GCache.cmpS1GLastBlockCost = 0;
        s1GCache.s1GBlock = 0;
        s1GCache.s2Start = 0;
        s1GCache.cmpS2Start = 0;
        s1GCache.s2End = 0;
        return;
    }
    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;
    // 计算ori_kv的token起止
    auto oriS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache);
    // 计算win和cmp部分s2起止和tailSize
    CalcBlockRangeAndTailSize(oriS2TokenRange, batchCache, s1GCache);
    // Calculate CostTable locally
    CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[s1GCache.bIdx],
                                                s1GCache.winS2TailSize, s1GCache.cmpS2TailSize);
    // 计算win和cmp部分的cost, block信息
    CalcWinS1GCache(s1GCache, splitInfo);
    CalcCmpS1GCache(s1GCache, splitInfo);
    // 汇总win和cmp部分的cost, block信息
    GatherWinAndCmpCache(s1GCache);
}

void SparseAttnSharedkvMetadataCpuKernel::CalcBatchCost(
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

void SparseAttnSharedkvMetadataCpuKernel::CalcCostInfo(SplitContext &splitContext)
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
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * kvHeadNum_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * kvHeadNum_;
    }
}

void SparseAttnSharedkvMetadataCpuKernel::AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
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
        if (assignContext.curBN2Idx == batchSize_ * kvHeadNum_) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / kvHeadNum_ != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / kvHeadNum_;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void SparseAttnSharedkvMetadataCpuKernel::AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
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

int64_t SparseAttnSharedkvMetadataCpuKernel::CalcCurBlockCost(AssignContext &assignContext)
{
    int64_t curCost = 0;
    if (assignContext.curS2Idx < assignContext.s1GCache.cmpS2Start) {
        curCost = assignContext.s1GCache.winS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GCache.cmpS2Start - 1U)) {
            curCost = assignContext.s1GCache.winS1GLastBlockCost;
        }
    } else {
        curCost = assignContext.s1GCache.cmpS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
            curCost = assignContext.s1GCache.cmpS1GLastBlockCost;
        }
    }
    return curCost;
}

void SparseAttnSharedkvMetadataCpuKernel::AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished || !supportFd) {
        return;
    }

    int64_t curCost = CalcCurBlockCost(assignContext);

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
        curCost = CalcCurBlockCost(assignContext);
    }
}

bool SparseAttnSharedkvMetadataCpuKernel::IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes)
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

void SparseAttnSharedkvMetadataCpuKernel::RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / kvHeadNum_;
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
    result.fdRes.fdWorkspaceIdx[result.numOfFdHead] = assignContext.curFdDataNum - assignContext.curKvSplitPart;
    result.fdRes.fdS2SplitNum[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.fdMSize[result.numOfFdHead] = curFdS1gSize;
    result.numOfFdHead++;
}

void SparseAttnSharedkvMetadataCpuKernel::UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext) {
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
    if (assignContext.curBN2Idx == batchSize_ * kvHeadNum_) {  // 所有负载全部分配完，设置最后一个核的右开区间，返回
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / kvHeadNum_ != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / kvHeadNum_;
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
        assignContext.curS2Idx = (supportFd) ? assignContext.s1GCache.winS2Start : 0;
    }
}

void SparseAttnSharedkvMetadataCpuKernel::ForceAssign(const SplitContext &splitContext, AssignContext &assignContext) {
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = CalcCurBlockCost(assignContext);

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

void SparseAttnSharedkvMetadataCpuKernel::AssignBlocksToCore(const SplitContext &splitContext, 
                                                                    AssignContext &assignContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    
    int64_t avgCost = assignContext.unassignedCost / (coreNum_ - assignContext.curCoreIdx);
    assignContext.coreCache = {};
    if (!supportFd) {
        assignContext.coreCache.costLimit = std::max(avgCost, costInfo.maxS1GCost);
    } else {
        assignContext.coreCache.costLimit = avgCost;
    }
    // 1、按整batch分配
    AssignByBatch(splitContext, assignContext);
    // 2、按行分配
    AssignByRow(splitContext, assignContext);
    // 3、按块分配
    AssignByBlock(splitContext, assignContext);
    // 4、强制分配
    if (assignContext.coreCache.block == 0 && supportFd) {
        ForceAssign(splitContext, assignContext);
    }
    result.bN2End[assignContext.curCoreIdx] = assignContext.curBN2Idx;
    result.gS1End[assignContext.curCoreIdx] = assignContext.curS1GIdx;
    result.s2End[assignContext.curCoreIdx] = assignContext.curS2Idx;
    result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);
    assignContext.unassignedCost -= assignContext.coreCache.cost;
    // 对之前的归约信息进行记录并清理
    if (IsNeedRecordFDInfo(assignContext, result)) {
        RecordFDInfo(splitContext, assignContext, result);
        assignContext.curKvSplitPart = 1U;
    }
    // 更新S2切分信息
    if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
        assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
        assignContext.curKvSplitPart++;
    }
}

void SparseAttnSharedkvMetadataCpuKernel::CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;

    if (coreNum_ == 0U) {
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

    for (uint32_t i = 0; i < coreNum_; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }
        assignContext.curCoreIdx = i;        
    AssignBlocksToCore(splitContext, assignContext, result);
    }
    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void SparseAttnSharedkvMetadataCpuKernel::SplitFD(SplitResult &splitRes)
{
    // 计算FD的总数据量
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    // 计算每个核处理的load
    uint64_t averageLoad = totalFDLoad / aivCoreNum_;
    uint32_t curCoreIndex = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        uint32_t curFDVectorNum = splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i] / averageLoad;
        uint32_t curAveMSize = splitRes.fdRes.fdMSize[i] / curFDVectorNum;
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

bool SparseAttnSharedkvMetadataCpuKernel::BalanceSchedule(SplitResult &splitRes) {
    SplitContext splitContext(batchSize_);

    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = batchSize_ * kvHeadNum_;
        splitRes.gS1End[0] = 0U;
        splitRes.s2End[0] = 0U;
        return true;
    }
    CalcCostInfo(splitContext);

    // 2、获取每个核的分配方案
    splitRes.maxCost = INT64_MAX;
    splitRes.usedCoreNum = 1U;
    
    CalcSplitPlan(splitRes.maxCost, splitContext, splitRes);
    // 3、存在FD任务，对FD进行负载均衡分配
    if (splitRes.numOfFdHead > 0U) {
        SplitFD(splitRes);
    }
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U);  // 至少使用1个core
    return true;
}
bool SparseAttnSharedkvMetadataCpuKernel::GenMetaData(SplitResult &splitRes) {
    optiling::detail::SasMetaData* metaDataPtr = (optiling::detail::SasMetaData*)metaData_->GetData();
    // FA Metadata Generate
    for (size_t i = 0; i < aicCoreNum_; ++i) {
        if (i >= splitRes.usedCoreNum) {
            metaDataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 0; // AIC disenable
            continue;
        }
        metaDataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 1; // AIC enable
        // FA START
        metaDataPtr->faMetadata[i][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i-1];
        metaDataPtr->faMetadata[i][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i-1];
        metaDataPtr->faMetadata[i][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i-1];
        // FA END
        metaDataPtr->faMetadata[i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
        metaDataPtr->faMetadata[i][FA_M_END_INDEX] = splitRes.gS1End[i];
        metaDataPtr->faMetadata[i][FA_S2_END_INDEX] = splitRes.s2End[i];
        // 
        metaDataPtr->faMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
    }

    // FD Metadata Generate
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            metaDataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        metaDataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        metaDataPtr->fdMetadata[i][FD_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        metaDataPtr->fdMetadata[i][FD_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        metaDataPtr->fdMetadata[i][FD_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        metaDataPtr->fdMetadata[i][FD_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        metaDataPtr->fdMetadata[i][FD_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        metaDataPtr->fdMetadata[i][FD_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    return true;
}
namespace {
    static const char *kernelType = "SparseAttnSharedkvMetadata";
    REGISTER_CPU_KERNEL(kernelType, SparseAttnSharedkvMetadataCpuKernel);
}

}; // namespace aicpu


