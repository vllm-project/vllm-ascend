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
 * \file kv_quant_sparse_attn_sharedkv_metadata_aicpu.cpp
 * \brief
 */

#include <cstdio>
#include <cmath>
#include "../../kv_quant_sparse_attn_sharedkv/op_kernel/kv_quant_sparse_attn_sharedkv_metadata.h"
#include "../../common/aicpu/cpu_context_util.h"
#include "kv_quant_sparse_attn_sharedkv_metadata_aicpu.h"

using namespace optiling;

namespace aicpu {
uint32_t
KvQuantSparseAttnSharedkvMetadataCpuKernel::Compute(CpuKernelContext &ctx) {
  bool success = Prepare(ctx) && BalanceSchedule() && GenMetaData();
  return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool KvQuantSparseAttnSharedkvMetadataCpuKernel::Prepare(
    CpuKernelContext &ctx) {
  // input
  actSeqLenQ_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenQ));
  actSeqLenOriKV_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenOriKV));
  actSeqLenCmpKV_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenCmpKV));
  SeqUsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::SeqUsedQ));
  SeqUsedKV_ = ctx.Input(static_cast<uint32_t>(ParamId::SeqUsedKV));
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
  GetAttrValueOpt(ctx, "ori_mask_mode", winMaskMode_);
  GetAttrValueOpt(ctx, "cmp_mask_mode", cmpMaskMode_);
  GetAttrValueOpt(ctx, "ori_win_left", winLeft_);
  GetAttrValueOpt(ctx, "ori_win_right", winRight_);
  GetAttrValueOpt(ctx, "layout_q", layoutQuery_);
  GetAttrValueOpt(ctx, "layout_kv", layoutKV_);
  GetAttrValueOpt(ctx, "has_ori_kv", hasOriKV_);
  GetAttrValueOpt(ctx, "has_cmp_kv", hasCmpKV_);

  coreNum_ = aicCoreNum_;
  sparseMode_ = static_cast<uint32_t>(SparseMode::BAND);
  preToken_ = (winLeft_ > -1) ? winLeft_ : INT64_MAX;
  nextToken_ = 0;
  attentionMode_ = 1;
  isS1G_ = (layoutQuery_ == "BSND" || layoutQuery_ == "BSH" || layoutQuery_ == "TND");

  return (ParamsCheck() && ParamsInit());
}

bool KvQuantSparseAttnSharedkvMetadataCpuKernel::ParamsCheck() {
  return true;
}

ValidSocVersion KvQuantSparseAttnSharedkvMetadataCpuKernel::ProcessSocVersion() {
    const std::string ascend910D = "Ascend910_95";
    if (socVersion_.find(ascend910D) != std::string::npos) {
        return ValidSocVersion::ASCEND910D;
    } else {
        return ValidSocVersion::ASCEND910B;
    }
    return ValidSocVersion::RESERVED_VERSION;
}

bool KvQuantSparseAttnSharedkvMetadataCpuKernel::ParamsInit() {
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

uint32_t KvQuantSparseAttnSharedkvMetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    // 1. 如果 SeqUsedQ_ 传了，直接使用
    if (SeqUsedQ_ != nullptr && SeqUsedQ_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t*>(SeqUsedQ_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // 2. SeqUsedQ_ 没传，判断 Layout
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

uint32_t KvQuantSparseAttnSharedkvMetadataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    // 1. 如果 SeqUsedKV_ 传了，直接使用
    if (SeqUsedKV_ != nullptr && SeqUsedKV_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t*>(SeqUsedKV_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // 2. SeqUsedKV_ 没传，判断 Layout
    if (layoutKV_ == "TND") {
        // 如果是 TND，尝试使用 actSeqLenOriKV_
        if (actSeqLenOriKV_ != nullptr && actSeqLenOriKV_->GetData() != nullptr) {
            const int32_t *s2Ptr = static_cast<const int32_t*>(actSeqLenOriKV_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 3. 如果不是 TND，或者 actSeqLenOriKV_ 为空，使用 kvSeqSize_
    return kvSeqSize_;
}

void KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcSplitInfo(SplitContext &splitContext)
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

int64_t KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcPreTokenLeftUp(
    uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(sparseMode_);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

int64_t KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcNextTokenLeftUp(
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

int64_t KvQuantSparseAttnSharedkvMetadataCpuKernel::WinCalcCost(
    uint32_t basicM, uint32_t basicS2)
{
    uint32_t winAlignCoefM = 16U;
    uint32_t winAlignCoefS2 = 64U;
    uint32_t winAlignBasicM = (basicM + winAlignCoefM - 1U) >> 4U;      // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t winAlignBasicS2 = (basicS2 + winAlignCoefS2 - 1U) >> 6U;   // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * winAlignBasicM + 10U * winAlignBasicS2);                 // 6：M轴系数，10：S2轴系数
}

int64_t KvQuantSparseAttnSharedkvMetadataCpuKernel::CmpCalcCost(
    uint32_t basicM, uint32_t basicS2)
{
    uint32_t cmpAlignCoefM = 16U;
    uint32_t cmpAlignCoefS2 = 64U;
    uint32_t cmpAlignBasicM = (basicM + cmpAlignCoefM - 1U) >> 4U;      // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t cmpAlignBasicS2 = (basicS2 + cmpAlignCoefS2 - 1U) >> 6U;   // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * cmpAlignBasicM + 10U * cmpAlignBasicS2);                 // 6：M轴系数，10：S2轴系数
}

void KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcCostTable(uint32_t s1NormalSize, 
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

Range<int64_t> KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache)
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcBatchCache(
    uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.s2Size = GetS2SeqSize(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
}

void KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcWinS1GCache(S1GCache &s1GCache, 
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcCmpS1GCache(S1GCache &s1GCache, 
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcBlockRangeAndTailSize(Range<int64_t> &oriS2TokenRange, 
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::GatherWinAndCmpCache(S1GCache &s1GCache)
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx,
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcBatchCost(
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcCostInfo(SplitContext &splitContext)
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext) {
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
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

int64_t KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcCurBlockCost(AssignContext &assignContext)
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext)
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::ForceAssign(const SplitContext &splitContext, AssignContext &assignContext) {
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

bool KvQuantSparseAttnSharedkvMetadataCpuKernel::IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes)
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / kvHeadNum_;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
                            (s1Size * groupSize_ - splitS1GIdx * mBaseSize_) : mBaseSize_;
    uint32_t curFdS1gSplitPart = (curFdS1gSize + gS1BaseSizeOfFd_ - 1U) / gS1BaseSizeOfFd_;
    uint32_t curFdS1gLastPartSize = curFdS1gSize - (gS1BaseSizeOfFd_ * (curFdS1gSplitPart - 1U));
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.fdRes.bN2IdxOfFdHead[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.gS1IdxOfFdHead[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.s2SplitNumOfFdHead[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.gS1SplitNumOfFdHead[result.numOfFdHead] = curFdS1gSplitPart;
    result.fdRes.gS1LastPartSizeOfFdHead[result.numOfFdHead] = curFdS1gLastPartSize;
    result.numOfFdHead++;
}

void KvQuantSparseAttnSharedkvMetadataCpuKernel::AssignBlocksToCore(const SplitContext &splitContext, 
                                                                    AssignContext &assignContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    result.fdRes.s2SplitStartIdxOfCore[assignContext.curCoreIdx] = assignContext.curKvSplitPart - 1U;
    
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext, 
                                                                SplitResult &result)
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
    // 负载分配
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

void KvQuantSparseAttnSharedkvMetadataCpuKernel::SplitFD(SplitResult &result)
{
    uint32_t totalFDLoad = 0;
    uint32_t totalFDHeadSplit = 0;
    // 计算FD的总数据量
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        totalFDLoad += result.fdRes.s2SplitNumOfFdHead[i] * result.fdRes.gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += result.fdRes.gS1SplitNumOfFdHead[i];
    }
    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, result.usedCoreNum * result.vecCubeRatio);
    double loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum);  // 初始化vector的负载上限
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0;
    uint32_t preTmpFDIndexEndOfFdHead = 0;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0;
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        uint32_t fDKVSplitNum = result.fdRes.s2SplitNumOfFdHead[i];
        for (uint32_t gS1SplitIdx = 0; gS1SplitIdx < result.fdRes.gS1SplitNumOfFdHead[i]; gS1SplitIdx++) {
            double remainSpace = loadThrOfVector - static_cast<double>(loadOfCurVector);  // 计算当前vector剩余负载空间
            // 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if (fDKVSplitNum > remainSpace * FD_TOLERANCE_RATIO) {
                result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1U;
                totalFDLoad -= static_cast<uint32_t>(loadOfCurVector);  // 当前未分配的总负载
                // 根据剩余负载和剩余可用vector更新负载上限，保证最后一个vector能分配所有负载
                loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum - curCoreIndex);
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    result.usedVecNumOfFd = curCoreIndex + 1;
}

bool KvQuantSparseAttnSharedkvMetadataCpuKernel::BalanceSchedule() {
    SplitContext splitContext(batchSize_);

    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes_.usedCoreNum = 1U;
        splitRes_.bN2End[0] = batchSize_ * kvHeadNum_;
        splitRes_.gS1End[0] = 0U;
        splitRes_.s2End[0] = 0U;
        return true;
    }
    CalcCostInfo(splitContext);

    // 2、获取每个核的分配方案
    splitRes_.maxCost = INT64_MAX;
    splitRes_.usedCoreNum = 1U;
    
    CalcSplitPlan(splitRes_.maxCost, splitContext, splitRes_);
    // 3、存在FD任务，对FD进行负载均衡分配
    if (splitRes_.numOfFdHead > 0U) {
        SplitFD(splitRes_);
    }
    splitRes_.usedCoreNum = std::max(splitRes_.usedCoreNum, 1U);  // 至少使用1个core
    return true;
}

bool KvQuantSparseAttnSharedkvMetadataCpuKernel::GenMetaData() {
    optiling::detail::SasMetaData* metaDataPtr = (optiling::detail::SasMetaData*)metaData_->GetData();

    for (size_t i = 0; i < coreNum_; ++i) {
        if (i < splitRes_.usedCoreNum) {
            metaDataPtr->coreMetadata[i].faMetadata[FA_CORE_ENABLE_INDEX] = 1;
        } else {
            metaDataPtr->coreMetadata[i].faMetadata[FA_CORE_ENABLE_INDEX] = 0;
            continue;
        }
        if (i == 0) {
            metaDataPtr->coreMetadata[i].faMetadata[FA_BN2_START_INDEX] = 0;
            metaDataPtr->coreMetadata[i].faMetadata[FA_M_START_INDEX] = 0;
            metaDataPtr->coreMetadata[i].faMetadata[FA_S2_START_INDEX] = 0;
        } else {
            metaDataPtr->coreMetadata[i].faMetadata[FA_BN2_START_INDEX] = splitRes_.bN2End[i-1];
            metaDataPtr->coreMetadata[i].faMetadata[FA_M_START_INDEX] = splitRes_.gS1End[i-1];
            metaDataPtr->coreMetadata[i].faMetadata[FA_S2_START_INDEX] = splitRes_.s2End[i-1];
        }

        metaDataPtr->coreMetadata[i].faMetadata[FA_BN2_END_INDEX] = splitRes_.bN2End[i];
        metaDataPtr->coreMetadata[i].faMetadata[FA_M_END_INDEX] = splitRes_.gS1End[i];
        metaDataPtr->coreMetadata[i].faMetadata[FA_S2_END_INDEX] = splitRes_.s2End[i];

        metaDataPtr->coreMetadata[i].faMetadata[FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = 0;
        metaDataPtr->coreMetadata[i].faMetadata[FA_FD_VECTOR_NUM_INDEX] = 0;
    }
    return true;
}

namespace {
    static const char *kernelType = "KvQuantSparseAttnSharedkvMetadata";
    REGISTER_CPU_KERNEL(kernelType, KvQuantSparseAttnSharedkvMetadataCpuKernel);
}

}; // namespace aicpu
