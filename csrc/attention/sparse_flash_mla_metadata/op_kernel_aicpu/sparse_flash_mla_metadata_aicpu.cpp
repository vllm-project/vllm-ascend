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
 * \file sparse_flash_mla_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <cstdio>
#include <cmath>
#include "sparse_flash_mla_metadata_aicpu.h"

using namespace optiling;

namespace aicpu {
uint32_t SparseFlashMlaMetadataCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    SplitResult splitRes{aicCoreNum_, aivCoreNum_};
    success = BalanceSchedule(splitRes) && GenMetadata(splitRes);
    return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool SparseFlashMlaMetadataCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensOriKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensOriKv));
    cuSeqlensCmpKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensCmpKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedOriKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedOriKv));
    sequsedCmpKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedCmpKv));
    cmpResidualKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cmpResidualKv));
    oriTopkLength_ = ctx.Input(static_cast<uint32_t>(ParamId::oriTopkLength));
    cmpTopkLength_ = ctx.Input(static_cast<uint32_t>(ParamId::cmpTopkLength));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    bool requiredAttrs = GetAttrValue(ctx, "num_heads_q", numHeadsQ_) &&
                         GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) && GetAttrValue(ctx, "head_dim", headDim_);
    if (!requiredAttrs) {
        return false;
    }

    // attributes optional
    GetAttrValueOpt(ctx, "soc_version", socVersion_);
    GetAttrValueOpt(ctx, "aic_core_num", aicCoreNum_);
    GetAttrValueOpt(ctx, "aiv_core_num", aivCoreNum_);
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_ori_kv", maxSeqlenOriKv_);
    GetAttrValueOpt(ctx, "max_seqlen_cmp_kv", maxSeqlenCmpKv_);
    GetAttrValueOpt(ctx, "ori_topk", oriTopK_);
    GetAttrValueOpt(ctx, "cmp_topk", cmpTopK_);
    GetAttrValueOpt(ctx, "cmp_ratio", cmpRatio_);
    GetAttrValueOpt(ctx, "ori_mask_mode", oriMaskMode_);
    GetAttrValueOpt(ctx, "cmp_mask_mode", cmpMaskMode_);
    GetAttrValueOpt(ctx, "ori_win_left", oriWinLeft_);
    GetAttrValueOpt(ctx, "ori_win_right", oriWinRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "has_ori_kv", hasOriKv_);
    GetAttrValueOpt(ctx, "has_cmp_kv", hasCmpKv_);

    return (ParamsCheck() && ParamsInit());
}

bool SparseFlashMlaMetadataCpuKernel::ParamsCheck()
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
            for (int i = 0; i < batchSize + 1; i++) {
                // 校验 cu_seqlens_q 元素非负
                if (cuSeqlensQPtr[i] < 0) {
                    KERNEL_LOG_ERROR("The elements in cu_seqlens_q should be >= 0, but got cu_seqlens_q[%d] = %d", i,
                                     cuSeqlensQPtr[i]);
                    return false;
                }
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
    // 校验 seqused_q 元素
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        const int32_t *sequsedQPtr = static_cast<const int32_t *>(sequsedQ_->GetData());
        for (int i = 0; i < batchSize; i++) {
            // 校验 seqused_q 元素非负
            if (sequsedQPtr[i] < 0) {
                KERNEL_LOG_ERROR("The elements in seqused_q should be >= 0, but got seqused_q[%d] = %d", i,
                                 sequsedQPtr[i]);
                return false;
            }
        }
    }
    if (hasOriKv_) {
        // 校验 cu_seqlens_ori_kv 元素
        if (layoutKv_ == "TND") {
            if (cuSeqlensOriKv_ != nullptr && cuSeqlensOriKv_->GetData() != nullptr) {
                const int32_t *cuSeqlensOriKvPtr = static_cast<const int32_t *>(cuSeqlensOriKv_->GetData());
                for (int i = 0; i < batchSize + 1; i++) {
                    // 校验 cu_seqlens_ori_kv 元素非负
                    if (cuSeqlensOriKvPtr[i] < 0) {
                        KERNEL_LOG_ERROR("The elements in cu_seqlens_ori_kv should be >= 0, "
                                         "but got cu_seqlens_ori_kv[%d] = %d",
                                         i, cuSeqlensOriKvPtr[i]);
                        return false;
                    }
                    // 校验 cu_seqlens_ori_kv 元素递增
                    if (i > 0 && cuSeqlensOriKvPtr[i - 1] > cuSeqlensOriKvPtr[i]) {
                        KERNEL_LOG_ERROR("The elements in cu_seqlens_ori_kv must be in ascending order, "
                                         "but got cu_seqlens_ori_kv[%d] = %d, cu_seqlens_ori_kv[%d] = %d",
                                         i - 1, cuSeqlensOriKvPtr[i - 1], i, cuSeqlensOriKvPtr[i]);
                        return false;
                    }
                }
            }
        }
        // 校验 seqused_ori_kv 元素
        if (sequsedOriKv_ != nullptr && sequsedOriKv_->GetData() != nullptr) {
            const int32_t *sequsedOriKvPtr = static_cast<const int32_t *>(sequsedOriKv_->GetData());
            for (int i = 0; i < batchSize; i++) {
                // 校验 seqused_ori_kv 元素非负
                if (sequsedOriKvPtr[i] < 0) {
                    KERNEL_LOG_ERROR("The elements in seqused_ori_kv should be >= 0, but got seqused_ori_kv[%d] = %d",
                                     i, sequsedOriKvPtr[i]);
                    return false;
                }
            }
        }
        // 校验 ori_topk_length 元素
        if (oriTopkLength_ != nullptr && oriTopkLength_->GetData() != nullptr) {
            // 校验 ori_topk_length 元素数量
            int32_t sumOfQuerySeq = GetSumOfQuerySeq();
            const int32_t *oriTopkLengthPtr = static_cast<const int32_t *>(oriTopkLength_->GetData());
            auto oriTopkLengthShape = oriTopkLength_->GetTensorShape();
            int32_t oriTopkLengthSize = layoutQ_ == "TND" ?
                                            oriTopkLengthShape->GetDimSize(0) * oriTopkLengthShape->GetDimSize(1) :
                                            oriTopkLengthShape->GetDimSize(0) * oriTopkLengthShape->GetDimSize(1) *
                                                oriTopkLengthShape->GetDimSize(2);
            if (oriTopkLengthSize < sumOfQuerySeq) {
                KERNEL_LOG_ERROR("The size of ori_topk_length %d should not be smaller than "
                                 "the sum of query sequence %d!",
                                 oriTopkLengthSize, sumOfQuerySeq);
                return false;
            }
            // 校验 ori_topk_length 元素非负
            for (int i = 0; i < oriTopkLengthSize; i++) {
                if (oriTopkLengthPtr[i] < 0) {
                    KERNEL_LOG_ERROR("The elements in ori_topk_length should be >= 0, but got ori_topk_length[%d] = %d",
                                     i, oriTopkLengthPtr[i]);
                    return false;
                }
            }
        }
    }
    if (hasCmpKv_) {
        if (layoutKv_ == "TND") {
            // 校验 cu_seqlens_cmp_kv 元素
            if (cuSeqlensCmpKv_ != nullptr && cuSeqlensCmpKv_->GetData() != nullptr) {
                const int32_t *cuSeqlensCmpKvPtr = static_cast<const int32_t *>(cuSeqlensCmpKv_->GetData());
                for (int i = 0; i < batchSize + 1; i++) {
                    // 校验 cu_seqlens_cmp_kv 元素非负
                    if (cuSeqlensCmpKvPtr[i] < 0) {
                        KERNEL_LOG_ERROR("The elements in cu_seqlens_cmp_kv should be >= 0, "
                                         "but got cu_seqlens_cmp_kv[%d] = %d",
                                         i, cuSeqlensCmpKvPtr[i]);
                        return false;
                    }
                    // 校验 cu_seqlens_cmp_kv 元素递增
                    if (i > 0 && cuSeqlensCmpKvPtr[i - 1] > cuSeqlensCmpKvPtr[i]) {
                        KERNEL_LOG_ERROR("The elements in cu_seqlens_cmp_kv must be in ascending order, "
                                         "but got cu_seqlens_cmp_kv[%d] = %d, cu_seqlens_cmp_kv[%d] = %d",
                                         i - 1, cuSeqlensCmpKvPtr[i - 1], i, cuSeqlensCmpKvPtr[i]);
                        return false;
                    }
                }
            }
        }
        // 校验 seqused_cmp_kv 元素
        if (sequsedCmpKv_ != nullptr && sequsedCmpKv_->GetData() != nullptr) {
            const int32_t *sequsedCmpKvPtr = static_cast<const int32_t *>(sequsedCmpKv_->GetData());
            for (int i = 0; i < batchSize; i++) {
                // 校验 seqused_cmp_kv 元素非负
                if (sequsedCmpKvPtr[i] < 0) {
                    KERNEL_LOG_ERROR("The elements in seqused_cmp_kv should be >= 0, but got seqused_cmp_kv[%d] = %d",
                                     i, sequsedCmpKvPtr[i]);
                    return false;
                }
            }
        }
        // 校验 cmp_residual_kv 元素
        if (cmpResidualKv_ != nullptr && cmpResidualKv_->GetData() != nullptr) {
            const int32_t *cmpResidualKvPtr = static_cast<const int32_t *>(cmpResidualKv_->GetData());
            for (int i = 0; i < batchSize; i++) {
                // 校验 cmp_residual_kv 元素非负
                if (cmpResidualKvPtr[i] < 0 || cmpResidualKvPtr[i] >= cmpRatio_) {
                    KERNEL_LOG_ERROR("The elements in cmp_residual_kv should be in [0, cmpRatio_), but got "
                                     "cmp_residual_kv[%d] = %d",
                                     i, cmpResidualKvPtr[i]);
                    return false;
                }
            }
        }
        // 校验 cmp_topk_length 元素
        if (cmpTopkLength_ != nullptr && cmpTopkLength_->GetData() != nullptr) {
            // 校验 cmp_topk_length 元素数量
            int32_t sumOfQuerySeq = GetSumOfQuerySeq();
            const int32_t *cmpTopkLengthPtr = static_cast<const int32_t *>(cmpTopkLength_->GetData());
            auto cmpTopkLengthShape = cmpTopkLength_->GetTensorShape();
            int32_t cmpTopkLengthSize = layoutQ_ == "TND" ?
                                            cmpTopkLengthShape->GetDimSize(0) * cmpTopkLengthShape->GetDimSize(1) :
                                            cmpTopkLengthShape->GetDimSize(0) * cmpTopkLengthShape->GetDimSize(1) *
                                                cmpTopkLengthShape->GetDimSize(2);
            if (cmpTopkLengthSize < sumOfQuerySeq) {
                KERNEL_LOG_ERROR("The size of cmp_topk_length %d should not be smaller than "
                                 "the sum of query sequence %d!",
                                 cmpTopkLengthSize, sumOfQuerySeq);
                return false;
            }
            // 校验 cmp_topk_length 元素非负
            for (int i = 0; i < cmpTopkLengthSize; i++) {
                if (cmpTopkLengthPtr[i] < 0) {
                    KERNEL_LOG_ERROR("The elements in cmp_topk_length should be >= 0, but got cmp_topk_length[%d] = %d",
                                     i, cmpTopkLengthPtr[i]);
                    return false;
                }
            }
        }
    }
    return true;
}

int32_t SparseFlashMlaMetadataCpuKernel::GetSumOfQuerySeq()
{
    int32_t batchSize = GetQueryBatchSize();
    // 如果sequsedQ_ 传了，使用sequsedQ_获取 BsSize
    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        if (sequsedQ_->GetTensorShape() != nullptr) {
            const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedQ_->GetData());
            int32_t queryBsSize = 0;
            for (int i = 0; i < batchSize; i++) {
                queryBsSize += seqUsedPtr[i];
            }
            return queryBsSize;
        }
    }
    // sequsedQ_ 没传，判断 Layout
    if (layoutQ_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensQ_获取 BsSize
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            if (cuSeqlensQ_->GetTensorShape() != nullptr) {
                const int32_t *s1Ptr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
                return s1Ptr[batchSize];
            }
        }
    }
    // 如果不是 TND，或者 cuSeqlensQ_ 为空，使用shape信息计算 BsSize
    return batchSize_ * maxSeqlenQ_;
}

int32_t SparseFlashMlaMetadataCpuKernel::GetQueryBatchSize()
{
    // 1. 如果sequsedQ_ 传了，使用sequsedQ_获取BatchSize
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

void SparseFlashMlaMetadataCpuKernel::CalcOriMaskMode()
{
    if (oriMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK)) {
        oriPreToken_ = INT64_MAX;
        oriNextToken_ = INT64_MAX;
        oriAttentionMode_ = NO_MASK;
    } else if (oriMaskMode_ == static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        oriPreToken_ = INT64_MAX;
        oriNextToken_ = 0;
        oriAttentionMode_ = HAS_MASK;
    } else { // SparseMode = 4
        oriPreToken_ = (oriWinLeft_ > -1) ? oriWinLeft_ : INT64_MAX;
        oriNextToken_ = (oriWinRight_ > -1) ? oriWinRight_ : INT64_MAX;
        oriAttentionMode_ = HAS_MASK;
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcCmpMaskMode()
{
    if (cmpMaskMode_ == static_cast<int32_t>(SparseMode::DEFAULT_MASK)) {
        cmpPreToken_ = INT64_MAX;
        cmpNextToken_ = INT64_MAX;
        cmpAttentionMode_ = NO_MASK;
    } else if (cmpMaskMode_ == static_cast<int32_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        cmpPreToken_ = INT64_MAX;
        cmpNextToken_ = 0;
        cmpAttentionMode_ = HAS_MASK;
    } else { // SparseMode = 4
        cmpPreToken_ = (oriWinLeft_ > -1) ? oriWinLeft_ : INT64_MAX;
        cmpNextToken_ = (oriWinRight_ > -1) ? oriWinRight_ : INT64_MAX;
        cmpAttentionMode_ = HAS_MASK;
    }
}

ValidSocVersion SparseFlashMlaMetadataCpuKernel::ProcessSocVersion()
{
    const std::string ascend950 = "Ascend950";
    if (socVersion_.find(ascend950) != std::string::npos) {
        return ValidSocVersion::ASCEND950;
    } else {
        return ValidSocVersion::ASCEND910;
    }
}

bool SparseFlashMlaMetadataCpuKernel::ParamsInit()
{
    batchSize_ = GetQueryBatchSize();
    CalcOriMaskMode();
    CalcCmpMaskMode();
    isS1G_ = (layoutQ_ == "BSND" || layoutQ_ == "BSH" || layoutQ_ == "TND");
    if (numHeadsKv_ == 0) {
        KERNEL_LOG_ERROR("num_heads_kv should not be 0.");
        return false;
    }
    ValidSocVersion validSocVersion = ProcessSocVersion();
    groupSize_ = numHeadsQ_ / numHeadsKv_;
    if (hasOriKv_ && oriTopK_ != 0) {
        isSparseOriKv_ = true;
    }
    if (hasCmpKv_ && cmpTopK_ != 0) {
        isSparseCmpKv_ = true;
    }
    if (validSocVersion == ValidSocVersion::ASCEND910) {
        mBaseSize_ = isSparseCmpKv_ ? groupSize_ : (256U / groupSize_) * groupSize_;
        s2BaseSize_ = 512U;
    } else if (validSocVersion == ValidSocVersion::ASCEND950) {
        if (groupSize_ > 64U) {
            isSplitG_ = true;
            aicCoreNum_ /= 2;
        }
        mBaseSize_ = groupSize_;
        s2BaseSize_ = 128U;
    } else {
        mBaseSize_ = groupSize_;
        s2BaseSize_ = 128U;
    }
    return true;
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetS1Idx(uint32_t s1Size, uint32_t s1GIdx)
{
    uint32_t s1GToken = s1GIdx * mBaseSize_;
    uint32_t s1Idx = 0;
    if (isS1G_) {
        s1Idx = s1GToken / static_cast<int64_t>(groupSize_);
    } else {
        s1Idx = s1GToken % static_cast<int64_t>(s1Size);
    }
    return s1Idx;
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetBsStride(uint32_t bIdx, uint32_t s1Idx)
{
    uint32_t bsStride = 0;
    if (layoutQ_ == "TND") {
        if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
            const int32_t *s1Ptr = static_cast<const int32_t *>(cuSeqlensQ_->GetData());
            bsStride = s1Ptr[bIdx] + s1Idx;
            return bsStride;
        }
    }
    bsStride = bIdx * static_cast<uint32_t>(maxSeqlenQ_) + s1Idx;
    return bsStride;
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetOriTopkLength(uint32_t bsStride)
{
    // 尝试使用 oriTopkLength_
    if (oriTopkLength_ != nullptr && oriTopkLength_->GetData() != nullptr) {
        const int32_t *oriTopkPtr = static_cast<const int32_t *>(oriTopkLength_->GetData());
        return static_cast<uint32_t>(oriTopkPtr[bsStride]);
    }
    // 如果不是 DEFAULT_MASK，使用 oriTopK_
    return static_cast<uint32_t>(oriTopK_);
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetCmpTopkLength(uint32_t bsStride)
{
    // 尝试使用 cmpTopkLength_
    if (cmpTopkLength_ != nullptr && cmpTopkLength_->GetData() != nullptr) {
        const int32_t *cmpTopkPtr = static_cast<const int32_t *>(cmpTopkLength_->GetData());
        return static_cast<uint32_t>(cmpTopkPtr[bsStride]);
    }
    // 如果不是 DEFAULT_MASK，使用 cmpTopK_
    return static_cast<uint32_t>(cmpTopK_);
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetS1SeqSize(uint32_t bIdx)
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
    // 3. 如果不是 TND，或者 cuSeqlensQ_ 为空，使用 maxSeqlenQ_
    return static_cast<uint32_t>(maxSeqlenQ_);
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetOriS2SeqSize(uint32_t bIdx)
{
    // 如果 sequsedOriKv_ 传了，直接使用
    if (sequsedOriKv_ != nullptr && sequsedOriKv_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedOriKv_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // sequsedOriKv_ 没传，判断 Layout
    if (layoutKv_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensOriKv_
        if (cuSeqlensOriKv_ != nullptr && cuSeqlensOriKv_->GetData() != nullptr) {
            const int32_t *s2Ptr = static_cast<const int32_t *>(cuSeqlensOriKv_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 如果 max_seqlen_ori_kv 没传入，且 ori_kv 为稀疏的，则尝试从 topk 中获取
    if (maxSeqlenOriKv_ == 0 && isSparseOriKv_) {
        return UINT32_MAX;
    }
    // 使用 max_seqlen_ori_kv
    return static_cast<uint32_t>(maxSeqlenOriKv_);
}

uint32_t SparseFlashMlaMetadataCpuKernel::GetCmpS2SeqSize(uint32_t bIdx)
{
    // 如果 sequsedCmpKv_ 传了，直接使用
    if (sequsedCmpKv_ != nullptr && sequsedCmpKv_->GetData() != nullptr) {
        const int32_t *seqUsedPtr = static_cast<const int32_t *>(sequsedCmpKv_->GetData());
        return static_cast<uint32_t>(seqUsedPtr[bIdx]);
    }
    // sequsedCmpKv_ 没传，判断 Layout
    if (layoutKv_ == "TND") {
        // 如果是 TND，尝试使用 cuSeqlensCmpKv_
        if (cuSeqlensCmpKv_ != nullptr && cuSeqlensCmpKv_->GetData() != nullptr) {
            const int32_t *s2Ptr = static_cast<const int32_t *>(cuSeqlensCmpKv_->GetData());
            return static_cast<uint32_t>(s2Ptr[bIdx + 1U] - s2Ptr[bIdx]);
        }
    }
    // 如果 max_seqlen_cmp_kv 没传入，且 cmp_kv 为稀疏的，则尝试从topk中获取
    if (maxSeqlenCmpKv_ == 0 && isSparseCmpKv_) {
        return UINT32_MAX;
    }
    // 使用 max_seqlen_cmp_kv
    return static_cast<uint32_t>(maxSeqlenCmpKv_);
}

uint64_t SparseFlashMlaMetadataCpuKernel::GetRevertS2Size(uint32_t bIdx)
{
    uint32_t cmpS2Size = GetCmpS2SeqSize(bIdx);
    if (cmpResidualKv_ != nullptr && cmpResidualKv_->GetData() != nullptr) {
        const int32_t *residualPtr = static_cast<const int32_t *>(cmpResidualKv_->GetData());
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_) + residualPtr[bIdx];
    } else {
        return static_cast<uint64_t>(cmpS2Size) * static_cast<uint64_t>(cmpRatio_);
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcSplitInfo(SplitContext &splitContext)
{
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        splitInfo.s1GBaseNum[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (static_cast<uint64_t>(s1Size) * groupSize_) % mBaseSize_;
        if (hasOriKv_) {
            uint32_t curOriS2Size = GetOriS2SeqSize(bIdx);
            splitInfo.oriS2BaseNum[bIdx] = (static_cast<uint64_t>(curOriS2Size) + s2BaseSize_ - 1U) / s2BaseSize_;
        }
        if (hasCmpKv_) {
            uint32_t curCmpS2Size = GetCmpS2SeqSize(bIdx);
            splitInfo.cmpS2BaseNum[bIdx] = (static_cast<uint64_t>(curCmpS2Size) + s2BaseSize_ - 1U) / s2BaseSize_;
        }
        if (splitInfo.s1GBaseNum[bIdx] != 0U &&
            (splitInfo.oriS2BaseNum[bIdx] != 0U || splitInfo.cmpS2BaseNum[bIdx] != 0U)) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcOriPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    if (mode == SparseMode::BAND) {
        return oriPreToken_ == INT64_MAX ? INT64_MAX :
                                           static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + oriPreToken_;
    }
    return oriPreToken_;
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcOriNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(oriMaskMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return oriNextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return oriNextToken_ == INT64_MAX ?
                       INT64_MAX :
                       static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + oriNextToken_;
        default:
            return oriNextToken_;
    }
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcCmpPreTokenLeftUp(uint32_t s1Size, uint64_t s2Size)
{
    auto mode = static_cast<SparseMode>(cmpMaskMode_);
    if (mode == SparseMode::BAND) {
        return cmpPreToken_ == INT64_MAX ? INT64_MAX :
                                           static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + cmpPreToken_;
    }
    return cmpPreToken_;
}

int64_t SparseFlashMlaMetadataCpuKernel::CalcCmpNextTokenLeftUp(uint32_t s1Size, uint64_t s2Size)
{
    auto mode = static_cast<SparseMode>(cmpMaskMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return cmpNextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return cmpNextToken_ == INT64_MAX ?
                       INT64_MAX :
                       static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + cmpNextToken_;
        default:
            return cmpNextToken_;
    }
}

int64_t SparseFlashMlaMetadataCpuKernel::OriCalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t oriAlignCoefM = 16U;
    uint32_t oriAlignCoefS2 = 64U;
    uint32_t oriAlignBasicM = (basicM + oriAlignCoefM - 1U) / oriAlignCoefM;
    uint32_t oriAlignBasicS2 = (basicS2 + oriAlignCoefS2 - 1U) / oriAlignCoefS2;
    return static_cast<int64_t>(COST_WEIGHT_M * oriAlignBasicM + COST_WEIGHT_S2 * oriAlignBasicS2);
}

int64_t SparseFlashMlaMetadataCpuKernel::CmpCalcCost(uint32_t basicM, uint32_t basicS2)
{
    uint32_t cmpAlignCoefM = 16U;
    uint32_t cmpAlignCoefS2 = 64U;
    uint32_t cmpAlignBasicM = (basicM + cmpAlignCoefM - 1U) / cmpAlignCoefM;
    uint32_t cmpAlignBasicS2 = (basicS2 + cmpAlignCoefS2 - 1U) / cmpAlignCoefS2;
    return static_cast<int64_t>(COST_WEIGHT_M * cmpAlignBasicM + COST_WEIGHT_S2 * cmpAlignBasicS2);
}

void SparseFlashMlaMetadataCpuKernel::CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
                                                    uint32_t oriS2TailSize, uint32_t cmpS2TailSize)
{
    // ori 部分 cost
    if (hasOriKv_) {
        typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK] = OriCalcCost(s1NormalSize, s2NormalSize);
        typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : OriCalcCost(s1GTailSize, s2NormalSize);
        typeCost_[ORI_NORMAL_BLOCK][ORI_TAIL_BLOCK] =
            (oriS2TailSize == 0U) ? 0U : OriCalcCost(s1NormalSize, oriS2TailSize);
        typeCost_[ORI_TAIL_BLOCK][ORI_TAIL_BLOCK] =
            (s1GTailSize == 0U || oriS2TailSize == 0U) ? 0U : OriCalcCost(s1GTailSize, oriS2TailSize);
    }
    // cmp 部分 cost
    if (hasCmpKv_) {
        typeCost_[CMP_NORMAL_BLOCK][CMP_NORMAL_BLOCK] = CmpCalcCost(s1NormalSize, s2NormalSize);
        typeCost_[CMP_TAIL_BLOCK][CMP_NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CmpCalcCost(s1GTailSize, s2NormalSize);
        typeCost_[CMP_NORMAL_BLOCK][CMP_TAIL_BLOCK] =
            (cmpS2TailSize == 0U) ? 0U : CmpCalcCost(s1NormalSize, cmpS2TailSize);
        typeCost_[CMP_TAIL_BLOCK][CMP_TAIL_BLOCK] =
            (s1GTailSize == 0U || cmpS2TailSize == 0U) ? 0U : CmpCalcCost(s1GTailSize, cmpS2TailSize);
    }
}

Range<int64_t> SparseFlashMlaMetadataCpuKernel::CalcS2TokenRange(uint32_t s1GIdx, const BatchCache &batchCache,
                                                                 bool isCmpKv)
{
    // actual seq == 0
    if (!isCmpKv) {
        if (batchCache.s1Size == 0U || batchCache.oriS2Size == 0U) {
            return std::make_pair(0, 0);
        }
    } else {
        if (batchCache.s1Size == 0U || batchCache.cmpRevertS2Size == 0U) {
            return std::make_pair(0, 0);
        }
    }

    // no mask
    uint32_t hasMask = 1;
    int64_t s2Size =
        isCmpKv ? static_cast<int64_t>(batchCache.cmpRevertS2Size) : static_cast<int64_t>(batchCache.oriS2Size);
    hasMask = isCmpKv ? cmpAttentionMode_ : oriAttentionMode_;
    if (!hasMask) {
        return std::make_pair(0, s2Size - 1);
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
            // start and end locate in tow or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = 0;
    int64_t s2LastToken = 0;
    if (!isCmpKv) {
        s2FirstToken = s1FirstToken - batchCache.oriPreTokenLeftUp;
        s2LastToken =
            batchCache.oriNextTokenLeftUp == INT64_MAX ? INT64_MAX : s1LastToken + batchCache.oriNextTokenLeftUp;
    } else {
        s2FirstToken = s1FirstToken - batchCache.cmpPreTokenLeftUp;
        s2LastToken =
            batchCache.cmpNextTokenLeftUp == INT64_MAX ? INT64_MAX : s1LastToken + batchCache.cmpNextTokenLeftUp;
    }
    return std::make_pair(s2FirstToken, s2LastToken);
}

void SparseFlashMlaMetadataCpuKernel::CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext,
                                                     BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    if (hasOriKv_) {
        batchCache.oriS2Size = GetOriS2SeqSize(bIdx);
        batchCache.oriPreTokenLeftUp = CalcOriPreTokenLeftUp(batchCache.s1Size, batchCache.oriS2Size);
        batchCache.oriNextTokenLeftUp = CalcOriNextTokenLeftUp(batchCache.s1Size, batchCache.oriS2Size);
    }
    if (hasCmpKv_) {
        batchCache.cmpRevertS2Size = GetRevertS2Size(bIdx);
        batchCache.cmpPreTokenLeftUp = CalcCmpPreTokenLeftUp(batchCache.s1Size, batchCache.cmpRevertS2Size);
        batchCache.cmpNextTokenLeftUp = CalcCmpNextTokenLeftUp(batchCache.s1Size, batchCache.cmpRevertS2Size);
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcOriS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo)
{
    // 处理 ori 部分 block 信息
    if (s1GCache.oriS2Start >= s1GCache.oriS2End) {
        // ori 范围无效, 则整个 s1g 行等效为空行
        s1GCache.oriS1GBlock = 0;
        s1GCache.oriS1GCost = 0;
        s1GCache.oriS1GLastBlockCost = 0;
        s1GCache.oriS1GNormalBlockCost = 0;
    } else {
        // 计算 ori 方向 Block 数量及 Cost
        s1GCache.oriS1GBlock = s1GCache.oriS2End - s1GCache.oriS2Start;
        // 判断 ori S2 方向是否包含尾块
        uint32_t curOriTailS2Num = (s1GCache.oriS2TailSize != 0U) ? 1U : 0U;
        uint32_t curOriNormalS2Num = s1GCache.oriS1GBlock - curOriTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) &&
            splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
            s1GCache.oriS1GCost = typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK] * curOriNormalS2Num +
                                  typeCost_[ORI_TAIL_BLOCK][ORI_TAIL_BLOCK] * curOriTailS2Num;
            s1GCache.oriS1GLastBlockCost = curOriTailS2Num > 0U ? typeCost_[ORI_TAIL_BLOCK][ORI_TAIL_BLOCK] :
                                                                  typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK];
            s1GCache.oriS1GNormalBlockCost = typeCost_[ORI_TAIL_BLOCK][ORI_NORMAL_BLOCK];
        } else {
            s1GCache.oriS1GCost = typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK] * curOriNormalS2Num +
                                  typeCost_[ORI_NORMAL_BLOCK][ORI_TAIL_BLOCK] * curOriTailS2Num;
            s1GCache.oriS1GLastBlockCost = curOriTailS2Num > 0U ? typeCost_[ORI_NORMAL_BLOCK][ORI_TAIL_BLOCK] :
                                                                  typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK];
            s1GCache.oriS1GNormalBlockCost = typeCost_[ORI_NORMAL_BLOCK][ORI_NORMAL_BLOCK];
        }
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcCmpS1GCache(S1GCache &s1GCache, const SplitInfo &splitInfo)
{
    // 处理cmp部分block信息
    if (s1GCache.cmpS2Start >= s1GCache.cmpS2End) {
        // Cmp范围无效, Cost保持为 0
        s1GCache.cmpS1GBlock = 0;
        s1GCache.cmpS1GCost = 0;
        s1GCache.cmpS1GLastBlockCost = 0;
        s1GCache.cmpS1GNormalBlockCost = 0;
    } else {
        // 计算 cmp 方向 Block 数量及 Cost
        s1GCache.cmpS1GBlock = s1GCache.cmpS2End - s1GCache.cmpS2Start;
        // 判断 Cmp S2 方向是否包含尾块
        uint32_t curCmpTailS2Num = (s1GCache.cmpS2TailSize != 0U) ? 1U : 0U; // Updated check using local var
        uint32_t curCmpNormalS2Num = s1GCache.cmpS1GBlock - curCmpTailS2Num;
        if (s1GCache.s1GIdx == (splitInfo.s1GBaseNum[s1GCache.bIdx] - 1U) &&
            splitInfo.s1GTailSize[s1GCache.bIdx] != 0U) {
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

void SparseFlashMlaMetadataCpuKernel::CalcOriBlockRange(const Range<int64_t> &oriS2TokenRange,
                                                        const BatchCache &batchCache, S1GCache &s1GCache)
{
    int64_t oriS2FirstToken = oriS2TokenRange.first;
    int64_t oriS2LastToken = oriS2TokenRange.second;
    s1GCache.oriS2Start = 0;
    // ori 部分 s2 起止和 tailSize
    if (oriS2FirstToken >= static_cast<int64_t>(batchCache.oriS2Size) || oriS2LastToken < 0 ||
        oriS2LastToken < oriS2FirstToken) {
        s1GCache.oriS2End = 0;
        s1GCache.oriS2TailSize = 0;
    } else {
        oriS2FirstToken =
            Clip(oriS2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.oriS2Size - 1U));
        oriS2LastToken = Clip(oriS2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.oriS2Size - 1U));
        // oriS2LastToken 与 topk 取最小
        uint32_t s1Idx = GetS1Idx(batchCache.s1Size, s1GCache.s1GIdx);
        uint32_t bsStride = GetBsStride(s1GCache.bIdx, s1Idx);
        uint32_t oriTopkSize = GetOriTopkLength(bsStride);
        uint32_t actOriS2Size = isSparseOriKv_ ?
                                    std::min(static_cast<uint32_t>(oriS2LastToken - oriS2FirstToken + 1), oriTopkSize) :
                                    static_cast<uint32_t>(oriS2LastToken - oriS2FirstToken + 1);
        s1GCache.oriS2End = actOriS2Size == 0 ? 0 : (actOriS2Size - 1) / s2BaseSize_ + 1U;
        s1GCache.oriS2TailSize = actOriS2Size % s2BaseSize_;
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcCmpBlockRange(const Range<int64_t> &cmpRevertS2TokenRange,
                                                        const BatchCache &batchCache, S1GCache &s1GCache)
{
    int64_t cmpRevertS2FirstToken = cmpRevertS2TokenRange.first;
    int64_t cmpRevertS2LastToken = cmpRevertS2TokenRange.second;
    s1GCache.cmpS2Start = s1GCache.oriS2End;
    // cmp 部分 s2 起止和 tailSize
    if (cmpRevertS2FirstToken >= static_cast<int64_t>(batchCache.cmpRevertS2Size) || cmpRevertS2LastToken < 0 ||
        cmpRevertS2LastToken < cmpRevertS2FirstToken) {
        s1GCache.cmpS2End = s1GCache.cmpS2Start;
        s1GCache.cmpS2TailSize = 0;
    } else {
        cmpRevertS2FirstToken =
            Clip(cmpRevertS2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.cmpRevertS2Size - 1U));
        cmpRevertS2LastToken =
            Clip(cmpRevertS2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.cmpRevertS2Size - 1U));
        // 如果压缩后长度为0，则直接返回
        if ((cmpRevertS2LastToken + 1) / cmpRatio_ == 0) {
            s1GCache.cmpS2End = s1GCache.cmpS2Start;
            s1GCache.cmpS2TailSize = 0;
            return;
        }
        // 获取压缩后的 token 索引
        uint64_t cmpS2FirstToken =
            (cmpRevertS2FirstToken + 1) / cmpRatio_ == 0 ? 0 : (cmpRevertS2FirstToken + 1) / cmpRatio_ - 1U;
        uint64_t cmpS2LastToken = (cmpRevertS2LastToken + 1) / cmpRatio_ - 1U;
        // cmpS2LastToken 与 topk 取最小
        uint32_t s1Idx = GetS1Idx(batchCache.s1Size, s1GCache.s1GIdx);
        uint32_t bsStride = GetBsStride(s1GCache.bIdx, s1Idx);
        uint32_t cmpTopkSize = GetCmpTopkLength(bsStride);
        uint32_t actCmpS2Size = isSparseCmpKv_ ?
                                    std::min(static_cast<uint32_t>(cmpS2LastToken - cmpS2FirstToken + 1), cmpTopkSize) :
                                    static_cast<uint32_t>(cmpS2LastToken - cmpS2FirstToken + 1);
        s1GCache.cmpS2End =
            actCmpS2Size == 0 ? s1GCache.cmpS2Start : s1GCache.cmpS2Start + (actCmpS2Size - 1) / s2BaseSize_ + 1U;
        s1GCache.cmpS2TailSize = actCmpS2Size % s2BaseSize_;
    }
}

void SparseFlashMlaMetadataCpuKernel::GatherOriAndCmpCache(S1GCache &s1GCache)
{
    s1GCache.s2Start = 0;
    if (s1GCache.cmpS1GBlock > 0) {
        s1GCache.s1GLastBlockCost = s1GCache.cmpS1GLastBlockCost;
        s1GCache.s2End = s1GCache.cmpS2End;
    } else {
        s1GCache.s1GLastBlockCost = s1GCache.oriS1GLastBlockCost;
        s1GCache.s2End = s1GCache.oriS2End;
    }
    s1GCache.s1GBlock = s1GCache.oriS1GBlock + s1GCache.cmpS1GBlock;
    s1GCache.s1GCost = s1GCache.oriS1GCost + s1GCache.cmpS1GCost;
}

void SparseFlashMlaMetadataCpuKernel::CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext,
                                                   const BatchCache &batchCache, S1GCache &s1GCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 如果s1G是空行，则直接返回
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.oriS1GNormalBlockCost = 0;
        s1GCache.oriS1GLastBlockCost = 0;
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
    // 计算 ori_kv 有效负载起止
    if (hasOriKv_) {
        // 计算 ori_kv 的 s2Token 起止
        auto oriS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache, ORI_KV);
        // 计算 ori_kv 的 s2Block 起止
        CalcOriBlockRange(oriS2TokenRange, batchCache, s1GCache);
    } else {
        // ori_kv s2Token 起止初始化为0
        s1GCache.oriS2Start = 0;
        s1GCache.oriS2End = s1GCache.oriS2Start;
        s1GCache.oriS2TailSize = 0;
    }
    // 计算 cmp_kv 有效负载起止
    if (hasCmpKv_) {
        // 计算 cmp_kv 的 s2Token 起止
        auto cmpRevertS2TokenRange = CalcS2TokenRange(s1GIdx, batchCache, CMP_KV);
        // 计算 cmp_kv 的 s2Block 起止
        CalcCmpBlockRange(cmpRevertS2TokenRange, batchCache, s1GCache);
    } else {
        // cmp_kv s2Token 起止初始化为0
        s1GCache.cmpS2Start = s1GCache.oriS2End;
        s1GCache.cmpS2End = s1GCache.cmpS2Start;
        s1GCache.cmpS2TailSize = 0;
    }
    // 计算基本块负载
    CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[s1GCache.bIdx], s1GCache.oriS2TailSize,
                  s1GCache.cmpS2TailSize);
    // 计算 ori 和 cmp 部分的 cost 和 block 信息
    CalcOriS1GCache(s1GCache, splitInfo);
    CalcCmpS1GCache(s1GCache, splitInfo);
    // 汇总 ori 和 cmp 部分的 cost 和 block 信息
    GatherOriAndCmpCache(s1GCache);
}

void SparseFlashMlaMetadataCpuKernel::CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (GetS1SeqSize(bIdx) == 0U) {
        return;
    }
    if (!hasOriKv_ && !hasCmpKv_) {
        return;
    } else if (!hasOriKv_) {
        if (GetCmpS2SeqSize(bIdx) == 0U) {
            return;
        }
    } else if (!hasCmpKv_) {
        if (GetOriS2SeqSize(bIdx) == 0U) {
            return;
        }
    } else {
        if ((GetOriS2SeqSize(bIdx) == 0U) && GetCmpS2SeqSize(bIdx) == 0U) {
            return;
        }
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

void SparseFlashMlaMetadataCpuKernel::CalcCostInfo(SplitContext &splitContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    // 计算 batch 的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * numHeadsKv_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * numHeadsKv_;
    }
}

void SparseFlashMlaMetadataCpuKernel::UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
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
    if (assignContext.curBN2Idx == batchSize_ * numHeadsKv_) { // 所有负载全部分配完，设置最后一个核的右开区间，返回
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / numHeadsKv_ != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / numHeadsKv_;
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
        assignContext.curS2Idx = (supportFd) ? assignContext.s1GCache.oriS2Start : 0;
    }
}

void SparseFlashMlaMetadataCpuKernel::AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo &costInfo = splitContext.costInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
    while (assignContext.bN2Cost == 0 ||
           IsWithinTolerance(assignContext.coreCache.costLimit,
                             costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
                             assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;
        // to the end
        if (assignContext.curBN2Idx == batchSize_ * numHeadsKv_) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / numHeadsKv_ != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / numHeadsKv_;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void SparseFlashMlaMetadataCpuKernel::AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
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

int64_t SparseFlashMlaMetadataCpuKernel::CalcCurBlockCost(const AssignContext &assignContext)
{
    int64_t curCost = 0;
    if (assignContext.curS2Idx < assignContext.s1GCache.cmpS2Start) {
        curCost = assignContext.s1GCache.oriS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GCache.cmpS2Start - 1U)) {
            curCost = assignContext.s1GCache.oriS1GLastBlockCost;
        }
    } else {
        curCost = assignContext.s1GCache.cmpS1GNormalBlockCost;
        if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
            curCost = assignContext.s1GCache.cmpS1GLastBlockCost;
        }
    }
    return curCost;
}

void SparseFlashMlaMetadataCpuKernel::AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished || !supportFd) {
        return;
    }

    int64_t curCost = CalcCurBlockCost(assignContext);

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
        curCost = CalcCurBlockCost(assignContext);
    }
}

void SparseFlashMlaMetadataCpuKernel::ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
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

bool SparseFlashMlaMetadataCpuKernel::IsNeedRecordFDInfo(const AssignContext &assignContext,
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

void SparseFlashMlaMetadataCpuKernel::RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext,
                                                   SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / numHeadsKv_;
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
    result.fdRes.fdMSize[result.numOfFdHead] = curFdS1gSize;
    result.numOfFdHead++;
}

void SparseFlashMlaMetadataCpuKernel::AssignBlocksToCore(const SplitContext &splitContext, AssignContext &assignContext,
                                                         SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    result.firstFdDataWorkspaceIdx[assignContext.curCoreIdx] =
        assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;
    int64_t avgCost = assignContext.unassignedCost / (aicCoreNum_ - assignContext.curCoreIdx);
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
    result.maxS2GBaseNum = std::max(assignContext.coreCache.block, result.maxS2GBaseNum);
    // 对之前的归约信息进行记录并清理
    if (IsNeedRecordFDInfo(assignContext, result)) {
        RecordFDInfo(splitContext, assignContext, result);
        assignContext.preFdDataNum += assignContext.curKvSplitPart;
        assignContext.curKvSplitPart = 1U;
    }
    // 更新S2切分信息
    if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
        assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
        assignContext.curKvSplitPart++;
    }
}

void SparseFlashMlaMetadataCpuKernel::CalcSplitPlan(int64_t costLimit, const SplitContext &splitContext,
                                                    SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;
    const SplitInfo &splitInfo = splitContext.splitInfo;
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
    // 负载分配
    for (uint32_t i = 0; i < aicCoreNum_; ++i) {
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

void SparseFlashMlaMetadataCpuKernel::SplitFD(SplitResult &splitRes)
{
    // 计算FD的总数据量
    uint64_t totalFDLoad = 0;
    for (uint32_t i = 0; i < splitRes.numOfFdHead; i++) {
        totalFDLoad += splitRes.fdRes.fdS2SplitNum[i] * splitRes.fdRes.fdMSize[i];
    }
    // 计算当前最大冗余vec核数
    uint32_t emptyVectorNum = aivCoreNum_ - splitRes.numOfFdHead;
    // 计算每个核处理的load
    uint64_t averageLoad = (totalFDLoad + aivCoreNum_ - 1U) / aivCoreNum_; // 向上取整，避免核负载为0
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

bool SparseFlashMlaMetadataCpuKernel::BalanceSchedule(SplitResult &splitRes)
{
    SplitContext splitContext(batchSize_);

    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes.usedCoreNum = 1U;
        splitRes.bN2End[0] = batchSize_ * numHeadsKv_;
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
    splitRes.usedCoreNum = std::max(splitRes.usedCoreNum, 1U); // 至少使用1个core
    return true;
}

bool SparseFlashMlaMetadataCpuKernel::GenMetadata(SplitResult &splitRes)
{
    optiling::detail::SmlaMetadata *metadataPtr = static_cast<optiling::detail::SmlaMetadata *>(metadata_->GetData());
    *metadataPtr = {};
    // FA Metadata Generate
    if (isSplitG_) {
        for (size_t i = 0; i < aicCoreNum_; i++) {
            // 单核s2基本块最大数量
            metadataPtr->faMetadata[2 * i][FA_S2_MAX_NUM] = splitRes.maxS2GBaseNum;
            metadataPtr->faMetadata[2 * i + 1][FA_S2_MAX_NUM] = splitRes.maxS2GBaseNum;

            if (i >= splitRes.usedCoreNum) {
                metadataPtr->faMetadata[2 * i][FA_CORE_ENABLE_INDEX] = 0;     // AIC disenable
                metadataPtr->faMetadata[2 * i + 1][FA_CORE_ENABLE_INDEX] = 0; // AIC disenable
                continue;
            }
            metadataPtr->faMetadata[2 * i][FA_CORE_ENABLE_INDEX] = 1;     // AIC enable
            metadataPtr->faMetadata[2 * i + 1][FA_CORE_ENABLE_INDEX] = 1; // AIC enable
            // FA START
            metadataPtr->faMetadata[2 * i][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
            metadataPtr->faMetadata[2 * i][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
            metadataPtr->faMetadata[2 * i][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];

            metadataPtr->faMetadata[2 * i + 1][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
            metadataPtr->faMetadata[2 * i + 1][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
            metadataPtr->faMetadata[2 * i + 1][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
            // FA END
            metadataPtr->faMetadata[2 * i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
            metadataPtr->faMetadata[2 * i][FA_M_END_INDEX] = splitRes.gS1End[i];
            metadataPtr->faMetadata[2 * i][FA_S2_END_INDEX] = splitRes.s2End[i];

            metadataPtr->faMetadata[2 * i + 1][FA_BN2_END_INDEX] = splitRes.bN2End[i];
            metadataPtr->faMetadata[2 * i + 1][FA_M_END_INDEX] = splitRes.gS1End[i];
            metadataPtr->faMetadata[2 * i + 1][FA_S2_END_INDEX] = splitRes.s2End[i];
            // firstFdDataWorkspace
            metadataPtr->faMetadata[2 * i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
            metadataPtr->faMetadata[2 * i + 1][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] =
                splitRes.firstFdDataWorkspaceIdx[i];
        }
    } else {
        for (size_t i = 0; i < aicCoreNum_; ++i) {
            if (i >= splitRes.usedCoreNum) {
                metadataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 0; // AIC disenable
                continue;
            }
            metadataPtr->faMetadata[i][FA_CORE_ENABLE_INDEX] = 1; // AIC enable
            // FA START
            metadataPtr->faMetadata[i][FA_BN2_START_INDEX] = i == 0 ? 0 : splitRes.bN2End[i - 1];
            metadataPtr->faMetadata[i][FA_M_START_INDEX] = i == 0 ? 0 : splitRes.gS1End[i - 1];
            metadataPtr->faMetadata[i][FA_S2_START_INDEX] = i == 0 ? 0 : splitRes.s2End[i - 1];
            // FA END
            metadataPtr->faMetadata[i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
            metadataPtr->faMetadata[i][FA_M_END_INDEX] = splitRes.gS1End[i];
            metadataPtr->faMetadata[i][FA_S2_END_INDEX] = splitRes.s2End[i];
            // firstFdDataWorkspace
            metadataPtr->faMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] = splitRes.firstFdDataWorkspaceIdx[i];
        }
    }

    // FD Metadata Generate
    for (size_t i = 0; i < aivCoreNum_; ++i) {
        if (i >= splitRes.fdRes.fdUsedVecNum) {
            metadataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        metadataPtr->fdMetadata[i][FD_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = splitRes.fdRes.fdIdx[i];
        metadataPtr->fdMetadata[i][FD_BN2_IDX_INDEX] = splitRes.fdRes.fdBN2Idx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_M_IDX_INDEX] = splitRes.fdRes.fdMIdx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_WORKSPACE_IDX_INDEX] = splitRes.fdRes.fdWorkspaceIdx[curFdIdx];
        metadataPtr->fdMetadata[i][FD_WORKSPACE_NUM_INDEX] = splitRes.fdRes.fdS2SplitNum[curFdIdx];
        metadataPtr->fdMetadata[i][FD_M_START_INDEX] = splitRes.fdRes.fdMStart[i];
        metadataPtr->fdMetadata[i][FD_M_NUM_INDEX] = splitRes.fdRes.fdMNum[i];
    }
    return true;
}

namespace {
static const char *kernelType = "SparseFlashMlaMetadata";
REGISTER_CPU_KERNEL(kernelType, SparseFlashMlaMetadataCpuKernel);
} // namespace

}; // namespace aicpu
