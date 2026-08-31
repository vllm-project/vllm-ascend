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
 * \file flash_mla_with_kvcache_metadata_check.h
 * \brief MLA metadata checker.
 * Base-size convention (D17/B3): mBaseSize=64 / s2BaseSize=128 are FIXED for MLA - the head fields
 * HEAD_M_BASE_SIZE_INDEX / HEAD_S2_BASE_SIZE_INDEX always carry 64/128 (forced in GenMetadata, not
 * derived); flash_mla kernel-side static_assert (flash_mla_block_cube_noquant_mla.h) is the invariant.
 */

#ifndef FLASH_MLA_WITH_KVCACHE_METADATA_CHECK_H
#define FLASH_MLA_WITH_KVCACHE_METADATA_CHECK_H

#include <unordered_set>
#include <string>
#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

class FlashMlaWithKvcacheMetadataCheck {
public:
    // m0070 参数序（与 meta.txt 文档一致）：无 cu_seqlens_kv/seqused_kv/win_*
    static inline aclnnStatus ParamsCheck(const aclTensor *cuSeqlensQOptional, const aclTensor *cacheSeqlensOptional,
                                          const aclTensor *sequsedQOptional, int64_t maxSeqlenQ,
                                          int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv,
                                          int64_t headDimQk, int64_t headDimV, int64_t maskMode, const char *layoutQ,
                                          const aclTensor *metadata);

private:
    static constexpr int64_t NONE_VALUE = -1;
    // MLA 硬约束（m0070）：kv head num 恒 1；head_dim_qk = q/k_cache 最后维 576（nope 512 + rope 64）；
    // head_dim_v = 512（value/输出宽）；mask {0,3}；q 布局 ∈ {TND,BNSD,BSND}
    // （layout_kv/layout_out 已从 metadata 算子移除：调度器不消费 KV 布局，输出不独立配置）
    static constexpr int64_t MLA_KV_HEADS = 1;
    static constexpr int64_t MLA_HEAD_DIM_QK = 576;
    static constexpr int64_t MLA_HEAD_DIM_V = 512;

    static inline bool IsTensorExist(const aclTensor *tensor);
    static inline aclnnStatus CheckSeqLens(bool isCu, int64_t batchSize, const aclTensor *seqLens);

    static inline aclnnStatus CheckBaseAttr(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                            int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDimQk,
                                            int64_t headDimV, const char *layoutQ);

    static inline aclnnStatus CheckMask(int64_t maskMode);

    static inline aclnnStatus CheckExistency(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                             const char *layoutQ,
                                             const aclTensor *cuSeqlensQOptional,
                                             const aclTensor *cacheSeqlensOptional,
                                             const aclTensor *metadata);

    static inline aclnnStatus CheckConsistency(int64_t batchSize, int64_t numHeadsQ,
                                               const aclTensor *cuSeqlensQOptional,
                                               const aclTensor *cacheSeqlensOptional,
                                               const aclTensor *sequsedQOptional);
};

inline aclnnStatus
FlashMlaWithKvcacheMetadataCheck::ParamsCheck(const aclTensor *cuSeqlensQOptional, const aclTensor *cacheSeqlensOptional,
                                              const aclTensor *sequsedQOptional, int64_t maxSeqlenQ,
                                              int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv,
                                              int64_t headDimQk, int64_t headDimV, int64_t maskMode, const char *layoutQ,
                                              const aclTensor *metadata)
{
    // batch size 由必传的 cacheSeqlens 长度推导（每 batch 一项），与 Python/dispatcher 侧同源（no batch_size attr）
    int64_t batchSize = NONE_VALUE;
    if (cacheSeqlensOptional != nullptr && cacheSeqlensOptional->GetViewShape().GetDimNum() > 0) {
        batchSize = cacheSeqlensOptional->GetViewShape().GetDim(0);
    }
    auto ret = CheckBaseAttr(batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDimQk, headDimV,
                             layoutQ);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckMask(maskMode);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckExistency(batchSize, maxSeqlenQ, maxSeqlenKv, layoutQ, cuSeqlensQOptional,
                         cacheSeqlensOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = CheckConsistency(batchSize, numHeadsQ, cuSeqlensQOptional, cacheSeqlensOptional, sequsedQOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

inline aclnnStatus
FlashMlaWithKvcacheMetadataCheck::CheckBaseAttr(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                                int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDimQk,
                                                int64_t headDimV, const char *layoutQ)
{
    int64_t MIN_BATCH = 0;
    int64_t MAX_BATCH = 65536;
    CHECK_COND(((batchSize == NONE_VALUE) || (batchSize > MIN_BATCH && batchSize < MAX_BATCH)), ACLNN_ERR_RUNTIME_ERROR,
               "batchSize must be %ld or between (%ld, %ld), but got %ld", NONE_VALUE, MIN_BATCH, MAX_BATCH, batchSize);
    CHECK_COND((maxSeqlenQ == NONE_VALUE || maxSeqlenQ >= 0), ACLNN_ERR_RUNTIME_ERROR,
               "maxSeqlenQ must be %ld or greater than or equal to 0, but got %ld", NONE_VALUE, maxSeqlenQ);
    CHECK_COND((maxSeqlenKv == NONE_VALUE || maxSeqlenKv >= 0), ACLNN_ERR_RUNTIME_ERROR,
               "maxSeqlenKv must be %ld or greater than or equal to 0, but got %ld", NONE_VALUE, maxSeqlenKv);

    CHECK_COND(numHeadsQ > 0, ACLNN_ERR_RUNTIME_ERROR, "numHeadsQ must be greater than 0, but got %ld", numHeadsQ);
    CHECK_COND(numHeadsKv == MLA_KV_HEADS, ACLNN_ERR_RUNTIME_ERROR,
               "MLA kv head num must be %ld, but got %ld", MLA_KV_HEADS, numHeadsKv);
    CHECK_COND(headDimQk == MLA_HEAD_DIM_QK, ACLNN_ERR_RUNTIME_ERROR,
               "MLA head_dim_qk (q/k_cache last dim, nope 512 + rope 64) must be %ld, but got %ld", MLA_HEAD_DIM_QK,
               headDimQk);
    CHECK_COND(headDimV == MLA_HEAD_DIM_V, ACLNN_ERR_RUNTIME_ERROR,
               "MLA head_dim_v (value/output width) must be %ld, but got %ld", MLA_HEAD_DIM_V, headDimV);

    static const std::unordered_set<std::string> layoutQSet = {"TND", "BNSD", "BSND"};
    CHECK_COND(layoutQSet.count(layoutQ) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "layoutQ only supports TND, BNSD, BSND, but got %s", layoutQ);

    return ACLNN_SUCCESS;
}

inline aclnnStatus FlashMlaWithKvcacheMetadataCheck::CheckMask(int64_t maskMode)
{
    constexpr int64_t NO_MASK = 0;
    constexpr int64_t CAUSAL_MASK = 3;

    static const std::unordered_set<int64_t> maskSet = {NO_MASK, CAUSAL_MASK};
    CHECK_COND(maskSet.count(maskMode) > 0, ACLNN_ERR_RUNTIME_ERROR,
               "maskMode only supports %ld, %ld, but got %ld", NO_MASK, CAUSAL_MASK, maskMode);
    return ACLNN_SUCCESS;
}

inline aclnnStatus
FlashMlaWithKvcacheMetadataCheck::CheckExistency(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
                                                 const char *layoutQ,
                                                 const aclTensor *cuSeqlensQOptional,
                                                 const aclTensor *cacheSeqlensOptional,
                                                 const aclTensor *metadata)
{
    CHECK_COND(metadata != nullptr, ACLNN_ERR_RUNTIME_ERROR, "metadata should be provided, but got null");

    // cuSeqlensQ：layoutQ 为 TND 时必传，非 TND（BNSD/BSND）时禁传（与主算子 seq_len_checker 判定一致）
    if (strcmp(layoutQ, "TND") == 0) {
        CHECK_COND(IsTensorExist(cuSeqlensQOptional), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutQ is TND, cuSeqlensQOptional should be provided, but got null");
    } else {
        CHECK_COND(!IsTensorExist(cuSeqlensQOptional), ACLNN_ERR_RUNTIME_ERROR,
                   "When layoutQ is not TND, cuSeqlensQOptional should not be provided, but got non-null");
    }

    CHECK_COND((batchSize != NONE_VALUE), ACLNN_ERR_RUNTIME_ERROR,
               "cacheSeqlens must be provided to derive batch_size, but got null");
    CHECK_COND(((maxSeqlenKv > 0) || IsTensorExist(cacheSeqlensOptional)), ACLNN_ERR_RUNTIME_ERROR,
               "At least one of maxSeqlenKv or cacheSeqlensOptional must be provided");

    return ACLNN_SUCCESS;
}

inline aclnnStatus
FlashMlaWithKvcacheMetadataCheck::CheckConsistency(int64_t batchSize, int64_t numHeadsQ,
                                                   const aclTensor *cuSeqlensQOptional,
                                                   const aclTensor *cacheSeqlensOptional,
                                                   const aclTensor *sequsedQOptional)
{
    if (batchSize <= 0) {
        return ACLNN_SUCCESS;
    }

    CHECK_COND(CheckSeqLens(true, batchSize, cuSeqlensQOptional) == ACLNN_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
               "cuSeqlensQOptional is not valid!");
    CHECK_COND(CheckSeqLens(false, batchSize, cacheSeqlensOptional) == ACLNN_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
               "cacheSeqlensOptional is not valid!");
    CHECK_COND(CheckSeqLens(false, batchSize, sequsedQOptional) == ACLNN_SUCCESS, ACLNN_ERR_RUNTIME_ERROR,
               "sequsedQOptional is not valid!");
    CHECK_COND(numHeadsQ > 0, ACLNN_ERR_RUNTIME_ERROR, "numHeadsQ must be greater than 0, but got %ld", numHeadsQ);

    return ACLNN_SUCCESS;
}

inline bool FlashMlaWithKvcacheMetadataCheck::IsTensorExist(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0) &&
           (tensor->GetData() != nullptr);
}

inline aclnnStatus
FlashMlaWithKvcacheMetadataCheck::CheckSeqLens(bool isCu, int64_t batchSize, const aclTensor *seqLens)
{
    if (seqLens == nullptr) {
        return ACLNN_SUCCESS;
    }
    // 空 tensor = 缺省 optional（get_valid_tensor 对 None 生成 (0,) int32）；必传性由 CheckExistency 保证
    if (seqLens->GetViewShape().GetDimNum() == 0 || seqLens->GetViewShape().GetDim(0) == 0) {
        return ACLNN_SUCCESS;
    }

    CHECK_COND(seqLens->GetViewShape().GetDimNum() == 1, ACLNN_ERR_RUNTIME_ERROR,
               "seqLens must be 1D tensor, but got %ld dims", seqLens->GetViewShape().GetDimNum());

    if (isCu) {
        CHECK_COND(seqLens->GetViewShape().GetDim(0) == batchSize + 1, ACLNN_ERR_RUNTIME_ERROR,
                   "cuSeqLens shape must be (batchSize+1,), but got %ld", seqLens->GetViewShape().GetDim(0));
    } else {
        CHECK_COND(seqLens->GetViewShape().GetDim(0) == batchSize, ACLNN_ERR_RUNTIME_ERROR,
                   "seqLens shape must be (batchSize,), but got %ld", seqLens->GetViewShape().GetDim(0));
    }

    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif

#endif // FLASH_MLA_WITH_KVCACHE_METADATA_CHECK_H
