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
 * \file quant_lightning_indexer_v2_metadata_check.h
 * \brief
 */

#include "log/log.h"
#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../../quant_lightning_indexer_v2/op_kernel/quant_lightning_indexer_v2_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr const char *QLI_V2_ACLNN_OP_NAME = "QuantLightningIndexerV2Metadata";

inline constexpr int64_t QLI_V2_QUANT_MODE_1 = 1;
inline constexpr int64_t QLI_V2_QUANT_MODE_2 = 2;
inline constexpr int64_t QLI_V2_QUANT_MODE_3 = 3;
inline constexpr int64_t QLI_V2_QUANT_MODE_4 = 4;
inline constexpr int64_t QLI_V2_QUANT_MODE_5 = 5;
inline constexpr int64_t QLI_V2_NO_MASK_MODE = 0;
inline constexpr int64_t QLI_V2_CAUSAL_MASK_MODE = 3;
inline constexpr int64_t QLI_V2_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t QLI_V2_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t QLI_V2_NUM_HEADS_Q_LOWER_BOUND = 1;
inline constexpr int64_t QLI_V2_NUM_HEADS_Q_UPPER_BOUND = 64;
inline constexpr int64_t QLI_V2_TOPK_LOWER_BOUND = 1;
inline constexpr int64_t QLI_V2_A5_TOPK_UPPER_BOUND = 8192;
inline constexpr int64_t QLI_V2_A3_TOPK_UPPER_BOUND = 2048;

inline bool IsTensorExistQliV2(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumQliV2(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeQliV2(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

inline bool IsTensorSourceQLiV2(const std::string &source) { return source != "batch_size"; }

inline int64_t GetRawShapeSizeQLiV2(const std::string &source, int64_t batchValue)
{
    if (source.find("cu_seqlens") != std::string::npos) {
        return batchValue + 1;
    }
    return batchValue;
}

int64_t GetQueryBatchSizeQliV2(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
                               const aclTensor *sequsedQOptional, const char *layoutQOptional, std::string &source)
{
    // 1. 如果sequsedQOptional 传了，使用sequsedQOptional获取BatchSize
    if (IsTensorExistQliV2(sequsedQOptional)) {
        source = "seqused_q";
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    // 2. 如果sequsedQOptional 没传，使用cuSeqlensQOptional获取BatchSize
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistQliV2(
                cuSeqlensQOptional)) { // 前序校验已保证layout_q = TND时，cu_seqlens_q必须传入，此通路必达
            source = "cu_seqlens_q";
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    source = "batch_size";
    // 3. 使用batchSize
    return batchSize;
}

int64_t GetKeyBatchSizeQliV2(int64_t batchSize, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedKOptional,
                             const char *layoutKOptional, std::string &source)
{
    // 1. 如果sequsedKOptional 传了，使用sequsedKOptional获取BatchSize
    if (IsTensorExistQliV2(sequsedKOptional)) {
        source = "seqused_q";
        return sequsedKOptional->GetViewShape().GetDim(0);
    }
    // 如果是 TND，必须使用 cuSeqlensKOptional获取BatchSize
    if (strcmp(layoutKOptional, "TND") == 0) {
        if (IsTensorExistQliV2(
                cuSeqlensKOptional)) { // 前序校验已保证layout_k = TND时，cu_seqlens_k必须传入，此通路必达
            source = "cu_seqlens_q";
            return cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    source = "batch_size";
    // 3. 使用batchSize
    return batchSize;
}

aclnnStatus CheckSingleParamQliV2(int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk,
                                  int64_t quantMode, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK,
                                  const char *layoutQOptional, const char *layoutKOptional, int64_t maskMode,
                                  int64_t cmpRatio, uint32_t aicCoreNum, uint32_t aivCoreNum,
                                  const std::string &socVersion)
{
    // num_heads_k 校验
    if (numHeadsK != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "num_heads_kv", std::to_string(numHeadsK),
                                              "The value of num_heads_kv should be 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // head_dim 校验
    if (headDim != 128) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "head_dim", std::to_string(numHeadsK),
                                              "The value of head_dim should be 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // batch_size 非负校验
    if (batchSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "batch_size", std::to_string(batchSize),
                                              "The value of batch_size should not be negative");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 校验
    if (maxSeqlenQ < -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                              "The value of max_seqlen_q should be >= -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_k 校验
    if (maxSeqlenK < -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "max_seqlen_k", std::to_string(maxSeqlenK),
                                              "The value of max_seqlen_k should be >= -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // mask_mode 校验
    if ((maskMode != QLI_V2_NO_MASK_MODE) && (maskMode != QLI_V2_CAUSAL_MASK_MODE)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "mask_mode", std::to_string(maskMode),
                                              "The value of mask_mode should be " +
                                                  std::to_string(QLI_V2_NO_MASK_MODE) + " or " +
                                                  std::to_string(QLI_V2_CAUSAL_MASK_MODE));
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q 校验
    if (layoutQOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "layout_q", "Layout_q is null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((strcmp(layoutQOptional, "TND") != 0) && (strcmp(layoutQOptional, "BSND") != 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "layout_q", layoutQOptional,
                                              "The value of layout_q must be TND or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_k 校验
    if (layoutKOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "layout_k", "Layout_k is null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((strcmp(layoutKOptional, "PA_BBND") != 0) && (strcmp(layoutQOptional, layoutKOptional) != 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "layout_q and layout_k",
                                               std::string(layoutQOptional) + " and " + std::string(layoutKOptional),
                                               "For layout_k != PA_BBND, layout_q and layout_k must be the same");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 A2/A3 参数
    if (socVersion.find("Ascend950") == std::string::npos) {
        // num_heads_q 校验
        CHECK_COND(numHeadsQ == QLI_V2_NUM_HEADS_Q_UPPER_BOUND, ACLNN_ERR_PARAM_INVALID,
                   "num_heads_q should be %lld, but got %lld", QLI_V2_NUM_HEADS_Q_UPPER_BOUND, numHeadsQ);
        // topk 校验
        CHECK_COND(topk >= QLI_V2_TOPK_LOWER_BOUND && topk <= QLI_V2_A3_TOPK_UPPER_BOUND, ACLNN_ERR_PARAM_INVALID,
                   "topk should be [%lld, %lld], but got %lld", QLI_V2_TOPK_LOWER_BOUND, QLI_V2_A3_TOPK_UPPER_BOUND,
                   topk);
        // quant_mode 校验
        CHECK_COND(quantMode == QLI_V2_QUANT_MODE_2, ACLNN_ERR_PARAM_INVALID, "quant_mode should be 2, but got %lld",
                   quantMode);
        // cmp_ratio 校验
        CHECK_COND((cmpRatio >= QLI_V2_CMP_RATIO_LOWER_BOUND) && (cmpRatio <= QLI_V2_CMP_RATIO_UPPER_BOUND) &&
                       ((cmpRatio & (cmpRatio - 1)) == 0),
                   ACLNN_ERR_PARAM_INVALID, "cmp_ratio should be 1/2/4/8/16/32/64/128, but got %lld", cmpRatio);
        CHECK_COND(strcmp(layoutKOptional, "PA_BBND") == 0, ACLNN_ERR_PARAM_INVALID,
                   "layout_k must be PA_BBND, but got %s", layoutKOptional);
    } else { // 校验 A5参数
        // num_heads_q 校验
        if (numHeadsQ < QLI_V2_NUM_HEADS_Q_LOWER_BOUND || numHeadsQ > QLI_V2_NUM_HEADS_Q_UPPER_BOUND) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "num_heads_q", std::to_string(numHeadsQ),
                                                  "The value of num_heads_q should be in range [" +
                                                      std::to_string(QLI_V2_NUM_HEADS_Q_LOWER_BOUND) + ", " +
                                                      std::to_string(QLI_V2_NUM_HEADS_Q_UPPER_BOUND) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // topk 校验
        if (topk < QLI_V2_TOPK_LOWER_BOUND || topk > QLI_V2_A5_TOPK_UPPER_BOUND) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "topk", std::to_string(topk),
                                                  "The value of topk should be in range [" +
                                                      std::to_string(QLI_V2_TOPK_LOWER_BOUND) + ", " +
                                                      std::to_string(QLI_V2_A5_TOPK_UPPER_BOUND) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // quant_mode 校验
        if ((quantMode != QLI_V2_QUANT_MODE_1) && (quantMode != QLI_V2_QUANT_MODE_2) &&
            (quantMode != QLI_V2_QUANT_MODE_3) && (quantMode != QLI_V2_QUANT_MODE_4) &&
            (quantMode != QLI_V2_QUANT_MODE_5)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                QLI_V2_ACLNN_OP_NAME, "quant_mode", std::to_string(quantMode),
                "The value of quant_mode should be in [" + std::to_string(QLI_V2_QUANT_MODE_1) + ", " +
                    std::to_string(QLI_V2_QUANT_MODE_2) + ", " + std::to_string(QLI_V2_QUANT_MODE_3) + ", " +
                    std::to_string(QLI_V2_QUANT_MODE_4) + ", " + std::to_string(QLI_V2_QUANT_MODE_5) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // cmp_ratio 校验
        if ((cmpRatio < QLI_V2_CMP_RATIO_LOWER_BOUND) || (cmpRatio > QLI_V2_CMP_RATIO_UPPER_BOUND)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "cmp_ratio", std::to_string(cmpRatio),
                                                  "The value of cmp_ratio should be in range [" +
                                                      std::to_string(QLI_V2_CMP_RATIO_LOWER_BOUND) + ", " +
                                                      std::to_string(QLI_V2_CMP_RATIO_UPPER_BOUND) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if ((strcmp(layoutKOptional, "TND") != 0) && (strcmp(layoutKOptional, "BSND") != 0) &&
            (strcmp(layoutKOptional, "PA_BBND") != 0)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "layout_k", layoutKOptional,
                                                  "The value of layout_k must be in [TND, BSND, PA_BBND]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 layout_q 为 BSND 时，max_seqlen_q 必须大于 0
    if ((strcmp(layoutQOptional, "BSND") == 0) && (maxSeqlenQ <= 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                               "When layout_q is BSND, the value of max_seqlen_q "
                                               "must be equal to the size of the second axis of q");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 layout_k 为 BSND 时，max_seqlen_k 必须大于 0
    if ((strcmp(layoutKOptional, "BSND") == 0) && (maxSeqlenK <= 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "max_seqlen_k", std::to_string(maxSeqlenK),
                                               "When layout_k is BSND, the value of max_seqlen_k "
                                               "must be equal to the size of the second axis of k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 核心数校验
    CHECK_COND(aicCoreNum > 0, ACLNN_ERR_PARAM_INVALID, "AIC num should be larger than 0, but got %u", aicCoreNum);
    CHECK_COND(aicCoreNum <= optiling::AIC_CORE_MAX_NUM, ACLNN_ERR_PARAM_INVALID,
               "The maximum supported AIC num is %u, but got %u", optiling::AIC_CORE_MAX_NUM, aicCoreNum);
    CHECK_COND(aivCoreNum > 0, ACLNN_ERR_PARAM_INVALID, "AIV num should be larger than 0, but got %u", aivCoreNum);
    CHECK_COND(aivCoreNum <= optiling::AIV_CORE_MAX_NUM, ACLNN_ERR_PARAM_INVALID,
               "The maximum supported AIV num is %u, but got %u", optiling::AIV_CORE_MAX_NUM, aivCoreNum);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceQliV2(int64_t maskMode, int64_t cmpRatio, const aclTensor *cuSeqlensQOptional,
                                const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
                                const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
                                int64_t maxSeqlenQ, int64_t maxSeqlenK, const char *layoutQOptional,
                                const char *layoutKOptional, const aclTensor *metadata)
{
    // cu_seqlens_q 存在性校验
    if ((strcmp(layoutQOptional, "TND") == 0) && !IsTensorExistQliV2(cuSeqlensQOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "cu_seqlens_q",
                                                 "When layout_q is TND, cu_seqlens_q must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q BSND, seqused_q 不存在时，max_seqlen_q 不能为-1
    if (strcmp(layoutQOptional, "BSND") == 0 && !IsTensorExistQliV2(sequsedQOptional) && (maxSeqlenQ <= -1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            QLI_V2_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
            "When layout_q is BSND and seqused_q is not provided, max_seqlen_q can not be -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cu_seqlens_k 存在性校验
    if ((strcmp(layoutKOptional, "TND") == 0) && !IsTensorExistQliV2(cuSeqlensKOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "cu_seqlens_k",
                                                 "When layout_k is TND, cu_seqlens_k must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // seqused_k 存在性校验
    if ((strcmp(layoutKOptional, "PA_BBND") == 0) && !IsTensorExistQliV2(sequsedKOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "seqused_k",
                                                 "When layout_k is PA_BBND, seqused_k must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_k BSND, seqused_k 不存在时，max_seqlen_k 不能为-1
    if (strcmp(layoutKOptional, "BSND") == 0 && !IsTensorExistQliV2(sequsedKOptional) && maxSeqlenK <= -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            QLI_V2_ACLNN_OP_NAME, "max_seqlen_k", std::to_string(maxSeqlenK),
            "When layout_k is BSND and seqused_k is not provided, max_seqlen_k can not be -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cmp_residual_k 存在性校验
    if ((cmpRatio != QLI_V2_CMP_RATIO_LOWER_BOUND) && (maskMode == QLI_V2_CAUSAL_MASK_MODE) &&
        !IsTensorExistQliV2(cmpResidualKOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            QLI_V2_ACLNN_OP_NAME, "cmp_residual_k",
            "When cmp_ratio is not 1 and mask_mode is CAUSAL, cmp_residual_k must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // metadata 存在性校验
    if (!IsTensorExistQliV2(metadata)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "metadata", "Metadata is nullptr");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencyQliV2(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
                                  const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
                                  const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
                                  const char *layoutQOptional, const char *layoutKOptional, const aclTensor *metadata)
{
    int64_t dimNum = -1;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    // 校验 cu_seqlens_q
    if (IsTensorExistQliV2(cuSeqlensQOptional)) {
        dimNum = GetDimNumQliV2(cuSeqlensQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME, "cu_seqlens_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(cuSeqlensQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "cu_seqlens_q", ToString(dataType).GetString(),
                                                  "The dtype of cu_seqlens_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cu_seqlens_k
    if (IsTensorExistQliV2(cuSeqlensKOptional)) {
        dimNum = GetDimNumQliV2(cuSeqlensKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME, "cu_seqlens_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(cuSeqlensKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "cu_seqlens_k", ToString(dataType).GetString(),
                                                  "The dtype of cu_seqlens_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_q
    if (IsTensorExistQliV2(sequsedQOptional)) {
        dimNum = GetDimNumQliV2(sequsedQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME, "seqused_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(sequsedQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "seqused_q", ToString(dataType).GetString(),
                                                  "The dtype of seqused_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_k
    if (IsTensorExistQliV2(sequsedKOptional)) {
        dimNum = GetDimNumQliV2(sequsedKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME, "seqused_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(sequsedKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "seqused_k", ToString(dataType).GetString(),
                                                  "The dtype of seqused_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cmp_residual_k
    if (IsTensorExistQliV2(cmpResidualKOptional)) {
        dimNum = GetDimNumQliV2(cmpResidualKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME, "cmp_residual_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(cmpResidualKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "cmp_residual_k",
                                                  ToString(dataType).GetString(),
                                                  "The dtype of cmp_residual_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 metadata
    if (IsTensorExistQliV2(metadata)) {
        dimNum = GetDimNumQliV2(metadata);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(QLI_V2_ACLNN_OP_NAME, "metadata", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeQliV2(metadata);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(QLI_V2_ACLNN_OP_NAME, "metadata", ToString(dataType).GetString(),
                                                  "The dtype of metadata must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素数
        if (metadata->GetViewShape().GetDim(0) != optiling::QLI_V2_METADATA_TOTAL_SIZE) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The element num of metadata must be %u, but got %lld",
                    optiling::QLI_V2_METADATA_TOTAL_SIZE, metadata->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验batch
    std::string querySource;
    std::string keySource;
    int64_t queryBatchSize =
        GetQueryBatchSizeQliV2(batchSize, cuSeqlensQOptional, sequsedQOptional, layoutQOptional, querySource);
    int64_t keyBatchSize =
        GetKeyBatchSizeQliV2(batchSize, cuSeqlensKOptional, sequsedKOptional, layoutKOptional, keySource);
    if (queryBatchSize != keyBatchSize) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            QLI_V2_ACLNN_OP_NAME, querySource + " and " + keySource,
            std::to_string(GetRawShapeSizeQLiV2(querySource, queryBatchSize)) + " and " +
                std::to_string(GetRawShapeSizeQLiV2(keySource, keyBatchSize)),
            "The batch_size obtained from query should be the same as that obtained from key");
        if (IsTensorSourceQLiV2(querySource) && IsTensorSourceQLiV2(keySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                QLI_V2_ACLNN_OP_NAME, querySource + " and " + keySource,
                std::to_string(GetRawShapeSizeQLiV2(querySource, queryBatchSize)) + " and " +
                    std::to_string(GetRawShapeSizeQLiV2(keySource, keyBatchSize)),
                "The batch_size obtained from query should be the same as that obtained from key");
        } else if (IsTensorSourceQLiV2(querySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                QLI_V2_ACLNN_OP_NAME, querySource, std::to_string(GetRawShapeSizeQLiV2(querySource, queryBatchSize)),
                "The batch_size obtained from query should be the same as that obtained from key");
        } else {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                QLI_V2_ACLNN_OP_NAME, keySource, std::to_string(GetRawShapeSizeQLiV2(keySource, keyBatchSize)),
                "The batch_size obtained from query should be the same as that obtained from key");
        }
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验TND场景q维度一致性
    if (strcmp(layoutQOptional, "TND") == 0 && IsTensorExistQliV2(sequsedQOptional)) {
        int64_t cuSeqlensQBatchSize = cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        CHECK_COND(
            cuSeqlensQBatchSize == queryBatchSize, ACLNN_ERR_PARAM_INVALID,
            "When layout_q is TND and seqused_q is passed, The batch_size obtained from cu_seqlens_q should be the "
            "same as that obtained from seqused_q, but got %lld and %lld",
            cuSeqlensQBatchSize, queryBatchSize);
        if (cuSeqlensQBatchSize != queryBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                QLI_V2_ACLNN_OP_NAME, "cu_seqlens_q and seqused_q",
                std::to_string(cuSeqlensQBatchSize) + " and " + std::to_string(queryBatchSize),
                "When layout_q is TND and seqused_q is passed, "
                "the shape size of cu_seqlens_q minus 1 must be equal to "
                "the shape size of seqused_q");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验TND场景k维度一致性
    if (strcmp(layoutKOptional, "TND") == 0 && IsTensorExistQliV2(sequsedKOptional)) {
        int64_t cuSeqlensKBatchSize = cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        if (cuSeqlensKBatchSize != keyBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                QLI_V2_ACLNN_OP_NAME, "cu_seqlens_k and seqused_k",
                std::to_string(cuSeqlensKBatchSize) + " and " + std::to_string(keyBatchSize),
                "When layout_k is TND and seqused_k is passed, "
                "the shape size of cu_seqlens_k minus 1 must be equal to "
                "the shape size of seqused_k");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cmp_residual_k 元素数
    auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
    if (IsTensorExistQliV2(cmpResidualKOptional) && (cmpResidualKBatch != queryBatchSize)) {
        if (IsTensorSourceQLiV2(querySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                QLI_V2_ACLNN_OP_NAME, "cmp_residual_k and " + querySource,
                std::to_string(cmpResidualKBatch) + " and " +
                    std::to_string(GetRawShapeSizeQLiV2(querySource, queryBatchSize)),
                "The batch_size of cmp_residual_k should match the valid batch size");
        } else {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                QLI_V2_ACLNN_OP_NAME, "cmp_residual_k", std::to_string(cmpResidualKBatch),
                "The batch_size of cmp_residual_k should match the valid batch size");
        }
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckQliV2(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
                             const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
                             const aclTensor *cmpResidualKOptional, int64_t numHeadsQ, int64_t numHeadsK,
                             int64_t headDim, int64_t topk, int64_t quantMode, int64_t batchSize, int64_t maxSeqlenQ,
                             int64_t maxSeqlenK, char *layoutQOptional, char *layoutKOptional, int64_t maskMode,
                             int64_t cmpRatio, const aclTensor *metadata, uint32_t aicCoreNum, uint32_t aivCoreNum,
                             const std::string &socVersion)
{
    auto ret =
        CheckSingleParamQliV2(numHeadsQ, numHeadsK, headDim, topk, quantMode, batchSize, maxSeqlenQ, maxSeqlenK,
                              layoutQOptional, layoutKOptional, maskMode, cmpRatio, aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceQliV2(maskMode, cmpRatio, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional,
                              sequsedKOptional, cmpResidualKOptional, maxSeqlenQ, maxSeqlenK, layoutQOptional,
                              layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencyQliV2(batchSize, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                                cmpResidualKOptional, layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
}
#endif
