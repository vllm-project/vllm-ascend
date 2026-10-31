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
 * \file sparse_flash_mla_metadata_check.h
 * \brief
 */

#include "log/log.h"
#include "opdev/format_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../../sparse_flash_mla/op_kernel/sparse_flash_mla_kernel_metadata.h"
#include <cstring>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr const char *SMLA_ACLNN_OP_NAME = "aclnnSparseFlashMlaMetadata";

enum class SparseModeSmla : uint8_t {
    DEFAULT_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    SPARSE_BUTT,
};

inline constexpr int64_t SMLA_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t SMLA_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t SMLA_NUM_HEADS_Q_LOWER_BOUND = 1;
inline constexpr int64_t SMLA_NUM_HEADS_Q_UPPER_BOUND = 128;

inline bool IsPowerOfTwoInRangeSmla(int64_t value, int64_t minValue, int64_t maxValue)
{
    return value >= minValue && value <= maxValue && ((value & (value - 1)) == 0);
}

inline bool IsCmpRatioSupportSmla(const char *socVersion, bool hasCmpKv, int64_t cmpTopk, int64_t cmpRatio)
{
    if (!hasCmpKv) {
        return cmpRatio == 0;
    }
    if (socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr) {
        return cmpRatio >= SMLA_CMP_RATIO_LOWER_BOUND && cmpRatio <= SMLA_CMP_RATIO_UPPER_BOUND;
    }
    return (cmpTopk > 0) ? (cmpRatio == 1 || cmpRatio == 2 || cmpRatio == 4) : (cmpRatio == 128);
}

inline bool IsTensorExistSmla(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

aclnnStatus CheckReservedOptionalTensorSmla(const aclTensor *tensor, const char *tensorName)
{
    if (!IsTensorExistSmla(tensor)) {
        return ACLNN_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, tensorName,
                                             std::string(tensorName) + " is reserved and does not support "
                                                                       "non-empty tensor");
    return ACLNN_ERR_PARAM_INVALID;
}

int64_t GetDimNumSmla(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeSmla(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

inline bool IsTensorSourceSmla(const std::string &source)
{
    return source != "batch_size";
}

inline int64_t GetRawShapeSizeSmla(const std::string &source, int64_t batchValue)
{
    if (source.find("cu_seqlens") != std::string::npos) {
        return batchValue + 1;
    }
    return batchValue;
}

inline std::string GetSourceDescSmla(const std::string &source)
{
    if (source == "batch_size") {
        return "batch_size";
    }
    if (source.find("cu_seqlens") != std::string::npos) {
        return "the shape size of " + source + " minus 1";
    }
    return "the shape size of " + source;
}

aclnnStatus CheckSingleParamSmla(int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv, int64_t maxSeqlenCmpKv,
                                 int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t oriTopk,
                                 int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode,
                                 int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional,
                                 const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv, uint32_t aicCoreNum,
                                 uint32_t aivCoreNum, const char *socVersion)
{
    // batch_size >= 0
    if (batchSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "batch_size", std::to_string(batchSize),
                                              "The value of batch_size must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q >= 0
    if (maxSeqlenQ < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                              "The value of max_seqlen_q must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_ori_kv >= 0
    if (maxSeqlenOriKv < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "max_seqlen_ori_kv", std::to_string(maxSeqlenOriKv),
                                              "The value of max_seqlen_ori_kv must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_cmp_kv >= 0
    if (maxSeqlenCmpKv < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "max_seqlen_cmp_kv", std::to_string(maxSeqlenCmpKv),
                                              "The value of max_seqlen_cmp_kv must be greater than or equal to 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q [1, 128]
    if (numHeadsQ < SMLA_NUM_HEADS_Q_LOWER_BOUND || numHeadsQ > SMLA_NUM_HEADS_Q_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "num_heads_q", std::to_string(numHeadsQ),
                                              "The current value is not within the valid range. "
                                              "The valid range is [" +
                                                  std::to_string(SMLA_NUM_HEADS_Q_LOWER_BOUND) + ", " +
                                                  std::to_string(SMLA_NUM_HEADS_Q_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv: 1
    if (numHeadsKv != 1) {
        OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "num_heads_kv", std::to_string(numHeadsKv), "1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (numHeadsQ % numHeadsKv != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(SMLA_ACLNN_OP_NAME, "num_heads_q, num_heads_kv",
                                               std::to_string(numHeadsQ) + ", " + std::to_string(numHeadsKv),
                                               "The value of num_heads_q must be divisible by "
                                               "that of num_heads_kv");
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t headRatio = numHeadsQ / numHeadsKv;
    if (socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr) {
        if (headRatio < SMLA_NUM_HEADS_Q_LOWER_BOUND || headRatio > SMLA_NUM_HEADS_Q_UPPER_BOUND) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "num_heads_q / num_heads_kv",
                                                  std::to_string(headRatio),
                                                  "The current value is not within the valid range. "
                                                  "The valid range is [" +
                                                      std::to_string(SMLA_NUM_HEADS_Q_LOWER_BOUND) + ", " +
                                                      std::to_string(SMLA_NUM_HEADS_Q_UPPER_BOUND) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (!IsPowerOfTwoInRangeSmla(headRatio, SMLA_NUM_HEADS_Q_LOWER_BOUND, SMLA_NUM_HEADS_Q_UPPER_BOUND)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "num_heads_q / num_heads_kv",
                                              std::to_string(headRatio),
                                              "The current value is not within the valid range. "
                                              "The valid range is power of two in [" +
                                                  std::to_string(SMLA_NUM_HEADS_Q_LOWER_BOUND) + ", " +
                                                  std::to_string(SMLA_NUM_HEADS_Q_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // head_dim: 512
    if (headDim != 512) {
        OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "head_dim", std::to_string(headDim), "512");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hasOriKv) {
        // ori_topk >= 0
        if (oriTopk < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "ori_topk", std::to_string(oriTopk),
                                                  "When has_ori_kv is true, the value of ori_topk must be "
                                                  "greater than or equal to 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr) && oriTopk != 0 && hasCmpKv) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "ori_topk", std::to_string(oriTopk),
                                                  "ori_topk is reserved and "
                                                  "the value of ori_topk must be 0 when has_cmp_kv is true");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr) {
            // ori_mask_mode: 0, 3, or 4
            if (oriMaskMode != static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK) &&
                oriMaskMode != static_cast<int64_t>(SparseModeSmla::RIGHT_DOWN_CAUSAL) &&
                oriMaskMode != static_cast<int64_t>(SparseModeSmla::BAND)) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "ori_mask_mode", std::to_string(oriMaskMode),
                                                      "When has_ori_kv is true, the value of ori_mask_mode "
                                                      "must be in [0, 3, 4]");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // A5 treats -1 as unlimited window
            if (oriWinLeft < -1 || oriWinRight < -1) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    SMLA_ACLNN_OP_NAME, "ori_win_left and ori_win_right",
                    std::to_string(oriWinLeft) + " and " + std::to_string(oriWinRight),
                    "When has_ori_kv is true, the value of ori_win_left, "
                    "ori_win_right must be greater than or equal to -1");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (oriTopk != 0) {
                if (oriMaskMode != static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK)) {
                    OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "ori_mask_mode", std::to_string(oriMaskMode), "0");
                    return ACLNN_ERR_PARAM_INVALID;
                }
                if (oriWinLeft < 0 || oriWinRight < 0) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        SMLA_ACLNN_OP_NAME, "ori_win_left, ori_win_right",
                        std::to_string(oriWinLeft) + ", " + std::to_string(oriWinRight),
                        "When has_ori_kv is true and ori_topk is non-zero (DSpark), "
                        "ori_win_left and ori_win_right must be non-negative");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else {
                if (oriMaskMode != static_cast<int64_t>(SparseModeSmla::BAND)) {
                    OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "ori_mask_mode", std::to_string(oriMaskMode), "4");
                    return ACLNN_ERR_PARAM_INVALID;
                }
                if (oriWinLeft != 127 || oriWinRight != 0) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        SMLA_ACLNN_OP_NAME, "ori_win_left and ori_win_right",
                        std::to_string(oriWinLeft) + " and " + std::to_string(oriWinRight),
                        "When has_ori_kv is true, the value of ori_win_left "
                        "must be 127 and the value of ori_win_right must be 0");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }
    }
    if (hasCmpKv) {
        // cmp_topk >= 0
        if (cmpTopk < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_topk", std::to_string(cmpTopk),
                                                  "When has_cmp_kv is true, the value of cmp_topk must be "
                                                  "greater than or equal to 0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr) && cmpTopk != 0 && cmpTopk != 512 &&
            cmpTopk != 1024) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_topk", std::to_string(cmpTopk),
                                                  "When has_cmp_kv is true, the value of cmp_topk must be "
                                                  "in [0, 512, 1024]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr) {
            // cmp_mask_mode: 0 or 3
            if (cmpMaskMode != static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK) &&
                cmpMaskMode != static_cast<int64_t>(SparseModeSmla::RIGHT_DOWN_CAUSAL)) {
                OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "cmp_mask_mode", std::to_string(cmpMaskMode), "0 or 3");
                return ACLNN_ERR_PARAM_INVALID;
            }
        } else {
            if (cmpMaskMode != static_cast<int64_t>(SparseModeSmla::RIGHT_DOWN_CAUSAL)) {
                OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "cmp_mask_mode", std::to_string(cmpMaskMode), "3");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (!IsCmpRatioSupportSmla(socVersion, hasCmpKv, cmpTopk, cmpRatio)) {
            if (socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_ratio", std::to_string(cmpRatio),
                                                      "When has_cmp_kv is true, the current value is not "
                                                      "within the valid range. The valid range is [" +
                                                          std::to_string(SMLA_CMP_RATIO_LOWER_BOUND) + ", " +
                                                          std::to_string(SMLA_CMP_RATIO_UPPER_BOUND) + "]");
            } else {
                int64_t expectedCmpRatio = (cmpTopk > 0) ? 4 : 128;
                if (cmpTopk > 0) {
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_ratio", std::to_string(cmpRatio),
                                                          "When has_cmp_kv is true and cmp_topk is non-zero"
                                                          "(CSA with cmp_sparse_indices), "
                                                          "the value of cmp_ratio must be 1, 2 or " +
                                                              std::to_string(expectedCmpRatio));
                } else {
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_ratio", std::to_string(cmpRatio),
                                                          "When has_cmp_kv is true and cmp_topk is 0"
                                                          "(HCA without cmp_sparse_indices), "
                                                          "the value of cmp_ratio must be " +
                                                              std::to_string(expectedCmpRatio));
                }
            }
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (!(socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr) &&
               !IsCmpRatioSupportSmla(socVersion, hasCmpKv, cmpTopk, cmpRatio)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_ratio", std::to_string(cmpRatio),
                                              "When has_cmp_kv is false, the value of cmp_ratio must be 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutQOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "layout_q", "layout_q cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutKvOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "layout_kv", "layout_kv cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q: BSND or TND
    if (strcmp(layoutQOptional, "TND") != 0 && strcmp(layoutQOptional, "BSND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(SMLA_ACLNN_OP_NAME, "layout_q", layoutQOptional, "TND or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv: BSND, TND, or PA_BBND
    if (strcmp(layoutKvOptional, "BSND") != 0 && strcmp(layoutKvOptional, "TND") != 0 &&
        strcmp(layoutKvOptional, "PA_BBND") != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "layout_kv", layoutKvOptional,
                                              "The value of layout_kv must be in [TND, BSND, PA_BBND]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutKvOptional, "PA_BBND") != 0 && strcmp(layoutQOptional, layoutKvOptional) != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            SMLA_ACLNN_OP_NAME, "layout_q, layout_kv",
            std::string(layoutQOptional) + ", " + std::string(layoutKvOptional),
            "When layout_kv is not PA_BBND, the values of layout_q, layout_kv must be the same");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 layout_q 为 BSND 时，max_seqlen_q 必须大于 0
    if (strcmp(layoutQOptional, "BSND") == 0 && maxSeqlenQ <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                              "When layout_q is BSND, the value of max_seqlen_q "
                                              "must be equal to the size of the second axis of q");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 has_ori_kv 且 layout_kv 为 BSND 时，max_seqlen_ori_kv 必须大于 0
    if (hasOriKv && strcmp(layoutKvOptional, "BSND") == 0 && maxSeqlenOriKv <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "max_seqlen_ori_kv", std::to_string(maxSeqlenOriKv),
                                              "When has_ori_kv is true and layout_kv is BSND, "
                                              "the value of max_seqlen_ori_kv "
                                              "must be equal to the size of the second axis of ori_kv");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 has_cmp_kv 且 layout_kv 为 BSND 时，max_seqlen_cmp_kv 必须大于 0
    if (hasCmpKv && strcmp(layoutKvOptional, "BSND") == 0 && maxSeqlenCmpKv <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "max_seqlen_cmp_kv", std::to_string(maxSeqlenCmpKv),
                                              "When has_cmp_kv is true and layout_kv is BSND, "
                                              "the value of max_seqlen_cmp_kv "
                                              "must be equal to the size of the second axis of cmp_kv");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 核数校验
    if (aicCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "aic_core_num", std::to_string(aicCoreNum),
                                              "The value of aic_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aicCoreNum > optiling::AIC_CORE_MAX_NUM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "aic_core_num", std::to_string(aicCoreNum),
                                              "The current value is not within the valid range. "
                                              "The valid range is [1, " +
                                                  std::to_string(optiling::AIC_CORE_MAX_NUM) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "aiv_core_num", std::to_string(aivCoreNum),
                                              "The value of aiv_core_num must be greater than 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (aivCoreNum > optiling::AIV_CORE_MAX_NUM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(SMLA_ACLNN_OP_NAME, "aiv_core_num", std::to_string(aivCoreNum),
                                              "The current value is not within the valid range. "
                                              "The valid range is [1, " +
                                                  std::to_string(optiling::AIV_CORE_MAX_NUM) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验切g模板核数
    if (numHeadsQ == 128) {
        if (aicCoreNum == 1 || aivCoreNum == 1) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                SMLA_ACLNN_OP_NAME, "num_heads_q and aic_core_num and aiv_core_num",
                std::to_string(numHeadsQ) + " and " + std::to_string(aicCoreNum) + " and " + std::to_string(aivCoreNum),
                "When num_heads_q is 128, the value of aic_core_num, "
                "aiv_core_num cannot be 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSmla(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                               const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedOriKvOptional,
                               const aclTensor *sequsedCmpKvOptional, const aclTensor *cmpResidualKvOptional,
                               const aclTensor *oriTopkLengthOptional, const aclTensor *cmpTopkLengthOptional,
                               int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio, int64_t oriMaskMode,
                               int64_t cmpMaskMode, bool hasOriKv, bool hasCmpKv, const char *layoutQOptional,
                               const char *layoutKvOptional, const char *socVersion, const aclTensor *metadata)
{
    // cu_seqlens_q 存在性校验
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (!IsTensorExistSmla(cuSeqlensQOptional)) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "cu_seqlens_q",
                                                     "When layout_q is TND, cu_seqlens_q cannot be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasOriKv) {
        // cu_seqlens_ori_kv 存在性校验
        if (strcmp(layoutKvOptional, "TND") == 0) {
            if (!IsTensorExistSmla(cuSeqlensOriKvOptional)) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "cu_seqlens_ori_kv",
                                                         "When has_ori_kv is true and layout_kv is TND, "
                                                         "cu_seqlens_ori_kv cannot be empty");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // seqused_ori_kv 存在性校验
        if (socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr) {
            if ((oriMaskMode != 0 || oriTopk == 0) && strcmp(layoutKvOptional, "PA_BBND") == 0) {
                if (!IsTensorExistSmla(sequsedOriKvOptional)) {
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "seqused_ori_kv",
                                                             "When has_ori_kv is true, ori_mask_mode != 0 or "
                                                             "ori_topk == 0, and layout_kv is PA_BBND, "
                                                             "seqused_ori_kv cannot be empty");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        } else {
            if (strcmp(layoutKvOptional, "PA_BBND") == 0) {
                if (!IsTensorExistSmla(sequsedOriKvOptional)) {
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "seqused_ori_kv",
                                                             "When has_ori_kv is true and layout_kv is PA_BBND, "
                                                             "seqused_ori_kv cannot be empty");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }
        // ori_topk_length 存在性校验
        if (oriTopk != 0 && oriMaskMode == static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK)) {
            if (!IsTensorExistSmla(oriTopkLengthOptional)) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "ori_topk_length",
                                                         "When has_ori_kv is true, ori_topk is not 0 and "
                                                         "ori_mask_mode is 0, ori_topk_length cannot be empty");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    if (hasCmpKv) {
        // cu_seqlens_cmp_kv 存在性校验
        if (strcmp(layoutKvOptional, "TND") == 0) {
            if (!IsTensorExistSmla(cuSeqlensCmpKvOptional)) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "cu_seqlens_cmp_kv",
                                                         "When has_cmp_kv is true and layout_kv is TND, "
                                                         "cu_seqlens_cmp_kv cannot be empty");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // seqused_cmp_kv 存在性校验
        if (socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr) {
            if ((cmpMaskMode != 0 || cmpTopk == 0) && strcmp(layoutKvOptional, "PA_BBND") == 0) {
                if (!IsTensorExistSmla(sequsedCmpKvOptional)) {
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "seqused_cmp_kv",
                                                             "When has_cmp_kv is true, cmp_mask_mode != 0 or "
                                                             "cmp_topk == 0, and layout_kv is PA_BBND, "
                                                             "seqused_cmp_kv cannot be empty");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        } else {
            if (strcmp(layoutKvOptional, "PA_BBND") == 0) {
                if (!IsTensorExistSmla(sequsedCmpKvOptional)) {
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "seqused_cmp_kv",
                                                             "When has_cmp_kv is true and layout_kv is PA_BBND, "
                                                             "seqused_cmp_kv cannot be empty");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }
        // cmp_residual_kv 存在性校验
        if (cmpRatio != 1 && cmpMaskMode == static_cast<int64_t>(SparseModeSmla::RIGHT_DOWN_CAUSAL)) {
            if (!IsTensorExistSmla(cmpResidualKvOptional)) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_residual_kv",
                                                         "When has_cmp_kv is true, cmp_ratio is not 1 and "
                                                         "cmp_mask_mode is 3, cmp_residual_kv cannot be empty");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // cmp_topk_length 存在性校验
        if (cmpTopk != 0 && cmpMaskMode == static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK)) {
            if (!IsTensorExistSmla(cmpTopkLengthOptional)) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_topk_length",
                                                         "When has_cmp_kv is true, cmp_topk is not 0 and "
                                                         "cmp_mask_mode is 0, cmp_topk_length cannot be empty");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // metadata 存在性校验
    if (!IsTensorExistSmla(metadata)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(SMLA_ACLNN_OP_NAME, "metadata", "metadata cannot be empty");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

int64_t GetQueryBatchSizeSmla(const aclTensor *sequsedQOptional, const aclTensor *cuSeqlensQOptional,
                              const char *layoutQOptional, int64_t batchSize, std::string *source)
{
    if (IsTensorExistSmla(sequsedQOptional)) {
        *source = "seqused_q";
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistSmla(cuSeqlensQOptional)) {
            *source = "cu_seqlens_q";
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    *source = "batch_size";
    return batchSize;
}

int64_t GetOriKvBatchSizeSmla(const aclTensor *sequsedOriKvOptional, const aclTensor *cuSeqlensOriKvOptional,
                              const char *layoutKvOptional, int64_t batchSize, std::string *source)
{
    if (IsTensorExistSmla(sequsedOriKvOptional)) {
        *source = "seqused_ori_kv";
        return sequsedOriKvOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutKvOptional, "TND") == 0) {
        if (IsTensorExistSmla(cuSeqlensOriKvOptional)) {
            *source = "cu_seqlens_ori_kv";
            return cuSeqlensOriKvOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    *source = "batch_size";
    return batchSize;
}

int64_t GetCmpKvBatchSizeSmla(const aclTensor *sequsedCmpKvOptional, const aclTensor *cuSeqlensCmpKvOptional,
                              const char *layoutKvOptional, int64_t batchSize, std::string *source)
{
    if (IsTensorExistSmla(sequsedCmpKvOptional)) {
        *source = "seqused_cmp_kv";
        return sequsedCmpKvOptional->GetViewShape().GetDim(0);
    }
    if (strcmp(layoutKvOptional, "TND") == 0) {
        if (IsTensorExistSmla(cuSeqlensCmpKvOptional)) {
            *source = "cu_seqlens_cmp_kv";
            return cuSeqlensCmpKvOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    *source = "batch_size";
    return batchSize;
}

std::string TopkLengthShapeToStringSmla(const aclTensor *topkLengthOptional)
{
    const auto &shape = topkLengthOptional->GetViewShape();
    std::string result;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            result += ", ";
        }
        result += std::to_string(shape.GetDim(i));
    }
    return result;
}

aclnnStatus CheckTopkLengthFirstDimSmla(const aclTensor *topkLengthOptional, const std::string &topkLengthName,
                                        int64_t queryBatchSize, const std::string &querySource)
{
    if (topkLengthOptional->GetViewShape().GetDim(0) == queryBatchSize) {
        return ACLNN_SUCCESS;
    }
    std::string incorrectShape = TopkLengthShapeToStringSmla(topkLengthOptional);
    if (IsTensorSourceSmla(querySource)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(SMLA_ACLNN_OP_NAME, topkLengthName, incorrectShape,
                                              "When layout_q is BSND, the size of the first axis of " + topkLengthName +
                                                  " must be equal to " + GetSourceDescSmla(querySource));
    } else {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            SMLA_ACLNN_OP_NAME, topkLengthName, incorrectShape,
            "When layout_q is BSND, the size of the first axis of " + topkLengthName + " must be equal to batch_size");
    }
    return ACLNN_ERR_PARAM_INVALID;
}

struct TopkLengthAxisSmla {
    int64_t index;
    const char *desc;
};

inline constexpr TopkLengthAxisSmla SMLA_TOPK_LENGTH_SECOND_AXIS{1, "second"};
inline constexpr TopkLengthAxisSmla SMLA_TOPK_LENGTH_THIRD_AXIS{2, "third"};

aclnnStatus CheckTopkLengthSingleDimSmla(const aclTensor *topkLengthOptional, const std::string &topkLengthName,
                                         TopkLengthAxisSmla axis, int64_t expectedValue,
                                         const std::string &expectedDesc, const char *layoutQOptional)
{
    if (topkLengthOptional->GetViewShape().GetDim(axis.index) == expectedValue) {
        return ACLNN_SUCCESS;
    }
    std::string incorrectShape = TopkLengthShapeToStringSmla(topkLengthOptional);
    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(SMLA_ACLNN_OP_NAME, topkLengthName, incorrectShape,
                                          "When layout_q is " + std::string(layoutQOptional) + ", the size of the " +
                                              axis.desc + " axis of " + topkLengthName + " must be equal to " +
                                              expectedDesc);
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus CheckConsistencySmla(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                                 const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
                                 const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
                                 const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
                                 const aclTensor *cmpTopkLengthOptional, int64_t batchSize, const char *layoutQOptional,
                                 const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv, int64_t oriTopk,
                                 int64_t cmpTopk, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t maxSeqlenQ,
                                 int64_t numHeadsKv, const char *socVersion, const aclTensor *metadata)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    int64_t dimNum = -1;
    if (!(socVersion != nullptr && strstr(socVersion, "Ascend950") != nullptr)) {
        if (CheckReservedOptionalTensorSmla(cmpTopkLengthOptional, "cmp_topk_length") != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cu_seqlens_q
    if (IsTensorExistSmla(cuSeqlensQOptional)) {
        // 校验 cu_seqlens_q 维度
        dimNum = GetDimNumSmla(cuSeqlensQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "cu_seqlens_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 cu_seqlens_q 数据类型
        dataType = GetDataTypeSmla(cuSeqlensQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cu_seqlens_q", ToString(dataType).GetString(),
                                                  "The dtype of cu_seqlens_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_q
    if (IsTensorExistSmla(sequsedQOptional)) {
        // 校验 seqused_q 维度
        dimNum = GetDimNumSmla(sequsedQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "seqused_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 seqused_q 数据类型
        dataType = GetDataTypeSmla(sequsedQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "seqused_q", ToString(dataType).GetString(),
                                                  "The dtype of seqused_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // ori_kv部分
    if (hasOriKv) {
        // 校验 cu_seqlens_ori_kv
        if (IsTensorExistSmla(cuSeqlensOriKvOptional)) {
            // 校验 cu_seqlens_ori_kv 维度
            dimNum = GetDimNumSmla(cuSeqlensOriKvOptional);
            if (dimNum != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "cu_seqlens_ori_kv", std::to_string(dimNum), "1");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cu_seqlens_ori_kv 数据类型
            dataType = GetDataTypeSmla(cuSeqlensOriKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cu_seqlens_ori_kv",
                                                      ToString(dataType).GetString(),
                                                      "The dtype of cu_seqlens_ori_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 seqused_ori_kv
        if (IsTensorExistSmla(sequsedOriKvOptional)) {
            // 校验 seqused_ori_kv 维度
            dimNum = GetDimNumSmla(sequsedOriKvOptional);
            if (dimNum != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "seqused_ori_kv", std::to_string(dimNum), "1");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 seqused_ori_kv 数据类型
            dataType = GetDataTypeSmla(sequsedOriKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "seqused_ori_kv",
                                                      ToString(dataType).GetString(),
                                                      "The dtype of seqused_ori_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 ori_topk_length
        if (oriTopk != 0 && oriMaskMode == static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK) &&
            IsTensorExistSmla(oriTopkLengthOptional)) {
            // 校验 ori_topk_length 维度
            dimNum = GetDimNumSmla(oriTopkLengthOptional);
            if (strcmp(layoutQOptional, "TND") == 0) {
                if (dimNum != 2) {
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(SMLA_ACLNN_OP_NAME, "ori_topk_length",
                                                             std::to_string(dimNum),
                                                             "The shape dim of ori_topk_length must be 2 "
                                                             "when layout_q is TND");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else if (strcmp(layoutQOptional, "BSND") == 0) {
                if (dimNum != 3) {
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(SMLA_ACLNN_OP_NAME, "ori_topk_length",
                                                             std::to_string(dimNum),
                                                             "The shape dim of ori_topk_length must be 3 "
                                                             "when layout_q is BSND");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
            // 校验 ori_topk_length 数据类型
            dataType = GetDataTypeSmla(oriTopkLengthOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "ori_topk_length",
                                                      ToString(dataType).GetString(),
                                                      "The dtype of ori_topk_length must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // cmp_kv部分
    if (hasCmpKv) {
        // 校验 cu_seqlens_cmp_kv
        if (IsTensorExistSmla(cuSeqlensCmpKvOptional)) {
            // 校验 cu_seqlens_cmp_kv 维度
            dimNum = GetDimNumSmla(cuSeqlensCmpKvOptional);
            if (dimNum != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "cu_seqlens_cmp_kv", std::to_string(dimNum), "1");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cu_seqlens_cmp_kv 数据类型
            dataType = GetDataTypeSmla(cuSeqlensCmpKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cu_seqlens_cmp_kv",
                                                      ToString(dataType).GetString(),
                                                      "The dtype of cu_seqlens_cmp_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 seqused_cmp_kv
        if (IsTensorExistSmla(sequsedCmpKvOptional)) {
            // 校验 seqused_cmp_kv 维度
            dimNum = GetDimNumSmla(sequsedCmpKvOptional);
            if (dimNum != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "seqused_cmp_kv", std::to_string(dimNum), "1");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 seqused_cmp_kv 数据类型
            dataType = GetDataTypeSmla(sequsedCmpKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "seqused_cmp_kv",
                                                      ToString(dataType).GetString(),
                                                      "The dtype of seqused_cmp_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_residual_kv
        if (IsTensorExistSmla(cmpResidualKvOptional)) {
            // 校验 cmp_residual_kv 维度
            dimNum = GetDimNumSmla(cmpResidualKvOptional);
            if (dimNum != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "cmp_residual_kv", std::to_string(dimNum), "1");
                return ACLNN_ERR_PARAM_INVALID;
            }
            // 校验 cmp_residual_kv 数据类型
            dataType = GetDataTypeSmla(cmpResidualKvOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_residual_kv",
                                                      ToString(dataType).GetString(),
                                                      "The dtype of cmp_residual_kv must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_topk_length
        if (cmpTopk != 0 && cmpMaskMode == static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK) &&
            IsTensorExistSmla(cmpTopkLengthOptional)) {
            // 校验 cmp_topk_length 维度
            dimNum = GetDimNumSmla(cmpTopkLengthOptional);
            if (strcmp(layoutQOptional, "TND") == 0) {
                if (dimNum != 2) {
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_topk_length",
                                                             std::to_string(dimNum),
                                                             "The shape dim of cmp_topk_length must be 2 "
                                                             "when layout_q is TND");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            } else if (strcmp(layoutQOptional, "BSND") == 0) {
                if (dimNum != 3) {
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_topk_length",
                                                             std::to_string(dimNum),
                                                             "The shape dim of cmp_topk_length must be 3 "
                                                             "when layout_q is BSND");
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
            // 校验 cmp_topk_length 数据类型
            dataType = GetDataTypeSmla(cmpTopkLengthOptional);
            if (dataType != aclDataType::ACL_INT32) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "cmp_topk_length",
                                                      ToString(dataType).GetString(),
                                                      "The dtype of cmp_topk_length must be int32");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    // 校验 metadata
    if (IsTensorExistSmla(metadata)) {
        // 校验 metadata 维度
        dimNum = GetDimNumSmla(metadata);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(SMLA_ACLNN_OP_NAME, "metadata", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素数
        if (metadata->GetViewShape().GetDim(0) != optiling::SMLA_METADATA_TOTAL_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPESIZE(SMLA_ACLNN_OP_NAME, "metadata",
                                          std::to_string(metadata->GetViewShape().GetDim(0)),
                                          std::to_string(optiling::SMLA_METADATA_TOTAL_SIZE));
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 数据类型
        dataType = GetDataTypeSmla(metadata);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(SMLA_ACLNN_OP_NAME, "metadata", ToString(dataType).GetString(),
                                                  "The dtype of metadata must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 q/kv 维度一致性
    std::string querySource;
    int64_t queryBatchSize =
        GetQueryBatchSizeSmla(sequsedQOptional, cuSeqlensQOptional, layoutQOptional, batchSize, &querySource);
    // 校验TND场景q维度一致性
    if (strcmp(layoutQOptional, "TND") == 0 && IsTensorExistSmla(sequsedQOptional)) {
        int64_t cuSeqlensQBatchSize = cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        if (cuSeqlensQBatchSize != queryBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(SMLA_ACLNN_OP_NAME, "cu_seqlens_q and seqused_q",
                                                       std::to_string(cuSeqlensQOptional->GetViewShape().GetDim(0)) +
                                                           " and " +
                                                           std::to_string(sequsedQOptional->GetViewShape().GetDim(0)),
                                                       "When layout_q is TND and seqused_q is passed, "
                                                       "the shape size of cu_seqlens_q minus 1 must be equal to "
                                                       "the shape size of seqused_q");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (hasOriKv) {
        std::string oriKvSource;
        int64_t oriKvBatchSize = GetOriKvBatchSizeSmla(sequsedOriKvOptional, cuSeqlensOriKvOptional, layoutKvOptional,
                                                       batchSize, &oriKvSource);
        // 校验q与ori_kv维度一致性
        if (queryBatchSize != oriKvBatchSize) {
            if (IsTensorSourceSmla(querySource) && IsTensorSourceSmla(oriKvSource)) {
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    SMLA_ACLNN_OP_NAME, querySource + " and " + oriKvSource,
                    std::to_string(GetRawShapeSizeSmla(querySource, queryBatchSize)) + " and " +
                        std::to_string(GetRawShapeSizeSmla(oriKvSource, oriKvBatchSize)),
                    "When has_ori_kv is true, " + GetSourceDescSmla(querySource) + " must be equal to " +
                        GetSourceDescSmla(oriKvSource));
            } else if (IsTensorSourceSmla(querySource)) {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    SMLA_ACLNN_OP_NAME, querySource, std::to_string(GetRawShapeSizeSmla(querySource, queryBatchSize)),
                    "When has_ori_kv is true, " + GetSourceDescSmla(querySource) + " must be equal to batch_size");
            } else {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    SMLA_ACLNN_OP_NAME, oriKvSource, std::to_string(GetRawShapeSizeSmla(oriKvSource, oriKvBatchSize)),
                    "When has_ori_kv is true, " + GetSourceDescSmla(oriKvSource) + " must be equal to batch_size");
            }
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验TND场景ori_kv维度一致性
        if (strcmp(layoutKvOptional, "TND") == 0 && IsTensorExistSmla(sequsedOriKvOptional)) {
            int64_t cuSeqlensOriKvBatchSize = cuSeqlensOriKvOptional->GetViewShape().GetDim(0) - 1;
            if (cuSeqlensOriKvBatchSize != oriKvBatchSize) {
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    SMLA_ACLNN_OP_NAME, "cu_seqlens_ori_kv and seqused_ori_kv",
                    std::to_string(cuSeqlensOriKvOptional->GetViewShape().GetDim(0)) + " and " +
                        std::to_string(sequsedOriKvOptional->GetViewShape().GetDim(0)),
                    "When layout_kv is TND and seqused_ori_kv is passed, "
                    "the shape size of cu_seqlens_ori_kv minus 1 must be "
                    "equal to the shape size of seqused_ori_kv");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 ori_topk_length 维度一致性
        if (oriTopk != 0 && oriMaskMode == static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK) &&
            IsTensorExistSmla(oriTopkLengthOptional)) {
            if (strcmp(layoutQOptional, "BSND") == 0) {
                // 校验 ori_topk_length 第一个维度
                aclnnStatus ret =
                    CheckTopkLengthFirstDimSmla(oriTopkLengthOptional, "ori_topk_length", queryBatchSize, querySource);
                if (ret != ACLNN_SUCCESS) {
                    return ret;
                }
                // 校验 ori_topk_length 第二个维度
                ret =
                    CheckTopkLengthSingleDimSmla(oriTopkLengthOptional, "ori_topk_length", SMLA_TOPK_LENGTH_SECOND_AXIS,
                                                 maxSeqlenQ, "max_seqlen_q", layoutQOptional);
                if (ret != ACLNN_SUCCESS) {
                    return ret;
                }
                // 校验 ori_topk_length 第三个维度
                ret =
                    CheckTopkLengthSingleDimSmla(oriTopkLengthOptional, "ori_topk_length", SMLA_TOPK_LENGTH_THIRD_AXIS,
                                                 numHeadsKv, "num_heads_kv", layoutQOptional);
                if (ret != ACLNN_SUCCESS) {
                    return ret;
                }
            } else if (strcmp(layoutQOptional, "TND") == 0) {
                // 校验 ori_topk_length 第二个维度
                aclnnStatus ret =
                    CheckTopkLengthSingleDimSmla(oriTopkLengthOptional, "ori_topk_length", SMLA_TOPK_LENGTH_SECOND_AXIS,
                                                 numHeadsKv, "num_heads_kv", layoutQOptional);
                if (ret != ACLNN_SUCCESS) {
                    return ret;
                }
            }
        }
    }
    if (hasCmpKv) {
        std::string cmpKvSource;
        int64_t cmpKvBatchSize = GetCmpKvBatchSizeSmla(sequsedCmpKvOptional, cuSeqlensCmpKvOptional, layoutKvOptional,
                                                       batchSize, &cmpKvSource);
        // 校验q与cmp_kv维度一致性
        if (queryBatchSize != cmpKvBatchSize) {
            if (IsTensorSourceSmla(querySource) && IsTensorSourceSmla(cmpKvSource)) {
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    SMLA_ACLNN_OP_NAME, querySource + " and " + cmpKvSource,
                    std::to_string(GetRawShapeSizeSmla(querySource, queryBatchSize)) + " and " +
                        std::to_string(GetRawShapeSizeSmla(cmpKvSource, cmpKvBatchSize)),
                    "When has_cmp_kv is true, " + GetSourceDescSmla(querySource) + " must be equal to " +
                        GetSourceDescSmla(cmpKvSource));
            } else if (IsTensorSourceSmla(querySource)) {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    SMLA_ACLNN_OP_NAME, querySource, std::to_string(GetRawShapeSizeSmla(querySource, queryBatchSize)),
                    "When has_cmp_kv is true, " + GetSourceDescSmla(querySource) + " must be equal to batch_size");
            } else {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    SMLA_ACLNN_OP_NAME, cmpKvSource, std::to_string(GetRawShapeSizeSmla(cmpKvSource, cmpKvBatchSize)),
                    "When has_cmp_kv is true, " + GetSourceDescSmla(cmpKvSource) + " must be equal to batch_size");
            }
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验TND场景cmp_kv维度一致性
        if (strcmp(layoutKvOptional, "TND") == 0 && IsTensorExistSmla(sequsedCmpKvOptional)) {
            int64_t cuSeqlensCmpKvBatchSize = cuSeqlensCmpKvOptional->GetViewShape().GetDim(0) - 1;
            if (cuSeqlensCmpKvBatchSize != cmpKvBatchSize) {
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    SMLA_ACLNN_OP_NAME, "cu_seqlens_cmp_kv and seqused_cmp_kv",
                    std::to_string(cuSeqlensCmpKvOptional->GetViewShape().GetDim(0)) + " and " +
                        std::to_string(sequsedCmpKvOptional->GetViewShape().GetDim(0)),
                    "When layout_kv is TND and seqused_cmp_kv is passed, "
                    "the shape size of cu_seqlens_cmp_kv minus 1 must be "
                    "equal to the shape size of seqused_cmp_kv");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_residual_kv 元素数
        if (IsTensorExistSmla(cmpResidualKvOptional)) {
            if (cmpResidualKvOptional->GetViewShape().GetDim(0) != queryBatchSize) {
                if (IsTensorSourceSmla(querySource)) {
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        SMLA_ACLNN_OP_NAME, "cmp_residual_kv and " + querySource,
                        std::to_string(cmpResidualKvOptional->GetViewShape().GetDim(0)) + " and " +
                            std::to_string(GetRawShapeSizeSmla(querySource, queryBatchSize)),
                        "The shape size of cmp_residual_kv must be equal to " + GetSourceDescSmla(querySource));
                } else {
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        SMLA_ACLNN_OP_NAME, "cmp_residual_kv",
                        std::to_string(cmpResidualKvOptional->GetViewShape().GetDim(0)),
                        "The shape size of cmp_residual_kv must be equal "
                        "to batch_size");
                }
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        // 校验 cmp_topk_length 维度一致性
        if (cmpTopk != 0 && cmpMaskMode == static_cast<int64_t>(SparseModeSmla::DEFAULT_MASK) &&
            IsTensorExistSmla(cmpTopkLengthOptional)) {
            if (strcmp(layoutQOptional, "BSND") == 0) {
                // 校验 cmp_topk_length 第一个维度
                aclnnStatus ret =
                    CheckTopkLengthFirstDimSmla(cmpTopkLengthOptional, "cmp_topk_length", queryBatchSize, querySource);
                if (ret != ACLNN_SUCCESS) {
                    return ret;
                }
                // 校验 cmp_topk_length 第二个维度
                ret =
                    CheckTopkLengthSingleDimSmla(cmpTopkLengthOptional, "cmp_topk_length", SMLA_TOPK_LENGTH_SECOND_AXIS,
                                                 maxSeqlenQ, "max_seqlen_q", layoutQOptional);
                if (ret != ACLNN_SUCCESS) {
                    return ret;
                }
                // 校验 cmp_topk_length 第三个维度
                ret =
                    CheckTopkLengthSingleDimSmla(cmpTopkLengthOptional, "cmp_topk_length", SMLA_TOPK_LENGTH_THIRD_AXIS,
                                                 numHeadsKv, "num_heads_kv", layoutQOptional);
                if (ret != ACLNN_SUCCESS) {
                    return ret;
                }
            } else if (strcmp(layoutQOptional, "TND") == 0) {
                // 校验 cmp_topk_length 第二个维度
                aclnnStatus ret =
                    CheckTopkLengthSingleDimSmla(cmpTopkLengthOptional, "cmp_topk_length", SMLA_TOPK_LENGTH_SECOND_AXIS,
                                                 numHeadsKv, "num_heads_kv", layoutQOptional);
                if (ret != ACLNN_SUCCESS) {
                    return ret;
                }
            }
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsCheck(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
                               const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
                               const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
                               const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
                               const aclTensor *cmpTopkLengthOptional, int64_t numHeadsQ, int64_t numHeadsKv,
                               int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenOriKv,
                               int64_t maxSeqlenCmpKv, int64_t oriTopk, int64_t cmpTopk, int64_t cmpRatio,
                               int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight,
                               const char *layoutQOptional, const char *layoutKvOptional, bool hasOriKv, bool hasCmpKv,
                               uint32_t aicCoreNum, uint32_t aivCoreNum, const char *socVersion,
                               const aclTensor *metaData)
{
    if (CheckSingleParamSmla(batchSize, maxSeqlenQ, maxSeqlenOriKv, maxSeqlenCmpKv, numHeadsQ, numHeadsKv, headDim,
                             oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                             layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv, aicCoreNum, aivCoreNum,
                             socVersion) == ACLNN_SUCCESS &&
        CheckExistenceSmla(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedOriKvOptional,
                           sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional,
                           oriTopk, cmpTopk, cmpRatio, oriMaskMode, cmpMaskMode, hasOriKv, hasCmpKv, layoutQOptional,
                           layoutKvOptional, socVersion, metaData) == ACLNN_SUCCESS &&
        CheckConsistencySmla(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
                             sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
                             cmpTopkLengthOptional, batchSize, layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv,
                             oriTopk, cmpTopk, oriMaskMode, cmpMaskMode, maxSeqlenQ, numHeadsKv, socVersion,
                             metaData) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}
} // namespace

#ifdef __cplusplus
}
#endif
