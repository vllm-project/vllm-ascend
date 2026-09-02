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
 * \file aclnn_ffn.cpp
 * \brief
 */

#include "aclnn_ffn.h"
#include "../ffn_layout.h"
#include "aclnn_ffn_v2.h"
#include "aclnn_ffn_v3.h"

#include <dlfcn.h>
#include <new>

#include "ffn.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t DIM_LIMIT_UPPER = 8;
static const int64_t DIM_LIMIT_LOWER = 2;
static const int64_t HIGH_PERFORMANCE = 1;
static const int64_t HIGH_PRECISION = 0;
static const int64_t DIM_NUM_ONE = 1;
static const int64_t DIM_NUM_TWO = 2;
static const int64_t DIM_NUM_THREE = 3;
static const int64_t JUDGE_EVEN = 2;
static const int64_t MAX_K = 65535;
using DtypeCheck = std::initializer_list<op::DataType>;

enum class ActiveType {
    FASTGELU = 0,
    RELU,
    SILU,
    GELU,
    GEGLU,
    SWIGLU,
    REGLU,
    INVALID_TYPE
};

class ActiveMap {
public:
    const char *activeName;
    ActiveType activeType;
};

constexpr class ActiveMap g_activeMap[] = {
    {"fastgelu", ActiveType::FASTGELU}, {"relu", ActiveType::RELU},   {"silu", ActiveType::SILU},
    {"gelu", ActiveType::GELU},         {"geglu", ActiveType::GEGLU}, {"swiglu", ActiveType::SWIGLU},
    {"reglu", ActiveType::REGLU},
};

struct FFNParams {
    const aclTensor *x = nullptr;
    const aclTensor *weight1 = nullptr;
    const aclTensor *weight2 = nullptr;
    const aclIntArray *expertTokensArr = nullptr;
    const aclTensor *expertTokens = nullptr;
    const aclTensor *bias1 = nullptr;
    const aclTensor *bias2 = nullptr;
    const aclTensor *scale = nullptr;
    const aclTensor *offset = nullptr;
    const aclTensor *deqScale1 = nullptr;
    const aclTensor *deqScale2 = nullptr;
    const aclTensor *antiquantScale1 = nullptr;
    const aclTensor *antiquantScale2 = nullptr;
    const aclTensor *antiquantOffset1 = nullptr;
    const aclTensor *antiquantOffset2 = nullptr;
    ActiveType activationType;
    int64_t innerPrecise;
    bool tokensIndexFlag;
    const aclTensor *y = nullptr;
};

static ActiveType GetActiveType(const char *activeType)
{
    for (const ActiveMap &item : g_activeMap) {
        size_t len = strlen(item.activeName);
        bool isValidActiveType = strlen(activeType) == len;
        // use for loop instead of strncasecmp to avoid possible out-of-bounds problems
        if (!isValidActiveType) {
            continue;
        }
        for (size_t i = 0; i < len; i++) {
            if (tolower(static_cast<unsigned char>(activeType[i])) != item.activeName[i]) {
                isValidActiveType = false;
                break;
            }
        }
        if (isValidActiveType) {
            OP_LOGD("aclnnFFN activeType is %s.", activeType);
            return item.activeType;
        }
    }
    return ActiveType::INVALID_TYPE;
}

static aclnnStatus CheckNotNull(const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2,
                                const aclTensor *y)
{
    CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(weight1 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "weight1 must not be nullptr.");
    CHECK_COND(weight2 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "weight2 must not be nullptr.");
    CHECK_COND(y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "y must not be nullptr.");
    return ACLNN_SUCCESS;
}

static inline bool CheckY(const FFNParams &ffnParams)
{
    auto xDimNum = ffnParams.x->GetViewShape().GetDimNum();
    auto yDimNum = ffnParams.y->GetViewShape().GetDimNum();
    // y's dimNum should be equal to x's dimNum
    if (yDimNum != xDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y's dimNum should be equal to x's dimNum, x dimNum is %zu, y dimNum is %zu",
                xDimNum, yDimNum);
        return false;
    }

    return true;
}

static bool CheckDtypeValidForFFNOp(const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2)
{
    CHECK_COND(x->GetDataType() == DataType::DT_FLOAT16, false,
               "x only support dtype float16 without expert tokens, but found %s.",
               op::ToString(x->GetDataType()).GetString());
    CHECK_COND(weight1->GetDataType() == DataType::DT_FLOAT16, false,
               "weight1 only support dtype float16 without expert tokens, but found %s.",
               op::ToString(weight1->GetDataType()).GetString());
    CHECK_COND(weight2->GetDataType() == DataType::DT_FLOAT16, false,
               "weight2 only support dtype float16 without expert tokens, but found %s.",
               op::ToString(weight2->GetDataType()).GetString());
    return true;
}

static bool IsArch35Fused(const FFNParams &ffnParams)
{
    return GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950 &&
           ffnParams.expertTokens == nullptr &&
           (ffnParams.x->GetDataType() == DataType::DT_BF16 ||
            ffnParams.x->GetDataType() == DataType::DT_FLOAT16) &&
           (ffnParams.activationType == ActiveType::GELU || ffnParams.activationType == ActiveType::SILU ||
            ffnParams.activationType == ActiveType::SWIGLU);
}

// arch35 fused 布局识别（公共规则 ../ffn_layout.h；维度取最后两维）
static ffnlayout::FfnLayout Arch35DetectLayout(const FFNParams &ffnParams)
{
    const auto &w1View = ffnParams.weight1->GetViewShape();
    const auto &w2View = ffnParams.weight2->GetViewShape();
    const auto &xView = ffnParams.x->GetViewShape();
    return ffnlayout::FfnDetectLayout(w1View.GetDim(w1View.GetDimNum() - 2),
                                      w1View.GetDim(w1View.GetDimNum() - 1),
                                      w2View.GetDim(w2View.GetDimNum() - 2),
                                      w2View.GetDim(w2View.GetDimNum() - 1),
                                      xView.GetDim(xView.GetDimNum() - 1),
                                      ffnParams.activationType == ActiveType::SWIGLU);
}

// arch35 fused：与权重校验同一公共规则（Arch35DetectLayout，含方阵 w2 消歧）；
// 非 arch35 场景一律 canonical 语义。
static bool IsLinearWeight(const FFNParams &ffnParams)
{
    if (!IsArch35Fused(ffnParams)) {
        return false;
    }
    return Arch35DetectLayout(ffnParams) == ffnlayout::FfnLayout::LINEAR;
}

static bool CheckBiasDimNum(const aclTensor *weight, const aclTensor *expertTokens, const aclTensor *bias,
                            const char *biasName, bool transposedWeight)
{
    bool hasExperts = (expertTokens != nullptr);
    size_t weightSize = hasExperts ? DIM_NUM_THREE : DIM_NUM_TWO;
    // bias dim size should be 2 when having experTokens, and be 1 when not having it
    size_t biasSize = hasExperts ? DIM_NUM_TWO : DIM_NUM_ONE;
    auto biasDimNum = bias->GetViewShape().GetDimNum();
    if (biasDimNum != biasSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "%s DimNum dim should be (2: has experts, 1: no experts), %s DimNum dim is %zu", biasName, biasName,
                biasDimNum);
        return false;
    }
    size_t weightNDimIdx = weightSize - 1;
    if (transposedWeight) {
        // arch35 fused gelu：weight 为 [N, K]（out, in），bias 长度对应 weight 行数
        weightNDimIdx = 0;
    }
    int64_t weightNDimValue = weight->GetViewShape().GetDim(weightNDimIdx);
    size_t biasNDimIdx = biasSize - 1;
    int64_t biasNDimValue = bias->GetViewShape().GetDim(biasNDimIdx);
    if (biasNDimValue != weightNDimValue) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s last dim should be %ld, but %s last dim is %ld", biasName, weightNDimValue,
                biasName, biasNDimValue);
        return false;
    }
    if (hasExperts) {
        int64_t expertTokensLength = expertTokens->GetViewShape().GetDim(0);
        int64_t biasEDimValue = bias->GetViewShape().GetDim(0);
        if (biasEDimValue != expertTokensLength) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s first dim should be %ld, but %s first dim is %ld", biasName,
                    expertTokensLength, biasName, biasEDimValue);
            return false;
        }
    }
    return true;
}

static bool CheckBias(const FFNParams &ffnParams)
{
    bool bias1CheckResult = true;
    bool bias2CheckResult = true;
    bool transposedWeight = IsLinearWeight(ffnParams);
    if (ffnParams.bias1 != nullptr) {
        bias1CheckResult = CheckBiasDimNum(ffnParams.weight1, ffnParams.expertTokens, ffnParams.bias1, "bias1",
                                           transposedWeight);
    }
    if (ffnParams.bias2 != nullptr) {
        bias2CheckResult = CheckBiasDimNum(ffnParams.weight2, ffnParams.expertTokens, ffnParams.bias2, "bias2",
                                           transposedWeight);
    }
    return bias1CheckResult && bias2CheckResult;
}

static bool GluShapeCheck(size_t weight1NDimValue, size_t weight2kDimValue)
{
    // remind by 2 to check if it is a odd number
    if (weight1NDimValue % 2 != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "weight1 NDimValue should be even number in glu case, but your weight1 "
                "NDimVakue is %zu",
                weight1NDimValue);
        return false;
    }
    // weight2 KDimValue should be equal to half of weight1 NDimValue in glu case
    if (weight2kDimValue != weight1NDimValue / 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "weight2 KDimValue should be equal to half of weight1 NDimValue in glu case, "
                "but your weight2 kDimValue is %zu and your weight1 NDimValue is  %zu",
                weight2kDimValue, weight1NDimValue);
        return false;
    }
    return true;
}

static aclnnStatus CheckFmapWeightShape(const FFNParams &ffnParams)
{
    auto xDimNum = ffnParams.x->GetViewShape().GetDimNum();
    // x's dim should within 2 ~ 8
    CHECK_COND(xDimNum >= DIM_LIMIT_LOWER && xDimNum <= DIM_LIMIT_UPPER, ACLNN_ERR_PARAM_INVALID,
               "x dim should within 2 ~ 8, but x dim is %zu", xDimNum);
    auto xKdimIndex = xDimNum - 1; // 1 represents the last dimension of x
    int64_t xKDimValue = ffnParams.x->GetViewShape().GetDim(xKdimIndex);

    bool hasExperts = (ffnParams.expertTokens != nullptr);
    // weight dim size should be 3 when having experTokens, and be 2 when not having it
    size_t weightSize = hasExperts ? DIM_NUM_THREE : DIM_NUM_TWO;
    size_t weight1DimNum = ffnParams.weight1->GetViewShape().GetDimNum();
    size_t weight2DimNum = ffnParams.weight2->GetViewShape().GetDimNum();
    CHECK_COND(weight1DimNum == weightSize, ACLNN_ERR_PARAM_INVALID,
               "weight1 DimNum dim should be (3: has experts, 2: no experts), but got %zu", weight1DimNum);
    CHECK_COND(weight2DimNum == weightSize, ACLNN_ERR_PARAM_INVALID,
               "weight2 DimNum dim should be (3: has experts, 2: no experts), but got %zu", weight2DimNum);
    size_t kDimIdx = weightSize - 2; // 2 represents the penultimate dimension of weight
    size_t nDimIdx = weightSize - 1; // 1 represents the last dimension of weight
    int64_t weight1kDimValue = ffnParams.weight1->GetViewShape().GetDim(kDimIdx);
    int64_t weight1NDimValue = ffnParams.weight1->GetViewShape().GetDim(nDimIdx);
    int64_t weight2kDimValue = ffnParams.weight2->GetViewShape().GetDim(kDimIdx);
    int64_t weight2NDimValue = ffnParams.weight2->GetViewShape().GetDim(nDimIdx);
    CHECK_COND(weight1kDimValue <= MAX_K && weight2kDimValue <= MAX_K, ACLNN_ERR_PARAM_INVALID,
               "kDimValue of weight1 "
               "%ld and weight2 %ld should both be equal to or less than 65535",
               weight1kDimValue, weight2kDimValue);

    // arch35 fused gelu/silu/swiglu：支持 canonical [K, N] 与 PyTorch Linear 布局 [N, K]（out, in）
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950 && !hasExperts &&
        (ffnParams.x->GetDataType() == DataType::DT_BF16 ||
         ffnParams.x->GetDataType() == DataType::DT_FLOAT16) &&
        (ffnParams.activationType == ActiveType::GELU || ffnParams.activationType == ActiveType::SILU ||
         ffnParams.activationType == ActiveType::SWIGLU)) {
        const auto &w1ViewShape = ffnParams.weight1->GetViewShape();
        const auto &w2ViewShape = ffnParams.weight2->GetViewShape();
        const int64_t w1Dim0 = w1ViewShape.GetDim(w1ViewShape.GetDimNum() - 2);
        const int64_t w1Dim1 = w1ViewShape.GetDim(w1ViewShape.GetDimNum() - 1);
        const int64_t w2Dim0 = w2ViewShape.GetDim(w2ViewShape.GetDimNum() - 2);
        const int64_t w2Dim1 = w2ViewShape.GetDim(w2ViewShape.GetDimNum() - 1);
        // 布局识别走公共规则（Arch35DetectLayout → ../ffn_layout.h）。
        const bool isSwiglu = (ffnParams.activationType == ActiveType::SWIGLU); // 下方 swiglu 分支校验使用
        const ffnlayout::FfnLayout layout = Arch35DetectLayout(ffnParams);
        if (layout == ffnlayout::FfnLayout::INVALID) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "weight1 shape [%ld, %ld] does not match x K=%ld (expect [K,N] or [N,K])", w1Dim0, w1Dim1,
                    xKDimValue);
            return ACLNN_ERR_PARAM_INVALID;
        }
        const bool layoutLinear = (layout == ffnlayout::FfnLayout::LINEAR);
        if (layoutLinear) {
            if (isSwiglu) {
                // linear swiglu：w1 [2H, K]，w2 [N, H]
                CHECK_COND(w1Dim0 % 2 == 0, ACLNN_ERR_PARAM_INVALID, "swiglu weight1 rows must be even");
                CHECK_COND(w1Dim0 / 2 == w2Dim1, ACLNN_ERR_PARAM_INVALID,
                           "swiglu weight1 rows/2[%ld] is not equal to weight2 last dim[%ld]", w1Dim0 / 2, w2Dim1);
                CHECK_COND(w2Dim0 > 0 && w1Dim0 > 0, ACLNN_ERR_PARAM_INVALID, "weight dims should be positive");
            } else {
                // linear：w1 [N1, K]，w2 [N2, N1]
                CHECK_COND(w1Dim0 == w2Dim1, ACLNN_ERR_PARAM_INVALID,
                           "weight1 rows[%ld] is not equal to weight2 last dim[%ld]", w1Dim0, w2Dim1);
                CHECK_COND(w2Dim0 > 0 && w1Dim0 > 0, ACLNN_ERR_PARAM_INVALID, "weight dims should be positive");
            }
        } else {
            if (isSwiglu) {
                // canonical swiglu：w1 [K, 2H]，w2 [H, N]
                CHECK_COND(w1Dim1 % 2 == 0, ACLNN_ERR_PARAM_INVALID, "swiglu weight1 last dim must be even");
                CHECK_COND(w1Dim1 / 2 == w2Dim0, ACLNN_ERR_PARAM_INVALID,
                           "swiglu weight1 last dim/2[%ld] is not equal to weight2 rows[%ld]", w1Dim1 / 2, w2Dim0);
                CHECK_COND(w2Dim1 > 0 && w1Dim1 > 0, ACLNN_ERR_PARAM_INVALID, "weight dims should be positive");
            } else {
                // canonical：w1 [K, N1]，w2 [N1, N2]
                CHECK_COND(w1Dim1 == w2Dim0, ACLNN_ERR_PARAM_INVALID,
                           "weight1 last dim[%ld] is not equal to weight2 rows[%ld]", w1Dim1, w2Dim0);
                CHECK_COND(w2Dim1 > 0 && w1Dim1 > 0, ACLNN_ERR_PARAM_INVALID, "weight dims should be positive");
            }
        }
        return ACLNN_SUCCESS;
    }

    CHECK_COND(xKDimValue == weight1kDimValue, ACLNN_ERR_PARAM_INVALID,
               "x KDimValue[%ld] is not equal to weight1 kDimValue[%ld]", xKDimValue, weight1kDimValue);

    bool isGlu = (static_cast<uint32_t>(ffnParams.activationType) >= static_cast<uint32_t>(ActiveType::GEGLU));
    if (isGlu) {
        CHECK_COND(GluShapeCheck(weight1NDimValue, weight2kDimValue), ACLNN_ERR_PARAM_INVALID, "GluShapeCheck failed.");
        return ACLNN_SUCCESS;
    }
    // check n1=k2, n2=k1
    CHECK_COND(weight1kDimValue == weight2NDimValue, ACLNN_ERR_PARAM_INVALID,
               "weight1 KDimValue[%ld] is not equal to weight2 NDimValue[%ld]", weight1kDimValue, weight2NDimValue);
    CHECK_COND(weight1NDimValue == weight2kDimValue, ACLNN_ERR_PARAM_INVALID,
               "weight1 NDimValue[%ld] is not equal to weight2 KDimValue[%ld]", weight1NDimValue, weight2kDimValue);
    if (hasExperts) {
        int64_t expertsNum = ffnParams.expertTokens->GetViewShape().GetDim(0);
        CHECK_COND(ffnParams.weight1->GetViewShape().GetDim(0) == expertsNum, ACLNN_ERR_PARAM_INVALID,
                   "weight1 length should be %ld, but got %ld", expertsNum,
                   ffnParams.weight1->GetViewShape().GetDim(0));
        CHECK_COND(ffnParams.weight2->GetViewShape().GetDim(0) == expertsNum, ACLNN_ERR_PARAM_INVALID,
                   "weight2 length should be %ld, but got %ld", expertsNum,
                   ffnParams.weight2->GetViewShape().GetDim(0));
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus IsAntiQuantEmpty(const FFNParams &ffnParams)
{
    CHECK_RET(ffnParams.antiquantScale1 == nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ffnParams.antiquantScale2 == nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ffnParams.antiquantOffset1 == nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ffnParams.antiquantOffset2 == nullptr, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus IsQuantEmpty(const FFNParams &ffnParams)
{
    CHECK_RET(ffnParams.scale == nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ffnParams.offset == nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ffnParams.deqScale1 == nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ffnParams.deqScale2 == nullptr, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckBiasDtype(const FFNParams &ffnParams, const DtypeCheck &biasSupportList)
{
    if (ffnParams.bias1 != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(ffnParams.bias1, biasSupportList, return ACLNN_ERR_PARAM_INVALID);
    }
    if (ffnParams.bias2 != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(ffnParams.bias2, biasSupportList, return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckQuantScaleAndOffset(const FFNParams &ffnParams, bool hasExperts, size_t weightNDimIdx)
{
    // 校验scale是否smooth 和 有无专家时scale和offset参数的维度数量和维度大小是否正确
    auto scaleDimNum = ffnParams.scale->GetViewShape().GetDimNum();
    CHECK_COND(scaleDimNum >= DIM_NUM_ONE, ACLNN_ERR_PARAM_INVALID,
               "scale dim num should be at least 1 in quant scenario, but currently is %zu", scaleDimNum);
    int64_t weight1N1 = ffnParams.weight1->GetViewShape().GetDim(weightNDimIdx);
    int64_t scaleLengthDim0 = ffnParams.scale->GetViewShape().GetDim(0);
    DataType deqScaleDtype = ffnParams.deqScale1->GetDataType();
    int64_t requiredLength = 1;

    if (hasExperts) {
        requiredLength = ffnParams.expertTokens->GetViewShape().GetDim(0);
        CHECK_COND(scaleDimNum == DIM_NUM_TWO || scaleDimNum == DIM_NUM_ONE, ACLNN_ERR_PARAM_INVALID,
                   "scale dim num should be 1 or 2 in quant scenario, but currently is %zu", scaleDimNum);
        CHECK_COND(scaleLengthDim0 == requiredLength, ACLNN_ERR_PARAM_INVALID,
                   "scale length should be %ld, but scale length is %ld", requiredLength, scaleLengthDim0);
        if (scaleDimNum == DIM_NUM_TWO) {
            int64_t scaleLengthDim1 = ffnParams.scale->GetViewShape().GetDim(1);
            CHECK_COND(scaleLengthDim1 == weight1N1, ACLNN_ERR_PARAM_INVALID,
                       "scale length should be %ld, but scale length is %ld", weight1N1, scaleLengthDim1);
            CHECK_COND(deqScaleDtype == DataType::DT_UINT64 || deqScaleDtype == DataType::DT_BF16 ||
                           deqScaleDtype == DataType::DT_INT64,
                       ACLNN_ERR_PARAM_INVALID,
                       "In per-channel mode, deqScaleDtype should be UINT64 or INT64 for output dtype is FLOAT16, "
                       "BFLOAT16 for output dtype is BFLOAT16, others are not supported.");
        }
    } else {
        CHECK_COND(scaleDimNum == 1 && (scaleLengthDim0 == 1 || scaleLengthDim0 == weight1N1), ACLNN_ERR_PARAM_INVALID,
                   "scale dimNum should be 1 and dimValue should be equal to n1 or 1 when no experts, but currently "
                   "dimNum is %zu, dimValue is %ld",
                   scaleDimNum, scaleLengthDim0);
        if (scaleLengthDim0 == weight1N1 && weight1N1 != 1) { // 1 represents the last dimension of weight1
            CHECK_COND(deqScaleDtype == DataType::DT_UINT64 || deqScaleDtype == DataType::DT_BF16 ||
                           deqScaleDtype == DataType::DT_INT64,
                       ACLNN_ERR_PARAM_INVALID,
                       "In per-channel mode, deqScaleDtype should be UINT64 or INT64 for output dtype is FLOAT16, "
                       "BFLOAT16 for output dtype is BFLOAT16, others are not supported.");
        }
    }
    int64_t offsetLength = ffnParams.offset->GetViewShape().GetDim(0);
    CHECK_COND(offsetLength == requiredLength, ACLNN_ERR_PARAM_INVALID,
               "offset length should be %ld, but offset length is %ld", requiredLength, offsetLength);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckQuantDimNumAndShape(const FFNParams &ffnParams)
{
    // check deqScale dim num
    bool hasExperts = (ffnParams.expertTokens != nullptr);
    size_t dequantParamSize = hasExperts ? DIM_NUM_TWO : DIM_NUM_ONE;
    auto dequantScale1DimNum = ffnParams.deqScale1->GetViewShape().GetDimNum();
    auto dequantScale2DimNum = ffnParams.deqScale2->GetViewShape().GetDimNum();
    CHECK_COND(dequantScale1DimNum == dequantParamSize, ACLNN_ERR_PARAM_INVALID,
               "deqScale1 DimNum should be (2: has experts, 1: no experts), but deqScale1 DimNum is %zu",
               dequantScale1DimNum);
    CHECK_COND(dequantScale2DimNum == dequantParamSize, ACLNN_ERR_PARAM_INVALID,
               "deqScale2 DimNum should be (2: has experts, 1: no experts), but deqScale2 DimNum is %zu",
               dequantScale2DimNum);
    auto offsetDimNum = ffnParams.offset->GetViewShape().GetDimNum();
    CHECK_COND(offsetDimNum == 1, ACLNN_ERR_PARAM_INVALID, "offset DimNum should be 1, but offset DimNum is %zu",
               offsetDimNum);
    // check deqScale last dim is equal to the corresponding weight
    size_t weightSize = hasExperts ? DIM_NUM_THREE : DIM_NUM_TWO;
    size_t weightNDimIdx = weightSize - 1;
    size_t dequantParamNDimIdx = dequantParamSize - 1;
    int64_t weight1NDimValue = ffnParams.weight1->GetViewShape().GetDim(weightNDimIdx);
    int64_t weight2NDimValue = ffnParams.weight2->GetViewShape().GetDim(weightNDimIdx);
    int64_t dequantScale1NDimValue = ffnParams.deqScale1->GetViewShape().GetDim(dequantParamNDimIdx);
    int64_t dequantScale2NDimValue = ffnParams.deqScale2->GetViewShape().GetDim(dequantParamNDimIdx);
    CHECK_COND(dequantScale1NDimValue == weight1NDimValue, ACLNN_ERR_PARAM_INVALID,
               "deqScale1 last dim should be %ld, but deqScale1 last dim is %ld", weight1NDimValue,
               dequantScale1NDimValue);
    CHECK_COND(dequantScale2NDimValue == weight2NDimValue, ACLNN_ERR_PARAM_INVALID,
               "deqScale2 last dim should be %ld, but deqScale2 last dim is %ld", weight2NDimValue,
               dequantScale2NDimValue);
    CHECK_COND(CheckQuantScaleAndOffset(ffnParams, hasExperts, weightNDimIdx) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected quant scale and offset params dim num or shape or dtype is not right!");

    if (hasExperts) {
        int64_t dequantScale1EDimValue = ffnParams.deqScale1->GetViewShape().GetDim(0);
        int64_t dequantScale2EDimValue = ffnParams.deqScale2->GetViewShape().GetDim(0);
        int64_t expertTokensLength = ffnParams.expertTokens->GetViewShape().GetDim(0);
        CHECK_COND(dequantScale1EDimValue == expertTokensLength, ACLNN_ERR_PARAM_INVALID,
                   "deqScale1 first dim should be %ld, but deqScale1 first dim is %ld", expertTokensLength,
                   dequantScale1EDimValue);
        CHECK_COND(dequantScale2EDimValue == expertTokensLength, ACLNN_ERR_PARAM_INVALID,
                   "deqScale2 first dim should be %ld, but deqScale2 first dim is %ld", expertTokensLength,
                   dequantScale2EDimValue);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckQuant(const FFNParams &ffnParams)
{
    // check quant params is not nullptr
    OP_CHECK_NULL(ffnParams.scale, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_NULL(ffnParams.offset, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_NULL(ffnParams.deqScale1, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_NULL(ffnParams.deqScale2, return ACLNN_ERR_PARAM_INVALID);

    DataType yDtype = ffnParams.y->GetDataType();
    const std::initializer_list<op::DataType> dequantDtypeSupportList = {
        op::DataType::DT_UINT64, op::DataType::DT_INT64, op::DataType::DT_FLOAT};
    if (yDtype == DataType::DT_BF16) {
        OP_CHECK_DTYPE_NOT_MATCH(ffnParams.deqScale1, op::DataType::DT_BF16, return ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(ffnParams.deqScale1, dequantDtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    }
    OP_CHECK_DTYPE_NOT_SAME(ffnParams.deqScale1, ffnParams.deqScale2, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(ffnParams.weight1, ffnParams.weight2, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(ffnParams.scale, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(ffnParams.offset, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);

    CHECK_COND(IsAntiQuantEmpty(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected quant input, but antiquant inputs is not empty!");
    CHECK_COND(CheckQuantDimNumAndShape(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected quant params dim num or shape or dtype is not right!");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckHighPrecisionBF16(const FFNParams &ffnParams)
{
    OP_CHECK_DTYPE_NOT_SAME(ffnParams.weight1, ffnParams.weight2, return ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(IsQuantEmpty(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected high precision, but quant inputs is not empty!");
    CHECK_COND(IsAntiQuantEmpty(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected high precision, but antiquant inputs is not empty!");
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
        // arch35 fused gelu：bias 与 linear 对齐，支持 bf16 / float32
        CHECK_COND(CheckBiasDtype(ffnParams, {DataType::DT_FLOAT, DataType::DT_BF16}) == ACLNN_SUCCESS,
                   ACLNN_ERR_PARAM_INVALID, "Detected high precision, bias dtype is not right!");
    } else {
        CHECK_COND(CheckBiasDtype(ffnParams, {DataType::DT_FLOAT}) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Detected high precision, bias dtype is not right!");
    }
    CHECK_COND(ffnParams.activationType != ActiveType::GEGLU, ACLNN_ERR_PARAM_INVALID,
               "Detected high precision, GEGLU is not supported!");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckHighPrecisionFP16(const FFNParams &ffnParams)
{
    OP_CHECK_DTYPE_NOT_MATCH(ffnParams.weight1, op::DataType::DT_FLOAT16, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(ffnParams.weight1, ffnParams.weight2, return ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(IsQuantEmpty(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected high precision, but quant inputs is not empty!");
    CHECK_COND(IsAntiQuantEmpty(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected high precision, but antiquant inputs is not empty!");
    OP_CHECK_DTYPE_NOT_SAME(ffnParams.x, ffnParams.weight1, return ACLNN_ERR_PARAM_INVALID);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
        CHECK_COND(CheckBiasDtype(ffnParams, {DataType::DT_FLOAT16, DataType::DT_FLOAT}) == ACLNN_SUCCESS,
                   ACLNN_ERR_PARAM_INVALID, "Detected high precision, bias dtype is not right!");
    } else {
        CHECK_COND(CheckBiasDtype(ffnParams, {DataType::DT_FLOAT16}) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Detected high precision, bias dtype is not right!");
    }
    CHECK_COND(ffnParams.activationType != ActiveType::GEGLU, ACLNN_ERR_PARAM_INVALID,
               "Detected high precision, GEGLU is not supported!");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAntiQuantDimNum(const FFNParams &ffnParams)
{
    // check dims of antiquant params
    bool hasExperts = (ffnParams.expertTokens != nullptr);
    // per group num = 1
    size_t antiquantParamSize = hasExperts ? DIM_NUM_TWO : DIM_NUM_ONE;
    // per group num > 1
    size_t groupAntiquantParamSize = hasExperts ? DIM_NUM_THREE : DIM_NUM_TWO;
    auto antiquantScale1DimNum = ffnParams.antiquantScale1->GetViewShape().GetDimNum();
    auto antiquantOffset1DimNum = ffnParams.antiquantOffset1->GetViewShape().GetDimNum();
    auto antiquantScale2DimNum = ffnParams.antiquantScale2->GetViewShape().GetDimNum();
    auto antiquantOffset2DimNum = ffnParams.antiquantOffset2->GetViewShape().GetDimNum();

    CHECK_COND((antiquantScale1DimNum == antiquantParamSize) || (antiquantScale1DimNum == groupAntiquantParamSize),
               ACLNN_ERR_PARAM_INVALID,
               "antiquantScale1 DimNum should be (3 or 2: has experts, 2 or 1: no experts), but DimNum is %zu",
               antiquantScale1DimNum);
    CHECK_COND(antiquantScale1DimNum == antiquantOffset1DimNum, ACLNN_ERR_PARAM_INVALID,
               "antiquantOffset1 DimNum should be equal to antiquantScale1, but antiquantOffset1 DimNum is %zu",
               antiquantOffset1DimNum);
    CHECK_COND(antiquantScale1DimNum == antiquantScale2DimNum, ACLNN_ERR_PARAM_INVALID,
               "antiquantScale2 DimNum should be equal to antiquantScale1, but antiquantScale2 DimNum is %zu",
               antiquantScale2DimNum);
    CHECK_COND(antiquantScale2DimNum == antiquantOffset2DimNum, ACLNN_ERR_PARAM_INVALID,
               "antiquantOffset2 DimNum should be equal to antiquantScale2, but antiquantOffset2 DimNum is %zu",
               antiquantOffset2DimNum);
    for (size_t i = 0; i < antiquantScale1DimNum; i++) {
        int64_t curScaleShape = ffnParams.antiquantScale1->GetViewShape().GetDim(i);
        int64_t curOffsetShape = ffnParams.antiquantOffset1->GetViewShape().GetDim(i);
        CHECK_COND(curScaleShape == curOffsetShape, ACLNN_ERR_PARAM_INVALID,
                   "antiquantScale1 shape should be equal to antiquantOffset1 shape, now is %ld vs %ld on dim %zu",
                   curScaleShape, curOffsetShape, i);
    }
    for (size_t i = 0; i < antiquantScale2DimNum; i++) {
        int64_t curScaleShape = ffnParams.antiquantScale2->GetViewShape().GetDim(i);
        int64_t curOffsetShape = ffnParams.antiquantOffset2->GetViewShape().GetDim(i);
        CHECK_COND(curScaleShape == curOffsetShape, ACLNN_ERR_PARAM_INVALID,
                   "antiquantScale2 shape should be equal to antiquantOffset2 shape, now is %ld vs %ld on dim %zu",
                   curScaleShape, curOffsetShape, i);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAntiQuantW4Shape(const FFNParams &ffnParams, const size_t weightNDimIdx)
{
    DataType w1Dtype = ffnParams.weight1->GetDataType();
    if (w1Dtype == DataType::DT_INT4) {
        int64_t weight1NDimValue = ffnParams.weight1->GetViewShape().GetDim(weightNDimIdx);
        int64_t weight2NDimValue = ffnParams.weight2->GetViewShape().GetDim(weightNDimIdx);
        CHECK_COND(weight1NDimValue % JUDGE_EVEN == 0, ACLNN_ERR_PARAM_INVALID,
                   "when w1 is int4, the last dimension of the shape N1 %ld should be an even number!",
                   weight1NDimValue);
        CHECK_COND(weight2NDimValue % JUDGE_EVEN == 0, ACLNN_ERR_PARAM_INVALID,
                   "when w2 is int4, the last dimension of the shape N2 %ld should be an even number!",
                   weight2NDimValue);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAntiQuantShapeE(const FFNParams &ffnParams)
{
    if (ffnParams.expertTokens != nullptr) {
        int64_t antiquantScale1EDimValue = ffnParams.antiquantScale1->GetViewShape().GetDim(0);
        int64_t antiquantOffset1EDimValue = ffnParams.antiquantOffset1->GetViewShape().GetDim(0);
        int64_t antiquantScale2EDimValue = ffnParams.antiquantScale2->GetViewShape().GetDim(0);
        int64_t antiquantOffset2EDimValue = ffnParams.antiquantOffset2->GetViewShape().GetDim(0);
        int64_t expertTokensLength = ffnParams.expertTokens->GetViewShape().GetDim(0);
        CHECK_COND(antiquantScale1EDimValue == expertTokensLength, ACLNN_ERR_PARAM_INVALID,
                   "antiquantScale1 first dim should be %ld, but antiquantScale1 first dim is %ld", expertTokensLength,
                   antiquantScale1EDimValue);
        CHECK_COND(antiquantScale2EDimValue == expertTokensLength, ACLNN_ERR_PARAM_INVALID,
                   "antiquantScale2 first dim should be %ld, but antiquantScale2 first dim is %ld", expertTokensLength,
                   antiquantScale2EDimValue);
        CHECK_COND(antiquantOffset1EDimValue == expertTokensLength, ACLNN_ERR_PARAM_INVALID,
                   "antiquantOffset1 first dim should be %ld, but antiquantOffset1 first dim is %ld",
                   expertTokensLength, antiquantOffset1EDimValue);
        CHECK_COND(antiquantOffset2EDimValue == expertTokensLength, ACLNN_ERR_PARAM_INVALID,
                   "antiquantOffset2 first dim should be %ld, but antiquantOffset2 first dim is %ld",
                   expertTokensLength, antiquantOffset2EDimValue);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAntiQuantShape(const FFNParams &ffnParams)
{
    // check last dim of antiquant params is equal to the corresponding weight
    size_t antiquantParamSize = ffnParams.antiquantScale1->GetViewShape().GetDimNum();
    size_t weightSize = ffnParams.expertTokens != nullptr ? DIM_NUM_THREE : DIM_NUM_TWO;
    size_t weightNDimIdx = weightSize - 1;
    size_t antiquantParamNDimIdx = antiquantParamSize - 1;
    int64_t weight1NDimValue = ffnParams.weight1->GetViewShape().GetDim(weightNDimIdx);
    int64_t weight2NDimValue = ffnParams.weight2->GetViewShape().GetDim(weightNDimIdx);
    int64_t antiquantScale1NDimValue = ffnParams.antiquantScale1->GetViewShape().GetDim(antiquantParamNDimIdx);
    int64_t antiquantOffset1NDimValue = ffnParams.antiquantOffset1->GetViewShape().GetDim(antiquantParamNDimIdx);
    int64_t antiquantScale2NDimValue = ffnParams.antiquantScale2->GetViewShape().GetDim(antiquantParamNDimIdx);
    int64_t antiquantOffset2NDimValue = ffnParams.antiquantOffset2->GetViewShape().GetDim(antiquantParamNDimIdx);
    CHECK_COND(antiquantScale1NDimValue == weight1NDimValue, ACLNN_ERR_PARAM_INVALID,
               "antiquantScale1 last dim should be %ld, but antiquantScale1 last dim is %ld", weight1NDimValue,
               antiquantScale1NDimValue);
    CHECK_COND(antiquantOffset1NDimValue == weight1NDimValue, ACLNN_ERR_PARAM_INVALID,
               "antiquantOffset1 last dim should be %ld, but antiquantOffset1 last dim is %ld", weight1NDimValue,
               antiquantOffset1NDimValue);
    CHECK_COND(antiquantScale2NDimValue == weight2NDimValue, ACLNN_ERR_PARAM_INVALID,
               "antiquantScale2 last dim should be %ld, but antiquantScale2 last dim is %ld", weight2NDimValue,
               antiquantScale2NDimValue);
    CHECK_COND(antiquantOffset2NDimValue == weight2NDimValue, ACLNN_ERR_PARAM_INVALID,
               "antiquantOffset2 last dim should be %ld, but antiquantOffset2 last dim is %ld", weight2NDimValue,
               antiquantOffset2NDimValue);

    CHECK_COND(CheckAntiQuantW4Shape(ffnParams, weightNDimIdx) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected antiquant w is int4, shape N is not an even number!");
    CHECK_COND(CheckAntiQuantShapeE(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected antiquant params dim0 and expert_tokens length are not equal");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAntiQuantPergroup(const FFNParams &ffnParams)
{
    bool hasExperts = (ffnParams.expertTokens != nullptr);
    size_t antiquantParamSize = ffnParams.antiquantScale1->GetViewShape().GetDimNum();
    if ((hasExperts && antiquantParamSize == DIM_NUM_THREE) || (!hasExperts && antiquantParamSize == DIM_NUM_TWO)) {
        size_t weightSize = ffnParams.weight1->GetViewShape().GetDimNum();
        size_t weightKDimIdx = weightSize - 2;
        int64_t weight1KDimValue = ffnParams.weight1->GetViewShape().GetDim(weightKDimIdx);
        int64_t weight2KDimValue = ffnParams.weight2->GetViewShape().GetDim(weightKDimIdx);
        DataType w1Dtype = ffnParams.weight1->GetDataType();
        DataType w2Dtype = ffnParams.weight2->GetDataType();
        size_t antiquantParamPergroupIdx = antiquantParamSize - 2;
        int64_t antiquantScale1GroupValue = ffnParams.antiquantScale1->GetViewShape().GetDim(antiquantParamPergroupIdx);
        int64_t antiquantScale2GroupValue = ffnParams.antiquantScale2->GetViewShape().GetDim(antiquantParamPergroupIdx);
        CHECK_COND(antiquantScale1GroupValue >= 1, ACLNN_ERR_PARAM_INVALID,
                   "antiquantScale1GroupValue %ld must be greater than or equal to 1.", antiquantScale1GroupValue);
        CHECK_COND(antiquantScale2GroupValue >= 1, ACLNN_ERR_PARAM_INVALID,
                   "antiquantScale2GroupValue %ld must be greater than or equal to 1.", antiquantScale2GroupValue);
        antiquantScale1GroupValue = antiquantScale1GroupValue > 1 ? antiquantScale1GroupValue : 1;
        antiquantScale2GroupValue = antiquantScale2GroupValue > 1 ? antiquantScale2GroupValue : 1;
        CHECK_COND((weight1KDimValue % antiquantScale1GroupValue) == 0, ACLNN_ERR_PARAM_INVALID,
                   "weight1KDimValue %ld must be divisible by antiquantScale1GroupValue %ld, with no remainder.",
                   weight1KDimValue, antiquantScale1GroupValue);
        CHECK_COND((weight2KDimValue % antiquantScale2GroupValue) == 0, ACLNN_ERR_PARAM_INVALID,
                   "weight2KDimValue %ld must be divisible by antiquantScale2GroupValue %ld, with no remainder.",
                   weight2KDimValue, antiquantScale2GroupValue);
        CHECK_COND((w1Dtype == DataType::DT_INT4) && (w2Dtype == DataType::DT_INT4), ACLNN_ERR_PARAM_INVALID,
                   "weight dtype must be INT4.");
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAntiQuant(const FFNParams &ffnParams, const DtypeCheck &supportList,
                                  const DtypeCheck &biasSupportList)
{
    // check antiquant params is not nullptr
    OP_CHECK_NULL(ffnParams.antiquantScale1, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_NULL(ffnParams.antiquantScale2, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_NULL(ffnParams.antiquantOffset1, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_NULL(ffnParams.antiquantOffset2, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(ffnParams.weight1, ffnParams.weight2, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(ffnParams.antiquantScale1, supportList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(ffnParams.antiquantOffset1, supportList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(ffnParams.antiquantScale2, supportList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(ffnParams.antiquantOffset2, supportList, return ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(IsQuantEmpty(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected antiquant input, but quant inputs is not empty!");
    CHECK_COND(CheckBiasDtype(ffnParams, biasSupportList) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected AntiQuant, bias dtype is not right!");
    CHECK_COND(CheckAntiQuantDimNum(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected antiquant params dim num is not right!");
    CHECK_COND(CheckAntiQuantShape(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected antiquant params shape is not right!");
    CHECK_COND(CheckAntiQuantPergroup(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected antiquant params per group num or weight dtype is not right!");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckGlu(const FFNParams &ffnParams)
{
    CHECK_RET(CheckDtypeValidForFFNOp(ffnParams.x, ffnParams.weight1, ffnParams.weight2), ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(ffnParams.innerPrecise == HIGH_PERFORMANCE, ACLNN_ERR_PARAM_INVALID,
               "Detected glu, inner precision is not right!");
    CHECK_COND(ffnParams.expertTokens == nullptr, ACLNN_ERR_PARAM_INVALID,
               "Detected glu, expert tokens should be nullptr, but current it is not nullptr!");
    CHECK_COND(CheckBiasDtype(ffnParams, {DataType::DT_FLOAT16}) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected glu, bias dtype is not right!");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckHighPerformance(const FFNParams &ffnParams)
{
    OP_CHECK_DTYPE_NOT_MATCH(ffnParams.weight1, op::DataType::DT_FLOAT16, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SAME(ffnParams.weight1, ffnParams.weight2, return ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(IsQuantEmpty(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected quant input, but quant inputs is not empty!");
    CHECK_COND(IsAntiQuantEmpty(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected antiquant input, but quant inputs is not empty!");
    CHECK_COND(CheckBiasDtype(ffnParams, {DataType::DT_FLOAT16}) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Detected HighPerformance, bias dtype is not right!");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOptionalParams(FFNParams &ffnParams)
{
    DataType xDtype = ffnParams.x->GetDataType();
    DataType w1Dtype = ffnParams.weight1->GetDataType();

    const DtypeCheck xSupportList = {DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT8};
    OP_CHECK_DTYPE_NOT_SUPPORT(ffnParams.x, xSupportList, return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_DTYPE_NOT_SAME(ffnParams.weight1, ffnParams.weight2, return ACLNN_ERR_PARAM_INVALID);
    const DtypeCheck wSupportList = {DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT8, DataType::DT_INT4};
    OP_CHECK_DTYPE_NOT_SUPPORT(ffnParams.weight1, wSupportList, return ACLNN_ERR_PARAM_INVALID);

    if (static_cast<uint32_t>(ffnParams.activationType) >= static_cast<uint32_t>(ActiveType::GEGLU)) {
        if (!IsArch35Fused(ffnParams)) {
            return CheckGlu(ffnParams); // 老 glu 路径（fp16 专用）；arch35 走 fused 路径
        }
    }

    if (xDtype == DataType::DT_BF16) {
        OP_CHECK_DTYPE_NOT_MATCH(ffnParams.y, op::DataType::DT_BF16, return ACLNN_ERR_PARAM_INVALID);
        if (w1Dtype == DataType::DT_BF16) {
            CHECK_COND(ffnParams.innerPrecise == HIGH_PRECISION, ACLNN_ERR_PARAM_INVALID,
                       "FFN bfloat16 is only support high precision now!");
            return CheckHighPrecisionBF16(ffnParams); // HIGH_PRECISION_BF16
        } else if (w1Dtype == DataType::DT_INT8 || w1Dtype == DataType::DT_INT4) {
            return CheckAntiQuant(ffnParams, {DataType::DT_BF16}, {DataType::DT_FLOAT}); // ANTIQUANT_BF16_W8+W4
        } else {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    DataType yDtype = ffnParams.y->GetDataType();
    if (xDtype == DataType::DT_INT8) {
        OP_CHECK_DTYPE_NOT_MATCH(ffnParams.weight1, op::DataType::DT_INT8, return ACLNN_ERR_PARAM_INVALID);
        CHECK_COND(CheckBiasDtype(ffnParams, {DataType::DT_INT32}) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Detected QUANT, bias dtype is not right!");
        if (yDtype == DataType::DT_BF16 || yDtype == DataType::DT_FLOAT16) {
            return CheckQuant(ffnParams); // QUANT
        } else {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    OP_CHECK_DTYPE_NOT_MATCH(ffnParams.y, op::DataType::DT_FLOAT16, return ACLNN_ERR_PARAM_INVALID);

    // mmDataType SHOULD BE DataType::DT_FLOAT16
    CHECK_COND(xDtype == DataType::DT_FLOAT16, ACLNN_ERR_PARAM_INVALID, "Detected xDtype is not right!");
    if (w1Dtype == DataType::DT_INT8 || w1Dtype == DataType::DT_INT4) {
        return CheckAntiQuant(ffnParams, {DataType::DT_FLOAT16}, {DataType::DT_FLOAT16}); // ANTIQUANT_FP16_W8+W4
    }
    if (ffnParams.innerPrecise == HIGH_PRECISION) {
        return CheckHighPrecisionFP16(ffnParams); // HIGH_PRECISION_FP16
    }
    return CheckHighPerformance(ffnParams); // HIGH_PERFORMANCE, expertTokens!=nullptr
}

static aclnnStatus CheckFormat(const FFNParams &ffnParams)
{
    bool is310P = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
    op::Format xFormat = ffnParams.x->GetStorageFormat();
    op::Format weight1Format = ffnParams.weight1->GetStorageFormat();
    op::Format weight2Format = ffnParams.weight2->GetStorageFormat();
    op::Format yFormat = ffnParams.y->GetStorageFormat();
    bool isXFormatValid = xFormat < Format::FORMAT_END &&
                          (!op::IsPrivateFormat(xFormat) || (is310P && xFormat == Format::FORMAT_FRACTAL_NZ));
    CHECK_COND(isXFormatValid, ACLNN_ERR_PARAM_INVALID, "format of x %s is invalid.",
               op::ToString(xFormat).GetString());

    bool isWeight1FormatValid =
        weight1Format < Format::FORMAT_END &&
        (!op::IsPrivateFormat(weight1Format) || (is310P && weight1Format == Format::FORMAT_FRACTAL_NZ));
    CHECK_COND(isWeight1FormatValid, ACLNN_ERR_PARAM_INVALID, "format of weight1 %s is invalid.",
               op::ToString(weight1Format).GetString());

    bool isWeight2FormatValid =
        weight2Format < Format::FORMAT_END &&
        (!op::IsPrivateFormat(weight2Format) || (is310P && weight2Format == Format::FORMAT_FRACTAL_NZ));
    CHECK_COND(isWeight2FormatValid, ACLNN_ERR_PARAM_INVALID, "format of weight2 %s is invalid.",
               op::ToString(weight2Format).GetString());

    bool isYFormatValid = yFormat < Format::FORMAT_END &&
                          (!op::IsPrivateFormat(yFormat) || (is310P && yFormat == Format::FORMAT_FRACTAL_NZ));
    CHECK_COND(isYFormatValid, ACLNN_ERR_PARAM_INVALID, "format of y %s is invalid.",
               op::ToString(yFormat).GetString());

    return ACLNN_SUCCESS;
}

inline static aclnnStatus CheckParam(FFNParams &ffnParams)
{
    CHECK_COND(CheckFormat(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid format.");
    CHECK_RET(CheckY(ffnParams), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFmapWeightShape(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckBias(ffnParams), ACLNN_ERR_PARAM_INVALID);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        auto xDimNum = ffnParams.x->GetViewShape().GetDimNum();
        CHECK_COND(xDimNum == DIM_LIMIT_LOWER, ACLNN_ERR_PARAM_INVALID,
                   "FFN only supports x dim 2 on ASCEND310P, but x dim is %zu", xDimNum);
        CHECK_COND(ffnParams.innerPrecise == HIGH_PERFORMANCE, ACLNN_ERR_PARAM_INVALID,
                   "FFN is only support high performance now!");
        CHECK_COND(ffnParams.activationType <= ActiveType::GELU, ACLNN_ERR_PARAM_INVALID,
                   "now activation types supported by ffn are fastgelu, gelu, relu, silu, please check activation.");
        OP_CHECK_DTYPE_NOT_MATCH(ffnParams.x, op::DataType::DT_FLOAT16, return ACLNN_ERR_PARAM_INVALID);
        OP_CHECK_DTYPE_NOT_MATCH(ffnParams.y, op::DataType::DT_FLOAT16, return ACLNN_ERR_PARAM_INVALID);
        CHECK_COND(ffnParams.expertTokens == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "FFN only supports no expert cases on ASCEND310P.");
        CHECK_RET(CheckHighPerformance(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(ffnParams.activationType != ActiveType::INVALID_TYPE, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckOptionalParams(ffnParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus InputsContiguousAndTransFormat(const aclTensor *tensor, const aclTensor *&reformatedTensor,
                                                  const std::string &tensorName, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    op::Format tensorFormat = tensor->GetStorageFormat();
    if (tensorFormat != Format::FORMAT_FRACTAL_NZ) {
        reformatedTensor = l0op::Contiguous(tensor, executor);
        CHECK_COND(reformatedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "%s Contiguous failed.", tensorName.c_str());

        bool is310P = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
        // only in 310P, tensor should be transdata to NZ format
        if (!is310P) {
            return ACLNN_SUCCESS;
        }

        reformatedTensor = l0op::TransData(reformatedTensor, Format::FORMAT_FRACTAL_NZ, 1, executor);
        CHECK_COND(reformatedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "%s TransData failed.", tensorName.c_str());
        return ACLNN_SUCCESS;
    }
    reformatedTensor = tensor;
    return ACLNN_SUCCESS;
}

static aclnnStatus OutputransFormat(const FFNParams &ffnParams, const aclTensor *tensor,
                                    const aclTensor *&reformatedTensor, aclOpExecutor *executor)
{
    op::Format requiredFormat = ffnParams.y->GetStorageFormat();
    reformatedTensor = l0op::TransData(tensor, requiredFormat, 1, executor);
    CHECK_COND(reformatedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR, "y TransData failed.");
    return ACLNN_SUCCESS;
}

static int64_t GetBatchSizeX(const aclTensor *x)
{
    if (x == nullptr || x->GetViewShape().GetDimNum() == 0) {
        return 1;
    }
    int64_t bs = 1;
    size_t xDimNum = x->GetViewShape().GetDimNum();
    if (xDimNum == 0) {
        return bs;
    }
    for (size_t i = 0; i < xDimNum - 1; i++) {
        bs *= x->GetViewShape().GetDim(i);
    }
    return bs;
}

static aclnnStatus CheckExpertTokens(FFNParams &ffnParams)
{
    size_t numExperts = ffnParams.expertTokensArr->Size();
    int64_t bs = GetBatchSizeX(ffnParams.x);
    int64_t totalNumExpertTokens;

    if (ffnParams.tokensIndexFlag) {
        int64_t prevTokensOffset = 0;
        for (size_t i = 0; i < numExperts; i++) {
            int64_t tokensOffset = (*ffnParams.expertTokensArr)[i];
            CHECK_COND(tokensOffset >= 0, ACLNN_ERR_PARAM_INVALID,
                       "Expert tokens index should not be smaller than zero, but %ld is found.", tokensOffset);
            int64_t tokens = tokensOffset - prevTokensOffset;
            CHECK_COND(
                tokens >= 0, ACLNN_ERR_PARAM_INVALID,
                "Expert tokens indices should not be decremental, but two consecutive numbers %ld and %ld are found.",
                prevTokensOffset, tokensOffset);
            prevTokensOffset = tokensOffset;
        }
        totalNumExpertTokens = (*ffnParams.expertTokensArr)[numExperts - 1];
    } else {
        totalNumExpertTokens = 0;
        for (size_t i = 0; i < numExperts; i++) {
            int64_t tokens = (*ffnParams.expertTokensArr)[i];
            CHECK_COND(tokens >= 0, ACLNN_ERR_PARAM_INVALID,
                       "Expert tokens should not be smaller than zero, but %ld is found.", tokens);
            totalNumExpertTokens += tokens;
        }
    }
    CHECK_COND(
        totalNumExpertTokens == bs, ACLNN_ERR_PARAM_INVALID,
        "Total number of expert tokens should be equal to the product of all x dimensions excluding the last one. "
        "But they are %ld and %ld respectively.",
        totalNumExpertTokens, bs);
    return ACLNN_SUCCESS;
}

static aclnnStatus ConvertExpertTokensAndCheckParams(FFNParams &ffnParams, aclOpExecutor *executor)
{
    if (ffnParams.expertTokens != nullptr && ffnParams.expertTokens->GetViewShape().GetDim(0) == 0) {
        ffnParams.expertTokens = nullptr;
    }

    if (ffnParams.expertTokensArr != nullptr && ffnParams.expertTokensArr->Size() != 0) {
        aclnnStatus ret = CheckExpertTokens(ffnParams);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ffnParams.expertTokens = executor->ConvertToTensor(ffnParams.expertTokensArr, op::ToOpDataType(ACL_INT64));
    }

    aclnnStatus ret = CheckParam(ffnParams);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    return ret;
}

// 校验并转换输入 tensor 为 contiguous 形式。
// 宏在文件作用域定义（非函数体内），避免宏名作用域混乱；加 FFN_ 前缀防止与其它头文件宏冲突。
#define FFN_CHECK_AND_CONVERT(param, param_name, conti_var) \
    const aclTensor *conti_var = nullptr; \
    CHECK_COND(InputsContiguousAndTransFormat(ffnParams.param, conti_var, #param_name, uniqueExecutor.get()) == \
                   ACLNN_SUCCESS, \
               ACLNN_ERR_PARAM_INVALID, "Convert " #param_name " to contiguous tensor failed.")

static aclnnStatus GetFFNResultByL0Api(FFNParams &ffnParams, const char *activation, uint64_t *workspaceSize,
                                       aclOpExecutor **executor)
{
    // create OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // check params
    aclnnStatus ret = ConvertExpertTokensAndCheckParams(ffnParams, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (ffnParams.x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor *reformatedX = nullptr;
    const aclTensor *reformatedWeight1 = nullptr;
    const aclTensor *reformatedWeight2 = nullptr;
    ret = InputsContiguousAndTransFormat(ffnParams.x, reformatedX, "x", uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = InputsContiguousAndTransFormat(ffnParams.weight1, reformatedWeight1, "weight1", uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    ret = InputsContiguousAndTransFormat(ffnParams.weight2, reformatedWeight2, "weight2", uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // when input tensor is from pytorch framework, its view_shape/ori_shape/storage_shape is different
    // view_shape is ND-dimension as we expected, but ori_shape/storage_shape is flattened.
    // Now storage_shape is used in tiling process to determine some scenarios like pergroup antiquant,
    // smooth quant and etc. So we call l0op::contiguous to keep them identical.
    // Currently quant_scale, antiquant_scale1,  antiquant_scale2 need to be processed.

    // 使用宏进行校验和转换
    FFN_CHECK_AND_CONVERT(expertTokens, expertTokens, contiExpertTokens);
    FFN_CHECK_AND_CONVERT(bias1, bias1, contiBias1);
    FFN_CHECK_AND_CONVERT(bias2, bias2, contiBias2);
    FFN_CHECK_AND_CONVERT(scale, scale, contiQuantScale);
    FFN_CHECK_AND_CONVERT(offset, offset, contiOffset);
    FFN_CHECK_AND_CONVERT(deqScale1, deqScale1, contiDeqScale1);
    FFN_CHECK_AND_CONVERT(deqScale2, deqScale2, contiDeqScale2);
    FFN_CHECK_AND_CONVERT(antiquantScale1, antiquantScale1, contiAntiScale1);
    FFN_CHECK_AND_CONVERT(antiquantScale2, antiquantScale2, contiAntiScale2);
    FFN_CHECK_AND_CONVERT(antiquantOffset1, antiquantOffset1, contiAntiOffset1);
    FFN_CHECK_AND_CONVERT(antiquantOffset2, antiquantOffset2, contiAntiOffset2);
#undef FFN_CHECK_AND_CONVERT

    // call l0 interface
    DataType yDtype = ffnParams.y->GetDataType();
    auto ffnResult = l0op::FFN(reformatedX, reformatedWeight1, reformatedWeight2, contiExpertTokens, contiBias1,
                               contiBias2, contiQuantScale, contiOffset, contiDeqScale1, contiDeqScale2,
                               contiAntiScale1, contiAntiScale2, contiAntiOffset1, contiAntiOffset2, activation,
                               ffnParams.innerPrecise, yDtype, ffnParams.tokensIndexFlag, uniqueExecutor.get());
    CHECK_RET(ffnResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *reformatedFFNResult = nullptr;
    ret = OutputransFormat(ffnParams, ffnResult, reformatedFFNResult, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // convert output tensor to contiguous tensor
    auto viewCopyResult = l0op::ViewCopy(reformatedFFNResult, ffnParams.y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // get workspace size
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

static aclnnStatus TransDataTypeDeqscale(FFNParams &ffnParams)
{
    auto deqScale1 = const_cast<aclTensor *>(ffnParams.deqScale1);
    auto deqScale2 = const_cast<aclTensor *>(ffnParams.deqScale2);
    DataType yDtype = ffnParams.y->GetDataType();
    if (deqScale1 != nullptr && deqScale2 != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(deqScale1, deqScale2, return ACLNN_ERR_PARAM_INVALID);
    }
    if (deqScale1 != nullptr && yDtype == DataType::DT_FLOAT16) {
        DataType deqScale1Dtype = ffnParams.deqScale1->GetDataType();
        if (deqScale1Dtype == DataType::DT_INT64) {
            deqScale1->SetDataType(op::DataType::DT_UINT64);
        }
    }
    if (deqScale2 != nullptr && yDtype == DataType::DT_FLOAT16) {
        DataType deqScale2Dtype = ffnParams.deqScale2->GetDataType();
        if (deqScale2Dtype == DataType::DT_INT64) {
            deqScale2->SetDataType(op::DataType::DT_UINT64);
        }
    }
    ffnParams.deqScale1 = deqScale1;
    ffnParams.deqScale2 = deqScale2;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFFNGetWorkspaceSize(const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2,
                                     const aclIntArray *expertTokens, const aclTensor *bias1, const aclTensor *bias2,
                                     const aclTensor *scale, const aclTensor *offset, const aclTensor *deqScale1,
                                     const aclTensor *deqScale2, const aclTensor *antiquantScale1,
                                     const aclTensor *antiquantScale2, const aclTensor *antiquantOffset1,
                                     const aclTensor *antiquantOffset2, const char *activation, int64_t innerPrecise,
                                     const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND310P, ACLNN_ERR_PARAM_INVALID,
               "aclnnFFNGetWorkspaceSize and aclnnFFN are not supported on Ascend310P Soc. Please use "
               "aclnnFFNV2GetWorkspaceSize and aclnnFFNV2!");
    CHECK_COND(activation != nullptr, ACLNN_ERR_PARAM_NULLPTR, "activation must not be nullptr.");
    ActiveType activationType = GetActiveType(activation);
    CHECK_COND(activationType != ActiveType::INVALID_TYPE, ACLNN_ERR_PARAM_INVALID,
               "the activation type %s is not supported by ffn operator, please select right activation according to "
               "the document",
               activation);
    CHECK_COND(CheckNotNull(x, weight1, weight2, y) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
               "one of required inputs for aclnnFFNGetWorkspaceSize is nullptr.");
    FFNParams ffnParams{x,
                        weight1,
                        weight2,
                        expertTokens,
                        nullptr,
                        bias1,
                        bias2,
                        scale,
                        offset,
                        deqScale1,
                        deqScale2,
                        antiquantScale1,
                        antiquantScale2,
                        antiquantOffset1,
                        antiquantOffset2,
                        activationType,
                        innerPrecise,
                        false,
                        y};
    aclnnStatus ret = TransDataTypeDeqscale(ffnParams);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    L2_DFX_PHASE_1(aclnnFFN,
                   DFX_IN(ffnParams.x, ffnParams.weight1, ffnParams.weight2, expertTokens, ffnParams.bias1,
                          ffnParams.bias2, ffnParams.scale, ffnParams.offset, ffnParams.deqScale1, ffnParams.deqScale2,
                          ffnParams.antiquantScale1, ffnParams.antiquantScale2, ffnParams.antiquantOffset1,
                          ffnParams.antiquantOffset2, activation, ffnParams.innerPrecise, -1, false),
                   DFX_OUT(ffnParams.y));
    return GetFFNResultByL0Api(ffnParams, activation, workspaceSize, executor);
}

aclnnStatus aclnnFFN(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFFN);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in FFN launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFFNV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2,
                                       const aclIntArray *expertTokens, const aclTensor *bias1, const aclTensor *bias2,
                                       const aclTensor *scale, const aclTensor *offset, const aclTensor *deqScale1,
                                       const aclTensor *deqScale2, const aclTensor *antiquantScale1,
                                       const aclTensor *antiquantScale2, const aclTensor *antiquantOffset1,
                                       const aclTensor *antiquantOffset2, const char *activation, int64_t innerPrecise,
                                       bool tokensIndexFlag, const aclTensor *y, uint64_t *workspaceSize,
                                       aclOpExecutor **executor)
{
    CHECK_COND(activation != nullptr, ACLNN_ERR_PARAM_NULLPTR, "activation must not be nullptr.");
    ActiveType activationType = GetActiveType(activation);
    CHECK_COND(activationType != ActiveType::INVALID_TYPE, ACLNN_ERR_PARAM_INVALID,
               "the activation type %s is not supported by ffn operator, please select right activation according to "
               "the document",
               activation);
    CHECK_COND(CheckNotNull(x, weight1, weight2, y) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
               "one of required inputs for aclnnFFNV2GetWorkspaceSize is nullptr.");
    FFNParams ffnParams{x,
                        weight1,
                        weight2,
                        expertTokens,
                        nullptr,
                        bias1,
                        bias2,
                        scale,
                        offset,
                        deqScale1,
                        deqScale2,
                        antiquantScale1,
                        antiquantScale2,
                        antiquantOffset1,
                        antiquantOffset2,
                        activationType,
                        innerPrecise,
                        tokensIndexFlag,
                        y};
    aclnnStatus ret = TransDataTypeDeqscale(ffnParams);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    L2_DFX_PHASE_1(aclnnFFNV2,
                   DFX_IN(ffnParams.x, ffnParams.weight1, ffnParams.weight2, expertTokens, ffnParams.bias1,
                          ffnParams.bias2, ffnParams.scale, ffnParams.offset, ffnParams.deqScale1, ffnParams.deqScale2,
                          ffnParams.antiquantScale1, ffnParams.antiquantScale2, ffnParams.antiquantOffset1,
                          ffnParams.antiquantOffset2, activation, ffnParams.innerPrecise, -1, tokensIndexFlag),
                   DFX_OUT(ffnParams.y));
    return GetFFNResultByL0Api(ffnParams, activation, workspaceSize, executor);
}

aclnnStatus aclnnFFNV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFFNV2);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in FFN launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFFNV3GetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2, const aclTensor *expertTokensOptional,
    const aclTensor *bias1Optional, const aclTensor *bias2Optional, const aclTensor *scaleOptional,
    const aclTensor *offsetOptional, const aclTensor *deqScale1Optional, const aclTensor *deqScale2Optional,
    const aclTensor *antiquantScale1Optional, const aclTensor *antiquantScale2Optional,
    const aclTensor *antiquantOffset1Optional, const aclTensor *antiquantOffset2Optional, const char *activation,
    int64_t innerPrecise, bool tokensIndexFlag, const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_COND(activation != nullptr, ACLNN_ERR_PARAM_NULLPTR, "activation must not be nullptr.");
    ActiveType activationType = GetActiveType(activation);
    CHECK_COND(activationType != ActiveType::INVALID_TYPE, ACLNN_ERR_PARAM_INVALID,
               "the activation type %s is not supported by ffn operator, please select right activation according to "
               "the document",
               activation);
    CHECK_COND(CheckNotNull(x, weight1, weight2, y) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
               "one of required inputs for aclnnFFNV3GetWorkspaceSize is nullptr.");
    FFNParams ffnParams{x,
                        weight1,
                        weight2,
                        nullptr,
                        expertTokensOptional,
                        bias1Optional,
                        bias2Optional,
                        scaleOptional,
                        offsetOptional,
                        deqScale1Optional,
                        deqScale2Optional,
                        antiquantScale1Optional,
                        antiquantScale2Optional,
                        antiquantOffset1Optional,
                        antiquantOffset2Optional,
                        activationType,
                        innerPrecise,
                        tokensIndexFlag,
                        y};
    aclnnStatus ret = TransDataTypeDeqscale(ffnParams);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    L2_DFX_PHASE_1(aclnnFFNV3,
                   DFX_IN(ffnParams.x, ffnParams.weight1, ffnParams.weight2, expertTokensOptional, ffnParams.bias1,
                          ffnParams.bias2, ffnParams.scale, ffnParams.offset, ffnParams.deqScale1, ffnParams.deqScale2,
                          ffnParams.antiquantScale1, ffnParams.antiquantScale2, ffnParams.antiquantOffset1,
                          ffnParams.antiquantOffset2, activation, ffnParams.innerPrecise, -1, tokensIndexFlag),
                   DFX_OUT(ffnParams.y));
    return GetFFNResultByL0Api(ffnParams, activation, workspaceSize, executor);
}

aclnnStatus aclnnFFNV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFFNV3);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in FFN launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
