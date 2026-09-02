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
 * \file ffn.cpp
 * \brief
 */

#include "ffn.h"
#include "../ffn_layout.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(FFN);

// 与 aclnn GetActiveType 同语义：大小写不敏感、长度精确匹配 "swiglu"
static bool IsSwigluActivation(const char *activation)
{
    static constexpr char swigluName[] = "swiglu";
    if (activation == nullptr) {
        return false;
    }
    for (size_t i = 0; i < sizeof(swigluName) - 1; ++i) {
        if (tolower(static_cast<unsigned char>(activation[i])) != swigluName[i]) {
            return false;
        }
    }
    return activation[sizeof(swigluName) - 1] == '\0';
}

// FFN 输出 shape = x 的 shape，最后一维替换为 weight2 的 N：
// canonical [K, N] 取 w2 最后一维；linear [N, K]（out, in）取 w2 倒数第二维。
static op::Shape GetFFNOutShape(const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2,
                                bool isSwiglu)
{
    op::Shape outShape = x->GetViewShape();
    size_t xDimNum = outShape.GetDimNum();
    size_t w1DimNum = weight1->GetViewShape().GetDimNum();
    size_t w2DimNum = weight2->GetViewShape().GetDimNum();
    if (xDimNum == 0 || w1DimNum < 2 || w2DimNum < 2) {
        return outShape;
    }
    int64_t xK = outShape.GetDim(xDimNum - 1);
    int64_t w1d0 = weight1->GetViewShape().GetDim(w1DimNum - 2);
    int64_t w1d1 = weight1->GetViewShape().GetDim(w1DimNum - 1);
    int64_t w2d0 = weight2->GetViewShape().GetDim(w2DimNum - 2);
    int64_t w2d1 = weight2->GetViewShape().GetDim(w2DimNum - 1);
    // 布局识别走公共规则（../ffn_layout.h）；INVALID/全方阵按 linear 计算，
    // 非法组合由 aclnn/tiling 层拦截报错。
    const bool isLinear = ffnlayout::FfnDetectLayout(w1d0, w1d1, w2d0, w2d1, xK, isSwiglu) ==
                          ffnlayout::FfnLayout::LINEAR;
    int64_t outN = isLinear ? w2d0 : w2d1;
    outShape.SetDim(xDimNum - 1, outN);
    return outShape;
}

const aclTensor *FFN(const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2,
                     const aclTensor *expertTokensOptional, const aclTensor *bias1Optional,
                     const aclTensor *bias2Optional, const aclTensor *scaleOptional, const aclTensor *offsetOptional,
                     const aclTensor *deqScale1Optional, const aclTensor *deqScale2Optional,
                     const aclTensor *antiquantScale1Optional, const aclTensor *antiquantScale2Optional,
                     const aclTensor *antiquantOffset1Optional, const aclTensor *antiquantOffset2Optional,
                     const char *activation, int64_t innerPrecise, const op::DataType yDtype, bool tokensIndexFlag,
                     aclOpExecutor *executor)
{
    L0_DFX(FFN, x, weight1, weight2, bias1Optional, bias2Optional, scaleOptional, offsetOptional, deqScale1Optional,
           deqScale2Optional, antiquantScale1Optional, antiquantScale2Optional, antiquantOffset1Optional,
           antiquantOffset2Optional, activation, innerPrecise, yDtype, tokensIndexFlag);
    op::Shape outShape = GetFFNOutShape(x, weight1, weight2, IsSwigluActivation(activation));
    auto ffnOut = executor->AllocTensor(outShape, outShape, yDtype, x->GetStorageFormat(),
                                        x->GetOriginalFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(FFN,
                                           OP_INPUT(x, weight1, weight2, expertTokensOptional, bias1Optional,
                                                    bias2Optional, scaleOptional, offsetOptional, deqScale1Optional,
                                                    deqScale2Optional, antiquantScale1Optional, antiquantScale2Optional,
                                                    antiquantOffset1Optional, antiquantOffset2Optional),
                                           OP_OUTPUT(ffnOut), OP_ATTR(activation, innerPrecise, -1, tokensIndexFlag));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FFN launch kernel failed.");
        return nullptr;
    }
    return ffnOut;
}

} // namespace l0op