/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_router_front.cpp  \brief L0 API. */
#include "dgemma_fused_router_front.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(DgemmaFusedRouterFront);

const aclTensor *DgemmaFusedRouterFront(
    const aclTensor *x, const aclTensor *scale, const aclTensor *projWeight,
    const aclTensor *normScratch, const aclTensor *logitsScratch,
    const aclTensor *perExpertScale, const aclTensor *syncScratch,
    float epsilon, int64_t hiddenSize, int64_t numExperts, int64_t topK,
    int64_t syncBase,
    aclTensor **topkIds,
    aclOpExecutor *executor)
{
    L0_DFX(DgemmaFusedRouterFront, x, scale, projWeight, normScratch,
           logitsScratch, perExpertScale, syncScratch, epsilon, hiddenSize, numExperts, topK, syncBase);
    const Format format = Format::FORMAT_ND;
    int64_t m = x->GetViewShape().GetDim(0);

    op::Shape outShape; outShape.AppendDim(m); outShape.AppendDim(topK);
    auto topkWeights = executor->AllocTensor(outShape, DataType::DT_FLOAT, format);
    OP_CHECK(topkWeights != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "topkWeights AllocTensor failed."), return nullptr);
    auto ids = executor->AllocTensor(outShape, DataType::DT_INT32, format);
    OP_CHECK(ids != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "topkIds AllocTensor failed."), return nullptr);
    *topkIds = ids;

    auto ret = INFER_SHAPE(DgemmaFusedRouterFront,
                           OP_INPUT(x, scale, projWeight, normScratch, logitsScratch, perExpertScale, syncScratch),
                           OP_OUTPUT(topkWeights, ids),
                           OP_ATTR(epsilon, hiddenSize, numExperts, topK, syncBase));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "DgemmaFusedRouterFront InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(DgemmaFusedRouterFront,
                                      OP_INPUT(x, scale, projWeight, normScratch, logitsScratch, perExpertScale, syncScratch),
                                      OP_OUTPUT(topkWeights, ids),
                                      OP_ATTR(epsilon, hiddenSize, numExperts, topK, syncBase));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
        "DgemmaFusedRouterFront ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return topkWeights;
}
} // namespace l0op
