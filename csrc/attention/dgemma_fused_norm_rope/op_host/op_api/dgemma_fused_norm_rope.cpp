/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_norm_rope.cpp
 *  \brief L0-level API for DgemmaFusedNormRope. */
#include "dgemma_fused_norm_rope.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(DgemmaFusedNormRope);
static constexpr DgemmaFusedNormRopeOutput kNullOutput{nullptr, nullptr, nullptr};

DgemmaFusedNormRopeOutput DgemmaFusedNormRope(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *qWeight, const aclTensor *kWeight,
    const aclTensor *cos, const aclTensor *sin,
    float epsilon, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    aclOpExecutor *executor)
{
    L0_DFX(DgemmaFusedNormRope, q, k, v, qWeight, kWeight, cos, sin,
           epsilon, numQHeads, numKvHeads, headDim);
    const Format format = Format::FORMAT_ND;
    auto qOut = executor->AllocTensor(q->GetViewShape(), q->GetDataType(), format);
    OP_CHECK(qOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "qOut AllocTensor failed."), return kNullOutput);
    auto kOut = executor->AllocTensor(k->GetViewShape(), k->GetDataType(), format);
    OP_CHECK(kOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "kOut AllocTensor failed."), return kNullOutput);
    auto vOut = executor->AllocTensor(v->GetViewShape(), v->GetDataType(), format);
    OP_CHECK(vOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "vOut AllocTensor failed."), return kNullOutput);

    auto ret = INFER_SHAPE(DgemmaFusedNormRope,
                           OP_INPUT(q, k, v, qWeight, kWeight, cos, sin),
                           OP_OUTPUT(qOut, kOut, vOut),
                           OP_ATTR(epsilon, numQHeads, numKvHeads, headDim));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return kNullOutput, "DgemmaFusedNormRope InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(DgemmaFusedNormRope,
                                      OP_INPUT(q, k, v, qWeight, kWeight, cos, sin),
                                      OP_OUTPUT(qOut, kOut, vOut),
                                      OP_ATTR(epsilon, numQHeads, numKvHeads, headDim));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return kNullOutput,
        "DgemmaFusedNormRope ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return DgemmaFusedNormRopeOutput{qOut, kOut, vOut};
}
} // namespace l0op
