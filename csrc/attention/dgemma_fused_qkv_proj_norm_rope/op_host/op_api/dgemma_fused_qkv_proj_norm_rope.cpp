/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_qkv_proj_norm_rope.cpp  \brief L0 API. */
#include "dgemma_fused_qkv_proj_norm_rope.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(DgemmaFusedQkvProjNormRope);
static constexpr DgemmaFusedQkvProjNormRopeOutput kNullOutput{nullptr, nullptr, nullptr, nullptr};

DgemmaFusedQkvProjNormRopeOutput DgemmaFusedQkvProjNormRope(
    const aclTensor *hidden, const aclTensor *wqkv,
    const aclTensor *qWeight, const aclTensor *kWeight,
    const aclTensor *cos, const aclTensor *sin,
    const aclTensor *qkvScratch, const aclTensor *qkvScratchOut,
    const aclTensor *qOut, const aclTensor *kOut, const aclTensor *vOut,
    float epsilon, int64_t numQHeads, int64_t numKvHeads, int64_t headDim, int64_t hiddenSize,
    int64_t syncBase,
    aclOpExecutor *executor)
{
    L0_DFX(DgemmaFusedQkvProjNormRope, hidden, wqkv, qWeight, kWeight, cos, sin,
           epsilon, numQHeads, numKvHeads, headDim, hiddenSize, syncBase);
    OP_CHECK(qOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "qOut is null."), return kNullOutput);
    OP_CHECK(kOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "kOut is null."), return kNullOutput);
    OP_CHECK(vOut != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "vOut is null."), return kNullOutput);

    auto ret = INFER_SHAPE(DgemmaFusedQkvProjNormRope,
                           OP_INPUT(hidden, wqkv, qWeight, kWeight, cos, sin, qkvScratch),
                           OP_OUTPUT(qkvScratchOut, qOut, kOut, vOut),
                           OP_ATTR(epsilon, numQHeads, numKvHeads, headDim, hiddenSize, syncBase));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return kNullOutput, "DgemmaFusedQkvProjNormRope InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(DgemmaFusedQkvProjNormRope,
                                      OP_INPUT(hidden, wqkv, qWeight, kWeight, cos, sin, qkvScratch),
                                      OP_OUTPUT(qkvScratchOut, qOut, kOut, vOut),
                                      OP_ATTR(epsilon, numQHeads, numKvHeads, headDim, hiddenSize, syncBase));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return kNullOutput,
        "DgemmaFusedQkvProjNormRope ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return DgemmaFusedQkvProjNormRopeOutput{qkvScratchOut, qOut, kOut, vOut};
}
} // namespace l0op
