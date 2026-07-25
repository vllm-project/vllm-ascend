/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "aclnn_fused_gdn_decode.h"
#include "fused_gdn_decode.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
struct Params {
    const aclTensor *mixedQkv{nullptr};
    const aclTensor *a{nullptr};
    const aclTensor *b{nullptr};
    const aclTensor *aLog{nullptr};
    const aclTensor *dtBias{nullptr};
    aclTensor *stateRef{nullptr};
    const aclTensor *ssmStateIndices{nullptr};
    aclTensor *out{nullptr};
};

static aclnnStatus CheckParams(const Params &p)
{
    OP_CHECK_NULL(p.mixedQkv, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(p.a, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(p.b, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(p.aLog, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(p.dtBias, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(p.stateRef, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(p.ssmStateIndices, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(p.out, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK(p.mixedQkv->GetDataType() == p.a->GetDataType() &&
                 p.mixedQkv->GetDataType() == p.b->GetDataType(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mixed_qkv/a/b dtype mismatch."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(p.aLog->GetDataType() == DataType::DT_FLOAT && p.dtBias->GetDataType() == DataType::DT_FLOAT,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "a_log and dt_bias must be fp32."),
             return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(p.ssmStateIndices->GetDataType() == DataType::DT_INT64,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ssm_state_indices must be int64."),
             return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnFusedGdnDecodeGetWorkspaceSize(
    const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
    const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
    const aclTensor *ssmStateIndices, float scale, float softplusThreshold,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedGdnDecode,
                   DFX_IN(mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, scale, softplusThreshold),
                   DFX_OUT(out, stateRef));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    Params params{mixedQkv, a, b, aLog, dtBias, stateRef, ssmStateIndices, out};
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto mixedQkvContig = l0op::Contiguous(mixedQkv, uniqueExecutor.get());
    auto aContig = l0op::Contiguous(a, uniqueExecutor.get());
    auto bContig = l0op::Contiguous(b, uniqueExecutor.get());
    auto aLogContig = l0op::Contiguous(aLog, uniqueExecutor.get());
    auto dtBiasContig = l0op::Contiguous(dtBias, uniqueExecutor.get());
    auto ssmStateIndicesContig = l0op::Contiguous(ssmStateIndices, uniqueExecutor.get());
    auto outContig = l0op::Contiguous(out, uniqueExecutor.get());

    auto outRet = l0op::FusedGdnDecode(mixedQkvContig, aContig, bContig, aLogContig, dtBiasContig,
                                       stateRef, ssmStateIndicesContig, scale, softplusThreshold,
                                       uniqueExecutor.get());
    CHECK_RET(outRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto copyRet = l0op::ViewCopy(outRet, outContig, uniqueExecutor.get());
    CHECK_RET(copyRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedGdnDecode(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedGdnDecode);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
