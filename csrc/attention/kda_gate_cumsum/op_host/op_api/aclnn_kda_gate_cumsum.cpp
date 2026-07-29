/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "aclnn_kda_gate_cumsum.h"
#include "kda_gate_cumsum.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#include <cstring>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t MAX_KDA_K_DIM = 256;

enum class KdaGateLayout : int64_t {
    BSND = 0,
    BNSD = 1,
    TND = 2,
    NTD = 3,
};

aclnnStatus KdaGateDataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

int64_t KdaGateDim(const aclTensor *tensor, size_t idx)
{
    return tensor->GetViewShape().GetDim(idx);
}

size_t KdaGateRank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

bool KdaGateSameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    size_t rank = KdaGateRank(lhs);
    if (rank != KdaGateRank(rhs)) {
        return false;
    }
    for (size_t idx = 0; idx < rank; ++idx) {
        if (KdaGateDim(lhs, idx) != KdaGateDim(rhs, idx)) {
            return false;
        }
    }
    return true;
}

aclnnStatus ParseKdaGateLayout(const char *layout, KdaGateLayout &parsed)
{
    CHECK_COND(layout != nullptr, ACLNN_ERR_PARAM_INVALID,
               "layout must be one of BSND, BNSD, TND or NTD.");
    if (std::strcmp(layout, "BSND") == 0) {
        parsed = KdaGateLayout::BSND;
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(layout, "BNSD") == 0) {
        parsed = KdaGateLayout::BNSD;
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(layout, "TND") == 0) {
        parsed = KdaGateLayout::TND;
        return ACLNN_SUCCESS;
    }
    if (std::strcmp(layout, "NTD") == 0) {
        parsed = KdaGateLayout::NTD;
        return ACLNN_SUCCESS;
    }
    CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
               "layout must be uppercase and one of BSND, BNSD, TND or NTD.");
    return ACLNN_ERR_PARAM_INVALID;
}

int64_t KdaGateSeqLen(const aclTensor *g, KdaGateLayout layout)
{
    if (layout == KdaGateLayout::TND) {
        return KdaGateDim(g, 0);
    }
    if (layout == KdaGateLayout::NTD) {
        return KdaGateDim(g, 1);
    }
    return layout == KdaGateLayout::BNSD ? KdaGateDim(g, 2) : KdaGateDim(g, 1);
}

aclnnStatus KdaGateCheckCuSeqlens(const aclIntArray *cuSeqlensOptional, int64_t seqlen)
{
    if (cuSeqlensOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const aclIntArray &cu = *cuSeqlensOptional;
    CHECK_COND(cu.Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional must contain at least [0, total_tokens].");
    CHECK_COND(cu[0] == 0, ACLNN_ERR_PARAM_INVALID, "cuSeqlensOptional[0] must be 0.");
    CHECK_COND(cu[cu.Size() - 1] == seqlen, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional last element must equal the sequence length.");
    for (size_t idx = 0; idx + 1 < cu.Size(); ++idx) {
        CHECK_COND(cu[idx] <= cu[idx + 1], ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must be nondecreasing.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus KdaGateCheckParams(
    const aclTensor *g,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    int64_t chunkSize,
    bool useGateInKernel,
    bool safeGate,
    double lowerBound,
    const char *layoutText,
    const aclTensor *gkOut)
{
    CHECK_COND(g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(gkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gkOut must not be nullptr.");
    CHECK_COND(chunkSize == 32 || chunkSize == 64 || chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize must be 32, 64 or 128.");
    size_t rank = KdaGateRank(g);
    CHECK_COND(rank == 3 || rank == 4, ACLNN_ERR_PARAM_INVALID,
               "g must be BSND/BNSD rank4 or TND/NTD rank3.");
    size_t kDimIdx = rank == 4 ? 3 : 2;
    CHECK_COND(KdaGateDim(g, kDimIdx) <= MAX_KDA_K_DIM, ACLNN_ERR_PARAM_INVALID,
               "K must be less than or equal to 256.");
    CHECK_COND(gkOut->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
               "gkOut must be float32.");
    CHECK_COND(KdaGateSameShape(gkOut, g), ACLNN_ERR_PARAM_INVALID, "gkOut shape must match g shape.");
    KdaGateLayout layout = KdaGateLayout::BSND;
    CHECK_RET(ParseKdaGateLayout(layoutText, layout) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    bool rankMatchesLayout = (rank == 4 && (layout == KdaGateLayout::BSND || layout == KdaGateLayout::BNSD)) ||
                             (rank == 3 && (layout == KdaGateLayout::TND || layout == KdaGateLayout::NTD));
    CHECK_COND(rankMatchesLayout, ACLNN_ERR_PARAM_INVALID,
               "layout rank does not match g rank: BSND/BNSD require rank 4 and TND/NTD require rank 3.");
    CHECK_RET(KdaGateCheckCuSeqlens(cuSeqlensOptional, KdaGateSeqLen(g, layout)) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(cuSeqlensOptional == nullptr || rank == 3 || KdaGateDim(g, 0) == 1, ACLNN_ERR_PARAM_INVALID,
               "rank4 varlen input with cuSeqlensOptional currently requires B=1.");
    int64_t hv = (layout == KdaGateLayout::BNSD) ? KdaGateDim(g, 1) :
                 ((layout == KdaGateLayout::NTD) ? KdaGateDim(g, 0) :
                  ((rank == 4) ? KdaGateDim(g, 2) : KdaGateDim(g, 1)));
    if (useGateInKernel) {
        CHECK_COND(aLogOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "aLogOptional must be provided when useGateInKernel is true.");
        CHECK_COND(safeGate, ACLNN_ERR_PARAM_INVALID,
                   "Only safe_gate raw-gate path is supported; set safeGate=true with lowerBound.");
        CHECK_COND(lowerBound >= -5.0 && lowerBound < 0.0, ACLNN_ERR_PARAM_INVALID,
                   "lowerBound must be in [-5, 0).");
        int64_t k = KdaGateDim(g, kDimIdx);
        CHECK_COND(KdaGateRank(aLogOptional) == 1 && KdaGateDim(aLogOptional, 0) == hv,
                   ACLNN_ERR_PARAM_INVALID, "aLogOptional shape must be [HV].");
        if (dtBiasOptional != nullptr) {
            size_t biasRank = KdaGateRank(dtBiasOptional);
            bool validBias = (biasRank == 1 && KdaGateDim(dtBiasOptional, 0) == hv * k) ||
                             (biasRank == 2 && KdaGateDim(dtBiasOptional, 0) == hv &&
                              KdaGateDim(dtBiasOptional, 1) == k);
            CHECK_COND(validBias, ACLNN_ERR_PARAM_INVALID, "dtBiasOptional shape must be [HV*K] or [HV, K].");
        }
    } else {
        CHECK_COND(!safeGate, ACLNN_ERR_PARAM_INVALID,
                   "safeGate only takes effect when useGateInKernel is true.");
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnKdaGateCumsumGetWorkspaceSize(
    const aclTensor *g,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    int64_t chunkSize,
    bool useGateInKernel,
    bool safeGate,
    double lowerBound,
    const char *layout,
    const aclTensor *gkOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnKdaGateCumsum, DFX_IN(g, aLogOptional, dtBiasOptional, cuSeqlensOptional), DFX_OUT(gkOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(KdaGateCheckParams(g, aLogOptional, dtBiasOptional, cuSeqlensOptional, chunkSize, useGateInKernel,
                                 safeGate, lowerBound, layout, gkOut) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaGateDataContiguous(g, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaGateDataContiguous(aLogOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(KdaGateDataContiguous(dtBiasOptional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto result = l0op::KdaGateCumsum(g, aLogOptional, dtBiasOptional, cuSeqlensOptional, chunkSize,
                                      useGateInKernel, safeGate, lowerBound, layout, gkOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKdaGateCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnKdaGateCumsum);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
