/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file aclnn_recurrent_kda.cpp
 * \brief
 */
#include "aclnn_recurrent_kda.h"
#include "recurrent_kda.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
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
constexpr size_t DIM0 = 0;
constexpr size_t DIM1 = 1;
constexpr size_t DIM2 = 2;
constexpr size_t DIM3 = 3;

enum class RecurrentKdaLayout {
    BSND,
    TND,
};

struct RecurrentKdaParams {
    const aclTensor *query = nullptr;
    const aclTensor *key = nullptr;
    const aclTensor *value = nullptr;
    const aclTensor *gate = nullptr;
    const aclTensor *beta = nullptr;
    aclTensor *stateRef = nullptr;
    const aclTensor *cuSeqlens = nullptr;
    const aclTensor *ssmStateIndicesOptional = nullptr;
    const aclTensor *aLogOptional = nullptr;
    const aclTensor *dtBiasOptional = nullptr;
    const aclTensor *numAcceptedTokensOptional = nullptr;
    const char *layout = "BSND";
    double scale = 1.0;
    bool outputFinalState = false;
    bool useQkL2normInKernel = false;
    bool useGateInKernel = false;
    bool useBetaSigmoidInKernel = false;
    bool allowNegEigval = false;
    bool safeGate = false;
    double lowerBound = -5.0;
    bool stateVFirst = true;
    const aclTensor *out = nullptr;
};

static const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16,
                                                                             op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> GATE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                            op::DataType::DT_BF16,
                                                                            op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> F32_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> INT_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32,
                                                                          op::DataType::DT_INT64};

size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

int64_t Dim(const aclTensor *tensor, size_t idx)
{
    return tensor->GetViewShape().GetDim(idx);
}

bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    if (Rank(lhs) != Rank(rhs)) {
        return false;
    }
    for (size_t i = 0; i < Rank(lhs); ++i) {
        if (Dim(lhs, i) != Dim(rhs, i)) {
            return false;
        }
    }
    return true;
}

bool ParseLayout(const char *layout, RecurrentKdaLayout &parsed)
{
    if (layout == nullptr || std::strcmp(layout, "BSND") == 0) {
        parsed = RecurrentKdaLayout::BSND;
        return true;
    }
    if (std::strcmp(layout, "TND") == 0) {
        parsed = RecurrentKdaLayout::TND;
        return true;
    }
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: layout must be BSND or TND.");
    return false;
}

bool CheckCuSeqlensShape(const aclTensor *cuSeqlens, const char *opName)
{
    if (Rank(cuSeqlens) != 1 || Dim(cuSeqlens, DIM0) < 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s: cu_seqlens must be a 1D tensor with at least two elements.", opName);
        return false;
    }
    return true;
}

bool CheckShape(const RecurrentKdaParams &params, RecurrentKdaLayout layout)
{
    if (!SameShape(params.query, params.key)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: query and key must have identical shape.");
        return false;
    }
    int64_t totalTokens = 0;
    int64_t denseSeqLen = 0;
    int64_t batch = 1;
    int64_t h = 0;
    int64_t hv = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
    if (layout == RecurrentKdaLayout::TND) {
        if (Rank(params.query) != 3 || Rank(params.value) != 3 || Rank(params.gate) != 3 || Rank(params.beta) != 2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "npu_recurrent_kda: TND expects q/k [T,H,K], v [T,HV,V], g [T,HV,K], beta [T,HV].");
            return false;
        }
        totalTokens = Dim(params.query, DIM0);
        denseSeqLen = totalTokens;
        h = Dim(params.query, DIM1);
        kDim = Dim(params.query, DIM2);
        hv = Dim(params.value, DIM1);
        vDim = Dim(params.value, DIM2);
        if (Dim(params.value, DIM0) != totalTokens || Dim(params.gate, DIM0) != totalTokens ||
            Dim(params.beta, DIM0) != totalTokens || Dim(params.gate, DIM1) != hv ||
            Dim(params.beta, DIM1) != hv || Dim(params.gate, DIM2) != kDim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: TND shape mismatch.");
            return false;
        }
    } else {
        if (Rank(params.query) != 4 || Rank(params.value) != 4 || Rank(params.gate) != 4 || Rank(params.beta) != 3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "npu_recurrent_kda: BSND expects q/k [B,T,H,K], v [B,T,HV,V], g [B,T,HV,K], beta [B,T,HV].");
            return false;
        }
        batch = Dim(params.query, DIM0);
        denseSeqLen = Dim(params.query, DIM1);
        totalTokens = batch * denseSeqLen;
        h = Dim(params.query, DIM2);
        kDim = Dim(params.query, DIM3);
        hv = Dim(params.value, DIM2);
        vDim = Dim(params.value, DIM3);
        if (Dim(params.value, DIM0) != batch || Dim(params.value, DIM1) != denseSeqLen ||
            Dim(params.gate, DIM0) != batch || Dim(params.gate, DIM1) != denseSeqLen ||
            Dim(params.gate, DIM2) != hv || Dim(params.gate, DIM3) != kDim ||
            Dim(params.beta, DIM0) != batch || Dim(params.beta, DIM1) != denseSeqLen ||
            Dim(params.beta, DIM2) != hv) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: BSND shape mismatch.");
            return false;
        }
    }
    if (h <= 0 || hv <= 0 || kDim <= 0 || vDim <= 0 || totalTokens <= 0 || denseSeqLen <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: all shape dimensions must be positive.");
        return false;
    }
    if (hv % h != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: HV must be divisible by H.");
        return false;
    }
    if (kDim != 128 || (vDim != 128 && vDim != 256)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "npu_recurrent_kda: K/V currently support only K=128,V=128 or K=128,V=256, but K=%ld,V=%ld.",
                kDim, vDim);
        return false;
    }
    if (!CheckCuSeqlensShape(params.cuSeqlens, "npu_recurrent_kda")) {
        return false;
    }
    int64_t seqNum = Dim(params.cuSeqlens, DIM0) - 1;
    if (!params.stateVFirst) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: state_v_first=false is not supported.");
        return false;
    }
    if (Rank(params.stateRef) != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "npu_recurrent_kda: state must be rank 4 [state_capacity, HV, V, K].");
        return false;
    }
    int64_t stateCapacity = Dim(params.stateRef, DIM0);
    if (stateCapacity <= 0 ||
        Dim(params.stateRef, DIM1) != hv || Dim(params.stateRef, DIM2) != vDim ||
        Dim(params.stateRef, DIM3) != kDim ||
        (params.ssmStateIndicesOptional == nullptr && stateCapacity != seqNum)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "npu_recurrent_kda: state must be [state_capacity, HV, V, K]; without ssm_state_indices, "
                "state_capacity must equal seq_num.");
        return false;
    }
    if (!SameShape(params.out, params.value)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: out output shape must match value.");
        return false;
    }
    if (params.ssmStateIndicesOptional != nullptr) {
        size_t rank = Rank(params.ssmStateIndicesOptional);
        bool packed1d = rank == 1 && Dim(params.ssmStateIndicesOptional, DIM0) >= totalTokens;
        bool speculative2d = rank == 2 && Dim(params.ssmStateIndicesOptional, DIM0) == seqNum &&
                             Dim(params.ssmStateIndicesOptional, DIM1) > 0;
        if (!packed1d && !speculative2d) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "npu_recurrent_kda: ssm_state_indices must be packed [T] or speculative [seq_num,max_step].");
            return false;
        }
    }
    if (params.numAcceptedTokensOptional != nullptr &&
        (Rank(params.numAcceptedTokensOptional) != 1 || Dim(params.numAcceptedTokensOptional, DIM0) != seqNum)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: num_accepted_tokens length must equal sequence number.");
        return false;
    }
    if (params.useGateInKernel && params.aLogOptional == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: A_log is required when use_gate_in_kernel=True.");
        return false;
    }
    if (!params.useGateInKernel && (params.safeGate || params.aLogOptional != nullptr || params.dtBiasOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "npu_recurrent_kda: A_log, dt_bias and safe_gate require use_gate_in_kernel=True.");
        return false;
    }
    if (params.aLogOptional != nullptr && (Rank(params.aLogOptional) != 1 || Dim(params.aLogOptional, DIM0) != hv)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: A_log must be float32 with shape [HV].");
        return false;
    }
    if (params.dtBiasOptional != nullptr) {
        bool dtBiasOk = (Rank(params.dtBiasOptional) == 1 && Dim(params.dtBiasOptional, DIM0) == hv * kDim) ||
                        (Rank(params.dtBiasOptional) == 2 && Dim(params.dtBiasOptional, DIM0) == hv &&
                         Dim(params.dtBiasOptional, DIM1) == kDim);
        if (!dtBiasOk) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: dt_bias must be float32 with shape [HV*K] or [HV,K].");
            return false;
        }
    }
    if (params.safeGate && (params.lowerBound < -5.0 || params.lowerBound >= 0.0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "npu_recurrent_kda: lower_bound must be in [-5, 0) when safe_gate=True.");
        return false;
    }
    return true;
}

bool CheckNotNull(const RecurrentKdaParams &params)
{
    OP_CHECK_NULL(params.query, return false);
    OP_CHECK_NULL(params.key, return false);
    OP_CHECK_NULL(params.value, return false);
    OP_CHECK_NULL(params.gate, return false);
    OP_CHECK_NULL(params.beta, return false);
    OP_CHECK_NULL(params.stateRef, return false);
    OP_CHECK_NULL(params.cuSeqlens, return false);
    OP_CHECK_NULL(params.out, return false);
    return true;
}

bool CheckDtypeValid(const RecurrentKdaParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.gate, GATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.beta, GATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.stateRef, STATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.cuSeqlens, INT_TYPE_SUPPORT_LIST, return false);
    if (params.ssmStateIndicesOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.ssmStateIndicesOptional, INT_TYPE_SUPPORT_LIST, return false);
    }
    if (params.aLogOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.aLogOptional, F32_TYPE_SUPPORT_LIST, return false);
    }
    if (params.dtBiasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.dtBiasOptional, F32_TYPE_SUPPORT_LIST, return false);
    }
    if (params.numAcceptedTokensOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.numAcceptedTokensOptional, INT_TYPE_SUPPORT_LIST, return false);
    }
    return true;
}

aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

void SetTensorOriginalShape(const aclTensor *tensor)
{
    if (tensor != nullptr) {
        tensor->SetOriginalShape(tensor->GetViewShape());
    }
}

void SetInputOriginalShape(RecurrentKdaParams &params)
{
    SetTensorOriginalShape(params.query);
    SetTensorOriginalShape(params.key);
    SetTensorOriginalShape(params.value);
    SetTensorOriginalShape(params.gate);
    SetTensorOriginalShape(params.beta);
    SetTensorOriginalShape(params.stateRef);
    SetTensorOriginalShape(params.cuSeqlens);
    SetTensorOriginalShape(params.ssmStateIndicesOptional);
    SetTensorOriginalShape(params.aLogOptional);
    SetTensorOriginalShape(params.dtBiasOptional);
    SetTensorOriginalShape(params.numAcceptedTokensOptional);
}

const aclTensor *MaybeCast(const aclTensor *tensor, DataType dataType, aclOpExecutor *executor)
{
    if (tensor == nullptr || tensor->GetDataType() == dataType) {
        return tensor;
    }
    return l0op::Cast(tensor, dataType, executor);
}

aclnnStatus PreProcess(RecurrentKdaParams &params, aclOpExecutor *executor)
{
    bool hasSsmStateIndices = params.ssmStateIndicesOptional != nullptr;
    bool hasNumAcceptedTokens = params.numAcceptedTokensOptional != nullptr;
    SetInputOriginalShape(params);
    CHECK_RET(DataContiguous(params.query, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.key, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.value, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.gate, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.beta, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.cuSeqlens, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.ssmStateIndicesOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.aLogOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.dtBiasOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(DataContiguous(params.numAcceptedTokensOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    params.gate = MaybeCast(params.gate, DataType::DT_FLOAT, executor);
    params.beta = MaybeCast(params.beta, DataType::DT_FLOAT, executor);
    params.cuSeqlens = MaybeCast(params.cuSeqlens, DataType::DT_INT64, executor);
    params.ssmStateIndicesOptional = MaybeCast(params.ssmStateIndicesOptional, DataType::DT_INT64, executor);
    params.numAcceptedTokensOptional = MaybeCast(params.numAcceptedTokensOptional, DataType::DT_INT64, executor);
    CHECK_RET(params.gate != nullptr && params.beta != nullptr && params.cuSeqlens != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(!hasSsmStateIndices || params.ssmStateIndicesOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(!hasNumAcceptedTokens || params.numAcceptedTokensOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (params.ssmStateIndicesOptional == nullptr && params.numAcceptedTokensOptional != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "npu_recurrent_kda: num_accepted_tokens requires ssm_state_indices.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnRecurrentKdaGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *gate,
    const aclTensor *beta,
    aclTensor *stateRef,
    const aclTensor *cuSeqlens,
    const aclTensor *ssmStateIndicesOptional,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *numAcceptedTokensOptional,
    const char *layout,
    double scale,
    bool outputFinalState,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool stateVFirst,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnRecurrentKda,
                   DFX_IN(query, key, value, gate, beta, stateRef, cuSeqlens, ssmStateIndicesOptional,
                          aLogOptional, dtBiasOptional, numAcceptedTokensOptional, layout, scale, outputFinalState,
                          useQkL2normInKernel, useGateInKernel, useBetaSigmoidInKernel, allowNegEigval, safeGate,
                          lowerBound, stateVFirst),
                   DFX_OUT(out, stateRef));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    RecurrentKdaParams params{query, key, value, gate, beta, stateRef, cuSeqlens, ssmStateIndicesOptional,
                              aLogOptional, dtBiasOptional, numAcceptedTokensOptional, layout, scale,
                              outputFinalState, useQkL2normInKernel, useGateInKernel, useBetaSigmoidInKernel,
                              allowNegEigval, safeGate, lowerBound, stateVFirst, out};

    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    RecurrentKdaLayout parsedLayout = RecurrentKdaLayout::BSND;
    CHECK_RET(ParseLayout(params.layout, parsedLayout), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params, parsedLayout), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(PreProcess(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclTensor *stateForKernel = params.stateRef;
    bool stateNeedViewCopy = !IsContiguous(params.stateRef);
    if (stateNeedViewCopy) {
        const aclTensor *contiguousState = params.stateRef;
        CHECK_RET(DataContiguous(contiguousState, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        stateForKernel = const_cast<aclTensor *>(contiguousState);
    }

    auto result = l0op::RecurrentKda(params.query, params.key, params.value, params.gate, params.beta,
                                     stateForKernel, params.cuSeqlens, params.ssmStateIndicesOptional,
                                     params.aLogOptional, params.dtBiasOptional, params.numAcceptedTokensOptional,
                                     params.layout, params.scale, params.useQkL2normInKernel,
                                     params.useGateInKernel, params.useBetaSigmoidInKernel, params.allowNegEigval,
                                     params.safeGate, params.lowerBound, params.stateVFirst, params.out, executorPtr);
    CHECK_RET(result[0] != nullptr && result[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (stateNeedViewCopy) {
        CHECK_RET(l0op::ViewCopy(result[1], params.stateRef, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRecurrentKda(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRecurrentKda);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
