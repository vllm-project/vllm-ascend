/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_chunk_fwd_h.h"
#include "chunk_fwd_h.h"
#include <dlfcn.h>
#include <new>

#include "aclnn_kernels/contiguous.h"
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

struct ChunkFwdHParams {
    const aclTensor *k = nullptr;
    const aclTensor *w = nullptr;
    const aclTensor *u = nullptr;
    const aclTensor *gOptional = nullptr;
    const aclTensor *gkOptional = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    bool outputFinalState = false;
    int64_t chunkSize = 64;
    bool saveNewValue = true;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    bool useExp2 = false;
    bool stateVFirst = false;
    const aclTensor *hOut = nullptr;
    const aclTensor *vNewOut = nullptr;
    const aclTensor *finalStateOut = nullptr;
};

static aclnnStatus CheckNotNull(ChunkFwdHParams params)
{
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.w != nullptr, ACLNN_ERR_PARAM_NULLPTR, "w must not be nullptr.");
    CHECK_COND(params.u != nullptr, ACLNN_ERR_PARAM_NULLPTR, "u must not be nullptr.");

    CHECK_COND(params.hOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hOut must not be nullptr.");
    CHECK_COND(params.vNewOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "vNewOut must not be nullptr.");
    CHECK_COND(params.outputFinalState == (params.finalStateOut != nullptr), ACLNN_ERR_PARAM_INVALID,
               "outputFinalState must equal the physical presence of finalStateOut.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(ChunkFwdHParams params)
{
    const aclTensor *tensors[] = {
        params.k,
        params.w,
        params.u,
        params.gOptional,
        params.gkOptional,
        params.initialStateOptional,
        params.hOut,
        params.vNewOut,
        params.finalStateOut,
    };
    const char *tensorNames[] = {
        "k",
        "w",
        "u",
        "gOptional",
        "gkOptional",
        "initialStateOptional",
        "hOut",
        "vNewOut",
        "finalStateOut",
    };
    for (size_t idx = 0; idx < sizeof(tensors) / sizeof(tensors[0]); ++idx) {
        if (tensors[idx] == nullptr) {
            continue;
        }
        const auto storageFormat = tensors[idx]->GetStorageFormat();
        const auto viewFormat = tensors[idx]->GetViewFormat();
        CHECK_COND(storageFormat == Format::FORMAT_ND && viewFormat == Format::FORMAT_ND &&
                       !IsPrivateFormat(storageFormat),
                   ACLNN_ERR_PARAM_INVALID,
                   "%s must use non-private ND storage/view format, but got storage=%d and view=%d.",
                   tensorNames[idx], static_cast<int>(storageFormat),
                   static_cast<int>(viewFormat));
    }
    const aclTensor *outputs[] = {params.hOut, params.vNewOut, params.finalStateOut};
    const char *outputNames[] = {"hOut", "vNewOut", "finalStateOut"};
    for (size_t idx = 0; idx < sizeof(outputs) / sizeof(outputs[0]); ++idx) {
        if (outputs[idx] == nullptr) {
            continue;
        }
        CHECK_COND(IsContiguous(outputs[idx]), ACLNN_ERR_PARAM_INVALID,
                   "%s must be contiguous because ChunkFwdH writes it directly.",
                   outputNames[idx]);
    }
    return ACLNN_SUCCESS;
}

static int64_t CountChunks(const aclIntArray *cuSeqlens, int64_t seqlen, int64_t chunkSize)
{
    if (cuSeqlens == nullptr) {
        return (seqlen + chunkSize - 1) / chunkSize;
    }
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens->Size(); ++seq) {
        const int64_t length = (*cuSeqlens)[seq + 1] - (*cuSeqlens)[seq];
        totalChunks += (length + chunkSize - 1) / chunkSize;
    }
    return totalChunks;
}

static aclnnStatus CheckVariableLengthInputs(const ChunkFwdHParams &params)
{
    const auto kShape = params.k->GetViewShape();
    CHECK_COND(kShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
               "k must be rank 4 before validating variable-length metadata.");
    if (params.cuSeqlensOptional == nullptr) {
        CHECK_COND(params.chunkIndicesOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional requires cuSeqlensOptional.");
        return ACLNN_SUCCESS;
    }

    CHECK_COND(kShape.GetDim(0) == 1, ACLNN_ERR_PARAM_INVALID,
               "Variable-length BNSD input requires B=1, but got B=%ld.", kShape.GetDim(0));
    CHECK_COND(params.cuSeqlensOptional->Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional must contain at least [0, total_tokens].");
    CHECK_COND((*params.cuSeqlensOptional)[0] == 0, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional[0] must be 0.");
    CHECK_COND((*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1] == kShape.GetDim(2),
               ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional last element must equal T=%ld.", kShape.GetDim(2));
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        CHECK_COND((*params.cuSeqlensOptional)[seq] < (*params.cuSeqlensOptional)[seq + 1],
                   ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must be strictly increasing; empty sequences are not supported.");
    }

    if (params.chunkIndicesOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const int64_t totalChunks = CountChunks(params.cuSeqlensOptional, kShape.GetDim(2), params.chunkSize);
    CHECK_COND(params.chunkIndicesOptional->Size() == static_cast<size_t>(totalChunks) * 2,
               ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional must contain exactly one (seq_id, chunk_id) pair per chunk.");
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        const int64_t length = (*params.cuSeqlensOptional)[seq + 1] - (*params.cuSeqlensOptional)[seq];
        const int64_t chunks = (length + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            CHECK_COND((*params.chunkIndicesOptional)[offset] == static_cast<int64_t>(seq) &&
                           (*params.chunkIndicesOptional)[offset + 1] == chunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunkIndicesOptional must use canonical sequence-major chunk order.");
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(ChunkFwdHParams params)
{
    auto kShape = params.k->GetViewShape();
    auto wShape = params.w->GetViewShape();
    auto uShape = params.u->GetViewShape();
    CHECK_COND(kShape.GetDimNum() == 4 && wShape.GetDimNum() == 4 && uShape.GetDimNum() == 4,
               ACLNN_ERR_PARAM_INVALID, "k, w and u must be rank-4 BNSD tensors.");
    CHECK_COND(kShape.GetDim(0) > 0 && kShape.GetDim(1) > 0 && kShape.GetDim(2) > 0 &&
                   uShape.GetDim(1) > 0,
               ACLNN_ERR_PARAM_INVALID,
               "B, HK, T and HV must all be positive, but got B=%ld, HK=%ld, T=%ld, HV=%ld.",
               kShape.GetDim(0), kShape.GetDim(1), kShape.GetDim(2), uShape.GetDim(1));
    CHECK_COND(kShape.GetDim(0) == wShape.GetDim(0) && kShape.GetDim(0) == uShape.GetDim(0) &&
                   wShape.GetDim(1) == uShape.GetDim(1) && kShape.GetDim(2) == wShape.GetDim(2) &&
                   kShape.GetDim(2) == uShape.GetDim(2) && kShape.GetDim(3) == wShape.GetDim(3),
               ACLNN_ERR_PARAM_INVALID,
               "k, w and u must match in B/T, w and u must match in HV, and k and w must match in K.");
    CHECK_COND(kShape.GetDim(3) == 128 && uShape.GetDim(3) == 128, ACLNN_ERR_PARAM_INVALID,
               "FwdH requires K=128 and V=128, but got K=%ld and V=%ld.",
               kShape.GetDim(3), uShape.GetDim(3));
    if (params.gOptional != nullptr) {
        CHECK_COND(uShape.GetDim(1) >= kShape.GetDim(1) && uShape.GetDim(1) % kShape.GetDim(1) == 0,
                   ACLNN_ERR_PARAM_INVALID,
                   "g-only requires HV >= HK and HV divisible by HK, but got HK=%ld and HV=%ld.",
                   kShape.GetDim(1), uShape.GetDim(1));
    } else {
        CHECK_COND(kShape.GetDim(1) == uShape.GetDim(1), ACLNN_ERR_PARAM_INVALID,
                   "gk-only requires the prepared kg head count to equal HV, but got kg heads=%ld and HV=%ld.",
                   kShape.GetDim(1), uShape.GetDim(1));
    }
    if (params.gOptional != nullptr) {
        auto gShape = params.gOptional->GetViewShape();
        CHECK_COND(gShape.GetDimNum() == 3 && gShape.GetDim(0) == uShape.GetDim(0) &&
                       gShape.GetDim(1) == uShape.GetDim(1) && gShape.GetDim(2) == uShape.GetDim(2),
                   ACLNN_ERR_PARAM_INVALID, "g must have shape [B, HV, T].");
    }
    const int64_t batch = kShape.GetDim(0);
    const int64_t hv = uShape.GetDim(1);
    const int64_t kDim = kShape.GetDim(3);
    const int64_t vDim = uShape.GetDim(3);
    const int64_t seqNum = params.cuSeqlensOptional == nullptr
                               ? batch
                               : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
    const int64_t totalChunks = CountChunks(params.cuSeqlensOptional, kShape.GetDim(2), params.chunkSize);
    auto hShape = params.hOut->GetViewShape();
    CHECK_COND(hShape.GetDimNum() == 5 && hShape.GetDim(0) == batch && hShape.GetDim(1) == hv &&
                   hShape.GetDim(2) == totalChunks,
               ACLNN_ERR_PARAM_INVALID,
               "hOut must be [B, HV, num_chunks, K, V] (or [B, HV, num_chunks, V, K]), "
               "where num_chunks=%ld.", totalChunks);
    const int64_t hK = params.stateVFirst ? hShape.GetDim(4) : hShape.GetDim(3);
    const int64_t hV = params.stateVFirst ? hShape.GetDim(3) : hShape.GetDim(4);
    CHECK_COND(hK == kDim && hV == vDim, ACLNN_ERR_PARAM_INVALID,
               "hOut state dimensions must be [K, V] when stateVFirst=false and [V, K] otherwise.");
    auto vNewShape = params.vNewOut->GetViewShape();
    CHECK_COND(vNewShape.GetDimNum() == 4 && vNewShape.GetDim(0) == batch &&
                   vNewShape.GetDim(1) == hv && vNewShape.GetDim(2) == kShape.GetDim(2) &&
                   vNewShape.GetDim(3) == vDim,
               ACLNN_ERR_PARAM_INVALID, "vNewOut must have shape [B, HV, T, V].");
    const aclTensor *states[] = {params.initialStateOptional, params.finalStateOut};
    const char *stateNames[] = {"initialStateOptional", "finalStateOut"};
    for (size_t idx = 0; idx < 2; ++idx) {
        if (states[idx] == nullptr) {
            continue;
        }
        auto stateShape = states[idx]->GetViewShape();
        CHECK_COND(stateShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
                   "%s must be rank 4.", stateNames[idx]);
        const int64_t stateK = params.stateVFirst ? stateShape.GetDim(3) : stateShape.GetDim(2);
        const int64_t stateV = params.stateVFirst ? stateShape.GetDim(2) : stateShape.GetDim(3);
        CHECK_COND(stateShape.GetDim(0) == seqNum &&
                       stateShape.GetDim(1) == hv && stateK == kDim && stateV == vDim,
                   ACLNN_ERR_PARAM_INVALID,
                   "%s must be [N, HV, K, V] when stateVFirst=false and [N, HV, V, K] otherwise.",
                   stateNames[idx]);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(ChunkFwdHParams params)
{
    auto inputDtype = params.k->GetDataType();
    CHECK_COND(inputDtype == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
               "k, w and u only support bfloat16.");
    CHECK_COND(params.w->GetDataType() == inputDtype && params.u->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "k, w and u must have the same dtype.");
    CHECK_COND(params.hOut->GetDataType() == inputDtype && params.vNewOut->GetDataType() == inputDtype,
               ACLNN_ERR_PARAM_INVALID, "hOut and vNewOut dtype must match k, w and u.");
    auto gateDtype = params.gOptional != nullptr ? params.gOptional->GetDataType() : params.gkOptional->GetDataType();
    CHECK_COND(gateDtype == DataType::DT_FLOAT || gateDtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "g/gk dtype must be bfloat16 or float32.");
    if (params.initialStateOptional != nullptr) {
        auto stateDtype = params.initialStateOptional->GetDataType();
        CHECK_COND(stateDtype == DataType::DT_BF16 || stateDtype == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "initialStateOptional dtype must be bfloat16 or float32.");
        if (params.finalStateOut != nullptr) {
            CHECK_COND(params.finalStateOut->GetDataType() == stateDtype, ACLNN_ERR_PARAM_INVALID,
                       "finalStateOut dtype must match initialStateOptional dtype.");
        }
    } else if (params.finalStateOut != nullptr) {
        auto stateDtype = params.finalStateOut->GetDataType();
        CHECK_COND(stateDtype == DataType::DT_BF16 || stateDtype == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "finalStateOut dtype must be bfloat16 or float32.");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkFwdHParams &params, aclOpExecutor *executorPtr)
{
    auto status = DataContiguous(params.k, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous(params.w, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = DataContiguous(params.u, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (params.gOptional != nullptr) {
        status = DataContiguous(params.gOptional, executorPtr);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    if (params.gkOptional != nullptr) {
        status = DataContiguous(params.gkOptional, executorPtr);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    if (params.initialStateOptional != nullptr) {
        status = DataContiguous(params.initialStateOptional, executorPtr);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckGateOptionalNonNull(const ChunkFwdHParams &params)
{
    CHECK_COND((params.gOptional != nullptr) != (params.gkOptional != nullptr), ACLNN_ERR_PARAM_INVALID,
               "Exactly one of g and gk must be provided.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckGkParams(const ChunkFwdHParams &params)
{
    if (params.gkOptional != nullptr) {
        auto gkShape = params.gkOptional->GetViewShape();
        CHECK_COND(gkShape.GetDimNum() == 4, ACLNN_ERR_PARAM_INVALID,
                   "gk must have rank 4 when provided, got rank %ld.", gkShape.GetDimNum());
        CHECK_COND(gkShape.GetDim(3) == params.k->GetViewShape().GetDim(3), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[3] (K) must match k.shape[3] (K).");
        CHECK_COND(gkShape.GetDim(2) == params.k->GetViewShape().GetDim(2), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[2] (T) must match k.shape[2] (T).");
        CHECK_COND(gkShape.GetDim(1) == params.u->GetViewShape().GetDim(1), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[1] (HV) must match u.shape[1] (HV).");
        CHECK_COND(gkShape.GetDim(0) == params.k->GetViewShape().GetDim(0), ACLNN_ERR_PARAM_INVALID,
                   "gk.shape[0] (B) must match k.shape[0] (B).");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(ChunkFwdHParams params)
{
    auto status = CheckNotNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGateOptionalNonNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    CHECK_COND(params.chunkSize == 64, ACLNN_ERR_PARAM_INVALID,
               "chunkSize must be 64, but got %ld.", params.chunkSize);
    CHECK_COND(params.saveNewValue, ACLNN_ERR_PARAM_INVALID, "saveNewValue must be true.");
    status = CheckVariableLengthInputs(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckFormat(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckShape(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGkParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckDtype(params);
}

aclnnStatus aclnnChunkFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    bool saveNewValue,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool useExp2,
    bool stateVFirst,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "executor must not be nullptr.");
    ChunkFwdHParams params{k,
                                         w,
                                         u,
                                         gOptional,
                                         gkOptional,
                                         initialStateOptional,
                                         outputFinalState,
                                         chunkSize,
                                         saveNewValue,
                                         cuSeqlensOptional,
                                         chunkIndicesOptional,
                                         useExp2,
                                         stateVFirst,
                                         hOut,
                                         vNewOut,
                                         finalStateOut};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnChunkFwdH,
                   DFX_IN(k, w, u, gOptional, gkOptional, initialStateOptional, outputFinalState, chunkSize,
                          saveNewValue, cuSeqlensOptional, chunkIndicesOptional, useExp2, stateVFirst),
                   DFX_OUT(hOut, vNewOut, finalStateOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = ParamsDataContiguous(params, executorPtr);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    auto result = l0op::ChunkFwdH(
        params.k, params.w, params.u, params.gOptional, params.gkOptional, params.initialStateOptional,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.outputFinalState, params.chunkSize,
        params.saveNewValue, params.useExp2, params.stateVFirst,
        params.hOut, params.vNewOut, params.finalStateOut, executorPtr);
    CHECK_RET(result[0] != nullptr && result[1] != nullptr &&
                  (!outputFinalState || result[2] != nullptr),
              ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult0 = l0op::ViewCopy(result[0], params.hOut, executorPtr);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(result[1], params.vNewOut, executorPtr);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (outputFinalState && params.finalStateOut != nullptr) {
        auto viewCopyResult2 = l0op::ViewCopy(result[2], params.finalStateOut, executorPtr);
        CHECK_RET(viewCopyResult2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}


aclnnStatus aclnnChunkFwdH(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkFwdH);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in ChunkFwdH launch aicore.");
    return ACLNN_SUCCESS;
}


#ifdef __cplusplus
}
#endif
