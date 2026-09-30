/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "aclnn_chunk_kda_fwd.h"
#include "chunk_kda_fwd.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t MAX_KDA_K_DIM = 256;
constexpr int64_t MAX_KDA_HEAD_NUM = 128;
constexpr int64_t KDA_STAGE_FULL = -1;
constexpr int64_t KDA_STAGE_GATE_PREPARE = 0;
constexpr int64_t KDA_STAGE_COUNT = 4;

constexpr int64_t MAX_KDA_VARLEN_SEQUENCES = 1024;

enum class KdaFwdLayout {
    BSND,
    BNSD,
    TND,
    NTD,
};

struct ChunkKdaFwdParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *aLogOptional = nullptr;
    const aclTensor *dtBiasOptional = nullptr;
    const aclTensor *initialStateOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    const char *layout = "BSND";
    double scale = 1.0;
    int64_t chunkSize = 64;
    bool safeGate = false;
    double lowerBound = -5.0;
    bool useGateInKernel = false;
    bool stateVFirst = false;
    const aclTensor *attnOut = nullptr;
    const aclTensor *finalStateOut = nullptr;
    const aclTensor *gkOut = nullptr;
    const aclTensor *aqkOut = nullptr;
    const aclTensor *akkOut = nullptr;
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
    const aclTensor *qgOut = nullptr;
    const aclTensor *kgOut = nullptr;
    const aclTensor *vNewOut = nullptr;
    const aclTensor *hOut = nullptr;
};

struct KdaShapeInfo {
    bool isRank3 = false;
    int64_t batch = 0;
    int64_t seqlen = 0;
    int64_t hNum = 0;
    int64_t hvNum = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
    int64_t seqNum = 0;
    int64_t totalChunks = 0;
};

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

const aclTensor *Transpose(const aclTensor *input, const std::vector<int64_t> &perm, aclOpExecutor *executor)
{
    const aclIntArray *permArray = executor->AllocIntArray(perm.data(), perm.size());
    if (permArray == nullptr) {
        return nullptr;
    }
    const aclTensor *transposed = l0op::Transpose(input, permArray, executor);
    if (transposed == nullptr) {
        return nullptr;
    }
    const aclTensor *materialized = l0op::Contiguous(transposed, executor);
    if (materialized == nullptr) {
        return nullptr;
    }
    const aclTensor *reshaped =
        l0op::Reshape(materialized, transposed->GetViewShape(), executor);
    if (reshaped == nullptr) {
        return nullptr;
    }
    reshaped->SetStorageShape(reshaped->GetViewShape());
    reshaped->SetOriginalShape(reshaped->GetViewShape());
    return reshaped;
}

const aclTensor *TransposeLastTwo(const aclTensor *input, aclOpExecutor *executor)
{
    const size_t rank = input->GetViewShape().GetDimNum();
    std::vector<int64_t> perm(rank);
    for (size_t idx = 0; idx < rank; ++idx) {
        perm[idx] = static_cast<int64_t>(idx);
    }
    std::swap(perm[rank - 2], perm[rank - 1]);
    return Transpose(input, perm, executor);
}

static int64_t Dim(const aclTensor *tensor, size_t idx)
{
    return tensor->GetViewShape().GetDim(idx);
}

static size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

static bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    if (lhs == nullptr || rhs == nullptr || Rank(lhs) != Rank(rhs)) {
        return false;
    }
    for (size_t idx = 0; idx < Rank(lhs); ++idx) {
        if (Dim(lhs, idx) != Dim(rhs, idx)) {
            return false;
        }
    }
    return true;
}

bool HasShape(const aclTensor *tensor, std::initializer_list<int64_t> expected)
{
    if (tensor == nullptr || Rank(tensor) != expected.size()) {
        return false;
    }
    size_t idx = 0;
    for (int64_t dim : expected) {
        if (Dim(tensor, idx++) != dim) {
            return false;
        }
    }
    return true;
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParseLayout(const char *layout, KdaFwdLayout &parsed)
{
    CHECK_COND(layout != nullptr, ACLNN_ERR_PARAM_INVALID,
               "layout must be uppercase and one of BSND, BNSD, TND or NTD.");
    if (std::strcmp(layout, "BSND") == 0) {
        parsed = KdaFwdLayout::BSND;
    } else if (std::strcmp(layout, "BNSD") == 0) {
        parsed = KdaFwdLayout::BNSD;
    } else if (std::strcmp(layout, "TND") == 0) {
        parsed = KdaFwdLayout::TND;
    } else if (std::strcmp(layout, "NTD") == 0) {
        parsed = KdaFwdLayout::NTD;
    } else {
        CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
                   "layout must be uppercase and one of BSND, BNSD, TND or NTD.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckCuSeqlens(const aclIntArray *cuSeqlens, int64_t seqlen)
{
    if (cuSeqlens == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(cuSeqlens->Size() >= 2, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional must contain at least [0, total_tokens].");
    CHECK_COND((*cuSeqlens)[0] == 0, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional[0] must be 0.");
    CHECK_COND((*cuSeqlens)[cuSeqlens->Size() - 1] == seqlen, ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional last element must equal the sequence length.");
    for (size_t idx = 0; idx + 1 < cuSeqlens->Size(); ++idx) {
        CHECK_COND((*cuSeqlens)[idx] <= (*cuSeqlens)[idx + 1], ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional must be nondecreasing.");
    }
    return ACLNN_SUCCESS;
}

int64_t CountChunks(const aclIntArray *cuSeqlens, int64_t seqlen, int64_t chunkSize)
{
    if (cuSeqlens == nullptr) {
        return (seqlen + chunkSize - 1) / chunkSize;
    }
    int64_t chunks = 0;
    for (size_t idx = 0; idx + 1 < cuSeqlens->Size(); ++idx) {
        chunks += ((*cuSeqlens)[idx + 1] - (*cuSeqlens)[idx] + chunkSize - 1) / chunkSize;
    }
    return chunks;
}

aclnnStatus CheckChunkIndices(const aclIntArray *chunkIndices, const aclIntArray *cuSeqlens,
                              int64_t totalChunks, int64_t chunkSize)
{
    if (chunkIndices == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(cuSeqlens != nullptr, ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional requires cuSeqlensOptional.");
    CHECK_COND(chunkIndices->Size() == static_cast<size_t>(totalChunks) * 2,
               ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional must contain exactly one (seq_id, chunk_id) pair per chunk.");
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens->Size(); ++seq) {
        const int64_t length = (*cuSeqlens)[seq + 1] - (*cuSeqlens)[seq];
        const int64_t chunks = (length + chunkSize - 1) / chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            CHECK_COND((*chunkIndices)[offset] == static_cast<int64_t>(seq) &&
                           (*chunkIndices)[offset + 1] == chunk,
                       ACLNN_ERR_PARAM_INVALID,
                       "chunkIndicesOptional must use canonical sequence-major chunk order.");
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveShapeInfo(const ChunkKdaFwdParams &params, KdaFwdLayout layout, KdaShapeInfo &info)
{
    info.isRank3 = layout == KdaFwdLayout::TND || layout == KdaFwdLayout::NTD;
    const size_t tensorRank = info.isRank3 ? 3 : 4;
    const size_t betaRank = info.isRank3 ? 2 : 3;
    CHECK_COND(Rank(params.q) == tensorRank && Rank(params.k) == tensorRank &&
                   Rank(params.v) == tensorRank && Rank(params.g) == tensorRank &&
                   Rank(params.beta) == betaRank,
               ACLNN_ERR_PARAM_INVALID,
               "q/k/v/g and beta ranks must match layout: rank3/rank2 for TND/NTD, rank4/rank3 for BSND/BNSD.");
    CHECK_COND(SameShape(params.q, params.k), ACLNN_ERR_PARAM_INVALID,
               "q and k must have identical shape.");

    if (layout == KdaFwdLayout::TND) {
        info.batch = 1;
        info.seqlen = Dim(params.q, 0);
        info.hNum = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.hvNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 2);
        CHECK_COND(HasShape(params.v, {info.seqlen, info.hvNum, info.vDim}) &&
                       HasShape(params.g, {info.seqlen, info.hvNum, info.kDim}) &&
                       HasShape(params.beta, {info.seqlen, info.hvNum}),
                   ACLNN_ERR_PARAM_INVALID, "TND expects v/g/beta as [T,HV,V], [T,HV,K], [T,HV].");
    } else if (layout == KdaFwdLayout::NTD) {
        info.batch = 1;
        info.hNum = Dim(params.q, 0);
        info.seqlen = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.hvNum = Dim(params.v, 0);
        info.vDim = Dim(params.v, 2);
        CHECK_COND(HasShape(params.v, {info.hvNum, info.seqlen, info.vDim}) &&
                       HasShape(params.g, {info.hvNum, info.seqlen, info.kDim}) &&
                       HasShape(params.beta, {info.hvNum, info.seqlen}),
                   ACLNN_ERR_PARAM_INVALID, "NTD expects v/g/beta as [HV,T,V], [HV,T,K], [HV,T].");
    } else if (layout == KdaFwdLayout::BSND) {
        info.batch = Dim(params.q, 0);
        info.seqlen = Dim(params.q, 1);
        info.hNum = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hvNum = Dim(params.v, 2);
        info.vDim = Dim(params.v, 3);
        CHECK_COND(HasShape(params.v, {info.batch, info.seqlen, info.hvNum, info.vDim}) &&
                       HasShape(params.g, {info.batch, info.seqlen, info.hvNum, info.kDim}) &&
                       HasShape(params.beta, {info.batch, info.seqlen, info.hvNum}),
                   ACLNN_ERR_PARAM_INVALID,
                   "BSND expects v/g/beta as [B,T,HV,V], [B,T,HV,K], [B,T,HV].");
    } else {
        info.batch = Dim(params.q, 0);
        info.hNum = Dim(params.q, 1);
        info.seqlen = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hvNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 3);
        CHECK_COND(HasShape(params.v, {info.batch, info.hvNum, info.seqlen, info.vDim}) &&
                       HasShape(params.g, {info.batch, info.hvNum, info.seqlen, info.kDim}) &&
                       HasShape(params.beta, {info.batch, info.hvNum, info.seqlen}),
                   ACLNN_ERR_PARAM_INVALID,
                   "BNSD expects v/g/beta as [B,HV,T,V], [B,HV,T,K], [B,HV,T].");
    }
    info.seqNum = params.cuSeqlensOptional == nullptr
                      ? info.batch
                      : static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;
    info.totalChunks = CountChunks(params.cuSeqlensOptional, info.seqlen, params.chunkSize);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtypes(const ChunkKdaFwdParams &params)
{
    const DataType dataType = params.q->GetDataType();
    CHECK_COND((dataType == DataType::DT_FLOAT16 || dataType == DataType::DT_BF16) &&
                   params.k->GetDataType() == dataType && params.v->GetDataType() == dataType,
               ACLNN_ERR_PARAM_INVALID, "q, k and v must use the same float16 or bfloat16 dtype.");
    const DataType gateType = params.g->GetDataType();
    CHECK_COND(gateType == DataType::DT_FLOAT || gateType == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "g must be float32 or bfloat16.");
    const DataType betaType = params.beta->GetDataType();
    CHECK_COND(betaType == DataType::DT_FLOAT || betaType == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "beta must be float32 or bfloat16.");
    if (params.aLogOptional != nullptr) {
        CHECK_COND(params.aLogOptional->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "aLogOptional must be float32.");
    }
    if (params.dtBiasOptional != nullptr) {
        CHECK_COND(params.dtBiasOptional->GetDataType() == DataType::DT_FLOAT, ACLNN_ERR_PARAM_INVALID,
                   "dtBiasOptional must be float32.");
    }
    if (params.initialStateOptional != nullptr) {
        CHECK_COND(params.initialStateOptional->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "initialStateOptional must be float32.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckStateShape(const aclTensor *state, const char *name, const KdaShapeInfo &info, bool stateVFirst)
{
    if (state == nullptr) {
        return ACLNN_SUCCESS;
    }
    const bool valid = stateVFirst
                           ? HasShape(state, {info.seqNum, info.hvNum, info.vDim, info.kDim})
                           : HasShape(state, {info.seqNum, info.hvNum, info.kDim, info.vDim});
    CHECK_COND(valid, ACLNN_ERR_PARAM_INVALID,
               "%s must be [N,HV,K,V] when stateVFirst=false and [N,HV,V,K] otherwise.", name);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputShapes(const ChunkKdaFwdParams &params, const KdaShapeInfo &info)
{
    const DataType dataType = params.q->GetDataType();
    const bool attnShapeValid = info.isRank3
                                    ? HasShape(params.attnOut, {info.seqlen, info.hvNum, info.vDim})
                                    : HasShape(params.attnOut,
                                               {info.batch, info.seqlen, info.hvNum, info.vDim});
    CHECK_COND(attnShapeValid && params.attnOut->GetDataType() == dataType,
               ACLNN_ERR_PARAM_INVALID,
               "attnOut must match q dtype and use fixed sequence-major TND/BSND layout.");
    if (params.gkOut != nullptr) {
        const bool valid = info.isRank3
                               ? HasShape(params.gkOut, {info.hvNum, info.seqlen, info.kDim})
                               : HasShape(params.gkOut,
                                          {info.batch, info.hvNum, info.seqlen, info.kDim});
        CHECK_COND(valid && params.gkOut->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "gkOut must be float32 in fixed head-major NTD/BNSD layout.");
    }
    const aclTensor *matrixOutputs[] = {params.aqkOut, params.akkOut};
    for (const aclTensor *output : matrixOutputs) {
        const bool valid = info.isRank3
                               ? HasShape(output, {info.hvNum, info.seqlen, params.chunkSize})
                               : HasShape(output,
                                          {info.batch, info.hvNum, info.seqlen, params.chunkSize});
        CHECK_COND(valid && output->GetDataType() == dataType, ACLNN_ERR_PARAM_INVALID,
                   "Aqk/Akk must match q dtype and use fixed head-major NTD/BNSD layout.");
    }
    const aclTensor *kOutputs[] = {params.wOut, params.qgOut, params.kgOut};
    for (const aclTensor *output : kOutputs) {
        if (output == nullptr) {
            continue;
        }
        const bool valid = info.isRank3
                               ? HasShape(output, {info.hvNum, info.seqlen, info.kDim})
                               : HasShape(output,
                                          {info.batch, info.hvNum, info.seqlen, info.kDim});
        CHECK_COND(valid && output->GetDataType() == dataType, ACLNN_ERR_PARAM_INVALID,
                   "w/qg/kg must match q dtype and use fixed head-major NTD/BNSD layout.");
    }
    const aclTensor *vOutputs[] = {params.uOut, params.vNewOut};
    for (const aclTensor *output : vOutputs) {
        if (output == nullptr) {
            continue;
        }
        const bool valid = info.isRank3
                               ? HasShape(output, {info.hvNum, info.seqlen, info.vDim})
                               : HasShape(output,
                                          {info.batch, info.hvNum, info.seqlen, info.vDim});
        CHECK_COND(valid && output->GetDataType() == dataType, ACLNN_ERR_PARAM_INVALID,
                   "u/vNew must match q dtype and use fixed head-major NTD/BNSD layout.");
    }
    if (params.hOut != nullptr) {
        const bool valid = params.stateVFirst
                               ? (info.isRank3
                                      ? HasShape(params.hOut,
                                                 {info.totalChunks, info.hvNum, info.vDim, info.kDim})
                                      : HasShape(params.hOut,
                                                 {info.batch, info.totalChunks, info.hvNum,
                                                  info.vDim, info.kDim}))
                               : (info.isRank3
                                      ? HasShape(params.hOut,
                                                 {info.totalChunks, info.hvNum, info.kDim, info.vDim})
                                      : HasShape(params.hOut,
                                                 {info.batch, info.totalChunks, info.hvNum,
                                                  info.kDim, info.vDim}));
        CHECK_COND(valid && params.hOut->GetDataType() == dataType, ACLNN_ERR_PARAM_INVALID,
                   "hOut must match q dtype, use fixed sequence-major layout, and follow stateVFirst.");
    }
    CHECK_RET(CheckStateShape(params.finalStateOut, "finalStateOut", info, params.stateVFirst) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    if (params.finalStateOut != nullptr) {
        CHECK_COND(params.finalStateOut->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "finalStateOut must be float32.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const ChunkKdaFwdParams &params, KdaFwdLayout &layout, KdaShapeInfo &info)
{
    CHECK_COND(params.q != nullptr && params.k != nullptr && params.v != nullptr &&
                   params.g != nullptr && params.beta != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "q, k, v, g and beta must not be nullptr.");
    CHECK_COND(params.attnOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "attnOut must not be nullptr.");
    CHECK_COND(params.aqkOut != nullptr && params.akkOut != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "aqkOut and akkOut must not be nullptr.");
    CHECK_COND(params.chunkSize == 64 || params.chunkSize == 128, ACLNN_ERR_PARAM_INVALID,
               "chunkSize must be 64 or 128.");
    CHECK_RET(ParseLayout(params.layout, layout) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ResolveShapeInfo(params, layout, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(info.hNum > 0 && info.hvNum >= info.hNum && info.hvNum % info.hNum == 0,
               ACLNN_ERR_PARAM_INVALID,
               "H and HV must be positive, HV must be greater than or equal to H, and HV must be divisible by H.");
    CHECK_COND(info.hNum <= MAX_KDA_HEAD_NUM && info.hvNum <= MAX_KDA_HEAD_NUM,
               ACLNN_ERR_PARAM_INVALID, "H and HV must be less than or equal to 128.");
    CHECK_COND(info.kDim >= 16 && info.kDim <= MAX_KDA_K_DIM && info.kDim % 16 == 0 &&
                   info.vDim >= 16 && info.vDim <= 256 && info.vDim % 16 == 0,
               ACLNN_ERR_PARAM_INVALID,
               "K/V must be multiples of 16, K must be <=256, and V must be <=256.");
    CHECK_RET(CheckDtypes(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckCuSeqlens(params.cuSeqlensOptional, info.seqlen) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(params.cuSeqlensOptional == nullptr || info.isRank3 || info.batch == 1,
               ACLNN_ERR_PARAM_INVALID,
               "rank4 varlen input with cuSeqlensOptional requires B=1.");
    CHECK_COND(params.cuSeqlensOptional == nullptr || info.seqNum <= MAX_KDA_VARLEN_SEQUENCES,
               ACLNN_ERR_PARAM_INVALID, "varlen input supports at most 1024 sequences.");
    CHECK_RET(CheckChunkIndices(params.chunkIndicesOptional, params.cuSeqlensOptional,
                                info.totalChunks, params.chunkSize) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckStateShape(params.initialStateOptional, "initialStateOptional", info,
                              params.stateVFirst) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    if (params.useGateInKernel) {
        CHECK_COND(params.aLogOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "aLogOptional is required when useGateInKernel is true.");
        CHECK_COND(HasShape(params.aLogOptional, {info.hvNum}), ACLNN_ERR_PARAM_INVALID,
                   "aLogOptional must have shape [HV].");
        if (params.dtBiasOptional != nullptr) {
            CHECK_COND(HasShape(params.dtBiasOptional, {info.hvNum * info.kDim}),
                       ACLNN_ERR_PARAM_INVALID, "dtBiasOptional must have shape [HV*K].");
        }
        if (params.safeGate) {
            CHECK_COND(params.lowerBound >= -5.0 && params.lowerBound < 0.0,
                       ACLNN_ERR_PARAM_INVALID,
                       "lowerBound must be in [-5, 0) when safeGate is true.");
        }
    }
    CHECK_RET(CheckOutputShapes(params, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus ContiguousInputs(ChunkKdaFwdParams &params, aclOpExecutor *executor)
{
    CHECK_RET(MakeContiguous(params.q, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.k, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.v, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.g, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.beta, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.aLogOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.dtBiasOptional, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(MakeContiguous(params.initialStateOptional, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

const aclTensor *AllocTensor(aclOpExecutor *executor, const op::Shape &shape, DataType dtype)
{
    return executor->AllocTensor(shape, dtype, Format::FORMAT_ND);
}

const aclTensor *AsRank4(const aclTensor *tensor, const op::Shape &shape, aclOpExecutor *executor)
{
    return l0op::Reshape(tensor, shape, executor);
}

bool IsAscend950()
{
    const char *socName = aclrtGetSocName();
    return socName != nullptr && std::strstr(socName, "Ascend950") != nullptr;
}
} // namespace

aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    double lowerBound,
    bool useGateInKernel,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *vNewOut,
    const aclTensor *hOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkKdaFwdParams params{
        q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional,
        cuSeqlensOptional, chunkIndicesOptional, layout, scale, chunkSize,
        safeGate, lowerBound, useGateInKernel, stateVFirst, attnOut, finalStateOut, gkOut,
        aqkOut, akkOut, wOut, uOut, qgOut, kgOut, vNewOut, hOut};
    L2_DFX_PHASE_1(
        aclnnChunkKdaFwd,
        DFX_IN(q, k, v, g, beta, aLogOptional, dtBiasOptional, initialStateOptional,
               cuSeqlensOptional, chunkIndicesOptional, layout, scale, chunkSize,
               safeGate, lowerBound, useGateInKernel, stateVFirst),
        DFX_OUT(attnOut, finalStateOut, gkOut, aqkOut, akkOut, wOut, uOut,
                qgOut, kgOut, vNewOut, hOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    KdaFwdLayout parsedLayout = KdaFwdLayout::BSND;
    KdaShapeInfo info;
    CHECK_RET(CheckParams(params, parsedLayout, info) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ContiguousInputs(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    const aclTensor *qHead = params.q;
    const aclTensor *kHead = params.k;
    const aclTensor *vHead = params.v;
    const aclTensor *gHead = params.g;
    const aclTensor *betaHead = params.beta;
    if (parsedLayout == KdaFwdLayout::BSND) {
        // beta is small and its scalar-per-token rows are not DMA friendly in
        // sequence-major form. Keep only this lightweight transpose.
        betaHead = Transpose(params.beta, {0, 2, 1}, executorPtr);
    } else if (parsedLayout == KdaFwdLayout::TND) {
        qHead = Transpose(params.q, {1, 0, 2}, executorPtr);
        kHead = Transpose(params.k, {1, 0, 2}, executorPtr);
        vHead = Transpose(params.v, {1, 0, 2}, executorPtr);
        gHead = Transpose(params.g, {1, 0, 2}, executorPtr);
        betaHead = Transpose(params.beta, {1, 0}, executorPtr);
    }
    CHECK_RET(qHead != nullptr && kHead != nullptr && vHead != nullptr &&
                  gHead != nullptr && betaHead != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    if (info.isRank3) {
        qHead = AsRank4(qHead, MakeShape({1, info.hNum, info.seqlen, info.kDim}), executorPtr);
        kHead = AsRank4(kHead, MakeShape({1, info.hNum, info.seqlen, info.kDim}), executorPtr);
        vHead = AsRank4(vHead, MakeShape({1, info.hvNum, info.seqlen, info.vDim}), executorPtr);
        gHead = AsRank4(gHead, MakeShape({1, info.hvNum, info.seqlen, info.kDim}), executorPtr);
        betaHead = AsRank4(betaHead, MakeShape({1, info.hvNum, info.seqlen}), executorPtr);
        CHECK_RET(qHead != nullptr && kHead != nullptr && vHead != nullptr &&
                      gHead != nullptr && betaHead != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    const op::Shape gkShape4 = MakeShape({info.batch, info.hvNum, info.seqlen, info.kDim});
    const op::Shape matrixShape4 =
        MakeShape({info.batch, info.hvNum, info.seqlen, params.chunkSize});
    const op::Shape kShape4 = MakeShape({info.batch, info.hvNum, info.seqlen, info.kDim});
    const op::Shape vShape4 = MakeShape({info.batch, info.hvNum, info.seqlen, info.vDim});
    const op::Shape hShape5 =
        MakeShape({info.batch, info.hvNum, info.totalChunks, info.kDim, info.vDim});
    const op::Shape hExportShape5 =
        params.stateVFirst
            ? MakeShape({info.batch, info.totalChunks, info.hvNum, info.vDim, info.kDim})
            : MakeShape({info.batch, info.totalChunks, info.hvNum, info.kDim, info.vDim});
    const op::Shape stateShape4 =
        MakeShape({info.seqNum, info.hvNum, info.kDim, info.vDim});
    const op::Shape placeholderShape = MakeShape({1});
    const bool useDenseA5FastPath =
        params.cuSeqlensOptional == nullptr && params.q->GetDataType() == DataType::DT_BF16 &&
        params.chunkSize == 64 && info.kDim == 128 && info.vDim == 128 &&
        info.seqlen % params.chunkSize == 0;
    const bool splitStages =
        IsAscend950() && info.totalChunks > 1 && !useDenseA5FastPath;

    const aclTensor *gkCompute = params.gkOut;
    if (gkCompute != nullptr && info.isRank3) {
        gkCompute = AsRank4(gkCompute, gkShape4, executorPtr);
    }
    if (gkCompute == nullptr) {
        gkCompute = AllocTensor(
            executorPtr, splitStages ? gkShape4 : placeholderShape,
            DataType::DT_FLOAT);
    }
    CHECK_RET(gkCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *aqkCompute = params.aqkOut;
    const aclTensor *akkCompute = params.akkOut;
    const aclTensor *wExport = params.wOut;
    const aclTensor *uExport = params.uOut;
    const aclTensor *qgExport = params.qgOut;
    const aclTensor *kgExport = params.kgOut;
    const aclTensor *vNewExport = params.vNewOut;
    const aclTensor *hExport = params.hOut;
    if (info.isRank3) {
        aqkCompute = aqkCompute == nullptr ? nullptr : AsRank4(aqkCompute, matrixShape4, executorPtr);
        akkCompute = akkCompute == nullptr ? nullptr : AsRank4(akkCompute, matrixShape4, executorPtr);
        wExport = wExport == nullptr ? nullptr : AsRank4(wExport, kShape4, executorPtr);
        uExport = uExport == nullptr ? nullptr : AsRank4(uExport, vShape4, executorPtr);
        qgExport = qgExport == nullptr ? nullptr : AsRank4(qgExport, kShape4, executorPtr);
        kgExport = kgExport == nullptr ? nullptr : AsRank4(kgExport, kShape4, executorPtr);
        vNewExport = vNewExport == nullptr ? nullptr : AsRank4(vNewExport, vShape4, executorPtr);
        if (hExport != nullptr) {
            hExport = AsRank4(hExport, hExportShape5, executorPtr);
        }
    }
    CHECK_RET((params.wOut == nullptr || wExport != nullptr) &&
                  (params.uOut == nullptr || uExport != nullptr) &&
                  (params.qgOut == nullptr || qgExport != nullptr) &&
                  (params.kgOut == nullptr || kgExport != nullptr) &&
                  (params.vNewOut == nullptr || vNewExport != nullptr) &&
                  (params.hOut == nullptr || hExport != nullptr),
              ACLNN_ERR_INNER_NULLPTR);
    if (aqkCompute == nullptr) {
        aqkCompute = AllocTensor(executorPtr, matrixShape4, params.q->GetDataType());
    }
    if (akkCompute == nullptr) {
        akkCompute = AllocTensor(executorPtr, matrixShape4, params.q->GetDataType());
    }
    const aclTensor *wCompute = wExport == nullptr
        ? AllocTensor(executorPtr, splitStages ? kShape4 : placeholderShape,
                      params.q->GetDataType())
        : wExport;
    const aclTensor *uCompute = uExport == nullptr
        ? AllocTensor(executorPtr, splitStages ? vShape4 : placeholderShape,
                      params.q->GetDataType())
        : uExport;
    const aclTensor *qgCompute = qgExport == nullptr
        ? AllocTensor(executorPtr, splitStages ? kShape4 : placeholderShape,
                      params.q->GetDataType())
        : qgExport;
    const aclTensor *kgCompute = kgExport == nullptr
        ? AllocTensor(executorPtr, splitStages ? kShape4 : placeholderShape,
                      params.q->GetDataType())
        : kgExport;
    const aclTensor *vNewCompute = vNewExport == nullptr
        ? AllocTensor(executorPtr, splitStages ? vShape4 : placeholderShape,
                      params.q->GetDataType())
        : vNewExport;
    const aclTensor *hCompute = AllocTensor(
        executorPtr, hExport == nullptr && !splitStages ? placeholderShape : hShape5,
        params.q->GetDataType());
    CHECK_RET(aqkCompute != nullptr && akkCompute != nullptr && wCompute != nullptr &&
                  uCompute != nullptr && qgCompute != nullptr && kgCompute != nullptr &&
                  vNewCompute != nullptr && hCompute != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *initialStateCompute = params.initialStateOptional;
    if (params.stateVFirst && initialStateCompute != nullptr) {
        initialStateCompute = TransposeLastTwo(initialStateCompute, executorPtr);
        CHECK_RET(initialStateCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const bool outputFinalState = params.finalStateOut != nullptr;
    const aclTensor *finalStateCompute = AllocTensor(
        executorPtr, outputFinalState || splitStages ? stateShape4 : placeholderShape,
        DataType::DT_FLOAT);
    CHECK_RET(finalStateCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *attnCompute = params.attnOut;
    if (info.isRank3) {
        attnCompute = AsRank4(
            params.attnOut, MakeShape({1, info.seqlen, info.hvNum, info.vDim}), executorPtr);
        CHECK_RET(attnCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor *qgScaledCompute = AllocTensor(
        executorPtr, splitStages ? kShape4 : placeholderShape,
        params.q->GetDataType());
    const aclTensor *uSeedCompute = AllocTensor(
        executorPtr, splitStages ? vShape4 : placeholderShape,
        params.q->GetDataType());
    CHECK_RET(qgScaledCompute != nullptr && uSeedCompute != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto launchStage = [&](int64_t stage) {
        return l0op::KdaChunkForward(
            qHead, kHead, vHead, gHead, betaHead, params.aLogOptional,
            params.dtBiasOptional, initialStateCompute, params.cuSeqlensOptional,
            params.chunkIndicesOptional, params.scale, params.chunkSize,
            params.safeGate, parsedLayout == KdaFwdLayout::BSND,
            params.useGateInKernel, params.lowerBound, attnCompute,
            finalStateCompute, gkCompute, aqkCompute, akkCompute, wCompute,
            uCompute, qgCompute, kgCompute, vNewCompute, hCompute,
            qgScaledCompute, uSeedCompute, stage, executorPtr);
    };
    l0op::KdaCoreOutputs result{};
    if (splitStages) {
        // Physical launch boundaries reset the A5 event state between the
        // prepare, post-WU, recurrent, and output pipelines.
        for (int64_t stage = KDA_STAGE_GATE_PREPARE; stage < KDA_STAGE_COUNT;
             ++stage) {
            result = launchStage(stage);
            for (const aclTensor *tensor : result) {
                CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
            }
        }
    } else {
        result = launchStage(KDA_STAGE_FULL);
        for (const aclTensor *tensor : result) {
            CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    if (outputFinalState) {
        const aclTensor *finalStateResult = result[1];
        if (params.stateVFirst) {
            finalStateResult = TransposeLastTwo(finalStateResult, executorPtr);
            CHECK_RET(finalStateResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        CHECK_RET(l0op::ViewCopy(finalStateResult, params.finalStateOut, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }
    if (hExport != nullptr) {
        const std::vector<int64_t> hPerm =
            params.stateVFirst ? std::vector<int64_t>{0, 2, 1, 4, 3}
                               : std::vector<int64_t>{0, 2, 1, 3, 4};
        const aclTensor *hResult = Transpose(result[10], hPerm, executorPtr);
        CHECK_RET(hResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(hResult, hExport, executorPtr) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaFwd(void *workspace, uint64_t workspaceSize,
                             aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaFwd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS,
               ACLNN_ERR_INNER, "ChunkKdaFwd launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
