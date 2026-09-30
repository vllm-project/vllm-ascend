/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "aclnn_chunk_kda_fwd_prepare.h"
#include "chunk_kda_fwd_prepare.h"
#include "../chunk_kda_fwd_prepare_output_mask.h"

#include <cstddef>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {

constexpr int64_t PREPARE_CHUNK_SIZE = 64;
constexpr int64_t PREPARE_K_DIM = 128;
constexpr int64_t PREPARE_V_DIM = 128;

enum class PrepareLayout {
    BNSD,
    BSND,
    NTD,
    TND,
};

struct ChunkKdaFwdPrepareParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    const aclTensor *aLogOptional = nullptr;
    const aclTensor *dtBiasOptional = nullptr;
    const aclIntArray *cuSeqlensOptional = nullptr;
    const aclIntArray *chunkIndicesOptional = nullptr;
    const char *layout = "BNSD";
    double scale = 1.0;
    int64_t chunkSize = PREPARE_CHUNK_SIZE;
    double epsilon = 1.0e-6;
    bool useQkL2normInKernel = false;
    bool useGateInKernel = false;
    bool useBetaSigmoidInKernel = false;
    bool allowNegEigval = false;
    bool safeGate = false;
    double lowerBound = -5.0;
    bool useExp2 = false;
    const aclTensor *gkOut = nullptr;
    const aclTensor *aqkOut = nullptr;
    const aclTensor *akkOut = nullptr;
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
    const aclTensor *qgOut = nullptr;
    const aclTensor *kgOut = nullptr;
    const aclTensor *qgScaledOut = nullptr;
    const aclTensor *qHatOut = nullptr;
    const aclTensor *kHatOut = nullptr;
    const aclTensor *qRstdOut = nullptr;
    const aclTensor *kRstdOut = nullptr;
    const aclTensor *betaEffOut = nullptr;
};

struct PrepareShapeInfo {
    bool packed = false;
    int64_t batch = 0;
    int64_t seqLen = 0;
    int64_t qkHeadNum = 0;
    int64_t valueHeadNum = 0;
    int64_t kDim = 0;
    int64_t vDim = 0;
};

size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

int64_t Dim(const aclTensor *tensor, size_t index)
{
    return tensor->GetViewShape().GetDim(index);
}

bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    if (lhs == nullptr || rhs == nullptr || Rank(lhs) != Rank(rhs)) {
        return false;
    }
    for (size_t dim = 0; dim < Rank(lhs); ++dim) {
        if (Dim(lhs, dim) != Dim(rhs, dim)) {
            return false;
        }
    }
    return true;
}

bool HasShape(const aclTensor *tensor,
              std::initializer_list<int64_t> expected)
{
    if (tensor == nullptr || Rank(tensor) != expected.size()) {
        return false;
    }
    size_t dim = 0;
    for (int64_t value : expected) {
        if (Dim(tensor, dim++) != value) {
            return false;
        }
    }
    return true;
}

aclnnStatus ParseLayout(const char *layout, PrepareLayout &parsed)
{
    CHECK_COND(layout != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "layout 不能为 nullptr。");
    if (std::strcmp(layout, "BNSD") == 0) {
        parsed = PrepareLayout::BNSD;
    } else if (std::strcmp(layout, "BSND") == 0) {
        parsed = PrepareLayout::BSND;
    } else if (std::strcmp(layout, "NTD") == 0) {
        parsed = PrepareLayout::NTD;
    } else if (std::strcmp(layout, "TND") == 0) {
        parsed = PrepareLayout::TND;
    } else {
        CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
                   "layout 必须为大写 BNSD、BSND、NTD 或 TND，当前值=%s。",
                   layout);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckNotNull(const ChunkKdaFwdPrepareParams &params)
{
    const aclTensor *inputs[] = {params.q, params.k, params.v, params.g,
                                 params.beta};
    const char *inputNames[] = {"q", "k", "v", "g", "beta"};
    for (size_t index = 0; index < sizeof(inputs) / sizeof(inputs[0]); ++index) {
        CHECK_COND(inputs[index] != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "%s 不能为 nullptr。", inputNames[index]);
    }

    const aclTensor *outputs[] = {
        params.gkOut, params.aqkOut, params.wOut,
        params.uOut, params.kgOut, params.qgScaledOut};
    const char *outputNames[] = {
        "gkOut", "aqkOut", "wOut", "uOut", "kgOut", "qgScaledOut"};
    for (size_t index = 0; index < sizeof(outputs) / sizeof(outputs[0]); ++index) {
        CHECK_COND(outputs[index] != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "%s 是必选公开输出，不能为 nullptr。", outputNames[index]);
    }
    return ACLNN_SUCCESS;
}

uint32_t GetOutputMask(const ChunkKdaFwdPrepareParams &params)
{
    const aclTensor *outputs[] = {
        params.gkOut,       params.aqkOut,  params.akkOut,
        params.wOut,        params.uOut,    params.qgOut,
        params.kgOut,       params.qgScaledOut,
        params.qHatOut,     params.kHatOut, params.qRstdOut,
        params.kRstdOut,    params.betaEffOut};
    uint32_t outputMask = 0;
    for (size_t index = 0; index < sizeof(outputs) / sizeof(outputs[0]); ++index) {
        if (outputs[index] != nullptr) {
            outputMask |= 1U << index;
        }
    }
    return outputMask;
}

int64_t GetOutputMode(const ChunkKdaFwdPrepareParams &params)
{
    const uint32_t outputMask = GetOutputMask(params);
    if (outputMask == optiling::PREPARE_REQUIRED_OUTPUT_MASK) {
        return optiling::PREPARE_OUTPUT_MODE_NONE;
    }
    if (outputMask == optiling::PREPARE_FORWARD_OUTPUT_MASK) {
        return optiling::PREPARE_OUTPUT_MODE_FORWARD;
    }
    if (outputMask == optiling::PREPARE_RECOMPUTE_OUTPUT_MASK) {
        return optiling::PREPARE_OUTPUT_MODE_RECOMPUTE;
    }
    if (outputMask == optiling::PREPARE_SAVE_OUTPUT_MASK) {
        return optiling::PREPARE_OUTPUT_MODE_SAVE;
    }
    return -1;
}

aclnnStatus CheckOutputMode(const ChunkKdaFwdPrepareParams &params)
{
    CHECK_COND(GetOutputMode(params) >= 0, ACLNN_ERR_PARAM_INVALID,
               "输出 nullptr 组合只支持 none/forward/recompute/save 四档，当前 outputMask=0x%x。",
               GetOutputMask(params));
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFormat(const ChunkKdaFwdPrepareParams &params)
{
    const aclTensor *tensors[] = {
        params.q,          params.k,        params.v,
        params.g,          params.beta,     params.aLogOptional,
        params.dtBiasOptional,
        params.gkOut,      params.aqkOut,   params.akkOut,
        params.wOut,       params.uOut,     params.qgOut,
        params.kgOut,      params.qgScaledOut,
        params.qHatOut,    params.kHatOut,  params.qRstdOut,
        params.kRstdOut,   params.betaEffOut};
    const char *names[] = {
        "q",          "k",        "v",          "g",          "beta",
        "aLogOptional", "dtBiasOptional",
        "gkOut",      "aqkOut",   "akkOut",     "wOut",       "uOut",
        "qgOut",      "kgOut",    "qgScaledOut", "qHatOut",   "kHatOut",
        "qRstdOut",   "kRstdOut", "betaEffOut"};
    for (size_t index = 0; index < sizeof(tensors) / sizeof(tensors[0]); ++index) {
        if (tensors[index] == nullptr) {
            continue;
        }
        const auto storageFormat = tensors[index]->GetStorageFormat();
        const auto viewFormat = tensors[index]->GetViewFormat();
        // 只拦私有（分形）格式：ND/NCHW/NCL/NHWC 等非私有拼写在逻辑维度上等价，
        // 张量本身已由 CheckContiguous 保证连续，不能因为拼写不同就拒绝调用方。
        CHECK_COND(!IsPrivateFormat(storageFormat) && !IsPrivateFormat(viewFormat),
                   ACLNN_ERR_PARAM_INVALID,
                   "%s must use a non-private storage/view format "
                   "(ND/NCHW/NCL/NHWC are all accepted); got storage=%d, view=%d.",
                   names[index], static_cast<int>(storageFormat),
                   static_cast<int>(viewFormat));
    }

    const aclTensor *outputs[] = {
        params.gkOut,       params.aqkOut,  params.akkOut,
        params.wOut,        params.uOut,    params.qgOut,
        params.kgOut,       params.qgScaledOut,
        params.qHatOut,     params.kHatOut, params.qRstdOut,
        params.kRstdOut,    params.betaEffOut};
    const char *outputNames[] = {
        "gkOut",       "aqkOut",  "akkOut",      "wOut",      "uOut",
        "qgOut",       "kgOut",   "qgScaledOut", "qHatOut",   "kHatOut",
        "qRstdOut",    "kRstdOut", "betaEffOut"};
    for (size_t index = 0; index < sizeof(outputs) / sizeof(outputs[0]); ++index) {
        if (outputs[index] == nullptr) {
            continue;
        }
        CHECK_COND(IsContiguous(outputs[index]), ACLNN_ERR_PARAM_INVALID,
                   "%s 由 kernel 直接写入，必须连续。", outputNames[index]);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckDtype(const ChunkKdaFwdPrepareParams &params)
{
    CHECK_COND(params.q->GetDataType() == DataType::DT_BF16 &&
                   params.k->GetDataType() == DataType::DT_BF16 &&
                   params.v->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "q/k/v 只支持 BF16。");
    const auto gateDtype = params.g->GetDataType();
    const auto betaDtype = params.beta->GetDataType();
    CHECK_COND(gateDtype == DataType::DT_BF16 ||
                   gateDtype == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "g 只支持 BF16 或 FP32。");
    CHECK_COND(betaDtype == DataType::DT_BF16 ||
                   betaDtype == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "beta 只支持 BF16 或 FP32。");
    if (params.aLogOptional != nullptr) {
        CHECK_COND(params.aLogOptional->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "aLogOptional 只支持 FP32。");
    }
    if (params.dtBiasOptional != nullptr) {
        CHECK_COND(params.dtBiasOptional->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID, "dtBiasOptional 只支持 FP32。");
    }

    const aclTensor *bf16Outputs[] = {
        params.aqkOut, params.akkOut, params.wOut, params.uOut,
        params.qgOut, params.kgOut, params.qgScaledOut,
        params.qHatOut, params.kHatOut};
    for (const aclTensor *output : bf16Outputs) {
        if (output == nullptr) {
            continue;
        }
        CHECK_COND(output->GetDataType() == DataType::DT_BF16,
                   ACLNN_ERR_PARAM_INVALID,
                   "Aqk/Akk/w/u/qg/kg/qgScaled/qHat/kHat 输出必须为 BF16。");
    }
    const aclTensor *fp32Outputs[] = {
        params.gkOut, params.qRstdOut, params.kRstdOut, params.betaEffOut};
    for (const aclTensor *output : fp32Outputs) {
        if (output == nullptr) {
            continue;
        }
        CHECK_COND(output->GetDataType() == DataType::DT_FLOAT,
                   ACLNN_ERR_PARAM_INVALID,
                   "gk/qRstd/kRstd/betaEff 输出必须为 FP32。");
    }
    return ACLNN_SUCCESS;
}

bool FitsUint32(int64_t value)
{
    return value > 0 &&
        static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max();
}

bool FitsDmaSourceStride(int64_t headNum, uint64_t rowBytes)
{
    return headNum > 0 && rowBytes > 0 &&
        static_cast<uint64_t>(headNum - 1) <=
            std::numeric_limits<uint32_t>::max() / rowBytes;
}

aclnnStatus CheckSequenceMajorDmaStride(
    const ChunkKdaFwdPrepareParams &params, PrepareLayout layout,
    const PrepareShapeInfo &info)
{
    if (layout != PrepareLayout::BSND && layout != PrepareLayout::TND) {
        return ACLNN_SUCCESS;
    }

    const uint64_t gateElementBytes =
        params.g->GetDataType() == DataType::DT_FLOAT ? sizeof(float) : 2U;
    const uint64_t betaElementBytes =
        params.beta->GetDataType() == DataType::DT_FLOAT ? sizeof(float) : 2U;
    const bool strideValid =
        FitsDmaSourceStride(info.qkHeadNum, PREPARE_K_DIM * 2U) &&
        FitsDmaSourceStride(info.valueHeadNum, PREPARE_V_DIM * 2U) &&
        FitsDmaSourceStride(info.valueHeadNum,
                            PREPARE_K_DIM * gateElementBytes) &&
        FitsDmaSourceStride(info.valueHeadNum, betaElementBytes);
    CHECK_COND(strideValid, ACLNN_ERR_PARAM_INVALID,
               "BSND/TND 输入的跨 head DMA stride 超过 uint32 范围，当前 HK=%ld,HV=%ld。",
               info.qkHeadNum, info.valueHeadNum);
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveShape(const ChunkKdaFwdPrepareParams &params,
                         PrepareLayout layout, PrepareShapeInfo &info)
{
    CHECK_COND(SameShape(params.q, params.k), ACLNN_ERR_PARAM_INVALID,
               "q 与 k shape 必须完全一致。");
    info.packed = layout == PrepareLayout::NTD || layout == PrepareLayout::TND;
    const size_t matrixRank = info.packed ? 3 : 4;
    const size_t scalarRank = info.packed ? 2 : 3;
    CHECK_COND(Rank(params.q) == matrixRank && Rank(params.v) == matrixRank &&
                   Rank(params.g) == matrixRank && Rank(params.beta) == scalarRank,
               ACLNN_ERR_PARAM_INVALID,
               "BNSD/BSND 要求 q/k/v/g 为 rank-4、beta 为 rank-3；"
               "NTD/TND 要求 q/k/v/g 为 rank-3、beta 为 rank-2。");

    if (layout == PrepareLayout::TND) {
        info.batch = 1;
        info.seqLen = Dim(params.q, 0);
        info.qkHeadNum = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.valueHeadNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 2);
        CHECK_COND(
            HasShape(params.v, {info.seqLen, info.valueHeadNum, info.vDim}) &&
                HasShape(params.g,
                         {info.seqLen, info.valueHeadNum, info.kDim}) &&
                HasShape(params.beta, {info.seqLen, info.valueHeadNum}),
            ACLNN_ERR_PARAM_INVALID,
            "TND 要求 v/g/beta 分别为 [T,HV,V]、[T,HV,K]、[T,HV]。");
    } else if (layout == PrepareLayout::NTD) {
        info.batch = 1;
        info.qkHeadNum = Dim(params.q, 0);
        info.seqLen = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.valueHeadNum = Dim(params.v, 0);
        info.vDim = Dim(params.v, 2);
        CHECK_COND(
            HasShape(params.v, {info.valueHeadNum, info.seqLen, info.vDim}) &&
                HasShape(params.g,
                         {info.valueHeadNum, info.seqLen, info.kDim}) &&
                HasShape(params.beta, {info.valueHeadNum, info.seqLen}),
            ACLNN_ERR_PARAM_INVALID,
            "NTD 要求 v/g/beta 分别为 [HV,T,V]、[HV,T,K]、[HV,T]。");
    } else if (layout == PrepareLayout::BSND) {
        info.batch = Dim(params.q, 0);
        info.seqLen = Dim(params.q, 1);
        info.qkHeadNum = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.valueHeadNum = Dim(params.v, 2);
        info.vDim = Dim(params.v, 3);
        CHECK_COND(
            HasShape(params.v,
                     {info.batch, info.seqLen, info.valueHeadNum, info.vDim}) &&
                HasShape(params.g,
                         {info.batch, info.seqLen, info.valueHeadNum,
                          info.kDim}) &&
                HasShape(params.beta,
                         {info.batch, info.seqLen, info.valueHeadNum}),
            ACLNN_ERR_PARAM_INVALID,
            "BSND 要求 v/g/beta 分别为 [B,T,HV,V]、[B,T,HV,K]、[B,T,HV]。");
    } else {
        info.batch = Dim(params.q, 0);
        info.qkHeadNum = Dim(params.q, 1);
        info.seqLen = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.valueHeadNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 3);
        CHECK_COND(
            HasShape(params.v,
                     {info.batch, info.valueHeadNum, info.seqLen, info.vDim}) &&
                HasShape(params.g,
                         {info.batch, info.valueHeadNum, info.seqLen,
                          info.kDim}) &&
                HasShape(params.beta,
                         {info.batch, info.valueHeadNum, info.seqLen}),
            ACLNN_ERR_PARAM_INVALID,
            "BNSD 要求 v/g/beta 分别为 [B,HV,T,V]、[B,HV,T,K]、[B,HV,T]。");
    }

    CHECK_COND(FitsUint32(info.batch) && FitsUint32(info.seqLen) &&
                   FitsUint32(info.qkHeadNum) &&
                   FitsUint32(info.valueHeadNum),
               ACLNN_ERR_PARAM_INVALID,
               "B/T/HK/HV 必须为正数且可由 uint32 表示，当前 B=%ld,T=%ld,HK=%ld,HV=%ld。",
               info.batch, info.seqLen, info.qkHeadNum, info.valueHeadNum);
    CHECK_COND(info.kDim == PREPARE_K_DIM && info.vDim == PREPARE_V_DIM,
               ACLNN_ERR_PARAM_INVALID,
               "当前实现要求 K=V=128，当前 K=%ld,V=%ld。", info.kDim,
               info.vDim);
    CHECK_COND(info.valueHeadNum >= info.qkHeadNum &&
                   info.valueHeadNum % info.qkHeadNum == 0,
               ACLNN_ERR_PARAM_INVALID,
               "GVA 要求 HV>=HK 且 HV 能被 HK 整除，当前 HK=%ld,HV=%ld。",
               info.qkHeadNum, info.valueHeadNum);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckModesAndOptionalInputs(
    const ChunkKdaFwdPrepareParams &params, const PrepareShapeInfo &info)
{
    CHECK_COND(params.chunkSize == PREPARE_CHUNK_SIZE,
               ACLNN_ERR_PARAM_INVALID,
               "chunkSize 仅支持 64，当前值=%ld。", params.chunkSize);
    // Tiling 和 kernel 均以 FP32 保存这些属性，必须按实际落盘精度校验。
    const float scaleFp32 = static_cast<float>(params.scale);
    const float epsilonFp32 = static_cast<float>(params.epsilon);
    const float lowerBoundFp32 = static_cast<float>(params.lowerBound);
    CHECK_COND(std::isfinite(scaleFp32) && std::isfinite(epsilonFp32) &&
                   epsilonFp32 > 0.0F && std::isfinite(lowerBoundFp32),
               ACLNN_ERR_PARAM_INVALID,
               "scale/lowerBound 转为 FP32 后必须为有限数，epsilon 转为 FP32 后必须为正有限数。");
    CHECK_COND(!params.allowNegEigval || params.useBetaSigmoidInKernel,
               ACLNN_ERR_PARAM_INVALID,
               "allowNegEigval=true 要求 useBetaSigmoidInKernel=true。");

    if (!params.useGateInKernel) {
        CHECK_COND(params.aLogOptional == nullptr &&
                       params.dtBiasOptional == nullptr && !params.safeGate,
                   ACLNN_ERR_PARAM_INVALID,
                   "useGateInKernel=false 时 aLogOptional、dtBiasOptional 必须为空且 safeGate 必须为 false。");
        return ACLNN_SUCCESS;
    }

    CHECK_COND(params.aLogOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "useGateInKernel=true 时 aLogOptional 必传。");
    CHECK_COND(HasShape(params.aLogOptional, {info.valueHeadNum}),
               ACLNN_ERR_PARAM_INVALID,
               "aLogOptional 必须为 [HV]，当前 HV=%ld。",
               info.valueHeadNum);
    if (params.dtBiasOptional != nullptr) {
        CHECK_COND(
            HasShape(params.dtBiasOptional,
                     {info.valueHeadNum * info.kDim}),
            ACLNN_ERR_PARAM_INVALID,
            "dtBiasOptional 必须为展平的 [HV*K]，当前 HV=%ld,K=%ld。",
            info.valueHeadNum, info.kDim);
    }
    if (params.safeGate) {
        CHECK_COND(lowerBoundFp32 >= -5.0F && lowerBoundFp32 < 0.0F,
                   ACLNN_ERR_PARAM_INVALID,
                   "safeGate=true 时 lowerBound 必须在 [-5,0) 内，当前值=%f。",
                   params.lowerBound);
    }
    return ACLNN_SUCCESS;
}

int64_t CountChunks(const aclIntArray *cuSeqlens, int64_t seqLen,
                    int64_t chunkSize)
{
    if (cuSeqlens == nullptr) {
        return (seqLen + chunkSize - 1) / chunkSize;
    }
    int64_t totalChunks = 0;
    for (size_t seq = 0; seq + 1 < cuSeqlens->Size(); ++seq) {
        const int64_t length = (*cuSeqlens)[seq + 1] - (*cuSeqlens)[seq];
        totalChunks += (length + chunkSize - 1) / chunkSize;
    }
    return totalChunks;
}

aclnnStatus CheckVariableLengthInputs(
    const ChunkKdaFwdPrepareParams &params, const PrepareShapeInfo &info)
{
    if (params.cuSeqlensOptional == nullptr) {
        CHECK_COND(params.chunkIndicesOptional == nullptr,
                   ACLNN_ERR_PARAM_INVALID,
                   "chunkIndicesOptional 要求同时传入 cuSeqlensOptional。");
        return ACLNN_SUCCESS;
    }
    CHECK_COND(info.packed || info.batch == 1, ACLNN_ERR_PARAM_INVALID,
               "rank-4 变长输入要求 B=1，当前 B=%ld。", info.batch);
    CHECK_COND(params.cuSeqlensOptional->Size() >= 2,
               ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional 至少包含 [0,total_tokens]。");
    CHECK_COND((*params.cuSeqlensOptional)[0] == 0,
               ACLNN_ERR_PARAM_INVALID,
               "cuSeqlensOptional[0] 必须为 0。");
    CHECK_COND(
        (*params.cuSeqlensOptional)[params.cuSeqlensOptional->Size() - 1] ==
            info.seqLen,
        ACLNN_ERR_PARAM_INVALID,
        "cuSeqlensOptional 末元素必须等于 T=%ld。", info.seqLen);
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        CHECK_COND((*params.cuSeqlensOptional)[seq] <=
                       (*params.cuSeqlensOptional)[seq + 1],
                   ACLNN_ERR_PARAM_INVALID,
                   "cuSeqlensOptional 必须非递减，错误位置=%zu。", seq);
    }

    if (params.chunkIndicesOptional == nullptr) {
        return ACLNN_SUCCESS;
    }
    const int64_t totalChunks = CountChunks(
        params.cuSeqlensOptional, info.seqLen, params.chunkSize);
    CHECK_COND(params.chunkIndicesOptional->Size() ==
                   static_cast<size_t>(totalChunks) * 2,
               ACLNN_ERR_PARAM_INVALID,
               "chunkIndicesOptional 必须为 [2*total_chunks]，total_chunks=%ld。",
               totalChunks);
    size_t offset = 0;
    for (size_t seq = 0; seq + 1 < params.cuSeqlensOptional->Size(); ++seq) {
        const int64_t length = (*params.cuSeqlensOptional)[seq + 1] -
            (*params.cuSeqlensOptional)[seq];
        const int64_t chunks =
            (length + params.chunkSize - 1) / params.chunkSize;
        for (int64_t chunk = 0; chunk < chunks; ++chunk) {
            CHECK_COND(
                (*params.chunkIndicesOptional)[offset] ==
                        static_cast<int64_t>(seq) &&
                    (*params.chunkIndicesOptional)[offset + 1] == chunk,
                ACLNN_ERR_PARAM_INVALID,
                "chunkIndicesOptional 必须按 sequence-major 顺序保存规范的 (seq_id,local_chunk_id)。");
            offset += 2;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputShapes(const ChunkKdaFwdPrepareParams &params,
                              const PrepareShapeInfo &info)
{
    const auto valueMatrixValid = [&](const aclTensor *tensor,
                                      int64_t dimension) {
        return info.packed
            ? HasShape(tensor,
                       {info.valueHeadNum, info.seqLen, dimension})
            : HasShape(tensor, {info.batch, info.valueHeadNum, info.seqLen,
                                dimension});
    };
    const auto qkMatrixValid = [&](const aclTensor *tensor) {
        return info.packed
            ? HasShape(tensor, {info.qkHeadNum, info.seqLen, info.kDim})
            : HasShape(tensor, {info.batch, info.qkHeadNum, info.seqLen,
                                info.kDim});
    };
    const auto valueScalarValid = [&](const aclTensor *tensor) {
        return info.packed
            ? HasShape(tensor, {info.valueHeadNum, info.seqLen})
            : HasShape(tensor,
                       {info.batch, info.valueHeadNum, info.seqLen});
    };
    const auto qkScalarValid = [&](const aclTensor *tensor) {
        return info.packed
            ? HasShape(tensor, {info.qkHeadNum, info.seqLen})
            : HasShape(tensor,
                       {info.batch, info.qkHeadNum, info.seqLen});
    };

    CHECK_COND(valueMatrixValid(params.gkOut, info.kDim),
               ACLNN_ERR_PARAM_INVALID,
               "gkOut 必须使用固定 head-major [B,HV,T,K]/[HV,T,K] shape。");
    CHECK_COND(valueMatrixValid(params.aqkOut, params.chunkSize),
               ACLNN_ERR_PARAM_INVALID,
               "aqkOut 必须使用固定 head-major [B,HV,T,64]/[HV,T,64] shape。");
    CHECK_COND(params.akkOut == nullptr ||
                   valueMatrixValid(params.akkOut, params.chunkSize),
               ACLNN_ERR_PARAM_INVALID,
               "非空 akkOut 必须使用固定 head-major [B,HV,T,64]/[HV,T,64] shape。");
    CHECK_COND(valueMatrixValid(params.wOut, info.kDim) &&
                   valueMatrixValid(params.kgOut, info.kDim) &&
                   valueMatrixValid(params.qgScaledOut, info.kDim),
               ACLNN_ERR_PARAM_INVALID,
               "wOut/kgOut/qgScaledOut 必须使用固定 head-major [B,HV,T,K]/[HV,T,K] shape。");
    CHECK_COND(params.qgOut == nullptr ||
                   valueMatrixValid(params.qgOut, info.kDim),
               ACLNN_ERR_PARAM_INVALID,
               "非空 qgOut 必须使用固定 head-major [B,HV,T,K]/[HV,T,K] shape。");
    CHECK_COND(valueMatrixValid(params.uOut, info.vDim),
               ACLNN_ERR_PARAM_INVALID,
               "uOut 必须使用固定 head-major [B,HV,T,V]/[HV,T,V] shape。");
    CHECK_COND((params.qHatOut == nullptr || qkMatrixValid(params.qHatOut)) &&
                   (params.kHatOut == nullptr || qkMatrixValid(params.kHatOut)),
                ACLNN_ERR_PARAM_INVALID,
                "非空 qHatOut/kHatOut 必须使用固定 head-major [B,HK,T,K]/[HK,T,K] shape。");
    CHECK_COND((params.qRstdOut == nullptr || qkScalarValid(params.qRstdOut)) &&
                   (params.kRstdOut == nullptr || qkScalarValid(params.kRstdOut)),
                ACLNN_ERR_PARAM_INVALID,
                "非空 qRstdOut/kRstdOut 必须使用固定 head-major [B,HK,T]/[HK,T] shape。");
    CHECK_COND(params.betaEffOut == nullptr ||
                   valueScalarValid(params.betaEffOut),
                ACLNN_ERR_PARAM_INVALID,
                "非空 betaEffOut 必须使用固定 head-major [B,HV,T]/[HV,T] shape。");
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const ChunkKdaFwdPrepareParams &params)
{
    aclnnStatus status = CheckNotNull(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckOutputMode(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    PrepareLayout layout;
    status = ParseLayout(params.layout, layout);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    PrepareShapeInfo info;
    status = ResolveShape(params, layout, info);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckModesAndOptionalInputs(params, info);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckVariableLengthInputs(params, info);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckFormat(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckDtype(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSequenceMajorDmaStride(params, layout, info);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckOutputShapes(params, info);
}

aclnnStatus MakeContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    if (tensor == nullptr || IsContiguous(tensor)) {
        return ACLNN_SUCCESS;
    }
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus MakeInputsContiguous(ChunkKdaFwdPrepareParams &params,
                                 aclOpExecutor *executor)
{
    const aclTensor **inputs[] = {
        &params.q, &params.k, &params.v, &params.g, &params.beta,
        &params.aLogOptional, &params.dtBiasOptional};
    for (const aclTensor **input : inputs) {
        const aclnnStatus status = MakeContiguous(*input, executor);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnChunkKdaFwdPrepareGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    double epsilon,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool useExp2,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *qgScaledOut,
    const aclTensor *qHatOut,
    const aclTensor *kHatOut,
    const aclTensor *qRstdOut,
    const aclTensor *kRstdOut,
    const aclTensor *betaEffOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "workspaceSize 不能为 nullptr。");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "executor 不能为 nullptr。");

    ChunkKdaFwdPrepareParams params{
        q,          k,         v,          g,          beta,
        aLogOptional, dtBiasOptional,
        cuSeqlensOptional, chunkIndicesOptional,
        layout,     scale,     chunkSize,  epsilon,
        useQkL2normInKernel, useGateInKernel, useBetaSigmoidInKernel,
        allowNegEigval, safeGate, lowerBound, useExp2,
        gkOut,      aqkOut,    akkOut,     wOut,       uOut,
        qgOut,      kgOut,     qgScaledOut,
        qHatOut,    kHatOut,   qRstdOut,   kRstdOut,   betaEffOut};

    L2_DFX_PHASE_1(
        aclnnChunkKdaFwdPrepare,
        DFX_IN(q, k, v, g, beta, aLogOptional, dtBiasOptional,
               cuSeqlensOptional, chunkIndicesOptional, layout, scale,
               chunkSize, epsilon, useQkL2normInKernel, useGateInKernel,
               useBetaSigmoidInKernel, allowNegEigval, safeGate, lowerBound,
               useExp2),
        DFX_OUT(gkOut, aqkOut, akkOut, wOut, uOut, qgOut, kgOut,
                qgScaledOut, qHatOut, kHatOut, qRstdOut, kRstdOut,
                betaEffOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr,
              ACLNN_ERR_INNER_CREATE_EXECUTOR);
    aclOpExecutor *executorPtr = uniqueExecutor.get();

    aclnnStatus status = CheckParams(params);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = MakeInputsContiguous(params, executorPtr);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    const auto result = l0op::ChunkKdaFwdPrepare(
        params.q, params.k, params.v, params.g, params.beta,
        params.aLogOptional, params.dtBiasOptional,
        params.cuSeqlensOptional, params.chunkIndicesOptional, params.layout,
        params.scale, params.chunkSize, params.epsilon,
        params.useQkL2normInKernel, params.useGateInKernel,
        params.useBetaSigmoidInKernel, params.allowNegEigval,
        params.safeGate, params.lowerBound, params.useExp2,
        params.gkOut, params.aqkOut, params.akkOut, params.wOut, params.uOut,
        params.qgOut, params.kgOut, params.qgScaledOut, params.qHatOut,
        params.kHatOut, params.qRstdOut, params.kRstdOut, params.betaEffOut,
        GetOutputMode(params), executorPtr);
    CHECK_RET(result[0] != nullptr && result[1] != nullptr &&
                  result[3] != nullptr && result[4] != nullptr &&
                  result[6] != nullptr && result[7] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkKdaFwdPrepare(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkKdaFwdPrepare);
    CHECK_COND(
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) ==
            ACLNN_SUCCESS,
        ACLNN_ERR_INNER, "ChunkKdaFwdPrepare AI Core 启动失败。");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
