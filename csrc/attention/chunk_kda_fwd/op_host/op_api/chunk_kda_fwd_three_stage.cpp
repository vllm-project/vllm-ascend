/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "chunk_kda_fwd_three_stage.h"

#include "../../../chunk_kda_fwd_prepare/op_host/chunk_kda_fwd_prepare_output_mask.h"
#include "../../../chunk_kda_fwd_prepare/op_host/op_api/chunk_kda_fwd_prepare.h"
#include "../../../chunk_kda_fwd_finalize/op_host/op_api/chunk_kda_fwd_finalize.h"
#include "../../../chunk_fwd_h/op_host/op_api/chunk_fwd_h.h"

#include <initializer_list>
#include <vector>

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace l0op {

namespace {

constexpr int64_t KDA_FWD_THREE_STAGE_AQK_COLUMNS = 64;

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

// 组合调用里所有中间张量都由 executor 分配，必须显式把 storage/original shape 与
// view shape 对齐，否则后续子算子的 tiling 会按退化的一维 storage shape 校验失败。
void NormalizeTensorMeta(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return;
    }
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageShape(tensor->GetViewShape());
    mutableTensor->SetOriginalShape(tensor->GetViewShape());
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
}

const aclTensor *AllocTensor(aclOpExecutor *executor, const op::Shape &shape, DataType dtype)
{
    const aclTensor *tensor = executor->AllocTensor(shape, dtype, Format::FORMAT_ND);
    NormalizeTensorMeta(tensor);
    return tensor;
}

// 名字与 l0op::ViewCopy 区分，避免在 namespace l0op 内产生重载歧义。
aclnnStatus CopyToOutput(const aclTensor *src, const aclTensor *dst, aclOpExecutor *executor)
{
    CHECK_COND(l0op::ViewCopy(src, dst, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR,
               "ChunkKdaFwd 三算子组合 ViewCopy 失败。");
    return ACLNN_SUCCESS;
}

// 公开输出与子算子期望的 head-major shape/dtype 完全一致时直接复用，
// 避免额外整张拷贝。
const aclTensor *ReuseOrAlloc(const aclTensor *publicOut, const op::Shape &shape, DataType dtype,
                              aclOpExecutor *executor)
{
    if (publicOut != nullptr && IsContiguous(publicOut) && publicOut->GetViewShape() == shape &&
        publicOut->GetDataType() == dtype) {
        NormalizeTensorMeta(publicOut);
        return publicOut;
    }
    return AllocTensor(executor, shape, dtype);
}

// ChunkFwdH 只接受 rank-4 BNSD 输入；rank-3 head-major 张量在这里补一维。
const aclTensor *MaybeReshapeToHeadMajor4(const aclTensor *tensor, bool packed, int64_t heads,
                                          int64_t seqLen, int64_t columns, const char *name,
                                          aclOpExecutor *executor)
{
    if (tensor == nullptr || !packed) {
        return tensor;
    }
    const aclTensor *reshaped =
        l0op::Reshape(tensor, MakeShape({1, heads, seqLen, columns}), executor);
    if (reshaped == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ChunkKdaFwd 三算子组合：%s 重排为 rank-4 BNSD 失败。", name);
    }
    return reshaped;
}

// 把 head-major 视图整理成连续张量，只使用 Transpose + Contiguous + Reshape。
const aclTensor *TransposeToContiguous(const aclTensor *input, const std::vector<int64_t> &perm,
                                       aclOpExecutor *executor)
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
    const aclTensor *reshaped = l0op::Reshape(materialized, transposed->GetViewShape(), executor);
    if (reshaped == nullptr) {
        return nullptr;
    }
    reshaped->SetStorageShape(reshaped->GetViewShape());
    reshaped->SetOriginalShape(reshaped->GetViewShape());
    return reshaped;
}

} // namespace

int64_t KdaFwdThreeStageOutputMode(const KdaFwdThreeStageArgs &args)
{
    const bool wantSavedIntermediates = args.wOut != nullptr || args.uOut != nullptr ||
                                        args.qgOut != nullptr || args.kgOut != nullptr ||
                                        args.vNewOut != nullptr;
    if (wantSavedIntermediates) {
        return optiling::PREPARE_OUTPUT_MODE_SAVE;
    }
    if (args.akkOut != nullptr) {
        // Aqk/Akk 是公开必选输出，但 qHat/kHat/qRstd/kRstd/betaEff 只服务反向
        // 重计算，前向链不需要，因此走只多搬 Akk 的 forward 档。
        return optiling::PREPARE_OUTPUT_MODE_FORWARD;
    }
    return optiling::PREPARE_OUTPUT_MODE_NONE;
}

aclnnStatus KdaFwdThreeStage(const KdaFwdThreeStageArgs &args, aclOpExecutor *executor)
{
    // 组合入口的输入/输出张量都可能来自 pyTorch ctypes 描述符，其 storage shape
    // 默认是展平的一维 numel；三个子算子的 tiling 按 storage shape 校验维度，
    // 因此这里统一把连续张量的 storage/original shape 对齐到 view shape。
    const aclTensor *contiguousTensors[] = {
        args.q,         args.k,        args.v,        args.g,      args.beta,
        args.aLog,      args.dtBias,   args.initialState,
        args.attnOut,   args.finalStateOut, args.gkOut,  args.aqkOut,
        args.akkOut,    args.wOut,     args.uOut,     args.qgOut,
        args.kgOut,     args.vNewOut,  args.hOut};
    for (const aclTensor *tensor : contiguousTensors) {
        NormalizeTensorMeta(tensor);
    }

    const int64_t chunkSize = args.chunkSize;
    const int64_t valueHeads = args.valueHeads;
    const bool packed = args.packed;
    // Prepare 的 outputMask 必须恰好等于三档之一：none 只写前向链必需量，
    // recompute 追加 qHat/kHat/qRstd/kRstd/betaEff，save 再追加 qg。
    const int64_t outputMode = KdaFwdThreeStageOutputMode(args);
    const bool needAkk = outputMode != optiling::PREPARE_OUTPUT_MODE_NONE;
    const bool needBackwardAux =
        outputMode == optiling::PREPARE_OUTPUT_MODE_RECOMPUTE ||
        outputMode == optiling::PREPARE_OUTPUT_MODE_SAVE;
    const bool needSavedQ = outputMode == optiling::PREPARE_OUTPUT_MODE_SAVE;

    const auto valueMatrix = [&](int64_t columns) {
        return packed ? MakeShape({valueHeads, args.seqLen, columns})
                      : MakeShape({args.batch, valueHeads, args.seqLen, columns});
    };
    const auto valueScalar = [&]() {
        return packed ? MakeShape({valueHeads, args.seqLen})
                      : MakeShape({args.batch, valueHeads, args.seqLen});
    };
    const auto qkMatrix = [&]() {
        return packed ? MakeShape({args.qkHeads, args.seqLen, args.kDim})
                      : MakeShape({args.batch, args.qkHeads, args.seqLen, args.kDim});
    };
    const auto qkScalar = [&]() {
        return packed ? MakeShape({args.qkHeads, args.seqLen})
                      : MakeShape({args.batch, args.qkHeads, args.seqLen});
    };

    // ---- Stage 1: ChunkKdaFwdPrepare -------------------------------------
    const aclTensor *gkCompute =
        ReuseOrAlloc(args.gkOut, valueMatrix(args.kDim), DataType::DT_FLOAT, executor);
    const aclTensor *aqkCompute = ReuseOrAlloc(args.aqkOut, valueMatrix(KDA_FWD_THREE_STAGE_AQK_COLUMNS),
                                               DataType::DT_BF16, executor);
    const aclTensor *akkCompute = needAkk
        ? ReuseOrAlloc(args.akkOut, valueMatrix(KDA_FWD_THREE_STAGE_AQK_COLUMNS), DataType::DT_BF16,
                       executor)
        : nullptr;
    const aclTensor *wCompute =
        ReuseOrAlloc(args.wOut, valueMatrix(args.kDim), DataType::DT_BF16, executor);
    const aclTensor *uCompute =
        ReuseOrAlloc(args.uOut, valueMatrix(args.vDim), DataType::DT_BF16, executor);
    const aclTensor *qgCompute =
        needSavedQ ? ReuseOrAlloc(args.qgOut, valueMatrix(args.kDim), DataType::DT_BF16, executor)
                   : nullptr;
    const aclTensor *kgCompute =
        ReuseOrAlloc(args.kgOut, valueMatrix(args.kDim), DataType::DT_BF16, executor);
    const aclTensor *qgScaledCompute = AllocTensor(executor, valueMatrix(args.kDim), DataType::DT_BF16);
    const aclTensor *qHatCompute =
        needBackwardAux ? AllocTensor(executor, qkMatrix(), DataType::DT_BF16) : nullptr;
    const aclTensor *kHatCompute =
        needBackwardAux ? AllocTensor(executor, qkMatrix(), DataType::DT_BF16) : nullptr;
    const aclTensor *qRstdCompute =
        needBackwardAux ? AllocTensor(executor, qkScalar(), DataType::DT_FLOAT) : nullptr;
    const aclTensor *kRstdCompute =
        needBackwardAux ? AllocTensor(executor, qkScalar(), DataType::DT_FLOAT) : nullptr;
    const aclTensor *betaEffCompute =
        needBackwardAux ? AllocTensor(executor, valueScalar(), DataType::DT_FLOAT) : nullptr;
    CHECK_COND(gkCompute != nullptr && aqkCompute != nullptr && wCompute != nullptr &&
                   uCompute != nullptr && kgCompute != nullptr && qgScaledCompute != nullptr &&
                   (!needAkk || akkCompute != nullptr) &&
                   (!needBackwardAux || (qHatCompute != nullptr && kHatCompute != nullptr &&
                                         qRstdCompute != nullptr && kRstdCompute != nullptr &&
                                         betaEffCompute != nullptr)) &&
                   (!needSavedQ || qgCompute != nullptr),
               ACLNN_ERR_INNER_NULLPTR,
               "ChunkKdaFwd 三算子组合：Prepare 输出张量分配失败。");

    const auto prepareResult = l0op::ChunkKdaFwdPrepare(
        args.q, args.k, args.v, args.g, args.beta, args.aLog, args.dtBias, args.cuSeqlens,
        args.chunkIndices, args.layout, args.scale, chunkSize, args.epsilon,
        args.useQkL2normInKernel, args.useGateInKernel, args.useBetaSigmoidInKernel,
        args.allowNegEigval, args.safeGate, args.lowerBound, args.useExp2, gkCompute,
        aqkCompute, akkCompute,
        wCompute, uCompute, qgCompute, kgCompute, qgScaledCompute, qHatCompute, kHatCompute,
        qRstdCompute, kRstdCompute, betaEffCompute, outputMode, executor);
    CHECK_COND(prepareResult[0] != nullptr && prepareResult[1] != nullptr &&
                   prepareResult[3] != nullptr && prepareResult[4] != nullptr &&
                   prepareResult[6] != nullptr && prepareResult[7] != nullptr,
               ACLNN_ERR_INNER_NULLPTR, "ChunkKdaFwd 三算子组合：ChunkKdaFwdPrepare 提交失败。");

    const aclTensor *wHead = MaybeReshapeToHeadMajor4(wCompute, packed, valueHeads, args.seqLen,
                                                      args.kDim, "w", executor);
    const aclTensor *uHead = MaybeReshapeToHeadMajor4(uCompute, packed, valueHeads, args.seqLen,
                                                      args.vDim, "u", executor);
    const aclTensor *kgHead = MaybeReshapeToHeadMajor4(kgCompute, packed, valueHeads, args.seqLen,
                                                       args.kDim, "kg", executor);
    const aclTensor *gkHead = MaybeReshapeToHeadMajor4(gkCompute, packed, valueHeads, args.seqLen,
                                                       args.kDim, "gk", executor);
    CHECK_COND(wHead != nullptr && uHead != nullptr && kgHead != nullptr && gkHead != nullptr,
               ACLNN_ERR_INNER_NULLPTR, "ChunkKdaFwd 三算子组合：head-major 重排失败。");

    // ---- Stage 2: ChunkFwdH ----------------------------------------------
    // state_v_first 由 ChunkFwdH 原生解释，不做 host 侧转置。
    const bool outputFinalState = args.finalStateOut != nullptr;
    const aclTensor *hCompute = AllocTensor(
        executor, MakeShape({args.batch, valueHeads, args.totalChunks, args.kDim, args.vDim}),
        DataType::DT_BF16);
    const aclTensor *vNewCompute =
        ReuseOrAlloc(args.vNewOut,
                     MakeShape({args.batch, valueHeads, args.seqLen, args.vDim}), DataType::DT_BF16,
                     executor);
    CHECK_COND(hCompute != nullptr && vNewCompute != nullptr, ACLNN_ERR_INNER_NULLPTR,
               "ChunkKdaFwd 三算子组合：FwdH 输出张量分配失败。");

    const auto fwdHResult = l0op::ChunkFwdH(
        kgHead, wHead, uHead, nullptr, gkHead, args.initialState, args.cuSeqlens, args.chunkIndices,
        outputFinalState, chunkSize, true, args.useExp2, args.stateVFirst, hCompute,
        vNewCompute, outputFinalState ? args.finalStateOut : nullptr, executor);
    CHECK_COND(fwdHResult[0] != nullptr && fwdHResult[1] != nullptr &&
                   (!outputFinalState || fwdHResult[2] != nullptr),
               ACLNN_ERR_INNER_NULLPTR, "ChunkKdaFwd 三算子组合：ChunkFwdH 提交失败。");

    // ---- Stage 3: ChunkKdaFwdFinalize ------------------------------------
    // Finalize 自己完成 attnOut 的布局落盘，rank-3 输入同样原生支持。
    const aclTensor *finalizeResult = l0op::ChunkKdaFwdFinalize(
        qgScaledCompute, aqkCompute, vNewCompute, hCompute, args.cuSeqlens, args.chunkIndices,
        args.attnLayout, args.stateVFirst, args.attnOut, executor);
    CHECK_COND(finalizeResult != nullptr, ACLNN_ERR_INNER_NULLPTR,
               "ChunkKdaFwd 三算子组合：ChunkKdaFwdFinalize 提交失败。");

    // ---- 回写：只处理无法直接复用的公开输出 ------------------------------
    if (args.vNewOut != nullptr && vNewCompute != args.vNewOut) {
        const aclTensor *vNewDst = args.vNewOut;
        if (packed && Rank(vNewDst) == 3) {
            vNewDst = l0op::Reshape(
                vNewDst, MakeShape({1, valueHeads, args.seqLen, args.vDim}), executor);
            CHECK_COND(vNewDst != nullptr, ACLNN_ERR_INNER_NULLPTR,
                       "ChunkKdaFwd 三算子组合：v_new 重排失败。");
        }
        CHECK_RET(CopyToOutput(vNewCompute, vNewDst, executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
    }
    if (args.hOut != nullptr) {
        // 内部 h 为 head-major [B,HV,Nc,K,V]，公开 hOut 为 chunk-major。
        const aclTensor *hSrc = TransposeToContiguous(hCompute, {0, 2, 1, 3, 4}, executor);
        CHECK_COND(hSrc != nullptr, ACLNN_ERR_INNER_NULLPTR,
                   "ChunkKdaFwd 三算子组合：h 布局转换失败。");
        const aclTensor *hDst = args.hOut;
        if (Rank(hDst) == 4) {
            hDst = l0op::Reshape(
                hDst, MakeShape({args.batch, args.totalChunks, valueHeads, args.kDim, args.vDim}),
                executor);
            CHECK_COND(hDst != nullptr, ACLNN_ERR_INNER_NULLPTR,
                       "ChunkKdaFwd 三算子组合：hOut 重排失败。");
        }
        CHECK_RET(CopyToOutput(hSrc, hDst, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

} // namespace l0op
