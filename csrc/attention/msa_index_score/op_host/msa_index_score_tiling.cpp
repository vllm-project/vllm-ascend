/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file msa_index_score_tiling.cpp
 * \brief MsaIndexScore Tiling 实现（A2/A3 / 950：PA BBND/BNBD + TND packed key；sparse_mode 0/3；
 *        PA key 允许 dim0 非连续，strideKvBlock 来自 GetInputStride）。
 */

#include "msa_index_score_tiling.h"
#include "../op_kernel/msa_index_score_common.h"

#include <limits>
#include <string>
#include "graph/utils/type_utils.h"

using namespace ge;
using namespace MsaIndexScoreNs;

namespace optiling {
namespace {
constexpr uint32_t MSA_QUERY_DIM_NUM = 3;
constexpr uint32_t MSA_KEY_TND_DIM_NUM = 3;
constexpr uint32_t MSA_KEY_PA_DIM_NUM = 4;
constexpr uint32_t MSA_BLOCK_TABLE_DIM_NUM = 2;
constexpr uint32_t MSA_SCALE_PA_DIM_NUM = 3;
constexpr uint32_t MSA_SCALE_TND_DIM_NUM = 2;
constexpr uint32_t MSA_ATTEN_MASK_DIM_NUM = 2;
constexpr uint32_t MSA_DIM_0 = 0;
constexpr uint32_t MSA_DIM_1 = 1;
constexpr uint32_t MSA_DIM_2 = 2;
constexpr uint32_t MSA_DIM_3 = 3;
constexpr uint32_t MSA_RANK_1D = 1;
constexpr uint32_t MSA_QLEN_MIN_DIM0 = 2; // prefix-sum 长度至少为 B+1

inline bool IsFp8ComputeDtype(ge::DataType dt)
{
    return dt == ge::DT_HIFLOAT8 || dt == ge::DT_FLOAT8_E5M2 || dt == ge::DT_FLOAT8_E4M3FN;
}

inline bool IsNonQuantQueryDtype(ge::DataType dt, bool isAscend950)
{
    if (dt == ge::DT_BF16 || dt == ge::DT_FLOAT16) {
        return true;
    }
    return isAscend950 && IsFp8ComputeDtype(dt);
}

inline uint32_t RoundUpU32(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

inline uint64_t GetDefaultStride0(const gert::Shape &shape)
{
    uint64_t stride = 1;
    for (size_t dim = 1; dim < shape.GetDimNum(); ++dim) {
        stride *= static_cast<uint64_t>(shape.GetDim(dim));
    }
    return stride;
}

// 逻辑 view：优先 OriginShape（aclCreateTensor 的 viewDims / torch sizes）。
// StorageShape 在 dim0 插空时可能是 [NP*gap, ...]，不能当 numPages。
inline const gert::Shape &KeyLogicalShape(const gert::StorageShape *keyShape, uint32_t keyLayout)
{
    const size_t expectRank = (keyLayout == MSA_KEY_LAYOUT_TND) ? MSA_KEY_TND_DIM_NUM : MSA_KEY_PA_DIM_NUM;
    const gert::Shape &origin = keyShape->GetOriginShape();
    if (origin.GetDimNum() == expectRank) {
        return origin;
    }
    return keyShape->GetStorageShape();
}

// aclnn TensorV2 优先 GetRequiredInputStride；
// 文档接口 GetInputStride 作为兼容回退。
inline const gert::Stride *GetKeyStrideDesc(gert::TilingContext *context)
{
    const gert::Stride *stride = context->GetRequiredInputStride(MSA_IDX_KEY);
    if (stride != nullptr) {
        return stride;
    }
    return context->GetInputStride(MSA_IDX_KEY);
}

inline bool ValidateKeyInnerAxesContiguous(gert::TilingContext *context, const gert::Stride *stride,
                                           const gert::Shape &shape)
{
    if (stride == nullptr || stride->GetDimNum() != shape.GetDimNum()) {
        return true;
    }
    uint64_t expectedStride = 1;
    for (int64_t i = static_cast<int64_t>(shape.GetDimNum()) - 1; i >= 1; --i) {
        const int64_t dimSize = shape.GetDim(static_cast<size_t>(i));
        if (dimSize <= 1) {
            // size-1 轴不参与寻址；PyTorch 对其 stride 不做连续约束（BNBD 且 N2=1 时常见）。
            continue;
        }
        const uint64_t actualStride = static_cast<uint64_t>(stride->GetStride(static_cast<size_t>(i)));
        if (actualStride != expectedStride) {
            OP_LOGE(context,
                    "key dim%ld must be contiguous, actual stride=%lu, expected=%lu. "
                    "Only dim0 (PA page axis) may be non-contiguous.",
                    i, actualStride, expectedStride);
            return false;
        }
        expectedStride *= static_cast<uint64_t>(dimSize);
    }
    return true;
}

// PA：strideKvBlock 取 key dim0 元素 stride。连续输入写入值与 shape 推算相同。
inline ge::graphStatus ResolveStrideKvBlock(gert::TilingContext *context, const MsaIndexScoreInfo &info,
                                            uint32_t defaultStrideKvBlock, uint32_t &strideKvBlock)
{
    strideKvBlock = defaultStrideKvBlock;

    const gert::StorageShape *keyShape = context->GetInputShape(MSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::Shape &kc = KeyLogicalShape(keyShape, info.keyLayout);
    if (kc.GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t defaultStride0 = GetDefaultStride0(kc);
    const gert::Stride *stride = GetKeyStrideDesc(context);
    if (stride == nullptr || stride->GetDimNum() != kc.GetDimNum()) {
        OP_LOGD(context->GetNodeName(), "key has no stride desc, treat as contiguous, strideKvBlock=%u.",
                strideKvBlock);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(!ValidateKeyInnerAxesContiguous(context, stride, kc),
                OP_LOGE(context, "key inner axes must stay contiguous."), return ge::GRAPH_FAILED);

    const int64_t actualStride0 = stride->GetStride(MSA_DIM_0);
    if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
        OP_CHECK_IF(actualStride0 < 0 || static_cast<uint64_t>(actualStride0) != defaultStride0,
                    OP_LOGE(context, "TND key dim0 must be contiguous, actual=%ld expected=%lu.", actualStride0,
                            defaultStride0),
                    return ge::GRAPH_FAILED);
        strideKvBlock = 0;
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(actualStride0 < 0 || static_cast<uint64_t>(actualStride0) < defaultStride0,
                OP_LOGE(context, "PA key dim0 stride must be >= %lu, got %ld.", defaultStride0, actualStride0),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint64_t>(actualStride0) > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                OP_LOGE(context, "PA key dim0 stride %ld exceeds uint32 strideKvBlock.", actualStride0),
                return ge::GRAPH_FAILED);
    strideKvBlock = static_cast<uint32_t>(actualStride0);
    OP_LOGI(context->GetNodeName(), "PA key strideKvBlock=%u (contiguous default=%lu).", strideKvBlock, defaultStride0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseLayoutKeyAttr(gert::TilingContext *context, const std::string &layoutKey, uint32_t &keyLayout)
{
    const std::string s = layoutKey.empty() ? "BBND" : layoutKey;
    if (s == "TND") {
        keyLayout = MSA_KEY_LAYOUT_TND;
    } else if (s == "BBND") {
        keyLayout = MSA_KEY_LAYOUT_BBND;
    } else if (s == "BNBD") {
        keyLayout = MSA_KEY_LAYOUT_BNBD;
    } else {
        OP_LOGE(context, "layout_key must be TND, BBND or BNBD, got %s.", s.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseAndCheck(gert::TilingContext *context, MsaIndexScoreInfo &info)
{
    info.platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(info.platformInfo == nullptr, OP_LOGE(context, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);
    auto ascendcPlatformEarly = platform_ascendc::PlatformAscendC(info.platformInfo);
    info.isAscend950 = (ascendcPlatformEarly.GetSocVersion() == platform_ascendc::SocVersion::ASCEND950);

    const gert::StorageShape *queryShape = context->GetInputShape(MSA_IDX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::StorageShape *keyShape = context->GetInputShape(MSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    const gert::StorageShape *blockTableShape = context->GetOptionalInputShape(MSA_IDX_BLOCK_TABLE);
    const gert::StorageShape *actualSeqQlenShape = context->GetOptionalInputShape(MSA_IDX_ACTUAL_SEQ_QLEN);
    OP_CHECK_IF(actualSeqQlenShape == nullptr, OP_LOGE(context, "TND query requires actual_seq_qlen."),
                return ge::GRAPH_FAILED);
    const gert::StorageShape *actualSeqKlenShape = context->GetOptionalInputShape(MSA_IDX_ACTUAL_SEQ_KLEN);
    OP_CHECK_IF(actualSeqKlenShape == nullptr, OP_LOGE(context, "actual_seq_klen is required."),
                return ge::GRAPH_FAILED);
    const gert::StorageShape *startLocShape = context->GetInputShape(MSA_IDX_START_LOC);
    OP_CHECK_NULL_WITH_CONTEXT(context, startLocShape);

    const gert::Shape &q = queryShape->GetStorageShape();
    const gert::Shape &qlen = actualSeqQlenShape->GetStorageShape();
    const gert::Shape &klen = actualSeqKlenShape->GetStorageShape();

    const auto *keyDesc = context->GetInputDesc(MSA_IDX_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDesc);
    info.keyDtype = keyDesc->GetDataType();
    OP_CHECK_IF(static_cast<ge::Format>(ge::GetPrimaryFormat(keyDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
                OP_LOGE(context, "FRACTAL_NZ key is not supported."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(q.GetDimNum() != MSA_QUERY_DIM_NUM,
                OP_LOGE(context, "query must be TND(3 dims), got %zu.", q.GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qlen.GetDimNum() != MSA_RANK_1D || qlen.GetDim(MSA_DIM_0) < MSA_QLEN_MIN_DIM0,
                OP_LOGE(context, "actual_seq_qlen must be 1D [B+1]."), return ge::GRAPH_FAILED);

    info.totalQ = static_cast<uint32_t>(q.GetDim(MSA_DIM_0));
    info.numQHeads = static_cast<uint32_t>(q.GetDim(MSA_DIM_1));
    info.headDim = static_cast<uint32_t>(q.GetDim(MSA_DIM_2));
    info.batch = static_cast<uint32_t>(qlen.GetDim(MSA_DIM_0) - 1);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *layoutKeyPtr = attrs->GetStr(MSA_ATTR_LAYOUT_KEY);
    const std::string layoutKey = (layoutKeyPtr == nullptr) ? "" : layoutKeyPtr;
    if (ParseLayoutKeyAttr(context, layoutKey, info.keyLayout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &kc = KeyLogicalShape(keyShape, info.keyLayout);

    if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
        info.totalK = static_cast<uint32_t>(kc.GetDim(MSA_DIM_0));
        info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
        info.blockSize = MSA_BLOCK_SIZE;
        info.numPages = 0;
        OP_CHECK_IF(kc.GetDimNum() != MSA_KEY_TND_DIM_NUM,
                    OP_LOGE(context, "layout_key=TND requires key rank 3 [T2,N2,D], got %zu.", kc.GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            static_cast<uint32_t>(kc.GetDim(MSA_DIM_2)) != info.headDim,
            OP_LOGE(context, "TND key headDim(%ld) must equal query headDim(%u).", kc.GetDim(MSA_DIM_2), info.headDim),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(blockTableShape != nullptr, OP_LOGE(context, "layout_key=TND must not pass block_table."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(klen.GetDimNum() != MSA_RANK_1D || static_cast<uint32_t>(klen.GetDim(MSA_DIM_0)) != info.batch + 1,
                    OP_LOGE(context, "TND actual_seq_klen must be [B+1] prefix-sum."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(blockTableShape == nullptr, OP_LOGE(context, "PageAttention (BBND/BNBD) requires block_table."),
                    return ge::GRAPH_FAILED);
        const gert::Shape &bt = blockTableShape->GetStorageShape();
        OP_CHECK_IF(bt.GetDimNum() != MSA_BLOCK_TABLE_DIM_NUM,
                    OP_LOGE(context, "block_table must have 2 dims, got %zu.", bt.GetDimNum()),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint32_t>(bt.GetDim(MSA_DIM_0)) != info.batch,
                    OP_LOGE(context, "block_table batch(%ld) must equal actual_seq_qlen batch(%u).",
                            bt.GetDim(MSA_DIM_0), info.batch),
                    return ge::GRAPH_FAILED);
        info.maxBlocksPerBatch = static_cast<uint32_t>(bt.GetDim(MSA_DIM_1));
        OP_CHECK_IF(kc.GetDimNum() != MSA_KEY_PA_DIM_NUM,
                    OP_LOGE(context, "layout_key=%s requires key rank 4, got %zu.",
                            info.keyLayout == MSA_KEY_LAYOUT_BNBD ? "BNBD" : "BBND", kc.GetDimNum()),
                    return ge::GRAPH_FAILED);
        info.numPages = static_cast<uint32_t>(kc.GetDim(MSA_DIM_0));
        if (info.keyLayout == MSA_KEY_LAYOUT_BBND) {
            info.blockSize = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
            info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_2));
            OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(MSA_DIM_3)) != info.headDim,
                        OP_LOGE(context, "BBND key headDim(%ld) must equal query headDim(%u).", kc.GetDim(MSA_DIM_3),
                                info.headDim),
                        return ge::GRAPH_FAILED);
        } else {
            info.numKvHeads = static_cast<uint32_t>(kc.GetDim(MSA_DIM_1));
            info.blockSize = static_cast<uint32_t>(kc.GetDim(MSA_DIM_2));
            OP_CHECK_IF(static_cast<uint32_t>(kc.GetDim(MSA_DIM_3)) != info.headDim,
                        OP_LOGE(context, "BNBD key headDim(%ld) must equal query headDim(%u).", kc.GetDim(MSA_DIM_3),
                                info.headDim),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(klen.GetDimNum() != MSA_RANK_1D || static_cast<uint32_t>(klen.GetDim(MSA_DIM_0)) != info.batch,
                    OP_LOGE(context, "PA actual_seq_klen must be [B] visible S2."), return ge::GRAPH_FAILED);
    }

    if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
        const gert::StorageShape *scoreShape = context->GetOutputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, scoreShape);
        const gert::Shape &scOut = scoreShape->GetStorageShape();
        OP_CHECK_IF(scOut.GetDimNum() != MSA_QUERY_DIM_NUM, OP_LOGE(context, "score must be 3D [N1,T1,stride]."),
                    return ge::GRAPH_FAILED);
        info.maxBlocksPerBatch = static_cast<uint32_t>(scOut.GetDim(MSA_DIM_2));
        OP_CHECK_IF(info.maxBlocksPerBatch == 0, OP_LOGE(context, "TND score last dim must be positive."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((info.maxBlocksPerBatch % MSA_SCORE_STRIDE_ALIGN) != 0,
                    OP_LOGE(context, "TND score last dim(%u) must be a multiple of %u.", info.maxBlocksPerBatch,
                            MSA_SCORE_STRIDE_ALIGN),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(info.blockSize != MSA_BLOCK_SIZE,
                OP_LOGE(context, "only blockSize == %u is supported, got %u.", MSA_BLOCK_SIZE, info.blockSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        info.headDim == 0 || (info.headDim % MSA_SCORE_STRIDE_ALIGN) != 0,
        OP_LOGE(context, "headDim(%u) must be a positive multiple of %u.", info.headDim, MSA_SCORE_STRIDE_ALIGN),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.headDim > MSA_K_TILE,
                OP_LOGE(context, "headDim(%u) larger than %u is not supported yet.", info.headDim, MSA_K_TILE),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.numKvHeads == 0 || (info.numQHeads % info.numKvHeads) != 0,
                OP_LOGE(context, "numQHeads(%u) must be divisible by numKvHeads(%u).", info.numQHeads, info.numKvHeads),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.batch == 0, OP_LOGE(context, "batch must be positive."), return ge::GRAPH_FAILED);
    // q_len/kv_len 全 0 跳过计算。InferShape 已把空 KV 的 score 末维钳到 16。
    OP_CHECK_IF(info.maxBlocksPerBatch == 0,
                OP_LOGE(context, "maxBlocksPerBatch must be positive (empty kv uses aligned score stride)."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(info.keyLayout != MSA_KEY_LAYOUT_TND && info.numPages == 0,
                OP_LOGE(context, "PA numPages must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(static_cast<uint32_t>(startLocShape->GetStorageShape().GetDim(MSA_DIM_0)) != info.batch,
                OP_LOGE(context, "start_loc size must equal batch."), return ge::GRAPH_FAILED);

    const auto *queryDesc = context->GetInputDesc(MSA_IDX_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    info.queryDtype = queryDesc->GetDataType();
    OP_CHECK_IF(!IsNonQuantQueryDtype(info.queryDtype, info.isAscend950),
                OP_LOGE(context, "query dtype must be bf16 or fp16 on A2/A3; Ascend 950 also allows hifloat8 / "
                                 "float8_e5m2 / float8_e4m3fn."),
                return ge::GRAPH_FAILED);

    info.isQuant = (info.keyDtype == ge::DT_INT8);
    if (info.isQuant) {
        OP_CHECK_IF(info.queryDtype != ge::DT_FLOAT16, OP_LOGE(context, "ND int8 key currently requires fp16 query."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(info.keyDtype != info.queryDtype, OP_LOGE(context, "non-quant key dtype must match query dtype."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(IsFp8ComputeDtype(info.queryDtype) && !info.isAscend950,
                    OP_LOGE(context, "FP8 query/key is only supported on Ascend 950."), return ge::GRAPH_FAILED);
    }

    const gert::StorageShape *scaleShape = context->GetOptionalInputShape(MSA_IDX_SCALE);
    if (info.isQuant) {
        OP_CHECK_IF(scaleShape == nullptr, OP_LOGE(context, "int8 key requires dequant scale."),
                    return ge::GRAPH_FAILED);
        const gert::Shape &sc = scaleShape->GetStorageShape();
        if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
            const bool ok2d = (sc.GetDimNum() == MSA_SCALE_TND_DIM_NUM &&
                               static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) == info.totalK &&
                               static_cast<uint32_t>(sc.GetDim(MSA_DIM_1)) == info.numKvHeads);
            const bool ok1d = (sc.GetDimNum() == MSA_RANK_1D && info.numKvHeads == 1 &&
                               static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) == info.totalK);
            OP_CHECK_IF(!ok2d && !ok1d,
                        OP_LOGE(context, "TND dequant scale must be [%u, %u] or [%u] (N2=1).", info.totalK,
                                info.numKvHeads, info.totalK),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(sc.GetDimNum() != MSA_SCALE_PA_DIM_NUM,
                        OP_LOGE(context, "PA dequant scale must be 3D [NP, N_kv, P], got rank %zu.", sc.GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(static_cast<uint32_t>(sc.GetDim(MSA_DIM_0)) != info.numPages ||
                            static_cast<uint32_t>(sc.GetDim(MSA_DIM_1)) != info.numKvHeads ||
                            static_cast<uint32_t>(sc.GetDim(MSA_DIM_2)) != info.blockSize,
                        OP_LOGE(context, "dequant scale shape must be [%u, %u, %u], got [%ld, %ld, %ld].",
                                info.numPages, info.numKvHeads, info.blockSize, sc.GetDim(MSA_DIM_0),
                                sc.GetDim(MSA_DIM_1), sc.GetDim(MSA_DIM_2)),
                        return ge::GRAPH_FAILED);
        }
        const auto *scaleDesc = context->GetOptionalInputDesc(MSA_IDX_SCALE);
        OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
        OP_CHECK_IF(scaleDesc->GetDataType() != ge::DT_FLOAT, OP_LOGE(context, "dequant scale dtype must be float32."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(scaleShape != nullptr, OP_LOGE(context, "non-quant path must not pass scale."),
                    return ge::GRAPH_FAILED);
    }

    const int64_t *sparseModeAttr = attrs->GetInt(MSA_ATTR_SPARSE_MODE);
    info.sparseMode = (sparseModeAttr == nullptr) ? MSA_SPARSE_MODE_RIGHT_DOWN : static_cast<uint32_t>(*sparseModeAttr);
    OP_CHECK_IF(info.sparseMode != MSA_SPARSE_MODE_DEFAULT && info.sparseMode != MSA_SPARSE_MODE_RIGHT_DOWN,
                OP_LOGE(context, "sparse_mode must be %u or %u, got %u.", MSA_SPARSE_MODE_DEFAULT,
                        MSA_SPARSE_MODE_RIGHT_DOWN, info.sparseMode),
                return ge::GRAPH_FAILED);

    const gert::StorageShape *attenMaskShape = context->GetOptionalInputShape(MSA_IDX_ATTEN_MASK);
    if (info.sparseMode == MSA_SPARSE_MODE_RIGHT_DOWN) {
        OP_CHECK_IF(attenMaskShape == nullptr,
                    OP_LOGE(context, "sparse_mode=%u requires atten_mask of shape [%u, %u].",
                            MSA_SPARSE_MODE_RIGHT_DOWN, MSA_ATTEN_MASK_SIZE, MSA_ATTEN_MASK_SIZE),
                    return ge::GRAPH_FAILED);
        const gert::Shape &am = attenMaskShape->GetStorageShape();
        OP_CHECK_IF(am.GetDimNum() != MSA_ATTEN_MASK_DIM_NUM ||
                        static_cast<uint32_t>(am.GetDim(MSA_DIM_0)) != MSA_ATTEN_MASK_SIZE ||
                        static_cast<uint32_t>(am.GetDim(MSA_DIM_1)) != MSA_ATTEN_MASK_SIZE,
                    OP_LOGE(context, "atten_mask must be [%u, %u], got rank=%zu dims=[%ld,%ld].", MSA_ATTEN_MASK_SIZE,
                            MSA_ATTEN_MASK_SIZE, am.GetDimNum(), am.GetDimNum() > 0 ? am.GetDim(MSA_DIM_0) : -1,
                            am.GetDimNum() > 1 ? am.GetDim(MSA_DIM_1) : -1),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(attenMaskShape != nullptr,
                    OP_LOGE(context, "sparse_mode=%u must not pass atten_mask.", MSA_SPARSE_MODE_DEFAULT),
                    return ge::GRAPH_FAILED);
    }

    const int64_t *initBlocksAttr = attrs->GetInt(MSA_ATTR_INIT_BLOCKS);
    const int64_t *localBlocksAttr = attrs->GetInt(MSA_ATTR_LOCAL_BLOCKS);
    info.initBlocks = (initBlocksAttr == nullptr) ? MSA_DEFAULT_INIT_BLOCKS : static_cast<uint32_t>(*initBlocksAttr);
    info.localBlocks =
        (localBlocksAttr == nullptr) ? MSA_DEFAULT_LOCAL_BLOCKS : static_cast<uint32_t>(*localBlocksAttr);
    OP_CHECK_IF(info.initBlocks > info.maxBlocksPerBatch || info.localBlocks > info.maxBlocksPerBatch,
                OP_LOGE(context, "init_blocks/local_blocks must be <= maxBlocksPerBatch."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DoTiling(gert::TilingContext *context, const MsaIndexScoreInfo &info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(info.platformInfo);
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(aicNum == 0, OP_LOGE(context, "aic core num is 0."), return ge::GRAPH_FAILED);

    const uint32_t scoreBlockStride = RoundUpU32(info.maxBlocksPerBatch, MSA_SCORE_STRIDE_ALIGN);
    // 950 C_to_UB：S 不进 GM；int8/TND 尾页仍要一页 K scratch。A2 与 950 回退（MSA_A5_USE_C2UB=0）走 8-stage S。
    const uint32_t sElemBytes = info.isQuant ? sizeof(float) : sizeof(uint16_t);
    const bool c2ub = info.isAscend950 && (MSA_A5_USE_C2UB != 0);
    const uint32_t sWsBytes = c2ub ? 0U : (aicNum * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM * sElemBytes);
    const bool useKScratch = info.isQuant || (info.keyLayout == MSA_KEY_LAYOUT_TND);
    uint32_t scratchElemBytes = sizeof(uint16_t);
    if (info.isAscend950 && !info.isQuant && IsFp8ComputeDtype(info.queryDtype)) {
        scratchElemBytes = 1U;
    }
    uint32_t kScratchElems = c2ub ? MSA_A5_K_SCRATCH_ELEM_NUM : MSA_K_SCRATCH_ELEM_NUM;
    // A2/A3 int8 四槽 K scratch；950 C2UB / 非量化 TND 尾页保持单槽。
    if (!info.isAscend950 && info.isQuant) {
        kScratchElems *= MSA_K_SCRATCH_STAGES_A2;
    }
    const uint32_t kScratchBytes = useKScratch ? (aicNum * kScratchElems * scratchElemBytes) : 0U;
    const uint32_t kScratchOffsetElems = sWsBytes / sizeof(float);

    MsaIndexScoreTilingData tilingData;
    tilingData.set_batch(info.batch);
    tilingData.set_totalQ(info.totalQ);
    tilingData.set_numQHeads(info.numQHeads);
    tilingData.set_numKvHeads(info.numKvHeads);
    tilingData.set_gqaGroup(info.numQHeads / info.numKvHeads);
    tilingData.set_headDim(info.headDim);
    tilingData.set_blockSize(info.blockSize);
    tilingData.set_maxBlocksPerBatch(info.maxBlocksPerBatch);
    tilingData.set_scoreBlockStride(scoreBlockStride);
    tilingData.set_usedCoreNum(aicNum);
    tilingData.set_isQuant(info.isQuant ? 1U : 0U);
    tilingData.set_sparseMode(info.sparseMode);
    tilingData.set_initBlocks(info.initBlocks);
    tilingData.set_localBlocks(info.localBlocks);
    tilingData.set_numPages(info.numPages);
    tilingData.set_keyLayout(info.keyLayout);
    tilingData.set_totalK(info.totalK);

    tilingData.set_strideQt(info.numQHeads * info.headDim);
    tilingData.set_strideQn(info.headDim);
    uint32_t defaultStrideKvBlock = 0;
    if (info.keyLayout == MSA_KEY_LAYOUT_TND) {
        defaultStrideKvBlock = 0;
        tilingData.set_strideKvToken(info.numKvHeads * info.headDim);
        tilingData.set_strideScalePage(info.numKvHeads);
        tilingData.set_strideScaleHead(1);
    } else if (info.keyLayout == MSA_KEY_LAYOUT_BNBD) {
        defaultStrideKvBlock = info.numKvHeads * info.blockSize * info.headDim;
        tilingData.set_strideKvToken(info.headDim);
        tilingData.set_strideScalePage(info.numKvHeads * info.blockSize);
        tilingData.set_strideScaleHead(info.blockSize);
    } else {
        defaultStrideKvBlock = info.blockSize * info.numKvHeads * info.headDim;
        tilingData.set_strideKvToken(info.numKvHeads * info.headDim);
        tilingData.set_strideScalePage(info.numKvHeads * info.blockSize);
        tilingData.set_strideScaleHead(info.blockSize);
    }
    uint32_t strideKvBlock = defaultStrideKvBlock;
    if (ResolveStrideKvBlock(context, info, defaultStrideKvBlock, strideKvBlock) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    tilingData.set_strideKvBlock(strideKvBlock);
    tilingData.set_strideOutHead(info.totalQ * scoreBlockStride);
    tilingData.set_strideOutToken(scoreBlockStride);
    tilingData.set_kScratchOffsetElems(kScratchOffsetElems);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    // MIX 1AIC:2AIV：CalcTschBlockDim 的 sliceNum 按 AIV 计数，内部再 / (aiv/aic)。
    // 传入 aicNum 会再除一次得到 blockDim=aic/2，只能打一半 Cube。
    // sliceNum = aivNum。
    // 整 batch q_len=0：queryS==0 → BlockDim=1，避免 totalTaskNum=0。
    if (info.totalQ == 0U) {
        context->SetBlockDim(1);
    } else {
        context->SetBlockDim(ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum));
    }

    size_t *workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] =
        ascendcPlatform.GetLibApiWorkSpaceSize() + static_cast<size_t>(sWsBytes) + static_cast<size_t>(kScratchBytes);
    // 非量化 PA 无 S GM / K scratch 时 user workspace 为 0，GetUserWorkspace 可能返回 nullptr。
    if (info.isAscend950 && sWsBytes == 0U && kScratchBytes == 0U) {
        workspaces[0] += 64U;
    }

    uint64_t tilingKey = MSA_TILING_KEY_FP16;
    if (info.isQuant) {
        tilingKey = MSA_TILING_KEY_FP16_INT8;
    } else if (info.queryDtype == ge::DT_HIFLOAT8) {
        tilingKey = MSA_TILING_KEY_HIFLOAT8;
    } else if (info.queryDtype == ge::DT_FLOAT8_E5M2) {
        tilingKey = MSA_TILING_KEY_FP8_E5M2;
    } else if (info.queryDtype == ge::DT_FLOAT8_E4M3FN) {
        tilingKey = MSA_TILING_KEY_FP8_E4M3FN;
    } else {
        tilingKey = (info.queryDtype == ge::DT_BF16) ? MSA_TILING_KEY_BF16 : MSA_TILING_KEY_FP16;
    }
    context->SetTilingKey(tilingKey);
    OP_LOGI(context->GetNodeName(), "MsaIndexScore tilingKey=%lu headDim=%u layout=%u strideKvBlock=%u", tilingKey,
            info.headDim, info.keyLayout, strideKvBlock);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus TilingPrepareForMsaIndexScore(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForMsaIndexScore(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("MsaIndexScore", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    MsaIndexScoreInfo info;
    if (ParseAndCheck(context, info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return DoTiling(context, info);
}

IMPL_OP_OPTILING(MsaIndexScore)
    .Tiling(TilingForMsaIndexScore)
    .TilingParse<MsaIndexScoreCompileInfo>(TilingPrepareForMsaIndexScore);

} // namespace optiling
