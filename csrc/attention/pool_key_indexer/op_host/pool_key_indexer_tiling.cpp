/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "pool_key_indexer_tiling.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include "register/op_impl_registry.h"
#include "../op_kernel/pool_key_indexer_template_tiling_key.h"

using namespace ge;
namespace optiling {

// ValueDepend 输入的 host 可见性判断: GetData() 对 device tensor 返回 device
// 地址(host 解引用段错误); 仅 kOnHost/kFollowing 可读, 其余由 kernel 从 GM 读值
static bool IsHostVisibleValue(const gert::Tensor *tensor)
{
    if (tensor == nullptr) {
        return false;
    }
    const auto placement = tensor->GetPlacement();
    return (placement == gert::kOnHost) || (placement == gert::kFollowing);
}

static uint32_t AlignUp(uint32_t val, uint32_t align)
{
    return (align == 0) ? val : (val + align - 1) / align * align;
}

static uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

static uint64_t RoundUp64(uint64_t val, uint64_t align)
{
    return (align == 0) ? val : (val + align - 1) / align * align;
}

ge::graphStatus PoolKeyIndexerTiling::ParseAndCheckParams(PoolKeyIndexerTilingInfo &info)
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t *topkPtr = attrs->GetInt(PKI_ATTR_TOPK);
    OP_CHECK_NULL_WITH_CONTEXT(context_, topkPtr);
    int64_t topkVal = *topkPtr;
    bool topkValid = (topkVal >= 1 && topkVal <= 2048) || topkVal == 3072 || topkVal == 4096 || topkVal == 5120 ||
                     topkVal == 6144 || topkVal == 7168 || topkVal == 8192;
    OP_CHECK_IF(
        !topkValid,
        OP_LOGE(context_, "topk(%ld) must be in [1, 2048] or one of {3072, 4096, 5120, 6144, 7168, 8192}", topkVal),
        return ge::GRAPH_FAILED);
    info.topk = static_cast<uint32_t>(topkVal);

    const int64_t *poolSizePtr = attrs->GetInt(PKI_ATTR_POOL_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, poolSizePtr);
    int64_t poolSizeVal = *poolSizePtr;
    OP_CHECK_IF(poolSizeVal < 1 || poolSizeVal > PKI_POOL_SIZE_LIMIT,
                OP_LOGE(context_, "pool_size(%ld) must be in [1, %u]", poolSizeVal, PKI_POOL_SIZE_LIMIT),
                return ge::GRAPH_FAILED);
    info.poolSize = static_cast<uint32_t>(poolSizeVal);

    const char *layoutQPtr = attrs->GetAttrPointer<char>(PKI_ATTR_LAYOUT_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context_, layoutQPtr);
    std::string layoutQStr(layoutQPtr);
    if (layoutQStr == "BSND") {
        info.layoutQ = PkiDataLayout::BSND;
    } else if (layoutQStr == "TND") {
        info.layoutQ = PkiDataLayout::TND;
    } else {
        OP_LOGE(context_, "layout_q must be BSND or TND, got %s", layoutQStr.c_str());
        return ge::GRAPH_FAILED;
    }

    const char *layoutKPtr = attrs->GetAttrPointer<char>(PKI_ATTR_LAYOUT_K);
    OP_CHECK_NULL_WITH_CONTEXT(context_, layoutKPtr);
    std::string layoutKStr(layoutKPtr);
    if (layoutKStr == "BSND") {
        info.layoutK = PkiDataLayout::BSND;
    } else if (layoutKStr == "TND") {
        info.layoutK = PkiDataLayout::TND;
    } else if (layoutKStr == "PA_BBND") {
        info.layoutK = PkiDataLayout::PA_BBND;
        info.pageAttention = true;
    } else {
        OP_LOGE(context_, "layout_k must be BSND, TND or PA_BBND, got %s", layoutKStr.c_str());
        return ge::GRAPH_FAILED;
    }

    // ---- 1. layout_q and layout_k must match unless PA ----
    if (!info.pageAttention && info.layoutQ != info.layoutK) {
        OP_LOGE(context_, "layout_q(%s) and layout_k(%s) must match in non-PA scenario", layoutQStr.c_str(),
                layoutKStr.c_str());
        return ge::GRAPH_FAILED;
    }

    // ---- 2/3. act_seq / block_table presence rules ----
    const gert::Tensor *actSeqQ = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_Q_INDEX);
    const gert::Tensor *actSeqK = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_K_INDEX);
    const gert::Tensor *blockTable = context_->GetOptionalInputTensor(PKI_BLOCK_TABLE_INDEX);

    if (info.pageAttention) {
        // PA scenario: actual_seq_k and block_table must be non-null
        OP_CHECK_IF(actSeqK == nullptr, OP_LOGE(context_, "PA scenario, actual_seq_k must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(blockTable == nullptr, OP_LOGE(context_, "PA scenario, block_table must not be null"),
                    return ge::GRAPH_FAILED);
        if (info.layoutQ == PkiDataLayout::BSND) {
            // PA + BSND: actual_seq_q must be null
            OP_CHECK_IF(actSeqQ != nullptr, OP_LOGE(context_, "PA + BSND, actual_seq_q must be null"),
                        return ge::GRAPH_FAILED);
        } else {
            // PA + TND: actual_seq_q must be non-null
            OP_CHECK_IF(actSeqQ == nullptr, OP_LOGE(context_, "PA + TND, actual_seq_q must not be null"),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // non-PA: block_table must be null
        OP_CHECK_IF(blockTable != nullptr, OP_LOGE(context_, "non-PA scenario, block_table must be null"),
                    return ge::GRAPH_FAILED);
        if (info.layoutQ == PkiDataLayout::BSND) {
            // BSND: actual_seq_q and actual_seq_k must be null
            OP_CHECK_IF(actSeqQ != nullptr, OP_LOGE(context_, "BSND scenario, actual_seq_q must be null"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(actSeqK != nullptr, OP_LOGE(context_, "BSND scenario, actual_seq_k must be null"),
                        return ge::GRAPH_FAILED);
        } else {
            // TND: actual_seq_q and actual_seq_k must be non-null
            OP_CHECK_IF(actSeqQ == nullptr, OP_LOGE(context_, "TND scenario, actual_seq_q must not be null"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(actSeqK == nullptr, OP_LOGE(context_, "TND scenario, actual_seq_k must not be null"),
                        return ge::GRAPH_FAILED);
        }
    }

    const int64_t *maskModePtr = attrs->GetInt(PKI_ATTR_MASK_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maskModePtr);
    info.maskMode = static_cast<int32_t>(*maskModePtr);
    if (info.maskMode != 0 && info.maskMode != 3) {
        OP_LOGE(context_, "mask_mode must be 0 or 3, got %d", info.maskMode);
        return ge::GRAPH_FAILED;
    }

    const int64_t *quantModePtr = attrs->GetInt(PKI_ATTR_QUANT_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantModePtr);
    info.quantMode = static_cast<int32_t>(*quantModePtr);
    if (info.quantMode != -1 && info.quantMode != 0 && info.quantMode != 1) {
        OP_LOGE(context_, "quant_mode must be -1, 0 or 1, got %d", info.quantMode);
        return ge::GRAPH_FAILED;
    }

    const bool *returnValuePtr = attrs->GetAttrPointer<bool>(PKI_ATTR_RETURN_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, returnValuePtr);
    info.returnValue = *returnValuePtr;

    // topk % poolSize == 0
    if (info.poolSize == 0 || info.topk % info.poolSize != 0) {
        OP_LOGE(context_, "topk(%u) must be divisible by poolSize(%u)", info.topk, info.poolSize);
        return ge::GRAPH_FAILED;
    }
    info.sparseCount = info.topk / info.poolSize;

    // Parse shapes
    const gert::StorageShape *queryShape = context_->GetInputShape(PKI_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    const gert::StorageShape *keyShape = context_->GetInputShape(PKI_POOL_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);

    info.queryDtype = context_->GetInputDesc(PKI_QUERY_INDEX)->GetDataType();
    info.keyDtype = context_->GetInputDesc(PKI_POOL_KEY_INDEX)->GetDataType();
    info.weightsDtype = context_->GetInputDesc(PKI_WEIGHTS_INDEX)->GetDataType();

    // ---- Required inputs non-null and non-empty ----
    const gert::Tensor *queryTensor = context_->GetInputTensor(PKI_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryTensor);
    const gert::Tensor *poolKeyTensor = context_->GetInputTensor(PKI_POOL_KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, poolKeyTensor);
    const gert::Tensor *weightsTensor = context_->GetInputTensor(PKI_WEIGHTS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsTensor);
    const gert::Tensor *poolTailKTensor = context_->GetInputTensor(PKI_POOL_TAIL_K_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, poolTailKTensor);
    OP_CHECK_IF(queryShape->GetStorageShape().GetShapeSize() == 0, OP_LOGE(context_, "query must not be empty tensor"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape->GetStorageShape().GetShapeSize() == 0, OP_LOGE(context_, "pool_key must not be empty tensor"),
                return ge::GRAPH_FAILED);

    // 捕获 pool_key 运行时 stride(0 轴非连续支持); 优先级: key_stride0 属性 >
    // tiling context 运行时 stride(仅图模式填充, eager/aclnn 返回 nullptr)
    auto keyStrides = context_->GetInputStride(PKI_POOL_KEY_INDEX);
    if (keyStrides == nullptr) {
        keyStrides = context_->GetDynamicInputStride(PKI_POOL_KEY_INDEX, 0);
    }
    if (keyStrides != nullptr && keyStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < keyStrides->GetDimNum(); i++) {
            keyStridesVec_.push_back(keyStrides->GetStride(i));
        }
        // 图模式下属性与运行时 stride 同时存在, 必须一致(否则以错误来源寻址)
        const int64_t *keyStride0Attr = attrs->GetAttrPointer<int64_t>(PKI_ATTR_KEY_STRIDE0);
        if (keyStride0Attr != nullptr && *keyStride0Attr >= 0 &&
            keyStridesVec_[0] != static_cast<uint32_t>(*keyStride0Attr)) {
            OP_LOGE(context_, "key_stride0 attr(%ld) conflicts with runtime stride(%u)", *keyStride0Attr,
                    keyStridesVec_[0]);
            return ge::GRAPH_FAILED;
        }
    }
    // k_descale 运行时 stride 捕获(量化场景 PA 0 轴非连续, 与 pool_key 一一对应);
    // 属性与运行时 stride 同时存在时必须一致
    auto kDescaleStrides = context_->GetInputStride(PKI_K_DESCALE_INDEX);
    if (kDescaleStrides == nullptr) {
        kDescaleStrides = context_->GetDynamicInputStride(PKI_K_DESCALE_INDEX, 0);
    }
    if (kDescaleStrides != nullptr && kDescaleStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < kDescaleStrides->GetDimNum(); i++) {
            keyDequantScaleStridesVec_.push_back(kDescaleStrides->GetStride(i));
        }
        const int64_t *kdsStride0Attr = attrs->GetAttrPointer<int64_t>(PKI_ATTR_K_DESCALE_STRIDE0);
        if (kdsStride0Attr != nullptr && *kdsStride0Attr >= 0 &&
            keyDequantScaleStridesVec_[0] != static_cast<uint32_t>(*kdsStride0Attr)) {
            OP_LOGE(context_, "k_descale_stride0 attr(%ld) conflicts with runtime stride(%u)", *kdsStride0Attr,
                    keyDequantScaleStridesVec_[0]);
            return ge::GRAPH_FAILED;
        }
    }
    const gert::StorageShape *weightsShape = context_->GetInputShape(PKI_WEIGHTS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightsShape);
    OP_CHECK_IF(weightsShape->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(context_, "weights must not be empty tensor"), return ge::GRAPH_FAILED);
    const gert::StorageShape *poolTailKShape = context_->GetInputShape(PKI_POOL_TAIL_K_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, poolTailKShape);
    OP_CHECK_IF(poolTailKShape->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE(context_, "pool_tail_k must not be empty tensor"), return ge::GRAPH_FAILED);

    // ---- A2/A3 does not support FP8 ----
    if (info.npuArch != NpuArch::DAV_3510 && info.queryDtype == ge::DT_FLOAT8_E4M3FN) {
        OP_LOGE(context_, "FP8 is only supported on Ascend950 (DAV_3510)");
        return ge::GRAPH_FAILED;
    }

    // ---- query / pool_key dtype must match ----
    if (info.queryDtype != info.keyDtype) {
        OP_LOGE(context_, "query dtype(%d) must match pool_key dtype(%d)", static_cast<int32_t>(info.queryDtype),
                static_cast<int32_t>(info.keyDtype));
        return ge::GRAPH_FAILED;
    }

    // ---- quantMode-driven dtype + descale validation ----
    const gert::Tensor *qDescaleTensor = context_->GetOptionalInputTensor(PKI_Q_DESCALE_INDEX);
    const gert::Tensor *kDescaleTensor = context_->GetOptionalInputTensor(PKI_K_DESCALE_INDEX);

    // ---- FP8 量化仅 arch35 (Ascend 950) 支持: 架构门禁 ----
    // arch22 (A2/A3) 硬件不支持 FP8, 必须在 tiling 层拒绝而非静默产出错误结果
    if ((info.quantMode == PKI_QUANT_FP8_PER_TOKEN || info.quantMode == PKI_QUANT_MXFP8) &&
        info.npuArch != NpuArch::DAV_3510) {
        OP_LOGE(context_, "quant_mode=%d is only supported on Ascend950 (DAV_3510), current arch=%d", info.quantMode,
                static_cast<int32_t>(info.npuArch));
        return ge::GRAPH_FAILED;
    }

    if (info.quantMode == PKI_QUANT_NONE) {
        // quantMode=-1: no quantization
        // descale must be null
        OP_CHECK_IF(qDescaleTensor != nullptr, OP_LOGE(context_, "quant_mode=-1, q_descale must be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kDescaleTensor != nullptr, OP_LOGE(context_, "quant_mode=-1, k_descale must be null"),
                    return ge::GRAPH_FAILED);
        // q/k must be FP16 or BF16
        OP_CHECK_IF(info.queryDtype != ge::DT_FLOAT16 && info.queryDtype != ge::DT_BF16,
                    OP_LOGE(context_, "quant_mode=-1, query dtype must be FP16 or BF16, got %d",
                            static_cast<int32_t>(info.queryDtype)),
                    return ge::GRAPH_FAILED);
        // weights must match query
        OP_CHECK_IF(info.weightsDtype != info.queryDtype,
                    OP_LOGE(context_, "quant_mode=-1, weights dtype must match query dtype"), return ge::GRAPH_FAILED);
    } else if (info.quantMode == PKI_QUANT_FP8_PER_TOKEN) {
        // quantMode=0: FP8 per-token-head
        // descale must be non-null
        OP_CHECK_IF(qDescaleTensor == nullptr, OP_LOGE(context_, "quant_mode=0, q_descale must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kDescaleTensor == nullptr, OP_LOGE(context_, "quant_mode=0, k_descale must not be null"),
                    return ge::GRAPH_FAILED);
        // q/k must be FP8_E4M3FN
        OP_CHECK_IF(info.queryDtype != ge::DT_FLOAT8_E4M3FN,
                    OP_LOGE(context_, "quant_mode=0, query dtype must be FP8_E4M3FN, got %d",
                            static_cast<int32_t>(info.queryDtype)),
                    return ge::GRAPH_FAILED);
        // descale must be FLOAT
        ge::DataType qScaleDtype = context_->GetOptionalInputDesc(PKI_Q_DESCALE_INDEX)->GetDataType();
        ge::DataType kScaleDtype = context_->GetOptionalInputDesc(PKI_K_DESCALE_INDEX)->GetDataType();
        OP_CHECK_IF(
            qScaleDtype != ge::DT_FLOAT,
            OP_LOGE(context_, "quant_mode=0, q_descale dtype must be FLOAT, got %d", static_cast<int32_t>(qScaleDtype)),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            kScaleDtype != ge::DT_FLOAT,
            OP_LOGE(context_, "quant_mode=0, k_descale dtype must be FLOAT, got %d", static_cast<int32_t>(kScaleDtype)),
            return ge::GRAPH_FAILED);
        // weights must be FP16 or BF16
        OP_CHECK_IF(info.weightsDtype != ge::DT_FLOAT16 && info.weightsDtype != ge::DT_BF16,
                    OP_LOGE(context_, "quant_mode=0, weights must be FP16 or BF16, got %d",
                            static_cast<int32_t>(info.weightsDtype)),
                    return ge::GRAPH_FAILED);
    } else if (info.quantMode == PKI_QUANT_MXFP8) {
        // quantMode=1: mxFP8
        // descale must be non-null
        OP_CHECK_IF(qDescaleTensor == nullptr, OP_LOGE(context_, "quant_mode=1, q_descale must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kDescaleTensor == nullptr, OP_LOGE(context_, "quant_mode=1, k_descale must not be null"),
                    return ge::GRAPH_FAILED);
        // q/k must be FP8_E4M3FN
        OP_CHECK_IF(info.queryDtype != ge::DT_FLOAT8_E4M3FN,
                    OP_LOGE(context_, "quant_mode=1, query dtype must be FP8_E4M3FN, got %d",
                            static_cast<int32_t>(info.queryDtype)),
                    return ge::GRAPH_FAILED);
        // descale must be FLOAT8_E8M0
        ge::DataType qScaleDtype = context_->GetOptionalInputDesc(PKI_Q_DESCALE_INDEX)->GetDataType();
        ge::DataType kScaleDtype = context_->GetOptionalInputDesc(PKI_K_DESCALE_INDEX)->GetDataType();
        OP_CHECK_IF(qScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(context_, "quant_mode=1, q_descale dtype must be FLOAT8_E8M0, got %d",
                            static_cast<int32_t>(qScaleDtype)),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(kScaleDtype != ge::DT_FLOAT8_E8M0,
                    OP_LOGE(context_, "quant_mode=1, k_descale dtype must be FLOAT8_E8M0, got %d",
                            static_cast<int32_t>(kScaleDtype)),
                    return ge::GRAPH_FAILED);
        // weights must be FP16 or BF16
        OP_CHECK_IF(info.weightsDtype != ge::DT_FLOAT16 && info.weightsDtype != ge::DT_BF16,
                    OP_LOGE(context_, "quant_mode=1, weights must be FP16 or BF16, got %d",
                            static_cast<int32_t>(info.weightsDtype)),
                    return ge::GRAPH_FAILED);
    }

    // N1, N2, D from query shape
    if (info.layoutQ == PkiDataLayout::BSND) {
        OP_CHECK_IF(queryShape->GetStorageShape().GetDimNum() != 4,
                    OP_LOGE(context_, "BSND query must be 4D, got %zu", queryShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
        info.bSize = queryShape->GetStorageShape().GetDim(0);
        info.s1Size = queryShape->GetStorageShape().GetDim(1);
        info.n1Size = queryShape->GetStorageShape().GetDim(2);
        info.headDim = queryShape->GetStorageShape().GetDim(3);
    } else {
        OP_CHECK_IF(queryShape->GetStorageShape().GetDimNum() != 3,
                    OP_LOGE(context_, "TND query must be 3D, got %zu", queryShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
        info.bSize = 0;                                        // determined from actual_seq_q
        info.s1Size = queryShape->GetStorageShape().GetDim(0); // T1
        info.n1Size = queryShape->GetStorageShape().GetDim(1);
        info.headDim = queryShape->GetStorageShape().GetDim(2);
    }

    info.n2Size = PKI_N2_FIXED;
    if (info.n1Size > PKI_N1_LIMIT) {
        OP_LOGE(context_, "N1(%u) must be <= %u", info.n1Size, PKI_N1_LIMIT);
        return ge::GRAPH_FAILED;
    }
    if (info.headDim != PKI_HEAD_DIM) {
        OP_LOGE(context_, "headDim(%u) must be 128", info.headDim);
        return ge::GRAPH_FAILED;
    }
    // mxFP8 尾维 [D/64, 2] 要求 headDim 是 64 的倍数
    if (info.quantMode == PKI_QUANT_MXFP8 && info.headDim % PKI_MX_SCALE_SHAPE_ALIGN != 0) {
        OP_LOGE(context_, "quant_mode=1 (mxFP8) requires headDim %% %u == 0, got %u", PKI_MX_SCALE_SHAPE_ALIGN,
                info.headDim);
        return ge::GRAPH_FAILED;
    }
    info.gSize = info.n1Size / info.n2Size;

    // S2 from key shape
    if (info.pageAttention) {
        OP_CHECK_IF(keyShape->GetStorageShape().GetDimNum() != 4,
                    OP_LOGE(context_, "PA_BBND key must be 4D (blockNum, blockSize, N2, D)"), return ge::GRAPH_FAILED);
        info.blockSize = keyShape->GetStorageShape().GetDim(1);
        if (info.blockSize == 0 || info.blockSize % PKI_BLOCK_SIZE_FACTOR != 0 ||
            info.blockSize > PKI_BLOCK_SIZE_LIMIT) {
            OP_LOGE(context_, "blockSize(%u) must be multiple of %u and <= %u", info.blockSize, PKI_BLOCK_SIZE_FACTOR,
                    PKI_BLOCK_SIZE_LIMIT);
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(2)) != info.n2Size,
                    OP_LOGE(context_, "PA key N2(%ld) must be %u", keyShape->GetStorageShape().GetDim(2), info.n2Size),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(3)) != info.headDim,
            OP_LOGE(context_, "PA key D(%ld) must be %u(128)", keyShape->GetStorageShape().GetDim(3), info.headDim),
            return ge::GRAPH_FAILED);
        // keyStride0 优先级: 属性 key_stride0(eager 路径来源) >
        // 运行时 stride(图模式) > 连续推导 blockSize*N2*headDim
        const int64_t *keyStride0Attr = attrs->GetAttrPointer<int64_t>(PKI_ATTR_KEY_STRIDE0);
        if (keyStride0Attr != nullptr && *keyStride0Attr >= 0) {
            uint64_t contiguousStride0 = static_cast<uint64_t>(info.blockSize) * info.n2Size * info.headDim;
            OP_CHECK_IF(static_cast<uint64_t>(*keyStride0Attr) < contiguousStride0,
                        OP_LOGE(context_, "key_stride0(%ld) must be >= contiguous stride(%lu)", *keyStride0Attr,
                                contiguousStride0),
                        return ge::GRAPH_FAILED);
            info.keyStride0 = static_cast<uint32_t>(*keyStride0Attr);
        } else if (!keyStridesVec_.empty() && keyStridesVec_[0] != 0) {
            info.keyStride0 = keyStridesVec_[0];
        } else {
            info.keyStride0 = info.blockSize * info.n2Size * info.headDim;
        }
        // S2 = maxBlockNumPerBatch * blockSize, 在 block_table 解析处设置
        // (kernel kSeqSize 用作 scoreGm 行距, 不可为 0)
        info.s2Size = 0;
        // k_descale stride0 接线(量化场景): 优先级同 key_stride0;
        // 连续 stride0: mode=0=blockSize*N2, mode=1=blockSize*N2*(D/32); 非量化保持 0
        if (info.quantMode == PKI_QUANT_FP8_PER_TOKEN || info.quantMode == PKI_QUANT_MXFP8) {
            const int64_t *kdsStride0Attr = attrs->GetAttrPointer<int64_t>(PKI_ATTR_K_DESCALE_STRIDE0);
            if (kdsStride0Attr != nullptr && *kdsStride0Attr >= 0) {
                // 属性指定: 校验不小于连续 stride(0 轴非连续只允许加 padding)
                uint64_t contigKdsStride0 =
                    (info.quantMode == PKI_QUANT_MXFP8) ?
                        static_cast<uint64_t>(info.blockSize) * info.n2Size * (info.headDim / PKI_MX_SCALE_GROUP_SIZE) :
                        static_cast<uint64_t>(info.blockSize) * info.n2Size;
                OP_CHECK_IF(static_cast<uint64_t>(*kdsStride0Attr) < contigKdsStride0,
                            OP_LOGE(context_, "k_descale_stride0(%ld) must be >= contiguous stride(%lu)",
                                    *kdsStride0Attr, contigKdsStride0),
                            return ge::GRAPH_FAILED);
                info.keyDequantScaleStride0 = static_cast<uint32_t>(*kdsStride0Attr);
            } else if (!keyDequantScaleStridesVec_.empty() && keyDequantScaleStridesVec_[0] != 0) {
                info.keyDequantScaleStride0 = keyDequantScaleStridesVec_[0];
            } else if (info.quantMode == PKI_QUANT_MXFP8) {
                info.keyDequantScaleStride0 = info.blockSize * info.n2Size * (info.headDim / PKI_MX_SCALE_GROUP_SIZE);
            } else {
                info.keyDequantScaleStride0 = info.blockSize * info.n2Size;
            }
        }
    } else if (info.layoutK == PkiDataLayout::BSND) {
        OP_CHECK_IF(keyShape->GetStorageShape().GetDimNum() != 4, OP_LOGE(context_, "BSND key must be 4D"),
                    return ge::GRAPH_FAILED);
        info.s2Size = keyShape->GetStorageShape().GetDim(1);
        if (info.n2Size != keyShape->GetStorageShape().GetDim(2)) {
            OP_LOGE(context_, "N2 mismatch: query N2=%u, key N2=%ld", info.n2Size,
                    keyShape->GetStorageShape().GetDim(2));
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(
            static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(3)) != info.headDim,
            OP_LOGE(context_, "BSND key D(%ld) must be %u(128)", keyShape->GetStorageShape().GetDim(3), info.headDim),
            return ge::GRAPH_FAILED);
        // BSND: kernel 按 tensorKeyOffset 连续寻址, stride0 属性仅 PA 分支消费;
        // 非 PA 下显式指定说明输入非连续, 显式拒绝
        {
            const int64_t *ks0Attr = attrs->GetAttrPointer<int64_t>(PKI_ATTR_KEY_STRIDE0);
            OP_CHECK_IF(ks0Attr != nullptr && *ks0Attr >= 0,
                        OP_LOGE(context_,
                                "key_stride0(%ld) is only supported with layout_k=PA_BBND, "
                                "but layout_k is BSND; pass a contiguous pool_key",
                                *ks0Attr),
                        return ge::GRAPH_FAILED);
            const int64_t *kds0Attr = attrs->GetAttrPointer<int64_t>(PKI_ATTR_K_DESCALE_STRIDE0);
            OP_CHECK_IF(kds0Attr != nullptr && *kds0Attr >= 0,
                        OP_LOGE(context_,
                                "k_descale_stride0(%ld) is only supported with layout_k=PA_BBND, "
                                "but layout_k is BSND; pass a contiguous k_descale",
                                *kds0Attr),
                        return ge::GRAPH_FAILED);
        }
        info.keyStride0 = 0;
    } else {
        OP_CHECK_IF(keyShape->GetStorageShape().GetDimNum() != 3, OP_LOGE(context_, "TND key must be 3D"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(1)) != info.n2Size,
                    OP_LOGE(context_, "TND key N2(%ld) must be %u", keyShape->GetStorageShape().GetDim(1), info.n2Size),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(2)) != info.headDim,
            OP_LOGE(context_, "TND key D(%ld) must be %u(128)", keyShape->GetStorageShape().GetDim(2), info.headDim),
            return ge::GRAPH_FAILED);
        // T2 = 累计 pool 总数; kSeqSize 用作 scoreGm 行距, 取 0 会导致行间覆盖
        info.s2Size = static_cast<uint32_t>(keyShape->GetStorageShape().GetDim(0));
        // TND: kernel 按 tensorKeyOffset 连续寻址, stride0 属性仅 PA 分支消费;
        // 非 PA 下显式指定说明输入非连续, 显式拒绝
        {
            const int64_t *ks0Attr = attrs->GetAttrPointer<int64_t>(PKI_ATTR_KEY_STRIDE0);
            OP_CHECK_IF(ks0Attr != nullptr && *ks0Attr >= 0,
                        OP_LOGE(context_,
                                "key_stride0(%ld) is only supported with layout_k=PA_BBND, "
                                "but layout_k is TND; pass a contiguous pool_key",
                                *ks0Attr),
                        return ge::GRAPH_FAILED);
            const int64_t *kds0Attr = attrs->GetAttrPointer<int64_t>(PKI_ATTR_K_DESCALE_STRIDE0);
            OP_CHECK_IF(kds0Attr != nullptr && *kds0Attr >= 0,
                        OP_LOGE(context_,
                                "k_descale_stride0(%ld) is only supported with layout_k=PA_BBND, "
                                "but layout_k is TND; pass a contiguous k_descale",
                                *kds0Attr),
                        return ge::GRAPH_FAILED);
        }
        info.keyStride0 = 0;
    }
    // block_table for PA
    if (info.pageAttention) {
        const gert::Tensor *blockTableTensor = context_->GetOptionalInputTensor(PKI_BLOCK_TABLE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, blockTableTensor);
        const gert::StorageShape *btShape = context_->GetOptionalInputShape(PKI_BLOCK_TABLE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, btShape);
        OP_CHECK_IF(btShape->GetStorageShape().GetDimNum() != 2,
                    OP_LOGE(context_, "block_table must be 2D (B, maxBlockNumPerSeq)"), return ge::GRAPH_FAILED);
        info.maxBlockNumPerBatch = btShape->GetStorageShape().GetDim(1);
        // S2 = maxBlockNumPerBatch * blockSize(每 batch 最大 pool 数上界);
        // kSeqSize 用作 scoreGm 行距, 取 0 会导致行间覆盖
        info.s2Size = info.maxBlockNumPerBatch * info.blockSize;
    }

    // ---- Tensor shape validation ----

    // weights shape: BSND (B,S1,N1) 3D / TND (T1,N1) 2D
    {
        const gert::StorageShape *wShape = context_->GetInputShape(PKI_WEIGHTS_INDEX);
        if (info.layoutQ == PkiDataLayout::BSND) {
            OP_CHECK_IF(
                wShape->GetStorageShape().GetDimNum() != 3,
                OP_LOGE(context_, "BSND weights must be 3D (B,S1,N1), got %zu", wShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(wShape->GetStorageShape().GetDim(0) != info.bSize,
                        OP_LOGE(context_, "weights B(%ld) must match query B(%u)", wShape->GetStorageShape().GetDim(0),
                                info.bSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(wShape->GetStorageShape().GetDim(1) != info.s1Size,
                        OP_LOGE(context_, "weights S1(%ld) must match query S1(%u)",
                                wShape->GetStorageShape().GetDim(1), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(wShape->GetStorageShape().GetDim(2) != info.n1Size,
                        OP_LOGE(context_, "weights N1(%ld) must match query N1(%u)",
                                wShape->GetStorageShape().GetDim(2), info.n1Size),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(
                wShape->GetStorageShape().GetDimNum() != 2,
                OP_LOGE(context_, "TND weights must be 2D (T1,N1), got %zu", wShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(wShape->GetStorageShape().GetDim(0) != info.s1Size,
                        OP_LOGE(context_, "weights T1(%ld) must match query T1(%u)",
                                wShape->GetStorageShape().GetDim(0), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(wShape->GetStorageShape().GetDim(1) != info.n1Size,
                        OP_LOGE(context_, "weights N1(%ld) must match query N1(%u)",
                                wShape->GetStorageShape().GetDim(1), info.n1Size),
                        return ge::GRAPH_FAILED);
        }
    }

    // pool_tail_k shape: (B,) 1D
    {
        const gert::StorageShape *ptkShape = context_->GetInputShape(PKI_POOL_TAIL_K_INDEX);
        OP_CHECK_IF(ptkShape->GetStorageShape().GetDimNum() != 1,
                    OP_LOGE(context_, "pool_tail_k must be 1D (B,), got %zu", ptkShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
        // B count checked in value-range validation below
    }

    // actual_seq_q shape (non-null, TND only): (B,) 1D prefix sum
    {
        const gert::Tensor *actSeqQT = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_Q_INDEX);
        if (actSeqQT != nullptr) {
            const gert::StorageShape *asqShape = context_->GetOptionalInputShape(PKI_ACTUAL_SEQ_Q_INDEX);
            OP_CHECK_NULL_WITH_CONTEXT(context_, asqShape);
            OP_CHECK_IF(asqShape->GetStorageShape().GetDimNum() != 1,
                        OP_LOGE(context_, "actual_seq_q must be 1D, got %zu", asqShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
        }
    }

    // actual_seq_k shape (non-null, TND/PA): (B,) 1D
    {
        const gert::Tensor *actSeqKT = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_K_INDEX);
        if (actSeqKT != nullptr) {
            const gert::StorageShape *askShape = context_->GetOptionalInputShape(PKI_ACTUAL_SEQ_K_INDEX);
            OP_CHECK_NULL_WITH_CONTEXT(context_, askShape);
            OP_CHECK_IF(askShape->GetStorageShape().GetDimNum() != 1,
                        OP_LOGE(context_, "actual_seq_k must be 1D, got %zu", askShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
        }
    }

    // q_descale shape (non-null, quantMode>=0): mode=0 为 BSND 3D / TND 2D;
    // mode=1 (mxFP8) 追加 [D/64, 2] 尾维 -> BSND 5D / TND 4D
    if (qDescaleTensor != nullptr) {
        const gert::StorageShape *qdsShape = context_->GetOptionalInputShape(PKI_Q_DESCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, qdsShape);
        if (info.layoutQ == PkiDataLayout::BSND) {
            size_t expectDims = (info.quantMode == PKI_QUANT_MXFP8) ? 5 : 3;
            OP_CHECK_IF(qdsShape->GetStorageShape().GetDimNum() != expectDims,
                        OP_LOGE(context_, "BSND q_descale must be %zuD, got %zu (quant_mode=%d)", expectDims,
                                qdsShape->GetStorageShape().GetDimNum(), info.quantMode),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(qdsShape->GetStorageShape().GetDim(0) != info.bSize,
                        OP_LOGE(context_, "q_descale B(%ld) must match query B(%u)",
                                qdsShape->GetStorageShape().GetDim(0), info.bSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(qdsShape->GetStorageShape().GetDim(1) != info.s1Size,
                        OP_LOGE(context_, "q_descale S1(%ld) must match query S1(%u)",
                                qdsShape->GetStorageShape().GetDim(1), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(qdsShape->GetStorageShape().GetDim(2) != info.n1Size,
                        OP_LOGE(context_, "q_descale N1(%ld) must match query N1(%u)",
                                qdsShape->GetStorageShape().GetDim(2), info.n1Size),
                        return ge::GRAPH_FAILED);
            if (info.quantMode == PKI_QUANT_MXFP8) {
                // mxFP8 尾维 [D/64, 2] 校验
                uint32_t expectScaleD = info.headDim / PKI_MX_SCALE_SHAPE_ALIGN;
                OP_CHECK_IF(
                    qdsShape->GetStorageShape().GetDim(3) != static_cast<int64_t>(expectScaleD) ||
                        qdsShape->GetStorageShape().GetDim(4) != static_cast<int64_t>(PKI_MX_E8M0_SCALE_PACK_NUM),
                    OP_LOGE(context_, "mxFP8 q_descale last dims must be [%u, %u]", expectScaleD,
                            PKI_MX_E8M0_SCALE_PACK_NUM),
                    return ge::GRAPH_FAILED);
            }
        } else {
            size_t expectDims = (info.quantMode == PKI_QUANT_MXFP8) ? 4 : 2;
            OP_CHECK_IF(qdsShape->GetStorageShape().GetDimNum() != expectDims,
                        OP_LOGE(context_, "TND q_descale must be %zuD, got %zu (quant_mode=%d)", expectDims,
                                qdsShape->GetStorageShape().GetDimNum(), info.quantMode),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(qdsShape->GetStorageShape().GetDim(0) != info.s1Size,
                        OP_LOGE(context_, "q_descale T1(%ld) must match query T1(%u)",
                                qdsShape->GetStorageShape().GetDim(0), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(qdsShape->GetStorageShape().GetDim(1) != info.n1Size,
                        OP_LOGE(context_, "q_descale N1(%ld) must match query N1(%u)",
                                qdsShape->GetStorageShape().GetDim(1), info.n1Size),
                        return ge::GRAPH_FAILED);
            if (info.quantMode == PKI_QUANT_MXFP8) {
                uint32_t expectScaleD = info.headDim / PKI_MX_SCALE_SHAPE_ALIGN;
                OP_CHECK_IF(
                    qdsShape->GetStorageShape().GetDim(2) != static_cast<int64_t>(expectScaleD) ||
                        qdsShape->GetStorageShape().GetDim(3) != static_cast<int64_t>(PKI_MX_E8M0_SCALE_PACK_NUM),
                    OP_LOGE(context_, "mxFP8 q_descale last dims must be [%u, %u]", expectScaleD,
                            PKI_MX_E8M0_SCALE_PACK_NUM),
                    return ge::GRAPH_FAILED);
            }
        }
    }

    // k_descale shape (non-null, quantMode>=0): mode=0 为 BSND/PA 3D 或 TND 2D;
    // mode=1 (mxFP8) 追加 [D/64, 2] 尾维 -> BSND/PA 5D / TND 4D
    if (kDescaleTensor != nullptr) {
        const gert::StorageShape *kdsShape = context_->GetOptionalInputShape(PKI_K_DESCALE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kdsShape);
        if (info.pageAttention) {
            size_t expectDims = (info.quantMode == PKI_QUANT_MXFP8) ? 5 : 3;
            OP_CHECK_IF(kdsShape->GetStorageShape().GetDimNum() != expectDims,
                        OP_LOGE(context_, "PA k_descale must be %zuD, got %zu (quant_mode=%d)", expectDims,
                                kdsShape->GetStorageShape().GetDimNum(), info.quantMode),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetStorageShape().GetDim(1) != info.blockSize,
                        OP_LOGE(context_, "k_descale blockSize(%ld) must match key blockSize(%u)",
                                kdsShape->GetStorageShape().GetDim(1), info.blockSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetStorageShape().GetDim(2) != info.n2Size,
                        OP_LOGE(context_, "k_descale N2(%ld) must match key N2(%u)",
                                kdsShape->GetStorageShape().GetDim(2), info.n2Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetStorageShape().GetDim(0) != keyShape->GetStorageShape().GetDim(0),
                        OP_LOGE(context_, "PA k_descale blockNum(%ld) must match key blockNum(%ld)",
                                kdsShape->GetStorageShape().GetDim(0), keyShape->GetStorageShape().GetDim(0)),
                        return ge::GRAPH_FAILED);
            if (info.quantMode == PKI_QUANT_MXFP8) {
                uint32_t expectScaleD = info.headDim / PKI_MX_SCALE_SHAPE_ALIGN;
                OP_CHECK_IF(
                    kdsShape->GetStorageShape().GetDim(3) != static_cast<int64_t>(expectScaleD) ||
                        kdsShape->GetStorageShape().GetDim(4) != static_cast<int64_t>(PKI_MX_E8M0_SCALE_PACK_NUM),
                    OP_LOGE(context_, "mxFP8 k_descale last dims must be [%u, %u]", expectScaleD,
                            PKI_MX_E8M0_SCALE_PACK_NUM),
                    return ge::GRAPH_FAILED);
            }
        } else if (info.layoutK == PkiDataLayout::BSND) {
            size_t expectDims = (info.quantMode == PKI_QUANT_MXFP8) ? 5 : 3;
            OP_CHECK_IF(kdsShape->GetStorageShape().GetDimNum() != expectDims,
                        OP_LOGE(context_, "BSND k_descale must be %zuD, got %zu (quant_mode=%d)", expectDims,
                                kdsShape->GetStorageShape().GetDimNum(), info.quantMode),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetStorageShape().GetDim(0) != info.bSize,
                        OP_LOGE(context_, "k_descale B(%ld) must match query B(%u)",
                                kdsShape->GetStorageShape().GetDim(0), info.bSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetStorageShape().GetDim(1) != info.s2Size,
                        OP_LOGE(context_, "BSND k_descale S2(%ld) must match key S2(%ld)",
                                kdsShape->GetStorageShape().GetDim(1), info.s2Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetStorageShape().GetDim(2) != info.n2Size,
                        OP_LOGE(context_, "k_descale N2(%ld) must match key N2(%u)",
                                kdsShape->GetStorageShape().GetDim(2), info.n2Size),
                        return ge::GRAPH_FAILED);
            if (info.quantMode == PKI_QUANT_MXFP8) {
                uint32_t expectScaleD = info.headDim / PKI_MX_SCALE_SHAPE_ALIGN;
                OP_CHECK_IF(
                    kdsShape->GetStorageShape().GetDim(3) != static_cast<int64_t>(expectScaleD) ||
                        kdsShape->GetStorageShape().GetDim(4) != static_cast<int64_t>(PKI_MX_E8M0_SCALE_PACK_NUM),
                    OP_LOGE(context_, "mxFP8 k_descale last dims must be [%u, %u]", expectScaleD,
                            PKI_MX_E8M0_SCALE_PACK_NUM),
                    return ge::GRAPH_FAILED);
            }
        } else {
            size_t expectDims = (info.quantMode == PKI_QUANT_MXFP8) ? 4 : 2;
            OP_CHECK_IF(kdsShape->GetStorageShape().GetDimNum() != expectDims,
                        OP_LOGE(context_, "TND k_descale must be %zuD, got %zu (quant_mode=%d)", expectDims,
                                kdsShape->GetStorageShape().GetDimNum(), info.quantMode),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetStorageShape().GetDim(0) != keyShape->GetStorageShape().GetDim(0),
                        OP_LOGE(context_, "TND k_descale T2(%ld) must match key T2(%ld)",
                                kdsShape->GetStorageShape().GetDim(0), keyShape->GetStorageShape().GetDim(0)),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(kdsShape->GetStorageShape().GetDim(1) != info.n2Size,
                        OP_LOGE(context_, "k_descale N2(%ld) must match key N2(%u)",
                                kdsShape->GetStorageShape().GetDim(1), info.n2Size),
                        return ge::GRAPH_FAILED);
            if (info.quantMode == PKI_QUANT_MXFP8) {
                uint32_t expectScaleD = info.headDim / PKI_MX_SCALE_SHAPE_ALIGN;
                OP_CHECK_IF(
                    kdsShape->GetStorageShape().GetDim(2) != static_cast<int64_t>(expectScaleD) ||
                        kdsShape->GetStorageShape().GetDim(3) != static_cast<int64_t>(PKI_MX_E8M0_SCALE_PACK_NUM),
                    OP_LOGE(context_, "mxFP8 k_descale last dims must be [%u, %u]", expectScaleD,
                            PKI_MX_E8M0_SCALE_PACK_NUM),
                    return ge::GRAPH_FAILED);
            }
        }
    }

    // sparse_indices_out shape: BSND (B,S1,topk+poolSize-1) 3D / TND (T1,topk+poolSize-1) 2D
    {
        const gert::StorageShape *idxOutShape = context_->GetOutputShape(PKI_SPARSE_INDICES_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, idxOutShape);
        int64_t outLastDim = static_cast<int64_t>(info.topk) + info.poolSize - 1;
        if (info.layoutQ == PkiDataLayout::BSND) {
            OP_CHECK_IF(idxOutShape->GetStorageShape().GetDimNum() != 3,
                        OP_LOGE(context_, "BSND sparse_indices_out must be 3D, got %zu",
                                idxOutShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(idxOutShape->GetStorageShape().GetDim(0) != info.bSize,
                        OP_LOGE(context_, "sparse_indices_out B(%ld) must match query B(%u)",
                                idxOutShape->GetStorageShape().GetDim(0), info.bSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(idxOutShape->GetStorageShape().GetDim(1) != info.s1Size,
                        OP_LOGE(context_, "sparse_indices_out S1(%ld) must match query S1(%u)",
                                idxOutShape->GetStorageShape().GetDim(1), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(idxOutShape->GetStorageShape().GetDim(2) != outLastDim,
                        OP_LOGE(context_, "sparse_indices_out last dim(%ld) must be topk+poolSize-1(%ld)",
                                idxOutShape->GetStorageShape().GetDim(2), outLastDim),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(idxOutShape->GetStorageShape().GetDimNum() != 2,
                        OP_LOGE(context_, "TND sparse_indices_out must be 2D, got %zu",
                                idxOutShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(idxOutShape->GetStorageShape().GetDim(0) != info.s1Size,
                        OP_LOGE(context_, "sparse_indices_out T1(%ld) must match query T1(%u)",
                                idxOutShape->GetStorageShape().GetDim(0), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(idxOutShape->GetStorageShape().GetDim(1) != outLastDim,
                        OP_LOGE(context_, "sparse_indices_out last dim(%ld) must be topk+poolSize-1(%ld)",
                                idxOutShape->GetStorageShape().GetDim(1), outLastDim),
                        return ge::GRAPH_FAILED);
        }
    }

    // sparse_values_out shape (returnValue=true): BSND (B,S1,topk//poolSize) 3D / TND (T1,topk//poolSize) 2D
    if (info.returnValue) {
        const gert::StorageShape *valOutShape = context_->GetOutputShape(PKI_SPARSE_VALUES_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, valOutShape);
        int64_t valLastDim = static_cast<int64_t>(info.sparseCount);
        if (info.layoutQ == PkiDataLayout::BSND) {
            OP_CHECK_IF(valOutShape->GetStorageShape().GetDimNum() != 3,
                        OP_LOGE(context_, "BSND sparse_values_out must be 3D, got %zu",
                                valOutShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(valOutShape->GetStorageShape().GetDim(0) != info.bSize,
                        OP_LOGE(context_, "sparse_values_out B(%ld) must match query B(%u)",
                                valOutShape->GetStorageShape().GetDim(0), info.bSize),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(valOutShape->GetStorageShape().GetDim(1) != info.s1Size,
                        OP_LOGE(context_, "sparse_values_out S1(%ld) must match query S1(%u)",
                                valOutShape->GetStorageShape().GetDim(1), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(valOutShape->GetStorageShape().GetDim(2) != valLastDim,
                        OP_LOGE(context_, "sparse_values_out last dim(%ld) must be topk/poolSize(%ld)",
                                valOutShape->GetStorageShape().GetDim(2), valLastDim),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(valOutShape->GetStorageShape().GetDimNum() != 2,
                        OP_LOGE(context_, "TND sparse_values_out must be 2D, got %zu",
                                valOutShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(valOutShape->GetStorageShape().GetDim(0) != info.s1Size,
                        OP_LOGE(context_, "sparse_values_out T1(%ld) must match query T1(%u)",
                                valOutShape->GetStorageShape().GetDim(0), info.s1Size),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(valOutShape->GetStorageShape().GetDim(1) != valLastDim,
                        OP_LOGE(context_, "sparse_values_out last dim(%ld) must be topk/poolSize(%ld)",
                                valOutShape->GetStorageShape().GetDim(1), valLastDim),
                        return ge::GRAPH_FAILED);
        }
    }

    // ---- Value-range validation (requires ValueDepend + support_aclnn) ----

    // #5/#6: actual_seq_q prefix-sum (TND): determines B, validates non-decreasing + last==T1
    if (info.layoutQ == PkiDataLayout::TND) {
        const gert::Tensor *asqTensor = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_Q_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, asqTensor);
        // B is derived from the shape (element count), available in both eager and
        // GE graph mode.
        int64_t asqCount = asqTensor->GetShapeSize();
        info.bSize = static_cast<uint32_t>(asqCount);
        const int64_t *asqData = IsHostVisibleValue(asqTensor) ? asqTensor->GetData<int64_t>() : nullptr;
        if (asqData == nullptr) {
            // Device tensor(GE 图模式 / eager Tensor 变体 NPU 输入): tiling 期值不可见。
            // 跳过 host 侧值校验; kernel 运行期从 GM 读前缀和(GetActualSeqLen)。
            OP_LOGI(context_, "TND: actual_seq_q data is not host-visible (device tensor), skip value validation");
        } else {
            int64_t prev = 0;
            for (int64_t i = 0; i < asqCount; ++i) {
                OP_CHECK_IF(asqData[i] < 0, OP_LOGE(context_, "actual_seq_q[%ld]=%ld must be >= 0", i, asqData[i]),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(
                    asqData[i] < prev,
                    OP_LOGE(context_, "actual_seq_q must be non-decreasing, [%ld]=%ld < prev=%ld", i, asqData[i], prev),
                    return ge::GRAPH_FAILED);
                prev = asqData[i];
            }
            if (asqCount > 0) {
                OP_CHECK_IF(asqData[asqCount - 1] != static_cast<int64_t>(info.s1Size),
                            OP_LOGE(context_, "actual_seq_q last(%ld) must equal query T1(%u)", asqData[asqCount - 1],
                                    info.s1Size),
                            return ge::GRAPH_FAILED);
            }
        }
    }

    // #5: actual_seq_k values: TND prefix-sum / PA non-prefix-sum
    if (info.layoutK == PkiDataLayout::TND || info.pageAttention) {
        const gert::Tensor *askTensor = context_->GetOptionalInputTensor(PKI_ACTUAL_SEQ_K_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, askTensor);
        const int64_t *askData = IsHostVisibleValue(askTensor) ? askTensor->GetData<int64_t>() : nullptr;
        int64_t askCount = askTensor->GetShapeSize();
        OP_CHECK_IF(static_cast<uint32_t>(askCount) != info.bSize,
                    OP_LOGE(context_, "actual_seq_k count(%ld) must equal B(%u)", askCount, info.bSize),
                    return ge::GRAPH_FAILED);

        if (askData == nullptr) {
            // Device tensor: tiling 期值不可见, 跳过 host 侧值校验,
            // kernel 运行期从 GM 读值; PA maxBlockNumPerSeq 检查同样推迟
            OP_LOGI(context_, "actual_seq_k data is not host-visible (device tensor), skip value validation");
        } else if (info.layoutK == PkiDataLayout::TND) {
            int64_t prev = 0;
            for (int64_t i = 0; i < askCount; ++i) {
                OP_CHECK_IF(askData[i] < 0, OP_LOGE(context_, "actual_seq_k[%ld]=%ld must be >= 0", i, askData[i]),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(
                    askData[i] < prev,
                    OP_LOGE(context_, "actual_seq_k must be non-decreasing, [%ld]=%ld < prev=%ld", i, askData[i], prev),
                    return ge::GRAPH_FAILED);
                prev = askData[i];
            }
            if (askCount > 0) {
                OP_CHECK_IF(askData[askCount - 1] != static_cast<int64_t>(keyShape->GetStorageShape().GetDim(0)),
                            OP_LOGE(context_, "actual_seq_k last(%ld) must equal key T2(%ld)", askData[askCount - 1],
                                    keyShape->GetStorageShape().GetDim(0)),
                            return ge::GRAPH_FAILED);
            }
        } else {
            // PA: non-prefix-sum, each value is current batch token count
            uint32_t maxTokens = 0;
            for (int64_t i = 0; i < askCount; ++i) {
                OP_CHECK_IF(askData[i] < 0, OP_LOGE(context_, "actual_seq_k[%ld]=%ld must be >= 0", i, askData[i]),
                            return ge::GRAPH_FAILED);
                if (static_cast<uint32_t>(askData[i]) > maxTokens) {
                    maxTokens = static_cast<uint32_t>(askData[i]);
                }
            }
            // #3: block_table dim(1) >= maxBlockNumPerSeq
            uint32_t maxBlockNumPerSeq = CeilDiv(maxTokens, info.blockSize);
            OP_CHECK_IF(info.maxBlockNumPerBatch < maxBlockNumPerSeq,
                        OP_LOGE(context_, "block_table dim(1)=%u must be >= maxBlockNumPerSeq=%u",
                                info.maxBlockNumPerBatch, maxBlockNumPerSeq),
                        return ge::GRAPH_FAILED);
        }
    }

    // #2: block_table dim(0) == B (PA, after B determined)
    if (info.pageAttention) {
        const gert::StorageShape *btShape = context_->GetOptionalInputShape(PKI_BLOCK_TABLE_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, btShape);
        OP_CHECK_IF(static_cast<uint32_t>(btShape->GetStorageShape().GetDim(0)) != info.bSize,
                    OP_LOGE(context_, "block_table dim(0)(%ld) must equal B(%u)", btShape->GetStorageShape().GetDim(0),
                            info.bSize),
                    return ge::GRAPH_FAILED);
    }

    // #4: pool_tail_k values: each must be in [0, pool_size-1]
    {
        const gert::Tensor *ptkTensor = context_->GetInputTensor(PKI_POOL_TAIL_K_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, ptkTensor);
        OP_CHECK_IF(static_cast<uint32_t>(ptkTensor->GetShapeSize()) != info.bSize,
                    OP_LOGE(context_, "pool_tail_k count(%ld) must equal B(%u)", ptkTensor->GetShapeSize(), info.bSize),
                    return ge::GRAPH_FAILED);
        const int64_t *ptkData = IsHostVisibleValue(ptkTensor) ? ptkTensor->GetData<int64_t>() : nullptr;
        if (ptkData != nullptr) {
            int64_t ptkCount = ptkTensor->GetShapeSize();
            for (int64_t i = 0; i < ptkCount; ++i) {
                OP_CHECK_IF(ptkData[i] < 0 || ptkData[i] >= static_cast<int64_t>(info.poolSize),
                            OP_LOGE(context_, "pool_tail_k[%ld]=%ld must be in [0, %u)", i, ptkData[i], info.poolSize),
                            return ge::GRAPH_FAILED);
            }
        }
    }

    // Validate pool_key contiguity: PA allows 0-axis non-contiguous, BSND/TND require fully contiguous
    if (CheckKeyContiguous(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PoolKeyIndexerTiling::CheckKeyContiguous(const PoolKeyIndexerTilingInfo &info) const
{
    // PA_BBND: axis 0 allows non-contiguous, check starts from axis 1
    // Non-PA: check starts from axis 0 (all axes must be contiguous)
    size_t checkStartIdx = (info.layoutK == PkiDataLayout::PA_BBND) ? 1 : 0;

    // ---- pool_key 连续性校验(失败日志含 tensor 名/轴号/实际 stride/期望 stride/shape) ----
    if (!keyStridesVec_.empty()) {
        const gert::StorageShape *keyShape = context_->GetInputShape(PKI_POOL_KEY_INDEX);
        if (keyShape != nullptr) {
            auto &shape = keyShape->GetStorageShape();
            std::vector<uint32_t> expectedStrides;
            if (info.layoutK == PkiDataLayout::BSND || info.layoutK == PkiDataLayout::PA_BBND) {
                expectedStrides = {static_cast<uint32_t>(shape.GetDim(1) * shape.GetDim(2) * shape.GetDim(3)),
                                   static_cast<uint32_t>(shape.GetDim(2) * shape.GetDim(3)),
                                   static_cast<uint32_t>(shape.GetDim(3)), 1U};
            } else { // TND
                expectedStrides = {static_cast<uint32_t>(shape.GetDim(1) * shape.GetDim(2)),
                                   static_cast<uint32_t>(shape.GetDim(2)), 1U};
            }
            std::string shapeStr = "shape=(";
            for (size_t i = 0; i < shape.GetDimNum(); ++i) {
                shapeStr += std::to_string(shape.GetDim(i)) + (i + 1 < shape.GetDimNum() ? "," : ")");
            }
            std::string actualStridesStr = "actual strides=(";
            for (size_t i = 0; i < keyStridesVec_.size(); ++i) {
                actualStridesStr += std::to_string(keyStridesVec_[i]) + (i + 1 < keyStridesVec_.size() ? "," : ")");
            }
            std::string expectedStridesStr = "expected strides=(";
            for (size_t i = 0; i < expectedStrides.size(); ++i) {
                expectedStridesStr += std::to_string(expectedStrides[i]) + (i + 1 < expectedStrides.size() ? "," : ")");
            }
            for (size_t i = checkStartIdx; i < expectedStrides.size(); ++i) {
                if (i < keyStridesVec_.size() && keyStridesVec_[i] != expectedStrides[i]) {
                    OP_LOGE(context_,
                            "pool_key only supports non-contiguous on the 0-axis, but axis[%zu] stride mismatch: "
                            "actual stride[%zu]=%u, expected stride[%zu]=%u, %s, %s, %s, layout_k=%s.",
                            i, i, keyStridesVec_[i], i, expectedStrides[i], shapeStr.c_str(), actualStridesStr.c_str(),
                            expectedStridesStr.c_str(),
                            info.layoutK == PkiDataLayout::PA_BBND ?
                                "PA_BBND" :
                                (info.layoutK == PkiDataLayout::BSND ? "BSND" : "TND"));
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }
    // ---- k_descale 非连续校验(量化场景, 与 pool_key 同规则: PA 仅 0 轴可非连续, 其余轴必须连续) ----
    if (info.quantMode == PKI_QUANT_FP8_PER_TOKEN || info.quantMode == PKI_QUANT_MXFP8) {
        if (!keyDequantScaleStridesVec_.empty()) {
            const gert::StorageShape *kdsShape = context_->GetOptionalInputShape(PKI_K_DESCALE_INDEX);
            if (kdsShape != nullptr) {
                auto &sShape = kdsShape->GetStorageShape();
                size_t dimNum = sShape.GetDimNum();
                // 逐轴推导连续 stride(倒序), 末轴 stride 恒 1
                std::vector<uint32_t> expectedStrides(dimNum, 1U);
                for (size_t i = dimNum - 1; i > 0; --i) {
                    expectedStrides[i - 1] = expectedStrides[i] * static_cast<uint32_t>(sShape.GetDim(i));
                }
                std::string shapeStr = "shape=(";
                for (size_t i = 0; i < dimNum; ++i) {
                    shapeStr += std::to_string(sShape.GetDim(i)) + (i + 1 < dimNum ? "," : ")");
                }
                std::string actualStridesStr = "actual strides=(";
                for (size_t i = 0; i < keyDequantScaleStridesVec_.size(); ++i) {
                    actualStridesStr += std::to_string(keyDequantScaleStridesVec_[i]) +
                                        (i + 1 < keyDequantScaleStridesVec_.size() ? "," : ")");
                }
                std::string expectedStridesStr = "expected strides=(";
                for (size_t i = 0; i < expectedStrides.size(); ++i) {
                    expectedStridesStr +=
                        std::to_string(expectedStrides[i]) + (i + 1 < expectedStrides.size() ? "," : ")");
                }
                for (size_t i = checkStartIdx; i < dimNum; ++i) {
                    if (i < keyDequantScaleStridesVec_.size() && keyDequantScaleStridesVec_[i] != expectedStrides[i]) {
                        OP_LOGE(context_,
                                "k_descale only supports non-contiguous on the 0-axis, but axis[%zu] stride mismatch: "
                                "actual stride[%zu]=%u, expected stride[%zu]=%u, %s, %s, %s, quant_mode=%d.",
                                i, i, keyDequantScaleStridesVec_[i], i, expectedStrides[i], shapeStr.c_str(),
                                actualStridesStr.c_str(), expectedStridesStr.c_str(), info.quantMode);
                        return ge::GRAPH_FAILED;
                    }
                }
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PoolKeyIndexerTiling::CalcTilingParams(PoolKeyIndexerTilingInfo &info)
{
    if (info.sparseCount > PKI_TOPK_6K) {
        tilingData_.set_s1BaseSize(PKI_S1_BASE_SIZE_SMALL);
        tilingData_.set_mBaseSizeMax(PKI_M_BASE_SIZE_SMALL);
    } else {
        tilingData_.set_s1BaseSize(PKI_S1_BASE_SIZE);
        tilingData_.set_mBaseSizeMax(PKI_M_BASE_SIZE);
    }
    tilingData_.set_mBaseSize(tilingData_.get_s1BaseSize() * info.gSize);
    tilingData_.set_s2BaseSize(PKI_S2_BASE_SIZE);
    tilingData_.set_trunkLen(GetTrunkLen(info.sparseCount));

    // Core count from platform
    tilingData_.set_usedCoreNum(info.aicNum);
    info.usedCoreNum = info.aicNum;

    return ge::GRAPH_SUCCESS;
}

uint32_t PoolKeyIndexerTiling::GetTrunkLen(uint32_t sparseCount)
{
    if (sparseCount <= PKI_TOPK_2K)
        return PKI_TRUNK_LEN_16K;
    if (sparseCount <= PKI_TOPK_3K)
        return PKI_TRUNK_LEN_12K;
    if (sparseCount <= PKI_TOPK_4K)
        return PKI_TRUNK_LEN_8K;
    if (sparseCount <= PKI_TOPK_5K)
        return PKI_TRUNK_LEN_4K;
    if (sparseCount <= PKI_TOPK_6K)
        return PKI_TRUNK_LEN_2K;
    return PKI_TRUNK_LEN_12K;
}

ge::graphStatus PoolKeyIndexerTiling::CalcWorkspaceSize(PoolKeyIndexerTilingInfo &info, uint64_t &workspaceSize)
{
    // workspace 布局须与 kernel 侧一致: | mm1ResGm(双缓冲score) | vec1ResGm(LD中间结果) |
    // vec1ParamGm(LD参数) |; MIX 核上 GetBlockNum() = aicNum, 直接使用 aicNum
    constexpr uint64_t PKI_BASE_TOPK = 2048;
    constexpr uint64_t PKI_TOPK_MAX_SIZE = 8192; // LD 区域按 topk 上限预留余量
    constexpr uint64_t PKI_SPARSE_COUNT_8K = 8192;
    constexpr uint64_t PKI_S2_BASE_SIZE = 512;
    constexpr uint64_t PKI_BLOCK_CUBE_SIZE = 16;
    constexpr uint64_t PKI_WS_DOUBLE = 2;
    constexpr uint64_t PKI_LD_PARAM_NUM = 16;
    // arch35(950) kernel 常量(见 op_kernel/arch35/pool_key_indexer_kernel_arch35.h)
    constexpr uint64_t PKI35_S2_BASE_SIZE = 128;
    constexpr uint64_t PKI35_S1_BASE_SIZE = 4;
    constexpr uint64_t PKI35_S1_BASE_SIZE_SMALL = 2;

    uint64_t offset = 0;
    uint32_t aicNum = info.aicNum;

    // kernel InitTilingData: sparseCount > BASE_TOPK 时 s1BaseSize = 8192/sparseCount*2, 否则为 8
    uint64_t s1BaseSize = (info.sparseCount > PKI_BASE_TOPK) ? (PKI_SPARSE_COUNT_8K / info.sparseCount) * 2 : 8;
    uint64_t mBaseSizeAlign = RoundUp64(s1BaseSize * info.gSize, PKI_BLOCK_CUBE_SIZE);

    // scoreGm: aicNum × 双缓冲 × mBaseSizeAlign × s2BaseSize × sizeof(SCORE_T)
    uint64_t scoreGmSize =
        RoundUp64(static_cast<uint64_t>(aicNum) * PKI_WS_DOUBLE * mBaseSizeAlign * PKI_S2_BASE_SIZE * PKI_SCORE_T_SIZE,
                  PKI_GM_ALIGN_BYTES);
    // arch35 kernel score 区域: aicNum × s1BaseSize(4|2) × Align(s2Size,128) × 4B;
    // TND/PA 的 s2Size 可能更大, 需取 max 覆盖
    uint64_t s1BaseSize35 = (info.sparseCount > PKI_BASE_TOPK) ? PKI35_S1_BASE_SIZE_SMALL : PKI35_S1_BASE_SIZE;
    uint64_t s2Size35Align = (info.s2Size + PKI35_S2_BASE_SIZE - 1) / PKI35_S2_BASE_SIZE * PKI35_S2_BASE_SIZE;
    uint64_t scoreGmSize35 =
        RoundUp64(static_cast<uint64_t>(aicNum) * s1BaseSize35 * s2Size35Align * PKI_SCORE_T_SIZE, PKI_GM_ALIGN_BYTES);
    scoreGmSize = std::max(scoreGmSize, scoreGmSize35);
    tilingData_.set_wsOffScore(offset);
    offset += scoreGmSize;

    // LD score: aicNum × s1BaseSize × 2(头/尾) × 2(idx/value) × TOPK_MAX × sizeof(SCORE_T)
    // 按 topk 上限 8192 预留，覆盖 kernel 实写的 BASE_TOPK(2048)
    uint64_t ldScoreSize = RoundUp64(static_cast<uint64_t>(aicNum) * s1BaseSize * PKI_WS_DOUBLE * PKI_WS_DOUBLE *
                                         PKI_TOPK_MAX_SIZE * PKI_SCORE_T_SIZE,
                                     32);
    tilingData_.set_wsOffLdScore(offset);
    offset += ldScoreSize;

    // LD param: aicNum × s1BaseSize × 2 × LD_PARAM_NUM × sizeof(int64_t)
    uint64_t ldIdxSize =
        RoundUp64(static_cast<uint64_t>(aicNum) * s1BaseSize * PKI_WS_DOUBLE * PKI_LD_PARAM_NUM * sizeof(int64_t), 32);
    tilingData_.set_wsOffLdIdx(offset);
    offset += ldIdxSize;

    workspaceSize = offset;
    OP_LOGI(context_->GetNodeName(), "PoolKeyIndexer workspace: scoreGm=%lu, ldScore=%lu, ldIdx=%lu, total=%lu",
            scoreGmSize, ldScoreSize, ldIdxSize, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

void PoolKeyIndexerTiling::SetTilingData(PoolKeyIndexerTilingInfo &info)
{
    tilingData_.set_bSize(info.bSize);
    tilingData_.set_gSize(info.gSize);
    tilingData_.set_s1Size(info.s1Size);
    tilingData_.set_s2Size(info.s2Size);
    tilingData_.set_sparseCount(info.sparseCount);
    tilingData_.set_topk(info.topk);
    tilingData_.set_poolSize(info.poolSize);
    tilingData_.set_maskMode(static_cast<uint32_t>(info.maskMode));
    // quant_mode: host(-1/0/1) → tpl(0/1/2) via +1 offset, see template_tiling_key.h
    tilingData_.set_quantMode(static_cast<uint32_t>(info.quantMode + 1));
    tilingData_.set_returnValue(info.returnValue ? 1 : 0);
    tilingData_.set_layoutQ(static_cast<uint32_t>(info.layoutQ));
    tilingData_.set_layoutK(static_cast<uint32_t>(info.layoutK));
    tilingData_.set_blockSize(info.blockSize);
    tilingData_.set_maxBlockNumPerBatch(info.maxBlockNumPerBatch);
    tilingData_.set_keyStride0(info.keyStride0);
    tilingData_.set_keyDequantScaleStride0(info.keyDequantScaleStride0);
    // 算子文档公式: S = Q @ K_pool^T * 1/sqrt(headDim), host 侧预计算缩放系数
    tilingData_.set_qkScale(1.0f / std::sqrt(static_cast<float>(info.headDim)));
}

ge::graphStatus PoolKeyIndexerTiling::DoTiling(PoolKeyIndexerTilingInfo *tilingInfo)
{
    auto &info = *tilingInfo;
    info.opName = context_->GetNodeName();

    // Get platform info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    info.npuArch = ascendcPlatform.GetCurNpuArch();
    info.aivNum = ascendcPlatform.GetCoreNumAiv();
    info.aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubSize);

    // Parse and validate
    auto ret = ParseAndCheckParams(info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Compute tiling params
    ret = CalcTilingParams(info);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Compute workspace
    uint64_t workspaceSize = 0;
    ret = CalcWorkspaceSize(info, workspaceSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // GetUserWorkspace() 会跳过系统保留区(GetLibApiWorkSpaceSize), 必须计入
    // 总 workspace, 否则 kernel 可用空间不足导致 GM 越界
    workspaceSize += ascendcPlatform.GetLibApiWorkSpaceSize();

    // Set tiling data fields
    SetTilingData(info);

    // Write tiling data
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // Set workspace size
    // GE 图模式 workspaces[0] 可能为 SIZE_MAX 哨兵, 必须直接覆写(取 max 会烘焙进图导致溢出/OOM)
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = static_cast<size_t>(workspaceSize);
    }

    // Set tiling key: map runtime attributes to template tiling key values
    // DT_Q/DT_K: FP16=1, BF16=27, FP8=2 (PKI_TPL_* in template_tiling_key.h)
    uint32_t tplDtQ = 0;
    if (info.queryDtype == ge::DT_FLOAT16) {
        tplDtQ = 1;
    } else if (info.queryDtype == ge::DT_BF16) {
        tplDtQ = 27;
    } else if (info.queryDtype == ge::DT_FLOAT8_E4M3FN) {
        tplDtQ = 2;
    }
    // quant_mode: host(-1/0/1) → tpl(0/1/2) via +1 offset
    uint32_t tplQuantMode = static_cast<uint32_t>(info.quantMode + 1);
    uint64_t tilingKey = GET_TPL_TILING_KEY(tplDtQ, tplDtQ, 3U, static_cast<uint32_t>(info.layoutQ),
                                            static_cast<uint32_t>(info.layoutK), static_cast<uint32_t>(info.maskMode),
                                            tplQuantMode, static_cast<uint32_t>(info.returnValue ? 1 : 0));
    context_->SetTilingKey(tilingKey);

    // Set blockDim (1:2 AIC:AIV)
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(info.aivNum, info.aicNum, info.aivNum);
    context_->SetBlockDim(blockDim);
    OP_LOGI(info.opName, "PoolKeyIndexer DoTiling: tilingKey=%lu, blockDim=%u, aicNum=%u, aivNum=%u, wsSize=%lu",
            tilingKey, blockDim, info.aicNum, info.aivNum, workspaceSize);
    if (info.pageAttention) {
        OP_LOGI(info.opName, "PoolKeyIndexer PA: keyStride0=%u, blockSize=%u, maxBlkPerBatch=%u, s2Size=%u",
                info.keyStride0, info.blockSize, info.maxBlockNumPerBatch, info.s2Size);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPoolKeyIndexer(gert::TilingContext *context)
{
    PoolKeyIndexerTiling tiling(context);
    PoolKeyIndexerTilingInfo info;
    return tiling.DoTiling(&info);
}

// 图模式 compile info 解析回调: tilingEntry 编译期将平台信息写入 CompileInfo,
// 框架据此生成 compile info JSON 供 FE 图编译期解析
static ge::graphStatus TilingPrepareForPoolKeyIndexer(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("PoolKeyIndexer", "TilingParseContext is nullptr!"),
                return ge::GRAPH_FAILED);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "TilingParse: platformInfo is nullptr"),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<PoolKeyIndexerCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context->GetNodeName(), "TilingParse: compileInfoPtr is nullptr"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);

    OP_LOGI(context->GetNodeName(), "TilingParse: soc=%d, npuArch=%d, aicNum=%lu, aivNum=%lu, ubSize=%lu, l1Size=%lu",
            static_cast<int32_t>(compileInfoPtr->socVersion), static_cast<int32_t>(compileInfoPtr->npuArch),
            compileInfoPtr->aicNum, compileInfoPtr->aivNum, compileInfoPtr->ubSize, compileInfoPtr->l1Size);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PoolKeyIndexer)
    .Tiling(TilingPoolKeyIndexer)
    .TilingParse<PoolKeyIndexerCompileInfo>(TilingPrepareForPoolKeyIndexer);

} // namespace optiling
