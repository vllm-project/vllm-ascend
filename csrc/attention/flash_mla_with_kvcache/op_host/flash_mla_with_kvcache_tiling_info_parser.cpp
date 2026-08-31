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
 * \file flash_mla_with_kvcache_tiling_info_parser.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "flash_mla_with_kvcache_tiling_info_parser.h"
#include "flash_mla_with_kvcache_tiling_utils.h"

using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace Ops::Base;
// using namespace AscendC;
namespace optiling {
namespace flash_mla_with_kvcache {

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetEmptyTensorFlag()
{
    auto checkEmptyTensor = [this](const gert::StorageShape *shape, const std::string &name) -> bool {
        if (shape == nullptr) {
            return false;
        }
        for (size_t i = 0; i < shape->GetStorageShape().GetDimNum(); i++) {
            if (shape->GetStorageShape().GetDim(i) == 0) {
                OP_LOGE(opName_, "Tensor %s has empty dimension at axis %zu, size is 0.", name.c_str(), i);
                return true;
            }
        }
        return false;
    };

    if (checkEmptyTensor(opParamInfo_.query.shape, QUERY_NAME) ||
        checkEmptyTensor(opParamInfo_.kCache.shape, K_CACHE_NAME) ||
        checkEmptyTensor(opParamInfo_.attnOut.shape, ATTN_OUT_NAME)) {
        emptyTensorFlag_ = true;
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of query"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of query"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kCache.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of k_cache"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.kCache.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of k_cache"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attnOut.shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Shape of atten_out"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attnOut.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "Desc of atten_out"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layoutQ == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_q"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutKV == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_kv"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutOut == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName_, "layout_out"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetCuSeqLenQSize(int64_t &size)
{
    if (opParamInfo_.cuSeqlensQ.tensor == nullptr) {
        OP_LOGE(opName_, "when %s's layout is %s, %s must be provided.", QUERY_NAME.c_str(),
                LayoutToSerialString(layoutQ_).c_str(), CU_SEQLENS_Q_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = opParamInfo_.cuSeqlensQ.tensor->GetShapeSize();
    if (shapeSize <= 1) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(opName_, "cu_seqlens_q", std::to_string(shapeSize).c_str(), "greater than 1");
        return ge::GRAPH_FAILED;
    }
    size = shapeSize - 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("FlashMlaWithKvcache", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_3510) {
        OP_LOGE(opName_, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void FlashMlaWithKvcacheInfoParser::GetOptionalInputParaInfo()
{
    GetOptionalInputParaSeqLengthInfo();
    GetOptionalInputParaPageAttentionInfo();
    GetOptionalInputParaMetadataInfo();
    GetOptionalInputParaMaskInfo();
}
void FlashMlaWithKvcacheInfoParser::GetOptionalInputParaMaskInfo()
{
    opParamInfo_.attnMask.tensor = context_->GetOptionalInputTensor(ATTN_MASK_INDEX);
    opParamInfo_.attnMask.desc = context_->GetOptionalInputDesc(ATTN_MASK_INDEX);
}

void FlashMlaWithKvcacheInfoParser::GetOptionalInputParaSeqLengthInfo()
{
    opParamInfo_.cacheSeqlens.tensor = context_->GetOptionalInputTensor(CACHE_SEQLENS_INDEX);
    opParamInfo_.cacheSeqlens.desc = context_->GetOptionalInputDesc(CACHE_SEQLENS_INDEX);
    opParamInfo_.cuSeqlensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqlensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.sequsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
}

void FlashMlaWithKvcacheInfoParser::GetOptionalInputParaPageAttentionInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void FlashMlaWithKvcacheInfoParser::GetOptionalInputParaMetadataInfo()
{
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
}

void FlashMlaWithKvcacheInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.kCache.desc = context_->GetInputDesc(K_CACHE_INDEX);
    opParamInfo_.kCache.shape = context_->GetInputShape(K_CACHE_INDEX);
    GetOptionalInputParaInfo();
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetStrides()
{
    // 用opbase版本判断：老版本opbase的输入非view, 视为tensorv1, k_cache必然连续
    if (context_->InputIsView(K_CACHE_INDEX) == false) {
        hasViewStride_ = false;
        return ge::GRAPH_SUCCESS;
    }
    // k == v 单 buffer (k_cache 同时承载 key 与 value 数据)
    keyStrides_ = context_->GetInputStride(K_CACHE_INDEX);
    valueStrides_ = keyStrides_;
    return ge::GRAPH_SUCCESS;
}

void FlashMlaWithKvcacheInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
    opParamInfo_.lseOut.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.lseOut.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    // 索引0: head_dim_v (Int, nope/value 段宽度, 默认 512)
    opParamInfo_.headDimV = attrs->GetAttrPointer<int64_t>(ATTR_HEAD_DIM_V_INDEX);

    // 索引1: softmax_scale (Float)
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);

    // 索引2: mask_mode (Int)
    opParamInfo_.maskMode = attrs->GetAttrPointer<int64_t>(ATTR_MASK_MODE_INDEX);

    // 索引3: max_seqlen_q (Int)
    opParamInfo_.maxSeqlenQ = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_Q_INDEX);

    // 索引4: max_seqlen_kv (Int)
    opParamInfo_.maxSeqlenKV = attrs->GetAttrPointer<int64_t>(ATTR_MAX_SEQLEN_KV_INDEX);

    // 索引5: layout_q (String)
    opParamInfo_.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);

    // 索引6: layout_kv (String)
    opParamInfo_.layoutKV = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);

    // 索引7: layout_out (String)
    opParamInfo_.layoutOut = attrs->GetStr(ATTR_LAYOUT_OUT_INDEX);

    // 索引8: return_softmax_lse (Int)
    opParamInfo_.returnSoftMaxLse = attrs->GetAttrPointer<int64_t>(ATTR_RETURN_LSE_INDEX);

    return ge::GRAPH_SUCCESS;
}

void FlashMlaWithKvcacheInfoParser::GetMaskParams()
{
    // 文档约束：不传或传入-1代表正无穷，默认值为-1；win_left/win_right 已从接口移除，恒为 -1
    winLeft_ = -1;
    winRight_ = -1;
    maskMode_ = (opParamInfo_.maskMode == nullptr) ? 0 : *opParamInfo_.maskMode;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FlashMlaWithKvcacheInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.kCache.desc->GetDataType();
    outputType_ = opParamInfo_.attnOut.desc->GetDataType();
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (layoutQ_ == FlashMlaWithKvcacheLayout::TND) {
        return GetCuSeqLenQSize(bSize_);
    } else { // BSH/BSND/BNSD
        if (queryShape_->CheckHasShapeB(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        bSize_ = queryShape_->GetShapeB();
        return ge::GRAPH_SUCCESS;
    }
}

void FlashMlaWithKvcacheInfoParser::GetQueryTSize()
{
    // 获取query的T基准值
    // 1、非TND/NTD时, 以query的batch_size维度为基准;
    // 2、TND/NTD时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    queryTSize_ = (queryShape_->HasShapeT()) ? static_cast<uint32_t>(queryShape_->GetShapeT()) : 0;
}

void FlashMlaWithKvcacheInfoParser::GetKeyTSize()
{
    keyTSize_ = (keyShape_->HasShapeT()) ? static_cast<uint32_t>(keyShape_->GetShapeT()) : 0;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetQkHeadDim()
{
    // q 最后维 = 576（nope 512 + rope 64 合并）。dSize(attention 计算宽度) = head_dim_v(nope 段),
    // dSizeRope 从 576 - head_dim_v 推导；仅 MLA D512：head_dim_v==512、rope==64。
    if (queryShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t headDimV = (opParamInfo_.headDimV == nullptr) ? arch35MLA::MLA_D_DIM_512 : *opParamInfo_.headDimV;
    int64_t lastDim = queryShape_->GetShapeD();
    if (lastDim <= headDimV || (lastDim - headDimV) != arch35MLA::MLA_ROPE_D_DIM_64) {
        OP_LOGE(opName_, "q last dim(%ld) must be head_dim_v(%ld) + rope(64).", lastDim, headDimV);
        return ge::GRAPH_FAILED;
    }
    qkHeadDim_ = headDimV;
    ropeHeadDim_ = lastDim - headDimV;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetS1Size()
{
    if (layoutQ_ == FlashMlaWithKvcacheLayout::TND) {
        s1Size_ = queryTSize_;
    } else {
        if (queryShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s1Size_ = static_cast<uint32_t>(queryShape_->GetShapeS());
    }
    return ge::GRAPH_SUCCESS;
}

void FlashMlaWithKvcacheInfoParser::GetKvIsContiguous()
{
    // 老版本opbase不支持strides, 无法判NC, 放过
    if (!hasViewStride_) {
        return;
    }

    int32_t keyDimIdx = 0;
    int32_t valDimIdx = 0;
    bool keyContig = false;
    bool valContig = false;
    if (keyStrides_ != nullptr) {
        const gert::Shape &keyShape = opParamInfo_.kCache.shape->GetStorageShape();
        keyContig =
            (CheckTensorContiguous(keyShape.GetDimNum(), keyShape, keyStrides_, keyDimIdx) == ge::GRAPH_SUCCESS);
        keyNonContigDim_ = keyContig ? -1 : keyDimIdx;
    }

    if (valueStrides_ != nullptr) {
        const gert::Shape &valueShape = opParamInfo_.kCache.shape->GetStorageShape();
        valContig =
            (CheckTensorContiguous(valueShape.GetDimNum(), valueShape, valueStrides_, valDimIdx) == ge::GRAPH_SUCCESS);
        valueNonContigDim_ = valContig ? -1 : valDimIdx;
    }

    if (layoutKV_ == FlashMlaWithKvcacheLayout::PA_BBND) {
        if (keyStrides_ != nullptr) {
            keyBnStride_ = keyStrides_->GetStride(0);
            keyN2Stride_ = keyStrides_->GetStride(2);
        }
        if (valueStrides_ != nullptr) {
            valueBnStride_ = valueStrides_->GetStride(0);
            valueN2Stride_ = valueStrides_->GetStride(2);
        }
    } else if (layoutKV_ == FlashMlaWithKvcacheLayout::PA_BNBD || layoutKV_ == FlashMlaWithKvcacheLayout::PA_NZ) {
        if (keyStrides_ != nullptr) {
            keyBnStride_ = keyStrides_->GetStride(0);
            keyN2Stride_ = keyStrides_->GetStride(1);
        }
        if (valueStrides_ != nullptr) {
            valueBnStride_ = valueStrides_->GetStride(0);
            valueN2Stride_ = valueStrides_->GetStride(1);
        }
    }
}

void FlashMlaWithKvcacheInfoParser::GetKvStorageMode()
{
    bool isPaLayout =
        (layoutKV_ == FlashMlaWithKvcacheLayout::PA_BBND || layoutKV_ == FlashMlaWithKvcacheLayout::PA_BNBD ||
         layoutKV_ == FlashMlaWithKvcacheLayout::PA_NZ);

    if (isPaLayout) {
        kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    }
    GetKvIsContiguous();
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetS2SizeForBatchContinuous()
{
    if (layoutKV_ == FlashMlaWithKvcacheLayout::TND) {
        s2Size_ = keyTSize_;
    } else {
        if (keyShape_->CheckHasShapeS(__func__) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        s2Size_ = keyShape_->GetShapeS();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetBlockSize()
{
    if (keyShape_->CheckHasShapeBlockSize(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    blockSize_ = keyShape_->GetShapeBlockSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetBlockNum()
{
    if (keyShape_->CheckHasShapeBlockNum(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    blockNum_ = keyShape_->GetShapeBlockNum();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetS2SizeForPageAttention()
{
    OP_CHECK_IF(
        opParamInfo_.blockTable.tensor == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "block_table", "provided",
                                              "When layout_kv is PA, block_table must be provided but got nullptr."),
        return ge::GRAPH_FAILED);
    if (GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetBlockNum() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = static_cast<int64_t>(maxBlockNumPerBatch_) * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 2、TENSOR_LIST时, 从kCache_的所有Tensor的S轴的最大值
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetValueHeadDim()
{
    // k==v：k_cache 最后维 576 承载 key(nope512+rope64) 与 value(nope512)。
    // vHeadDim = head_dim_v（nope/value 段宽）；同时校验 k_cache 最后维 = head_dim_v + rope。
    if (valueShape_->CheckHasShapeD(__func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    int64_t headDimV = (opParamInfo_.headDimV == nullptr) ? arch35MLA::MLA_D_DIM_512 : *opParamInfo_.headDimV;
    int64_t lastDim = valueShape_->GetShapeD();
    if (lastDim <= headDimV || (lastDim - headDimV) != arch35MLA::MLA_ROPE_D_DIM_64) {
        OP_LOGE(opName_, "k_cache last dim(%ld) must be head_dim_v(%ld) + rope(64).", lastDim, headDimV);
        return ge::GRAPH_FAILED;
    }
    vHeadDim_ = headDimV;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetInAndOutLayout()
{
    // Layout枚举很多，kernel和tiling用同一个枚举，
    auto itQ = layoutMap.find(opParamInfo_.layoutQ);
    if (itQ == layoutMap.end()) {
        std::string reason = "layout_q: " + std::string(opParamInfo_.layoutQ) + " is not supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_q", opParamInfo_.layoutQ, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutQ_ = itQ->second;

    auto itKV = layoutMap.find(opParamInfo_.layoutKV);
    if (itKV == layoutMap.end()) {
        std::string reason = "layout_kv: " + std::string(opParamInfo_.layoutKV) + " is not supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_kv", opParamInfo_.layoutKV, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutKV_ = itKV->second;

    auto itOut = layoutMap.find(opParamInfo_.layoutOut);
    if (itOut == layoutMap.end()) {
        std::string reason = "layout_out: " + std::string(opParamInfo_.layoutOut) + " is not supported.";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_out", opParamInfo_.layoutOut, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    layoutOut_ = itOut->second;

    // 路由约束（与 op_host/checkers/common_checker.cpp 的新布局矩阵对齐）：
    //   q ∈ {TND, BNSD, BSND}；kv 仅分页布局 {PA_NZ, PA_BBND, PA_BNBD}（连续 KV 不放行）；out 必须等于 q。
    if (layoutQ_ != FlashMlaWithKvcacheLayout::TND && layoutQ_ != FlashMlaWithKvcacheLayout::BNSD &&
        layoutQ_ != FlashMlaWithKvcacheLayout::BSND) {
        std::string reason = "The value of layout_q must be TND/BNSD/BSND";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_q", opParamInfo_.layoutQ, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (layoutKV_ != FlashMlaWithKvcacheLayout::PA_NZ && layoutKV_ != FlashMlaWithKvcacheLayout::PA_BBND &&
        layoutKV_ != FlashMlaWithKvcacheLayout::PA_BNBD) {
        std::string reason =
            "The value of layout_kv must be PA_NZ/PA_BBND/PA_BNBD (paged KV only, continuous KV is not supported)";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_kv", opParamInfo_.layoutKV, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (layoutOut_ != layoutQ_) {
        std::string reason = "The value of layout_out must be TND/BNSD/BSND (layout_out must equal layout_q)";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_out", opParamInfo_.layoutOut, reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetN1Size()
{
    // 从 Q 形状获取 N1 值
    if (queryShape_ != nullptr && queryShape_->HasShapeN()) {
        n1Size_ = static_cast<uint32_t>(queryShape_->GetShapeN());
    } else {
        OP_LOGE(opName_, "Failed to get N1 size from query shape.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetN2Size()
{
    // 从 K 形状获取 N2 值
    if (keyShape_ != nullptr && keyShape_->HasShapeN()) {
        n2Size_ = static_cast<uint32_t>(keyShape_->GetShapeN());
    } else {
        OP_LOGE(opName_, "Failed to get N2 size from key shape.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void FlashMlaWithKvcacheInfoParser::SetFaShape()
{
    queryShape_ = std::make_shared<FlashMlaWithKvcacheTilingShape>(opParamInfo_.query.shape->GetStorageShape(),
                                                                   layoutQ_, QUERY_NAME, opName_);
    // 单个 k_cache 同时承载 key 与 value（k == v 语义），key/value 形状按同一张量推导
    keyShape_ = std::make_shared<FlashMlaWithKvcacheTilingShape>(opParamInfo_.kCache.shape->GetStorageShape(),
                                                                 layoutKV_, K_CACHE_NAME, opName_);
    valueShape_ = std::make_shared<FlashMlaWithKvcacheTilingShape>(opParamInfo_.kCache.shape->GetStorageShape(),
                                                                   layoutKV_, K_CACHE_NAME, opName_);
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetGSize()
{
    // 获取G基准值
    if (n2Size_ == 0U) {
        OP_LOGE(opName_, "Kv Heads(%ld) should not be zero.", n2Size_);
        return ge::GRAPH_FAILED;
    }
    if (n1Size_ % n2Size_ != 0U) {
        std::string shapeStr = ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                               ToString(opParamInfo_.kCache.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "query and k_cache", shapeStr.c_str(),
                                               "N of query must be an integer multiple of the same axis of k_cache");
        return ge::GRAPH_FAILED;
    }
    gSize_ = n1Size_ / n2Size_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::GetActualSeqInfo()
{
    // 不物化 int64 —— kernel ActualSeqLensParser 以 ACTLEN_T=uint32_t
    // 直接读取 host 传入的 INT32 GM buffer（cu_seqlens_q / cache_seqlens），此处只统计元素数。
    //   cu_seqlens_q  [b+1]：TND 累计语义（含头部 0）→ actualLenQDims = b + 1
    //   cache_seqlens [b]  ：每 batch 的实际 KV-cache 长度（BY_BATCH 语义）→ actualLenKvDims = b
    actualLenQDims_ = (opParamInfo_.cuSeqlensQ.tensor == nullptr) ? 0 : opParamInfo_.cuSeqlensQ.tensor->GetShapeSize();
    actualLenKvDims_ =
        (opParamInfo_.cacheSeqlens.tensor == nullptr) ? 0 : opParamInfo_.cacheSeqlens.tensor->GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

void FlashMlaWithKvcacheInfoParser::GenerateFeatureInfo(FlashMlaWithKvcacheTilingInfo &faInfo)
{
    // PagedAttention 参数组
    faInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    faInfo.blockSize = blockSize_;
    faInfo.blockTypeSize = sizeof(float);

    // Mask 参数组（maskMode ∈ {NO_MASK, RIGHT_DOWN}；attenMaskFlag = attn_mask 张量存在，FIA 语义）
    faInfo.maskMode = maskMode_;
    faInfo.attenMaskFlag = (opParamInfo_.attnMask.tensor != nullptr) &&
                           (opParamInfo_.attnMask.tensor->GetStorageShape().GetShapeSize() != 0);
    faInfo.winLeft = winLeft_;
    faInfo.winRight = winRight_;

    // preTokens/nextTokens 由 maskMode 推导（FIA UpdatePreNextTokenBySparseMode 语义）
    faInfo.preTokens = MASK_MODE_INT_MAX;
    faInfo.nextTokens = MASK_MODE_INT_MAX;
    if (maskMode_ == static_cast<int64_t>(MaskMode::CAUSAL)) {
        faInfo.nextTokens = 0;
    }

    // SoftmaxLSE 参数组
    faInfo.softmaxLseFlag = softmaxLseFlag_;
    faInfo.totalLseSize =
        (opParamInfo_.lseOut.shape == nullptr) ? 0 : opParamInfo_.lseOut.shape->GetStorageShape().GetShapeSize();

    // INT32 seq-lens buffer 元素数
    faInfo.actualLenQDims = actualLenQDims_;
    faInfo.actualLenKvDims = actualLenKvDims_;

    // 公共参数组 - 其他属性
    faInfo.maxSeqQ = maxSeqQ_;
    faInfo.maxSeqKv = maxSeqKv_;
}

void FlashMlaWithKvcacheInfoParser::GenerateLayoutInfo(FlashMlaWithKvcacheTilingInfo &faInfo)
{
    faInfo.qLayout = layoutQ_;
    faInfo.kvLayout = layoutKV_;
    faInfo.outLayout = layoutOut_;
    // kernel 侧 FLASH_MLA_WITH_KVCACHE_LAYOUT 数值（flash_mla_with_kvcache_public_define_arch35.h），限定路由下恒为
    // NTD(5)
    switch (layoutOut_) {
        case FlashMlaWithKvcacheLayout::BSH:
        case FlashMlaWithKvcacheLayout::BSND:
            faInfo.kernelOutputLayout = arch35MLA::KERNEL_LAYOUT_BSH;
            break;
        case FlashMlaWithKvcacheLayout::BNSD:
            faInfo.kernelOutputLayout = arch35MLA::KERNEL_LAYOUT_BNSD;
            break;
        case FlashMlaWithKvcacheLayout::TND:
            faInfo.kernelOutputLayout = arch35MLA::KERNEL_LAYOUT_TND;
            break;
        case FlashMlaWithKvcacheLayout::NBSD:
            faInfo.kernelOutputLayout = arch35MLA::KERNEL_LAYOUT_NBSD;
            break;
        case FlashMlaWithKvcacheLayout::NTD:
            faInfo.kernelOutputLayout = arch35MLA::KERNEL_LAYOUT_NTD;
            break;
        default:
            faInfo.kernelOutputLayout = arch35MLA::KERNEL_LAYOUT_NTD;
            break;
    }
}

void FlashMlaWithKvcacheInfoParser::GenerateInfo(FlashMlaWithKvcacheTilingInfo &faInfo)
{
    faInfo.opName = opName_;
    faInfo.platformInfo = platformInfo_;
    faInfo.opParamInfo = opParamInfo_;
    GenerateAxisInfo(faInfo);
    GenerateDtypeInfo(faInfo);
    faInfo.batchContinuousFlag = (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS);
    faInfo.kvStorageMode = kvStorageMode_;
    faInfo.emptyTensorFlag = emptyTensorFlag_;

    faInfo.totalOutputSize = opParamInfo_.attnOut.shape->GetStorageShape().GetShapeSize();
    faInfo.totalBlockNum = blockNum_;
    faInfo.softmaxScale = softmaxScale_;
    faInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    faInfo.keyBnStride = keyBnStride_;
    faInfo.keyN2Stride = keyN2Stride_;
    faInfo.valueBnStride = valueBnStride_;
    faInfo.valueN2Stride = valueN2Stride_;

    faInfo.hasViewStride = hasViewStride_;
    faInfo.keyNonContigDim = keyNonContigDim_;
    faInfo.valueNonContigDim = valueNonContigDim_;

    GenerateFeatureInfo(faInfo);
    GenerateLayoutInfo(faInfo);
}

void FlashMlaWithKvcacheInfoParser::GenerateAxisInfo(FlashMlaWithKvcacheTilingInfo &faInfo)
{
    faInfo.bSize = bSize_;
    faInfo.n1Size = n1Size_;
    faInfo.n2Size = n2Size_;
    faInfo.s1Size = s1Size_;
    faInfo.s2Size = s2Size_;
    faInfo.gSize = gSize_;
    faInfo.qkHeadDim = qkHeadDim_;
    faInfo.vHeadDim = vHeadDim_;
    faInfo.ropeHeadDim = ropeHeadDim_;
    faInfo.qTSize = queryTSize_;
    faInfo.kTSize = keyTSize_;
}

void FlashMlaWithKvcacheInfoParser::GenerateDtypeInfo(FlashMlaWithKvcacheTilingInfo &faInfo)
{
    faInfo.inputQType = inputQType_;
    faInfo.inputKvType = inputKvType_;
    faInfo.outputType = outputType_;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::Parse(FlashMlaWithKvcacheTilingInfo &faInfo)
{
    OP_LOGI(faInfo.opName, "enter FlashMlaWithKvcacheInfoParser::Parse!");
    if (context_ == nullptr) {
        OP_LOGE(faInfo.opName, "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence() || ge::GRAPH_SUCCESS != GetEmptyTensorFlag()) {
        return ge::GRAPH_FAILED;
    }
    GetInOutDataType();

    if (ge::GRAPH_SUCCESS != GetInAndOutLayout()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetStrides()) {
        return ge::GRAPH_FAILED;
    }
    GetKvStorageMode();
    if (emptyTensorFlag_) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseAxisInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ParseFeatureInfo()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(faInfo);
    OP_LOGI(faInfo.opName, "end FlashMlaWithKvcacheInfoParser::Parse!");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::ParseAxisInfo()
{
    SetFaShape();
    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size()) {
        return ge::GRAPH_FAILED;
    }

    GetQueryTSize();

    if (ge::GRAPH_SUCCESS != GetQkHeadDim() || ge::GRAPH_SUCCESS != GetValueHeadDim()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size()) {
        return ge::GRAPH_FAILED;
    }

    GetKeyTSize();

    if (ge::GRAPH_SUCCESS != GetGSize() || ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheInfoParser::ParseFeatureInfo()
{
    // Mask 参数组解析；合法集 {NO_MASK(0), RIGHT_DOWN(3)}（FIA MLA 仅验证过这两个 sparseMode）
    GetMaskParams();
    if (maskMode_ != static_cast<int64_t>(MaskMode::NO_MASK) && maskMode_ != static_cast<int64_t>(MaskMode::CAUSAL)) {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "mask_mode", std::to_string(maskMode_).c_str(),
                                  "0 (no mask) or 3 (right-down causal)");
        return ge::GRAPH_FAILED;
    }
    if (maskMode_ == static_cast<int64_t>(MaskMode::NO_MASK) && opParamInfo_.attnMask.tensor != nullptr) {
        OP_LOGE(opName_, "When mask_mode=0 (no mask), attn_mask must not be provided.");
        return ge::GRAPH_FAILED;
    }

    // SeqLengths 参数组解析
    if (ge::GRAPH_SUCCESS != GetActualSeqInfo()) {
        return ge::GRAPH_FAILED;
    }

    // SoftmaxLSE 参数组解析
    // 文档约束：return_softmax_lse 默认值为 0
    returnSoftmaxLse_ = (opParamInfo_.returnSoftMaxLse == nullptr) ? 0 : *opParamInfo_.returnSoftMaxLse;
    softmaxLseFlag_ = (returnSoftmaxLse_ != 0);

    softmaxScale_ = (opParamInfo_.softmaxScale == nullptr) ? 1.0f : *opParamInfo_.softmaxScale;
    maxSeqQ_ = (opParamInfo_.maxSeqlenQ == nullptr) ? -1 : *opParamInfo_.maxSeqlenQ;
    maxSeqKv_ = (opParamInfo_.maxSeqlenKV == nullptr) ? -1 : *opParamInfo_.maxSeqlenKV;

    return ge::GRAPH_SUCCESS;
}
} // namespace flash_mla_with_kvcache
} // namespace optiling
