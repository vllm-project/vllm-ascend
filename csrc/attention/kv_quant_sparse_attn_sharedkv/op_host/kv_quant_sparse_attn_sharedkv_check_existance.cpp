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
 * \file kv_quant_sparse_attn_sharedkv_check_existance.cpp
 * \brief
 */

#include "kv_quant_sparse_attn_sharedkv_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

static constexpr uint32_t TopK_SIZE = 512;
static constexpr uint32_t DIM_0 = 0;
static constexpr uint32_t DIM_1 = 1;
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t DIM_3 = 3;

ge::graphStatus KvQuantSASTilingCheck::CheckParaExistenceAntiquant() const
{
    if (kvLayout_ == SASLayout::BSND) {
        return ge::GRAPH_SUCCESS;
    }  else if (kvLayout_ == SASLayout::PA_ND) {
        OP_CHECK_IF(opParamInfo_.sequsedKv.tensor == nullptr,
            OP_LOGE(opName_, "when layout_kv is PA_ND, actualSeqLengthsKv must not be null"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((opParamInfo_.oriBlockTable.tensor == nullptr) && (opParamInfo_.cmpBlockTable.tensor == nullptr),
            OP_LOGE(opName_, "when layout_kv is PA_ND, oriBlockTable and cmpBlockTable must be one "),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckCmpSparseIndicesExistence() ||
        ge::GRAPH_SUCCESS != CheckSWAExistence() ||
        ge::GRAPH_SUCCESS != CheckCFAExistence() ||
        ge::GRAPH_SUCCESS != CheckSCFAExistence() ||
        ge::GRAPH_SUCCESS != CheckCmpRatioExistence() ||
        ge::GRAPH_SUCCESS != CheckUnrequiredParaExistence() ||
        ge::GRAPH_SUCCESS != CheckParaExistenceAntiquant()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

 ge::graphStatus KvQuantSASTilingCheck::CheckUnrequiredParaExistence() const
{
    OP_CHECK_IF(opParamInfo_.oriSparseIndices.tensor != nullptr || opParamInfo_.oriSparseIndices.desc != nullptr,
                OP_LOGE(opName_, "oriSparseIndices is not supported now, it must be nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.cuSeqLensOriKv.tensor != nullptr || opParamInfo_.cuSeqLensOriKv.desc != nullptr,
                OP_LOGE(opName_, "cuSeqLensOriKv is not supported now, it must be nullptr."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.cuSeqLensCmpKv.tensor != nullptr || opParamInfo_.cuSeqLensCmpKv.desc != nullptr,
                OP_LOGE(opName_, "cuSeqLensCmpKv is not supported now, it must be nullptr."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckCmpSparseIndicesExistence()
{
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        if (qLayout_ == SASLayout::BSND) {
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_3) != 512 && opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_3) != 1024) {
                OP_LOGE(opName_, "When qLayout is BNSD, topK should be 512 or 1024, but got %ld", opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(3));
                return ge::GRAPH_FAILED;
            }
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_1) != s1Size_) {
                OP_LOGE(opName_, "When qLayout is BNSD, cmpSparseIndices's S should be eaque to s1Size:%u, but got %ld", s1Size_, opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(1));
                return ge::GRAPH_FAILED;
            }
        } else {
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_2) != 512 && opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_2) != 1024) {
                OP_LOGE(opName_, "When qLayout is TND, topK should be 512 or 1024, but got %ld", opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(2));
                return ge::GRAPH_FAILED;
            }
            if (opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_0) != qTSize_) {
                OP_LOGE(opName_, "When qLayout is TND, cmpSparseIndices's T should be eaque to qTSize:%u, but got %ld", qTSize_, opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(0));
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSWAExistence()
{
    if (perfMode_ != SASTemplateMode::SWA_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.oriBlockTable.tensor == nullptr,
        OP_LOGE(opName_, "oriBlockTable must not be empty when cmpKv is not provided. "),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckCFAExistence()
{
    if (perfMode_ != SASTemplateMode::CFA_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr && opParamInfo_.cmpKv.tensor != nullptr,
        OP_LOGE(opName_, "oriKv must not be empty when cmpKv is provided and cmpSparseIndices is not provided."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.cmpKv.tensor == nullptr && opParamInfo_.cmpRatio != nullptr,
        OP_LOGE(opName_, "cmpKv must not be empty when cmpKv is provided and cmpSparseIndices is not provided."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.cmpKv.tensor != nullptr && opParamInfo_.cmpRatio == nullptr,
        OP_LOGE(opName_, "cmpRatio must not be empty when cmpKv is provided and cmpSparseIndices is not provided."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.cmpKv.tensor != nullptr && opParamInfo_.cmpBlockTable.tensor == nullptr,
        OP_LOGE(opName_, "cmpBlockTable must not be empty when cmpKv is provided and cmpSparseIndices is not provided."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckSCFAExistence()
{
    if (perfMode_ != SASTemplateMode::SCFA_TEMPLATE_MODE) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(opParamInfo_.oriKv.tensor != nullptr && opParamInfo_.cmpKv.tensor == nullptr && opParamInfo_.cmpSparseIndices.tensor != nullptr,
        OP_LOGE(opName_, "cmpKv must not be empty when cmpKv and cmpSparseIndices are provided."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr && opParamInfo_.cmpKv.tensor != nullptr && opParamInfo_.cmpSparseIndices.tensor != nullptr,
        OP_LOGE(opName_, "oriKv must not be empty when cmpKv and cmpSparseIndices are provided."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr && opParamInfo_.cmpKv.tensor == nullptr && opParamInfo_.cmpSparseIndices.tensor != nullptr,
        OP_LOGE(opName_, "oriKv and cmpKv must not be empty when cmpKv and cmpSparseIndices are provided."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckCmpRatioExistence()
{
    if (perfMode_ == SASTemplateMode::SWA_TEMPLATE_MODE) {
        OP_CHECK_IF(*opParamInfo_.cmpRatio != 1 && *opParamInfo_.cmpRatio != 128 && *opParamInfo_.cmpRatio != 4,
            OP_LOGE(opName_, "when SWA mode, cmpRatio must be 1 or 4 or 128, but got %u", *opParamInfo_.cmpRatio),
            return ge::GRAPH_FAILED);
    } else if (perfMode_ == SASTemplateMode::CFA_TEMPLATE_MODE) {
        OP_CHECK_IF(*opParamInfo_.cmpRatio != 128 && *opParamInfo_.cmpRatio != 4,
            OP_LOGE(opName_, "when CFA mode, cmpRatio must be 4 or 128, but got %u", *opParamInfo_.cmpRatio),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(*opParamInfo_.cmpRatio != 128 && *opParamInfo_.cmpRatio != 4,
            OP_LOGE(opName_, "when SCFA mode, cmpRatio must be 4 or 128, but got %u", *opParamInfo_.cmpRatio),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

}