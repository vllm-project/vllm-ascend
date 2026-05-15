/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_attn_sharedkv_check_feature.cpp
 * \brief
 */

#include "kv_quant_sparse_attn_sharedkv_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

ge::graphStatus KvQuantSASTilingCheck::CheckFeatureAntiquantShape() const
{
    OP_CHECK_IF(bSize_ <= 0,
        OP_LOGE(opName_, "batch_size should be greater than 0, but got %u", bSize_),
        return ge::GRAPH_FAILED);
        
    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == SASLayout::TND),
            OP_LOGE(opName_, "T_size of query should be greater than 0, but got %u", qTSize_),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ != 64,
            OP_LOGE(opName_, "q_head_num should be 1, but got %u", n1Size_),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(n2Size_ != 1,
        OP_LOGE(opName_, "kv_head_num should be 1, but got %u", n2Size_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % n2Size_ != 0,
        OP_LOGE(opName_, "q_head_num(%u) must be divisible by kv_head_num(%u)", n1Size_, n2Size_),
        return ge::GRAPH_FAILED);

    std::vector<uint32_t> gSizeSupportList = {64};
    OP_CHECK_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
        OP_LOGE(opName_, "group num should be 64, but got %u", gSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qkHeadDim_ != 512, // 512:当前不泛化
        OP_LOGE(opName_, "q_head_dim only support 512, but got %u", qkHeadDim_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckFeatureAntiquantLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQ;
    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutQuery) == layoutSupportList.end(),
        OP_LOGE(opName_, "layoutQuery only supports BSND/TND, but got %s", layoutQuery.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckFeatureAntiquantDtype() const
{
    OP_CHECK_IF(qType_ != ge::DT_BF16 && qType_ != ge::DT_FLOAT16,
        OP_LOGE(opName_, "query dtype only support %s and %s, but got %s",
            SASDataTypeToSerialString(ge::DT_BF16).c_str(), SASDataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            SASDataTypeToSerialString(qType_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckFeatureAntiquantAttr() const
{
    OP_CHECK_IF(*opParamInfo_.kvQuantMode != 1,
        OP_LOGE(opName_, "kv_quant_mode_ should be 1, but got %d",
        *opParamInfo_.kvQuantMode),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(*opParamInfo_.tileSize != 64, // 64:当前不泛化
        OP_LOGE(opName_, "tile_size should be 64, but got %ld",
        *opParamInfo_.tileSize),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(*opParamInfo_.ropeHeadDim != 64, // 64:当前不泛化
        OP_LOGE(opName_, "rope_head_dim should be 64, but got %d",
        *opParamInfo_.ropeHeadDim),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckFeatureAntiquantPa() const
{
    OP_CHECK_IF(oriBlockSize_ <= 0 || oriBlockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
        OP_LOGE(opName_, "when page attention is enabled, oriBlockSize_(%ld) should be in range (0, %u].",
        oriBlockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);

    if (cmpBlockSize_ != 0){
        OP_CHECK_IF(cmpBlockSize_ <= 0 || cmpBlockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
            OP_LOGE(opName_, "when page attention is enabled, cmpBlockSize_(%ld) should be in range (0, %u].",
            cmpBlockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckFeatureAntiquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureAntiquantAttr() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KvQuantSASTilingCheck::CheckFeature() const
{
    return CheckFeatureAntiquant();
}

}