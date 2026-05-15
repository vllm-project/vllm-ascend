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
    if (ge::GRAPH_SUCCESS != CheckDequantScaleNotExistence()) {
        return ge::GRAPH_FAILED;
    }

    return CheckParaExistenceAntiquant();
}

ge::graphStatus KvQuantSASTilingCheck::CheckDequantScaleNotExistence()
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
        OP_LOGE(opName_, "rope_head_dim should be 64, but got %ld",
        *opParamInfo_.ropeHeadDim),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

}