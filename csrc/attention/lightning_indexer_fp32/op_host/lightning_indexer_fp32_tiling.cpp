/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../lightning_indexer/op_host/lightning_indexer_tiling.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(LightningIndexerFp32, LITilingData)

static ge::graphStatus TilingForLightningIndexerFp32(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    LITilingInfo info;
    LIInfoParser parser(context, true);
    if (parser.ParseAndCheck(info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    LightningIndexerTiling tiling(context);
    return tiling.DoTiling(&info);
}

IMPL_OP_OPTILING(LightningIndexerFp32).Tiling(TilingForLightningIndexerFp32);
} // namespace optiling
