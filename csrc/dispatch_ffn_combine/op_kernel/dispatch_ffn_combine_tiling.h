/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_ffn_combine_tiling.h
 * \brief
 */

#include "moe_init_routing_quant_v2/moe_init_routing_v2_tiling.h"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

#ifndef ASCENDC_DISPATCH_FFN_COMBINE_TILING_H
#define ASCENDC_DISPATCH_FFN_COMBINE_TILING_H
struct DispatchFFNCombineInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t expertPerRank;
    uint32_t maxOutputSize;
    uint32_t isTransposeB;
    uint32_t isWeightNz;
    uint32_t aivNum;
    uint32_t totalUbSize;
    uint32_t topK;
    uint32_t worldSize;
};

struct CoCTiling {
    int32_t m0 = -1;
    int32_t k0 = -1;
    int32_t n0 = -1;
    int32_t swizzleDirect = -1;
    int32_t swizzleOffset = -1;
    int32_t ubMoveNum = -1;
    int32_t pValue = -1;
    int32_t commNpuSplit = -1;
    int32_t commDataSplit = -1;
    int32_t lenPerLoop = -1;
    uint64_t initRoutingQuantTilingKey;
    optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
};

struct DispatchFFNCombineTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    DispatchFFNCombineInfo dispatchFFNCombineInfo;
    CoCTiling cocTiling;
};
#endif