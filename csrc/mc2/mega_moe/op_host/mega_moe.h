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
 * \file mega_moe.h
 * \brief
 */

#ifndef MEGA_MOE_TILING
#define MEGA_MOE_TILING

#include <cstdint>
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_base.h"

namespace optiling {

struct MegaMoeConfig {
    uint32_t contextIndex = 0U;
    uint32_t xIndex = 1U;
    uint32_t topkIdsIndex = 2U;
    uint32_t topkWeightsIndex = 3U;
    uint32_t weight1Index = 4U;
    uint32_t weight2Index = 5U;
    uint32_t weightScales1Index = 6U;
    uint32_t weightScales2Index = 7U;
    uint32_t bias1Index = 8U;
    uint32_t bias2Index = 9U;
    uint32_t xActiveMaskIndex = 10U;
    uint32_t scalesIndex = 11U;
    uint32_t yIndex = 0U;
    uint32_t expertTokenNumsIndex = 1U;
    uint32_t attrMoeExpertNumIndex = 0U;
    uint32_t attrEpWorldSizeIndex = 1U;
    uint32_t attrCclBufferSizeIndex = 2U;
    uint32_t attrMaxRecvTokenNumIndex = 3U;
    uint32_t attrDispatchQuantModeIndex = 4U;
    uint32_t attrDispatchQuantOutDtypeIndex = 5U;
    uint32_t attrCombineQuantModeIndex = 6U;
    uint32_t attrCommAlgIndex = 7U;
    uint32_t attrNumMaxTokensPerRankIndex = 8U;
    uint32_t attrActivationIndex = 9U;
    uint32_t attrActivationClampIndex = 10U;
    uint32_t attrActivationOutDtypeIndex = 11U;
    uint32_t attrTransposeWeight1Index = 12U;
    uint32_t attrTransposeWeight2Index = 13U;
    uint32_t attrWeight1InterleaveIndex = 14U;
    uint32_t attrTopoTypeIndex = 15U;
    uint32_t attrRankNumPerServerIndex = 16U;
    bool isMc2Context = false;
};

ge::graphStatus MegaMoeTilingFuncImplPublic(gert::TilingContext* context, MegaMoeConfig& config);

}

#endif