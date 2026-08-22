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
 * \file fia_tiling_data_noquant_gqa.h
 * \brief
 */

#ifndef FIA_TILING_DATA_NOQUANT_GQA_H_
#define FIA_TILING_DATA_NOQUANT_GQA_H_

#include "../fia_tiling_data_public.h"

namespace optiling {

class NoQuantTilingArch35 {
public:
    FiaBaseParams fiaBaseParams;
    FiaAttenMaskParams fiaAttenMaskParams;
    FiaPseParams fiaPseParams;
    FiaSystemPrefixParams fiaSystemPrefixParams;
    FiaPageAttentionParams fiaPageAttentionParams;
    FiaLeftPaddingParams fiaLeftPaddingParams;
    FiaPostQuantParams fiaPostQuantParams;
    FiaWorkspaceParams fiaWorkspaceParams;
    FiaS1OuterSplitCoreParams fiaS1OuterSplitCoreParams;
    FiaEmptyTensorParams fiaEmptyTensorParams;
};

class FusedInferAttentionScoreTilingData {
public:
    NoQuantTilingArch35 baseTiling;
    FiaMetaData fiaMetaData;
};

} // namespace optiling
#endif