/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tbe_tiling_api.h"

using namespace optiling;

namespace optiling {
bool GetTbeTiling(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoForV2, Conv3dBackpropV2TBETilingData& tbeTilingForV2)
{
    (void)context;
    (void)runInfoForV2;
    (void)tbeTilingForV2;
    return true;
}
bool GetTbeTiling(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
    Conv3dBackpropV2TBETilingData& tbeTilingForV2, const optiling::OpTypeV2 opType)
{
    (void)context;
    (void)runInfoV2;
    (void)tbeTilingForV2;
    (void)opType;
    return true;
}

bool GetTbeTiling(gert::TilingContext* context, Conv3dBackpropV2TBETilingData& tbeTilingForV2, const optiling::OpTypeV2 opType)
{
    (void)context;
    (void)tbeTilingForV2;
    (void)opType;
    return true;
}
}