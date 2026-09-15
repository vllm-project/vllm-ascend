/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TURBOQUANT_RESHAPE_AND_CACHE_V310_TILING_H
#define TURBOQUANT_RESHAPE_AND_CACHE_V310_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "../op_kernel/turboquant_reshape_and_cache_v310_tiling_data.h"

namespace optiling {

struct TurboquantReshapeAndCacheV310CompileInfo {
    uint32_t coreNum{0};
};

class TurboquantReshapeAndCacheV310Tiling {
public:
    explicit TurboquantReshapeAndCacheV310Tiling(gert::TilingContext *context) : context_(context) {}

    ge::graphStatus Run();

private:
    ge::graphStatus ParseInputs();
    ge::graphStatus ComputeSplit();
    ge::graphStatus PostTiling();

    gert::TilingContext *context_{nullptr};
    TurboquantReshapeAndCacheV310TilingData tilingData_{};
    uint64_t tilingKey_{0};
    uint64_t workspaceSize_{0};
    uint32_t bits_{3};
};

ge::graphStatus TilingForTurboquantReshapeAndCacheV310(gert::TilingContext *context);

}  // namespace optiling

#endif  // TURBOQUANT_RESHAPE_AND_CACHE_V310_TILING_H
