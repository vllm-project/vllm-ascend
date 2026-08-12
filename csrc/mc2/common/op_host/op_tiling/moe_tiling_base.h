/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_tiling_base.h
 * \brief
 */

#ifndef MOE_TILING_BASE_H
#define MOE_TILING_BASE_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
#include "mc2_log.h"
#include "op_host/tiling_base.h"
#include "mc2_tiling_struct.h"
#include "matmul_formulaic_tiling.h"
#include "mc2_tiling_utils.h"
#include "platform/platform_infos_def.h"

namespace optiling {

class MoeTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MoeTilingBase(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context) {};

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    NpuArch npuArch_;
};
} // namespace optiling

#endif // MOE_TILING_BASE_H
