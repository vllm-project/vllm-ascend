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
 * \file attn_res_fwd_tiling.h
 * \brief AttnResFwd host tiling header
 */
#ifndef ATTN_RES_FWD_TILING_H
#define ATTN_RES_FWD_TILING_H

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"
#include "../op_kernel/attn_res_fwd_tiling_data.h"
#include "../op_kernel/tiling_key_attn_res_fwd.h"

namespace optiling {

struct AttnResFwdCompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

struct AttnResFwdInfo {
    const char *opName = "AttnResFwd";
};

class AttnResFwdTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AttnResFwdTiling(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    }
    ~AttnResFwdTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

protected:
    void InitCompileInfo();
    void PrintTilingData();
    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus GetNormEps();
    ge::graphStatus GetNeedBackward();
    ge::graphStatus FillStagingFields();
    uint64_t EstimateUbComputeBytes(bool resident) const;
    bool CanResidentAllBlocks(uint32_t blockCount, uint32_t hiddenSize) const;
    uint32_t CalcMaxResidentRows(uint32_t hiddenSize) const;
    uint32_t GetMinStagingBytes() const;

    AttnResFwdCompileInfo compileInfo_;
    AttnResFwd::AttnResFwdTilingData tilingData_;
    AttnResFwdInfo inputParams_;
    uint64_t tilingKey_{TILING_KEY_BF16_RELOAD};
    uint64_t workspaceSize_{0UL};
};

} // namespace optiling
#endif // ATTN_RES_FWD_TILING_H
