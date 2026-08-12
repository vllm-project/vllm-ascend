/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/* !
 * \file new_mc2_matmul_tiling_cfg.h
 * \brief
 */

#ifndef __NEW_MC2_MATMUL_TILING_CFG_H__
#define __NEW_MC2_MATMUL_TILING_CFG_H__

#pragma once
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_data.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_common_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_cfg.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "mc2_tiling_struct.h"

namespace Mc2MatmulHelper {
class NewMc2MatmulTilingCfg : public optiling::Mc2MatMulTilingCfg {
public:
    NewMc2MatmulTilingCfg(const void *compileInfoIn, const void *argsIn, uint32_t baseMLimit = 0,
                          bool needUpdateIn = true)
        : Mc2MatMulTilingCfg(needUpdateIn, compileInfoIn, argsIn), baseMLimit_(baseMLimit)
    {
    }

    void SetMatMulV3TilingData(optiling::MC2MatmulV3TilingData &tilingData);
    void Update(const optiling::Mc2TilingResult &result) override;
    void SetRankDim(uint64_t rankDim);
    void SetCommCnt(uint64_t commCnt);

private:
    void SetMMTilingData() const;
    void SetTailCntAndType() const;
    void DealBaseBlock() const;

    int32_t baseMLimit_{0};
    uint64_t rankDim_{0};
    uint64_t commCnt_{0};
    optiling::MC2MatmulV3TilingData *mc2MmV3TilingData_{nullptr};
    Mc2MatMulV3TilingData *mmv3TilingData_{nullptr};
};

} // namespace Mc2MatmulHelper

#endif
