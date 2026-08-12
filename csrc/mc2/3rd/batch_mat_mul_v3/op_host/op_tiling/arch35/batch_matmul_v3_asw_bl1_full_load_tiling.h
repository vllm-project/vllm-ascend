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
 * \file batch_matmul_v3_asw_bl1_full_load_tiling.h
 * \brief
 */

#ifndef __OP_HOST_BATCH_MATMUL_V3_ASW_BL1_FULL_LOAD_TILING_H__
#define __OP_HOST_BATCH_MATMUL_V3_ASW_BL1_FULL_LOAD_TILING_H__

#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_base_tiling_advanced.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_asw_full_load_tiling.h"

namespace optiling {
namespace Mc2batch_matmul_v3_advanced {
using namespace mc2_matmul_v3_advanced;
class Mc2BatchMatMulV3AswBL1FullLoadTiling : public Mc2MatMulV3AswFullLoadTiling {
public:
    Mc2BatchMatMulV3AswBL1FullLoadTiling(gert::TilingContext *context, Mc2MatMulTilingCfg &cfg)
        : Mc2MatMulV3AswFullLoadTiling(context, cfg) {};

    ~Mc2BatchMatMulV3AswBL1FullLoadTiling() override {};

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

protected:
    bool isBl1MulCoreLoad_{false};
};
} // namespace Mc2batch_matmul_v3_advanced
} // namespace optiling
#endif // __OP_HOST_BATCH_MATMUL_V3_ASW_BL1_FULL_LOAD_TILING_H__
