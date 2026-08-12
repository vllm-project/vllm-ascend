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
 * \file matmul_v3_asw_basic_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_ASW_BASIC_TILING_H__
#define __OP_HOST_MATMUL_V3_ASW_BASIC_TILING_H__

#include "matmul_v3_base_tiling_advanced.h"
#include "matmul_v3_asw_tiling.h"

namespace optiling {
namespace mc2_matmul_v3_advanced {
class Mc2MatMulV3AswBasicApiTiling : public Mc2MatMulV3AswTiling {
public:
    Mc2MatMulV3AswBasicApiTiling(gert::TilingContext *context, Mc2MatMulTilingCfg &cfg)
        : Mc2MatMulV3AswTiling(context, cfg) {};

    ~Mc2MatMulV3AswBasicApiTiling() override {};

protected:
    bool IsCapable() override;

    uint64_t GetTilingKey() const override;
};
} // namespace mc2_matmul_v3_advanced
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_ASW_BASIC_TILING_H__
