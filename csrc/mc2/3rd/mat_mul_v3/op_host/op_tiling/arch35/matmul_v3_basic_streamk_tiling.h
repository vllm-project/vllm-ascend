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
 * \file matmul_v3_basic_streamk_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_BASIC_STREAMK_H__
#define __OP_HOST_MATMUL_V3_BASIC_STREAMK_H__

#include "matmul_v3_base_tiling_advanced.h"

namespace optiling {
namespace mc2_matmul_v3_advanced {
class Mc2MatMulV3BasicStreamKTiling : public Mc2MatMulV3BaseTiling {
public:
    Mc2MatMulV3BasicStreamKTiling(gert::TilingContext *context, Mc2MatMulTilingCfg &cfg)
        : Mc2MatMulV3BaseTiling(context, cfg) {};

    ~Mc2MatMulV3BasicStreamKTiling() override {};

protected:
    bool IsCapable() override;

    ge::graphStatus DoOpTiling() override;

    uint64_t GetTilingKey() const override;

    std::vector<size_t> GetWorkspaceSize() const override;

private:
    bool CheckStreamKSKTiling() const;

    bool CheckStreamKDPSKTiling() const;

    Mc2MatMulV3L0C2Out GetL0C2OutFlag() const;

    uint64_t mCnt_{1};
    uint64_t nCnt_{1};
    uint64_t totalMNCnt_{1};
    Mc2MatMulV3L0C2Out l0C2Out_{Mc2MatMulV3L0C2Out::ON_THE_FLY};
};
} // namespace mc2_matmul_v3_advanced
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_BASIC_STREAM_K_H__
