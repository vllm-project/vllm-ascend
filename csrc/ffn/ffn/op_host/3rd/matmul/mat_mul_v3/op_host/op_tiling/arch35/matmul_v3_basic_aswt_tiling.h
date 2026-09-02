/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_basic_aswt_tiling.h
 * \brief
 */
#pragma once

#include "matmul_v3_asw_tiling.h"
#include "matmul_v3_common_advanced.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3BasicAswtTiling : public MatMulV3AswTiling {
public:
    MatMulV3BasicAswtTiling(gert::TilingContext* context, MatMulTilingCfg& cfg) : MatMulV3AswTiling(context, cfg){};
    ~MatMulV3BasicAswtTiling() override = default;
    void CheckFp32SplitK();
    void CheckApiLevelAndModel();
    void CheckIsSplitN();

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetTilingData(TilingResult& tiling) const override;
    void DoBL1FullLoad();
    void DoAL1FullLoad();
    MatMulV3FullLoad fullLoad_{MatMulV3FullLoad::NONE_FULL_LOAD};
    MatMulV3L0C2Out l0C2Out_{MatMulV3L0C2Out::ON_THE_FLY};
    MatMulV3ApiLevel apiLevel_{MatMulV3ApiLevel::BASIC_LEVEL};
    MatMulV3Model model_{MatMulV3Model::BASIC};
    bool isSlice_{false};

private:
    void ResetFullLoadLoadBalance();

    void CalcTailBasicBlockBL1Full();
    void CalcTailBasicBlockAL1Full();
    bool CheckBL1FullLoad() const;
    bool CheckAL1FullLoad() const;
};
} // namespace matmul_v3_advanced
} // namespace optiling
