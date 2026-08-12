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
 * \file matmul_v3_asw_tiling.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_ASW_TILING_H__
#define __OP_HOST_MATMUL_V3_ASW_TILING_H__

#include "matmul_v3_base_tiling_advanced.h"

namespace optiling {
namespace mc2_matmul_v3_advanced {
class Mc2MatMulV3AswTiling : public Mc2MatMulV3BaseTiling {
public:
    Mc2MatMulV3AswTiling(gert::TilingContext *context, Mc2MatMulTilingCfg &cfg)
        : Mc2MatMulV3BaseTiling(context, cfg) {};

    ~Mc2MatMulV3AswTiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    };

    ge::graphStatus DoOpTiling() override;

    ge::graphStatus DoNormOpTiling();

    uint64_t GetTilingKey() const override;

    Mc2MatMulV3Model aswtModel_{Mc2MatMulV3Model::BASIC};

private:
    struct CalcParams {
        uint64_t baseStart;
        uint64_t baseEnd;
        bool isNegativeSign;
        double bestBalance;
        uint64_t baseM;
        uint64_t baseN;
        uint64_t baseK;
    };

    void CalcBasicBlock();
    void FormulateBasicBlock();
    void FormulateLoadBalanceBlock();
    void CalcTailBasicBlock();
    void OptimizeEdgeBasicBlock();
    void GetOuterAxisTailCnt(bool nLoadBalance, uint64_t &baseTailSplitCnt, uint64_t &tailMain);
    void CalcSingleX(uint64_t &higherSingleX, uint64_t &lowerSingleX);
    // 单边大场景处理
    uint64_t UpdateBaseBlock(uint64_t baseBlock, bool isMLarger);
    void CalcLargeSingleSide(uint64_t maxMN, uint64_t &targetBase, bool isMLarger);
    void HandleLargeSingleSide(uint64_t minMN, uint64_t maxMN, bool isMLarger);
    // 两边都比较大场景处理
    bool UpdateBothBaseBlock(double balance, CalcParams &params, uint64_t currentBaseM, uint64_t currentBaseN,
                             uint64_t baseK);
    bool CalcBestBalance(CalcParams &params, bool isMLarger);
    void HandleLargeBothSides(uint64_t higherSingleX, uint64_t lowerSingleX, uint64_t minMN, bool isMLarger);
};
} // namespace mc2_matmul_v3_advanced
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_ASW_TILING_H__
