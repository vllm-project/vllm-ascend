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
 * \file mc2_matmul_tiling_cfg.cpp
 * \brief
 */

#include "mc2_matmul_tiling_cfg.h"
#include "util/math_util.h"

namespace Mc2MatmulHelper {

constexpr int32_t BASEM_ALIGN_SIZE = 16;

void Mc2MatmulTilingCfg::SetMatMulV3TilingData(Mc2MatMulV3TilingData &tilingData)
{
    mc2MmV3TilingData_ = &tilingData;
}

void Mc2MatmulTilingCfg::Update(const optiling::Mc2TilingResult &result)
{
    mmv3TilingData_ = static_cast<Mc2MatMulV3TilingData *>(result.tilingData);

    if ((mmv3TilingData_ == nullptr) || (mc2MmV3TilingData_ == nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT("Mc2MatmulTilingCfg", "mc2MmV3TilingData_, mmv3TilingData_");
        return;
    }

    SetMMTilingData();

    SetTailCntAndType();
}

void Mc2MatmulTilingCfg::SetMMTilingData() const
{
    DealBaseBlock();
    mc2MmV3TilingData_->tCubeTiling.usedCoreNum = mmv3TilingData_->tCubeTiling.usedCoreNum;
    mc2MmV3TilingData_->tCubeTiling.M = mmv3TilingData_->tCubeTiling.M;
    mc2MmV3TilingData_->tCubeTiling.N = mmv3TilingData_->tCubeTiling.N;
    mc2MmV3TilingData_->tCubeTiling.Ka = mmv3TilingData_->tCubeTiling.Ka;
    mc2MmV3TilingData_->tCubeTiling.Kb = mmv3TilingData_->tCubeTiling.Kb;
    mc2MmV3TilingData_->tCubeTiling.depthA1 = mmv3TilingData_->tCubeTiling.depthA1;
    mc2MmV3TilingData_->tCubeTiling.depthB1 = mmv3TilingData_->tCubeTiling.depthB1;
    mc2MmV3TilingData_->tCubeTiling.stepM = mmv3TilingData_->tCubeTiling.stepM;
    mc2MmV3TilingData_->tCubeTiling.stepN = mmv3TilingData_->tCubeTiling.stepN;
    mc2MmV3TilingData_->tCubeTiling.isBias = mmv3TilingData_->tCubeTiling.isBias;
    mc2MmV3TilingData_->tCubeTiling.transLength = mmv3TilingData_->tCubeTiling.transLength;
    mc2MmV3TilingData_->tCubeTiling.iterateOrder = mmv3TilingData_->tCubeTiling.iterateOrder;
    mc2MmV3TilingData_->tCubeTiling.shareMode = mmv3TilingData_->tCubeTiling.shareMode;
    mc2MmV3TilingData_->tCubeTiling.shareL1Size = mmv3TilingData_->tCubeTiling.shareL1Size;
    mc2MmV3TilingData_->tCubeTiling.shareL0CSize = mmv3TilingData_->tCubeTiling.shareL0CSize;
    mc2MmV3TilingData_->tCubeTiling.shareUbSize = mmv3TilingData_->tCubeTiling.shareUbSize;
    mc2MmV3TilingData_->tCubeTiling.batchM = mmv3TilingData_->tCubeTiling.batchM;
    mc2MmV3TilingData_->tCubeTiling.batchN = mmv3TilingData_->tCubeTiling.batchN;
    mc2MmV3TilingData_->tCubeTiling.singleBatchM = mmv3TilingData_->tCubeTiling.singleBatchM;
    mc2MmV3TilingData_->tCubeTiling.singleBatchN = mmv3TilingData_->tCubeTiling.singleBatchN;
    mc2MmV3TilingData_->tCubeTiling.stepKa = mmv3TilingData_->tCubeTiling.stepKa;
    mc2MmV3TilingData_->tCubeTiling.stepKb = mmv3TilingData_->tCubeTiling.stepKb;
    mc2MmV3TilingData_->tCubeTiling.depthAL1CacheUB = mmv3TilingData_->tCubeTiling.depthAL1CacheUB;
    mc2MmV3TilingData_->tCubeTiling.depthBL1CacheUB = mmv3TilingData_->tCubeTiling.depthBL1CacheUB;
    mc2MmV3TilingData_->tCubeTiling.dbL0A = mmv3TilingData_->tCubeTiling.dbL0A;
    mc2MmV3TilingData_->tCubeTiling.dbL0B = mmv3TilingData_->tCubeTiling.dbL0B;
    mc2MmV3TilingData_->tCubeTiling.dbL0C = mmv3TilingData_->tCubeTiling.dbL0C;
    mc2MmV3TilingData_->tCubeTiling.ALayoutInfoB = mmv3TilingData_->tCubeTiling.ALayoutInfoB;
    mc2MmV3TilingData_->tCubeTiling.ALayoutInfoS = mmv3TilingData_->tCubeTiling.ALayoutInfoS;
    mc2MmV3TilingData_->tCubeTiling.ALayoutInfoN = mmv3TilingData_->tCubeTiling.ALayoutInfoN;
    mc2MmV3TilingData_->tCubeTiling.ALayoutInfoG = mmv3TilingData_->tCubeTiling.ALayoutInfoG;
    mc2MmV3TilingData_->tCubeTiling.ALayoutInfoD = mmv3TilingData_->tCubeTiling.ALayoutInfoD;
    mc2MmV3TilingData_->tCubeTiling.BLayoutInfoB = mmv3TilingData_->tCubeTiling.BLayoutInfoB;
    mc2MmV3TilingData_->tCubeTiling.BLayoutInfoS = mmv3TilingData_->tCubeTiling.BLayoutInfoS;
    mc2MmV3TilingData_->tCubeTiling.BLayoutInfoN = mmv3TilingData_->tCubeTiling.BLayoutInfoN;
    mc2MmV3TilingData_->tCubeTiling.BLayoutInfoG = mmv3TilingData_->tCubeTiling.BLayoutInfoG;
    mc2MmV3TilingData_->tCubeTiling.BLayoutInfoD = mmv3TilingData_->tCubeTiling.BLayoutInfoD;
    mc2MmV3TilingData_->tCubeTiling.CLayoutInfoB = mmv3TilingData_->tCubeTiling.CLayoutInfoB;
    mc2MmV3TilingData_->tCubeTiling.CLayoutInfoS1 = mmv3TilingData_->tCubeTiling.CLayoutInfoS1;
    mc2MmV3TilingData_->tCubeTiling.CLayoutInfoS2 = mmv3TilingData_->tCubeTiling.CLayoutInfoS2;
    mc2MmV3TilingData_->tCubeTiling.CLayoutInfoN = mmv3TilingData_->tCubeTiling.CLayoutInfoN;
    mc2MmV3TilingData_->tCubeTiling.CLayoutInfoG = mmv3TilingData_->tCubeTiling.CLayoutInfoG;
    mc2MmV3TilingData_->tCubeTiling.BatchNum = mmv3TilingData_->tCubeTiling.BatchNum;
    mc2MmV3TilingData_->tCubeTiling.mxTypePara = mmv3TilingData_->tCubeTiling.mxTypePara;
}

void Mc2MatmulTilingCfg::SetTailCntAndType() const
{
    mc2MmV3TilingData_->kTailCnt = mmv3TilingData_->kTailCnt;
    if (baseMLimit_ < 0) {
        OP_LOGE_FOR_INVALID_VALUE("Mc2MatmulTilingCfg", "baseMLimit_", std::to_string(baseMLimit_).c_str(),
                                  "should be >= 0");
        return;
    } else if (baseMLimit_ == 0) {
        mc2MmV3TilingData_->mTailCnt = mmv3TilingData_->mTailCnt;
        mc2MmV3TilingData_->nTailCnt = mmv3TilingData_->nTailCnt;
        mc2MmV3TilingData_->mBaseTailSplitCnt = mmv3TilingData_->mBaseTailSplitCnt;
        mc2MmV3TilingData_->nBaseTailSplitCnt = mmv3TilingData_->nBaseTailSplitCnt;
        mc2MmV3TilingData_->mTailMain = mmv3TilingData_->mTailMain;
        mc2MmV3TilingData_->nTailMain = mmv3TilingData_->nTailMain;
        mc2MmV3TilingData_->isHf32 = mmv3TilingData_->isHf32;
        mc2MmV3TilingData_->aswWindowLen = mmv3TilingData_->aswWindowLen;
        return;
    }

    int64_t baseM = static_cast<int64_t>(mc2MmV3TilingData_->tCubeTiling.baseM);
    int64_t baseN = static_cast<int64_t>(mc2MmV3TilingData_->tCubeTiling.baseN);
    int64_t nValue = static_cast<int64_t>(mc2MmV3TilingData_->tCubeTiling.N);
    int64_t coreNum = static_cast<int64_t>(mc2MmV3TilingData_->tCubeTiling.usedCoreNum);
    if ((baseM == 0) || (baseN == 0) || (coreNum == 0)) {
        OP_LOGE_FOR_INVALID_VALUE(
            "Mc2MatmulTilingCfg", "baseM, baseN, coreNum",
            (std::to_string(baseM) + ", " + std::to_string(baseN) + ", " + std::to_string(coreNum)).c_str(),
            "all should be non-zero");
        return;
    }
    int64_t mCnt = static_cast<int64_t>(((static_cast<int64_t>(baseMLimit_) + baseM - 1) / baseM) *
                                        static_cast<int64_t>(rankDim_) * static_cast<int64_t>(commCnt_));
    int64_t nCnt = static_cast<int64_t>((nValue + baseN - 1) / baseN);
    int64_t mnCnt = static_cast<int64_t>(mCnt * nCnt);
    int64_t tailCnt = static_cast<int64_t>(mnCnt % coreNum);

    int64_t mTailCnt = 1;
    int64_t nTailCnt = 1;

    if (tailCnt != 0) {
        while ((mTailCnt + 1) * nTailCnt * tailCnt <= coreNum) {
            mTailCnt += 1;
            if (mTailCnt * (nTailCnt + 1) * tailCnt <= coreNum) {
                nTailCnt += 1;
            }
        }
    }

    mc2MmV3TilingData_->mTailCnt = static_cast<unsigned int>(mTailCnt);
    mc2MmV3TilingData_->nTailCnt = static_cast<unsigned int>(nTailCnt);
    mc2MmV3TilingData_->mBaseTailSplitCnt = mmv3TilingData_->mBaseTailSplitCnt;
    mc2MmV3TilingData_->nBaseTailSplitCnt = mmv3TilingData_->nBaseTailSplitCnt;
    mc2MmV3TilingData_->mTailMain = mmv3TilingData_->mTailMain;
    mc2MmV3TilingData_->nTailMain = mmv3TilingData_->nTailMain;
    mc2MmV3TilingData_->isHf32 = mmv3TilingData_->isHf32;
    mc2MmV3TilingData_->aswWindowLen = mmv3TilingData_->aswWindowLen;
}

void Mc2MatmulTilingCfg::DealBaseBlock() const
{
    if (baseMLimit_ < 0) {
        OP_LOGE_FOR_INVALID_VALUE("Mc2MatmulTilingCfg", "baseMLimit_", std::to_string(baseMLimit_).c_str(),
                                  "should be >= 0");
        return;
    } else if ((baseMLimit_ > 0) && (mmv3TilingData_->tCubeTiling.baseM > baseMLimit_)) {
        mc2MmV3TilingData_->tCubeTiling.singleCoreM = baseMLimit_;
        mc2MmV3TilingData_->tCubeTiling.baseM = Ops::Base::CeilAlign(baseMLimit_, BASEM_ALIGN_SIZE);
    } else {
        mc2MmV3TilingData_->tCubeTiling.singleCoreM = mmv3TilingData_->tCubeTiling.singleCoreM;
        mc2MmV3TilingData_->tCubeTiling.baseM = mmv3TilingData_->tCubeTiling.baseM;
    }

    mc2MmV3TilingData_->tCubeTiling.singleCoreN = mmv3TilingData_->tCubeTiling.singleCoreN;
    mc2MmV3TilingData_->tCubeTiling.singleCoreK = mmv3TilingData_->tCubeTiling.singleCoreK;
    mc2MmV3TilingData_->tCubeTiling.baseN = mmv3TilingData_->tCubeTiling.baseN;
    mc2MmV3TilingData_->tCubeTiling.baseK = mmv3TilingData_->tCubeTiling.baseK;
}

void Mc2MatmulTilingCfg::SetRankDim(uint64_t rankDim)
{
    rankDim_ = rankDim;
}

void Mc2MatmulTilingCfg::SetCommCnt(uint64_t commCnt)
{
    commCnt_ = commCnt;
}

} // namespace Mc2MatmulHelper
