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
 * \file new_mc2_matmul_tiling_cfg.cpp
 * \brief
 */

#include "new_mc2_matmul_tiling_cfg.h"

namespace Mc2MatmulHelper {
void NewMc2MatmulTilingCfg::SetMatMulV3TilingData(optiling::MC2MatmulV3TilingData &tilingData)
{
    mc2MmV3TilingData_ = &tilingData;
}

void NewMc2MatmulTilingCfg::Update(const optiling::Mc2TilingResult &result)
{
    mmv3TilingData_ = static_cast<Mc2MatMulV3TilingData *>(result.tilingData);

    if ((mmv3TilingData_ == nullptr) || (mc2MmV3TilingData_ == nullptr)) {
        OP_LOGE_WITH_INVALID_INPUT("Mc2MatmulTilingCfg", "mc2MmV3TilingData_, mmv3TilingData_");
        return;
    }

    SetMMTilingData();

    SetTailCntAndType();
}

void NewMc2MatmulTilingCfg::SetMMTilingData() const
{
    DealBaseBlock();
    mc2MmV3TilingData_->matmulTiling.set_usedCoreNum(mmv3TilingData_->tCubeTiling.usedCoreNum);
    mc2MmV3TilingData_->matmulTiling.set_M(mmv3TilingData_->tCubeTiling.M);
    mc2MmV3TilingData_->matmulTiling.set_N(mmv3TilingData_->tCubeTiling.N);
    mc2MmV3TilingData_->matmulTiling.set_Ka(mmv3TilingData_->tCubeTiling.Ka);
    mc2MmV3TilingData_->matmulTiling.set_Kb(mmv3TilingData_->tCubeTiling.Kb);
    mc2MmV3TilingData_->matmulTiling.set_depthA1(mmv3TilingData_->tCubeTiling.depthA1);
    mc2MmV3TilingData_->matmulTiling.set_depthB1(mmv3TilingData_->tCubeTiling.depthB1);
    mc2MmV3TilingData_->matmulTiling.set_stepM(mmv3TilingData_->tCubeTiling.stepM);
    mc2MmV3TilingData_->matmulTiling.set_stepN(mmv3TilingData_->tCubeTiling.stepN);
    mc2MmV3TilingData_->matmulTiling.set_isBias(mmv3TilingData_->tCubeTiling.isBias);
    mc2MmV3TilingData_->matmulTiling.set_transLength(mmv3TilingData_->tCubeTiling.transLength);
    mc2MmV3TilingData_->matmulTiling.set_iterateOrder(mmv3TilingData_->tCubeTiling.iterateOrder);
    mc2MmV3TilingData_->matmulTiling.set_shareMode(mmv3TilingData_->tCubeTiling.shareMode);
    mc2MmV3TilingData_->matmulTiling.set_shareL1Size(mmv3TilingData_->tCubeTiling.shareL1Size);
    mc2MmV3TilingData_->matmulTiling.set_shareL0CSize(mmv3TilingData_->tCubeTiling.shareL0CSize);
    mc2MmV3TilingData_->matmulTiling.set_shareUbSize(mmv3TilingData_->tCubeTiling.shareUbSize);
    mc2MmV3TilingData_->matmulTiling.set_batchM(mmv3TilingData_->tCubeTiling.batchM);
    mc2MmV3TilingData_->matmulTiling.set_batchN(mmv3TilingData_->tCubeTiling.batchN);
    mc2MmV3TilingData_->matmulTiling.set_singleBatchM(mmv3TilingData_->tCubeTiling.singleBatchM);
    mc2MmV3TilingData_->matmulTiling.set_singleBatchN(mmv3TilingData_->tCubeTiling.singleBatchN);
    mc2MmV3TilingData_->matmulTiling.set_stepKa(mmv3TilingData_->tCubeTiling.stepKa);
    mc2MmV3TilingData_->matmulTiling.set_stepKb(mmv3TilingData_->tCubeTiling.stepKb);
    mc2MmV3TilingData_->matmulTiling.set_depthAL1CacheUB(mmv3TilingData_->tCubeTiling.depthAL1CacheUB);
    mc2MmV3TilingData_->matmulTiling.set_depthBL1CacheUB(mmv3TilingData_->tCubeTiling.depthBL1CacheUB);
    mc2MmV3TilingData_->matmulTiling.set_dbL0A(mmv3TilingData_->tCubeTiling.dbL0A);
    mc2MmV3TilingData_->matmulTiling.set_dbL0B(mmv3TilingData_->tCubeTiling.dbL0B);
    mc2MmV3TilingData_->matmulTiling.set_dbL0C(mmv3TilingData_->tCubeTiling.dbL0C);
    mc2MmV3TilingData_->matmulTiling.set_ALayoutInfoB(mmv3TilingData_->tCubeTiling.ALayoutInfoB);
    mc2MmV3TilingData_->matmulTiling.set_ALayoutInfoS(mmv3TilingData_->tCubeTiling.ALayoutInfoS);
    mc2MmV3TilingData_->matmulTiling.set_ALayoutInfoN(mmv3TilingData_->tCubeTiling.ALayoutInfoN);
    mc2MmV3TilingData_->matmulTiling.set_ALayoutInfoG(mmv3TilingData_->tCubeTiling.ALayoutInfoG);
    mc2MmV3TilingData_->matmulTiling.set_ALayoutInfoD(mmv3TilingData_->tCubeTiling.ALayoutInfoD);
    mc2MmV3TilingData_->matmulTiling.set_BLayoutInfoB(mmv3TilingData_->tCubeTiling.BLayoutInfoB);
    mc2MmV3TilingData_->matmulTiling.set_BLayoutInfoS(mmv3TilingData_->tCubeTiling.BLayoutInfoS);
    mc2MmV3TilingData_->matmulTiling.set_BLayoutInfoN(mmv3TilingData_->tCubeTiling.BLayoutInfoN);
    mc2MmV3TilingData_->matmulTiling.set_BLayoutInfoG(mmv3TilingData_->tCubeTiling.BLayoutInfoG);
    mc2MmV3TilingData_->matmulTiling.set_BLayoutInfoD(mmv3TilingData_->tCubeTiling.BLayoutInfoD);
    mc2MmV3TilingData_->matmulTiling.set_CLayoutInfoB(mmv3TilingData_->tCubeTiling.CLayoutInfoB);
    mc2MmV3TilingData_->matmulTiling.set_CLayoutInfoS1(mmv3TilingData_->tCubeTiling.CLayoutInfoS1);
    mc2MmV3TilingData_->matmulTiling.set_CLayoutInfoS2(mmv3TilingData_->tCubeTiling.CLayoutInfoS2);
    mc2MmV3TilingData_->matmulTiling.set_CLayoutInfoN(mmv3TilingData_->tCubeTiling.CLayoutInfoN);
    mc2MmV3TilingData_->matmulTiling.set_CLayoutInfoG(mmv3TilingData_->tCubeTiling.CLayoutInfoG);
    mc2MmV3TilingData_->matmulTiling.set_BatchNum(mmv3TilingData_->tCubeTiling.BatchNum);
    mc2MmV3TilingData_->matmulTiling.set_mxTypePara(mmv3TilingData_->tCubeTiling.mxTypePara);
}

void NewMc2MatmulTilingCfg::SetTailCntAndType() const
{
    mc2MmV3TilingData_->set_kTailCnt(mmv3TilingData_->kTailCnt);
    if (baseMLimit_ < 0) {
        OP_LOGE_FOR_INVALID_VALUE("Mc2MatmulTilingCfg", "baseMLimit_", std::to_string(baseMLimit_).c_str(),
                                  "should be >= 0");
        return;
    } else if (baseMLimit_ == 0) {
        mc2MmV3TilingData_->set_mTailCnt(mmv3TilingData_->mTailCnt);
        mc2MmV3TilingData_->set_nTailCnt(mmv3TilingData_->nTailCnt);
        mc2MmV3TilingData_->set_mBaseTailSplitCnt(mmv3TilingData_->mBaseTailSplitCnt);
        mc2MmV3TilingData_->set_nBaseTailSplitCnt(mmv3TilingData_->nBaseTailSplitCnt);
        mc2MmV3TilingData_->set_mTailMain(mmv3TilingData_->mTailMain);
        mc2MmV3TilingData_->set_nTailMain(mmv3TilingData_->nTailMain);
        return;
    }

    uint64_t baseM = static_cast<uint64_t>(mc2MmV3TilingData_->matmulTiling.get_baseM());
    uint64_t baseN = static_cast<uint64_t>(mc2MmV3TilingData_->matmulTiling.get_baseN());
    uint64_t nValue = static_cast<uint64_t>(mc2MmV3TilingData_->matmulTiling.get_N());
    uint64_t coreNum = static_cast<uint64_t>(mc2MmV3TilingData_->matmulTiling.get_usedCoreNum());
    if ((baseM == 0) || (baseN == 0) || (coreNum == 0)) {
        OP_LOGE_FOR_INVALID_VALUE(
            "Mc2MatmulTilingCfg", "baseM, baseN, coreNum",
            (std::to_string(baseM) + ", " + std::to_string(baseN) + ", " + std::to_string(coreNum)).c_str(),
            "all should be non-zero");
        return;
    }
    uint64_t mCnt = ((baseMLimit_ + baseM - 1) / baseM) * rankDim_ * commCnt_;
    uint64_t nCnt = (nValue + baseN - 1) / baseN;
    uint64_t mnCnt = mCnt * nCnt;
    uint64_t tailCnt = mnCnt % coreNum;

    uint64_t mTailCnt = 1;
    uint64_t nTailCnt = 1;

    if (tailCnt != 0) {
        while ((mTailCnt + 1) * nTailCnt * tailCnt <= coreNum) {
            mTailCnt += 1;
            if (mTailCnt * (nTailCnt + 1) * tailCnt <= coreNum) {
                nTailCnt += 1;
            }
        }
    }

    mc2MmV3TilingData_->set_mTailCnt(mTailCnt);
    mc2MmV3TilingData_->set_nTailCnt(nTailCnt);
    mc2MmV3TilingData_->set_mBaseTailSplitCnt(mmv3TilingData_->mBaseTailSplitCnt);
    mc2MmV3TilingData_->set_nBaseTailSplitCnt(mmv3TilingData_->nBaseTailSplitCnt);
    mc2MmV3TilingData_->set_mTailMain(mmv3TilingData_->mTailMain);
    mc2MmV3TilingData_->set_nTailMain(mmv3TilingData_->nTailMain);
    mc2MmV3TilingData_->set_isHf32(mmv3TilingData_->isHf32);
}

void NewMc2MatmulTilingCfg::DealBaseBlock() const
{
    if (baseMLimit_ < 0) {
        OP_LOGE_FOR_INVALID_VALUE("Mc2MatmulTilingCfg", "baseMLimit_", std::to_string(baseMLimit_).c_str(),
                                  "should be >= 0");
        return;
    } else if ((baseMLimit_ > 0) && (mmv3TilingData_->tCubeTiling.baseM > baseMLimit_)) {
        mc2MmV3TilingData_->matmulTiling.set_singleCoreM(baseMLimit_);
        mc2MmV3TilingData_->matmulTiling.set_baseM(baseMLimit_);
    } else {
        mc2MmV3TilingData_->matmulTiling.set_singleCoreM(mmv3TilingData_->tCubeTiling.singleCoreM);
        mc2MmV3TilingData_->matmulTiling.set_baseM(mmv3TilingData_->tCubeTiling.baseM);
    }

    mc2MmV3TilingData_->matmulTiling.set_singleCoreN(mmv3TilingData_->tCubeTiling.singleCoreN);
    mc2MmV3TilingData_->matmulTiling.set_singleCoreK(mmv3TilingData_->tCubeTiling.singleCoreK);
    mc2MmV3TilingData_->matmulTiling.set_baseN(mmv3TilingData_->tCubeTiling.baseN);
    mc2MmV3TilingData_->matmulTiling.set_baseK(mmv3TilingData_->tCubeTiling.baseK);
}

void NewMc2MatmulTilingCfg::SetRankDim(uint64_t rankDim)
{
    rankDim_ = rankDim;
}

void NewMc2MatmulTilingCfg::SetCommCnt(uint64_t commCnt)
{
    commCnt_ = commCnt;
}

} // namespace Mc2MatmulHelper
