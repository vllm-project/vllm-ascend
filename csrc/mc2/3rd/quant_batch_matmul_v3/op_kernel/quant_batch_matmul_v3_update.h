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
 * \file quant_batch_matmul_v3_update.h
 * \brief
 */
#ifndef MC2_QUANT_BATCH_MATMUL_V3_UPDATE_H
#define MC2_QUANT_BATCH_MATMUL_V3_UPDATE_H

#include "quant_batch_matmul_v3_base.h"

namespace AscendC {

struct QBmmUpdateInfo {
    uint64_t mBaseTail; // 只量化mm用
    uint64_t nBaseTail; // 只量化mm用
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;
};

class Mc2QuantBatchMatmulV3Update {
public:
    __aicore__ inline Mc2QuantBatchMatmulV3Update()
    {
    }
    template <int x1Format, int x2Format, bool aTrans, bool bTrans>
    __aicore__ inline void Init(const TCubeTiling *mmTiling, const QBmmBaseBlockArgs &params);
    template <int x1Format, int x2Format, bool aTrans, bool bTrans>
    __aicore__ inline void UpdateBlockParamsAndCalcGmOffset(QBmmBaseBlockArgs &params, QBmmBlockOffset &offset,
                                                            uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void UpdateBlockParams(QBmmBaseBlockArgs &params, uint64_t mTileIndex, uint64_t nTileIndex);
    template <int x1Format, int x2Format, bool aTrans, bool bTrans>
    __aicore__ inline void CalcGMOffset(QBmmBaseBlockArgs &params, QBmmBlockOffset &offset);

private:
    QBmmUpdateInfo info_;
    const TCubeTiling *mmTiling_;
};

template <int x1Format, int x2Format, bool aTrans, bool bTrans>
__aicore__ inline void Mc2QuantBatchMatmulV3Update::Init(const TCubeTiling *mmTiling, const QBmmBaseBlockArgs &params)
{
    mmTiling_ = mmTiling;
    info_.nBaseTail =
        static_cast<uint64_t>(mmTiling_->N) - (params.nTotalCnt - 1) * mmTiling_->singleCoreN; // n方向上的base尾块
    info_.mBaseTail =
        static_cast<uint64_t>(mmTiling_->M) - (params.mTotalCnt - 1) * mmTiling_->singleCoreM; // m方向上的base尾块
    if constexpr (aTrans) {                                                                    // (k, m)
        info_.alignedKaSize = DequantBmm::Align(mmTiling_->Ka, BMM_BLOCK_NUM);
    } else { // (m, k)
        info_.alignedKaSize = DequantBmm::Align(mmTiling_->Ka, K0_INT8);
    }
    if constexpr (bTrans) { // (n, k)
        info_.alignedKbSize = DequantBmm::Align(mmTiling_->Kb, K0_INT8);
    } else { // (k, n)
        info_.alignedKbSize = DequantBmm::Align(mmTiling_->Kb, BMM_BLOCK_NUM);
    }
}

template <int x1Format, int x2Format, bool aTrans, bool bTrans>
__aicore__ inline void
Mc2QuantBatchMatmulV3Update::UpdateBlockParamsAndCalcGmOffset(QBmmBaseBlockArgs &params, QBmmBlockOffset &offset,
                                                              uint64_t mTileIndex, uint64_t nTileIndex)
{
    UpdateBlockParams(params, mTileIndex, nTileIndex);
    CalcGMOffset<x1Format, x2Format, aTrans, bTrans>(params, offset);
}

__aicore__ inline void Mc2QuantBatchMatmulV3Update::UpdateBlockParams(QBmmBaseBlockArgs &params, uint64_t mTileIndex,
                                                                      uint64_t nTileIndex)
{
    if ((mTileIndex == (params.mTileCntL2 - 1)) && (nTileIndex == (params.nTileCntL2 - 1)) &&
        (params.index == (params.totalTileCnt - 1))) {
        params.singleCoreM = info_.mBaseTail;
        params.singleCoreN = info_.nBaseTail;
    } else if ((mTileIndex == (params.mTileCntL2 - 1)) && (params.index >= (params.mCntUse - 1) * params.nCntUse)) {
        params.singleCoreM = info_.mBaseTail;
        params.singleCoreN = mmTiling_->baseN;
    } else if ((nTileIndex == (params.nTileCntL2 - 1)) && ((params.index + 1) % params.nCntUse == 0)) {
        params.singleCoreM = mmTiling_->baseM;
        params.singleCoreN = info_.nBaseTail;
    } else {
        params.singleCoreM = mmTiling_->baseM;
        params.singleCoreN = mmTiling_->baseN;
    }
}

template <int x1Format, int x2Format, bool aTrans, bool bTrans>
__aicore__ inline void Mc2QuantBatchMatmulV3Update::CalcGMOffset(QBmmBaseBlockArgs &params, QBmmBlockOffset &offset)
{
    uint64_t mCntIndex = params.index / params.nCntUse;
    uint64_t nCntIndex = params.index - mCntIndex * params.nCntUse;
    // tiling已保证baseM/K/N低轴16/32对齐
    // 前面的m都能保证是singleCoreM(baseM)的倍数，m尾块只会出现在m轴最后
    uint64_t mOffset = mCntIndex * mmTiling_->singleCoreM + params.mTileAddrOffset;
    if constexpr (DequantBmm::GetFormat(x1Format) == CubeFormat::ND) {
        if constexpr (aTrans) { // (k, m)
            offset.offsetA = mOffset;
        } else { // (m, k)
            offset.offsetA = mOffset * mmTiling_->Ka;
        }
    } else if constexpr (DequantBmm::GetFormat(x1Format) == CubeFormat::NZ) {
        if constexpr (aTrans) { // (m1, k1, k0, m0)
            offset.offsetA = mOffset * info_.alignedKaSize;
        } else { // (k1, m1, m0, k0)
            offset.offsetA = mOffset * K0_INT8;
        }
    }
    // 前面的n都能保证是singleCoreN(baseN)的倍数，n尾块只会出现在n轴最后
    uint64_t nOffset = nCntIndex * mmTiling_->singleCoreN + params.nTileAddrOffset;
    if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::ND) {
        if constexpr (bTrans) { // (n, k)
            offset.offsetB = nOffset * mmTiling_->Kb;
        } else { // (k, n)
            offset.offsetB = nOffset;
        }
    } else if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::NZ) {
        if constexpr (bTrans) { // (k1, n1, n0, k0)
            offset.offsetB = nOffset * K0_INT8;
        } else { // (n1, k1, k0, n0)
            offset.offsetB = nOffset * info_.alignedKbSize;
        }
    }
    // 输出只支持ND
    offset.offsetC = mOffset * mmTiling_->N + nOffset;
    // scale的计算当perchannel统一处理，实际参与mm/ub计算时会区分perchannel/pertensor
    offset.offsetScale = nOffset;
    // 当前基本块算法无batch轴，所以只考虑单维度，可选输入当存在时计算
    offset.offsetBias = nOffset;
    // 即使是纯cube场景，也可以计算pertoken偏移，这样mm输入输出的update接口可以共用
    offset.offsetPertoken = mOffset;
}
} // namespace AscendC
#endif // QUANT_BATCH_MATMUL_V3_UPDATE_H
