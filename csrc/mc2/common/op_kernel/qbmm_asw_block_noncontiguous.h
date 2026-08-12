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
 * \file qbmm_asw_block_noncontiguous.h
 * \brief
 */
#ifndef QBMM_ASW_BLOCK_NONCONTIGUOUS_H
#define QBMM_ASW_BLOCK_NONCONTIGUOUS_H

#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_asw_block.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_base.h"

namespace Mc2QuantBatchMatmulV3 {

class Mc2QuantBmmAswBlockNonContiguous : public Mc2QuantBmmAswBlock {
public:
    __aicore__ inline Mc2QuantBmmAswBlockNonContiguous()
    {
    }
    template <bool aTrans, bool bTrans, class x1Type, class scaleType, CubeFormat formatX2>
    __aicore__ inline void CalcGMOffset(const uint32_t strideCount, const uint32_t *batchWeight);
};

template <bool aTrans, bool bTrans, class x1Type, class scaleType, CubeFormat formatX2>
__aicore__ inline void Mc2QuantBmmAswBlockNonContiguous::CalcGMOffset(const uint32_t strideCount,
                                                                      const uint32_t *batchWeight)
{
    uint64_t mOffset = params_.mIndex * tilingData_->matmulTiling.baseM + params_.mSplitAddrOffset;
    uint64_t nOffset = params_.nIndex * tilingData_->matmulTiling.baseN + params_.nSplitAddrOffset;

    // Currently only supports aTrans being false. To support aTrans being true, 'offsetA' must be recalculated.
    uint32_t idx = static_cast<uint32_t>(offset_.batchAOffset);
    offset_.offsetA = mOffset * tilingData_->matmulTiling.Ka;
    offset_.offsetA += strideCount * batchWeight[idx] * tilingData_->matmulTiling.Ka;

    // Currently only supports CubeFormat being ND. To support other formats, 'offsetB' must be recalculated.
    if constexpr (bTrans) {
        offset_.offsetB = nOffset * tilingData_->matmulTiling.Kb;
    } else {
        offset_.offsetB = nOffset;
    }
    offset_.offsetB += offset_.batchBOffset * tilingData_->matmulTiling.N * tilingData_->matmulTiling.Kb;

    offset_.offsetC = mOffset * tilingData_->matmulTiling.N + nOffset;
    // idx will < 64
    idx = static_cast<uint32_t>(offset_.batchCOffset);
    offset_.offsetC += strideCount * batchWeight[idx] * tilingData_->matmulTiling.N;
    offset_.offsetPerTokenScale = mOffset;
    offset_.offsetScale = nOffset;
    // Perblock mode do not support bias now, if need to support, "offsetBias" must be recalculated.
}

} // namespace Mc2QuantBatchMatmulV3
#endif // MC2_QBMM_ASW_BLOCK_NONCONTIGUOUS_H
