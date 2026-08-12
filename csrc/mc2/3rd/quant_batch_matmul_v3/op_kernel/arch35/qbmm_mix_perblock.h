/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qbmm_mix_perblock.h
 * \brief
 */

#ifndef MC2_QBMM_MIX_PERBLOCK_H
#define MC2_QBMM_MIX_PERBLOCK_H

#include "qbmm_api_utils.h"
#include "qbmm_perblock_api_utils.h"
#include "quant_batch_matmul_v3_tiling_data.h"
namespace Mc2QuantBatchMatmulV3 {

template <typename x1Type, typename x2Type, typename biasType, typename yType, CubeFormat formatX1, CubeFormat formatX2,
          CubeFormat formatY, bool aTrans, bool bTrans>
class MatMulPerBlockASW {
public:
    __aicore__ inline MatMulPerBlockASW()
    {
    }
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale,
                                GM_ADDR cGM, GM_ADDR workSpace, const void *tilingData, TPipe *que);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessWithoutBatch();

protected:
    const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *quantBmmTilingData_;
    Mc2QuantBmmAswBlock block_;
    MatMulPerBlock<x1Type, x2Type, float, biasType, float, yType, formatX1, formatX2, formatY, aTrans, bTrans, float,
                   Mc2QuantBatchMatmulV3::Mc2QuantBmmAswBlock>
        mm_;
};

template <typename x1Type, typename x2Type, typename biasType, typename yType, CubeFormat formatX1, CubeFormat formatX2,
          CubeFormat formatY, bool aTrans, bool bTrans>
__aicore__ inline void
MatMulPerBlockASW<x1Type, x2Type, biasType, yType, formatX1, formatX2, formatY, aTrans, bTrans>::Init(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR cGM, GM_ADDR workSpace,
    const void *tilingData, TPipe *que)
{
    quantBmmTilingData_ = static_cast<const DequantBmm::Mc2QuantBatchMatmulV3TilingDataParams *>(tilingData);
    uint32_t blockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        blockIdx = blockIdx / AscendC::GetTaskRation();
    }
    block_.Init(quantBmmTilingData_, blockIdx);
    mm_.Init(quantBmmTilingData_->matmulTiling, aGM, bGM, bias, scale, perTokenScale, cGM, block_, workSpace, que);
}

template <typename x1Type, typename x2Type, typename biasType, typename yType, CubeFormat formatX1, CubeFormat formatX2,
          CubeFormat formatY, bool aTrans, bool bTrans>
__aicore__ inline void
MatMulPerBlockASW<x1Type, x2Type, biasType, yType, formatX1, formatX2, formatY, aTrans, bTrans>::Process()
{
    if (quantBmmTilingData_->params.batchC == 1UL) {
        block_.offset_.batchCOffset = 0UL;
        block_.offset_.batchAOffset = 0UL;
        block_.offset_.batchBOffset = 0UL;
        ProcessWithoutBatch();
    } else {
        ProcessWithBatch<MatMulPerBlockASW>(block_, *this);
    }
    mm_.AivEnd();
    mm_.AicEnd();
}

template <typename x1Type, typename x2Type, typename biasType, typename yType, CubeFormat formatX1, CubeFormat formatX2,
          CubeFormat formatY, bool aTrans, bool bTrans>
__aicore__ inline void
MatMulPerBlockASW<x1Type, x2Type, biasType, yType, formatX1, formatX2, formatY, aTrans, bTrans>::ProcessWithoutBatch()
{
    for (uint64_t roundIndex = 0; roundIndex < block_.params_.round; ++roundIndex) {
        block_.UpdateBasicIndex(roundIndex);
        // 1. Set single core param
        block_.template UpdateBlockParams<bTrans, formatX2>(roundIndex);
        if (block_.params_.singleCoreM <= 0 || block_.params_.singleCoreN <= 0) {
            continue;
        }
        if ASCEND_IS_AIC {
            block_.template CalcGMOffset<aTrans, bTrans, x1Type, float, formatX2>();
        }
        block_.UpdatePerBlockMmParam<aTrans, bTrans>();
        mm_.Iterate();
    }
}
} // namespace Mc2QuantBatchMatmulV3
#endif // QBMM_MIX_PERBLOCK_H
