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
 * \file weight_quant_batch_matmul_v2_reg_base.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCHMATMUL_V2_REG_BASE_H
#define WEIGHT_QUANT_BATCHMATMUL_V2_REG_BASE_H

#include "weight_quant_batch_matmul_v2_reg_base_common.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35 {

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz = false>
class Mc2WeightQuantBatchMatmulV2RegBaseKernel
    : public Mc2WeightQuantBatchMatmulV2RegBaseCommonKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                            hasAntiQuantOffset, antiQuantType, weightNz> {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2RegBaseKernel(){};
    __aicore__ inline void Process();
};

/*
 * 该函数作用为通过 IterMatmulOut 每次移动一个 baseM 或 baseN，并循环遍历 KL1，
 * 计算好 AL1 和 BL1 的搬运时刻和 index 后，调用 compute 进行计算
 */
template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          bool hasAntiQuantOffset, Mc2QuantType antiQuantType, bool weightNz>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2RegBaseKernel<xType, wType, biasType, yType, aTrans, bTrans,
                                                                hasAntiQuantOffset, antiQuantType, weightNz>::Process()
{
    uint16_t usedCoreNum = this->tiling_->cubeBlockDimM * this->tiling_->cubeBlockDimN;
    if ((this->curBlockIdx_) >= usedCoreNum) {
        return;
    }

    AscendC::TEventID eventIdsMte1ToMte2[MAX_AL1_BUF_NUM];
    AscendC::TEventID biasEventIdsMte1ToMte2[DOUBLE_BUFFER];
    this->InitSync(eventIdsMte1ToMte2, biasEventIdsMte1ToMte2);
    while (this->IterMatmulOut()) {
        for (int32_t kFactorIdx = 0; kFactorIdx < this->kSingleCoreIterNum_; kFactorIdx++) {
            this->GetAL1(kFactorIdx, eventIdsMte1ToMte2);
            this->GetBL1(kFactorIdx);
            this->GetBiasL1(kFactorIdx, biasEventIdsMte1ToMte2);
            this->IterateMatmul(kFactorIdx);
            this->PostProcess(kFactorIdx, eventIdsMte1ToMte2, biasEventIdsMte1ToMte2);
        }
        this->GetTensorC();
    }
    this->EndSync(eventIdsMte1ToMte2, biasEventIdsMte1ToMte2);
}
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35
#endif // WEIGHT_QUANT_BATCHMATMUL_V2_REG_BASE_H
