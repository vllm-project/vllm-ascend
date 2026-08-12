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
 * \file qbmm_perblock_api_utils_noncontiguous.h
 * \brief
 */

#ifndef QBMM_PERBLOCK_API_UTILS_NONCONTIGUOUS_H
#define QBMM_PERBLOCK_API_UTILS_NONCONTIGUOUS_H

#include "qbmm_mix_perblock_noncontiguous.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_api_utils.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_asw_block.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_base.h"
#include "../../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_perblock_api_utils.h"

namespace Mc2QuantBatchMatmulV3 {

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
class MatMulPerBlockNonContiguous : public MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS> {
public:
    __aicore__ inline MatMulPerBlockNonContiguous(){};
    __aicore__ inline void Iterate(const uint64_t tmpScaleX1Offset, const uint32_t *batchWeight,
                                   const uint64_t strideCount, const bool offsetFlag);
    __aicore__ inline void ProcessAivSingleK(const uint64_t tmpScaleMOffset, const uint32_t *batchWeight,
                                             const uint64_t strideCount, const bool offsetFlag);
    __aicore__ inline void UpdateUBOffsetC(const uint32_t *batchWeight, const uint64_t strideCount,
                                           const bool offsetFlag);
};

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlockNonContiguous<MATMUL_PERBLOCK_FUNC_PARAMS>::Iterate(const uint64_t tmpScaleMOffset,
                                                                                         const uint32_t *batchWeight,
                                                                                         const uint64_t strideCount,
                                                                                         const bool offsetFlag)
{
    if ASCEND_IS_AIC {
        MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::ProcessAicSingleK();
    }
    if ASCEND_IS_AIV {
        if (this->isPertile_) {
            MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::ProcessAivSingleKPertile();
        } else {
            ProcessAivSingleK(tmpScaleMOffset, batchWeight, strideCount, offsetFlag);
        }
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlockNonContiguous<MATMUL_PERBLOCK_FUNC_PARAMS>::UpdateUBOffsetC(
    const uint32_t *batchWeight, const uint64_t strideCount, const bool offsetFlag)
{
    if (offsetFlag) {
        uint32_t idx = static_cast<uint32_t>(this->block_->offset_.batchCOffset);
        this->block_->ubParams_.offsetC -=
            this->block_->offset_.batchCOffset * this->matmulTiling_->N * this->matmulTiling_->M;
        this->block_->ubParams_.offsetC += strideCount * batchWeight[idx] * this->matmulTiling_->N;
    }
}

MATMUL_PERBLOCK_CLASS_TEM_PARAMS
__aicore__ inline void MatMulPerBlockNonContiguous<MATMUL_PERBLOCK_FUNC_PARAMS>::ProcessAivSingleK(
    const uint64_t tmpScaleMOffset, const uint32_t *batchWeight, const uint64_t strideCount, const bool offsetFlag)
{
    this->block_->UpdatePerBlockUBParam();
    UpdateUBOffsetC(batchWeight, strideCount, offsetFlag);
    // Recalculate scaleOffsetX1 with new scaleM.
    uint32_t idx = static_cast<uint32_t>(this->block_->offset_.batchAOffset);
    uint64_t scaleOffsetX1 = this->scaleK_ * tmpScaleMOffset * batchWeight[idx];
    uint64_t scaleOffsetX2 = this->block_->offset_.batchBOffset * this->scaleK_ * this->scaleN_;
    // Currently only supports aTrans being false. To support aTrans being true, 'scaleOffsetX1' must be recalculated.
    scaleOffsetX1 += this->block_->ubParams_.offsetScaleM * this->scaleK_;
    if constexpr (bTrans) {
        scaleOffsetX2 += this->block_->ubParams_.offsetScaleN * this->scaleK_;
    } else {
        scaleOffsetX2 += this->block_->ubParams_.offsetScaleN;
    }
    GM_ADDR scaleX1AddrOff = this->scaleX1Addr_ + sizeof(ptScaleType) * (scaleOffsetX1);
    GM_ADDR scaleX2AddrOff = this->scaleX2Addr_ + sizeof(scaleType) * (scaleOffsetX2);
    LocalTensor<l0cDtype> mmAddUb = this->vecQueAdd_.template AllocTensor<l0cDtype>();
    for (uint64_t kb = 0, kOffset = 0; kb < this->matmulTiling_->Ka; kb += this->matmulTiling_->baseK, kOffset++) {
        ptScaleType scaleX1 = 0;
        scaleType scaleX2 = 0;
        // Currently only supports aTrans being false. To support aTrans being true, 'scaleX1' must be recalculated.
        scaleX1 = *((__gm__ ptScaleType *)(scaleX1AddrOff + sizeof(ptScaleType) * kOffset));
        if constexpr (bTrans) {
            scaleX2 = *((__gm__ scaleType *)(scaleX2AddrOff + sizeof(scaleType) * kOffset));
        } else {
            scaleX2 = *((__gm__ scaleType *)(scaleX2AddrOff + sizeof(scaleType) * kOffset * this->scaleN_));
        }
        scaleType scaleMul = scaleX1 * scaleX2;
        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + this->crossPingPongID_);
        auto mmUbInput = this->crossPingPongID_ == 0 ? this->mmResPing_ : this->mmResPong_;
        uint64_t mmAddUbAddr = reinterpret_cast<uint64_t>(mmAddUb.GetPhyAddr());
        uint64_t mmUbInputAddr = reinterpret_cast<uint64_t>(mmUbInput.GetPhyAddr());
        AscendC::PipeBarrier<PIPE_V>();
        if (kb == 0) {
            MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::template AivPerTensor<true>(
                (__ubuf__ l0cDtype *)mmAddUbAddr, (__ubuf__ l0cDtype *)mmUbInputAddr, scaleMul,
                this->block_->ubParams_.validM, this->block_->ubParams_.validN, this->block_->ubParams_.singleN);
        } else {
            MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::template AivPerTensor<false>(
                (__ubuf__ l0cDtype *)mmAddUbAddr, (__ubuf__ l0cDtype *)mmUbInputAddr, scaleMul,
                this->block_->ubParams_.validM, this->block_->ubParams_.validN, this->block_->ubParams_.singleN);
        }
        AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + this->crossPingPongID_);
        this->crossPingPongID_ = (this->crossPingPongID_ + 1UL) & 1UL;
    }
    MatMulPerBlock<MATMUL_PERBLOCK_FUNC_PARAMS>::AivPostProcess(mmAddUb);
    this->vecQueAdd_.FreeTensor(mmAddUb);
}


} // namespace Mc2QuantBatchMatmulV3
#endif // QBMM_PERBLOCK_API_UTILS_NONCONTIGUOUS_H
