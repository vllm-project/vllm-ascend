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
 * \file weight_quant_batch_matmul_v2_fixpipe_stage2.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_FIXPIPE_STAGE2_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_FIXPIPE_STAGE2_H
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../tool.h"

namespace Mc2WeightQuantBatchMatmulV2 {
using AscendC::BLOCK_CUBE;
using AscendC::FixpipeParamsV220;
using AscendC::LoadData2DParams;

// a * weight_fp16 + bias_fp16 = y_fp16
template <bool hasBias>
class Mc2WeightQuantBatchMatmulV2FixpipeStage2 {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2FixpipeStage2(){};
    __aicore__ inline void Init(const LocalTensor<half> &a16L0A, const LocalTensor<half> &w16L0B,
                                const LocalTensor<float> &biasBT, const LocalTensor<float> &y32C01,
                                const GlobalTensor<half> &yGm)
    {
        // L0A DB
        a16L0A_ = a16L0A;
        w16L0B_ = w16L0B;
        yGm_ = yGm;
        y32C01_ = y32C01;

        if constexpr (hasBias) {
            biasBT_ = biasBT;
        }

        fixpipeParams_.ndNum = 1;
        fixpipeParams_.quantPre = QuantMode_t::F322F16;
    }

    __aicore__ inline void SetMadRelatedParams(uint32_t mSize, uint32_t nSize, uint32_t kSize, uint32_t mL1Size)
    {
        mBaseFracBlk_ = CeilDiv(mSize, FP16_BLOCK_SIZE); // m维度补齐16后的长度，/16
        baseMLen_ = mBaseFracBlk_ * FP16_BLOCK_SIZE;

        // Seta16InA01Params
        uint32_t kBaseFracBlk = CeilDiv(kSize, FP16_BLOCK_SIZE); // k维度补齐16后的长度，/16
        kaFracStride_ = kBaseFracBlk * FRAC_SIZE_HALF;

        a16InA01Params_.repeatTimes = static_cast<uint8_t>(kBaseFracBlk);
        a16InA01Params_.srcStride = static_cast<uint8_t>(CeilDiv(mL1Size, static_cast<uint32_t>(BLOCK_CUBE)));
        a16InA01Params_.ifTranspose = false;

        // Setw16InB01Params
        kbFracStride_ = kaFracStride_; // 初始的kb stride和ka的stride相同，且无需更新(stage1返回的默认一致)
        nProcessFracBlk_ = CeilDiv(nSize, FP16_BLOCK_SIZE);

        // 按照行循环去搬运
        w16InB01Params_.repeatTimes = static_cast<uint8_t>(kBaseFracBlk);
        w16InB01Params_.srcStride = 1;
        w16InB01Params_.dstGap = static_cast<uint16_t>(nProcessFracBlk_ - 1);
        w16InB01Params_.ifTranspose = true;

        // SetMmadParams
        mmadParams_.m = baseMLen_;
        mmadParams_.n = nSize;
        mmadParams_.k = kSize;

        // SetFixpipeParams
        fixpipeParams_.srcStride = baseMLen_;
        fixpipeParams_.nSize = nSize;
        fixpipeParams_.mSize = mSize;
    }

    __aicore__ inline void SetFixBiasParams(uint32_t nOriSize, uint32_t biasSize)
    {
        fixpipeParams_.dstStride = nOriSize;

        if constexpr (hasBias) {
            // bias l1ToBt指令需要搬运长度64B对齐
            bias16InBTParams_.blockCount = 1;
            bias16InBTParams_.blockLen = CeilDiv(biasSize * sizeof(half), 64UL);
        }
    }

    __aicore__ inline void ReSetParams(uint32_t kSize)
    {
        // reset w16InB01Params
        uint32_t kFracBlk = CeilDiv(kSize, FP16_BLOCK_SIZE);
        w16InB01Params_.repeatTimes = kFracBlk;

        // l0a上的ka stride需要实时更新，避免mmad将脏数据混入计算
        kaFracStride_ = kFracBlk * FRAC_SIZE_HALF;

        // reset mmadParams
        mmadParams_.k = kSize;
    }

    __aicore__ inline void LoadA16InA2(const LocalTensor<half> &a16A1)
    {
        for (uint64_t i = 0; i < mBaseFracBlk_; ++i) {
            LoadData(a16L0A_[i * kaFracStride_], a16A1[i * FRAC_SIZE_HALF], a16InA01Params_); // nz->zz
        }
    }

    __aicore__ inline void LoadBias16InBT(const LocalTensor<half> &bias16L1)
    {
        DataCopy(biasBT_, bias16L1, bias16InBTParams_);
    }

    __aicore__ inline void LoadW16InB2Trans(const LocalTensor<half> &w16B1)
    {
        for (uint64_t i = 0; i < nProcessFracBlk_; ++i) {
            LoadData(w16L0B_[pingPongOffset_ + i * FRAC_SIZE_HALF], w16B1[i * kbFracStride_],
                     w16InB01Params_); // nz->zn
        }
    }

    __aicore__ inline void ComputeY32(const uint64_t nOffset)
    {
        // C += A * B
        mmadParams_.cmatrixInitVal = false;
        Mmad(y32C01_[nOffset * baseMLen_], a16L0A_, w16L0B_[pingPongOffset_], mmadParams_);
    }

    __aicore__ inline void ComputeY32Bias(const uint64_t nOffset)
    {
        if constexpr (hasBias) {
            // C = A * B + bias
            mmadParams_.cmatrixInitVal = false;
            Mmad(y32C01_[nOffset * baseMLen_], a16L0A_, w16L0B_[pingPongOffset_], biasBT_[nOffset], mmadParams_);
        } else {
            mmadParams_.cmatrixInitVal = true;
            Mmad(y32C01_[nOffset * baseMLen_], a16L0A_, w16L0B_[pingPongOffset_], mmadParams_);
        }
    }

    __aicore__ inline void Process1(const LocalTensor<half> &w16B1)
    {
        pingPongOffset_ = (processCount_ & 1) ? pingpongSize_ : 0;
        LoadW16InB2Trans(w16B1);
    }

    __aicore__ inline void Process2(const uint64_t nOffset, const bool isFirst, const bool isLast,
                                    const uint64_t outOffset)
    {
        if (isFirst) {
            ComputeY32Bias(nOffset);
        } else {
            ComputeY32(nOffset);
        }
        processCount_++;
        if (isLast) {
            event_t eventIdMToFix = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_FIX));
            SetFlag<HardEvent::M_FIX>(eventIdMToFix);
            WaitFlag<HardEvent::M_FIX>(eventIdMToFix);
            FixpOutY16Gm(outOffset, nOffset);
        }
    }
    __aicore__ inline void FixpOutY16Gm(const uint64_t outOffset, const uint64_t nOffset)
    {
        Fixpipe(yGm_[outOffset], y32C01_[nOffset * baseMLen_], fixpipeParams_);
    }

private:
    LocalTensor<half> a16L0A_;
    LocalTensor<half> w16L0B_;
    LocalTensor<float> biasBT_;
    LocalTensor<float> y32C01_;
    GlobalTensor<half> yGm_;
    LoadData2DParams a16InA01Params_;
    LoadData2DParams w16InB01Params_;
    DataCopyParams bias16InBTParams_;
    MmadParams mmadParams_;
    FixpipeParamsV220 fixpipeParams_;
    uint32_t processCount_ = 0;
    // l1的切分固定，可用的db空间为32*256
    static constexpr uint64_t pingpongSize_ = 32 * 256;
    uint32_t kaFracStride_; // L1上两列分形的间隔(Nz)，k0LenAlign*16，单位：element
    uint32_t kbFracStride_;
    uint32_t mBaseFracBlk_;
    uint32_t baseMLen_;
    uint32_t nProcessFracBlk_; // n维度分形数量，n0LenAlign/16
    uint64_t pingPongOffset_ = 0;
};
} // namespace Mc2WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_FIXPIPE_STAGE2_H
