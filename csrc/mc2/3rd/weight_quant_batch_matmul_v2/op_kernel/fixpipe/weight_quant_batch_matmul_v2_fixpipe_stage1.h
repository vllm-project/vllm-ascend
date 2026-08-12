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
 * \file weight_quant_batch_matmul_v2_fixpipe_stage1.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_FIXPIPE_STAGE1_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_FIXPIPE_STAGE1_H
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../tool.h"

namespace Mc2WeightQuantBatchMatmulV2 {

using AscendC::DataCopyCO12DstParams;
using AscendC::DataCopyParams;
using AscendC::LoadData2dTransposeParams;
using AscendC::MmadParams;

template <bool hasAntiqOffset>
class Mc2WeightQuantBatchMatmulV2FixpipeStage1 {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2FixpipeStage1()
    {
    }

    __aicore__ inline void Init(const LocalTensor<int8_t> &weightS8L0a, const LocalTensor<int8_t> &diagS8L0b,
                                const LocalTensor<int32_t> &antiqOffsetBT, const LocalTensor<uint64_t> &antiqScaleFP,
                                const LocalTensor<int32_t> &weightS32L0c)
    {
        // L0A DB
        weightS8L0a_ = weightS8L0a;
        diagS8L0b_ = diagS8L0b;
        if constexpr (hasAntiqOffset) {
            antiqOffsetBT_ = antiqOffsetBT;
        }
        antiqScaleFP_ = antiqScaleFP;
        // L0C DB
        weightS32L0c_ = weightS32L0c;

        antiqOffsetParams_.blockCount = 1;
        antiqScaleParams_.blockCount = 1;
    }

    __aicore__ inline void SetOriShape(const uint32_t nOriSize)
    {
        weightParams_.srcStride = CeilDiv(nOriSize, INT8_ONE_BLK_SIZE);
        SetAntiqOffsetToBTParams(nOriSize);
        SetAntiqScaleToFixpipeParams(nOriSize);
    }

    __aicore__ inline void SetParams(const uint32_t kSize, const uint32_t nSize)
    {
        SetWeightToL0aParams(kSize, nSize);
        SetMmadParams(kSize, nSize);
    }

    __aicore__ inline void SetWeightToL0aParams(const uint32_t kSize, const uint32_t nSize)
    {
        weightParams_.repeatTimes = CeilDiv(kSize, INT8_ONE_BLK_SIZE);
        weightParams_.dstGap = CeilDiv(nSize, INT8_ONE_BLK_SIZE) * INT8_FRAC_NUM - 1;
    }

    // weight8 A1->A2
    __aicore__ inline void WeightToL0A(const LocalTensor<int8_t> &weightS8L1, const uint64_t pingpongOffset)
    {
        LoadDataWithTranspose(weightS8L0a_[pingpongOffset], weightS8L1, weightParams_);
    }

    __aicore__ inline void SetAntiqOffsetToBTParams(const uint32_t nSize)
    {
        // 4是int32的字节数，antiqoffset在L1中是int32，右移2位
        constexpr uint32_t btBlkSizeS32 = BT_BLK_SIZE >> 2;
        antiqOffsetParams_.blockLen = CeilDiv(nSize, btBlkSizeS32);
    }

    // antiqoffset C1->BT
    __aicore__ inline void AntiqOffsetToBT(const LocalTensor<int32_t> &antiqOffsetL1)
    {
        DataCopy(antiqOffsetBT_, antiqOffsetL1, antiqOffsetParams_);
    }

    __aicore__ inline void SetAntiqScaleToFixpipeParams(const uint32_t nSize)
    {
        // 8是uint64的字节数，antiqscale在L1中是uint64，右移3位
        constexpr uint32_t fbBlkSizeU64 = FIXP_BLK_SIZE >> 3;
        antiqScaleParams_.blockLen = CeilDiv(nSize, fbBlkSizeU64);
    }

    // antiqscale C1->FB
    __aicore__ inline void AntiqScaleToFixpipe(const LocalTensor<uint64_t> &antiqScaleL1)
    {
        DataCopy(antiqScaleFP_, antiqScaleL1, antiqScaleParams_);
    }

    __aicore__ inline void SetMmadParams(const uint32_t kSize, const uint32_t nSize)
    {
        mmadParams_.m = kSize;
        mmadParams_.n = nSize;
        mmadParams_.k = nSize;
        if constexpr (hasAntiqOffset) {
            mmadParams_.cmatrixInitVal = false;
        } else {
            mmadParams_.cmatrixInitVal = true;
        }
    }

    // weight * diag + antiqoffset
    __aicore__ inline void Compute(const uint64_t pingpongOffset, const uint64_t biasOffset)
    {
        if constexpr (hasAntiqOffset) {
            Mmad(weightS32L0c_[pingpongOffset], weightS8L0a_[pingpongOffset], diagS8L0b_, antiqOffsetBT_[biasOffset],
                 mmadParams_);
        } else {
            Mmad(weightS32L0c_[pingpongOffset], weightS8L0a_[pingpongOffset], diagS8L0b_, mmadParams_);
        }
    }

    // weight16 CO1->B1
    __aicore__ inline void WeightFixpDequant(const LocalTensor<half> &weightFP16L1, uint64_t pingpongOffset,
                                             uint64_t antiqScaleOffset, const uint32_t kSize, const uint32_t nSize)
    {
        SetFixPipeConfig(antiqScaleFP_[antiqScaleOffset]);
        DataCopyCO12DstParams params;
        params.nSize = nSize;
        params.mSize = kSize;
        params.dstStride = 256; // 做常量化优化，stride固定成256, 后续stage基于该值执行
        params.srcStride = kSize;
        params.quantPre = QuantMode_t::VDEQF16;
        params.reluPre = 0;
        params.channelSplit = 0;
        params.nz2ndEn = 0;
        DataCopy(weightFP16L1, weightS32L0c_[pingpongOffset], params);
    }

    __aicore__ inline void Process(const LocalTensor<half> &weightFP16L1, const LocalTensor<int8_t> &weightS8L1,
                                   const uint64_t biasOffset, const uint32_t kSize, const uint32_t nSize)
    {
        uint64_t pingpongOffset = (processCount_ & 1) * pingpongSize_;
        WeightToL0A(weightS8L1, pingpongOffset);
        event_t eventIdMte1ToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
        SetFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
        WaitFlag<HardEvent::MTE1_M>(eventIdMte1ToM);
        Compute(pingpongOffset, biasOffset);
        event_t eventIdMToFix = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_FIX));
        SetFlag<HardEvent::M_FIX>(eventIdMToFix);
        WaitFlag<HardEvent::M_FIX>(eventIdMToFix);
        WeightFixpDequant(weightFP16L1, pingpongOffset, biasOffset, kSize, nSize);
        processCount_++;
    }

    __aicore__ inline uint64_t Process1(const LocalTensor<int8_t> &weightS8L1)
    {
        uint64_t pingpongOffset = (processCount_ & 1) * pingpongSize_;
        WeightToL0A(weightS8L1, pingpongOffset);
        processCount_++;
        return pingpongOffset;
    }

    __aicore__ inline void Process2(const LocalTensor<half> &weightFP16L1, const uint64_t biasOffset,
                                    const uint64_t pingpongOffset, const uint32_t kSize, const uint32_t nSize)
    {
        Compute(pingpongOffset, biasOffset);
        event_t eventIdMToFix = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_FIX));
        SetFlag<HardEvent::M_FIX>(eventIdMToFix);
        WaitFlag<HardEvent::M_FIX>(eventIdMToFix);
        WeightFixpDequant(weightFP16L1, pingpongOffset, biasOffset, kSize, nSize);
    }

private:
    LoadData2dTransposeParams weightParams_;
    DataCopyParams antiqOffsetParams_;
    DataCopyParams antiqScaleParams_;
    MmadParams mmadParams_;
    uint64_t processCount_ = 0;
    // l1的切分固定，可用的db空间为32*256
    static constexpr uint64_t pingpongSize_ = 32 * 256;
    LocalTensor<int8_t> weightS8L0a_;
    LocalTensor<int8_t> diagS8L0b_;
    LocalTensor<int32_t> antiqOffsetBT_;
    LocalTensor<uint64_t> antiqScaleFP_;
    LocalTensor<int32_t> weightS32L0c_;
};
} // namespace Mc2WeightQuantBatchMatmulV2
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_FIXPIPE_STAGE1_H
