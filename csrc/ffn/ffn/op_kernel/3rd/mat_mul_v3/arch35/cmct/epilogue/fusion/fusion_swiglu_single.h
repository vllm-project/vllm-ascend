/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fusion_swiglu_single.h
 * \brief swiglu 单 matmul epilogue
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../utils/common_utils.h"
#include "../../utils/device_utils.h"
#include "fusion_regbase_act.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <typename DataTypeOut_, typename DataTypeIn_>
class FusionSwigluSingle {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_; // mmad 输出类型，fp32
    // 输出列宽 = 输入列宽的一半，epilogue 据此半宽 Cast / copy-out
    static constexpr bool HALF_WIDTH_OUT = true;
    static constexpr bool kRegBaseAct = true;
    __aicore__ inline FusionSwigluSingle(){};

    struct Arguments {
        GM_ADDR inputGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR inputGmAddr{nullptr};
    };

    int64_t stageSize_{0};
    AscendC::LocalTensor<DataTypeIn> inputLocal_;

    template <class LocalTensor>
    __aicore__ inline void Init(Params const& params, LocalTensor ubTensor, int64_t ubCalcM, int64_t ubCalcN,
                                int64_t& ubOffset, int64_t& stageSize)
    {
        int64_t lastUBSize = AscendC::TOTAL_UB_SIZE - ubOffset * sizeof(DataTypeIn);
        ASCENDC_ASSERT((lastUBSize > ubCalcN * sizeof(DataTypeIn)), {
            KERNEL_LOG(KERNEL_ERROR, , "ub size limit %ld, %ld!", lastUBSize, ubCalcN * sizeof(DataTypeIn));
        });
        static constexpr int64_t stageNum = 2;
        stageSize_ = AscendC::Std::min(
            static_cast<int64_t>(lastUBSize / stageNum / sizeof(DataTypeIn) / ubCalcN * ubCalcN), ubCalcM * ubCalcN);
        inputLocal_ = ubTensor[ubOffset];
        ubOffset += stageSize_;
        stageSize = stageSize_;
    }

    __aicore__ inline void operator()(const AscendC::LocalTensor<DataTypeIn>& srcLocal,
                                      AscendC::LocalTensor<DataTypeIn>& outputLocal, int64_t offset, int64_t curAivM,
                                      int64_t curAivN, int64_t strideN, int64_t stageSize)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        int64_t rows = stageSize / curAivN;
        if constexpr (kRegBaseAct && !AscendC::IsSameType<DataTypeOut, DataTypeIn>::value) {
            // RegBase：寄存器内 silu(g)*u + Cast RINT，半宽就地写回
            auto dstView = outputLocal.template ReinterpretCast<DataTypeOut>();
            RegBaseAct::RegSwigluSingleB16<DataTypeOut>(
                (__ubuf__ DataTypeOut *)dstView.GetPhyAddr(),
                (__ubuf__ DataTypeIn *)srcLocal.GetPhyAddr(), static_cast<uint32_t>(rows),
                static_cast<uint32_t>(curAivN));
        } else {
            int64_t outWidth = curAivN / 2;
            for (int64_t r = 0; r < rows; ++r) {
                AscendC::LocalTensor<DataTypeIn> srcRow = srcLocal[r * curAivN];
                AscendC::LocalTensor<DataTypeIn> outRow = outputLocal[r * outWidth];
                AscendC::Silu(outRow, srcRow, static_cast<uint32_t>(outWidth));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(outRow, outRow, srcRow[outWidth], static_cast<uint32_t>(outWidth));
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __host_aicore__ static Params InitParams(Arguments const& /* args */, GM_ADDR /* workspaceGm */)
    {
        return {};
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
