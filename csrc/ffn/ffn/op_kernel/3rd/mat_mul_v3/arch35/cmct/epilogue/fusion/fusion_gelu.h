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
 * \file fusion_gelu.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "math/erf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../utils/common_utils.h"
#include "../../utils/device_utils.h"
#include "fusion_regbase_act.h"

namespace Cmct {
namespace Gemm {
namespace Block {
enum class GeluApproxiMate : uint8_t { ERF = 0, TANH = 1 };

constexpr float SCALAR_ONE = 1.0;
constexpr float BETA = 0.044715;
constexpr float ALPHA = -1.5957691;
constexpr float REQ_SQRT2 = 0.70710678;
constexpr float SCALAR_HALF = 0.5;
constexpr uint16_t ZERO_FLAG = 0;

template <typename DataTypeOut_, typename DataTypeIn_, GeluApproxiMate approxiMate>
class FusionGelu {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    // ERF 且输出为窄类型（bf16/fp16）时走 RegBase：寄存器内激活+Cast，免 UB 中转/逐指令 barrier
    static constexpr bool kRegBaseAct = (approxiMate == GeluApproxiMate::ERF);
    __aicore__ inline FusionGelu(){};

    struct Arguments {
        GM_ADDR inputGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR inputGmAddr{nullptr};
    };

    int64_t stageSize_{0};
    int64_t ubCalcM_{0};
    int64_t ubCalcN_{0};
    int64_t strideN_{0};
    AscendC::LocalTensor<DataTypeIn> inputLocal_;

    template <class LocalTensor>
    __aicore__ inline void Init(Params const& params, LocalTensor ubTensor, int64_t ubCalcM, int64_t ubCalcN,
                                int64_t& ubOffset, int64_t& stageSize)
    {
        static constexpr int64_t stageNum = (approxiMate == GeluApproxiMate::ERF) ? 2 : 1;
        int64_t lastUBSize = AscendC::TOTAL_UB_SIZE - ubOffset * sizeof(DataTypeIn);
        ASCENDC_ASSERT((lastUBSize > ubCalcN * sizeof(DataTypeIn)), {
            KERNEL_LOG(KERNEL_ERROR, , "ub size limit %ld, %ld!", lastUBSize, ubCalcN * sizeof(DataTypeIn));
        });
        stageSize_ = AscendC::Std::min(
            static_cast<int64_t>(lastUBSize / stageNum / sizeof(DataTypeIn) / ubCalcN * ubCalcN), ubCalcM * ubCalcN);
        inputLocal_ = ubTensor[ubOffset];
        ubOffset += (approxiMate == GeluApproxiMate::ERF) ? stageSize_ : 0;
        stageSize = stageSize_;
    }

    __aicore__ inline void operator()(const AscendC::LocalTensor<DataTypeIn>& srcLocal,
                                      AscendC::LocalTensor<DataTypeIn>& outputLocal, int64_t offset, int64_t curAivM,
                                      int64_t curAivN, int64_t strideN, int64_t stageSize)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(ZERO_FLAG);
        if constexpr (kRegBaseAct && !AscendC::IsSameType<DataTypeOut, DataTypeIn>::value) {
            // RegBase：寄存器内 gelu(erf-Pade)+Cast RINT，输出就地写回 outputLocal（bf16 视图，
            // 半宽覆盖、读先于写）。fixpipe(fp32)→VF 读的可见性由既有 CrossCoreFlag 协议承担
            // （AIC 置位于 PIPE_FIX、AIV 等待于 PIPE_V，标准跨核同步模式），
            // 不插逐指令 PipeBarrier。outputLocal 由 epilogue 传 cLocal_[stageOffset] 原位视图。
            auto dstView = outputLocal.template ReinterpretCast<DataTypeOut>();
            RegBaseAct::RegGeluErfB16<DataTypeOut>(
                (__ubuf__ DataTypeOut *)dstView.GetPhyAddr(),
                (__ubuf__ DataTypeIn *)srcLocal.GetPhyAddr(), static_cast<uint32_t>(stageSize));
        } else if constexpr (approxiMate == GeluApproxiMate::ERF) {
            AscendC::Muls(inputLocal_, srcLocal, REQ_SQRT2, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Erf(outputLocal, inputLocal_, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(outputLocal, outputLocal, SCALAR_ONE, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(outputLocal, outputLocal, SCALAR_HALF, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(outputLocal, outputLocal, srcLocal, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Mul(outputLocal, srcLocal, srcLocal, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(outputLocal, srcLocal, outputLocal, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(outputLocal, outputLocal, BETA, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(outputLocal, srcLocal, outputLocal, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(outputLocal, outputLocal, ALPHA, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(outputLocal, outputLocal, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(outputLocal, outputLocal, SCALAR_ONE, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(outputLocal, srcLocal, outputLocal, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __host_aicore__ static Params InitParams(Arguments const /* &args */, GM_ADDR /* workspaceGm */) { return {}; }
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
