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
 * \file fusion_swiglu.h
 * \brief swiglu(up) epilogue：从 GM 读取 gate(fp32)，输出 = silu(gate) * up。
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../utils/common_utils.h"
#include "../../utils/device_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <typename DataTypeOut_, typename DataTypeIn_>
class FusionSwiglu {
public:
    using DataTypeOut = DataTypeOut_;
    static constexpr bool kRegBaseAct = false;
    using DataTypeIn = DataTypeIn_; // mmad 输出类型，fp32
    __aicore__ inline FusionSwiglu(){};

    struct Arguments {
        GM_ADDR inputGmAddr{nullptr}; // gate，fp32，row-major [M,H]
    };

    struct Params {
        GM_ADDR inputGmAddr{nullptr};
    };

    static constexpr uint16_t ZERO_FLAG = 0;
    AscendC::LocalTensor<DataTypeIn> inputLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::GlobalTensor<DataTypeIn> inputGlobal_;
    int64_t stageSize_{0};

    template <class LocalTensor>
    __aicore__ inline void Init(Params const& params, LocalTensor ubTensor, int64_t ubCalcM, int64_t ubCalcN,
                                int64_t& ubOffset, int64_t& stageSize)
    {
        static constexpr int64_t stageNum = 2;
        inputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeIn *>(params.inputGmAddr));
        int64_t lastUBSize = AscendC::TOTAL_UB_SIZE - ubOffset * sizeof(DataTypeIn);
        ASCENDC_ASSERT((lastUBSize > ubCalcN * sizeof(DataTypeIn)), {
            KERNEL_LOG(KERNEL_ERROR, , "ub size limit %ld, %ld!", lastUBSize, ubCalcN * sizeof(DataTypeIn));
        });
        stageSize_ = AscendC::Std::min(
            static_cast<int64_t>(lastUBSize / stageNum / sizeof(DataTypeIn_) / ubCalcN * ubCalcN), ubCalcM * ubCalcN);
        inputLocal_ = ubTensor[ubOffset];
        ubOffset += stageSize_;
        stageSize = stageSize_;
    }

    __aicore__ inline void operator()(const AscendC::LocalTensor<DataTypeIn>& srcLocal,
                                      AscendC::LocalTensor<DataTypeOut>& outputLocal, int64_t offset, int64_t curAivM,
                                      int64_t curAivN, int64_t strideN, int64_t stageSize)
    {
        int64_t curAivNAlign = Cmct::Gemm::Align(curAivN, AscendC::AuxGetC0Size<half>());
        constexpr int64_t kUbBlockSize = 32;
        int64_t dstGapBlocks = (curAivNAlign * sizeof(DataTypeIn) -
                                Cmct::Gemm::Align(curAivN * sizeof(DataTypeIn), kUbBlockSize)) /
                               kUbBlockSize;
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(stageSize / curAivNAlign),
                                              static_cast<uint32_t>(curAivN * sizeof(DataTypeIn)),
                                              static_cast<uint32_t>((strideN - curAivN) * sizeof(DataTypeIn)),
                                              static_cast<uint32_t>(dstGapBlocks), 0};
        AscendC::DataCopyPadExtParams<DataTypeIn> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(inputLocal_, inputGlobal_[offset], copyParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

        // outputLocal = silu(gate) * up = gate / (1 + e^-gate) * up
        AscendC::Muls(outputLocal, inputLocal_, -1.0f, stageSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(outputLocal, outputLocal, stageSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(outputLocal, outputLocal, 1.0f, stageSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Reciprocal(outputLocal, outputLocal, stageSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(outputLocal, inputLocal_, outputLocal, stageSize);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(outputLocal, outputLocal, srcLocal, stageSize);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR /* workspaceGm */)
    {
        return {args.inputGmAddr};
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
