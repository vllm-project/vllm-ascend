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
 * \file fusion_mul.h
 * \brief
 */

#pragma once
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "../../utils/common_utils.h"
#include "../../utils/device_utils.h"
#include "../../tile/x3_copy_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <typename DataTypeOut_, typename MmDataTypeIn_, typename X3Type_ = MmDataTypeIn_>
class FusionMul {
public:
    using DataTypeOut = DataTypeOut_;
    static constexpr bool kRegBaseAct = false;
    using MmDataTypeIn = MmDataTypeIn_;
    using X3Type = X3Type_;
    __aicore__ inline FusionMul(){};

    struct Arguments {
        GM_ADDR inputGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR inputGmAddr{nullptr};
        bool x3BatchBroadcast{false};
        int64_t x3M{0};
    };

    static constexpr uint16_t ZERO_FLAG = 0;
    static constexpr bool highPrecision = !AscendC::IsSameType<MmDataTypeIn, X3Type>::value;
    AscendC::LocalTensor<X3Type> inputLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::LocalTensor<MmDataTypeIn> castedInputLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::GlobalTensor<X3Type> inputGlobal_; // mul input x3 GM
    int64_t stageSize_ = 0;

    int64_t ubCalcM_{0};
    int64_t ubCalcN_{0};
    int64_t strideN_{0};
    bool needNdDma_{false};
    bool fixp1v2_{false};
    bool x3BatchBroadcast_{false};
    int64_t x3M_{0};

    template <class LocalTensor>
    __aicore__ inline void Init(Params const& params, LocalTensor ubTensor, int64_t ubCalcM, int64_t ubCalcN,
                                int64_t& ubOffset, int64_t& stageSize, uint64_t m = 0, bool needNdDma = false)
    {
        static constexpr int64_t stageNum = highPrecision ? 3 : 2;
        inputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ X3Type*>(params.inputGmAddr));
        int64_t lastUBSize = AscendC::TOTAL_UB_SIZE - ubOffset * sizeof(MmDataTypeIn);
        stageSize_ = AscendC::Std::min(
            static_cast<int64_t>(lastUBSize / stageNum / sizeof(MmDataTypeIn) / ubCalcN * ubCalcN), ubCalcM * ubCalcN);
        needNdDma_ = needNdDma;
        x3BatchBroadcast_ = params.x3BatchBroadcast;
        x3M_ = params.x3M;
        if (needNdDma_) {
            int64_t batchSize = m * ubCalcN;
            if (batchSize <= 0) {
                stageSize_ = 0;
                inputLocal_ = ubTensor[ubOffset].template ReinterpretCast<X3Type>();
                stageSize = stageSize_;
                return;
            }
            stageSize_ = stageSize_ / batchSize * batchSize;
        }
        if (m > 0) {
            fixp1v2_ = true;
        }
        inputLocal_ = ubTensor[ubOffset].template ReinterpretCast<X3Type>();
        ubOffset += stageSize_;
        if constexpr (highPrecision) {
            castedInputLocal_ = ubTensor[ubOffset].template ReinterpretCast<MmDataTypeIn>();
            ubOffset += stageSize_;
        }
        stageSize = stageSize_;
    }

    __aicore__ inline void operator()(const AscendC::LocalTensor<MmDataTypeIn>& srcLocal,
                                      AscendC::LocalTensor<DataTypeOut>& outputLocal, int64_t offset, int64_t curAivM,
                                      int64_t curAivN, int64_t strideN, int64_t stageSize)
    {
        bool copyOk = Detail::CopyFusionX3<X3Type, X3Type>(inputLocal_, inputGlobal_, offset, curAivM, curAivN, strideN,
                                                           stageSize, needNdDma_, fixp1v2_, x3BatchBroadcast_, x3M_,
                                                           true);
        if (!copyOk) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

        if constexpr (highPrecision) {
            AscendC::Cast(castedInputLocal_, inputLocal_, AscendC::RoundMode::CAST_NONE, stageSize);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(outputLocal, castedInputLocal_, srcLocal, stageSize);
        } else {
            AscendC::Mul(outputLocal, inputLocal_, srcLocal, stageSize);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR /* workspaceGm */)
    {
        return {args.inputGmAddr, false, 0};
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Cmct
