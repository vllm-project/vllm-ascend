/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_SINGLECORE_BASE_H
#define ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_SINGLECORE_BASE_H

#include "../utils/device_utils.h"
#include "../utils/math_utils.h"
#include "kernel_operator.h"
#include "utils.h"

using AscendC::HardEvent;
using AscendC::Hardware;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
template <Hardware Src, Hardware Dst, uint8_t Stages_, typename Derived>
struct PipelineStageSingleCoreBase {
    static constexpr uint8_t Stages = Stages_;
    static constexpr HardEvent ForwardHardEvent = detail::GetQueEvt<Src, Dst, true>();
    static constexpr HardEvent BackwardHardEvent = detail::GetQueEvt<Src, Dst, false>();
    TEventID backwardEventId[Stages];

    DEVICE
    PipelineStageSingleCoreBase()
    {
    }

    DEVICE
    void ProducerWait(PipelineState<Stages> const &state) const
    {
        ProducerWait(state.index());
    }

    DEVICE
    void ProducerRelease(PipelineState<Stages> const &state) const
    {
        static_cast<const Derived *>(this)->ProducerRelease(state.index());
    }

    DEVICE
    void ConsumerWait(PipelineState<Stages> const &state) const
    {
        static_cast<const Derived *>(this)->ConsumerWait(state.index());
    }

    DEVICE
    void ConsumerRelease(PipelineState<Stages> const &state) const
    {
        ConsumerRelease(state.index());
    }

    DEVICE
    void ProducerWait(uint64_t index) const
    {
        WaitFlag<BackwardHardEvent>(backwardEventId[index]);
    }

    DEVICE
    void ConsumerRelease(uint64_t index) const
    {
        SetFlag<BackwardHardEvent>(backwardEventId[index]);
    }

    DEVICE void Clear() const
    {
// 多set需要wait掉
#pragma unroll
        for (uint8_t i = 0; i < Stages; ++i) {
            ProducerWait(backwardEventId[i]);
        }
    }
};
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
