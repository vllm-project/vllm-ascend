/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_SINGLECORE_H
#define ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_SINGLECORE_H

#include "../utils/device_utils.h"
#include "../utils/math_utils.h"
#include "kernel_operator.h"
#include "pipeline_stage_singlecore_base.h"
#include "utils.h"

#if defined(__CCE_KT_TEST__)
#include <set>
std::set<std::tuple<HardEvent, int>> s;
#endif

using AscendC::HardEvent;
using AscendC::Hardware;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::WaitFlag;

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
template <Hardware Src, Hardware Dst, uint8_t Stages_>
struct PipelineStageSingleCore
    : public PipelineStageSingleCoreBase<Src, Dst, Stages_, PipelineStageSingleCore<Src, Dst, Stages_>> {
    static constexpr uint8_t Stages = Stages_;
    using Base = PipelineStageSingleCoreBase<Src, Dst, Stages_, PipelineStageSingleCore<Src, Dst, Stages_>>;
    TEventID forwardEventId_;

    using Base::ConsumerWait;
    using Base::ProducerRelease;

    DEVICE
    PipelineStageSingleCore()
    {
        forwardEventId_ = GetTPipePtr()->AllocEventID<Base::ForwardHardEvent>();
        for (uint8_t i = 0; i < Stages; ++i) {
            Base::backwardEventId[i] = GetTPipePtr()->AllocEventID<Base::BackwardHardEvent>();
            Base::ConsumerRelease(i);
        }
    }

    DEVICE
    void ProducerRelease(uint64_t index) const
    {
        SetFlag<Base::ForwardHardEvent>(forwardEventId_);
    }

    DEVICE
    void ConsumerWait(uint64_t index) const
    {
        WaitFlag<Base::ForwardHardEvent>(forwardEventId_);
    }
};
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
