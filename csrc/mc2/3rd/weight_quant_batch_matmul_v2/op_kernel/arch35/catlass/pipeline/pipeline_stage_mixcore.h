/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_MIXCORE_H
#define ARCH35_CATLASS_PIPELINE_PIPELINE_STAGE_MIXCORE_H

#include "../utils/device_utils.h"
#include "../utils/math_utils.h"
#include "kernel_operator.h"
#include "pipeline_state.h"
#include "utils.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {

#if defined(__CCE_KT_TEST__)
template <uint8_t modeId, pipe_t pipe>
DEVICE void CrossCoreWaitFlag(uint16_t flagId)
{
    auto strPipe = detail::PipelineToStr(pipe);
    X_LOG("CrossCoreWaitFlag<%d,%s>(%d)", modeId, strPipe.c_str(), flagId);
}

template <uint8_t modeId, pipe_t pipe>
DEVICE void CrossCoreSetFlag(uint16_t flagId)
{
    auto strPipe = detail::PipelineToStr(pipe);
    X_LOG("CrossCoreSetFlag<%d,%s>(%d)", modeId, strPipe.c_str(), flagId);
}
#endif

template <pipe_t ProducerPipeline, pipe_t ConsumerPipeline, int32_t SubBlockDim, uint8_t SyncMode = 4,
          uint16_t FlagId = 1>
struct PipelineStageMixCore {
    static constexpr uint64_t FLAG_ID_MAX = 16;

    template <uint8_t Stages>
    DEVICE static void ProducerWaitAfterStages(PipelineState<Stages> const &state)
    {
        if (state.count() >= Stages) {
            CrossCoreWaitFlag<SyncMode, ProducerPipeline>(FlagId);
        }
    }

    DEVICE
    static void ProducerRelease()
    {
        CrossCoreSetFlag<SyncMode, ProducerPipeline>(FlagId);
    }

    DEVICE
    static void ConsumerWait()
    {
        if constexpr (SubBlockDim == 2) {
            CrossCoreWaitFlag<SyncMode, ConsumerPipeline>(FlagId + FLAG_ID_MAX);
        }
        CrossCoreWaitFlag<SyncMode, ConsumerPipeline>(FlagId);
    }

    DEVICE
    static void ConsumerRelease()
    {
        if constexpr (SubBlockDim == 2) {
            CrossCoreSetFlag<SyncMode, ConsumerPipeline>(FlagId + FLAG_ID_MAX);
        }
        CrossCoreSetFlag<SyncMode, ConsumerPipeline>(FlagId);
    }

    template <uint8_t Stages>
    DEVICE static void Clear(PipelineState<Stages> const &state)
    {
        if ASCEND_IS_AIV {
            for (uint8_t i = 0; i < Min(static_cast<uint64_t>(Stages), state.count()); ++i) {
                ProducerWait();
            }
        }
    }

private:
    DEVICE
    static void ProducerWait()
    {
        CrossCoreWaitFlag<SyncMode, ProducerPipeline>(FlagId);
    }
};
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
