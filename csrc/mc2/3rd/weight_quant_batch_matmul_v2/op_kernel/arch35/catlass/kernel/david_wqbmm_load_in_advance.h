/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_KERNEL_DAVID_WQBMM_LOAD_IN_ADVANCE_H
#define ARCH35_CATLASS_KERNEL_DAVID_WQBMM_LOAD_IN_ADVANCE_H

#include "../utils/device_utils.h"

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {
template <typename ProblemShape, typename BlockMainloop, typename TileScheduler>
class wqbmmv2 {
public:
    struct Params {
        ProblemShape problemShape;
        typename BlockMainloop::Params mainloop;
        typename BlockMainloop::TileShapeL1 tileShape;
    };

public:
    DEVICE void operator()(Params const &params)
    {
        BlockMainloop blockMainloop;
        blockMainloop.Init(params.problemShape, params.mainloop);

        auto states = typename BlockMainloop::PipelineStateTuple{};
        auto pipelines = typename BlockMainloop::PipelineTuple{};

        TileScheduler tileScheduler(params.problemShape, params.mainloop);

        decltype(auto) mIterLoad = tileScheduler.MIter();
        decltype(auto) nIterLoad = tileScheduler.NIter();
        decltype(auto) mBiasIterLoad = tileScheduler.MIter();
        decltype(auto) nBiasIterLoad = tileScheduler.NIter();
        decltype(auto) itersLoad = AscendC::Std::tie(mIterLoad, nIterLoad, mBiasIterLoad, nBiasIterLoad);
        if (tileScheduler.IsValid()) {
            blockMainloop.LoadInAdvance(params.problemShape, pipelines, states, params.mainloop, itersLoad);
        }

        decltype(auto) mIterCompute = tileScheduler.MIterRef();
        decltype(auto) nIterCompute = tileScheduler.NIterRef();
        decltype(auto) mBiasIterCompute = tileScheduler.MIter();
        decltype(auto) itersCompute = AscendC::Std::tie(mIterCompute, nIterCompute, mBiasIterCompute, nIterCompute);

        // 尾核不执行Process
        while (tileScheduler.IsValid()) {
            blockMainloop.Process(params.problemShape, pipelines, states, params.mainloop, itersLoad, itersCompute);
            tileScheduler.FetchNextWork();
        }

        blockMainloop.ClearPipeline(pipelines, states);
    }
};
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
