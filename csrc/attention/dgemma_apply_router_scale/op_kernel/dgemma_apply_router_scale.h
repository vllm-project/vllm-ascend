/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef DGEMMA_APPLY_ROUTER_SCALE_KERNEL_H
#define DGEMMA_APPLY_ROUTER_SCALE_KERNEL_H
#include "kernel_operator.h"
#include "dgemma_apply_router_scale_tiling_data.h"
namespace DgemmaApplyRouterScale {
using namespace AscendC;

class KernelDgemmaApplyRouterScale {
public:
    __aicore__ inline KernelDgemmaApplyRouterScale() {}

    __aicore__ inline void Init(GM_ADDR weights, GM_ADDR ids, GM_ADDR scale,
                                GM_ADDR out,
                                const DgemmaApplyRouterScaleTilingData *tiling)
    {
        numElems_ = tiling->numElems;
        weightsGm_.SetGlobalBuffer((__gm__ float *)weights, numElems_);
        idsGm_.SetGlobalBuffer((__gm__ int32_t *)ids, numElems_);
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale);
        outGm_.SetGlobalBuffer((__gm__ float *)out, numElems_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        for (uint32_t i = blockIdx; i < numElems_; i += blockNum) {
            const float w = weightsGm_.GetValue(i);
            const int32_t expert = idsGm_.GetValue(i);
            const float s = scaleGm_.GetValue((uint32_t)expert);
            outGm_.SetValue(i, w * s);
        }
    }

private:
    GlobalTensor<float> weightsGm_, outGm_;
    GlobalTensor<int32_t> idsGm_;
    GlobalTensor<float> scaleGm_;
    uint32_t numElems_;
};
} // namespace DgemmaApplyRouterScale
#endif
