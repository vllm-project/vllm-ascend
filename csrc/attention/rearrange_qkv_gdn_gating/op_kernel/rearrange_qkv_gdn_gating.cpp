// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vllm-ascend project

#include "rearrange_qkv_gdn_gating.h"

using namespace rearrange_qkv_gdn_gating_impl;

extern "C" __global__ __aicore__ void rearrange_qkv_gdn_gating(
    GM_ADDR x, GM_ADDR a, GM_ADDR b, GM_ADDR aLog, GM_ADDR dtBias,
    GM_ADDR y, GM_ADDR g, GM_ADDR betaOut, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    if ASCEND_IS_AIC {
        if (GetBlockIdx() >= tilingData.dmaCoreNum) {
            return;
        }
        TPipe pipe;
        RearrangeQkvDmaStage dmaStage;
        dmaStage.Init(x, y, tilingData, &pipe);
        dmaStage.Process();
    }
    if ASCEND_IS_AIV {
        const uint64_t subBlockNum = static_cast<uint64_t>(GetSubBlockNum());
        if (subBlockNum == 0) {
            return;
        }
        const uint64_t aivCoreNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t activeAivCoreNum = tilingData.tokens < aivCoreNum ? tilingData.tokens : aivCoreNum;
        const uint64_t aivCoreId = (static_cast<uint64_t>(GetBlockIdx()) / subBlockNum) * subBlockNum +
                                   static_cast<uint64_t>(GetSubBlockIdx());
        if (aivCoreId >= activeAivCoreNum) {
            return;
        }
        TPipe pipe;
        GdnGatingStage<DTYPE_ALOG> gatingStage;
        gatingStage.Init(a, b, aLog, dtBias, g, betaOut, tilingData, &pipe);
        gatingStage.Process();
    }
}
