// SPDX-License-Identifier: Apache-2.0
#include "mla_serial_q_prepare_vector.h"
extern "C" __global__ __aicore__ void mla_serial_q_prepare(
    GM_ADDR up, GM_ADDR cos, GM_ADDR sin, GM_ADDR packed, GM_ADDR qRope, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(t,tiling);
    TPipe pipe;
    MlaPrepareStage v(pipe); v.Init();
    const uint32_t groups=(t.heads+t.headsPerTask-1)/t.headsPerTask;
    for (uint32_t task=GetBlockIdx(); task<t.tokens*groups; task+=GetBlockNum()) {
        const uint32_t token=task/groups, head=(task%groups)*t.headsPerTask;
        const uint32_t rows=t.heads-head<t.headsPerTask?t.heads-head:t.headsPerTask;
        const uint64_t first=uint64_t(token)*t.heads+head;
        v.Pack(up, first*(t.nope+64), packed, first*t.nope, t.nope, rows);
        v.Rope(up,first*(t.nope+64)+t.nope,t.useRope?cos:nullptr,t.useRope?sin:nullptr,
               token,qRope,first*64,rows,t.nope+64,64);
    }
}
