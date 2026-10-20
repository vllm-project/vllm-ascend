// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void mla_serial_q_pack(
    GM_ADDR up,GM_ADDR packed,GM_ADDR workspace,GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(t,tiling);
    constexpr uint32_t QUEUE_ELEMENTS=8192;
    TPipe pipe;
    TQueBind<TPosition::VECIN,TPosition::VECOUT,1> queue;
    pipe.InitBuffer(queue,2,QUEUE_ELEMENTS*sizeof(bfloat16_t));
    GlobalTensor<bfloat16_t> source,target;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(up));
    target.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(packed));
    const uint32_t groups=(t.tokens+t.tileTokens-1)/t.tileTokens;
    for(uint32_t task=GetBlockIdx();task<t.heads*groups;task+=GetBlockNum()) {
        const uint32_t head=task%t.heads,token=(task/t.heads)*t.tileTokens;
        const uint32_t rows=t.tokens-token<t.tileTokens?t.tokens-token:t.tileTokens;
        auto x=queue.AllocTensor<bfloat16_t>();
        const uint64_t from=(uint64_t(token)*t.heads+head)*(t.nope+64);
        DataCopyPad(x,source[from],DataCopyExtParams{static_cast<uint16_t>(rows),t.nope*2,
                    (t.heads*(t.nope+64)-t.nope)*2,0,0},DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
        queue.EnQue(x); x=queue.DeQue<bfloat16_t>();
        DataCopyPad(target[(uint64_t(head)*t.tokens+token)*t.nope],x,
                    DataCopyExtParams{1,rows*t.nope*2,0,0,0});
        queue.FreeTensor(x);
    }
}
