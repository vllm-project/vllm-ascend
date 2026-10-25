// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void moe_counts_cumsum_kernel(GM_ADDR input, GM_ADDR output, uint32_t count) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    AscendC::GlobalTensor<int64_t> x, y;
    x.SetGlobalBuffer((__gm__ int64_t*)input,count);
    y.SetGlobalBuffer((__gm__ int64_t*)output,count);
    // Open-source MoE models typically have no more than 1024 experts.
    // This specialized scan assumes counts fit in UB; capacity overflow
    // is outside its scope.
    // Align the UB extent to 32 bytes for DataCopyPad.
    const uint32_t aligned_count = (count + 3) / 4 * 4;
    AscendC::LocalTensor<int64_t> values(AscendC::TPosition::VECCALC, 0, aligned_count);
    AscendC::DataCopyExtParams copy{1,count*8,0,0,0};
    AscendC::DataCopyPadExtParams<int64_t> padding{false,0,0,0};
    AscendC::DataCopyPad(values,x,copy,padding);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    int64_t prefix=0;
    uint32_t i=0;
    for(;i+8<=count;i+=8){
        int64_t v0=values.GetValue(i+0);
        int64_t v1=values.GetValue(i+1);
        int64_t v2=values.GetValue(i+2);
        int64_t v3=values.GetValue(i+3);
        int64_t v4=values.GetValue(i+4);
        int64_t v5=values.GetValue(i+5);
        int64_t v6=values.GetValue(i+6);
        int64_t v7=values.GetValue(i+7);
        v0+=prefix;
        v1+=v0;
        v2+=v1;
        v3+=v2;
        v4+=v3;
        v5+=v4;
        v6+=v5;
        v7+=v6;
        values.SetValue(i+0,v0);
        values.SetValue(i+1,v1);
        values.SetValue(i+2,v2);
        values.SetValue(i+3,v3);
        values.SetValue(i+4,v4);
        values.SetValue(i+5,v5);
        values.SetValue(i+6,v6);
        values.SetValue(i+7,v7);
        prefix=v7;
    }
    for(;i<count;++i){prefix+=values.GetValue(i);values.SetValue(i,prefix);}
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopyPad(y,values,copy);
    AscendC::PipeBarrier<PIPE_ALL>();
}

namespace vllm_ascend {
void moe_counts_cumsum_impl(void* stream, void* input, void* output, uint32_t count)
{
    moe_counts_cumsum_kernel<<<1, nullptr, stream>>>((uint8_t*)input, (uint8_t*)output, count);
}
}  // namespace vllm_ascend
