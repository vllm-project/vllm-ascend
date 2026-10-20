// SPDX-License-Identifier: Apache-2.0
#include "mla_serial_norm_cache_vector.h"
extern "C" __global__ __aicore__ void mla_serial_norm_cache(
    GM_ADDR down, GM_ADDR gammaKv, GM_ADDR cos, GM_ADDR sin, GM_ADDR slots, GM_ADDR gammaQ,
    GM_ADDR cache, GM_ADDR cacheRope, GM_ADDR q, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(t,tiling);
    TPipe pipe;
    MlaVectorStage v(pipe); v.Init();
    // Separate local loops retain each gamma in UB; negative slots only mask cache.
    for(uint32_t token=GetBlockIdx();token<t.tokens;token+=GetBlockNum())
        v.Norm(down,uint64_t(token)*(t.rank+576)+576,gammaQ,q,uint64_t(token)*t.rank,t.rank);
    GlobalTensor<int32_t> mapping; mapping.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(slots));
    for (uint32_t token=GetBlockIdx(); token<t.tokens; token+=GetBlockNum()) {
        const int32_t slot=mapping.GetValue(token);
        if (slot<0 || uint32_t(slot)>=t.cacheBlocks*128) continue;
        const uint64_t row=uint64_t(token)*(t.rank+576);
        const uint64_t target=uint64_t(slot/128)*t.cacheStride+uint64_t(slot%128)*512;
        const uint64_t ropeTarget=uint64_t(slot/128)*t.ropeStride+uint64_t(slot%128)*64;
        v.Norm(down,row,gammaKv,cache,target,512);
        v.Rope(down,row+512,t.useRope?cos:nullptr,t.useRope?sin:nullptr,token,cacheRope,ropeTarget);
    }
}
