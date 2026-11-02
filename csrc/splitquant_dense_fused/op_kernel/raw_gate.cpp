// SPDX-License-Identifier: BSD-3-Clause
#define ASCENDC_CUBE_ONLY
#include "raw_cube_base.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void moe_gate_raw_nz(GM_ADDR ap,GM_ADDR bp,GM_ADDR idp,GM_ADDR cp,uint32_t m,uint32_t h,uint32_t inter,uint32_t topk,uint32_t experts,uint32_t tn,uint32_t gp,GM_ADDR workspace,GM_ADDR tiling){
 if ASCEND_IS_AIV {return;}
 TPipe pipe;Raw::Dot64 dot(&pipe);dot.Init();
 GlobalTensor<uint8_t>a,b;a.SetGlobalBuffer((__gm__ uint8_t*)ap);b.SetGlobalBuffer((__gm__ uint8_t*)bp);
 GlobalTensor<int32_t>c;c.SetGlobalBuffer((__gm__ int32_t*)cp);
 GlobalTensor<int64_t>ids;ids.SetGlobalBuffer((__gm__ int64_t*)idp);
 uint32_t n=2*inter,nt=n/64;dot.LoadA(a,0);
 for(uint32_t task=GetBlockIdx();task<topk*nt;task+=GetBlockNum()){
  uint32_t route=task/nt,ni=(task%nt)*64,e=static_cast<uint32_t>(ids.GetValue(route));if(e>=experts)e=0;
  dot.Run(b,(uint64_t(e)*nt+ni/64)*65536,n,c,uint64_t(route)*16*n+ni);
 }
}
