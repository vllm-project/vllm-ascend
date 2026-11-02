// SPDX-License-Identifier: BSD-3-Clause
#define ASCENDC_CUBE_ONLY
#include "raw_cube_base.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void splitquant_sparse_raw_nz(GM_ADDR ap,GM_ADDR bp,GM_ADDR cp,uint32_t n,uint32_t g,uint32_t projections,uint32_t n0,uint32_t n1,uint32_t tn,GM_ADDR workspace,GM_ADDR tiling){
 if ASCEND_IS_AIV {return;}
 TPipe pipe;Raw::Dot64 dot(&pipe);dot.Init();
 GlobalTensor<uint8_t>a,b;a.SetGlobalBuffer((__gm__ uint8_t*)ap);b.SetGlobalBuffer((__gm__ uint8_t*)bp);
 GlobalTensor<int32_t>c;c.SetGlobalBuffer((__gm__ int32_t*)cp);
 uint32_t chunks=g/16,t0=n0/64,t1=n1/64,t2=(n-n0-n1)/64;
 if(projections==1){t0=n/64;t1=0;t2=0;}
 uint32_t nt=t0+t1+t2,lastA=0xffffffffU;
 for(uint32_t task=GetBlockIdx();task<chunks*nt;task+=GetBlockNum()){
  uint32_t ch=task/nt,ni=task%nt,proj,begin;
  if(projections==1){proj=0;begin=ni*64;}
  else if(ni<t0){proj=0;begin=ni*64;}
  else if(ni<t0+t1){proj=1;begin=n0+(ni-t0)*64;}
  else{proj=2;begin=n0+n1+(ni-t0-t1)*64;}
  uint32_t aIndex=proj*chunks+ch;
  if(aIndex!=lastA){dot.LoadA(a,uint64_t(aIndex)*16384);lastA=aIndex;}
  dot.Run(b,(uint64_t(begin/64)*(g/16)+ch)*65536,n,c,uint64_t(ch)*16*n+begin);
 }
}
