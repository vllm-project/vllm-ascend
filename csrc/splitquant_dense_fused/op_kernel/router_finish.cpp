// SPDX-License-Identifier: BSD-3-Clause
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void moe_router_finish_bf16(GM_ADDR wp,GM_ADDR ip,GM_ADDR op,GM_ADDR oid,uint32_t m,uint32_t ids64,uint32_t normalize){
 TPipe pipe;TBuf<TPosition::VECCALC> weights,values,denom,scalar,scratch,typed,rawids,longids;
 pipe.InitBuffer(weights,128);pipe.InitBuffer(values,128);pipe.InitBuffer(denom,128);pipe.InitBuffer(scalar,128);pipe.InitBuffer(scratch,128);pipe.InitBuffer(typed,64);pipe.InitBuffer(rawids,128);pipe.InitBuffer(longids,128);
 auto wf=weights.Get<float>();auto vf=values.Get<float>();auto df=denom.Get<float>();auto sf=scalar.Get<float>();auto tmp=scratch.Get<float>();auto wt=typed.Get<bfloat16_t>();auto ri=rawids.Get<uint8_t>();auto oi=longids.Get<int64_t>();
 GlobalTensor<float> w;w.SetGlobalBuffer((__gm__ float*)wp);GlobalTensor<uint8_t> ids;ids.SetGlobalBuffer((__gm__ uint8_t*)ip);GlobalTensor<bfloat16_t> out;out.SetGlobalBuffer((__gm__ bfloat16_t*)op);GlobalTensor<int64_t> outids;outids.SetGlobalBuffer((__gm__ int64_t*)oid);
 for(uint32_t task=GetBlockIdx();task<(m+1)/2;task+=GetBlockNum()){
  uint32_t first=task*2,rows=(m-first<2?m-first:2),count=rows*8;
  DataCopyPad(wf,w[first*8],{1,count*4,0,0,0},{false,0,0,0});DataCopyPad(ri,ids[uint64_t(first)*8*(ids64?8:4)],{1,count*(ids64?8U:4U),0,0,0},{false,0,0,0});PipeBarrier<PIPE_ALL>();
  if(normalize){
   ReduceSum(sf,wf,tmp,8);PipeBarrier<PIPE_V>();
   if(rows==2){ReduceSum(sf[8],wf[8],tmp,8);PipeBarrier<PIPE_V>();}
   PipeBarrier<PIPE_ALL>();Duplicate(df,sf.GetValue(0),8);if(rows==2)Duplicate(df[8],sf.GetValue(8),8);PipeBarrier<PIPE_V>();Div(vf,wf,df,count);PipeBarrier<PIPE_V>();Cast(wt,vf,RoundMode::CAST_RINT,count);
  }else{Cast(wt,wf,RoundMode::CAST_RINT,count);}
  for(uint32_t j=0;j<count;++j)oi.SetValue(j,ids64?rawids.Get<int64_t>().GetValue(j):int64_t(rawids.Get<int32_t>().GetValue(j)));
  PipeBarrier<PIPE_ALL>();DataCopyPad(out[first*8],wt,{1,count*2,0,0,0});DataCopyPad(outids[first*8],oi,{1,count*8,0,0,0});PipeBarrier<PIPE_ALL>();
 }
}
