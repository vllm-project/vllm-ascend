// BSD-3-Clause. Four independent rows share metadata and vector instruction setup.
#pragma once
#include "kernel_operator.h"
namespace SplitQuantPrefill {
using namespace AscendC;
template<HardEvent E> __aicore__ inline void Sync(){auto id=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));SetFlag<E>(id);WaitFlag<E>(id);}
template<class T> class Epilogue {
 static constexpr uint32_t Rows=4,N=128,Groups=16,Outputs=Rows*N,Elements=Groups*Outputs;
 TBuf<TPosition::VECCALC> pi,wsb,wob,val,scales,acc,typed,metadata;
public:
 __aicore__ inline void Run(TPipe* pipe,GM_ADDR pp,GM_ADDR wsp,GM_ADDR wop,GM_ADDR sp,GM_ADDR sump,GM_ADDR bp,GM_ADDR yp,uint32_t m,uint32_t n,uint32_t g,uint32_t hb,uint32_t n0,uint32_t n1,uint32_t multi){
  GlobalTensor<int32_t> p;p.SetGlobalBuffer((__gm__ int32_t*)pp);
  GlobalTensor<float> ws,wo,s,sum;ws.SetGlobalBuffer((__gm__ float*)wsp);wo.SetGlobalBuffer((__gm__ float*)wop);s.SetGlobalBuffer((__gm__ float*)sp);sum.SetGlobalBuffer((__gm__ float*)sump);
  GlobalTensor<T> bias,y;bias.SetGlobalBuffer((__gm__ T*)bp);y.SetGlobalBuffer((__gm__ T*)yp);
  pipe->InitBuffer(pi,Elements*4);pipe->InitBuffer(wsb,Elements*4);pipe->InitBuffer(wob,Elements*4);pipe->InitBuffer(val,Elements*4);pipe->InitBuffer(scales,(Groups*Rows*8+Groups*Rows)*4);
  pipe->InitBuffer(acc,Outputs*4);pipe->InitBuffer(typed,Elements*2);pipe->InitBuffer(metadata,2*Rows*Groups*8*4);
  auto ip=pi.Get<int32_t>();auto wf=wsb.Get<float>();auto of=wob.Get<float>();auto vf=val.Get<float>();auto sf=scales.Get<float>();auto af=acc.Get<float>();auto tt=typed.template Get<T>();auto meta=metadata.Get<float>();
  uint32_t nt=n/N,rt=(m+Rows-1)/Rows;
  for(uint32_t task=GetBlockIdx();task<rt*nt;task+=GetBlockNum()){
   uint32_t row=task/nt*Rows,ni=task%nt*N,nrows=(m-row<Rows?m-row:Rows);
   uint32_t projection=multi?(ni<n0?0:(ni<n0+n1?1:2)):0;
   Duplicate(af,0.0f,Outputs);PipeBarrier<PIPE_V>();
   for(uint32_t first=0;first<g;first+=Groups){
    uint32_t count=(g-first<Groups?g-first:Groups),total=count*Outputs;
    Duplicate(meta,0.0f,2*Rows*Groups*8);
    if(nrows<Rows)Duplicate(ip.ReinterpretCast<float>(),0.0f,total);
    PipeBarrier<PIPE_V>();Sync<HardEvent::V_MTE2>();
    for(uint32_t r=0;r<nrows;++r){
     DataCopyPad(ip[r*N],p[(first*m+row+r)*n+ni],{static_cast<uint16_t>(count),N*4,(m*n-N)*4,(Rows-1)*N*4/32,0},{false,0,0,0});
     DataCopyPad(meta[r*8],s[((projection*m+row+r)*g+first)*8],{static_cast<uint16_t>(count),32,0,Rows-1,0},{false,0,0,0});
     DataCopyPad(meta[Rows*Groups*8+r*8],sum[((projection*m+row+r)*g+first)*8],{static_cast<uint16_t>(count),32,0,Rows-1,0},{false,0,0,0});
    }
    DataCopyPad(wf,ws[first*n+ni],{static_cast<uint16_t>(count),N*4,(n-N)*4,(Rows-1)*N*4/32,0},{false,0,0,0});
    DataCopyPad(of,wo[first*n+ni],{static_cast<uint16_t>(count),N*4,(n-N)*4,(Rows-1)*N*4/32,0},{false,0,0,0});
    Sync<HardEvent::MTE2_V>();Sync<HardEvent::MTE2_S>();
    // First row of each group was loaded; UB-to-UB copy preserves every bit.
    for(uint32_t r=1;r<Rows;++r){
     DataCopyParams copy{static_cast<uint16_t>(count),N*4/32,(Rows-1)*N*4/32,(Rows-1)*N*4/32};
     DataCopy(wf[r*N],wf,copy);DataCopy(of[r*N],of,copy);
    }
    PipeBarrier<PIPE_V>();
    Cast(vf,ip,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();
    // INT32 partials are consumed: reuse that exact allocation as Gather indices.
    ArithProgression(ip,static_cast<int32_t>(0),static_cast<int32_t>(32),Groups*Rows);PipeBarrier<PIPE_V>();
    auto index=ip.ReinterpretCast<uint32_t>();auto packedScalar=sf[Groups*Rows*8];
    Gather(packedScalar,meta[Rows*Groups*8],index,0,Groups*Rows);PipeBarrier<PIPE_V>();
    Brcb(sf,packedScalar,static_cast<uint8_t>(Groups*Rows/8),{1,8});PipeBarrier<PIPE_V>();
    Mul(of,of,sf,64,count*Rows,{1,1,0,16,16,1});
    Mul(of[64],of[64],sf,64,count*Rows,{1,1,0,16,16,1});PipeBarrier<PIPE_V>();
    Sub(vf,vf,of,total);PipeBarrier<PIPE_V>();
    Cast(tt,vf,RoundMode::CAST_RINT,total);PipeBarrier<PIPE_V>();Cast(vf,tt,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();
    Gather(packedScalar,meta,index,0,Groups*Rows);PipeBarrier<PIPE_V>();
    Brcb(sf,packedScalar,static_cast<uint8_t>(Groups*Rows/8),{1,8});PipeBarrier<PIPE_V>();
    Mul(vf,vf,sf,64,count*Rows,{1,1,0,16,16,1});
    Mul(vf[64],vf[64],sf,64,count*Rows,{1,1,0,16,16,1});PipeBarrier<PIPE_V>();
    Cast(tt,vf,RoundMode::CAST_RINT,total);PipeBarrier<PIPE_V>();Cast(vf,tt,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();
    Cast(tt,wf,RoundMode::CAST_RINT,total);PipeBarrier<PIPE_V>();Cast(wf,tt,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();
    Mul(vf,vf,wf,total);PipeBarrier<PIPE_V>();Cast(tt,vf,RoundMode::CAST_RINT,total);PipeBarrier<PIPE_V>();Cast(vf,tt,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();
    for(uint32_t j=0;j<count;++j){Add(af,af,vf[j*Outputs],Outputs);PipeBarrier<PIPE_V>();}
    Sync<HardEvent::V_MTE2>();
   }
   Cast(tt,af,RoundMode::CAST_RINT,Outputs);PipeBarrier<PIPE_V>();
   if(hb){
    Cast(af,tt,RoundMode::CAST_NONE,Outputs);PipeBarrier<PIPE_V>();
    DataCopyPad(ip.template ReinterpretCast<T>(),bias[ni],{1,N*2,0,0,0},{false,0,0,0});Sync<HardEvent::MTE2_V>();
    Cast(sf,ip.template ReinterpretCast<T>(),RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();
    for(uint32_t r=0;r<nrows;++r)Add(af[r*N],af[r*N],sf,N);
    PipeBarrier<PIPE_V>();Cast(tt,af,RoundMode::CAST_RINT,Outputs);PipeBarrier<PIPE_V>();
   }
   Sync<HardEvent::V_MTE3>();
   DataCopyPad(y[row*n+ni],tt,{static_cast<uint16_t>(nrows),N*2,0,(n-N)*2,0});
   Sync<HardEvent::MTE3_V>();Sync<HardEvent::V_MTE2>();
  }
 }
};
}
