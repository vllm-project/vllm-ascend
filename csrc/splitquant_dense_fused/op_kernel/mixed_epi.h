#pragma once
namespace Mixed {
template<class T> class Epilogue {
 static constexpr uint32_t TileN=128, ChunkG=32, Elements=TileN*ChunkG;
 TPipe& pipe;
 TBuf<TPosition::VECCALC> pi,wsbuf,wobuf,vbuf,scale,acc,typed,scalars;
public:
 __aicore__ inline Epilogue(TPipe& p):pipe(p){}
 __aicore__ inline void Run(GM_ADDR pp,GM_ADDR wsp,GM_ADDR wop,GM_ADDR sp,GM_ADDR sump,
                            GM_ADDR bp,GM_ADDR yp,uint32_t m,uint32_t n,uint32_t g,uint32_t hb,uint32_t projections,uint32_t n0,uint32_t n1,uint32_t tn,uint32_t cores) {
  GlobalTensor<int32_t> p; p.SetGlobalBuffer((__gm__ int32_t*)pp);
  GlobalTensor<float> ws,wo,s,asum;
  ws.SetGlobalBuffer((__gm__ float*)wsp); wo.SetGlobalBuffer((__gm__ float*)wop);
  s.SetGlobalBuffer((__gm__ float*)sp); asum.SetGlobalBuffer((__gm__ float*)sump);
  GlobalTensor<T> wtyped;wtyped.SetGlobalBuffer((__gm__ T*)wsp);
  GlobalTensor<T> b,y; b.SetGlobalBuffer((__gm__ T*)bp); y.SetGlobalBuffer((__gm__ T*)yp);
  pipe.InitBuffer(pi,Elements*4); pipe.InitBuffer(wsbuf,Elements*4);
  pipe.InitBuffer(wobuf,Elements*4); pipe.InitBuffer(vbuf,Elements*4);
  pipe.InitBuffer(scale,Elements*4); pipe.InitBuffer(typed,Elements*2);
  pipe.InitBuffer(acc,TileN*4); pipe.InitBuffer(scalars,ChunkG*8*4*2);
  auto ip=pi.Get<int32_t>(); auto wf=wsbuf.Get<float>(); auto of=wobuf.Get<float>();
  auto vf=vbuf.Get<float>(); auto sf=scale.Get<float>(); auto af=acc.Get<float>();
  auto tt=typed.Get<T>(); auto meta=scalars.Get<float>();
  uint32_t ratio=GetTaskRation(),core=ratio?GetBlockIdx()/ratio:GetBlockIdx(),sub=GetSubBlockIdx();
  uint32_t t0=(n0+tn-1)/tn,t1=(n1+tn-1)/tn,t2=(n-n0-n1+tn-1)/tn;if(projections==1){t0=(n+tn-1)/tn;t1=t2=0;}uint32_t nt=t0+t1+t2;
  CrossCoreSetFlag<2,PIPE_MTE2>(9);
  for(uint32_t task=core;task<nt;task+=cores){
   uint32_t proj,begin,end;
   if(projections==1){proj=0;begin=task*tn;end=n;}else if(task<t0){proj=0;begin=task*tn;end=n0;}else if(task<t0+t1){proj=1;begin=n0+(task-t0)*tn;end=n0+n1;}else{proj=2;begin=n0+n1+(task-t0-t1)*tn;end=n;}
   uint32_t ni=begin+sub*(tn/2),len=ni<end?(end-ni<tn/2?end-ni:tn/2):0,row=0;
   CrossCoreWaitFlag(8);
   // Each task owns distinct partial columns; one-task lookahead cannot overwrite current input.
   CrossCoreSetFlag<2,PIPE_MTE2>(9);
   if(len>0){
   Duplicate(af,0.0f,TileN); PipeBarrier<PIPE_V>();
   for(uint32_t first=0;first<g;first+=ChunkG) {
    const uint32_t count=(ChunkG < g-first ? ChunkG : g-first),total=count*len;
    // N is a multiple of 16; each local row therefore needs no padding.
    DataCopyPad(ip,p[(first*m+row)*n+ni],
        {static_cast<uint16_t>(count),len*4,(m*n-len)*4,0,0},{false,0,0,0});
    if(hb&2){DataCopyPad(of.ReinterpretCast<T>(),wtyped[first*n+ni],{static_cast<uint16_t>(count),len*2,(n-len)*2,0,0},{false,0,0,0});}
    else{DataCopyPad(wf,ws[first*n+ni],{static_cast<uint16_t>(count),len*4,(n-len)*4,0,0},{false,0,0,0});}
    DataCopyPad(meta,s[(proj*g+first)*8],{1,count*32,0,0,0},{false,0,0,0});
    Sync<HardEvent::MTE2_V>(); Sync<HardEvent::MTE2_S>();
    if(hb&2){Cast(wf,of.ReinterpretCast<T>(),RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();}
    Cast(vf,ip,RoundMode::CAST_NONE,total); PipeBarrier<PIPE_V>();
    Cast(tt,vf,RoundMode::CAST_RINT,total); PipeBarrier<PIPE_V>();
    Cast(vf,tt,RoundMode::CAST_NONE,total); PipeBarrier<PIPE_V>();
    for(uint32_t i=0;i<count;++i) Duplicate(sf[i*len],meta.GetValue(i*8),len);
    PipeBarrier<PIPE_V>();
    Mul(vf,vf,sf,total); PipeBarrier<PIPE_V>();
    Cast(tt,vf,RoundMode::CAST_RINT,total); PipeBarrier<PIPE_V>();
    Cast(vf,tt,RoundMode::CAST_NONE,total); PipeBarrier<PIPE_V>();
    if(!(hb&2)){Cast(tt,wf,RoundMode::CAST_RINT,total);PipeBarrier<PIPE_V>();Cast(wf,tt,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();}
    Mul(vf,vf,wf,total); PipeBarrier<PIPE_V>();
    Cast(tt,vf,RoundMode::CAST_RINT,total); PipeBarrier<PIPE_V>();
    Cast(vf,tt,RoundMode::CAST_NONE,total); PipeBarrier<PIPE_V>();
    for(uint32_t i=0;i<count;++i) {
     Add(af,af,vf[i*len],len); PipeBarrier<PIPE_V>();
    }
    Sync<HardEvent::V_MTE2>();
   }
   Cast(tt,af,RoundMode::CAST_RINT,len); PipeBarrier<PIPE_V>();
   if(hb&1) {
    Cast(af,tt,RoundMode::CAST_NONE,len); PipeBarrier<PIPE_V>();
    // pi is no longer in use. Separate bias storage avoids a V->MTE2 hazard.
    DataCopyPad(ip.ReinterpretCast<T>(),b[ni],{1,len*2,0,0,0},{false,0,0,0});
    Sync<HardEvent::MTE2_V>();
    Cast(vf,ip.ReinterpretCast<T>(),RoundMode::CAST_NONE,len); PipeBarrier<PIPE_V>();
    Add(af,af,vf,len); PipeBarrier<PIPE_V>();
    Cast(tt,af,RoundMode::CAST_RINT,len); PipeBarrier<PIPE_V>();
   }
   Sync<HardEvent::V_MTE3>();
   DataCopyPad(y[row*n+ni],tt,{1,len*2,0,0,0});
   Sync<HardEvent::MTE3_V>();
   // Bias reuses pi after the group loop. Wait before the next task DMA
   // overwrites that storage while the previous vector cast is still reading.
   Sync<HardEvent::V_MTE2>();
   }
  }
 }
};
}
