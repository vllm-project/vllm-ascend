// V2: batch DMA and dtype-rounded elementwise work across independent groups.
// Retain sequential FP32 group addition, then activation-dtype sum and bias.
#include "kernel_operator.h"
#include "prefill_epilogue.h"
using namespace AscendC;
namespace {
template<HardEvent E> __aicore__ inline void Sync() {
 auto id=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
 SetFlag<E>(id); WaitFlag<E>(id);
}
template<class T,bool ZeroOffset=false> class Epilogue {
 static constexpr uint32_t TileN=128, ChunkG=32, Elements=TileN*ChunkG;
 TPipe pipe;
 TBuf<TPosition::VECCALC> pi,wsbuf,wobuf,vbuf,scale,acc,typed,scalars;
public:
 __aicore__ inline void Run(GM_ADDR pp,GM_ADDR wsp,GM_ADDR wop,GM_ADDR sp,GM_ADDR sump,
                            GM_ADDR bp,GM_ADDR yp,uint32_t m,uint32_t n,uint32_t g,uint32_t hb,uint32_t n0,uint32_t n1) {
  GlobalTensor<int32_t> p; p.SetGlobalBuffer((__gm__ int32_t*)pp);
  GlobalTensor<float> ws,wo,s,asum;
  ws.SetGlobalBuffer((__gm__ float*)wsp); wo.SetGlobalBuffer((__gm__ float*)wop);
  s.SetGlobalBuffer((__gm__ float*)sp); asum.SetGlobalBuffer((__gm__ float*)sump);
  GlobalTensor<T> b,y; b.SetGlobalBuffer((__gm__ T*)bp); y.SetGlobalBuffer((__gm__ T*)yp);
  pipe.InitBuffer(pi,Elements*4); pipe.InitBuffer(wsbuf,Elements*4);
  pipe.InitBuffer(wobuf,Elements*4); pipe.InitBuffer(vbuf,Elements*4);
  pipe.InitBuffer(scale,Elements*4); pipe.InitBuffer(typed,Elements*2);
  pipe.InitBuffer(acc,TileN*4); pipe.InitBuffer(scalars,ChunkG*8*4*2);
  auto ip=pi.Get<int32_t>(); auto wf=wsbuf.Get<float>(); auto of=wobuf.Get<float>();
  auto vf=vbuf.Get<float>(); auto sf=scale.Get<float>(); auto af=acc.Get<float>();
  auto tt=typed.Get<T>(); auto meta=scalars.Get<float>();
  const uint32_t nt=(n+TileN-1)/TileN;
  for(uint32_t task=GetBlockIdx();task<m*nt;task+=GetBlockNum()) {
   const uint32_t row=task/nt,ni=(task%nt)*TileN,len=(TileN < n-ni ? TileN : n-ni);
   const uint32_t projection=(ni<n0?0:(ni<n0+n1?1:2));
   Duplicate(af,0.0f,TileN); PipeBarrier<PIPE_V>();
   for(uint32_t first=0;first<g;first+=ChunkG) {
    const uint32_t count=(ChunkG < g-first ? ChunkG : g-first),total=count*len;
    // N is a multiple of 16; each local row therefore needs no padding.
    DataCopyPad(ip,p[(first*m+row)*n+ni],
        {static_cast<uint16_t>(count),len*4,(m*n-len)*4,0,0},{false,0,0,0});
    DataCopyPad(wf,ws[first*n+ni],
        {static_cast<uint16_t>(count),len*4,(n-len)*4,0,0},{false,0,0,0});
    if constexpr(!ZeroOffset){
    DataCopyPad(of,wo[first*n+ni],
        {static_cast<uint16_t>(count),len*4,(n-len)*4,0,0},{false,0,0,0});
    DataCopy(meta[ChunkG*8],asum[((projection*m+row)*g+first)*8],count*8);
    }
    DataCopy(meta,s[((projection*m+row)*g+first)*8],count*8);
    Sync<HardEvent::MTE2_V>(); Sync<HardEvent::MTE2_S>();
    Cast(vf,ip,RoundMode::CAST_NONE,total); PipeBarrier<PIPE_V>();
    if constexpr(!ZeroOffset){
    for(uint32_t i=0;i<count;++i) Duplicate(sf[i*len],meta.GetValue(ChunkG*8+i*8),len);
    PipeBarrier<PIPE_V>();
    Mul(of,of,sf,total); PipeBarrier<PIPE_V>();
    Sub(vf,vf,of,total); PipeBarrier<PIPE_V>();
    }
    Cast(tt,vf,RoundMode::CAST_RINT,total); PipeBarrier<PIPE_V>();
    Cast(vf,tt,RoundMode::CAST_NONE,total); PipeBarrier<PIPE_V>();
    for(uint32_t i=0;i<count;++i) Duplicate(sf[i*len],meta.GetValue(i*8),len);
    PipeBarrier<PIPE_V>();
    Mul(vf,vf,sf,total); PipeBarrier<PIPE_V>();
    Cast(tt,vf,RoundMode::CAST_RINT,total); PipeBarrier<PIPE_V>();
    Cast(vf,tt,RoundMode::CAST_NONE,total); PipeBarrier<PIPE_V>();
    Cast(tt,wf,RoundMode::CAST_RINT,total); PipeBarrier<PIPE_V>();
    Cast(wf,tt,RoundMode::CAST_NONE,total); PipeBarrier<PIPE_V>();
    Mul(vf,vf,wf,total); PipeBarrier<PIPE_V>();
    Cast(tt,vf,RoundMode::CAST_RINT,total); PipeBarrier<PIPE_V>();
    Cast(vf,tt,RoundMode::CAST_NONE,total); PipeBarrier<PIPE_V>();
    for(uint32_t i=0;i<count;++i) {
     Add(af,af,vf[i*len],len); PipeBarrier<PIPE_V>();
    }
    Sync<HardEvent::V_MTE2>();
   }
   Cast(tt,af,RoundMode::CAST_RINT,len); PipeBarrier<PIPE_V>();
   if(hb) {
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
};
}

extern "C" __global__ __aicore__ void splitquant_multi_epi(GM_ADDR p,GM_ADDR ws,GM_ADDR wo,GM_ADDR s,GM_ADDR sum,GM_ADDR b,GM_ADDR y,uint32_t m,uint32_t n,uint32_t g,uint32_t hb,uint32_t n0,uint32_t n1){if(m>=4 && n>=1024){TPipe pipe;SplitQuantPrefill::Epilogue<bfloat16_t> op;op.Run(&pipe,p,ws,wo,s,sum,b,y,m,n,g,hb,n0,n1,1);}else{Epilogue<bfloat16_t> op;op.Run(p,ws,wo,s,sum,b,y,m,n,g,hb,n0,n1);}}

extern "C" __global__ __aicore__ void splitquant_multi_epi_zero(GM_ADDR p,GM_ADDR ws,GM_ADDR wo,GM_ADDR s,GM_ADDR sum,GM_ADDR b,GM_ADDR y,uint32_t m,uint32_t n,uint32_t g,uint32_t hb,uint32_t n0,uint32_t n1){Epilogue<bfloat16_t,true> op;op.Run(p,ws,wo,s,sum,b,y,m,n,g,hb,n0,n1);}
