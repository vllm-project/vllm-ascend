// BSD-3-Clause. Fuse INT4 dot, offset/scale rounding and group reduction.
#include "kernel_operator.h"
using namespace AscendC;
namespace {
template<HardEvent E> __aicore__ inline void Sync(){auto e=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));SetFlag<E>(e);WaitFlag<E>(e);}
template<class T> class Decode {
 static constexpr uint32_t N=16,G=8,L=N*G*128,D=N*G;
 TPipe pipe;
 TBuf<TPosition::VECCALC> wb,wh,pf,ab,ah,af,dot,wsb,wob,sb,sumb,acc,typed,biasbuf;
public:
 __aicore__ inline void Run(GM_ADDR ap,GM_ADDR wp,GM_ADDR wsp,GM_ADDR wop,GM_ADDR sp,GM_ADDR sump,GM_ADDR bp,GM_ADDR yp,uint32_t n,uint32_t g,uint32_t hb){
  GlobalTensor<uint8_t> a,w;a.SetGlobalBuffer((__gm__ uint8_t*)ap);w.SetGlobalBuffer((__gm__ uint8_t*)wp);
  GlobalTensor<float> ws,wo,s,sum;ws.SetGlobalBuffer((__gm__ float*)wsp);wo.SetGlobalBuffer((__gm__ float*)wop);s.SetGlobalBuffer((__gm__ float*)sp);sum.SetGlobalBuffer((__gm__ float*)sump);
  GlobalTensor<T> b,y;b.SetGlobalBuffer((__gm__ T*)bp);y.SetGlobalBuffer((__gm__ T*)yp);
  pipe.InitBuffer(wb,L/2);pipe.InitBuffer(wh,L*2);pipe.InitBuffer(pf,L*4);
  pipe.InitBuffer(ab,G*64);pipe.InitBuffer(ah,G*128*2);pipe.InitBuffer(af,G*128*4);
  pipe.InitBuffer(dot,D*4);pipe.InitBuffer(wsb,D*4);pipe.InitBuffer(wob,D*4);pipe.InitBuffer(sb,D*4);pipe.InitBuffer(sumb,D*4);pipe.InitBuffer(acc,N*4);pipe.InitBuffer(typed,D*2);pipe.InitBuffer(biasbuf,N*2);
  auto wr=wb.Get<uint8_t>();auto wt=wh.Get<half>();auto prod=pf.Get<float>();auto ar=ab.Get<uint8_t>();auto at=ah.Get<half>();auto av=af.Get<float>();
  auto dv=dot.Get<float>();auto wf=wsb.Get<float>();auto of=wob.Get<float>();auto sv=sb.Get<float>();auto sums=sumb.Get<float>();auto out=acc.Get<float>();auto tt=typed.template Get<T>();auto bt=biasbuf.template Get<T>();
  for(uint32_t tile=GetBlockIdx();tile<n/N;tile+=GetBlockNum()){
   uint32_t ni=tile*N;
   Duplicate(out,0.0f,N);PipeBarrier<PIPE_V>();
   for(uint32_t first=0;first<g;first+=G){
    uint32_t count=(g-first<G?g-first:G),rows=count*N,elements=rows*128;
    DataCopyPad(wr,w[(tile*g+first)*N*64],{1,count*N*64,0,0,0},{false,0,0,0});
    DataCopyPad(ar,a[first*64],{1,count*64,0,0,0},{false,0,0,0});
    DataCopyPad(wf,ws[first*n+ni],{static_cast<uint16_t>(count),N*4,(n-N)*4,0,0},{false,0,0,0});
    DataCopyPad(of,wo[first*n+ni],{static_cast<uint16_t>(count),N*4,(n-N)*4,0,0},{false,0,0,0});
    DataCopyPad(sv,s[first*8],{1,count*32,0,0,0},{false,0,0,0});
    DataCopyPad(sums,sum[first*8],{1,count*32,0,0,0},{false,0,0,0});
    Sync<HardEvent::MTE2_V>();Sync<HardEvent::MTE2_S>();
    Cast(wt,wr.ReinterpretCast<int4b_t>(),RoundMode::CAST_NONE,elements);PipeBarrier<PIPE_V>();
    Cast(prod,wt,RoundMode::CAST_NONE,elements);PipeBarrier<PIPE_V>();
    Cast(at,ar.ReinterpretCast<int4b_t>(),RoundMode::CAST_NONE,count*128);PipeBarrier<PIPE_V>();
    Cast(av,at,RoundMode::CAST_NONE,count*128);PipeBarrier<PIPE_V>();
    for(uint32_t j=0;j<count;++j){
     Mul(prod[j*N*128],prod[j*N*128],av[j*128],64,N,{1,1,1,16,16,0});
     Mul(prod[j*N*128+64],prod[j*N*128+64],av[j*128+64],64,N,{1,1,1,16,16,0});
    }
    PipeBarrier<PIPE_V>();
    Add(prod,prod,prod[64],64,rows,{1,1,1,16,16,16});PipeBarrier<PIPE_V>();
    WholeReduceSum(dv,prod,64,rows,1,1,16);PipeBarrier<PIPE_V>();
    // Expand metadata only after all compact scalar reads, backwards for alias safety.
    for(uint32_t jj=count;jj>0;--jj){uint32_t j=jj-1;float scale=sv.GetValue(j*8),ss=sums.GetValue(j*8);Duplicate(sv[j*N],scale,N);Duplicate(sums[j*N],ss,N);}
    PipeBarrier<PIPE_V>();
    Mul(of,of,sums,rows);PipeBarrier<PIPE_V>();Sub(dv,dv,of,rows);PipeBarrier<PIPE_V>();
    Cast(tt,dv,RoundMode::CAST_RINT,rows);PipeBarrier<PIPE_V>();Cast(dv,tt,RoundMode::CAST_NONE,rows);PipeBarrier<PIPE_V>();
    Mul(dv,dv,sv,rows);PipeBarrier<PIPE_V>();Cast(tt,dv,RoundMode::CAST_RINT,rows);PipeBarrier<PIPE_V>();Cast(dv,tt,RoundMode::CAST_NONE,rows);PipeBarrier<PIPE_V>();
    Cast(tt,wf,RoundMode::CAST_RINT,rows);PipeBarrier<PIPE_V>();Cast(wf,tt,RoundMode::CAST_NONE,rows);PipeBarrier<PIPE_V>();
    Mul(dv,dv,wf,rows);PipeBarrier<PIPE_V>();Cast(tt,dv,RoundMode::CAST_RINT,rows);PipeBarrier<PIPE_V>();Cast(dv,tt,RoundMode::CAST_NONE,rows);PipeBarrier<PIPE_V>();
    for(uint32_t j=0;j<count;++j){Add(out,out,dv[j*N],N);PipeBarrier<PIPE_V>();}
    Sync<HardEvent::V_MTE2>();
   }
   Cast(tt,out,RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();
   if(hb){Cast(out,tt,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();DataCopyPad(bt,b[ni],{1,N*2,0,0,0},{false,0,0,0});Sync<HardEvent::MTE2_V>();Cast(dv,bt,RoundMode::CAST_NONE,N);PipeBarrier<PIPE_V>();Add(out,out,dv,N);PipeBarrier<PIPE_V>();Cast(tt,out,RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();}
   Sync<HardEvent::V_MTE3>();DataCopyPad(y[ni],tt,{1,N*2,0,0,0});Sync<HardEvent::MTE3_V>();Sync<HardEvent::V_MTE2>();
  }
 }
};
}
#define ENTRY(TAG,T) \
extern "C" __global__ __aicore__ void splitquant_decode_##TAG(GM_ADDR a,GM_ADDR w,GM_ADDR ws,GM_ADDR wo,GM_ADDR s,GM_ADDR sum,GM_ADDR b,GM_ADDR y,uint32_t n,uint32_t g,uint32_t hb){Decode<T> op;op.Run(a,w,ws,wo,s,sum,b,y,n,g,hb);}
ENTRY(bf16,bfloat16_t)
ENTRY(fp16,half)
