#include "kernel_operator.h"
using namespace AscendC;
namespace {
template <HardEvent E> __aicore__ inline void SyncStage() {
 auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
 SetFlag<E>(event); WaitFlag<E>(event);
}

template<class T> class Quantize {
public:
 __aicore__ inline void Run(GM_ADDR xp,GM_ADDR cpp,GM_ADDR cnp,GM_ADDR ap,GM_ADDR sp,GM_ADDR sump,uint32_t m,uint32_t k) {
  const bool sparse_output=(m&0x80000000U)!=0;m&=0x7fffffffU;
  uint32_t g=k/128;
  GlobalTensor<T> x; x.SetGlobalBuffer((__gm__ T*)xp);
  GlobalTensor<float> cp,cn,s,asum;cp.SetGlobalBuffer((__gm__ float*)cpp);cn.SetGlobalBuffer((__gm__ float*)cnp);s.SetGlobalBuffer((__gm__ float*)sp);asum.SetGlobalBuffer((__gm__ float*)sump);
  GlobalTensor<uint8_t> a;a.SetGlobalBuffer((__gm__ uint8_t*)ap);
  pipe.InitBuffer(sparsebuf,1024);pipe.InitBuffer(in,256);pipe.InitBuffer(f,512);pipe.InitBuffer(z,512);pipe.InitBuffer(tmp,512);pipe.InitBuffer(red,512);pipe.InitBuffer(o,64);pipe.InitBuffer(t,256);pipe.InitBuffer(code,512);pipe.InitBuffer(packed,64);pipe.InitBuffer(clips,64);
  auto it=in.template Get<T>();auto ft=f.template Get<float>();auto zt=z.template Get<float>();auto tt=t.template Get<T>();auto rt=red.template Get<float>();auto ot=o.template Get<float>();auto tmpf=tmp.template Get<float>();auto ci=code.template Get<int32_t>();auto pt=packed.template Get<uint8_t>();auto cf=clips.template Get<float>();
  for(uint32_t task=GetBlockIdx();task<m*g;task+=GetBlockNum()) {
   uint32_t row=task/g,group=task%g;
   DataCopyPad(it,x[task*128],{1,256,0,0,0},{false,0,0,0});
   DataCopyPad(cf,cp[group],{1,4,0,0,0},{false,0,0,0});
   DataCopyPad(cf[8],cn[group],{1,4,0,0,0},{false,0,0,0});PipeBarrier<PIPE_ALL>();
   Cast(ft,it,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
   Adds(tmpf,ft,0.0f,128);PipeBarrier<PIPE_V>();ReduceMax(ot,tmpf,rt,128);PipeBarrier<PIPE_ALL>();float hi=ot.GetValue(0);if(hi<0)hi=0;
   Adds(tmpf,ft,0.0f,128);PipeBarrier<PIPE_V>();ReduceMin(ot,tmpf,rt,128);PipeBarrier<PIPE_ALL>();float lo=ot.GetValue(0);if(lo>0)lo=0;
   float pos=Round(cf.GetValue(0),zt,tt),neg=Round(cf.GetValue(8),zt,tt);
   hi=Round(hi*pos,zt,tt);lo=Round(lo*neg,zt,tt);
   float scale=Round((hi>-lo?hi:-lo)/7.0f,zt,tt);if(scale==0)scale=1;
   Duplicate(zt,scale,128);PipeBarrier<PIPE_V>();Div(zt,ft,zt,128);PipeBarrier<PIPE_V>();
   Cast(tt,zt,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(zt,tt,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
   Maxs(zt,zt,-8.0f,128);PipeBarrier<PIPE_V>();Mins(zt,zt,7.0f,128);PipeBarrier<PIPE_V>();
   Cast(ci,zt,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(zt,ci,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
   ReduceSum(ot,zt,rt,128);PipeBarrier<PIPE_ALL>();float sum=ot.GetValue(0);
   // Like ops-nn FlatQuant: vector conversion emits packed signed INT4.
   auto half_codes=t.template Get<half>();
   Cast(half_codes,zt,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();
   Cast(packed.template Get<int4b_t>(),half_codes,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
   ot.SetValue(0,scale);ot.SetValue(8,sum);PipeBarrier<PIPE_ALL>();
   if(sparse_output){
    auto diagonal=sparsebuf.Get<uint16_t>();Duplicate(diagonal,uint16_t(0),512);PipeBarrier<PIPE_V>();
    DataCopy(diagonal[(group%16)*32],pt.ReinterpretCast<uint16_t>(),32);SyncStage<HardEvent::V_MTE3>();
    DataCopyPad(a[uint64_t(group)*1024],diagonal.ReinterpretCast<uint8_t>(),{1,1024,0,0,0});
   }else{DataCopyPad(a[(group*m+row)*64],pt,{1,64,0,0,0});}
   DataCopyPad(s[task*8],ot,{1,4,0,0,0});DataCopyPad(asum[task*8],ot[8],{1,4,0,0,0});PipeBarrier<PIPE_ALL>();
  }
 }
private:
 __aicore__ inline float Round(float v,LocalTensor<float> f,LocalTensor<T> t){Duplicate(f,v,16);PipeBarrier<PIPE_V>();Cast(t,f,RoundMode::CAST_RINT,16);PipeBarrier<PIPE_V>();Cast(f,t,RoundMode::CAST_NONE,16);PipeBarrier<PIPE_ALL>();return f.GetValue(0);}
 TPipe pipe;TBuf<TPosition::VECCALC> sparsebuf,in,f,z,tmp,red,o,t,code,packed,clips;
};

// Cube fast path for the exported group rotation: rotate each 128-wide
// activation group in UB and quantize it before writing the packed ABI.  This
// avoids materializing a dense block-diagonal [K,K] matrix and a rotated [M,K]
// tensor between two separate launches.
template<class T, bool VectorTile=false, bool Compensated=false, bool Prefill=false, bool PreRotated=false> class RotateQuantize {
public:
 __aicore__ inline void Run(GM_ADDR xp,GM_ADDR gtp,GM_ADDR cpp,GM_ADDR cnp,
                            GM_ADDR ap,GM_ADDR sp,GM_ADDR sump,uint32_t m,uint32_t k,uint32_t projections) {
  const bool sparse_output=(m&0x80000000U)!=0;m&=0x7fffffffU;
  uint32_t g=k/128;
  GlobalTensor<T> x,gt; x.SetGlobalBuffer((__gm__ T*)xp); gt.SetGlobalBuffer((__gm__ T*)gtp);
  GlobalTensor<float> cp,cn,s,asum; cp.SetGlobalBuffer((__gm__ float*)cpp); cn.SetGlobalBuffer((__gm__ float*)cnp); s.SetGlobalBuffer((__gm__ float*)sp); asum.SetGlobalBuffer((__gm__ float*)sump);
  GlobalTensor<uint8_t> a; a.SetGlobalBuffer((__gm__ uint8_t*)ap);
  pipe.InitBuffer(sparsebuf,1024);pipe.InitBuffer(in,256); pipe.InitBuffer(f,512); pipe.InitBuffer(z,512); pipe.InitBuffer(tmp,512); pipe.InitBuffer(red,512); pipe.InitBuffer(o,64); pipe.InitBuffer(t,256); pipe.InitBuffer(code,512); pipe.InitBuffer(packed,64); pipe.InitBuffer(clips,64); pipe.InitBuffer(rotation,Prefill?65536:32768); if constexpr(VectorTile) pipe.InitBuffer(products,65536);
  if constexpr(Compensated){
   pipe.InitBuffer(compact,32768);
   pipe.InitBuffer(exactHigh,512);pipe.InitBuffer(exactLow,512);
   pipe.InitBuffer(exactNextHigh,512);pipe.InitBuffer(exactNextLow,512);
   pipe.InitBuffer(exactWork,512);pipe.InitBuffer(gatherHigh,512);
   pipe.InitBuffer(gatherLow,512);pipe.InitBuffer(gatherIndex,Prefill?1536:512);
  }
  auto it=in.template Get<T>(); auto ft=f.template Get<float>(); auto zt=z.template Get<float>(); auto tt=t.template Get<T>(); auto rt=red.template Get<float>(); auto ot=o.template Get<float>(); auto tmpf=tmp.template Get<float>(); auto ci=code.template Get<int32_t>(); auto pt=packed.template Get<uint8_t>(); auto cf=clips.template Get<float>(); auto rot=rotation.template Get<T>();
  if constexpr(Prefill) {
   auto maps=gatherIndex.template Get<int32_t>();
   for(uint32_t slot=0;slot<3;++slot){
    uint32_t width=(slot==0?4:(slot==1?2:1));
    ArithProgression(maps[slot*128],static_cast<int32_t>(width*4),static_cast<int32_t>(4),128);PipeBarrier<PIPE_V>();
    Mins(maps[slot*128],maps[slot*128],static_cast<int32_t>(127*4),128);PipeBarrier<PIPE_V>();
   }
  }
  const uint32_t rowTiles=(m+7)/8;
  const uint32_t tasks=Prefill?g*rowTiles:m*g;
  for(uint32_t tile=GetBlockIdx();tile<tasks;tile+=GetBlockNum()) {
   const uint32_t group=Prefill?tile/rowTiles:tile%g;
   const uint32_t firstRow=Prefill?(tile%rowTiles)*8:tile/g;
   const uint32_t rowCount=Prefill?(m-firstRow<8?m-firstRow:8):1;
   if constexpr(Prefill) {
    auto stage=compact.template Get<T>();
    DataCopyPad(stage,gt[group*16384],{1,32768,0,0,0},{false,0,0,0});PipeBarrier<PIPE_ALL>();
    Cast(rotation.template Get<float>(),stage,RoundMode::CAST_NONE,16384);PipeBarrier<PIPE_V>();
   }
   for(uint32_t localRow=0;localRow<rowCount;++localRow) {
   const uint32_t row=firstRow+localRow,task=row*g+group;
   DataCopyPad(it,x[task*128],{1,256,0,0,0},{false,0,0,0});
   if constexpr(!Prefill && !PreRotated) {DataCopyPad(rot,gt[group*16384],{1,32768,0,0,0},{false,0,0,0});} PipeBarrier<PIPE_ALL>();
   Cast(ft,it,RoundMode::CAST_NONE,128); PipeBarrier<PIPE_V>();
   if constexpr(!PreRotated) {
   if constexpr(Compensated) {
    auto pf=products.template Get<float>();auto reduced=compact.template Get<float>();
    if constexpr(Prefill) {
     auto cachedRotation=rotation.template Get<float>();
     Mul(pf,cachedRotation,ft,64,128,{1,1,1,16,16,0});PipeBarrier<PIPE_V>();
     Mul(pf[64],cachedRotation[64],ft[64],64,128,{1,1,1,16,16,0});PipeBarrier<PIPE_V>();
    } else {
     Cast(pf,rot,RoundMode::CAST_NONE,16384);PipeBarrier<PIPE_V>();
     Mul(pf,pf,ft,64,128,{1,1,1,16,16,0});PipeBarrier<PIPE_V>();
     Mul(pf[64],pf[64],ft[64],64,128,{1,1,1,16,16,0});PipeBarrier<PIPE_V>();
    }
    Add(reduced,pf,pf[64],64,128,{1,1,1,8,16,16});PipeBarrier<PIPE_V>();
    WholeReduceSum(zt,reduced,64,128,1,1,8);PipeBarrier<PIPE_V>();
    // BF16 products are exact FP32. Bound the seven-level FP32 summation
    // error by 16*u*sum(abs(products)), u=2^-24 (with conservative slack).
    // The rotation buffer is no longer needed after products are formed.
    if constexpr(Prefill) {
     Abs(pf,pf,16384);PipeBarrier<PIPE_V>();
     Add(reduced,pf,pf[64],64,128,{1,1,1,8,16,16});PipeBarrier<PIPE_V>();
    } else {
     auto absolute=rotation.template Get<float>();
     Abs(reduced,pf,64,128,{1,1,8,16});PipeBarrier<PIPE_V>();
     Abs(absolute,pf[64],64,128,{1,1,8,16});PipeBarrier<PIPE_V>();
     Add(reduced,reduced,absolute,8192);PipeBarrier<PIPE_V>();
    }
    WholeReduceSum(tmpf,reduced,64,128,1,1,8);PipeBarrier<PIPE_ALL>();
    // Batch the reliable-rounding proof; compensate only uncertain groups.
    auto midbits=code.template Get<int32_t>();
    auto maskbits=exactWork.template Get<int32_t>();
    Duplicate(maskbits,static_cast<int32_t>(0xFFFF0000U),128);PipeBarrier<PIPE_V>();
    And(midbits.template ReinterpretCast<uint16_t>(),zt.template ReinterpretCast<uint16_t>(),maskbits.template ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();
    Duplicate(maskbits,static_cast<int32_t>(0x8000U),128);PipeBarrier<PIPE_V>();
    Or(midbits.template ReinterpretCast<uint16_t>(),midbits.template ReinterpretCast<uint16_t>(),maskbits.template ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();
    auto margin=gatherHigh.template Get<float>();
    auto bound=gatherLow.template Get<float>();
    Sub(margin,zt,midbits.template ReinterpretCast<float>(),128);PipeBarrier<PIPE_V>();
    Abs(margin,margin,128);PipeBarrier<PIPE_V>();
    Muls(bound,tmpf,0.00000095367431640625f,128);PipeBarrier<PIPE_V>();
    Sub(margin,margin,bound,128);PipeBarrier<PIPE_V>();
    ReduceMin(ot,margin,rt,128);PipeBarrier<PIPE_ALL>();
    if(ot.GetValue(0)<=0.0f) {
    for(uint32_t column=0;column<128;++column) {
     float fast=zt.GetValue(column);
     union Bits {float f;uint32_t u;} value,mid;
     value.f=fast;mid.u=(value.u&0xFFFF0000U)|0x8000U;
     float distance=fast-mid.f;if(distance<0.0f)distance=-distance;
     if(distance>tmpf.GetValue(column)*0.00000095367431640625f)continue;
     // Only values with uncertain BF16 rounding need compensated reduction.
     auto high=exactHigh.template Get<float>();auto lower=exactLow.template Get<float>();
     auto nh=exactNextHigh.template Get<float>();auto nl=exactNextLow.template Get<float>();
     auto work=exactWork.template Get<float>();
     if constexpr(Prefill) {Mul(high,rotation.template Get<float>()[column*128],ft,128);}
     else {Adds(high,pf[column*128],0.0f,128);}PipeBarrier<PIPE_V>();
     Duplicate(lower,0.0f,128);PipeBarrier<PIPE_V>();
     for(uint32_t width=64;width>0;width/=2) {
      auto rh=high[width];auto rl=lower[width];
      if(width<8) {
       auto ix=gatherIndex.template Get<int32_t>();
       if constexpr(Prefill){ix=ix[(width==4?0:(width==2?1:2))*128];}
       else {
        ArithProgression(ix,static_cast<int32_t>(width*4),static_cast<int32_t>(4),128);PipeBarrier<PIPE_V>();
        Mins(ix,ix,static_cast<int32_t>(127*4),128);PipeBarrier<PIPE_V>();
       }
       rh=gatherHigh.template Get<float>();rl=gatherLow.template Get<float>();
       Gather(rh,high,ix.template ReinterpretCast<uint32_t>(),0,128);
       Gather(rl,lower,ix.template ReinterpretCast<uint32_t>(),0,128);PipeBarrier<PIPE_V>();
      }
      Add(nh,high,rh,width,1,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
      Sub(work,nh,high,width,1,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
      Sub(nl,nh,work,width,1,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
      Sub(nl,high,nl,width,1,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
      Sub(work,rh,work,width,1,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
      Add(nl,nl,work,width,1,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
      Add(nl,nl,lower,width,1,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
      Add(nl,nl,rl,width,1,{1,1,1,8,8,8});PipeBarrier<PIPE_V>();
      auto oldHigh=high;high=nh;nh=oldHigh;
      auto oldLow=lower;lower=nl;nl=oldLow;
     }
     PipeBarrier<PIPE_ALL>();
     volatile float hi=high.GetValue(0),lo=lower.GetValue(0);
     float rounded=hi+lo;float tail=(hi-rounded)+lo;
     value.f=rounded;
     if((value.u&65535U)==32768U && tail!=0.0f) {
      bool up=tail>0.0f,positive=(value.u&0x80000000U)==0;
      value.u+=(up==positive)?1U:static_cast<uint32_t>(-1);rounded=value.f;
     }
     zt.SetValue(column,rounded);
    }
    } // uncertain group
    PipeBarrier<PIPE_ALL>();
   } else if constexpr(VectorTile) {
    auto pf=products.template Get<float>();
    Cast(pf,rot,RoundMode::CAST_NONE,16384); PipeBarrier<PIPE_V>();
    // Each repeat computes one output column's low/high 64 input lanes.
    Mul(pf,pf,ft,64,128,{1,1,1,16,16,0}); PipeBarrier<PIPE_V>();
    Mul(pf[64],pf[64],ft[64],64,128,{1,1,1,16,16,0}); PipeBarrier<PIPE_V>();
    Add(pf,pf,pf[64],64,128,{1,1,1,16,16,16}); PipeBarrier<PIPE_V>();
    WholeReduceSum(zt,pf,64,128,1,1,16); PipeBarrier<PIPE_V>();
   } else {
   Duplicate(zt,0.0f,128); PipeBarrier<PIPE_V>();
   for(uint32_t i=0;i<128;++i) {
    Cast(tmpf,rot[i*128],RoundMode::CAST_NONE,128); PipeBarrier<PIPE_V>();
    Muls(tmpf,tmpf,ft.GetValue(i),128); PipeBarrier<PIPE_V>();
    Add(zt,zt,tmpf,128); PipeBarrier<PIPE_V>();
   }
   }
   Cast(it,zt,RoundMode::CAST_RINT,128); PipeBarrier<PIPE_V>();
   Cast(ft,it,RoundMode::CAST_NONE,128); PipeBarrier<PIPE_V>();
   } // rotation skipped for pre-rotated dense numerical contract
   Adds(tmpf,ft,0.0f,128);ReduceMax(ot,tmpf,rt,128);PipeBarrier<PIPE_ALL>();float rawHi=ot.GetValue(0);if(rawHi<0)rawHi=0;
   Adds(tmpf,ft,0.0f,128);ReduceMin(ot,tmpf,rt,128);PipeBarrier<PIPE_ALL>();float rawLo=ot.GetValue(0);if(rawLo>0)rawLo=0;
   for(uint32_t projection=0;projection<projections;++projection) {
   DataCopyPad(cf,cp[projection*g+group],{1,4,0,0,0},{false,0,0,0});
   DataCopyPad(cf[8],cn[projection*g+group],{1,4,0,0,0},{false,0,0,0});PipeBarrier<PIPE_ALL>();
   float hi=rawHi;
   float lo=rawLo;
   float pos=Round(cf.GetValue(0),zt,tt),neg=Round(cf.GetValue(8),zt,tt);
   hi=Round(hi*pos,zt,tt); lo=Round(lo*neg,zt,tt);
   float scale=Round((hi>-lo?hi:-lo)/7.0f,zt,tt); if(scale==0)scale=1;
   Duplicate(zt,scale,128); PipeBarrier<PIPE_V>(); Div(zt,ft,zt,128); PipeBarrier<PIPE_V>();
   Cast(tt,zt,RoundMode::CAST_RINT,128); PipeBarrier<PIPE_V>(); Cast(zt,tt,RoundMode::CAST_NONE,128); PipeBarrier<PIPE_V>();
   Maxs(zt,zt,-8.0f,128); PipeBarrier<PIPE_V>(); Mins(zt,zt,7.0f,128); PipeBarrier<PIPE_V>();
   Cast(ci,zt,RoundMode::CAST_RINT,128); PipeBarrier<PIPE_V>(); Cast(zt,ci,RoundMode::CAST_NONE,128); PipeBarrier<PIPE_V>();
   ReduceSum(ot,zt,rt,128); PipeBarrier<PIPE_ALL>(); float sum=ot.GetValue(0);
   // Like ops-nn FlatQuant: vector conversion emits packed signed INT4.
   auto half_codes=t.template Get<half>();
   Cast(half_codes,zt,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();
   Cast(packed.template Get<int4b_t>(),half_codes,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
   ot.SetValue(0,scale); ot.SetValue(8,sum); PipeBarrier<PIPE_ALL>();
   if(sparse_output){
    auto diagonal=sparsebuf.Get<uint16_t>();Duplicate(diagonal,uint16_t(0),512);PipeBarrier<PIPE_V>();
    DataCopy(diagonal[(group%16)*32],pt.ReinterpretCast<uint16_t>(),32);SyncStage<HardEvent::V_MTE3>();
    DataCopyPad(a[((uint64_t(projection)*((g+15)/16)+group/16)*16+group%16)*1024],diagonal.ReinterpretCast<uint8_t>(),{1,1024,0,0,0});
   }else{DataCopyPad(a[(group*projections*m+projection*m+row)*64],pt,{1,64,0,0,0});}
   DataCopyPad(s[((projection*m+row)*g+group)*8],ot,{1,4,0,0,0}); DataCopyPad(asum[((projection*m+row)*g+group)*8],ot[8],{1,4,0,0,0}); PipeBarrier<PIPE_ALL>();
   } // independent projection quantization
   } // row tile
  }
 }
private:
 __aicore__ inline float Round(float v,LocalTensor<float> f,LocalTensor<T> t){Duplicate(f,v,16);PipeBarrier<PIPE_V>();Cast(t,f,RoundMode::CAST_RINT,16);PipeBarrier<PIPE_V>();Cast(f,t,RoundMode::CAST_NONE,16);PipeBarrier<PIPE_ALL>();return f.GetValue(0);}
 TPipe pipe; TBuf<TPosition::VECCALC> sparsebuf,in,f,z,tmp,red,o,t,code,packed,clips,rotation,products,compact,exactHigh,exactLow,exactNextHigh,exactNextLow,exactWork,gatherHigh,gatherLow,gatherIndex;
};
}

extern "C" __global__ __aicore__ void splitquant_multi_quant(GM_ADDR x,GM_ADDR gt,GM_ADDR cp,GM_ADDR cn,GM_ADDR a,GM_ADDR s,GM_ADDR sum,uint32_t m,uint32_t k,uint32_t projections){if((m&0x7fffffffU)>=4 && k>=4096){RotateQuantize<bfloat16_t,true,true,true> op;op.Run(x,gt,cp,cn,a,s,sum,m,k,projections);}else{RotateQuantize<bfloat16_t,true,true> op;op.Run(x,gt,cp,cn,a,s,sum,m,k,projections);}}

extern "C" __global__ __aicore__ void splitquant_multi_quant_plain(GM_ADDR x,GM_ADDR gt,GM_ADDR cp,GM_ADDR cn,GM_ADDR a,GM_ADDR s,GM_ADDR sum,uint32_t m,uint32_t k,uint32_t projections){RotateQuantize<bfloat16_t,false,false,false,true> op;op.Run(x,gt,cp,cn,a,s,sum,m,k,projections);}
