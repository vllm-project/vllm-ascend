// SPDX-License-Identifier: BSD-3-Clause
#include "moe_common.h"
#include "compact_silu.h"
#include "paged_silu.h"
namespace Moe {
template<class T> class InputQuant {
 TPipe& pipe;TBuf<TPosition::VECCALC> in,f,z,backup,red,typed,codes,packed,meta,sparsebuf;
public:
 __aicore__ inline InputQuant(TPipe& p):pipe(p){}
 __aicore__ inline void Run(GM_ADDR xp,GM_ADDR cpp,GM_ADDR cnp,GM_ADDR ap,GM_ADDR sp,uint32_t m,uint32_t h,uint32_t gp,uint32_t scalarRound){
  GlobalTensor<T> x;x.SetGlobalBuffer((__gm__ T*)xp);GlobalTensor<float> cp,cn,s;cp.SetGlobalBuffer((__gm__ float*)cpp);cn.SetGlobalBuffer((__gm__ float*)cnp);s.SetGlobalBuffer((__gm__ float*)sp);GlobalTensor<uint8_t>a;a.SetGlobalBuffer((__gm__ uint8_t*)ap);
  pipe.InitBuffer(in,256);pipe.InitBuffer(f,512);pipe.InitBuffer(z,512);pipe.InitBuffer(backup,512);pipe.InitBuffer(red,512);pipe.InitBuffer(typed,256);pipe.InitBuffer(codes,512);pipe.InitBuffer(packed,64);pipe.InitBuffer(meta,128);pipe.InitBuffer(sparsebuf,1024);
  auto it=in.template Get<T>();auto ft=f.Get<float>();auto zt=z.Get<float>();auto bt=backup.Get<float>();auto rt=red.Get<float>();auto tt=typed.template Get<T>();auto ci=codes.Get<int32_t>();auto pt=packed.Get<uint8_t>();auto mf=meta.Get<float>();uint32_t g=h/128;
  for(uint32_t task=GetBlockIdx();task<m*g;task+=GetBlockNum()){
   uint32_t row=task/g,group=task%g;
   DataCopyPad(it,x[uint64_t(task)*128],{1,256,0,0,0},{false,0,0,0});DataCopyPad(mf,cp[group],{1,4,0,0,0},{false,0,0,0});DataCopyPad(mf[8],cn[group],{1,4,0,0,0},{false,0,0,0});PipeBarrier<PIPE_ALL>();Cast(ft,it,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
   float scale=Quant<T>(ft,zt,bt,rt,mf[24],tt,ci,mf.GetValue(0),mf.GetValue(8),(scalarRound&1));
   Cast(typed.Get<half>(),zt,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(packed.Get<int4b_t>(),typed.Get<half>(),RoundMode::CAST_NONE,128);PipeBarrier<PIPE_ALL>();mf.SetValue(16,scale);PipeBarrier<PIPE_ALL>();
   if(m==1){
    auto sparse=sparsebuf.Get<uint16_t>();Duplicate(sparse,uint16_t(0),gp*32);PipeBarrier<PIPE_V>();
    DataCopy(sparse[(group%gp)*32],pt.ReinterpretCast<uint16_t>(),32);Fence<HardEvent::V_MTE3>();
    DataCopyPad(a[(uint64_t(group/gp)*16+group%gp)*gp*64],sparse.ReinterpretCast<uint8_t>(),{1,gp*64,0,0,0});
   }else{DataCopyPad(a[(uint64_t(group)*m+row)*64],pt,{1,64,0,0,0});}
   if(scalarRound&2){Duplicate(mf[16],scale,8);PipeBarrier<PIPE_ALL>();DataCopyPad(s[uint64_t(task)*8],mf[16],{1,32,0,0,0});}
   else{DataCopyPad(s[uint64_t(task)*8],mf[16],{1,4,0,0,0});}PipeBarrier<PIPE_ALL>();
  }
 }
};
template<class T,bool Compact=false,bool Paged=false> class Post {
 static constexpr uint32_t C=(Compact||Paged)?16:8,L=128,El=2*C*L;
 CompactSilu compact;PagedSilu paged;
 TPipe& pipe;TBuf<TPosition::VECCALC> lut,partial,ws,values,scales,typed,acc,q,meta,codes,idsbuf;
public:
 __aicore__ inline Post(TPipe& p):pipe(p){}
 __aicore__ inline void Run(GM_ADDR pp,GM_ADDR wsp,GM_ADDR sp,GM_ADDR hp,GM_ADDR hcp,GM_ADDR hcn,GM_ADDR idp,GM_ADDR lp,uint32_t m,uint32_t h,uint32_t inter,uint32_t topk,uint32_t experts,uint32_t scalarRound,uint32_t lutBytes=32768){
  GlobalTensor<int32_t> p;p.SetGlobalBuffer((__gm__ int32_t*)pp);GlobalTensor<float>w,s,cp,cn;w.SetGlobalBuffer((__gm__ float*)wsp);s.SetGlobalBuffer((__gm__ float*)sp);cp.SetGlobalBuffer((__gm__ float*)hcp);cn.SetGlobalBuffer((__gm__ float*)hcn);GlobalTensor<T>wt;wt.SetGlobalBuffer((__gm__ T*)wsp);GlobalTensor<T>out,lutgm;out.SetGlobalBuffer((__gm__ T*)hp);lutgm.SetGlobalBuffer((__gm__ T*)lp);GlobalTensor<int64_t> ids;ids.SetGlobalBuffer((__gm__ int64_t*)idp);
  pipe.InitBuffer(lut,Paged?32768:Compact?10240:131072);if constexpr(Compact)compact.Init(pipe);if constexpr(Paged)paged.Init(pipe);pipe.InitBuffer(partial,El*4);pipe.InitBuffer(ws,El*4);pipe.InitBuffer(values,El*4);pipe.InitBuffer(scales,El*4);pipe.InitBuffer(typed,El*2);pipe.InitBuffer(acc,256*4);pipe.InitBuffer(q,4*512);pipe.InitBuffer(meta,576);pipe.InitBuffer(codes,512);pipe.InitBuffer(idsbuf,32);
  auto lt=lut.template Get<T>();auto ip=partial.Get<int32_t>();auto wf=ws.Get<float>();auto vf=values.Get<float>();auto sf=scales.Get<float>();auto tt=typed.template Get<T>();auto af=acc.Get<float>();auto qf=q.Get<float>();auto mf=meta.Get<float>();auto ci=codes.Get<int32_t>();auto id=idsbuf.Get<int64_t>();
  DataCopyPad(lt,lutgm,{1,Paged?lutBytes:Compact?10240U:131072U,0,0,0},{false,0,0,0});PipeBarrier<PIPE_ALL>();
  uint32_t g=h/128,ng=inter/128,n=2*inter,r=m*topk;
  for(uint32_t task=GetBlockIdx();task<r*ng;task+=GetBlockNum()){
   uint32_t route=task/ng,ni=(task%ng)*128,token=route/topk;
   DataCopyPad(id,ids[route],{1,8,0,0,0},{false,0,0,0});DataCopyPad(mf[128],cp[task%ng],{1,4,0,0,0},{false,0,0,0});DataCopyPad(mf[136],cn[task%ng],{1,4,0,0,0},{false,0,0,0});PipeBarrier<PIPE_ALL>();
   uint32_t e=static_cast<uint32_t>(id.GetValue(0));if(e>=experts)e=0;
   Duplicate(af,0.0f,256);PipeBarrier<PIPE_V>();
   for(uint32_t first=0;first<g;first+=C){
    uint32_t count=(g-first<C?g-first:C),total=2*count*128;
    DataCopyPad(ip,p[(uint64_t(route)*g+first)*n+ni],{static_cast<uint16_t>(count),512,(n-128)*4,0,0},{false,0,0,0});
    DataCopyPad(ip[count*128],p[(uint64_t(route)*g+first)*n+inter+ni],{static_cast<uint16_t>(count),512,(n-128)*4,0,0},{false,0,0,0});
    if(scalarRound&4){
     DataCopyPad(sf.ReinterpretCast<T>(),wt[(uint64_t(e)*g+first)*n+ni],{static_cast<uint16_t>(count),256,(n-128)*2,0,0},{false,0,0,0});
     DataCopyPad(sf.ReinterpretCast<T>()[count*128],wt[(uint64_t(e)*g+first)*n+inter+ni],{static_cast<uint16_t>(count),256,(n-128)*2,0,0},{false,0,0,0});
    }else{
    DataCopyPad(wf,w[(uint64_t(e)*g+first)*n+ni],{static_cast<uint16_t>(count),512,(n-128)*4,0,0},{false,0,0,0});
    DataCopyPad(wf[count*128],w[(uint64_t(e)*g+first)*n+inter+ni],{static_cast<uint16_t>(count),512,(n-128)*4,0,0},{false,0,0,0});
    }
    DataCopyPad(mf,s[(uint64_t(token)*g+first)*8],{1,count*32,0,0,0},{false,0,0,0});PipeBarrier<PIPE_ALL>();
    if(scalarRound&4){Cast(wf,sf.ReinterpretCast<T>(),RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();}
    Cast(vf,ip,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();Cast(tt,vf,RoundMode::CAST_RINT,total);PipeBarrier<PIPE_V>();Cast(vf,tt,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();
    if(scalarRound&2){
     BinaryRepeatParams rp{1,1,0,16,16,1};
     Mul(vf,vf,mf,uint64_t(64),static_cast<uint8_t>(count),rp);
     Mul(vf[64],vf[64],mf,uint64_t(64),static_cast<uint8_t>(count),rp);
     Mul(vf[count*128],vf[count*128],mf,uint64_t(64),static_cast<uint8_t>(count),rp);
     Mul(vf[count*128+64],vf[count*128+64],mf,uint64_t(64),static_cast<uint8_t>(count),rp);
    }else{
     for(uint32_t j=0;j<2*count;++j)Duplicate(sf[j*128],mf.GetValue((j%count)*8),128);
     PipeBarrier<PIPE_V>();Mul(vf,vf,sf,total);
    }
    PipeBarrier<PIPE_V>();Cast(tt,vf,RoundMode::CAST_RINT,total);PipeBarrier<PIPE_V>();Cast(vf,tt,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();
    if(!(scalarRound&4)){Cast(tt,wf,RoundMode::CAST_RINT,total);PipeBarrier<PIPE_V>();Cast(wf,tt,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();}Mul(vf,vf,wf,total);PipeBarrier<PIPE_V>();Cast(tt,vf,RoundMode::CAST_RINT,total);PipeBarrier<PIPE_V>();Cast(vf,tt,RoundMode::CAST_NONE,total);PipeBarrier<PIPE_V>();
    for(uint32_t j=0;j<count;++j){Add(af,af,vf[j*128],128);PipeBarrier<PIPE_V>();Add(af[128],af[128],vf[(count+j)*128],128);PipeBarrier<PIPE_V>();}
    Fence<HardEvent::V_MTE2>();
   }
   Cast(tt,af,RoundMode::CAST_RINT,256);PipeBarrier<PIPE_ALL>();
   if constexpr(Paged){paged.Eval(typed.Get<uint16_t>(),typed.Get<int16_t>(),lut.Get<uint16_t>());}
   else if constexpr(Compact){compact.Eval(typed.Get<uint16_t>(),typed.Get<int16_t>(),lut.Get<uint16_t>());}
   else{
   // Widen signed bit patterns, mask to unsigned16, then gather by byte offset.
   Cast(qf,typed.Get<int16_t>(),RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
   Cast(ci,qf,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();
   auto offsets=codes.Get<uint32_t>();auto bitmask=q.Get<uint32_t>();
   Duplicate(bitmask,uint32_t(65535),128);PipeBarrier<PIPE_V>();
   // CANN8.5 And count overload counts 16-bit lanes, even for uint32!
   And(codes.Get<uint16_t>(),codes.Get<uint16_t>(),q.Get<uint16_t>(),256);PipeBarrier<PIPE_V>();
   ShiftLeft(offsets,offsets,uint32_t(1),128);PipeBarrier<PIPE_V>();
   Gather(typed.Get<uint16_t>(),lut.Get<uint16_t>(),offsets,uint32_t(0),uint32_t(128));
   }
   PipeBarrier<PIPE_ALL>();Cast(af,tt,RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();Mul(qf,af,af[128],128);PipeBarrier<PIPE_V>();Cast(tt,qf,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(qf,tt,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
   float scale=Quant<T>(qf,qf[128],qf[256],qf[384],mf[96],tt,ci,mf.GetValue(128),mf.GetValue(136),(scalarRound&1));
   Muls(qf[128],qf[128],scale,128);PipeBarrier<PIPE_V>();Cast(tt,qf[128],RoundMode::CAST_RINT,128);PipeBarrier<PIPE_ALL>();DataCopyPad(out[uint64_t(route)*inter+ni],tt,{1,256,0,0,0});PipeBarrier<PIPE_ALL>();
  }
 }
};
template<class T> class Scatter {
 static constexpr uint32_t C=64;
 TPipe& pipe;TBuf<TPosition::VECCALC> data,prod,broadcast,acc,typed,output,meta;
public:
 __aicore__ inline Scatter(TPipe& p):pipe(p){}
 __aicore__ inline void Run(GM_ADDR yp,GM_ADDR idp,GM_ADDR wp,GM_ADDR op,uint32_t m,uint32_t h,uint32_t topk){
  GlobalTensor<T> y,w,out;y.SetGlobalBuffer((__gm__ T*)yp);w.SetGlobalBuffer((__gm__ T*)wp);out.SetGlobalBuffer((__gm__ T*)op);GlobalTensor<int64_t> ids;ids.SetGlobalBuffer((__gm__ int64_t*)idp);
  pipe.InitBuffer(data,8*C*2);pipe.InitBuffer(prod,8*C*4);pipe.InitBuffer(broadcast,8*C*4);pipe.InitBuffer(acc,C*4);pipe.InitBuffer(typed,8*C*2);pipe.InitBuffer(output,C*2);pipe.InitBuffer(meta,192);
  auto dt=data.template Get<T>();auto vf=prod.Get<float>();auto weights=broadcast.Get<float>();auto af=acc.Get<float>();auto tt=typed.template Get<T>();auto ot=output.template Get<T>();auto ix=meta.Get<int64_t>();auto wt=meta.template Get<T>()[32];auto wf=meta.Get<float>()[32];
  uint32_t nt=h/C;
  for(uint32_t task=GetBlockIdx();task<m*nt;task+=GetBlockNum()){
   uint32_t token=task/nt,ni=(task%nt)*C,used=0;
   DataCopyPad(ix,ids[uint64_t(token)*topk],{1,topk*8,0,0,0},{false,0,0,0});
   DataCopyPad(wt,w[uint64_t(token)*topk],{1,topk*2,0,0,0},{false,0,0,0});
   DataCopyPad(dt,y[uint64_t(token)*topk*h+ni],{static_cast<uint16_t>(topk),C*2,(h-C)*2,0,0},{false,0,0,0});
   Fence<HardEvent::MTE2_V>();Fence<HardEvent::MTE2_S>();
   Cast(wf,wt,RoundMode::CAST_NONE,16);Cast(vf,dt,RoundMode::CAST_NONE,topk*C);Fence<HardEvent::V_S>();
   for(uint32_t j=0;j<topk;++j)Duplicate(weights[j*C],wf.GetValue(j),C);
   PipeBarrier<PIPE_V>();Mul(vf,vf,weights,topk*C);PipeBarrier<PIPE_V>();Cast(tt,vf,RoundMode::CAST_RINT,topk*C);PipeBarrier<PIPE_V>();Cast(vf,tt,RoundMode::CAST_NONE,topk*C);PipeBarrier<PIPE_V>();
   Duplicate(af,0.0f,C);PipeBarrier<PIPE_V>();
   for(uint32_t j=0;j<topk;++j){
    uint32_t slot=0;int64_t best=0x7fffffffffffffffLL;
    for(uint32_t k=0;k<topk;++k)if(!(used&(1U<<k))&&ix.GetValue(k)<best){best=ix.GetValue(k);slot=k;}
    used|=1U<<slot;Add(af,af,vf[slot*C],C);PipeBarrier<PIPE_V>();Cast(ot,af,RoundMode::CAST_RINT,C);PipeBarrier<PIPE_V>();Cast(af,ot,RoundMode::CAST_NONE,C);PipeBarrier<PIPE_V>();
   }
   Fence<HardEvent::V_MTE3>();DataCopyPad(out[uint64_t(token)*h+ni],ot,{1,C*2,0,0,0});Fence<HardEvent::MTE3_V>();Fence<HardEvent::V_MTE2>();
  }
 }
};
}
#define VECTOR_ENTRIES(TAG,T) \
extern "C" __global__ __aicore__ void moe_quant_##TAG(GM_ADDR x,GM_ADDR cp,GM_ADDR cn,GM_ADDR a,GM_ADDR s,uint32_t m,uint32_t h,uint32_t gp,uint32_t scalarRound){TPipe pipe;Moe::InputQuant<T> k(pipe);k.Run(x,cp,cn,a,s,m,h,gp,scalarRound);} \
extern "C" __global__ __aicore__ void moe_post_##TAG(GM_ADDR p,GM_ADDR ws,GM_ADDR s,GM_ADDR out,GM_ADDR cp,GM_ADDR cn,GM_ADDR ids,GM_ADDR lut,uint32_t m,uint32_t h,uint32_t i,uint32_t topk,uint32_t e,uint32_t scalarRound){TPipe pipe;Moe::Post<T> k(pipe);k.Run(p,ws,s,out,cp,cn,ids,lut,m,h,i,topk,e,scalarRound);} \
extern "C" __global__ __aicore__ void moe_scatter_##TAG(GM_ADDR y,GM_ADDR ids,GM_ADDR w,GM_ADDR out,uint32_t m,uint32_t h,uint32_t topk){TPipe pipe;Moe::Scatter<T> k(pipe);k.Run(y,ids,w,out,m,h,topk);}
VECTOR_ENTRIES(bf16,bfloat16_t)
VECTOR_ENTRIES(fp16,half)

extern "C" __global__ __aicore__ void moe_post_compact_bf16(GM_ADDR p,GM_ADDR ws,GM_ADDR s,GM_ADDR out,GM_ADDR cp,GM_ADDR cn,GM_ADDR ids,GM_ADDR lut,uint32_t m,uint32_t h,uint32_t i,uint32_t topk,uint32_t e,uint32_t scalarRound){TPipe pipe;Moe::Post<bfloat16_t,true> k(pipe);k.Run(p,ws,s,out,cp,cn,ids,lut,m,h,i,topk,e,scalarRound);}

extern "C" __global__ __aicore__ void moe_post_paged_bf16(GM_ADDR p,GM_ADDR ws,GM_ADDR s,GM_ADDR out,GM_ADDR cp,GM_ADDR cn,GM_ADDR ids,GM_ADDR lut,uint32_t m,uint32_t h,uint32_t i,uint32_t topk,uint32_t e,uint32_t scalarRound,uint32_t lutBytes){TPipe pipe;Moe::Post<bfloat16_t,false,true> k(pipe);k.Run(p,ws,s,out,cp,cn,ids,lut,m,h,i,topk,e,scalarRound,lutBytes);}
