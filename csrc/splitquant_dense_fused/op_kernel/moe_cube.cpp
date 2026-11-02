// SPDX-License-Identifier: BSD-3-Clause
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "constant_cube_config.h"
using namespace AscendC;
namespace MoeCube {
constexpr auto Config=GetMDLConfig(false,false,0,false,false,false,true);
template<class Header> __aicore__ inline void Load(Header& dst,GM_ADDR src){auto* d=reinterpret_cast<int32_t*>(&dst);auto* s=reinterpret_cast<__gm__ int32_t*>(src);for(uint32_t j=0;j<(sizeof(Header)+3)/4;++j)d[j]=s[j];}
template<bool NzWeight=false,bool Fixed=false> class Gate {
 using A=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int4b_t,false>;
 using B=matmul::MatmulType<TPosition::GM,(NzWeight?CubeFormat::NZ:CubeFormat::ND),int4b_t,!NzWeight>;
 using C=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int32_t,false>;
 TPipe* pipe;TCubeTiling tile{};static constexpr auto& GateMode=Const::Mode<Fixed,A,B,C,768,256,512>::Value;matmul::MatmulImpl<A,B,C,C,GateMode> mm;
 GlobalTensor<int4b_t>a,b;GlobalTensor<int32_t>c;
public:
 __aicore__ inline Gate(TPipe* p):pipe(p){}
 __aicore__ inline void Run(GM_ADDR ap,GM_ADDR bp,GM_ADDR ids,GM_ADDR cp,uint32_t m,uint32_t h,uint32_t inter,uint32_t topk,uint32_t experts,uint32_t tn,uint32_t gp,GM_ADDR tp){
  if ASCEND_IS_AIV {return;}
  a.SetGlobalBuffer((__gm__ int4b_t*)ap);b.SetGlobalBuffer((__gm__ int4b_t*)bp);c.SetGlobalBuffer((__gm__ int32_t*)cp);Load(tile,tp);mm.SetSubBlockIdx(0);mm.Init(&tile,pipe);
  if constexpr(Fixed){
   mm.SetOrgShape(16,1536,2048,2048,1536);mm.SetTensorA(a,false);
   for(uint32_t task=GetBlockIdx();task<topk*2;task+=GetBlockNum()){
    uint32_t route=task/2,ni=(task%2)*768;int64_t raw=(reinterpret_cast<__gm__ int64_t*>(ids))[route];uint32_t e=static_cast<uint32_t>(raw);if(e>=experts)e=0;
    mm.SetTensorB(b[(uint64_t(e)*1536+ni)*2048],false);mm.IterateAll(c[uint64_t(route)*16*1536+ni],0);PipeBarrier<PIPE_ALL>();SetAtomicNone();
   }
   mm.End();return;
  }
  uint32_t g=h/128,n=2*inter,nt=(n+tn-1)/tn,r=m*topk,chunks=(g+gp-1)/gp;
  uint32_t groups=m==1?chunks:g;
  for(uint32_t task=GetBlockIdx();task<r*groups*nt;task+=GetBlockNum()){
   uint32_t route=task/(groups*nt),group=(task/nt)%groups,ni=(task%nt)*tn,len=(n-ni<tn?n-ni:tn),token=route/topk;
   int64_t raw=(reinterpret_cast<__gm__ int64_t*>(ids))[route];uint32_t e=static_cast<uint32_t>(raw);if(e>=experts)e=0;
   if(m==1){
    uint32_t rows=(g-group*gp<gp?g-group*gp:gp);
    mm.SetOrgShape(16,n,gp*128,h,n);mm.SetSingleShape(rows,len,rows*128);
    mm.SetTensorA(a[uint64_t(group)*16*gp*128],false);mm.SetTensorB(b[(uint64_t(e)*n+ni)*h+uint64_t(group)*gp*128*(NzWeight?64:1)],!NzWeight);
    mm.IterateAll(c[(uint64_t(route)*g+group*gp)*n+ni],0);
   }else{
    mm.SetOrgShape(1,n,128,h,n);mm.SetSingleShape(1,len,128);
    mm.SetTensorA(a[(uint64_t(group)*m+token)*128],false);mm.SetTensorB(b[(uint64_t(e)*n+ni)*h+uint64_t(group)*128*(NzWeight?64:1)],!NzWeight);
    mm.IterateAll(c[(uint64_t(route)*g+group)*n+ni],0);
   }
   PipeBarrier<PIPE_ALL>();SetAtomicNone();
  }
  mm.End();
 }
};
template<class T,bool Rotation,bool NzWeight=false> class Dense {
 using A=matmul::MatmulType<TPosition::GM,CubeFormat::ND,T,false>;
 using B=matmul::MatmulType<TPosition::GM,(NzWeight?CubeFormat::NZ:CubeFormat::ND),T,(NzWeight?false:!Rotation)>;
 using C=matmul::MatmulType<TPosition::GM,CubeFormat::ND,T,false>;
 TPipe* pipe;TCubeTiling tile{};matmul::MatmulImpl<A,B,C,C,Config> mm;
 GlobalTensor<T>a,b,c;
public:
 __aicore__ inline Dense(TPipe* p):pipe(p){}
 __aicore__ inline void Run(GM_ADDR ap,GM_ADDR bp,GM_ADDR ids,GM_ADDR cp,uint32_t r,uint32_t h,uint32_t inter,uint32_t experts,uint32_t tn,GM_ADDR tp){
  if ASCEND_IS_AIV {return;}
  a.SetGlobalBuffer((__gm__ T*)ap);b.SetGlobalBuffer((__gm__ T*)bp);c.SetGlobalBuffer((__gm__ T*)cp);Load(tile,tp);mm.SetSubBlockIdx(0);mm.Init(&tile,pipe);mm.SetHF32(false,0);
  if constexpr(Rotation){
   uint32_t g=inter/128,mt=(r+15)/16,nt=128/tn,cols=tn;
   for(uint32_t task=GetBlockIdx();task<g*mt*nt;task+=GetBlockNum()){
    uint32_t group=task/(mt*nt),row=((task%(mt*nt))/nt)*16,ni=(task%nt)*cols,len=(r-row<16?r-row:16);
    mm.SetOrgShape(r,128,inter,128,inter);mm.SetSingleShape(len,cols,128);
    mm.SetTensorA(a[uint64_t(row)*inter+group*128],false);mm.SetTensorB(b[uint64_t(group)*16384+ni],false);mm.IterateAll(c[uint64_t(row)*inter+group*128+ni],0);PipeBarrier<PIPE_ALL>();SetAtomicNone();
   }
  }else{
   uint32_t nt=(h+tn-1)/tn;
   for(uint32_t task=GetBlockIdx();task<r*nt;task+=GetBlockNum()){
    uint32_t route=task/nt,ni=(task%nt)*tn,len=(h-ni<tn?h-ni:tn);int64_t raw=(reinterpret_cast<__gm__ int64_t*>(ids))[route];uint32_t e=static_cast<uint32_t>(raw);if(e>=experts)e=0;
    mm.SetOrgShape(1,h,inter,inter,h);mm.SetSingleShape(1,len,inter);mm.SetTensorA(a[uint64_t(route)*inter],false);mm.SetTensorB(b[uint64_t(e)*h*inter+uint64_t(ni)*inter],!NzWeight);mm.IterateAll(c[uint64_t(route)*h+ni],0);PipeBarrier<PIPE_ALL>();SetAtomicNone();
   }
  }
  mm.End();
 }
};
}
extern "C" __global__ __aicore__ void moe_gate(GM_ADDR a,GM_ADDR b,GM_ADDR ids,GM_ADDR c,uint32_t m,uint32_t h,uint32_t i,uint32_t k,uint32_t e,uint32_t tn,uint32_t gp,GM_ADDR workspace,GM_ADDR tiling){TPipe pipe;MoeCube::Gate<> op(&pipe);op.Run(a,b,ids,c,m,h,i,k,e,tn,gp,tiling);}
#define DENSE_ENTRIES(TAG,T) \
extern "C" __global__ __aicore__ void moe_rotate_##TAG(GM_ADDR a,GM_ADDR b,GM_ADDR ids,GM_ADDR c,uint32_t r,uint32_t h,uint32_t i,uint32_t e,uint32_t tn,GM_ADDR workspace,GM_ADDR tiling){TPipe pipe;MoeCube::Dense<T,true> op(&pipe);op.Run(a,b,ids,c,r,h,i,e,tn,tiling);} \
extern "C" __global__ __aicore__ void moe_down_##TAG(GM_ADDR a,GM_ADDR b,GM_ADDR ids,GM_ADDR c,uint32_t r,uint32_t h,uint32_t i,uint32_t e,uint32_t tn,GM_ADDR workspace,GM_ADDR tiling){TPipe pipe;MoeCube::Dense<T,false> op(&pipe);op.Run(a,b,ids,c,r,h,i,e,tn,tiling);}
DENSE_ENTRIES(bf16,bfloat16_t)
DENSE_ENTRIES(fp16,half)

#define NZ_ENTRIES(TAG,T) extern "C" __global__ __aicore__ void moe_down_nz_##TAG(GM_ADDR a,GM_ADDR b,GM_ADDR ids,GM_ADDR c,uint32_t r,uint32_t h,uint32_t i,uint32_t e,uint32_t tn,GM_ADDR workspace,GM_ADDR tiling){TPipe pipe;MoeCube::Dense<T,false,true> op(&pipe);op.Run(a,b,ids,c,r,h,i,e,tn,tiling);}
NZ_ENTRIES(bf16,bfloat16_t)
NZ_ENTRIES(fp16,half)

extern "C" __global__ __aicore__ void moe_gate_nz(GM_ADDR a,GM_ADDR b,GM_ADDR ids,GM_ADDR c,uint32_t m,uint32_t h,uint32_t i,uint32_t k,uint32_t e,uint32_t tn,uint32_t gp,GM_ADDR workspace,GM_ADDR tiling){TPipe pipe;MoeCube::Gate<true> op(&pipe);op.Run(a,b,ids,c,m,h,i,k,e,tn,gp,tiling);}

extern "C" __global__ __aicore__ void moe_gate_const_nz(GM_ADDR a,GM_ADDR b,GM_ADDR ids,GM_ADDR c,uint32_t m,uint32_t h,uint32_t i,uint32_t k,uint32_t e,uint32_t tn,uint32_t gp,GM_ADDR workspace,GM_ADDR tiling){TPipe pipe;MoeCube::Gate<true,true> op(&pipe);op.Run(a,b,ids,c,m,h,i,k,e,tn,gp,tiling);}
