// SPDX-License-Identifier: BSD-3-Clause
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
template<HardEvent E> __aicore__ inline void Sync(){auto id=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));SetFlag<E>(id);WaitFlag<E>(id);}
#include "mixed_epi.h"
namespace Mixed {
class Cube {
 static constexpr auto Config=GetMDLConfig(false,false,0,false,false,false,true);
 using A=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int4b_t,false>;
 using B=matmul::MatmulType<TPosition::GM,CubeFormat::NZ,int4b_t,false>;
 using C=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int32_t,false>;
 TPipe& pipe;TCubeTiling tile{};matmul::MatmulImpl<A,B,C,C,Config> mm;
public:
 __aicore__ inline Cube(TPipe& p):pipe(p){}
 __aicore__ inline void Run(GM_ADDR ap,GM_ADDR wp,GM_ADDR cp,uint32_t n,uint32_t g,uint32_t projections,uint32_t n0,uint32_t n1,uint32_t tn,uint32_t cores,GM_ADDR tp){
  GlobalTensor<int4b_t>a,w;GlobalTensor<int32_t>c;a.SetGlobalBuffer((__gm__ int4b_t*)ap);w.SetGlobalBuffer((__gm__ int4b_t*)wp);c.SetGlobalBuffer((__gm__ int32_t*)cp);
  auto src=(__gm__ int32_t*)tp;auto dst=(int32_t*)&tile;for(uint32_t j=0;j<(sizeof(TCubeTiling)+3)/4;++j)dst[j]=src[j];mm.SetSubBlockIdx(0);mm.Init(&tile,&pipe);
  uint32_t chunks=(g+15)/16,t0=(n0+tn-1)/tn,t1=(n1+tn-1)/tn,t2=(n-n0-n1+tn-1)/tn;if(projections==1){t0=(n+tn-1)/tn;t1=t2=0;}uint32_t nt=t0+t1+t2;
  for(uint32_t task=GetBlockIdx();task<nt;task+=cores){
   uint32_t proj,begin,end;
   if(projections==1){proj=0;begin=task*tn;end=n;}else if(task<t0){proj=0;begin=task*tn;end=n0;}else if(task<t0+t1){proj=1;begin=n0+(task-t0)*tn;end=n0+n1;}else{proj=2;begin=n0+n1+(task-t0-t1)*tn;end=n;}
   CrossCoreWaitFlag(9);uint32_t vn=(end-begin<tn?end-begin:tn);
   for(uint32_t ch=0;ch<chunks;++ch){
    uint32_t rows=(g-ch*16<16?g-ch*16:16);mm.SetOrgShape(16,n,2048,g*128,n);mm.SetSingleShape(rows,vn,rows*128);mm.SetTensorA(a[(proj*chunks+ch)*16*2048],false);mm.SetTensorB(w[uint64_t(begin)*g*128+uint64_t(ch)*2048*64],false);mm.IterateAll(c[ch*16*n+begin],0);PipeBarrier<PIPE_ALL>();SetAtomicNone();
   }
   CrossCoreSetFlag<2,PIPE_FIX>(8);
  }
  CrossCoreWaitFlag(9);mm.End();
 }
};
}
extern "C" __global__ __aicore__ void splitquant_sparse_mix_bf16(GM_ADDR a,GM_ADDR w,GM_ADDR partial,GM_ADDR ws,GM_ADDR s,GM_ADDR b,GM_ADDR y,uint32_t n,uint32_t g,uint32_t p,uint32_t n0,uint32_t n1,uint32_t tn,uint32_t cores,uint32_t hb,GM_ADDR workspace,GM_ADDR tiling){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 TPipe pipe;
 if ASCEND_IS_AIC {Mixed::Cube op(pipe);op.Run(a,w,partial,n,g,p,n0,n1,tn,cores,tiling);}
 if ASCEND_IS_AIV {Mixed::Epilogue<bfloat16_t> op(pipe);op.Run(partial,ws,ws,s,s,b,y,1,n,g,hb,p,n0,n1,tn,cores);}
}
