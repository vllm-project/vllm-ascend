// BSD-3-Clause. Reuse all 16 physical Cube rows for separate exact INT4 groups.
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "constant_cube_config.h"
using namespace AscendC;
namespace {
template<bool NzWeight=false,bool Fixed=false> class SparseCube {
 static constexpr auto Config=GetMDLConfig(false,false,0,false,false,false,true);
 using A=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int4b_t,false>;
 using B=matmul::MatmulType<TPosition::GM,(NzWeight?CubeFormat::NZ:CubeFormat::ND),int4b_t,!NzWeight>;
 using C=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int32_t,false>;
 TPipe* pipe;TCubeTiling tile{};static constexpr auto& Mode=Const::Mode<Fixed,A,B,C,256,128,256>::Value;matmul::MatmulImpl<A,B,C,C,Mode> mm;
public:
 __aicore__ inline SparseCube(TPipe* p):pipe(p){}
 __aicore__ inline void Run(GM_ADDR ap,GM_ADDR wp,GM_ADDR cp,uint32_t n,uint32_t g,uint32_t p,uint32_t n0,uint32_t n1,uint32_t tn,GM_ADDR tp){
  if ASCEND_IS_AIV {return;}
  GlobalTensor<int4b_t>a,w;GlobalTensor<int32_t>c;
  a.SetGlobalBuffer((__gm__ int4b_t*)ap);w.SetGlobalBuffer((__gm__ int4b_t*)wp);c.SetGlobalBuffer((__gm__ int32_t*)cp);
  auto src=(__gm__ int32_t*)tp;auto dst=(int32_t*)&tile;
  for(uint32_t j=0;j<(sizeof(TCubeTiling)+3)/4;++j)dst[j]=src[j];
  mm.SetSubBlockIdx(0);mm.Init(&tile,pipe);
  uint32_t chunks=(g+15)/16,t0=(n0+tn-1)/tn,t1=(n1+tn-1)/tn,t2=(n-n0-n1+tn-1)/tn;
  if(p==1){t0=(n+tn-1)/tn;t1=0;t2=0;}
  uint32_t nt=t0+t1+t2;
  if constexpr(Fixed)mm.SetOrgShape(16,n,2048,g*128,n);
  for(uint32_t task=GetBlockIdx();task<chunks*nt;task+=GetBlockNum()){
   uint32_t ch=task/nt,ni=task%nt,proj,begin,end;
   if(p==1){proj=0;begin=ni*tn;end=n;}
   else if(ni<t0){proj=0;begin=ni*tn;end=n0;}
   else if(ni<t0+t1){proj=1;begin=n0+(ni-t0)*tn;end=n0+n1;}
   else{proj=2;begin=n0+n1+(ni-t0-t1)*tn;end=n;}
   uint32_t rows=(g-ch*16<16?g-ch*16:16),vn=(end-begin<tn?end-begin:tn);
   if constexpr(!Fixed){mm.SetOrgShape(16,n,2048,g*128,n);mm.SetSingleShape(rows,vn,rows*128);}
   mm.SetTensorA(a[(proj*chunks+ch)*16*2048],false);
   mm.SetTensorB(w[uint64_t(begin)*g*128+uint64_t(ch)*2048*(NzWeight?64:1)],!NzWeight);
   mm.IterateAll(c[ch*16*n+begin],0);PipeBarrier<PIPE_ALL>();SetAtomicNone();
  }
  mm.End();
 }
};
}
extern "C" __global__ __aicore__ void splitquant_sparse_cube(GM_ADDR a,GM_ADDR w,GM_ADDR c,uint32_t n,uint32_t g,uint32_t p,uint32_t n0,uint32_t n1,uint32_t tn,GM_ADDR workspace,GM_ADDR tiling){TPipe pipe;SparseCube<> op(&pipe);op.Run(a,w,c,n,g,p,n0,n1,tn,tiling);}

extern "C" __global__ __aicore__ void splitquant_sparse_cube_nz(GM_ADDR a,GM_ADDR w,GM_ADDR c,uint32_t n,uint32_t g,uint32_t p,uint32_t n0,uint32_t n1,uint32_t tn,GM_ADDR workspace,GM_ADDR tiling){TPipe pipe;SparseCube<true> op(&pipe);op.Run(a,w,c,n,g,p,n0,n1,tn,tiling);}

extern "C" __global__ __aicore__ void splitquant_sparse_cube_const_nz(GM_ADDR a,GM_ADDR w,GM_ADDR c,uint32_t n,uint32_t g,uint32_t p,uint32_t n0,uint32_t n1,uint32_t tn,GM_ADDR workspace,GM_ADDR tiling){TPipe pipe;SparseCube<true,true> op(&pipe);op.Run(a,w,c,n,g,p,n0,n1,tn,tiling);}
