// BSD-3-Clause. Joint scheduling, independent activation quantization per projection.
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
namespace {
class MultiCube {
 static constexpr auto Config=GetMDLConfig(false,false,0,false,false,false,true);
 using A=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int4b_t,false>;
 using B=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int4b_t,true>;
 using C=matmul::MatmulType<TPosition::GM,CubeFormat::ND,int32_t,false>;
 TPipe pipe;TCubeTiling tile{};matmul::MatmulImpl<A,B,C,C,Config> mm;
public:
 __aicore__ inline void Run(GM_ADDR ap,GM_ADDR wp,GM_ADDR cp,uint32_t m,uint32_t n,uint32_t g,uint32_t projections,uint32_t n0,uint32_t n1,GM_ADDR tp){
  if ASCEND_IS_AIV {return;}
  GlobalTensor<int4b_t> a,w;GlobalTensor<int32_t> c;
  a.SetGlobalBuffer((__gm__ int4b_t*)ap);w.SetGlobalBuffer((__gm__ int4b_t*)wp);c.SetGlobalBuffer((__gm__ int32_t*)cp);
  auto src=(__gm__ int32_t*)tp;auto dst=(int32_t*)&tile;
  for(uint32_t j=0;j<(sizeof(TCubeTiling)+3)/4;++j)dst[j]=src[j];
  mm.SetSubBlockIdx(0);mm.Init(&tile,&pipe);
  uint32_t t0=(n0+1023)/1024,t1=(n1+1023)/1024,t2=(n-n0-n1+1023)/1024,nt=t0+t1+t2,mt=(m+15)/16;
  for(uint32_t task=GetBlockIdx();task<g*mt*nt;task+=GetBlockNum()){
   uint32_t group=task/(mt*nt),mi=(task%(mt*nt))/nt*16,ni=task%nt;
   uint32_t projection,begin,end;
   if(ni<t0){projection=0;begin=ni*1024;end=n0;}
   else if(ni<t0+t1){projection=1;begin=n0+(ni-t0)*1024;end=n0+n1;}
   else {projection=2;begin=n0+n1+(ni-t0-t1)*1024;end=n;}
   uint32_t vm=(m-mi<16?m-mi:16),vn=(end-begin<1024?end-begin:1024);
   mm.SetOrgShape(projections*m,n,128,128,n);mm.SetSingleShape(vm,vn,128);
   mm.SetTensorA(a[(group*projections*m+projection*m+mi)*128],false);
   mm.SetTensorB(w[(group*n+begin)*128],true);
   mm.IterateAll(c[(group*m+mi)*n+begin],0);
   PipeBarrier<PIPE_ALL>();SetAtomicNone();
  }
  mm.End();
 }
};
}
extern "C" __global__ __aicore__ void splitquant_multi_cube(GM_ADDR a,GM_ADDR w,GM_ADDR c,uint32_t m,uint32_t n,uint32_t g,uint32_t projections,uint32_t n0,uint32_t n1,GM_ADDR workspace,GM_ADDR tiling){MultiCube op;op.Run(a,w,c,m,n,g,projections,n0,n1,tiling);}
