// Independent 128x128 group rotations; no block-diagonal KxK tensor.
#define ASCENDC_CUBE_ONLY
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
namespace {
class GroupedRotation {
 static constexpr auto Config=GetMDLConfig(false,false,0,false,false,false,true);
 using Operand=matmul::MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t,false>;
 using Matmul=matmul::MatmulImpl<Operand,Operand,Operand,Operand,Config>;
 TPipe* pipe; TCubeTiling tiling{}; Matmul mm;
 GlobalTensor<bfloat16_t> x,r,y;
public:
 __aicore__ inline GroupedRotation(TPipe* p):pipe(p){}
 __aicore__ inline void Run(GM_ADDR xp,GM_ADDR rp,GM_ADDR yp,uint32_t m,uint32_t k,GM_ADDR tp) {
  if ASCEND_IS_AIV {return;}
  x.SetGlobalBuffer((__gm__ bfloat16_t*)xp);r.SetGlobalBuffer((__gm__ bfloat16_t*)rp);y.SetGlobalBuffer((__gm__ bfloat16_t*)yp);
  auto* dst=reinterpret_cast<int32_t*>(&tiling);auto* src=reinterpret_cast<__gm__ int32_t*>(tp);
  for(uint32_t i=0;i<(sizeof(TCubeTiling)+3)/4;++i)dst[i]=src[i];
  mm.SetSubBlockIdx(0);mm.Init(&tiling,pipe);
  const uint32_t groups=k/128,mt=(m+15)/16;
  for(uint32_t task=GetBlockIdx();task<groups*mt;task+=GetBlockNum()) {
   uint32_t group=task/mt,row=(task%mt)*16,rows=(m-row<16?m-row:16);
   mm.SetOrgShape(m,128,k,128,k);
   mm.SetSingleShape(rows,128,128);
   mm.SetTensorA(x[row*k+group*128],false);
   mm.SetTensorB(r[group*16384],false);
   mm.IterateAll(y[row*k+group*128],0);
   PipeBarrier<PIPE_ALL>();SetAtomicNone();
  }
  mm.End();
 }
};
}
extern "C" __global__ __aicore__ void splitquant_rotate_bf16(GM_ADDR x,GM_ADDR r,GM_ADDR y,uint32_t m,uint32_t k,GM_ADDR workspace,GM_ADDR tiling) {
 TPipe pipe;GroupedRotation op(&pipe);op.Run(x,r,y,m,k,tiling);
}
