// BSD-3-Clause. Exact INT4 group-as-M packing, one independent sparse row per task.
#include "kernel_operator.h"
using namespace AscendC;
namespace {
template<HardEvent E> __aicore__ inline void Sync(){auto e=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));SetFlag<E>(e);WaitFlag<E>(e);}
}
extern "C" __global__ __aicore__ void splitquant_sparse_pack(GM_ADDR ap,GM_ADDR sp,uint32_t groups,uint32_t projections){
 TPipe pipe;TBuf<TPosition::VECCALC> rowbuf;pipe.InitBuffer(rowbuf,1024);
 auto row=rowbuf.Get<uint16_t>();GlobalTensor<uint16_t>a,s;
 a.SetGlobalBuffer((__gm__ uint16_t*)ap);s.SetGlobalBuffer((__gm__ uint16_t*)sp);
 uint32_t chunks=(groups+15)/16;
 for(uint32_t task=GetBlockIdx();task<groups*projections;task+=GetBlockNum()){
  uint32_t p=task/groups,g=task%groups;
  Duplicate(row,static_cast<uint16_t>(0),512);Sync<HardEvent::V_MTE2>();
  DataCopyPad(row[(g%16)*32],a[(g*projections+p)*32],{1,64,0,0,0},{false,0,0,0});Sync<HardEvent::MTE2_MTE3>();
  DataCopyPad(s[((p*chunks+g/16)*16+g%16)*512],row,{1,1024,0,0,0});
  Sync<HardEvent::MTE3_V>();Sync<HardEvent::MTE3_MTE2>();
 }
}
