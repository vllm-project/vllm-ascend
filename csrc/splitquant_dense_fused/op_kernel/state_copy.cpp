// BSD-3-Clause. Raw contiguous-row copy; no view materialization or arithmetic.
#include "kernel_operator.h"
using namespace AscendC;
namespace {
template<HardEvent E> __aicore__ inline void Sync(){auto e=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));SetFlag<E>(e);WaitFlag<E>(e);}
}
extern "C" __global__ __aicore__ void splitquant_state_copy(GM_ADDR src,GM_ADDR dst,uint64_t bytes,uint64_t source_offset,uint64_t dest_offset){
 TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,16384);GlobalTensor<uint8_t>s,d;
 s.SetGlobalBuffer((__gm__ uint8_t*)src);d.SetGlobalBuffer((__gm__ uint8_t*)dst);
 auto b=buf.Get<uint8_t>();uint64_t tiles=(bytes+8191)/8192;
 for(uint64_t tile=GetBlockIdx();tile<tiles;tile+=GetBlockNum()){
  uint64_t offset=tile*8192;uint32_t len=(bytes-offset<8192?bytes-offset:8192);
  auto local=b[(tile%2)*8192];
  DataCopyPad(local,s[source_offset+offset],{1,len,0,0,0},{false,0,0,0});Sync<HardEvent::MTE2_MTE3>();
  DataCopyPad(d[dest_offset+offset],local,{1,len,0,0,0});Sync<HardEvent::MTE3_MTE2>();
 }
}

extern "C" __global__ __aicore__ void splitquant_state_index_copy(GM_ADDR src,GM_ADDR dst,GM_ADDR indices,uint64_t bytes,uint64_t stride,uint64_t rows,uint64_t batches,uint32_t write){
 TPipe pipe;TBuf<TPosition::VECCALC> buf;pipe.InitBuffer(buf,16384);GlobalTensor<uint8_t>s,d;
 s.SetGlobalBuffer((__gm__ uint8_t*)src);d.SetGlobalBuffer((__gm__ uint8_t*)dst);
 auto b=buf.Get<uint8_t>();uint64_t tiles=(bytes+8191)/8192;
 for(uint64_t task=GetBlockIdx();task<batches*tiles;task+=GetBlockNum()){
  uint64_t batch=task/tiles,offset=(task%tiles)*8192;int64_t row=((__gm__ int64_t*)indices)[batch];if(row<0||uint64_t(row)>=rows)continue;
  uint32_t len=(bytes-offset<8192?bytes-offset:8192);auto local=b[(task%2)*8192];
  uint64_t so=(write?batch*bytes:uint64_t(row)*stride)+offset,doff=(write?uint64_t(row)*stride:batch*bytes)+offset;
  DataCopyPad(local,s[so],{1,len,0,0,0},{false,0,0,0});Sync<HardEvent::MTE2_MTE3>();
  DataCopyPad(d[doff],local,{1,len,0,0,0});Sync<HardEvent::MTE3_MTE2>();
 }
}
