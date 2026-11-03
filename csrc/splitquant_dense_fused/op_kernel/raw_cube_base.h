#pragma once
#include "kernel_operator.h"
namespace Raw {
using namespace AscendC;
class Dot64 {
 TPipe* pipe;
 TBuf<TPosition::A1> ab;TBuf<TPosition::B1> bb;
 TBuf<TPosition::A2> al;TBuf<TPosition::B2> bl;TBuf<TPosition::CO1> cl;
public:
 __aicore__ inline Dot64(TPipe* p):pipe(p){}
 __aicore__ inline void Init(){
  pipe->InitBuffer(ab,16384);pipe->InitBuffer(bb,65536);
  pipe->InitBuffer(al,16384);pipe->InitBuffer(bl,65536);pipe->InitBuffer(cl,4096);
 }
 __aicore__ inline void LoadA(const GlobalTensor<uint8_t>& a,uint64_t offset){
  Nd2NzParams params{};params.ndNum=1;params.nValue=16;params.dValue=1024;params.srcDValue=1024;params.dstNzC0Stride=16;params.dstNzNStride=1;
  DataCopy(ab.Get<uint8_t>(),a[offset],params);
  SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
  LoadData2DParams la{};la.repeatTimes=32;la.srcStride=1;
  LoadData(al.Get<int4b_t>(),ab.Get<int4b_t>(),la);
  SetFlag<HardEvent::MTE1_M>(EVENT_ID0);WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
 }
 __aicore__ inline void Run(const GlobalTensor<uint8_t>& b,uint64_t offset,uint32_t n,const GlobalTensor<int32_t>& c,uint64_t cOffset){
  DataCopyParams bd{1,2048,0,0};DataCopy(bb.Get<uint8_t>(),b[offset],bd);
  SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
  LoadData2DParams lb{};lb.repeatTimes=128;lb.srcStride=1;LoadData(bl.Get<int4b_t>(),bb.Get<int4b_t>(),lb);
  SetFlag<HardEvent::MTE1_M>(EVENT_ID0);WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
  MmadParams mm{};mm.m=16;mm.n=64;mm.k=2048;mm.cmatrixInitVal=true;
  Mmad(cl.Get<int32_t>(),al.Get<int4b_t>(),bl.Get<int4b_t>(),mm);
  SetFlag<HardEvent::M_MTE1>(EVENT_ID0);WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);PipeBarrier<PIPE_ALL>();
  SetFlag<HardEvent::M_FIX>(EVENT_ID0);WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
  FixpipeParamsV220 fp{};fp.nSize=64;fp.mSize=16;fp.srcStride=16;fp.dstStride=n;
  Fixpipe<int32_t,int32_t,CFG_ROW_MAJOR>(c[cOffset],cl.Get<int32_t>(),fp);
  SetFlag<HardEvent::FIX_M>(EVENT_ID0);WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
  SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
 }
};
}
