// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
using namespace AscendC;
namespace Moe {
class PagedSilu {
 TBuf<TPosition::VECCALC> scratch;
public:
 __aicore__ inline void Init(TPipe& pipe){pipe.InitBuffer(scratch,3328);}
 __aicore__ inline void Eval(LocalTensor<uint16_t> out,LocalTensor<int16_t> input,LocalTensor<uint16_t> lut){
  auto f=scratch.Get<float>();auto code=scratch.Get<uint32_t>()[128];auto page=scratch.Get<uint32_t>()[256];auto low=scratch.Get<uint32_t>()[384];auto off=scratch.Get<uint32_t>()[512];auto mask=scratch.Get<uint32_t>()[640];auto base=scratch.Get<int16_t>()[1536];
  Duplicate(mask,uint32_t(65535),128);Cast(f,input,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
  Cast(code.ReinterpretCast<int32_t>(),f,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();
  And(code.ReinterpretCast<uint16_t>(),code.ReinterpretCast<uint16_t>(),mask.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();
  ShiftRight(page,code,uint32_t(7),128);Duplicate(mask,uint32_t(127),128);PipeBarrier<PIPE_V>();
  And(low.ReinterpretCast<uint16_t>(),code.ReinterpretCast<uint16_t>(),mask.ReinterpretCast<uint16_t>(),256);ShiftLeft(page,page,uint32_t(2),128);PipeBarrier<PIPE_V>();
  Gather(off,lut.ReinterpretCast<uint32_t>(),page,uint32_t(0),uint32_t(128));PipeBarrier<PIPE_V>();
  ShiftLeft(low,low,uint32_t(1),128);ShiftRight(page,page,uint32_t(1),128);PipeBarrier<PIPE_V>();
  Gather(base.ReinterpretCast<uint16_t>(),lut[1024],page,uint32_t(0),uint32_t(128));Add(off.ReinterpretCast<int32_t>(),off.ReinterpretCast<int32_t>(),low.ReinterpretCast<int32_t>(),128);PipeBarrier<PIPE_V>();
  Gather(out,lut[1536],off,uint32_t(0),uint32_t(128));PipeBarrier<PIPE_V>();
  Add(out.ReinterpretCast<int16_t>(),out.ReinterpretCast<int16_t>(),base,128);PipeBarrier<PIPE_ALL>();
 }
};
}
