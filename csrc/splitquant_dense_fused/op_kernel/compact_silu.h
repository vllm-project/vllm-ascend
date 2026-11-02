// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
using namespace AscendC;
namespace Moe {
class CompactSilu {
 TBuf<TPosition::VECCALC> scratch;
public:
 __aicore__ inline void Init(TPipe& pipe){pipe.InitBuffer(scratch,8192);}
 __aicore__ inline void Eval(LocalTensor<uint16_t> out,LocalTensor<int16_t> input,LocalTensor<uint16_t> lut){
  auto code=scratch.Get<float>();auto mag=scratch.Get<float>()[128];auto idx=scratch.Get<float>()[256];auto a=scratch.Get<float>()[384];auto b=scratch.Get<float>()[512];auto direct=scratch.Get<float>()[640];auto off=scratch.Get<uint32_t>()[896];auto masks=scratch.Get<uint16_t>()[2176];
  auto sign=scratch.Get<uint8_t>()[4352];auto tiny=scratch.Get<uint8_t>()[4416];auto special=scratch.Get<uint8_t>()[4480];auto low=scratch.Get<uint8_t>()[4544];auto low2=scratch.Get<uint8_t>()[4608];auto high=scratch.Get<uint8_t>()[4672];auto high2=scratch.Get<uint8_t>()[4736];auto useDirect=scratch.Get<uint8_t>()[4800];
  Duplicate(masks,uint16_t(0),256);Cast(code,input,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();Cast(off.ReinterpretCast<int32_t>(),code,RoundMode::CAST_RINT,128);Duplicate(b.ReinterpretCast<uint32_t>(),uint32_t(65535),128);PipeBarrier<PIPE_V>();And(off.ReinterpretCast<uint16_t>(),off.ReinterpretCast<uint16_t>(),b.ReinterpretCast<uint16_t>(),256);PipeBarrier<PIPE_V>();Cast(code,off.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
  CompareScalar(sign,code,32768.0f,CMPMODE::GE,128);Adds(mag,code,-32768.0f,128);PipeBarrier<PIPE_V>();Select(mag,sign,mag,code,SELMODE::VSEL_TENSOR_TENSOR_MODE,128);PipeBarrier<PIPE_V>();
  Adds(idx,mag,-14976.0f,128);PipeBarrier<PIPE_V>();Adds(a,idx,2176.0f,128);PipeBarrier<PIPE_V>();Select(idx,sign,a,idx,SELMODE::VSEL_TENSOR_TENSOR_MODE,128);PipeBarrier<PIPE_V>();
  CompareScalar(tiny,mag,256.0f,CMPMODE::LT,128);Adds(a,mag,4352.0f,128);PipeBarrier<PIPE_V>();Adds(b,a,256.0f,128);PipeBarrier<PIPE_V>();Select(a,sign,b,a,SELMODE::VSEL_TENSOR_TENSOR_MODE,128);PipeBarrier<PIPE_V>();Select(idx,tiny,a,idx,SELMODE::VSEL_TENSOR_TENSOR_MODE,128);PipeBarrier<PIPE_V>();
  CompareScalar(special,mag,32640.0f,CMPMODE::GE,128);Adds(a,mag,-27776.0f,128);PipeBarrier<PIPE_V>();Adds(b,a,128.0f,128);PipeBarrier<PIPE_V>();Select(a,sign,b,a,SELMODE::VSEL_TENSOR_TENSOR_MODE,128);PipeBarrier<PIPE_V>();Select(idx,special,a,idx,SELMODE::VSEL_TENSOR_TENSOR_MODE,128);PipeBarrier<PIPE_V>();
  CompareScalar(low,mag,256.0f,CMPMODE::GE,128);CompareScalar(low2,mag,14976.0f,CMPMODE::LT,128);CompareScalar(high,mag,17152.0f,CMPMODE::GE,128);CompareScalar(high2,mag,32640.0f,CMPMODE::LT,128);PipeBarrier<PIPE_V>();And(low.ReinterpretCast<uint16_t>(),low.ReinterpretCast<uint16_t>(),low2.ReinterpretCast<uint16_t>(),32);And(high.ReinterpretCast<uint16_t>(),high.ReinterpretCast<uint16_t>(),high2.ReinterpretCast<uint16_t>(),32);PipeBarrier<PIPE_V>();Or(useDirect.ReinterpretCast<uint16_t>(),low.ReinterpretCast<uint16_t>(),high.ReinterpretCast<uint16_t>(),32);Duplicate(a,0.0f,128);PipeBarrier<PIPE_V>();Select(idx,useDirect,a,idx,SELMODE::VSEL_TENSOR_TENSOR_MODE,128);PipeBarrier<PIPE_V>();
  Cast(off.ReinterpretCast<int32_t>(),idx,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();ShiftLeft(off,off,uint32_t(1),128);PipeBarrier<PIPE_V>();
  Adds(direct,code,-128.0f,128);PipeBarrier<PIPE_V>();Adds(a,direct,-65536.0f,128);PipeBarrier<PIPE_V>();Select(direct,sign,a,direct,SELMODE::VSEL_TENSOR_TENSOR_MODE,128);Duplicate(b,-32768.0f,128);PipeBarrier<PIPE_V>();Select(a,sign,b,code,SELMODE::VSEL_TENSOR_TENSOR_MODE,128);PipeBarrier<PIPE_V>();Select(direct,high,a,direct,SELMODE::VSEL_TENSOR_TENSOR_MODE,128);PipeBarrier<PIPE_V>();Maxs(direct,direct,-32768.0f,128);PipeBarrier<PIPE_V>();Mins(direct,direct,32767.0f,128);PipeBarrier<PIPE_V>();
  Gather(out,lut,off,uint32_t(0),uint32_t(128));PipeBarrier<PIPE_ALL>();Cast(code,out.ReinterpretCast<int16_t>(),RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();Select(code,useDirect,direct,code,SELMODE::VSEL_TENSOR_TENSOR_MODE,128);PipeBarrier<PIPE_V>();Cast(out.ReinterpretCast<int16_t>(),code,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_ALL>();
 }
};
}
