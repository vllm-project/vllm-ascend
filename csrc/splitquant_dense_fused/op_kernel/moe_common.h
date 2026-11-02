// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "kernel_operator.h"
using namespace AscendC;
namespace Moe {
template<HardEvent E> __aicore__ inline void Fence(){auto e=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));SetFlag<E>(e);WaitFlag<E>(e);}
__aicore__ inline float RoundBF16Bits(float x){union Bits{float f;uint32_t u;} v;v.f=x;v.u=(v.u+0x7fffU+((v.u>>16)&1U))&0xffff0000U;return v.f;}
template<class T> __aicore__ inline float Round(float x,LocalTensor<float> z,LocalTensor<T> t,bool scalarRound=false){if(scalarRound){union Bits{float f;uint32_t u;} v;v.f=x;if((v.u&0x7f800000U)!=0x7f800000U)return RoundBF16Bits(x);}
 Duplicate(z,x,16);PipeBarrier<PIPE_V>();Cast(t,z,RoundMode::CAST_RINT,16);PipeBarrier<PIPE_V>();Cast(z,t,RoundMode::CAST_NONE,16);PipeBarrier<PIPE_ALL>();return z.GetValue(0);}
// f survives reductions and dtype-rounding scratch. z becomes integer-valued F32.
template<class T> __aicore__ inline float Quant(LocalTensor<float> f,LocalTensor<float> z,LocalTensor<float> backup,LocalTensor<float> red,LocalTensor<float> scalar,LocalTensor<T> t,LocalTensor<int32_t> codes,float cp,float cn,bool scalarRound=false){
 Adds(backup,f,0.0f,128);PipeBarrier<PIPE_V>();ReduceMax(scalar,backup,red,128);PipeBarrier<PIPE_ALL>();float hi=scalar.GetValue(0);if(hi<0)hi=0;
 Adds(backup,f,0.0f,128);PipeBarrier<PIPE_V>();ReduceMin(scalar,backup,red,128);PipeBarrier<PIPE_ALL>();float lo=scalar.GetValue(0);if(lo>0)lo=0;
 cp=Round<T>(cp,z,t,scalarRound);cn=Round<T>(cn,z,t,scalarRound);hi=Round<T>(hi*cp,z,t,scalarRound);lo=Round<T>(lo*cn,z,t,scalarRound);
 float scale=Round<T>((hi>-lo?hi:-lo)/7.0f,z,t,scalarRound);if(scale==0)scale=1;
 Duplicate(z,scale,128);PipeBarrier<PIPE_V>();Div(z,f,z,128);PipeBarrier<PIPE_V>();Cast(t,z,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(z,t,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
 // Integer bounds commute with round, and clamping first prevents int32 overflow.
 Maxs(z,z,-8.0f,128);PipeBarrier<PIPE_V>();Mins(z,z,7.0f,128);PipeBarrier<PIPE_V>();Cast(codes,z,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(z,codes,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();return scale;
}
}
