#include "compact_silu.h"
extern "C" __global__ __aicore__ void moe_compact_silu_debug(GM_ADDR xp,GM_ADDR lp,GM_ADDR yp,uint32_t n){
 TPipe pipe;TBuf<TPosition::VECCALC> data,table;pipe.InitBuffer(data,256);pipe.InitBuffer(table,10240);Moe::CompactSilu op;op.Init(pipe);auto x=data.Get<bfloat16_t>();auto lut=table.Get<uint16_t>();GlobalTensor<bfloat16_t> in,out;GlobalTensor<uint16_t> gmLut;in.SetGlobalBuffer((__gm__ bfloat16_t*)xp);out.SetGlobalBuffer((__gm__ bfloat16_t*)yp);gmLut.SetGlobalBuffer((__gm__ uint16_t*)lp);DataCopyPad(lut,gmLut,{1,10240,0,0,0},{false,0,0,0});PipeBarrier<PIPE_ALL>();
 for(uint32_t task=GetBlockIdx();task<(n+127)/128;task+=GetBlockNum()){uint32_t off=task*128,len=(n-off<128?n-off:128);DataCopyPad(x,in[off],{1,len*2,0,0,0},{false,0,0,0});PipeBarrier<PIPE_ALL>();op.Eval(x.ReinterpretCast<uint16_t>(),x.ReinterpretCast<int16_t>(),lut);DataCopyPad(out[off],x,{1,len*2,0,0,0});PipeBarrier<PIPE_ALL>();}
}
