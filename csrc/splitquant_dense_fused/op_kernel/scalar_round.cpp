#include "moe_common.h"
extern "C" __global__ __aicore__ void moe_scalar_round_debug(GM_ADDR xp,GM_ADDR yp,uint32_t n){
 TPipe pipe;TBuf<TPosition::VECCALC> data,output;pipe.InitBuffer(data,512);pipe.InitBuffer(output,512);auto x=data.Get<float>();auto y=output.Get<float>();GlobalTensor<float> in,out;in.SetGlobalBuffer((__gm__ float*)xp);out.SetGlobalBuffer((__gm__ float*)yp);
 for(uint32_t task=GetBlockIdx();task<(n+127)/128;task+=GetBlockNum()){uint32_t off=task*128,len=(n-off<128?n-off:128);DataCopyPad(x,in[off],{1,len*4,0,0,0},{false,0,0,0});PipeBarrier<PIPE_ALL>();for(uint32_t j=0;j<len;++j)y.SetValue(j,Moe::RoundBF16Bits(x.GetValue(j)));PipeBarrier<PIPE_ALL>();DataCopyPad(out[off],y,{1,len*4,0,0,0});PipeBarrier<PIPE_ALL>();}
}
