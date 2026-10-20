// SPDX-License-Identifier: Apache-2.0
#include "mla_serial_q_finalize_rope.h"
using namespace AscendC;
constexpr uint32_t KEY_DIM=512;
constexpr uint32_t KEY_TILE_HEADS=128;
constexpr uint32_t KEY_ELEMENTS=KEY_TILE_HEADS*KEY_DIM;
class MlaFinalizeStage {
public:
    __aicore__ inline explicit MlaFinalizeStage(TPipe& pipe):pipe_(pipe),rope_(pipe) {}
    __aicore__ inline void Init()
    {
        pipe_.InitBuffer(key_,1,KEY_ELEMENTS*sizeof(bfloat16_t));
        rope_.Init();
    }
    __aicore__ inline void CopyKey(GM_ADDR key,GM_ADDR out,uint32_t token,uint32_t head,
                                   uint32_t tokenRows,uint32_t headRows,uint32_t tokens,uint32_t heads)
    {
        GlobalTensor<bfloat16_t> source,target;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(key));
        target.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out));
        auto x=key_.AllocTensor<bfloat16_t>();
        DataCopyPad(x,source[(uint64_t(head)*tokens+token)*KEY_DIM],
                    DataCopyExtParams{static_cast<uint16_t>(headRows),tokenRows*KEY_DIM*2,(tokens-tokenRows)*KEY_DIM*2,0,0},
                    DataCopyPadExtParams<bfloat16_t>{false,0,0,0});
        key_.EnQue(x); x=key_.DeQue<bfloat16_t>();
        // UB is [head,token,512]. UB stride uses32B blocks; GM stride uses bytes.
        for(uint32_t row=0;row<tokenRows;++row)
            DataCopyPad(target[(uint64_t(token+row)*heads+head)*KEY_DIM],x[row*KEY_DIM],
                        DataCopyExtParams{static_cast<uint16_t>(headRows),KEY_DIM*2,
                                          (tokenRows-1)*KEY_DIM*2/32,0,0});
        key_.FreeTensor(x);
    }
    __aicore__ inline void Rope(GM_ADDR up,GM_ADDR cos,GM_ADDR sin,GM_ADDR out,
                                uint32_t token,uint32_t head,uint32_t rows,uint32_t heads,uint32_t nope)
    {
        for(uint32_t done=0;done<rows;done+=MAX_ROPE_ROWS) {
            const uint32_t count=rows-done<MAX_ROPE_ROWS?rows-done:MAX_ROPE_ROWS;
            const uint64_t cell=uint64_t(token)*heads+head+done;
            rope_.Rope(up,cell*(nope+ROPE_DIM)+nope,cos,sin,token,out,cell*ROPE_DIM,
                        count,nope+ROPE_DIM,ROPE_DIM);
        }
    }
private:
    TPipe& pipe_;
    // Pure key movement has no Vector arithmetic; one bound buffer serves MTE2/MTE3.
    TQueBind<TPosition::VECIN,TPosition::VECOUT,1> key_;
    MlaFinalizeRope rope_;
};
extern "C" __global__ __aicore__ void mla_serial_q_finalize(
    GM_ADDR up,GM_ADDR key,GM_ADDR cos,GM_ADDR sin,GM_ADDR qNope,GM_ADDR qRope,GM_ADDR workspace,GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(t,tiling);
    TPipe pipe;
    MlaFinalizeStage v(pipe); v.Init();
    const uint32_t headGroups=(t.heads+t.tileHeads-1)/t.tileHeads;
    const uint32_t tokenGroups=(t.tokens+t.tileTokens-1)/t.tileTokens;
    for(uint32_t task=GetBlockIdx();task<tokenGroups*headGroups;task+=GetBlockNum()) {
        const uint32_t token=(task/headGroups)*t.tileTokens,head=(task%headGroups)*t.tileHeads;
        const uint32_t tokenRows=t.tokens-token<t.tileTokens?t.tokens-token:t.tileTokens;
        const uint32_t headRows=t.heads-head<t.tileHeads?t.heads-head:t.tileHeads;
        v.CopyKey(key,qNope,token,head,tokenRows,headRows,t.tokens,t.heads);
        for(uint32_t row=0;row<tokenRows;++row)
            v.Rope(up,t.useRope?cos:nullptr,t.useRope?sin:nullptr,qRope,
                    token+row,head,headRows,t.heads,t.nope);
    }
}
