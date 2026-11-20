# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen A3 MTP3 verify, with four-token DMA and vectorized gate preprocessing."""

import tilelang
from tilelang import language as T


def build_window(
    rows, pages, tokens, state_stride, replay_stride, index_stride, gate_stride, b_token_stride, v_token_stride
):
    tile = 128
    # Simultaneously live arrays get independent 32-byte-aligned UB offsets.
    arrays = [
        ("checkpoint_bf16", "bfloat16_t", tile * 128),
        ("checkpoint", "float", tile * 128),
        ("scratch", "float", tile * 128),
        ("query_bf16", "bfloat16_t", 512),
        ("key_bf16", "bfloat16_t", 512),
        ("value_bf16", "bfloat16_t", 4 * tile),
        ("queries", "float", 512),
        ("keys", "float", 512),
        ("values", "float", 4 * tile),
        ("outputs", "bfloat16_t", 4 * tile),
        ("a_bf16", "bfloat16_t", 128),
        ("b_bf16", "bfloat16_t", 128),
        ("a_fp32", "float", 128),
        ("b_fp32", "float", 128),
        ("bias_bf16", "bfloat16_t", 32),
        ("bias_fp32", "float", 32),
        ("a_log_fp32", "float", 32),
        ("gate_lanes", "float", 32),
        ("rounded", "bfloat16_t", 16),
        ("records", "float", 4 * 264),
        ("projection", "float", tile),
        ("outer", "float", tile * 8),
    ]
    size = 0
    definitions = []
    for name, typ, count in arrays:
        definitions.append(f"auto {name} = ascend_ub.GetWithOffset<{typ}>({count},{size});")
        size += (count * (2 if typ == "bfloat16_t" else 4) + 31) // 32 * 32
    assert size <= 196352, size
    cpp = _BODY
    values = {
        "@BUFFERS@": "\n".join(definitions),
        "@INDEX@": str(index_stride),
        "@STATE@": str(state_stride),
        "@REPLAY@": str(replay_stride),
        "@A@": str(gate_stride),
        "@B@": str(b_token_stride),
        "@V@": str(v_token_stride),
    }
    for tag, value in values.items():
        cpp = cpp.replace(tag, value)

    @T.prim_func
    def gdn_window_verify(
        log_a: T.Tensor((24,), "float32"),
        a: T.Tensor((tokens, 24), "bfloat16", strides=(gate_stride, 1)),
        b: T.Tensor((tokens, 24), "bfloat16", strides=(b_token_stride, 1)),
        bias: T.Tensor((24,), "bfloat16"),
        q: T.Tensor((tokens, 8, 128), "bfloat16"),
        k: T.Tensor((tokens, 8, 128), "bfloat16"),
        v: T.Tensor((tokens, 24, 128), "bfloat16", strides=(v_token_stride, 128, 1)),
        state: T.Tensor((1, 24, 128, 128), "bfloat16", strides=(state_stride, 16384, 128, 1)),
        replay: T.Tensor((1, 24, 4, 257), "float32", strides=(replay_stride, 1028, 257, 1)),
        starts: T.Tensor((rows + 1,), "int32"),
        slots: T.Tensor((rows,), "int32", strides=(index_stride,)),
        out: T.Tensor((tokens, 24, 128), "bfloat16"),
    ):
        with T.Kernel(rows * 12, is_npu=True) as (core, sub):  # noqa: SIM117 - TileLang parses nested scopes
            with T.Scope("V"):
                # Retain physical stride parameters in the TileLang parser's closure.
                T.evaluate(
                    tokens + state_stride + replay_stride + index_stride + gate_stride + b_token_stride + v_token_stride
                )
                T._src_code(cpp)

    return tilelang.compile(
        gdn_window_verify,
        target="ascendc",
        platform="A3",
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        },
    )


_BODY = r"""
{
using namespace AscendC;
constexpr int tile = 128;
const int job = (GetBlockIdx()/2)*2 + GetSubBlockIdx();
const int row = job / 24;
const int head = job % 24;

const int slot=slots.GetValue(row*@INDEX@), begin=starts.GetValue(row), count=starts.GetValue(row+1)-begin;
@BUFFERS@
#define DEEP_G2U(buf,src,n,width,stride,Tp) do { \
  DataCopyExtParams p={uint16_t(n),uint32_t((width)*sizeof(Tp)),uint32_t(((stride)-(width))*sizeof(Tp)),0,0}; \
  DataCopyPadExtParams<Tp> pad={false,0,0,0}; DataCopyPad((buf),(src),p,pad); } while(0)
#define DEEP_U2G(dst,buf,n,width,stride,Tp) do { \
  DataCopyExtParams p={uint16_t(n),uint32_t((width)*sizeof(Tp)),0,uint32_t(((stride)-(width))*sizeof(Tp)),0}; \
  DataCopyPad((dst),(buf),p); } while(0)
#define DEEP_SYNC(event) do { SetFlag<HardEvent::event>(0); WaitFlag<HardEvent::event>(0); } while(0)
#define DEEP_VEC() PipeBarrier<PIPE_V>()
if (count>0 && slot>0) {
    state.SetGlobalBuffer((__gm__ bfloat16_t*)state_handle+((int64_t)slot)*@STATE@LL);
    replay.SetGlobalBuffer((__gm__ float*)replay_handle+((int64_t)slot)*@REPLAY@LL);
    // DMA once per row/head; MTE pads 24 BF16 gates to 32 elements per UB row.
    DEEP_G2U(checkpoint_bf16,state[head*16384],1,tile*128,tile*128,bfloat16_t);
    DEEP_G2U(query_bf16,q[begin*1024+(head/3)*128],count,128,1024,bfloat16_t);
    DEEP_G2U(key_bf16,k[begin*1024+(head/3)*128],count,128,1024,bfloat16_t);
    DEEP_G2U(value_bf16,v[begin*@V@+head*128],count,tile,@V@,bfloat16_t);
    DEEP_G2U(a_bf16,a[begin*@A@],count,24,@A@,bfloat16_t);
    DEEP_G2U(b_bf16,b[begin*@B@],count,24,@B@,bfloat16_t);
    DEEP_G2U(bias_bf16,bias[0],1,24,24,bfloat16_t);
    DEEP_G2U(a_log_fp32,log_a[0],1,24,24,float);
    DEEP_SYNC(MTE2_V);
    Cast(checkpoint,checkpoint_bf16,RoundMode::CAST_NONE,tile*128);
    Cast(queries,query_bf16,RoundMode::CAST_NONE,count*128);
    Cast(keys,key_bf16,RoundMode::CAST_NONE,count*128);
    Cast(values,value_bf16,RoundMode::CAST_NONE,count*tile);
    Cast(a_fp32,a_bf16,RoundMode::CAST_NONE,count*32);
    Cast(b_fp32,b_bf16,RoundMode::CAST_NONE,count*32);
    Cast(bias_fp32,bias_bf16,RoundMode::CAST_NONE,24);
    DEEP_VEC(); Muls(queries,queries,0x1.6a09e6p-4f,count*128);
    DEEP_SYNC(V_S);
    // A_log is shared across the entire window. The reciprocal stays scalar and beta rounds to BF16.
    float xs[4],dec[4],bet[4];
    const float bias_value=bias_fp32.GetValue(head),log_value=a_log_fp32.GetValue(head);
    for(int st=0;st<count;++st) {
        xs[st]=a_fp32.GetValue(st*32+head)+bias_value;
        gate_lanes.SetValue(st,xs[st]); gate_lanes.SetValue(8+st,-b_fp32.GetValue(st*32+head));
    }
    gate_lanes.SetValue(4,log_value);
    DEEP_SYNC(S_V); Exp(gate_lanes,gate_lanes,16); DEEP_SYNC(V_S);
    const float exp_a=gate_lanes.GetValue(4);
    for(int st=0;st<count;++st) {
        bet[st]=1.0f/(gate_lanes.GetValue(8+st)+1.0f);
        gate_lanes.SetValue(st,gate_lanes.GetValue(st)+1.0f);
    }
    DEEP_SYNC(S_V); Ln(gate_lanes,gate_lanes,8); DEEP_SYNC(V_S);
    for(int st=0;st<count;++st) gate_lanes.SetValue(st,-exp_a*(xs[st]<=20.0f ? gate_lanes.GetValue(st) : xs[st]));
    DEEP_SYNC(S_V); Exp(gate_lanes,gate_lanes,8); DEEP_SYNC(V_S);
    for(int st=0;st<count;++st) {
        dec[st]=gate_lanes.GetValue(st); gate_lanes.SetValue(8+st,bet[st]); records.SetValue(st*264+256,dec[st]);
    }
    DEEP_SYNC(S_V);
    Cast(rounded,gate_lanes[8],RoundMode::CAST_RINT,8);
    DEEP_VEC();
    Cast(gate_lanes[8],rounded,RoundMode::CAST_NONE,8); DEEP_SYNC(V_S);
    for(int st=0;st<count;++st) bet[st]=gate_lanes.GetValue(8+st);
    // Retain the original keys-half reduction tree and separate FP32 mul/add updates.
    const BinaryRepeatParams broadcast(1,1,1,16,16,0),sum(1,1,1,16,16,16),expand(1,0,1,16,1,0);
    for(int st=0;st<count;++st) {
        auto key=keys[st*128],query=queries[st*128],record=records[st*264];
        Muls(checkpoint,checkpoint,dec[st],tile*128); DEEP_VEC();
        Mul(scratch,checkpoint,key,uint64_t(64),tile,broadcast);
        Mul(scratch[64],checkpoint[64],key[64],uint64_t(64),tile,broadcast);
        DEEP_VEC(); Add(scratch,scratch,scratch[64],uint64_t(64),tile,sum);
        DEEP_VEC(); WholeReduceSum(projection,scratch,uint64_t(64),tile,1,1,16);
        DEEP_VEC(); Sub(record,values[st*tile],projection,tile);
        DEEP_VEC(); Muls(record,record,bet[st],tile);
        DEEP_VEC(); Brcb(outer,record,tile/8,BrcbRepeatParams(1,8)); DEEP_VEC();
        Mul(scratch,outer,key,uint64_t(64),tile,expand);
        Mul(scratch[64],outer,key[64],uint64_t(64),tile,expand);
        DEEP_VEC(); Add(checkpoint,checkpoint,scratch,tile*128); DEEP_VEC();
        Mul(scratch,checkpoint,query,uint64_t(64),tile,broadcast);
        Mul(scratch[64],checkpoint[64],query[64],uint64_t(64),tile,broadcast);
        DEEP_VEC(); Add(scratch,scratch,scratch[64],uint64_t(64),tile,sum);
        DEEP_VEC(); WholeReduceSum(projection,scratch,uint64_t(64),tile,1,1,16);
        DEEP_VEC();
        Cast(outputs[st*tile],projection,RoundMode::CAST_RINT,tile);
        DataCopy(record[128],key,128);
    }
    PipeBarrier<PIPE_ALL>();
    DEEP_U2G(out[(begin*24+head)*128],outputs,count,tile,3072,bfloat16_t);
    // MTE advances 264 FP32 in UB but writes exactly 257 FP32 per contiguous GM record.
    DEEP_U2G(replay[head*1028],records,count,257,257,float);
} else if(count>0) {
    Duplicate(outputs,bfloat16_t(0),count*tile); DEEP_SYNC(V_MTE3);
    DEEP_U2G(out[(begin*24+head)*128],outputs,count,tile,3072,bfloat16_t);
}
PipeBarrier<PIPE_ALL>();
#undef DEEP_G2U
#undef DEEP_U2G
#undef DEEP_SYNC
#undef DEEP_VEC
}
"""
