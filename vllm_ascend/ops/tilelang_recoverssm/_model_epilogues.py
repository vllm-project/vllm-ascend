# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend C source fragments used by the A3 mixed-core Verify experiment."""

PACK_BUFFERS = [
    ("pack_tile", "bfloat16_t", 8 * 128),
    ("pack_fp", "float", 4 * 128),
    ("pack_stat", "float", 8),
    ("pack_reduce", "float", 128),
    ("reduce_tmp", "float", 128),
    ("pack_ab", "bfloat16_t", 4 * 32),
    ("pack_a", "float", 4 * 32),
    ("pack_b", "float", 4 * 32),
    ("pack_bias", "float", 24),
    ("pack_alog", "float", 24),
    ("pack_x", "float", 16),
    ("pack_decay", "float", 16),
    ("pack_beta", "float", 16),
    ("pack_ones", "float", 16),
]

PACK_BINDINGS = r"""
GlobalTensor<bfloat16_t> raw_q, raw_k, raw_a, raw_b;
GlobalTensor<float> raw_alog, raw_bias;
raw_q.SetGlobalBuffer((__gm__ bfloat16_t*)q_raw_handle);
raw_k.SetGlobalBuffer((__gm__ bfloat16_t*)k_raw_handle);
raw_a.SetGlobalBuffer((__gm__ bfloat16_t*)a_raw_handle);
raw_b.SetGlobalBuffer((__gm__ bfloat16_t*)b_raw_handle);
raw_alog.SetGlobalBuffer((__gm__ float*)log_a_handle);
raw_bias.SetGlobalBuffer((__gm__ float*)bias_handle);
Duplicate(pack_ones,1.0f,16);
if (sid == 0) {
    G2U(pack_bias,raw_bias[0],1,24,24,float);
    G2U(pack_alog,raw_alog[0],1,24,24,float);
    SYNC(MTE2_V);
    Exp(pack_alog,pack_alog,24);
    SYNC(V_S);
}
"""

# The two AIVs own disjoint 32-byte-aligned packed/norm regions.
# The C side waits once per valid job, including empty rows and invalid pages.
PACK_JOB = r"""
#define PACK_JOB(index) do { \
    const int pj=(index), pr=pj/8, ph=pj%8; \
    const int pb=lg.GetValue(pr), pe=lg.GetValue(pr+1); \
    const int pn=(pb>=0 && pe<=@TOKENS@ && pe>pb && pe-pb<=4) ? pe-pb : 0; \
    SYNC(MTE3_V); \
    Duplicate(pack_tile,bfloat16_t(0.0f),8*128); \
    Duplicate(pack_stat,0.0f,8); \
    SYNC(V_MTE2); \
    if (pn>0) { \
        if (sid==0) { G2U(pack_tile,raw_k[pb*@K_STRIDE@+ph*128],pn,128,@K_STRIDE@,bfloat16_t); } \
        else { G2U(pack_tile,raw_q[pb*@Q_STRIDE@+ph*128],pn,128,@Q_STRIDE@,bfloat16_t); } \
    } \
    SYNC(MTE2_V); \
    Cast(pack_fp,pack_tile,RoundMode::CAST_NONE,4*128); BV(); \
    Mul(pack_fp,pack_fp,pack_fp,4*128); BV(); \
    for (int i=0; i<4; ++i) { \
        ReduceSum(pack_reduce,pack_fp[i*128],reduce_tmp,128); SYNC(V_S); \
        pack_stat.SetValue(i,pack_reduce.GetValue(0)); SYNC(S_V); \
    } \
    Adds(pack_stat,pack_stat,1e-6f,8); BV(); Sqrt(pack_stat,pack_stat,8); BV(); \
    Div(pack_stat,pack_ones,pack_stat,8); \
    SYNC(V_MTE3); \
    U2G(kg[pj*2048+sid*512],pack_tile,2,512,1024,bfloat16_t); \
    U2G(ag[pj*48+32+sid*8],pack_stat,1,8,8,float); \
    if (sid==0 && pn>0) { \
        G2U(pack_ab,raw_a[pb*@GATE@],pn,24,@GATE@,bfloat16_t); SYNC(MTE2_V); \
        Cast(pack_a,pack_ab,RoundMode::CAST_NONE,pn*32); SYNC(V_MTE2); \
        G2U(pack_ab,raw_b[pb*@GATE@],pn,24,@GATE@,bfloat16_t); SYNC(MTE2_V); \
        Cast(pack_b,pack_ab,RoundMode::CAST_NONE,pn*32); \
        Duplicate(pack_x,0.0f,16); Duplicate(pack_beta,0.0f,16); SYNC(V_S); \
        for (int i=0; i<pn; ++i) { \
            for (int h=0; h<3; ++h) { \
                pack_x.SetValue(i*4+h,pack_a.GetValue(i*32+ph*3+h)+pack_bias.GetValue(ph*3+h)); \
                pack_beta.SetValue(i*4+h,-pack_b.GetValue(i*32+ph*3+h)); \
            } \
        } \
        SYNC(S_V); \
        Exp(pack_decay,pack_x,16); BV(); Adds(pack_decay,pack_decay,1.0f,16); BV(); \
        Ln(pack_decay,pack_decay,16); \
        Exp(pack_beta,pack_beta,16); BV(); Adds(pack_beta,pack_beta,1.0f,16); BV(); \
        Div(pack_beta,pack_ones,pack_beta,16); SYNC(V_S); \
        for (int i=0; i<pn; ++i) { \
            for (int h=0; h<3; ++h) { \
                const int p=i*4+h; \
                const float x=pack_x.GetValue(p); \
                const float sp=x>20.0f ? x : pack_decay.GetValue(p); \
                pack_decay.SetValue(p,-pack_alog.GetValue(ph*3+h)*sp); \
            } \
        } \
        SYNC(S_V); Exp(pack_decay,pack_decay,16); SYNC(V_MTE3); \
        U2G(ag[pj*48],pack_decay,1,16,16,float); \
        U2G(ag[pj*48+16],pack_beta,1,16,16,float); \
    } \
    CrossCoreSetFlag<2,PIPE_MTE3>(7); \
} while (0)
"""

NORM_BUFFERS = [
    ("norm_z", "bfloat16_t", 4 * 256),
    ("norm_weight_bf", "bfloat16_t", 128),
    ("norm_weight_fp", "float", 128),
    ("norm_x", "float", 4 * 256),
    ("norm_gate", "float", 4 * 256),
    ("norm_exp", "float", 4 * 256),
    ("norm_square", "float", 4 * 256),
    ("norm_stat", "float", 8),
]

VECTOR_RSTD_BUFFERS = [
    ("norm_reductions", "float", 64),
    ("norm_rstd_offsets", "uint32_t", 8),
    ("norm_rstd_values", "float", 8),
    ("norm_rstd_blocks", "float", 64),
]

VECTOR_RSTD_INIT = r"""
for (int i=0; i<8; ++i) {
    norm_rstd_offsets.SetValue(i, uint32_t(i*32));
}
SYNC(S_V);
"""

NORM_BINDINGS = r"""
GlobalTensor<bfloat16_t> norm_z_gm, norm_weight_gm;
norm_z_gm.SetGlobalBuffer((__gm__ bfloat16_t*)norm_z_raw_handle);
norm_weight_gm.SetGlobalBuffer((__gm__ bfloat16_t*)norm_w_raw_handle);
G2U(norm_weight_bf,norm_weight_gm[0],1,128,128,bfloat16_t);
SYNC(MTE2_V);
Cast(norm_weight_fp,norm_weight_bf,RoundMode::CAST_NONE,128);
"""

# Start from the same rounded BF16 attention result as the separate RMSNorm.
NORM_APPLY = r"""
G2U(norm_z,norm_z_gm[begin*@Z_STRIDE@+head*128],count,width,@Z_STRIDE@,bfloat16_t);
SYNC(MTE2_V);
const int norm_elements=count*width;
Cast(norm_x,outputs,RoundMode::CAST_NONE,norm_elements);
Cast(norm_gate,norm_z,RoundMode::CAST_NONE,norm_elements); BV();
Mul(norm_square,norm_x,norm_x,norm_elements);
Muls(norm_exp,norm_gate,-1.0f,norm_elements); BV();
Exp(norm_exp,norm_exp,norm_elements); BV();
Adds(norm_exp,norm_exp,1.0f,norm_elements); BV();
Div(norm_gate,norm_gate,norm_exp,norm_elements); BV();
for (int t=0; t<count; ++t) {
    for (int h=0; h<owned; ++h) {
        const int row=t*width+h*128;
        ReduceSum(norm_stat,norm_square[row],reduce_tmp,128); BV();
        Muls(norm_stat,norm_stat,1.0f/128.0f,8); BV();
        Adds(norm_stat,norm_stat,@NORM_EPS@,8); BV();
        Rsqrt(norm_stat,norm_stat,8); SYNC(V_S);
        const float rstd=norm_stat.GetValue(0);
        SYNC(S_V);
        Muls(norm_x[row],norm_x[row],rstd,128); BV();
        Mul(norm_x[row],norm_x[row],norm_weight_fp,128); BV();
        Mul(norm_x[row],norm_x[row],norm_gate[row],128); BV();
        Cast(outputs[row],norm_x[row],RoundMode::CAST_RINT,128); BV();
    }
}
"""


NORM_FP32_BINDINGS = r"""
GlobalTensor<bfloat16_t> norm_z_gm;
GlobalTensor<float> norm_weight_gm;
norm_z_gm.SetGlobalBuffer((__gm__ bfloat16_t*)norm_z_raw_handle);
norm_weight_gm.SetGlobalBuffer((__gm__ float*)norm_w_raw_handle);
G2U(norm_weight_fp,norm_weight_gm[0],1,128,128,float);
SYNC(MTE2_V);
"""

VECTOR_RSTD_APPLY = r"""
const int norm_rows=count*owned;
Duplicate(norm_reductions,0.0f,64); BV();
for (int i=0; i<norm_rows; ++i) {
    ReduceSum(norm_reductions[i*8],norm_square[i*128],reduce_tmp,128); BV();
}
Muls(norm_reductions,norm_reductions,1.0f/128.0f,64); BV();
Adds(norm_reductions,norm_reductions,@NORM_EPS@,64); BV();
Rsqrt(norm_reductions,norm_reductions,64); BV();
Gather(norm_rstd_values,norm_reductions,norm_rstd_offsets,uint32_t(0),8); BV();
Brcb(norm_rstd_blocks,norm_rstd_values,1,{1,8}); BV();
for (int i=0; i<norm_rows; ++i) {
    const int row=i*128;
    // One FP32 data block of identical rstd values broadcasts across both repeats.
    Mul(norm_exp[row],norm_x[row],norm_rstd_blocks[i*8],64,2,{1,1,0,8,8,0}); BV();
    Mul(norm_x[row],norm_exp[row],norm_weight_fp,128); BV();
}
Mul(norm_x,norm_x,norm_gate,norm_elements); BV();
Cast(outputs,norm_x,RoundMode::CAST_RINT,norm_elements); BV();
"""


def norm_apply(activation, *, vector_rstd=False):
    if activation not in ("silu", "swish", "sigmoid"):
        raise ValueError(f"Unsupported gated RMSNorm activation: {activation}")
    source = NORM_APPLY
    if vector_rstd:
        source = source.split("for (int t=0; t<count; ++t)", 1)[0] + VECTOR_RSTD_APPLY
    if activation == "sigmoid":
        division = "Div(norm_gate,norm_gate,norm_exp,norm_elements); BV();"
        source = source.replace(division, "Duplicate(norm_gate,1.0f,norm_elements); BV();\n" + division)
    return source
