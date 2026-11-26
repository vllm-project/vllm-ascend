# SPDX-License-Identifier: Apache-2.0
import tilelang.ascend.language as T
from .kda_gate import gate_projection_program
from .kda_projection import pack_weight, program
from .kda_recurrent import small_batch_program
from .kda_recurrent_simd import small_batch_program as simd_middle_program


def abi_program(c, page_layout):
    B, L, I, H, D = T.dynamic("B"), 6, 4096, 64, 128
    PITCH = 3 * H * D + H + 2 * D
    state_rows = 16
    gate_shape = (H // 2, 2 * D, D)
    conv_stride, conv_row, state_stride, corr_stride, kg_stride = page_layout
    cs_shape = (T.dynamic("conv_rows"), D)
    state_shape = (T.dynamic("state_rows"), D)
    corr_shape = (T.dynamic("correction_rows"), D)
    kg_shape = (T.dynamic("kg_rows"), 2 * D)
    metadata_shape = (3, B)
    sfu_precision = "ftz_true"
    M, K, N, CORES = 64, 256, 128, 32
    PCOLS = PITCH
    taps = T.Unroll

    @T.prim_func
    def main(
        X: T.Tensor((B * L, I), "bfloat16"),
        Wi: T.Tensor((I // 16, PITCH, 16), "bfloat16"),
        Wo: T.Tensor((H * D // 16, I, 16), "bfloat16"),
        Out: T.Tensor((T.ceildiv(B * L, M) * M, I), "bfloat16"),
        P: T.Tensor((T.ceildiv(B * L, M) * M, PCOLS), "bfloat16"),
        WF: T.Tensor(gate_shape, "bfloat16"),
        WG: T.Tensor(gate_shape, "bfloat16"),
        CW: T.Tensor((H, 3, 4, D), "bfloat16"),
        CS: T.Tensor(cs_shape, "bfloat16"),
        State: T.Tensor(state_shape, "float32"),
        Accepted: T.Tensor(metadata_shape, "int32"),
        A: T.Tensor((H,), "float32"),
        DT: T.Tensor((H, D), "float32"),
        NW: T.Tensor((D,), "bfloat16"),
        Y: T.Tensor((B * L, H * D), "bfloat16"),
        Correction: T.Tensor(corr_shape, "float32"),
        GF: T.Tensor((T.ceildiv(B * L, 32) * 32, H, D), "bfloat16"),
        GG: T.Tensor((T.ceildiv(B * L, 32) * 32, H, D), "bfloat16"),
        KG: T.Tensor(kg_shape, "bfloat16"),
    ):
        with T.MixedKernel(CORES, sids=2) as (bx, sid):
            # Preserve tensor and shape arguments for the Cython launch adapter.
            operand = T.alloc_l1((32,16), "bfloat16")
            accum = T.alloc_l0c((32,32), "float32")
            shared = T.alloc_shared((16,32), "float32")
            T.copy(X[:32,:16], operand)
            T.gemm(operand, operand, accum, transpose_B=True, clear_accum=True)
            T.dual_copy(accum, shared)
            tmp = T.alloc_shared((1,), "float32")
            with T.SimtVF(threads=1):
                tmp[0] = (T.cast(B + CS.shape[0] + State.shape[0] + Correction.shape[0] + KG.shape[0],"float32") + T.cast(X[0,0],"float32") + T.cast(Wi[0,0,0],"float32")
                    + T.cast(Wo[0,0,0],"float32") + T.cast(P[0,0],"float32")
                    + T.cast(WF[0,0,0],"float32") + T.cast(WG[0,0,0],"float32")
                    + T.cast(CW[0,0,0,0],"float32") + T.cast(CS[0,0],"float32")
                    + State[0,0] + T.cast(Accepted[0,0],"float32") + A[0] + DT[0,0]
                    + T.cast(NW[0],"float32") + T.cast(Y[0,0],"float32") + Correction[0,0]
                    + T.cast(GF[0,0,0],"float32") + T.cast(GG[0,0,0],"float32") + T.cast(KG[0,0],"float32"))
            with T.SimtVF(threads=1):
                Out[0,0] = T.cast(tmp[0]+shared[0,0],"bfloat16")
    return main

def compose(c, layout, lower):
    import re
    names = ("q", "g", "mid", "o")
    m = c.tile_rows
    # Each phase reuses the physical L1, UB and L0 buffers.
    n = 784 if m <= 80 else 400 if m <= 160 else 272 if m <= 240 else 208
    mid_program = simd_middle_program if c.simd else small_batch_program
    inline_gates = c.batch == 1
    funcs = (lambda c,layout: program(4096,24896,m,n,256 if m in (96,112) else 128,cache_hint=4),
             lambda c,layout: gate_projection_program(c),
             lambda c,layout: mid_program(c, page_layout=layout, batch_gates=not inline_gates, fast_math=True),
             lambda c,layout: program(8192,4096,m,128,256))
    helpers = []
    calls = {}
    all_args = {}
    shared_bytes = 0
    for name, make in zip(names, funcs):
        phase = lower(make(c,layout))
        source = phase.kernel_source
        for f in phase.device_mod.functions.values():
            shared_bytes = max(shared_bytes, int(f.attrs.get("dyn_shared_memory_buf", 0)))
        sig = re.search(r'extern "C" __global__ .*?void main_kernel\(([^)]*)\) \{', source)
        if not sig:
            raise RuntimeError("Unrecognized TileLang Ascend entry signature")
        fields = [field.strip() for field in sig[1].split(",")]
        args = [field.rsplit(" ",1)[-1] for field in fields]
        rename = {"X":"X", "W":"Wi", "Y":"P", "Meta":"Accepted"} if name=="q" else ({"X":"Y", "W":"Wo","Y":"Out","Meta":"Accepted"} if name=="o" else {})
        if name=="mid" and not inline_gates:
            rename={"WF":"GF","WG":"GG"}
        if name=="g":
            rename={"rows":"(B * 6)"}
        mapped=[rename.get(a,a) for a in args]
        all_args.update((m,re.sub(r"\b"+a+r"$",m,field)) for a,m,field in zip(args,mapped,fields) if m.isidentifier())
        calls[name]="piece_"+name+"("+", ".join(mapped)+");\n"
        source=(source[:sig.start()]+"__aicore__ inline void piece_"+name+"("+sig[1]+") {"+source[sig.end():]).replace("  asc_init();","")
        source=source.replace("main_kernel_", "piece_"+name+"_")
        if name in ("q","o"):
            source=source.replace("compact_nz_weight",name+"_compact_nz_weight").replace("global_nz_weight",name+"_global_nz_weight")
        helpers.append(source)
    fields=sorted(all_args,key=lambda n:("*" not in all_args[n],n))
    sign=", ".join(all_args[k] for k in fields)
    fence="asc_sync_pipe(PIPE_ALL);\n"
    gates = "" if inline_gates else calls["g"]+fence+"asc_sync_inter_arrive(PIPE_FIX,15); asc_sync_inter_wait(PIPE_FIX,15);\n"
    cb = (calls["q"]+fence+"asc_sync_inter_arrive(PIPE_FIX,12); asc_sync_inter_wait(PIPE_MTE2,12);\n"+fence+gates+
          "asc_sync_intra_arrive(PIPE_FIX,10); asc_sync_intra_arrive(PIPE_FIX,26);\n"+calls["mid"]+fence+
          "asc_sync_intra_wait(PIPE_MTE2,11); asc_sync_intra_wait(PIPE_MTE2,27);\n"+calls["o"]+fence)
    vb = ("asc_sync_intra_wait(PIPE_MTE2,10);\n"+calls["mid"]+fence+
          "asc_sync_inter_arrive(PIPE_MTE3,13); asc_sync_inter_wait(PIPE_MTE3,13);\n"+
          "asc_sync_intra_arrive(PIPE_MTE3,11);\n")
    for tag,body in [('c',cb),('v',vb)]:
        helpers.append("__aicore__ inline int fused_tilelang_body_"+tag+"("+sign+") {\n"+body+"return 0;\n}")
    return "\n".join(helpers), shared_bytes, fields

# SPDX-License-Identifier: Apache-2.0
"""Single-launch BF16 KDA verify for six-token GLM-5.3 requests on Ascend."""
import tilelang.ascend.language as T


from functools import lru_cache
from types import SimpleNamespace

@lru_cache(maxsize=128)
def compile_hybrid(eps, lower_bound, layout, m, single, simd):
    from tilelang.jit.kernel import JITKernel
    import re
    cfg=SimpleNamespace(eps=eps, lower_bound=lower_bound,batch=1 if single else 2,
                        tile_rows=m,simd=simd,tokens=6,heads=64,hidden=4096,dim=128)
    class ComposedKernel(JITKernel):
        def _compile_artifact(self, *args, **kwargs):
            lower = super()._compile_artifact
            helpers, shared_bytes, callee_fields = compose(cfg, layout, lambda f: lower(f, *args[1:], **kwargs))
            artifact=lower(*args,**kwargs)
            for symbol, func in list(artifact.device_mod.functions.items()):
                artifact.device_mod.update_func(symbol, func.with_attr("dyn_shared_memory_buf", shared_bytes))
            sig=re.search(r'extern "C" __global__ .*?void main_kernel\(([^)]*)\) \{',artifact.kernel_source)
            assert "__mix__(1, 2)" in sig[0], "Adapter must launch paired Cube and Vector cores"
            fields=[field.strip().rsplit(" ",1)[-1] for field in sig[1].split(",")]
            assert callee_fields == fields, (callee_fields, fields)
            params=", ".join(fields)
            entry=sig[0]+"\nasc_init();\nif ASC_IS_AIC { fused_tilelang_body_c("+params+"); }\nif ASC_IS_AIV { fused_tilelang_body_v("+params+"); }\n}\n"
            artifact.kernel_source=helpers+"\n"+entry
            return artifact

    return ComposedKernel(abi_program(cfg,layout), target="ascend",execution_backend="cython")

class WholeKDA:
    """Bind BF16 verify weights and paged ReplaySSM storage for a single layer.

    metadata contains conv page IDs, checkpoint page IDs, and accepted lengths.
    Prepare metadata with the scheduler before calling the layer.
    """

    def __init__(self, layer, batch=None):
        import tilelang
        import torch
        import torch_npu
        from types import SimpleNamespace

        conv, state, corr, kg = layer.kv_cache
        if layer._conv_state_dim_first:
            conv = conv.transpose(-1, -2)
        if conv.stride(2) != 1 or any(not t[0].is_contiguous() for t in (state, corr, kg)):
            raise ValueError("Unsupported ReplaySSM page layout")
        strides = (conv.stride(0), conv.stride(1), state.stride(0), corr.stride(0), kg.stride(0))
        widths = (128, 128, 128, 128, 256)
        if any(stride % width for stride, width in zip(strides, widths)):
            raise ValueError("Cache strides must be multiples of the corresponding head width")
        layout = tuple(stride // width for stride, width in zip(strides, widths))
        self.config = (layer.o_norm.eps, layer.kda_lower_bound, layout)

        def flat(t, width):
            size = 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
            if size % width:
                raise ValueError("Cache ends in an incomplete head row")
            return t.as_strided((size // width, width), (width, 1))

        def weight(module):
            return torch_npu.npu_format_cast(module.weight, 2).contiguous()

        self.win = pack_weight(weight(layer.in_proj_qkvbfg_a),0,0)
        self.wo = pack_weight(weight(layer.o_proj),0,0)
        self.wf = weight(layer.f_b_proj).reshape(32, 256, 128)
        self.wg = weight(layer.g_b_proj).reshape(32, 256, 128)
        self.cw = torch.stack(
            [m.weight.reshape(64, 128, 4) for m in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)], dim=1
        ).permute(0, 1, 3, 2).bfloat16().contiguous()
        self.a = layer.A_log.reshape(64).float().contiguous()
        self.dt = layer.dt_bias.reshape(64, 128).float().contiguous()
        self.nw = layer.o_norm.weight.reshape(128).bfloat16().contiguous()
        self.conv, self.state, self.corr, self.kg = [
            flat(t, w) for t, w in zip((conv, state, corr, kg), (128, 128, 128, 256))
        ]

    @staticmethod
    def allocate(x):
        rows = x.shape[0]
        padded = (rows + 63) // 64 * 64
        gates = (((rows + 31) // 32) * 32, 64, 128)
        return x.new_empty((padded, 24896)), x.new_empty((rows, 8192)), x.new_empty((padded, 4096)), x.new_empty(gates), x.new_empty(gates)

    def __call__(self, x, metadata, workspace=None):
        import torch
        rows = x.shape[0]
        if x.shape != (rows, 4096) or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("Expected contiguous BF16 hidden states with width 4096")
        if rows == 0 or rows % 6 or metadata.shape != (3, rows // 6):
            raise ValueError("Expected six tokens per request and three metadata fields")
        if metadata.dtype != torch.int32 or not metadata.is_contiguous():
            raise ValueError("Expected contiguous int32 metadata")
        buffers = self.allocate(x) if workspace is None else workspace
        if len(buffers) == 3:
            gate_shape = ((rows + 31) // 32 * 32, 64, 128)
            buffers = (*buffers, x.new_empty(gate_shape), x.new_empty(gate_shape))
        p, y, out, gf, gg = buffers
        batch = rows // 6
        m = (rows + 15) // 16 * 16 if batch <= 50 else 64
        kernel = compile_hybrid(*self.config, m, batch == 1, batch >= 10)
        kernel(x, self.win, self.wo, out, p, self.wf, self.wg, self.cw, self.conv,
                    self.state, metadata, self.a, self.dt, self.nw, y, self.corr, gf, gg, self.kg)
        return out[:rows]

