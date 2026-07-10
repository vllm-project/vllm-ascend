"""
Minimal probe for the "independent NZ list" EPLB approach for W4A8MXFP.
RUN THIS ON THE REAL DEPLOYMENT DEVICE (the one where allow_internal_format is ON /
where inference actually runs), not on a box where NZ falls back to ND.

It checks the 3 uncertain adaptation points:
  1. Does unbind(0)+clone of a packed FRACTAL_NZ_C0_16 weight give a *standalone*
     tensor that is still NZ with storage_offset == 0?
  2. Does copy_ work between two standalone (contiguous, offset-0) NZ tensors?
     (every copy_ failure so far was on a *non-contiguous packed slice*; this is
     the writeback form the list approach actually uses.)
  3. Does npu_grouped_matmul accept weight as a *per-expert list* for fp8_e4m3 x
     fp4_e2m1 (best-effort; the reliable way is the device_op tweak noted below).

Each test is isolated in try/except so one failure doesn't hide the others.
"""

import torch
import torch_npu

NZ = 29
ND = 2
DEV = "npu:0"

# Shapes mirror the real w13_weight: packed (E, 2I, H//2) uint8, group_size 32.
E, TWO_I, H_HALF = 8, 4096, 2048
GROUP_SIZE = 32


def fmt(t):
    return f"format={torch_npu.get_npu_format(t)} offset={t.storage_offset()} " \
           f"contig={t.is_contiguous()} shape={tuple(t.shape)} stride={t.stride()}"


def build_packed_nz():
    """Reproduce the load-time cast: uint8 ND -> NZ(C0_16) -> transpose(1,2)."""
    w_nd = torch.randint(0, 255, (E, TWO_I, H_HALF), dtype=torch.uint8, device=DEV)
    w_nz = torch_npu.npu_format_cast(
        w_nd, NZ, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
    )
    w_nz = w_nz.transpose(1, 2)  # (E, H//2, 2I), matches real w13_weight
    return w_nd, w_nz


def main():
    torch.npu.set_device(0)
    w_nd, w_nz = build_packed_nz()
    print("[setup] packed w_nz:", fmt(w_nz))

    # ---- TEST 1: unbind + clone gives standalone NZ, offset 0? -------------------
    # Try both: clone of the transposed slice, and clone of the un-transposed slice.
    print("\n==== TEST 1: unbind(0)+clone format/offset ====")
    try:
        # (a) clone the transposed per-expert slice directly
        sl_t = list(w_nz.unbind(0))[3]            # expert 3, (H//2, 2I), non-contig, offset!=0
        print("  slice[3] (transposed):", fmt(sl_t))
        cl_t = sl_t.clone()
        print("  clone(slice[3]):       ", fmt(cl_t))
    except Exception as e:
        print("  clone(transposed slice) FAIL:", repr(e)[:160])
    try:
        # (b) un-transpose first, then unbind+clone the contiguous NZ block (w4a8 style)
        w_nz_contig = w_nz.transpose(1, 2)         # back to (E, 2I, H//2) contiguous NZ
        sl_c = list(w_nz_contig.unbind(0))[3]      # expert 3, (2I, H//2)
        print("  slice[3] (un-transposed):", fmt(sl_c))
        cl_c = sl_c.clone()
        print("  clone(un-transposed):    ", fmt(cl_c))
    except Exception as e:
        print("  clone(un-transposed slice) FAIL:", repr(e)[:160])

    # ---- TEST 2: copy_ between two STANDALONE NZ tensors (the writeback) ---------
    print("\n==== TEST 2: copy_ standalone NZ <- standalone NZ (writeback) ====")
    try:
        w_nz_contig = w_nz.transpose(1, 2)
        experts = [e.clone() for e in w_nz_contig.unbind(0)]  # standalone NZ, offset 0 each
        dst = experts[3]
        src = experts[5].clone()                              # another standalone NZ payload
        print("  dst:", fmt(dst))
        print("  src:", fmt(src))
        dst.copy_(src)                                        # NZ <- NZ, both standalone
        print("  copy_ standalone NZ<-NZ: OK")
    except Exception as e:
        print("  copy_ standalone NZ<-NZ FAIL:", repr(e)[:200])

    # ---- TEST 2b: copy_ standalone NZ <- ND (in case buffers arrive as ND) -------
    print("\n==== TEST 2b: copy_ standalone NZ <- ND ====")
    try:
        w_nz_contig = w_nz.transpose(1, 2)
        dst = w_nz_contig.unbind(0)[3].clone()
        nd_buf = torch.randint(0, 255, tuple(dst.shape), dtype=torch.uint8, device=DEV)  # ND
        dst.copy_(nd_buf)
        print("  copy_ standalone NZ<-ND: OK")
    except Exception as e:
        print("  copy_ standalone NZ<-ND FAIL:", repr(e)[:200])

    # ---- TEST 3: npu_grouped_matmul with weight as a per-expert list -------------
    # Best-effort; numerics are junk, we only care whether the op ACCEPTS the list
    # form for fp8_e4m3 x fp4_e2m1. Compare against the known-good single-tensor form.
    print("\n==== TEST 3: npu_grouped_matmul weight=[packed] vs weight=list ====")
    try:
        T = 16
        H = H_HALF * 2  # real hidden
        x = torch.randint(0, 255, (T, H), dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn)
        x_scale = torch.ones(T, dtype=torch.float8_e8m0fnu, device=DEV)
        # weight_scale (antiquant_scale): (E, 2I, H//group_size)
        w_scale = torch.ones(E, TWO_I, H // GROUP_SIZE, dtype=torch.float8_e8m0fnu, device=DEV)
        # group_list: cumulative tokens per expert (all tokens to expert 0 for simplicity)
        group_list = torch.tensor([T] + [T] * (E - 1), dtype=torch.int64, device=DEV)
        w_nz_contig = w_nz.transpose(1, 2)  # (E, 2I, H//2) NZ

        def run(weight, wscale):
            return torch_npu.npu_grouped_matmul(
                x=[x], weight=weight, scale=None, antiquant_scale=wscale, scale_dtype=None,
                per_token_scale=[x_scale], per_token_scale_dtype=torch.float8_e8m0fnu,
                split_item=2, group_type=0, group_list=group_list,
                x_dtype=torch.float8_e4m3fn, weight_dtype=torch_npu.float4_e2m1fn_x2,
                output_dtype=torch.bfloat16,
            )[0]

        try:
            out = run([w_nz_contig], [w_scale])
            print("  weight=[packed 3D]: OK, out", tuple(out.shape))
        except Exception as e:
            print("  weight=[packed 3D] FAIL:", repr(e)[:200])

        try:
            w_list = [e.clone() for e in w_nz_contig.unbind(0)]   # per-expert NZ list
            s_list = [s for s in w_scale.unbind(0)]
            out = run(w_list, s_list)
            print("  weight=per-expert list: OK, out", tuple(out.shape))
        except Exception as e:
            print("  weight=per-expert list FAIL:", repr(e)[:200])
    except Exception as e:
        print("  TEST 3 setup FAIL:", repr(e)[:200])

    print("\n[done]")


if __name__ == "__main__":
    main()
