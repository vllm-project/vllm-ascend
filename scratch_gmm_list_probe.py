"""
Single-op probe: does npu_grouped_matmul (W4A8MXFP / fp8_e4m3 x fp4_e2m1) accept a
PER-EXPERT weight LIST instead of one packed tensor?  RUN ON THE REAL DEVICE.

Strategy: build the packed weight + scales exactly like process_weights does, then call
npu_grouped_matmul TWICE with identical inputs except the weight form:
    (A) baseline  weight=[packed 3D]         <- known-good; validates the whole setup
    (B) list      weight=list(unbind(0))     <- the form the independent-tensor route needs
If (A) fails  -> setup is wrong (most likely scale shape/dtype); the in-model device_op
               change is then the reliable fallback.
If (A) ok, (B) ok, outputs match -> list form supported.
If (A) ok, (B) fails/mismatch     -> list form NOT usable as-is (report the error).
"""

import torch
import torch_npu

NZ = 29
DEV = "npu:0"

E = 8            # experts
H = 2048         # hidden
I = 1024         # intermediate  -> gate_up output = 2I
GS = 32          # group size
TWO_I = 2 * I
H_HALF = H // 2  # 4-bit packed along K
TOK_PER_E = 4
T = E * TOK_PER_E

MXFP_KW = dict(customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2)


def e8m0_ones(shape):
    """All-1.0 e8m0 tensor (exponent bias 127) built without aclnnInplaceOne."""
    return torch.full(shape, 127, dtype=torch.uint8, device=DEV).view(torch.float8_e8m0fnu)


def build():
    # packed weight: ND (E, 2I, H/2) -> NZ -> transpose(1,2) == what the layer holds
    w_nd = torch.randint(0, 255, (E, TWO_I, H_HALF), dtype=torch.uint8, device=DEV)
    weight = torch_npu.npu_format_cast(w_nd, NZ, **MXFP_KW).transpose(1, 2)  # (E, H/2, 2I) NZ

    # weight_scale: replicate fp8.py process transform
    #   pre: (E, 2I, H/GS) e8m0  ->  reshape(g,n,k//2,2).view(uint8).transpose(-3,-2)
    scale = e8m0_ones((E, TWO_I, H // GS))
    g, n, k = scale.shape
    weight_scale = scale.reshape(g, n, k // 2, 2).view(torch.uint8).transpose(-3, -2)  # (E, H/GS/2, 2I, 2)

    # activation + per-token scale
    x = torch.randint(0, 255, (T, H), dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn)
    x_scale = e8m0_ones((T,))

    # group_list: cumulative token count per expert (group_type=0)
    group_list = torch.tensor([TOK_PER_E * (i + 1) for i in range(E)], dtype=torch.int64, device=DEV)
    return weight, weight_scale, x, x_scale, group_list


def call(weight, weight_scale, x, x_scale, group_list):
    return torch_npu.npu_grouped_matmul(
        x=[x],
        weight=weight,
        scale=None,
        antiquant_scale=weight_scale,
        scale_dtype=None,
        per_token_scale=[x_scale],
        per_token_scale_dtype=torch.float8_e8m0fnu,
        split_item=2,
        group_type=0,
        group_list=group_list,
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        output_dtype=torch.bfloat16,
    )[0]


def main():
    torch.npu.set_device(0)
    weight, weight_scale, x, x_scale, group_list = build()
    print(f"[setup] weight {tuple(weight.shape)}  weight_scale {tuple(weight_scale.shape)}  "
          f"x {tuple(x.shape)}  group_list {group_list.tolist()}")

    print("\n==== (A) baseline: weight=[packed 3D] ====")
    ok_ref = False
    try:
        out_ref = call([weight], [weight_scale], x, x_scale, group_list)
        print("  OK  out", tuple(out_ref.shape), out_ref.dtype)
        ok_ref = True
    except Exception as e:
        print("  FAIL:", repr(e)[:260])
        print("  -> setup issue (likely scale shape/dtype). Use the in-model device_op change instead.")

    print("\n==== (B) per-expert list: weight=list(unbind(0)) ====")
    try:
        w_list = list(weight.unbind(0))
        s_list = list(weight_scale.unbind(0))
        print(f"  per-expert weight {tuple(w_list[0].shape)}  scale {tuple(s_list[0].shape)}  (x{len(w_list)})")
        out_list = call(w_list, s_list, x, x_scale, group_list)
        print("  OK  out", tuple(out_list.shape), out_list.dtype)
        if ok_ref:
            same = torch.allclose(out_ref.float(), out_list.float(), atol=1e-2, rtol=1e-2)
            maxdiff = (out_ref.float() - out_list.float()).abs().max().item()
            print(f"  matches baseline: {same}  (max_abs_diff={maxdiff:.4g})")
    except Exception as e:
        print("  FAIL:", repr(e)[:260])

    print("\n[done]")


if __name__ == "__main__":
    main()
