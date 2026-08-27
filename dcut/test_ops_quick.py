#!/usr/bin/env python3
"""Quick test: load dcut torch extension and test operator registration."""
import sys, os

# Load dcut torch extension
ext_path = "/data/c00954457/tmp/VLLM0.23.0/dcut/kernel/build/torch_extension/dcut_torch_ops.so"
if os.path.exists(ext_path):
    print(f"[1] Loading torch extension: {ext_path}")
    import torch
    torch.ops.load_library(ext_path)
    print("    OK")
else:
    print(f"[!] Extension not found: {ext_path}")
    sys.exit(1)

# List all registered _C_ascend ops that contain "recurrent" or "dcut"
print("\n[2] Searching for registered ops...")
import torch_npu  # ensure _C_ascend dispatcher is ready
ops = [name for name in dir(torch.ops._C_ascend) if "recurrent" in name.lower() or "dcut" in name.lower()]
for o in sorted(ops):
    print(f"    {o}")
if not ops:
    print("    (none found in dir(); checking hasattr instead)")
    for name in ("npu_dcut_recurrent_gated_delta_rule", "npu_recurrent_gated_delta_rule"):
        exists = hasattr(torch.ops._C_ascend, name)
        print(f"    hasattr({name}) = {exists}")

# Call dcut op with the CORRECT signature:
# (q, k, v, state, *, beta, scale, query_start_loc,
#  ssm_state_indices, num_accepted_tokens, g, gk, zero_padded_output) -> Tensor
print("\n[3] Calling npu_dcut_recurrent_gated_delta_rule (correct sig) ...")
import torch
dev = "npu:0"
dt  = torch.bfloat16
B, S, H, D = 2, 4, 8, 64
T = B * S  # packed sequence length
# 3D packed format: (T, H, D)
q  = torch.randn(T, H, D,  device=dev, dtype=dt)
k  = torch.randn(T, H, D,  device=dev, dtype=dt)
v  = torch.randn(T, H, D,  device=dev, dtype=dt)
g  = torch.randn(T, H, D,  device=dev, dtype=torch.float32)  # gate must be float32
# 4D state: (B, H, D, D_state)
state = torch.randn(B, H, D, D, device=dev, dtype=dt)
# 2D beta: (T, H)
beta  = torch.randn(T, H,    device=dev, dtype=dt)
# query_start_loc: cumulative packed-token offsets, shape (B+1,)
qsl   = torch.tensor([0, S, 2*S], device=dev, dtype=torch.int32)
# ssm_state_indices: (B, S) where 1 <= S <= 16
ssi   = torch.tensor([[0]*S]*B, device=dev, dtype=torch.int32)
nat   = torch.tensor([S]*B, device=dev, dtype=torch.int32)

try:
    out = torch.ops._C_ascend.npu_dcut_recurrent_gated_delta_rule(
        q, k, v, state,
        g=g,
        beta=beta,
        scale=1.0,
        query_start_loc=qsl,
        ssm_state_indices=ssi,
        num_accepted_tokens=nat,
    )
    print(f"    OK  output shape: {out.shape if not isinstance(out,tuple) else [o.shape for o in out]}")
except Exception as e:
    print(f"    FAIL: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")
