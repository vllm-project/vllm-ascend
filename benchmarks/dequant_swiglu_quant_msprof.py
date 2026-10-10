# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner for msprof op collection of the fused dequant_swiglu_quant op.

msprof op needs a plain command line; this script just creates the inputs and
calls the op repeatedly so the profiler can capture steady-state launches.

Usage (from the repo root):
  msprof op --kernel-name=dequant_swiglu_quant --output=./profiling \
      --aic-metrics=PipeUtilization \
      python benchmarks/dequant_swiglu_quant_msprof.py 2048 4096

Positional args: tokens, x_last_dim (2H). Optional env-style flags via argv[3]:
  nogroup  : shared-experts path, swiglu_mode=1, clamp 0.0 (default)
  clamp7   : shared-experts path, swiglu_mode=1, clamp 7.0
  group:N  : MoE group_index path with N groups, swiglu_mode=0
"""

import sys

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
X_LAST_DIM = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
MODE = sys.argv[3] if len(sys.argv) > 3 else "nogroup"
ITERS = 30

SWIGLU_MODE = 1
CLAMP_LIMIT = 0.0
GROUPS = None
if MODE == "clamp7":
    CLAMP_LIMIT = 7.0
elif MODE.startswith("group:"):
    GROUPS = int(MODE.split(":")[1])
    SWIGLU_MODE = 0

torch.manual_seed(7)
x = torch.randint(-1000, 1000, (TOKENS, X_LAST_DIM), dtype=torch.int32,
                  device="npu")
if GROUPS is None:
    weight_scale = torch.randn(X_LAST_DIM, dtype=torch.float32,
                               device="npu") * 0.001
else:
    # The group path requires weight_scale [group_num, 2H].
    weight_scale = torch.randn(GROUPS, X_LAST_DIM, dtype=torch.float32,
                               device="npu") * 0.001
activation_scale = torch.rand(TOKENS, 1, dtype=torch.float32, device="npu") * 4 + 0.5
group_index = None
if GROUPS is not None:
    base = TOKENS // GROUPS
    sizes = [base] * GROUPS
    for i in range(TOKENS - base * GROUPS):
        sizes[i] += 1
    # The op expects per-group row counts (dst_list_type=1), not a cumsum.
    group_index = torch.tensor(sizes, dtype=torch.int64, device="npu")

for _ in range(10):
    torch.ops._C_ascend.npu_dequant_swiglu_quant(
        x=x, weight_scale=weight_scale, activation_scale=activation_scale,
        bias=None, quant_scale=None, quant_offset=None,
        group_index=group_index, activate_left=True, quant_mode=1,
        swiglu_mode=SWIGLU_MODE, clamp_limit=CLAMP_LIMIT,
        glu_alpha=1.0, glu_bias=0.0)
torch.npu.synchronize()
for _ in range(ITERS):
    torch.ops._C_ascend.npu_dequant_swiglu_quant(
        x=x, weight_scale=weight_scale, activation_scale=activation_scale,
        bias=None, quant_scale=None, quant_offset=None,
        group_index=group_index, activate_left=True, quant_mode=1,
        swiglu_mode=SWIGLU_MODE, clamp_limit=CLAMP_LIMIT,
        glu_alpha=1.0, glu_bias=0.0)
torch.npu.synchronize()
print("done")
