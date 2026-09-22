# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner for msprof op collection of the fused rms_norm_cast custom op.

msprof op needs a plain command line; this script just creates the inputs and
calls the op repeatedly so the profiler can capture steady-state launches.

Usage (from the repo root):
  msprof op --kernel-name=RmsNormCast --output=./profiling \
      --aic-metrics=PipeUtilization \
      python benchmarks/rms_norm_cast_msprof.py 2048
"""

import sys

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
HIDDEN = 7168
ITERS = 30

x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="npu")
gamma = torch.randn(HIDDEN, dtype=torch.bfloat16, device="npu")

for _ in range(10):
    torch.ops._C_ascend.npu_rms_norm_cast(x, gamma, 1e-6)
torch.npu.synchronize()
for _ in range(ITERS):
    torch.ops._C_ascend.npu_rms_norm_cast(x, gamma, 1e-6)
torch.npu.synchronize()
print("done")
