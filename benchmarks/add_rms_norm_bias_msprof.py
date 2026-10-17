# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runner for msprof op collection of the fused add_rms_norm_bias custom op.

msprof op needs a plain command line; this script just creates the inputs and
calls the op repeatedly so the profiler can capture steady-state launches.

Usage (from the repo root):
  msprof op --kernel-name=AddRmsNormBias --output=./profiling \
      --aic-metrics=PipeUtilization \
      python benchmarks/add_rms_norm_bias_msprof.py 2048
"""

import sys

import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
HIDDEN = int(sys.argv[2]) if len(sys.argv) > 2 else 7168
ITERS = 30

x1 = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="npu")
x2 = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="npu")
gamma = torch.randn(HIDDEN, dtype=torch.bfloat16, device="npu")
beta = torch.randn(HIDDEN, dtype=torch.bfloat16, device="npu")

for _ in range(10):
    torch.ops._C_ascend.npu_add_rms_norm_bias(x1, x2, gamma, beta, 1e-6)
torch.npu.synchronize()
for _ in range(ITERS):
    torch.ops._C_ascend.npu_add_rms_norm_bias(x1, x2, gamma, beta, 1e-6)
torch.npu.synchronize()
print("done")
