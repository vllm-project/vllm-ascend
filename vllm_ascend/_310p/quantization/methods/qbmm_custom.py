#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Custom W8A8 batch matmul (QuantBatchMatmulV3X) routing for 310P.

The custom kernel beats the CANN builtin at both regimes on 310P
(prefill M=2048: 1.5-1.9x on the large shapes, and the builtin crashes on
K=12288/N=4096 there; decode M=1: ~5%). Two integration hazards this module
exists to handle:

- Layout: the kernel is numerically correct ONLY with a plain-NZ weight and
  transpose_x2=True. Serving stores linear weights as
  maybe_trans_nz(w).transpose(0, 1); transposing that view back yields the
  original [N, K] NZ tensor, which is exactly the required operand - no
  re-processing, same storage.
- torch.compile: a raw _C_ascend op inside the compiled model region fails
  AOT capture (gb0090) even with a fake impl registered, so the call is
  wrapped through vLLM's direct_register_custom_op - dynamo sees an opaque
  vllm-namespace op with a proper fake, and the aclnn call stays inside it.

Opt-in via VLLM_CUSTOM_QBMM=1; default serving is untouched. The op itself
was renamed from QuantBatchMatmulV3 to QuantBatchMatmulV3X so its aclnn
symbol can no longer shadow the builtin that torch_npu.npu_quant_matmul
resolves (the "vendor hijack").
"""

import os

import torch
import torch_npu

from vllm.utils.torch_utils import direct_register_custom_op

_ENV = "VLLM_CUSTOM_QBMM"

# The custom kernel's tiling is validated for the model's projection shapes
# (K/N up to 24576); the 248k-row vocab head fails tiling (ret -1). Shapes
# past this bound fall back to the builtin inside the opaque op, where the
# branch is invisible to dynamo.
_MAX_DIM = 32768


def _kernel_supports(weight_t: torch.Tensor) -> bool:
    # weight_t is [K, N]. Tiling rejects K/N past _MAX_DIM, and an odd
    # K % 32 tail (the partial K-fractal MTE2 stride is only block-aligned
    # for an even tail). Both are hard tiling failures in the op, so screen
    # them here and let the builtin take those shapes.
    k, n = weight_t.shape[0], weight_t.shape[-1]
    return k <= _MAX_DIM and n <= _MAX_DIM and (k % 32) % 2 == 0


def _qbmm_v3x(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    scale: torch.Tensor,
    pertoken_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # weight_t is the serving-layout [K, N] NZ.T view; transpose back to the
    # plain-NZ [N, K] tensor the kernel requires, and flag transpose_x2.
    if not _kernel_supports(weight_t):
        return torch_npu.npu_quant_matmul(
            x,
            weight_t,
            scale,
            pertoken_scale=pertoken_scale,
            bias=bias,
            output_dtype=torch.float16,
        )
    return torch.ops._C_ascend.quant_batch_matmul_v3_x(
        x,
        weight_t.transpose(0, 1),
        scale,
        pertoken_scale=pertoken_scale,
        bias=bias,
        transpose_x2=True,
    )


def _qbmm_v3x_fake(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    scale: torch.Tensor,
    pertoken_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], weight_t.shape[-1], dtype=torch.float16)


# Registration must run OUTSIDE any compiled region (direct_register_custom_op
# runs torch.library schema inference, which dynamo refuses to trace), yet
# AFTER the _C_ascend op library is loaded - which is later than this module's
# import in the engine process. ensure_registered() is therefore called from
# the schemes' process_weights_after_loading (eager, post-load, pre-compile);
# the import-time attempt below is a best-effort fast path.
_ENABLED = False


def ensure_registered() -> bool:
    global _ENABLED
    if _ENABLED:
        return True
    if os.environ.get(_ENV, "") != "1":
        return False
    if not hasattr(torch.ops._C_ascend, "quant_batch_matmul_v3_x"):
        return False
    direct_register_custom_op(
        op_name="qbmm_v3x",
        op_func=_qbmm_v3x,
        mutates_args=[],
        fake_impl=_qbmm_v3x_fake,
    )
    _ENABLED = True
    return True


ensure_registered()


def custom_qbmm_enabled() -> bool:
    return _ENABLED


def qbmm(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    scale: torch.Tensor,
    *,
    pertoken_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out = torch.ops.vllm.qbmm_v3x(x, weight_t, scale, pertoken_scale, bias)
    if output_dtype is not None and out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out
