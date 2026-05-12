#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""PTO megakernel drop-in for ``chunk_gated_delta_rule`` — prefill only.

Activated via ``VLLM_ASCEND_PTO_CHUNK_GDN=1``.  Falls back transparently to
the Triton implementation for:
  - Non-zero ``initial_state`` (decode step with prior recurrent state)
  - Missing ``cu_seqlens`` (non-varlen path)
  - Multi-device pipeline-parallel (PCP) groups (world_size > 1)
  - Mismatched Q/K/V head dimensions
  - Non-NPU device

GQA is supported: if ``v.shape[2] > q.shape[2]`` the GQA path is taken.

The PTO path only executes for **prefill** (zero initial_state, cu_seqlens
present).  Decode transparently falls back to Triton so normal inference
(prefill + decode) works correctly end-to-end.
"""
from __future__ import annotations

import torch
from einops import rearrange


def _needs_triton_fallback(
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.LongTensor | None,
) -> bool:
    if initial_state is not None and torch.any(initial_state != 0):
        return True
    return cu_seqlens is None


@torch.compiler.disable
def chunk_gated_delta_rule_pto(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    prebuilt_meta=None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    *,
    _triton_impl,
):
    """PTO megakernel drop-in for ``chunk_gated_delta_rule``.

    Runs the fused Bisheng megakernel for prefill.  Falls back to Triton for
    decode (``initial_state != 0``) so end-to-end inference is unaffected.
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "Use bfloat16 or float16, not float32."
    assert beta.ndim == 3, "beta must be [B, T, H] (head_first=False)."

    if head_first:
        q, k, v, beta, g = (rearrange(x, "b h t ... -> b t h ...") for x in (q, k, v, beta, g))

    if scale is None:
        scale = float(k.shape[-1] ** -0.5)

    if use_qk_l2norm_in_kernel:
        from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd
        q, k = l2norm_fwd(q), l2norm_fwd(k)

    def _triton():
        return _triton_impl(
            q, k, v, g, beta,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state, cu_seqlens=cu_seqlens,
            prebuilt_meta=prebuilt_meta, head_first=False,
            use_qk_l2norm_in_kernel=False,
        )

    if q.device.type != "npu":
        return _triton()
    if _needs_triton_fallback(initial_state, cu_seqlens):
        return _triton()
    if q.shape[3] != v.shape[3]:
        return _triton()
    if v.shape[2] != q.shape[2] and v.shape[2] % q.shape[2] != 0:
        return _triton()
    try:
        from vllm.distributed import get_pcp_group
        if get_pcp_group().world_size > 1:
            return _triton()
    except Exception:
        pass

    from vllm_ascend.ops.pto_chunk_gdn.mega_kernel import run_mega_kernel

    kh = q.shape[2]
    cu32 = cu_seqlens.to(torch.int32).contiguous()
    stream = torch.npu.current_stream()._as_parameter_

    with torch.autograd.profiler.record_function("PTO_mega_kernel"):
        result = run_mega_kernel(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
            g.float(),
            beta.to(torch.float16),
            cu32,
            stream=stream,
            chunk_size=128,
            scale=scale,
            key_heads=kh,
            return_final_state=output_final_state,
        )

    if output_final_state:
        o, fs = result
        o = o.to(q.dtype)
        fs = fs.to(q.dtype)
    else:
        o = result.to(q.dtype)  # type: ignore[assignment]
        fs = None

    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, fs
