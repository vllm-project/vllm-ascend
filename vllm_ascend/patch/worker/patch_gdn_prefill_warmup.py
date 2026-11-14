#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# mypy: ignore-errors

import torch
import torch_npu  # noqa: F401
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule

_GDN_PREFILL_WARMUP_TOKENS = 64
_GDN_PREFILL_WARMUP_SIGNATURES: set[tuple[object, ...]] = set()
_ASCEND_GDN_FORWARD_CORE = QwenGatedDeltaNetAttention._forward_core


def _warmup_gdn_prefill_kernels(
    self,
    qkv_or_qkvz: torch.Tensor,
    v_dim: int,
) -> None:
    """Warm Ascend GDN prefill kernels before KV-cache allocation.

    During the V1 profile run, GDN receives no attention metadata. The Ascend
    forward core therefore exits before the gating and chunked-prefill kernels
    execute. Without an explicit warmup, the first real prefill may trigger
    Triton autotuning after most free HBM has already been assigned to KV cache.

    One 64-token warmup is run for each process-level kernel signature.
    Successful signatures are shared across GDN layers, while failed attempts
    remain retryable.
    """
    qkv_dim = int(qkv_or_qkvz.shape[-1]) - int(v_dim)
    if qkv_dim <= 0:
        logger.warning(
            "Skipping Ascend GDN prefill warmup for layer %s: invalid qkv dimension %d (input=%d, v_dim=%d).",
            getattr(self, "prefix", "<unnamed>"),
            qkv_dim,
            qkv_or_qkvz.shape[-1],
            v_dim,
        )
        return

    num_k_heads = self.num_k_heads // self.tp_size
    num_v_heads = self.num_v_heads // self.tp_size
    _, state_dtype = self.get_state_dtype()
    pcp_world_size = get_pcp_group().world_size
    use_fused_chunk = AscendGatedDeltaNetAttention._probe_fused_chunk() and pcp_world_size == 1

    signature = (
        str(qkv_or_qkvz.device),
        qkv_or_qkvz.dtype,
        state_dtype,
        qkv_dim,
        num_k_heads,
        num_v_heads,
        self.head_k_dim,
        self.head_v_dim,
        pcp_world_size,
        use_fused_chunk,
    )
    if signature in _GDN_PREFILL_WARMUP_SIGNATURES:
        return

    device = qkv_or_qkvz.device
    dtype = qkv_or_qkvz.dtype
    warmup_tokens = _GDN_PREFILL_WARMUP_TOKENS

    def _run_warmup() -> None:
        dummy_mixed_qkv = torch.randn(
            warmup_tokens,
            qkv_dim,
            device=device,
            dtype=dtype,
        )
        query, key, value = self.rearrange_mixed_qkv(dummy_mixed_qkv)
        dummy_a = torch.randn(
            warmup_tokens,
            num_v_heads,
            device=device,
            dtype=dtype,
        )
        dummy_b = torch.randn_like(dummy_a)
        g, beta = DeviceOperator.fused_gdn_gating(
            self.A_log,
            dummy_a,
            dummy_b,
            self.dt_bias,
        )
        cu_seqlens = torch.tensor(
            [0, warmup_tokens],
            device=device,
            dtype=torch.int32,
        )

        if use_fused_chunk:
            # Live fused-CANN state layout: [num_seqs, heads, value_dim, key_dim].
            initial_state = torch.zeros(
                1,
                num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
                device=device,
                dtype=state_dtype,
            )
            AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
                scale=self.head_k_dim**-0.5,
            )
        else:
            # Live FLA fallback layout after the SSM state transpose: [N, H, K, V].
            initial_state = torch.zeros(
                1,
                num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                device=device,
                dtype=state_dtype,
            )
            chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                prebuilt_meta=None,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )

    try:
        _run_warmup()
        if device.type == "npu":
            torch.npu.synchronize()
    except Exception:
        logger.warning(
            "Ascend GDN prefill warmup failed for layer %s "
            "(tokens=%d, qkv_dim=%d, dtype=%s, state_dtype=%s, "
            "pcp_world_size=%d, fused=%s). The first real prefill will retry "
            "kernel initialization.",
            getattr(self, "prefix", "<unnamed>"),
            warmup_tokens,
            qkv_dim,
            dtype,
            state_dtype,
            pcp_world_size,
            use_fused_chunk,
            exc_info=True,
        )
    else:
        _GDN_PREFILL_WARMUP_SIGNATURES.add(signature)
        logger.debug(
            "Ascend GDN prefill warmup completed for layer %s (tokens=%d, qkv_dim=%d, fused=%s).",
            getattr(self, "prefix", "<unnamed>"),
            warmup_tokens,
            qkv_dim,
            use_fused_chunk,
        )
    finally:
        if device.type == "npu":
            torch.npu.empty_cache()


def _forward_core_with_prefill_warmup(
    self,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
):
    if get_forward_context().attn_metadata is None:
        _warmup_gdn_prefill_kernels(self, mixed_qkv, 0)
        return
    return _ASCEND_GDN_FORWARD_CORE(
        self,
        mixed_qkv,
        b,
        a,
        core_attn_out,
    )


QwenGatedDeltaNetAttention._warmup_prefill_kernels = _warmup_gdn_prefill_kernels
QwenGatedDeltaNetAttention._forward_core = _forward_core_with_prefill_warmup
