# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncsByType,
    get_conv_copy_spec,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

import vllm_ascend.patch.worker.patch_mamba_utils  # noqa: F401
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

_COPY_FUNCS: MambaStateCopyFuncsByType = {
    MambaAttentionBackendEnum.MAMBA2: (get_conv_copy_spec, get_temporal_copy_spec)
}


@dataclass
class _TestConfig:
    block_size: int = 16
    num_blocks: int = 32
    num_layers: int = 2
    max_num_reqs: int = 8
    conv_width: int = 4
    conv_inner_dim: int = 64
    temporal_state_dim: int = 128
    dtype: torch.dtype = torch.float16


def _postprocess_reference(
    cfg: _TestConfig,
    conv_states: list[torch.Tensor],
    temporal_states: list[torch.Tensor],
    block_ids_per_req: list[list[int]],
    mamba_state_indices: list[int],
    running_tokens: list[int],
    accepted_tokens: list[int],
    conv_dim_first: bool,
) -> list[int]:
    """Save aligned states using ordinary CPU tensor copies."""
    normalized_accepted = accepted_tokens.copy()
    for req_idx, (block_ids, src_idx, running, accepted) in enumerate(
        zip(block_ids_per_req, mamba_state_indices, running_tokens, accepted_tokens, strict=True)
    ):
        new_computed = running + accepted - 1
        aligned = new_computed // cfg.block_size * cfg.block_size
        if aligned < running:
            continue

        bias = aligned - running
        dst_idx = aligned // cfg.block_size - 1
        src_block = block_ids[src_idx]
        dst_block = block_ids[dst_idx]
        temporal_src_block = block_ids[src_idx + bias]
        for conv, temporal in zip(conv_states, temporal_states, strict=True):
            # Present both layouts as (block, state_len, dim) for the reference.
            conv_by_position = conv.transpose(1, 2) if conv_dim_first else conv
            # A shifted copy can read and write overlapping slices of one block.
            conv_by_position[dst_block, : cfg.conv_width - bias].copy_(conv_by_position[src_block, bias:].clone())
            temporal[dst_block].copy_(temporal[temporal_src_block])

        if src_idx == dst_idx:
            normalized_accepted[req_idx] = 1
    return normalized_accepted


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_matches_python_postprocess_mamba():
    cfg = _TestConfig()
    device = torch.device("npu:0")
    torch.manual_seed(42)

    num_computed_tokens = [60, 30, 45, 10]
    num_scheduled_tokens = [5, 3, 8, 6]
    num_draft_tokens = [2, 0, 3, 0]
    accepted_tokens = [3, 2, 4, 2]
    mamba_state_indices = [3, 1, 2, 0]
    block_ids_per_req = [
        list(range(8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
    ]
    num_reqs = len(accepted_tokens)
    running_tokens = [
        computed + scheduled - draft
        for computed, scheduled, draft in zip(num_computed_tokens, num_scheduled_tokens, num_draft_tokens, strict=True)
    ]

    layer_names = [f"layer_{i}" for i in range(cfg.num_layers)]
    conv_dim_first = is_conv_state_dim_first()
    conv_shape = (cfg.conv_inner_dim, cfg.conv_width) if conv_dim_first else (cfg.conv_width, cfg.conv_inner_dim)
    temporal_shape = (cfg.temporal_state_dim,)
    mamba_spec = MambaSpec(
        block_size=cfg.block_size,
        shapes=(conv_shape, temporal_shape),
        dtypes=(cfg.dtype, cfg.dtype),
        mamba_cache_mode="align",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=mamba_spec)],
    )

    conv_states = [torch.randn(cfg.num_blocks, *conv_shape, dtype=cfg.dtype, device=device) for _ in layer_names]
    temporal_states = [
        torch.randn(cfg.num_blocks, *temporal_shape, dtype=cfg.dtype, device=device) for _ in layer_names
    ]
    expected_conv = [state.to(device="cpu", copy=True) for state in conv_states]
    expected_temporal = [state.to(device="cpu", copy=True) for state in temporal_states]
    expected_accepted = _postprocess_reference(
        cfg,
        expected_conv,
        expected_temporal,
        block_ids_per_req,
        mamba_state_indices,
        running_tokens,
        accepted_tokens,
        conv_dim_first,
    )

    forward_context = {
        name: SimpleNamespace(kv_cache=[conv, temporal])
        for name, conv, temporal in zip(layer_names, conv_states, temporal_states, strict=True)
    }
    block_table = torch.tensor(block_ids_per_req, dtype=torch.int32, device=device)
    gpu_ctx = MambaSpecDecodeGPUContext.create(
        max_num_reqs=cfg.max_num_reqs,
        kv_cache_config=kv_cache_config,
        copy_funcs=_COPY_FUNCS,
        forward_context=forward_context,
        device=device,
    )
    gpu_ctx.initialize_from_forward_context(kv_cache_config, forward_context, _COPY_FUNCS, [block_table])

    # The align path snapshots the entire persistent accepted-token buffer.
    accepted_tokens_gpu = torch.zeros(cfg.max_num_reqs, dtype=torch.int32, device=device)
    accepted_tokens_gpu[:num_reqs] = torch.tensor(accepted_tokens, dtype=torch.int32, device=device)
    # Model Runner V2 supplies the post-step computed count for each request slot.
    new_computed_tokens = [
        running + accepted - 1 for running, accepted in zip(running_tokens, accepted_tokens, strict=True)
    ]
    gpu_ctx.run_fused_postprocess_align(
        num_reqs=num_reqs,
        num_accepted_tokens_gpu=accepted_tokens_gpu,
        state_idx_gpu=torch.tensor(mamba_state_indices, dtype=torch.int32, device=device),
        new_num_computed_tokens_gpu=torch.tensor(new_computed_tokens, dtype=torch.int32, device=device),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
    )
    torch.accelerator.synchronize()

    for i in range(cfg.num_layers):
        torch.testing.assert_close(conv_states[i].cpu(), expected_conv[i])
        torch.testing.assert_close(temporal_states[i].cpu(), expected_temporal[i])
    torch.testing.assert_close(accepted_tokens_gpu[:num_reqs].cpu(), torch.tensor(expected_accepted, dtype=torch.int32))
