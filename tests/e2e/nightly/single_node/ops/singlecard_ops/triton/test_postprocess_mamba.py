# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.mamba_utils import (
    MambaCopyBuffers,
    MambaSpecDecodeGPUContext,
    collect_mamba_copy_meta,
    do_mamba_copy_block,
)

import vllm_ascend.patch.worker.patch_mamba_utils  # noqa: F401

MambaStateCopyFunc = Callable[..., Any]
_COPY_FUNCS: tuple[MambaStateCopyFunc, ...] = (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)


def postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: Any,
    requests: dict[str, Any],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
):
    assert input_batch.mamba_state_idx_cpu is not None
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    mamba_state_idx_cpu = input_batch.mamba_state_idx_cpu
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        aligned_new_computed_tokens = new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx_cpu[i]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                src_block_idx,
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(copy_bufs)


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


class _MockCpuGpuBuffer:
    def __init__(self, size: int, dtype: torch.dtype, device: torch.device):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.gpu = torch.zeros(size, dtype=dtype, device=device)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)


def _make_scheduler_output(
    num_scheduled_tokens: dict[str, int],
    scheduled_spec_decode_tokens: dict[str, list] | None = None,
) -> SchedulerOutput:
    cached = CachedRequestData.make_empty()
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens or {},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=set(),
    )


def _make_mock_attention(conv_state: torch.Tensor, temporal_state: torch.Tensor) -> MagicMock:
    attention = MagicMock()
    attention.kv_cache = [conv_state, temporal_state]
    return attention


def _make_states(
    cfg: _TestConfig, layer_names: list[str], device: torch.device
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    dict[str, MagicMock],
    dict[str, MagicMock],
]:
    conv_py = [
        torch.randn(cfg.num_blocks, cfg.conv_width, cfg.conv_inner_dim, dtype=cfg.dtype, device=device)
        for _ in layer_names
    ]
    temporal_py = [
        torch.randn(cfg.num_blocks, cfg.temporal_state_dim, dtype=cfg.dtype, device=device) for _ in layer_names
    ]
    conv_gpu = [s.clone() for s in conv_py]
    temporal_gpu = [s.clone() for s in temporal_py]
    fwd_py = {name: _make_mock_attention(c, t) for name, c, t in zip(layer_names, conv_py, temporal_py)}
    fwd_gpu = {name: _make_mock_attention(c, t) for name, c, t in zip(layer_names, conv_gpu, temporal_gpu)}
    return conv_py, temporal_py, conv_gpu, temporal_gpu, fwd_py, fwd_gpu


def _make_kv_cache_config(cfg: _TestConfig, layer_names: list[str]) -> KVCacheConfig:
    mamba_spec = MambaSpec(
        block_size=cfg.block_size,
        shapes=((cfg.conv_width, cfg.conv_inner_dim), (cfg.temporal_state_dim,)),
        dtypes=(cfg.dtype, cfg.dtype),
        mamba_cache_mode="all",
    )
    return KVCacheConfig(
        num_blocks=cfg.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=mamba_spec)],
    )


def _make_input_batch(req_ids: list[str], num_accepted_tokens: list[int], mamba_state_idx: list[int]) -> MagicMock:
    batch = MagicMock()
    batch.req_ids = req_ids
    batch.req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
    batch.num_accepted_tokens_cpu = np.array(num_accepted_tokens, dtype=np.int32)
    batch.mamba_state_idx_cpu = np.array(mamba_state_idx, dtype=np.int32)
    return batch


def _make_requests(
    req_ids: list[str],
    num_computed_tokens: list[int],
    block_ids_per_req: list[list[int]],
) -> dict[str, MagicMock]:
    requests: dict[str, MagicMock] = {}
    for i, req_id in enumerate(req_ids):
        req = MagicMock()
        req.num_computed_tokens = num_computed_tokens[i]
        req.block_ids = {0: block_ids_per_req[i]}
        requests[req_id] = req
    return requests


def _make_copy_bufs(cfg: _TestConfig, kv_cache_config: KVCacheConfig, device: torch.device) -> MambaCopyBuffers:
    return MambaCopyBuffers.create(
        max_num_reqs=cfg.max_num_reqs,
        kv_cache_config=kv_cache_config,
        copy_funcs=_COPY_FUNCS,
        make_buffer=lambda n, dtype: _MockCpuGpuBuffer(n, dtype, device),
    )


def _make_gpu_ctx(cfg: _TestConfig, kv_cache_config: KVCacheConfig, device: torch.device) -> MambaSpecDecodeGPUContext:
    return MambaSpecDecodeGPUContext.create(
        max_num_reqs=cfg.max_num_reqs,
        kv_cache_config=kv_cache_config,
        num_state_types=2,
        device=device,
        make_buffer=lambda n, dtype: _MockCpuGpuBuffer(n, dtype, device),
    )


def _run_gpu_postprocess(
    gpu_ctx: MambaSpecDecodeGPUContext,
    *,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    copy_funcs: tuple,
    block_table: torch.Tensor,
    req_ids: list[str],
    num_accepted_tokens: list[int],
    mamba_state_idx: list[int],
    num_scheduled_tokens: dict[str, int],
    num_computed_tokens: list[int],
    num_draft_tokens: dict[str, int],
    device: torch.device,
) -> None:
    def t(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    gpu_ctx.initialize_from_forward_context(kv_cache_config, forward_context, copy_funcs, [block_table])
    gpu_ctx.run_fused_postprocess(
        num_reqs=len(req_ids),
        num_accepted_tokens_gpu=t(num_accepted_tokens),
        mamba_state_idx_gpu=t(mamba_state_idx),
        num_scheduled_tokens_gpu=t([num_scheduled_tokens[r] for r in req_ids]),
        num_computed_tokens_gpu=t(num_computed_tokens),
        num_draft_tokens_gpu=t([num_draft_tokens.get(r, 0) for r in req_ids]),
    )
    torch.accelerator.synchronize()


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU required")
def test_matches_python_postprocess_mamba():
    cfg = _TestConfig()
    device = torch.device("npu:0")
    torch.manual_seed(42)

    req_ids = ["req_0", "req_1", "req_2", "req_3"]
    num_computed_tokens = [60, 30, 45, 10]
    num_scheduled_tokens = {"req_0": 5, "req_1": 3, "req_2": 8, "req_3": 6}
    num_draft_tokens = {"req_0": 2, "req_1": 0, "req_2": 3, "req_3": 0}
    num_accepted_tokens = [3, 2, 4, 2]
    mamba_state_idx = [3, 1, 2, 0]
    block_ids_per_req = [
        list(range(8)),
        list(range(8, 16)),
        list(range(16, 24)),
        list(range(24, 32)),
    ]

    layer_names = [f"layer_{i}" for i in range(cfg.num_layers)]
    kv_cache_config = _make_kv_cache_config(cfg, layer_names)

    (
        conv_states_py,
        temporal_states_py,
        conv_states_gpu,
        temporal_states_gpu,
        forward_context_py,
        forward_context_gpu,
    ) = _make_states(cfg, layer_names, device)

    scheduler_output = _make_scheduler_output(
        num_scheduled_tokens,
        {k: [None] * v for k, v in num_draft_tokens.items() if v > 0},
    )
    input_batch_py = _make_input_batch(req_ids, num_accepted_tokens.copy(), mamba_state_idx.copy())
    requests = _make_requests(req_ids, num_computed_tokens, block_ids_per_req)
    copy_bufs = _make_copy_bufs(cfg, kv_cache_config, device)

    postprocess_mamba(
        scheduler_output,
        kv_cache_config,
        input_batch_py,
        requests,
        forward_context_py,
        _COPY_FUNCS,
        copy_bufs,
    )
    torch.accelerator.synchronize()

    gpu_ctx = _make_gpu_ctx(cfg, kv_cache_config, device)
    block_table_gpu = torch.zeros(len(req_ids), 8, dtype=torch.int32, device=device)
    for i, block_ids in enumerate(block_ids_per_req):
        block_table_gpu[i, : len(block_ids)] = torch.tensor(block_ids, dtype=torch.int32)

    _run_gpu_postprocess(
        gpu_ctx,
        kv_cache_config=kv_cache_config,
        forward_context=forward_context_gpu,
        copy_funcs=_COPY_FUNCS,
        block_table=block_table_gpu,
        req_ids=req_ids,
        num_accepted_tokens=num_accepted_tokens,
        mamba_state_idx=mamba_state_idx,
        num_scheduled_tokens=num_scheduled_tokens,
        num_computed_tokens=num_computed_tokens,
        num_draft_tokens=num_draft_tokens,
        device=device,
    )

    for i in range(cfg.num_layers):
        torch.testing.assert_close(conv_states_gpu[i], conv_states_py[i])
        torch.testing.assert_close(temporal_states_gpu[i], temporal_states_py[i])

    expected_accepted = torch.tensor(
        input_batch_py.num_accepted_tokens_cpu[: len(req_ids)],
        dtype=torch.int32,
        device=device,
    )
    torch.testing.assert_close(gpu_ctx.num_accepted_tokens_out[: len(req_ids)], expected_accepted)
