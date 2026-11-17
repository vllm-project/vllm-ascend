# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN accepted-prefix commit using stock transferable conv/SSM pages."""

import logging
from typing import Any

import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

from vllm_ascend.ops.triton.gdn.gdn_recoverssm_commit import gdn_recoverssm_commit
from vllm_ascend.ops.triton.mamba.compact_conv_state import compact_conv_state
from vllm_ascend.ops.triton.mamba.prepare_commit_plan import prepare_commit_plan
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


class GDNRecoverSSMCommitContext:
    """One persistent binding per GDN group; replay records are private scratch.

    State caches retain the normal GDN KV page layout. Replay records are
    indexed by source page and overwritten on every verify, so they never
    have to cross the P/D connector.
    """

    layers: tuple[Any, ...]
    bindings: tuple[tuple[torch.Tensor, ...], ...]
    states: tuple[torch.Tensor, ...]
    convs: tuple[torch.Tensor, ...]
    replays: list[torch.Tensor]
    spec_query_len: int
    max_num_reqs: int
    conv_history_len: int
    replay_bytes: int
    state_table: torch.Tensor
    conv_tables: tuple[torch.Tensor, ...]
    plan: tuple[torch.Tensor, ...]
    commit_steps: int

    @classmethod
    def create(cls, layers, *, spec_query_len: int, max_num_reqs: int):
        if not layers or spec_query_len < 1 or max_num_reqs < 1:
            raise ValueError("GDN RecoverSSM needs layers and positive capacities")
        init_device_properties_triton()
        self = cls()
        self.layers = tuple(layers)
        self.bindings = tuple(tuple(layer.kv_cache) for layer in layers)
        self.states = tuple(binding[1] for binding in self.bindings)
        self.convs = tuple(
            binding[0] if is_conv_state_dim_first() else binding[0].transpose(-1, -2) for binding in self.bindings
        )
        state, conv = self.states[0], self.convs[0]
        pages, heads, value_dim, key_dim = state.shape
        self.spec_query_len = spec_query_len
        self.max_num_reqs = max_num_reqs
        self.conv_history_len = conv.shape[2] - spec_query_len + 1
        if self.conv_history_len < 1:
            raise ValueError("GDN convolution cache is shorter than its speculative window")
        self.replays = []
        self.replay_bytes = len(layers) * pages * heads * spec_query_len * (value_dim + key_dim + 1) * 4
        logging.getLogger(__name__).info(
            "Qwen GDN RecoverSSM private replay allocation: %.3f GiB, layers=%d pages=%d local_heads=%d Q=%d",
            self.replay_bytes / 2**30,
            len(layers),
            pages,
            heads,
            spec_query_len,
        )
        for layer, checkpoint, convolution in zip(layers, self.states, self.convs):
            if (
                checkpoint.shape != state.shape
                or checkpoint.dtype != state.dtype
                or checkpoint.stride()[1:] != state.stride()[1:]
                or checkpoint.dtype not in (torch.float32, torch.bfloat16)
                or checkpoint.device != state.device
                or convolution.shape != conv.shape
            ):
                raise ValueError("GDN RecoverSSM layers require matching BF16/FP32 state and convolution layouts")
            try:
                replay = torch.empty(
                    (pages, heads, spec_query_len, value_dim + key_dim + 1), dtype=torch.float32, device=state.device
                )
            except torch.OutOfMemoryError as error:
                raise RuntimeError(
                    f"Qwen GDN RecoverSSM needs {self.replay_bytes / 2**30:.3f} GiB private replay memory; "
                    "lower --gpu-memory-utilization to leave space outside the KV cache budget"
                ) from error
            layer._qwen_gdn_replay = replay
            self.replays.append(replay)
        self.state_table = torch.tensor(
            [
                [[tensor.data_ptr() for tensor in tensors], [tensor.stride(0) for tensor in tensors]]
                for tensors in (self.states, self.replays)
            ],
            dtype=torch.int64,
            device=state.device,
        )
        self.conv_tables = tuple(
            torch.tensor(values, dtype=torch.int64, device=state.device)
            for values in (
                [tensor.data_ptr() for tensor in self.convs],
                [tensor.stride(0) for tensor in self.convs],
                [tensor.stride(1) for tensor in self.convs],
                [tensor.stride(2) for tensor in self.convs],
            )
        )
        self.plan = tuple(torch.empty(max_num_reqs, dtype=torch.int32, device=state.device) for _ in range(6))
        self.commit_steps = 0
        return self

    def commit(self, accepted, state_indices, query_start_loc, request_indices=None):
        if any(
            any(current is not bound for current, bound in zip(layer.kv_cache, binding))
            for layer, binding in zip(self.layers, self.bindings)
        ):
            raise RuntimeError("GDN RecoverSSM caches were rebound; rebuild the commit context")
        count = state_indices.numel()
        if count > self.max_num_reqs:
            raise ValueError("GDN RecoverSSM commit exceeds its request capacity")
        if count == 0:
            return
        lens, state_lens, boundary_lens, final, boundary, recovery = (tensor[:count] for tensor in self.plan)
        prepare_commit_plan(
            accepted,
            state_indices,
            query_start_loc,
            lens,
            state_lens,
            boundary_lens,
            final,
            boundary,
            recovery,
            spec_query_len=self.spec_query_len,
            null_block_id=0,
            request_indices=request_indices,
        )
        gdn_recoverssm_commit(
            self.states[0],
            self.replays[0],
            self.state_table,
            state_indices,
            lens,
            final,
            boundary,
            recovery,
            spec_query_len=self.spec_query_len,
            align_mode=False,
        )
        compact_conv_state(
            self.convs[0],
            self.conv_tables[0],
            self.conv_tables[1],
            self.conv_tables[2],
            self.conv_tables[3],
            state_indices,
            lens,
            final,
            boundary,
            recovery,
            conv_history_len=self.conv_history_len,
            align_mode=False,
            dim_stride=None,
            null_block_id=0,
        )
        self.commit_steps += 1
