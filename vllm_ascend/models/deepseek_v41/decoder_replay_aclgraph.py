# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent, breakable ACL graphs for DeepSeek-V4.1 decoder replay."""

from collections.abc import Callable
from copy import copy
from typing import TYPE_CHECKING

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.utils.math_utils import round_up
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CudaGraphManager

from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

from .decoder_replay_state import DecoderReplayInputs

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.block_table import BlockTables
    from vllm.v1.worker.gpu.input_batch import InputBuffers
    from vllm.v1.worker.utils import AttentionGroup

    from .decoder_replay_state import DecoderReplayState


class DecoderReplayAclGraphManager(CudaGraphManager):
    """Capture replay layers by padded token count on stable NPU buffers."""

    def __init__(
        self,
        config: VllmConfig,
        device: torch.device,
        state: "DecoderReplayState",
        input_ids_dtype: torch.dtype,
    ) -> None:
        compilation = copy(config.compilation_config)
        layers = state.layers
        sizes = compilation.decoder_replay_cudagraph_capture_sizes
        if not sizes:
            bound = min(
                compilation.max_cudagraph_capture_size,
                config.scheduler_config.max_num_batched_tokens,
                config.scheduler_config.max_num_seqs * layers.window,
            )
            sizes, size = [bound], layers.window
            while size < bound:
                sizes.append(size)
                size += max(layers.window, 1 << max(size.bit_length() - 4, 0))
        tp_size = config.parallel_config.tensor_parallel_size if layers.sequence_parallel else 1
        sizes = sorted({round_up(size, tp_size) for size in sizes})
        if not sizes or sizes[-1] > config.scheduler_config.max_num_batched_tokens:
            raise ValueError("decoder replay ACL graph sizes must fit max_num_batched_tokens")
        compilation.cudagraph_capture_sizes = sizes
        compilation.max_cudagraph_capture_size = sizes[-1]
        replay_config = copy(config)
        replay_config.compilation_config = compilation
        super().__init__(replay_config, device, CUDAGraphMode.PIECEWISE, 1)
        self.state = state
        self.runner = BreakableACLGraphWrapper(layers.run_layers, replay_config)

        hf_config = config.model_config.hf_config
        hc = hf_config.hc_mult
        hidden = hf_config.hidden_size
        count = sizes[-1]
        sharded = count // tp_size
        self.input_buffers = (
            torch.zeros((sharded, hc, hidden), dtype=config.model_config.dtype, device=device),
            torch.zeros((sharded, hc), dtype=torch.float32, device=device),
            torch.zeros(count, dtype=torch.int64, device=device),
            torch.zeros(sharded, dtype=input_ids_dtype, device=device),
        )

    def capture_replay_graphs(
        self,
        input_buffers: "InputBuffers",
        block_tables: "BlockTables",
        attn_groups: list[list["AttentionGroup"]],
    ) -> None:
        state = self.state

        def create_forward_fn(desc: BatchExecutionDescriptor, warmup: bool) -> Callable[[CUDAGraphMode], None]:
            num_reqs = min(desc.num_tokens, state.config.scheduler_config.max_num_seqs)
            batch = AscendInputBatch.make_dummy(num_reqs, desc.num_tokens, input_buffers)
            state.prepare(
                DecoderReplayInputs(
                    batch,
                    block_tables.get_dummy_block_tables(num_reqs),
                    block_tables.get_dummy_slot_mappings(desc.num_tokens),
                    attn_groups,
                    torch.zeros(num_reqs, dtype=torch.int32, device=self.device),
                ),
                CUDAGraphMode.NONE,
                capture_desc=desc,
            )
            assert state.layers.rows is not None
            token_count = desc.num_tokens
            tp_size = state.config.parallel_config.tensor_parallel_size if state.layers.sequence_parallel else 1
            sharded_count = token_count // tp_size
            args = (
                self.input_buffers[0][:sharded_count],
                self.input_buffers[1][:sharded_count],
                self.input_buffers[2][:token_count],
                self.input_buffers[3][:sharded_count],
            )

            def forward_fn(mode: CUDAGraphMode) -> None:
                with state.forward_context(mode):
                    self.runner(*args)

            return forward_fn

        self.capture(create_forward_fn, "Capturing decoder replay ACL graphs")
        state.layers.rows = None
        state.layers.context_factory = None

    def run(
        self,
        rows: torch.Tensor,
        states: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor, ...]:
        assert all(state is not None for state in states)
        inputs = tuple(state for state in states if state is not None)
        assert len(inputs) == len(self.input_buffers)
        for buffer, value in zip(self.input_buffers, inputs):
            assert value.shape[0] <= buffer.shape[0]
            buffer[: value.shape[0]].copy_(value)
        count = rows.numel()
        tp_size = self.state.config.parallel_config.tensor_parallel_size if self.state.layers.sequence_parallel else 1
        sharded_count = count // tp_size
        outputs = self.runner(
            self.input_buffers[0][:sharded_count],
            self.input_buffers[1][:sharded_count],
            self.input_buffers[2][:count],
            self.input_buffers[3][:sharded_count],
        )
        return tuple(outputs)
