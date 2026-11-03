# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_tp_group, tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.utils.math_utils import round_up
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import should_skip_dp_coordination
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import get_mc2_mask, get_mrv2_in_profile_run
from vllm_ascend.attention.dsa_v41 import AscendDSAV41Metadata, AscendDSAV41MetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.ops.triton.decoder_replay import gather_replay_batch
from vllm_ascend.utils import enable_dsa_cp, should_skip_allreduce_across_dp_group
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

from .decoder_replay_layers import AscendDecoderReplayLayers


@dataclass(frozen=True)
class DecoderReplayInputs:
    batch: AscendInputBatch
    block_tables: tuple[torch.Tensor, ...]
    slot_mappings: torch.Tensor
    attn_groups: list[list[AttentionGroup]]
    encoder_replay_start: torch.Tensor


class DecoderReplayState:
    def __init__(self, config: VllmConfig, layers: AscendDecoderReplayLayers, device: torch.device) -> None:
        self.config = config
        self.layers = layers
        num_reqs = config.scheduler_config.max_num_seqs
        capacity = round_up(config.scheduler_config.max_num_batched_tokens, config.parallel_config.tensor_parallel_size)
        self.keeps_rows = np.zeros(num_reqs, dtype=np.bool_)
        self.rows = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.positions = torch.zeros_like(self.rows)
        self.offsets = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        self.dropped_before = CpuGpuBuffer(num_reqs + 1, dtype=torch.int32, device=device, pin_memory=False)
        self.replay_start = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        self.slots: torch.Tensor | None = None
        self.builders: list[tuple[int, list[str], AscendDSAV41MetadataBuilder]] = []
        self.metadata: dict[str, AscendDSAV41Metadata] = {}
        self.num_tokens = 0
        self.num_actual_tokens = 0
        self.dp_tokens: torch.Tensor | None = None

    def _initialize(self, inputs: DecoderReplayInputs) -> None:
        self.slots = torch.empty(
            (inputs.slot_mappings.shape[0], self.rows.numel()), dtype=torch.int64, device=self.rows.device
        )
        for group_id, groups in enumerate(inputs.attn_groups):
            for group in groups:
                if is_circular_kv_cache_spec(group.kv_cache_spec):
                    continue
                builder_cls = group.backend.get_builder_cls()
                if issubclass(builder_cls, AscendDSAV41MetadataBuilder):
                    builder = builder_cls(group.kv_cache_spec, group.layer_names, self.config, self.rows.device)
                    self.builders.append((group_id, group.layer_names, builder))
        self._gather(inputs, warmup=True)

    def _gather(self, inputs: DecoderReplayInputs, *, warmup: bool = False) -> None:
        batch = inputs.batch
        assert self.slots is not None
        gather_replay_batch(
            batch.query_start_loc[: batch.num_reqs + 1],
            self.dropped_before.gpu[: batch.num_reqs + 1],
            batch.seq_lens,
            inputs.encoder_replay_start,
            batch.positions,
            inputs.slot_mappings,
            self.rows,
            self.offsets,
            self.positions,
            self.slots,
            self.replay_start,
            self.num_tokens,
            self.layers.window,
            warmup=warmup,
        )

    def prepare(self, inputs: DecoderReplayInputs, mode: CUDAGraphMode, *, for_capture: bool = False) -> None:
        self.layers.rows = None
        self.layers.context_factory = None
        if self.slots is None:
            self._initialize(inputs)
        if for_capture or mode != CUDAGraphMode.NONE or get_mrv2_in_profile_run():
            return
        parallel = self.config.parallel_config
        if parallel.data_parallel_size > 1 and should_skip_allreduce_across_dp_group(self.config):
            return
        batch = inputs.batch
        num_reqs = batch.num_reqs
        query_lens = np.diff(batch.query_start_loc_np[: num_reqs + 1])
        trim_requests = batch.is_prefilling_np[:num_reqs] & ~self.keeps_rows[batch.idx_mapping_np[:num_reqs]]
        if batch.is_dummy:
            trim_requests = np.zeros(num_reqs, dtype=np.bool_)
        kept_lens = np.where(trim_requests, np.minimum(query_lens, self.layers.window), query_lens)
        trims = bool((kept_lens < query_lens).any())
        self.num_actual_tokens = int(kept_lens.sum())
        self.num_tokens = self.num_actual_tokens
        if enable_dsa_cp():
            self.num_tokens = round_up(self.num_tokens, parallel.tensor_parallel_size)
        self.dp_tokens = None
        if parallel.data_parallel_size > 1:
            agreed = torch.zeros((2, parallel.data_parallel_size), dtype=torch.int32)
            agreed[:, get_dp_group().rank_in_group] = torch.tensor([trims, self.num_tokens])
            if not should_skip_dp_coordination():
                dist.all_reduce(agreed, group=get_dp_group().cpu_group)
            trims = bool(agreed[0].any())
            self.dp_tokens = agreed[1]
        if not trims:
            return
        offsets = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(kept_lens, out=offsets[1:])
        dropped = batch.query_start_loc_np[: num_reqs + 1] - offsets
        self.dropped_before.np[: num_reqs + 1] = dropped
        self.dropped_before.copy_to_gpu(num_reqs + 1)
        self._gather(inputs)
        self.metadata = self._build_metadata(inputs, offsets)
        self.layers.rows = self.rows[: self.num_tokens]
        self.layers.num_actual_rows = self.num_actual_tokens
        self.layers.context_factory = self.forward_context

    def _build_metadata(self, inputs: DecoderReplayInputs, offsets: np.ndarray) -> dict[str, AscendDSAV41Metadata]:
        batch = inputs.batch
        num_reqs = batch.num_reqs
        assert self.slots is not None
        positions = self.positions[: self.num_tokens]
        rope = get_cos_and_sin_dsa(positions, use_cache=False)
        batch_shared: dict[str, Any] = {}
        group_shared: dict[int, dict[str, Any]] = {}
        metadata: dict[str, AscendDSAV41Metadata] = {}
        for group_id, layer_names, builder in self.builders:
            common = AscendCommonAttentionMetadata(
                query_start_loc=self.offsets[: num_reqs + 1],
                query_start_loc_cpu=torch.from_numpy(offsets),
                seq_lens=batch.seq_lens[:num_reqs],
                seq_lens_cpu=torch.from_numpy(batch.seq_lens_np[:num_reqs]),
                num_reqs=num_reqs,
                num_actual_tokens=self.num_actual_tokens,
                num_input_tokens=self.num_tokens,
                max_query_len=max(int(np.diff(offsets).max()), batch.max_query_len or 0),
                max_seq_len=self.config.model_config.max_model_len,
                block_table_tensor=inputs.block_tables[group_id][:num_reqs],
                slot_mapping=self.slots[group_id, : self.num_tokens],
                positions=positions,
                attn_state=batch.attn_state,
                is_prefilling=torch.from_numpy(batch.is_prefilling_np[:num_reqs]),
                replay_start=self.replay_start[:num_reqs],
            )
            value = builder.build(
                0,
                common,
                num_actual_reqs=num_reqs,
                common_v41_batch_metadata=batch_shared,
                common_v41_metadata=group_shared.setdefault(group_id, {}),
                rope_views=rope,
            )
            for task in builder.take_device_metadata_tasks():
                task.run()
            for name in layer_names:
                metadata[name] = value
        return metadata

    @contextmanager
    def forward_context(self) -> Iterator[None]:
        outer = get_forward_context()
        mask = get_mc2_mask()
        saved_mask = mask.clone() if mask is not None else None
        try:
            with set_forward_context(
                self.metadata,
                self.config,
                num_tokens=self.num_tokens,
                num_tokens_across_dp=self.dp_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                context = get_forward_context()
                context.additional_kwargs["eplb_heat_collection_status"] = outer.additional_kwargs.get(
                    "eplb_heat_collection_status", False
                )
                replay_mask = context.additional_kwargs.get("mc2_mask")
                if replay_mask is not None:
                    replay_mask[self.num_actual_tokens :].fill_(False)
                if outer.is_padding is not None:
                    padding = outer.is_padding
                    if self.layers.sequence_parallel:
                        padding = tensor_model_parallel_all_gather(padding, dim=0)
                    context.is_padding = padding.index_select(0, self.rows[: self.num_tokens])
                    context.is_padding[self.num_actual_tokens :].fill_(True)
                    if self.layers.sequence_parallel:
                        tp = get_tp_group()
                        context.is_padding = context.is_padding.chunk(tp.world_size)[tp.rank_in_group]
                yield
        finally:
            if saved_mask is not None:
                mask.copy_(saved_mask)
