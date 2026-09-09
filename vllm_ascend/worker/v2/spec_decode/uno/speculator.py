# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend MRV2 implementation of UNO shared-model parallel drafting."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers

try:
    from vllm.v1.spec_decode.uno import UNO_LORA_INT_ID, uno_query_layout
    from vllm.v1.spec_decode.uno_noise import fill_uno_noise

    UPSTREAM_UNO_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name != "vllm.v1.spec_decode.uno":
        raise
    UPSTREAM_UNO_AVAILABLE = False
    UNO_LORA_INT_ID = 1_000_003

    def _uno_unavailable(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("Ascend Uno requires a vLLM revision containing PR #55947")

    uno_query_layout = _uno_unavailable
    fill_uno_noise = _uno_unavailable


MAX_ASCEND_UNO_SPECULATIVE_TOKENS = 15


class AscendUnoSpeculator(DraftModelSpeculator):
    """Run the target model once on one seed and K-1 noisy rows per request."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        if not UPSTREAM_UNO_AVAILABLE:
            raise RuntimeError("Ascend Uno requires a vLLM revision containing PR #55947")
        if device.type != "npu":
            raise ValueError("Ascend Uno requires an NPU device")
        assert vllm_config.speculative_config is not None
        num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
        )
        assert num_speculative_tokens is not None
        if num_speculative_tokens > MAX_ASCEND_UNO_SPECULATIVE_TOKENS:
            raise ValueError(
                "Ascend Uno requires num_speculative_tokens <= "
                f"{MAX_ASCEND_UNO_SPECULATIVE_TOKENS}"
            )
        super().__init__(vllm_config, device)
        self.runner = runner
        self.uno_lora_id = UNO_LORA_INT_ID
        self.lora_request = LoRARequest(
            lora_name="uno",
            lora_int_id=self.uno_lora_id,
            lora_path=self.speculative_config.uno_lora_path,
        )
        self._lora_hook: Callable[[tuple[int, ...] | None], None] | None = None
        self._step = 0
        self.query_start_loc_cpu = (
            torch.arange(self.max_num_reqs + 1, dtype=torch.int32)
            * self.num_speculative_steps
        )
        self.sample_indices = torch.arange(
            self.max_num_reqs * self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.sample_columns = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)

        # Ascend metadata builders require a CPU sequence-length mirror.
        del self.input_buffers
        self.input_buffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # UNO shares the target model, whose graph must not capture draft LoRA state.
        return

    def capture(self) -> None:
        return

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return target_model

    def load_model(self, target_model: nn.Module) -> None:
        self.model = target_model
        layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self.draft_attn_layer_names = {
            name
            for name, layer in layers.items()
            if layer.get_kv_cache_spec(self.vllm_config) is not None
        }
        if not self.draft_attn_layer_names:
            raise ValueError("Uno requires target attention layers with a KV cache")
        self._install_lora_hook()

    def _install_lora_hook(self) -> None:
        if self.runner is None:
            raise RuntimeError("Uno requires access to the model runner's LoRA manager")
        runner = self.runner
        runner._ensure_lora_enabled()
        active_mapping: tuple[int, ...] = ()

        def ensure_adapter() -> None:
            if self.uno_lora_id not in runner.lora_manager.list_adapters():
                runner.lora_manager.add_adapter(self.lora_request)

        def set_mapping(mapping: tuple[int, ...] | None) -> None:
            nonlocal active_mapping
            if mapping is None:
                base_mapping = (0,) * len(active_mapping)
                runner._set_active_loras(base_mapping, base_mapping, set())
                active_mapping = ()
                return
            active_mapping = mapping
            ensure_adapter()
            runner._set_active_loras(mapping, mapping, {self.lora_request})

        ensure_adapter()
        self._lora_hook = set_mapping

    @contextmanager
    def _draft_lora(self, mapping: tuple[int, ...]) -> Iterator[None]:
        if self._lora_hook is None:
            raise RuntimeError("Uno draft adapter callback has not been installed")
        try:
            self._lora_hook(mapping)
            yield
        finally:
            self._lora_hook(None)

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        if len(kv_cache_config.kv_cache_groups) != 1:
            raise ValueError("Uno requires one homogeneous full-attention KV group")
        spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        if (
            type(spec) is not FullAttentionSpec
            or spec.sliding_window is not None
            or spec.attention_chunk_size is not None
        ):
            raise ValueError("Uno requires homogeneous full attention")
        if block_tables.cp_size != 1:
            raise ValueError("Uno currently supports only single-device execution")
        self.model_state = model_state
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.target_input_buffers = target_input_buffers
        self.target_attn_groups = target_attn_groups
        self.attn_groups = target_attn_groups

    def _prepare_inputs(
        self,
        input_batch: InputBatch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        prepare_slots: bool = True,
    ) -> tuple[int, tuple[int, ...], np.ndarray]:
        num_reqs = input_batch.num_reqs
        k = self.num_speculative_steps
        num_queries = num_reqs * k
        _, _, mapping = uno_query_layout(
            num_reqs, k, self.uno_lora_id, self.device
        )

        valid_ends = input_batch.query_start_loc[1 : num_reqs + 1] - num_rejected
        last_positions = input_batch.positions[valid_ends.long() - 1]
        offsets = torch.arange(k, dtype=torch.int64, device=self.device)
        positions = last_positions[:, None] + 1 + offsets
        flat_positions = positions.reshape(-1)
        self.input_buffers.positions[:num_queries].copy_(flat_positions)

        req_state_indices = input_batch.idx_mapping[:num_reqs]
        sampled_seeds = last_sampled[req_state_indices]
        prefill_seeds = next_prefill_tokens[req_state_indices]
        seed_ids = torch.where(num_sampled > 0, sampled_seeds, prefill_seeds)
        flat_ids = self.input_buffers.input_ids[:num_queries]
        flat_ids.copy_(seed_ids.repeat_interleave(k))
        is_noise = torch.arange(num_queries, device=self.device) % k != 0
        noise_seeds = (
            torch.arange(num_reqs, device=self.device, dtype=torch.int64)
            + (self.speculative_config.uno_noise_seed & ((1 << 62) - 1))
        ).repeat_interleave(k)
        self._step += 1
        noise_high = self.speculative_config.uno_mask_token_id
        assert noise_high is not None
        fill_uno_noise(flat_ids, is_noise, noise_seeds, self._step, 1, noise_high)

        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(
            self.query_start_loc_cpu[: num_reqs + 1]
        )
        seq_lens = (last_positions + 1 + k).clamp(max=self.max_model_len)
        self.input_buffers.seq_lens[:num_reqs].copy_(seq_lens)
        self.input_buffers.seq_lens[num_reqs:].zero_()

        if prepare_slots:
            block_table = self.block_tables.input_block_tables[0][:num_reqs]
            block_size = self.block_tables.kernel_block_sizes[0]
            block_numbers = positions // block_size
            safe_blocks = block_numbers.clamp(max=block_table.shape[1] - 1)
            block_ids = block_table.gather(1, safe_blocks.long()).to(torch.int64)
            slots = block_ids * block_size + positions % block_size
            valid_slots = (
                (positions < self.max_model_len)
                & (block_numbers < block_table.shape[1])
                & (block_ids != 0)
            )
            slots.masked_fill_(~valid_slots, PAD_SLOT_ID)
            self.block_tables.slot_mappings[0, :num_queries].copy_(
                slots.reshape(-1)
            )
            self.block_tables.slot_mappings[0, num_queries:].fill_(PAD_SLOT_ID)

        self._copy_request_inputs(
            num_reqs, input_batch.idx_mapping, temperature, seeds
        )
        seq_lens_cpu = seq_lens.to(device="cpu", dtype=torch.int32)
        self.input_buffers.seq_lens_cpu[:num_reqs].copy_(seq_lens_cpu)
        self.input_buffers.seq_lens_cpu[num_reqs:].zero_()
        return num_queries, mapping, seq_lens_cpu.numpy()

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync=None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs=None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del attn_metadata, slot_mappings, last_hidden_states, aux_hidden_states
        del dp_sync, mm_inputs, is_profile
        num_reqs = input_batch.num_reqs
        num_queries, mapping, seq_lens_np = self._prepare_inputs(
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            prepare_slots=not skip_attn_for_dummy_run,
        )

        draft_attn_metadata = None
        draft_slot_mappings = None
        if not skip_attn_for_dummy_run:
            block_tables = [
                table[:num_reqs] for table in self.block_tables.input_block_tables
            ]
            slots = self.block_tables.slot_mappings[:, :num_queries]
            draft_attn_metadata = build_attn_metadata(
                attn_groups=self.attn_groups,
                num_reqs=num_reqs,
                num_tokens=num_queries,
                query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc_cpu[: num_reqs + 1],
                max_query_len=self.num_speculative_steps,
                seq_lens=self.input_buffers.seq_lens[:num_reqs],
                max_seq_len=self.max_model_len,
                block_tables=block_tables,
                slot_mappings=slots,
                kv_cache_config=self.kv_cache_config,
                seq_lens_np=seq_lens_np,
                seq_lens_cpu_upper_bound=self.input_buffers.seq_lens_cpu[:num_reqs],
                positions=self.input_buffers.positions[:num_queries],
                attn_state=AscendAttentionState.DecodeOnly,
                is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
                causal=True,
            )
            draft_slot_mappings = build_slot_mappings_by_layer(
                slots, self.kv_cache_config
            )

        self._prepare_eplb_forward(num_queries)
        with self._draft_lora(mapping):
            with set_forward_context(
                draft_attn_metadata,
                self.vllm_config,
                num_tokens=num_queries,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=draft_slot_mappings,
                batch_descriptor=BatchDescriptor(
                    num_tokens=num_queries,
                    has_lora=True,
                    num_active_loras=1,
                ),
            ):
                hidden_states = self.model(
                    input_ids=self.input_buffers.input_ids[:num_queries],
                    positions=self.input_buffers.positions[:num_queries],
                    inputs_embeds=None,
                )
            idx_mapping = input_batch.idx_mapping[:num_reqs].repeat_interleave(
                self.num_speculative_steps
            )
            draft_tokens = self.sample_draft(
                hidden_states[self.sample_indices[:num_queries]],
                self.input_buffers.positions[:num_queries],
                idx_mapping,
                self.temperature,
                self.seeds,
                self.sample_columns[:num_queries],
                self.draft_logits,
            )
        self.draft_tokens[:num_reqs].copy_(
            draft_tokens.view(num_reqs, self.num_speculative_steps)
        )
        return self.draft_tokens[:num_reqs]
