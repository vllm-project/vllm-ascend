# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend Model Runner V1 support for Uno shared-model drafting."""

from collections.abc import Callable
from dataclasses import replace

import torch
import torch.nn as nn
from typing_extensions import override
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionState,
)
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

try:
    from vllm.v1.spec_decode.uno import (
        UNO_LORA_INT_ID,
        UnoProposer,
        uno_query_layout,
    )
    from vllm.v1.spec_decode.uno_noise import fill_uno_noise

    UPSTREAM_UNO_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name != "vllm.v1.spec_decode.uno":
        raise

    UPSTREAM_UNO_AVAILABLE = False
    UNO_LORA_INT_ID = 1_000_003

    class UnoProposer:  # type: ignore[no-redef]
        """Import placeholder for vLLM revisions predating Uno."""

    def _uno_unavailable(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("Ascend Uno requires a vLLM revision containing PR #55947")

    uno_query_layout = _uno_unavailable
    fill_uno_noise = _uno_unavailable

# Ascend fused-infer-attention supports at most 16 query tokens per request in
# its speculative TND layout. Uno's target verification consumes K + 1 rows.
MAX_ASCEND_UNO_SPECULATIVE_TOKENS = 15


class AscendUnoProposer(UnoProposer, AscendSpecDecodeBaseProposer):
    """Run Uno's eager draft pass on the shared target model on Ascend.

    The upstream proposer owns the Uno algorithm. This subclass replaces only
    the CUDA-specific construction and forward context, and preserves Ascend's
    attention metadata and padded-batch helpers through the second base class.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        if not UPSTREAM_UNO_AVAILABLE:
            raise RuntimeError("Ascend Uno requires a vLLM revision containing PR #55947")
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.use_uno()
        if device.type != "npu":
            raise ValueError("Ascend Uno requires an NPU device")
        num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens
        assert num_speculative_tokens is not None
        if num_speculative_tokens > MAX_ASCEND_UNO_SPECULATIVE_TOKENS:
            raise ValueError(f"Ascend Uno requires num_speculative_tokens <= {MAX_ASCEND_UNO_SPECULATIVE_TOKENS}")

        self.runner = runner
        AscendSpecDecodeBaseProposer.__init__(
            self,
            vllm_config,
            device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self.uno_lora_id = UNO_LORA_INT_ID
        self.lora_request = LoRARequest(
            lora_name="uno",
            lora_int_id=self.uno_lora_id,
            lora_path=self.speculative_config.uno_lora_path,
        )
        self._lora_hook: Callable[[tuple[int, ...] | None], None] | None = None
        self._pending_lora_map: tuple[int, ...] = ()
        self._step = 0
        self.max_query_tokens = self.max_batch_size * self.num_speculative_tokens
        self.input_ids = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.full(
            (self.max_query_tokens,),
            PADDING_SLOT_ID,
            dtype=torch.int64,
            device=device,
        )

    @override
    def load_model(self, target_model: nn.Module) -> None:
        """Share target weights and accept only Ascend full-attention layers."""
        self.model = target_model
        all_attn = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self._draft_attn_layer_names = {
            name for name, layer in all_attn.items() if layer.get_kv_cache_spec(self.vllm_config) is not None
        }
        if not self._draft_attn_layer_names:
            raise ValueError("Ascend Uno requires target attention layers with a KV cache")
        for name in self._draft_attn_layer_names:
            backend = all_attn[name].get_attn_backend()
            if not issubclass(backend, AscendAttentionBackend):
                raise ValueError("Ascend Uno currently requires AscendAttentionBackend")
        self.attn_layer_names = sorted(self._draft_attn_layer_names)

    @override
    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        # UnoProposer validates that the shared cache is one homogeneous group;
        # its super() then resolves to AscendSpecDecodeBaseProposer here.
        super().initialize_attn_backend(kv_cache_config, kernel_block_sizes)

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: AscendCommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, AscendCommonAttentionMetadata]:
        batch_size = cad.batch_size()
        k = self.num_speculative_tokens
        num_queries = batch_size * k
        is_noise, sample_indices, self._pending_lora_map = uno_query_layout(
            batch_size,
            k,
            self.uno_lora_id,
            self.device,
        )

        valid_ends = cad.query_start_loc[1 : batch_size + 1]
        seq_lens = cad.seq_lens
        if num_rejected_tokens_gpu is not None:
            valid_ends = valid_ends - num_rejected_tokens_gpu
            seq_lens = seq_lens - num_rejected_tokens_gpu
        first_positions = target_positions[valid_ends.long() - 1] + 1
        offsets = torch.arange(k, device=self.device)
        positions = first_positions[:, None] + offsets
        self.positions[:num_queries].copy_(positions.reshape(-1))
        self.input_ids[:num_queries].copy_(next_token_ids.repeat_interleave(k))

        req_seeds = (
            torch.arange(batch_size, device=self.device, dtype=torch.int64)
            + (self.speculative_config.uno_noise_seed & ((1 << 62) - 1))
        ).repeat_interleave(k)
        self._step += 1
        noise_high = self.speculative_config.uno_mask_token_id
        assert noise_high is not None
        fill_uno_noise(
            self.input_ids[:num_queries],
            is_noise,
            req_seeds,
            self._step,
            1,
            noise_high,
        )

        block_numbers = positions // self.block_size
        block_table = cad.block_table_tensor[:batch_size]
        safe_blocks = block_numbers.clamp(max=block_table.shape[1] - 1)
        block_ids = block_table.gather(1, safe_blocks.long()).to(torch.int64)
        slots = block_ids * self.block_size + positions % self.block_size
        valid_slots = (positions < self.max_model_len) & (block_numbers < block_table.shape[1]) & (block_ids != 0)
        slots.masked_fill_(~valid_slots, PADDING_SLOT_ID)
        self._slot_mapping_buffer[:num_queries].copy_(slots.reshape(-1))

        query_start_loc_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * k
        query_start_loc = query_start_loc_cpu.to(self.device, non_blocking=True)
        upper = cad.seq_lens_cpu_upper_bound
        seq_lens_cpu = None
        if num_rejected_tokens_gpu is None:
            # The synchronous runner has already reconciled rejected tokens on
            # the host. Preserve that exact mirror so FIA metadata construction
            # does not introduce an NPU-to-CPU synchronization in every draft.
            base_seq_lens_cpu = cad._seq_lens_cpu if cad._seq_lens_cpu is not None else cad.seq_lens_cpu
            if base_seq_lens_cpu is not None:
                seq_lens_cpu = base_seq_lens_cpu[:batch_size] + k
        new_cad = replace(
            cad,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens + k,
            _seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=None if upper is None else upper + k,
            num_reqs=batch_size,
            num_actual_tokens=num_queries,
            num_input_tokens=num_queries,
            max_query_len=k,
            max_seq_len=cad.max_seq_len + k,
            block_table_tensor=block_table,
            slot_mapping=self._slot_mapping_buffer[:num_queries],
            causal=True,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            positions=self.positions[:num_queries],
            positions_cpu=None,
            attn_state=AscendAttentionState.ChunkedPrefill,
            decode_token_per_req=k,
            context_parallel_metadata=None,
        )
        return num_queries, sample_indices, new_cad

    @override
    @torch.inference_mode()
    def propose(
        self,
        num_speculative_tokens: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: AscendCommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        del mm_embed_inputs, slot_mappings
        self._last_draft_probs = None
        self._pending_lora_map = ()
        k = num_speculative_tokens
        if not 0 <= k <= self.speculative_config.num_speculative_tokens:
            raise ValueError("Uno draft length exceeds the configured capacity")
        self.num_speculative_tokens = k
        if k == 0:
            return torch.empty(
                common_attn_metadata.batch_size(),
                0,
                device=self.device,
                dtype=torch.int64,
            )

        try:
            num_tokens, sample_indices, cad = self.set_inputs_first_pass(
                target_token_ids,
                next_token_ids,
                target_positions,
                target_hidden_states,
                token_indices_to_sample,
                common_attn_metadata,
                num_rejected_tokens_gpu,
            )
            _, per_layer_metadata = self.build_per_group_and_layer_attn_metadata(cad)
            with self._draft_lora(self._pending_lora_map):
                with set_ascend_forward_context(
                    per_layer_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens,
                    num_actual_tokens=num_tokens,
                    aclgraph_runtime_mode=CUDAGraphMode.NONE,
                    model_instance=self.model,
                    is_draft_model=False,
                ):
                    hidden_states = self.model(
                        input_ids=self.input_ids[:num_tokens],
                        positions=self.positions[:num_tokens],
                        inputs_embeds=None,
                    )
                draft_ids, draft_probs = self._sample_draft_tokens(
                    hidden_states[sample_indices],
                    sampling_metadata,
                )
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1,
                    k,
                    draft_probs.shape[-1],
                ).contiguous()
            return draft_ids.view(-1, k)
        except Exception:
            self._last_draft_probs = None
            raise
        finally:
            self._pending_lora_map = ()

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile: bool = False,
        **_: object,
    ) -> None:
        del with_prefill, num_reqs, num_tokens_across_dp
        del aclgraph_runtime_mode, batch_descriptor, dummy_compute_logits
        if in_graph_capturing or num_tokens == 0:
            return
        k = self.speculative_config.num_speculative_tokens
        batch_size = min(self.max_batch_size, (num_tokens + k - 1) // k)
        num_queries = batch_size * k
        _, _, mapping = uno_query_layout(
            batch_size,
            k,
            self.uno_lora_id,
            self.device,
        )
        self.input_ids[:num_queries].zero_()
        self.positions[:num_queries].zero_()
        self._slot_mapping_buffer[:num_queries].fill_(PADDING_SLOT_ID)
        with self._draft_lora(mapping):
            with set_ascend_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_queries,
                num_actual_tokens=num_queries,
                in_profile_run=is_profile,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                model_instance=self.model,
                is_draft_model=False,
            ):
                hidden_states = self.model(
                    input_ids=self.input_ids[:num_queries],
                    positions=self.positions[:num_queries],
                    inputs_embeds=None,
                )
            logits = self.model.compute_logits(hidden_states)
            if self._enable_probabilistic_draft_probs:
                assert self.runner is not None
                metadata = replace(
                    self.runner.input_batch.sampling_metadata,
                    temperature=torch.ones(num_queries, device=self.device),
                    all_greedy=False,
                    all_random=True,
                )
                self._sample_from_logits(logits, metadata)
