# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/uno.py
"""UNO shared-model parallel drafting for the Ascend model runner (V2).

UNO drafts with the target model itself plus a gated rank-128 LoRA adapter,
so one forward over ``k`` rows per request produces a whole draft block: one
base-weighted seed row and ``k - 1`` adapter rows. It needs no separate draft
checkpoint, hence ``load_draft_model`` returns the target model unchanged.

Stage 1 scope (this file): eager execution, linear (non-tree) verification,
``rejection_sample_method == "standard"``. CUDA-graph capture is not
implemented for the draft pass on Ascend yet, so ``init_cudagraph_manager``
forces ``CUDAGraphMode.NONE``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_factory,
    build_attn_metadata_wrapper,
)
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.spec_decode.uno.lora import (
    base_prompt_mapping,
    draft_lora_mapping,
)
from vllm_ascend.worker.v2.spec_decode.uno.prepare import prepare_uno_inputs

logger = init_logger(__name__)

# LoRA id UNO registers its gate adapter under. High enough that a user adapter
# id collision is practically impossible; the target-side wrapper below still
# handles the case where it happens.
UNO_LORA_ID = 1_000_003

UNO_USER_LORA_ERROR = (
    "UNO does not support user LoRA adapters yet: --enable-lora serves only "
    "UNO's own draft adapter, so runtime LoRA loading and default multimodal "
    "LoRAs are refused (request-specific LoRA requests are refused before queueing)"
)

# Ascend attention backends UNO is validated on. A backend must be listed here
# and must build the draft's uniform-batch decode metadata; anything else is
# refused at startup rather than failing inside the attention kernel.
UNO_ASCEND_ATTN_BACKENDS: tuple[str, ...] = ("ASCEND", "ASCEND_MLA")


def uno_device_refusal(device: torch.device) -> str | None:
    """Why UNO cannot run on ``device``, or None when it can."""
    if device.type != "npu":
        return f"UNO requires an Ascend NPU device, got '{device.type}'"
    return None


def uno_attention_backend_refusal(backend_name: str) -> str | None:
    """Why UNO cannot run on the attention backend ``backend_name``, or None."""
    if backend_name in UNO_ASCEND_ATTN_BACKENDS:
        return None
    return (
        f"UNO on Ascend is validated only with the "
        f"{' and '.join(UNO_ASCEND_ATTN_BACKENDS)} attention backends, not "
        f"{backend_name}."
    )


def validate_uno_config(vllm_config: VllmConfig) -> None:
    """Refuse UNO configurations Ascend does not support, before building.

    The feature-level checks that only need ``additional_config`` live in
    ``AscendConfig._validate_uno``; this covers what needs the full
    ``VllmConfig`` (model shape, attention layout, scheduler sizing).
    """
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    model_config = vllm_config.model_config
    scheduler_config = vllm_config.scheduler_config

    if speculative_config.num_speculative_tokens_per_batch_size is not None:
        raise ValueError("UNO requires a fixed num_speculative_tokens")
    if speculative_config.rejection_sample_method != "standard":
        raise ValueError("UNO requires rejection_sample_method='standard'")
    if speculative_config.use_local_argmax_reduction:
        raise ValueError("UNO requires full draft logits; local argmax is unsupported")
    if speculative_config.enable_adaptive_verification:
        raise ValueError(
            "Adaptive verification is not supported with method='uno'; disable enable_adaptive_verification."
        )
    if model_config.runner_type != "generate":
        raise ValueError("UNO requires a generate (decoder) model")
    if model_config.is_diffusion or model_config.is_multimodal_model or model_config.is_encoder_decoder:
        raise ValueError("UNO requires a text-only decoder model")
    if model_config.is_hybrid or model_config.is_attention_free or model_config.get_sliding_window() is not None:
        raise ValueError("UNO requires a model with full attention in every layer")
    if not vllm_config.scheduler_config.async_scheduling:
        raise ValueError("UNO requires asynchronous scheduling")
    parallel_config = vllm_config.parallel_config
    draft_tp = speculative_config.draft_tensor_parallel_size
    if draft_tp not in (None, parallel_config.tensor_parallel_size):
        raise ValueError(
            "UNO requires the draft to use the target's tensor_parallel_size, got "
            f"draft_tensor_parallel_size={draft_tp} and tensor_parallel_size={parallel_config.tensor_parallel_size}."
        )
    if parallel_config.use_ubatching:
        raise ValueError("UNO does not support dual batch overlap")

    # The draft pass packs k rows per request into one query, so the scheduler
    # must be able to hold max_num_seqs * k draft rows in a single batch.
    required_tokens = scheduler_config.max_num_seqs * speculative_config.num_speculative_tokens
    if scheduler_config.max_num_batched_tokens < required_tokens:
        raise ValueError(
            "UNO requires max_num_batched_tokens >= max_num_seqs * "
            f"num_speculative_tokens ({required_tokens}) for draft LoRA routing"
        )

    attention_config = vllm_config.attention_config
    requested = (attention_config.backend, *attention_config.backend_per_kind.values())
    for backend in requested:
        if backend is not None:
            refusal = uno_attention_backend_refusal(backend.name)
            if refusal is not None:
                raise ValueError(refusal)


class AscendUnoSpeculator(DraftModelSpeculator):
    """UNO draft pass on Ascend: eager, shared target model, gated LoRA."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        refusal = uno_device_refusal(device)
        if refusal is not None:
            raise ValueError(refusal)
        validate_uno_config(vllm_config)
        super().__init__(vllm_config, device)

        self.k = self.num_speculative_steps
        ascend_config = vllm_config.additional_config
        assert isinstance(ascend_config, dict)
        lora_path = ascend_config.get("uno_lora_path")
        if lora_path is None:
            raise ValueError("UNO requires 'uno_lora_path' in additional_config")
        self.lora_request = LoRARequest("uno", UNO_LORA_ID, lora_path)

        # AscendInputBuffers adds `seq_lens_cpu`, which Ascend attention
        # metadata needs. Replace the base buffers created by super().__init__.
        del self.input_buffers
        self.input_buffers: AscendInputBuffers = AscendInputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )

        self._lora_hook: Callable[[tuple[int, int] | None], None] | None = None
        # Draft row -> request-state slot. int64 on purpose: an int32 slot
        # times a per-request stride overflows past 2**31 rows of state.
        self.sample_idx_mapping = torch.full((self.max_num_reqs * self.k,), -1, dtype=torch.int64, device=device)
        self.sample_col = torch.arange(self.k, dtype=torch.int32, device=device).repeat(self.max_num_reqs)
        self._step = 0
        self.num_eager_proposals = 0

    # ---- LoRA routing -----------------------------------------------------

    def set_lora_manager(self, lora_manager: Any) -> None:
        """Route the draft rows through UNO's adapter with the public LoRA API.

        Each draft pass installs its mapping through the worker manager's
        ``set_active_adapters``, the same call the runner makes for a target
        step, so the native Punica metadata is rebuilt in place.

        The mapping is deliberately not restored after the draft: the runner
        installs the next target step's own mapping, and the manager rebuilds
        it because it differs from the draft mapping. A target step whose rows
        carry no adapter takes the ``no_lora`` early-exit in the Ascend Punica
        wrapper, so verification pays no LoRA cost.
        """
        if lora_manager is None:
            raise RuntimeError(
                "UNO requires an initialized LoRA manager. Enable LoRA with --enable-lora "
                "--max-lora-rank 128 so the gated draft adapter can be registered."
            )
        set_active_adapters = lora_manager.set_active_adapters
        set_active_adapters({self.lora_request}, None)

        def set_target_adapters(requests: set[Any], mapping: Any) -> None:
            # A prebuilt EngineCoreRequest can carry UNO's adapter id because it
            # skips the frontend check. Move UNO's adapter to a fresh id first
            # so the request still gets the adapter it asked for.
            uno_id = self.lora_request.lora_int_id
            if any(request.lora_int_id == uno_id and request is not self.lora_request for request in requests):
                self._move_adapter_off(lora_manager, requests)
            set_active_adapters(requests, mapping)

        lora_manager.set_active_adapters = set_target_adapters

        def set_draft_mapping(shape: tuple[int, int] | None) -> None:
            if shape is None:
                return
            num_reqs, num_tokens = shape
            uno_id = self.lora_request.lora_int_id
            set_active_adapters(
                {self.lora_request},
                LoRAMapping(
                    draft_lora_mapping(num_tokens, num_reqs, self.k, uno_id),
                    base_prompt_mapping(num_tokens),
                    is_prefill=True,
                ),
            )

        self._lora_hook = set_draft_mapping

    def _move_adapter_off(self, lora_manager: Any, requests: set[Any]) -> None:
        """Re-register UNO's adapter under an id no current request carries."""
        taken = {request.lora_int_id for request in requests}
        taken |= set(lora_manager.list_adapters())
        old_id = self.lora_request.lora_int_id
        new_id = max(taken | {old_id}) + 1
        moved = LoRARequest(self.lora_request.lora_name, new_id, self.lora_request.lora_path)
        lora_manager.add_adapter(moved)
        lora_manager.remove_adapter(old_id)
        self.lora_request = moved
        logger.warning(
            "A request asked for LoRA adapter id %d, which UNO's adapter was "
            "registered under; UNO's adapter now uses id %d so the request "
            "gets its own adapter.",
            old_id,
            new_id,
        )

    def set_lora_hook(self, hook: Callable[[tuple[int, int] | None], None]) -> None:
        self._lora_hook = hook

    @contextmanager
    def _draft_lora(self, num_reqs: int, num_tokens: int) -> Iterator[None]:
        if self._lora_hook is None:
            raise RuntimeError("UNO adapter routing has not been initialized")
        try:
            self._lora_hook((num_reqs, num_tokens))
            yield
        finally:
            self._lora_hook(None)

    # ---- model / attention ------------------------------------------------

    def load_draft_model(self, target_model: nn.Module, target_attn_layer_names: set[str]) -> nn.Module:
        """UNO drafts on the target model, so there is no draft to load."""
        return target_model

    def load_model(self, target_model: nn.Module) -> None:
        self.model = target_model
        self._validate_local_argmax_reduction()
        from vllm.config import get_layers_from_vllm_config

        layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        # Every target attention layer with a KV cache is also a draft layer:
        # the draft pass writes its own KV rows into the same cache.
        self.draft_attn_layer_names = {
            name for name, layer in layers.items() if layer.get_kv_cache_spec(self.vllm_config) is not None
        }
        if not self.draft_attn_layer_names:
            raise ValueError("UNO requires shared target attention with a KV cache")
        for name in self.draft_attn_layer_names:
            refusal = uno_attention_backend_refusal(layers[name].get_attn_backend().get_name())
            if refusal is not None:
                raise ValueError(refusal)
        self.supports_mm_inputs = False

    def set_attn(
        self,
        model_state: ModelState,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        target_input_buffers: InputBuffers,
        target_attn_groups: list[list[AttentionGroup]],
    ) -> None:
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise ValueError("UNO requires one homogeneous full-attention KV group")
        spec = groups[0].kv_cache_spec
        if (
            type(spec) is not FullAttentionSpec
            or spec.sliding_window is not None
            or spec.attention_chunk_size is not None
        ):
            raise ValueError("UNO requires homogeneous full attention")
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        # Ascend attention backends do not implement in-place draft decode
        # metadata updates, so every draft step rebuilds its metadata eagerly.
        # Nothing to assert here: the eager path below does not rely on it.
        self.supports_decode_metadata_update = False

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        """Stage 1 drafts eagerly: no draft ACL graph on Ascend yet."""
        self.cudagraph_manager = None
        self._graph_attn_metadata = {}
        if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL:
            logger.info_once(
                "UNO drafts eagerly on Ascend: the draft pass does not capture a graph yet. "
                "Set enforce_eager=True to silence this."
            )

    def capture(self) -> None:
        # Eager drafting only; nothing to capture.
        return None

    # ---- draft pass -------------------------------------------------------

    def _build_uno_attn_metadata(
        self,
        num_reqs: int,
        num_tokens: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
    ) -> dict[str, Any] | None:
        """Build uniform-batch decode metadata for the draft rows.

        The draft query is ``k`` rows per request, so the sequence-length upper
        bound must include those rows; otherwise the causal mask truncates the
        draft block. Ascend's metadata builder needs ``positions`` forwarded,
        which the flat upstream path does not do.
        """
        positions = self.input_buffers.positions[:num_tokens]
        draft_seq_lens = (seq_lens_cpu_upper_bound[:num_reqs] + self.k).clamp(max=self.max_model_len)
        with (
            build_attn_metadata_wrapper(),
            build_attn_metadata_factory(
                positions,
                num_tokens,
                self.draft_is_prefilling[:num_reqs],
                attn_state=AscendAttentionState.DecodeOnly,
                parallel_config=self.vllm_config.parallel_config,
            ),
        ):
            return self._build_uniform_attn_metadata(
                _eager_desc(num_reqs, num_tokens),
                num_reqs,
                self.k,
                draft_seq_lens,
                step=self.k,
                causal=True,
            )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor],
    ) -> None:
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=slot_mappings,
        ):
            hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )
        count = num_reqs * self.k
        tokens = self.sample_draft(
            hidden_states[:count],
            self.input_buffers.positions[:count],
            self.sample_idx_mapping[:count],
            self.temperature,
            self.seeds,
            self.sample_col[:count],
            self.draft_logits,
            **_sample_draft_step_kwargs(self),
        )
        self.draft_tokens[:num_reqs].copy_(tokens.view(num_reqs, self.k))

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
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        n = input_batch.num_reqs
        if n == 0:
            return self.draft_tokens[:0]
        count = n * self.k
        self._copy_request_inputs(n, input_batch.idx_mapping, temperature, seeds)

        if dummy_run and skip_attn_for_dummy_run:
            self.input_buffers.input_ids.zero_()
            self.input_buffers.positions.zero_()
            self.sample_idx_mapping.fill_(-1)
            with self._draft_lora(n, count):
                self._generate_draft(n, count, None, {})
            return self.draft_tokens[:n]

        if not (dummy_run or is_profile):
            # Startup and dummy proposals must not move the noise step the
            # first served proposal starts from.
            self._step += 1

        ascend_config = self.vllm_config.additional_config
        assert isinstance(ascend_config, dict)
        mask_token_id = ascend_config.get("uno_mask_token_id")
        noise_high = mask_token_id if mask_token_id is not None else self.vocab_size
        prepare_uno_inputs(
            self.input_buffers,
            self.block_tables.slot_mappings[0],
            self.sample_idx_mapping,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            seeds,
            self.block_tables.input_block_tables[0],
            self.block_tables.block_sizes[0],
            self.k,
            self.max_model_len,
            int(ascend_config.get("uno_noise_seed", 0)),
            int(noise_high),
            self._step,
        )

        if dummy_run:
            self.block_tables.slot_mappings.fill_(PAD_SLOT_ID)
            self.sample_idx_mapping.fill_(-1)

        self.draft_max_seq_len = min(
            int(input_batch.seq_lens_cpu_upper_bound[:n].max()) + self.k,
            self.max_model_len,
        )
        draft_attn = self._build_uno_attn_metadata(
            n,
            count,
            input_batch.seq_lens_cpu_upper_bound,
        )
        slots = build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :count],
            self.kv_cache_config,
        )
        with self._draft_lora(n, count):
            self._generate_draft(n, count, draft_attn, slots)
            if dummy_run or is_profile:
                pass
            else:
                self.num_eager_proposals += 1
        return self.draft_tokens[:n]


def _eager_desc(num_reqs: int, num_tokens: int):
    """Build an eager (no graph) batch descriptor."""
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

    return BatchExecutionDescriptor(CUDAGraphMode.NONE, num_tokens, num_reqs)


def _sample_draft_step_kwargs(speculator: DraftModelSpeculator) -> dict[str, Any]:
    """Return the step-index kwargs ``sample_draft`` takes on this vLLM lane.

    vLLM main added ``spec_step_idx`` / ``num_steps`` for multi-step drafts;
    v0.31.0 does not take them. UNO drafts a single block, so the defaults are
    correct either way.
    """
    if vllm_version_is("0.31.0"):
        return {}
    return {"spec_step_idx": 0, "num_steps": 1}


__all__ = [
    "AscendUnoSpeculator",
    "UNO_LORA_ID",
    "UNO_USER_LORA_ERROR",
    "uno_attention_backend_refusal",
    "uno_device_refusal",
    "validate_uno_config",
]
