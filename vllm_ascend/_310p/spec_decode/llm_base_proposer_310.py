#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata

from vllm_ascend._310p.attention.dflash_hybrid_draft_graph_safe_attention import (
    DFlashHybridDraftAttentionInputs310,
    copy_dflash_hybrid_draft_attention_inputs_310,
)
from vllm_ascend._310p.attention.metadata_builder import (
    get_dflash_hybrid_draft_attention_inputs_310,
)
from vllm_ascend._310p.dflash_full_and_piecewise import (
    is_310p_dflash_full_and_piecewise,
)
from vllm_ascend._310p.dflash_full_decode_only import (
    is_310p_dflash_full_decode_only,
)
from vllm_ascend._310p.ops.rotary_embedding import (
    AscendRotaryEmbedding310,
    clear_full_decode_draft_rope_310,
    get_full_decode_draft_rope_buffers_310,
    prepare_full_decode_draft_rope_310,
)
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

_original_run_merged_draft = AscendSpecDecodeBaseProposer._run_merged_draft
_original_load_model = AscendSpecDecodeBaseProposer.load_model
_original_compute_draft_step_slot_mapping = (
    AscendSpecDecodeBaseProposer._compute_draft_step_slot_mapping
)


class DFlashHybridDraftForwardACLGraphWrapper310(ACLGraphWrapper):
    """Refresh device metadata captured by the six-layer Draft FULL island."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hybrid_draft_staging_by_descriptor_310: dict[
            Any,
            dict[str, DFlashHybridDraftAttentionInputs310],
        ] = {}

    @staticmethod
    def _collect_private_inputs_310(
        per_layer_metadata: dict[str, Any],
    ) -> dict[str, DFlashHybridDraftAttentionInputs310]:
        result = {}
        for layer_name, metadata in per_layer_metadata.items():
            inputs = get_dflash_hybrid_draft_attention_inputs_310(metadata)
            if inputs is None:
                raise RuntimeError(
                    "310P DFlash Hybrid Draft FULL metadata is missing the "
                    f"private device contract for {layer_name}"
                )
            result[layer_name] = inputs
        if not result:
            raise RuntimeError(
                "310P DFlash Hybrid Draft FULL received no layer metadata"
            )
        return result

    @staticmethod
    def _copy_runtime_inputs_310(
        captured: dict[str, DFlashHybridDraftAttentionInputs310],
        current: dict[str, DFlashHybridDraftAttentionInputs310],
    ) -> None:
        if captured.keys() != current.keys():
            raise RuntimeError(
                "310P DFlash Hybrid Draft FULL layer metadata changed after capture"
            )
        copied: set[tuple[int, int]] = set()
        for layer_name, destination in captured.items():
            source = current[layer_name]
            identity = (
                destination.valid_num_reqs.data_ptr(),
                source.valid_num_reqs.data_ptr(),
            )
            if identity in copied:
                continue
            copied.add(identity)
            copy_dflash_hybrid_draft_attention_inputs_310(
                destination,
                source,
            )

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        descriptor = forward_context.batch_descriptor
        entry = self.concrete_aclgraph_entries.get(descriptor)
        is_full = (
            is_310p_dflash_full_and_piecewise(self.vllm_config)
            and forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        )
        current_inputs = None
        if is_full:
            current_metadata = forward_context.attn_metadata
            if not isinstance(current_metadata, dict):
                raise RuntimeError(
                    "310P DFlash Hybrid Draft FULL replay requires one "
                    "per-layer attention metadata mapping for the current "
                    "Draft substep"
                )
            current_inputs = self._collect_private_inputs_310(current_metadata)
            captured_inputs = self._hybrid_draft_staging_by_descriptor_310.get(
                descriptor
            )
            if (
                captured_inputs is not None
                and entry is not None
                and entry.aclgraph is not None
            ):
                self._copy_runtime_inputs_310(
                    captured_inputs,
                    current_inputs,
                )
            logger.debug(
                "[310p-dflash-full-and-piecewise/draft-island] "
                "event=%s descriptor_tokens=%d "
                "layer_metadata=%d",
                (
                    "device-metadata-refresh"
                    if captured_inputs is not None
                    else "device-metadata-capture-source"
                ),
                int(descriptor.num_tokens),
                len(current_metadata),
            )
        result = super().__call__(*args, **kwargs)
        if is_full and current_inputs is not None:
            entry = self.concrete_aclgraph_entries.get(descriptor)
            if (
                entry is not None
                and entry.aclgraph is not None
                and descriptor
                not in self._hybrid_draft_staging_by_descriptor_310
            ):
                self._hybrid_draft_staging_by_descriptor_310[descriptor] = (
                    current_inputs
                )
        return result


class AscendSpecDecodeBaseProposer310(AscendSpecDecodeBaseProposer):
    """310P proposer overrides for NPU-specific spec-decode workarounds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_hybrid_draft_slot_mapping_310()

    def load_model(self, model: torch.nn.Module) -> None:
        """Load normally, then narrow Hybrid FULL to Draft model forward."""
        _original_load_model(self, model)
        AscendSpecDecodeBaseProposer310._install_hybrid_draft_forward_full_island_310(
            self
        )

    def _install_hybrid_draft_forward_full_island_310(self) -> None:
        """Move Hybrid FULL capture from merged proposal to model forward."""
        if not (
            is_310p_dflash_full_and_piecewise(self.vllm_config)
            and self.use_cuda_graph
        ):
            return

        if isinstance(self._runnable, ACLGraphWrapper):
            self._runnable = self._runnable.unwrap()
        if isinstance(self.model, ACLGraphWrapper):
            return

        self.model = DFlashHybridDraftForwardACLGraphWrapper310(
            self.model,
            self.vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            use_eagle=self.use_eagle,
            enable_enpu=self.enable_enpu,
            component="draft",
            retained_input_provider=self._full_decode_draft_retained_inputs,
        )

    def _initialize_hybrid_draft_slot_mapping_310(self) -> None:
        """Allocate stable Hybrid Draft slot-mapping work buffers.

        Production 310P workers patch selected methods onto the public proposer
        class instead of instantiating this subclass directly. Keep allocation
        in a standalone method so that patch_idex_310 can invoke it immediately
        after the original public constructor, before graph capture starts.
        """
        if not is_310p_dflash_full_and_piecewise(self.vllm_config):
            return
        reference = self.slot_mapping_group[0]
        capacity = int(reference.shape[0])
        device = reference.device
        if not hasattr(self, "_hybrid_draft_slot_gather_indices_310"):
            self._hybrid_draft_slot_gather_indices_310 = torch.empty(
                (capacity, 1),
                dtype=torch.int64,
                device=device,
            )
            for suffix, dtype in (("i32", torch.int32), ("i64", torch.int64)):
                setattr(
                    self,
                    f"_hybrid_draft_slot_block_ids_{suffix}_310",
                    torch.empty(capacity, dtype=dtype, device=device),
                )
                setattr(
                    self,
                    f"_hybrid_draft_slot_starts_{suffix}_310",
                    torch.empty(capacity, dtype=dtype, device=device),
                )
                setattr(
                    self,
                    f"_hybrid_draft_slot_offsets_{suffix}_310",
                    torch.empty(capacity, dtype=dtype, device=device),
                )
                setattr(
                    self,
                    f"_hybrid_draft_slot_output_{suffix}_310",
                    torch.empty(capacity, dtype=dtype, device=device),
                )

    def _compute_draft_step_slot_mapping(
        self,
        block_table_for_slot: torch.Tensor,
        clamped_positions: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        if not is_310p_dflash_full_and_piecewise(self.vllm_config):
            return _original_compute_draft_step_slot_mapping(
                self,
                block_table_for_slot,
                clamped_positions,
                block_size,
            )

        logical_positions = (
            clamped_positions[0] if self.uses_mrope else clamped_positions
        )
        num_tokens = int(logical_positions.shape[0])
        capacity = int(self._hybrid_draft_slot_gather_indices_310.shape[0])
        if num_tokens > capacity:
            raise RuntimeError(
                "310P DFlash Hybrid Draft slot mapping exceeds persistent "
                f"capacity: tokens={num_tokens}, capacity={capacity}"
            )

        if block_table_for_slot.dtype == torch.int64:
            table_suffix = "i64"
        elif block_table_for_slot.dtype == torch.int32:
            table_suffix = "i32"
        else:
            raise TypeError(
                "310P DFlash Hybrid Draft slot mapping requires int32/int64, "
                f"got table={block_table_for_slot.dtype}, "
                f"positions={logical_positions.dtype}"
            )

        gather_indices = self._hybrid_draft_slot_gather_indices_310[
            :num_tokens
        ]
        block_ids = getattr(
            self,
            f"_hybrid_draft_slot_block_ids_{table_suffix}_310",
        )[:num_tokens]
        # The persistent destination consumed by attention is int32. Keep the
        # complete address arithmetic in that dtype as well: the 310P int64
        # Add kernel faults for unaligned dynamic lengths (for example C7's
        # 112 slots), while all legal cache slot ids already fit the public
        # int32 slot_mapping contract.
        block_starts = self._hybrid_draft_slot_starts_i32_310[:num_tokens]
        block_offsets = self._hybrid_draft_slot_offsets_i32_310[:num_tokens]
        output = self._hybrid_draft_slot_output_i32_310[:num_tokens]

        gather_indices.copy_(
            (logical_positions // block_size).view(-1, 1),
        )
        torch.gather(
            block_table_for_slot,
            dim=1,
            index=gather_indices,
            out=block_ids.view(-1, 1),
        )
        block_starts.copy_(block_ids)
        block_starts.mul_(block_size)
        block_offsets.copy_(logical_positions)
        block_offsets.remainder_(block_size)
        block_offsets.neg_()
        torch.sub(block_starts, block_offsets, out=output)
        return output

    def _prepare_full_decode_draft_rope(
        self,
        *,
        query_positions: torch.Tensor,
        query_actual_tokens: int,
        descriptor_tokens: int,
        runtime_mode: CUDAGraphMode,
    ) -> bool:
        """Refresh stable query/context RoPE inputs for compiled FDO draft."""
        runner = getattr(self, "runner", None)
        scope_config = getattr(runner, "vllm_config", self.vllm_config)
        uses_full_decode_only = is_310p_dflash_full_decode_only(scope_config)
        uses_hybrid_graph = is_310p_dflash_full_and_piecewise(scope_config)
        uses_precomputed_rope = uses_full_decode_only or uses_hybrid_graph
        if getattr(self, "method", None) != "dflash" or not uses_precomputed_rope:
            return False
        # FDO and Hybrid compile the rotary branch while FULL precomputed
        # buffers are selected. The compiled callable keeps those stable
        # addresses when runtime dispatch later selects NONE or PIECEWISE, so
        # every call in either configured mode must refresh the same buffers
        # from its current logical positions.
        query_descriptor_tokens = descriptor_tokens
        if uses_hybrid_graph:
            max_query_tokens = getattr(self, "max_query_tokens", None)
            if isinstance(max_query_tokens, int) and max_query_tokens > 0:
                query_descriptor_tokens = min(
                    query_descriptor_tokens,
                    max_query_tokens,
                )
        if query_actual_tokens > query_descriptor_tokens:
            raise RuntimeError(
                "310P DFlash FULL draft RoPE active query extent exceeds "
                "the Draft query capacity: "
                f"actual={query_actual_tokens}, "
                f"query_descriptor={query_descriptor_tokens}, "
                f"target_descriptor={descriptor_tokens}"
            )
        if runtime_mode != CUDAGraphMode.FULL:
            query_positions = self._get_positions(query_descriptor_tokens)
        elif query_positions.ndim == 1 and query_positions.shape[0] > query_descriptor_tokens:
            query_positions = query_positions[:query_descriptor_tokens]
        if (
            query_positions.ndim != 1
            or query_positions.shape[0] != query_descriptor_tokens
        ):
            raise RuntimeError(
                "310P DFlash FULL draft RoPE requires one Draft-query-sized "
                f"query position vector, got shape={tuple(query_positions.shape)}, "
                f"query_descriptor={query_descriptor_tokens}, "
                f"target_descriptor={descriptor_tokens}"
            )

        context_positions_buffer = self._context_positions_buffer
        context_actual_tokens = int(getattr(self, "_dflash_num_context", 0))
        context_descriptor_tokens = max(
            descriptor_tokens,
            context_actual_tokens,
        )
        if context_positions_buffer.shape[0] < context_descriptor_tokens:
            raise RuntimeError(
                "310P DFlash FULL context-position buffer is smaller than the "
                f"descriptor: capacity={context_positions_buffer.shape[0]}, "
                f"descriptor={context_descriptor_tokens}"
            )
        context_positions = context_positions_buffer[:context_descriptor_tokens]
        if not 0 <= context_actual_tokens <= context_descriptor_tokens:
            raise RuntimeError(
                "310P DFlash FULL context extent is outside the descriptor: "
                f"actual={context_actual_tokens}, "
                f"descriptor={context_descriptor_tokens}"
            )

        draft_rotary = getattr(self, "_full_decode_draft_rotary_310", None)
        if draft_rotary is None:
            draft_rotary = next(
                (module for module in self.model.modules() if isinstance(module, AscendRotaryEmbedding310)),
                None,
            )
            if draft_rotary is None:
                raise RuntimeError("310P DFlash FULL draft model has no Ascend rotary embedding")
            self._full_decode_draft_rotary_310 = draft_rotary

        capacity_tokens = max(
            int(getattr(runner, "max_num_tokens", descriptor_tokens)),
            descriptor_tokens,
        )
        prepare_full_decode_draft_rope_310(
            draft_rotary.cos_sin_cache,
            query_positions=query_positions,
            query_actual_tokens=query_actual_tokens,
            context_positions=context_positions,
            context_actual_tokens=context_actual_tokens,
            capacity_tokens=capacity_tokens,
        )
        (
            query_cos,
            query_sin,
            context_cos,
            context_sin,
        ) = get_full_decode_draft_rope_buffers_310()
        assert query_cos is not None and query_sin is not None
        assert context_cos is not None and context_sin is not None
        self._full_decode_draft_query_rope_cos_310 = query_cos
        self._full_decode_draft_query_rope_sin_310 = query_sin
        self._full_decode_draft_context_rope_cos_310 = context_cos
        self._full_decode_draft_context_rope_sin_310 = context_sin
        logger.debug(
            "[310p-dflash-full-decode-only/rope-precompute] "
            "component=draft actual_runtime=%s descriptor_tokens=%d "
            "query_actual=%d "
            "context_actual=%d query_positions_ptr=%d "
            "context_positions_ptr=%d query_cos_ptr=%d context_cos_ptr=%d",
            runtime_mode.name,
            descriptor_tokens,
            query_actual_tokens,
            context_actual_tokens,
            query_positions.data_ptr(),
            context_positions.data_ptr(),
            query_cos.data_ptr(),
            context_cos.data_ptr(),
        )
        return True

    def _finish_full_decode_draft_rope(self, prepared: bool) -> None:
        if prepared:
            clear_full_decode_draft_rope_310()

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
    ) -> torch.Tensor:
        AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
        try:
            result = _original_run_merged_draft(
                self,
                num_input_tokens,
                batch_size,
                token_indices_to_sample,
                target_positions,
                inputs_embeds,
                multi_steps_attn_metadata,
                num_tokens,
                is_prefill,
            )
        finally:
            AscendRotaryEmbedding310.set_rope_position_flag_310p(False)
        return result

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        if not self.needs_extra_input_slots:
            # 310P workaround for MTP:
            # The NPU implementation of the slice assign
            #   self.input_ids[:num_tokens-1] = target_token_ids[1:]
            # can corrupt the tail element (index num_tokens-1) of the
            # persistent drafter input_ids buffer. We save/restore it to
            # avoid feeding garbage to the draft model or later GatherV2.
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1

            num_tokens = target_token_ids.shape[0]

            # Protected shift (310P specific)
            tail_save = self.input_ids[num_tokens - 1].clone()
            self.input_ids[: num_tokens - 1] = target_token_ids[1:]
            self.input_ids[num_tokens - 1] = tail_save

            # Replace the last token with the next token.
            self.input_ids[token_indices_to_sample] = next_token_ids

            assert self.runner is not None

            # 310P does not support PCP/DCP, so we skip all PCP handling.
            ori_token_indices_to_sample = None
            query_lens_d = None

            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]

            self._set_positions(num_tokens, target_positions)
            self.hidden_states[:num_tokens] = target_hidden_states.view(num_tokens, -1)

            return num_tokens, token_indices_to_sample, cad, (query_lens_d, ori_token_indices_to_sample)
        return super().set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )
