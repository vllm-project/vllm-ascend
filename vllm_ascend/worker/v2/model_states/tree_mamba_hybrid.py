# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState

if TYPE_CHECKING:
    from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder
    from vllm_ascend.ops.triton.spec_decode.tree.gdn_tree_pages import GDNTreePagePointers
    from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
    from vllm_ascend.worker.v2.spec_decode.tree.metrics import TreeGraphStats
    from vllm_ascend.worker.v2.spec_decode.tree.shared_metadata import SharedGDNTreeMetadata


def _pad_batch_rows(table: torch.Tensor, num_reqs: int) -> torch.Tensor:
    """Extend a per-request draft-tree table to the padded batch size.

    The runner builds the tree tables for the requests it verified, while the
    attention metadata carries the padded batch and the builder selects table
    rows by batch row (FULL cudagraph replay). Zero is the neutral filler: a
    padded row reads as a linear row (``tree_num_nodes == 0``) and owns no
    verified token, so its entries never become a node of a draft tree.
    """
    if table.shape[0] >= num_reqs:
        return table[:num_reqs]
    filler = table.new_zeros((num_reqs - table.shape[0], *table.shape[1:]))
    return torch.cat((table, filler), dim=0)


@dataclass
class AscendTreeMambaHybridAttnMetadata(MambaHybridAttnMetadata):
    """Mamba metadata that also carries the draft-tree layout.

    ``AscendGDNAttentionMetadataBuilder`` needs the tree parent ids to point each
    node's recurrent state init at its parent's slot; every other builder ignores
    them.
    """

    tree_parents: torch.Tensor | None = None
    tree_num_nodes: torch.Tensor | None = None
    # Accepted draft nodes of the *previous* step and its sampled token count,
    # both ``[num_reqs]``-sized batches. The short conv needs them to rebuild the
    # committed history from the node the sampler accepted; ``None`` (or
    # ``num_sampled == 0``) keeps the history the previous op left behind.
    tree_prev_path: torch.Tensor | None = None
    tree_prev_num_sampled: torch.Tensor | None = None
    tree_depths: torch.Tensor | None = None
    shared_gdn_tree: "SharedGDNTreeMetadata | None" = None

    def get_extra_attn_kwargs(
        self,
        attn_metadata_builder: Any,
        num_reqs: int,
    ) -> dict[str, Any]:
        kwargs = super().get_extra_attn_kwargs(attn_metadata_builder, num_reqs)
        if not kwargs or self.tree_parents is None:
            return kwargs
        # Lazy import to avoid a model-state <-> attention-builder cycle.
        from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder

        if not isinstance(attn_metadata_builder, AscendGDNAttentionMetadataBuilder):
            # Mamba2 / short-conv have no draft-tree state rollback.
            return kwargs
        # The runner copies the tree layout of the requests it verified; a FULL
        # graph replays with padded rows, which the builder selects by batch row
        # as well, so the tables have to cover the padded batch. A padded row
        # reads as a linear row (``tree_num_nodes == 0``) and owns no token, so
        # the extra entries are never used as a node.
        kwargs["tree_parents"] = _pad_batch_rows(self.tree_parents, num_reqs)
        if self.tree_num_nodes is not None:
            kwargs["tree_num_nodes"] = _pad_batch_rows(self.tree_num_nodes, num_reqs)
        if self.tree_prev_path is not None and self.tree_prev_num_sampled is not None:
            kwargs["prev_path_node_ids"] = self.tree_prev_path[:num_reqs]
            kwargs["prev_num_sampled"] = self.tree_prev_num_sampled[:num_reqs]
        tree = self.shared_gdn_tree
        if (
            tree is not None
            and attn_metadata_builder.spec_tree_slots == tree.slots
            and attn_metadata_builder.spec_conv_width == tree.conv_width
        ):
            kwargs["shared_gdn_tree"] = tree
        return kwargs


class AscendTreeMambaHybridModelState(AscendMambaHybridModelState):
    """Tree-specific state snapshots; native chains keep upstream postprocessing."""

    # GDN state slot the next step has to resume from, staged by the model
    # commit coordinator while verifying a draft tree (see ``tree.state_rollback``); ``None``
    # means the linear-chain convention (``num_sampled``) applies.
    _tree_resume_state_cursor: torch.Tensor | None = None
    _gdn_page_pointers: "GDNTreePagePointers | None" = None
    # Accepted draft path of the step being committed, in batch order, plus its
    # sampled token count; kept per request so the next step can re-anchor the
    # short-conv history (``0`` marks a request without a fresh tree path).
    _tree_path: torch.Tensor | None = None
    _tree_num_sampled: torch.Tensor | None = None
    # Which rows of that batch the tree op actually verified (see
    # ``tree_row_mask``); a row it did not verify was committed by the linear
    # operator, which leaves the committed columns where they are.
    _tree_is_tree_row: torch.Tensor | None = None
    _tree_prev_path: torch.Tensor | None = None
    _tree_prev_num_sampled: torch.Tensor | None = None
    tree_graph_stats: "TreeGraphStats | None" = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import tree_state_slot_count

        slots = tree_state_slot_count(self.vllm_config)
        assert slots is not None
        self._tree_slots = slots
        self._gdn_tree_graph_groups = None
        self._gdn_tree_graph_pools: dict = {}

    def custom_sampler(self, sampler):
        from vllm_ascend.worker.v2.spec_decode.tree.rejection_sampler import TreeRejectionSampler

        spec_config = self.vllm_config.speculative_config
        assert spec_config is not None
        return sampler, TreeRejectionSampler(
            sampler,
            spec_config,
            self.device,
            async_scheduling=self.vllm_config.scheduler_config.async_scheduling,
        )

    def _select_gdn_tree_graph_inputs(
        self, attn_groups: list[list[AttentionGroup]], full_row: bool, tree_slots: int
    ) -> dict[int, "AscendGDNHostMetadataBuilder"]:
        previous = self._gdn_tree_graph_groups
        if previous is not None and previous[0] is attn_groups and previous[1] == full_row:
            return previous[2]
        from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder
        from vllm_ascend.ops.triton.spec_decode.tree.gdn_tree_pages import GDNTreePageBatch

        refresh_builders = {}
        for index, groups in enumerate(attn_groups):
            for group in groups:
                builder = group.get_metadata_builder(0)
                if type(builder) is AscendGDNHostMetadataBuilder:
                    builder.select_full_tree_inputs(
                        full_row and builder.spec_tree_slots == tree_slots, self._gdn_tree_graph_pools
                    )
                    if full_row and tree_slots == GDNTreePageBatch.SLOTS and len(groups) == 1:
                        refresh_builders[index] = builder
        self._gdn_tree_graph_groups = (attn_groups, full_row, refresh_builders)
        return refresh_builders

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        assert ubatch_idx == 0, "DBO is not supported on Ascend"

        def build_metadata(**kwargs):
            return self._build_tree_attention_metadata(
                input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config, for_capture
            )

        device_metadata = getattr(self, "device_metadata", None)
        if device_metadata is None:
            self.attn_metadata = build_metadata()
        else:
            self.attn_metadata = device_metadata.run_build(
                build_metadata,
                attn_groups=attn_groups,
                full_graph_mode=cudagraph_mode == CUDAGraphMode.FULL,
                for_cudagraph_capture=for_capture,
            )
        return self.attn_metadata

    def _build_tree_attention_metadata(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        num_input_tokens = num_tokens
        if cudagraph_mode == CUDAGraphMode.PIECEWISE and input_batch.is_dummy:
            num_input_tokens = input_batch.num_tokens_after_padding

        tree_slots = self._tree_slots
        refresh_builders = None
        if (
            self.vllm_config.parallel_config.pipeline_parallel_size == 1
            and self.vllm_config.scheduler_config.max_num_seqs == 1
            and not self.vllm_config.scheduler_config.async_scheduling
        ):
            full_row = (
                (for_capture or cudagraph_mode == CUDAGraphMode.FULL)
                and input_batch.num_reqs_after_padding == 1
                and input_batch.num_tokens_after_padding == tree_slots
                and (input_batch.num_reqs == 0 or (input_batch.num_reqs == 1 and input_batch.num_tokens == tree_slots))
            )
            refresh_builders = self._select_gdn_tree_graph_inputs(attn_groups, full_row, tree_slots)
        real_reqs = input_batch.num_reqs
        is_prefilling_np = input_batch.is_prefilling_np[:real_reqs]
        if input_batch.prefill_runs_as_decode_np is not None:
            is_prefilling_np = is_prefilling_np & ~input_batch.prefill_runs_as_decode_np[:real_reqs]
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[:real_reqs] = torch.from_numpy(is_prefilling_np)
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[:real_reqs] = self.num_accepted_tokens_gpu[input_batch.idx_mapping[:real_reqs]]
            draft_counts = np.full(num_reqs, -1, dtype=np.int32)
            if input_batch.num_draft_tokens_per_req is not None:
                drafts = input_batch.num_draft_tokens_per_req[:real_reqs]
                scheduled = input_batch.num_scheduled_tokens[:real_reqs]
                is_verify = ~is_prefilling_np & (drafts > 0) & (scheduled == drafts + 1) & (scheduled <= tree_slots)
                draft_counts[:real_reqs] = np.where(is_verify, drafts, -1)
                if cudagraph_mode == CUDAGraphMode.FULL and num_reqs > real_reqs and is_verify.all():
                    padding_widths = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])[real_reqs:]
                    if np.all(padding_widths == tree_slots):
                        draft_counts[real_reqs:] = padding_widths - 1
            num_decode_draft_tokens_cpu = torch.from_numpy(draft_counts)

        # A padded or path-less request keeps 0, which leaves committed conv history untouched.
        path_buffer, num_sampled_buffer = self._tree_buffers(self.num_accepted_tokens_gpu.device)
        tree_prev_path = path_buffer.new_zeros((num_reqs, path_buffer.shape[1]))
        tree_prev_path[: input_batch.num_reqs] = path_buffer[input_batch.idx_mapping]
        tree_prev_num_sampled = num_sampled_buffer.new_zeros((num_reqs,))
        tree_prev_num_sampled[: input_batch.num_reqs] = num_sampled_buffer[input_batch.idx_mapping]

        model_specific_metadata = AscendTreeMambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            # ``None`` unless the runner populated the draft-tree layout.
            tree_parents=input_batch.tree_parents,
            tree_num_nodes=input_batch.tree_num_nodes,
            tree_prev_path=tree_prev_path,
            tree_prev_num_sampled=tree_prev_num_sampled,
            tree_depths=input_batch.tree_depths,
        )
        # Share topology for a real, complete BS1 FULL verify.
        if (
            not for_capture
            and not input_batch.is_dummy
            and cudagraph_mode == CUDAGraphMode.FULL
            and input_batch.num_reqs == num_reqs == 1
            and input_batch.num_tokens == num_tokens == tree_slots
            and num_decode_draft_tokens_cpu is not None
            and int(num_decode_draft_tokens_cpu[0]) == tree_slots - 1
            and not bool(is_prefilling[0])
        ):
            self._prepare_shared_gdn_tree(model_specific_metadata, attn_groups, input_batch.req_ids[0])
        # Varlen graphs capture evenly split dummy rows (possibly one token
        # each), but must allow the descriptor's full K+1 replay query bound.
        max_query_len = getattr(input_batch, "max_query_len", None)
        if max_query_len is None:
            max_query_len = input_batch.num_scheduled_tokens.max().item()
        # Validate the shared row once. Only groups with a warm descriptor for
        # this same request can bypass common metadata construction below.
        tree = model_specific_metadata.shared_gdn_tree
        refresh_group = None
        if (
            refresh_builders
            and tree is not None
            and tree.request_id is not None
            and num_tokens == num_input_tokens == max_query_len == tree.slots
            and input_batch.query_start_loc_np.shape == (2,)
            and tuple(input_batch.query_start_loc_np) == (0, tree.slots)
            and num_accepted_tokens is not None
            and num_accepted_tokens.shape == (1,)
        ):
            row_seq_lens = input_batch.seq_lens[:num_reqs]

            def refresh_group(index, block_table):
                builder = refresh_builders.get(index)
                if builder is not None:
                    return builder.try_refresh_tree_group(block_table, row_seq_lens, tree, num_accepted_tokens)
                return None

        metadata = build_attn_metadata(
            refresh_tree_group=refresh_group,
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_actual_reqs=input_batch.num_reqs,
            num_tokens=num_tokens,
            num_actual_tokens=input_batch.num_tokens,
            num_input_tokens=num_input_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            parallel_config=self.vllm_config.parallel_config,
            seq_lens_np=input_batch.seq_lens_np,
            seq_lens_host_exact=input_batch.seq_lens_host_exact,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            model_specific_attn_metadata=model_specific_metadata,
            full_graph_mode=cudagraph_mode == CUDAGraphMode.FULL,
            tree_visibility=input_batch.tree_visibility,
            for_cudagraph_capture=for_capture,
        )
        tree = model_specific_metadata.shared_gdn_tree
        if tree is not None and tree.page_batch is not None:
            tree.page_batch.finish(tree.init_state_columns)
        return metadata

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        """Validate greedy verification and reset the request's Tree state."""
        params = new_req_data.sampling_params
        if params is not None and (
            params.temperature != 0
            or params.logprobs is not None
            or params.logprob_token_ids
            or params.structured_outputs is not None
            or params.logit_bias
            or params.allowed_token_ids
            or params.bad_words
            or params.min_tokens
            or params.presence_penalty != 0
            or params.frequency_penalty != 0
            or params.repetition_penalty != 1
        ):
            raise ValueError("MegaTree requires greedy sampling without logit processors or output logprobs")
        super().add_request(req_index, new_req_data)
        # The path stored here belongs to the request that occupied this slot
        # before. A new request prefills its own conv state, so its first tree
        # step must keep the history the prefill left (``0``) instead of
        # re-anchoring from the previous occupant's accepted nodes.
        if self._tree_prev_num_sampled is not None:
            self._tree_prev_num_sampled[req_index] = 0
        if self._tree_prev_path is not None:
            self._tree_prev_path[req_index] = 0

    def stage_tree_acceptance(self, acceptance: "TreeAcceptance", num_rejected: torch.Tensor) -> None:
        """Stage the accepted GDN leaf through the common tree commit boundary."""
        from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import tree_row_mask

        # The verifier returns a zero-based node ID, with zero for the root.
        self._tree_resume_state_cursor = (acceptance.last_accepted_node_id + 1).to(torch.int32)
        self._tree_path = acceptance.path_node_ids
        self._tree_num_sampled = acceptance.num_sampled
        self._tree_is_tree_row = tree_row_mask(acceptance.num_sampled, num_rejected)

    def _tree_buffers(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-request storage of the previous step's accepted draft path.

        Keep a stable row wide enough for the sampler's padded draft-depth
        path and for the tree topology workspace, whichever is larger.
        """
        path_width = max(self._tree_slots - 1, self.vllm_config.num_speculative_tokens)
        # Per-request storage: ``idx_mapping`` values address the request slots
        # of the persistent spec-decode state.
        num_reqs = self.num_accepted_tokens_gpu.shape[0]
        path = self._tree_prev_path
        if path is None or path.shape != (num_reqs, path_width) or path.device != device:
            path = torch.zeros((num_reqs, path_width), dtype=torch.int32, device=device)
            self._tree_prev_num_sampled = torch.zeros((num_reqs,), dtype=torch.int32, device=device)
            self._tree_prev_path = path
        assert self._tree_prev_num_sampled is not None
        return path, self._tree_prev_num_sampled

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        resume_state_cursor = self._tree_resume_state_cursor
        path = self._tree_path
        # Native chain decoding retains the upstream state lifecycle.
        if resume_state_cursor is None and path is None and self._tree_prev_path is None:
            super().postprocess_state(idx_mapping, num_sampled, num_computed_tokens)
            return

        committed = False
        if (
            resume_state_cursor is not None
            and idx_mapping.device.type == "npu"
            and path is not None
            and self._tree_num_sampled is not None
            and not self.vllm_config.speculative_config.enforce_eager
            and self.vllm_config.compilation_config.cudagraph_mode
            in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY)
        ):
            from vllm_ascend.worker.v2.spec_decode.tree.qwen_state_graph import QwenTreeStateGraphs

            graphs = getattr(self, "_tree_state_graphs", None)
            if graphs is None:
                graphs = self._tree_state_graphs = QwenTreeStateGraphs(self, stats=self.tree_graph_stats)
            committed = graphs.run(
                idx_mapping,
                resume_state_cursor,
                path,
                self._tree_num_sampled,
                self._tree_is_tree_row,
                num_computed_tokens,
            )
        if not committed:
            self._postprocess_tree_state_tensors(
                idx_mapping,
                resume_state_cursor if resume_state_cursor is not None else num_sampled,
                path,
                self._tree_num_sampled,
                self._tree_is_tree_row,
                num_computed_tokens,
            )
        self._tree_resume_state_cursor = None
        self._tree_path = None
        self._tree_num_sampled = None
        self._tree_is_tree_row = None

    def _postprocess_tree_state_tensors(
        self, idx_mapping, resume_state_cursor, path, num_sampled, tree_rows, num_computed_tokens
    ) -> None:
        """The same accepted-state computation is used eagerly and in a graph."""
        if path is not None or self._tree_prev_path is not None:
            from vllm_ascend.ops.triton.spec_decode.tree.state_commit import update_tree_acceptance_history_

            previous_path, previous_sampled = self._tree_buffers(idx_mapping.device)
            update_tree_acceptance_history_(previous_path, previous_sampled, idx_mapping, path, num_sampled, tree_rows)
        super().postprocess_state(idx_mapping, resume_state_cursor, num_computed_tokens)

    def _prepare_shared_gdn_tree(
        self,
        metadata: AscendTreeMambaHybridAttnMetadata,
        attn_groups,
        request_id: str,
    ) -> None:
        """Create the shared draft-tree geometry before the layer loop.

        Enqueue the topology work before building the leading FIA groups so
        the device can run it while the host builds attention metadata.
        """
        if (
            metadata.tree_parents is None
            or metadata.tree_num_nodes is None
            or metadata.tree_depths is None
            or metadata.tree_prev_path is None
            or metadata.tree_prev_num_sampled is None
            or metadata.num_accepted_tokens is None
        ):
            return
        # Lazy imports to keep the model-state <-> attention-builder cycle
        # out of the module level (same reason as get_extra_attn_kwargs).
        from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder
        from vllm_ascend.ops.triton.spec_decode.tree.gdn_tree_pages import GDNTreePageBatch, GDNTreePagePointers
        from vllm_ascend.worker.v2.spec_decode.tree.shared_metadata import SharedGDNTreeMetadata

        conv_width = None
        for cache_group in attn_groups:
            for attn_group in cache_group:
                builder = attn_group.get_metadata_builder(0)
                if (
                    isinstance(builder, AscendGDNAttentionMetadataBuilder)
                    and builder.spec_tree_slots == metadata.tree_parents.shape[1] + 1
                    and builder.spec_conv_width is not None
                ):
                    conv_width = builder.spec_conv_width
                    break
            if conv_width is not None:
                break
        if conv_width is None:
            return
        page_batch = None
        if (
            not self.vllm_config.scheduler_config.async_scheduling
            and metadata.tree_parents.device.type == "npu"
            and metadata.tree_parents.shape[1] + 1 == GDNTreePageBatch.SLOTS
        ):
            if self._gdn_page_pointers is None:
                self._gdn_page_pointers = GDNTreePagePointers()
            page_batch = GDNTreePageBatch(self._gdn_page_pointers)
        metadata.shared_gdn_tree = SharedGDNTreeMetadata.create(
            metadata.tree_parents,
            metadata.tree_num_nodes,
            metadata.tree_depths,
            metadata.num_accepted_tokens,
            metadata.tree_prev_path,
            metadata.tree_prev_num_sampled,
            conv_width,
            request_id=request_id,
            use_fused_topology=not self.vllm_config.scheduler_config.async_scheduling,
            page_batch=page_batch,
        )
