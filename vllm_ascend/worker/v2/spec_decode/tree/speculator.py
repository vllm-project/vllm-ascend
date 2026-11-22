import copy
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import (
    AscendDFlashSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeLayout
from vllm_ascend.worker.v2.spec_decode.tree.mega_tree_builder import DFlash2MegaTreeBuilder
from vllm_ascend.worker.v2.spec_decode.tree.metrics import TreeGraphStats
from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape

logger = init_logger("vllm." + __name__)


class AscendDFlash2TreeSpeculator(AscendDFlashSpeculator):
    """Run one DFlash2 draft forward and publish a MegaTree proposal."""

    _speculator_name = "DFlash2Tree"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        if self.use_local_argmax_reduction:
            raise ValueError("MegaTree requires use_local_argmax_reduction=False")

        tree_cfg = get_ascend_config().tree_spec_config
        assert tree_cfg is not None
        self.budget = tree_cfg.budget
        self.candidate_size = tree_cfg.candidate_size
        self.tree_shape = TreeShape(self.num_speculative_steps, self.budget)
        self.tree_graph_stats = TreeGraphStats()
        self.tree_builder: DFlash2MegaTreeBuilder
        self._tree_finalized = True
        self.tree_kv_compact = None
        self.tree_commit = None
        # Persistent so FULL replay can update hidden without re-entering Python.
        self._draft_hidden_buf = torch.empty(
            self.max_num_reqs * self.num_query_per_req,
            self.hidden_size,
            dtype=self.dtype,
            device=device,
        )
        # The draft forward still projects every depth. Its published tokens
        # have exactly B columns, including when B is smaller than that depth.
        if self.budget != self.draft_tokens.shape[1]:
            self.draft_tokens = torch.zeros(
                self.max_num_reqs,
                self.budget,
                dtype=self.draft_tokens.dtype,
                device=device,
            )
        self.tree_parents = torch.full(
            (self.max_num_reqs, self.budget),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.tree_depths = torch.zeros(
            (self.max_num_reqs, self.budget),
            dtype=torch.int32,
            device=device,
        )
        self.tree_num_nodes = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.tree_visibility = torch.zeros(
            (self.max_num_reqs, self.budget, self.budget),
            dtype=torch.bool,
            device=device,
        )
        self.tree_first_child = torch.full(
            (self.max_num_reqs, self.budget + 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.tree_next_sibling = torch.full(
            (self.max_num_reqs, self.budget + 1),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.tree = self._view_tree_layout(self.max_num_reqs)
        logger.info(
            "MegaTree enabled: depth=%d candidates=%d budget=%d",
            self.num_speculative_steps,
            self.candidate_size,
            self.budget,
        )
        self._tree_graphs: dict[int, Any] = {}

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        # The fixed-width draft is captured directly. Keep the native model
        # loader and sharing hooks while avoiding a separate RoPE trace.
        config = self.vllm_config
        self.vllm_config = copy.copy(config)
        self.vllm_config.compilation_config = copy.copy(config.compilation_config)
        self.vllm_config.compilation_config.mode = CompilationMode.NONE
        try:
            return super().load_draft_model(target_model, target_attn_layer_names)
        finally:
            self.vllm_config = config

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self.tree_builder = DFlash2MegaTreeBuilder(self.model)

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self.supports_decode_metadata_update = False

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
        dp_sync_state: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        if num_speculative_tokens not in (None, 0, self.num_speculative_steps):
            raise ValueError(
                "Tree speculation requires a fixed draft depth; dynamic speculative token counts are not supported."
            )
        self._draft_graph_used = False
        acceptance = None if dummy_run else input_batch.tree_acceptance
        if acceptance is not None:
            tensors = [last_hidden_states]
            if aux_hidden_states:
                tensors.extend(aux_hidden_states)
            self.tree_commit.compact_queries(
                tensors,
                input_batch.query_start_loc,
                acceptance.path_node_ids,
                acceptance.active,
                input_batch.positions,
                input_batch=input_batch,
            )
        self._tree_finalized = False
        tokens = super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state,
            dummy_run,
            skip_attn_for_dummy_run,
            mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )
        if num_speculative_tokens == 0:
            # Upstream skips proposals when the remaining output budget needs
            # no drafts. Its context-only pass still updates the drafter KV.
            return tokens
        # FULL replay only runs draft forward; tree construction has its own graph.
        if not dummy_run:
            self.tree_graph_stats.record_lookup("draft", self._draft_graph_used)
            if self._draft_graph_used:
                self.tree_graph_stats.record_replay("draft")
            self._build_or_replay_tree(input_batch.num_reqs)
            return self.tree.tokens
        return tokens

    def _prepare_graph_replay(
        self,
        input_batch,
        num_graph_reqs,
        last_hidden_states,
        aux_hidden_states,
        num_rejected,
        dummy_run,
    ) -> None:
        self._draft_graph_used = True
        self._store_hidden_states(input_batch.num_tokens, last_hidden_states, aux_hidden_states)
        self.prepare_context_anchor(input_batch.num_reqs, input_batch.query_start_loc, num_rejected)
        self._precompute_context_kv(0, input_batch.num_tokens, dummy_run)
        # FA3 reprovisions the graph's scheduler tiling from the lengths
        # that prepare_dflash_inputs wrote into the persistent draft buffers.
        for groups in self.attn_groups:
            for group in groups:
                if group.supports_draft_decode_metadata_update:
                    group.get_metadata_builder().update_draft_decode_metadata()

    def capture(self) -> None:
        logger.info("Capturing model for %s speculator...", self._speculator_name)
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.fill_(-1)
        existing_draft_graphs = set(self.query_cudagraph_manager.graphs)
        self.query_cudagraph_manager.capture(
            self._run_draft_forward,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=self._group_causal,
            # Tree stages context eagerly in _prepare_graph_replay; the draft
            # graph must keep capturing only the parallel query forward.
            prepare_forward=lambda num_reqs, attn_metadata: None,
            progress_bar_desc=f"Capturing {self._speculator_name.lower()} CUDA graphs",
        )
        for _ in self.query_cudagraph_manager.graphs.keys() - existing_draft_graphs:
            self.tree_graph_stats.record_capture("draft")
        self.tree_graph_stats.set_workspace_bytes(
            "draft", "hidden", self.tree_graph_stats.tensor_bytes((self._draft_hidden_buf,))
        )
        self._capture_tree_graphs()
        if self.tree_kv_compact is not None:
            sizes = self._draft_capture_num_reqs()
            self.tree_kv_compact.capture(sizes)
            missing = set(sizes) - self.tree_kv_compact._graphs.keys()
            if missing:
                raise RuntimeError(f"FULL DFlash2 tree KV compaction capture failed for batches {sorted(missing)}")

    def _draft_capture_num_reqs(self) -> list[int]:
        manager = self.query_cudagraph_manager
        if manager is None:
            return []
        seen: set[int] = set()
        qlen = max(int(self.num_query_per_req), 1)
        for descs in manager._capture_descs.values():
            for desc in descs:
                n = desc.num_reqs
                if not n:
                    n = desc.num_tokens // qlen
                if n:
                    seen.add(int(n))
        return sorted(seen)

    def _build_tree_from_draft_hidden_states(self, num_reqs: int) -> None:
        """Project the DFlash2 candidate lattice and build the next tree."""
        num_sample = num_reqs * self.num_speculative_steps
        hidden = self._draft_hidden_buf[self.sample_indices[:num_sample]]
        candidate_ids, logits = self.model.compute_candidates(hidden, top_k=self.candidate_size)
        lattice_shape = (num_reqs, self.num_speculative_steps, self.candidate_size)
        query_ids = self.input_buffers.input_ids[: num_reqs * self.num_query_per_req]
        root_token_ids = query_ids.view(num_reqs, self.num_query_per_req)[:, 0]
        self.tree = self.tree_builder.build(
            candidate_ids.view(lattice_shape),
            logits.view(lattice_shape),
            hidden.view(num_reqs, self.num_speculative_steps, -1),
            root_token_ids,
            self._view_tree_layout(num_reqs),
        )

    def _capture_tree_graphs(self) -> None:
        manager = self.query_cudagraph_manager
        if manager is None or not manager.needs_capture():
            return
        sizes = self._draft_capture_num_reqs()
        if not sizes:
            return
        from vllm.platforms import current_platform

        pool = current_platform.get_global_graph_pool()
        for num_reqs in sizes:
            try:
                with torch.inference_mode():
                    self._build_tree_from_draft_hidden_states(num_reqs)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph, pool=pool):
                        self._build_tree_from_draft_hidden_states(num_reqs)
                self._tree_graphs[num_reqs] = graph
                self.tree_graph_stats.record_capture("tree")
                logger.info("Captured MegaTree ACLGraph for %d requests", num_reqs)
            except Exception as exc:
                raise RuntimeError(f"MegaTree ACLGraph capture failed for {num_reqs} requests") from exc

    def _run_draft_forward(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Draft model forward only; tree construction uses a separate ACLGraph."""
        hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._draft_hidden_buf[:num_tokens_padded].copy_(hidden)
        self._tree_finalized = False

    def _build_or_replay_tree(self, num_reqs: int) -> None:
        if self._tree_finalized:
            return

        # Match the actual request count, independently of the padded batch
        # used by the draft-forward graph. Uncaptured sizes use eager build.
        graph = self._tree_graphs.get(num_reqs)
        self.tree_graph_stats.record_lookup("tree", graph is not None)
        if graph is not None:
            graph.replay()
            self.tree_graph_stats.record_replay("tree")
            self.tree = self._view_tree_layout(num_reqs)
        else:
            self._build_tree_from_draft_hidden_states(num_reqs)
        self._tree_finalized = True

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        self._run_draft_forward(
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._build_or_replay_tree(num_reqs)

    def _view_tree_layout(self, num_reqs: int) -> TreeLayout:
        """Views into persistent buffers."""
        budget = self.budget
        return TreeLayout(
            tokens=self.draft_tokens[:num_reqs, :budget],
            depths=self.tree_depths[:num_reqs, :budget],
            parents=self.tree_parents[:num_reqs, :budget],
            num_nodes=self.tree_num_nodes[:num_reqs],
            visibility=self.tree_visibility[:num_reqs, :budget, :budget],
            first_child=self.tree_first_child[:num_reqs, : budget + 1],
            next_sibling=self.tree_next_sibling[:num_reqs, : budget + 1],
        )
