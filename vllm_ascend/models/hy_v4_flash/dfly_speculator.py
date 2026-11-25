# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy

import torch
from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import get_draft_load_config, process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import get_quant_config
from vllm.model_executor.models.utils import WeightsMapper
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from vllm_ascend import envs
from vllm_ascend.worker.v2.utils import communicator_switch

from .dfly import CHECKPOINT_PREFIX, MODEL_PREFIX, FlashDFlyDraft

logger = init_logger(__name__)


def build_reuse_indices(target_rows, anchor, width, window, active):
    """Combine the target's recalled history with the causal draft block."""
    starts = anchor[:, None] - window
    history = target_rows.masked_fill((target_rows < 0) | (target_rows >= starts), -1)
    batch = target_rows.shape[0]
    local = starts + torch.arange(window, device=anchor.device)
    offsets = torch.arange(width, device=anchor.device)
    positions = anchor[:, None] + offsets
    query = positions[:, None, :].expand(batch, width, width)
    query = query.masked_fill(offsets[None, :] > offsets[:, None], -1)
    result = torch.cat(
        (
            history[:, None, :].expand(-1, width, -1),
            local.clamp_min(-1)[:, None, :].expand(-1, width, -1),
            query,
        ),
        dim=-1,
    )
    return result.masked_fill(~active[:, None, None], -1)


class FlashDFlySpeculator(DraftModelSpeculator):
    can_reuse_target_attn_metadata = False

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        self.num_query_per_req = self.num_speculative_steps + 1
        self.offsets = torch.arange(self.num_query_per_req, device=device)
        self.cudagraph_manager = None

    def load_draft_model(self, target_model, target_attn_layer_names):
        config = copy(self.vllm_config)
        config.model_config = self.draft_model_config
        config.parallel_config = self.speculative_config.draft_parallel_config
        config.load_config = get_draft_load_config(config)
        # The target loader has already mapped checkpoint prefixes into the
        # language-model wrapper. Resolve the draft's exclusions from HF again.
        config.quant_config = (
            get_quant_config(config.model_config, config.load_config)
            if config.model_config.quantization is not None
            else None
        )
        if config.quant_config is not None:
            overrides = {
                f"{CHECKPOINT_PREFIX}{i}.": f"{MODEL_PREFIX}.layers.{i}."
                for i in range(config.model_config.hf_text_config.num_hidden_layers)
            }
            overrides[CHECKPOINT_PREFIX] = f"{MODEL_PREFIX}."
            config.quant_config.apply_vllm_mapper(WeightsMapper(orig_to_new_prefix=overrides))
            config.quant_config.packed_modules_mapping = FlashDFlyDraft.packed_modules_mapping.copy()

        if config.load_config.load_format not in ("auto", "safetensors", "fastsafetensors"):
            raise ValueError("DFly draft loading requires safetensors")
        loader = DefaultModelLoader(config.load_config)
        with set_current_vllm_config(config), set_default_torch_dtype(self.dtype):
            with torch.device(self.device):
                draft = FlashDFlyDraft(config)
            loader.load_weights(draft, self.draft_model_config)
            process_weights_after_loading(draft, self.draft_model_config, self.device)
        self.target = getattr(target_model, "language_model", target_model)
        target_layers = self.target.model.layers
        self.index_layers = tuple(target_layers[i].self_attn for i in draft.config.target_index_layer)
        self.kv_layers = tuple(target_layers[i].self_attn for i in draft.config.target_kv_layers)
        for attn in set(self.index_layers):
            if attn.indexer is None:
                raise ValueError("DFly target index must be produced by a sparse attention layer")
            attn.topk_snapshot = torch.empty(
                (self.max_num_tokens, draft.config.target_index_topk), dtype=torch.int64, device=self.device
            )
        for layer, source in zip(draft.layers, self.kv_layers, strict=True):
            layer.self_attn.bind_target(source)
        logger.info(
            "HYV4 Flash DFly loaded %d layers (FFN width %d), proposing %d tokens per step.",
            len(draft.layers),
            draft.ffn_width,
            self.num_speculative_steps,
        )
        return draft

    def set_attn(self, model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups):
        self.model_state = model_state

    def init_cudagraph_manager(self, cudagraph_mode):
        if not cudagraph_mode.has_full_cudagraphs() or self.speculative_config.enforce_eager:
            return
        enable_dp_graph = self.vllm_config.additional_config.get("hy4flash_dfly_dp_graph", False)
        if not isinstance(enable_dp_graph, bool):
            raise ValueError("hy4flash_dfly_dp_graph must be a boolean")
        # DP replicas propose independently, including empty batches. Only a
        # local dense draft can replay without matching calls on other ranks.
        if self.dp_size > 1:
            if not enable_dp_graph:
                return
            if not self._dp_graph_is_local():
                logger.warning_once(
                    "HYV4 DFly DP ACLGraph requires TP1 with local Embedding, LMHead and draft projections; "
                    "keeping eager drafting for this parallel layout."
                )
                return
        compilation = self.vllm_config.compilation_config
        width = self.num_query_per_req
        max_reqs = min(self.max_num_reqs, self.max_num_tokens // width)
        graph_batches = sorted(
            {size for size in compilation.decode_cudagraph_capture_request_sizes or () if 0 < size <= max_reqs}
        )
        if not graph_batches:
            return
        # The draft has dense FFNs, even though the target is MoE. Do not let
        # set_forward_context coordinate draft capture batches across DP.
        # Copy the config: the target still needs its original DP/EP metadata.
        self.graph_vllm_config = copy(self.vllm_config)
        self.graph_vllm_config.parallel_config = copy(self.vllm_config.parallel_config)
        self.graph_vllm_config.parallel_config.is_moe_model = False
        self.cudagraph_manager = CudaGraphManager(
            self.graph_vllm_config,
            self.device,
            CUDAGraphMode.FULL_DECODE_ONLY,
            {width: max_reqs},
            decode_request_sizes=graph_batches,
        )
        # Draft replay follows target replay while target-derived inputs are live.
        self.cudagraph_manager.pool = torch.npu.graph_pool_handle()
        capacity = max(graph_batches)
        self.graph_ids = torch.empty((capacity, width), dtype=torch.long, device=self.device)
        self.graph_hidden = torch.empty((capacity, self.hidden_size), dtype=self.dtype, device=self.device)
        self.graph_active = torch.empty(capacity, dtype=torch.bool, device=self.device)
        self.graph_source_rows = torch.empty(capacity, dtype=torch.long, device=self.device)
        self.graph_query_positions = torch.empty_like(self.graph_ids)
        # Fixed storage for raw inputs; preparation is captured with the draft.
        self.graph_raw_ends = torch.zeros(capacity, dtype=torch.int32, device=self.device)
        self.graph_raw_rejected = torch.zeros_like(self.graph_raw_ends)
        self.graph_raw_sampled = torch.zeros_like(self.graph_raw_ends)
        self.graph_raw_mapping = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.graph_raw_live = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        # Select a prebuilt mask by host batch size. Replay requires one copy,
        # not per-step zero/fill kernels on the default stream.
        slots = torch.arange(capacity, device=self.device)
        self.graph_live_masks = slots[None, :] < torch.arange(capacity + 1, device=self.device)[:, None]
        self.graph_raw_positions = torch.zeros(self.max_num_tokens, dtype=torch.long, device=self.device)
        self.graph_raw_hidden = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=self.device
        )
        self.graph_raw_previous = torch.zeros(self.max_num_reqs, dtype=torch.long, device=self.device)
        rope = getattr(self.model_state, "rope_state", None)
        self.graph_rope_dims = getattr(rope, "num_dims", 0)
        # Preserve the RoPE capability check: M-RoPE needs a per-request
        # delta table, independently of the unshifted KV query positions.
        delta = getattr(getattr(rope, "prefill_delta", None), "gpu", None)
        rope_supported = rope is None or (
            self.graph_rope_dims > 0
            and isinstance(delta, torch.Tensor)
            and delta.ndim == 1
            and delta.numel() <= self.max_num_reqs
            and delta.device == self.graph_ids.device
        )
        # Freeze this decision for both capture and replay. Re-reading the
        # environment later could replay a graph with stale raw inputs.
        self.graph_prepare_inputs = envs.VLLM_ASCEND_HY4_DRAFT_PREP_GRAPH and rope_supported
        self.graph_raw_rope_delta = torch.zeros(
            self.max_num_reqs,
            dtype=delta.dtype if rope_supported and rope is not None else torch.long,
            device=self.device,
        )
        if not rope_supported:
            logger.warning_once("HYV4 DFly input preparation graph: unsupported RoPE state; using body-only graphs.")
        if self.dp_size > 1:
            logger.info(
                "HYV4 DFly local DP ACLGraph enabled for %d replicas; capture request sizes: %s.",
                self.dp_size,
                graph_batches,
            )

    def _dp_graph_is_local(self):
        for parallel in (self.vllm_config.parallel_config, self.speculative_config.draft_parallel_config):
            if any(
                getattr(parallel, field) != 1
                for field in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "prefill_context_parallel_size",
                    "decode_context_parallel_size",
                )
            ):
                return False
        # Fine-grained Embedding/LMHead TP can communicate across the DP axis
        # even when the model's TP size is one. Inspect the loaded modules.
        embedding = self.target.model.embed_tokens
        if any(getattr(module, "tp_size", None) != 1 for module in (embedding, self.target.lm_head)):
            return False
        if getattr(embedding, "token_exchange_group", None) is not None:
            return False
        reduce_group = getattr(embedding, "output_reduce_group", None)
        if reduce_group is not None and reduce_group.world_size > 1:
            return False
        return all(getattr(module, "tp_size", 1) == 1 for module in self.model.modules())

    def capture(self):
        if self.cudagraph_manager is None:
            return

        self.graph_block_tables = tuple(
            torch.zeros(
                (self.graph_ids.shape[0], cdiv(self.max_model_len, source.kv_cache.shape[2])),
                dtype=torch.int32,
                device=self.device,
            )
            for source in self.kv_layers
        )

        def prepare(desc, warmup):
            batch = desc.num_reqs
            count = desc.num_tokens
            ids = self.graph_ids[:batch]
            ids.fill_(self.model.config.mask_token_id)
            ids[:, 0].zero_()
            flat_positions = self.input_buffers.positions[:count]
            flat_positions.view_as(ids).copy_(self.offsets)
            positions = flat_positions.expand(self.graph_rope_dims, -1) if self.graph_rope_dims else flat_positions
            target_hidden = self.graph_hidden[:batch].zero_()
            active = self.graph_active[:batch].zero_()
            source_rows = self.graph_source_rows[:batch].zero_()
            query_positions = self.graph_query_positions[:batch].copy_(self.offsets)
            tables = tuple(table[:batch] for table in self.graph_block_tables)
            result = self.draft_tokens[:batch]

            def forward(mode):
                with set_forward_context(None, self.graph_vllm_config, num_tokens=count, cudagraph_runtime_mode=mode):
                    if self.graph_prepare_inputs:
                        self._prepare_graph_inputs(batch)
                    self._forward_queries(
                        ids, positions, target_hidden, active, result, source_rows, query_positions, tables
                    )

            return forward

        with communicator_switch():
            self.cudagraph_manager.capture(prepare, progress_bar_desc="Capturing HYV4 DFly ACLGraphs")

    def _prepare_graph_inputs(self, batch):
        live = self.graph_raw_live[:batch]
        rows = (self.graph_raw_ends[:batch] - self.graph_raw_rejected[:batch] - 1).long().clamp_min(0)
        rows = rows.masked_fill(~live, 0)
        anchors = self.graph_raw_positions[rows] + 1
        active = live & (self.graph_raw_sampled[:batch] > 0) & (anchors + self.num_query_per_req <= self.max_model_len)
        anchors = anchors.masked_fill(~active, 0)
        query = anchors[:, None] + self.offsets
        # Mask stale padded request slots before gathering any per-request data.
        mapping = self.graph_raw_mapping[:batch].masked_fill(~live, 0)
        previous = self.graph_raw_previous[mapping].masked_fill(~active, 0)
        self.graph_ids[:batch].fill_(self.model.config.mask_token_id)
        self.graph_ids[:batch, 0].copy_(previous)
        self.graph_ids[:batch].masked_fill_(~live[:, None], 0)
        self.graph_query_positions[:batch].copy_(query.masked_fill(~live[:, None], 0))
        if self.graph_rope_dims:
            delta = self.graph_raw_rope_delta[mapping].long()
            rope_positions = (query + delta[:, None]).masked_fill(~active[:, None], 0)
        else:
            rope_positions = self.graph_query_positions[:batch]
        self.input_buffers.positions[: batch * self.num_query_per_req].copy_(rope_positions.flatten())
        self.graph_hidden[:batch].copy_(self.graph_raw_hidden[rows].masked_fill(~live[:, None], 0))
        self.graph_source_rows[:batch].copy_(rows)
        self.graph_active[:batch].copy_(active)
        for table in self.graph_block_tables:
            table[:batch].masked_fill_(~live[:, None], 0)

    def _forward_queries(
        self, ids, positions, target_hidden, active, result, source_rows, query_positions, block_tables
    ):
        try:
            for layer, index_layer, kv_layer, table in zip(
                self.model.layers, self.index_layers, self.kv_layers, block_tables, strict=True
            ):
                selected = build_reuse_indices(
                    index_layer.topk_snapshot[source_rows],
                    query_positions[:, 0],
                    self.num_query_per_req,
                    self.model.config.context_local_window,
                    active,
                )
                layer.self_attn.context = (kv_layer.kv_cache.shape[2], table, selected, query_positions)
            self._draft_tokens(ids, positions, target_hidden, active, result)
        finally:
            for layer in self.model.layers:
                layer.self_attn.context = None

    def _draft_tokens(self, ids, positions, target_hidden, active, result):
        embeddings = self.target.embed_input_ids(ids.flatten())
        anchor_embeddings = embeddings[:: self.num_query_per_req]
        anchor_embeddings.copy_(self.model.mtp_fusion(anchor_embeddings, target_hidden))
        hidden = self.model(embeddings, positions).view(ids.shape[0], self.num_query_per_req, -1)
        previous = ids[:, 0].contiguous()
        for step in range(self.num_speculative_steps):
            corrected = self.model.hidden_correction(hidden[:, step + 1], self.target.embed_input_ids(previous))
            previous = self.model.logits_processor(self.target.lm_head, corrected).argmax(dim=-1)
            result[:, step] = previous.masked_fill(~active, -1)

    def propose(
        self,
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
        dp_sync_state=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
        num_speculative_tokens=None,
    ):
        batch = input_batch.num_reqs
        result = self.draft_tokens[:batch]
        if batch == 0 or num_speculative_tokens == 0:
            return result.fill_(-1)
        if dummy_run:
            # Profile the draft workspace without reading unbound cache pages.
            if is_profile:
                count = batch * self.num_query_per_req
                positions = self.offsets.repeat(batch)
                embeddings = self.target.embed_input_ids(positions)
                target_config = self.vllm_config.model_config
                if target_config.uses_mrope:
                    positions = positions.expand(target_config.mrope_num_dims, count)
                self.model(embeddings, positions)
            return result.fill_(-1)

        manager = self.cudagraph_manager
        if manager is not None and self.graph_prepare_inputs:
            desc = manager.dispatch(batch, batch * self.num_query_per_req, self.num_query_per_req, 0)
            if desc in manager.graphs:
                padded = desc.num_reqs
                self.graph_raw_live[:padded].copy_(self.graph_live_masks[batch, :padded])
                for dest, value in (
                    (self.graph_raw_ends, input_batch.query_start_loc[1 : batch + 1]),
                    (self.graph_raw_rejected, num_rejected[:batch]),
                    (self.graph_raw_sampled, num_sampled[:batch]),
                    (self.graph_raw_mapping, input_batch.idx_mapping[:batch]),
                ):
                    dest[:batch].copy_(value)
                self.graph_raw_positions[: input_batch.positions.numel()].copy_(input_batch.positions)
                self.graph_raw_hidden[: last_hidden_states.shape[0]].copy_(last_hidden_states)
                self.graph_raw_previous[: last_sampled.shape[0]].copy_(last_sampled[:, 0])
                if self.graph_rope_dims:
                    rope = self.model_state.rope_state
                    delta = rope.prefill_delta.gpu
                    if rope.num_dims != self.graph_rope_dims or delta.ndim != 1 or delta.numel() > self.max_num_reqs:
                        raise RuntimeError("HYV4 DFly RoPE state changed after graph capture")
                    self.graph_raw_rope_delta[: delta.numel()].copy_(delta)
                for dest, source in zip(self.graph_block_tables, self.kv_layers, strict=True):
                    table = attn_metadata[source.layer_name].block_table
                    dest[:batch].copy_(table[:batch, : dest.shape[1]])
                manager.run_fullgraph(desc)
                logger.info_once("FULL ACLGraph replay is active for HYV4 Flash DFly draft (%d requests).", padded)
                logger.info_once(
                    "HYV4 DFly raw-input preparation is captured in the draft graph (dp_rank=%d, rope_dims=%d).",
                    self.vllm_config.parallel_config.data_parallel_rank,
                    self.graph_rope_dims,
                )
                return result

        if manager is not None and self.graph_prepare_inputs:
            logger.warning_once("HYV4 DFly input preparation graph miss: batch=%d; executing eager draft.", batch)
        elif manager is not None:
            logger.warning_once("HYV4 DFly uses body-only graphs; input preparation graph is disabled or unsupported.")

        source_rows = (input_batch.query_start_loc[1 : batch + 1] - num_rejected[:batch] - 1).long()
        safe_rows = source_rows.clamp_min(0)
        anchors = (input_batch.positions[safe_rows] + 1).long()
        active = (num_sampled[:batch] > 0) & (anchors + self.num_query_per_req <= self.max_model_len)
        anchors = anchors.masked_fill(~active, 0)
        query_positions = anchors[:, None] + self.offsets
        positions = query_positions.flatten()
        rope = getattr(self.model_state, "rope_state", None)
        if rope is not None:
            delta = rope.prefill_delta.gpu.index_select(0, input_batch.idx_mapping).long()
            rope_positions = (query_positions + delta[:, None]).masked_fill(~active[:, None], 0)
            positions = rope_positions.flatten().expand(rope.num_dims, -1)

        previous = last_sampled[input_batch.idx_mapping, 0].long().masked_fill(~active, 0)
        ids = torch.full(
            (batch, self.num_query_per_req), self.model.config.mask_token_id, dtype=torch.long, device=self.device
        )
        ids[:, 0] = previous
        target_hidden = last_hidden_states[safe_rows]

        tables = tuple(attn_metadata[source.layer_name].block_table[:batch] for source in self.kv_layers)
        manager = self.cudagraph_manager
        if manager is not None:
            desc = manager.dispatch(batch, ids.numel(), self.num_query_per_req, 0)
            if desc in manager.graphs:
                padded = desc.num_reqs
                self.graph_ids[:batch].copy_(ids)
                self.input_buffers.positions[: ids.numel()].copy_(positions[0] if positions.ndim == 2 else positions)
                self.graph_hidden[:batch].copy_(target_hidden)
                self.graph_active[:batch].copy_(active)
                self.graph_source_rows[:batch].copy_(safe_rows)
                self.graph_query_positions[:batch].copy_(query_positions)
                for graph_table, table in zip(self.graph_block_tables, tables, strict=True):
                    graph_table[:batch].copy_(table[:, : graph_table.shape[1]])
                if padded > batch:
                    self.graph_ids[batch:padded].zero_()
                    self.input_buffers.positions[ids.numel() : desc.num_tokens].zero_()
                    self.graph_hidden[batch:padded].zero_()
                    self.graph_active[batch:padded].zero_()
                    self.graph_source_rows[batch:padded].zero_()
                    self.graph_query_positions[batch:padded].zero_()
                    for table in self.graph_block_tables:
                        table[batch:padded].zero_()
                manager.run_fullgraph(desc)
                logger.info_once("FULL ACLGraph replay is active for HYV4 Flash DFly draft (%d requests).", padded)
                return result
        self._forward_queries(ids, positions, target_hidden, active, result, safe_rows, query_positions, tables)
        return result
