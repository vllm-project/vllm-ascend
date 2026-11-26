# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import copy
from typing import NamedTuple

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
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from vllm_ascend.worker.v2.utils import communicator_switch

from .dfly import CHECKPOINT_PREFIX, MODEL_PREFIX, FlashDFlyDraft

logger = init_logger(__name__)


class DFlyProposeInputs(NamedTuple):
    """Target outputs used by the upstream HYV4 DFly preparation contract."""

    query_start_loc: torch.Tensor
    positions: torch.Tensor
    idx_mapping: torch.Tensor
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    last_sampled: torch.Tensor
    last_hidden_states: torch.Tensor


class DFlyAttentionMetadata(NamedTuple):
    """C8 gather state; query K/V stay temporary instead of writing target KV."""

    block_size: int
    block_table: torch.Tensor
    indices: torch.Tensor
    query_positions: torch.Tensor


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
        self._draft_vllm_config = copy(vllm_config)
        self._draft_vllm_config.model_config = self.draft_model_config
        self._draft_vllm_config.parallel_config = copy(self.speculative_config.draft_parallel_config)
        self._draft_vllm_config.parallel_config.enable_expert_parallel = False
        self._draft_vllm_config.parallel_config.is_moe_model = False
        self._init_query_buffers()

    def _init_query_buffers(self):
        self.query_capacity = min(self.max_num_reqs, self.max_num_tokens // self.num_query_per_req)
        self.target_hidden_states = torch.zeros(
            (self.query_capacity, self.hidden_size), dtype=self.dtype, device=self.device
        )
        self.active_requests = torch.zeros(self.query_capacity, dtype=torch.bool, device=self.device)
        self.source_rows = torch.zeros(self.query_capacity, dtype=torch.long, device=self.device)
        self.anchors = torch.zeros_like(self.source_rows)
        self.mrope_positions = None
        self.query_block_tables = None
        self.query_indices = None

    def load_draft_model(self, target_model, target_attn_layer_names):
        config = copy(self._draft_vllm_config)
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
        self._draft_vllm_config = config
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
        rope = getattr(model_state, "rope_state", None)
        if rope is not None:
            if rope.num_dims <= 0 or rope.prefill_delta.gpu.ndim != 1:
                raise ValueError("HYV4 DFly requires a one-dimensional per-request RoPE delta")
            self.mrope_positions = torch.zeros(
                (rope.num_dims, self.max_num_tokens), dtype=torch.long, device=self.device
            )

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
        self.cudagraph_manager = CudaGraphManager(
            self._draft_vllm_config,
            self.device,
            CUDAGraphMode.FULL_DECODE_ONLY,
            {width: max_reqs},
            decode_request_sizes=graph_batches,
        )
        # Keep target-derived hidden/topk storage independent of draft outputs.
        self.cudagraph_manager.pool = torch.npu.graph_pool_handle()
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

    @torch.inference_mode()
    def capture(self):
        manager = self.cudagraph_manager
        if manager is None or not manager.needs_capture():
            return

        def create_forward_fn(desc, warmup):
            batch = desc.num_tokens // self.num_query_per_req
            # Follow upstream HYV4 DFly: prepare an all-padding query and its
            # attention state OUTSIDE the graph. Only the query model is captured.
            self._prepare_query(None, 0, batch)
            metadata = self._query_attention_state(desc, 0, None)
            return lambda mode: self._run_query_model(desc.num_tokens, metadata, mode)

        with communicator_switch():
            manager.capture(create_forward_fn, progress_bar_desc="Capturing HYV4 DFly ACLGraphs")

    def _prepare_query(self, inputs, num_reqs, padded_batch):
        if self.device.type == "npu":
            from .dfly_prepare import try_prepare_query

            if try_prepare_query(self, inputs, num_reqs, padded_batch):
                return
        self._prepare_query_reference(inputs, num_reqs, padded_batch)

    def _prepare_query_reference(self, inputs, num_reqs, padded_batch):
        """Torch adaptation of upstream HYV4 DFly's input preparation reference.

        Source: vLLM 4fd24951, common/dfly_inputs.py and its preparation tests.
        Preparation stays eager on NPU; the installed vLLM 0.33 lacks that
        upstream module. Greedy drafting needs no copied sampling distribution.
        """
        width = self.num_query_per_req
        count = padded_batch * width
        ids = self.input_buffers.input_ids[:count].view(padded_batch, width)
        positions = self.input_buffers.positions[:count].view(padded_batch, width)
        ids.fill_(self.model.config.mask_token_id)
        ids[:, 0].zero_()
        positions.copy_(self.offsets)
        self.input_buffers.is_padding[:count].fill_(True)
        self.input_buffers.query_start_loc[: padded_batch + 1].copy_(
            torch.arange(padded_batch + 1, dtype=torch.int32, device=self.device) * width
        )
        self.input_buffers.seq_lens[:padded_batch].fill_(width)
        self.target_hidden_states[:padded_batch].zero_()
        self.active_requests[:padded_batch].zero_()
        self.source_rows[:padded_batch].zero_()
        self.anchors[:padded_batch].zero_()
        if self.mrope_positions is not None:
            self.mrope_positions[:, :count].zero_()
        if num_reqs == 0:
            return

        rows = (inputs.query_start_loc[1 : num_reqs + 1] - inputs.num_rejected[:num_reqs] - 1).long()
        anchors = inputs.positions[rows] + 1
        active = (inputs.num_sampled[:num_reqs] > 0) & (anchors + width <= self.max_model_len)
        mapping = inputs.idx_mapping[:num_reqs].long()
        previous = inputs.last_sampled[mapping, 0].masked_fill(~active, 0)
        query = torch.where(active[:, None], anchors[:, None] + self.offsets, self.offsets)
        ids[:num_reqs, 0].copy_(previous)
        positions[:num_reqs].copy_(query)
        self.input_buffers.is_padding[: num_reqs * width].copy_((~active[:, None]).expand(-1, width).flatten())
        self.input_buffers.seq_lens[:num_reqs].copy_(anchors.masked_fill(~active, 0) + width)
        self.target_hidden_states[:num_reqs].copy_(inputs.last_hidden_states[rows])
        self.active_requests[:num_reqs].copy_(active)
        self.source_rows[:num_reqs].copy_(rows)
        self.anchors[:num_reqs].copy_(anchors)
        if self.mrope_positions is not None:
            delta = self.model_state.rope_state.prefill_delta.gpu[mapping].long()
            rope = (query + delta[:, None]).masked_fill(~active[:, None], 0)
            self.mrope_positions[:, : num_reqs * width].copy_(rope.flatten())

    def _init_attention_buffers(self):
        if self.query_block_tables is not None:
            return
        self.query_block_tables = tuple(
            torch.zeros(
                (self.query_capacity, cdiv(self.max_model_len, source.kv_cache.shape[2])),
                dtype=torch.int32,
                device=self.device,
            )
            for source in self.kv_layers
        )
        candidates = self.model.config.target_index_topk + (self.model.config.context_local_window or 0)
        self.query_indices = tuple(
            torch.full(
                (self.query_capacity, self.num_query_per_req, candidates + self.num_query_per_req),
                -1,
                dtype=torch.int64,
                device=self.device,
            )
            for _ in self.kv_layers
        )

    def _query_attention_state(self, desc, num_reqs, target_metadata):
        """Update persistent C8 metadata before either eager execution or replay.

        Unlike upstream MLA, DFlyAttention gathers history and concatenates
        temporary query K/V. It must not create writable slots into target KV.
        Native attention lengths are bounded by the fixed candidate width and
        are computed from these indices, not by a captured host max_seq_len.
        """
        self._init_attention_buffers()
        batch = desc.num_tokens // self.num_query_per_req
        positions = self.input_buffers.positions[: desc.num_tokens].view(batch, self.num_query_per_req)
        metadata = {}
        for layer, index, source, table, indices in zip(
            self.model.layers,
            self.index_layers,
            self.kv_layers,
            self.query_block_tables,
            self.query_indices,
            strict=True,
        ):
            prepared = False
            if num_reqs and self.device.type == "npu":
                from .dfly_prepare import try_prepare_c8_metadata

                prepared = try_prepare_c8_metadata(
                    index.topk_snapshot,
                    target_metadata[source.layer_name].block_table,
                    table,
                    indices,
                    self.active_requests,
                    self.source_rows,
                    self.anchors,
                    num_reqs,
                    batch,
                    self.model.config.context_local_window or 0,
                )
            if not prepared:
                table[:batch].zero_()
                indices[:batch].fill_(-1)
            if num_reqs and not prepared:
                current = target_metadata[source.layer_name].block_table
                table[:num_reqs].copy_(current[:num_reqs, : table.shape[1]])
                selected = build_reuse_indices(
                    index.topk_snapshot[self.source_rows[:num_reqs]],
                    self.anchors[:num_reqs],
                    self.num_query_per_req,
                    self.model.config.context_local_window or 0,
                    self.active_requests[:num_reqs],
                )
                indices[:num_reqs].copy_(selected)
            metadata[layer.self_attn.layer_name] = DFlyAttentionMetadata(
                source.kv_cache.shape[2], table[:batch], indices[:batch], positions
            )
        return metadata

    def _run_query_model(self, count, metadata, cg_mode):
        """Shared model-only execution boundary, as in upstream DFly."""
        batch = count // self.num_query_per_req
        ids = self.input_buffers.input_ids[:count].view(batch, self.num_query_per_req)
        positions = (
            self.input_buffers.positions[:count] if self.mrope_positions is None else self.mrope_positions[:, :count]
        )
        try:
            for layer in self.model.layers:
                layer.self_attn.context = metadata[layer.self_attn.layer_name]
            # C8 metadata is passed explicitly to DFlyAttention.context. It is
            # not metadata for the generic Ascend attention backend.
            with set_forward_context(
                None,
                self._draft_vllm_config,
                num_tokens=count,
                cudagraph_runtime_mode=cg_mode,
                is_padding=self.input_buffers.is_padding[:count],
            ):
                self._draft_tokens(
                    ids,
                    positions,
                    self.target_hidden_states[:batch],
                    self.active_requests[:batch],
                    self.draft_tokens[:batch],
                )
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
        count = batch * self.num_query_per_req
        if count > self.max_num_tokens or batch > self.query_capacity:
            raise ValueError("DFly query block exceeds max_num_batched_tokens")
        result = self.draft_tokens[:batch]
        if batch == 0 or num_speculative_tokens == 0:
            return result.fill_(-1)
        if dummy_run:
            # Memory profiling precedes KV allocation; keep its no-cache path.
            if is_profile:
                positions = self.offsets.repeat(batch)
                embeddings = self.target.embed_input_ids(positions)
                target_config = self.vllm_config.model_config
                if target_config.uses_mrope:
                    positions = positions.expand(target_config.mrope_num_dims, count)
                self.model(embeddings, positions)
            return result.fill_(-1)

        desc = BatchExecutionDescriptor(CUDAGraphMode.NONE, count, batch)
        manager = self.cudagraph_manager
        if manager is not None:
            desc = manager.dispatch(batch, count, self.num_query_per_req, 0)
        assert desc.num_tokens % self.num_query_per_req == 0
        padded_batch = desc.num_tokens // self.num_query_per_req
        inputs = DFlyProposeInputs(
            input_batch.query_start_loc,
            input_batch.positions,
            input_batch.idx_mapping,
            num_sampled,
            num_rejected,
            last_sampled,
            last_hidden_states,
        )
        self._prepare_query(inputs, batch, padded_batch)
        metadata = self._query_attention_state(desc, batch, attn_metadata)
        if desc.cg_mode == CUDAGraphMode.FULL:
            manager.run_fullgraph(desc)
            logger.info_once("FULL ACLGraph replay is active for HYV4 Flash DFly draft (%d requests).", padded_batch)
        else:
            self._run_query_model(desc.num_tokens, metadata, desc.cg_mode)
        return result
