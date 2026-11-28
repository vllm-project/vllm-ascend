# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 Flash model state."""

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import logger as graph_logger
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.ops.triton.engram_lookback import _gather_lookback_kernel
from vllm_ascend.ops.vocab_parallel_embedding import AscendVocabParallelEmbedding
from vllm_ascend.utils import lmhead_tp_enable
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from .attention_ops import uses_decode_fusions
from .config import uses_eager_prefill_fusions, uses_prefill_ops


@triton.jit
def _target_embedding_prepare_kernel(
    ids,
    weight,
    output,
    ID_STRIDE: tl.constexpr,
    WEIGHT_ROW: tl.constexpr,
    WEIGHT_COL: tl.constexpr,
    OUTPUT_ROW: tl.constexpr,
    OUTPUT_COL: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(ids + row * ID_STRIDE).to(tl.int64)
    token = tl.maximum(token, 0)
    values = tl.load(weight + token * WEIGHT_ROW + col * WEIGHT_COL, col < HIDDEN, other=0)
    tl.store(output + row * OUTPUT_ROW + col * OUTPUT_COL, values, col < HIDDEN)


def _embedding_eligible(embedding, ids, output):
    """Metadata-only gate; no device data reads, reductions or uploads."""
    if type(embedding) is not AscendVocabParallelEmbedding:
        return False
    if type(embedding.quant_method) is not UnquantizedEmbeddingMethod:
        return False
    if embedding.tp_size != 1 or embedding.token_exchange_group is not None:
        return False
    reduce_group = getattr(embedding, "output_reduce_group", None)
    if reduce_group is not None and reduce_group.world_size != 1:
        return False
    weight = embedding.weight
    return (
        ids.ndim == 1
        and output.ndim == weight.ndim == 2
        and ids.dtype in (torch.int32, torch.int64)
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.device.type == "npu"
        and ids.device == weight.device == output.device
        and output.shape[0] >= ids.numel()
        and output.shape[1] == weight.shape[1]
        and weight.shape[0] >= embedding.num_embeddings
        and ids.stride(0) > 0
        and min(weight.stride()) > 0
        and min(output.stride()) > 0
    )


def try_prepare_target_embeddings(state, scheduled_encoder_inputs, input_batch, req_states):
    """Return preallocated buffer view, or None without touching device state.

    Covers pure decode with no media work, no prompt embeddings, no pruning,
    no PCP/TP exchange, no quantized/LoRA/custom embedding implementation.
    Padding rows remain untouched, matching the original prepare_inputs_embeds.
    """
    if (
        scheduled_encoder_inputs
        or getattr(input_batch, "has_prefill", True)
        or input_batch.is_dummy
        or state.prompt_embeds_state is not None
        or getattr(state, "pcp_context", None) is not None
    ):
        return None
    core = getattr(state.model, "language_model", state.model).model
    if not getattr(core, "uses_dfly", False):
        return None
    embedding = core.embed_tokens
    if state.supports_mm_inputs:
        runner = state.encoder_runner
        if runner.is_realtime or state.mm_pruner is not None:
            return None
        output = runner.inputs_embeds
    else:
        output = state.inputs_embeds
    ids = input_batch.input_ids[: input_batch.num_tokens]
    if not _embedding_eligible(embedding, ids, output):
        return None
    # Match AscendModelState.prepare_inputs_embeds's host metadata update.
    setter = getattr(state.model, "set_embedding_num_tokens_padded", None)
    if setter is not None:
        setter(input_batch.num_tokens_after_padding)
    if ids.numel():
        _target_embedding_prepare_kernel[(ids.numel(), triton.cdiv(output.shape[1], 256))](
            ids,
            embedding.weight,
            output,
            ids.stride(0),
            *embedding.weight.stride(),
            *output.stride(),
            output.shape[1],
            256,
        )
    return output[: input_batch.num_tokens_after_padding]


class FlashAclGraphManager(ModelAclGraphManager):
    def capture(self, model, model_state, *args, **kwargs):
        super().capture(model, model_state, *args, **kwargs)
        self._short_graphs = {}
        if not uses_decode_fusions(self.vllm_config):
            return
        layers = [
            module
            for module in model.modules()
            if getattr(module, "indexer", None) is not None and hasattr(module, "_short_indexpool_capture")
        ]
        if not layers:
            return
        self._short_limit = min((layer.indexer.budget + 1) * layer.indexer.ratio for layer in layers)
        long_graphs = self.graphs
        self.graphs = {}
        try:
            for layer in layers:
                layer._short_indexpool_capture = True
            super().capture(model, model_state, *args, **kwargs)
            self._short_graphs = self.graphs
        finally:
            self.graphs = long_graphs
            for layer in layers:
                layer._short_indexpool_capture = False

    def _init_candidates(self):
        super()._init_candidates()
        decode_graphs = [
            desc for desc in self._capture_descs.get(CUDAGraphMode.FULL, ()) if desc.uniform_token_count is not None
        ]
        self._capture_descs.clear()
        self._candidates.clear()
        if decode_graphs:
            self._capture_descs[CUDAGraphMode.FULL] = decode_graphs

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None, num_ubatches=1):
        if uniform_token_count is not None:
            desc = super().dispatch(
                num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len, num_ubatches
            )
            if desc in self.graphs:
                return desc
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
            num_ubatches=num_ubatches,
        )

    def run_fullgraph(self, desc):
        metadata = self.model_runner.model_state.attn_metadata
        batch = next(iter(metadata.values()), None) if isinstance(metadata, dict) else None
        short_graph = getattr(self, "_short_graphs", {}).get(desc)
        use_short = (
            short_graph is not None and batch is not None and max(batch.seq_lens, default=0) <= self._short_limit
        )
        original = self.graphs[desc]
        try:
            if use_short:
                self.graphs[desc] = short_graph
            output = super().run_fullgraph(desc)
        finally:
            self.graphs[desc] = original
        graph_logger.info_once(
            "FULL ACLGraph replay is active for HYV4 Flash target (%d requests, %d query tokens).",
            desc.num_reqs,
            desc.num_tokens,
        )
        return output


logger = init_logger_ascend(__name__)


class FlashModelState(AscendModelState):
    aclgraph_manager_cls = FlashAclGraphManager

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self._decode_optimizations = uses_decode_fusions(vllm_config) and device.type == "npu"
        self._graph_decode = False
        self._reuse_eager_embeddings = (
            uses_prefill_ops(vllm_config)
            and vllm_config.model_config.enforce_eager
            and vllm_config.parallel_config.prefill_context_parallel_size == 1
        )
        self._use_prefill_fusions = (
            self._reuse_eager_embeddings and device.type == "npu" and uses_eager_prefill_fusions(vllm_config)
        )
        language_model = getattr(model, "language_model", model)
        self._flash_model = language_model.model
        self._prefill_vocab_embedding = None
        if self._use_prefill_fusions and self._flash_model.start_layer == 0:
            embedding = self._flash_model.embed_tokens
            if (
                self.prompt_embeds_state is None
                and self.mm_pruner is None
                and not self._flash_model.uses_dfly
                and embedding.hy4_prefill_vocab_parameters() is not None
            ):
                self._prefill_vocab_embedding = embedding
            first_layer = self._flash_model.layers[0]
            first_engram = 1 in (self._flash_model.config.engram_layer_ids or ())
            if self._prefill_vocab_embedding is not None and not first_layer.sequence_parallel and not first_engram:
                first_layer.hc_attn_layer.configure_prefill_broadcast(
                    embedding,
                    vllm_config.scheduler_config.max_num_batched_tokens,
                )
        self._defer_hc_head = (
            self._use_prefill_fusions
            and vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
            and vllm_config.lora_config is None
            and vllm_config.parallel_config.pipeline_parallel_size == 1
            and vllm_config.parallel_config.data_parallel_size == 1
            and vllm_config.parallel_config.decode_context_parallel_size == 1
            and not vllm_config.parallel_config.enable_batch_sharded_sampling
            and not lmhead_tp_enable()
            and self._flash_model.engram_hash is not None
        )
        self._graph_hashes = None
        if model.token_lookback_depth and vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            engram = self._flash_model.engram_hash
            capacity = min(
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.scheduler_config.max_num_seqs * vllm_config.uniform_decode_query_len,
            )
            self._graph_hashes = torch.full(
                (len(engram.layout.layer_ids), capacity, engram.offsets.shape[1]), -1, dtype=torch.int32, device=device
            )
            self._engram_embeddings = tuple(
                self._flash_model.layers[layer - 1].engram_layer.multi_head_embedding
                for layer in engram.layout.layer_ids
            )
            self._graph_rows = tuple(
                torch.zeros(
                    (capacity, embedding.head_end - embedding.head_start, embedding.dim),
                    dtype=torch.float32,
                    device=device,
                )
                for embedding in self._engram_embeddings
            )

    def _prepare_prefill_inputs(self, input_batch, req_states):
        rope_positions = getattr(input_batch, "hy4_rope_positions", None)
        if rope_positions is None:
            inputs = super().prepare_inputs(input_batch, req_states)
        else:
            inputs = {"positions": rope_positions}
            inputs.update(self.prepare_engram_inputs(input_batch, req_states))
        indices = getattr(input_batch, "hy4_vocab_indices", None)
        if indices is not None and not self.supports_mm_inputs:
            inputs["inputs_embeds"] = self._prefill_vocab_embedding.embed_hy4_local_indices(indices)
            input_batch.hy4_text_vocab_embeddings = True
        if getattr(input_batch, "hy4_text_vocab_embeddings", False):
            inputs["first_ihc_vocab"] = True
        worker = self.prompt_logprobs_worker
        if (
            self._defer_hc_head
            and not input_batch.is_dummy
            and not self.kvpp_is_dummy_run
            and input_batch.num_tokens == input_batch.num_tokens_after_padding
            and input_batch.logits_indices.numel() > 0
            and (
                input_batch.num_computed_prefill_tokens_np < req_states.prefill_len.np[input_batch.idx_mapping_np]
            ).all()
            and worker is not None
            and not worker.uses_prompt_logprobs[input_batch.idx_mapping_np].any()
            and not any(req_id in worker.token_id_scores for req_id in input_batch.req_ids)
        ):
            inputs["defer_hc_head"] = True
        return inputs

    def compute_sample_logits(self, hidden_states):
        if self._defer_hc_head and hidden_states.ndim == 3:
            hidden_states = self._flash_model.finish_hidden_states(hidden_states)
        return super().compute_sample_logits(hidden_states)

    def custom_sampler(self, sampler):
        if self._use_prefill_fusions:
            from vllm.v1.worker.gpu.sample.sampler import Sampler

            if type(sampler) is Sampler:
                from vllm_ascend.worker.v2.hy4_prefill_greedy import PrefillSampler

                return PrefillSampler(sampler), None
        return super().custom_sampler(sampler)

    def prepare_attn(self, input_batch, cudagraph_mode, *args, **kwargs):
        self._graph_decode = cudagraph_mode == CUDAGraphMode.FULL
        return super().prepare_attn(input_batch, cudagraph_mode, *args, **kwargs)

    def prefill_vocab_parameters(self):
        embedding = self._prefill_vocab_embedding
        return None if embedding is None else embedding.hy4_prefill_vocab_parameters()

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, req_states) -> torch.Tensor:
        if self._decode_optimizations:
            result = try_prepare_target_embeddings(self, scheduled_encoder_inputs, input_batch, req_states)
            if result is not None:
                return result
        num_tokens = input_batch.num_tokens
        if (
            not self._reuse_eager_embeddings
            or num_tokens != input_batch.num_tokens_after_padding
            or not self.supports_mm_inputs
            or self.prompt_embeds_state is not None
            or self.mm_pruner is not None
        ):
            return super().prepare_inputs_embeds(scheduled_encoder_inputs, input_batch, req_states)

        set_padded_tokens = getattr(self.model, "set_embedding_num_tokens_padded", None)
        if set_padded_tokens is not None:
            set_padded_tokens(num_tokens)
        input_ids = input_batch.input_ids[:num_tokens]
        self.execute_mm_encoder(scheduled_encoder_inputs)
        mm_embeds, is_mm_embed = self.gather_mm_embeddings(input_batch)
        indices = getattr(input_batch, "hy4_vocab_indices", None)
        needs_mm_token_mask = is_mm_embed is not None and getattr(self.model, "_has_oov_mm_tokens", False)
        if indices is not None and not mm_embeds and not needs_mm_token_mask:
            embeddings = self._prefill_vocab_embedding.embed_hy4_local_indices(indices)
            input_batch.hy4_text_vocab_embeddings = True
        else:
            embeddings = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )
        buffer = self.encoder_runner.inputs_embeds

        # Eager Prefill consumes the newly produced embeddings immediately.
        return embeddings.to(device=buffer.device, dtype=buffer.dtype).contiguous()

    def prepare_engram_dummy_inputs(self, num_reqs, num_tokens):
        if self._graph_hashes is None:
            return {}
        return {
            "prepared_engram": (
                self._graph_hashes[:, :num_tokens],
                tuple(rows[:num_tokens] for rows in self._graph_rows),
            )
        }

    def _can_fuse_rope_hash(self, input_batch):
        return (
            self._decode_optimizations
            and self._graph_decode
            and not input_batch.is_dummy
            and self._graph_hashes is not None
            and self.model.token_lookback_depth > 0
            and input_batch.num_tokens_after_padding > 0
            and self.rope_state is not None
            # Original RopeState launches once for every mapping entry, while
            # hash uses num_reqs. Fall back if those scopes differ (padding).
            and input_batch.idx_mapping.ndim == 1
            and input_batch.idx_mapping.shape[0] == input_batch.num_reqs
            and all(
                0 <= int(b) - int(a) <= 6
                for a, b in zip(
                    input_batch.query_start_loc_np[: input_batch.num_reqs],
                    input_batch.query_start_loc_np[1 : input_batch.num_reqs + 1],
                )
            )
        )

    def prepare_inputs(self, input_batch, req_states):
        if self._reuse_eager_embeddings:
            return self._prepare_prefill_inputs(input_batch, req_states)
        if not self._can_fuse_rope_hash(input_batch):
            return super().prepare_inputs(input_batch, req_states)
        # Parent prepare_inputs performs only RoPE preparation followed by the
        # Engram hook. Compute both here before either result is consumed.
        model_inputs = self.prepare_engram_inputs(input_batch, req_states, rope_state=self.rope_state)
        model_inputs["positions"] = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        return model_inputs

    def prepare_engram_inputs(self, input_batch, req_states, *, rope_state=None):
        depth = self.model.token_lookback_depth
        if depth == 0:
            return {}
        history = getattr(input_batch, "hy4_lookback_token_ids", None)
        if history is not None:
            return {"lookback_token_ids": history}
        if (
            self._graph_decode
            and not input_batch.is_dummy
            and all(
                0 <= int(b) - int(a) <= 6
                for a, b in zip(
                    input_batch.query_start_loc_np[: input_batch.num_reqs],
                    input_batch.query_start_loc_np[1 : input_batch.num_reqs + 1],
                )
            )
            and self._decode_optimizations
        ):
            from .engram_hash import fused_engram_hashes

            inputs = self.prepare_engram_dummy_inputs(
                input_batch.num_reqs_after_padding, input_batch.num_tokens_after_padding
            )
            hashes, rows = inputs["prepared_engram"]
            packed_hashes = None
            if rows and all(
                output.shape == rows[0].shape
                and embedding.embedding.weight.device.type == "cpu"
                and embedding.embedding.weight.dtype == torch.bfloat16
                and not embedding.fp8
                for embedding, output in zip(self._engram_embeddings, rows, strict=True)
            ):
                # Keep graph-visible hash addresses/strides unchanged. The flat
                # staging allocation is reused, but its compact layer stride
                # follows this batch's NTOK, so .cpu() needs no Slice packing.
                storage = getattr(self, "_engram_packed_hash_storage", None)
                if storage is None:
                    storage = self._engram_packed_hash_storage = torch.empty(
                        self._graph_hashes.numel(), dtype=hashes.dtype, device=hashes.device
                    )
                packed_hashes = storage[: hashes.numel()].view(hashes.shape)
            state = self._flash_model.engram_hash
            if state.layout.lookup_table is None:
                raise ValueError("Missing checkpoint Engram token map")
            fused_engram_hashes(
                input_batch.input_ids,
                input_batch.query_start_loc,
                input_batch.idx_mapping,
                req_states.num_computed_tokens.gpu,
                req_states.all_token_ids.gpu,
                state.token_map,
                state.multipliers,
                state.primes,
                state.offsets,
                state.pad_id,
                state.layout.pad_token_id,
                state.visual_token_ids,
                hashes,
                num_reqs=input_batch.num_reqs,
                rope_state=rope_state,
                prefill_lens=req_states.prefill_len.gpu,
                packed_output=packed_hashes,
            )
            self._prepare_engram_rows(hashes, rows, packed_hashes=packed_hashes)
            return inputs
        history = torch.empty((input_batch.num_reqs, depth), dtype=torch.int32, device=self.device)
        if input_batch.is_dummy:
            history.fill_(-1)
        elif input_batch.num_reqs:
            all_token_ids = req_states.all_token_ids.gpu
            _gather_lookback_kernel[(input_batch.num_reqs,)](
                history,
                input_batch.idx_mapping,
                req_states.num_computed_tokens.gpu,
                all_token_ids,
                all_token_ids.stride(0),
                input_batch.num_reqs,
                DEPTH=depth,
                BLOCK_DEPTH=triton.next_power_of_2(depth),
            )
        if self._graph_decode:
            num_tokens = input_batch.num_tokens_after_padding
            inputs = self.prepare_engram_dummy_inputs(input_batch.num_reqs_after_padding, num_tokens)
            hashes, rows = inputs["prepared_engram"]
            if input_batch.is_dummy:
                hashes.fill_(-1)
                for output in rows:
                    output.zero_()
            else:
                self._flash_model.engram_hash.prepare_batch_hashes(
                    input_batch.input_ids,
                    history,
                    input_batch.query_start_loc_np[: input_batch.num_reqs + 1],
                    hashes,
                )
                self._prepare_engram_rows(hashes, rows)
            return inputs
        return {"lookback_token_ids": history}

    def _prepare_engram_rows(self, hashes, rows, *, packed_hashes=None):
        if self._decode_optimizations:
            from .engram_ops import try_prepare_engram_rows

            if try_prepare_engram_rows(self, hashes, rows, packed_hashes):
                return
        embeddings = self._engram_embeddings
        if rows and all(
            output.shape == rows[0].shape
            and embedding.embedding.weight.device.type == "cpu"
            and embedding.embedding.weight.dtype == torch.bfloat16
            and not embedding.fp8
            for embedding, output in zip(embeddings, rows, strict=True)
        ):
            # Each layer keeps its existing shard and clamp rules. Transfer the
            # hashes and native BF16 rows together to avoid per-layer barriers.
            if packed_hashes is not None:
                if (
                    packed_hashes.shape != hashes.shape
                    or packed_hashes.dtype != hashes.dtype
                    or packed_hashes.device != hashes.device
                    or not packed_hashes.is_contiguous()
                ):
                    raise ValueError("Invalid packed Engram hash staging view")
                host_hashes = packed_hashes.cpu()
            else:
                host_hashes = hashes.cpu()
            host_rows = [
                embedding.gather_rows(layer_hashes)[0]
                for embedding, layer_hashes in zip(embeddings, host_hashes, strict=True)
            ]
            device_rows = torch.stack(host_rows).to(hashes.device)
            fused_rows = False
            if self._decode_optimizations:
                from .engram_ops import scatter_engram_rows

                fused_rows = scatter_engram_rows(device_rows, rows)
            if not fused_rows:
                for output, layer_rows in zip(rows, device_rows, strict=True):
                    output.copy_(layer_rows)
            logger.info_once("HYV4 Decode batches CPU-offloaded BF16 Engram lookups")
            return

        for embedding, layer_hashes, output in zip(embeddings, hashes, rows, strict=True):
            output.copy_(embedding(layer_hashes))
