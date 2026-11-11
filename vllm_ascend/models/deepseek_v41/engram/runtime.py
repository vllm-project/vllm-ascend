# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ElasticBuffer integration with the current upstream hash/history contract."""

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_world_group, in_the_same_node_as
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import logger

from . import ops as _ops  # noqa: F401
from .elastic import ElasticEngramEmbedding, create_engram_process_group
from .executor import EngramMultiStreamExecutor
from .hash_state import engram_dead_mask


class ElasticEngramRuntime:
    """Own tables and metadata; runners only stage coordinates before replay.

    Hash history remains owned by upstream: V1 uses physical KV slots and V2
    passes its authoritative device lookback. No CPU output-token mirror is
    introduced. All table operations run inside forward, including FULL replay.
    """

    def __init__(self, model, vllm_config):
        self.model = model
        self.vllm_config = vllm_config
        self.tables = {}
        self.executor = None
        self.inputs = None

    def _validate_topology(self, expected_world_size: int) -> None:
        """Collectively verify the target domain before allocating Engram resources."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("Engram topology validation requires an initialized distributed process group")
        world_size = torch.distributed.get_world_size()
        if world_size != expected_world_size:
            raise RuntimeError(
                "Engram requires the PyTorch distributed world to span all configured "
                f"ranks; runtime={world_size}, configured={expected_world_size}"
            )
        cpu_group = get_world_group().cpu_group
        cpu_world_size = torch.distributed.get_world_size(cpu_group)
        if cpu_world_size != expected_world_size:
            raise RuntimeError(
                "Engram requires the CPU world group to span all configured "
                f"ranks; cpu_world_size={cpu_world_size}, configured={expected_world_size}"
            )
        same_node = in_the_same_node_as(cpu_group)
        if len(same_node) != expected_world_size:
            raise RuntimeError(
                "Engram same-node probe returned an incomplete domain; "
                f"probe_ranks={len(same_node)}, configured={expected_world_size}"
            )
        if not all(same_node):
            failed_ranks = [rank for rank, local in enumerate(same_node) if not local]
            raise RuntimeError(
                "ElasticBuffer Engram requires all participating ranks on the same node "
                f"and shared-memory namespace; configured={expected_world_size}, failed_ranks={failed_ranks}"
            )
        logger.info_once(
            "Engram topology verified: configured_world_size=%d runtime_world_size=%d "
            "cpu_world_size=%d node_local=True",
            expected_world_size,
            world_size,
            cpu_world_size,
        )

    def initialize(self):
        if self.tables:
            return
        from vllm_ascend.ascend_config import get_ascend_config

        size = get_ascend_config().engram_config.group_size
        try:
            self._validate_topology(size)
            for slot, layer_id in enumerate(self.model.config.engram_layer_ids):
                group, owns_group = create_engram_process_group(size, layer_id=layer_id, expected_world_size=size)
                rows = self.model.config.engram_num_embeddings[slot]
                minimum = sum(size for order in self.model.engram_layout.primes[slot] for size in order)
                try:
                    table = ElasticEngramEmbedding(
                        rows,
                        self.model.config.engram_head_dim,
                        size,
                        device=self.model.topk_indices_buffer.device,
                        group=group,
                        owns_group=owns_group,
                        layer_id=layer_id,
                        minimum_rows=minimum,
                    )
                except BaseException:
                    torch.distributed.destroy_process_group(group)
                    raise
                self.tables[layer_id] = table
                local_error = None
                try:
                    from cann_ops_transformer.ops import ElasticBuffer

                    table._check_fp8_runtime(ElasticBuffer)
                except Exception as exc:
                    local_error = exc
                ready = table._sync_phase_status(local_error is None, "runtime-abi")
                if local_error is not None:
                    raise local_error
                if not ready:
                    raise RuntimeError("Engram runtime ABI validation failed on another rank")
                table.load_checkpoint(self.model.engram_weight_root, f"layers.{layer_id}.engram.embed.weight")
                table.offload_weights()
                layer = self.model.layers[layer_id].engram
                layer.embed_tokens = table
                layer.layer_id, layer.layer_hash_index = layer_id, slot
                layer.rotation = self.model.engram_rotation if self.model.engram_rotated else None
                table.n_hash_cols = len(self.model.engram_layout.primes[slot]) * len(
                    self.model.engram_layout.primes[slot][0]
                )
                table.dim = table.width
            if self.model._engram_overlap_enabled:
                layers = {layer: self.model.layers[layer].engram for layer in self.tables}
                self.executor = EngramMultiStreamExecutor(tuple(layers), hash_fn=self, engram_layers=layers)
                for layer in layers.values():
                    layer.executor = self.executor
        except BaseException:
            self.close()
            raise

    def _buffers(self):
        if self.inputs is None:
            model, config = self.model, self.vllm_config
            device = model.engram_rotation.device
            reqs = config.scheduler_config.max_num_seqs
            depth = model.engram_hash.lookback_depth
            scheduler_block = getattr(getattr(config, "cache_config", None), "block_size", None)
            block_size = min(model.engram_hash.block_size, scheduler_block or model.engram_hash.block_size)
            blocks = (config.model_config.max_model_len + block_size - 1) // block_size
            self.inputs = {
                "query": torch.zeros(reqs + 1, dtype=torch.int32, device=device),
                "lookback": torch.full((reqs, depth), -1, dtype=torch.int32, device=device),
                "lookback_dead": torch.zeros((reqs, depth), dtype=torch.bool, device=device),
                "slots": torch.full((model._engram_max_tokens,), -1, dtype=torch.int64, device=device),
                "blocks": torch.zeros((reqs, blocks), dtype=torch.int32, device=device),
                "valid": torch.zeros(1, dtype=torch.int32, device=device),
                "dead": torch.zeros(model._engram_max_tokens, dtype=torch.bool, device=device),
            }
        return self.inputs

    def stage(
        self,
        input_ids,
        positions,
        padded_tokens=None,
        lookback_token_ids=None,
        query_start_loc=None,
        slot_mapping=None,
        block_table=None,
        *,
        token_indices=None,
        force_dummy=False,
        cg_mode=None,
        current_dead_mask=None,
        multimodal_spans=None,
    ):
        if token_indices is not None:
            raise ValueError("ElasticBuffer Engram does not support PCP token sharding")
        if cg_mode is None and is_forward_context_available():
            cg_mode = get_forward_context().cudagraph_runtime_mode
        if self.model._engram_overlap_enabled and cg_mode == CUDAGraphMode.PIECEWISE:
            raise RuntimeError("ElasticBuffer Engram overlap requires eager or FULL_DECODE_ONLY")
        capacity = self.model._engram_max_tokens
        if positions.shape[0] > capacity or (padded_tokens is not None and padded_tokens > capacity):
            raise ValueError("Engram token count exceeds metadata capacity")
        buffers = self._buffers()
        buffers["query"].zero_()
        buffers["valid"].zero_()
        buffers["lookback"].fill_(-1)
        buffers["lookback_dead"].zero_()
        buffers["slots"].fill_(-1)
        buffers["blocks"].zero_()
        buffers["dead"].zero_()
        if query_start_loc is not None and not force_dummy and query_start_loc.numel() > 1:
            if query_start_loc.numel() > buffers["query"].numel():
                raise ValueError("Engram request count exceeds metadata capacity")
            buffers["valid"].copy_(query_start_loc[-1:])
            buffers["query"].copy_(buffers["valid"].expand_as(buffers["query"]))
            buffers["query"][: query_start_loc.numel()].copy_(query_start_loc)
            if lookback_token_ids is not None:
                buffers["lookback"][: lookback_token_ids.shape[0]].copy_(lookback_token_ids)
            if slot_mapping is not None:
                count = min(positions.shape[0], slot_mapping.numel())
                buffers["slots"][:count].copy_(slot_mapping[:count])
            if block_table is not None:
                buffers["blocks"][: block_table.shape[0], : block_table.shape[1]].copy_(block_table)
            if current_dead_mask is not None:
                buffers["dead"][: current_dead_mask.numel()].copy_(current_dead_mask)
            if multimodal_spans and positions.numel():
                # Span coordinates are CPU metadata. Compare them with actual
                # device positions, never the async scheduler's optimistic CPU
                # computed-token count, including for the lookback window.
                token_indices = torch.arange(positions.shape[0], device=positions.device)
                history_offsets = torch.arange(buffers["lookback"].shape[1], device=positions.device) + 1
                for req, spans in enumerate(multimodal_spans):
                    if not spans:
                        continue
                    query = buffers["query"]
                    active = (token_indices >= query[req]) & (token_indices < query[req + 1])
                    start = positions.index_select(0, query[req : req + 1].long().clamp(0, positions.numel() - 1))
                    history_positions = start - history_offsets
                    for offset, length in spans:
                        buffers["dead"][: positions.numel()] |= (
                            active & (positions >= offset) & (positions < offset + length)
                        )
                        buffers["lookback_dead"][req] |= (history_positions >= offset) & (
                            history_positions < offset + length
                        )
        return {"engram_elastic_inputs": buffers}

    def graph_inputs(self, padded_tokens=None, *, prime=False):
        if padded_tokens is not None and padded_tokens > self.model._engram_max_tokens:
            raise ValueError("Engram token count exceeds metadata capacity")
        buffers = self._buffers()
        if prime:
            # Capture/dummy forwards may follow a real request. They must not
            # write its slot history using stale coordinates and dummy IDs.
            buffers["query"].zero_()
            buffers["valid"].zero_()
            buffers["slots"].fill_(-1)
            buffers["lookback"].fill_(-1)
            buffers["lookback_dead"].zero_()
            buffers["dead"].zero_()
        return {"engram_elastic_inputs": buffers}

    def forward_packed(
        self,
        input_ids,
        positions,
        query,
        lookback,
        slots,
        blocks,
        valid,
        dead,
        lookback_dead=None,
        single_token_per_request=False,
    ):
        model = self.model
        image = model.config.image_token_id
        image_pad = getattr(model.config, "image_pad_token_id", image + 1)
        mask = engram_dead_mask(input_ids, image, image_pad) | dead[: input_ids.shape[0]]
        mask = mask | (input_ids < 0) | (torch.arange(input_ids.shape[0], device=input_ids.device) >= valid)
        lookback_dead = (
            engram_dead_mask(lookback, image, image_pad)
            if lookback_dead is None
            else lookback_dead | engram_dead_mask(lookback, image, image_pad)
        )
        vocab_size = model.engram_hash.token_map.numel()
        mask = mask | (input_ids >= vocab_size)
        lookback_dead = lookback_dead | (lookback >= vocab_size)
        safe_ids = input_ids.clamp(0, vocab_size - 1)
        safe_lookback = lookback.clamp(-1, vocab_size - 1)
        if not model.engram_hash.ensure_cache():
            shape = (input_ids.shape[0], len(model.config.engram_layer_ids), self.hash_columns)
            return input_ids.new_zeros(shape, dtype=torch.int32), torch.zeros_like(mask)
        hashes = model.engram_hash(
            safe_ids, positions, query, mask, safe_lookback, lookback_dead, slots[: input_ids.shape[0]], blocks
        )
        return hashes, ~mask

    @property
    def hash_columns(self):
        config = self.model.config
        return (config.engram_max_ngram_size - 1) * config.engram_n_heads

    def begin(self, input_ids, positions, buffers):
        tensors = tuple(
            buffers[key] for key in ("query", "lookback", "slots", "blocks", "valid", "dead", "lookback_dead")
        )
        if self.executor is None:
            return self.forward_packed(input_ids, positions, *tensors)
        first = self.model.layers[self.model.config.engram_layer_ids[0]].engram
        return torch.ops.vllm.engram_elastic_begin(
            input_ids, positions, *tensors, first.layer_name, len(self.tables), self.hash_columns
        )

    def retire(self, *, failed=False):
        if self.executor is not None:
            if failed:
                self.executor.abort()
            else:
                self.executor.release()

    def close(self):
        errors = []
        if self.executor is not None:
            try:
                self.executor.close()
            except BaseException as exc:
                errors.append(exc)
        for layer, table in tuple(self.tables.items()):
            try:
                table.destroy()
                del self.tables[layer]
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError("Failed to close all ElasticBuffer Engram resources") from errors[0]
        compilation = getattr(self.vllm_config, "compilation_config", None)
        context = getattr(compilation, "static_forward_context", None)
        if context is not None:
            for layer_id in self.model.config.engram_layer_ids:
                layer = self.model.layers[layer_id].engram
                if context.get(layer.layer_name) is layer:
                    del context[layer.layer_name]
