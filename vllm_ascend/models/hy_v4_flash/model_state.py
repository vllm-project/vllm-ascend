# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.triton_utils import triton

from vllm_ascend import envs
from vllm_ascend.ops.triton.engram_lookback import _gather_lookback_kernel
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from .aclgraph import FlashAclGraphManager

logger = init_logger(__name__)


def _prep_fusion_enabled(value):
    # Unset independent controls retain the legacy group-switch behavior.
    if value is None:
        return envs.VLLM_ASCEND_HY4_TARGET_PREP_EXTRA_FUSIONS
    return value


def _triton_hash_enabled():
    return envs.VLLM_ASCEND_HY4_ENGRAM_TRITON_HASH or envs.VLLM_ASCEND_HY4_TARGET_ROPE_HASH_FUSION is True


class FlashModelState(AscendModelState):
    aclgraph_manager_cls = FlashAclGraphManager

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        self._graph_decode = False
        self._graph_hashes = None
        if model.token_lookback_depth and vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self._flash_model = getattr(model, "language_model", model).model
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

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, req_states):
        # Experimental work remains outside Target/Draft model graphs.
        if _prep_fusion_enabled(envs.VLLM_ASCEND_HY4_TARGET_EMBED_FUSION):
            from .target_embedding_triton import try_prepare_target_embeddings

            result = try_prepare_target_embeddings(self, scheduled_encoder_inputs, input_batch, req_states)
            if result is not None:
                return result
        return super().prepare_inputs_embeds(scheduled_encoder_inputs, input_batch, req_states)

    def prepare_attn(self, input_batch, cudagraph_mode, *args, **kwargs):
        self._graph_decode = cudagraph_mode == CUDAGraphMode.FULL
        return super().prepare_attn(input_batch, cudagraph_mode, *args, **kwargs)

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
            _prep_fusion_enabled(envs.VLLM_ASCEND_HY4_TARGET_ROPE_HASH_FUSION)
            and _triton_hash_enabled()
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
            and _triton_hash_enabled()
        ):
            use_extra = _prep_fusion_enabled(envs.VLLM_ASCEND_HY4_TARGET_ROPE_HASH_FUSION)
            if not use_extra and rope_state is None:
                from .engram_hash_triton import fused_engram_hashes
            else:
                from .engram_hash_triton_extra import fused_engram_hashes

            inputs = self.prepare_engram_dummy_inputs(
                input_batch.num_reqs_after_padding, input_batch.num_tokens_after_padding
            )
            hashes, rows = inputs["prepared_engram"]
            packed_hashes = None
            if (
                use_extra
                and rows
                and all(
                    output.shape == rows[0].shape
                    and embedding.embedding.weight.device.type == "cpu"
                    and embedding.embedding.weight.dtype == torch.bfloat16
                    and not embedding.fp8
                    for embedding, output in zip(self._engram_embeddings, rows, strict=True)
                )
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
                **(
                    {
                        "rope_state": rope_state,
                        "prefill_lens": req_states.prefill_len.gpu,
                        "packed_output": packed_hashes,
                    }
                    if use_extra or rope_state is not None
                    else {}
                ),
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
        if envs.VLLM_ASCEND_HY4_ENGRAM_ROW_PREP:
            from .engram_row_prepare import try_prepare_engram_rows

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
            if envs.VLLM_ASCEND_HY4_TARGET_PREP_EXTRA_FUSIONS:
                from .engram_row_scatter import scatter_engram_rows

                fused_rows = scatter_engram_rows(device_rows, rows)
            if not fused_rows:
                for output, layer_rows in zip(rows, device_rows, strict=True):
                    output.copy_(layer_rows)
            logger.info_once("HYV4 Decode batches CPU-offloaded BF16 Engram lookups")
            return

        for embedding, layer_hashes, output in zip(embeddings, hashes, rows, strict=True):
            output.copy_(embedding(layer_hashes))
