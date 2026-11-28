# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from copy import copy

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_ep_group, get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.utils import copy_pp_buffers, is_pp_missing_parameter
from vllm.models.hy_v4.nvidia.model import HYV4Model
from vllm.models.hy_v4_flash.model import HYV4FlashDecoderLayer, HYV4FlashForCausalLM, HYV4FlashModel
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.device_config import get_ascend_device_type, is_950
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.logger import init_logger_ascend
from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather, sp_shard

from .attention import AscendHYV4FlashAttention, FlashAttentionBackend
from .config import METHOD, uses_prefill_ops
from .engram import FlashEngram, FlashNgramHashState
from .engram_ops import PrefetchedEngramRows
from .ihc import FlashHCHeadLayer, FlashHCLayer
from .moe import AscendFlashMoE

logger = init_logger_ascend(__name__)

FLASHCOMM_950_PEER_BYTES = 768 * 1024


def validate_runtime(vllm_config: VllmConfig) -> None:
    model = vllm_config.model_config
    config = model.hf_text_config
    config.validate_flash_config()
    parallel = vllm_config.parallel_config
    if parallel.pipeline_parallel_size > 1 and not (
        model.enforce_eager
        and uses_prefill_ops(vllm_config)
        and config.enable_ihc
        and vllm_config.speculative_config is None
        and not parallel.use_sequence_parallel_moe
    ):
        raise ValueError(
            "HYV4 Flash Ascend pipeline parallelism requires eager iHC prefill "
            "with speculative decoding and sequence-parallel MoE disabled"
        )
    for field in (
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, field) != 1:
            raise ValueError(f"HYV4 Flash Ascend requires {field}=1")
    if parallel.enable_eplb or parallel.enable_dbo:
        raise ValueError("HYV4 Flash Ascend does not support EPLB or dual batch overlap")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("HYV4 Flash Ascend requires MRV2")
    if vllm_config.compilation_config.cudagraph_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY):
        raise ValueError("HYV4 Flash Ascend supports eager or FULL_DECODE_ONLY ACL graphs")
    if model.dtype != torch.bfloat16:
        raise ValueError("HYV4 Flash Ascend requires BF16 compute")
    if not FlashAttentionBackend.supports_kv_cache_dtype(vllm_config.cache_config.cache_dtype):
        raise ValueError("HYV4 Flash Ascend requires BF16 or FP8 KV cache")
    if model.quantization == "fp8":
        block = getattr(vllm_config.quant_config, "weight_block_size", None)
        if tuple(block or ()) != (128, 128):
            raise ValueError("HYV4 Flash Ascend requires FP8 weights with 128x128 block scales")
    elif model.quantization is not None:
        raise ValueError("HYV4 Flash Ascend accepts BF16 or native block-wise FP8 checkpoints")
    if get_ascend_config().enable_fused_mc2:
        raise ValueError("HYV4 Flash Ascend requires fused MC2 disabled")
    spec = vllm_config.speculative_config
    if spec is not None and getattr(spec, "custom_method", None) != METHOD:
        raise ValueError("HYV4 Flash Ascend requires DFly for speculative decoding")
    # Packed cache pages include IndexPool and position state. Mooncake V2 pull
    # copies these replicated MLA pages whole, including across unequal TP sizes.
    transfer = vllm_config.kv_transfer_config
    if transfer is not None and (
        transfer.kv_connector != "MooncakeConnectorV2" or transfer.get_from_extra_config("use_layerwise", False)
    ):
        raise ValueError("HYV4 Flash Ascend PD requires MooncakeConnectorV2 in pull mode")
    if vllm_config.lora_config is not None:
        raise ValueError("HYV4 Flash Ascend does not support LoRA")
    if config.kv_lora_rank != 512 or config.qk_rope_head_dim != 64:
        raise ValueError("HYV4 Flash requires a 512-wide MLA latent and 64 rotary dimensions")
    if config.engram_layer_ids and not config.enable_ihc:
        raise ValueError("HYV4 Flash Engram requires iHC")
    if not (config.learnable_sink and config.gated_mla and config.gating_type == "elementwise"):
        raise ValueError("HYV4 Flash requires learned sinks and elementwise MLA gates")
    if (
        getattr(config, "gate_lora_rank", None) is not None
        or getattr(config, "o_lora_rank", None) is not None
        or getattr(config, "gate_groups", 1) != 1
        or getattr(config, "o_groups", 1) != 1
        or getattr(config, "full_nope", False)
        or not getattr(config, "qk_norm", True)
        or config.qk_rope_head_dim != config.swa_qk_rope_head_dim
    ):
        raise ValueError("HYV4 Flash Ascend requires ungrouped MLA projections and equal rotary widths")
    for field in ("num_attention_heads", "swa_num_attention_heads"):
        if getattr(config, field) % parallel.tensor_parallel_size:
            raise ValueError(f"{field} must be divisible by tensor_parallel_size")


class AscendHYV4FlashDecoderLayer(HYV4FlashDecoderLayer):
    attention_cls = AscendHYV4FlashAttention
    hc_layer_cls = FlashHCLayer
    moe_cls = AscendFlashMoE

    def __init__(self, config, vllm_config, cache_config=None, quant_config=None, prefix="", topk_indices_buffer=None):
        super().__init__(config, vllm_config, cache_config, quant_config, prefix, topk_indices_buffer)
        if self.self_attn.use_cann_prefill:
            self.hc_attn_layer.configure_prefill_post()
            self.hc_mlp_layer.configure_prefill_post()
        if vllm_config.kv_transfer_config is not None and vllm_config.kv_transfer_config.kv_role == "kv_consumer":
            self.hc_attn_layer.configure_decode_post()
            self.hc_mlp_layer.configure_decode_post()
        self.sequence_parallel = vllm_config.parallel_config.use_sequence_parallel_moe and self.block_type == "moe"
        if self.sequence_parallel:
            layers = config.mlp_layer_types
            engram_layers = config.engram_layer_ids or ()
            index = self.layer_idx
            # Engram consumes full token rows before the decoder layer.
            self.input_is_sharded = index > 0 and layers[index - 1] != "dense" and index + 1 not in engram_layers
            self.keep_output_sharded = (
                index + 1 < len(layers) and layers[index + 1] != "dense" and index + 2 not in engram_layers
            )
            self.self_attn.o_proj.reduce_results = False
            self.self_attn.sequence_parallel_output = True
            if self.enable_ihc:
                self.hc_attn_layer.hc_pre.sequence_parallel_input = self.input_is_sharded
                self.hc_mlp_layer.hc_pre.sequence_parallel_input = True

    def _forward_normal(self, positions, hidden_states, residual):
        if not self.sequence_parallel:
            return super()._forward_normal(positions, hidden_states, residual)

        num_tokens = positions.shape[-1]
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.input_is_sharded:
            hidden_states = sp_all_gather(hidden_states)[:num_tokens]
        else:
            residual = sp_shard(residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        residual = hidden_states + residual
        hidden_states = self.mlp(self.post_attention_layernorm(residual))
        if not self.keep_output_sharded:
            hidden_states = sp_all_gather(hidden_states)[:num_tokens]
            residual = sp_all_gather(residual)[:num_tokens]
        return hidden_states, residual

    def _forward_ihc(self, positions, hidden_states):
        if not self.sequence_parallel:
            return super()._forward_ihc(positions, hidden_states)

        num_tokens = positions.shape[-1]
        hidden_states = self.hc_attn_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_attn_layer.pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        if self.input_is_sharded:
            hidden_states = sp_all_gather(hidden_states)[:num_tokens]
        else:
            residual = sp_shard(residual)
            post_gates = sp_shard(post_gates)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.hc_attn_layer.post(hidden_states, residual, post_gates)

        hidden_states = self.hc_mlp_layer.prepare_input(hidden_states)
        hidden_states, post_gates, residual = self.hc_mlp_layer.pre(hidden_states)
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post_gates)
        if not self.keep_output_sharded:
            hidden_states = sp_all_gather(hidden_states)[:num_tokens]
        return hidden_states, None


class AscendHYV4FlashModel(HYV4FlashModel):
    decoder_layer_cls = AscendHYV4FlashDecoderLayer
    hc_head_cls = FlashHCHeadLayer
    engram_cls = FlashEngram
    engram_hash_cls = FlashNgramHashState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.parallel_config.use_sequence_parallel_moe and get_ascend_device_type() == AscendDeviceType.A5:
            # Large prefill transfers can select the multi-channel HCCL path.
            # Cap each peer's payload on the shared TP and routed EP groups.
            for group in (get_tp_group(), get_ep_group()):
                group.device_communicator.max_sequence_collective_peer_bytes = FLASHCOMM_950_PEER_BYTES
        pipeline_parallel = vllm_config.parallel_config.pipeline_parallel_size > 1
        if pipeline_parallel:
            # The HYV4 backbone partitions decoder layers; the upstream Flash
            # constructor would additionally attach Engram to off-stage placeholders.
            HYV4Model.__init__(self, vllm_config=vllm_config, prefix=prefix)
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.pipeline_parallel = pipeline_parallel
        self.engram_hash_indices = {
            layer_id - 1: index
            for index, layer_id in enumerate(self.config.engram_layer_ids or ())
            if self.start_layer <= layer_id - 1 < self.end_layer
        }
        if pipeline_parallel:
            self._init_stage_engram(vllm_config, prefix)
        self.uses_dfly = vllm_config.speculative_config is not None
        self._engram_prefetch_stream = None
        if self.engram_hash is not None:
            self.engram_hash.use_fused_prefill = is_950() and uses_prefill_ops(vllm_config)

    def _init_stage_engram(self, vllm_config, prefix):
        config = self.config
        local_ids = [layer_index + 1 for layer_index in self.engram_hash_indices]
        self.loaded_parts = {}
        self.engram_hash = None
        if local_ids:
            # Hash seeds and offsets use the global order of Engram layers.
            self.engram_hash = self.engram_hash_cls(config, dummy=vllm_config.load_config.load_format == "dummy")
            for layer_id in local_ids:
                self.layers[layer_id - 1].engram_layer = self.engram_cls(
                    vllm_config, self.engram_hash.layout, layer_id, f"{prefix}.layers.{layer_id - 1}.engram_layer"
                )
        self.token_lookback_depth = config.engram_max_ngram_size - 1 if local_ids else 0
        # The inherited checkpoint verifier checks every declared Engram layer.
        # Give it the stage's list without changing the shared model config.
        self.config = copy(config)
        self.config.engram_layer_ids = local_ids

    def load_weights(self, weights):
        if self.pipeline_parallel:
            weights = ((name, value) for name, value in weights if not is_pp_missing_parameter(name, self))
        return super().load_weights(weights)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
        prepared_engram=None,
        *,
        defer_hc_head=False,
        check_hc_head_rows=None,
        first_ihc_vocab=False,
    ):
        if first_ihc_vocab and input_ids is not None:
            get_forward_context().additional_kwargs["hy4_first_ihc_token_ids"] = input_ids
        if (
            prepared_engram is None
            and self.engram_hash is not None
            and self.engram_hash.use_fused_prefill
            and input_ids is not None
            and lookback_token_ids is not None
            and input_ids.device.type == "npu"
        ):
            metadata = get_forward_context().attn_metadata
            layer_name = self.layers[self.start_layer].self_attn.layer_name
            batch = metadata.get(layer_name) if isinstance(metadata, dict) else None
            if batch is not None and batch.prefill_reqs and all(batch.prefill_reqs):
                embeddings = {
                    index: self.layers[layer].engram_layer.multi_head_embedding
                    for layer, index in self.engram_hash_indices.items()
                }
                if all(embedding.map_host_weight for embedding in embeddings.values()):
                    prepared_engram = self._prefetch_engram(
                        input_ids, lookback_token_ids, layer_name, batch, embeddings
                    )
        if self.pipeline_parallel and self.engram_hash is not None:
            return self._forward_stage_with_engram(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                lookback_token_ids,
                prepared_engram,
                defer_hc_head,
            )
        if not defer_hc_head or self.engram_hash is None:
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                lookback_token_ids,
                prepared_engram,
            )
        hidden = self._forward_prefill_hidden(input_ids, positions, inputs_embeds, lookback_token_ids, prepared_engram)
        if check_hc_head_rows is not None:
            reference = self.finish_hidden_states(hidden)[check_hc_head_rows]
            selected = self.finish_hidden_states(hidden[check_hc_head_rows])
            different = torch.count_nonzero(reference != selected).item()
            max_abs = (reference.float() - selected.float()).abs().max().item()
            logger.info(
                "HY4 HC_HEAD_CHECK tokens=%d rows=%d different=%d max_abs=%g",
                hidden.shape[0],
                selected.shape[0],
                different,
                max_abs,
            )
        return hidden

    def finish_hidden_states(self, hidden):
        return self.norm(self.hc_head(hidden))

    def _forward_prefill_hidden(self, input_ids, positions, inputs_embeds, lookback_token_ids, prepared_engram):
        if input_ids is None:
            raise ValueError("Flash Engram requires original input token IDs")
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        if prepared_engram is None:
            hashes = input_ids.new_empty(
                (
                    len(self.config.engram_layer_ids),
                    input_ids.numel(),
                    self.engram_hash.offsets.shape[1],
                ),
                dtype=torch.int32,
            )
            self.engram_hash.prepare_hashes_with_output(
                input_ids,
                lookback_token_ids,
                self.layers[0].self_attn.layer_name,
                hashes,
            )
            rows = None
        else:
            hashes, rows = prepared_engram
        hash_indices = {layer_id - 1: index for index, layer_id in enumerate(self.config.engram_layer_ids)}
        for index, layer in enumerate(self.layers):
            if index in hash_indices:
                hidden = layer.hc_attn_layer.prepare_input(hidden)
                i = hash_indices[index]
                if rows is None:
                    hidden = layer.engram_layer(hidden, hashes[i])
                else:
                    hidden = layer.engram_layer(hidden, hashes[i], embedding_rows=rows[i])
            hidden, _ = layer(positions, hidden, None)
        return hidden

    def _forward_stage_with_engram(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        lookback_token_ids,
        prepared_engram,
        defer_hc_head,
    ):
        if input_ids is None and prepared_engram is None:
            raise ValueError("Flash Engram requires original input token IDs")
        pp = get_pp_group()
        if pp.is_first_rank:
            hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden = intermediate_tensors["hidden_states"]
            copy_pp_buffers(intermediate_tensors, self.pp_buffers)

        if prepared_engram is None:
            layer_name = self.layers[self.start_layer].self_attn.layer_name
            hashes = self._prepare_engram_hashes(input_ids, lookback_token_ids, layer_name)
            rows = None
        else:
            hashes, rows = prepared_engram
        for layer_index in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_index]
            if layer_index in self.engram_hash_indices:
                index = self.engram_hash_indices[layer_index]
                hidden = layer.hc_attn_layer.prepare_input(hidden)
                if rows is None:
                    hidden = layer.engram_layer(hidden, hashes[index])
                else:
                    hidden = layer.engram_layer(hidden, hashes[index], embedding_rows=rows[index])
            hidden, _ = layer(positions, hidden, None)
        if pp.is_last_rank:
            return hidden if defer_hc_head else self.finish_hidden_states(hidden)
        buffers = {name: tensor[: hidden.shape[0]] for name, tensor in self.pp_buffers.items()}
        return IntermediateTensors({"hidden_states": hidden.flatten(1), **buffers})

    def _prepare_engram_hashes(self, input_ids, lookback, layer_name):
        hashes = input_ids.new_empty(
            (len(self.engram_hash.layout.layer_ids), input_ids.numel(), self.engram_hash.offsets.shape[1]),
            dtype=torch.int32,
        )
        self.engram_hash.prepare_hashes_with_output(input_ids, lookback, layer_name, hashes)
        return hashes

    def _prefetch_engram(self, input_ids, lookback, layer_name, batch, embeddings):
        if self._engram_prefetch_stream is None:
            self._engram_prefetch_stream = torch.npu.Stream(device=input_ids.device)
        stream = self._engram_prefetch_stream
        compute_stream = torch.npu.current_stream(device=input_ids.device)
        stream.wait_stream(compute_stream)
        with torch.npu.stream(stream):
            hashes = self._prepare_engram_hashes(input_ids, lookback, layer_name)
            for tensor in (input_ids, lookback, batch.query_start_loc_device):
                if tensor is not None and tensor.device == input_ids.device:
                    tensor.record_stream(stream)
            outputs = {}
            for index, embedding in embeddings.items():
                rows = embedding.prefetch_rows(hashes[index])
                ready = torch.npu.Event()
                ready.record(stream)
                outputs[index] = (rows, ready)
        hashes.record_stream(compute_stream)
        return hashes, PrefetchedEngramRows(outputs)

    def embed_input_ids(self, input_ids):
        # Async speculative batches can contain -1 at discarded query slots.
        return super().embed_input_ids(input_ids.clamp_min(0) if self.uses_dfly else input_ids)


class AscendHYV4FlashForCausalLM(HYV4FlashForCausalLM):
    supports_pp = True
    model_cls = AscendHYV4FlashModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        validate_runtime(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._pack_prefill_engram_weights = uses_prefill_ops(vllm_config)
        transfer = vllm_config.kv_transfer_config
        self.logits_processor.use_npu_fp32_head = (
            is_950()
            and transfer is not None
            and (
                transfer.kv_role == "kv_consumer"
                or (transfer.kv_role == "kv_producer" and os.getenv("HY4_PREFILL_FP32_HEAD") == "1")
            )
        )

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
        prepared_engram=None,
        *,
        defer_hc_head=False,
        check_hc_head_rows=None,
        first_ihc_vocab=False,
    ):
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids,
            prepared_engram,
            defer_hc_head=defer_hc_head,
            check_hc_head_rows=check_hc_head_rows,
            first_ihc_vocab=first_ihc_vocab,
        )

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        if self._pack_prefill_engram_weights:
            for module in self.model.modules():
                if isinstance(module, FlashEngram):
                    module.pack_prefill_weights()
        return loaded

    @staticmethod
    def get_model_state_cls():
        from .model_state import FlashModelState

        return FlashModelState
