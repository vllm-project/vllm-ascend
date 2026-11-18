# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_ep_group, get_tp_group
from vllm.models.hy_v4_flash.model import HYV4FlashDecoderLayer, HYV4FlashForCausalLM, HYV4FlashModel

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather, sp_shard

from .attention import AscendHYV4FlashAttention, FlashAttentionBackend
from .dfly_config import METHOD
from .engram import FlashEngram, FlashNgramHashState
from .ihc import FlashHCHeadLayer, FlashHCLayer
from .moe import AscendFlashMoE

FLASHCOMM_950_PEER_BYTES = 768 * 1024


def validate_runtime(vllm_config: VllmConfig) -> None:
    model = vllm_config.model_config
    config = model.hf_text_config
    config.validate_flash_config()
    parallel = vllm_config.parallel_config
    for field in (
        "data_parallel_size",
        "pipeline_parallel_size",
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
    if spec is not None and (
        getattr(spec, "custom_method", None) != METHOD or vllm_config.cache_config.enable_prefix_caching
    ):
        raise ValueError("HYV4 Flash Ascend requires DFly with prefix caching disabled for speculative decoding")
    if vllm_config.kv_transfer_config is not None or vllm_config.lora_config is not None:
        raise ValueError("HYV4 Flash Ascend does not support KV transfer or LoRA")
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
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.uses_dfly = vllm_config.speculative_config is not None

    def embed_input_ids(self, input_ids):
        # Async speculative batches can contain -1 at discarded query slots.
        return super().embed_input_ids(input_ids.clamp_min(0) if self.uses_dfly else input_ids)


class AscendHYV4FlashForCausalLM(HYV4FlashForCausalLM):
    model_cls = AscendHYV4FlashModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        validate_runtime(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @staticmethod
    def get_model_state_cls():
        from .model_state import FlashModelState

        return FlashModelState
