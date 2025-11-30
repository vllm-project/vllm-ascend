from typing import Any, Callable, Tuple

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              get_world_group)
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors
from xlite._C import AttnMeta, AttnMHA, Model, ModelConfig, Runtime

import vllm_ascend.envs as envs_ascend
from vllm_ascend.attention.attention_v1 import (AscendAttentionState,
                                                AscendMetadata)
from vllm_ascend.utils import is_enable_nz


class XliteModel:

    def initialize(
            self, runnable: nn.Module,
            vllm_config: VllmConfig) -> Tuple[Model, int, int, torch.dtype]:
        raise NotImplementedError(
            "Xlite Model initialize function not implemented.")


class LlamaXliteModel(XliteModel):

    def initialize(
            self, runnable: nn.Module,
            vllm_config: VllmConfig) -> Tuple[Model, int, int, torch.dtype]:
        params_dict = dict(runnable.named_parameters())
        layers = runnable.model.layers

        hf_config = vllm_config.model_config.hf_config
        config = ModelConfig()
        config.vocab_size = hf_config.vocab_size
        config.hidden_size = hf_config.hidden_size
        config.n_layers = hf_config.num_hidden_layers
        config.n_heads = hf_config.num_attention_heads
        config.n_kv_heads = hf_config.num_key_value_heads
        config.head_dim = hf_config.head_dim
        config.rope_head_dim = hf_config.head_dim
        config.norm_eps = hf_config.rms_norm_eps
        config.rope_theta = hf_config.rope_theta
        config.softmax_scale = hf_config.head_dim**-0.5
        config.n_dense_layers = hf_config.num_hidden_layers
        config.intermediate_size = hf_config.intermediate_size
        config.def_tp_size = get_tensor_model_parallel_world_size()
        config.def_dp_size = 1
        config.moe_ep_size = 1
        config.moe_tp_size = 1

        config.attn_type = AttnMHA
        config.weight_nz = is_enable_nz()
        scheduler_config = vllm_config.scheduler_config
        max_batch_size = scheduler_config.max_num_seqs
        max_seq_len = scheduler_config.max_model_len
        config.max_m = scheduler_config.max_num_batched_tokens
        config.max_batch_size = max_batch_size
        config.max_seq_len = max_seq_len
        config.block_size = vllm_config.cache_config.block_size

        xlite_model = Model()
        xlite_model.embed = params_dict.get("model.embed_tokens.weight")
        xlite_model.norm = params_dict.get("model.norm.weight")
        xlite_model.head = params_dict.get("lm_head.weight")
        xlite_model.attn_norm = [
            layer.input_layernorm.weight for layer in layers
        ]
        xlite_model.attn_out = [
            layer.self_attn.o_proj.weight for layer in layers
        ]
        xlite_model.mha_qkv = [
            layer.self_attn.qkv_proj.weight for layer in layers
        ]
        xlite_model.mlp_norm = [
            layer.post_attention_layernorm.weight for layer in layers
        ]
        xlite_model.mlp_up_gate = [
            layer.mlp.gate_up_proj.weight for layer in layers
        ]
        xlite_model.mlp_down = [layer.mlp.down_proj.weight for layer in layers]
        mha_qkv_bias = [
            layer.self_attn.qkv_proj.bias for layer in layers
            if hasattr(layer.self_attn.qkv_proj, "bias")
            and layer.self_attn.qkv_proj.bias is not None
        ]
        q_norm = [
            layer.self_attn.q_norm.weight for layer in layers
            if hasattr(layer.self_attn, "q_norm")
        ]
        k_norm = [
            layer.self_attn.k_norm.weight for layer in layers
            if hasattr(layer.self_attn, "k_norm")
        ]
        if len(mha_qkv_bias) != hf_config.num_hidden_layers:
            config.qkv_bias = False
        else:
            config.qkv_bias = True
            xlite_model.mha_qkv_bias = mha_qkv_bias

        if (len(q_norm) != hf_config.num_hidden_layers
                or len(k_norm) != hf_config.num_hidden_layers):
            config.qk_norm = False
        else:
            config.qk_norm = True
            xlite_model.mha_q_norm = q_norm
            xlite_model.mha_k_norm = k_norm

        rank = torch.distributed.get_rank()
        xlite_model.init(config, rank)

        if xlite_model.embed is None:
            raise ValueError("xlite_model.embed is None, cannot get dtype!")

        freq_cis = self.precompute_freqs_cis(config.head_dim, max_seq_len,
                                             xlite_model.embed.dtype,
                                             hf_config.rope_theta)

        return (xlite_model, freq_cis, config.hidden_size,
                xlite_model.embed.dtype)

    def precompute_freqs_cis(self,
                             dim: int,
                             end: int,
                             dtype: torch.dtype,
                             theta: float = 10000.0):
        freqs = 1.0 / (theta**(torch.arange(
            0, dim, 2, dtype=torch.float32, device='cpu')[:(dim // 2)] / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        cos_cache = freqs.cos().to(dtype)
        sin_cache = freqs.sin().to(dtype)
        freq_cis = torch.cat((cos_cache, sin_cache), dim=-1)
        return freq_cis.to(device='npu')


def xlite_model_init(
        runnable: nn.Module,
        vllm_config: VllmConfig) -> Tuple[Model, int, int, torch.dtype]:
    strategy_map = {
        "LlamaForCausalLM": LlamaXliteModel,
        "Qwen2ForCausalLM": LlamaXliteModel,
        "Qwen3ForCausalLM": LlamaXliteModel,
    }

    architecture = vllm_config.model_config.architectures[0]
    strategy_class = strategy_map.get(architecture)
    if not strategy_class:
        raise ValueError(f"{architecture} not supported!")
    return strategy_class().initialize(runnable, vllm_config)


class XliteWrapper:
    """
    xlite graph wrapper
    """

    def __init__(self, runnable: nn.Module, vllm_config: VllmConfig):
        self.runnable = runnable

        rank = torch.distributed.get_rank()
        local_rank = get_world_group().local_rank
        self.xlite_rt = Runtime(local_rank, 0, rank,
                                get_tensor_model_parallel_world_size())

        (self.xlite_model, self.freq_cis, self.hidden_size,
         self.dtype) = xlite_model_init(runnable, vllm_config)

        rt_pool_size = self.xlite_model.get_tensor_pool_size()
        if rank == 0:
            logger.info(f"xlite runtime pool size: {rt_pool_size} MB")
        if self.xlite_rt.init_tensor_pool(rt_pool_size) != 0:
            raise ValueError(
                f"xlite wrapper init failed! runtime pool size: {rt_pool_size} MB"
            )

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"xlite wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def register_kv_caches(self, kv_caches: Any):
        self.kv_caches = kv_caches

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor,
                                                    list[torch.Tensor]]:
        forward_context = get_forward_context()
        attn_metadata: Any = forward_context.attn_metadata
        if attn_metadata is None:
            return self.runnable(input_ids, positions, intermediate_tensors,
                                 inputs_embeds)

        attn_metadata = next(iter(attn_metadata.values()), None)
        if attn_metadata is None or not isinstance(attn_metadata,
                                                   AscendMetadata):
            return self.runnable(input_ids, positions, intermediate_tensors,
                                 inputs_embeds)

        with_prefill = attn_metadata.attn_state not in [
            AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding
        ]

        if not with_prefill or envs_ascend.VLLM_ASCEND_ENABLE_XLITE > 1:
            batch = attn_metadata.num_prefills + attn_metadata.num_decodes
            seq_lens = attn_metadata.seq_lens[:batch]
            query_lens = attn_metadata.query_lens[:batch]
            cached_lens = seq_lens - query_lens

            xlite_attn_metadata = AttnMeta()
            xlite_attn_metadata.lens = query_lens.tolist()
            xlite_attn_metadata.cached_lens = cached_lens.tolist()
            xlite_attn_metadata.is_prefills = [
                False
            ] * attn_metadata.num_decodes + [True] * attn_metadata.num_prefills
            xlite_attn_metadata.block_tables = attn_metadata.block_tables
            xlite_attn_metadata.slot_mapping = attn_metadata.slot_mapping
            xlite_attn_metadata.positions = positions

            hidden_states = torch.empty(attn_metadata.num_actual_tokens,
                                        self.hidden_size,
                                        device=input_ids.device,
                                        dtype=self.dtype)
            # xlite utilizes a self-managed stream for operator dispatch,
            # and it is necessary to synchronize with the current stream
            # of torch.npu here to ensure the validity of the NPU tensors
            # in the prepare input process.
            torch.npu.synchronize()

            if inputs_embeds is None:
                self.xlite_model.forward(self.xlite_rt, input_ids,
                                         xlite_attn_metadata, self.kv_caches,
                                         self.freq_cis, hidden_states)
            else:
                self.xlite_model.forward_with_inputs_embeds(
                    self.xlite_rt, inputs_embeds, xlite_attn_metadata,
                    self.kv_caches, self.freq_cis, hidden_states)
            return hidden_states
        else:
            return self.runnable(input_ids, positions, intermediate_tensors,
                                 inputs_embeds)
