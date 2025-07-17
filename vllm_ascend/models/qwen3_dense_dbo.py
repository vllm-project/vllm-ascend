from collections.abc import Iterable
from typing import List, Optional, Union

import torch
import vllm.envs as envs
from torch import nn
from torch.nn.parameter import Parameter
from transformers import Qwen3Config
from vllm.attention import AttentionMetadata, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              PPMissingLayer, maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

import vllm_ascend.envs as envs_ascend
from vllm_ascend.multistream.base import MSEventKey
from vllm_ascend.multistream.context import (
    advance_step_multistream_layer_context, get_multistream_comm_context,
    get_multistream_layer_context, set_multistream_context)
from vllm_ascend.multistream.layers import (
    MultiStreamPostQwen3TransformerLayer, MultiStreamPreQwen3TransformerLayer)
from vllm_ascend.multistream.metadata import (MultiStreamConfig,
                                              MultiStreamStepMetadata,
                                              make_multistream_metadata_ds)
from vllm_ascend.multistream.ms_split import find_best_split_point

VLLM_ASCEND_ENABLE_DBO: bool = envs_ascend.VLLM_ASCEND_ENABLE_DBO


class CustomRowParallelLinearDBO(RowParallelLinear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, input_
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        assert self.quant_method is not None
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            current_ms_metadata = get_multistream_comm_context()
            if current_ms_metadata is None:
                output = tensor_model_parallel_all_reduce(output_parallel)
            else:
                current_ms_metadata.before_comm_event.record()
                with torch.npu.stream(current_ms_metadata.comm_stream):
                    current_ms_metadata.before_comm_event.wait()
                    output = tensor_model_parallel_all_reduce(output_parallel)
                    current_ms_metadata.after_comm_event.record()
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias


class CustomQwen3DBOMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = CustomRowParallelLinearDBO(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

    def _forward_ms_mlp(self, x):
        current_ms_metadata = get_multistream_comm_context()
        assert current_ms_metadata is not None
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class CustomQwen3DBOAttention(Qwen3Attention):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 head_dim: Optional[int] = None,
                 rms_norm_eps: float = 1e-06,
                 qkv_bias: bool = False,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER) -> None:
        super().__init__(hidden_size, num_heads, num_kv_heads, max_position,
                         head_dim, rms_norm_eps, qkv_bias, rope_theta,
                         cache_config, quant_config, rope_scaling, prefix,
                         attn_type)

        # rewrite o_proj, enable multi_stream
        self.o_proj = CustomRowParallelLinearDBO(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def forward(self, positions: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)

        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k, cos=cos, sin=sin, skip_index_select=True)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class CustomQwen3DBODecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = CustomQwen3DBOAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )

        self.mlp = CustomQwen3DBOMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
            self, positions: torch.Tensor, cos: torch.Tensor,
            sin: torch.Tensor, hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(positions=positions,
                                       cos=cos,
                                       sin=sin,
                                       hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def _forward_ms_layer(self, positions: List[torch.Tensor],
                          hidden_states: List[torch.Tensor],
                          residual: List[torch.Tensor],
                          attn_metadata: List[AttentionMetadata],
                          cos_sin_cache) -> tuple[torch.Tensor, torch.Tensor]:
        layer_index, ms_metadata, _ = get_multistream_layer_context()
        assert layer_index >= 0 and ms_metadata is not None
        num_micro_batches = ms_metadata.ms_config.num_micro_batches
        assert isinstance(self.mlp, CustomQwen3DBOMLP)
        assert len(positions) == num_micro_batches
        assert len(hidden_states) == num_micro_batches
        assert residual is not None
        assert attn_metadata is not None

        for i in range(num_micro_batches):
            cos_sin = cos_sin_cache.index_select(0, positions[i])
            last_dim = cos_sin.size()[-1]
            cos, sin = cos_sin.reshape(-1, 2,
                                       last_dim // 2).repeat(1, 1,
                                                             2).chunk(2,
                                                                      dim=-2)
            cos, sin = cos.view(1, -1, 1, last_dim).contiguous(), sin.view(
                1, -1, 1, last_dim).contiguous()
            ms_metadata.try_wait_event(layer_index - 1, i,
                                       MSEventKey.FFN_AR_FINISH)
            context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.ATTN_COM_FINISH],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.ATTN_AR_FINISH])

            with set_multistream_context(context, i):
                forward_context = get_forward_context()
                origin_attn_metadata = forward_context.attn_metadata
                forward_context.attn_metadata = attn_metadata[i]
                # input layer norm
                hidden_states[i], residual[
                    i] = self._forward_ms_op_input_layernorm(
                        hidden_states[i], residual[i])
                # ATTENTION
                hidden_states[i] = self._forward_ms_op_attn(
                    positions[i], cos, sin, hidden_states[i])
                forward_context.attn_metadata = origin_attn_metadata

        for i in range(num_micro_batches):
            ms_metadata.try_wait_event(layer_index, i,
                                       MSEventKey.ATTN_AR_FINISH)
            context = MultiStreamStepMetadata(
                comm_stream=ms_metadata.communicate_stream,
                before_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.FFN_COM_FINISH],
                after_comm_event=ms_metadata.ms_events[layer_index][i][
                    MSEventKey.FFN_AR_FINISH])

            with set_multistream_context(context, i):
                # post attention layer norm
                hidden_states[i], residual[
                    i] = self._forward_ms_op_post_attn_layernorm(
                        hidden_states[i], residual[i])
                # MLP
                hidden_states[i] = self.mlp._forward_ms_mlp(
                    hidden_states[i])  # MLP内部有wait和record

        return hidden_states, residual

    def _forward_ms_op_input_layernorm(self, hidden_states: torch.Tensor,
                                       residual: torch.Tensor):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        return hidden_states, residual

    def _forward_ms_op_attn(self, positions: torch.Tensor, cos: torch.Tensor,
                            sin: torch.Tensor, hidden_states: torch.Tensor):
        hidden_states = self.self_attn(positions=positions,
                                       cos=cos,
                                       sin=sin,
                                       hidden_states=hidden_states)

        return hidden_states

    def _forward_ms_op_post_attn_layernorm(self, hidden_states: torch.Tensor,
                                           residual: torch.Tensor):
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": CustomQwen3DBODecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class CustomQwen3DBOModel(Qwen2Model):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         decoder_layer_type=CustomQwen3DBODecoderLayer)

        self.cos_sin_cache = self.layers[0].self_attn.rotary_emb.cos_sin_cache
        config = vllm_config.model_config.hf_config

        if VLLM_ASCEND_ENABLE_DBO:
            self.multistream_config = MultiStreamConfig(min_total_tokens_to_split=128, imbalance_ratio=1)
            multistream_metadata = make_multistream_metadata_ds(
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                causal_lm=getattr(config, "causal_lm", True),
                multistream_config=self.multistream_config,
            )
            self.ms_pre_layer = MultiStreamPreQwen3TransformerLayer(
                multistream_metadata)
            self.ms_post_layer = MultiStreamPostQwen3TransformerLayer(
                multistream_metadata)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        graph_enable: Optional[bool] = True
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        can_run_ms = self.can_run_ms()
        # VLLM_ASCEND_ENABLE_DBO & NO graph_enable & can_run_ms -> DBO
        run_multi_stream = True if VLLM_ASCEND_ENABLE_DBO and not graph_enable and can_run_ms else False

        # run default model
        if not run_multi_stream:
            cos_sin = self.cos_sin_cache.index_select(0, positions)
            last_dim = cos_sin.size()[-1]
            cos, sin = cos_sin.reshape(-1, 2,
                                       last_dim // 2).repeat(1, 1,
                                                             2).chunk(2,
                                                                      dim=-2)
            cos, sin = cos.view(1, -1, 1, last_dim).contiguous(), sin.view(
                1, -1, 1, last_dim).contiguous()
            for layer in self.layers[self.start_layer:self.end_layer]:
                hidden_states, residual = layer(positions=positions,
                                                cos=cos,
                                                sin=sin,
                                                hidden_states=hidden_states,
                                                residual=residual)
        # run multi_stream dbo
        else:
            hidden_states, residual = self._forward_ms_layers(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                cos_sin_cache=self.cos_sin_cache)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def can_run_ms(self):
        if not envs.VLLM_USE_V1:
            return False

        if self.multistream_config is None:
            return False

        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None:
            return False

        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            return False

        split_bs_point, split_token_index = find_best_split_point(
            attn_metadata.query_lens,
            self.multistream_config.min_total_tokens_to_split,
            self.multistream_config.imbalance_ratio)
        if split_bs_point == -1 or split_token_index == -1:
            return False

        return True

    def _forward_ms_layers(self, positions: torch.Tensor,
                           hidden_states: torch.Tensor, residual: torch.Tensor,
                           cos_sin_cache):
        attn_metadata, (positions, hidden_states,
                        residual) = self.ms_pre_layer(
                            [positions, hidden_states, residual])

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer._forward_ms_layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                attn_metadata=attn_metadata,
                cos_sin_cache=cos_sin_cache)
            advance_step_multistream_layer_context()

        [hidden_states,
         residual] = self.ms_post_layer([hidden_states, residual])

        return hidden_states, residual


class CustomQwen3DBOForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = CustomQwen3DBOModel(vllm_config=vllm_config,
                                         prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        graph_enable: Optional[bool] = True
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, graph_enable)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
