import torch
import torch.nn as nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform


class MiniMaxM2DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = getattr(config, "num_key_value_heads", self.n_heads)
        self.n_local_heads = self.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.norm_eps = config.rms_norm_eps

        self.input_layernorm = RMSNorm(self.hidden_size, eps=self.norm_eps)

        self.self_attn = ColumnParallelLinear(
            self.hidden_size,
            3 * self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.n_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn.o_proj",
        )

        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=self.norm_eps)

        intermediate_size = getattr(config, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = 4 * self.hidden_size

        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.gate_up_proj",
        )

        self.down_proj = RowParallelLinear(
            intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.down_proj",
        )

        self.act_fn = SiluAndMul()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states.clone()
        hidden_states = self.input_layernorm(hidden_states)

        qkv, _ = self.self_attn(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        k = k.view(-1, self.n_local_kv_heads, self.head_dim)
        v = v.view(-1, self.n_local_kv_heads, self.head_dim)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        attn_output = attn_output.view(-1, self.n_heads * self.head_dim)

        attn_output, _ = self.o_proj(attn_output)
        hidden_states = residual + attn_output

        residual = hidden_states.clone()
        hidden_states = self.post_attention_layernorm(hidden_states)

        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, residual


@support_torch_compile
class MiniMaxM2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.device = current_platform.device_type

        from vllm.distributed import get_pp_group

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniMaxM2DecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.distributed import get_pp_group

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = None

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            from vllm.sequence import IntermediateTensors

            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class AscendMiniMaxM2ForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    model_cls = MiniMaxM2Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        from vllm.distributed import get_pp_group

        self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states