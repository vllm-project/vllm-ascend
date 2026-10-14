# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only IQuestQ1 model with hybrid attention.

Ported from the upstream vLLM plugin ``IQuestLab/vllm-iquest-q1`` for Ascend NPU.

Every layer here is stock vLLM (``QKVParallelLinear`` / ``FusedMoEFactory`` /
``GateLinear`` / ``get_rope`` / ``SiluAndMul`` / ...), so vllm-ascend swaps them
for its Ascend implementations automatically via ``REGISTERED_ASCEND_OPS``.
The only Ascend-specific piece is the attention layer: ``IQuestQ1AscendAttention``
plus its backend in ``attention.py``, which folds in the learned sink key using
the softmax LSE returned by ``npu_fused_infer_attention_score_v2``.

v0 scope: no MTP / speculative decoding, no quantization.
Run with ``--enforce-eager`` (the sink path is not ACL-graph compatible yet).
"""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.model import str_dtype_to_torch_dtype
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

logger = init_logger(__name__)


def get_layer_sliding_window_size(
    first_layers_types: list[str],
    last_layers_types: list[str],
    hybrid_layers_types_block: list[str],
    num_hybrid_layers_types_block: int,
    layer_idx: int,
    sliding_window_size: int,
    is_mtp_layer: bool = False,
) -> int | None:
    if is_mtp_layer:
        return None

    num_first_layers = len(first_layers_types)
    num_hybrid_block_layers = (
        len(hybrid_layers_types_block) * num_hybrid_layers_types_block
    )

    if layer_idx < num_first_layers:
        current_layer_type = first_layers_types[layer_idx]
    elif layer_idx < num_first_layers + num_hybrid_block_layers:
        effective_layer_idx = layer_idx - num_first_layers
        current_layer_type = hybrid_layers_types_block[
            effective_layer_idx % len(hybrid_layers_types_block)
        ]
    else:
        effective_layer_idx = layer_idx - num_first_layers - num_hybrid_block_layers
        current_layer_type = last_layers_types[effective_layer_idx]

    if current_layer_type == "full_attention":
        return None
    else:
        return sliding_window_size


class IQuestQ1RMSNorm(nn.Module):
    """RMSNorm (equivalent to T5LayerNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        result = (self.weight * hidden_states).to(input_dtype)
        return result

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class IQuestQ1MoEBlock(nn.Module):
    """IQuestQ1 top-k softmax routing with tensor or expert parallel experts."""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        router_dtype: torch.dtype = torch.float32,
        tp_size: int | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.gate = GateLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=None if router_dtype == torch.float32 else router_dtype,
            out_dtype=router_dtype,
            force_fp32_compute=router_dtype == torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.experts = FusedMoEFactory(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=True,
            renormalize=True,
            router_logits_dtype=router_dtype,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        return final_hidden_states.view(orig_shape)


class IQuestQ1DenseMLP(nn.Module):
    """MLP for dense layers and optional shared expert (with optional gate)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
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
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)
        return out


class IQuestQ1Attention(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        is_mtp_layer: bool = False,
        draft_sliding_window: int | None = None,
        draft_rope_theta: float | None = None,
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.cache_config = vllm_config.cache_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 4096)

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        softmax_scale = getattr(config, "softmax_scale", None)
        self.scaling = self.head_dim**-0.5 if softmax_scale is None else softmax_scale
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.tp_size = tp_size
        self.tp_rank = get_tensor_model_parallel_rank()
        self.q_norm = IQuestQ1RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_dim = getattr(config, "rotary_dim", None)
        rope_parameters = dict(getattr(config, "rope_parameters", None) or {})
        if self.rotary_dim is not None:
            rope_parameters["partial_rotary_factor"] = self.rotary_dim / self.head_dim

        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta:
            rope_parameters["rope_theta"] = rope_theta

        layer_idx = extract_layer_index(prefix)
        use_hybrid_layers = getattr(config, "use_hybrid_layers", False)
        real_sliding_window = None
        if use_hybrid_layers:
            first_layers_types = config.first_layers_types
            hybrid_layers_types_block = config.hybrid_layers_types_block
            num_hybrid_layers_types_block = config.num_hybrid_layers_block
            last_layers_types = config.last_layers_types
            sliding_window_size = config.sliding_window
            real_sliding_window = get_layer_sliding_window_size(
                first_layers_types=first_layers_types,
                last_layers_types=last_layers_types,
                hybrid_layers_types_block=hybrid_layers_types_block,
                num_hybrid_layers_types_block=num_hybrid_layers_types_block,
                layer_idx=layer_idx,
                sliding_window_size=sliding_window_size,
                is_mtp_layer=is_mtp_layer,
            )
            if real_sliding_window:
                rope_parameters["rope_theta"] = config.swa_rope_theta

        if draft_sliding_window is not None:
            real_sliding_window = draft_sliding_window
        if draft_rope_theta is not None:
            rope_parameters["rope_theta"] = draft_rope_theta

        no_rope_layers = getattr(config, "no_rope_layers", [])
        current_layer_no_rope = layer_idx in no_rope_layers

        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=max_position_embeddings,
                rope_parameters=rope_parameters,
                is_neox_style=True,
            )
            if not current_layer_no_rope
            else None
        )

        self.shared_kv_num_layers = config.shared_kv_num_layers
        kv_sharing_target_layer_name = None
        self.cross_kv_cache = False
        if self.shared_kv_num_layers and not is_mtp_layer:
            # use shared kv cache
            # attn name is like:
            # 'model.layers.0.self_attn.attn', 'model.layers.1.self_attn.attn'
            self.shared_kv_source_begin = config.shared_kv_source_begin
            self.shared_kv_target_begin = config.shared_kv_target_begin
            if (
                layer_idx >= self.shared_kv_target_begin
                and layer_idx < self.shared_kv_target_begin + self.shared_kv_num_layers
            ):
                current_layer_name = f"{prefix}.attn"
                layer_offset = layer_idx - self.shared_kv_target_begin
                target_layer_idx = self.shared_kv_source_begin + layer_offset
                kv_sharing_target_layer_name = current_layer_name.replace(
                    f"layers.{layer_idx}", f"layers.{target_layer_idx}"
                )
                self.cross_kv_cache = True
                self.k_norm = None
            else:
                self.k_norm = IQuestQ1RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.k_norm = IQuestQ1RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.enable_sink_attention = getattr(config, "enable_sink_attention", False)
        attn_cls: type[nn.Module] = Attention
        sink_args = {}
        if self.enable_sink_attention:
            from .attention import IQuestQ1AscendAttention

            self.sink_k = nn.Parameter(
                torch.zeros(self.num_kv_heads, self.head_dim), requires_grad=False
            )
            set_weight_attrs(self.sink_k, {"weight_loader": self.sinks_k_weight_loader})
            attn_cls = IQuestQ1AscendAttention
            sink_args["sink_key"] = self.sink_k

        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=self.cache_config,
            per_layer_sliding_window=real_sliding_window,
            quant_config=quant_config,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            prefix=f"{prefix}.attn",
            **sink_args,
        )

    def sinks_k_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        replicas = max(1, self.tp_size // self.total_num_kv_heads)
        weight_shard_start = (self.tp_rank // replicas) * self.num_kv_heads
        weight_shard_end = weight_shard_start + self.num_kv_heads
        loaded_weight = loaded_weight[weight_shard_start:weight_shard_end]
        default_weight_loader(param, loaded_weight)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        if self.cross_kv_cache:
            if self.rotary_emb is not None:
                q, _ = self.rotary_emb(positions, q, None)
            attn_output = self.attn(q, None, None)
        else:
            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim
            )
            assert self.k_norm is not None
            k_by_head = self.k_norm(k_by_head)
            k = k_by_head.view(k.shape)

            if self.rotary_emb:
                q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class IQuestQ1DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        self.layer_idx = extract_layer_index(prefix)

        self.self_attn = IQuestQ1Attention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )
        self.use_sandwich_norm = self.layer_idx == 0

        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        if self.layer_idx not in mlp_only_layers:
            self.mlp = IQuestQ1MoEBlock(
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                router_dtype=str_dtype_to_torch_dtype(config.moe_router_dtype),
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = IQuestQ1DenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.dense_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.attention_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_out_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.feed_forward_norm = IQuestQ1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if self.use_sandwich_norm:
            self.ffn_out_norm = IQuestQ1RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

        if self.layer_idx == 0:
            self.attn_out_scale = getattr(config, "first_layer_attn_out_scale", 1.0)
            self.ffn_out_scale = getattr(config, "first_layer_ffn_out_scale", 1.0)
        else:
            self.attn_out_scale = getattr(config, "attn_out_scale", 1.0)
            self.ffn_out_scale = getattr(config, "ffn_out_scale", 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Self Attention
        if self.use_sandwich_norm:
            norm_hidden_states = self.attention_norm(hidden_states)
            attn_output = self.self_attn(
                positions=positions, hidden_states=norm_hidden_states
            )
            h = hidden_states + self.attn_out_norm(attn_output) * self.attn_out_scale

            # fully connected
            hidden_states = self.mlp(self.feed_forward_norm(h))
            output = h + self.ffn_out_norm(hidden_states) * self.ffn_out_scale
            return output
        else:
            x = self.attention_norm(hidden_states)
            attn_output = self.self_attn(positions=positions, hidden_states=x)
            h = x + self.attn_out_norm(attn_output) * self.attn_out_scale

            # fully connected
            ffn_out = self.mlp(self.feed_forward_norm(h))
            output = h + ffn_out * self.ffn_out_scale
            return output


@support_torch_compile
class IQuestQ1Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = IQuestQ1DecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config

        self.vocab_size = config.vocab_size
        self.config = config
        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = IQuestQ1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

        self.use_oe_embedding = getattr(config, "use_over_encoding", False)
        if self.use_oe_embedding:
            raise NotImplementedError(
                "IQuestQ1 over-encoding embeddings are not supported"
            )
        self.enable_sink_attention = getattr(config, "enable_sink_attention", False)

    def embed_input_ids(
        self, input_ids: torch.Tensor, oe_input_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        oe_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_input_ids(
                    input_ids, oe_input_ids=oe_input_ids
                )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(
                positions,
                hidden_states,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    if not self.enable_sink_attention and "sink_k" in name:
                        logger.warning_once("sink attention feature is disabled")
                        continue

                    if "experts.fc" in name:
                        # Shape: [experts, 2 * intermediate_size, hidden_size].
                        name = name.replace(
                            "experts.fc", "experts.routed_experts.w13_weight"
                        )
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        for expert_id in range(self.config.num_experts):
                            # w1 shard
                            weight_loader(
                                param,
                                loaded_weight[expert_id][
                                    : self.config.intermediate_size
                                ],
                                name,
                                shard_id="w1",
                                expert_id=expert_id,
                            )
                            # w3 shard
                            weight_loader(
                                param,
                                loaded_weight[expert_id][
                                    self.config.intermediate_size :
                                ],
                                name,
                                shard_id="w3",
                                expert_id=expert_id,
                            )
                        loaded_params.add(name)
                        continue

                    if not self.use_oe_embedding and "over_encoding" in name:
                        logger.warning_once("over encoding feature is disabled")
                        continue

                    if "experts.proj" in name:
                        # Shape: [experts, hidden_size, intermediate_size].
                        name = name.replace(
                            "experts.proj", "experts.routed_experts.w2_weight"
                        )
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        for expert_id in range(self.config.num_experts):
                            weight_loader(
                                param,
                                loaded_weight[expert_id],
                                name,
                                shard_id="w2",
                                expert_id=expert_id,
                            )
                        loaded_params.add(name)
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name

                    if "router.weight" in name:
                        name = name.replace("router.weight", "gate.weight")

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class IQuestQ1ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"mtp_layers.": None})

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = IQuestQ1DecoderLayer,
    ):
        super().__init__()
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None and speculative_config.method == "mtp":
            raise ValueError(
                "IQuestQ1 requires method='eagle' with a standalone recursive draft"
            )
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = IQuestQ1Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(
            config.vocab_size, scale=getattr(config, "logit_scale", 1.0)
        )
        self.logits_processor.head_dtype = (
            torch.float32
            if config.enable_lm_head_fp32
            else vllm_config.model_config.dtype
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        oe_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            oe_input_ids=oe_input_ids,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
