from itertools import islice

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV2MLAAttention,
    DeepseekV2Model,
    Indexer,
    _get_llama_4_scaling,
    yarn_get_mscale,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.sequence import IntermediateTensors


def _is_index_cache_skip_layer(config, layer_id: int) -> bool:
    """True if this layer reuses top-k indices from a previous indexer-bearing
    layer according to the IndexCache pattern configured in the model config.

    This is a pure function extracted from ``_deepseek_v2_mla_attention_init``
    so it can be reused both at model-init time and at weight-loading time.
    """
    index_topk_freq = getattr(config, "index_topk_freq", 1)
    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)

    if index_topk_pattern is None:
        return max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
    if 0 <= layer_id < len(index_topk_pattern):
        return index_topk_pattern[layer_id] == "S"
    return False


def _should_skip_indexer_init(
    config: DeepseekV2Config | DeepseekV3Config,
    prefix: str,
    skip_topk: bool,
) -> bool:
    if not skip_topk:
        return False

    layer_id = extract_layer_index(prefix)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is not None and layer_id >= num_hidden_layers:
        return False

    # GLM-5.2 describes checkpoint-level shared indexers explicitly. Runtime
    # IndexCache overrides on GLM-5.1 only skip top-k computation; its
    # checkpoint still contains an Indexer for every layer.
    indexer_types = getattr(config, "indexer_types", None)
    indexer_type = indexer_types[layer_id] if indexer_types is not None and layer_id < len(indexer_types) else None
    if isinstance(indexer_type, str) and indexer_type.lower() == "shared":
        return True

    # GLM-5.1 IndexCache: layers that reuse another layer's top-k indices
    # can drop the indexer entirely — don't allocate k_cache, don't compute
    # k_li, don't scatter. The checkpoint still carries Indexer weights for
    # every layer; those are filtered out by the load_weights patch so the
    # AutoWeightsLoader never sees them.
    return bool(getattr(config, "use_index_cache", False))


def _deepseek_v2_mla_attention_init(
    self,
    vllm_config: VllmConfig,
    config: DeepseekV2Config | DeepseekV3Config,
    hidden_size: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int | None,
    kv_lora_rank: int,
    max_position_embeddings: int = 8192,
    cache_config: CacheConfig | None = None,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
    topk_indices_buffer: torch.Tensor | None = None,
    input_size: int | None = None,
    reduce_results: bool = True,
) -> None:
    # 这里不能使用 super().__init__()，因为当前函数定义在原类之外，
    # 最后通过赋值的方式替换 DeepseekV2MLAAttention.__init__。
    nn.Module.__init__(self)

    self.hidden_size = hidden_size
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    self.v_head_dim = v_head_dim

    self.q_lora_rank = q_lora_rank
    self.kv_lora_rank = kv_lora_rank

    self.num_heads = num_heads
    tp_size = get_tensor_model_parallel_world_size()
    assert num_heads % tp_size == 0
    self.num_local_heads = num_heads // tp_size

    self.scaling = self.qk_head_dim**-0.5
    self.max_position_embeddings = max_position_embeddings

    # Use input_size for projection input dimensions if provided,
    # otherwise default to hidden_size (used in Eagle3 Deepseek with MLA).
    proj_input_size = input_size if input_size is not None else self.hidden_size

    if self.q_lora_rank is not None:
        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            proj_input_size,
            [
                self.q_lora_rank,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
    else:
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            proj_input_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )

    if self.q_lora_rank is not None:
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
    else:
        self.q_proj = ColumnParallelLinear(
            proj_input_size,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

    self.kv_a_layernorm = RMSNorm(
        self.kv_lora_rank,
        eps=config.rms_norm_eps,
    )

    self.kv_b_proj = ColumnParallelLinear(
        self.kv_lora_rank,
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        bias=False,
        quant_config=quant_config,
        prefix=f"{prefix}.kv_b_proj",
    )

    self.o_proj = RowParallelLinear(
        self.num_heads * self.v_head_dim,
        self.hidden_size,
        bias=False,
        reduce_results=reduce_results,
        quant_config=quant_config,
        prefix=f"{prefix}.o_proj",
    )

    if config.rope_parameters["rope_type"] != "default":
        config.rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if config.rope_parameters.get(
                "apply_yarn_scaling",
                True,
            )
            else "deepseek_llama_scaling"
        )

    self.rotary_emb = get_rope(
        qk_rope_head_dim,
        max_position=max_position_embeddings,
        rope_parameters=config.rope_parameters,
        is_neox_style=False,
    )

    if config.rope_parameters["rope_type"] != "default" and config.rope_parameters["rope_type"] == "deepseek_yarn":
        mscale_all_dim = config.rope_parameters.get(
            "mscale_all_dim",
            False,
        )
        scaling_factor = config.rope_parameters["factor"]
        mscale = yarn_get_mscale(
            scaling_factor,
            float(mscale_all_dim),
        )
        self.scaling = self.scaling * mscale * mscale

    self.is_v32 = hasattr(config, "index_topk")

    # IndexCache config.
    #
    # skip_topk controls top-k reuse. Indexer initialization is skipped only
    # when the checkpoint marks this layer as sharing another layer's Indexer.
    layer_id = extract_layer_index(prefix)
    _skip_topk = _is_index_cache_skip_layer(config, layer_id)

    skip_indexer_init = _should_skip_indexer_init(config, prefix, _skip_topk)
    if self.is_v32 and not skip_indexer_init:
        self.indexer_rope_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=not getattr(
                config,
                "indexer_rope_interleave",
                False,
            ),
        )

        self.indexer = Indexer(
            vllm_config,
            config,
            hidden_size,
            q_lora_rank,
            quant_config,
            cache_config,
            topk_indices_buffer,
            f"{prefix}.indexer",
            is_inplace_rope=self.indexer_rope_emb.enabled(),
        )
    else:
        self.indexer_rope_emb = None
        self.indexer = None

    mla_modules = MLAModules(
        kv_a_layernorm=self.kv_a_layernorm,
        kv_b_proj=self.kv_b_proj,
        rotary_emb=self.rotary_emb,
        o_proj=self.o_proj,
        fused_qkv_a_proj=(self.fused_qkv_a_proj if self.q_lora_rank is not None else None),
        kv_a_proj_with_mqa=(self.kv_a_proj_with_mqa if self.q_lora_rank is None else None),
        q_a_layernorm=(self.q_a_layernorm if self.q_lora_rank is not None else None),
        q_b_proj=(self.q_b_proj if self.q_lora_rank is not None else None),
        q_proj=(self.q_proj if self.q_lora_rank is None else None),
        indexer=self.indexer,
        indexer_rotary_emb=self.indexer_rope_emb,
        is_sparse=self.is_v32,
        topk_indices_buffer=topk_indices_buffer,
    )

    self.mla_attn = MultiHeadLatentAttentionWrapper(
        self.hidden_size,
        self.num_local_heads,
        self.scaling,
        self.qk_nope_head_dim,
        self.qk_rope_head_dim,
        self.v_head_dim,
        self.q_lora_rank,
        self.kv_lora_rank,
        mla_modules,
        cache_config,
        quant_config,
        prefix,
        skip_topk=_skip_topk,
    )


DeepseekV2MLAAttention.__init__ = _deepseek_v2_mla_attention_init


def _patched_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided to DeepseekV2Model.forward")
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
    llama_4_scaling: torch.Tensor | None
    if llama_4_scaling_config is not None:
        llama_4_scaling = _get_llama_4_scaling(
            original_max_position_embeddings=llama_4_scaling_config["original_max_position_embeddings"],
            scaling_beta=llama_4_scaling_config["beta"],
            positions=positions,
        )
    else:
        llama_4_scaling = None

    aux_hidden_states = []
    for idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        if idx in self.aux_hidden_state_layers:
            aux_hidden_state = hidden_states + residual
            if aux_hidden_state.shape[0] != positions.shape[0]:
                aux_hidden_state = tensor_model_parallel_all_gather(aux_hidden_state, 0)
                aux_hidden_state = aux_hidden_state[: positions.shape[0]]
            aux_hidden_states.append(aux_hidden_state)
        hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

    if hidden_states.shape[0] != positions.shape[0]:
        combined_states = torch.cat([hidden_states, residual], dim=-1)
        combined_states = tensor_model_parallel_all_gather(combined_states, 0)
        combined_states = combined_states[: positions.shape[0]]
        hidden_size = self.hidden_size
        hidden_states, residual = combined_states.split([hidden_size, hidden_size], dim=-1)
        residual = residual.contiguous()

    if self.end_layer in self.aux_hidden_state_layers:
        aux_hidden_states.append(hidden_states + residual)

    hidden_states, _ = self.norm(hidden_states, residual)
    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states


DeepseekV2Model.forward = _patched_forward


# ============================================================================
# IndexCache: weight-loading patch for GLM5.1
# ============================================================================
#
# GLM-5.1 checkpoints include Indexer weights for *every* layer, but layers
# where _should_skip_indexer_init returns True don't create an Indexer module.
# The AutoWeightsLoader raises ValueError when a checkpoint weight has no
# matching parameter, so we filter out those weights before they reach the
# loader.
#
# This executes at import time (patch_gqa_c8 runs after us — but patch_gqa_c8
# *wraps* the current load_weights, so our filter naturally composes inside
# its chain: patch_gqa_c8 → our filter → original).
#
#     _patched_c8_load_weights  (C8 scale interception)
#       → _patched_glm_load_weights  (IndexCache filtering)
#         → original Glm4MoeForCausalLM.load_weights


def _is_skipped_indexer_weight(name: str, skip_ids: frozenset[int]) -> bool:
    """Return True if *name* belongs to an Indexer whose layer is in *skip_ids*."""
    if ".indexer." not in name:
        return False
    parts = name.split(".")
    try:
        i = parts.index("layers")
        return int(parts[i + 1]) in skip_ids
    except (ValueError, IndexError):
        return False


def _apply_index_cache_weight_patch() -> None:
    """Wrap ``Glm4MoeForCausalLM.load_weights`` to filter unused Indexer
    weights for GLM-5.1 IndexCache skip layers.

    Safe to call at import time — ``patch_gqa_c8`` wraps the *current*
    ``load_weights`` afterward, so our filter composes naturally.
    """
    from vllm.model_executor.models.glm4_moe import Glm4MoeForCausalLM

    _prev_load_weights = Glm4MoeForCausalLM.load_weights

    def _patched_glm_load_weights(self, weights):
        config = getattr(self, "config", None)
        if config is not None and getattr(config, "use_index_cache", False):
            num_layers = getattr(config, "num_hidden_layers", 0)
            skip_ids = frozenset(lid for lid in range(num_layers) if _is_index_cache_skip_layer(config, lid))
            if skip_ids:
                weights = ((name, w) for name, w in weights if not _is_skipped_indexer_weight(name, skip_ids))
        return _prev_load_weights(self, weights)

    Glm4MoeForCausalLM.load_weights = _patched_glm_load_weights


_apply_index_cache_weight_patch()
