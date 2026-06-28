import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
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
    Indexer,
    yarn_get_mscale,
)
from vllm.model_executor.models.utils import extract_layer_index


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
    # PR #45895 的关键修改是：
    # 1. 在创建 Indexer 前先计算当前层是否 skip_topk；
    # 2. skip_topk 的 backbone 层不创建 Indexer；
    # 3. MTP/nextn 层即使命中 skip pattern，也必须创建完整 Indexer。
    _skip_topk = False
    _index_topk_freq = getattr(
        config,
        "index_topk_freq",
        1,
    )
    _index_topk_pattern = getattr(
        config,
        "index_topk_pattern",
        None,
    )
    _index_skip_topk_offset = getattr(
        config,
        "index_skip_topk_offset",
        2,
    )

    layer_id = extract_layer_index(prefix)

    if _index_topk_pattern is None:
        _skip_topk = (
            max(
                layer_id - _index_skip_topk_offset + 1,
                0,
            )
            % _index_topk_freq
            != 0
        )
    elif 0 <= layer_id < len(_index_topk_pattern):
        _skip_topk = _index_topk_pattern[layer_id] == "S"

    # Skip pattern only governs backbone layers.
    #
    # MTP/nextn layers must always build a complete Indexer. The MTP
    # implementation computes top-k indices at draft step 0, then changes
    # skip_topk dynamically during the remaining speculative iterations.
    _num_hidden_layers = getattr(
        config,
        "num_hidden_layers",
        None,
    )
    is_mtp_layer = _num_hidden_layers is not None and layer_id >= _num_hidden_layers

    if self.is_v32 and (not _skip_topk or is_mtp_layer):
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
