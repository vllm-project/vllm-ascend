from vllm.model_executor.layers.mla import MLAModules
from vllm.model_executor.layers.mla import MLAModules
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.attention.layer import MLAAttention
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper


def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    scale: float,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    q_lora_rank: int | None,
    kv_lora_rank: int,
    mla_modules: MLAModules,
    cache_config: CacheConfig | None = None,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
) -> None:

    super(MultiHeadLatentAttentionWrapper, self).__init__()
    self.hidden_size = hidden_size
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    self.v_head_dim = v_head_dim
    self.q_lora_rank = q_lora_rank
    self.kv_lora_rank = kv_lora_rank
    self.num_heads = num_heads
    self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
    self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
    self.q_a_layernorm = mla_modules.q_a_layernorm
    self.q_b_proj = mla_modules.q_b_proj
    self.q_proj = mla_modules.q_proj
    self.kv_a_layernorm = mla_modules.kv_a_layernorm
    self.kv_b_proj = mla_modules.kv_b_proj
    self.rotary_emb = mla_modules.rotary_emb
    self.o_proj = mla_modules.o_proj
    self.indexer = mla_modules.indexer
    self.indexer_rope_emb = mla_modules.indexer_rotary_emb
    self.is_sparse = mla_modules.is_sparse

    if self.indexer is not None:
        assert hasattr(self.indexer, "topk_tokens")
        self.topk_tokens = self.indexer.topk_tokens
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

    self.mla_attn = MLAAttention(
        num_heads=self.num_heads,
        scale=scale,
        qk_nope_head_dim=self.qk_nope_head_dim,
        qk_rope_head_dim=self.qk_rope_head_dim,
        v_head_dim=self.v_head_dim,
        q_lora_rank=self.q_lora_rank,
        kv_lora_rank=self.kv_lora_rank,
        cache_config=cache_config,
        quant_config=quant_config,
        prefix=f"{prefix}.attn",
        kv_b_proj=self.kv_b_proj,
        use_sparse=self.is_sparse,
        indexer=self.indexer,
        ## Adding missing arguments
        rotary_emb=self.rotary_emb,
        q_proj=self.q_proj,
        o_proj=self.o_proj
    )

    self.prefix = prefix