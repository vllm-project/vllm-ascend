from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.rotary_embedding import get_rope

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding


def get_rotary_embedding(config: "PretrainedConfig", *, dtype: torch.dtype) -> "RotaryEmbedding":
    """Resolve the full-head NeoX table used by runtime and calibration.

    Qwen3 builds RoPE through the same vLLM factory. Matching its resolved
    parameters also allows the factory to reuse the model's cached embedding.
    """
    parameters = dict(getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {})
    parameters.setdefault("rope_theta", float(getattr(config, "rope_theta", 10000.0)))
    partial = parameters.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
    if (
        "mrope_section" in parameters
        or getattr(config, "dual_chunk_attention_config", None) is not None
        or not getattr(config, "rope_is_neox_style", True)
        or partial != 1.0
    ):
        raise ValueError("KVTC requires full-head, 1-D NeoX RoPE")
    head_size = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    rotary_emb = get_rope(
        head_size=int(head_size),
        max_position=int(config.max_position_embeddings),
        rope_parameters=parameters,
        is_neox_style=True,
        dtype=dtype,
    )
    if rotary_emb.rotary_dim != head_size:
        raise ValueError("KVTC requires full-head RoPE")
    return rotary_emb
