"""Apply all LongCat-Flash patches. Call once before vllm serve starts.

Usage: python -c "from vllm_ascend.models.longcat.apply import apply; apply()"
"""


def apply() -> None:
    """Apply longcat model compat patches (dual-attention, MLA rotary, layernorm)."""
    from vllm_ascend.models.longcat.fix_dual_attention import patch as _da
    from vllm_ascend.models.longcat.fix_mla_rotary import patch as _mr
    from vllm_ascend.models.longcat.fix_layernorm_dtype import patch as _ln

    _da()
    _mr()
    _ln()
    print("[longcat] Patches applied: dual_attention, mla_rotary, layernorm_dtype")
