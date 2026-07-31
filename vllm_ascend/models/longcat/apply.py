"""Apply all LongCat-Flash patches. Call once before vllm serve starts.

Usage: python -c "from vllm_ascend.models.longcat.apply import apply; apply()"
"""


def apply() -> None:
    """Apply all LongCat model patches (dual-attention, MLA rotary, layernorm, EP zero-expert)."""
    from vllm_ascend.models.longcat.fix_dual_attention import patch as _da
    from vllm_ascend.models.longcat.fix_mla_rotary import patch as _mr
    from vllm_ascend.models.longcat.fix_layernorm_dtype import patch as _ln
    from vllm_ascend.ops.fused_moe.fix_ep_zero_expert import patch as _ep

    _da()
    _mr()
    _ln()
    _ep()
    print("[longcat] Patches applied: dual_attention, mla_rotary, layernorm_dtype, ep_zero_expert")
