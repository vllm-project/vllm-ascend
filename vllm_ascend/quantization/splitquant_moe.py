"""Opt-in six-kernel AscendC MoE; immutable weight preparation at load time."""
import os
import torch
from . import splitquant_moe_native as native


@torch.no_grad()
def prepare(experts):
    if torch.count_nonzero(experts.down_proj_offset).item() or torch.count_nonzero(experts.gate_up_proj_offset).item():
        raise ValueError('MoE  requires zero quantization offsets')
    experts._prepared = native.prepare(
        experts.catlass_gate_up_packed, experts.gate_up_proj_scale,
        experts.down_proj, experts.down_proj_scale,
        experts.activation_clip_max, experts.activation_clip_min,
        experts.hidden_activation_clip_max, experts.hidden_activation_clip_min,
        experts.down_group_trans, experts.hidden_activation_clip_max.dtype)
    # The  model owns the group-major container. Alias the Parameter and
    # buffer to it, releasing the redundant ND packed allocation after load.
    experts.gate_up_proj.data = experts._prepared[0]
    experts.catlass_gate_up_packed = experts.gate_up_proj.data
    if os.getenv('SPLITQUANT_ROUTER_QUANT_FUSED') == '1':
        from .splitquant_router_quant import install
        install()


def forward(experts, x, indices, weights):
    if not hasattr(experts, '_prepared'):
        experts.prepare_catlass()
    # Explicitly report fallback instead of claiming unsupported shapes use .
    if x.shape[0] > 32:
        experts.moe_fallback_calls = getattr(experts, 'moe_fallback_calls', 0) + 1
        return experts._forward_reference(x, indices, weights)
    result = native.apply(experts._prepared, x, indices.long(), weights.to(x.dtype))
    if os.getenv('SPLITQUANT_MOE_VERIFY') == '1' and not getattr(experts, 'moe_verification_checks', 0):
        reference = experts._forward_reference(x, indices, weights)
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        experts.moe_verification_checks = 1
    experts.moe_calls = getattr(experts, 'moe_calls', 0) + 1
    return result
