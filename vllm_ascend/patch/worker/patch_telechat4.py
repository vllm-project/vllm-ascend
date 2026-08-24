import torch
from vllm.model_executor.models.telechat4 import (
    Telechat4MHC
)


def _patch_forward(
    self, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute mHC pre-mixing: aggregate n-stream -> 1-stream.

    Args:
        hidden_states: [nt, n*C] - n-stream hidden states (flattened)
    Returns:
        aggregated: [nt, C] - single-stream input for the sub-layer
        h_res: [nt, n, n] - residual mixing matrix (comb_mix)
        h_post: [nt, n, 1] - stream expansion weights (post_mix)
    """
    n = self.n
    C = self.hidden_size
    #self._ensure_cache()
    s, _ = hidden_states.shape
    x = hidden_states.reshape(s, 1, n, C).transpose(1, 0)
    h_in, h_post, h_res, _, _, _, _, _, _ = torch.ops._C_ascend.mhc_pre_clamp_sinkhorn(
        x=x, phi=self.hc_fn, alpha=self.hc_scale, bias=self.hc_base,
        hcMult=self.n, numIters=self.sinkhorn_iterations, hcEps=self.norm_eps,
        normEps=self.norm_eps, outFlag=False,
        clamp_min=self.h_res_clamp_min, clamp_max=self.h_res_clamp_max
    )

    h_in = h_in.reshape(s, C) # b s c -> s b c
    h_res = h_res.reshape(s, n, n).permute(0,2,1) # b s n*n -> b*s n n -> s b n n
    h_post = h_post.reshape(s, n, 1)
    return h_in, h_res, h_post

def _patch_fused_h_res_h_post_bda_inference(
    self,
    h_res: torch.Tensor,
    original_residual: torch.Tensor,
    h_post: torch.Tensor,
    layer_output_with_bias: tuple[torch.Tensor, torch.Tensor | None],
) -> torch.Tensor:
    """
    Fused residual mixing + post expansion + bias-dropout-add.

    Args:
        h_res: [nt, n, n] - comb_mix from forward()
        original_residual: [nt, n*C] - the n-stream input before aggregation
        h_post: [nt, n, 1] - post_mix from forward()
        layer_output_with_bias: (x [nt,C], bias None)
    Returns:
        output: [nt, n*C] - updated n-stream residual for next layer
    """
    x, _ = layer_output_with_bias

    n = self.n
    C = self.hidden_size

    hidden_states = torch.ops._C_ascend.mhc_post(
        x=original_residual.reshape(-1, n, C),
        hRes=h_res,
        hOut=x,
        hPost=h_post.squeeze(-1)
    )

    return hidden_states.reshape(-1, n*C)

Telechat4MHC.forward = _patch_forward
Telechat4MHC.fused_h_res_h_post_bda_inference = _patch_fused_h_res_h_post_bda_inference
