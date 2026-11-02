"""Opt-in  M=1 router finish/input quantization fusion."""
import os
import torch
import torch_npu


def install():
    from vllm_ascend.models.qwen3_splitquant import SplitQuantMoeBlock
    if hasattr(SplitQuantMoeBlock, '_router_quant_original'):
        return
    original = SplitQuantMoeBlock.forward
    SplitQuantMoeBlock._router_quant_original = original

    def forward(self, x):
        enabled = all(os.getenv(name) == '1' for name in (
            'SPLITQUANT_DENSE', 'SPLITQUANT_MOE',
            'SPLITQUANT_FUSED_ROUTER', 'SPLITQUANT_ROUTER_QUANT_FUSED'))
        # The fused router path is optional.  Older or partially built native
        # libraries may not export these operators; fall back to the regular
        # MoE implementation instead of raising AttributeError at runtime.
        npu_ops = torch.ops.npu
        if not all(hasattr(npu_ops, name) for name in (
                'splitquant_router_quant', 'splitquant_moe_fused_prequant')):
            return original(self, x)
        if not (enabled and x.dtype == torch.bfloat16 and x.numel() == 2048
                and x.shape[-1] == 2048 and self.top_k == 8
                and hasattr(self.experts, '_prepared')
                and self.experts._prepared[0].shape[0] == 128):
            return original(self, x)
        shape = x.shape
        x = x.reshape(1, 2048)
        transform = self.experts.input_group_trans.to(x)
        if os.getenv('SPLITQUANT_RAW_ROTATION') == '1':
            x = torch.ops.npu.splitquant_raw_rotation(x.contiguous(), transform.contiguous(), 0)
            self.experts.raw_rotation_calls = getattr(self.experts, 'raw_rotation_calls', 0) + 1
        else:
            x = torch.einsum('mgk,gkl->mgl', x.reshape(1, 16, 128), transform).reshape_as(x)
        logits, _ = self.gate(x)
        assert logits.shape[1] == 128
        x = torch_npu.npu_format_cast(x.contiguous(), 2)
        w, ids, _ = torch_npu.npu_moe_gating_top_k_softmax(
            x=logits.float(), finished=None, k=8)
        up, sc, down, cp, cn, hp, hn, gt, lut = self.experts._prepared
        w, ids, a, scale = torch.ops.npu.splitquant_router_quant(
            w.contiguous(), ids.contiguous(), x, cp, cn, self.norm_topk_prob)
        out = torch.ops.npu.splitquant_moe_fused_prequant(
            x, up, sc, down, cp, cn, hp, hn, gt, ids, w, lut, a, scale)
        if os.getenv('SPLITQUANT_MOE_VERIFY') == '1' and not getattr(
                self.experts, 'moe_verification_checks', 0):
            reference = self.experts._forward_reference(x, ids, w)
            torch.testing.assert_close(out, reference, rtol=0, atol=0)
            self.experts.moe_verification_checks = 1
        for name in ('router_calls', 'moe_calls',
                     'router_quant_calls', 'moe_prequant_calls'):
            setattr(self.experts, name, getattr(self.experts, name, 0) + 1)
        return out.reshape(shape)

    SplitQuantMoeBlock.forward = forward
