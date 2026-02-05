import torch
import torch_npu
from vllm.model_executor.models.deepencoder import RelPosAttention, add_decomposed_rel_pos

origin_forward = RelPosAttention.forward
def new_forward(self, x: torch.Tensor) -> torch.Tensor:
    B, H, W, _ = x.shape
    # qkv with shape (3, B, nHead, H * W, C)
    qkv = (
        self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    )
    # q, k, v with shape (B * nHead, H * W, C)
    q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

    rel_h, rel_w = None, None
    if self.use_rel_pos:
        rel_h, rel_w = add_decomposed_rel_pos(
            q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
        )

    q = q.view(B, self.num_heads, H * W, -1)
    k = k.view(B, self.num_heads, H * W, -1)
    v = v.view(B, self.num_heads, H * W, -1)

    if self.use_rel_pos:
        rel_h = rel_h.view(
            B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3)
        )
        rel_w = rel_w.view(
            B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3)
        )
        attn_bias = (rel_h + rel_w).view(
            B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4)
        )
        x = torch_npu.npu_prompt_flash_attention(
            q, k, v,
            pse_shift=attn_bias,
            input_layout="BNSD",
            scale_value=self.scale,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            num_heads=self.num_heads)

    else:
        x = torch_npu.npu_prompt_flash_attention(
            q, k, v,
            input_layout="BNSD",
            scale_value=self.scale,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            num_heads=self.num_heads)

    x = (
        x.view(B, self.num_heads, H, W, -1)
        .permute(0, 2, 3, 1, 4)
        .reshape(B, H, W, -1)
    )

    x = self.proj(x)

    return x


RelPosAttention.forward = new_forward