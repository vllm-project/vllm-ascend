import torch
import torch_npu
from vllm.model_executor.models.deepencoder import RelPosAttention, add_decomposed_rel_pos

from vllm_ascend.utils import vllm_version_is


class AscendRelPosAttention(RelPosAttention):
    def _init_attention(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: tuple[int, int] | None = None,
        *,
        use_triton_attention: bool = False,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        # #55629 inserted a positional parameter on main. Use named arguments
        # so input_size / rel_pos_zero_init cannot silently bind to other fields.
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size,
            **({} if vllm_version_is("0.28.0") else {"use_triton_attention": use_triton_attention}),
        )

    if vllm_version_is("0.28.0"):
        __init__ = _init_attention
    else:

        def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            use_rel_pos: bool = False,
            use_triton_attention: bool = False,
            rel_pos_zero_init: bool = True,
            input_size: tuple[int, int] | None = None,
        ) -> None:
            self._init_attention(
                dim,
                num_heads,
                qkv_bias,
                use_rel_pos,
                rel_pos_zero_init,
                input_size,
                use_triton_attention=use_triton_attention,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        rel_h, rel_w = None, None
        if self.use_rel_pos:
            rel_h, rel_w = add_decomposed_rel_pos(q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        q = q.view(B, self.num_heads, H * W, -1)
        k = k.view(B, self.num_heads, H * W, -1)
        v = v.view(B, self.num_heads, H * W, -1)

        if self.use_rel_pos:
            assert rel_h is not None and rel_w is not None
            rel_h = rel_h.view(B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3))
            rel_w = rel_w.view(B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3))
            attn_bias = (rel_h + rel_w).view(B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4))
            x = torch_npu.npu_prompt_flash_attention(
                q,
                k,
                v,
                pse_shift=attn_bias,
                input_layout="BNSD",
                scale_value=self.scale,
                num_heads=self.num_heads,
            )
        else:
            x = torch_npu.npu_prompt_flash_attention(
                q,
                k,
                v,
                input_layout="BNSD",
                scale_value=self.scale,
                num_heads=self.num_heads,
            )

        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x
