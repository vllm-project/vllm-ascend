import torch
import torch.nn.functional as F
import einops
import torch_npu

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ops.mm_encoder_attention import (
    AscendMMEncoderAttention as _Base,
    MIN_PAD_SIZE,
    MAX_PAD_SIZE,
)


class AscendMMEncoderAttention310(_Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ):
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        q, k, v = self.reshape_qkv_to_3d(query, key, value, bsz, q_len, kv_len)
        
        enable_pad = (
            envs_ascend.USE_OPTIMIZED_MODEL
            and self.head_size > MIN_PAD_SIZE
            and self.head_size < MAX_PAD_SIZE
        )

        origin_shape = q.shape[-1]
        if enable_pad:
            pad_len = MAX_PAD_SIZE - origin_shape
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        origin_dim = origin_shape
        cur_dim = q.shape[-1]
        pad16 = (16 - cur_dim % 16) % 16
        if pad16:
            q = F.pad(q, (0, pad16), mode="constant", value=0)
            k = F.pad(k, (0, pad16), mode="constant", value=0)
            v = F.pad(v, (0, pad16), mode="constant", value=0)

        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0, (bsz + 1) * q_len, step=q_len,
                dtype=torch.int32, device=query.device,
            )

        total_q_tokens = bsz * q_len
        context_flat = q.new_empty((total_q_tokens, self.num_heads, q.shape[-1]))

        st = 0
        seg_lens = torch.diff(cu_seqlens).to("cpu", dtype=torch.int64).tolist()
        for seg_len in seg_lens:
            seg_len = int(seg_len)
            ed = st + seg_len

            q_i = q[st:ed].unsqueeze(0)   # [1, S, H, D]
            k_i = k[st:ed].unsqueeze(0)
            v_i = v[st:ed].unsqueeze(0)

            qs = int(q_i.shape[1])
            kvs = int(k_i.shape[1])

            out_i = torch_npu.npu_prompt_flash_attention(
                q_i, k_i, v_i,
                input_layout="BSND",
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                scale_value=self.head_size**-0.5,
                pre_tokens=qs,
                next_tokens=kvs,
            )
            context_flat[st:ed] = out_i[0]
            st = ed

        context_flat = context_flat[..., :origin_dim]
        context_layer = einops.rearrange(
            context_flat, "(b s) h d -> b s h d", b=bsz
        ).contiguous()
        return context_layer
