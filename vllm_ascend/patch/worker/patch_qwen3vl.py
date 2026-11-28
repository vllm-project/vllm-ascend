import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionBlock,
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
    pos_embed_interpolate_native,
)
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding
from vllm_ascend.utils import enable_sp


def tensor_parallel_wrap(func):
    def wrap(*args, **kwargs):
        deepstack_input_embeds = func(*args, **kwargs)
        if deepstack_input_embeds is None or not enable_sp():
            return deepstack_input_embeds
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        deepstack_input_embeds.tensors = {
            k: v.chunk(tp_size)[tp_rank] for k, v in deepstack_input_embeds.tensors.items()
        }
        return deepstack_input_embeds

    return wrap


def forward_with_split_qkv_rmsnorm_mrope(self, positions: torch.Tensor, hidden_states: torch.Tensor):
    qkv, _ = self.qkv_proj(hidden_states)
    if isinstance(self.rotary_emb, AscendMRotaryEmbedding):
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        if cos_sin.device != qkv.device:
            cos_sin = cos_sin.to(qkv.device)
        if cos_sin.dtype != qkv.dtype:
            cos_sin = cos_sin.to(qkv.dtype)
        q, k, v, _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            cos_sin=cos_sin,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            eps=self.q_norm.variance_epsilon,
            mrope_section=self.rotary_emb.mrope_section,
            is_interleaved=self.rotary_emb.mrope_interleaved,
            rope_dim=self.rotary_emb.rotary_dim,
        )
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


Qwen3Attention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3MoeAttention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = tensor_parallel_wrap(
    Qwen3VLForConditionalGeneration._get_deepstack_input_embeds
)


def _fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
    outputs = []
    for t, h, w in grid_thw:
        outputs.append(
            pos_embed_interpolate_native(
                self.pos_embed.weight,
                t,
                h,
                w,
                self.num_grid_per_side,
                self.spatial_merge_size,
                self.dtype,
            )
        )
    return torch.cat(outputs, dim=0)


Qwen3_VisionTransformer.fast_pos_embed_interpolate = _fast_pos_embed_interpolate


def patch_qwen3_vl_moe_pp_layer_range():
    try:
        from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM
    except Exception:
        return

    if not hasattr(Qwen3MoeLLMForCausalLM, "start_layer"):
        Qwen3MoeLLMForCausalLM.start_layer = property(lambda self: self.model.start_layer)

    if not hasattr(Qwen3MoeLLMForCausalLM, "end_layer"):
        Qwen3MoeLLMForCausalLM.end_layer = property(lambda self: self.model.end_layer)


patch_qwen3_vl_moe_pp_layer_range()


if HAS_TRITON:
    from vllm_ascend.ops.triton.qwen_prefill import pack_vision_qk, vision_add_norm

    _original_vision_block = Qwen3_VisionBlock.forward

    def _vision_block_forward(
        self, x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, max_seqlen, sequence_lengths
    ):
        cos, sin = rotary_pos_emb_cos, rotary_pos_emb_sin
        attention = self.attn
        dim = attention.hidden_size_per_attention_head
        if not (
            x.device.type == "npu"
            and x.ndim == 3
            and x.shape[1:] == (1, 1152)
            and x.size(0) > 0
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and dim == 72
            and cos is not None
            and sin is not None
            and cos.shape == sin.shape == (x.size(0), dim // 2)
            and cos.is_contiguous()
            and sin.is_contiguous()
            and cos.device == sin.device == x.device
            and cos.dtype == sin.dtype == x.dtype
            and hasattr(torch.ops.npu, "npu_add_layer_norm")
            and isinstance(self.norm2, torch.nn.LayerNorm)
            and self.norm2.normalized_shape == (1152,)
            and self.norm2.weight is not None
            and self.norm2.bias is not None
            and self.norm2.weight.is_contiguous()
            and self.norm2.bias.is_contiguous()
        ):
            return _original_vision_block(self, x, cu_seqlens, cos, sin, max_seqlen, sequence_lengths)

        tokens = x.size(0)
        heads = attention.num_attention_heads_per_partition
        packed, _ = attention.qkv(self.norm1(x))
        qkv = packed.reshape(tokens, 3, heads, dim)
        rotated = attention.apply_rotary_emb(pack_vision_qk(qkv), cos, sin)
        q, k = rotated.reshape(2, 1, tokens, heads, dim).unbind(0)
        context = attention.attn(
            query=q,
            key=k,
            value=qkv[:, 2].unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        projected, _ = attention.proj(context.reshape(tokens, 1, heads * dim).contiguous())
        residual, normalized = vision_add_norm(x, projected.contiguous(), self.norm2)
        return residual + self.mlp(normalized)

    Qwen3_VisionBlock.forward = _vision_block_forward
