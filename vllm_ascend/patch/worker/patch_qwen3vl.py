import torch
import torch_npu
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionAttention
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
    pos_embed_interpolate_native,
)

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding
from vllm_ascend.utils import enable_sp


_QWEN_VISION_ATTENTION_ORIGINAL_FORWARD = Qwen2_5_VisionAttention.forward
_QWEN3_VISION_PREPARE_METADATA_ORIGINAL = Qwen3_VisionTransformer.prepare_encoder_metadata


def _qwen3_vision_prepare_metadata_with_expanded_rope(self, *args, **kwargs):
    metadata = _QWEN3_VISION_PREPARE_METADATA_ORIGINAL(self, *args, **kwargs)
    cos = metadata["rotary_pos_emb_cos"]
    sin = metadata["rotary_pos_emb_sin"]
    if cos is not None and sin is not None:
        metadata["rotary_pos_emb_cos"] = torch.cat((cos, cos), dim=-1)
        metadata["rotary_pos_emb_sin"] = torch.cat((sin, sin), dim=-1)
    return metadata


def _qwen_vision_attention_forward_with_preexpanded_rope(
    self,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb_cos: torch.Tensor,
    rotary_pos_emb_sin: torch.Tensor,
    max_seqlen: torch.Tensor,
    sequence_lengths: torch.Tensor | None,
) -> torch.Tensor:
    head_dim = self.hidden_size_per_attention_head
    use_preexpanded_rope = (
        rotary_pos_emb_cos is not None
        and rotary_pos_emb_sin is not None
        and rotary_pos_emb_cos.shape[-1] == head_dim
        and rotary_pos_emb_sin.shape[-1] == head_dim
    )
    if not use_preexpanded_rope:
        return _QWEN_VISION_ATTENTION_ORIGINAL_FORWARD(
            self,
            x,
            cu_seqlens,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
            max_seqlen,
            sequence_lengths,
        )

    qkv, _ = self.qkv(x)
    seq_len, batch_size, _ = qkv.shape
    num_heads = self.num_attention_heads_per_partition
    qkv = qkv.view(seq_len, batch_size, 3, num_heads, head_dim).permute(1, 0, 2, 3, 4)
    qk, v = qkv[:, :, :2], qkv[:, :, 2]
    qk = qk.permute(2, 0, 1, 3, 4).reshape(2 * batch_size, seq_len, num_heads, head_dim)
    qk = qk.contiguous()
    cos = rotary_pos_emb_cos.reshape(1, seq_len, 1, head_dim)
    sin = rotary_pos_emb_sin.reshape(1, seq_len, 1, head_dim)
    qk = torch_npu.npu_rotary_mul(
        qk,
        cos,
        sin,
        rotary_mode="half",
    )
    q, k = qk.view(2, batch_size, seq_len, num_heads, head_dim).unbind(dim=0)
    context_layer = self.attn(
        query=q,
        key=k,
        value=v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        sequence_lengths=sequence_lengths,
    )
    context_layer = context_layer.permute(1, 0, 2, 3).reshape(seq_len, batch_size, -1).contiguous()
    output, _ = self.proj(context_layer)
    return output


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
if get_current_hardware_profile().supports(HardwareCapability.VIT_PREEXPANDED_ROPE):
    Qwen3_VisionTransformer.prepare_encoder_metadata = (
        _qwen3_vision_prepare_metadata_with_expanded_rope
    )
    Qwen2_5_VisionAttention.forward = _qwen_vision_attention_forward_with_preexpanded_rope
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
