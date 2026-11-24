from contextlib import suppress

import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
    pos_embed_interpolate_native,
)
from vllm.utils.torch_utils import _encode_layer_name

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


_orig_qwen3_attn_init = Qwen3Attention.__init__


def _qwen3_attn_init_with_qkv_rope_cache(self, *args, **kwargs):
    _orig_qwen3_attn_init(self, *args, **kwargs)
    self._use_qkv_rope_cache = False
    self._encoded_layer_name = ""
    head_dim = getattr(self, "head_dim", 0)
    q_size = getattr(self, "q_size", 0)
    kv_size = getattr(self, "kv_size", 0)
    rotary_emb = getattr(self, "rotary_emb", None)
    q_norm = getattr(self, "q_norm", None)
    k_norm = getattr(self, "k_norm", None)
    attn = getattr(self, "attn", None)
    attn_type = getattr(attn, "attn_type", None) if attn is not None else None
    if (
        attn is not None
        and hasattr(attn, "layer_name")
        and "ENCODER_ONLY" not in str(attn_type)
        and 0 < head_dim <= 256
        and (head_dim & (head_dim - 1)) == 0
        and 0 < q_size <= 8192
        and (q_size & (q_size - 1)) == 0
        and 0 < kv_size <= 2048
        and (kv_size & (kv_size - 1)) == 0
        and q_size % head_dim == 0
        and kv_size % head_dim == 0
        and rotary_emb is not None
        and getattr(rotary_emb, "rotary_dim", None) == head_dim
        and not getattr(rotary_emb, "is_neox_style", True)
        and hasattr(rotary_emb, "cos_sin_cache")
        and q_norm is not None
        and k_norm is not None
        and getattr(q_norm, "bias", None) is None
        and getattr(k_norm, "bias", None) is None
    ):
        with suppress(Exception):
            import vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope  # noqa: F401
        if hasattr(torch.ops.vllm, "qkv_rmsnorm_rope_cache"):
            self._use_qkv_rope_cache = True
            self._encoded_layer_name = _encode_layer_name(attn.layer_name)


def forward_with_split_qkv_rmsnorm_mrope(self, positions: torch.Tensor, hidden_states: torch.Tensor):
    qkv, _ = self.qkv_proj(hidden_states)
    if getattr(self, "_use_qkv_rope_cache", False):
        q, k, v = torch.ops.vllm.qkv_rmsnorm_rope_cache(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            self._encoded_layer_name,
            self.q_size,
            self.kv_size,
            self.head_dim,
            self.q_norm.variance_epsilon,
        )
    elif isinstance(self.rotary_emb, AscendMRotaryEmbedding):
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


Qwen3Attention.__init__ = _qwen3_attn_init_with_qkv_rope_cache
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
