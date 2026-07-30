import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
    pos_embed_interpolate_native,
)
from vllm.sequence import IntermediateTensors

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding
from vllm_ascend.worker.ubatch_utils import get_ubatch_runtime_manager


def _ubatch_token_slice(num_tokens: int) -> tuple[int, int]:
    """Return the [start, stop) token range this call should read/clear.

    Under ubatch overlap each worker thread must operate on the slice
    ``[start:stop]`` bound by ``UBatchRuntimeManager.exec`` (the same
    thread-local mechanism used by ``get_cos_and_sin_slice`` in rotary
    embedding). When ubatch is disabled or the caller is not on a worker
    thread, fall back to the whole ``[0:num_tokens]`` range so non-ubatch
    runs are unaffected.
    """
    token_slice = get_ubatch_runtime_manager().get_current_token_slice()
    if token_slice is not None:
        return token_slice.start, token_slice.stop
    return 0, num_tokens


def _patched_get_deepstack_input_embeds(self, num_tokens: int):
    if not getattr(self, "deepstack_input_embeds", None):
        return None  # If vision tower is skipped

    # Keep the upstream resize guard: if the buffer is smaller than the full
    # token count, grow (and zero) it first. This must use the *full*
    # num_tokens (not the ubatch slice) because the buffer is shared across
    # ubatches and written once by the vision encoder before any ubatch
    # worker reads from it.
    if num_tokens > self.deepstack_input_embeds[0].size(0):
        self._resize_deepstack_input_embeds(num_tokens)

    start, stop = _ubatch_token_slice(num_tokens)
    deepstack_input_embeds = IntermediateTensors(
        {
            f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][start:stop]
            for idx in range(self.deepstack_num_level)
        }
    )

    # TP chunk under flash_comm_v1 (mirrors the original tensor_parallel_wrap):
    # when SP/flash_comm is active each rank only needs its own shard.
    try:
        flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
    except (AssertionError, AttributeError, KeyError):
        flash_comm_v1_enabled = False
    if flash_comm_v1_enabled:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        deepstack_input_embeds.tensors = {
            k: v.chunk(tp_size)[tp_rank] for k, v in deepstack_input_embeds.tensors.items()
        }
    return deepstack_input_embeds


def _patched_clear_deepstack_input_embeds(self, num_tokens: int) -> None:
    if not getattr(self, "deepstack_input_embeds", None):
        return

    start, stop = _ubatch_token_slice(num_tokens)
    if stop > start:
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][start:stop].zero_()


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


def _apply_deepstack_patches(model_cls) -> None:
    """Patch DeepStack cross-layer buffer accessors to be both TP-aware and
    ubatch-thread-safe.

    This combines the original tensor_parallel_wrap (TP chunk under
    flash_comm_v1) with ubatch overlap support: each worker thread reads/clears
    the [start:stop] slice bound by UBatchRuntimeManager.exec. The resize guard
    from the upstream implementation is preserved. Sets the
    ``_ubatch_deepstack_patched`` marker so ``NPUModelRunner._ubatch_blocked_reason``
    lifts the DeepStack gate and allows ubatch overlap on these models.
    """
    model_cls._get_deepstack_input_embeds = _patched_get_deepstack_input_embeds
    model_cls._clear_deepstack_input_embeds = _patched_clear_deepstack_input_embeds
    model_cls._ubatch_deepstack_patched = True


_apply_deepstack_patches(Qwen3VLForConditionalGeneration)

# Qwen3OmniMoeThinker does not inherit from Qwen3VLForConditionalGeneration but
# has an identical DeepStack buffer implementation, so apply the same patches.
try:
    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    _apply_deepstack_patches(Qwen3OmniMoeThinkerForConditionalGeneration)
except ImportError:
    pass


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
