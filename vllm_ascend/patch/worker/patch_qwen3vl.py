import torch
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
    pos_embed_interpolate_native,
    Qwen3VLMultiModalProcessor,
    _cached_tensor,
)
from vllm.model_executor.models.utils import _merge_multimodal_embeddings

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding


def tensor_parallel_wrap(func):
    def wrap(*args, **kwargs):
        deepstack_input_embeds = func(*args, **kwargs)
        if deepstack_input_embeds is None:
            return deepstack_input_embeds
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

def _create_final_video_embeddings(
    self,
    video_embeddings: torch.Tensor,
    num_tokens_per_frame: list[int],
    timestamps: list[float],
    video_grid_thw: list[int],
    retention_mask: torch.Tensor,
) -> torch.Tensor:
    """Create final embeddings that combine video embeddings with
    text embeddings of indicator tokens.

    These final embeddings contain:
    - Actual video embeddings in positions corresponding to video content
    - Text embeddings for indicator tokens (<img>, </img>, and
      frame separation text) in their respective positions

    These embeddings will replace the placeholder embeddings to create
    input_embeds for the LLM.
    """

    # Generate video replacement token IDs using get_video_repl
    # This tokenizes each frame separator independently, then uses pre-tokenized
    # special tokens to ensure consistent tokenization regardless of
    # num_tokens_per_frame values.
    video_repl = Qwen3VLMultiModalProcessor.get_video_repl(
        tokens_per_frame=num_tokens_per_frame,
        tokenizer=self._tokenizer,
        timestamps=timestamps,
        vision_start_token_id=self.config.vision_start_token_id,
        vision_end_token_id=self.config.vision_end_token_id,
        video_token_id=self.config.video_token_id,
        select_token_id=self.is_multimodal_pruning_enabled,
    )

    repl_token_ids = torch.tensor(video_repl.full, device=video_embeddings.device)
    embed_token_id = _cached_tensor(
        self.config.video_token_id, repl_token_ids.device
    )
    is_video_embed = torch.isin(repl_token_ids, embed_token_id)

    # Get text embeddings for indicator tokens (has only `visual_dim``).
    text_embeddings = self.get_language_model().embed_input_ids(repl_token_ids)

    if self.use_deepstack:
        (
            deepstack_input_embeds,
            multimodal_embeddings,
        ) = self._compute_deepstack_embeds(
            inputs_embeds=text_embeddings,
            multimodal_embeddings=[video_embeddings],
            is_multimodal=is_video_embed,
        )
    else:
        deepstack_input_embeds = None
        multimodal_embeddings = [video_embeddings]

    merged_embeddings = _merge_multimodal_embeddings(
        inputs_embeds=text_embeddings,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_video_embed,
    )

    to_concat = [merged_embeddings]
    if deepstack_input_embeds is not None:
        to_concat.append(
            deepstack_input_embeds.permute(1, 0, 2).reshape(
                deepstack_input_embeds.shape[1], -1
            )
        )

    expanded_positions = None
    if self.is_multimodal_pruning_enabled:
        is_vision_start = repl_token_ids.eq(self.config.vision_start_token_id)
        expanded_positions = self._get_expanded_positions(
            device=merged_embeddings.device,
            seq_len=merged_embeddings.shape[0],
            video_grid_thw=video_grid_thw,
            num_tokens_per_frame=num_tokens_per_frame,
            timestamps=timestamps,
            is_video_embed=is_video_embed,
            is_vision_start=is_vision_start,
            retention_mask=retention_mask,
        )
        to_concat.append(expanded_positions)

    final_video_embeddings = torch.cat(to_concat, dim=-1)

    return final_video_embeddings

Qwen3VLForConditionalGeneration._create_final_video_embeddings = _create_final_video_embeddings

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
