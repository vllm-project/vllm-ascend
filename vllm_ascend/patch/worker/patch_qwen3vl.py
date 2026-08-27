import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionAttention
from vllm.model_executor.models.qwen3 import Qwen3Attention
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionMLP,
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
    pos_embed_interpolate_native,
)

from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding
from vllm_ascend.utils import enable_sp, enable_vision_sp


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


# ---------------------------------------------------------------------------
# VIT Sequence Parallelism (TP+SP hybrid) — Module E: registration
# ---------------------------------------------------------------------------
# Patches are applied at import time. Each wrapper checks enable_vision_sp()
# at runtime: when SP is off, the original forward is called (zero overhead);
# when SP is on, the SP forward (which calls VisionSPStrategy) is called.

_orig_vision_attention_forward = Qwen2_5_VisionAttention.forward
_orig_vision_mlp_forward = Qwen3_VisionMLP.forward
_orig_vision_transformer_forward = Qwen3_VisionTransformer.forward


def _sp_vision_transformer_forward(self, x, grid_thw, *, encoder_metadata=None):
    """SP-mode VisionTransformer.forward.

    Compared to the original forward (qwen3_vl.py:800), this version:
    - Pads total_seq to be divisible by tp_size (padding is trimmed at exit)
    - Splits the sequence across TP ranks before the blocks loop (SP entry)
    - AllGathers before each deepstack merger and before the final merger (SP exit)
    - Trims padding from the final output
    """
    hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
    hidden_states = self.patch_embed(hidden_states)

    if encoder_metadata is None:
        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
        else:
            grid_thw_list = grid_thw.tolist()
        encoder_metadata = self.prepare_encoder_metadata(grid_thw_list)

    pos_embeds = encoder_metadata["pos_embeds"]
    hidden_states = hidden_states + pos_embeds
    hidden_states = hidden_states.unsqueeze(1)  # [total_seq, 1, hidden]

    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    total_seq = hidden_states.shape[0]

    # --- SP entry padding ---
    # total_seq may not be divisible by tp_size (e.g. tp_size >= 8).
    # Pad with zeros so each rank gets an equal-sized shard.
    # The padded tokens' outputs are discarded at SP exit.
    padded_seq = ((total_seq + tp_size - 1) // tp_size) * tp_size
    if padded_seq != total_seq:
        # Shallow-copy metadata dict so we don't mutate the caller's copy
        # (the same metadata may be reused across forward calls, e.g. cudagraph).
        encoder_metadata = dict(encoder_metadata)
        pad_len = padded_seq - total_seq
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, 0, 0, pad_len))
        # Pad rotary cos/sin to match the AllGathered full_seq inside blocks.
        # Pad with the last valid entry so RoPE produces finite values for
        # dummy tokens (their output is discarded at exit).
        cos = encoder_metadata.get("rotary_pos_emb_cos")
        sin = encoder_metadata.get("rotary_pos_emb_sin")
        if cos is not None and cos.shape[0] == total_seq:
            cos_pad = cos[-1:].expand(pad_len, -1)
            encoder_metadata["rotary_pos_emb_cos"] = torch.cat([cos, cos_pad], dim=0)
        if sin is not None and sin.shape[0] == total_seq:
            sin_pad = sin[-1:].expand(pad_len, -1)
            encoder_metadata["rotary_pos_emb_sin"] = torch.cat([sin, sin_pad], dim=0)
        # Pad cu_seqlens by appending padded_seq as a dummy segment endpoint
        # so FIA sees a token-count matching the padded input. The dummy
        # segment isolates padded tokens' attention from real tokens.
        cu_seqlens = encoder_metadata.get("cu_seqlens")
        if cu_seqlens is not None and cu_seqlens[-1].item() == total_seq:
            dummy = torch.tensor([padded_seq], dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            encoder_metadata["cu_seqlens"] = torch.cat([cu_seqlens, dummy])

    # SP entry: split sequence evenly across TP ranks
    local_seq = padded_seq // tp_size
    hidden_states = hidden_states[tp_rank * local_seq : (tp_rank + 1) * local_seq]

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        # Block.forward is NOT patched — it calls self.attn() and self.mlp()
        # which ARE patched. Residual Add and LayerNorm operate on local_seq
        # (both are element-wise, so local_seq is correct).
        hidden_states = blk(
            hidden_states,
            cu_seqlens=encoder_metadata["cu_seqlens"],
            rotary_pos_emb_cos=encoder_metadata["rotary_pos_emb_cos"],
            rotary_pos_emb_sin=encoder_metadata["rotary_pos_emb_sin"],
            max_seqlen=encoder_metadata["max_seqlen"],
            sequence_lengths=encoder_metadata.get("sequence_lengths"),
        )
        if layer_num in self.deepstack_visual_indexes:
            # Deepstack mergers require full_seq (they reshape by
            # spatial_merge_size). AllGather, TRIM padding to total_seq,
            # then run merger. Trimming is critical: without it, the merger
            # produces padded_seq/4 tokens while the final merger produces
            # total_seq/4, causing torch.cat dim=0 mismatch.
            full_hs = tensor_model_parallel_all_gather(hidden_states, dim=0)
            if padded_seq != total_seq:
                full_hs = full_hs[:total_seq]
            deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
            deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](full_hs)
            deepstack_feature_lists.append(deepstack_feature)

    # SP exit: AllGather to restore full (padded) seq for the final merger
    hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
    # Trim padding back to original total_seq
    if padded_seq != total_seq:
        hidden_states = hidden_states[:total_seq]
    hidden_states = self.merger(hidden_states)
    hidden_states = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)
    return hidden_states


def _sp_vision_attention_forward(
    self,
    x,
    cu_seqlens,
    rotary_pos_emb_cos,
    rotary_pos_emb_sin,
    max_seqlen,
    sequence_lengths,
):
    """SP-mode VisionAttention.forward (Ulysses-style SP for attention).

    Original flow (TP-only, qwen2_5_vl.py:398-456):
        x [full_seq] -> qkv -> RoPE -> FA -> o_proj(+AllReduce) -> [full_seq]

    SP flow (TP+SP):
        x [local_seq] -> AllGather+qkv -> RoPE -> FA -> AllToAll+o_proj(+AllReduce) -> [local_seq]
    """
    import einops

    from vllm_ascend.ops.vision_sp_strategy import get_vision_sp_strategy

    strategy = get_vision_sp_strategy()

    # x: [local_seq, 1, hidden] (SP sharded on dim=0)
    # Step 1: AllGather & Matmul (qkv)
    #   strategy: all_gather(x, dim=0) -> [full_seq, 1, hidden]
    #             then quant_method.apply(qkv_layer, full_x, bias)
    #             -> [full_seq, 1, 3*local_h*head_dim]
    x = strategy.allgather_and_matmul(x, self.qkv)

    seq_len, batch_size, _ = x.shape
    qkv = einops.rearrange(
        x,
        "s b (three head head_dim) -> b s three head head_dim",
        three=3,
        head=self.num_attention_heads_per_partition,
    )

    # Step 2: RoPE
    #   cos/sin are the FULL versions (all ranks hold a copy, not SP-sharded).
    #   x is already full_seq after Step 1's AllGather, so dimensions match.
    if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
        qk, v = qkv[:, :, :2], qkv[:, :, 2]
        qk_reshaped = einops.rearrange(qk, "b s two head head_dim -> (two b) s head head_dim", two=2)
        qk_reshaped = qk_reshaped.contiguous()
        qk_rotated = self.apply_rotary_emb(qk_reshaped, rotary_pos_emb_cos, rotary_pos_emb_sin)
        qk_rotated = qk_rotated.view(
            2,
            batch_size,
            seq_len,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        q, k = qk_rotated.unbind(dim=0)
    else:
        q, k, v = qkv.unbind(dim=2)

    # Step 3: FA (Flash Attention / FIA)
    #   Each rank computes attention for its local_h heads over full_seq.
    #   cu_seqlens and max_seqlen are the full versions — SP does not change
    #   sequence boundaries, only which tokens each rank stores between layers.
    context_layer = self.attn(
        query=q,
        key=k,
        value=v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        sequence_lengths=sequence_lengths,
    )
    context_layer = einops.rearrange(context_layer, "b s h d -> s b (h d)", b=batch_size).contiguous()
    # [full_seq, 1, local_h*head_dim]

    # Step 4: AllToAll & Matmul (o_proj)
    #   strategy: all_to_all(context, scatter=seq, gather=head)
    #             -> [local_seq, 1, all_h*head_dim]
    #             then quant_method.apply(proj, gathered, bias)
    #             -> [local_seq, 1, hidden] (full, no AllReduce needed)
    #   o_proj is ReplicatedLinear when SP is enabled (full weight on every rank)
    output = strategy.alltoall_matmul(context_layer, self.proj)
    return output


def _sp_vision_mlp_forward(self, x):
    """SP-mode VisionMLP.forward (AllGather/ReduceScatter SP for MLP).

    Original flow (TP-only, qwen3_vl.py:408-410):
        x [full_seq] -> linear_fc1 -> act_fn -> linear_fc2(+AllReduce) -> [full_seq]

    SP flow (TP+SP):
        x [local_seq] -> AllGather+linear_fc1 -> act_fn -> linear_fc2+ReduceScatter -> [local_seq]
    """
    from vllm_ascend.ops.vision_sp_strategy import (
        get_vision_sp_strategy,
        vision_matmul_and_reducescatter,
    )

    strategy = get_vision_sp_strategy()
    # x: [local_seq, 1, hidden]
    # Step 1: AllGather & FFN UP
    up_out = strategy.allgather_and_matmul(x, self.linear_fc1)
    # [full_seq, 1, local_ffn_dim]
    # Step 2: Gelu (element-wise, same as TP-only)
    act_out = self.act_fn(up_out)
    # [full_seq, 1, local_ffn_dim]
    # Step 3: FFN Down & ReduceScatter
    #   Standalone function (no fused version planned for ReduceScatter)
    down_out = vision_matmul_and_reducescatter(act_out, self.linear_fc2)
    # [local_seq, 1, hidden]
    return down_out


# --- Runtime dispatch wrappers ---


def _vision_attention_forward_wrapper(self, *args, **kwargs):
    if enable_vision_sp():
        return _sp_vision_attention_forward(self, *args, **kwargs)
    return _orig_vision_attention_forward(self, *args, **kwargs)


def _vision_mlp_forward_wrapper(self, x):
    if enable_vision_sp():
        return _sp_vision_mlp_forward(self, x)
    return _orig_vision_mlp_forward(self, x)


def _vision_transformer_forward_wrapper(self, *args, **kwargs):
    if enable_vision_sp():
        return _sp_vision_transformer_forward(self, *args, **kwargs)
    return _orig_vision_transformer_forward(self, *args, **kwargs)


# --- Apply patches ---
Qwen2_5_VisionAttention.forward = _vision_attention_forward_wrapper
Qwen3_VisionMLP.forward = _vision_mlp_forward_wrapper
Qwen3_VisionTransformer.forward = _vision_transformer_forward_wrapper


# ---------------------------------------------------------------------------
# Path A: Replace o_proj with ReplicatedLinear when SP is enabled
# ---------------------------------------------------------------------------
# When vision SP is enabled, o_proj must use a full (replicated) weight
# [H, N*head_dim] on every rank so that after AllToAll each rank can compute
# the complete output for its local_seq without AllReduce. ReplicatedLinear's
# weight_loader loads the checkpoint weight without TP sharding, so every rank
# naturally receives the full weight at load time -- no runtime AllGather
# needed.
#
# When SP is NOT enabled, o_proj stays RowParallelLinear (original behavior).
# enable_vision_sp() is determined by config and stable throughout inference;
# it is only reset by clear_enable_sp() during startup reinit (before model
# construction), so the layer type is always consistent with the runtime flag.
# ---------------------------------------------------------------------------

_orig_vision_attn_init = Qwen2_5_VisionAttention.__init__


def _patched_vision_attn_init(self, *args, **kwargs):
    _orig_vision_attn_init(self, *args, **kwargs)
    if enable_vision_sp():
        tp_size = get_tensor_model_parallel_world_size()
        orig_proj = self.proj
        output_size = orig_proj.weight.shape[0]
        input_size = orig_proj.weight.shape[1] * tp_size
        has_bias = orig_proj.bias is not None
        if not isinstance(orig_proj.quant_method, UnquantizedLinearMethod):
            raise NotImplementedError(
                "ReplicatedLinear o_proj with quantized ViT is not yet "
                "supported. Please disable vision SP or use unquantized ViT."
            )
        self.proj = ReplicatedLinear(
            input_size=input_size,
            output_size=output_size,
            bias=has_bias,
        )


Qwen2_5_VisionAttention.__init__ = _patched_vision_attn_init
