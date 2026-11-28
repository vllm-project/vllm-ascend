from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm_ascend.attention.kimi_c8_quant import merge_and_quantize_query
from vllm_ascend.attention.kimi_c8_static import write_static_fp8_cache


@dataclass
class KimiC8Projection:
    q: torch.Tensor | None
    q_descale: torch.Tensor | None
    q_c: torch.Tensor
    kv_no_split: torch.Tensor
    written_kv: tuple[torch.Tensor, torch.Tensor]


def project_and_write_kimi_c8(
    impl,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    k_descale: torch.Tensor | float,
    *,
    num_decode_tokens: int,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    cache_writer: Callable | None = None,
) -> KimiC8Projection:
    """Reuse native projections for TP-local C8 Q and PA_NZ cache writes.

    No collectives or connector lifecycle are hidden here. The caller provides
    positions covering all rows (decode first), and can reuse q_c/kv_no_split
    plus full-precision (k_pe, k_nope) for native prefill preprocessing.
    """
    tokens = hidden_states.shape[0]
    if hidden_states.ndim != 2 or not 0 <= num_decode_tokens <= tokens or slots.numel() != tokens:
        raise ValueError("C8 projections require one slot per token and a valid decode prefix")
    if (impl.kv_lora_rank, impl.qk_rope_head_dim, impl.num_kv_heads) != (512, 64, 1):
        raise ValueError("Kimi C8 projections require latent/aux/KV-head geometry 512/64/1")
    if kv_cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("Kimi C8 cache must store E4M3 bytes")
    for name in ("fused_qkv_a_proj", "q_proj"):
        projection = getattr(impl, name, None)
        if projection is not None and getattr(projection, "weight", False) is None:
            raise RuntimeError(f"Kimi C8 must retain source {name} weights for native projections")
    if impl.use_mla_rope:
        if cos is None or sin is None or cos.shape[0] < tokens or sin.shape[0] < tokens:
            raise ValueError("C8 RoPE projections require cos/sin covering every token")
        cos, sin = cos[:tokens], sin[:tokens]

    if impl.fused_qkv_a_proj is not None:
        qkv = impl.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_no_split = qkv.split([impl.q_lora_rank, 576], dim=-1)
        q_c = impl.q_a_layernorm(q_c)
    else:
        q_c = hidden_states
        kv_no_split = impl.kv_a_proj_with_mqa(hidden_states)[0]
    kv_latent, k_pe = kv_no_split.view(tokens, 1, 576).split([512, 64], dim=-1)
    k_nope = impl.kv_a_layernorm(kv_latent.contiguous()).reshape(tokens, 1, 512)
    if impl.use_mla_rope:
        k_pe = impl.rope_single(k_pe, cos, sin)

    q = q_descale = None
    if num_decode_tokens:
        q_nope, q_pe = impl._q_proj_and_k_up_proj(q_c[:num_decode_tokens])
        if impl.use_mla_rope:
            q_pe = impl.rope_single(q_pe, cos[:num_decode_tokens], sin[:num_decode_tokens])
        q, q_descale = merge_and_quantize_query(q_nope, q_pe)
    merged_kv = torch.cat((k_nope, k_pe), dim=-1).reshape(tokens, 576)
    writer = write_static_fp8_cache if cache_writer is None else cache_writer
    writer(merged_kv, kv_cache.view(torch.uint8), slots, k_descale)
    return KimiC8Projection(q, q_descale, q_c, kv_no_split, (k_pe, k_nope))


def _supports_kimi_c8_prolog_runtime(impl) -> bool:
    geometry = tuple(
        getattr(impl, name, None)
        for name in ("num_heads", "num_kv_heads", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim")
    )
    transfer = getattr(getattr(impl, "vllm_config", None), "kv_transfer_config", None)
    if (
        getattr(impl, "kimi_c8_static_fp8", False) is not True
        or geometry[0] is None
        or not 1 <= geometry[0] <= 128
        or geometry[1:] != (1, 1536, 512, 128, 64)
        or getattr(impl, "use_mla_rope", True)
        or getattr(impl, "pcp_enabled", False)
        or getattr(impl, "is_pcp_decode_sharded", False)
        or getattr(transfer, "is_kv_producer", False)
        or getattr(getattr(impl, "fused_qkv_a_proj", None), "input_size", None) != 7168
    ):
        return False
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod

    for projection in (impl.fused_qkv_a_proj, impl.q_proj):
        scheme = getattr(getattr(projection, "quant_method", None), "quant_method", None)
        if type(scheme) is not AscendW8A8MXFP8DynamicLinearMethod:
            return False
    norm_eps = getattr(impl.kv_a_layernorm, "variance_epsilon", None)
    return norm_eps is not None and norm_eps == getattr(impl.q_a_layernorm, "variance_epsilon", None)


def prepare_kimi_c8_prolog_runtime(impl) -> bool:
    """Prepare all persistent fused-Prolog resources before KV memory sizing.

    Call after each weight load, after invalidating the old private projection
    cache. TP-local decode supports Graph; producer prefill retains native projections.
    """
    if not _supports_kimi_c8_prolog_runtime(impl):
        return False
    from vllm_ascend.attention.kimi_c8_prolog import (
        copy_prepared_weight,
        load_quant_mla_prolog,
        prepare_quant_mla_prolog_weights,
    )

    op = getattr(impl, "_kimi_c8_prolog_op", None)
    precheck = getattr(impl, "_kimi_c8_prolog_precheck", None)
    if op is None:
        op, precheck = load_quant_mla_prolog()
    weights = impl._prepare_prolog_v3_weights()
    prepared = prepare_quant_mla_prolog_weights(weights)
    prepared["quant_scale_kv"] = impl.fak_descale_float.float().reciprocal().reshape(1)
    norms = (
        impl.q_a_layernorm.weight.detach().float().contiguous(),
        impl.kv_a_layernorm.weight.detach().float().contiguous(),
    )
    # The public precheck reads the static scale. Run only at load/reload,
    # before KV memory sizing and outside every capture/decode step.
    device = norms[0].device
    precheck(
        **prepared,
        x=torch.empty((1, 7168), dtype=torch.float8_e4m3fn, device=device),
        descale_x=torch.empty((1, 224), dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu),
        norm_weight_qa=norms[0],
        norm_weight_kva=norms[1],
        kv_cache=torch.empty((1, 128, 1, 576), dtype=torch.float8_e4m3fn, device=device),
        cache_index=torch.full((1,), -1, dtype=torch.int64, device=device),
        norm_eps=impl.kv_a_layernorm.variance_epsilon,
    )
    # Captured Graphs retain these addresses across a same-geometry weight reload.
    previous = getattr(impl, "_kimi_c8_prolog_weights", None)
    if previous is not None:
        for name, value in prepared.items():
            current = previous.get(name)
            if current is None or (current.shape, current.dtype, current.device) != (
                value.shape,
                value.dtype,
                value.device,
            ):
                raise RuntimeError("C8 Prolog weight geometry/device changed; reload and recapture the Graph")
            if current.data_ptr() != value.data_ptr():
                copy_prepared_weight(current, value)
            prepared[name] = current
    previous_norms = getattr(impl, "_kimi_c8_prolog_norm_weights", None)
    if previous_norms is not None:
        for current, value in zip(previous_norms, norms):
            current.copy_(value)
        norms = previous_norms
    # Prepared entries retain the required NZ buffers; discard the redundant
    # native-V3 cache owner without rebuilding projections during decode.
    impl._prolog_v3_weights = None
    impl._kimi_c8_prolog_weights = prepared
    impl._kimi_c8_prolog_norm_weights = norms
    impl._kimi_c8_prolog_op, impl._kimi_c8_prolog_precheck = op, precheck
    if getattr(impl, "_kimi_c8_prolog_cache", None) is None:
        capacity = impl.vllm_config.scheduler_config.max_num_seqs * (impl.vllm_config.num_speculative_tokens + 1)
        blocks = (capacity + 127) // 128
        impl._kimi_c8_prolog_cache = torch.empty(blocks * 128 * 576, dtype=torch.uint8, device=device).view(
            blocks, 1, 18, 128, 32
        )
        impl._kimi_c8_prolog_slots = torch.arange(blocks * 128, dtype=torch.int64, device=device)
    return True


def project_and_write_kimi_c8_prolog(
    impl,
    hidden_states: torch.Tensor,
    kv_cache: torch.Tensor,
    physical_slots: torch.Tensor,
    *,
    num_decode_tokens: int,
    is_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the fused decode Prolog using model-load persistent resources.

    None selects generic projections for unsupported geometry or execution.
    An eligible unprepared runtime is an error: decode must not allocate
    projection/norm buffers after KV memory has already been budgeted.
    """
    if (
        hidden_states.ndim != 2
        or hidden_states.shape[1] != 7168
        or not 0 < num_decode_tokens == hidden_states.shape[0]
        or not _supports_kimi_c8_prolog_runtime(impl)
    ):
        return None
    import torch_npu

    from vllm_ascend.attention.kimi_c8_static import pa_nz_prolog_view
    from vllm_ascend.device.kimi_c8_cache_write import copy_prolog_kv_to_cache

    prepared = getattr(impl, "_kimi_c8_prolog_weights", None)
    norms = getattr(impl, "_kimi_c8_prolog_norm_weights", None)
    op = getattr(impl, "_kimi_c8_prolog_op", None)
    if prepared is None or norms is None or op is None:
        raise RuntimeError("C8 Prolog runtime must be prepared during model weight loading")
    workspace = impl._kimi_c8_prolog_cache
    logical_cache = pa_nz_prolog_view(workspace)
    if physical_slots.dtype not in (torch.int32, torch.int64) or physical_slots.numel() != hidden_states.shape[0]:
        raise ValueError("C8 Prolog requires one integer physical cache slot per token")
    slots = physical_slots.reshape(-1).to(torch.int64).contiguous()
    if not is_graph and bool((slots >= kv_cache.shape[0] * kv_cache.shape[3]).any()):
        raise ValueError("C8 Prolog physical cache slot exceeds the cache capacity")
    # The installed operator masks negative/past-capacity slots on device.
    token_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(hidden_states.contiguous(), dst_type=torch.float8_e4m3fn)
    q, descale, _, _ = op(
        **prepared,
        x=token_x,
        descale_x=dynamic_scale.view(torch.uint8).reshape(hidden_states.shape[0], 224).view(torch.float8_e8m0fnu),
        norm_weight_qa=norms[0],
        norm_weight_kva=norms[1],
        kv_cache=logical_cache,
        cache_index=impl._kimi_c8_prolog_slots[: hidden_states.shape[0]],
        norm_eps=impl.kv_a_layernorm.variance_epsilon,
    )
    copy_prolog_kv_to_cache(workspace, kv_cache, slots)
    return q, descale.squeeze(-1)
