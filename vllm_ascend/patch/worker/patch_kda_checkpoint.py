# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""Patch upstream KDA checkpoint helpers and KimiGatedDeltaNetAttention for Ascend Triton KDA."""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import types
from collections.abc import Callable
from typing import Any

import torch

from vllm_ascend.ops.kda_checkpoint import (
    CHECKPOINT_LIST_FIELDS,
    CHECKPOINT_TENSOR_FIELDS,
    KDA_PREFILL_CHECKPOINT_ALIGNMENT,
    build_kda_kv_cache_spec,
    clone_with_updates,
    export_kda_prefill_checkpoint,
    get_triton_kda_checkpoint_buffer,
    initialize_kda_checkpoints,
)
from vllm_ascend.ops.triton.kda.kda import chunk_kda_with_fused_gate as ascend_chunk_kda_with_fused_gate

_KDA_CKPT_PATCH_ATTR = "_ascend_kda_checkpoint_patched"
_KDA_CKPT_HOOK_VAR_ATTR = "_ascend_kda_checkpoint_hook_var"


# Neither pinned vLLM baseline (v0.34.0 nor vllm-main-verified) has native Triton KDA
# checkpoint_state support; these probes strictly guard against future upstream additions.
def _safe_param_names(fn: Any) -> frozenset[str] | None:
    try:
        return frozenset(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None


def _code_references_name(fn: Any, name: str) -> bool | None:
    try:
        root_code = getattr(inspect.unwrap(fn), "__code__", None)
    except Exception:
        return None
    if not isinstance(root_code, types.CodeType):
        return None
    stack = [root_code]
    while stack:
        code = stack.pop()
        if name in code.co_names or name in code.co_varnames or name in code.co_freevars:
            return True
        stack.extend(c for c in code.co_consts if isinstance(c, types.CodeType))
    return False


def _layer_has_native_kda_checkpoint(layer_cls: Any) -> bool:
    cls_dict = getattr(layer_cls, "__dict__", {})
    for method_name in ("get_kv_cache_checkpoint_alignment", "get_prefill_checkpoint_alignment"):
        method = cls_dict.get(method_name)
        if method is not None and not getattr(method, _KDA_CKPT_PATCH_ATTR, False):
            return True
    orig_forward = cls_dict.get("_forward", getattr(layer_cls, "_forward", None))
    return (
        orig_forward is not None
        and not getattr(orig_forward, _KDA_CKPT_PATCH_ATTR, False)
        and _code_references_name(orig_forward, "checkpoint_state") is True
    )


def _non_bool_int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _select_checkpoint_raw_qkv(raw_qkv_ns: Any, attn_metadata: Any, q: Any) -> tuple[Any, bool]:
    num_decodes = _non_bool_int(getattr(attn_metadata, "num_decodes", 0))
    decode_tokens = _non_bool_int(getattr(attn_metadata, "num_decode_tokens", 0)) if num_decodes > 0 else 0
    if decode_tokens <= 0:
        return raw_qkv_ns, False
    q_tokens = q.shape[1] if isinstance(q, torch.Tensor) and q.ndim >= 2 else None
    if q_tokens is not None and isinstance(raw_qkv_ns, torch.Tensor) and q_tokens == raw_qkv_ns.shape[0]:
        return raw_qkv_ns, False
    return raw_qkv_ns[decode_tokens:], True


def _scatter_masked_tensor(value: torch.Tensor, full_mask: torch.Tensor, pad_val: int) -> torch.Tensor:
    if value.shape[0] == 0:
        return value.new_full((full_mask.shape[0], *value.shape[1:]), pad_val)
    # Clamp to [0, base_len - 1] so False rows safely gather before torch.where overwrites.
    src_indices = (full_mask.to(dtype=torch.int64).cumsum(dim=0) - 1).clamp(min=0, max=value.shape[0] - 1)
    gathered = value.index_select(0, src_indices)
    mask_view = full_mask.view(full_mask.shape[0], *([1] * (value.ndim - 1)))
    return torch.where(mask_view, gathered, torch.full_like(gathered, pad_val))


def _scatter_checkpoint_with_mask(
    checkpoint: Any,
    base_len: int,
    full_mask: torch.Tensor,
    pad_values: dict[str, int],
    *,
    full_mask_list: list[bool] | None,
) -> Any:
    updates: dict[str, Any] = {}
    cached_masks: dict[Any, torch.Tensor] = {}
    for field_name, pad_val in pad_values.items():
        value = getattr(checkpoint, field_name, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == base_len:
            mask = cached_masks.get(value.device)
            if mask is None:
                mask = full_mask if full_mask.device == value.device else full_mask.to(value.device, non_blocking=True)
                cached_masks[value.device] = mask
            updates[field_name] = _scatter_masked_tensor(value, mask, pad_val)

    for field_name in CHECKPOINT_LIST_FIELDS:
        value = getattr(checkpoint, field_name, None)
        if full_mask_list is not None:
            if isinstance(value, (tuple, list)) and len(value) == base_len:
                it = iter(value)
                scattered = [next(it) if keep else 0 for keep in full_mask_list]
                updates[field_name] = tuple(scattered) if isinstance(value, tuple) else scattered
        elif value is not None:
            # Clear stale host lists on device-only fallback without triggering D2H .tolist().
            updates[field_name] = None
    return clone_with_updates(checkpoint, updates) if updates else checkpoint


def _align_checkpoint_rows(
    checkpoint: Any,
    num_seqs: int,
    num_decodes: int,
    peeled_decodes: bool,
    *,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> Any:
    """Align compacted checkpoint rows with ``num_seqs`` rows (handling decodes & zero-len pads)."""
    if checkpoint is None or num_seqs <= 0:
        return checkpoint
    ckpt_offsets = getattr(checkpoint, "checkpoint_offsets", None)
    if not isinstance(ckpt_offsets, torch.Tensor) or ckpt_offsets.ndim < 1:
        return checkpoint

    base_len = ckpt_offsets.shape[0]
    if peeled_decodes and num_decodes > 0 and base_len == num_decodes + num_seqs:
        updates: dict[str, Any] = {
            name: val[num_decodes:]
            for name in CHECKPOINT_TENSOR_FIELDS
            if isinstance((val := getattr(checkpoint, name, None)), torch.Tensor)
            and val.ndim >= 1
            and val.shape[0] == base_len
        }
        for name in CHECKPOINT_LIST_FIELDS:
            val = getattr(checkpoint, name, None)
            if isinstance(val, (tuple, list)) and len(val) == base_len:
                updates[name] = val[num_decodes:]
        return clone_with_updates(checkpoint, updates) if updates else checkpoint

    prefix_rows = 0 if peeled_decodes else max(num_decodes, 0)
    if base_len >= num_seqs or prefix_rows > num_seqs:
        return checkpoint

    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    pad_values = {"checkpoint_offsets": 0, "state_indices": NULL_BLOCK_ID, "request_rows": 0, "block_cols": -1}
    if prefix_rows > 0 and prefix_rows + base_len == num_seqs:
        updates = {}
        for name, pad_val in pad_values.items():
            val = getattr(checkpoint, name, None)
            if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[0] == base_len:
                prefix = torch.full((prefix_rows, *val.shape[1:]), pad_val, dtype=val.dtype, device=val.device)
                updates[name] = torch.cat((prefix, val), dim=0)
        for name in CHECKPOINT_LIST_FIELDS:
            val = getattr(checkpoint, name, None)
            if isinstance(val, (tuple, list)) and len(val) == base_len:
                updates[name] = ((0,) * prefix_rows + val) if isinstance(val, tuple) else ([0] * prefix_rows + val)
        return clone_with_updates(checkpoint, updates) if updates else checkpoint

    cu_cpu = next(
        (
            c
            for c in (cu_seqlens_cpu, cu_seqlens)
            if isinstance(c, torch.Tensor) and c.device.type == "cpu" and c.ndim == 1 and c.shape[0] == num_seqs + 1
        ),
        None,
    )
    if cu_cpu is not None:
        live_cpu = (cu_cpu[prefix_rows + 1 : num_seqs + 1] - cu_cpu[prefix_rows:num_seqs]) > 0
        if int(live_cpu.sum().item()) != base_len:
            return checkpoint
        full_mask_cpu = torch.zeros((num_seqs,), dtype=torch.bool)
        full_mask_cpu[prefix_rows:] = live_cpu
        return _scatter_checkpoint_with_mask(
            checkpoint, base_len, full_mask_cpu, pad_values, full_mask_list=full_mask_cpu.tolist()
        )

    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.ndim != 1 or cu_seqlens.shape[0] != num_seqs + 1:
        return checkpoint
    live_dev = (cu_seqlens[prefix_rows + 1 : num_seqs + 1] - cu_seqlens[prefix_rows:num_seqs]) > 0
    full_mask_dev = (
        torch.cat((cu_seqlens.new_zeros((prefix_rows,), dtype=torch.bool), live_dev), dim=0)
        if prefix_rows > 0
        else live_dev
    )
    return _scatter_checkpoint_with_mask(checkpoint, base_len, full_mask_dev, pad_values, full_mask_list=None)


def _make_kda_dispatch_wrapper(
    hook_var: contextvars.ContextVar[Callable[..., Any] | None] | None = None,
) -> tuple[Callable[..., Any], contextvars.ContextVar[Callable[..., Any] | None]]:
    active_hook = hook_var or contextvars.ContextVar("ascend_kda_checkpoint_hook", default=None)

    @functools.wraps(ascend_chunk_kda_with_fused_gate)
    def _dispatch(*args: Any, **kwargs: Any) -> Any:
        hook = active_hook.get()
        if hook is not None:
            return hook(ascend_chunk_kda_with_fused_gate, *args, **kwargs)
        return ascend_chunk_kda_with_fused_gate(*args, **kwargs)

    setattr(_dispatch, _KDA_CKPT_PATCH_ATTR, True)
    setattr(_dispatch, _KDA_CKPT_HOOK_VAR_ATTR, active_hook)
    return _dispatch, active_hook


def _install_kda_dispatch_binding(
    target_mod: Any,
    dispatch_fn: Callable[..., Any] | None,
    hook_var: contextvars.ContextVar[Callable[..., Any] | None] | None,
) -> tuple[Callable[..., Any] | None, contextvars.ContextVar[Callable[..., Any] | None] | None]:
    current = getattr(target_mod, "chunk_kda_with_fused_gate", None)
    if current is None:
        return dispatch_fn, hook_var
    if getattr(current, _KDA_CKPT_PATCH_ATTR, False):
        return current, getattr(current, _KDA_CKPT_HOOK_VAR_ATTR, hook_var)

    param_names = _safe_param_names(current)
    if param_names is None or "checkpoint_state" in param_names:
        return dispatch_fn, hook_var
    if dispatch_fn is None:
        dispatch_fn, hook_var = _make_kda_dispatch_wrapper(hook_var)
    elif hook_var is None:
        hook_var = getattr(dispatch_fn, _KDA_CKPT_HOOK_VAR_ATTR, None)

    target_mod.chunk_kda_with_fused_gate = dispatch_fn  # type: ignore[attr-defined]
    return dispatch_fn, hook_var


def _extract_eager_break_decorator(fn: Any) -> tuple[Any, Callable[[Any], Any] | None]:
    inner = getattr(fn, "__wrapped__", None)
    code = getattr(fn, "__code__", None)
    if inner is None or not isinstance(code, types.CodeType):
        return fn, None
    if "BreakableCUDAGraphCapture" not in code.co_names or "add_eager" not in code.co_names:
        return fn, None
    fn_globals = getattr(fn, "__globals__", None)
    decorator = fn_globals.get("eager_break_during_capture") if isinstance(fn_globals, dict) else None
    if not callable(decorator):
        try:
            bc_mod = importlib.import_module("vllm.compilation.breakable_cudagraph")
            decorator = getattr(bc_mod, "eager_break_during_capture", None)
        except ImportError:
            decorator = None
    return (inner, decorator) if callable(decorator) else (fn, None)


def _wrap_kimi_gdn_forward(
    layer_cls: Any,
    orig_forward: Any,
    hook_var: contextvars.ContextVar[Callable[..., Any] | None],
) -> None:
    import vllm.forward_context as forward_context_mod
    import vllm.model_executor.layers.mamba.mamba_utils as mamba_utils_mod

    inner_forward, eager_break_decorator = _extract_eager_break_decorator(orig_forward)

    @functools.wraps(inner_forward)
    def _forward_with_checkpoint(
        self: Any, mixed_qkv: Any, g1: Any, g2: Any, beta: Any, core_attn_out: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if not getattr(self, "_ascend_internal_checkpoint_enabled", False):
            return inner_forward(self, mixed_qkv, g1, g2, beta, core_attn_out, *args, **kwargs)
        attn_raw = forward_context_mod.get_forward_context().attn_metadata
        attn_meta = attn_raw.get(self.prefix) if isinstance(attn_raw, dict) else None
        checkpoint = getattr(attn_meta, "checkpoint", None)
        if attn_meta is None or checkpoint is None or _non_bool_int(getattr(attn_meta, "num_prefills", 0)) <= 0:
            return inner_forward(self, mixed_qkv, g1, g2, beta, core_attn_out, *args, **kwargs)

        trimmed = mixed_qkv[: _non_bool_int(getattr(attn_meta, "num_actual_tokens", mixed_qkv.shape[0]))]
        non_spec_indx = getattr(attn_meta, "non_spec_token_indx", None)
        raw_qkv_ns = (
            trimmed.index_select(0, non_spec_indx)
            if getattr(attn_meta, "spec_sequence_masks", None) is not None and non_spec_indx is not None
            else trimmed
        )
        num_decodes = _non_bool_int(getattr(attn_meta, "num_decodes", 0))
        dim_first = mamba_utils_mod.is_conv_state_dim_first()

        def _chunk_kda_and_export(base_chunk_fn: Callable[..., Any], *call_args: Any, **call_kwargs: Any) -> Any:
            cu_seqlens = call_kwargs.get("cu_seqlens")
            initial_state = call_kwargs.get("initial_state")
            q_arg = call_kwargs.get("q") if "q" in call_kwargs else (call_args[0] if call_args else None)
            num_seqs = (
                initial_state.shape[0]
                if initial_state is not None
                else (cu_seqlens.numel() - 1 if cu_seqlens is not None else 0)
            )
            raw_qkv_for_export, peeled = _select_checkpoint_raw_qkv(raw_qkv_ns, attn_meta, q_arg)
            active_ckpt = _align_checkpoint_rows(
                checkpoint,
                num_seqs,
                num_decodes,
                peeled,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=getattr(attn_meta, "non_spec_query_start_loc_cpu", None),
            )
            ckpt_state, ckpt_offsets = get_triton_kda_checkpoint_buffer(
                active_ckpt, self._triton_checkpoint_spec, num_seqs, reference_tensor=self.kv_cache[1]
            )
            call_kwargs["checkpoint_state"] = ckpt_state
            call_kwargs["checkpoint_offsets"] = ckpt_offsets
            output, last_state = base_chunk_fn(*call_args, **call_kwargs)
            export_kda_prefill_checkpoint(
                self._checkpoint_exporter,
                active_ckpt,
                raw_qkv=raw_qkv_for_export,
                conv_state=self.kv_cache[0],
                conv_state_is_dim_first=dim_first,
                recurrent_checkpoint=ckpt_state,
                recurrent_state=self.kv_cache[1],
                cu_seqlens=cu_seqlens,
            )
            return output, last_state

        token = hook_var.set(_chunk_kda_and_export)
        try:
            return inner_forward(self, mixed_qkv, g1, g2, beta, core_attn_out, *args, **kwargs)
        finally:
            hook_var.reset(token)

    setattr(_forward_with_checkpoint, _KDA_CKPT_PATCH_ATTR, True)
    # Keep @eager_break_during_capture outermost so capture skips host/graph-break work.
    wrapped = (
        eager_break_decorator(_forward_with_checkpoint)
        if eager_break_decorator is not None
        else _forward_with_checkpoint
    )
    setattr(wrapped, _KDA_CKPT_PATCH_ATTR, True)
    layer_cls._forward = wrapped


def _patch_kimi_gdn_linear_attn(
    dispatch_fn: Callable[..., Any] | None,
    hook_var: contextvars.ContextVar[Callable[..., Any] | None] | None,
) -> None:
    try:
        kimi_gdn_mod = importlib.import_module("vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn")
    except ImportError:
        return
    layer_cls = getattr(kimi_gdn_mod, "KimiGatedDeltaNetAttention", None)
    if layer_cls is None or _layer_has_native_kda_checkpoint(layer_cls):
        return

    dispatch_fn, hook_var = _install_kda_dispatch_binding(kimi_gdn_mod, dispatch_fn, hook_var)
    orig_init = layer_cls.__init__
    if not getattr(orig_init, _KDA_CKPT_PATCH_ATTR, False):

        @functools.wraps(orig_init)
        def _init_with_checkpoints(self: Any, *args: Any, **kwargs: Any) -> None:
            orig_init(self, *args, **kwargs)
            vllm_config = (
                kwargs.get("vllm_config") or (args[1] if len(args) >= 2 else None) or getattr(self, "vllm_config", None)
            )
            if getattr(self, "_triton_checkpoint_spec", None) is None:
                initialize_kda_checkpoints(self, vllm_config)
            else:
                self._ascend_internal_checkpoint_enabled = True

        setattr(_init_with_checkpoints, _KDA_CKPT_PATCH_ATTR, True)
        layer_cls.__init__ = _init_with_checkpoints

    def _get_alignment(self: Any, *_args: Any) -> int | None:
        return KDA_PREFILL_CHECKPOINT_ALIGNMENT if getattr(self, "_ascend_internal_checkpoint_enabled", False) else None

    setattr(_get_alignment, _KDA_CKPT_PATCH_ATTR, True)
    cls_dict = getattr(layer_cls, "__dict__", {})
    for attr_name in ("get_prefill_checkpoint_alignment", "get_kv_cache_checkpoint_alignment"):
        if attr_name not in cls_dict:
            setattr(layer_cls, attr_name, _get_alignment)

    orig_get_kv_spec = getattr(layer_cls, "get_kv_cache_spec", None)
    if orig_get_kv_spec is not None and not getattr(orig_get_kv_spec, _KDA_CKPT_PATCH_ATTR, False):

        @functools.wraps(orig_get_kv_spec)
        def _get_kv_cache_spec(self: Any, vllm_config: Any) -> Any:
            return build_kda_kv_cache_spec(self, vllm_config, orig_get_kv_spec(self, vllm_config))

        setattr(_get_kv_cache_spec, _KDA_CKPT_PATCH_ATTR, True)
        layer_cls.get_kv_cache_spec = _get_kv_cache_spec

    orig_forward = getattr(layer_cls, "_forward", None)
    if orig_forward is not None and hook_var is not None and not getattr(orig_forward, _KDA_CKPT_PATCH_ATTR, False):
        if _code_references_name(orig_forward, "checkpoint_state") is False:
            _wrap_kimi_gdn_forward(layer_cls, orig_forward, hook_var)


def apply_patch() -> None:
    """Apply KDA checkpoint patches for Triton bindings and legacy attention layers."""
    try:
        nvidia_kda_mod = importlib.import_module("vllm.models.kimi_k3.nvidia.ops.third_party.kda")
        dispatch_fn, hook_var = _install_kda_dispatch_binding(nvidia_kda_mod, None, None)
    except ImportError:
        dispatch_fn, hook_var = None, None
    _patch_kimi_gdn_linear_attn(dispatch_fn, hook_var)


apply_patch()
