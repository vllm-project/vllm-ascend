# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Route vLLM's sparse-indexer per-row top-k to the AscendC TopKPerRow kernel.

``vllm/model_executor/layers/indexer_topk.py`` resolves its top-k backend in
``SparseIndexerTopk._resolve_auto``: ``cooperative`` needs CUDA and
``persistent`` needs CUDA plus ``topk_tokens in (512, 1024, 2048)``, so on NPU
the chain falls through to ``per_row``. That backend calls
``vllm._custom_ops.top_k_per_row_decode``, whose body is
``torch.ops._C.top_k_per_row_decode`` -- an operator that only exists when vLLM
is built with CUDA. On Ascend the call raises, and with it every model that
drives the DSA indexer through vLLM's own implementation.

``torch.ops._C_ascend.top_k_per_row_prefill`` / ``.top_k_per_row_decode`` take
the same argument list (see ``csrc/attention/top_k_per_row`` and the schemas in
``csrc/torch_binding.cpp``), so rebinding the two ``vllm._custom_ops`` module
attributes is enough: the indexer looks the names up on the module at call time,
so no vLLM source has to be modified and no import order has to be preserved.

The rebind happens at import time, but the *choice* between the Ascend operator
and vLLM's own one is made per call. That is deliberate: the Ascend extension
(``vllm_ascend.vllm_ascend_C``, which registers ``torch.ops._C_ascend``) is
imported lazily by ``vllm_ascend.utils.enable_custom_op`` so that CANN's RTS is
not initialised before ``torch.npu.set_device()``, so ``_C_ascend`` may not exist
yet when this module is imported inside a worker. Resolving the operator per call
means the routing starts working as soon as the operator is registered, and
falls back to vLLM's implementation -- the pre-patch behaviour -- while it is not.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

import vllm._custom_ops as vllm_custom_ops

PREFILL_ENTRY = "top_k_per_row_prefill"
DECODE_ENTRY = "top_k_per_row_decode"


def _ascend_op(name: str) -> Callable[..., Any] | None:
    """``torch.ops._C_ascend.<name>`` when the operator is registered, else None."""
    ascend_ops = getattr(torch.ops, "_C_ascend", None)
    if ascend_ops is None:
        return None
    return getattr(ascend_ops, name, None)


# Indirection so a test can drive the dispatch without an NPU, and so the real
# lookup stays late-bound (see the module docstring).
_LOOKUP: Callable[[str], Any] = _ascend_op


def _dispatch(name: str, fallback: Callable[..., Any] | None, args: tuple) -> None:
    op = _LOOKUP(name)
    if op is not None:
        op(*args)
        return
    if fallback is not None:
        fallback(*args)
        return
    raise RuntimeError(
        f"neither torch.ops._C_ascend.{name} nor vLLM's own implementation is "
        "available on this device; the TopKPerRow operator was not built"
    )


def install(ops_module: Any = None) -> bool:
    """Rebind vLLM's per-row top-k entry points onto the AscendC operator.

    Returns True when the entry points were replaced, False when they were
    already rebound by an earlier call. ``ops_module`` is injectable so the unit
    tests can drive the rebind without touching the real ``vllm._custom_ops``.
    """
    target = vllm_custom_ops if ops_module is None else ops_module

    current_prefill = getattr(target, PREFILL_ENTRY, None)
    if getattr(current_prefill, "_ascend_dispatch", False):
        return False
    # Keep whatever vLLM installed, so the wrappers can fall back to it while the
    # Ascend operator is not registered (or not built at all, e.g. on CPU).
    fallback_prefill = current_prefill
    fallback_decode = getattr(target, DECODE_ENTRY, None)

    # Same positional signatures as vllm._custom_ops.top_k_per_row_{prefill,decode}.
    # The CUDA kernels take stride0/stride1; the Ascend operator derives the row
    # width from the tensor, so the strides are accepted and ignored there.
    def top_k_per_row_prefill(
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        topk_tokens: int,
    ) -> None:
        _dispatch(
            PREFILL_ENTRY,
            fallback_prefill,
            (
                logits,
                cu_seqlen_ks,
                cu_seqlen_ke,
                raw_topk_indices,
                num_rows,
                stride0,
                stride1,
                topk_tokens,
            ),
        )

    def top_k_per_row_decode(
        logits: torch.Tensor,
        next_n: int,
        seq_lens: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        topk_tokens: int,
    ) -> None:
        _dispatch(
            DECODE_ENTRY,
            fallback_decode,
            (
                logits,
                next_n,
                seq_lens,
                raw_topk_indices,
                num_rows,
                stride0,
                stride1,
                topk_tokens,
            ),
        )

    top_k_per_row_prefill._ascend_dispatch = True  # type: ignore[attr-defined]
    top_k_per_row_decode._ascend_dispatch = True  # type: ignore[attr-defined]

    setattr(target, PREFILL_ENTRY, top_k_per_row_prefill)
    setattr(target, DECODE_ENTRY, top_k_per_row_decode)
    return True


INSTALLED = install()
