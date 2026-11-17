# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Route vLLM's sparse-indexer per-row top-k to the AscendC TopKPerRow kernel.

``vllm/model_executor/layers/indexer_topk.py`` resolves its decode top-k kernel
in ``SparseIndexerTopk._resolve_auto``: ``cooperative`` and ``persistent`` both
require CUDA, so on NPU the chain falls through to ``per_row``. That backend then
calls ``vllm._custom_ops.top_k_per_row_decode``, whose body dispatches to
``torch.ops._C.top_k_per_row_decode`` -- an operator that only exists when vLLM is
built with CUDA. On Ascend the call therefore raises, and with it every model
that drives the DSA indexer through vLLM's own implementation.

``torch.ops._C_ascend.top_k_per_row_prefill`` / ``.top_k_per_row_decode`` accept
the same argument lists (see ``csrc/attention/top_k_per_row``) and implement them
on top of the TopKPerRow aclnn operator, so rebinding the two
``vllm._custom_ops`` module attributes is enough: the indexer looks the names up
on the module at call time, so no vLLM source has to be modified and no import
order has to be preserved.

The rebind is a no-op while the Ascend operator is not registered, which keeps
this patch independent of the operator PR: it starts routing as soon as the
``_C_ascend`` bindings exist.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

import vllm._custom_ops as vllm_custom_ops

PREFILL_ENTRY = "top_k_per_row_prefill"
DECODE_ENTRY = "top_k_per_row_decode"


def _ascend_op(name: str) -> Callable[..., Any] | None:
    """``torch.ops._C_ascend.<name>`` when the operator is built, else None."""
    ascend_ops = getattr(torch.ops, "_C_ascend", None)
    if ascend_ops is None:
        return None
    return getattr(ascend_ops, name, None)


def install(ops_module: Any = None, get_op: Callable[[str], Any] | None = None) -> bool:
    """Rebind vLLM's per-row top-k entry points onto the AscendC operator.

    Returns True when both entry points were replaced. ``ops_module`` and
    ``get_op`` are injectable so the unit test can drive the rebind without an
    NPU or a compiled operator.
    """
    target = vllm_custom_ops if ops_module is None else ops_module
    lookup = _ascend_op if get_op is None else get_op

    prefill_op = lookup(PREFILL_ENTRY)
    decode_op = lookup(DECODE_ENTRY)
    if prefill_op is None or decode_op is None:
        return False

    # Same positional signatures as vllm._custom_ops.top_k_per_row_{prefill,decode}.
    # The CUDA kernels take stride0/stride1; the Ascend operator indexes rows by
    # their logical width, so the strides are validated but not used.
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
        prefill_op(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            raw_topk_indices,
            num_rows,
            stride0,
            stride1,
            topk_tokens,
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
        decode_op(
            logits,
            next_n,
            seq_lens,
            raw_topk_indices,
            num_rows,
            stride0,
            stride1,
            topk_tokens,
        )

    setattr(target, PREFILL_ENTRY, top_k_per_row_prefill)
    setattr(target, DECODE_ENTRY, top_k_per_row_decode)
    return True


INSTALLED = install()
