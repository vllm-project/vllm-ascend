# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""vllm ``releases/v0.27.1`` does not ship the pluggable kv-cache-dtype
mechanism that the ``kvquant_27`` branch added (``register_kv_cache_dtype``
injects ``fp8 -> torch.float8_e4m3fn`` into ``STR_DTYPE_TO_TORCH_DTYPE`` in
place, so Ascend's MLA/DSA kernels get native float8 instead of upstream's
``torch.uint8`` raw bytes).

On vllm builds that lack ``register_kv_cache_dtype`` this patch reproduces
that effect directly: it mutates
``vllm.utils.torch_utils.STR_DTYPE_TO_TORCH_DTYPE["fp8"]`` to
``torch.float8_e4m3fn`` in the live per-process dict, so the call sites in
``vllm_ascend/models/deepseek_v4/{model,indexer}.py`` that do
``kv_cache_dtype_str_to_dtype("fp8", ...)`` resolve to ``float8_e4m3fn``.

This runs as a worker patch (``adapt_patch(is_global_patch=False)`` in
``NPUWorker.__init__``) because the dtype is resolved inside each spawned
Worker process — the launcher's platform patch does not propagate to the
workers' fresh interpreters. It is applied before model loading, which is
when ``kv_cache_dtype_str_to_dtype`` is called.

On vllm builds that *do* provide ``register_kv_cache_dtype`` (kvquant_27 /
future main) this patch is a no-op: the eager handler import in
``vllm_ascend/worker/worker.py`` already flipped the dtype, so we must not
touch the dict again (idempotency, and to avoid masking the handler path).
"""

import torch
import vllm.utils.torch_utils as _torch_utils


def _apply_fp8_kv_cache_dtype_flip() -> None:
    try:
        from vllm.config.cache import register_kv_cache_dtype  # noqa: F401
    except ImportError:
        # No pluggable mechanism: flip the dtype in place to match what
        # Fp8AscendHandler.torch_dtype() would have injected.
        _torch_utils.STR_DTYPE_TO_TORCH_DTYPE["fp8"] = torch.float8_e4m3fn
        # int8 already maps to torch.int8 upstream; nothing to flip.
        return

    # Pluggable mechanism present: the handler import in worker.py already
    # injected fp8 -> float8_e4m3fn. Leave the dict untouched so the handler
    # path stays the source of truth (and so is_quantized/quant_mode queries
    # routed through the handler remain consistent).
    return


_apply_fp8_kv_cache_dtype_flip()
