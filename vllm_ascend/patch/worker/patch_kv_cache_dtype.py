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

Worker-side dtype Literal widening: the launcher's platform patches
(``patch/platform/patch_kv_cache_dtype.py`` and
``patch/platform/patch_indexer_kv_dtype.py``) let ``--kv-cache-dtype int8``
and ``--attention_config.indexer_kv_dtype int8`` through CLI/pydantic
validation, but only in the main process. Worker processes run in fresh
interpreters, so the ``CacheDType`` / ``IndexerKVDType`` Literals still reject
``"int8"`` there. This patch widens both in every worker, mirroring the
platform patches. Without it, ``vllm/v1/attention/selector.py::get_attn_backend``
dies with ``AssertionError: Invalid kv_cache_dtype: int8`` when the attention
backend is selected during model init, and the speculative draft path
(``DraftModelProposer._create_draft_vllm_config`` -> ``replace``) re-validates
``AttentionConfig`` and raises a pydantic ``literal_error``.
"""

import typing

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


def _widen_cache_dtype_literal_for_worker() -> None:
    """Widen ``CacheConfig.cache_dtype`` to accept ``"int8"`` in the worker.

    Mirrors ``patch/platform/patch_kv_cache_dtype.py``: update the module-level
    Literal, the ``CacheConfig`` annotation and dataclass field type, then
    rebuild the pydantic schema.

    Fork trap: the worker is forked from the main process, where the platform
    patch has already widened ``vllm.config.cache.CacheDType`` and rebuilt
    ``CacheConfig``; those changes are inherited, so the early-return branch
    below is the NORMAL path in the real worker. However, the selector module
    (``vllm.v1.attention.selector``) may have been imported in the main process
    BEFORE the platform patch widened the Literal, so its module-global
    ``CacheDType`` still binds the ORIGINAL Literal even though the cache
    module's attribute is widened. ``get_attn_backend``'s assertion
    (``assert kv_cache_dtype in get_args(CacheDType)``) reads that stale module
    global. The rebind below must therefore run UNCONDITIONALLY, in both
    branches.
    """
    import pydantic.dataclasses as _pdc
    from vllm.config import cache as _cache_mod
    from vllm.config.cache import CacheConfig

    existing_args = getattr(_cache_mod.CacheDType, "__args__", ())
    if "int8" not in existing_args:
        # Derive from the current upstream Literal and append "int8" so no
        # upstream member is dropped.
        widened = typing.Literal[*existing_args + ("int8",)]  # type: ignore[valid-type]

        _cache_mod.CacheDType = widened
        CacheConfig.__annotations__["cache_dtype"] = widened
        CacheConfig.__dataclass_fields__["cache_dtype"].type = widened

        _pdc.rebuild_dataclass(
            CacheConfig,
            force=True,
            raise_errors=True,
            _parent_namespace_depth=1,
        )
    else:
        # Already widened by the platform patch in the main process and
        # inherited via fork; nothing to patch here.
        widened = _cache_mod.CacheDType

    # Unconditional rebind: the selector's module-global CacheDType must point
    # at the widened Literal regardless of import order / fork inheritance.
    try:
        import vllm.v1.attention.selector as _selector
    except ImportError:
        pass
    else:
        _selector.CacheDType = widened


def _widen_indexer_kv_dtype_literal_for_worker() -> None:
    """Widen ``AttentionConfig.indexer_kv_dtype`` to accept ``"int8"`` in the worker.

    Mirrors ``patch/platform/patch_indexer_kv_dtype.py``. Needed because the
    speculative draft path runs in the worker and re-instantiates
    ``AttentionConfig`` via ``replace()``, which passes every field explicitly
    (including the default ``indexer_kv_dtype="auto"``) and re-validates it
    against the Literal.
    """
    import pydantic.dataclasses as _pdc
    from vllm.config import attention as _attention_mod
    from vllm.config.attention import AttentionConfig

    existing_args = getattr(_attention_mod.IndexerKVDType, "__args__", ())
    if "int8" in existing_args:
        return

    widened = typing.Literal[*existing_args + ("int8",)]  # type: ignore[valid-type]

    _attention_mod.IndexerKVDType = widened
    AttentionConfig.__annotations__["indexer_kv_dtype"] = widened
    AttentionConfig.__dataclass_fields__["indexer_kv_dtype"].type = widened

    _pdc.rebuild_dataclass(
        AttentionConfig,
        force=True,
        raise_errors=True,
        _parent_namespace_depth=1,
    )


_apply_fp8_kv_cache_dtype_flip()
_widen_cache_dtype_literal_for_worker()
_widen_indexer_kv_dtype_literal_for_worker()
