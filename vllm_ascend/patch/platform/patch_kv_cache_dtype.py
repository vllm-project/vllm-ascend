# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patch target: vllm/config/cache.py
# - DeepSeek V4 / similar Ascend sparse-attention models store the MLA KV
#   cache in ``int8``; the A5 QFA path stores it as C8-MXFP (FP8 KV + E8M0
#   scales) behind ``--kv-cache-dtype mxfp8``. Current vLLM main rejects both
#   strings in ``CacheConfig._validate_cache_dtype`` via
#   ``is_known_kv_cache_dtype``, which reads the mutable
#   ``vllm.config.cache.KV_CACHE_DTYPES`` mirror list -- the same list backs
#   the ``get_attn_backend`` assertion in ``vllm/v1/attention/selector.py``.
#   (Older releases instead gated on the ``CacheDType`` Literal with a
#   pydantic ``literal_error``.) This patch widens both: it appends
#   ``"int8"``/``"mxfp8"`` to ``KV_CACHE_DTYPES`` and widens the
#   ``CacheDType`` Literal by rebuilding the pydantic dataclass schema in
#   place, without touching the upstream vllm tree.
#
# Why a platform patch:
#   ``vllm serve`` builds its argparse parser in
#   ``AsyncEngineArgs.add_cli_args`` (``vllm/engine/arg_utils.py``) before
#   ``current_platform.pre_register_and_update(parser)`` runs
#   ``adapt_patch(is_global_patch=True)`` (i.e. this module). Appending to
#   the mirror list makes the later ``CacheConfig`` construction accept the
#   Ascend dtypes; the Literal rebuild additionally keeps ``get_args``
#   consumers (e.g. the CLI choices widening in
#   ``vllm_ascend/platform.py::pre_register_and_update``) in sync. Worker
#   processes re-apply the same widening via
#   ``patch/worker/patch_kv_cache_dtype.py``.

import typing

import pydantic.dataclasses as _pdc
from vllm.config import cache as _cache_mod
from vllm.config.cache import CacheConfig
from vllm.logger import logger

# Ascend-only ``--kv-cache-dtype`` values: ``"int8"`` (DeepSeek V4 / similar
# Ascend MLA KV cache dtype) and ``"mxfp8"`` (C8-MXFP KV cache on the QFA
# path).
_ASCEND_CACHE_DTYPES = ("int8", "mxfp8")


def _apply_kv_cache_dtype_int8_patch() -> None:
    """Widen ``CacheConfig.cache_dtype`` to accept ``"int8"`` and ``"mxfp8"``.

    A pydantic dataclass caches its core schema / validator at class-build
    time from the field annotations. Mutating the annotation alone is not
    enough; the schema must be rebuilt. Three places hold the old Literal:

      1. ``vllm.config.cache.CacheDType`` -- the module-level symbol
         re-exported / imported by other modules (e.g.
         ``vllm/v1/attention/selector.py`` reads it via ``get_args`` at
         runtime).
      2. ``CacheConfig.__annotations__["cache_dtype"]`` -- the inline
         annotation on the class.
      3. ``CacheConfig.__dataclass_fields__["cache_dtype"].type`` -- the
         stdlib dataclass field type, which pydantic's
         ``collect_dataclass_fields`` reads when rebuilding (not the
         ``__pydantic_fields__`` FieldInfo, which it overwrites).

    After updating all three we call ``rebuild_dataclass(force=True)`` to
    regenerate the validator + core schema. This is idempotent and safe to
    call multiple times / across plugin entry points (the ``force`` flag
    rebuilds even though ``__pydantic_complete__`` is already True).
    """
    # Widen the mutable mirror list first: ``is_known_kv_cache_dtype`` (the
    # ``_validate_cache_dtype`` field validator and the ``get_attn_backend``
    # assertion) reads this list, so a widened Literal alone still dies with
    # "Invalid kv_cache_dtype" at CacheConfig construction. The mirror list
    # only exists on newer vLLM main; releases (e.g. v0.32.0) gate on the
    # Literal alone, so the append is main-only.
    mirror_list = getattr(_cache_mod, "KV_CACHE_DTYPES", None)
    if mirror_list is not None:
        for dtype in _ASCEND_CACHE_DTYPES:
            if dtype not in mirror_list:
                mirror_list.append(dtype)

    # Idempotency: if both are already accepted, nothing to do.
    existing_args = getattr(_cache_mod.CacheDType, "__args__", ())
    missing = tuple(d for d in _ASCEND_CACHE_DTYPES if d not in existing_args)
    if not missing:
        return

    # Derive from the current upstream Literal and append the Ascend-only
    # dtypes so no upstream member is dropped (a hardcoded list goes stale
    # as upstream adds members). Subscripting with the plain tuple is the
    # same __getitem__ call as star-unpacking it and parses on Python 3.10
    # (PEP 646 starred subscripts are 3.11+).
    widened = typing.Literal[existing_args + missing]  # type: ignore[valid-type]

    _cache_mod.CacheDType = widened
    CacheConfig.__annotations__["cache_dtype"] = widened
    CacheConfig.__dataclass_fields__["cache_dtype"].type = widened

    # Rebuild the pydantic dataclass schema so the new Literal is enforced.
    # ``_parent_namespace_depth=1`` makes pydantic resolve types against this
    # module's namespace (where the new Literal lives).
    _pdc.rebuild_dataclass(
        CacheConfig,
        force=True,
        raise_errors=True,
        _parent_namespace_depth=1,
    )
    logger.info("Patched CacheConfig.cache_dtype to accept 'int8' and 'mxfp8' (Ascend KV cache dtypes).")


_apply_kv_cache_dtype_int8_patch()
