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
#   cache in ``int8``. Upstream's ``CacheConfig.cache_dtype`` field is typed
#   as ``CacheDType = Literal["auto", "float16", ...]`` and rejects ``"int8"``
#   at CLI/config validation time with a pydantic ``literal_error`` -- even
#   though ``kv_cache_dtype_str_to_dtype`` supports ``int8`` (it maps to
#   ``torch.int8`` upstream, see ``vllm/utils/torch_utils.py``).
#   This patch widens the accepted ``Literal`` to include ``"int8"`` without
#   touching the upstream vllm tree, by rebuilding the pydantic dataclass
#   schema in place.
#
# Why a platform patch:
#   ``vllm serve`` builds its argparse parser in
#   ``AsyncEngineArgs.add_cli_args`` (``vllm/engine/arg_utils.py``). The
#   ``--kv-cache-dtype`` choices are generated from the upstream
#   ``CacheDType`` Literal while the parser is being built, *before*
#   ``current_platform.pre_register_and_update(parser)`` runs
#   ``adapt_patch(is_global_patch=True)`` (i.e. this module). So the argparse
#   choices are widened separately in
#   ``vllm_ascend/platform.py::pre_register_and_update`` (same pattern as the
#   ``--quantization`` choice). Rebuilding the schema here makes the new
#   ``Literal`` visible to the pydantic validation of ``CacheConfig`` when it
#   is constructed later from the parsed args -- in time for validation to
#   accept ``int8``.

import typing

import pydantic.dataclasses as _pdc
from vllm.config import cache as _cache_mod
from vllm.config.cache import CacheConfig
from vllm.logger import logger

# The upstream Literal that gates ``cache_dtype``. We widen it to also accept
# ``"int8"`` (DeepSeek V4 Ascend MLA KV cache dtype) and ``"mxfp8"`` (C8-MXFP
# KV cache on the QFA path). Keep the existing members verbatim so other
# behavior is unchanged.
_ORIG_CACHE_DTYPE = _cache_mod.CacheDType
_CACHE_DTYPE_WITH_ASCEND = typing.Literal[
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "int4_per_token_head",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
    "nvfp4_4over6",
    "int8",
    "mxfp8",
]


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
    # Idempotency: if int8 is already accepted, nothing to do.
    existing_args = getattr(_ORIG_CACHE_DTYPE, "__args__", ())
    if "int8" in existing_args and "mxfp8" in existing_args:
        return

    _cache_mod.CacheDType = _CACHE_DTYPE_WITH_ASCEND
    CacheConfig.__annotations__["cache_dtype"] = _CACHE_DTYPE_WITH_ASCEND
    CacheConfig.__dataclass_fields__["cache_dtype"].type = _CACHE_DTYPE_WITH_ASCEND

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
