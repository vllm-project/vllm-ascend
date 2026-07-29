"""DSA sparse-cache EngineCore hooks."""

from __future__ import annotations

from functools import wraps

import vllm_ascend.patch.dsa_sparse.patch_kv_cache_utils  # noqa: F401
from vllm_ascend.patch.dsa_sparse import patch_kv_cache_utils as dsa_kv_utils
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core import kv_cache_utils as kv_utils
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
import vllm.v1.engine.core as engine_core_mod
from vllm.v1.engine.core import EngineCore

from vllm_ascend.dsa_sparse.dsa_config import (
    attach_dsa_sparse_cache_attrs,
    is_dsa_sparse_config_enabled,
)
from vllm_ascend.dsa_sparse.dsa_spec_utils import is_dsa_indexer_spec

_DSA_ENGINE_CORE_INIT_WRAPPER_ATTR = (
    "_vllm_ascend_dsa_engine_core_init_wrapper")
_DSA_INITIALIZE_KV_CACHES_WRAPPER_ATTR = (
    "_vllm_ascend_dsa_initialize_kv_caches_wrapper")


def _verify_engine_core_kv_configs_hook(vllm_config, boundary: str) -> None:
    if not is_dsa_sparse_config_enabled(vllm_config):
        return
    kv_fn = kv_utils.get_kv_cache_configs
    engine_fn = engine_core_mod.get_kv_cache_configs
    kv_ok = dsa_kv_utils.is_dsa_get_kv_cache_configs_wrapper(kv_fn)
    engine_ok = dsa_kv_utils.is_dsa_get_kv_cache_configs_wrapper(engine_fn)
    if kv_ok and engine_ok and engine_fn is kv_fn:
        return
    raise RuntimeError(
        "DSA sparse-cache is enabled, but EngineCore KV-cache config hook is "
        f"not using the vllm-ascend DSA wrapper at {boundary}. "
        f"kv_utils.get_kv_cache_configs="
        f"{dsa_kv_utils.describe_callable(kv_fn)} dsa_wrapper={kv_ok}; "
        f"engine_core.get_kv_cache_configs="
        f"{dsa_kv_utils.describe_callable(engine_fn)} dsa_wrapper={engine_ok}; "
        f"same_object={engine_fn is kv_fn}; "
        f"kv_cache_utils_patched="
        f"{getattr(kv_utils, '_dsa_kv_cache_utils_patched', None)}")


engine_core_mod.get_kv_cache_configs = kv_utils.get_kv_cache_configs


def is_dsa_engine_core_init_wrapper(fn) -> bool:
    return bool(getattr(fn, _DSA_ENGINE_CORE_INIT_WRAPPER_ATTR, False))


def is_dsa_initialize_kv_caches_wrapper(fn) -> bool:
    return bool(getattr(fn, _DSA_INITIALIZE_KV_CACHES_WRAPPER_ATTR, False))


def install_dsa_engine_core_patches() -> None:
    """Install DSA EngineCore hooks and re-assert them after other patches.

    vllm-ascend has several platform patches that also wrap EngineCore or the
    EngineCore subprocess entrypoint.  In the source-fork version of DSA these
    hooks lived directly in vLLM, but after moving them to vllm-ascend they must
    be resilient to import order.  Therefore the guard checks the *current*
    callable object rather than a historical "patched once" flag.
    """
    engine_core_mod.get_kv_cache_configs = kv_utils.get_kv_cache_configs

    if not is_dsa_initialize_kv_caches_wrapper(
            EngineCore._initialize_kv_caches):
        original_initialize_kv_caches = EngineCore._initialize_kv_caches

        @wraps(original_initialize_kv_caches)
        def _dsa_sparse_initialize_kv_caches(self: EngineCore, vllm_config):
            attach_dsa_sparse_cache_attrs(vllm_config)
            # `vllm.v1.engine.core` imports get_kv_cache_configs by value.
            # Refresh the alias at the last safe boundary before KV planning.
            engine_core_mod.get_kv_cache_configs = (
                kv_utils.get_kv_cache_configs)
            _verify_engine_core_kv_configs_hook(
                vllm_config, "before_initialize_kv_caches")
            scheduler_kv_cache_config = original_initialize_kv_caches(
                self, vllm_config)
            if is_dsa_sparse_config_enabled(vllm_config):
                dsa_kv_utils.report_dsa_kv_cache_config_or_raise(
                    vllm_config,
                    scheduler_kv_cache_config,
                )
                group_specs = tuple(
                    type(group.kv_cache_spec).__name__
                    for group in scheduler_kv_cache_config.kv_cache_groups)
                if not any(
                        is_dsa_indexer_spec(group.kv_cache_spec)
                        for group in
                        scheduler_kv_cache_config.kv_cache_groups):
                    raise RuntimeError(
                        "DSA sparse-cache is enabled after EngineCore "
                        "KV-cache initialization, but scheduler KV-cache "
                        "groups do not contain IndexerKVSpec. "
                        f"group_specs={group_specs} "
                        f"get_kv_cache_configs_alias_ok="
                        f"{engine_core_mod.get_kv_cache_configs is kv_utils.get_kv_cache_configs}"
                    )
            return scheduler_kv_cache_config

        setattr(_dsa_sparse_initialize_kv_caches,
                _DSA_INITIALIZE_KV_CACHES_WRAPPER_ATTR, True)
        EngineCore._dsa_sparse_original_initialize_kv_caches = (
            original_initialize_kv_caches)
        EngineCore._initialize_kv_caches = _dsa_sparse_initialize_kv_caches
        EngineCore._dsa_sparse_initialize_kv_caches_patched = True

    if not is_dsa_engine_core_init_wrapper(EngineCore.__init__):
        original_init = EngineCore.__init__

        @wraps(original_init)
        def _dsa_sparse_engine_core_init(self: EngineCore, *args,
                                         **kwargs) -> None:
            original_init(self, *args, **kwargs)

            vllm_config = self.vllm_config
            if not is_dsa_sparse_config_enabled(vllm_config):
                return
            if self.request_block_hasher is not None:
                return

            scheduler_block_size = (
                vllm_config.cache_config.block_size
                * vllm_config.parallel_config.decode_context_parallel_size
                * vllm_config.parallel_config.prefill_context_parallel_size
            )
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo)
            init_none_hash(caching_hash_fn)
            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn)

        setattr(_dsa_sparse_engine_core_init,
                _DSA_ENGINE_CORE_INIT_WRAPPER_ATTR, True)
        EngineCore._dsa_sparse_original_init = original_init
        EngineCore.__init__ = _dsa_sparse_engine_core_init
        EngineCore._dsa_sparse_engine_core_init_patched = True


install_dsa_engine_core_patches()
