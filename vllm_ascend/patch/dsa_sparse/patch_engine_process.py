"""Re-apply DSA sparse-cache patches inside EngineCore subprocesses."""

from __future__ import annotations

from vllm.v1.engine import utils as engine_utils
import vllm.v1.engine.core_client as core_client_mod
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.core_client import EngineCoreClient

from vllm_ascend.dsa_sparse.dsa_config import (
    attach_dsa_sparse_cache_attrs,
    is_dsa_sparse_config_enabled,
)

_DSA_RUN_ENGINE_CORE_WRAPPER_ATTR = (
    "_vllm_ascend_dsa_run_engine_core_wrapper")


def _is_dsa_enabled_on_config(vllm_config) -> bool:
    if vllm_config is None:
        return False
    attach_dsa_sparse_cache_attrs(vllm_config)
    return is_dsa_sparse_config_enabled(vllm_config)


def _get_manager_vllm_config(args, kwargs):
    if "vllm_config" in kwargs:
        return kwargs["vllm_config"]
    if len(args) >= 4:
        return args[3]
    return None


def _get_launch_core_vllm_config(args, kwargs):
    if "vllm_config" in kwargs:
        return kwargs["vllm_config"]
    if args:
        return args[0]
    return None


def _get_make_client_vllm_config(args, kwargs):
    if "vllm_config" in kwargs:
        return kwargs["vllm_config"]
    if len(args) >= 3:
        return args[2]
    return None


def _install_dsa_runtime_patches() -> None:
    # A spawned EngineCore imports this module to resolve the process target
    # before vllm-ascend's platform patches have necessarily been imported.
    # Some of those patches (notably patch_balance_schedule) capture and
    # replace EngineCoreProc.run_engine_core at import time.  Do not let them
    # capture the DSA wrapper itself: wrapping that platform entrypoint again
    # would otherwise create DSA -> platform -> DSA recursion.
    current_run_engine_core = EngineCoreProc.run_engine_core
    if is_dsa_run_engine_core_wrapper(current_run_engine_core):
        original_run_engine_core = getattr(
            EngineCoreProc,
            "_dsa_sparse_original_run_engine_core",
            None,
        )
        if original_run_engine_core is None:
            raise RuntimeError(
                "DSA EngineCore entrypoint wrapper has no original callable"
            )
        EngineCoreProc.run_engine_core = original_run_engine_core

    try:
        from vllm_ascend.patch.dsa_sparse.patch_runtime import (
            install_dsa_runtime_patches,
        )

        install_dsa_runtime_patches()
    finally:
        # Platform patch imports may have replaced the class entrypoint.  Keep
        # DSA outermost so the child installs its KV-cache aliases before the
        # selected platform entrypoint constructs EngineCore.
        ensure_dsa_engine_core_entrypoint()


def is_dsa_run_engine_core_wrapper(fn) -> bool:
    return bool(getattr(fn, _DSA_RUN_ENGINE_CORE_WRAPPER_ATTR, False))


def verify_dsa_runtime_patches_installed() -> None:
    from vllm.v1.core import kv_cache_utils as kv_utils
    from vllm.v1.engine.core_client import EngineCoreClient
    import vllm.v1.engine.core as engine_core_mod
    from vllm.v1.engine.core import EngineCore
    from vllm_ascend.patch.dsa_sparse import patch_engine_core
    from vllm_ascend.patch.dsa_sparse import patch_kv_cache_utils as dsa_kv_utils
    from vllm_ascend.patch.dsa_sparse.patch_deepseek_v2 import (
        is_dsa_indexer_cache_spec_patch_installed,
    )

    checks = {
        "kv_cache_utils_patched": bool(
            getattr(kv_utils, "_dsa_kv_cache_utils_patched", False)),
        "kv_cache_utils_get_configs_is_dsa_wrapper":
        dsa_kv_utils.is_dsa_get_kv_cache_configs_wrapper(
            kv_utils.get_kv_cache_configs),
        "engine_core_get_configs_is_dsa_wrapper":
        dsa_kv_utils.is_dsa_get_kv_cache_configs_wrapper(
            engine_core_mod.get_kv_cache_configs),
        "engine_core_get_kv_cache_configs_alias":
        engine_core_mod.get_kv_cache_configs is kv_utils.get_kv_cache_configs,
        "engine_core_init_patched": bool(
            getattr(EngineCore, "_dsa_sparse_engine_core_init_patched",
                    False)),
        "engine_core_init_is_dsa_wrapper":
        patch_engine_core.is_dsa_engine_core_init_wrapper(EngineCore.__init__),
        "engine_core_initialize_kv_caches_patched": bool(
            getattr(EngineCore,
                    "_dsa_sparse_initialize_kv_caches_patched", False)),
        "engine_core_initialize_kv_caches_is_dsa_wrapper":
        patch_engine_core.is_dsa_initialize_kv_caches_wrapper(
            EngineCore._initialize_kv_caches),
        "engine_core_proc_entrypoint_patched": bool(
            getattr(EngineCoreProc,
                    "_dsa_sparse_run_engine_core_patched", False)),
        "engine_core_proc_entrypoint_is_dsa_wrapper":
        is_dsa_run_engine_core_wrapper(EngineCoreProc.run_engine_core),
        "engine_core_make_client_patched": bool(
            getattr(EngineCoreClient, "_dsa_sparse_make_client_patched",
                    False)),
        "dsa_indexer_spec_patch_installed": (
            is_dsa_indexer_cache_spec_patch_installed()),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "DSA sparse-cache runtime patches are incomplete in the "
            f"EngineCore process: {checks}. "
            f"kv_utils.get_kv_cache_configs="
            f"{dsa_kv_utils.describe_callable(kv_utils.get_kv_cache_configs)}; "
            f"engine_core.get_kv_cache_configs="
            f"{dsa_kv_utils.describe_callable(engine_core_mod.get_kv_cache_configs)}; "
            f"EngineCore.__init__="
            f"{dsa_kv_utils.describe_callable(EngineCore.__init__)}; "
            f"EngineCore._initialize_kv_caches="
            f"{dsa_kv_utils.describe_callable(EngineCore._initialize_kv_caches)}; "
            f"EngineCoreProc.run_engine_core="
            f"{dsa_kv_utils.describe_callable(EngineCoreProc.run_engine_core)}")


def _prepare_dsa_engine_bootstrap(vllm_config) -> bool:
    if vllm_config is not None:
        attach_dsa_sparse_cache_attrs(vllm_config)
    if not _is_dsa_enabled_on_config(vllm_config):
        return False

    _install_dsa_runtime_patches()
    ensure_dsa_engine_core_entrypoint()
    verify_dsa_runtime_patches_installed()
    return True


def _reattach_dsa_config_from_additional_config(kwargs) -> None:
    vllm_config = kwargs.get("vllm_config")
    if vllm_config is not None:
        attach_dsa_sparse_cache_attrs(vllm_config)


def _dsa_sparse_run_engine_core(*args, **kwargs):
    # EngineCoreProc is spawned as a fresh interpreter.  Platform patches
    # installed in the parent process are not present there, so the DSA
    # scheduler/KV-cache aliases must be re-installed before EngineCore is
    # constructed and starts generating KV-cache configs.
    _reattach_dsa_config_from_additional_config(kwargs)
    if _is_dsa_enabled_on_config(kwargs.get("vllm_config")):
        _install_dsa_runtime_patches()
        verify_dsa_runtime_patches_installed()
    original_run_engine_core = EngineCoreProc._dsa_sparse_original_run_engine_core
    return original_run_engine_core(*args, **kwargs)


setattr(_dsa_sparse_run_engine_core, _DSA_RUN_ENGINE_CORE_WRAPPER_ATTR, True)


def ensure_dsa_engine_core_entrypoint() -> None:
    current_run_engine_core = EngineCoreProc.run_engine_core
    if is_dsa_run_engine_core_wrapper(current_run_engine_core):
        return

    EngineCoreProc._dsa_sparse_original_run_engine_core = (
        current_run_engine_core)
    EngineCoreProc.run_engine_core = _dsa_sparse_run_engine_core
    EngineCoreProc._dsa_sparse_run_engine_core_patched = True


ensure_dsa_engine_core_entrypoint()


if not getattr(EngineCoreClient, "_dsa_sparse_make_client_patched", False):
    _original_make_client = EngineCoreClient.make_client

    def _dsa_sparse_make_client(*args, **kwargs):
        # This is the first common vLLM boundary after VllmConfig exists and
        # before any multiprocessing EngineCore target is selected.  Install
        # the DSA runtime patches here so later launch-path aliases cannot
        # silently bypass the split KV-cache hooks.
        _prepare_dsa_engine_bootstrap(_get_make_client_vllm_config(args, kwargs))
        return _original_make_client(*args, **kwargs)

    EngineCoreClient.make_client = staticmethod(_dsa_sparse_make_client)
    EngineCoreClient._dsa_sparse_make_client_patched = True

if not getattr(engine_utils.CoreEngineProcManager,
               "_dsa_sparse_engine_proc_manager_init_patched", False):
    _original_core_engine_proc_manager_init = (
        engine_utils.CoreEngineProcManager.__init__)

    def _dsa_sparse_core_engine_proc_manager_init(self, *args, **kwargs):
        # Other platform patches may also wrap EngineCoreProc.run_engine_core.
        # Re-check immediately before vLLM creates the subprocess target so
        # the child enters through the DSA bootstrap wrapper.
        _prepare_dsa_engine_bootstrap(_get_manager_vllm_config(args, kwargs))
        return _original_core_engine_proc_manager_init(self, *args, **kwargs)

    engine_utils.CoreEngineProcManager.__init__ = (
        _dsa_sparse_core_engine_proc_manager_init)
    engine_utils.CoreEngineProcManager._dsa_sparse_engine_proc_manager_init_patched = True


if not getattr(engine_utils, "_dsa_sparse_launch_core_engines_patched", False):
    _original_launch_core_engines = engine_utils.launch_core_engines

    def _dsa_sparse_launch_core_engines(*args, **kwargs):
        # This is the call site used by MPClient.  Patch it directly so DSA
        # does not depend on whether vLLM imported launch_core_engines before
        # or after vllm-ascend's platform patches.
        _prepare_dsa_engine_bootstrap(_get_launch_core_vllm_config(
            args, kwargs))
        return _original_launch_core_engines(*args, **kwargs)

    engine_utils.launch_core_engines = _dsa_sparse_launch_core_engines
    core_client_mod.launch_core_engines = _dsa_sparse_launch_core_engines
    engine_utils._dsa_sparse_launch_core_engines_patched = True
