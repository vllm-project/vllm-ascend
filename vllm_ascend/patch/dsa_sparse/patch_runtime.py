"""Install DSA sparse-cache patches that must exist in every process.

This module is loaded through both the Ascend platform patch path and vLLM's
``vllm.general_plugins`` path.  Do not rely only on import side effects here:
some vllm-ascend patches wrap the same vLLM functions later, and Python module
imports are cached.  The explicit installer below re-asserts the current
callable objects each time a stable process-local bootstrap point is reached.
"""


def install_dsa_runtime_patches() -> None:
    # Import order is semantic: interface/output dataclasses must exist before
    # the model and scheduler wrappers consume them.
    # isort: off
    # v0.23 owns Ascend KV specs in ``vllm_ascend.core``; the legacy
    # patch_kv_cache_interface module no longer exists.
    import vllm_ascend.core.kv_cache_interface  # noqa: F401,E402
    import vllm_ascend.patch.dsa_sparse.patch_scheduler_output  # noqa: F401,E402
    import vllm_ascend.patch.dsa_sparse.patch_kv_cache_decoupling  # noqa: F401,E402
    import vllm_ascend.patch.dsa_sparse.patch_deepseek_v2  # noqa: F401,E402
    import vllm_ascend.patch.dsa_sparse.patch_request  # noqa: F401,E402
    import vllm_ascend.patch.dsa_sparse.patch_scheduler  # noqa: F401,E402

    from vllm_ascend.patch.dsa_sparse import patch_engine_core
    from vllm_ascend.patch.dsa_sparse import patch_kv_cache_utils
    # isort: on

    patch_kv_cache_utils.install_dsa_kv_cache_utils_patch()
    patch_engine_core.install_dsa_engine_core_patches()


install_dsa_runtime_patches()
