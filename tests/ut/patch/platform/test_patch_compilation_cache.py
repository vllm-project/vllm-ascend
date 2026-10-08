import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.patch.platform.patch_compilation_cache import (
    _compiler_manager_initialize_cache,
    _device_isolated_cache_path,
    _support_torch_compile,
    _try_load_aot_compiled_fn,
)


@contextmanager
def _patch_npu_device(device_index: int):
    with (
        patch(
            "vllm_ascend.patch.platform.patch_compilation_cache.torch.npu.is_available",
            return_value=True,
        ),
        patch(
            "vllm_ascend.patch.platform.patch_compilation_cache.torch.npu.current_device",
            return_value=device_index,
        ),
    ):
        yield


def test_device_isolated_cache_path_adds_npu_index():
    path = os.path.join(
        "/root/.cache/vllm",
        "torch_compile_cache",
        "torch_aot_compile",
        "hash",
        "rank_1_7",
    )

    with _patch_npu_device(15):
        isolated_path = _device_isolated_cache_path(path)

    assert isolated_path == os.path.join(
        "/root/.cache/vllm",
        "torch_compile_cache",
        "torch_aot_compile",
        "hash",
        "rank_1_7",
        "dev15",
    )


def test_device_isolated_cache_path_keeps_prefix_after_device():
    path = os.path.join(
        "/root/.cache/vllm",
        "torch_compile_cache",
        "hash",
        "rank_1_7",
        "language_model",
    )

    with _patch_npu_device(5):
        isolated_path = _device_isolated_cache_path(path)

    assert isolated_path.endswith(os.path.join("rank_1_7", "dev5", "language_model"))


def test_device_isolated_cache_path_replaces_stale_device_directory():
    path = os.path.join(
        "/root/.cache/vllm",
        "torch_compile_cache",
        "torch_aot_compile",
        "hash",
        "rank_1_7",
        "dev15",
        "model",
    )
    isolated_path = os.path.join(
        "/root/.cache/vllm",
        "torch_compile_cache",
        "torch_aot_compile",
        "hash",
        "rank_1_7",
        "dev5",
        "model",
    )

    with _patch_npu_device(5):
        assert _device_isolated_cache_path(path) == isolated_path


def test_device_isolated_cache_path_is_idempotent():
    path = os.path.join(
        "/root/.cache/vllm",
        "torch_compile_cache",
        "torch_aot_compile",
        "hash",
        "rank_1_7",
        "dev5",
        "model",
    )

    with _patch_npu_device(5):
        assert _device_isolated_cache_path(path) == path


def test_device_isolated_cache_path_does_not_modify_other_paths():
    path = os.path.join("/tmp", "rank_1_7")

    with _patch_npu_device(5):
        assert _device_isolated_cache_path(path) == path


def test_device_isolated_cache_path_ignores_missing_device():
    path = os.path.join(
        "/root/.cache/vllm",
        "torch_compile_cache",
        "torch_aot_compile",
        "hash",
        "rank_1_7",
    )

    with (
        patch(
            "vllm_ascend.patch.platform.patch_compilation_cache.torch.npu.is_available",
            return_value=False,
        ),
        patch(
            "vllm_ascend.patch.platform.patch_compilation_cache.torch.npu.current_device",
            side_effect=RuntimeError("NPU is not initialized"),
        ),
    ):
        assert _device_isolated_cache_path(path) == path


def test_aot_load_uses_device_isolated_path():
    from vllm_ascend.patch.platform import patch_compilation_cache as patch_module

    path = os.path.join(
        "/root/.cache/vllm",
        "torch_compile_cache",
        "torch_aot_compile",
        "hash",
        "rank_1_7",
    )
    model = object()

    with (
        _patch_npu_device(15),
        patch.object(patch_module, "_original_try_load_aot_compiled_fn") as load,
    ):
        _try_load_aot_compiled_fn(model, path)

    load.assert_called_once_with(
        model,
        os.path.join(path, "dev15"),
    )


def test_aot_save_uses_device_isolated_path():
    from vllm_ascend.patch.platform import patch_compilation_cache as patch_module

    saved_paths = []

    class CompiledModel:
        def save_aot_compiled_function(self):
            saved_paths.append((self._aot_compilation_path, self._aot_cache_dir))

    path = os.path.join(
        "/root/.cache/vllm",
        "torch_compile_cache",
        "torch_aot_compile",
        "hash",
        "rank_1_7",
    )

    with (
        _patch_npu_device(15),
        patch.object(patch_module, "_original_support_torch_compile", return_value=CompiledModel),
    ):
        patched_cls = _support_torch_compile(CompiledModel)

        instance = patched_cls()
        instance._aot_compilation_path = os.path.join(path, "model")
        instance._aot_cache_dir = path
        instance.save_aot_compiled_function()

    assert saved_paths == [
        (
            os.path.join(path, "dev15", "model"),
            os.path.join(path, "dev15"),
        )
    ]


def test_compiler_manager_cache_uses_device_isolated_path():
    from vllm_ascend.patch.platform import patch_compilation_cache as patch_module

    path = os.path.join(
        "/root/.cache/vllm",
        "torch_compile_cache",
        "hash",
        "rank_1_7",
        "language_model",
    )
    isolated_path = os.path.join(
        "/root/.cache/vllm",
        "torch_compile_cache",
        "hash",
        "rank_1_7",
        "dev5",
        "language_model",
    )
    manager = SimpleNamespace(
        compilation_config=SimpleNamespace(local_cache_dir=path),
    )

    with (
        _patch_npu_device(5),
        patch.object(patch_module, "_original_compiler_manager_initialize_cache") as initialize,
    ):
        _compiler_manager_initialize_cache(
            manager,
            cache_dir=path,
            disable_cache=True,
            prefix="language_model",
        )

    assert manager.compilation_config.local_cache_dir == isolated_path
    initialize.assert_called_once_with(
        manager,
        cache_dir=isolated_path,
        disable_cache=True,
        prefix="language_model",
    )
