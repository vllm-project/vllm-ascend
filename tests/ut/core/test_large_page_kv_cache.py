from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import uses_large_page_allocator
from vllm_ascend.patch.platform import patch_kv_cache_utils
from vllm_ascend.platform import NPUPlatform


def _large_page_config() -> SimpleNamespace:
    return SimpleNamespace(cache_config=SimpleNamespace(kv_cache_allocator="large_page"))


def _parallel_config(**overrides) -> SimpleNamespace:
    values = {
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_absent_allocator_field_reads_as_uniform_page() -> None:
    assert not uses_large_page_allocator(SimpleNamespace())
    assert not uses_large_page_allocator(SimpleNamespace(kv_cache_allocator="uniform_page"))
    assert uses_large_page_allocator(SimpleNamespace(kv_cache_allocator="large_page"))


def test_large_page_grouping_delegates_to_upstream(monkeypatch) -> None:
    vllm_config = _large_page_config()
    specs: dict[str, KVCacheSpec] = {}
    expected: list = [Mock()]
    upstream_grouping = Mock(return_value=expected)
    monkeypatch.setattr(patch_kv_cache_utils, "_orig_get_kv_cache_groups", upstream_grouping)
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "get_qwen4_exp_kv_cache_groups",
        Mock(side_effect=AssertionError("Ascend grouping must not run under large_page")),
    )

    groups = patch_kv_cache_utils._get_ascend_kv_cache_groups(vllm_config, specs)

    assert groups is expected
    upstream_grouping.assert_called_once_with(vllm_config, specs)


def test_large_page_block_sizes_delegate_to_upstream(monkeypatch) -> None:
    vllm_config = _large_page_config()
    kv_cache_config = SimpleNamespace()
    upstream_resolve = Mock(return_value=(64, 64))
    monkeypatch.setattr(
        patch_kv_cache_utils,
        "_orig_resolve_kv_cache_block_sizes",
        upstream_resolve,
    )

    resolved = patch_kv_cache_utils._ascend_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)

    assert resolved == (64, 64)
    upstream_resolve.assert_called_once_with(kv_cache_config, vllm_config)


@pytest.mark.parametrize(
    ("parallel_overrides", "kv_transfer_config", "expected"),
    [
        ({"decode_context_parallel_size": 2}, None, "decode context parallelism"),
        ({"prefill_context_parallel_size": 2}, None, "prefill context parallelism"),
        ({}, Mock(), "KV transfer"),
    ],
)
def test_rejects_unsupported_large_page_combinations(
    parallel_overrides: dict[str, int],
    kv_transfer_config: Mock | None,
    expected: str,
) -> None:
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_allocator="large_page"),
        parallel_config=_parallel_config(**parallel_overrides),
        kv_transfer_config=kv_transfer_config,
    )

    with pytest.raises(ValueError, match=expected):
        NPUPlatform._reject_large_page_combinations(vllm_config)


def test_accepts_plain_large_page_run() -> None:
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_allocator="large_page"),
        parallel_config=_parallel_config(),
        kv_transfer_config=None,
    )

    NPUPlatform._reject_large_page_combinations(vllm_config)


def test_uniform_page_keeps_unsupported_combinations() -> None:
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(kv_cache_allocator="uniform_page"),
        parallel_config=_parallel_config(decode_context_parallel_size=2),
        kv_transfer_config=Mock(),
    )

    NPUPlatform._reject_large_page_combinations(vllm_config)
