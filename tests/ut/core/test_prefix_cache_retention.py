# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

# Load the helper directly so this check does not import vLLM.
_MODULE_PATH = Path(__file__).resolve().parents[3] / "vllm_ascend" / "core" / "prefix_cache_retention.py"
_SPEC = importlib.util.spec_from_file_location("prefix_cache_retention_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_retention = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_retention)


def test_next_retention_interval_rounds_hybrid_mamba_block_size():
    # Qwen3.8-27B align mode: CLI block size 32 becomes scheduler block size 1536.
    assert _retention.next_retention_interval(4096, 1536) == 4608
    assert _retention.next_retention_interval(4608, 1536) == 4608
    assert _retention.next_retention_interval(0, 1536) == 0
    assert _retention.next_retention_interval(-8, 1536) == 0


def test_retention_interval_error_names_effective_block_size_and_next_value():
    groups = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=32)),
            SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=1536)),
        ]
    )

    def reject_unaligned(retention_interval, scheduler_block_size, kv_cache_config):
        del kv_cache_config
        raise ValueError(
            f"prefix_cache_retention_interval ({retention_interval}) "
            "must be non-negative and a multiple of scheduler_block_size "
            f"({scheduler_block_size})."
        )

    with pytest.raises(ValueError, match="4608") as exc_info:
        _retention.validate_retention_interval(4096, 1536, groups, reject_unaligned)
    message = str(exc_info.value)
    assert "scheduler_block_size is 1536" in message
    assert "[32, 1536]" in message
    assert "VLLM_PREFIX_CACHE_RETENTION_INTERVAL" in message


def test_retention_interval_error_keeps_unrelated_validation_failures():
    def reject_unused(retention_interval, scheduler_block_size, kv_cache_config):
        del retention_interval, scheduler_block_size, kv_cache_config
        raise ValueError("prefix_cache_retention_interval is set but this model has no Mamba group")

    with pytest.raises(ValueError, match="no Mamba group"):
        _retention.validate_retention_interval(4096, 1536, SimpleNamespace(kv_cache_groups=[]), reject_unused)
