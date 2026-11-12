# SPDX-License-Identifier: Apache-2.0
"""Retired Host flags must fail before allocation or singleton config reuse."""

from types import SimpleNamespace

import pytest

from vllm_ascend import ascend_config


@pytest.mark.parametrize(
    "legacy",
    [
        {"enable_cpu_kv_store": True},
        {"tp_shared_host_kv": True},
        {"mode": "fused_overlap", "enable_cpu_kv_store": True},
        {"mode": "baseline"},
        {"selection_topk_block_size": 1},
        {"shared_host_store_port": 8650},
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_retired_config_rejected_even_without_manager_offload(legacy, enabled):
    cfg = SimpleNamespace(additional_config={"dsa_sparse_attention_config": legacy})
    with pytest.raises(ValueError, match="removed.*sparse_kv_offload_config"):
        ascend_config.SparseKVOffloadConfig.from_additional_config(cfg, {"enabled": enabled})


def test_retired_config_cannot_bypass_validation_via_cached_singleton(monkeypatch):
    cfg = SimpleNamespace(additional_config={"dsa_sparse_attention_config": {"enable_cpu_kv_store": True}})
    monkeypatch.setattr(ascend_config, "_INIT_VLLM_CONFIG", cfg)
    monkeypatch.setattr(
        ascend_config, "_ASCEND_CONFIG", SimpleNamespace(ascend_compilation_config=object(), eplb_config=object())
    )
    with pytest.raises(ValueError, match="removed"):
        ascend_config.init_ascend_config(cfg)


@pytest.mark.parametrize("legacy", [None, {}])
def test_empty_retired_section_does_not_change_default_device_configuration(legacy):
    cfg = SimpleNamespace(additional_config={"dsa_sparse_attention_config": legacy}, speculative_config=None)
    resolved = ascend_config.SparseKVOffloadConfig.from_additional_config(cfg, {})
    assert not resolved.enabled


@pytest.mark.parametrize("legacy", [True, [], "fused_overlap"])
def test_malformed_retired_section_is_not_silently_ignored(legacy):
    cfg = SimpleNamespace(additional_config={"dsa_sparse_attention_config": legacy})
    with pytest.raises(TypeError, match="must be a dict"):
        ascend_config.SparseKVOffloadConfig.from_additional_config(cfg, {})
