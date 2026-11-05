# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Exercise the platform patch with synthetic configs, without model allocation."""

import importlib.util
import runpy
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def shard_selector(monkeypatch):
    config = SimpleNamespace(
        engram_config=SimpleNamespace(dp_shared_memory=False),
        model_config=SimpleNamespace(
            architecture="DeepseekV41ForCausalLM",
            hf_text_config=SimpleNamespace(engram_n_heads=8, engram_max_ngram_size=4),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    original = Mock(return_value=16)
    config_module = ModuleType("vllm.config")
    config_module.get_current_vllm_config = lambda: config
    engram_module = ModuleType("vllm.config.engram")
    engram_module.EngramConfig = type("EngramConfig", (), {})
    engram_module.model_has_engram_layers = Mock(return_value=True)
    distributed = ModuleType("vllm.distributed")
    distributed.parallel_state = SimpleNamespace(_engram_dp_shard_size=original)
    # Load only this patch, avoiding platform registration and accelerator imports.
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "vllm.config", config_module)
        imports.setitem(sys.modules, "vllm.config.engram", engram_module)
        imports.setitem(sys.modules, "vllm.distributed", distributed)
        imports.setattr(importlib.util, "find_spec", lambda _: ModuleSpec("vllm.config.engram", loader=None))
        root = Path(__file__).resolve().parents[4]
        runpy.run_path(str(root / "vllm_ascend/patch/platform/patch_engram_config.py"))
    return config, original, distributed.parallel_state._engram_dp_shard_size


@pytest.mark.parametrize(
    "heads,ngram,tp,candidate,expected",
    [
        (8, 4, 1, 16, 8),  # Regression: 24 heads cannot be split over EDP16.
        (8, 4, 2, 8, 4),
        (8, 4, 4, 4, 2),
        (8, 4, 8, 2, 1),
        (8, 4, 1, 8, 8),  # Already compatible.
        (8, 4, 1, 1, 1),  # Upstream node-boundary fallback.
        (8, 4, 2, 3, 3),  # Non-power-of-two candidates remain supported.
        (8, 5, 1, 16, 16),  # Derive the head count instead of hardcoding 24.
        (5, 4, 1, 8, 1),  # No compatible multi-rank group.
        (6, 4, 2, 8, 1),
    ],
)
def test_nonshared_group_uses_largest_head_compatible_divisor(shard_selector, heads, ngram, tp, candidate, expected):
    config, original, select = shard_selector
    text = config.model_config.hf_text_config
    text.engram_n_heads, text.engram_max_ngram_size = heads, ngram
    config.parallel_config.tensor_parallel_size = tp
    original.return_value = candidate
    assert select(32, 16, tp) == expected
    original.assert_called_once_with(32, 16, tp)
    # Every compatible divisor of the upstream candidate is at most the result.
    compatible = [n for n in range(1, candidate + 1) if candidate % n == 0 and heads * (ngram - 1) % (tp * n) == 0]
    assert expected == max(compatible)


@pytest.mark.parametrize("shared", [False, True])
def test_pcp_replica_size_is_not_mistaken_for_tp(shard_selector, shared):
    config, original, select = shard_selector
    config.parallel_config.tensor_parallel_size = 2
    config.engram_config.dp_shared_memory = shared
    # Upstream DP includes PCP in replica_size; Ascend DP x PCP does not.
    # In either case only TP participates in the uniform head-shard constraint.
    original.return_value = 8
    assert select(32, 8, 4) == (8 if shared else 4)
    assert select(32, 16, 2) == (8 if shared else 4)


@pytest.mark.parametrize("candidate", [1, 8, 16])
def test_shared_table_group_is_not_head_sharded(shard_selector, candidate):
    config, original, select = shard_selector
    config.engram_config.dp_shared_memory = True
    original.return_value = candidate
    assert select(16, 16, 1) == candidate


@pytest.mark.parametrize("model", [None, SimpleNamespace(architecture="OtherModel")])
def test_unrelated_models_do_not_require_engram_head_fields(shard_selector, model):
    config, original, select = shard_selector
    config.model_config = model
    assert select(16, 16, 1) == 16


def test_disabled_engram_leaves_upstream_selection_unchanged(shard_selector):
    config, original, select = shard_selector
    config.engram_config = None
    config.model_config = None
    assert select(16, 16, 1) == 16


@pytest.mark.parametrize("heads,ngram,tp", [(0, 4, 1), (8, 1, 1), (-8, 4, 1), (5, 4, 2)])
def test_invalid_heads_or_tp_cannot_be_repaired_by_reducing_edp(shard_selector, heads, ngram, tp):
    config, _, select = shard_selector
    text = config.model_config.hf_text_config
    text.engram_n_heads, text.engram_max_ngram_size = heads, ngram
    config.parallel_config.tensor_parallel_size = tp
    with pytest.raises(ValueError, match="positive hash-head count divisible by TP"):
        select(16, 16, tp)


def test_upstream_failure_is_not_hidden(shard_selector):
    _, original, select = shard_selector
    original.side_effect = RuntimeError("node discovery failed")
    with pytest.raises(RuntimeError, match="node discovery failed"):
        select(16, 16, 1)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_native_node_partition_is_retained(shard_selector, monkeypatch, shared, contiguous):
    pytest.importorskip("vllm")
    from vllm.distributed import parallel_state

    from vllm_ascend.patch.platform import patch_engram_config

    config, original, select = shard_selector
    original.side_effect = patch_engram_config._original_engram_dp_shard_size
    config.engram_config.dp_shared_memory = shared
    monkeypatch.setattr(parallel_state, "get_node_count", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: SimpleNamespace(cpu_group=object()))
    monkeypatch.setattr(
        parallel_state,
        "in_the_same_node_as",
        lambda _, start: [rank // 16 == start // 16 if contiguous else rank % 2 == start % 2 for rank in range(32)],
    )
    assert select(32, 32, 1) == ((16 if shared else 8) if contiguous else 1)
