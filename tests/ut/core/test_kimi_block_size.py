# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.kv_cache_interface import MambaSpec

from tests.ut.core.test_packed_sfa_cache import make_packed_specs, packed_config
from vllm_ascend.models.kimi_k3 import block_size as sizing


def search_config():
    config = packed_config()
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.cp_kv_cache_interleave_size = 128
    config.cache_config.block_size = 128
    config.cache_config.user_specified_block_size = False
    config.cache_config.mamba_block_size = 128
    config.cache_config.mamba_page_size_padded = None
    config.kv_transfer_config = None
    return config


@pytest.mark.parametrize("draft_layers,steps,expected", [(0, 0, 8192), (3, 3, 11776), (5, 3, 10752)])
def test_auto_block_capacity_regressions(draft_layers, steps, expected):
    config = search_config()
    specs = make_packed_specs(draft_layers=draft_layers, speculative_steps=steps)
    assert sizing.select_packed_sfa_block_size(config, [specs], 1) == expected
    assert config.cache_config.block_size == 128
    assert all(spec.block_size == 10752 for spec in specs.values())


def test_pp_uses_global_groups_even_for_recurrent_only_stage():
    config = search_config()
    specs = make_packed_specs()
    recurrent = {name: spec for name, spec in specs.items() if isinstance(spec, MambaSpec)}
    history = {name: spec for name, spec in specs.items() if not isinstance(spec, MambaSpec)}
    assert sizing.select_packed_sfa_block_size(config, [recurrent, history], 1) == 10752
    assert sizing.select_packed_sfa_block_size(config, [history, recurrent], 1) == 10752


def test_auto_block_respects_checkpoint_prefix_and_cp_interleave():
    config = search_config()
    config.parallel_config.decode_context_parallel_size = 16
    config.parallel_config.cp_kv_cache_interleave_size = 768
    config.cache_config.prefix_match_unit = 256
    specs = make_packed_specs(state_heads=6)
    selected = sizing.select_packed_sfa_block_size(config, [specs], 1536)
    assert selected % 1536 == selected % 768 == selected % 256 == selected % 128 == 0


def test_auto_block_replayssm_consumes_actual_persistent_state_specs():
    config = search_config()
    specs = make_packed_specs()
    recovered = {
        name: replace(spec, num_speculative_blocks=0) if isinstance(spec, MambaSpec) else spec
        for name, spec in specs.items()
    }
    selected = sizing.select_packed_sfa_block_size(config, [recovered], 1)
    assert selected % 128 == 0
    assert all(spec.num_speculative_blocks == 0 for spec in recovered.values() if isinstance(spec, MambaSpec))


def test_auto_block_reports_unsatisfiable_alignment():
    with pytest.raises(ValueError, match="set --block-size explicitly"):
        sizing.select_packed_sfa_block_size(search_config(), [make_packed_specs()], 65537)


@pytest.mark.parametrize("interleave", [0, 64, 192])
def test_auto_block_rejects_interleave_smaller_than_native_pages(interleave):
    config = search_config()
    config.parallel_config.decode_context_parallel_size = 16
    config.parallel_config.cp_kv_cache_interleave_size = interleave
    with pytest.raises(ValueError, match="positive multiple of the 128-token kernel block"):
        sizing.select_packed_sfa_block_size(config, [make_packed_specs(state_heads=6)], 1)


def test_auto_block_preserves_explicit_value_without_inspecting_layers(monkeypatch):
    config = search_config()
    config.cache_config.user_specified_block_size = True
    config.cache_config.block_size = 1536
    config.kv_transfer_config = object()
    gather = Mock(side_effect=AssertionError("Explicit block sizes must not gather or search."))
    monkeypatch.setattr(sizing, "get_mamba_prefill_checkpoint_alignment", gather)
    sizing.auto_select_packed_sfa_block_size(config)
    assert config.cache_config.block_size == 1536
    gather.assert_not_called()


def test_auto_block_requires_explicit_size_for_unqualified_connector():
    config = search_config()
    config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV1")
    with pytest.raises(ValueError, match="set --block-size explicitly for other KV connectors"):
        sizing.auto_select_packed_sfa_block_size(config)


@pytest.mark.parametrize("connector", [None, "MooncakeConnectorV2", "MooncakePullConnector"])
def test_auto_block_updates_loaded_target_and_mamba_configs(monkeypatch, connector):
    config = search_config()
    if connector is not None:
        config.kv_transfer_config = SimpleNamespace(kv_connector=connector)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.worker.v2.attn_utils",
        SimpleNamespace(get_kv_cache_spec=lambda _: make_packed_specs()),
    )
    monkeypatch.setattr(sizing, "get_mamba_prefill_checkpoint_alignment", lambda _: 1536)
    sizing.auto_select_packed_sfa_block_size(config)
    assert config.cache_config.block_size == 10752
    assert config.cache_config.mamba_block_size == 10752
    assert not config.cache_config.user_specified_block_size


def test_auto_block_gathers_recurrent_only_pp_stage_over_cpu_group(monkeypatch):
    config = search_config()
    config.parallel_config.pipeline_parallel_size = 2
    specs = make_packed_specs()
    recurrent = {name: spec for name, spec in specs.items() if isinstance(spec, MambaSpec)}
    history = {name: spec for name, spec in specs.items() if not isinstance(spec, MambaSpec)}
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.worker.v2.attn_utils", SimpleNamespace(get_kv_cache_spec=lambda _: recurrent)
    )
    group = SimpleNamespace(world_size=2, cpu_group=object())
    monkeypatch.setattr(sizing, "get_pp_group", lambda: group)
    monkeypatch.setattr(sizing, "get_mamba_prefill_checkpoint_alignment", lambda _: 128)

    def gather(output, local, *, group):
        assert group is not None
        output[:] = [local, (history, 1536)]

    collect = Mock(side_effect=gather)
    monkeypatch.setattr(sizing.dist, "all_gather_object", collect)
    sizing.auto_select_packed_sfa_block_size(config)
    assert config.cache_config.block_size == 10752
    assert collect.call_args.kwargs["group"] is group.cpu_group
