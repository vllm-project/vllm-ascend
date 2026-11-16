# SPDX-License-Identifier: Apache-2.0

"""Tests for the backend-agnostic CPTopology derivation and common validation."""

from types import SimpleNamespace

import pytest

from vllm_ascend.attention.context_parallel.common_cp import is_pcp_decode_sharding_enabled
from vllm_ascend.attention.context_parallel.topology import (
    CPTopology,
    DCPGroupAxis,
    resolve_cp_topology,
    validate_common_topology,
)


def test_default_topology_is_disabled() -> None:
    topology = CPTopology(tp_size=2, pcp_size=1, dcp_size=1)
    assert topology.is_pcp_enabled is False
    assert topology.is_dcp_enabled is False
    assert topology.dcp_axis is DCPGroupAxis.DISABLED
    assert topology.dcp_reuses_tp_ranks is False


def test_standalone_dcp_uses_tp_axis() -> None:
    topology = CPTopology(tp_size=2, pcp_size=1, dcp_size=2)
    assert topology.is_dcp_enabled is True
    assert topology.dcp_axis is DCPGroupAxis.TP
    assert topology.dcp_reuses_tp_ranks is True


def test_stacked_dcp_equal_pcp_uses_pcp_axis() -> None:
    topology = CPTopology(tp_size=2, pcp_size=2, dcp_size=2)
    assert topology.dcp_axis is DCPGroupAxis.PCP
    assert topology.dcp_reuses_tp_ranks is False


def test_full_grid_dcp_equals_tp_times_pcp() -> None:
    topology = CPTopology(tp_size=2, pcp_size=2, dcp_size=4)
    assert topology.dcp_axis is DCPGroupAxis.TP_PCP
    assert topology.dcp_reuses_tp_ranks is True


def test_logical_dcp_rank_order_pcp_major() -> None:
    full_grid = CPTopology(tp_size=2, pcp_size=2, dcp_size=4)
    assert full_grid.pcp_face_ranks() == (0, 2)
    # The PCP coordinate varies before the TP coordinate: (pcp=0,tp=0),
    # (pcp=1,tp=0), (pcp=0,tp=1), (pcp=1,tp=1).
    assert full_grid.dcp_logical_rank_order() == (0, 2, 1, 3)
    assert CPTopology(tp_size=1, pcp_size=4, dcp_size=4).dcp_logical_rank_order() == (0, 1, 2, 3)


def test_invalid_dcp_size_rejected() -> None:
    topology = CPTopology(tp_size=2, pcp_size=2, dcp_size=3)
    with pytest.raises(ValueError, match="DCP=3"):
        _ = topology.dcp_axis


def test_interleave_must_divide_block_size() -> None:
    cache_config = SimpleNamespace(block_size=128)
    invalid = CPTopology(tp_size=2, pcp_size=1, dcp_size=2, interleave_size=96)
    with pytest.raises(
        ValueError,
        match=r"cp_kv_cache_interleave_size\(96\) must divide the cache block_size\(128\)",
    ):
        validate_common_topology(invalid, cache_config=cache_config)

    validate_common_topology(
        CPTopology(tp_size=2, pcp_size=1, dcp_size=2, interleave_size=64), cache_config=cache_config
    )
    # The interleave requirement only applies when decode context parallel is on.
    validate_common_topology(
        CPTopology(tp_size=2, pcp_size=1, dcp_size=1, interleave_size=96), cache_config=cache_config
    )


def test_gqa_stacked_requires_equal_sizes() -> None:
    gqa_config = SimpleNamespace(use_mla=False)
    unequal = CPTopology(tp_size=1, pcp_size=2, dcp_size=4)
    with pytest.raises(ValueError, match=r"PCP\+DCP.*PCP=2, DCP=4"):
        validate_common_topology(unequal, gqa_config)

    validate_common_topology(CPTopology(tp_size=1, pcp_size=2, dcp_size=2), gqa_config)
    # MLA models keep the full TP x PCP grid decodes allowed.
    validate_common_topology(unequal, SimpleNamespace(use_mla=True))


def test_resolve_topology_reads_parallel_config() -> None:
    parallel_config = SimpleNamespace(
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=4,
        cp_kv_cache_interleave_size=128,
        pipeline_parallel_size=1,
    )
    topology = resolve_cp_topology(SimpleNamespace(parallel_config=parallel_config))
    assert topology == CPTopology(
        tp_size=2,
        pcp_size=2,
        dcp_size=4,
        interleave_size=128,
        pipeline_parallel_size=1,
    )

    # Partial configs (tests and draft helpers) fall back to 1 for the optional
    # fields; the PCP/DCP sizes are always read from the config.
    partial_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2, decode_context_parallel_size=2)
    )
    partial_topology = resolve_cp_topology(partial_config)
    assert (partial_topology.tp_size, partial_topology.interleave_size) == (1, 1)
    assert partial_topology.pipeline_parallel_size == 1


def test_pcp_only_mrv2_message_preserved() -> None:
    topology = CPTopology(tp_size=1, pcp_size=2, dcp_size=1)
    with pytest.raises(ValueError, match="Prefill Context Parallel"):
        validate_common_topology(topology, vllm_config=SimpleNamespace(use_v2_model_runner=False))
    validate_common_topology(topology, vllm_config=SimpleNamespace(use_v2_model_runner=True))


def test_parallel_config_fields_roundtrip() -> None:
    topology = CPTopology(tp_size=2, pcp_size=2, dcp_size=4, interleave_size=128, pipeline_parallel_size=2)
    fields = topology.to_parallel_config_fields()
    assert fields == {
        "tensor_parallel_size": 2,
        "prefill_context_parallel_size": 2,
        "decode_context_parallel_size": 4,
        "cp_kv_cache_interleave_size": 128,
        "pipeline_parallel_size": 2,
    }
    recreated = CPTopology(
        tp_size=fields["tensor_parallel_size"],
        pcp_size=fields["prefill_context_parallel_size"],
        dcp_size=fields["decode_context_parallel_size"],
        interleave_size=fields["cp_kv_cache_interleave_size"],
        pipeline_parallel_size=fields["pipeline_parallel_size"],
    )
    assert recreated == topology


@pytest.mark.parametrize(("pcp_size", "dcp_size"), [(2, 1), (1, 1), (2, 2), (1, 2)])
def test_is_pcp_decode_sharding_enabled_matches_parallel_fields(pcp_size: int, dcp_size: int) -> None:
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
        ),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
    )
    assert is_pcp_decode_sharding_enabled(config) is (pcp_size > 1 and dcp_size == 1)

    # Read-only KPool models keep replicated decodes even when PCP is enabled.
    kpool_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
        ),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_kpool=4)),
    )
    assert is_pcp_decode_sharding_enabled(kpool_config) is False
