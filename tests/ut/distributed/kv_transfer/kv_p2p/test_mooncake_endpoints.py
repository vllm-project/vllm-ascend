"""Golden tests for the shared mooncake v1 handshake port formulas.

The expected values are the legacy inline formulas, rewritten here as plain
literals on purpose: the assertions must not be derived from the code under
test. The pull and layerwise formula families are intentionally different; see
``mooncake_v1/common/endpoints.py`` for why they are not unified.
"""

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.endpoints import (
    layerwise_handshake_port,
    layerwise_scheduler_base_port,
    layerwise_worker_base_port,
    pull_base_port,
    pull_device_index,
    pull_handshake_port,
)

KV_PORT = 30000


@pytest.mark.parametrize("dp_index", [0, 1])
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("pcp_size", [1, 2])
def test_pull_base_port_matches_legacy(dp_index, tp_size, pp_size, pcp_size):
    legacy = KV_PORT + dp_index * tp_size * pp_size * pcp_size
    actual = pull_base_port(kv_port=KV_PORT, dp_index=dp_index, tp_size=tp_size, pp_size=pp_size, pcp_size=pcp_size)
    assert actual == legacy


@pytest.mark.parametrize("tp_rank,tp_size", [(0, 2), (1, 2), (0, 4), (3, 4)])
@pytest.mark.parametrize("pp_rank,pp_size", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("pcp_rank,pcp_size", [(0, 1), (0, 2), (1, 2)])
def test_pull_device_index_matches_legacy(tp_rank, tp_size, pp_rank, pp_size, pcp_rank, pcp_size):
    legacy = (pp_rank * pcp_size + pcp_rank) * tp_size + tp_rank
    actual = pull_device_index(pp_rank=pp_rank, pcp_size=pcp_size, pcp_rank=pcp_rank, tp_size=tp_size, tp_rank=tp_rank)
    assert actual == legacy


@pytest.mark.parametrize("tp_rank,tp_size", [(0, 2), (1, 2), (0, 4), (3, 4)])
@pytest.mark.parametrize("pp_rank,pp_size", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("pcp_rank,pcp_size", [(0, 1), (0, 2), (1, 2)])
def test_pull_handshake_port_matches_legacy(tp_rank, tp_size, pp_rank, pp_size, pcp_rank, pcp_size):
    base_port = KV_PORT + 0 * tp_size * pp_size * pcp_size
    legacy = base_port + (pp_rank * pcp_size + pcp_rank) * tp_size + tp_rank
    actual = pull_handshake_port(
        base_port=base_port,
        pp_rank=pp_rank,
        pcp_size=pcp_size,
        pcp_rank=pcp_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    assert actual == legacy


def test_pull_handshake_ports_unique_per_rank():
    tp_size, pp_size, pcp_size = 2, 2, 2
    base_port = KV_PORT
    ports = {
        pull_handshake_port(
            base_port=base_port,
            pp_rank=pp_rank,
            pcp_size=pcp_size,
            pcp_rank=pcp_rank,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        for tp_rank in range(tp_size)
        for pp_rank in range(pp_size)
        for pcp_rank in range(pcp_size)
    }
    assert len(ports) == tp_size * pp_size * pcp_size


def test_pull_dp_replicas_get_distinct_ports():
    tp_size, pp_size, pcp_size = 4, 2, 2
    base_dp0 = pull_base_port(kv_port=KV_PORT, dp_index=0, tp_size=tp_size, pp_size=pp_size, pcp_size=pcp_size)
    base_dp1 = pull_base_port(kv_port=KV_PORT, dp_index=1, tp_size=tp_size, pp_size=pp_size, pcp_size=pcp_size)
    assert base_dp1 - base_dp0 == tp_size * pp_size * pcp_size


@pytest.mark.parametrize("dp_rank", [0, 1, 3])
@pytest.mark.parametrize("tp_size", [2, 4])
def test_layerwise_scheduler_base_port_matches_legacy(dp_rank, tp_size):
    legacy = KV_PORT + dp_rank * tp_size
    assert layerwise_scheduler_base_port(kv_port=KV_PORT, dp_rank=dp_rank, tp_size=tp_size) == legacy


@pytest.mark.parametrize("dp_rank", [0, 1, 3])
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("pcp_rank,pcp_size", [(0, 1), (0, 2), (1, 2)])
def test_layerwise_worker_base_port_matches_legacy(dp_rank, tp_size, pcp_rank, pcp_size):
    legacy = KV_PORT + dp_rank * pcp_size * tp_size + pcp_rank * tp_size
    actual = layerwise_worker_base_port(
        kv_port=KV_PORT, dp_rank=dp_rank, pcp_size=pcp_size, tp_size=tp_size, pcp_rank=pcp_rank
    )
    assert actual == legacy


@pytest.mark.parametrize("dp_rank", [0, 1])
@pytest.mark.parametrize("tp_rank,tp_size", [(0, 2), (1, 2), (0, 4), (3, 4)])
def test_layerwise_handshake_port_matches_legacy(dp_rank, tp_rank, tp_size):
    base_port = KV_PORT + dp_rank * tp_size
    legacy = base_port + tp_rank
    assert layerwise_handshake_port(base_port=base_port, tp_rank=tp_rank) == legacy


def test_layerwise_bases_equal_when_pcp_disabled():
    scheduler_base = layerwise_scheduler_base_port(kv_port=KV_PORT, dp_rank=2, tp_size=4)
    worker_base = layerwise_worker_base_port(kv_port=KV_PORT, dp_rank=2, pcp_size=1, tp_size=4, pcp_rank=0)
    assert scheduler_base == worker_base == KV_PORT + 2 * 4


def test_layerwise_bases_differ_when_pcp_enabled():
    # The layerwise scheduler asserts prefill_context_parallel_size == 1 for KV
    # consumers, so the two bases only meet with pcp disabled. Enabling pcp keeps
    # the historical divergence on purpose (do not unify the formulas).
    scheduler_base = layerwise_scheduler_base_port(kv_port=KV_PORT, dp_rank=1, tp_size=4)
    worker_base = layerwise_worker_base_port(kv_port=KV_PORT, dp_rank=1, pcp_size=2, tp_size=4, pcp_rank=0)
    assert worker_base != scheduler_base
    assert worker_base - scheduler_base == 4
