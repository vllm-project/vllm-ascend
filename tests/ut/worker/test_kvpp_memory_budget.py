"""Exercise actual worker budget hooks without loading a model."""

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.core.kv_cache_placement import KVPPCacheMemoryBudget
from vllm_ascend.worker.worker import NPUWorker


def _worker(override=None):
    return SimpleNamespace(
        vllm_config=object(),
        cache_config=SimpleNamespace(num_gpu_blocks_override=override, kv_cache_memory_bytes=8 << 20),
        _kvpp_cache_allocation_plan=SimpleNamespace(
            plan_memory=lambda _config, available: KVPPCacheMemoryBudget.create(available, 65536, 16384)
        ),
        _apply_kv_offload_decode_memory_constraints=lambda available: available,
        init_snapshot=SimpleNamespace(free_memory=16 << 20),
    )


@pytest.mark.parametrize("override", [None, 1, 96])
def test_worker_budget_accepts_fitting_override(override):
    worker = _worker(override)
    assert NPUWorker._apply_kvpp_memory_constraints(worker, 8 << 20) == 96 * 65536


@pytest.mark.parametrize("override", [-1, 0, 97])
def test_worker_rejects_invalid_override_before_engine_planning(override):
    with pytest.raises(ValueError, match="num_gpu_blocks_override"):
        NPUWorker._apply_kvpp_memory_constraints(_worker(override), 8 << 20)


def test_explicit_memory_budget_cannot_skip_staging():
    worker = _worker()
    calls = []
    worker.model_runner = SimpleNamespace(profile_run=lambda: calls.append("profile"))
    worker._apply_kvpp_memory_constraints = MethodType(NPUWorker._apply_kvpp_memory_constraints, worker)
    assert NPUWorker.determine_available_memory(worker) == 96 * 65536
    assert calls == ["profile"]


def test_disabled_kvpp_does_not_change_budget_or_override():
    worker = _worker(999999)
    worker._kvpp_cache_allocation_plan = None
    assert NPUWorker._apply_kvpp_memory_constraints(worker, 8 << 20) == 8 << 20


def test_elastic_scale_up_cannot_bypass_staging_budget(monkeypatch):
    worker = _worker()
    worker.model_runner = SimpleNamespace(get_kv_cache_spec=lambda: {})
    worker.vllm_config = SimpleNamespace(kv_transfer_config=None)
    monkeypatch.setenv("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "1")
    with (
        patch("vllm_ascend.worker.worker.get_gva_layerwise_config", return_value=None),
        patch("vllm_ascend.worker.worker.KVPPConfig.from_vllm_config", return_value=SimpleNamespace(size=2)),
        pytest.raises(NotImplementedError, match="elastic EP scale-up"),
    ):
        NPUWorker.get_kv_cache_spec(worker)
