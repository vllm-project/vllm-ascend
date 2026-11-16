# SPDX-License-Identifier: Apache-2.0
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config.parallel import ParallelConfig

from vllm_ascend._310p import worker_310p as worker_module
from vllm_ascend._310p.worker_310p import NPUWorker310

_PORTS_PER_REPLICA = worker_module._HCCL_PORTS_PER_DP_REPLICA


def make_worker(**overrides):
    config = SimpleNamespace(
        data_parallel_size=8,
        data_parallel_rank_local=7,
        data_parallel_index=7,
        assigned_physical_gpu_ids=None,
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        world_size=1,
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    config.get_local_gpu_index = partial(ParallelConfig.get_local_gpu_index, config)
    worker = object.__new__(NPUWorker310)
    worker.parallel_config = config
    worker.local_rank = 0
    worker.rank = 0
    return worker


def test_eight_local_replicas_use_valid_disjoint_ports(monkeypatch):
    for rank in range(8):
        monkeypatch.setenv("HCCL_IF_BASE_PORT", "50000")
        worker = make_worker(data_parallel_rank_local=rank, data_parallel_index=rank)
        with patch.object(worker_module.torch.npu, "device_count", return_value=8):
            worker._configure_parallel_device()
        assert worker.local_rank == rank
        assert worker_module.os.environ["HCCL_IF_BASE_PORT"] == str(50000 + rank * _PORTS_PER_REPLICA)


@pytest.mark.parametrize(
    "options",
    [
        {"distributed_executor_backend": "ray"},
        {"distributed_executor_backend": "external_launcher"},
        {"data_parallel_backend": "ray"},
        {"nnodes_within_dp": 2},
    ],
)
def test_launcher_owned_device_index_is_preserved(monkeypatch, options):
    monkeypatch.setenv("HCCL_IF_BASE_PORT", "60000")
    worker = make_worker(**options)
    with patch.object(worker_module.torch.npu, "device_count", return_value=1):
        worker._configure_parallel_device()
    assert worker.local_rank == 0


def test_device_ids_are_sliced_per_dp_rank_before_the_worker():
    """vLLM gives each DP engine core its own --device-ids shard."""
    from vllm.v1.engine.utils import get_physical_gpu_ids_for_local_dp_rank

    def shard(local_dp_rank, ids):
        return get_physical_gpu_ids_for_local_dp_rank(
            "ASCEND_RT_VISIBLE_DEVICES", local_dp_rank, 2, 2, user_assigned_gpu_ids=ids
        )

    assert shard(0, [0, 1, 2, 3]) == [0, 1]
    assert shard(1, [0, 1, 2, 3]) == [2, 3]
    # Non-contiguous and out-of-order lists keep their user order after slicing.
    assert shard(0, [3, 1]) == [3, 1]
    assert shard(1, [3, 1, 0, 2]) == [0, 2]


def test_dp_shard_is_not_shifted_again_by_the_worker(monkeypatch):
    """local_rank indexes the shard, so no second DP offset is applied."""
    monkeypatch.setenv("HCCL_IF_BASE_PORT", "60000")
    worker = make_worker(
        data_parallel_size=2,
        data_parallel_rank_local=1,
        data_parallel_index=1,
        assigned_physical_gpu_ids=[2, 3],
        world_size=2,
    )
    with patch("vllm.platforms.interface.set_assigned_physical_gpu_ids") as publish:
        worker._configure_parallel_device()
    publish.assert_called_once_with([2, 3])
    assert worker.local_rank == 0
    assert worker_module.os.environ["HCCL_IF_BASE_PORT"] == "60032"


def test_assigned_device_out_of_shard_is_rejected(monkeypatch):
    monkeypatch.setenv("HCCL_IF_BASE_PORT", "60000")
    worker = make_worker(assigned_physical_gpu_ids=[2, 3], world_size=2)
    worker.local_rank = 2
    with (
        patch("vllm.platforms.interface.set_assigned_physical_gpu_ids"),
        pytest.raises(ValueError, match="outside 2 visible devices"),
    ):
        worker._configure_parallel_device()


def test_dp_index_fallback_matches_upstream(monkeypatch):
    monkeypatch.setenv("HCCL_IF_BASE_PORT", "60000")
    worker = make_worker(data_parallel_rank_local=None, data_parallel_index=3)
    with patch.object(worker_module.torch.npu, "device_count", return_value=8):
        worker._configure_parallel_device()
    assert worker.local_rank == 3
    assert worker_module.os.environ["HCCL_IF_BASE_PORT"] == "60096"


@pytest.mark.parametrize("base", ["1023", "65504", "66000"])
def test_port_range_overflow_fails_before_device_initialization(monkeypatch, base):
    monkeypatch.setenv("HCCL_IF_BASE_PORT", base)
    worker = make_worker()
    with (
        patch.object(worker_module.torch.npu, "device_count", return_value=8),
        pytest.raises(ValueError, match="HCCL_IF_BASE_PORT"),
    ):
        worker._configure_parallel_device()
    assert worker_module.os.environ["HCCL_IF_BASE_PORT"] == base


def test_single_replica_leaves_hccl_base_port_untouched(monkeypatch):
    monkeypatch.setenv("HCCL_IF_BASE_PORT", "60000")
    worker = make_worker(data_parallel_size=1, data_parallel_rank_local=0, data_parallel_index=0)
    with patch.object(worker_module.torch.npu, "device_count", return_value=8):
        worker._configure_parallel_device()
    assert worker_module.os.environ["HCCL_IF_BASE_PORT"] == "60000"


def test_dp_shard_feeds_the_final_device_mapping(monkeypatch):
    """Slice -> logical id -> device the worker actually opens."""
    import vllm.platforms.interface as interface
    from vllm.platforms import current_platform
    from vllm.v1.engine.utils import get_physical_gpu_ids_for_local_dp_rank

    def devices(local_dp_rank, ids):
        shard = get_physical_gpu_ids_for_local_dp_rank(
            "ASCEND_RT_VISIBLE_DEVICES", local_dp_rank, 2, 2, user_assigned_gpu_ids=ids
        )
        monkeypatch.setattr(interface, "_assigned_physical_gpu_ids", None, raising=False)
        interface.set_assigned_physical_gpu_ids(shard)
        monkeypatch.delenv(current_platform.device_control_env_var, raising=False)
        return [current_platform.logical_device_id_to_visible_device_id(i) for i in range(len(shard))]

    assert devices(0, [0, 1, 2, 3]) == [0, 1]
    assert devices(1, [0, 1, 2, 3]) == [2, 3]
    # Out-of-order lists are preserved, so the mapping is not positional-only.
    assert devices(1, [3, 1, 0, 2]) == [0, 2]


def test_non_integer_base_port_reports_the_variable(monkeypatch):
    monkeypatch.setenv("HCCL_IF_BASE_PORT", "abc")
    worker = make_worker()
    with (
        patch.object(worker_module.torch.npu, "device_count", return_value=8),
        pytest.raises(ValueError, match="HCCL_IF_BASE_PORT must be an integer, got 'abc'"),
    ):
        worker._configure_parallel_device()


def test_invalid_device_index_is_rejected(monkeypatch):
    monkeypatch.setenv("HCCL_IF_BASE_PORT", "60000")
    worker = make_worker()
    with (
        patch.object(worker_module.torch.npu, "device_count", return_value=4),
        pytest.raises(ValueError, match="outside 4 visible devices"),
    ):
        worker._configure_parallel_device()


def _config(data_parallel_size: int = 2, speculative_config=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(use_mla=False, enable_sleep_mode=False),
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_expert_parallel=False,
        ),
        speculative_config=speculative_config,
        kv_transfer_config=None,
    )


def test_validate_config_does_not_reject_data_parallel():
    from vllm_ascend._310p.worker.v2.model_runner import NPUModelRunner310V2

    NPUModelRunner310V2._validate_config(_config())


def test_pipeline_parallel_is_still_rejected():
    from vllm_ascend._310p.worker.v2.model_runner import NPUModelRunner310V2

    vllm_config = _config(data_parallel_size=1)
    vllm_config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(NotImplementedError, match="pipeline_parallel_size"):
        NPUModelRunner310V2._validate_config(vllm_config)
