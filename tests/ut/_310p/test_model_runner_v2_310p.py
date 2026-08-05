# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.model_executor.triton_dispatcher import _get_kernel_impl
from vllm.sampling_params import SamplingParams

from vllm_ascend._310p.worker.v2.block_table import Ascend310PBlockTables
from vllm_ascend._310p.worker.v2.model_runner import NPUModelRunner310V2
from vllm_ascend._310p.worker.v2.sampler import Ascend310PGreedySampler


def test_310p_slot_mapping_kernel_is_registered() -> None:
    kernel_name = "vllm_ascend.worker.v2.block_table._compute_slot_mappings_kernel"

    assert _get_kernel_impl(kernel_name) is not None


def _make_vllm_config(**parallel_overrides):
    parallel = {
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "enable_expert_parallel": False,
    }
    parallel.update(parallel_overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        speculative_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        lora_config=None,
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        model_config=SimpleNamespace(enable_sleep_mode=False),
    )


def test_first_release_config_accepts_tensor_parallelism() -> None:
    config = _make_vllm_config()
    config.parallel_config.tensor_parallel_size = 2

    NPUModelRunner310V2._validate_first_release_config(config)


@pytest.mark.parametrize(
    "setting",
    [
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ],
)
def test_first_release_config_rejects_non_tp_parallelism(setting: str) -> None:
    config = _make_vllm_config(**{setting: 2})

    with pytest.raises(NotImplementedError, match="only supports tensor parallelism"):
        NPUModelRunner310V2._validate_first_release_config(config)


def test_first_release_config_rejects_speculative_decoding() -> None:
    config = _make_vllm_config()
    config.speculative_config = SimpleNamespace(method="mtp")

    with pytest.raises(NotImplementedError, match="deferred to the second"):
        NPUModelRunner310V2._validate_first_release_config(config)


def test_first_release_config_rejects_prefix_caching() -> None:
    config = _make_vllm_config()
    config.cache_config.enable_prefix_caching = True

    with pytest.raises(NotImplementedError, match="deferred to the second"):
        NPUModelRunner310V2._validate_first_release_config(config)


def test_first_release_config_rejects_expert_parallelism() -> None:
    config = _make_vllm_config(enable_expert_parallel=True)

    with pytest.raises(NotImplementedError, match="Expert parallelism"):
        NPUModelRunner310V2._validate_first_release_config(config)


def test_first_release_sampler_accepts_only_greedy() -> None:
    sampler = Ascend310PGreedySampler()
    sampler.add_request(0, 4, SamplingParams(temperature=0))

    with pytest.raises(NotImplementedError, match="only supports greedy"):
        sampler.add_request(0, 4, SamplingParams(temperature=1))


def test_block_tables_use_cpu_metadata_for_gather_and_slot_mapping() -> None:
    block_tables = Ascend310PBlockTables(
        block_sizes=[4],
        max_num_reqs=3,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[4],
        device=torch.device("cpu"),
        kernel_block_sizes=[4],
    )
    block_tables.append_block_ids(0, ([10, 11],), overwrite=True)
    block_tables.append_block_ids(1, ([20],), overwrite=True)
    block_tables.apply_staged_writes()

    gathered = block_tables.gather_block_tables(np.array([1, 0], dtype=np.int32), num_reqs_padded=3)
    torch.testing.assert_close(gathered[0][0, :2], torch.tensor([20, 0], dtype=torch.int32))
    torch.testing.assert_close(gathered[0][1, :2], torch.tensor([10, 11], dtype=torch.int32))
    torch.testing.assert_close(gathered[0][2], torch.zeros_like(gathered[0][2]))

    slots = block_tables.compute_slot_mappings(
        np.array([1, 0], dtype=np.int32),
        np.array([0, 2, 4], dtype=np.int32),
        np.array([0, 1, 4, 5], dtype=np.int64),
        num_tokens_padded=8,
    )
    expected = torch.tensor([[80, 81, 44, 45, -1, -1, -1, -1]], dtype=torch.int32)
    torch.testing.assert_close(slots, expected)


def test_block_tables_reject_device_metadata() -> None:
    block_tables = Ascend310PBlockTables(
        block_sizes=[4],
        max_num_reqs=1,
        max_num_batched_tokens=4,
        max_num_blocks_per_group=[1],
        device=torch.device("cpu"),
        kernel_block_sizes=[4],
    )

    with pytest.raises(TypeError, match="CPU request-state mirror"):
        block_tables.gather_block_tables(torch.empty(1, device="meta", dtype=torch.int32), 1)


def test_worker_selects_v2_runner_when_enabled() -> None:
    from vllm_ascend._310p.worker_310p import NPUWorker310

    worker = object.__new__(NPUWorker310)
    worker.use_v2_model_runner = True
    worker.vllm_config = MagicMock()
    worker.device = torch.device("cpu")

    with patch(
        "vllm_ascend._310p.worker.v2.model_runner.NPUModelRunner310V2",
        return_value=MagicMock(),
    ) as runner_cls:
        worker._create_model_runner()

    runner_cls.assert_called_once_with(worker.vllm_config, worker.device)


def test_worker_keeps_v1_runner_when_v2_is_disabled() -> None:
    from vllm_ascend._310p.worker_310p import NPUWorker310

    worker = object.__new__(NPUWorker310)
    worker.use_v2_model_runner = False
    worker.vllm_config = MagicMock()
    worker.device = torch.device("cpu")

    with patch(
        "vllm_ascend._310p.worker_310p.NPUModelRunner310",
        return_value=MagicMock(),
    ) as runner_cls:
        worker._create_model_runner()

    runner_cls.assert_called_once_with(worker.vllm_config, worker.device)
