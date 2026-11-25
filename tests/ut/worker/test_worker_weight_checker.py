# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the NPUWorker weight-checker methods.

The checksum/reset implementation comes from vLLM's ``gpu_worker.Worker``.
These tests cover the rank-key prefix, checksum/reset behavior over a real
model, and the Host VMM reset restriction.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from vllm.config import ParallelConfig
from vllm.v1.worker import gpu_worker

from vllm_ascend._310p.worker_310p import NPUWorker310
from vllm_ascend.worker import worker as npu_worker_module
from vllm_ascend.worker.worker import NPUWorker


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)
        nn.init.ones_(self.linear.weight)
        # Buffers are not covered by the checker: loading never restores them.
        self.register_buffer("scale", torch.tensor(2.0))


@pytest.fixture
def mock_parallel_groups(monkeypatch):
    group = SimpleNamespace(rank_in_group=0)
    for name in ("get_pp_group", "get_pcp_group", "get_tp_group"):
        monkeypatch.setattr(gpu_worker, name, lambda: group)


@pytest.fixture
def mock_ascend_config(monkeypatch):
    config = SimpleNamespace(input_embedding_host_vmm=False)
    monkeypatch.setattr(npu_worker_module, "get_ascend_config", lambda: config)
    return config


def _make_worker(dp_rank: int) -> NPUWorker:
    worker = object.__new__(NPUWorker)
    worker.parallel_config = ParallelConfig(data_parallel_size=2, data_parallel_rank=dp_rank)
    # Dense DP resets data_parallel_rank to 0 in each engine.
    worker.parallel_config.reconfigure_for_independent_dp_rank()
    return worker


def test_weight_checker_methods_exist():
    for cls in (NPUWorker, NPUWorker310):
        for method in ("compute_weight_checksums", "reset_weights", "_weight_checksum_prefix"):
            assert callable(getattr(cls, method, None)), f"{cls.__name__}.{method}"


def test_dense_dp_replicas_get_distinct_key_prefixes(mock_parallel_groups):
    prefixes = [_make_worker(dp_rank)._weight_checksum_prefix() for dp_rank in (0, 1)]
    assert prefixes == ["dp0:pp0:pcp0:tp0:", "dp1:pp0:pcp0:tp0:"]


def test_compute_weight_checksums_prefixes_keys(mock_parallel_groups, mock_ascend_config):
    worker = _make_worker(dp_rank=0)
    worker.model_runner = cast(Any, SimpleNamespace(get_model=lambda: _TinyModel()))

    checksums = worker.compute_weight_checksums()

    assert set(checksums) == {"dp0:pp0:pcp0:tp0:linear.weight"}
    assert all(len(digest) == 64 for digest in checksums.values())
    assert all(all(c in "0123456789abcdef" for c in digest) for digest in checksums.values())


def test_reset_weights_zeroes_params_not_buffers(mock_parallel_groups, mock_ascend_config):
    model = _TinyModel()
    worker = _make_worker(dp_rank=0)
    worker.model_runner = cast(Any, SimpleNamespace(get_model=lambda: model))

    worker.reset_weights()

    assert torch.all(model.linear.weight == 0)
    assert model.scale.item() == 2.0


def test_reset_weights_rejects_host_vmm_before_mutating_params(mock_ascend_config):
    mock_ascend_config.input_embedding_host_vmm = True
    model = _TinyModel()
    worker = _make_worker(dp_rank=0)
    worker.model_runner = cast(Any, SimpleNamespace(get_model=lambda: model))

    with pytest.raises(RuntimeError, match="Weight reset is unavailable with Host VMM"):
        worker.reset_weights()

    assert torch.all(model.linear.weight == 1)
