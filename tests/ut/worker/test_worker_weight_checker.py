# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the NPUWorker weight-checker methods.

The checksum/reset behavior comes from the paired vLLM's
``vllm.model_executor.model_loader.weight_checksum`` (PR #51350) and the worker
methods themselves are inherited from vLLM's ``gpu_worker.Worker``; these tests
pin the contract the NPU worker relies on: method presence, the rank-key
prefix, and the checksum/reset behavior over a real model.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from vllm.config import ParallelConfig
from vllm.v1.worker import gpu_worker

from vllm_ascend._310p.worker_310p import NPUWorker310
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


def test_compute_weight_checksums_prefixes_keys(mock_parallel_groups):
    worker = _make_worker(dp_rank=0)
    worker.model_runner = cast(Any, SimpleNamespace(get_model=lambda: _TinyModel()))

    checksums = worker.compute_weight_checksums()

    assert set(checksums) == {"dp0:pp0:pcp0:tp0:linear.weight"}
    assert all(len(digest) == 64 for digest in checksums.values())
    assert all(all(c in "0123456789abcdef" for c in digest) for digest in checksums.values())


def test_reset_weights_zeroes_params_not_buffers(mock_parallel_groups):
    model = _TinyModel()
    worker = _make_worker(dp_rank=0)
    worker.model_runner = cast(Any, SimpleNamespace(get_model=lambda: model))

    worker.reset_weights()

    assert torch.all(model.linear.weight == 0)
    assert model.scale.item() == 2.0
