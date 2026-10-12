# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the NPUWorker weight-checker methods.

The checksum/reset behavior comes from the paired vLLM's
``vllm.model_executor.model_loader.weight_checksum`` (PR #51350); these tests
only cover the NPUWorker wiring: method presence, rank-key prefixes, model
access for both runner shapes, and the clear error for an outdated vLLM.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config import ParallelConfig

from vllm_ascend._310p.worker_310p import NPUWorker310
from vllm_ascend.worker import worker as npu_worker_module
from vllm_ascend.worker.worker import NPUWorker

UPSTREAM_CHECKSUM_MODULE = "vllm.model_executor.model_loader.weight_checksum"


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
        monkeypatch.setattr(npu_worker_module, name, lambda: group)


@pytest.fixture
def mock_ascend_config(monkeypatch):
    monkeypatch.setattr(
        npu_worker_module,
        "get_ascend_config",
        lambda: SimpleNamespace(weight_nz_mode=0),
    )


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
    pytest.importorskip(UPSTREAM_CHECKSUM_MODULE)
    worker = _make_worker(dp_rank=0)
    worker.model_runner = SimpleNamespace(get_model=lambda: _TinyModel())

    checksums = worker.compute_weight_checksums()

    assert set(checksums) == {"dp0:pp0:pcp0:tp0:linear.weight"}
    assert all(len(digest) == 64 for digest in checksums.values())
    assert all(all(c in "0123456789abcdef" for c in digest) for digest in checksums.values())


def test_reset_weights_zeroes_params_not_buffers(mock_parallel_groups, mock_ascend_config):
    pytest.importorskip(UPSTREAM_CHECKSUM_MODULE)
    model = _TinyModel()
    worker = _make_worker(dp_rank=0)
    worker.model_runner = SimpleNamespace(get_model=lambda: model)

    worker.reset_weights()

    assert torch.all(model.linear.weight == 0)
    assert model.scale.item() == 2.0


def test_methods_work_with_v2_runner_shape(mock_parallel_groups, mock_ascend_config):
    pytest.importorskip(UPSTREAM_CHECKSUM_MODULE)
    model = _TinyModel()
    # The v2 model runner only exposes the model attribute, no get_model().
    worker = _make_worker(dp_rank=0)
    worker.model_runner = SimpleNamespace(model=model)

    checksums = worker.compute_weight_checksums()
    assert set(checksums) == {"dp0:pp0:pcp0:tp0:linear.weight"}

    worker.reset_weights()
    assert torch.all(model.linear.weight == 0)


def test_missing_utils_raise_clear_error(monkeypatch):
    monkeypatch.setattr(npu_worker_module, "compute_tensor_digests", None)
    monkeypatch.setattr(npu_worker_module, "zero_weights", None)
    worker = object.__new__(NPUWorker)

    with pytest.raises(RuntimeError, match="42d9f8ccdb"):
        worker.compute_weight_checksums()
    with pytest.raises(RuntimeError, match="42d9f8ccdb"):
        worker.reset_weights()
