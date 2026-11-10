# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for ``nz_warmup``."""

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.model_executor.warmup import nz_warmup


@pytest.fixture(autouse=True)
def reset_state():
    nz_warmup._NZ_WARMED = False
    nz_warmup._NZ_THREAD_STARTED = False
    nz_warmup._NZ_THREAD = None
    yield
    nz_warmup._NZ_WARMED = False
    nz_warmup._NZ_THREAD_STARTED = False
    nz_warmup._NZ_THREAD = None


class _FakeThread:
    instances: list["_FakeThread"] = []

    def __init__(self, target=None, name=None, daemon=False):
        self.target = target
        self.name = name
        self.daemon = daemon
        _FakeThread.instances.append(self)

    def start(self):
        pass


def test_warm_nz_format_cast_runs_once():
    with patch.object(nz_warmup, "maybe_trans_nz") as mock_trans, patch.object(torch.npu, "synchronize"):
        nz_warmup.warm_nz_format_cast("test")
        first_calls = mock_trans.call_count
        nz_warmup.warm_nz_format_cast("test")

    assert mock_trans.call_count == first_calls
    assert nz_warmup._NZ_WARMED is True


def test_warm_nz_format_cast_swallows_failures():
    with patch.object(nz_warmup, "maybe_trans_nz", side_effect=RuntimeError("boom")):
        nz_warmup.warm_nz_format_cast("test")

    assert nz_warmup._NZ_WARMED is True
