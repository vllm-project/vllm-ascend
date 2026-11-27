# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the public wrapper; kernel arithmetic is covered by NPU nightly tests."""

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.ops.triton import deepseek_expert_pool as op


def test_operator_import_is_independent_of_worker_and_offload(monkeypatch):
    original = builtins.__import__

    def import_without_upper_layers(name, *args, **kwargs):
        if name.startswith(("vllm_ascend.worker", "vllm_ascend.distributed.expert_transfer")):
            raise AssertionError(f"operator depends on an upper layer: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_upper_layers)
    spec = importlib.util.spec_from_file_location("isolated_expert_pool", Path(op.__file__))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert module.TOP_K == 6
    assert (module.UNRESTRICTED, module.CONTRIBUTOR, module.RESTRICTED) == (0, 1, 2)


def test_batch_helpers_import_without_loading_routing_or_pool(monkeypatch):
    original = builtins.__import__

    def import_without_routing(name, *args, **kwargs):
        if name.startswith("vllm_ascend.ops") or name.endswith(".dspark_expert_pool"):
            raise AssertionError(f"batch helpers load routing implementation: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_routing)
    path = Path(op.__file__).parents[2] / "distributed/expert_transfer/expert_offload_utils.py"
    spec = importlib.util.spec_from_file_location("isolated_offload_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    runner = SimpleNamespace(model=None)
    module.before_execute(runner)
    assert module.prepare_batch(runner, None, 6, None) == 6
    module.prepare_inputs(runner, None, None)
    module.prepare_capture(None, None)
    assert callable(module.expert_offload_weights)


def make_inputs(rows=6, experts=256):
    return {
        "logits": torch.randn(rows, experts * 2)[:, ::2],
        "roles": torch.ones(rows, dtype=torch.int32),
        "bias": torch.zeros(experts),
        "raw": torch.full((rows, experts), -7.0),
        "votes": torch.full((experts,), 9, dtype=torch.int32),
        "suffix_experts": 2,
        "scaling": 1.5,
    }


@pytest.fixture
def kernels(monkeypatch):
    build, finish = MagicMock(), MagicMock()
    monkeypatch.setattr(op, "_build_pool", build)
    monkeypatch.setattr(op, "_finish_route", finish)
    return build, finish


@pytest.mark.parametrize(
    "name,value",
    [
        ("logits", torch.zeros(6, 256, dtype=torch.float16)),
        ("logits", torch.zeros(6, 256, dtype=torch.bfloat16)),
        ("logits", torch.zeros(256)),
        ("logits", torch.zeros(6, 5)),
        ("roles", torch.zeros(5, dtype=torch.int32)),
        ("roles", torch.zeros(6, dtype=torch.int64)),
        ("roles", torch.zeros(12, dtype=torch.int32)[::2]),
        ("bias", torch.zeros(255)),
        ("bias", torch.zeros(256, dtype=torch.float16)),
        ("bias", torch.zeros(512)[::2]),
        ("raw", torch.zeros(6, 255)),
        ("raw", torch.zeros(6, 256, dtype=torch.float16)),
        ("raw", torch.zeros(6, 512)[:, ::2]),
        ("votes", torch.zeros(255, dtype=torch.int32)),
        ("votes", torch.zeros(256, dtype=torch.int64)),
        ("votes", torch.zeros(512, dtype=torch.int32)[::2]),
        ("bias", torch.empty(256, device="meta")),
        ("suffix_experts", -1),
        ("suffix_experts", 7),
        ("suffix_experts", True),
        ("scaling", float("nan")),
        ("scaling", float("inf")),
    ],
)
def test_invalid_metadata_is_rejected_before_scratch_mutation(kernels, name, value):
    inputs = make_inputs()
    inputs[name] = value
    raw, votes = inputs["raw"].clone(), inputs["votes"].clone()
    with pytest.raises(ValueError):
        op.route_experts(**inputs)
    torch.testing.assert_close(inputs["raw"], raw)
    torch.testing.assert_close(inputs["votes"], votes)
    for kernel in kernels:
        kernel.__getitem__.assert_not_called()


def test_empty_batch_returns_empty_outputs_without_launch_or_scratch_mutation(kernels):
    inputs = make_inputs(rows=0)
    weights, ids = op.route_experts(**inputs)
    assert weights.shape == ids.shape == (0, op.TOP_K)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert torch.all(inputs["votes"] == 9)
    for kernel in kernels:
        kernel.__getitem__.assert_not_called()


def test_strided_logits_use_public_wrapper_and_preserve_inputs(kernels):
    inputs = make_inputs()
    snapshots = {name: inputs[name].clone() for name in ("logits", "roles", "bias")}

    def build(*args, **kwargs):
        args[3].fill_(0.5)
        args[4].add_(1)

    def finish(*args, **kwargs):
        args[4].fill_(0.25)
        args[5].fill_(3)

    kernels[0].__getitem__.return_value.side_effect = build
    kernels[1].__getitem__.return_value.side_effect = finish
    weights, ids = op.route_experts(**inputs)
    assert weights.shape == ids.shape == (6, op.TOP_K)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert torch.all(weights == 0.25) and torch.all(ids == 3)
    assert torch.all(inputs["raw"] == 0.5) and torch.all(inputs["votes"] == 1)
    for name, before in snapshots.items():
        torch.testing.assert_close(inputs[name], before)
    args = kernels[0].__getitem__.return_value.call_args.args
    assert args[0] is inputs["logits"]
    assert args[5:7] == inputs["logits"].stride()
