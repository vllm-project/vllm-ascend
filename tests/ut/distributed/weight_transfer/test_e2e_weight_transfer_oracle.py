#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Tests for the weight-transfer E2E oracle's two server configurations.

The oracle only means something while the reference server and the live-update
lane differ in *nothing but* the load path: same dtype, graph mode, NZ layout,
memory budget, max length, executor backend, sleep-mode allocation pool and
config overrides. A future edit that touches one builder and not the other would
silently turn the comparison into apples-to-oranges, so the parity and the
caching contract are pinned here.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.e2e.pull_request.rlhf import weight_transfer_test_utils as utils
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    MODEL_CASES,
    live_update_serve_args,
    reference_serve_args,
    reference_signature,
)

CASE = MODEL_CASES[0]
PORT = 8123
GPU_MEMORY_UTILIZATION = 0.75


def _flag_map(args: list[str]) -> dict[str, str | bool]:
    """Split ``--flag value`` pairs; a bare ``--flag`` maps to ``True``."""
    flags: dict[str, str | bool] = {}
    index = 0
    while index < len(args):
        flag = args[index]
        assert flag.startswith("--"), f"expected a flag, got {flag!r}"
        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            flags[flag] = args[index + 1]
            index += 2
        else:
            flags[flag] = True
            index += 1
    return flags


def test_reference_and_lane_differ_only_in_the_load_path():
    lane = _flag_map(
        live_update_serve_args(
            CASE,
            backend="hccl",
            port=PORT,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
    )
    reference = _flag_map(
        reference_serve_args(
            CASE,
            port=PORT,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
    )

    # The lane starts dummy and transfers live; the reference loads W at startup
    # under the case's own model name and tokenizer.
    assert lane.pop("--load-format") == "dummy"
    assert lane.pop("--weight-transfer-config") == '{"backend": "hccl"}'
    assert reference.pop("--tokenizer") == CASE.model
    assert reference.pop("--served-model-name") == CASE.model
    assert "--load-format" not in reference

    assert lane == reference


def test_reference_and_lane_share_the_measurement_conditions():
    reference = _flag_map(reference_serve_args(CASE, port=PORT, gpu_memory_utilization=0.45))
    lane = _flag_map(live_update_serve_args(CASE, backend="npu_ipc", port=PORT, gpu_memory_utilization=0.45))

    # Everything that could move a logprob has to match, including the graph
    # mode and the NZ layout, or the comparison proves nothing.
    for flag in (
        "--dtype",
        "--compilation-config",
        "--max-model-len",
        "--gpu-memory-utilization",
        "--tensor-parallel-size",
        "--additional-config",
        "--hf-overrides",
        "--distributed-executor-backend",
    ):
        assert lane[flag] == reference[flag], flag
    assert lane["--compilation-config"].startswith('{"cudagraph_mode": "FULL_DECODE_ONLY"')
    # The one-card lane sleeps level 2 before every transfer, which only hands
    # HBM back while the engine's weights and KV cache live in the CaMem pool.
    # Both builders enable it so the reference is a startup load under the very
    # same allocation pool instead of a differently-allocated one; the reference
    # itself never sleeps.
    assert lane["--enable-sleep-mode"] is reference["--enable-sleep-mode"] is True


@pytest.fixture
def reference_cache():
    """Give every test a private copy of the reference cache."""
    with patch.object(utils, "_REFERENCE_SIGNATURES", {}):
        yield utils._REFERENCE_SIGNATURES


class _FakeServer:
    instances: list["_FakeServer"] = []

    def __init__(self, model, *, vllm_serve_args, **kwargs) -> None:
        self.model = model
        self.vllm_serve_args = vllm_serve_args
        self.__class__.instances.append(self)

    def __enter__(self) -> "_FakeServer":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def get_client(self):
        return MagicMock()


def test_reference_signature_starts_one_server_per_measurement_conditions(reference_cache):
    """Lanes that share model + memory budget + TP must reuse one startup.

    The reference never depends on the transport or on the packed flag, so paying
    for a fresh startup-load server per lane would double the suite's cost for no
    extra coverage.
    """
    _FakeServer.instances.clear()
    signature = [("text", (0.5,))]

    with (
        patch("tests.e2e.conftest.RemoteOpenAIServer", _FakeServer),
        patch.object(utils, "fixed_startup_checkpoint", MagicMock()),
        patch.object(utils, "wait_for_free_device_memory") as mock_wait,
        patch.object(utils, "generation_signature", return_value=signature) as mock_generate,
    ):
        first = reference_signature(
            MagicMock(case_id=CASE.id),
            CASE,
            port=PORT,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=1,
            device_index=0,
        )
        second = reference_signature(
            MagicMock(case_id=CASE.id),
            CASE,
            port=PORT + 1,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=1,
            device_index=0,
        )
        # A different memory budget is a different measurement condition.
        third = reference_signature(
            MagicMock(case_id=CASE.id),
            CASE,
            port=PORT + 2,
            gpu_memory_utilization=0.45,
            tensor_parallel_size=1,
            device_index=0,
        )

    assert first == second == signature
    assert third == signature
    assert len(_FakeServer.instances) == 2
    assert mock_generate.call_count == 2
    # One gate per startup, not per call: a cache hit must not stop the world.
    assert mock_wait.call_count == 2
