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

Checkpoint-source checks cover learned router state and fused projection
expansion, which a shared startup/update data source could otherwise omit.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.e2e.pull_request.rlhf.weight_transfer import utils
from tests.e2e.pull_request.rlhf.weight_transfer.models.deepseek import DEEPSEEK_CASES
from tests.e2e.pull_request.rlhf.weight_transfer.models.glm import GLM_CASES
from tests.e2e.pull_request.rlhf.weight_transfer.models.minimax import MINIMAX_CASES
from tests.e2e.pull_request.rlhf.weight_transfer.models.qwen import QWEN_CASES
from tests.e2e.pull_request.rlhf.weight_transfer.models.registry import MODEL_CASES, pytest_model_cases
from tests.e2e.pull_request.rlhf.weight_transfer.utils import (
    live_update_serve_args,
    reference_serve_args,
    reference_signature,
)
from tests.ut.distributed.weight_transfer.weight_transfer_test_helpers import checkpoint_source, router_meta_model

CASE = MODEL_CASES[0]
ORACLE_CASES = MODEL_CASES
PORT = 8123
GPU_MEMORY_UTILIZATION = 0.75
PACKED_BUFFER_FLOOR_BYTES = 2**30
PACKED_BUFFER_HEADROOM_BYTES = 128 * 2**20


@pytest.mark.parametrize("backend", ["npu_ipc", "hccl"])
def test_model_matrix_separates_model_and_transport_coverage(backend):
    parameters = pytest_model_cases(backend)
    representative = next(case for case in MODEL_CASES if case.model == "Qwen/Qwen3.5-35B-A3B")
    expected = {(representative.id, False), (representative.id, True)}
    if backend == "npu_ipc":
        expected |= {(case.id, False) for case in MODEL_CASES}

    assert {(case.id, packed) for param in parameters for case, packed in [param.values]} == expected
    assert len(parameters) == len(expected)
    for parameter in parameters:
        case, packed = parameter.values
        assert parameter.id == f"{'packed' if packed else 'unpacked'}-{case.id}"
        assert any(mark.name == "e2e_model" and mark.args == (case.model,) for mark in parameter.marks)


def test_model_matrix_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported weight transfer backend"):
        pytest_model_cases("unknown")


@pytest.mark.parametrize("backend", ["npu_ipc", "hccl"])
@pytest.mark.parametrize(
    "model_cases",
    [QWEN_CASES, GLM_CASES, MINIMAX_CASES, DEEPSEEK_CASES],
    ids=["qwen", "glm", "minimax", "deepseek"],
)
def test_family_selection_preserves_parameters_and_markers(backend, model_cases):
    expected = [parameter for parameter in pytest_model_cases(backend) if parameter.values[0] in model_cases]
    actual = pytest_model_cases(backend, model_cases=model_cases)
    assert actual == expected


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


@pytest.mark.parametrize("case", ORACLE_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
@pytest.mark.parametrize("backend", ["npu_ipc", "hccl"])
def test_reference_and_lane_differ_only_in_the_load_path(case, tensor_parallel_size, backend):
    lane = _flag_map(
        live_update_serve_args(
            case,
            backend=backend,
            port=PORT,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=tensor_parallel_size,
        )
    )
    reference = _flag_map(
        reference_serve_args(
            case,
            port=PORT,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=tensor_parallel_size,
        )
    )

    # The lane starts dummy and transfers live; the reference loads W at startup
    # under the case's own model name and tokenizer.
    assert lane.pop("--load-format") == "dummy"
    assert lane.pop("--weight-transfer-config") == f'{{"backend": "{backend}"}}'
    assert reference.pop("--tokenizer") == case.model
    assert reference.pop("--served-model-name") == case.model
    assert "--load-format" not in reference

    assert lane == reference
    assert lane["--tensor-parallel-size"] == str(tensor_parallel_size)


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
    compilation_config = lane["--compilation-config"]
    assert isinstance(compilation_config, str)
    assert compilation_config.startswith('{"cudagraph_mode": "FULL_DECODE_ONLY"')
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

    def __exit__(self, *exc) -> None:
        # Never swallow an exception raised inside the ``with`` block.
        return None

    def get_client(self):
        return MagicMock()


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_reference_signature_starts_one_server_per_measurement_conditions(reference_cache, tensor_parallel_size):
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
            tensor_parallel_size=tensor_parallel_size,
            device_index=0,
        )
        second = reference_signature(
            MagicMock(case_id=CASE.id),
            CASE,
            port=PORT + 1,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=tensor_parallel_size,
            device_index=0,
        )
        # A different memory budget is a different measurement condition.
        third = reference_signature(
            MagicMock(case_id=CASE.id),
            CASE,
            port=PORT + 2,
            gpu_memory_utilization=0.45,
            tensor_parallel_size=tensor_parallel_size,
            device_index=0,
        )

    assert first == second == signature
    assert third == signature
    assert len(_FakeServer.instances) == 2
    assert mock_generate.call_count == 2
    # One gate per startup, not per call: a cache hit must not stop the world.
    assert mock_wait.call_count == 2 * tensor_parallel_size
    assert [call.args[0] for call in mock_wait.call_args_list] == list(range(tensor_parallel_size)) * 2


@pytest.mark.parametrize(
    ("model_name", "layer_index", "checkpoint_gate", "buffers", "legacy_buffers", "added_buffers"),
    [
        pytest.param(
            "Qwen/Qwen3.5-35B-A3B",
            3,
            "model.language_model.layers.3.mlp.gate",
            (),
            {},
            {},
            id="qwen3.5-unchanged",
        ),
        pytest.param(
            "kylesayrs/DeepSeek-V4-Flash-bf16",
            0,
            "model.layers.0.mlp.gate",
            (("tid2eid", (16, 2), torch.int64, False),),
            {"model.layers.0.mlp.gate.tid2eid": ((16, 2), torch.int64)},
            {},
            id="deepseek-v4-integer-table",
        ),
        pytest.param(
            "ZhipuAI/GLM-5.1",
            3,
            "model.layers.3.mlp.gate",
            (("e_score_correction_bias", (8,), torch.float32, True),),
            {},
            {"model.layers.3.mlp.gate.e_score_correction_bias": ((8,), torch.float32)},
            id="glm-5.1-fp32-router-bias",
        ),
        pytest.param(
            "MiniMax/MiniMax-M3",
            3,
            "language_model.model.layers.3.block_sparse_moe.gate",
            (("e_score_correction_bias", (8,), torch.float32, True),),
            {},
            {"language_model.model.layers.3.block_sparse_moe.e_score_correction_bias": ((8,), torch.float32)},
            id="minimax-m3-fp32-router-bias",
        ),
    ],
)
def test_checkpoint_source_buffer_metadata(
    model_name, layer_index, checkpoint_gate, buffers, legacy_buffers, added_buffers
):
    """Pin the metadata delta using representative HF state, without downloads.

    Real buffer registration exercises state_dict persistence. The DeepSeek
    table is deliberately non-persistent to preserve the old integer-buffer
    contract regardless of how a Transformers version registers it.
    """
    case = next(case for case in MODEL_CASES if case.model == model_name)
    model, gate = router_meta_model(layer_index)
    for name, shape, dtype, persistent in buffers:
        gate.register_buffer(name, torch.empty(shape, dtype=dtype, device="meta"), persistent=persistent)
    # Runtime-only floating/complex buffers must never enter the payload.
    model.register_buffer("inv_freq", torch.empty(8, device="meta"), persistent=False)
    model.register_buffer("complex_runtime", torch.empty(8, dtype=torch.complex64, device="meta"), persistent=False)

    unchanged_baseline_metadata = {f"{checkpoint_gate}.weight": ((8, 16), torch.bfloat16), **legacy_buffers}
    source = checkpoint_source(model, case)
    actual = {meta.name: (meta.shape, meta.dtype) for meta in source.metadata()}
    assert actual == unchanged_baseline_metadata | added_buffers
    weights = dict(source)
    assert {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in weights.items()} == actual
    # Exercise expert-count resolution even for models without integer buffers.
    table = source._make_tensor("synthetic.tid2eid", (16, 2), torch.int64)
    assert torch.all((table >= 0) & (table < source._num_experts))
    assert torch.all(table[:, 0] != table[:, 1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    ("tensor_bytes", "expected_capacity"),
    [
        pytest.param(
            PACKED_BUFFER_FLOOR_BYTES - PACKED_BUFFER_HEADROOM_BYTES - 4,
            PACKED_BUFFER_FLOOR_BYTES,
            id="below-growth-threshold",
        ),
        pytest.param(
            PACKED_BUFFER_FLOOR_BYTES - PACKED_BUFFER_HEADROOM_BYTES,
            PACKED_BUFFER_FLOOR_BYTES,
            id="at-growth-threshold",
        ),
        pytest.param(
            PACKED_BUFFER_FLOOR_BYTES - PACKED_BUFFER_HEADROOM_BYTES + 4,
            PACKED_BUFFER_FLOOR_BYTES + 4,
            id="above-growth-threshold",
        ),
        pytest.param(
            PACKED_BUFFER_FLOOR_BYTES + 4,
            PACKED_BUFFER_FLOOR_BYTES + PACKED_BUFFER_HEADROOM_BYTES + 4,
            id="tensor-exceeds-floor",
        ),
    ],
)
def test_packed_buffer_size_uses_largest_tensor_bytes(dtype, tensor_bytes, expected_capacity):
    """Exercise the 1-GiB floor and growth without allocating large tensors."""
    source = MagicMock(spec=utils.FixedRandomWeightSource)
    source.metadata.return_value = [
        utils.ParamMeta("small_before", torch.bfloat16, (8, 16)),
        utils.ParamMeta("largest", dtype, (1, tensor_bytes // dtype.itemsize)),
        utils.ParamMeta("small_after", torch.float32, (8,)),
    ]
    assert utils.packed_buffer_size_for(source) == expected_capacity
    source.metadata.assert_called_once_with()
    source.__iter__.assert_not_called()
