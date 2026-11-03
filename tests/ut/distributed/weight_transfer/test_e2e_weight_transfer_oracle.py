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

Checkpoint-source checks preserve the existing models' buffer metadata
through the helper migration.
"""

from unittest.mock import MagicMock, call, patch

import pytest
import torch
from transformers import PretrainedConfig

from tests.e2e.pull_request.two_card.rlhf.weight_transfer.cases import (
    DEEPSEEK_CASES,
    GLM_CASES,
    MODEL_CASES,
    QWEN_CASES,
    pytest_model_cases,
)
from tests.e2e.rlhf.common import server_visible_devices
from tests.e2e.rlhf.weight_transfer import oracle, weight_source
from tests.e2e.rlhf.weight_transfer.oracle import (
    live_update_serve_args,
    reference_serve_args,
    reference_signature,
)

CASE = MODEL_CASES[0]
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

    assert {(parameter.values[0].id, parameter.values[1]) for parameter in parameters} == expected
    assert len(parameters) == len(expected)
    for parameter in parameters:
        case, packed = parameter.values
        assert parameter.id == f"{'packed' if packed else 'unpacked'}-{case.id}"
        assert any(mark.name == "e2e_model" and mark.args == (case.model,) for mark in parameter.marks)
        skip_marks = [mark for mark in parameter.marks if mark.name == "skip"]
        assert len(skip_marks) == int(packed or case.skip_reason is not None)
        if skip_marks:
            assert case.id in skip_marks[0].kwargs["reason"]
            assert ("#18763" if packed else case.skip_reason) in skip_marks[0].kwargs["reason"]

    for family in (QWEN_CASES, GLM_CASES, DEEPSEEK_CASES):
        expected_family = [parameter for parameter in parameters if parameter.values[0] in family]
        assert pytest_model_cases(backend, model_cases=family) == expected_family


def test_model_matrix_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported weight transfer backend"):
        pytest_model_cases("unknown")


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


@pytest.mark.parametrize(
    ("case", "backend", "tensor_parallel_size"),
    [
        pytest.param(QWEN_CASES[0], "npu_ipc", 2, id="qwen-ipc-tp2"),
        pytest.param(QWEN_CASES[0], "hccl", 1, id="qwen-hccl-tp1"),
        pytest.param(GLM_CASES[0], "npu_ipc", 2, id="glm-ipc-tp2"),
    ],
)
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

    compilation_config = lane["--compilation-config"]
    assert isinstance(compilation_config, str)
    assert compilation_config.startswith('{"cudagraph_mode": "FULL_DECODE_ONLY"')
    # The IPC lane sleeps level 2 before every transfer, which only hands
    # HBM back while the engine's weights and KV cache live in the CaMem pool.
    # Both builders enable it so the reference is a startup load under the very
    # same allocation pool instead of a differently-allocated one; the reference
    # itself never sleeps.
    assert lane["--enable-sleep-mode"] is reference["--enable-sleep-mode"] is True


@pytest.mark.parametrize(
    ("model_name", "layer_index", "checkpoint_gate", "buffers", "expected_buffers"),
    [
        pytest.param("Qwen/Qwen3.5-35B-A3B", 3, "model.language_model.layers.3.mlp.gate", (), {}, id="qwen3.5"),
        pytest.param(
            "kylesayrs/DeepSeek-V4-Flash-bf16",
            0,
            "model.layers.0.mlp.gate",
            (("tid2eid", (16, 2), torch.int64, False),),
            {"model.layers.0.mlp.gate.tid2eid": ((16, 2), torch.int64)},
            id="deepseek-v4",
        ),
        pytest.param(
            "zai-org/GLM-5.1",
            3,
            "model.layers.3.mlp.gate",
            (("e_score_correction_bias", (8,), torch.float32, True),),
            {},
            id="glm-5.1",
        ),
    ],
)
def test_checkpoint_source_preserves_existing_buffer_metadata(
    model_name, layer_index, checkpoint_gate, buffers, expected_buffers
):
    """Preserve main's non-floating buffer selection for the existing models."""
    case = next(case for case in MODEL_CASES if case.model == model_name)
    model = torch.nn.Module()
    parent = model
    for component in ("model", "layers", str(layer_index), "mlp"):
        child = torch.nn.Module()
        parent.add_module(component, child)
        parent = child
    gate = torch.nn.Linear(16, 8, bias=False, device="meta")
    parent.add_module("gate", gate)
    for name, shape, dtype, persistent in buffers:
        gate.register_buffer(name, torch.empty(shape, dtype=dtype, device="meta"), persistent=persistent)
    model.register_buffer("inv_freq", torch.empty(8, device="meta"), persistent=False)
    model.register_buffer("complex_runtime", torch.empty(8, dtype=torch.complex64, device="meta"), persistent=False)
    meta_config = PretrainedConfig()
    config = (
        PretrainedConfig(**{case.meta_config_attribute: meta_config}) if case.meta_config_attribute else meta_config
    )
    with (
        patch.object(weight_source.AutoConfig, "from_pretrained", return_value=config),
        patch.object(weight_source.AutoModelForCausalLM, "from_config", return_value=model),
    ):
        source = weight_source.FixedRandomWeightSource(case, torch.device("cpu"))
    expected = {f"{checkpoint_gate}.weight": ((8, 16), torch.bfloat16), **expected_buffers}
    metadata = source.metadata() if weight_source.vllm_version_is("0.32.0") else source.metadata
    assert {meta.name: (meta.shape, meta.dtype) for meta in metadata} == expected
    weights = dict(source)
    assert {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in weights.items()} == expected
    for name, tensor in weights.items():
        if name.endswith(".tid2eid"):
            assert torch.all((tensor >= 0) & (tensor < source._num_experts))
            assert torch.all(tensor[:, 0] != tensor[:, 1])


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
    source = MagicMock(spec=weight_source.FixedRandomWeightSource)
    metadata = [
        weight_source.ParamMeta("small_before", torch.bfloat16, (8, 16)),
        weight_source.ParamMeta("largest", dtype, (1, tensor_bytes // dtype.itemsize)),
        weight_source.ParamMeta("small_after", torch.float32, (8,)),
    ]
    if weight_source.vllm_version_is("0.32.0"):
        source.metadata.return_value = metadata
    else:
        source.metadata = metadata
    assert weight_source.packed_buffer_size_for(source) == expected_capacity
    if weight_source.vllm_version_is("0.32.0"):
        source.metadata.assert_called_once_with()
    else:
        assert source.metadata is metadata
    source.__iter__.assert_not_called()


@pytest.mark.parametrize(("backend", "tp"), [("npu_ipc", 2), ("hccl", 1)])
def test_servers_share_environment_and_wait_for_all_devices(backend, tp):
    case = CASE
    source = MagicMock()
    reference, dummy = ["startup"], ["dummy"]
    env = {"ASCEND_RT_VISIBLE_DEVICES": "6,7" if tp == 2 else "6"}
    with (
        patch.object(oracle, "reference_signature", return_value=reference) as startup,
        patch.object(oracle, "wait_for_free_device_memory") as wait,
        patch("tests.e2e.conftest.RemoteOpenAIServer") as server_factory,
        patch.object(oracle, "generation_signature", return_value=dummy) as generate,
        patch.object(oracle, "get_open_port", return_value=8123) as get_port,
    ):
        with oracle.weight_transfer_servers(
            source,
            case,
            backend=backend,
            tensor_parallel_size=tp,
            device_index=6,
            gpu_memory_utilization=0.45,
            env_dict=env,
        ) as (server, actual_reference, actual_dummy):
            assert server is server_factory.return_value.__enter__.return_value
            assert actual_reference == reference
            assert actual_dummy == dummy
            generate.assert_called_once_with(server.get_client(), case.model)
            assert get_port.call_count == 2
            startup.assert_called_once_with(
                source,
                case,
                port=8123,
                gpu_memory_utilization=0.45,
                tensor_parallel_size=tp,
                device_index=6,
                env_dict=env,
            )
            assert wait.call_args_list == [call(6 + rank, 0.45) for rank in range(tp)]
            assert server_factory.call_args.kwargs["env_dict"] == env
        server_factory.return_value.__exit__.assert_called_once()
    assert env == {"ASCEND_RT_VISIBLE_DEVICES": "6,7" if tp == 2 else "6"}


@pytest.mark.parametrize(
    ("visible", "device_index", "world_size", "expected"),
    [
        (None, 0, 2, "0,1"),
        (None, 6, 1, "6"),
        ("4,5", 1, 1, "5"),
        ("6,2,7,3", 1, 2, "2,7"),
    ],
)
def test_server_visibility_maps_parent_logical_devices(monkeypatch, visible, device_index, world_size, expected):
    if visible is None:
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    assert server_visible_devices(device_index, world_size) == expected


@pytest.mark.parametrize(
    ("visible", "device_index", "world_size"),
    [("4,5", 1, 2), ("4,4", 0, 2), ("", 0, 1), ("4,x", 0, 2), ("4,5", -1, 1), ("4,5", 0, 0)],
)
def test_server_visibility_rejects_invalid_selection(monkeypatch, visible, device_index, world_size):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError):
        server_visible_devices(device_index, world_size)


@pytest.mark.parametrize(
    ("memory", "device", "visible"),
    [
        pytest.param(0.45, 0, "0,1", id="memory-budget"),
        pytest.param(GPU_MEMORY_UTILIZATION, 1, "0,1", id="logical-device"),
        pytest.param(GPU_MEMORY_UTILIZATION, 0, "4,5", id="physical-visibility"),
    ],
)
def test_reference_cache_reuses_only_matching_conditions(memory: float, device: int, visible: str):
    def measure(port: int, memory: float, device: int, env: dict[str, str]):
        return reference_signature(
            MagicMock(),
            CASE,
            port=port,
            gpu_memory_utilization=memory,
            tensor_parallel_size=2,
            device_index=device,
            env_dict=env,
        )

    with (
        patch.object(oracle, "_REFERENCE_SIGNATURES", {}),
        patch("tests.e2e.conftest.RemoteOpenAIServer") as server_factory,
        patch.object(oracle, "fixed_startup_checkpoint", MagicMock()),
        patch.object(oracle, "wait_for_free_device_memory") as wait,
        patch.object(oracle, "generation_signature", side_effect=[[("first", ())], [("second", ())]]) as generate,
    ):
        env = {"ASCEND_RT_VISIBLE_DEVICES": "0,1"}
        first = measure(PORT, GPU_MEMORY_UTILIZATION, 0, env)
        # Ports are incidental: the same measurement must reuse its reference.
        assert measure(PORT + 1, GPU_MEMORY_UTILIZATION, 0, env) == first
        env = {"ASCEND_RT_VISIBLE_DEVICES": visible}
        second = measure(PORT, memory, device, env)
        assert measure(PORT, memory, device, env.copy()) == second
    assert first != second
    assert generate.call_count == server_factory.call_count == 2
    # Cache hits do not wait; each startup gates every TP rank.
    assert wait.call_args_list == [call(0, GPU_MEMORY_UTILIZATION), call(1, GPU_MEMORY_UTILIZATION)] + [
        call(device + rank, memory) for rank in range(2)
    ]


@pytest.mark.parametrize("invalid", [None, "dummy", "first-update", "second-update", "missing-update"])
def test_oracle_requires_observable_payload_and_every_update(invalid):
    reference = [("reference", (0.5,))]
    dummy = [("dummy", (0.1,))]
    updates = [reference] * oracle.UPDATE_ROUNDS
    if invalid == "dummy":
        dummy = reference
    elif invalid == "first-update":
        updates[0] = dummy
    elif invalid == "second-update":
        updates[-1] = dummy
    elif invalid == "missing-update":
        updates.pop()
    if invalid is None:
        oracle.assert_weight_update_matches_reference(dummy, reference, updates, CASE)
    else:
        with pytest.raises(AssertionError):
            oracle.assert_weight_update_matches_reference(dummy, reference, updates, CASE)
