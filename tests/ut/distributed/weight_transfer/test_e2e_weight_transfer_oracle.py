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
from transformers import PretrainedConfig

from tests.e2e.pull_request.rlhf import weight_transfer_test_utils as utils
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    MODEL_CASES,
    live_update_serve_args,
    reference_serve_args,
    reference_signature,
)

CASE = MODEL_CASES[0]
MINIMAX_M3_CASE = next(case for case in MODEL_CASES if case.model == "MiniMaxAI/MiniMax-M3")
ORACLE_CASES = (CASE, MINIMAX_M3_CASE)
PORT = 8123
GPU_MEMORY_UTILIZATION = 0.75
PACKED_BUFFER_FLOOR_BYTES = 2**30
PACKED_BUFFER_HEADROOM_BYTES = 128 * 2**20


def _checkpoint_source(model, case):
    # Only supply num_experts when the case does not override an expert field.
    # Otherwise an incorrect fallback could pass by reading this default.
    overrides = (
        case.hf_overrides.get(case.meta_config_attribute, {}) if case.meta_config_attribute else case.hf_overrides
    )
    meta_config = PretrainedConfig()
    if not any(name in overrides for name in ("n_routed_experts", "num_local_experts", "num_experts")):
        meta_config.num_experts = 8
    config = (
        PretrainedConfig(**{case.meta_config_attribute: meta_config}) if case.meta_config_attribute else meta_config
    )
    with (
        patch.object(utils.AutoConfig, "from_pretrained", return_value=config),
        patch.object(utils.AutoModelForCausalLM, "from_config", return_value=model),
    ):
        source = utils.FixedRandomWeightSource(case, torch.device("cpu"))
    assert source._num_experts == 8
    return source


def _router_meta_model(layer_index):
    model = torch.nn.Module()
    parent = model
    for component in ("model", "layers", str(layer_index), "mlp"):
        child = torch.nn.Module()
        parent.add_module(component, child)
        parent = child
    gate = torch.nn.Linear(16, 8, bias=False, device="meta")
    parent.add_module("gate", gate)
    return model, gate


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
def test_reference_and_lane_differ_only_in_the_load_path(case):
    lane = _flag_map(
        live_update_serve_args(
            case,
            backend="hccl",
            port=PORT,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
    )
    reference = _flag_map(
        reference_serve_args(
            case,
            port=PORT,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        )
    )

    # The lane starts dummy and transfers live; the reference loads W at startup
    # under the case's own model name and tokenizer.
    assert lane.pop("--load-format") == "dummy"
    assert lane.pop("--weight-transfer-config") == '{"backend": "hccl"}'
    assert reference.pop("--tokenizer") == case.model
    assert reference.pop("--served-model-name") == case.model
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
            "zai-org/GLM-5.1",
            3,
            "model.layers.3.mlp.gate",
            (("e_score_correction_bias", (8,), torch.float32, True),),
            {},
            {"model.layers.3.mlp.gate.e_score_correction_bias": ((8,), torch.float32)},
            id="glm-5.1-fp32-router-bias",
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
            "MiniMaxAI/MiniMax-M3",
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
    model, gate = _router_meta_model(layer_index)
    for name, shape, dtype, persistent in buffers:
        gate.register_buffer(name, torch.empty(shape, dtype=dtype, device="meta"), persistent=persistent)
    # Runtime-only floating/complex buffers must never enter the payload.
    model.register_buffer("inv_freq", torch.empty(8, device="meta"), persistent=False)
    model.register_buffer("complex_runtime", torch.empty(8, dtype=torch.complex64, device="meta"), persistent=False)

    unchanged_baseline_metadata = {f"{checkpoint_gate}.weight": ((8, 16), torch.bfloat16), **legacy_buffers}
    source = _checkpoint_source(model, case)
    actual = {meta.name: (meta.shape, meta.dtype) for meta in source.metadata()}
    assert actual == unchanged_baseline_metadata | added_buffers
    weights = dict(source)
    assert {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in weights.items()} == actual
    # Exercise expert-count resolution even for models without integer buffers.
    table = source._make_tensor("synthetic.tid2eid", (16, 2), torch.int64)
    assert torch.all((table >= 0) & (table < source._num_experts))
    assert torch.all(table[:, 0] != table[:, 1])


def test_minimax_checkpoint_source_includes_parameter_router_bias():
    """Keep the name mapping valid if HF represents the bias as a parameter."""
    model, gate = _router_meta_model(3)
    gate.register_parameter(
        "e_score_correction_bias", torch.nn.Parameter(torch.empty(8, dtype=torch.float32, device="meta"))
    )
    source = _checkpoint_source(model, MINIMAX_M3_CASE)
    prefix = "language_model.model.layers.3.block_sparse_moe"
    expected = {
        f"{prefix}.gate.weight": ((8, 16), torch.bfloat16),
        f"{prefix}.e_score_correction_bias": ((8,), torch.bfloat16),
    }
    assert {meta.name: (meta.shape, meta.dtype) for meta in source.metadata()} == expected
    assert {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in source} == expected


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


def test_minimax_checkpoint_source_splits_fused_projections():
    model = MagicMock()
    model.named_parameters.return_value = [
        ("model.layers.0.mlp.gate_up_proj.weight", torch.empty((12, 16), device="meta")),
        ("model.layers.3.mlp.shared_experts.gate_up_proj.weight", torch.empty((12, 16), device="meta")),
        ("model.layers.3.mlp.experts.gate_up_proj", torch.empty((2, 12, 16), device="meta")),
    ]
    model.named_buffers.return_value = []
    model.state_dict.return_value = {}
    source = _checkpoint_source(model, MINIMAX_M3_CASE)
    actual = {meta.name: (meta.shape, meta.dtype) for meta in source.metadata()}
    expected_names = {
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight",
        "language_model.model.layers.3.block_sparse_moe.shared_experts.up_proj.weight",
        *{
            f"language_model.model.layers.3.block_sparse_moe.experts.{expert}.{shard}.weight"
            for expert in range(2)
            for shard in ("w1", "w3")
        },
    }
    assert actual == {name: ((6, 16), torch.bfloat16) for name in expected_names}
