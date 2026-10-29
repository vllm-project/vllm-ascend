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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import load_file
from transformers import Qwen3Config

from tests.e2e.pull_request.rlhf import weight_transfer_test_utils as utils
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    MODEL_CASES,
    FixedRandomWeightSource,
    WeightUpdateModelCase,
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


@pytest.fixture
def random_weight_cases(tmp_path):
    cases = []
    for name, layers in (("target", 2), ("draft", 1)):
        directory = tmp_path / name
        Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=layers,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            tie_word_embeddings=False,
        ).save_pretrained(directory)
        cases.append(WeightUpdateModelCase(id=name, model=str(directory), hf_overrides={}))
    return cases


def test_random_target_payload_is_unchanged_when_draft_is_added(random_weight_cases):
    target_case, draft_case = random_weight_cases
    plain = FixedRandomWeightSource(target_case, torch.device("cpu"))
    combined = FixedRandomWeightSource(target_case, torch.device("cpu"), draft_case=draft_case)
    assert plain.draft_source is None
    assert combined.draft_source is not None
    assert plain.metadata() == combined.metadata()
    for (name, expected), (actual_name, actual) in zip(plain, combined, strict=True):
        assert name == actual_name
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(combined.draft_source.metadata()) < len(combined.metadata())
    assert all(not meta.name.startswith("draft.") for meta in combined.draft_source.metadata())


def test_draft_payload_is_repeatable_independent_and_does_not_reseed_global_rng(random_weight_cases):
    target_case, draft_case = random_weight_cases
    source = FixedRandomWeightSource(target_case, torch.device("cpu"), draft_case=draft_case)
    rng_state = torch.get_rng_state()
    first = dict(source.draft_source)
    second = dict(source.draft_source)
    assert torch.equal(torch.get_rng_state(), rng_state)
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    changed = FixedRandomWeightSource(target_case, torch.device("cpu"), draft_case=draft_case, draft_seed=999)
    assert not torch.equal(first["model.norm.weight"], dict(changed.draft_source)["model.norm.weight"])
    assert not torch.equal(first["model.norm.weight"], dict(source)["model.norm.weight"])


def test_target_seed_does_not_change_explicit_draft_payload(random_weight_cases):
    target_case, draft_case = random_weight_cases
    first = FixedRandomWeightSource(target_case, torch.device("cpu"), seed=1, draft_case=draft_case, draft_seed=3)
    second = FixedRandomWeightSource(target_case, torch.device("cpu"), seed=2, draft_case=draft_case, draft_seed=3)
    assert not torch.equal(dict(first)["model.norm.weight"], dict(second)["model.norm.weight"])
    for name, tensor in first.draft_source:
        torch.testing.assert_close(tensor, dict(second.draft_source)[name], rtol=0, atol=0)


def test_random_checkpoints_contain_exact_target_and_draft_payloads(random_weight_cases, tmp_path):
    target_case, draft_case = random_weight_cases
    source = FixedRandomWeightSource(target_case, torch.device("cpu"), draft_case=draft_case)
    destination = tmp_path / "checkpoint"
    # Local configs must never be passed to hf_hub_download.
    with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("unexpected download")):
        source.write_checkpoint(destination)
    for directory, expected_source, case in (
        (destination, source, target_case),
        (destination / "draft", source.draft_source, draft_case),
    ):
        checkpoint = load_file(str(directory / "model.safetensors"))
        payload = dict(expected_source)
        assert checkpoint.keys() == payload.keys()
        for name, tensor in payload.items():
            torch.testing.assert_close(checkpoint[name], tensor, rtol=0, atol=0)
        assert (directory / "config.json").read_bytes() == (Path(case.model) / "config.json").read_bytes()


def test_target_only_checkpoint_keeps_its_original_layout(random_weight_cases, tmp_path):
    case, _ = random_weight_cases
    source = FixedRandomWeightSource(case, torch.device("cpu"))
    destination = tmp_path / "target-only"
    source.write_checkpoint(destination)
    assert {path.name for path in destination.iterdir()} == {"config.json", "model.safetensors"}
    checkpoint = load_file(str(destination / "model.safetensors"))
    for name, tensor in source:
        torch.testing.assert_close(checkpoint[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("is_draft", [False, True])
def test_hccl_client_selects_only_the_start_endpoint(is_draft):
    from tests.e2e.pull_request.two_card.rlhf.state_transitions import test_hccl_weight_transfer as hccl

    client = hccl._WeightUpdateClient("http://unused")
    client.is_draft = is_draft
    with patch.object(client, "_post") as post:
        client.start_weight_update()
        client.update_weights({"names": ["model.norm.weight"]})
        client.finish_weight_update()
    assert [call.args[0] for call in post.call_args_list] == [
        "start_draft_weight_update" if is_draft else "start_weight_update",
        "update_weights",
        "finish_weight_update",
    ]


@pytest.mark.parametrize("is_draft", [False, True])
def test_hccl_send_uses_the_selected_source_and_factory_engine(is_draft):
    from tests.e2e.pull_request.two_card.rlhf.state_transitions import test_hccl_weight_transfer as hccl

    server, engine = MagicMock(), MagicMock()
    client = hccl._WeightUpdateClient("http://unused")
    source = object()
    with patch.object(hccl, "_post") as post:
        hccl._send_update(server, engine, client, source, is_draft=is_draft)
    assert client.is_draft is is_draft
    assert engine.source is source
    engine.send_weights.assert_called_once_with()
    assert [call.args[1] for call in post.call_args_list] == ["pause", "resume"]


def test_hccl_send_resumes_after_a_trainer_failure():
    from tests.e2e.pull_request.two_card.rlhf.state_transitions import test_hccl_weight_transfer as hccl

    server, engine = MagicMock(), MagicMock()
    engine.send_weights.side_effect = RuntimeError("transfer failed")
    with patch.object(hccl, "_post") as post, pytest.raises(RuntimeError, match="transfer failed"):
        hccl._send_update(server, engine, hccl._WeightUpdateClient("http://unused"), object(), is_draft=True)
    assert [call.args[1] for call in post.call_args_list] == ["pause", "resume"]
