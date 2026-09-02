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

import json
import stat
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config import CUDAGraphMode

import vllm_ascend._310p.dflash_fdo_numerical_probe as numerical_probe
import vllm_ascend.spec_decode.llm_base_proposer as llm_base_proposer
import vllm_ascend.worker.model_runner_v1 as model_runner_v1
from vllm_ascend import envs


def _config(method: str | None, mode: CUDAGraphMode):
    return SimpleNamespace(
        speculative_config=(SimpleNamespace(method=method) if method is not None else None),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
    )


def _enabled_environ(tmp_path):
    return {
        numerical_probe.PROBE_DIR_ENV: str(tmp_path / "trace"),
        numerical_probe.PROBE_COMPONENT_ENV: "target",
        numerical_probe.PROBE_LAYER_ENV: "model.layers.17",
        numerical_probe.PROBE_DATASET_REQUEST_ENV: "12",
        numerical_probe.PROBE_MAX_ITERATIONS_ENV: "7",
        numerical_probe.PROBE_MAX_RECORDS_ENV: "31",
        numerical_probe.PROBE_MAX_BYTES_ENV: "65536",
    }


def test_probe_configuration_is_default_off_without_filesystem_work(tmp_path):
    config = numerical_probe.FdoNumericalProbeConfig.from_environ(
        _config("dflash", CUDAGraphMode.FULL_DECODE_ONLY),
        environ={},
    )

    assert config.enabled is False
    assert config.output_dir is None
    assert not list(tmp_path.iterdir())


def test_probe_configuration_selects_component_layer_and_bounds(tmp_path):
    with patch.object(numerical_probe, "is_310p", return_value=True):
        config = numerical_probe.FdoNumericalProbeConfig.from_environ(
            _config("dflash", CUDAGraphMode.FULL_DECODE_ONLY),
            environ=_enabled_environ(tmp_path),
        )

    assert config.enabled is True
    assert config.output_dir == tmp_path / "trace"
    assert config.component == "target"
    assert config.layer == "model.layers.17"
    assert config.dataset_request == 12
    assert config.max_iterations == 7
    assert config.max_records == 31
    assert config.max_bytes == 65536
    assert not config.output_dir.exists()


def test_probe_configuration_uses_centralized_env_registry(tmp_path):
    values = _enabled_environ(tmp_path)

    assert set(values).issubset(envs.env_variables)
    with (
        patch.multiple(envs, create=True, **values),
        patch.object(numerical_probe, "is_310p", return_value=True),
    ):
        config = numerical_probe.FdoNumericalProbeConfig.from_environ(_config("dflash", CUDAGraphMode.FULL_DECODE_ONLY))

    assert config.enabled is True
    assert config.output_dir == tmp_path / "trace"
    assert config.component == "target"
    assert config.layer == "model.layers.17"
    assert config.dataset_request == 12
    assert config.max_iterations == 7
    assert config.max_records == 31
    assert config.max_bytes == 65536


@pytest.mark.parametrize(
    ("is_310p_platform", "method", "mode"),
    [
        (False, "dflash", CUDAGraphMode.FULL_DECODE_ONLY),
        (True, "mtp", CUDAGraphMode.FULL_DECODE_ONLY),
        (True, None, CUDAGraphMode.FULL_DECODE_ONLY),
        (True, "dflash", CUDAGraphMode.FULL),
    ],
)
def test_enabled_probe_rejects_outside_310p_dflash_eager_or_fdo(
    tmp_path,
    is_310p_platform,
    method,
    mode,
):
    with (
        patch.object(numerical_probe, "is_310p", return_value=is_310p_platform),
        pytest.raises(numerical_probe.FdoNumericalProbeConfigError),
    ):
        numerical_probe.FdoNumericalProbeConfig.from_environ(
            _config(method, mode),
            environ=_enabled_environ(tmp_path),
        )


@pytest.mark.parametrize(
    "mode",
    [
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ],
)
def test_enabled_probe_accepts_comparable_eager_fdo_and_hybrid_modes(
    tmp_path,
    mode,
):
    with patch.object(numerical_probe, "is_310p", return_value=True):
        config = numerical_probe.FdoNumericalProbeConfig.from_environ(
            _config("dflash", mode),
            environ=_enabled_environ(tmp_path),
        )

    assert config.enabled is True
    assert config.mode == mode.name


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("component", "unknown"),
        ("max_iterations", "0"),
        ("max_records", "-1"),
        ("max_bytes", "1023"),
    ],
)
def test_enabled_probe_rejects_invalid_selection_or_bounds(
    tmp_path,
    key,
    value,
):
    environ = _enabled_environ(tmp_path)
    env_key = {
        "component": numerical_probe.PROBE_COMPONENT_ENV,
        "max_iterations": numerical_probe.PROBE_MAX_ITERATIONS_ENV,
        "max_records": numerical_probe.PROBE_MAX_RECORDS_ENV,
        "max_bytes": numerical_probe.PROBE_MAX_BYTES_ENV,
    }[key]
    environ[env_key] = value

    with (
        patch.object(numerical_probe, "is_310p", return_value=True),
        pytest.raises(numerical_probe.FdoNumericalProbeConfigError),
    ):
        numerical_probe.FdoNumericalProbeConfig.from_environ(
            _config("dflash", CUDAGraphMode.FULL_DECODE_ONLY),
            environ=environ,
        )


def _identity(**overrides):
    fields = {
        "mode": "FULL_DECODE_ONLY",
        "component": "target",
        "tp_rank": 1,
        "dataset_request": 12,
        "generated_prefix": (17, 23, 42),
        "speculative_iteration": 9,
        "draft_substep": None,
        "descriptor": 32,
        "actual_tokens": 16,
        "active_rows": (0, 3),
        "semantic_role": "final_hidden",
        "shape": (4, 8),
        "dtype": "torch.float16",
    }
    fields.update(overrides)
    return numerical_probe.ProbeTraceIdentity(**fields)


def _writer_config(tmp_path, **overrides):
    fields = {
        "enabled": True,
        "output_dir": tmp_path / "trace",
        "mode": "FULL_DECODE_ONLY",
        "component": "target",
        "layer": None,
        "max_iterations": 16,
        "max_records": 4,
        "max_bytes": 1024 * 1024,
    }
    fields.update(overrides)
    return numerical_probe.FdoNumericalProbeConfig(**fields)


def test_trace_identity_contains_every_alignment_field():
    identity = _identity()

    assert identity.to_manifest_dict() == {
        "mode": "FULL_DECODE_ONLY",
        "component": "target",
        "tp_rank": 1,
        "dataset_request": 12,
        "generated_prefix": [17, 23, 42],
        "speculative_iteration": 9,
        "draft_substep": None,
        "descriptor": 32,
        "actual_tokens": 16,
        "active_rows": [0, 3],
        "semantic_role": "final_hidden",
        "shape": [4, 8],
        "dtype": "torch.float16",
    }


def test_writer_commits_owner_only_artifact_before_manifest(tmp_path):
    writer = numerical_probe.BoundedProbeWriter(_writer_config(tmp_path))
    tensor = torch.arange(32, dtype=torch.float16).reshape(4, 8)

    record = writer.write_tensor(_identity(), tensor)

    trace_dir = tmp_path / "trace"
    artifact = trace_dir / record["artifact"]
    manifest = trace_dir / "manifest.jsonl"
    assert stat.S_IMODE(trace_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600
    assert not list(trace_dir.glob("*.tmp"))
    line = json.loads(manifest.read_text().strip())
    assert line == record
    assert line["complete"] is True
    assert line["identity"] == _identity().to_manifest_dict()
    loaded = numerical_probe.load_probe_records(trace_dir)
    torch.testing.assert_close(loaded[0].tensor, tensor)
    assert loaded[0].identity == _identity()


def test_writer_rejects_iteration_record_and_byte_overflow_atomically(tmp_path):
    identity = _identity()
    tensor = torch.arange(32, dtype=torch.float16).reshape(4, 8)

    iteration_writer = numerical_probe.BoundedProbeWriter(_writer_config(tmp_path / "iteration", max_iterations=9))
    with pytest.raises(numerical_probe.FdoNumericalProbeLimitError):
        iteration_writer.write_tensor(replace(identity, speculative_iteration=9), tensor)
    assert not (tmp_path / "iteration" / "trace" / "manifest.jsonl").exists()

    record_writer = numerical_probe.BoundedProbeWriter(_writer_config(tmp_path / "records", max_records=1))
    record_writer.write_tensor(identity, tensor)
    with pytest.raises(numerical_probe.FdoNumericalProbeLimitError):
        record_writer.write_tensor(replace(identity, semantic_role="logits"), tensor)
    records_dir = tmp_path / "records" / "trace"
    assert len((records_dir / "manifest.jsonl").read_text().splitlines()) == 1
    assert len(list(records_dir.glob("*.pt"))) == 1

    byte_writer = numerical_probe.BoundedProbeWriter(_writer_config(tmp_path / "bytes", max_bytes=64))
    with pytest.raises(numerical_probe.FdoNumericalProbeLimitError):
        byte_writer.write_tensor(identity, tensor)
    bytes_dir = tmp_path / "bytes" / "trace"
    assert not (bytes_dir / "manifest.jsonl").exists()
    assert not list(bytes_dir.glob("*.pt"))
    assert not list(bytes_dir.glob("*.tmp"))


def test_loader_fails_closed_for_truncated_or_missing_artifacts(tmp_path):
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir(mode=0o700)
    (trace_dir / "manifest.jsonl").write_text('{"complete": true')
    with pytest.raises(numerical_probe.FdoNumericalProbeArtifactError):
        numerical_probe.load_probe_records(trace_dir)

    (trace_dir / "manifest.jsonl").write_text(
        json.dumps(
            {
                "complete": True,
                "artifact": "missing.pt",
                "artifact_bytes": 10,
                "sha256": "0" * 64,
                "identity": _identity().to_manifest_dict(),
            }
        )
        + "\n"
    )
    with pytest.raises(numerical_probe.FdoNumericalProbeArtifactError):
        numerical_probe.load_probe_records(trace_dir)


def test_comparator_excludes_padding_and_reports_logit_evidence():
    eager = torch.tensor(
        [
            [4.0, 3.0, 1.0, 0.0],
            [900.0, 900.0, 900.0, 900.0],
            [1.0, 2.0, 3.0, 4.0],
            [800.0, 800.0, 800.0, 800.0],
        ]
    )
    fdo = torch.tensor(
        [
            [4.0, 2.0, 1.0, 0.0],
            [-900.0, -900.0, -900.0, -900.0],
            [1.0, 2.0, 4.0, 3.0],
            [-800.0, -800.0, -800.0, -800.0],
        ]
    )
    eager_identity = _identity(
        mode="NONE",
        semantic_role="logits",
        shape=(4, 4),
        dtype="torch.float32",
        active_rows=(0, 2),
    )
    fdo_identity = replace(eager_identity, mode="FULL_DECODE_ONLY")

    result = numerical_probe.compare_probe_tensors(
        eager_identity,
        eager,
        fdo_identity,
        fdo,
        topk=1,
    )

    assert result.active_shape == (2, 4)
    assert result.exact_unequal_count == 3
    assert result.eager_all_finite is True
    assert result.fdo_all_finite is True
    assert result.max_abs_difference == pytest.approx(1.0)
    assert result.mean_abs_difference == pytest.approx(0.375)
    assert result.max_relative_difference == pytest.approx(1 / 3)
    assert result.mean_relative_difference == pytest.approx(5 / 48)
    assert 0.9 < result.cosine_similarity < 1.0
    assert result.topk_overlap_mean == pytest.approx(0.5)
    assert result.eager_selected_token_ids == (0, 3)
    assert result.fdo_selected_token_ids == (0, 2)
    assert result.eager_selected_logits_in_fdo == pytest.approx((4.0, 3.0))
    assert result.fdo_selected_logits_in_eager == pytest.approx((4.0, 3.0))
    assert result.eager_argmax_margins == pytest.approx((1.0, 1.0))
    assert result.fdo_argmax_margins == pytest.approx((2.0, 1.0))


def test_comparator_ignores_all_inactive_lane_differences():
    eager = torch.zeros((4, 3))
    fdo = eager.clone()
    fdo[1] = 99
    fdo[3] = -99
    eager_identity = _identity(
        mode="NONE",
        semantic_role="hidden",
        shape=(4, 3),
        dtype="torch.float32",
        active_rows=(0, 2),
    )

    result = numerical_probe.compare_probe_tensors(
        eager_identity,
        eager,
        replace(eager_identity, mode="FULL_DECODE_ONLY"),
        fdo,
    )

    assert result.exact_unequal_count == 0
    assert result.max_abs_difference == 0.0
    assert result.cosine_similarity == 1.0
    assert result.topk_overlap_mean is None


def test_comparator_reports_nonfinite_active_values_without_numeric_claims():
    eager = torch.ones((4, 3))
    fdo = eager.clone()
    fdo[0, 1] = torch.nan
    eager_identity = _identity(
        mode="NONE",
        semantic_role="hidden",
        shape=(4, 3),
        dtype="torch.float32",
        active_rows=(0, 2),
    )

    result = numerical_probe.compare_probe_tensors(
        eager_identity,
        eager,
        replace(eager_identity, mode="FULL_DECODE_ONLY"),
        fdo,
    )

    assert result.eager_all_finite is True
    assert result.fdo_all_finite is False
    assert result.exact_unequal_count == 1
    assert result.max_abs_difference is None
    assert result.mean_abs_difference is None
    assert result.cosine_similarity is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("component", "draft"),
        ("tp_rank", 0),
        ("dataset_request", 11),
        ("generated_prefix", (17, 23)),
        ("speculative_iteration", 8),
        ("draft_substep", 0),
        ("descriptor", 64),
        ("actual_tokens", 15),
        ("active_rows", (0, 2)),
        ("semantic_role", "logits"),
        ("shape", (4, 4)),
        ("dtype", "torch.float32"),
    ],
)
def test_comparator_rejects_unaligned_records(field, value):
    eager_identity = _identity(mode="NONE")
    fdo_identity = replace(
        eager_identity,
        mode="FULL_DECODE_ONLY",
        **{field: value},
    )

    with pytest.raises(numerical_probe.ProbeTraceAlignmentError):
        numerical_probe.compare_probe_tensors(
            eager_identity,
            torch.zeros(eager_identity.shape, dtype=torch.float16),
            fdo_identity,
            torch.zeros(
                fdo_identity.shape,
                dtype=getattr(
                    torch,
                    fdo_identity.dtype.removeprefix("torch."),
                ),
            ),
        )


def test_target_boundary_probe_records_post_model_active_evidence(tmp_path):
    config = _writer_config(
        tmp_path,
        component="target",
        dataset_request=12,
        max_records=12,
    )
    probe = numerical_probe.TargetBoundaryProbe(config)

    probe.record_after_model(
        tp_rank=1,
        generated_prefix=(101, 102, 103),
        input_ids=torch.tensor([11, 12, 0, 0], dtype=torch.int32),
        positions=torch.tensor([41, 42, 0, 0], dtype=torch.int64),
        sample_indices=torch.tensor([0, 1], dtype=torch.int64),
        selected_hidden=torch.arange(6, dtype=torch.float16).reshape(2, 3),
        logits=torch.tensor(
            [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]],
            dtype=torch.float32,
        ),
        descriptor=4,
        actual_tokens=2,
        runtime_mode=CUDAGraphMode.FULL,
    )

    records = numerical_probe.load_probe_records(tmp_path / "trace" / "rank1")
    by_role = {record.identity.semantic_role: record for record in records}
    assert set(by_role) == {
        "input_ids",
        "positions",
        "sample_indices",
        "selected_hidden",
        "logits",
        "graph_runtime",
    }
    assert all(record.identity.tp_rank == 1 for record in records)
    assert all(record.identity.dataset_request == 12 for record in records)
    assert all(record.identity.generated_prefix == (101, 102, 103) for record in records)
    assert all(record.identity.speculative_iteration == 0 for record in records)
    assert all(record.identity.descriptor == 4 for record in records)
    assert all(record.identity.actual_tokens == 2 for record in records)
    assert by_role["input_ids"].identity.active_rows == (0, 1)
    assert by_role["positions"].identity.active_rows == (0, 1)
    assert by_role["selected_hidden"].identity.active_rows == (0, 1)
    assert by_role["logits"].identity.active_rows == (0, 1)
    assert by_role["graph_runtime"].tensor.tolist() == [[1, 4, 2]]


def test_target_runner_delegates_only_enabled_speculative_runtime(monkeypatch):
    runner = object.__new__(model_runner_v1.NPUModelRunner)
    recorder = SimpleNamespace(record_after_model=Mock())
    runner._fdo_target_numerical_probe = recorder
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.requests = {"req-0": SimpleNamespace(output_token_ids=[101, 102, 103])}
    monkeypatch.setattr(
        model_runner_v1,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )
    tensors = {
        "input_ids": torch.tensor([11, 12, 0, 0], dtype=torch.int32),
        "positions": torch.tensor([41, 42, 0, 0], dtype=torch.int64),
        "sample_indices": torch.tensor([0, 1], dtype=torch.int64),
        "selected_hidden": torch.zeros((2, 3), dtype=torch.float16),
        "logits": torch.zeros((2, 5), dtype=torch.float32),
    }

    runner._record_target_numerical_probe(
        **tensors,
        descriptor=4,
        actual_tokens=2,
        runtime_mode=CUDAGraphMode.FULL,
        spec_decode_metadata=object(),
    )

    recorder.record_after_model.assert_called_once_with(
        tp_rank=1,
        generated_prefix=(101, 102, 103),
        descriptor=4,
        actual_tokens=2,
        runtime_mode=CUDAGraphMode.FULL,
        **tensors,
    )

    runner._fdo_target_numerical_probe = None
    runner.input_batch = None
    runner.requests = None
    runner._record_target_numerical_probe(
        input_ids=object(),
        positions=object(),
        sample_indices=object(),
        selected_hidden=object(),
        logits=object(),
        descriptor=4,
        actual_tokens=2,
        runtime_mode=CUDAGraphMode.FULL,
        spec_decode_metadata=object(),
    )


def test_target_runner_identifies_a_multi_request_batch(monkeypatch):
    runner = object.__new__(model_runner_v1.NPUModelRunner)
    recorder = SimpleNamespace(record_after_model=Mock())
    runner._fdo_target_numerical_probe = recorder
    runner.input_batch = SimpleNamespace(req_ids=["req-0", "req-1"])
    runner.requests = {
        "req-0": SimpleNamespace(output_token_ids=[101]),
        "req-1": SimpleNamespace(output_token_ids=[201, 202]),
    }
    monkeypatch.setattr(
        model_runner_v1,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )

    runner._record_target_numerical_probe(
        input_ids=torch.arange(4, dtype=torch.int32),
        positions=torch.arange(4, dtype=torch.int32),
        sample_indices=torch.tensor([1, 3], dtype=torch.int64),
        selected_hidden=torch.zeros((2, 3), dtype=torch.float16),
        logits=torch.zeros((2, 5), dtype=torch.float32),
        descriptor=4,
        actual_tokens=4,
        runtime_mode=CUDAGraphMode.FULL,
        spec_decode_metadata=object(),
    )

    call = recorder.record_after_model.call_args.kwargs
    assert call["generated_prefix"] == (101, -1, 201, 202, -1)


def test_target_runner_delegates_target_layer_capture(monkeypatch):
    runner = object.__new__(model_runner_v1.NPUModelRunner)
    layer_recorder = SimpleNamespace(record_after_model=Mock())
    runner._fdo_target_numerical_probe = None
    runner._fdo_target_layer_probe = layer_recorder
    runner.input_batch = SimpleNamespace(req_ids=["req-0", "req-1"])
    runner.requests = {
        "req-0": SimpleNamespace(output_token_ids=[101]),
        "req-1": SimpleNamespace(output_token_ids=[201, 202]),
    }
    monkeypatch.setattr(
        model_runner_v1,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )

    runner._record_target_numerical_probe(
        input_ids=torch.arange(4, dtype=torch.int32),
        positions=torch.arange(4, dtype=torch.int32),
        sample_indices=torch.tensor([1, 3], dtype=torch.int64),
        selected_hidden=torch.zeros((2, 3), dtype=torch.float16),
        logits=torch.zeros((2, 5), dtype=torch.float32),
        descriptor=4,
        actual_tokens=4,
        runtime_mode=CUDAGraphMode.FULL,
        spec_decode_metadata=object(),
    )

    layer_recorder.record_after_model.assert_called_once_with(
        tp_rank=1,
        generated_prefix=(101, -1, 201, 202, -1),
        descriptor=4,
        actual_tokens=4,
        runtime_mode=CUDAGraphMode.FULL,
    )


def test_target_runner_records_layer_probe_during_prefill_without_spec_metadata(
    monkeypatch,
):
    """Catch the prefill layer trace being dropped by a SPEC-only early return."""
    runner = object.__new__(model_runner_v1.NPUModelRunner)
    boundary_recorder = SimpleNamespace(record_after_model=Mock())
    layer_recorder = SimpleNamespace(record_after_model=Mock())
    runner._fdo_target_numerical_probe = boundary_recorder
    runner._fdo_target_layer_probe = layer_recorder
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.requests = {"req-0": SimpleNamespace(output_token_ids=[])}
    monkeypatch.setattr(
        model_runner_v1,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )

    runner._record_target_numerical_probe(
        input_ids=torch.arange(82, dtype=torch.int32),
        positions=torch.arange(82, dtype=torch.int32),
        sample_indices=torch.tensor([81], dtype=torch.int64),
        selected_hidden=torch.zeros((1, 3), dtype=torch.float16),
        logits=torch.zeros((1, 5), dtype=torch.float32),
        descriptor=160,
        actual_tokens=82,
        runtime_mode=CUDAGraphMode.PIECEWISE,
        spec_decode_metadata=None,
    )

    boundary_recorder.record_after_model.assert_not_called()
    layer_recorder.record_after_model.assert_called_once_with(
        tp_rank=0,
        generated_prefix=(),
        descriptor=160,
        actual_tokens=82,
        runtime_mode=CUDAGraphMode.PIECEWISE,
    )


class _TargetProbeNorm(torch.nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states + 0.1
        return hidden_states + 0.1, residual + 0.2


class _TargetProbeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_v_heads = 1
        self.tp_size = 1
        self.head_v_dim = 3
        self.in_proj_qkvz = _TargetProbeProjection(3, 6, 0.1)
        self.in_proj_ba = _TargetProbeProjection(3, 2, 0.2)
        self.norm = _TargetProbeGateNorm()
        self.out_proj = _TargetProbeProjection(3, 3, 0.4)

    def forward(self, hidden_states, output):
        qkvz, _ = self.in_proj_qkvz(hidden_states)
        self.in_proj_ba(hidden_states)
        core = qkvz[:, :3]
        gate = qkvz[:, 3:]
        normalized = self.norm(core, gate)
        projected, _ = self.out_proj(normalized)
        output.copy_(projected)


class _TargetProbeProjection(torch.nn.Module):
    def __init__(self, input_width: int, output_width: int, offset: float):
        super().__init__()
        self.input_size_per_partition = input_width
        self.output_size_per_partition = output_width
        self.offset = offset

    def forward(self, hidden_states):
        repeats = (self.output_size_per_partition + hidden_states.shape[1] - 1) // hidden_states.shape[1]
        output = hidden_states.repeat(1, repeats)[:, : self.output_size_per_partition]
        return output + self.offset, None


class _TargetProbeGateNorm(torch.nn.Module):
    def forward(self, core, gate):
        return core + gate


class _TargetProbeLayer(torch.nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset
        self.input_layernorm = _TargetProbeNorm()
        self.linear_attn = _TargetProbeAttention()
        self.post_attention_layernorm = _TargetProbeNorm()
        self.mlp = _DraftProbeTensorOp(offset + 0.5)

    def forward(self, hidden_states, residual=None):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        attention_output = torch.empty_like(hidden_states)
        self.linear_attn(hidden_states=hidden_states, output=attention_output)
        hidden_states, residual = self.post_attention_layernorm(attention_output, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + self.offset, residual


def test_target_layer_probe_records_all_selected_layer_boundaries(tmp_path):
    config = _writer_config(
        tmp_path,
        component="target_layer",
        layer="all",
        dataset_request=12,
        max_records=40,
        max_bytes=1024 * 1024,
    )
    probe = numerical_probe.TargetLayerProbe(
        config,
        max_num_tokens=4,
        hidden_size=3,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    layers = torch.nn.ModuleList([_TargetProbeLayer(1), _TargetProbeLayer(2)])
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.decoder = torch.nn.Module()
    model.model.language_model.decoder.layers = layers
    probe.bind(model)
    hidden = torch.arange(12, dtype=torch.float16).reshape(4, 3)
    output0, residual0 = layers[0](hidden)
    layers[1](output0, residual0)

    probe.record_after_model(
        tp_rank=0,
        generated_prefix=(101, -1, 201, -1),
        descriptor=4,
        actual_tokens=2,
        runtime_mode=CUDAGraphMode.FULL,
    )

    records = numerical_probe.load_probe_records(tmp_path / "trace" / "rank0")
    by_role = {record.identity.semantic_role: record for record in records}
    assert set(by_role) == {
        "target_layer.0.input",
        "target_layer.0.input_norm.hidden",
        "target_layer.0.attention.input",
        "target_layer.0.attention.qkvz",
        "target_layer.0.attention.ba",
        "target_layer.0.attention.core",
        "target_layer.0.attention.gate",
        "target_layer.0.attention.norm",
        "target_layer.0.attention.out_proj.input",
        "target_layer.0.attention.out_proj.output",
        "target_layer.0.attention.output",
        "target_layer.0.post_norm.hidden",
        "target_layer.0.post_norm.residual",
        "target_layer.0.mlp.output",
        "target_layer.0.output",
        "target_layer.0.residual",
        "target_layer.1.input",
        "target_layer.1.input_norm.hidden",
        "target_layer.1.input_norm.residual",
        "target_layer.1.attention.input",
        "target_layer.1.attention.qkvz",
        "target_layer.1.attention.ba",
        "target_layer.1.attention.core",
        "target_layer.1.attention.gate",
        "target_layer.1.attention.norm",
        "target_layer.1.attention.out_proj.input",
        "target_layer.1.attention.out_proj.output",
        "target_layer.1.attention.output",
        "target_layer.1.post_norm.hidden",
        "target_layer.1.post_norm.residual",
        "target_layer.1.mlp.output",
        "target_layer.1.output",
        "target_layer.1.residual",
        "graph_runtime",
    }
    torch.testing.assert_close(by_role["target_layer.0.input"].tensor, hidden)
    torch.testing.assert_close(by_role["target_layer.0.output"].tensor, output0)
    assert by_role["target_layer.1.output"].identity.active_rows == (0, 1)
    assert by_role["graph_runtime"].tensor.tolist() == [[1, 4, 2]]


def test_draft_boundary_probe_records_inputs_and_remapped_proposals(tmp_path):
    config = _writer_config(
        tmp_path,
        component="draft",
        dataset_request=12,
        max_records=12,
    )
    probe = numerical_probe.DraftBoundaryProbe(config)

    probe.record_after_model(
        tp_rank=0,
        generated_prefix=(101, 102, 103),
        target_token_ids=torch.tensor([103], dtype=torch.int32),
        target_positions=torch.tensor([44], dtype=torch.int64),
        target_hidden=torch.arange(3, dtype=torch.float16).reshape(1, 3),
        next_token_ids=torch.tensor([[103, -1]], dtype=torch.int32),
        proposed_token_ids=torch.tensor(
            [[501, 502, 503]],
            dtype=torch.int64,
        ),
        descriptor=16,
        actual_tokens=1,
        runtime_mode=CUDAGraphMode.FULL,
    )

    records = numerical_probe.load_probe_records(tmp_path / "trace" / "rank0")
    by_role = {record.identity.semantic_role: record for record in records}
    assert set(by_role) == {
        "target_token_ids",
        "target_positions",
        "target_hidden_input",
        "next_token_ids",
        "proposed_token_ids",
        "graph_runtime",
    }
    assert all(record.identity.component == "draft" for record in records)
    assert all(record.identity.dataset_request == 12 for record in records)
    assert all(record.identity.generated_prefix == (101, 102, 103) for record in records)
    assert all(record.identity.draft_substep is None for record in records)
    assert by_role["proposed_token_ids"].tensor.tolist() == [[501, 502, 503]]
    assert by_role["graph_runtime"].tensor.tolist() == [[1, 16, 1]]


def test_rejection_loop_probe_records_accepted_token_contract(tmp_path):
    config = _writer_config(
        tmp_path,
        component="rejection",
        dataset_request=12,
        max_records=8,
    )
    probe = numerical_probe.RejectionLoopProbe(config)

    probe.record_after_sample(
        tp_rank=0,
        generated_prefix=(101, 102, 103),
        draft_token_ids=torch.tensor([501, 502, 601], dtype=torch.int32),
        num_draft_tokens=torch.tensor([2, 1], dtype=torch.int32),
        sampled_token_ids=torch.tensor(
            [[501, 901, -1], [701, -1, -1]],
            dtype=torch.int32,
        ),
        valid_sampled_token_count=torch.tensor([2, 1], dtype=torch.int64),
        descriptor=16,
        actual_tokens=5,
    )

    records = numerical_probe.load_probe_records(tmp_path / "trace" / "rank0")
    by_role = {record.identity.semantic_role: record for record in records}
    assert set(by_role) == {
        "draft_token_ids",
        "num_draft_tokens",
        "sampled_token_ids",
        "valid_sampled_token_count",
    }
    assert all(record.identity.component == "rejection" for record in records)
    assert all(record.identity.generated_prefix == (101, 102, 103) for record in records)
    assert by_role["draft_token_ids"].tensor.tolist() == [501, 502, 601]
    assert by_role["num_draft_tokens"].tensor.tolist() == [2, 1]
    assert by_role["sampled_token_ids"].tensor.tolist() == [
        [501, 901, -1],
        [701, -1, -1],
    ]
    assert by_role["valid_sampled_token_count"].tensor.tolist() == [2, 1]


def test_model_runner_rejection_probe_is_noop_when_disabled():
    runner = object.__new__(model_runner_v1.NPUModelRunner)
    runner._fdo_rejection_numerical_probe = None

    runner._record_rejection_numerical_probe(
        sampled_token_ids=object(),
        spec_decode_metadata=object(),
        descriptor=16,
        actual_tokens=5,
    )


def test_draft_proposer_delegates_only_when_probe_is_enabled(monkeypatch):
    proposer = object.__new__(llm_base_proposer.AscendSpecDecodeBaseProposer)
    recorder = SimpleNamespace(record_after_model=Mock())
    proposer._fdo_draft_numerical_probe = recorder
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["req-0"]),
        requests={"req-0": SimpleNamespace(output_token_ids=[101, 102, 103])},
    )
    monkeypatch.setattr(
        llm_base_proposer,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )
    tensors = {
        "target_token_ids": torch.tensor([103], dtype=torch.int32),
        "target_positions": torch.tensor([44], dtype=torch.int64),
        "target_hidden": torch.zeros((1, 3), dtype=torch.float16),
        "next_token_ids": torch.tensor([[103, -1]], dtype=torch.int32),
        "proposed_token_ids": torch.tensor([[501, 502, 503]]),
    }

    proposer._record_draft_numerical_probe(
        **tensors,
        descriptor=16,
        actual_tokens=1,
        runtime_mode=CUDAGraphMode.FULL,
    )

    recorder.record_after_model.assert_called_once_with(
        tp_rank=0,
        generated_prefix=(101, 102, 103),
        descriptor=16,
        actual_tokens=1,
        runtime_mode=CUDAGraphMode.FULL,
        **tensors,
    )

    proposer._fdo_draft_numerical_probe = None
    proposer.runner = None
    proposer._record_draft_numerical_probe(
        target_token_ids=object(),
        target_positions=object(),
        target_hidden=object(),
        next_token_ids=object(),
        proposed_token_ids=object(),
        descriptor=16,
        actual_tokens=1,
        runtime_mode=CUDAGraphMode.FULL,
    )


class _DraftProbeTensorOp(torch.nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states + self.offset


class _DraftProbeTupleOp(torch.nn.Module):
    def __init__(self, hidden_offset: float, residual_offset: float):
        super().__init__()
        self.hidden_offset = hidden_offset
        self.residual_offset = residual_offset

    def forward(self, hidden_states, residual):
        return hidden_states + self.hidden_offset, residual + self.residual_offset


class _DraftProbeTupleReturningNorm(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 0.25, hidden_states + 0.5


class _DraftProbeRotaryOp(torch.nn.Module):
    def forward(self, positions, query, key):
        return query + 0.6, key + 0.7


class _DraftProbeAttentionKernel(torch.nn.Module):
    def forward(self, query, key, value):
        return query + key + value


class _DraftProbeOutputProjection(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 0.8, None


class _DraftProbeAttention(torch.nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.q_size = 3
        self.kv_size = 3
        self.q_norm = _DraftProbeTensorOp(offset + 0.1)
        self.k_norm = _DraftProbeTensorOp(offset + 0.2)
        self.rotary_emb = _DraftProbeRotaryOp()
        self.attn = _DraftProbeAttentionKernel()
        self.o_proj = _DraftProbeOutputProjection()

    def forward(self, hidden_states, *args, **kwargs):
        query = self.q_norm(hidden_states)
        key = self.k_norm(hidden_states)
        positions = torch.arange(hidden_states.shape[0])
        query, key = self.rotary_emb(positions, query, key)
        output = self.attn(query, key, hidden_states)
        output, _ = self.o_proj(output)
        return output


class _DraftProbeLayer(torch.nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset
        self.input_layernorm = _DraftProbeTensorOp(offset + 0.1)
        self.self_attn = _DraftProbeAttention(offset + 0.2)
        self.post_attention_layernorm = _DraftProbeTupleOp(
            offset + 0.3,
            offset + 0.4,
        )
        self.mlp = _DraftProbeTensorOp(offset + 0.5)

    def forward(self, hidden_states, residual):
        self.input_layernorm(hidden_states)
        self.self_attn(hidden_states)
        self.post_attention_layernorm(hidden_states, residual)
        self.mlp(hidden_states)
        return hidden_states + self.offset, residual + self.offset * 10


def test_draft_layer_probe_records_selected_layer_hidden_and_residual(tmp_path):
    config = _writer_config(
        tmp_path,
        component="layer",
        layer="0,2",
        dataset_request=12,
        max_records=64,
        max_bytes=1024 * 1024,
    )
    probe = numerical_probe.DraftLayerProbe(
        config,
        max_num_tokens=8,
        hidden_size=3,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    layers = torch.nn.ModuleList([_DraftProbeLayer(1), _DraftProbeLayer(2), _DraftProbeLayer(3)])
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    probe.bind(model)

    # The same rotary module is also used by DFlash context-KV precompute.
    # Calls outside the selected decoder layer must not enter the layer probe.
    layers[0].self_attn.rotary_emb(
        torch.arange(2),
        torch.zeros((2, 1), dtype=torch.float16),
        torch.zeros((2, 1), dtype=torch.float16),
    )

    embedded = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    hidden = embedded
    residual = hidden + 100
    for layer in layers:
        hidden, residual = layer(hidden, residual)
    probe.record_after_model(
        tp_rank=0,
        generated_prefix=(101, 102),
        draft_input_ids=torch.tensor([7, 8], dtype=torch.int32),
        draft_positions=torch.tensor([21, 22], dtype=torch.int32),
        draft_embeddings=embedded,
        descriptor=2,
        actual_tokens=2,
        runtime_mode=CUDAGraphMode.FULL,
    )

    records = numerical_probe.load_probe_records(tmp_path / "trace" / "rank0")
    by_role = {record.identity.semantic_role: record for record in records}
    assert set(by_role) == {
        "draft_layer.0.input",
        "draft_layer.0.input_norm",
        "draft_layer.0.attention",
        "draft_layer.0.q_norm.input",
        "draft_layer.0.q_norm.output",
        "draft_layer.0.k_norm.input",
        "draft_layer.0.k_norm.output",
        "draft_layer.0.rope.positions",
        "draft_layer.0.rope.q",
        "draft_layer.0.rope.k",
        "draft_layer.0.attn.q",
        "draft_layer.0.attn.k",
        "draft_layer.0.attn.v",
        "draft_layer.0.attn.output",
        "draft_layer.0.o_proj.output",
        "draft_layer.0.post_norm.hidden",
        "draft_layer.0.post_norm.residual",
        "draft_layer.0.mlp",
        "draft_layer.0.hidden",
        "draft_layer.0.residual",
        "draft_layer.2.input",
        "draft_layer.2.input_norm",
        "draft_layer.2.attention",
        "draft_layer.2.q_norm.input",
        "draft_layer.2.q_norm.output",
        "draft_layer.2.k_norm.input",
        "draft_layer.2.k_norm.output",
        "draft_layer.2.rope.positions",
        "draft_layer.2.rope.q",
        "draft_layer.2.rope.k",
        "draft_layer.2.attn.q",
        "draft_layer.2.attn.k",
        "draft_layer.2.attn.v",
        "draft_layer.2.attn.output",
        "draft_layer.2.o_proj.output",
        "draft_layer.2.post_norm.hidden",
        "draft_layer.2.post_norm.residual",
        "draft_layer.2.mlp",
        "draft_layer.2.hidden",
        "draft_layer.2.residual",
        "draft_input.input_ids",
        "draft_input.positions",
        "draft_input.embeddings",
        "graph_runtime",
    }
    torch.testing.assert_close(by_role["draft_input.embeddings"].tensor, embedded)
    torch.testing.assert_close(by_role["draft_layer.0.input"].tensor, embedded)
    torch.testing.assert_close(
        by_role["draft_layer.0.input_norm"].tensor,
        embedded + 1.1,
    )
    torch.testing.assert_close(
        by_role["draft_layer.0.attention"].tensor,
        embedded * 3 + 4.8,
    )
    torch.testing.assert_close(
        by_role["draft_layer.0.post_norm.hidden"].tensor,
        embedded + 1.3,
    )
    torch.testing.assert_close(
        by_role["draft_layer.0.post_norm.residual"].tensor,
        embedded + 101.4,
    )
    torch.testing.assert_close(
        by_role["draft_layer.0.mlp"].tensor,
        embedded + 1.5,
    )
    torch.testing.assert_close(by_role["draft_layer.0.hidden"].tensor, hidden - 5)
    torch.testing.assert_close(
        by_role["draft_layer.0.residual"].tensor,
        residual - 50,
    )
    torch.testing.assert_close(by_role["draft_layer.2.input"].tensor, embedded + 3)
    torch.testing.assert_close(by_role["draft_layer.2.hidden"].tensor, hidden)
    torch.testing.assert_close(by_role["draft_layer.2.residual"].tensor, residual)
    assert all(record.identity.component == "layer" for record in records)
    assert all(record.identity.active_rows == (0, 1) for role, record in by_role.items() if role != "graph_runtime")
    assert by_role["graph_runtime"].tensor.tolist() == [[1, 2, 2]]


def test_draft_layer_probe_records_only_runtime_context_rows(tmp_path):
    config = _writer_config(
        tmp_path,
        component="layer",
        layer="0",
        dataset_request=12,
        max_records=64,
        max_bytes=1024 * 1024,
    )
    probe = numerical_probe.DraftLayerProbe(
        config,
        max_num_tokens=8,
        hidden_size=3,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    layers = torch.nn.ModuleList([_DraftProbeLayer(1)])
    draft_model = SimpleNamespace(layers=layers)
    model = SimpleNamespace(model=draft_model)
    probe.bind(model)

    assert draft_model._fdo_context_probe is probe
    captured_context_states = torch.arange(12, dtype=torch.float16).reshape(4, 3)
    context_states = captured_context_states[:2]
    normed_captured_context_states = captured_context_states + 10
    normed_context_states = context_states + 10
    captured_context_positions = torch.tensor([21, 22, 23, 24], dtype=torch.int32)
    context_positions = captured_context_positions[:2]
    captured_slot_mapping = torch.tensor([31, 32, 33, 34], dtype=torch.int32)
    slot_mapping = captured_slot_mapping[:2]
    captured_k_norm_input = captured_context_states + 20
    captured_k_norm_output = captured_context_states + 30
    captured_k_rope = captured_context_states + 40
    captured_value = captured_context_states + 50
    k_norm_input = captured_k_norm_input[:2]
    k_norm_output = captured_k_norm_output[:2]
    k_rope = captured_k_rope[:2]
    value = captured_value[:2]
    probe.capture_context_inputs(
        context_states=captured_context_states,
        context_positions=captured_context_positions,
        normed_context_states=normed_captured_context_states,
        slot_mapping=captured_slot_mapping,
    )
    probe.capture_context_k_norm(
        layer_index=0,
        k_norm_input=captured_k_norm_input,
        k_norm_output=captured_k_norm_output,
    )
    probe.capture_context_rope(
        layer_index=0,
        k_rope=captured_k_rope,
        value=captured_value,
    )

    hidden = context_states
    residual = context_states + 100
    hidden, residual = layers[0](hidden, residual)
    probe.record_after_model(
        tp_rank=0,
        generated_prefix=(101, 102),
        draft_input_ids=torch.tensor([7, 8], dtype=torch.int32),
        draft_positions=torch.tensor([21, 22], dtype=torch.int32),
        draft_embeddings=context_states,
        descriptor=2,
        actual_tokens=2,
        context_actual_tokens=2,
        runtime_mode=CUDAGraphMode.FULL,
    )

    records = numerical_probe.load_probe_records(tmp_path / "trace" / "rank0")
    by_role = {record.identity.semantic_role: record for record in records}
    expected = {
        "draft_context.input": context_states,
        "draft_context.positions": context_positions.to(torch.int64),
        "draft_context.hidden_norm": normed_context_states,
        "draft_context.slot_mapping": slot_mapping.to(torch.int64),
        "draft_context.layer.0.k_norm.input": k_norm_input,
        "draft_context.layer.0.k_norm.output": k_norm_output,
        "draft_context.layer.0.rope.k": k_rope,
        "draft_context.layer.0.v": value,
    }
    for role, tensor in expected.items():
        torch.testing.assert_close(by_role[role].tensor, tensor)
        assert by_role[role].identity.active_rows == (0, 1)


def test_draft_layer_probe_normalizes_tuple_input_norm_output(tmp_path):
    config = _writer_config(
        tmp_path,
        component="layer",
        layer="0",
        max_records=64,
        max_bytes=1024 * 1024,
    )
    probe = numerical_probe.DraftLayerProbe(
        config,
        max_num_tokens=8,
        hidden_size=3,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    layer = _DraftProbeLayer(1)
    layer.input_layernorm = _DraftProbeTupleReturningNorm()
    model = SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([layer])))
    probe.bind(model)

    hidden = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    layer(hidden, hidden + 100)
    probe.record_after_model(
        tp_rank=0,
        generated_prefix=(),
        draft_input_ids=torch.tensor([7, 8], dtype=torch.int32),
        draft_positions=torch.tensor([21, 22], dtype=torch.int32),
        draft_embeddings=hidden,
        descriptor=2,
        actual_tokens=2,
        runtime_mode=CUDAGraphMode.NONE,
    )

    records = numerical_probe.load_probe_records(tmp_path / "trace" / "rank0")
    by_role = {record.identity.semantic_role: record for record in records}
    torch.testing.assert_close(
        by_role["draft_layer.0.input_norm"].tensor,
        hidden + 0.25,
    )


@pytest.mark.parametrize("selection", [None, "", "-1", "0,0", "a", "7"])
def test_draft_layer_probe_rejects_missing_or_invalid_selection(
    tmp_path,
    selection,
):
    config = _writer_config(
        tmp_path,
        component="layer",
        layer=selection,
    )
    probe = numerical_probe.DraftLayerProbe(
        config,
        max_num_tokens=8,
        hidden_size=3,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    model = SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([_DraftProbeLayer(1)])))

    with pytest.raises(numerical_probe.FdoNumericalProbeConfigError):
        probe.bind(model)


def test_draft_layer_probe_all_selection_is_bounded_by_model_layers(tmp_path):
    config = _writer_config(
        tmp_path,
        component="layer",
        layer="all",
    )
    probe = numerical_probe.DraftLayerProbe(
        config,
        max_num_tokens=8,
        hidden_size=3,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=torch.nn.ModuleList([_DraftProbeLayer(1), _DraftProbeLayer(2)]))
    )

    probe.bind(model)

    assert probe.selected_layers == (0, 1)


def test_draft_proposer_delegates_layer_capture_after_model(monkeypatch):
    proposer = object.__new__(llm_base_proposer.AscendSpecDecodeBaseProposer)
    recorder = SimpleNamespace(record_after_model=Mock())
    proposer._fdo_draft_layer_probe = recorder
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["req-0"]),
        requests={"req-0": SimpleNamespace(output_token_ids=[101, 102])},
    )
    proposer.input_ids = torch.arange(16, dtype=torch.int32)
    proposer.positions = torch.arange(16, dtype=torch.int32) + 20
    proposer._dflash_num_context = 7
    proposer._get_positions = Mock(return_value=proposer.positions[:16])
    draft_embeddings = torch.arange(32, dtype=torch.float16).reshape(16, 2)
    proposer.model = SimpleNamespace(embed_input_ids=Mock(return_value=draft_embeddings))
    monkeypatch.setattr(
        llm_base_proposer,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=1),
    )

    proposer._record_draft_layer_numerical_probe(
        descriptor=16,
        actual_tokens=16,
        runtime_mode=CUDAGraphMode.FULL,
    )
    recorder.record_after_model.assert_called_once()
    call = recorder.record_after_model.call_args.kwargs
    assert call["tp_rank"] == 1
    assert call["generated_prefix"] == (101, 102)
    torch.testing.assert_close(call["draft_input_ids"], proposer.input_ids[:16])
    torch.testing.assert_close(call["draft_positions"], proposer.positions[:16])
    torch.testing.assert_close(call["draft_embeddings"], draft_embeddings)
    proposer.model.embed_input_ids.assert_called_once()
    assert call["descriptor"] == 16
    assert call["actual_tokens"] == 16
    assert call["context_actual_tokens"] == 7
    assert call["runtime_mode"] == CUDAGraphMode.FULL

    proposer._fdo_draft_layer_probe = None
    proposer.runner = None
    proposer._record_draft_layer_numerical_probe(
        descriptor=16,
        actual_tokens=16,
        runtime_mode=CUDAGraphMode.FULL,
    )


def test_draft_layer_probe_identifies_a_multi_request_batch(monkeypatch):
    proposer = object.__new__(llm_base_proposer.AscendSpecDecodeBaseProposer)
    recorder = SimpleNamespace(record_after_model=Mock())
    proposer._fdo_draft_layer_probe = recorder
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["req-0", "req-1"]),
        requests={
            "req-0": SimpleNamespace(output_token_ids=[101]),
            "req-1": SimpleNamespace(output_token_ids=[201, 202]),
        },
    )
    proposer.input_ids = torch.arange(32, dtype=torch.int32)
    proposer.positions = torch.arange(32, dtype=torch.int32) + 20
    proposer._get_positions = Mock(return_value=proposer.positions)
    proposer.model = SimpleNamespace(
        embed_input_ids=Mock(return_value=torch.arange(64, dtype=torch.float16).reshape(32, 2))
    )
    monkeypatch.setattr(
        llm_base_proposer,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )

    proposer._record_draft_layer_numerical_probe(
        descriptor=32,
        actual_tokens=32,
        runtime_mode=CUDAGraphMode.FULL,
    )

    call = recorder.record_after_model.call_args.kwargs
    assert call["generated_prefix"] == (101, -1, 201, 202, -1)
