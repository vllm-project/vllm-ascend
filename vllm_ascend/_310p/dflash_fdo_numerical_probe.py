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

"""Bounded, opt-in numerical probes for 310P DFlash graph diagnosis."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig

from vllm_ascend import envs
from vllm_ascend.utils import is_310p

PROBE_DIR_ENV = envs.FDO_PROBE_DIR_ENV
PROBE_COMPONENT_ENV = envs.FDO_PROBE_COMPONENT_ENV
PROBE_LAYER_ENV = envs.FDO_PROBE_LAYER_ENV
PROBE_DATASET_REQUEST_ENV = envs.FDO_PROBE_DATASET_REQUEST_ENV
PROBE_MAX_ITERATIONS_ENV = envs.FDO_PROBE_MAX_ITERATIONS_ENV
PROBE_MAX_RECORDS_ENV = envs.FDO_PROBE_MAX_RECORDS_ENV
PROBE_MAX_BYTES_ENV = envs.FDO_PROBE_MAX_BYTES_ENV

_COMPONENTS = frozenset({"boundary", "target", "target_layer", "draft", "rejection", "layer"})
_COMPARABLE_MODES = frozenset(
    {
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    }
)


class FdoNumericalProbeConfigError(ValueError):
    """Raised when an explicitly enabled probe is unsafe or out of scope."""


class FdoNumericalProbeLimitError(RuntimeError):
    """Raised before a probe would cross a configured resource bound."""


class FdoNumericalProbeArtifactError(RuntimeError):
    """Raised when an artifact set is incomplete, corrupt, or inconsistent."""


class ProbeTraceAlignmentError(ValueError):
    """Raised rather than comparing tensors from different logical work."""


def _positive_int(environ: Mapping[str, str], key: str, default: int) -> int:
    raw_value = environ.get(key, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise FdoNumericalProbeConfigError(f"{key} must be a positive integer, got {raw_value!r}") from exc
    if value <= 0:
        raise FdoNumericalProbeConfigError(f"{key} must be a positive integer, got {raw_value!r}")
    return value


def _central_probe_environ() -> dict[str, str]:
    values = {
        PROBE_DIR_ENV: envs.VLLM_ASCEND_310P_DFLASH_FDO_PROBE_DIR,
        PROBE_COMPONENT_ENV: envs.VLLM_ASCEND_310P_DFLASH_FDO_PROBE_COMPONENT,
        PROBE_LAYER_ENV: envs.VLLM_ASCEND_310P_DFLASH_FDO_PROBE_LAYER,
        PROBE_DATASET_REQUEST_ENV: envs.VLLM_ASCEND_310P_DFLASH_FDO_PROBE_DATASET_REQUEST,
        PROBE_MAX_ITERATIONS_ENV: envs.VLLM_ASCEND_310P_DFLASH_FDO_PROBE_MAX_ITERATIONS,
        PROBE_MAX_RECORDS_ENV: envs.VLLM_ASCEND_310P_DFLASH_FDO_PROBE_MAX_RECORDS,
        PROBE_MAX_BYTES_ENV: envs.VLLM_ASCEND_310P_DFLASH_FDO_PROBE_MAX_BYTES,
    }
    return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class FdoNumericalProbeConfig:
    """Immutable probe settings resolved once during runner construction."""

    enabled: bool
    output_dir: Path | None
    mode: str | None
    component: str = "boundary"
    layer: str | None = None
    dataset_request: int = 0
    max_iterations: int = 64
    max_records: int = 256
    max_bytes: int = 128 * 1024 * 1024

    @classmethod
    def from_environ(
        cls,
        vllm_config: VllmConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> FdoNumericalProbeConfig:
        values = _central_probe_environ() if environ is None else environ
        output_dir_raw = values.get(PROBE_DIR_ENV)
        if not output_dir_raw:
            return cls(enabled=False, output_dir=None, mode=None)

        speculative_config = vllm_config.speculative_config
        mode = vllm_config.compilation_config.cudagraph_mode
        if (
            not is_310p()
            or speculative_config is None
            or speculative_config.method != "dflash"
            or mode not in _COMPARABLE_MODES
        ):
            raise FdoNumericalProbeConfigError(
                "the numerical probe is restricted to 310P DFlash Eager, "
                "PIECEWISE, FULL_DECODE_ONLY, and FULL_AND_PIECEWISE"
            )

        component = values.get(PROBE_COMPONENT_ENV, "boundary")
        if component not in _COMPONENTS:
            raise FdoNumericalProbeConfigError(
                f"unsupported probe component {component!r}; expected one of {sorted(_COMPONENTS)}"
            )

        layer = values.get(PROBE_LAYER_ENV) or None
        dataset_request_raw = values.get(PROBE_DATASET_REQUEST_ENV, "0")
        try:
            dataset_request = int(dataset_request_raw)
        except (TypeError, ValueError) as exc:
            raise FdoNumericalProbeConfigError(f"{PROBE_DATASET_REQUEST_ENV} must be a non-negative integer") from exc
        if dataset_request < 0:
            raise FdoNumericalProbeConfigError(f"{PROBE_DATASET_REQUEST_ENV} must be a non-negative integer")
        max_iterations = _positive_int(values, PROBE_MAX_ITERATIONS_ENV, 64)
        max_records = _positive_int(values, PROBE_MAX_RECORDS_ENV, 256)
        max_bytes = _positive_int(
            values,
            PROBE_MAX_BYTES_ENV,
            128 * 1024 * 1024,
        )
        if max_bytes < 1024:
            raise FdoNumericalProbeConfigError(f"{PROBE_MAX_BYTES_ENV} must be at least 1024 bytes")

        return cls(
            enabled=True,
            output_dir=Path(output_dir_raw),
            mode=mode.name,
            component=component,
            layer=layer,
            dataset_request=dataset_request,
            max_iterations=max_iterations,
            max_records=max_records,
            max_bytes=max_bytes,
        )


@dataclass(frozen=True)
class ProbeTraceIdentity:
    """Identity required to align one tensor across Eager and FDO traces."""

    mode: str
    component: str
    tp_rank: int
    dataset_request: int
    generated_prefix: tuple[int, ...]
    speculative_iteration: int
    draft_substep: int | None
    descriptor: int
    actual_tokens: int
    active_rows: tuple[int, ...]
    semantic_role: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.mode or not self.component or not self.semantic_role:
            raise FdoNumericalProbeArtifactError("mode, component, and semantic role must be non-empty")
        for field_name in (
            "tp_rank",
            "dataset_request",
            "speculative_iteration",
            "descriptor",
            "actual_tokens",
        ):
            if getattr(self, field_name) < 0:
                raise FdoNumericalProbeArtifactError(f"{field_name} must be non-negative")
        if self.draft_substep is not None and self.draft_substep < 0:
            raise FdoNumericalProbeArtifactError("draft_substep must be non-negative when present")
        if not self.shape or any(dimension < 0 for dimension in self.shape):
            raise FdoNumericalProbeArtifactError("shape must be non-empty")
        if len(set(self.active_rows)) != len(self.active_rows):
            raise FdoNumericalProbeArtifactError("active rows must be unique")
        if any(row < 0 or row >= self.shape[0] for row in self.active_rows):
            raise FdoNumericalProbeArtifactError("active rows must address the tensor's leading dimension")

    def to_manifest_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for field_name in ("generated_prefix", "active_rows", "shape"):
            result[field_name] = list(result[field_name])
        return result

    @classmethod
    def from_manifest_dict(cls, value: Mapping[str, Any]) -> ProbeTraceIdentity:
        fields = dict(value)
        for field_name in ("generated_prefix", "active_rows", "shape"):
            fields[field_name] = tuple(fields[field_name])
        try:
            return cls(**fields)
        except (KeyError, TypeError, ValueError) as exc:
            raise FdoNumericalProbeArtifactError("invalid trace identity") from exc


@dataclass(frozen=True)
class LoadedProbeRecord:
    identity: ProbeTraceIdentity
    tensor: torch.Tensor
    manifest: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for block in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(trace_dir: Path) -> list[dict[str, Any]]:
    manifest_path = trace_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    try:
        content = manifest_path.read_bytes()
    except OSError as exc:
        raise FdoNumericalProbeArtifactError(f"cannot read probe manifest {manifest_path}") from exc
    if content and not content.endswith(b"\n"):
        raise FdoNumericalProbeArtifactError("truncated probe manifest record")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise FdoNumericalProbeArtifactError(f"invalid probe manifest record at line {line_number}") from exc
        if not isinstance(record, dict) or record.get("complete") is not True:
            raise FdoNumericalProbeArtifactError(f"incomplete probe manifest record at line {line_number}")
        records.append(record)
    return records


def load_probe_records(trace_dir: str | Path) -> tuple[LoadedProbeRecord, ...]:
    """Load only complete, hash-verified probe artifacts."""
    directory = Path(trace_dir)
    loaded: list[LoadedProbeRecord] = []
    for record in _read_manifest(directory):
        try:
            artifact_name = record["artifact"]
            artifact_bytes = int(record["artifact_bytes"])
            expected_sha256 = record["sha256"]
            identity = ProbeTraceIdentity.from_manifest_dict(record["identity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FdoNumericalProbeArtifactError("probe manifest record is missing required fields") from exc
        if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name:
            raise FdoNumericalProbeArtifactError("unsafe probe artifact name")
        artifact_path = directory / artifact_name
        if not artifact_path.is_file():
            raise FdoNumericalProbeArtifactError(f"missing probe artifact {artifact_name}")
        if artifact_path.stat().st_size != artifact_bytes:
            raise FdoNumericalProbeArtifactError(f"probe artifact size mismatch for {artifact_name}")
        if _sha256_file(artifact_path) != expected_sha256:
            raise FdoNumericalProbeArtifactError(f"probe artifact hash mismatch for {artifact_name}")
        try:
            tensor = torch.load(
                artifact_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise FdoNumericalProbeArtifactError(f"cannot load probe artifact {artifact_name}") from exc
        if not isinstance(tensor, torch.Tensor):
            raise FdoNumericalProbeArtifactError(f"probe artifact {artifact_name} is not a tensor")
        if tuple(tensor.shape) != identity.shape or str(tensor.dtype) != identity.dtype:
            raise FdoNumericalProbeArtifactError(f"probe artifact identity mismatch for {artifact_name}")
        loaded.append(LoadedProbeRecord(identity, tensor, record))
    return tuple(loaded)


class BoundedProbeWriter:
    """Commit bounded tensor records with the manifest written last."""

    def __init__(self, config: FdoNumericalProbeConfig) -> None:
        if not config.enabled or config.output_dir is None:
            raise FdoNumericalProbeConfigError("a bounded writer requires an enabled probe configuration")
        self._config = config
        self._trace_dir = config.output_dir
        self._trace_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._trace_dir.chmod(0o700)
        existing = _read_manifest(self._trace_dir)
        self._records_written = len(existing)
        self._artifact_bytes = sum(int(record.get("artifact_bytes", 0)) for record in existing)

    def write_tensor(
        self,
        identity: ProbeTraceIdentity,
        tensor: torch.Tensor,
    ) -> dict[str, Any]:
        if identity.speculative_iteration >= self._config.max_iterations:
            raise FdoNumericalProbeLimitError("probe speculative-iteration bound reached")
        if self._records_written >= self._config.max_records:
            raise FdoNumericalProbeLimitError("probe record bound reached")
        if tuple(tensor.shape) != identity.shape or str(tensor.dtype) != identity.dtype:
            raise FdoNumericalProbeArtifactError("tensor shape or dtype does not match trace identity")

        canonical_identity = json.dumps(
            identity.to_manifest_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        identity_digest = hashlib.sha256(canonical_identity).hexdigest()[:16]
        artifact_name = f"{self._records_written:06d}-{identity_digest}.pt"
        artifact_path = self._trace_dir / artifact_name
        temporary_path = self._trace_dir / f".{artifact_name}.tmp"
        if artifact_path.exists() or temporary_path.exists():
            raise FdoNumericalProbeArtifactError(f"probe artifact collision for {artifact_name}")

        try:
            torch.save(tensor.detach().cpu(), temporary_path)
            temporary_path.chmod(0o600)
            artifact_bytes = temporary_path.stat().st_size
            if self._artifact_bytes + artifact_bytes > self._config.max_bytes:
                raise FdoNumericalProbeLimitError("probe byte bound reached")
            artifact_sha256 = _sha256_file(temporary_path)
            os.replace(temporary_path, artifact_path)
            artifact_path.chmod(0o600)

            record: dict[str, Any] = {
                "complete": True,
                "artifact": artifact_name,
                "artifact_bytes": artifact_bytes,
                "sha256": artifact_sha256,
                "identity": identity.to_manifest_dict(),
            }
            encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            manifest_path = self._trace_dir / "manifest.jsonl"
            file_descriptor = os.open(
                manifest_path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                written = os.write(file_descriptor, encoded)
                if written != len(encoded):
                    raise FdoNumericalProbeArtifactError("short write while committing probe manifest record")
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
            manifest_path.chmod(0o600)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            if artifact_path.exists():
                artifact_path.unlink()
            raise

        self._records_written += 1
        self._artifact_bytes += artifact_bytes
        return record


@dataclass(frozen=True)
class ProbeTensorComparison:
    active_shape: tuple[int, ...]
    exact_unequal_count: int
    eager_all_finite: bool
    fdo_all_finite: bool
    max_abs_difference: float | None
    mean_abs_difference: float | None
    max_relative_difference: float | None
    mean_relative_difference: float | None
    cosine_similarity: float | None
    topk_overlap_mean: float | None = None
    eager_selected_token_ids: tuple[int, ...] | None = None
    fdo_selected_token_ids: tuple[int, ...] | None = None
    eager_selected_logits_in_fdo: tuple[float, ...] | None = None
    fdo_selected_logits_in_eager: tuple[float, ...] | None = None
    eager_argmax_margins: tuple[float, ...] | None = None
    fdo_argmax_margins: tuple[float, ...] | None = None


def _assert_aligned_identities(
    eager_identity: ProbeTraceIdentity,
    fdo_identity: ProbeTraceIdentity,
) -> None:
    if eager_identity.mode != CUDAGraphMode.NONE.name:
        raise ProbeTraceAlignmentError(f"expected Eager mode NONE, got {eager_identity.mode!r}")
    if fdo_identity.mode != CUDAGraphMode.FULL_DECODE_ONLY.name:
        raise ProbeTraceAlignmentError(f"expected comparison mode FULL_DECODE_ONLY, got {fdo_identity.mode!r}")
    eager_fields = eager_identity.to_manifest_dict()
    fdo_fields = fdo_identity.to_manifest_dict()
    eager_fields.pop("mode")
    fdo_fields.pop("mode")
    if eager_fields != fdo_fields:
        differing = sorted(field for field in eager_fields if eager_fields.get(field) != fdo_fields.get(field))
        raise ProbeTraceAlignmentError(f"unaligned trace identities: {', '.join(differing)}")


def _argmax_evidence(
    eager: torch.Tensor,
    fdo: torch.Tensor,
    topk: int,
) -> dict[str, Any]:
    if eager.ndim != 2 or eager.shape[-1] == 0:
        raise ProbeTraceAlignmentError("logit evidence requires a non-empty two-dimensional tensor")
    if topk <= 0 or topk > eager.shape[-1]:
        raise ProbeTraceAlignmentError(f"topk must be in [1, {eager.shape[-1]}], got {topk}")
    eager_topk = torch.topk(eager, topk, dim=-1).indices
    fdo_topk = torch.topk(fdo, topk, dim=-1).indices
    overlaps = []
    for eager_row, fdo_row in zip(eager_topk, fdo_topk, strict=True):
        overlap = len(set(eager_row.tolist()) & set(fdo_row.tolist()))
        overlaps.append(overlap / topk)

    eager_selected = torch.argmax(eager, dim=-1)
    fdo_selected = torch.argmax(fdo, dim=-1)
    row_indices = torch.arange(eager.shape[0])
    eager_cross = fdo[row_indices, eager_selected]
    fdo_cross = eager[row_indices, fdo_selected]

    def _margins(logits: torch.Tensor) -> tuple[float, ...]:
        if logits.shape[-1] == 1:
            return tuple(0.0 for _ in range(logits.shape[0]))
        top_two = torch.topk(logits, 2, dim=-1).values
        return tuple((top_two[:, 0] - top_two[:, 1]).tolist())

    return {
        "topk_overlap_mean": sum(overlaps) / len(overlaps),
        "eager_selected_token_ids": tuple(eager_selected.tolist()),
        "fdo_selected_token_ids": tuple(fdo_selected.tolist()),
        "eager_selected_logits_in_fdo": tuple(eager_cross.tolist()),
        "fdo_selected_logits_in_eager": tuple(fdo_cross.tolist()),
        "eager_argmax_margins": _margins(eager),
        "fdo_argmax_margins": _margins(fdo),
    }


def compare_probe_tensors(
    eager_identity: ProbeTraceIdentity,
    eager_tensor: torch.Tensor,
    fdo_identity: ProbeTraceIdentity,
    fdo_tensor: torch.Tensor,
    *,
    topk: int = 10,
) -> ProbeTensorComparison:
    """Strictly align active lanes and compute offline numerical evidence."""
    _assert_aligned_identities(eager_identity, fdo_identity)
    for label, identity, tensor in (
        ("Eager", eager_identity, eager_tensor),
        ("FDO", fdo_identity, fdo_tensor),
    ):
        if tuple(tensor.shape) != identity.shape:
            raise ProbeTraceAlignmentError(f"{label} tensor shape does not match its trace identity")
        if str(tensor.dtype) != identity.dtype:
            raise ProbeTraceAlignmentError(f"{label} tensor dtype does not match its trace identity")

    active_indices = torch.tensor(eager_identity.active_rows, dtype=torch.long)
    eager_active = eager_tensor.detach().cpu().index_select(0, active_indices)
    fdo_active = fdo_tensor.detach().cpu().index_select(0, active_indices)
    eager_finite = bool(torch.isfinite(eager_active).all().item())
    fdo_finite = bool(torch.isfinite(fdo_active).all().item())
    unequal_count = int(torch.ne(eager_active, fdo_active).sum().item())
    common: dict[str, Any] = {
        "active_shape": tuple(eager_active.shape),
        "exact_unequal_count": unequal_count,
        "eager_all_finite": eager_finite,
        "fdo_all_finite": fdo_finite,
    }
    if not eager_finite or not fdo_finite:
        return ProbeTensorComparison(
            **common,
            max_abs_difference=None,
            mean_abs_difference=None,
            max_relative_difference=None,
            mean_relative_difference=None,
            cosine_similarity=None,
        )

    eager_float = eager_active.to(torch.float64)
    fdo_float = fdo_active.to(torch.float64)
    absolute = torch.abs(eager_float - fdo_float)
    denominator = torch.maximum(torch.abs(eager_float), torch.abs(fdo_float)).clamp_min(torch.finfo(torch.float64).eps)
    relative = absolute / denominator
    eager_flat = eager_float.flatten()
    fdo_flat = fdo_float.flatten()
    norm_product = torch.linalg.vector_norm(eager_flat) * torch.linalg.vector_norm(fdo_flat)
    if norm_product == 0:
        cosine = 1.0 if unequal_count == 0 else 0.0
    else:
        cosine = float(torch.dot(eager_flat, fdo_flat) / norm_product)

    logit_evidence: dict[str, Any] = {}
    if "logits" in eager_identity.semantic_role:
        logit_evidence = _argmax_evidence(eager_float, fdo_float, topk)
    return ProbeTensorComparison(
        **common,
        max_abs_difference=float(absolute.max().item()),
        mean_abs_difference=float(absolute.mean().item()),
        max_relative_difference=float(relative.max().item()),
        mean_relative_difference=float(relative.mean().item()),
        cosine_similarity=cosine,
        **logit_evidence,
    )


class TargetBoundaryProbe:
    """Export target inputs and outputs only after a real model invocation."""

    def __init__(self, config: FdoNumericalProbeConfig) -> None:
        if not config.enabled or config.output_dir is None:
            raise FdoNumericalProbeConfigError("target boundary probing requires an enabled configuration")
        if config.component not in {"boundary", "target"}:
            raise FdoNumericalProbeConfigError(f"target probe cannot serve component {config.component!r}")
        self._config = config
        self._writers: dict[int, BoundedProbeWriter] = {}
        self._iterations: dict[int, int] = {}
        self._exhausted_ranks: set[int] = set()

    def _writer(self, tp_rank: int) -> BoundedProbeWriter:
        writer = self._writers.get(tp_rank)
        if writer is None:
            assert self._config.output_dir is not None
            rank_config = replace(
                self._config,
                output_dir=self._config.output_dir / f"rank{tp_rank}",
            )
            writer = BoundedProbeWriter(rank_config)
            self._writers[tp_rank] = writer
        return writer

    @staticmethod
    def _token_major_positions(positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim <= 1:
            return positions
        return positions.movedim(-1, 0)

    def record_after_model(
        self,
        *,
        tp_rank: int,
        generated_prefix: tuple[int, ...],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        sample_indices: torch.Tensor,
        selected_hidden: torch.Tensor,
        logits: torch.Tensor,
        descriptor: int,
        actual_tokens: int,
        runtime_mode: CUDAGraphMode,
    ) -> None:
        if tp_rank in self._exhausted_ranks:
            return
        iteration = self._iterations.get(tp_rank, 0)
        positions = self._token_major_positions(positions)
        if input_ids.ndim == 0 or positions.ndim == 0:
            raise FdoNumericalProbeArtifactError("target inputs must expose a token dimension")
        if actual_tokens > input_ids.shape[0] or actual_tokens > positions.shape[0]:
            raise FdoNumericalProbeArtifactError("actual target tokens exceed the recorded input buffers")
        selected_rows = selected_hidden.shape[0]
        if logits.shape[0] != selected_rows or sample_indices.numel() != selected_rows:
            raise FdoNumericalProbeArtifactError("target hidden, logits, and sample-index rows must align")

        runtime_code = 1 if runtime_mode == CUDAGraphMode.FULL else 0
        runtime_identity = torch.tensor(
            [[runtime_code, descriptor, actual_tokens]],
            dtype=torch.int64,
            device=input_ids.device,
        )
        tensors = (
            ("input_ids", input_ids, tuple(range(actual_tokens))),
            ("positions", positions, tuple(range(actual_tokens))),
            (
                "sample_indices",
                sample_indices.reshape(-1),
                tuple(range(sample_indices.numel())),
            ),
            (
                "selected_hidden",
                selected_hidden,
                tuple(range(selected_rows)),
            ),
            ("logits", logits, tuple(range(selected_rows))),
            ("graph_runtime", runtime_identity, (0,)),
        )
        writer = self._writer(tp_rank)
        try:
            for semantic_role, tensor, active_rows in tensors:
                identity = ProbeTraceIdentity(
                    mode=self._config.mode or "",
                    component="target",
                    tp_rank=tp_rank,
                    dataset_request=self._config.dataset_request,
                    generated_prefix=generated_prefix,
                    speculative_iteration=iteration,
                    draft_substep=None,
                    descriptor=descriptor,
                    actual_tokens=actual_tokens,
                    active_rows=active_rows,
                    semantic_role=semantic_role,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                )
                writer.write_tensor(identity, tensor)
        except FdoNumericalProbeLimitError:
            self._exhausted_ranks.add(tp_rank)
            return
        self._iterations[tp_rank] = iteration + 1


def create_target_boundary_probe(
    vllm_config: VllmConfig,
) -> TargetBoundaryProbe | None:
    """Create the target observer only for an explicitly enabled component."""
    config = FdoNumericalProbeConfig.from_environ(vllm_config)
    if not config.enabled or config.component not in {"boundary", "target"}:
        return None
    return TargetBoundaryProbe(config)


class TargetLayerProbe:
    """Capture target decoder layer boundaries into persistent device buffers."""

    def __init__(
        self,
        config: FdoNumericalProbeConfig,
        *,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not config.enabled or config.output_dir is None:
            raise FdoNumericalProbeConfigError("target layer probing requires an enabled configuration")
        if config.component != "target_layer":
            raise FdoNumericalProbeConfigError(f"target layer probe cannot serve component {config.component!r}")
        self._config = config
        self._max_num_tokens = max_num_tokens
        self._hidden_size = hidden_size
        self._dtype = dtype
        self._device = device
        self._writers: dict[int, BoundedProbeWriter] = {}
        self._iterations: dict[int, int] = {}
        self._exhausted_ranks: set[int] = set()
        self._buffers: dict[tuple[int, str], torch.Tensor] = {}
        self._captured_roles: dict[int, set[str]] = {}
        self._role_row_multipliers: dict[tuple[int, str], int] = {}
        self._hook_handles: list[Any] = []
        self.selected_layers: tuple[int, ...] = ()
        self._layer_roles = (
            "input",
            "input_norm.hidden",
            "input_norm.residual",
            "attention.input",
            "attention.output",
            "post_norm.hidden",
            "post_norm.residual",
            "mlp.output",
            "output",
            "residual",
        )

    def _resolve_layer_selection(self, num_layers: int) -> tuple[int, ...]:
        selection = self._config.layer
        if selection == "all":
            return tuple(range(num_layers))
        if not selection:
            raise FdoNumericalProbeConfigError(f"{PROBE_LAYER_ENV} is required for a target layer probe")
        try:
            indices = tuple(int(value) for value in selection.split(","))
        except ValueError as exc:
            raise FdoNumericalProbeConfigError(f"{PROBE_LAYER_ENV} must be 'all' or comma-separated indices") from exc
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(index < 0 or index >= num_layers for index in indices)
        ):
            raise FdoNumericalProbeConfigError(f"invalid target layer selection {selection!r} for {num_layers} layers")
        return indices

    def bind(self, model: torch.nn.Module) -> None:
        if self._hook_handles:
            raise FdoNumericalProbeConfigError("target layer probe is already bound")
        inner_model = getattr(model, "model", None)
        candidates = (
            inner_model,
            getattr(inner_model, "language_model", None),
            getattr(model, "language_model", None),
        )
        layers = None
        for candidate in candidates:
            candidate_layers = getattr(candidate, "layers", None)
            if candidate_layers is not None:
                layers = candidate_layers
                break
        if layers is None and isinstance(model, torch.nn.Module):
            layers = next(
                (
                    module
                    for name, module in model.named_modules()
                    if name.endswith(".layers") and isinstance(module, torch.nn.ModuleList)
                ),
                None,
            )
        if layers is None:
            raise FdoNumericalProbeConfigError(
                "target model does not expose decoder layers through a named *.layers ModuleList"
            )
        self.selected_layers = self._resolve_layer_selection(len(layers))
        for layer_index in self.selected_layers:
            self._captured_roles[layer_index] = set()
            for role in self._layer_roles:
                self._buffers[(layer_index, role)] = torch.empty(
                    (self._max_num_tokens, self._hidden_size),
                    dtype=self._dtype,
                    device=self._device,
                )
                self._role_row_multipliers[(layer_index, role)] = 1

            def copy_role(
                value: Any,
                *,
                index: int,
                role: str,
            ) -> None:
                if not isinstance(value, torch.Tensor) or value.ndim < 2:
                    raise FdoNumericalProbeArtifactError(f"target decoder {role} must expose rows and features")
                flattened = value.reshape(value.shape[0], -1)
                buffer = self._buffers[(index, role)]
                if flattened.shape[0] > buffer.shape[0] or (flattened.shape[1] != buffer.shape[1]):
                    raise FdoNumericalProbeArtifactError(f"target decoder {role} exceeds its capture buffer")
                buffer[: flattened.shape[0]].copy_(flattened)
                self._captured_roles[index].add(role)

            def capture_input(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                value = args[0] if args else kwargs.get("hidden_states")
                copy_role(value, index=index, role="input")

            def capture_output(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                if not isinstance(output, tuple) or len(output) != 2:
                    raise FdoNumericalProbeArtifactError("target decoder output must be (hidden, residual)")
                copy_role(output[0], index=index, role="output")
                copy_role(output[1], index=index, role="residual")

            def capture_norm_output(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
                prefix: str,
            ) -> None:
                if isinstance(output, tuple):
                    if len(output) != 2:
                        raise FdoNumericalProbeArtifactError(f"target decoder {prefix} must return hidden/residual")
                    copy_role(
                        output[0],
                        index=index,
                        role=f"{prefix}.hidden",
                    )
                    copy_role(
                        output[1],
                        index=index,
                        role=f"{prefix}.residual",
                    )
                else:
                    copy_role(
                        output,
                        index=index,
                        role=f"{prefix}.hidden",
                    )

            def capture_attention_input(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                value = args[0] if args else kwargs.get("hidden_states")
                copy_role(value, index=index, role="attention.input")

            def capture_attention_output(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                _output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                value = kwargs.get("output")
                if value is None and len(args) > 1:
                    value = args[1]
                copy_role(value, index=index, role="attention.output")

            def capture_mlp_output(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                value = output[0] if isinstance(output, tuple) else output
                copy_role(value, index=index, role="mlp.output")

            def capture_projection_output(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
                role: str,
            ) -> None:
                value = output[0] if isinstance(output, tuple) else output
                copy_role(value, index=index, role=role)

            def capture_gate_norm_input(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                values = args[:2]
                if len(values) != 2:
                    values = (
                        kwargs.get("hidden_states"),
                        kwargs.get("residual"),
                    )
                copy_role(values[0], index=index, role="attention.core")
                copy_role(values[1], index=index, role="attention.gate")

            def capture_projection_input(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
                role: str,
            ) -> None:
                value = args[0] if args else kwargs.get("hidden_states")
                copy_role(value, index=index, role=role)

            layer = layers[layer_index]
            input_norm = getattr(layer, "input_layernorm", None)
            attention = getattr(
                layer,
                "linear_attn",
                getattr(layer, "self_attn", None),
            )
            post_norm = getattr(layer, "post_attention_layernorm", None)
            mlp = getattr(layer, "mlp", None)
            if any(module is None for module in (input_norm, attention, post_norm, mlp)):
                raise FdoNumericalProbeConfigError("target layer probe requires norm, attention, and MLP modules")
            qkvz_projection = getattr(attention, "in_proj_qkvz", None)
            ba_projection = getattr(attention, "in_proj_ba", None)
            gate_norm = getattr(attention, "norm", None)
            output_projection = getattr(attention, "out_proj", None)
            linear_attention_hooks: tuple[Any, ...] = ()
            if all(
                module is not None
                for module in (
                    qkvz_projection,
                    ba_projection,
                    gate_norm,
                    output_projection,
                )
            ):
                heads = int(attention.num_v_heads) // int(attention.tp_size)
                head_dim = int(attention.head_v_dim)
                dynamic_roles = {
                    "attention.qkvz": (
                        1,
                        int(qkvz_projection.output_size_per_partition),
                    ),
                    "attention.ba": (
                        1,
                        int(ba_projection.output_size_per_partition),
                    ),
                    "attention.core": (heads, head_dim),
                    "attention.gate": (heads, head_dim),
                    "attention.norm": (heads, head_dim),
                    "attention.out_proj.input": (
                        1,
                        int(output_projection.input_size_per_partition),
                    ),
                    "attention.out_proj.output": (1, self._hidden_size),
                }
                for role, (row_multiplier, width) in dynamic_roles.items():
                    self._buffers[(layer_index, role)] = torch.empty(
                        (self._max_num_tokens * row_multiplier, width),
                        dtype=self._dtype,
                        device=self._device,
                    )
                    self._role_row_multipliers[(layer_index, role)] = row_multiplier
                linear_attention_hooks = (
                    qkvz_projection.register_forward_hook(
                        partial(
                            capture_projection_output,
                            role="attention.qkvz",
                        )
                    ),
                    ba_projection.register_forward_hook(
                        partial(
                            capture_projection_output,
                            role="attention.ba",
                        )
                    ),
                    gate_norm.register_forward_pre_hook(
                        capture_gate_norm_input,
                        with_kwargs=True,
                    ),
                    gate_norm.register_forward_hook(
                        partial(
                            capture_projection_output,
                            role="attention.norm",
                        )
                    ),
                    output_projection.register_forward_pre_hook(
                        partial(
                            capture_projection_input,
                            role="attention.out_proj.input",
                        ),
                        with_kwargs=True,
                    ),
                    output_projection.register_forward_hook(
                        partial(
                            capture_projection_output,
                            role="attention.out_proj.output",
                        )
                    ),
                )
            self._hook_handles.extend(
                (
                    layer.register_forward_pre_hook(
                        capture_input,
                        with_kwargs=True,
                    ),
                    input_norm.register_forward_hook(partial(capture_norm_output, prefix="input_norm")),
                    attention.register_forward_pre_hook(
                        capture_attention_input,
                        with_kwargs=True,
                    ),
                    attention.register_forward_hook(
                        capture_attention_output,
                        with_kwargs=True,
                    ),
                    post_norm.register_forward_hook(partial(capture_norm_output, prefix="post_norm")),
                    mlp.register_forward_hook(capture_mlp_output),
                    layer.register_forward_hook(capture_output),
                    *linear_attention_hooks,
                )
            )

    def _writer(self, tp_rank: int) -> BoundedProbeWriter:
        writer = self._writers.get(tp_rank)
        if writer is None:
            assert self._config.output_dir is not None
            writer = BoundedProbeWriter(
                replace(
                    self._config,
                    output_dir=self._config.output_dir / f"rank{tp_rank}",
                )
            )
            self._writers[tp_rank] = writer
        return writer

    def record_after_model(
        self,
        *,
        tp_rank: int,
        generated_prefix: tuple[int, ...],
        descriptor: int,
        actual_tokens: int,
        runtime_mode: CUDAGraphMode,
    ) -> None:
        if tp_rank in self._exhausted_ranks:
            return
        if not self.selected_layers:
            raise FdoNumericalProbeArtifactError("target layer probe was not bound before recording")
        if descriptor > self._max_num_tokens or actual_tokens > descriptor:
            raise FdoNumericalProbeArtifactError("target layer descriptor exceeds its persistent capture buffer")
        iteration = self._iterations.get(tp_rank, 0)
        runtime_code = 1 if runtime_mode == CUDAGraphMode.FULL else 0
        tensors: list[tuple[str, torch.Tensor, tuple[int, ...]]] = []
        for layer_index in self.selected_layers:
            for role in sorted(self._captured_roles[layer_index]):
                row_multiplier = self._role_row_multipliers[(layer_index, role)]
                descriptor_rows = descriptor * row_multiplier
                active_rows = tuple(range(actual_tokens * row_multiplier))
                tensors.append(
                    (
                        f"target_layer.{layer_index}.{role}",
                        self._buffers[(layer_index, role)][:descriptor_rows],
                        active_rows,
                    )
                )
        tensors.append(
            (
                "graph_runtime",
                torch.tensor(
                    [[runtime_code, descriptor, actual_tokens]],
                    dtype=torch.int64,
                    device=self._device,
                ),
                (0,),
            )
        )

        writer = self._writer(tp_rank)
        try:
            for semantic_role, tensor, tensor_active_rows in tensors:
                identity = ProbeTraceIdentity(
                    mode=self._config.mode or "",
                    component="target_layer",
                    tp_rank=tp_rank,
                    dataset_request=self._config.dataset_request,
                    generated_prefix=generated_prefix,
                    speculative_iteration=iteration,
                    draft_substep=None,
                    descriptor=descriptor,
                    actual_tokens=actual_tokens,
                    active_rows=tensor_active_rows,
                    semantic_role=semantic_role,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                )
                writer.write_tensor(identity, tensor)
        except FdoNumericalProbeLimitError:
            self._exhausted_ranks.add(tp_rank)
            return
        self._iterations[tp_rank] = iteration + 1


def create_target_layer_probe(
    vllm_config: VllmConfig,
    *,
    max_num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> TargetLayerProbe | None:
    """Create an opt-in target decoder layer observer."""
    config = FdoNumericalProbeConfig.from_environ(vllm_config)
    if not config.enabled or config.component != "target_layer":
        return None
    return TargetLayerProbe(
        config,
        max_num_tokens=max_num_tokens,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
    )


class DraftBoundaryProbe:
    """Export matched DFlash inputs and remapped proposal IDs after replay."""

    def __init__(self, config: FdoNumericalProbeConfig) -> None:
        if not config.enabled or config.output_dir is None:
            raise FdoNumericalProbeConfigError("draft boundary probing requires an enabled configuration")
        if config.component not in {"boundary", "draft"}:
            raise FdoNumericalProbeConfigError(f"draft probe cannot serve component {config.component!r}")
        self._config = config
        self._writers: dict[int, BoundedProbeWriter] = {}
        self._iterations: dict[int, int] = {}
        self._exhausted_ranks: set[int] = set()

    def _writer(self, tp_rank: int) -> BoundedProbeWriter:
        writer = self._writers.get(tp_rank)
        if writer is None:
            assert self._config.output_dir is not None
            writer = BoundedProbeWriter(
                replace(
                    self._config,
                    output_dir=self._config.output_dir / f"rank{tp_rank}",
                )
            )
            self._writers[tp_rank] = writer
        return writer

    def record_after_model(
        self,
        *,
        tp_rank: int,
        generated_prefix: tuple[int, ...],
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden: torch.Tensor,
        next_token_ids: torch.Tensor,
        proposed_token_ids: torch.Tensor,
        descriptor: int,
        actual_tokens: int,
        runtime_mode: CUDAGraphMode,
    ) -> None:
        if tp_rank in self._exhausted_ranks:
            return
        iteration = self._iterations.get(tp_rank, 0)
        target_positions = TargetBoundaryProbe._token_major_positions(target_positions)
        if target_token_ids.ndim == 0 or target_positions.ndim == 0 or target_hidden.ndim == 0:
            raise FdoNumericalProbeArtifactError("draft inputs must expose a token dimension")
        target_rows = target_token_ids.shape[0]
        if target_positions.shape[0] != target_rows or target_hidden.shape[0] != target_rows:
            raise FdoNumericalProbeArtifactError("draft token, position, and hidden input rows must align")
        runtime_code = 1 if runtime_mode == CUDAGraphMode.FULL else 0
        runtime_identity = torch.tensor(
            [[runtime_code, descriptor, actual_tokens]],
            dtype=torch.int64,
            device=target_token_ids.device,
        )
        tensors = (
            ("target_token_ids", target_token_ids, tuple(range(target_rows))),
            ("target_positions", target_positions, tuple(range(target_rows))),
            ("target_hidden_input", target_hidden, tuple(range(target_rows))),
            (
                "next_token_ids",
                next_token_ids,
                tuple(range(next_token_ids.shape[0])),
            ),
            (
                "proposed_token_ids",
                proposed_token_ids,
                tuple(range(proposed_token_ids.shape[0])),
            ),
            ("graph_runtime", runtime_identity, (0,)),
        )
        writer = self._writer(tp_rank)
        try:
            for semantic_role, tensor, active_rows in tensors:
                identity = ProbeTraceIdentity(
                    mode=self._config.mode or "",
                    component="draft",
                    tp_rank=tp_rank,
                    dataset_request=self._config.dataset_request,
                    generated_prefix=generated_prefix,
                    speculative_iteration=iteration,
                    draft_substep=None,
                    descriptor=descriptor,
                    actual_tokens=actual_tokens,
                    active_rows=active_rows,
                    semantic_role=semantic_role,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                )
                writer.write_tensor(identity, tensor)
        except FdoNumericalProbeLimitError:
            self._exhausted_ranks.add(tp_rank)
            return
        self._iterations[tp_rank] = iteration + 1


def create_draft_boundary_probe(
    vllm_config: VllmConfig,
) -> DraftBoundaryProbe | None:
    """Create the draft observer only for an explicitly enabled component."""
    config = FdoNumericalProbeConfig.from_environ(vllm_config)
    if not config.enabled or config.component not in {"boundary", "draft"}:
        return None
    return DraftBoundaryProbe(config)


class RejectionLoopProbe:
    """Export the accepted-token contract after the public rejection sampler."""

    def __init__(self, config: FdoNumericalProbeConfig) -> None:
        if not config.enabled or config.output_dir is None:
            raise FdoNumericalProbeConfigError("rejection probing requires an enabled configuration")
        if config.component != "rejection":
            raise FdoNumericalProbeConfigError(f"rejection probe cannot serve component {config.component!r}")
        self._config = config
        self._writers: dict[int, BoundedProbeWriter] = {}
        self._iterations: dict[int, int] = {}
        self._exhausted_ranks: set[int] = set()

    def _writer(self, tp_rank: int) -> BoundedProbeWriter:
        writer = self._writers.get(tp_rank)
        if writer is None:
            assert self._config.output_dir is not None
            writer = BoundedProbeWriter(
                replace(
                    self._config,
                    output_dir=self._config.output_dir / f"rank{tp_rank}",
                )
            )
            self._writers[tp_rank] = writer
        return writer

    def record_after_sample(
        self,
        *,
        tp_rank: int,
        generated_prefix: tuple[int, ...],
        draft_token_ids: torch.Tensor,
        num_draft_tokens: torch.Tensor,
        sampled_token_ids: torch.Tensor,
        valid_sampled_token_count: torch.Tensor,
        descriptor: int,
        actual_tokens: int,
    ) -> None:
        if tp_rank in self._exhausted_ranks:
            return
        if draft_token_ids.ndim != 1 or num_draft_tokens.ndim != 1:
            raise FdoNumericalProbeArtifactError("rejection draft inputs must be one-dimensional")
        if sampled_token_ids.ndim != 2 or valid_sampled_token_count.ndim != 1:
            raise FdoNumericalProbeArtifactError("rejection sampled tokens must be request-major")
        num_reqs = num_draft_tokens.shape[0]
        if sampled_token_ids.shape[0] != num_reqs or valid_sampled_token_count.shape[0] != num_reqs:
            raise FdoNumericalProbeArtifactError("rejection request rows must align")
        if int(num_draft_tokens.sum().item()) != draft_token_ids.numel():
            raise FdoNumericalProbeArtifactError("flattened draft tokens do not match per-request widths")
        if bool(torch.any(valid_sampled_token_count > sampled_token_ids.shape[1]).item()):
            raise FdoNumericalProbeArtifactError("valid sampled-token count exceeds the sampler row width")

        iteration = self._iterations.get(tp_rank, 0)
        request_rows = tuple(range(num_reqs))
        tensors = (
            (
                "draft_token_ids",
                draft_token_ids,
                tuple(range(draft_token_ids.shape[0])),
            ),
            ("num_draft_tokens", num_draft_tokens, request_rows),
            ("sampled_token_ids", sampled_token_ids, request_rows),
            (
                "valid_sampled_token_count",
                valid_sampled_token_count,
                request_rows,
            ),
        )
        writer = self._writer(tp_rank)
        try:
            for semantic_role, tensor, active_rows in tensors:
                identity = ProbeTraceIdentity(
                    mode=self._config.mode or "",
                    component="rejection",
                    tp_rank=tp_rank,
                    dataset_request=self._config.dataset_request,
                    generated_prefix=generated_prefix,
                    speculative_iteration=iteration,
                    draft_substep=None,
                    descriptor=descriptor,
                    actual_tokens=actual_tokens,
                    active_rows=active_rows,
                    semantic_role=semantic_role,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                )
                writer.write_tensor(identity, tensor)
        except FdoNumericalProbeLimitError:
            self._exhausted_ranks.add(tp_rank)
            return
        self._iterations[tp_rank] = iteration + 1


def create_rejection_loop_probe(
    vllm_config: VllmConfig,
) -> RejectionLoopProbe | None:
    """Create the rejection observer only for explicit diagnostic runs."""
    config = FdoNumericalProbeConfig.from_environ(vllm_config)
    if not config.enabled or config.component != "rejection":
        return None
    return RejectionLoopProbe(config)


class DraftLayerProbe:
    """Capture selected DFlash decoder outputs into persistent device buffers.

    Forward hooks only enqueue device-to-device copies. Artifact export happens
    after the draft invocation, so an ACL graph records stable buffer writes and
    never contains filesystem or device-to-host work.
    """

    def __init__(
        self,
        config: FdoNumericalProbeConfig,
        *,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if not config.enabled or config.output_dir is None:
            raise FdoNumericalProbeConfigError("draft layer probing requires an enabled configuration")
        if config.component != "layer":
            raise FdoNumericalProbeConfigError(f"draft layer probe cannot serve component {config.component!r}")
        self._config = config
        self._max_num_tokens = max_num_tokens
        self._hidden_size = hidden_size
        self._dtype = dtype
        self._device = device
        self._writers: dict[int, BoundedProbeWriter] = {}
        self._iterations: dict[int, int] = {}
        self._exhausted_ranks: set[int] = set()
        self._buffers: dict[tuple[int, str], torch.Tensor] = {}
        self._hook_handles: list[Any] = []
        self._active_layers: set[int] = set()
        self._context_buffers: dict[str, torch.Tensor] = {}
        self._context_rows = 0
        self._context_position_width = 0
        self._context_has_slot_mapping = False
        self._context_knorm_layers: set[int] = set()
        self._context_rope_layers: set[int] = set()
        self.selected_layers: tuple[int, ...] = ()
        self._layer_roles = (
            "input",
            "input_norm",
            "attention",
            "post_norm.hidden",
            "post_norm.residual",
            "mlp",
            "hidden",
            "residual",
        )
        self._attention_roles = (
            "q_norm.input",
            "q_norm.output",
            "k_norm.input",
            "k_norm.output",
            "rope.positions",
            "rope.q",
            "rope.k",
            "attn.q",
            "attn.k",
            "attn.v",
            "attn.output",
            "o_proj.output",
        )

    def _resolve_layer_selection(self, num_layers: int) -> tuple[int, ...]:
        selection = self._config.layer
        if selection == "all":
            return tuple(range(num_layers))
        if not selection:
            raise FdoNumericalProbeConfigError(f"{PROBE_LAYER_ENV} is required for a layer probe")
        try:
            indices = tuple(int(value) for value in selection.split(","))
        except ValueError as exc:
            raise FdoNumericalProbeConfigError(f"{PROBE_LAYER_ENV} must be 'all' or comma-separated indices") from exc
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(index < 0 or index >= num_layers for index in indices)
        ):
            raise FdoNumericalProbeConfigError(f"invalid draft layer selection {selection!r} for {num_layers} layers")
        return indices

    def bind(self, model: torch.nn.Module) -> None:
        if self._hook_handles:
            raise FdoNumericalProbeConfigError("draft layer probe is already bound")
        draft_model = getattr(model, "model", None)
        layers = getattr(draft_model, "layers", None)
        if layers is None:
            raise FdoNumericalProbeConfigError("draft model does not expose model.layers")
        self.selected_layers = self._resolve_layer_selection(len(layers))
        self._context_buffers = {
            "input": torch.empty(
                (self._max_num_tokens, self._hidden_size),
                dtype=self._dtype,
                device=self._device,
            ),
            "positions": torch.empty(
                (self._max_num_tokens, 3),
                dtype=torch.int64,
                device=self._device,
            ),
            "hidden_norm": torch.empty(
                (self._max_num_tokens, self._hidden_size),
                dtype=self._dtype,
                device=self._device,
            ),
            "slot_mapping": torch.empty(
                self._max_num_tokens,
                dtype=torch.int64,
                device=self._device,
            ),
        }
        draft_model._fdo_context_probe = self
        for layer_index in self.selected_layers:
            layer = layers[layer_index]
            required_modules = {
                "input_norm": getattr(layer, "input_layernorm", None),
                "attention": getattr(layer, "self_attn", None),
                "post_norm": getattr(layer, "post_attention_layernorm", None),
                "mlp": getattr(layer, "mlp", None),
            }
            if any(module is None for module in required_modules.values()):
                raise FdoNumericalProbeConfigError(
                    "draft layer probe requires input norm, attention, post norm, and MLP modules"
                )
            attention_module = required_modules["attention"]
            assert attention_module is not None
            required_attention_modules = {
                "q_norm": getattr(attention_module, "q_norm", None),
                "k_norm": getattr(attention_module, "k_norm", None),
                "rotary_emb": getattr(attention_module, "rotary_emb", None),
                "attn": getattr(attention_module, "attn", None),
                "o_proj": getattr(attention_module, "o_proj", None),
            }
            if any(module is None for module in required_attention_modules.values()):
                raise FdoNumericalProbeConfigError(
                    "draft attention probe requires Q/K norm, rotary, attention, and output projection modules"
                )
            q_size = int(attention_module.q_size)
            kv_size = int(attention_module.kv_size)
            for role in self._layer_roles:
                self._buffers[(layer_index, role)] = torch.empty(
                    (self._max_num_tokens, self._hidden_size),
                    dtype=self._dtype,
                    device=self._device,
                )
            attention_widths = {
                "q_norm.input": q_size,
                "q_norm.output": q_size,
                "k_norm.input": kv_size,
                "k_norm.output": kv_size,
                "rope.positions": 1,
                "rope.q": q_size,
                "rope.k": kv_size,
                "attn.q": q_size,
                "attn.k": kv_size,
                "attn.v": kv_size,
                "attn.output": q_size,
                "o_proj.output": self._hidden_size,
            }
            for role, width in attention_widths.items():
                self._buffers[(layer_index, role)] = torch.empty(
                    (self._max_num_tokens, width),
                    dtype=(torch.int64 if role == "rope.positions" else self._dtype),
                    device=self._device,
                )
            for role in (
                "context.k_norm.input",
                "context.k_norm.output",
                "context.rope.k",
                "context.v",
            ):
                self._buffers[(layer_index, role)] = torch.empty(
                    (self._max_num_tokens, kv_size),
                    dtype=self._dtype,
                    device=self._device,
                )

            def copy_role(
                tensor: Any,
                *,
                index: int,
                role: str,
            ) -> None:
                if index not in self._active_layers:
                    return
                if not isinstance(tensor, torch.Tensor):
                    raise FdoNumericalProbeArtifactError(f"draft decoder {role} value must be a tensor")
                flattened = tensor.reshape(tensor.shape[0], -1)
                buffer = self._buffers[(index, role)]
                if flattened.shape[1] != buffer.shape[1]:
                    raise FdoNumericalProbeArtifactError(
                        f"draft decoder {role} width mismatch: actual={flattened.shape[1]}, expected={buffer.shape[1]}"
                    )
                buffer[: flattened.shape[0]].copy_(flattened)

            def capture_input(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                self._active_layers.add(index)
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None and args:
                    hidden_states = args[0]
                if not isinstance(hidden_states, torch.Tensor):
                    raise FdoNumericalProbeArtifactError("draft decoder hidden input must be a tensor")
                rows = hidden_states.shape[0]
                self._buffers[(index, "input")][:rows].copy_(hidden_states)

            def capture_output(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                try:
                    if not isinstance(output, tuple) or len(output) != 2:
                        raise FdoNumericalProbeArtifactError("draft decoder layer output must be (hidden, residual)")
                    hidden_states, residual = output
                    if not isinstance(hidden_states, torch.Tensor) or not isinstance(residual, torch.Tensor):
                        raise FdoNumericalProbeArtifactError(
                            "draft decoder hidden and residual outputs must be tensors"
                        )
                    rows = hidden_states.shape[0]
                    self._buffers[(index, "hidden")][:rows].copy_(hidden_states)
                    self._buffers[(index, "residual")][:rows].copy_(residual)
                finally:
                    self._active_layers.discard(index)

            def capture_tensor_output(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
                role: str,
            ) -> None:
                value = output[0] if role == "input_norm" and isinstance(output, tuple) else output
                if not isinstance(value, torch.Tensor):
                    raise FdoNumericalProbeArtifactError(f"draft decoder {role} output must be a tensor")
                rows = value.shape[0]
                self._buffers[(index, role)][:rows].copy_(value)

            def capture_post_norm(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                if not isinstance(output, tuple) or len(output) != 2:
                    raise FdoNumericalProbeArtifactError("draft decoder post norm output must be (hidden, residual)")
                hidden_states, residual = output
                if not isinstance(hidden_states, torch.Tensor) or not isinstance(residual, torch.Tensor):
                    raise FdoNumericalProbeArtifactError("draft decoder post norm outputs must be tensors")
                rows = hidden_states.shape[0]
                self._buffers[(index, "post_norm.hidden")][:rows].copy_(hidden_states)
                self._buffers[(index, "post_norm.residual")][:rows].copy_(residual)

            def capture_first_input(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
                role: str,
            ) -> None:
                value = args[0] if args else kwargs.get("hidden_states")
                copy_role(value, index=index, role=role)

            def capture_tensor_role(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
                role: str,
            ) -> None:
                copy_role(output, index=index, role=role)

            def capture_rope(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                if not isinstance(output, tuple) or len(output) != 2:
                    raise FdoNumericalProbeArtifactError("draft rotary output must be (query, key)")
                copy_role(output[0], index=index, role="rope.q")
                copy_role(output[1], index=index, role="rope.k")

            def capture_rope_inputs(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                positions = args[0] if args else kwargs.get("positions")
                copy_role(positions, index=index, role="rope.positions")

            def capture_attn_inputs(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
                *,
                index: int = layer_index,
            ) -> None:
                values = args[:3]
                if len(values) != 3:
                    values = tuple(kwargs.get(name) for name in ("query", "key", "value"))
                for role, value in zip(("attn.q", "attn.k", "attn.v"), values):
                    copy_role(value, index=index, role=role)

            def capture_o_proj(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                index: int = layer_index,
            ) -> None:
                value = output[0] if isinstance(output, tuple) else output
                copy_role(value, index=index, role="o_proj.output")

            self._hook_handles.extend(
                (
                    layer.register_forward_pre_hook(
                        capture_input,
                        with_kwargs=True,
                    ),
                    required_modules["input_norm"].register_forward_hook(
                        partial(capture_tensor_output, role="input_norm")
                    ),
                    required_modules["attention"].register_forward_hook(
                        partial(capture_tensor_output, role="attention")
                    ),
                    required_attention_modules["q_norm"].register_forward_pre_hook(
                        partial(capture_first_input, role="q_norm.input"),
                        with_kwargs=True,
                    ),
                    required_attention_modules["q_norm"].register_forward_hook(
                        partial(capture_tensor_role, role="q_norm.output")
                    ),
                    required_attention_modules["k_norm"].register_forward_pre_hook(
                        partial(capture_first_input, role="k_norm.input"),
                        with_kwargs=True,
                    ),
                    required_attention_modules["k_norm"].register_forward_hook(
                        partial(capture_tensor_role, role="k_norm.output")
                    ),
                    required_attention_modules["rotary_emb"].register_forward_hook(capture_rope),
                    required_attention_modules["rotary_emb"].register_forward_pre_hook(
                        capture_rope_inputs,
                        with_kwargs=True,
                    ),
                    required_attention_modules["attn"].register_forward_pre_hook(
                        capture_attn_inputs,
                        with_kwargs=True,
                    ),
                    required_attention_modules["attn"].register_forward_hook(
                        partial(capture_tensor_role, role="attn.output")
                    ),
                    required_attention_modules["o_proj"].register_forward_hook(capture_o_proj),
                    required_modules["post_norm"].register_forward_hook(capture_post_norm),
                    required_modules["mlp"].register_forward_hook(partial(capture_tensor_output, role="mlp")),
                    layer.register_forward_hook(capture_output),
                )
            )

    def capture_context_inputs(
        self,
        *,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        normed_context_states: torch.Tensor,
        slot_mapping: (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None),
    ) -> None:
        """Enqueue stable D2D copies for the context-KV common inputs."""
        if context_states.ndim != 2 or normed_context_states.ndim != 2:
            raise FdoNumericalProbeArtifactError("draft context states and normalized states must be 2D")
        rows = context_states.shape[0]
        if (
            rows > self._max_num_tokens
            or normed_context_states.shape != context_states.shape
            or context_states.shape[1] != self._hidden_size
        ):
            raise FdoNumericalProbeArtifactError("draft context inputs exceed their persistent capture buffers")
        token_major_positions = TargetBoundaryProbe._token_major_positions(context_positions)
        if token_major_positions.ndim == 1:
            flattened_positions = token_major_positions.reshape(-1, 1)
        else:
            flattened_positions = token_major_positions.reshape(token_major_positions.shape[0], -1)
        if flattened_positions.shape[0] != rows or flattened_positions.shape[1] > 3:
            raise FdoNumericalProbeArtifactError("draft context positions do not align with context states")

        self._context_buffers["input"][:rows].copy_(context_states)
        self._context_buffers["hidden_norm"][:rows].copy_(normed_context_states)
        self._context_buffers["positions"][:rows, : flattened_positions.shape[1]].copy_(flattened_positions)
        self._context_rows = rows
        self._context_position_width = flattened_positions.shape[1]
        self._context_has_slot_mapping = False
        self._context_knorm_layers.clear()
        self._context_rope_layers.clear()

        selected_slot_mapping = slot_mapping
        if isinstance(slot_mapping, (list, tuple)):
            primary_layer = self.selected_layers[0]
            selected_slot_mapping = slot_mapping[primary_layer]
        if selected_slot_mapping is not None:
            flattened_slots = selected_slot_mapping.reshape(-1)
            if flattened_slots.shape[0] != rows:
                raise FdoNumericalProbeArtifactError("draft context slot mapping does not align with context states")
            self._context_buffers["slot_mapping"][:rows].copy_(flattened_slots)
            self._context_has_slot_mapping = True

    def capture_context_k_norm(
        self,
        *,
        layer_index: int,
        k_norm_input: torch.Tensor,
        k_norm_output: torch.Tensor,
    ) -> None:
        """Capture K immediately after norm and before RoPE can mutate it."""
        if layer_index not in self.selected_layers:
            return
        if self._context_rows == 0:
            raise FdoNumericalProbeArtifactError("draft context layer was captured before its common inputs")
        for role, tensor in (
            ("context.k_norm.input", k_norm_input),
            ("context.k_norm.output", k_norm_output),
        ):
            flattened = tensor.reshape(tensor.shape[0], -1)
            buffer = self._buffers[(layer_index, role)]
            if flattened.shape[0] != self._context_rows or flattened.shape[1] != buffer.shape[1]:
                raise FdoNumericalProbeArtifactError(f"draft {role} does not align with its context capture buffer")
            buffer[: self._context_rows].copy_(flattened)
        self._context_knorm_layers.add(layer_index)

    def capture_context_rope(
        self,
        *,
        layer_index: int,
        k_rope: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Capture context K after RoPE and its paired V projection."""
        if layer_index not in self.selected_layers:
            return
        if self._context_rows == 0:
            raise FdoNumericalProbeArtifactError("draft context RoPE was captured before its common inputs")
        for role, tensor in (
            ("context.rope.k", k_rope),
            ("context.v", value),
        ):
            flattened = tensor.reshape(tensor.shape[0], -1)
            buffer = self._buffers[(layer_index, role)]
            if flattened.shape[0] != self._context_rows or flattened.shape[1] != buffer.shape[1]:
                raise FdoNumericalProbeArtifactError(f"draft {role} does not align with its context capture buffer")
            buffer[: self._context_rows].copy_(flattened)
        self._context_rope_layers.add(layer_index)

    def _writer(self, tp_rank: int) -> BoundedProbeWriter:
        writer = self._writers.get(tp_rank)
        if writer is None:
            assert self._config.output_dir is not None
            writer = BoundedProbeWriter(
                replace(
                    self._config,
                    output_dir=self._config.output_dir / f"rank{tp_rank}",
                )
            )
            self._writers[tp_rank] = writer
        return writer

    def record_after_model(
        self,
        *,
        tp_rank: int,
        generated_prefix: tuple[int, ...],
        draft_input_ids: torch.Tensor,
        draft_positions: torch.Tensor,
        draft_embeddings: torch.Tensor,
        descriptor: int,
        actual_tokens: int,
        runtime_mode: CUDAGraphMode,
        context_actual_tokens: int | None = None,
    ) -> None:
        if tp_rank in self._exhausted_ranks:
            return
        if not self.selected_layers:
            raise FdoNumericalProbeArtifactError("draft layer probe was not bound before recording")
        if descriptor > self._max_num_tokens or actual_tokens > descriptor:
            raise FdoNumericalProbeArtifactError("draft layer descriptor exceeds its persistent capture buffer")
        context_rows = self._context_rows if context_actual_tokens is None else context_actual_tokens
        if not 0 <= context_rows <= self._context_rows:
            raise FdoNumericalProbeArtifactError(
                "runtime draft context rows exceed the captured context buffer: "
                f"actual={context_rows}, captured={self._context_rows}"
            )
        draft_positions = TargetBoundaryProbe._token_major_positions(draft_positions)
        if (
            draft_input_ids.ndim == 0
            or draft_positions.ndim == 0
            or draft_embeddings.ndim != 2
            or draft_input_ids.shape[0] < descriptor
            or draft_positions.shape[0] < descriptor
            or draft_embeddings.shape[0] < descriptor
            or draft_embeddings.shape[1] != self._hidden_size
        ):
            raise FdoNumericalProbeArtifactError("draft layer inputs do not cover the recorded descriptor")
        iteration = self._iterations.get(tp_rank, 0)
        active_rows = tuple(range(actual_tokens))
        tensors: list[tuple[str, torch.Tensor, tuple[int, ...]]] = [
            (
                "draft_input.input_ids",
                draft_input_ids[:descriptor],
                active_rows,
            ),
            (
                "draft_input.positions",
                draft_positions[:descriptor],
                active_rows,
            ),
            (
                "draft_input.embeddings",
                draft_embeddings[:descriptor],
                active_rows,
            ),
        ]
        if context_rows:
            context_active_rows = tuple(range(context_rows))
            positions = self._context_buffers["positions"][:context_rows, : self._context_position_width]
            if self._context_position_width == 1:
                positions = positions.reshape(-1)
            tensors.extend(
                (
                    (
                        "draft_context.input",
                        self._context_buffers["input"][:context_rows],
                        context_active_rows,
                    ),
                    (
                        "draft_context.positions",
                        positions,
                        context_active_rows,
                    ),
                    (
                        "draft_context.hidden_norm",
                        self._context_buffers["hidden_norm"][:context_rows],
                        context_active_rows,
                    ),
                )
            )
            if self._context_has_slot_mapping:
                tensors.append(
                    (
                        "draft_context.slot_mapping",
                        self._context_buffers["slot_mapping"][:context_rows],
                        context_active_rows,
                    )
                )
            for layer_index in self.selected_layers:
                if layer_index not in self._context_knorm_layers or layer_index not in self._context_rope_layers:
                    continue
                for role in (
                    "context.k_norm.input",
                    "context.k_norm.output",
                    "context.rope.k",
                    "context.v",
                ):
                    tensors.append(
                        (
                            f"draft_context.layer.{layer_index}.{role.removeprefix('context.')}",
                            self._buffers[(layer_index, role)][:context_rows],
                            context_active_rows,
                        )
                    )
        for layer_index in self.selected_layers:
            for role in self._layer_roles:
                tensors.append(
                    (
                        f"draft_layer.{layer_index}.{role}",
                        self._buffers[(layer_index, role)][:descriptor],
                        active_rows,
                    )
                )
            for role in self._attention_roles:
                tensors.append(
                    (
                        f"draft_layer.{layer_index}.{role}",
                        self._buffers[(layer_index, role)][:descriptor],
                        active_rows,
                    )
                )
        runtime_code = 1 if runtime_mode == CUDAGraphMode.FULL else 0
        runtime_identity = torch.tensor(
            [[runtime_code, descriptor, actual_tokens]],
            dtype=torch.int64,
            device=self._device,
        )
        tensors.append(("graph_runtime", runtime_identity, (0,)))

        writer = self._writer(tp_rank)
        try:
            for semantic_role, tensor, tensor_active_rows in tensors:
                identity = ProbeTraceIdentity(
                    mode=self._config.mode or "",
                    component="layer",
                    tp_rank=tp_rank,
                    dataset_request=self._config.dataset_request,
                    generated_prefix=generated_prefix,
                    speculative_iteration=iteration,
                    draft_substep=None,
                    descriptor=descriptor,
                    actual_tokens=actual_tokens,
                    active_rows=tensor_active_rows,
                    semantic_role=semantic_role,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                )
                writer.write_tensor(identity, tensor)
        except FdoNumericalProbeLimitError:
            self._exhausted_ranks.add(tp_rank)
            return
        self._iterations[tp_rank] = iteration + 1


def create_draft_layer_probe(
    vllm_config: VllmConfig,
    *,
    max_num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> DraftLayerProbe | None:
    """Create an opt-in draft layer observer with fixed device buffers."""
    config = FdoNumericalProbeConfig.from_environ(vllm_config)
    if not config.enabled or config.component != "layer":
        return None
    return DraftLayerProbe(
        config,
        max_num_tokens=max_num_tokens,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
    )
