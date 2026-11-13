from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .quant_layout import BITS, INT_TYPES, build_quant_layout

logger = logging.getLogger(__name__)

KVTC_FILE_VERSION = "v3-worker-quant"
QUANT_DTYPES = ("float32", "bfloat16", "int8", "int4")
QUANT_BLOCK_SIZES = (1, 16, 64, 256, 1024)
INT4_BLOCK_SIZES = (8, 16, 64, 256, 1024)
NPU_ERROR_BYTES_PER_VALUE = 16


def quant_group_bits(group_size: int, dtype_name: str) -> int:
    """Per-token payload and one FP16 scale/offset pair for integer groups."""
    return group_size * BITS[dtype_name] + (32 if dtype_name in INT_TYPES else 0)


def resolve_quant_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name != "npu":
        raise ValueError(f"Unsupported quantization device: {device_name}")

    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError("NPU quantization requested, but torch_npu is not installed") from error

    missing_ops = [name for name in ("npu_dynamic_quant_asymmetric", "npu_anti_quant") if not hasattr(torch_npu, name)]
    if missing_ops:
        raise RuntimeError("NPU quantization requires missing torch_npu APIs: " + ", ".join(missing_ops))
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU quantization requested, but no NPU is available")
    return torch.device("npu")


def simulate_quantization_error(values: torch.Tensor, dtype_name: str) -> float:
    if dtype_name == "float32":
        return 0.0
    if dtype_name == "bfloat16":
        reconstructed = values.to(torch.bfloat16).to(torch.float32)
    else:
        qmin, qmax = (-128, 127) if dtype_name == "int8" else (-8, 7)
        quant_values = values.to(torch.bfloat16).to(torch.float32)
        minimum = quant_values.amin(dim=1, keepdim=True)
        maximum = quant_values.amax(dim=1, keepdim=True)
        value_range = maximum - minimum
        scale = torch.where(
            value_range > 0,
            value_range / (qmax - qmin),
            torch.ones_like(value_range),
        )
        offset = qmin - minimum / scale
        quantized = torch.clamp(torch.round(quant_values / scale + offset), qmin, qmax)
        stored_scale = scale.to(torch.float16).to(torch.float32)
        stored_offset = offset.to(torch.float16).to(torch.float32)
        reconstructed = (quantized - stored_offset) * stored_scale
        reconstructed = torch.where(value_range > 0, reconstructed, quant_values)
        reconstructed = reconstructed.to(torch.bfloat16).to(torch.float32)
    error_norm = torch.linalg.vector_norm(values - reconstructed)
    return float(error_norm.square().item())


@dataclass(frozen=True)
class QuantizationErrorTable:
    rank: int
    source_norm: float
    tail_errors: tuple[float, ...]
    block_errors: dict[tuple[int, int, str], float]


def _prefix_block_errors(
    feature_errors: torch.Tensor,
    block_sizes: tuple[int, ...],
    dtype_name: str,
    max_budget: int,
) -> dict[tuple[int, int, str], float]:
    prefix = torch.cat((torch.zeros(1, dtype=torch.float64), feature_errors.to(torch.float64).cumsum(0)))
    errors = {}
    rank = feature_errors.shape[0]
    for size in block_sizes:
        if size > rank or quant_group_bits(size, dtype_name) > max_budget:
            continue
        block_values = prefix[size:] - prefix[:-size]
        for start, error in enumerate(block_values.tolist()):
            errors[(start, size, dtype_name)] = error
    return errors


def _build_cpu_integer_errors(
    projected_data: torch.Tensor,
    max_budget: int,
) -> dict[tuple[int, int, str], float]:
    errors = {}
    candidates = [
        (dtype_name, size)
        for dtype_name, block_sizes in (
            ("int8", QUANT_BLOCK_SIZES),
            ("int4", INT4_BLOCK_SIZES),
        )
        for size in block_sizes
        if size <= projected_data.shape[1] and quant_group_bits(size, dtype_name) <= max_budget
    ]
    total = sum(projected_data.shape[1] - size + 1 for _, size in candidates)
    with tqdm(
        total=total,
        desc="Quantization errors (CPU)",
        unit="block",
        dynamic_ncols=True,
        disable=None,
    ) as progress:
        for dtype_name, size in candidates:
            for start in range(projected_data.shape[1] - size + 1):
                errors[(start, size, dtype_name)] = simulate_quantization_error(
                    projected_data[:, start : start + size], dtype_name
                )
                progress.update()
    return errors


def _npu_integer_batch_errors(
    source: torch.Tensor,
    dtype_name: str,
) -> torch.Tensor:
    import torch_npu

    batch_count, token_count, width = source.shape
    quant_input = source.to(torch.bfloat16).reshape(-1, width)
    dst_type = torch.quint4x2 if dtype_name == "int4" else torch.int8
    quantized, scale, quant_offset = torch_npu.npu_dynamic_quant_asymmetric(
        quant_input,
        dst_type=dst_type,
    )
    stored_scale = scale.to(torch.float16).to(torch.float32)
    stored_offset = (-quant_offset).to(torch.float16).to(torch.float32)
    expanded_scale = stored_scale.reshape(-1).repeat_interleave(width)
    expanded_offset = stored_offset.reshape(-1).repeat_interleave(width)
    kwargs = {
        "offset": expanded_offset,
        "dst_dtype": torch.bfloat16,
    }
    if dtype_name == "int4":
        kwargs["src_dtype"] = torch.quint4x2
    reconstructed = torch_npu.npu_anti_quant(
        quantized.reshape(1, -1),
        expanded_scale,
        **kwargs,
    ).reshape(batch_count, token_count, width)
    return (source - reconstructed.to(torch.float32)).square().sum(dim=(1, 2))


def _build_npu_integer_errors(
    projected_data: torch.Tensor,
    max_budget: int,
    workspace_mb: int,
) -> dict[tuple[int, int, str], float]:
    errors = {}
    candidates = [
        (dtype_name, size)
        for dtype_name, block_sizes in (
            ("int8", QUANT_BLOCK_SIZES),
            ("int4", INT4_BLOCK_SIZES),
        )
        for size in block_sizes
        if size <= projected_data.shape[1] and quant_group_bits(size, dtype_name) <= max_budget
    ]
    total = sum(projected_data.shape[1] - size + 1 for _, size in candidates)
    workspace_bytes = workspace_mb * 1024 * 1024
    token_count = projected_data.shape[0]
    with tqdm(
        total=total,
        desc="Quantization errors (NPU)",
        unit="block",
        dynamic_ncols=True,
        disable=None,
    ) as progress:
        for dtype_name, size in candidates:
            window_count = projected_data.shape[1] - size + 1
            bytes_per_window = token_count * size * NPU_ERROR_BYTES_PER_VALUE
            batch_size = max(1, workspace_bytes // bytes_per_window)
            logger.info(
                "NPU quantization errors dtype=%s width=%s windows=%s batch=%s",
                dtype_name,
                size,
                window_count,
                batch_size,
            )
            for first in range(0, window_count, batch_size):
                starts = list(range(first, min(first + batch_size, window_count)))
                source = torch.stack([projected_data[:, start : start + size] for start in starts])
                batch_errors = _npu_integer_batch_errors(source, dtype_name)
                values = batch_errors.cpu().tolist()
                errors.update(((start, size, dtype_name), error) for start, error in zip(starts, values))
                progress.update(len(starts))
                del source, batch_errors
    return errors


def build_quantization_error_table(
    projected_data: torch.Tensor,
    max_budget: int,
    npu_workspace_mb: int = 512,
) -> QuantizationErrorTable:
    if projected_data.ndim != 2 or projected_data.shape[0] == 0:
        raise ValueError("No held-out projections were provided for quantization")
    if projected_data.device.type not in ("cpu", "npu"):
        raise ValueError(f"Unsupported quantization tensor device: {projected_data.device.type}")

    started = time.perf_counter()
    rank = projected_data.shape[1]
    feature_energy = projected_data.square().sum(dim=0).cpu().to(torch.float64)
    tail_errors_tensor = torch.cat(
        (
            torch.flip(torch.cumsum(torch.flip(feature_energy, (0,)), dim=0), (0,)),
            torch.zeros(1, dtype=torch.float64),
        )
    )
    source_norm = math.sqrt(float(feature_energy.sum().item()))

    block_errors = {}
    for size in QUANT_BLOCK_SIZES:
        if size <= rank and quant_group_bits(size, "float32") <= max_budget:
            for start in range(rank - size + 1):
                block_errors[(start, size, "float32")] = 0.0

    bf16_reconstructed = projected_data.to(torch.bfloat16).to(torch.float32)
    bf16_feature_errors = (projected_data - bf16_reconstructed).square().sum(dim=0).cpu()
    block_errors.update(
        _prefix_block_errors(
            bf16_feature_errors,
            QUANT_BLOCK_SIZES,
            "bfloat16",
            max_budget,
        )
    )
    del bf16_reconstructed, bf16_feature_errors

    if projected_data.device.type == "npu":
        block_errors.update(
            _build_npu_integer_errors(
                projected_data,
                max_budget,
                npu_workspace_mb,
            )
        )
        torch.npu.synchronize()
    else:
        block_errors.update(_build_cpu_integer_errors(projected_data, max_budget))

    logger.info(
        "Built %s quantization block errors on %s in %.2fs",
        len(block_errors),
        projected_data.device.type,
        time.perf_counter() - started,
    )
    return QuantizationErrorTable(
        rank=rank,
        source_norm=source_norm,
        tail_errors=tuple(tail_errors_tensor.tolist()),
        block_errors=block_errors,
    )


@dataclass(frozen=True)
class DPRecord:
    error: float
    cost: int
    previous: DPRecord | None
    group: tuple[int, str] | None


def _pareto_frontier(records: list[DPRecord]) -> dict[int, DPRecord]:
    best_by_cost = {}
    for record in records:
        current = best_by_cost.get(record.cost)
        if current is None or record.error < current.error:
            best_by_cost[record.cost] = record
    frontier = {}
    best_error = math.inf
    for cost in sorted(best_by_cost):
        record = best_by_cost[cost]
        if record.error < best_error:
            frontier[cost] = record
            best_error = record.error
    return frontier


def _assign_quantization(
    error_table: QuantizationErrorTable,
    original_feature_count: int,
    compression_ratio: int,
) -> list[tuple[int, str]]:
    """Choose contiguous quantization groups within a per-token bit budget.

    Keep Pareto frontiers of error and storage cost for each retained prefix.
    Select the prefix minimizing quantization error plus omitted tail energy,
    then reconstruct its groups and merge adjacent floating-point groups.
    """
    rank = error_table.rank
    budget = math.floor(16 * original_feature_count / compression_ratio)
    frontiers = [{} for _ in range(rank + 1)]

    started = time.perf_counter()
    with tqdm(
        total=rank * len(QUANT_DTYPES),
        desc=f"DP {compression_ratio}x",
        unit="state",
        dynamic_ncols=True,
        disable=None,
    ) as progress:
        for end in range(1, rank + 1):
            candidates = []
            for dtype_name in QUANT_DTYPES:
                block_sizes = INT4_BLOCK_SIZES if dtype_name == "int4" else QUANT_BLOCK_SIZES
                for size in block_sizes:
                    start = end - size
                    group_cost = quant_group_bits(size, dtype_name)
                    if start < 0 or group_cost > budget:
                        continue
                    quant_error = error_table.block_errors[(start, size, dtype_name)]
                    if start == 0:
                        candidates.append(DPRecord(quant_error, group_cost, None, (size, dtype_name)))
                        continue
                    for previous in frontiers[start].values():
                        cost = previous.cost + group_cost
                        if cost <= budget:
                            candidates.append(
                                DPRecord(
                                    previous.error + quant_error,
                                    cost,
                                    previous,
                                    (size, dtype_name),
                                )
                            )
                progress.update()
            frontiers[end] = _pareto_frontier(candidates)

    best = None
    best_end = None
    best_total_error = math.inf
    for end in range(1, rank + 1):
        omitted_error = error_table.tail_errors[end]
        for record in frontiers[end].values():
            total_error = record.error + omitted_error
            if total_error < best_total_error or (
                total_error == best_total_error and (best is None or record.cost < best.cost)
            ):
                best = record
                best_end = end
                best_total_error = total_error
    if best is None:
        raise ValueError(
            f"Compression ratio {compression_ratio} has a {budget}-bit budget, "
            "which cannot fit a non-empty quantization schema"
        )

    relative_error = math.sqrt(best_total_error) / error_table.source_norm if error_table.source_norm else 0.0

    selected = best
    schema = []
    while best is not None:
        schema.append(best.group)
        best = best.previous
    schema.reverse()
    merged = []
    for size, dtype_name in schema:
        if merged and dtype_name in ("float32", "bfloat16") and merged[-1][1] == dtype_name:
            merged[-1] = (merged[-1][0] + size, dtype_name)
        else:
            merged.append((size, dtype_name))
    build_quant_layout(merged, block_size=1, basis_rank=rank)
    used_bits = sum(quant_group_bits(size, dtype_name) for size, dtype_name in merged)
    logger.info(
        "DP ratio=%sx p=%s rank=%s budget=%s bits_per_rank=%.4f "
        "used=%s effective_ratio=%.4fx retained=%s tail_error=%.7g "
        "quant_error=%.7g total_error=%.7g relative_error=%.7g "
        "time=%.2fs schema=%s",
        compression_ratio,
        original_feature_count,
        rank,
        budget,
        budget / rank,
        used_bits,
        16 * original_feature_count / used_bits,
        best_end,
        error_table.tail_errors[best_end],
        selected.error,
        best_total_error,
        relative_error,
        time.perf_counter() - started,
        merged,
    )

    feature_start = 0
    for group_index, (size, dtype_name) in enumerate(schema):
        feature_end = feature_start + size
        group_error = error_table.block_errors[(feature_start, size, dtype_name)]
        group_energy = error_table.tail_errors[feature_start] - error_table.tail_errors[feature_end]
        relative_group_error = math.sqrt(group_error / group_energy) if group_energy > 0 else 0.0
        logger.info(
            "DP group=%s range=%s:%s dtype=%s bits=%s energy=%.7g error=%.7g relative_error=%.7g",
            group_index,
            feature_start,
            feature_end,
            dtype_name,
            quant_group_bits(size, dtype_name),
            group_energy,
            group_error,
            relative_group_error,
        )
        feature_start = feature_end
    return merged


def assign_quantizations(
    projected_data: torch.Tensor,
    original_feature_count: int,
    compression_ratios: list[int],
    npu_workspace_mb: int = 512,
) -> dict[int, list[tuple[int, str]]]:
    max_budget = max(math.floor(16 * original_feature_count / ratio) for ratio in compression_ratios)
    error_table = build_quantization_error_table(
        projected_data,
        max_budget,
        npu_workspace_mb,
    )
    return {ratio: _assign_quantization(error_table, original_feature_count, ratio) for ratio in compression_ratios}


def assign_quantization(
    projected_data: torch.Tensor,
    original_feature_count: int,
    compression_ratio: int,
) -> list[tuple[int, str]]:
    return assign_quantizations(
        projected_data,
        original_feature_count,
        [compression_ratio],
    )[compression_ratio]


def load_pca_artifact(path: Path, workers: list[str]) -> dict:
    if not path.is_file():
        raise ValueError(f"PCA calibration file does not exist: {path}")
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise ValueError(f"PCA calibration file must contain a dictionary: {path}")

    output = {"version": KVTC_FILE_VERSION}
    for matrix_name in ("keys", "values"):
        matrix_params = artifact.get(matrix_name)
        if not isinstance(matrix_params, dict):
            raise ValueError(f"PCA calibration file is missing {matrix_name!r}: {path}")
        artifact_workers = sorted(key for key in matrix_params if key != "quant")
        if artifact_workers != sorted(workers):
            raise ValueError(
                f"PCA calibration {matrix_name} workers {artifact_workers} do not match dump workers {sorted(workers)}"
            )

        output[matrix_name] = {}
        for worker in workers:
            worker_params = matrix_params.get(worker)
            if not isinstance(worker_params, dict):
                raise ValueError(f"PCA calibration file has invalid {matrix_name}/{worker} parameters")
            mu = worker_params.get("mu")
            basis = worker_params.get("basis")
            if not isinstance(mu, torch.Tensor) or mu.ndim != 1:
                raise ValueError(f"PCA calibration {matrix_name}/{worker}/mu must be a 1-D tensor")
            if (
                not isinstance(basis, torch.Tensor)
                or basis.ndim != 2
                or basis.shape[0] != mu.shape[0]
                or basis.shape[1] == 0
            ):
                raise ValueError(
                    f"PCA calibration {matrix_name}/{worker}/basis must have shape ({mu.shape[0]}, rank>0)"
                )
            output[matrix_name][worker] = {"mu": mu, "basis": basis, "quant": {}}

    logger.info(
        "Reusing PCA parameters from %s (source version=%s)",
        path,
        artifact.get("version", "<missing>"),
    )
    return output
