"""Pure-Python KVTC schema and pool accounting, shared by scheduler and workers.

No NPU initialization and no tensor allocation here. A compression ratio is
the TOTAL target relative to BF16, including quantization metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
from dataclasses import dataclass

BITS = {"float32": 32, "bfloat16": 16, "int8": 8, "int4": 4}
STORAGE_BYTES = {"float32": 4, "bfloat16": 2, "int8": 1, "int4": 4}
INT_TYPES = ("int8", "int4")


def integer(value, name, minimum=0):
    if isinstance(value, bool) or type(value).__module__.startswith("torch"):
        raise ValueError(f"{name} must be a host integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a host integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def ratio_key(ratio):
    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio < 1 or not ratio.is_integer():
        raise ValueError("Quantized KVTC requires integer ratios >= 1")
    return str(int(ratio))


@dataclass(frozen=True)
class QuantGroup:
    start: int
    size: int
    dtype: str
    payload_start: int
    payload_end: int
    metadata_index: int | None

    @property
    def end(self):
        return self.start + self.size


@dataclass(frozen=True)
class QuantLayout:
    groups: tuple[QuantGroup, ...]
    block_size: int
    feature_count: int
    payload_elements: tuple[tuple[str, int], ...]
    metadata_count: int
    # Buckets combine groups with identical width/type into one native call.
    buckets: tuple[tuple[int, ...], ...]

    @property
    def gemm_dim(self):
        # Padding is DEVICE-ONLY. Never truncate calibrated groups to align C.
        return (self.feature_count + 15) // 16 * 16

    @property
    def block_bytes(self):
        return sum(n * STORAGE_BYTES[t] for t, n in self.payload_elements) + (self.block_size * self.metadata_count * 4)

    @property
    def schema(self):
        return tuple((g.size, g.dtype) for g in self.groups)


def build_quant_layout(schema, *, block_size, basis_rank):
    block_size = integer(block_size, "block_size", 1)
    basis_rank = integer(basis_rank, "basis_rank", 1)
    if not isinstance(schema, (list, tuple)) or not schema:
        raise ValueError("Quant schema must be a nonempty list of (size, dtype)")
    offsets = {name: 0 for name in BITS}
    groups, bucket_map = [], {}
    feature_count = metadata_count = 0
    for entry in schema:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"Invalid quant group: {entry!r}")
        size, dtype = entry
        size = integer(size, "group size", 1)
        if not isinstance(dtype, str) or dtype not in BITS:
            raise ValueError(f"Unsupported quant dtype: {dtype!r}")
        if dtype == "int4" and (size < 8 or size % 8):
            raise ValueError("Packed INT4 requires group size divisible by 8")
        count = block_size * size // (8 if dtype == "int4" else 1)
        mi = metadata_count if dtype in INT_TYPES else None
        groups.append(QuantGroup(feature_count, size, dtype, offsets[dtype], offsets[dtype] + count, mi))
        bucket_map.setdefault((size, dtype), []).append(len(groups) - 1)
        offsets[dtype] += count
        feature_count += size
        metadata_count += int(mi is not None)
    if feature_count > basis_rank:
        raise ValueError(f"Schema retains {feature_count} features; basis has {basis_rank}")
    return QuantLayout(
        tuple(groups),
        block_size,
        feature_count,
        tuple((t, n) for t, n in offsets.items() if n),
        metadata_count,
        tuple(tuple(v) for v in bucket_map.values()),
    )


def checkpoint_layouts(params, *, tp_size, k, block_size, kcr, vcr):
    """Validate all TP schemas so every process derives identical CPU slot IDs."""
    if not isinstance(params, dict) or params.get("version") != "v3-worker-quant":
        raise ValueError("Expected v3-worker-quant calibration; use kvtc_quantization=false for PCA-only")
    result = {}
    for rank in range(integer(tp_size, "tp_size", 1)):
        worker = f"tp_{rank}_pp_0"
        for kind, ratio in (("keys", kcr), ("values", vcr)):
            try:
                item = params[kind][worker]
                mu, basis = item["mu"], item["basis"]
                schema = item["quant"][ratio_key(ratio)]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Missing calibration/schema for {kind}/{worker}, ratio={ratio}") from exc
            if tuple(mu.shape) != (k,) or len(basis.shape) != 2 or basis.shape[0] != k:
                raise ValueError(f"Invalid mu/basis shape for {kind}/{worker}, expected K={k}")
            layout = build_quant_layout(schema, block_size=block_size, basis_rank=int(basis.shape[1]))
            # DP budgets are per token, not rounded after multiplication by B.
            budget_bits = math.floor(16 * k / float(ratio))
            if layout.block_bytes * 8 > block_size * budget_bits:
                raise ValueError(f"{kind}/{worker}: schema exceeds the {ratio}x byte budget")
            result[(kind, worker)] = layout
    return result


@dataclass(frozen=True)
class PoolGeometry:
    raw_block_bytes: int
    k: int
    block_size: int
    compressed_block_bytes: int
    max_comp_blocks: int
    max_sink_blocks: int
    fingerprint: str

    @property
    def num_cpu_blocks(self):
        return self.max_comp_blocks + self.max_sink_blocks

    @classmethod
    def calculate(
        cls,
        *,
        capacity,
        raw_block_bytes,
        block_size,
        kcr,
        vcr,
        raw_ratio,
        quantized,
        layouts=None,
    ):
        capacity = integer(capacity, "capacity", 1)
        raw_block_bytes = integer(raw_block_bytes, "raw_block_bytes", 1)
        block_size = integer(block_size, "block_size", 1)
        if raw_block_bytes % (4 * block_size):
            raise ValueError("Expected uniform BF16 K/V block geometry")
        k = raw_block_bytes // (4 * block_size)
        if not all(math.isfinite(float(x)) for x in (kcr, vcr, raw_ratio)):
            raise ValueError("Ratios must be finite")
        if not 0 <= raw_ratio <= 1 or (kcr > 0) != (vcr > 0):
            raise ValueError("Invalid raw ratio or partial K/V compression")
        base = capacity // raw_block_bytes
        comp_bytes = comp = 0
        schemas = []
        if kcr <= 0 and vcr <= 0:
            sink = base
        else:
            sink = int(base * raw_ratio)
            if quantized:
                if not layouts:
                    raise ValueError("Quantized pool requires all TP layouts")
                workers = sorted({w for kind, w in layouts if kind == "keys"})
                comp_bytes = max(layouts[("keys", w)].block_bytes + layouts[("values", w)].block_bytes for w in workers)
                schemas = [(kind, w, layouts[(kind, w)].schema) for w in workers for kind in ("keys", "values")]
            else:
                # Preserve the existing PCA-only unaligned quota.
                qk, qv = int(k // kcr), int(k // vcr)
                if min(qk // 16 * 16, qv // 16 * 16) <= 0 or max(qk, qv) > k:
                    raise ValueError("Invalid PCA-only compressed dimensions")
                comp_bytes = block_size * (qk + qv) * 2
            comp = (base - sink) * raw_block_bytes // comp_bytes
        if comp + sink < 2:
            raise ValueError("CPU budget needs a null slot and at least one usable slot")
        signature = (
            1,
            capacity,
            raw_block_bytes,
            block_size,
            kcr,
            vcr,
            raw_ratio,
            quantized,
            comp_bytes,
            comp,
            sink,
            schemas,
        )
        fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        return cls(raw_block_bytes, k, block_size, comp_bytes, comp, sink, fingerprint)
