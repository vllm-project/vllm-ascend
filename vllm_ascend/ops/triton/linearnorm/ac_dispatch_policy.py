"""A+C G3 workload-selection candidate, derived from the frozen E2c guard.

This is an exploratory successor to the frozen boundary-guard policy. It stays stdlib-only: no
Triton, torch, device probing, runtime measurement parsing, or empirical UB
arithmetic.

E2c falsified one boundary estimate, so pair-capable selection with
N_BND > 0 is temporarily outside G2's calibrated domain. This is a generic
head-tile structural rule, not a case/SoC/version whitelist. It does not
claim that all boundary shapes overflow or repair the byte estimator.

G3 is a separate, host-only candidate performance selection after G1/G2:
tail-only chooses simple M1, and pair work requires at least 12 iterations on
every active core. The threshold is a B3-derived hypothesis, not a universal
performance guarantee. No case, SoC, CANN or toolchain approval list is used.

Relative to v1 (``wp1_ac_integrated_pair_capable_dispatch_0001``) it changes:

  * ``select_dispatch`` is restructured into two separable gates plus a
    terminal shape/resource selection:
      G1 semantic/layout (pure geometry + layout, no hardware info);
      G2 resource (calibrated screening estimate <= resolved capacity
         envelope, ``estimated_margin > 0`` to clear);
      G3 workload: after G2 clears, tail-only selects simple M1; pair work
         selects M1 when min_active_tokens // 2 < 12, otherwise retains the
         resource-feasible pair variant. This is a B3-derived hypothesis.
  * The unconditional ``soc == "unknown"`` gate is removed: SoC is an
    informational decision field only.
  * The old per-config G3 evidence/enablement gate is removed: registry
    coverage and performance reports do not gate selection. The registry
    remains diagnostic; the new G3 choice uses only the actual partition.
  * Capacity is observed from C2 only (the frozen-image backend's
    ``triton.backends.ascend.runtime.utils.ub_size_in_kbytes``, target-bound);
    the version-combo capacity whitelist (C3) is removed.  ``capacity_state ==
    "unknown"`` fails closed to M1 with an auditable reason -- never a
    fabricated budget.
  * Evidence classes and the registry loader remain as DIAGNOSTIC utilities
    (calibration anchors, experiment records); selection never consults them.

All fail-closed outcomes of v1 are preserved (any gate failure -> M1), but the
reason strings are more accurate and the decision records the resolved
capacity state, the screening estimate and the selected variant.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import regex as re

_POLICY_DIR = Path(__file__).resolve().parent
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCHEMA_ID = "phase2_wp1_ac_evidence_registry_v2"
SCHEMA_VERSION = 2
SOCR_UNKNOWN = "unknown"
_PROVENANCE_CATEGORIES = ("allocator", "numerics", "formal_performance")
_G3_MIN_PAIR_ITERATIONS = 12  # Exploratory threshold, not a validated universal optimum.

# ---------------------------------------------------------------------------
# Stdlib-only sibling module load (ac_ub_demand.py, same candidate directory).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Demand:
    module: Any

    def capacity_resolve(self, **kwargs):
        return self.module.capacity_resolve(**kwargs)

    def demand_estimate(self, **kwargs):
        return self.module.demand_estimate(**kwargs)

    def estimated_margin(self, *args):
        return self.module.estimated_margin(*args)

    def pair_boundary_screen_supported(self, num_q_heads: int) -> bool:
        return self.module.pair_boundary_screen_supported(num_q_heads)


_DEMAND_CACHE: _Demand | None = None


def _load_demand() -> _Demand:
    global _DEMAND_CACHE
    if _DEMAND_CACHE is not None:
        return _DEMAND_CACHE
    module_name = "wp1_ac_integrated_g3_workload_demand"
    spec = importlib.util.spec_from_file_location(module_name, _POLICY_DIR / "ac_ub_demand.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _DEMAND_CACHE = _Demand(module)
    return _DEMAND_CACHE


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer, not bool or another numeric type")
    return value


def _require_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return value


def _require_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a real bool")
    return value


class Variant(str, Enum):
    M1_FALLBACK = "m1_fallback"
    M2_TAIL_ONLY = "m2_tail_only"
    M2_PAIR_CAPABLE = "m2_pair_capable"


@dataclass(frozen=True)
class ShapeKey:
    """Shape, math, and layout fields bound by an evidence class."""

    num_q_heads: int
    num_kv_heads: int
    head_size: int
    has_gate: bool
    rope_dim: int
    has_bias: bool
    is_interleaved: bool
    mrope_section: tuple[int, int, int]
    eps: float
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for value, label in (
            (self.num_q_heads, "num_q_heads"),
            (self.num_kv_heads, "num_kv_heads"),
            (self.head_size, "head_size"),
            (self.rope_dim, "rope_dim"),
        ):
            _require_int(value, label)
        if self.num_q_heads <= 0 or self.num_kv_heads <= 0 or self.head_size <= 0:
            raise ValueError("shape dimensions must be positive")
        if self.rope_dim <= 0 or self.rope_dim > self.head_size or self.rope_dim % 2:
            raise ValueError("rope_dim must be positive, even, and no greater than head_size")
        _require_bool(self.has_gate, "has_gate")
        _require_bool(self.has_bias, "has_bias")
        _require_bool(self.is_interleaved, "is_interleaved")
        if type(self.mrope_section) is not tuple or len(self.mrope_section) != 3:
            raise TypeError("mrope_section must be a three-integer tuple")
        for value in self.mrope_section:
            _require_int(value, "mrope_section entry")
            if value < 0:
                raise ValueError("mrope_section entries must be non-negative")
        if 2 * sum(self.mrope_section) != self.rope_dim:
            raise ValueError("2 * sum(mrope_section) must equal rope_dim")
        if type(self.eps) is not float or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be a finite positive float")
        _require_str(self.dtype, "dtype")
        if self.dtype not in {"bfloat16", "float16"}:
            raise ValueError("dtype must be exactly bfloat16 or float16")


@dataclass(frozen=True)
class Partition:
    """The wrapper's front/tail core partition, represented without tensors."""

    vector_core_count: int
    front_core_num: int
    tail_core_num: int
    tokens_each_front_core: int
    tokens_each_tail_core: int

    def __post_init__(self) -> None:
        values = (
            self.vector_core_count,
            self.front_core_num,
            self.tail_core_num,
            self.tokens_each_front_core,
            self.tokens_each_tail_core,
        )
        for value, label in zip(
            values,
            (
                "vector_core_count",
                "front_core_num",
                "tail_core_num",
                "tokens_each_front_core",
                "tokens_each_tail_core",
            ),
        ):
            _require_int(value, label)
        if self.vector_core_count <= 0:
            raise ValueError("vector_core_count must be positive")
        if not 0 <= self.front_core_num <= self.vector_core_count:
            raise ValueError("front_core_num is outside the core count")
        if not 0 <= self.tail_core_num <= self.vector_core_count:
            raise ValueError("tail_core_num is outside the core count")
        if self.front_core_num + self.tail_core_num > self.vector_core_count:
            raise ValueError("active cores exceed the core count")
        if self.tokens_each_front_core < 0 or self.tokens_each_tail_core < 0:
            raise ValueError("per-core token counts must be non-negative")

    @property
    def active_core_count(self) -> int:
        return self.front_core_num + self.tail_core_num

    @property
    def active_loop_lengths(self) -> tuple[int, ...]:
        return (self.tokens_each_front_core,) * self.front_core_num + (self.tokens_each_tail_core,) * self.tail_core_num

    @property
    def max_active_tokens(self) -> int:
        return max(self.active_loop_lengths, default=0)

    @property
    def min_active_tokens(self) -> int:
        return min(self.active_loop_lengths, default=0)


@dataclass(frozen=True)
class CompilerConfig:
    """Compiler identity required for a class record match.

    ``multibuffer=False``, ``num_stages=1`` and ``num_warps=32`` are the frozen
    M2 config; they are not inferred from a shape or estimated from UB.
    """

    pair_capable: bool
    multibuffer: bool
    num_stages: int
    num_warps: int

    def __post_init__(self) -> None:
        _require_bool(self.pair_capable, "pair_capable")
        _require_bool(self.multibuffer, "multibuffer")
        _require_int(self.num_stages, "num_stages")
        _require_int(self.num_warps, "num_warps")
        if self.num_stages <= 0 or self.num_warps <= 0:
            raise ValueError("compiler stage/warp counts must be positive")


def _validate_provenance_dict(provenance: Any) -> dict:
    if type(provenance) is not dict:
        raise ValueError("provenance must be an object keyed by approval category")
    for category, chain in provenance.items():
        _require_str(category, "provenance category")
        if category not in _PROVENANCE_CATEGORIES:
            raise ValueError(f"unknown provenance category {category!r}")
        if type(chain) is not list:
            raise ValueError(f"provenance[{category}] must be a list")
        for record in chain:
            if type(record) is not dict:
                raise ValueError(f"provenance[{category}] entries must be objects")
            for key in ("record_path", "record_sha256", "verdict", "auth_ref"):
                _require_str(record.get(key, ""), f"provenance[{category}].{key}")
            _require_str(record.get("record_sha256", ""), f"provenance[{category}].record_sha256")
            if not _SHA256_RE.fullmatch(record.get("record_sha256", "")):
                raise ValueError(f"provenance[{category}].record_sha256 must be a lowercase SHA-256")
    return dict(provenance)


@dataclass(frozen=True)
class DispatchDecision:
    variant: Variant
    block_m: int
    pair_capable: bool
    reason: str
    profile_name: str | None
    source_sha256: str | None
    compiler: CompilerConfig | None
    partition: Partition
    gate: str = "g1_semantic"
    capacity_state: str | None = None
    envelope_bytes: int | None = None
    demand_estimate_bytes: int | None = None
    estimated_margin_bytes: int | None = None
    feasible_variants: tuple[str, ...] = ()


def make_partition(num_tokens: int, vector_core_count: int) -> Partition:
    """Reproduce the existing front/tail arithmetic using host integers."""

    _require_int(num_tokens, "num_tokens")
    _require_int(vector_core_count, "vector_core_count")
    if num_tokens < 0 or vector_core_count <= 0:
        raise ValueError("num_tokens must be non-negative and core count positive")

    if num_tokens == 0:
        return Partition(
            vector_core_count=vector_core_count,
            front_core_num=0,
            tail_core_num=0,
            tokens_each_front_core=0,
            tokens_each_tail_core=0,
        )

    front_core_num = vector_core_count
    if num_tokens % vector_core_count != 0:
        front_core_num = num_tokens % vector_core_count
    tokens_each_front_core = (num_tokens + vector_core_count - 1) // vector_core_count
    tail_core_num = 0
    if num_tokens > vector_core_count:
        tail_core_num = vector_core_count - front_core_num
    tokens_each_tail_core = num_tokens // vector_core_count
    return Partition(
        vector_core_count=vector_core_count,
        front_core_num=front_core_num,
        tail_core_num=tail_core_num,
        tokens_each_front_core=tokens_each_front_core,
        tokens_each_tail_core=tokens_each_tail_core,
    )


def _m1_fallback(
    partition: Partition,
    reason: str,
    *,
    gate: str = "g1_semantic",
    capacity_state: str | None = None,
    envelope_bytes: int | None = None,
    demand_estimate_bytes: int | None = None,
    estimated_margin_bytes: int | None = None,
    feasible_variants: tuple[str, ...] = (),
) -> DispatchDecision:
    return DispatchDecision(
        variant=Variant.M1_FALLBACK,
        block_m=1,
        pair_capable=False,
        reason=reason,
        profile_name=None,
        source_sha256=None,
        compiler=None,
        partition=partition,
        gate=gate,
        capacity_state=capacity_state,
        envelope_bytes=envelope_bytes,
        demand_estimate_bytes=demand_estimate_bytes,
        estimated_margin_bytes=estimated_margin_bytes,
        feasible_variants=feasible_variants,
    )


def layout_contract(
    qkv,
    q_weight,
    k_weight,
    cos_sin,
    q_bias,
    k_bias,
    num_tokens,
    q_size,
    gate_size,
    kv_size,
    head_size,
    rope_dim,
) -> tuple[bool, str]:
    """Normalized frozen-ABI layout/pointer contract, returned as (valid, contract).

    ``valid`` is the strict gate: every tensor must have the exact shape the
    kernel's raw-pointer arithmetic assumes, be contiguous (so strides are
    canonical), and share the dtype and device of ``qkv``; bias tensors must
    appear as a pair (both None or both present).  The contract string carries
    the normalized identity but never gates by itself.  Unchanged from v1.
    """

    def describe(t, expected_shape):
        try:
            shape = tuple(t.shape)
            contig = bool(t.is_contiguous())
            dtype = str(t.dtype)
            device = str(t.device)
        except Exception:
            return False, "unreadable", None, None
        ok = shape == expected_shape and contig
        return ok, f"sh{'-'.join(map(str, shape))}/c{int(contig)}/{dtype}@{device}", dtype, device

    qkv_ok, qkv_tag, qkv_dtype, qkv_device = describe(qkv, (num_tokens, q_size + gate_size + 2 * kv_size))
    cos_ok, cos_tag, cos_dtype, cos_device = describe(cos_sin, (3, num_tokens, rope_dim))
    w_ok, w_tag, w_dtype, w_device = describe(q_weight, (head_size,))
    kw_ok, kw_tag, kw_dtype, kw_device = describe(k_weight, (head_size,))
    bias_pair_ok = (q_bias is None) == (k_bias is None)
    if q_bias is not None:
        qb_ok, qb_tag, qb_dtype, qb_device = describe(q_bias, (head_size,))
        kb_ok, kb_tag, kb_dtype, kb_device = describe(k_bias, (head_size,))
        bias_identity_ok = (
            qb_dtype == qkv_dtype and qb_device == qkv_device and kb_dtype == qkv_dtype and kb_device == qkv_device
        )
    else:
        qb_ok = kb_ok = True
        qb_tag = kb_tag = "absent"
        bias_identity_ok = True
    other_identity_ok = (
        cos_dtype == qkv_dtype
        and cos_device == qkv_device
        and w_dtype == qkv_dtype
        and w_device == qkv_device
        and kw_dtype == qkv_dtype
        and kw_device == qkv_device
    )
    identity_ok = qkv_dtype is not None and qkv_device is not None and other_identity_ok and bias_identity_ok
    valid = qkv_ok and cos_ok and w_ok and kw_ok and bias_pair_ok and qb_ok and kb_ok and identity_ok
    contract = (
        f"qkv={qkv_tag};cos={cos_tag};w={w_tag};kw={kw_tag};"
        f"qb={qb_tag};kb={kb_tag};pair={int(bias_pair_ok)};"
        f"d={qkv_dtype};dev={qkv_device}"
    )
    return valid, contract


# ---------------------------------------------------------------------------
# G3 evidence classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceClass:
    """A representative-point-backed evidence class (registry schema v2).

    Not an exact whitelist: a candidate is covered when every variable
    dimension falls within a representative point or a documented
    monotone-subset rule (BHQ subset / exact boundary point / per-dimension
    ranges).  ``soc`` is informational only and may be ``"unknown"``.
    """

    name: str
    variant: Variant
    toolchain_compat: dict
    capacity_class: str
    representative_points: tuple[dict, ...]
    coverage_rules: dict
    approvals: dict
    provenance: dict
    calibration: tuple[dict, ...] = ()
    soc: str = "informational_only_not_matched"
    notes: str = ""

    def __post_init__(self) -> None:
        _require_str(self.name, "class name")
        if not self.name:
            raise ValueError("class name must be non-empty")
        if self.variant not in (Variant.M2_TAIL_ONLY, Variant.M2_PAIR_CAPABLE):
            raise ValueError("only M2 variants may be evidence classes")
        if type(self.toolchain_compat) is not dict or not self.toolchain_compat:
            raise ValueError("class toolchain_compat must be a non-empty object")
        _require_str(self.capacity_class, "capacity_class")
        if not self.capacity_class:
            raise ValueError("class capacity_class must be non-empty")
        if type(self.representative_points) is not tuple or not self.representative_points:
            raise ValueError("class representative_points must be a non-empty tuple")
        if type(self.coverage_rules) is not dict:
            raise ValueError("class coverage_rules must be an object")
        if type(self.approvals) is not dict:
            raise ValueError("class approvals must be an object")
        for category in _PROVENANCE_CATEGORIES:
            if category not in self.approvals:
                raise ValueError(f"class approvals missing {category}")
            _require_bool(self.approvals[category], f"approvals[{category}]")
        if type(self.provenance) is not dict:
            raise ValueError("class provenance must be an object")
        for category in _PROVENANCE_CATEGORIES:
            if self.approvals[category] and not self.provenance.get(category):
                raise ValueError(f"approved {category} requires a non-empty provenance chain")
        if type(self.calibration) is not tuple:
            raise ValueError("class calibration must be a tuple")
        _require_str(self.soc, "class soc")
        _require_str(self.notes, "class notes")

    @property
    def fully_verified(self) -> bool:
        return all(self.approvals[c] for c in _PROVENANCE_CATEGORIES)


def _version_in_interval(version: tuple[int, int], interval: dict) -> bool:
    lo = tuple(interval["min"])
    hi = tuple(interval["max"])
    return lo <= version <= hi


def _parse_fingerprint_for_compat(fingerprint: str, demand) -> dict | None:
    return demand.module.parse_toolchain_fingerprint(fingerprint)


def toolchain_matches_compat(fingerprint: str, compat: dict, demand) -> bool:
    """A live fingerprint is compatible iff every compat dist falls in range.

    Not an exact version whitelist: any patch within a pinned major.minor
    interval is compatible; ``"unknown"`` / unparsable never matches.
    """
    parsed = _parse_fingerprint_for_compat(fingerprint, demand)
    if parsed is None:
        return False
    for name, interval in compat.items():
        if name not in parsed:
            return False
        if not _version_in_interval(parsed[name], interval):
            return False
    return True


def _bhq(num_q_heads: int) -> int:
    return num_q_heads if num_q_heads < 12 else 12


def _point_structure(geom: dict) -> tuple[int, int, int] | None:
    """(bhq, n_full, n_bnd) for a representative-point geometry."""
    try:
        num_q_heads = int(geom.get("num_q_heads", 0))
    except (TypeError, ValueError):
        return None
    if num_q_heads <= 0:
        return None
    bhq = _bhq(num_q_heads)
    n_full = num_q_heads // bhq
    n_bnd = num_q_heads - n_full * bhq
    return (bhq, n_full, n_bnd)


def class_covers_shape(cls: EvidenceClass, shape: ShapeKey) -> bool:
    """Representative-point / boundary coverage for the shape dimensions.

    Rules (pre-registered, tested):
      * structural keys (has_gate, dtype) must exactly match a representative
        point;
      * BHQ subset (``bhq_subset``): candidate BHQ equals a representative
        point's BHQ and the candidate's N_FULL_TILES <= that point's
        N_FULL_TILES (fewer tile iterations is a semantic sub-structure);
      * boundary (N_BND > 0) requires an EXACT (N_FULL, N_BND) entry in
        ``coverage_rules.boundary_points`` AND an exact-matching representative
        point; an N_BND == 0 representative point never covers N_BND > 0;
      * each variable dimension (num_kv_heads, head_size, rope_dim) must fall
        in the corresponding coverage range; single-point ranges cover only
        the exact value.
    """
    bhq = _bhq(shape.num_q_heads)
    n_full = shape.num_q_heads // bhq
    n_bnd = shape.num_q_heads - n_full * bhq
    rules = cls.coverage_rules
    if not rules.get("bhq_subset", False):
        return False

    def _in_range(value: int, key: str) -> bool:
        rng = rules.get(key)
        if not isinstance(rng, list) or len(rng) != 2:
            return False
        lo, hi = rng
        if type(lo) is not int or type(hi) is not int:
            return False
        return lo <= value <= hi

    if not (
        _in_range(shape.num_kv_heads, "num_kv_heads_range")
        and _in_range(shape.head_size, "head_size_range")
        and _in_range(shape.rope_dim, "rope_dim_range")
    ):
        return False

    # Structural exact keys + tile-structure coverage.
    covered = False
    for point in cls.representative_points:
        geom = point.get("geometry")
        if type(geom) is not dict:
            continue
        if geom.get("has_gate") != shape.has_gate:
            continue
        if geom.get("dtype") != shape.dtype:
            continue
        point_structure = _point_structure(geom)
        if point_structure is None:
            continue
        point_bhq, point_n_full, point_n_bnd = point_structure
        if point_bhq != bhq:
            continue
        if n_bnd == 0:
            # An N_BND == 0 candidate is covered by any BHQ-equal point with
            # at least as many full tiles (a strict sub-structure).  An
            # N_BND > 0 point also covers it (fewer tiles, no boundary).
            if n_full <= point_n_full:
                covered = True
                break
        else:
            # An N_BND > 0 candidate needs an EXACT (N_FULL, N_BND) match.
            if n_full == point_n_full and n_bnd == point_n_bnd:
                covered = True
                break
    if not covered:
        return False

    # Boundary heads additionally require an exact (N_FULL, N_BND) entry.
    if n_bnd > 0:
        boundary_points = rules.get("boundary_points")
        if not isinstance(boundary_points, list):
            return False
        expected = f"[{n_full},{n_bnd}]"
        if expected not in {str(point) for point in boundary_points}:
            return False
    return True


# ---------------------------------------------------------------------------
# Registry loading (schema v2)
# ---------------------------------------------------------------------------


def _parse_geometry_from_dict(d: dict) -> ShapeKey:
    mrope = d.get("mrope_section")
    if type(mrope) is not list or len(mrope) != 3:
        raise ValueError("geometry.mrope_section must be a three-integer list")
    return ShapeKey(
        num_q_heads=_require_int(d.get("num_q_heads"), "geometry.num_q_heads"),
        num_kv_heads=_require_int(d.get("num_kv_heads"), "geometry.num_kv_heads"),
        head_size=_require_int(d.get("head_size"), "geometry.head_size"),
        has_gate=_require_bool(d.get("has_gate"), "geometry.has_gate"),
        rope_dim=_require_int(d.get("rope_dim"), "geometry.rope_dim"),
        has_bias=_require_bool(d.get("has_bias", False), "geometry.has_bias"),
        is_interleaved=_require_bool(d.get("is_interleaved", True), "geometry.is_interleaved"),
        mrope_section=(
            _require_int(mrope[0], "mrope_section[0]"),
            _require_int(mrope[1], "mrope_section[1]"),
            _require_int(mrope[2], "mrope_section[2]"),
        ),
        eps=float(d.get("eps", 1e-6)),
        dtype=_require_str(d.get("dtype", "bfloat16"), "geometry.dtype"),
    )


def _parse_class_from_dict(d: dict) -> EvidenceClass:
    if type(d) is not dict:
        raise ValueError("class entry must be an object")
    variant = _require_str(d.get("variant"), "variant")
    try:
        variant_enum = Variant(variant)
    except ValueError:
        raise ValueError(f"unknown variant {variant!r}") from None
    toolchain_compat_raw = d.get("toolchain_compat")
    if type(toolchain_compat_raw) is not dict or not toolchain_compat_raw:
        raise ValueError("class toolchain_compat must be a non-empty object")
    toolchain_compat = {}
    for name, interval in toolchain_compat_raw.items():
        if type(interval) is not dict:
            raise ValueError(f"class toolchain_compat[{name}] must be an object")
        for bound in ("min", "max"):
            value = interval.get(bound)
            if type(value) is not list or len(value) != 2:
                raise ValueError(f"class toolchain_compat[{name}].{bound} must be a two-int list")
            toolchain_compat.setdefault(name, {})[bound] = (
                _require_int(value[0], f"toolchain_compat[{name}].{bound}[0]"),
                _require_int(value[1], f"toolchain_compat[{name}].{bound}[1]"),
            )
    rep_raw = d.get("representative_points")
    if type(rep_raw) is not list or not rep_raw:
        raise ValueError("class representative_points must be a non-empty list")
    representative_points: list[dict] = []
    for point in rep_raw:
        if type(point) is not dict:
            raise ValueError("representative point must be an object")
        geom = point.get("geometry")
        if type(geom) is not dict:
            raise ValueError("representative point geometry must be an object")
        _parse_geometry_from_dict(geom)  # strict type check
        outcome = _require_str(point.get("outcome"), "representative point outcome")
        if outcome not in ("allocator_accepted", "allocator_rejected"):
            raise ValueError("representative point outcome must be allocator_accepted/rejected")
        representative_points.append(dict(point))
    coverage_raw = d.get("coverage_rules")
    if type(coverage_raw) is not dict:
        raise ValueError("class coverage_rules must be an object")
    if type(coverage_raw.get("boundary_points", [])) is not list:
        raise ValueError("coverage_rules.boundary_points must be a list")
    for key in ("num_kv_heads_range", "head_size_range", "rope_dim_range"):
        value = coverage_raw.get(key)
        if type(value) is not list or len(value) != 2:
            raise ValueError(f"coverage_rules.{key} must be a two-int list")
        _require_int(value[0], f"coverage_rules.{key}[0]")
        _require_int(value[1], f"coverage_rules.{key}[1]")
    approvals_raw = d.get("approvals")
    if type(approvals_raw) is not dict:
        raise ValueError("class approvals must be an object")
    approvals = {c: _require_bool(approvals_raw.get(c, False), f"approvals.{c}") for c in _PROVENANCE_CATEGORIES}
    provenance = _validate_provenance_dict(d.get("provenance", {}))
    calibration_raw = d.get("calibration")
    calibration = tuple(calibration_raw) if type(calibration_raw) is list else ()
    soc = _require_str(d.get("soc", "informational_only_not_matched"), "soc")
    notes = _require_str(d.get("notes", ""), "notes")
    return EvidenceClass(
        name=_require_str(d.get("name"), "name"),
        variant=variant_enum,
        toolchain_compat=toolchain_compat,
        capacity_class=_require_str(d.get("capacity_class"), "capacity_class"),
        representative_points=tuple(representative_points),
        coverage_rules=dict(coverage_raw),
        approvals=approvals,
        provenance=provenance,
        calibration=calibration,
        soc=soc,
        notes=notes,
    )


def load_evidence_registry(path: Any) -> tuple[list[EvidenceClass], list[str]]:
    """Load and validate the Git-frozen evidence registry (schema v2), fail-closed.

    Returns ``(classes, errors)``.  Any read / parse / schema / per-class
    validation problem is collected in ``errors`` and the whole registry is
    excluded, so the caller's class set is empty and every request falls back
    to M1.
    """

    errors: list[str] = []
    if not isinstance(path, (str, Path)):
        return [], [f"registry path must be a path, got {type(path).__name__}"]
    registry_path = Path(path)
    if not registry_path.exists():
        return [], [f"registry not found: {registry_path}"]
    if registry_path.is_symlink():
        return [], [f"registry must not be a symlink: {registry_path}"]
    if not registry_path.is_file():
        return [], [f"registry is not a regular file: {registry_path}"]
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], [f"registry is not readable JSON: {exc}"]
    if type(raw) is not dict:
        return [], ["registry root must be an object"]
    if raw.get("schema_id") != SCHEMA_ID:
        return [], [f"registry schema_id mismatch: {raw.get('schema_id')!r}"]
    if raw.get("schema_version") != SCHEMA_VERSION:
        return [], [f"registry schema_version mismatch: {raw.get('schema_version')!r}"]
    classes_raw = raw.get("classes")
    if type(classes_raw) is not list:
        return [], ["registry classes must be a list"]
    classes: list[EvidenceClass] = []
    for index, entry in enumerate(classes_raw):
        try:
            classes.append(_parse_class_from_dict(entry))
        except (TypeError, ValueError) as exc:
            errors.append(f"class[{index}] rejected: {exc}")
            # Fail closed on the whole registry: any invalid class invalidates
            # the entire trusted set, so the caller's class list is empty and
            # every request falls back to M1.
            return [], errors
    return classes, errors


# ---------------------------------------------------------------------------
# select_dispatch (three-layer)
# ---------------------------------------------------------------------------


def _desired_variant(partition: Partition) -> Variant:
    if partition.max_active_tokens <= 1:
        return Variant.M2_TAIL_ONLY
    return Variant.M2_PAIR_CAPABLE


def select_dispatch(
    *,
    shape: ShapeKey,
    num_tokens: int,
    vector_core_count: int,
    soc: str,
    resource_profile: str,
    current_source_sha256: str,
    current_compiler_config: CompilerConfig,
    current_layout_contract: str | None = None,
    current_layout_valid: bool,
    current_capacity: dict,
) -> DispatchDecision:
    """Select an M2 variant only after semantic, resource and workload gates.

    Three separable gates plus a terminal selection:

      G1 semantic/layout (no hardware info) -- source-SHA sanity, compiler
         identity, layout validity (frozen ABI contract), active cores,
         screening-domain dtype, non-empty identity strings;
      G2 resource -- calibrated screening estimate vs resolved capacity
         envelope; clears only when ``estimated_margin > 0`` (exact-fit
         rejects) AND the pair path is in the calibrated N_BND == 0 domain.
         E2c falsified the boundary estimate at Hq=16, so N_BND > 0 pair
         requests temporarily fall back to M1. Tail-only is unaffected;
         ``capacity_state == "unknown"`` also fails closed to M1;
      G3 workload -- after G2 clears, tail-only selects the simple M1 path;
         pair work selects M1 when ``min_active_tokens // 2 < 12``.  Pair
         work at/above the trial threshold preserves the M2 decision.  No
         evidence class, per-config approval or toolchain-version gate.

    ``current_capacity`` is the raw observed capacity candidates
    (``{"c1": {...}, "c2": {...}}``, see the wrapper) plus the toolchain
    fingerprint under which C2 was observed.  The evidence registry
    (``classes``) is deliberately NOT an argument: registry coverage is a
    diagnostic/experiment record only and must not gate selection.
    """
    demand = _load_demand()

    partition = make_partition(num_tokens, vector_core_count)
    for value, label in ((soc, "current SoC"), (resource_profile, "current resource profile")):
        _require_str(value, label)
        if not value:
            return _m1_fallback(partition, f"{label} must be a non-empty string", gate="g1_semantic")
    # NOTE: soc == "unknown" is informational only; no unconditional gate.

    _require_str(current_source_sha256, "current_source_sha256")
    if not _SHA256_RE.fullmatch(current_source_sha256):
        return _m1_fallback(partition, "current source identity is not a lowercase SHA-256", gate="g1_semantic")
    if not isinstance(current_compiler_config, CompilerConfig):
        return _m1_fallback(partition, "current compiler identity is invalid", gate="g1_semantic")
    if type(current_layout_valid) is not bool:
        return _m1_fallback(partition, "current layout validity must be a bool", gate="g1_semantic")
    if not current_layout_valid:
        return _m1_fallback(partition, "current layout does not satisfy the frozen ABI contract", gate="g1_semantic")
    if partition.active_core_count == 0:
        return _m1_fallback(partition, "no active cores", gate="g1_semantic")
    if current_layout_contract is not None:
        _require_str(current_layout_contract, "current_layout_contract")
    if shape.dtype not in {"bfloat16", "float16"}:
        return _m1_fallback(
            partition,
            "semantic: dtype outside screening domain",
            gate="g1_semantic",
        )

    desired = _desired_variant(partition)

    # ---- G2: resource (calibrated screening estimate vs capacity envelope) ----
    capacity = demand.capacity_resolve(
        c1=current_capacity.get("c1", {}),
        c2=current_capacity.get("c2", {}),
        toolchain_fingerprint=current_capacity.get("toolchain_fingerprint", ""),
    )
    capacity_state = capacity["capacity_state"]
    envelope_bytes = capacity["capacity_bytes"]
    if capacity_state != "valid" or envelope_bytes is None:
        return _m1_fallback(
            partition,
            "resource: capacity unknown",
            gate="g2_resource",
            capacity_state=capacity_state,
            feasible_variants=(desired.value,),
        )

    demand_kwargs = dict(
        block_m=2,
        num_q_heads=shape.num_q_heads,
        num_kv_heads=shape.num_kv_heads,
        head_size=shape.head_size,
        has_gate=shape.has_gate,
        rope_dim=shape.rope_dim,
        dtype=shape.dtype,
        has_bias=shape.has_bias,
        multibuffer=current_compiler_config.multibuffer,
    )
    if desired is Variant.M2_PAIR_CAPABLE:
        est = demand.demand_estimate(variant="pair", **demand_kwargs)
    else:
        est = demand.demand_estimate(variant="tail", **demand_kwargs)
    margin = demand.estimated_margin(envelope_bytes, est.demand_estimate_bytes)
    if desired is Variant.M2_PAIR_CAPABLE and not demand.pair_boundary_screen_supported(shape.num_q_heads):
        return _m1_fallback(
            partition,
            "resource: pair boundary liveness uncalibrated after E2c allocator overflow",
            gate="g2_resource",
            capacity_state=capacity_state,
            envelope_bytes=envelope_bytes,
            demand_estimate_bytes=est.demand_estimate_bytes,
            estimated_margin_bytes=margin,
        )
    if margin <= 0:
        return _m1_fallback(
            partition,
            f"resource: demand estimate {est.demand_estimate_bytes} B exceeds envelope {envelope_bytes} B",
            gate="g2_resource",
            capacity_state=capacity_state,
            envelope_bytes=envelope_bytes,
            demand_estimate_bytes=est.demand_estimate_bytes,
            estimated_margin_bytes=margin,
            feasible_variants=(desired.value,),
        )

    # ---- G3: workload benefit (distinct from G2 resource feasibility) ----
    # P3's one-token tail-only route has no pair work to amortize, and its
    # integrated B3 paired result showed no clear benefit. Keep the historical
    # tail-only instance frozen for evidence, but use simple M1 in this trial.
    if desired is Variant.M2_TAIL_ONLY:
        return _m1_fallback(
            partition,
            "workload: tail-only has no pair work; resource feasible, simple M1 selected",
            gate="g3_workload",
            capacity_state=capacity_state,
            envelope_bytes=envelope_bytes,
            demand_estimate_bytes=est.demand_estimate_bytes,
            estimated_margin_bytes=margin,
            feasible_variants=(desired.value,),
        )

    min_pair_iterations = partition.min_active_tokens // 2
    if min_pair_iterations < _G3_MIN_PAIR_ITERATIONS:
        return _m1_fallback(
            partition,
            "workload: min_pair_iterations "
            f"{min_pair_iterations} below trial threshold {_G3_MIN_PAIR_ITERATIONS}; "
            "resource feasible, M1 selected",
            gate="g3_workload",
            capacity_state=capacity_state,
            envelope_bytes=envelope_bytes,
            demand_estimate_bytes=est.demand_estimate_bytes,
            estimated_margin_bytes=margin,
            feasible_variants=(desired.value,),
        )

    # ---- Terminal selection: G1/G2 cleared and G3 pair work sufficient ----
    # The pair variant comes from the real per-core partition.  The wrapper's
    # existing tail-only applicability guard remains a correctness backstop.
    # object SHA / source SHA / compile-product identity are acceptance and
    # diagnostic artefacts, never per-call enablement conditions; the live
    # source SHA is recorded here as a diagnostic field only.
    return DispatchDecision(
        variant=desired,
        block_m=2,
        pair_capable=desired is Variant.M2_PAIR_CAPABLE,
        reason=(
            "shape/resource selection: partition-derived M2 variant "
            "(semantics and resource cleared; performance pending)"
        ),
        profile_name=None,
        source_sha256=current_source_sha256,
        compiler=current_compiler_config,
        partition=partition,
        gate="shape_resource",
        capacity_state=capacity_state,
        envelope_bytes=envelope_bytes,
        demand_estimate_bytes=est.demand_estimate_bytes,
        estimated_margin_bytes=margin,
        feasible_variants=(desired.value,),
    )


__all__ = [
    "CompilerConfig",
    "DispatchDecision",
    "EvidenceClass",
    "Partition",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SOCR_UNKNOWN",
    "ShapeKey",
    "Variant",
    "class_covers_shape",
    "layout_contract",
    "load_evidence_registry",
    "make_partition",
    "select_dispatch",
    "toolchain_matches_compat",
]
