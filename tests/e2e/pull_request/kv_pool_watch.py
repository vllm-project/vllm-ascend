# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""KV pool watch helpers for PR-level e2e tests.

Counters are read as deltas; hybrid cases reset local KV between rounds.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

HITS_METRIC = "vllm:external_prefix_cache_hits"
QUERIES_METRIC = "vllm:external_prefix_cache_queries"
LOAD_KEYS_METRIC = "vllm:ascend_store_load_get_keys"
DELAYED_BLOCKS_METRIC = "vllm:ascend_store_delayed_release_blocks"


@dataclass(frozen=True)
class PoolSnapshot:
    """Point-in-time pool counters; always compare deltas."""

    hits: float
    queries: float
    load_keys: float
    delayed_blocks: float = 0.0

    def delta_queries(self, later: "PoolSnapshot") -> float:
        return later.queries - self.queries

    def delta_hits(self, later: "PoolSnapshot") -> float:
        return later.hits - self.hits

    def delta_ratio(self, later: "PoolSnapshot") -> float:
        dq = self.delta_queries(later)
        if dq <= 0:
            return 0.0
        return self.delta_hits(later) / dq


def snapshot_metrics(source: Any) -> PoolSnapshot:
    """Read counters from a runner whose ``get_metrics()`` returns a dict or a metric list."""
    metrics = source.get_metrics()
    raw: dict[str, float] = {}
    if isinstance(metrics, dict):
        raw = {name: float(value) for name, value in metrics.items()}
    else:
        for metric in metrics:
            # Histogram metrics expose count/sum/buckets, not ``value``.
            value = getattr(metric, "value", None)
            if value is None:
                continue
            raw[metric.name] = float(value)
    return PoolSnapshot(
        hits=raw.get(HITS_METRIC, 0.0),
        queries=raw.get(QUERIES_METRIC, 0.0),
        load_keys=raw.get(LOAD_KEYS_METRIC, 0.0),
        delayed_blocks=raw.get(DELAYED_BLOCKS_METRIC, 0.0),
    )


def expected_hit_ratio(token_len: int, block_size: int) -> float:
    """Max hit ratio for a fully cached prompt (non-spec cases)."""
    if token_len <= 0:
        return 0.0
    return (token_len // block_size) * block_size / token_len


def expected_hit_token_count(prompt_tokens: int, block_size: int, lcm_block_size: int | None = None) -> int:
    """Exact hit tokens reported by the engine (spec/hybrid and non-spec)."""
    if prompt_tokens <= 0:
        return 0
    hits = (prompt_tokens // block_size) * block_size
    if lcm_block_size:
        # Only whole lcm blocks are transferable; the tail never hits.
        saveable = (prompt_tokens // lcm_block_size) * lcm_block_size
        hits = min(hits, saveable)
        final_block_start = ((prompt_tokens - 1) // lcm_block_size) * lcm_block_size
        if hits > final_block_start:
            hits -= lcm_block_size
    if hits == prompt_tokens:
        hits -= 1
    return max(0, hits)


def assert_hit_ratio(
    before: PoolSnapshot,
    after: PoolSnapshot,
    *,
    prompt_tokens: int,
    block_size: int,
    tolerance: float = 0.9,
    tag: str = "",
) -> float:
    """H1/H2 for non-spec cases: hit exists and ratio >= expected * tolerance."""
    ratio = before.delta_ratio(after)
    expected = expected_hit_ratio(prompt_tokens, block_size)
    floor = expected * tolerance

    assert before.delta_hits(after) > 0, (
        f"[{tag}] H1 failed: no external hit "
        f"(hits Δ{before.delta_hits(after):.0f}, queries Δ{before.delta_queries(after):.0f})"
    )
    assert ratio >= floor, (
        f"[{tag}] H2 failed: delta hit ratio {ratio:.4f} < {floor:.4f} "
        f"(expected {expected:.4f}, hits Δ{before.delta_hits(after):.0f}, "
        f"queries Δ{before.delta_queries(after):.0f})"
    )
    return ratio


def assert_hit_tokens(before: PoolSnapshot, after: PoolSnapshot, *, expected: int, tag: str = "") -> float:
    """H1 + exact hit-token assertion for spec/hybrid cases."""
    delta = before.delta_hits(after)
    assert delta > 0, (
        f"[{tag}] H1 failed: no external hit "
        f"(hits Δ{delta:.0f}, queries Δ{before.delta_queries(after):.0f}); "
        f"expected {expected} hit tokens"
    )
    assert delta == expected, (
        f"[{tag}] H2 failed: hit tokens Δ{delta:.0f} != expected {expected} "
        f"(queries Δ{before.delta_queries(after):.0f}); "
        "spec-decode/hybrid must assert exact tokens, not a ratio"
    )
    return delta


def assert_no_hit(before: PoolSnapshot, after: PoolSnapshot, *, tag: str = "") -> None:
    """R4: a disjoint prompt must not hit (key isolation)."""
    dh = before.delta_hits(after)
    assert dh == 0, (
        f"[{tag}] isolation failed: expected 0 external hits for a disjoint prompt, "
        f"got Δ{dh} (key collision or stale entry)"
    )


def assert_output_identical(cold_tokens: Sequence[int], hit_tokens: Sequence[int], *, tag: str = "") -> None:
    """H3: pool-hit tokens must match cold-start tokens exactly."""
    cold = list(cold_tokens)
    hit = list(hit_tokens)
    if cold == hit:
        return
    first_diff = next(
        (i for i, (c, h) in enumerate(zip(cold, hit)) if c != h),
        min(len(cold), len(hit)),
    )
    num_diff = sum(1 for c, h in zip(cold, hit) if c != h) + abs(len(cold) - len(hit))
    raise AssertionError(
        f"[{tag}] H3 failed: silent data corruption, pool-hit tokens differ from the cold start\n"
        f"  len(cold)={len(cold)} len(hit)={len(hit)} num_diff={num_diff} "
        f"first_diff={first_diff} cold_tail={cold[-8:]} hit_tail={hit[-8:]}"
    )


# Model-independent token ids; all PR pool vocabs exceed base + span.
POOL_PROMPT_TOKEN_BASE = 1000
POOL_PROMPT_TOKEN_SPAN = 7000
POOL_MAX_TOKENS = 16


def build_pool_prompts(prompt_tokens: int) -> tuple[list[int], list[int]]:
    """Return two token-id prompts of ``prompt_tokens``; the first block differs (R4)."""

    def _build(seed: int) -> list[int]:
        return [
            POOL_PROMPT_TOKEN_BASE + (seed + index * 7) % POOL_PROMPT_TOKEN_SPAN
            for index in range(prompt_tokens)
        ]

    return _build(0), _build(POOL_PROMPT_TOKEN_SPAN // 2)


def run_pool_hit_sequence(
    runner: Any,
    *,
    block_size: int,
    prompt_tokens: int,
    replicas: int = 1,
    max_tokens: int = POOL_MAX_TOKENS,
    lcm_block_size: int | None = None,
    expected_hit_tokens: int | None = None,
    reset_local_cache: bool = False,
    check_output: bool = True,
    tolerance: float = 0.9,
    tag: str = "",
) -> dict[str, Any]:
    """Run R1 cold write -> R2/R3 hit -> R4 disjoint miss.

    ``replicas`` scales with DP size. ``lcm_block_size`` / ``expected_hit_tokens``
    enable exact hit-token checks. ``reset_local_cache`` is for hybrid cases.
    ``check_output=False`` skips H3 (DSpark and dummy-weight cases).
    """
    prompt_a, prompt_b = build_pool_prompts(prompt_tokens)
    if expected_hit_tokens is None:
        expected_hits = expected_hit_token_count(prompt_tokens, block_size, lcm_block_size)
    else:
        # Measured override for models whose hash granularity equals lcm.
        expected_hits = expected_hit_tokens

    def _reset_local() -> None:
        if not reset_local_cache:
            return
        try:
            model = runner.model
        except RuntimeError as exc:
            raise AssertionError(
                "reset_local_cache requires a runner exposing `.model` (VllmRunner); "
                "DPVllmRunner does not support it"
            ) from exc
        model.reset_prefix_cache(reset_connector=False)

    def _generate(ids: list[int]) -> list[list[int]]:
        _reset_local()
        outputs = runner.generate_greedy([ids] * replicas, max_tokens)
        return [list(token_ids) for token_ids, _ in outputs]

    def _assert_outputs(cold: list[list[int]], hit: list[list[int]], round_tag: str) -> None:
        assert len(hit) == len(cold), f"[{tag} {round_tag}] replica count changed: {len(hit)} != {len(cold)}"
        for index, hit_tokens in enumerate(hit):
            assert_output_identical(cold[index], hit_tokens, tag=f"{tag} {round_tag} replica{index}")

    def _assert_hit(before: PoolSnapshot, after: PoolSnapshot, round_tag: str) -> float:
        round_full_tag = f"{tag} {round_tag}"
        if lcm_block_size:
            # One prompt copy per DP rank, so hits scale with replicas.
            return assert_hit_tokens(before, after, expected=expected_hits * replicas, tag=round_full_tag)
        return assert_hit_ratio(
            before,
            after,
            prompt_tokens=prompt_tokens,
            block_size=block_size,
            tolerance=tolerance,
            tag=round_full_tag,
        )

    # R1: cold start writes the pool; writes are visible when generate() returns.
    cold = _generate(prompt_a)
    snap1 = snapshot_metrics(runner)

    # R2: first hit
    hit_1 = _generate(prompt_a)
    snap2 = snapshot_metrics(runner)
    ratio_r2 = _assert_hit(snap1, snap2, "R2")
    if check_output:
        _assert_outputs(cold, hit_1, "R2")

    # R3: repeated hit must not decay
    hit_2 = _generate(prompt_a)
    snap3 = snapshot_metrics(runner)
    ratio_r3 = _assert_hit(snap2, snap3, "R3")
    if check_output:
        _assert_outputs(cold, hit_2, "R3")

    # R4: disjoint prompt must not hit.
    _generate(prompt_b)
    snap4 = snapshot_metrics(runner)
    assert_no_hit(snap3, snap4, tag=f"{tag} R4")

    summary = {
        "r2": ratio_r2,
        "r3": ratio_r3,
        "expected_hits": float(expected_hits),
        "expected_hits_total": float(expected_hits * replicas),
        "replicas": replicas,
        "load_keys": snap4.load_keys,
        # A/B comparison: R1 cold outputs and R3 hit-round outputs
        "cold_outputs": cold,
        "hit_outputs": hit_2,
    }
    metric_label = "hit_tokens" if lcm_block_size else "delta ratios"
    print(
        f"[kv_pool] {tag} {metric_label} R2={ratio_r2:.4f} R3={ratio_r3:.4f} "
        f"expected_hits={expected_hits} replicas={replicas} "
        f"expected_hits_total={expected_hits * replicas} "
        f"load_keys_total={snap4.load_keys:.0f}",
        flush=True,
    )
    return summary


# --- A/B runner -------------------------------------------------------------

AB_TTFT_LIMIT = 1.08
AB_THROUGHPUT_LIMIT = 0.92
AB_SAMPLES = 10
AB_WARMUP = 2
# Let the B leg release device memory and child processes before starting A.
AB_LEG_GAP_SECONDS = 10
AB_ARTIFACT = os.environ.get("KV_POOL_AB_ARTIFACT", "pool_ab_report.json")
# B-leg default 16K workload; all six PR cases pass prompt_tokens explicitly.
POOL_PROMPT_TOKENS = 16388
PREFIX_HITS_METRIC = "vllm:prefix_cache_hits"


def _preflight_probe() -> str | None:
    """Return a reason when the environment is unusable, else None."""
    return None


@dataclass
class LegResult:
    acc: bool
    hit_tokens: float
    outputs: list[list[int]]
    ttft_p50: float
    ttft_p95: float
    throughput: float
    env: dict[str, str] = field(default_factory=dict)
    ttft_mean: float = 0.0
    ttft_min: float = 0.0
    ttft_max: float = 0.0


@dataclass
class PoolABReport:
    case: str
    a: LegResult
    b: LegResult
    verdict: str
    label: str

    @property
    def delta_ttft(self) -> float:
        if not self.a.ttft_p50 or not self.b.ttft_p50:
            return 0.0
        return self.a.ttft_p50 / self.b.ttft_p50 - 1.0

    @property
    def delta_throughput(self) -> float:
        if not self.a.throughput or not self.b.throughput:
            return 0.0
        return self.a.throughput / self.b.throughput - 1.0

    def emit(self) -> None:
        for leg, result in (("A", self.a), ("B", self.b)):
            print(
                f"[kv_pool_ab] case={self.case} leg={leg} acc={int(result.acc)} "
                f"hit_tokens={result.hit_tokens:.0f} ttft_p50={result.ttft_p50:.2f} "
                f"ttft_p95={result.ttft_p95:.2f} throughput={result.throughput:.1f} "
                f"delta_ttft={self.delta_ttft:+.4f} delta_throughput={self.delta_throughput:+.4f} "
                f"verdict={self.verdict} label={self.label}",
                flush=True,
            )
        payload = asdict(self)
        payload["delta_ttft"] = self.delta_ttft
        payload["delta_throughput"] = self.delta_throughput
        with open(AB_ARTIFACT, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")


def _prefix_cache_hits(source: Any) -> float:
    """Local prefix-cache hit tokens; used to prove the B leg really hits locally."""
    metrics = source.get_metrics()
    if isinstance(metrics, dict):
        return float(metrics.get(PREFIX_HITS_METRIC, 0.0))
    for metric in metrics:
        if metric.name == PREFIX_HITS_METRIC and getattr(metric, "value", None) is not None:
            return float(metric.value)
    return 0.0


def _env_fingerprint() -> dict[str, str]:
    return {
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "vllm_use_v2_model_runner": os.environ.get("VLLM_USE_V2_MODEL_RUNNER", ""),
        "kv_pool_ab_mode": os.environ.get("KV_POOL_AB_MODE", "record"),
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def _collect_perf(
    runner: Any, prompt_tokens: int, *, reset_local_cache: bool = False
) -> tuple[float, float, float, float, float, float]:
    """Sample the hit state; max_tokens=1 latency is the TTFT proxy."""
    prompt, _ = build_pool_prompts(prompt_tokens)
    ttft: list[float] = []
    produced = 0
    started = time.perf_counter()
    for index in range(AB_WARMUP + AB_SAMPLES):
        if reset_local_cache:
            # DPVllmRunner does not expose `.model`.
            with contextlib.suppress(RuntimeError):
                runner.model.reset_prefix_cache(reset_connector=False)
        begin = time.perf_counter()
        runner.generate_greedy([prompt], 1)
        elapsed = (time.perf_counter() - begin) * 1000.0
        if index >= AB_WARMUP:
            ttft.append(elapsed)
            produced += prompt_tokens
    wall = time.perf_counter() - started
    mean = sum(ttft) / len(ttft) if ttft else 0.0
    minimum = min(ttft) if ttft else 0.0
    maximum = max(ttft) if ttft else 0.0
    return _percentile(ttft, 0.50), _percentile(ttft, 0.95), produced / wall, mean, minimum, maximum


def _control_hit_outputs(runner: Any, prompt_tokens: int, max_tokens: int) -> list[list[int]]:
    """B leg: cold write + one local-hit round; no local hit is CONTROL_DEFECT."""
    prompt, _ = build_pool_prompts(prompt_tokens)
    runner.generate_greedy([prompt], max_tokens)
    before = _prefix_cache_hits(runner)
    outputs = [list(ids) for ids, _ in runner.generate_greedy([prompt], max_tokens)]
    after = _prefix_cache_hits(runner)
    assert after > before, (
        f"CONTROL_DEFECT: B leg got no local prefix-cache hit "
        f"(prefix_cache_hits Δ{after - before:.0f})"
    )
    return outputs


def _classify(
    a_err: str | None,
    b_err: str | None,
    a_env: bool,
    b_env: bool,
    token_diff: bool,
    perf_bad: bool,
) -> str:
    if a_env or b_env:
        return "ENV_BLOCKED"
    if a_err and b_err:
        return "TEST_DEFECT"
    if b_err:
        return "CONTROL_DEFECT"
    if a_err or token_diff or perf_bad:
        return "POOL_DEFECT"
    return "PASS"


def run_pool_ab(
    case_name: str,
    build_leg: Callable[[bool, str], Any],
    accuracy: Callable[[Any], None],
    hit: Callable[[Any], dict[str, Any]] | None = None,
    *,
    prompt_tokens: int = POOL_PROMPT_TOKENS,
    max_tokens: int = POOL_MAX_TOKENS,
    token_identity: bool = True,
    perf_reset_local: bool = False,
    gate: bool = True,
) -> PoolABReport:
    mode = os.environ.get("KV_POOL_AB_MODE", "record")
    blocked = _preflight_probe()
    if blocked is not None:
        empty = LegResult(False, 0.0, [], 0.0, 0.0, 0.0, _env_fingerprint())
        report = PoolABReport(case_name, empty, empty, "FAIL", "ENV_BLOCKED")
        report.emit()
        raise AssertionError(f"ENV_BLOCKED: {blocked}")

    def _run(use_pool: bool, leg: str) -> tuple[LegResult | None, str | None, bool, bool]:
        """Return (result, error, original-assertion failed, env failed)."""
        accuracy_ok = False
        built = False
        try:
            with build_leg(use_pool, f"{case_name}_{leg}") as runner:
                built = True
                accuracy(runner)
                accuracy_ok = True
                if use_pool:
                    summary = hit(runner) if hit is not None else {}
                    outputs = summary.get("hit_outputs", [])
                    hits = float(summary.get("r3", 0.0))
                else:
                    outputs = _control_hit_outputs(runner, prompt_tokens, max_tokens)
                    # B leg records local-hit output, not pool hits.
                    hits = 0.0
                p50, p95, throughput, ttft_mean, ttft_min, ttft_max = _collect_perf(
                    runner, prompt_tokens, reset_local_cache=perf_reset_local
                )
                result = LegResult(
                    True,
                    hits,
                    outputs,
                    p50,
                    p95,
                    throughput,
                    _env_fingerprint(),
                    ttft_mean=ttft_mean,
                    ttft_min=ttft_min,
                    ttft_max=ttft_max,
                )
                return result, None, False, False
        except AssertionError as exc:
            return None, str(exc), not accuracy_ok, False
        except Exception as exc:
            # Engine/store failures are ENV_BLOCKED, not pool assertions.
            return None, f"{type(exc).__name__}: {exc}", False, not built

    b, b_err, b_fatal, b_env = _run(False, "b")
    if b_err is None:
        # Wait for the B leg to release workers and device memory.
        time.sleep(AB_LEG_GAP_SECONDS)
        a, a_err, a_fatal, a_env = _run(True, "a")
    else:
        # Control group failed: skip the pool leg.
        a, a_err, a_fatal, a_env = None, None, False, False
    empty = LegResult(False, 0.0, [], 0.0, 0.0, 0.0, _env_fingerprint())
    a_result = a or empty
    b_result = b or empty
    token_diff = token_identity and a_err is None and b_err is None and a_result.outputs != b_result.outputs
    perf_bad = (
        a_err is None
        and b_err is None
        and (
            a_result.ttft_p50 > b_result.ttft_p50 * AB_TTFT_LIMIT
            or a_result.ttft_p95 > b_result.ttft_p95 * AB_TTFT_LIMIT
            or a_result.throughput < b_result.throughput * AB_THROUGHPUT_LIMIT
        )
    )
    label = _classify(a_err, b_err, a_env, b_env, token_diff, perf_bad)
    report = PoolABReport(case_name, a_result, b_result, "PASS" if label == "PASS" else "FAIL", label)
    report.emit()
    if a_env or b_env:
        raise AssertionError(f"[{case_name}] ENV_BLOCKED: A={a_err or 'ok'}; B={b_err or 'ok'}")
    if a_fatal or b_fatal:
        raise AssertionError(
            f"[{case_name}] {label}: original assertions failed "
            f"(A={a_err or 'ok'}; B={b_err or 'ok'})"
        )
    if gate and mode == "gate" and label != "PASS":
        raise AssertionError(f"[{case_name}] {label}: A/B comparison failed")
    return report
