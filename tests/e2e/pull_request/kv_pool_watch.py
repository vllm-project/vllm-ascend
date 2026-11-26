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

import regex as re

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

    def delta_queries(self, later: PoolSnapshot) -> float:
        return later.queries - self.queries

    def delta_hits(self, later: PoolSnapshot) -> float:
        return later.hits - self.hits

    def delta_ratio(self, later: PoolSnapshot) -> float:
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
        return [POOL_PROMPT_TOKEN_BASE + (seed + index * 7) % POOL_PROMPT_TOKEN_SPAN for index in range(prompt_tokens)]

    return _build(0), _build(POOL_PROMPT_TOKEN_SPAN // 2)


POOL_SECRET = "8"
# The fact repeats evenly through the cached prefix, one per padding gap. A
# trailing padding gap keeps the last fact away from the question.
POOL_FACT_REPEATS = 10
POOL_ANSWER_RE = re.compile(r"answer\s*:\s*\$?\s*(-?\d+)")


def build_pool_accuracy_prompt(
    tokenizer: Any,
    target_tokens: int,
    max_model_len: int,
    secret: str = POOL_SECRET,
) -> list[int]:
    """Long natural-language prompt; the fact repeats evenly in the cached prefix."""
    padding = "This paragraph is padding for a long context test. "
    fact = f"The secret number is {secret}. "
    question = "\nQuestion: What is the secret number? The secret number is"
    unit = len(tokenizer.encode(padding, add_special_tokens=False))
    fact_unit = len(tokenizer.encode(fact, add_special_tokens=False))
    gaps = POOL_FACT_REPEATS + 1
    repeats = max(1, (target_tokens - POOL_FACT_REPEATS * fact_unit) // (gaps * unit))
    counts = [repeats] * gaps

    def _render() -> str:
        parts: list[str] = []
        for index, count in enumerate(counts):
            parts.append(padding * count)
            if index < POOL_FACT_REPEATS:
                parts.append(fact)
        parts.append(question)
        return "".join(parts)

    while True:
        ids = tokenizer.encode(_render(), add_special_tokens=False)
        missing = target_tokens - len(ids)
        if missing <= 0:
            break
        step = missing // (gaps * unit)
        if step:
            counts = [count + step for count in counts]
        else:
            # Top up one padding sentence at a time to stay inside the
            # max_model_len headroom.
            counts[counts.index(min(counts))] += 1
    assert len(ids) >= target_tokens, f"prompt {len(ids)} < target {target_tokens}"
    if len(ids) > max_model_len - 32:
        raise AssertionError(f"prompt {len(ids)} exceeds max_model_len-32={max_model_len - 32}")
    return ids


def extract_pool_answer(text: str) -> str:
    match = POOL_ANSWER_RE.search(text)
    if match is not None:
        return match.group(1)
    fallback = re.search(r"-?\d+", text)
    assert fallback is not None, f"no answer in output: {text!r}"
    return fallback.group(0)


def run_pool_hit_sequence(
    runner: Any,
    *,
    block_size: int,
    prompt_tokens: int,
    prompt_ids: list[int] | None = None,
    replicas: int = 1,
    max_tokens: int = POOL_MAX_TOKENS,
    lcm_block_size: int | None = None,
    expected_hit_tokens: int | None = None,
    reset_local_cache: bool = False,
    check_output: bool = True,
    answer_secret: str | None = None,
    tolerance: float = 0.9,
    tag: str = "",
) -> dict[str, Any]:
    """Run R1 cold write -> R2/R3 hit -> R4 disjoint miss.

    ``prompt_ids`` supplies a measured long natural-language prompt;
    ``answer_secret`` asserts the cold and hit answers. ``replicas`` scales with
    DP size. ``lcm_block_size`` / ``expected_hit_tokens`` enable exact hit-token
    checks. ``reset_local_cache`` is for hybrid cases. ``check_output=False``
    skips H3 (DSpark and dummy-weight cases).
    """
    prompt_a = prompt_ids if prompt_ids is not None else build_pool_prompts(prompt_tokens)[0]
    prompt_b = build_pool_prompts(prompt_tokens)[1]
    effective_tokens = len(prompt_a)
    if expected_hit_tokens is None:
        expected_hits = expected_hit_token_count(effective_tokens, block_size, lcm_block_size)
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
                "reset_local_cache requires a runner exposing `.model` (VllmRunner); DPVllmRunner does not support it"
            ) from exc
        model.reset_prefix_cache(reset_connector=False)

    def _generate(ids: list[int]) -> tuple[list[list[int]], list[str]]:
        _reset_local()
        outputs = runner.generate_greedy([ids] * replicas, max_tokens)
        return [list(token_ids) for token_ids, _ in outputs], [text for _, text in outputs]

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
            prompt_tokens=effective_tokens,
            block_size=block_size,
            tolerance=tolerance,
            tag=round_full_tag,
        )

    # R1: cold start writes the pool; writes are visible when generate() returns.
    cold, cold_texts = _generate(prompt_a)
    snap1 = snapshot_metrics(runner)

    # R2: first hit
    hit_1, hit_1_texts = _generate(prompt_a)
    snap2 = snapshot_metrics(runner)
    ratio_r2 = _assert_hit(snap1, snap2, "R2")
    if check_output:
        _assert_outputs(cold, hit_1, "R2")

    # R3: repeated hit must not decay
    hit_2, hit_2_texts = _generate(prompt_a)
    snap3 = snapshot_metrics(runner)
    ratio_r3 = _assert_hit(snap2, snap3, "R3")
    if check_output:
        _assert_outputs(cold, hit_2, "R3")

    if answer_secret is not None:
        for round_tag, texts in (("R1", cold_texts), ("R2", hit_1_texts), ("R3", hit_2_texts)):
            answers = [extract_pool_answer(text) for text in texts]
            assert all(answer == answer_secret for answer in answers), (
                f"[{tag} {round_tag}] answer mismatch: {answers} != {answer_secret!r}; texts={texts!r}"
            )

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
        "cold_texts": cold_texts,
        "hit_texts": hit_2_texts,
    }
    metric_label = "hit_tokens" if lcm_block_size else "delta ratios"
    print(
        f"[kv_pool] {tag} {metric_label} R2={ratio_r2:.4f} R3={ratio_r3:.4f} "
        f"expected_hits={expected_hits} replicas={replicas} "
        f"expected_hits_total={expected_hits * replicas} "
        f"prompt_tokens={effective_tokens} "
        f"load_keys_total={snap4.load_keys:.0f}",
        flush=True,
    )
    return summary


# --- A/B runner -------------------------------------------------------------

AB_THROUGHPUT_LIMIT = 0.85
# 2 warmup rounds of 2 requests, 1 s settle, then 32 timed requests at
# concurrency 16 (2 waves).
AB_WARMUP_ROUNDS = 2
AB_WARMUP_REQUESTS = 2
AB_CONCURRENCY = 16
AB_REQUESTS = AB_CONCURRENCY * 2
AB_PERF_SETTLE_SECONDS = 1.0
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
    throughput: float
    env: dict[str, str] = field(default_factory=dict)
    local_hit_tokens: float = 0.0
    acceptance: list[float] = field(default_factory=list)


@dataclass
class PoolABReport:
    case: str
    a: LegResult
    b: LegResult
    verdict: str
    label: str

    @property
    def delta_throughput(self) -> float:
        if not self.a.throughput or not self.b.throughput:
            return 0.0
        return self.a.throughput / self.b.throughput - 1.0

    def emit(self) -> None:
        for leg, result in (("A", self.a), ("B", self.b)):
            print(
                f"[kv_pool_ab] case={self.case} leg={leg} acc={int(result.acc)} "
                f"hit_tokens={result.hit_tokens:.0f} local_hit_tokens={result.local_hit_tokens:.0f} "
                f"throughput={result.throughput:.1f} "
                f"delta_throughput={self.delta_throughput:+.4f} "
                f"verdict={self.verdict} label={self.label}",
                flush=True,
            )
        payload = asdict(self)
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
        "kv_pool_ab_mode": os.environ.get("KV_POOL_AB_MODE", "record"),
    }


def _collect_perf(
    runner: Any,
    prompt_tokens: int,
    *,
    reset_local_cache: bool = False,
    seed_concurrency: bool = False,
) -> float:
    """Return input throughput for 32 requests at concurrency 16."""
    prompt, _ = build_pool_prompts(prompt_tokens)
    if seed_concurrency:
        # Hybrid align only: seed the tail block at the target concurrency so
        # the first timed wave does not measure an under-populated block.
        runner.generate_greedy([prompt] * AB_CONCURRENCY, 1)
    for _ in range(AB_WARMUP_ROUNDS):
        runner.generate_greedy([prompt] * AB_WARMUP_REQUESTS, 1)
    # Let scheduling and in-flight async writes settle before timing.
    time.sleep(AB_PERF_SETTLE_SECONDS)
    if reset_local_cache:
        # Drop the warmup-filled local blocks so the timed requests hit the pool.
        with contextlib.suppress(RuntimeError):
            runner.model.reset_prefix_cache(reset_connector=False)
    begin = time.perf_counter()
    for _ in range(AB_REQUESTS // AB_CONCURRENCY):
        runner.generate_greedy([prompt] * AB_CONCURRENCY, 1)
    elapsed = time.perf_counter() - begin
    return AB_REQUESTS * prompt_tokens / elapsed


def _control_hit_outputs(
    runner: Any,
    prompt_tokens: int,
    max_tokens: int,
    prompt_ids: list[int] | None = None,
) -> tuple[list[list[int]], float]:
    """B leg: cold write + one local-hit round; no local hit is CONTROL_DEFECT."""
    prompt = prompt_ids if prompt_ids is not None else build_pool_prompts(prompt_tokens)[0]
    runner.generate_greedy([prompt], max_tokens)
    before = _prefix_cache_hits(runner)
    outputs = [list(ids) for ids, _ in runner.generate_greedy([prompt], max_tokens)]
    after = _prefix_cache_hits(runner)
    local_hits = after - before
    assert local_hits > 0, f"CONTROL_DEFECT: B leg got no local prefix-cache hit (prefix_cache_hits Δ{local_hits:.0f})"
    return outputs, local_hits


def _report_token_diff(a: LegResult, b: LegResult) -> None:
    """Print where the pool-hit and local-hit outputs diverge."""
    print(
        f"[kv_pool_ab_diff] a_hit_tokens={a.hit_tokens:.0f} "
        f"b_local_hit_tokens={b.local_hit_tokens:.0f} "
        f"replicas={len(a.outputs)}/{len(b.outputs)}",
        flush=True,
    )
    for index, (left, right) in enumerate(zip(a.outputs, b.outputs)):
        if left == right:
            print(f"[kv_pool_ab_diff] replica{index} identical len={len(left)}", flush=True)
            continue
        limit = min(len(left), len(right))
        first = next((pos for pos in range(limit) if left[pos] != right[pos]), limit)
        num = sum(1 for pos in range(limit) if left[pos] != right[pos]) + abs(len(left) - len(right))
        print(
            f"[kv_pool_ab_diff] replica{index} len={len(left)}/{len(right)} "
            f"first_diff={first} num_diff={num} "
            f"a_tail={left[-8:]} b_tail={right[-8:]}",
            flush=True,
        )


@dataclass
class LegOutcome:
    """Result of one leg, including the phase that failed.

    ``env`` marks engine/store failures and perf noise; only those map to
    ENV_BLOCKED. Assertion failures keep the phase so the classifier can tell a
    broken control group from a pool defect.
    """

    result: LegResult | None = None
    error: str | None = None
    phase: str = "ok"
    env: bool = False


def _classify(a: LegOutcome, b: LegOutcome, token_diff: bool, perf_bad: bool, acc_bad: bool = False) -> str:
    if a.env or b.env:
        return "ENV_BLOCKED"
    if b.phase == "skipped" and a.error is not None:
        # Pool leg ran first and failed, so the control leg was skipped.
        return "POOL_DEFECT"
    if b.error is not None:
        # A is skipped when B fails. A failure in the shared accuracy function
        # is a test defect; a failure in the local-hit control is not.
        return "TEST_DEFECT" if b.phase == "accuracy" else "CONTROL_DEFECT"
    if a.error is not None or token_diff or perf_bad or acc_bad:
        return "POOL_DEFECT"
    return "PASS"


def run_pool_ab(
    case_name: str,
    build_leg: Callable[[bool, str], Any],
    accuracy: Callable[[Any], Sequence[float] | None],
    hit: Callable[[Any], dict[str, Any]] | None = None,
    *,
    prompt_tokens: int = POOL_PROMPT_TOKENS,
    prompt_ids: list[int] | None = None,
    perf_prompt_tokens: int | None = None,
    max_tokens: int = POOL_MAX_TOKENS,
    token_identity: bool = True,
    perf_reset_local: bool = False,
    perf_seed_concurrency: bool = False,
    gate: bool = True,
    pool_first: bool = False,
    acceptance_ratio: float | None = None,
) -> PoolABReport:
    perf_tokens = perf_prompt_tokens or prompt_tokens
    blocked = _preflight_probe()
    if blocked is not None:
        empty = LegResult(False, 0.0, [], 0.0, _env_fingerprint())
        report = PoolABReport(case_name, empty, empty, "FAIL", "ENV_BLOCKED")
        report.emit()
        raise AssertionError(f"ENV_BLOCKED: {blocked}")

    def _run(use_pool: bool, leg: str) -> LegOutcome:
        phase = "build"
        try:
            with build_leg(use_pool, f"{case_name}_{leg}") as runner:
                phase = "accuracy"
                rates = accuracy(runner)
                phase = "hit"
                if use_pool:
                    summary = hit(runner) if hit is not None else {}
                    outputs = summary.get("hit_outputs", [])
                    hits = float(summary.get("r3", 0.0))
                    local_hits = 0.0
                else:
                    outputs, local_hits = _control_hit_outputs(runner, prompt_tokens, max_tokens, prompt_ids)
                    # B leg records local-hit output, not pool hits.
                    hits = 0.0
                phase = "perf"
                throughput = _collect_perf(
                    runner,
                    perf_tokens,
                    reset_local_cache=perf_reset_local and use_pool,
                    seed_concurrency=perf_seed_concurrency,
                )
                result = LegResult(
                    True,
                    hits,
                    outputs,
                    throughput,
                    _env_fingerprint(),
                    local_hit_tokens=local_hits,
                    acceptance=list(rates) if rates else [],
                )
                return LegOutcome(result=result)
        except AssertionError as exc:
            # An assertion while starting the leg is an environment problem;
            # later assertions belong to the accuracy/hit phase.
            return LegOutcome(error=str(exc), phase=phase, env=phase == "build")
        except Exception as exc:
            # Engine/store failures are ENV_BLOCKED, not pool assertions.
            return LegOutcome(error=f"{type(exc).__name__}: {exc}", phase=phase, env=True)

    if pool_first:
        # Diagnostic order: surface a pool-leg hang without waiting for the
        # control leg first.
        a = _run(True, "a")
        if a.error is None:
            time.sleep(AB_LEG_GAP_SECONDS)
            b = _run(False, "b")
        else:
            b = LegOutcome(error="skipped: pool leg failed", phase="skipped")
    else:
        b = _run(False, "b")
        if b.error is None:
            # Wait for the B leg to release workers and device memory.
            time.sleep(AB_LEG_GAP_SECONDS)
            a = _run(True, "a")
        else:
            # Control group failed: skip the pool leg.
            a = LegOutcome(error="skipped: control leg failed", phase="skipped")
    empty = LegResult(False, 0.0, [], 0.0, _env_fingerprint())
    a_result = a.result or empty
    b_result = b.result or empty
    token_diff = (
        token_identity and a.result is not None and b.result is not None and a_result.outputs != b_result.outputs
    )
    perf_bad = (
        a.result is not None
        and b.result is not None
        and a_result.throughput < b_result.throughput * AB_THROUGHPUT_LIMIT
    )
    if token_diff:
        _report_token_diff(a_result, b_result)
    acc_bad = False
    if (
        acceptance_ratio is not None
        and a.result is not None
        and b.result is not None
        and a_result.acceptance
        and b_result.acceptance
    ):
        pairs = list(zip(a_result.acceptance, b_result.acceptance))
        acc_bad = any(a_val < b_val * acceptance_ratio for a_val, b_val in pairs)
        print(
            f"[kv_pool_acceptance_ab] a={a_result.acceptance} b={b_result.acceptance} "
            f"ratio={acceptance_ratio} bad={int(acc_bad)}",
            flush=True,
        )
    label = _classify(a, b, token_diff, perf_bad, acc_bad)
    report = PoolABReport(case_name, a_result, b_result, "PASS" if label == "PASS" else "FAIL", label)
    report.emit()
    if gate and label != "PASS":
        a_text = "skipped" if a.phase == "skipped" else (a.error or "ok")
        raise AssertionError(f"[{case_name}] {label}: A={a_text}; B={b.error or 'ok'}")
    return report
