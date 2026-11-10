# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""KV pool watch helpers for PR-level e2e tests.

口径来源：``Proposals/ascend_project/pool_ci/analysis.md §0``
- H1/H2/H3 三级命中断言；spec-decode/hybrid 用精确命中 token 数
- 必须用快照差值，禁止读累计值（冷启动 miss 会永久抬高分母）
- 普通命中用例必须显式关闭本地 prefix cache；hybrid（mamba align）例外，
  轮次之间只清本地 KV（``reset_connector=False``），保留池内容

Typical use inside a test::

    from tests.e2e.pull_request.kv_pool_watch import run_pool_hit_sequence

    with VllmRunner(
        ..., enable_prefix_caching=False, disable_log_stats=False, kv_transfer_config=pool_kv_transfer_config
    ) as runner:
        run_pool_hit_sequence(runner, block_size=128, prompt_tokens=16384)

Hybrid / MTP cases (Qwen3.8-27B, GLM-5.3-Flash) pass the lcm block size and
reset the local prefix cache between rounds::

    run_pool_hit_sequence(
        runner,
        block_size=128,
        prompt_tokens=lcm + 4,
        lcm_block_size=lcm,
        reset_local_cache=True,
    )

The low-level helpers (``snapshot_metrics`` / ``assert_hit_ratio`` /
``assert_hit_tokens`` / ``assert_no_hit`` / ``assert_output_identical``) stay
public so a test can build a custom sequence when the shared one does not fit.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

HITS_METRIC = "vllm:external_prefix_cache_hits"
QUERIES_METRIC = "vllm:external_prefix_cache_queries"
LOAD_KEYS_METRIC = "vllm:ascend_store_load_get_keys"
DELAYED_BLOCKS_METRIC = "vllm:ascend_store_delayed_release_blocks"


@dataclass(frozen=True)
class PoolSnapshot:
    """Cumulative counters captured at a point in time.

    ``delayed_blocks`` is a gauge (``ascend_store_delayed_release_blocks``):
    blocks the store still accounts for while releasing them. It is reported
    for diagnostics only. Measured on DSV4 TP4 16K, the pool write is visible
    as soon as ``generate()`` returns, so the hit sequence never polls this
    gauge and never inserts a fence.
    """

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
    """Read current pool counters from a runner or model exposing ``get_metrics()``.

    ``VllmRunner.get_metrics()`` and ``DPVllmRunner.get_metrics()`` both
    return ``name -> value`` (DP counters summed over ranks); a raw
    ``PromMetric`` list is still accepted for callers that pass engine metrics
    directly.

    ``LLM.get_metrics()`` requires ``disable_log_stats=False``; otherwise it
    raises ``AssertionError: Stat logging disabled``.
    """
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
    """Theoretical max hit ratio for a fully cached prompt (non-spec cases).

    Only whole blocks are transferable, so the tail below ``block_size``
    can never hit. Verified on hardware: 801 tokens / 128 block -> 768/801
    = 95.88%.
    """
    if token_len <= 0:
        return 0.0
    return (token_len // block_size) * block_size / token_len


def expected_hit_token_count(prompt_tokens: int, block_size: int, lcm_block_size: int | None = None) -> int:
    """Exact number of hit tokens the engine can report.

    Mirrors ``pool_scheduler`` (see ``_get_num_new_uncached_and_cached_tokens``):

    - non-spec: whole ``block_size`` blocks; if the hit covers the full prompt
      the scheduler drops one token, so 16,384 -> 16,383;
    - spec / MTP (``use_eagle``): when the hit reaches the prompt's final
      ``lcm_block_size`` block, that whole block is trimmed. ``lcm + 4`` ->
      ``lcm`` hits; ``2 * lcm`` -> ``lcm`` hits; exactly ``lcm`` -> 0 hits.
      Hybrid cases also pass ``lcm_block_size`` so that an ``lcm + 4`` prompt
      asserts the exact token count; with a non-zero tail the result is the
      same ``lcm`` regardless of whether the engine trims.
    """
    if prompt_tokens <= 0:
        return 0
    hits = (prompt_tokens // block_size) * block_size
    if lcm_block_size:
        # The pool only stores whole lcm blocks; a prompt tail shorter than
        # lcm is never transferred. Measured on GLM-5.3-Flash dummy
        # (lcm=1152, prompt=16,388): the engine reports 16,128 hits, not the
        # 16,384 block-granularity value.
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
    """H1 + H2: there is a hit, and the delta ratio reaches expected * tolerance.

    Only for non-spec cases. Spec-decode / hybrid must use
    :func:`assert_hit_tokens` instead.
    """
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
    """H1 + exact hit-token assertion for spec-decode / hybrid cases.

    ``expected`` comes from :func:`expected_hit_token_count`. Do not replace
    this with a 0.9 ratio: with a 16K-class lcm block the ratio still passes
    while the actual payload is wrong, and an lcm-aligned prompt legitimately
    has 0 hits.
    """
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
    """Negative assertion: a prompt with no shared prefix must not hit the pool.

    A non-zero delta means key collision or write-overwrite, which is worse
    than a miss.
    """
    dh = before.delta_hits(after)
    assert dh == 0, (
        f"[{tag}] isolation failed: expected 0 external hits for a disjoint prompt, "
        f"got Δ{dh} (key collision or stale entry)"
    )


def assert_output_identical(cold_tokens: Sequence[int], hit_tokens: Sequence[int], *, tag: str = "") -> None:
    """H3: the pool-hit run must reproduce the cold-start tokens exactly.

    Greedy decoding over a fixed token-id prompt is deterministic, so any
    divergence means corrupted KV data rather than sampling noise.
    """
    cold, hit = list(cold_tokens), list(hit_tokens)
    if cold == hit:
        return
    diff_positions = [index for index, (a, b) in enumerate(zip(cold, hit)) if a != b]
    raise AssertionError(
        f"[{tag}] H3 failed: silent data corruption, pool-hit tokens differ from the cold start\n"
        f"  len cold={len(cold)} hit={len(hit)} "
        f"num_diff={len(diff_positions) + abs(len(cold) - len(hit))} "
        f"first_diff_positions={diff_positions[:3]}\n"
        f"  cold_tail={cold[-8:]}\n  hit_tail ={hit[-8:]}"
    )


# Deterministic token-id prompts. Token IDs are model-independent here: every
# PR pool model has a vocabulary far larger than base + span, so these are
# valid embeddings for all of them.
POOL_PROMPT_TOKEN_BASE = 1000
POOL_PROMPT_TOKEN_SPAN = 7000
POOL_MAX_TOKENS = 16


def build_pool_prompts(prompt_tokens: int) -> tuple[list[int], list[int]]:
    """Return ``(prompt_a, prompt_b)`` as raw token ids of ``prompt_tokens``.

    Tokenizer-free on purpose: ``DPVllmRunner`` does not expose ``model``, so a
    DP test cannot build prompts through ``get_tokenizer()``. The two prompts
    differ in their very first token, which is enough to guarantee distinct
    block hashes for R4 (key isolation).
    """

    def _build(seed: int) -> list[int]:
        return [POOL_PROMPT_TOKEN_BASE + (seed + index * 7) % POOL_PROMPT_TOKEN_SPAN for index in range(prompt_tokens)]

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
) -> dict[str, float]:
    """Run R1 cold write -> R2/R3 hit -> R4 disjoint miss.

    Criteria: ``Proposals/ascend_project/pool_ci/analysis.md`` §0.1 / §0.2.

    ``replicas`` sends identical copies of a prompt in one call. The DP runner
    shards prompts across ranks, so ``replicas=dp_size`` exercises every DP
    rank in the same round instead of only rank 0.

    ``lcm_block_size`` switches R2/R3 to the exact hit-token assertion used by
    spec-decode and hybrid models; ``expected_hit_tokens`` overrides the
    generic formula with a measured constant when the model's hash granularity
    equals lcm (GLM-5.3-Flash dummy). ``reset_local_cache`` calls
    ``runner.model.reset_prefix_cache(reset_connector=False)`` before every
    round (hybrid cases must keep the local prefix cache enabled, and without
    the reset R2/R3 would be served locally with 0 external hits).
    ``check_output=False`` skips H3 for cases where cold/hit token identity is
    not meaningful (#11 dummy weights) or not comparable (#3 DSpark); H1/H2/R4
    stay active.

    Calls go through ``runner.generate_greedy`` rather than ``runner.model`` so
    that both ``VllmRunner`` and ``DPVllmRunner`` are supported.
    """
    prompt_a, prompt_b = build_pool_prompts(prompt_tokens)
    if expected_hit_tokens is None:
        expected_hits = expected_hit_token_count(prompt_tokens, block_size, lcm_block_size)
    else:
        # Hybrid models whose hash granularity equals lcm (dense Mamba/GDN)
        # need a measured constant: the scheduler drops the final engine block
        # before grouping into lcm chunks, which the generic formula does not
        # model. Record the measured value next to the case.
        expected_hits = expected_hit_tokens

    # DPVllmRunner sums counters over all DP ranks, and each rank serves one
    # replica of the prompt, so the pooled delta is ``replicas`` times the
    # per-prompt expectation. Ratios are scale-invariant and keep the
    # per-prompt value.
    expected_total = expected_hits * replicas

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
            return assert_hit_tokens(before, after, expected=expected_total, tag=round_full_tag)
        return assert_hit_ratio(
            before,
            after,
            prompt_tokens=prompt_tokens,
            block_size=block_size,
            tolerance=tolerance,
            tag=round_full_tag,
        )

    # R1: cold start, no pool content yet -> writes the pool.
    # No fence between R1 and R2 on purpose: measured on DSV4 TP4 16K, writes
    # are visible as soon as generate() returns (0.9999 hit on R2/R3 with no
    # sleep). If R2 ever misses intermittently in CI, check backend flush
    # behavior first -- do not paper over it with a sleep.
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

    # R4: disjoint prompt must not hit (negative assertion: key isolation)
    _generate(prompt_b)
    snap4 = snapshot_metrics(runner)
    assert_no_hit(snap3, snap4, tag=f"{tag} R4")

    summary = {
        "r2": ratio_r2,
        "r3": ratio_r3,
        "expected_hits": float(expected_hits),
        "expected_hits_total": float(expected_total),
        "load_keys": snap4.load_keys,
        # A/B 对照用：R1 冷输出与 R3 命中轮输出
        "cold_outputs": cold,
        "hit_outputs": hit_2,
    }
    if lcm_block_size:
        detail = f"hit_tokens R2={ratio_r2:.0f} R3={ratio_r3:.0f}"
    else:
        detail = f"delta_ratio R2={ratio_r2:.4f} R3={ratio_r3:.4f}"
    print(
        f"[kv_pool] {tag} {detail} expected_hits={expected_hits} replicas={replicas} "
        f"expected_hits_total={expected_total} load_keys_total={snap4.load_keys:.0f}",
        flush=True,
    )
    return summary


# --- A/B two-leg runner -----------------------------------------------------

AB_TTFT_LIMIT = 1.08
AB_THROUGHPUT_LIMIT = 0.92
AB_SAMPLES = 10
AB_WARMUP = 2
# Leave time for worker processes and NPU memory to be reclaimed between legs.
AB_LEG_GAP_SECONDS = 10
AB_ARTIFACT = os.environ.get("KV_POOL_AB_ARTIFACT", "pool_ab_report.json")
# Default 16K-class load for the performance segment; every PR case passes its
# own prompt_tokens explicitly.
POOL_PROMPT_TOKENS = 16388
PREFIX_HITS_METRIC = "vllm:prefix_cache_hits"


def _preflight_probe() -> str | None:
    """Return a reason string when the environment cannot run the A/B legs.

    CI replaces this with real probes (fabric-mem export, hugepages, loop,
    cross-node links, ports); the default does not block.
    """
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


@dataclass
class PoolABReport:
    case: str
    a: LegResult
    b: LegResult
    verdict: str
    label: str

    @property
    def delta_ttft(self) -> float:
        return (self.a.ttft_p50 / self.b.ttft_p50 - 1.0) if self.b.ttft_p50 else 0.0

    @property
    def delta_throughput(self) -> float:
        return (self.a.throughput / self.b.throughput - 1.0) if self.b.throughput else 0.0

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
    """Local prefix-cache hit tokens, used to prove the B leg really hits."""
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
) -> tuple[float, float, float]:
    """Sample the hit state; end-to-end latency with max_tokens=1 proxies TTFT.

    Hybrid legs keep the local prefix cache on. Only the pooled leg must drop
    the local KV before every sample (``reset_connector=False``), otherwise
    every sample after the first measures a local hit instead of the pool. The
    control leg intentionally keeps its warm local cache.
    """
    prompt, _ = build_pool_prompts(prompt_tokens)
    ttft: list[float] = []
    produced = 0
    started = time.perf_counter()
    for index in range(AB_WARMUP + AB_SAMPLES):
        if reset_local_cache:
            try:
                runner.model.reset_prefix_cache(reset_connector=False)
            except RuntimeError:
                pass  # DPVllmRunner does not expose `.model`
        begin = time.perf_counter()
        runner.generate_greedy([prompt], 1)
        elapsed = (time.perf_counter() - begin) * 1000.0
        if index >= AB_WARMUP:
            ttft.append(elapsed)
            produced += prompt_tokens
    wall = time.perf_counter() - started
    return _percentile(ttft, 0.50), _percentile(ttft, 0.95), produced / wall


def _control_hit_outputs(runner: Any, prompt_tokens: int, max_tokens: int) -> list[list[int]]:
    """B leg: one cold round to fill the local cache, then one hit round.

    A missing local hit is a control defect, not a pool defect.
    """
    prompt, _ = build_pool_prompts(prompt_tokens)
    runner.generate_greedy([prompt], max_tokens)
    before = _prefix_cache_hits(runner)
    outputs = [list(ids) for ids, _ in runner.generate_greedy([prompt], max_tokens)]
    after = _prefix_cache_hits(runner)
    assert after > before, (
        f"CONTROL_DEFECT: B leg got no local prefix-cache hit (prefix_cache_hits Δ{after - before:.0f})"
    )
    return outputs


def _classify(a_err: str | None, b_err: str | None, token_diff: bool, perf_bad: bool) -> str:
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
    """Run leg A (pool on, local cache off) then leg B (pool off, local cache on).

    Legs run serially in the same pytest process: HCCL/torch.distributed groups
    are process-wide singletons. The original assertions of both legs always
    fail the test; ``record`` mode only records the pool-side labels
    (``POOL_DEFECT`` / ``CONTROL_DEFECT``).
    """
    mode = os.environ.get("KV_POOL_AB_MODE", "record")
    blocked = _preflight_probe()
    if blocked is not None:
        empty = LegResult(False, 0.0, [], 0.0, 0.0, 0.0, _env_fingerprint())
        report = PoolABReport(case_name, empty, empty, "FAIL", "ENV_BLOCKED")
        report.emit()
        raise AssertionError(f"ENV_BLOCKED: {blocked}")

    def _run(use_pool: bool, leg: str) -> tuple[LegResult | None, str | None, bool]:
        """Return ``(result, error, original_assertion_failed)``.

        The third value keeps the original case assertions non-degradable: an
        engine that cannot start or an ``accuracy()`` failure must fail in both
        ``record`` and ``gate`` modes.
        """
        accuracy_ok = False
        try:
            with build_leg(use_pool, f"{case_name}_{leg}") as runner:
                accuracy(runner)
                accuracy_ok = True
                if use_pool:
                    summary = hit(runner) if hit is not None else {}
                    outputs = summary.get("hit_outputs", [])
                    hits = float(summary.get("r3", 0.0))
                else:
                    outputs = _control_hit_outputs(runner, prompt_tokens, max_tokens)
                    # The control leg owns local cache hits, not pool hits.
                    hits = 0.0
                p50, p95, throughput = _collect_perf(
                    runner, prompt_tokens, reset_local_cache=perf_reset_local and use_pool
                )
                result = LegResult(True, hits, outputs, p50, p95, throughput, _env_fingerprint())
                return result, None, False
        except AssertionError as exc:
            return None, str(exc), not accuracy_ok

    a, a_err, a_fatal = _run(True, "a")
    # Wait until worker processes and NPU memory are reclaimed, then run B.
    time.sleep(AB_LEG_GAP_SECONDS)
    b, b_err, b_fatal = _run(False, "b")
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
    label = _classify(a_err, b_err, token_diff, perf_bad)
    report = PoolABReport(case_name, a_result, b_result, "PASS" if label == "PASS" else "FAIL", label)
    report.emit()
    if a_fatal or b_fatal:
        raise AssertionError(
            f"[{case_name}] {label}: original assertions failed (A={a_err or 'ok'}; B={b_err or 'ok'})"
        )
    if gate and mode == "gate" and label != "PASS":
        raise AssertionError(f"[{case_name}] {label}: A/B comparison failed")
    return report
