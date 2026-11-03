# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""KV pool watch helpers for PR-level e2e tests.

Criteria source: ``Proposals/ascend_project/pool_ci/analysis.md §0``
- H1/H2/H3 three-level hit assertions
- Must use snapshot deltas, never cumulative values (cold-start misses would
  permanently inflate the denominator)
- Hit cases must explicitly disable the local prefix cache

Typical use inside a test::

    from tests.e2e.pull_request.kv_pool_watch import run_pool_hit_sequence

    with VllmRunner(
        ..., enable_prefix_caching=False, disable_log_stats=False, kv_transfer_config=pool_kv_transfer_config
    ) as runner:
        run_pool_hit_sequence(runner, block_size=128, prompt_tokens=16384)

The low-level helpers (``snapshot_metrics`` / ``assert_hit_ratio`` /
``assert_no_hit`` / ``assert_output_identical``) stay
public so a test can build a custom sequence when the shared one does not fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

HITS_METRIC = "vllm:external_prefix_cache_hits"
QUERIES_METRIC = "vllm:external_prefix_cache_queries"
LOAD_KEYS_METRIC = "vllm:ascend_store_load_get_keys"
DELAYED_BLOCKS_METRIC = "vllm:ascend_store_delayed_release_blocks"


@dataclass(frozen=True)
class PoolSnapshot:
    """Cumulative counters captured at a point in time.

    ``delayed_blocks`` is a gauge (``ascend_store_delayed_release_blocks``):
    blocks held by the store because the async save has not finished yet.
    A non-zero value right after a request means the pool write is still in
    flight, so the next read must not be assumed to hit without a fence.
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

    Both ``VllmRunner.get_metrics()`` and ``DPVllmRunner.get_metrics()``
    return ``name -> value`` (the latter sums over DP ranks); the raw
    ``PromMetric`` list from ``LLM.get_metrics()`` is also accepted.

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
    """Theoretical max hit ratio for a fully cached prompt.

    Only whole blocks are transferable, so the tail below ``block_size``
    can never hit. Verified on hardware: 801 tokens / 128 block -> 768/801
    = 95.88%.
    """
    if token_len <= 0:
        return 0.0
    return (token_len // block_size) * block_size / token_len


def assert_hit_ratio(
    before: PoolSnapshot,
    after: PoolSnapshot,
    *,
    prompt_tokens: int,
    block_size: int,
    tolerance: float = 0.9,
    tag: str = "",
) -> float:
    """H1 + H2: there is a hit, and the delta ratio reaches expected * tolerance."""
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
    """H3: pool-hit output must match the cold-start output token by token."""
    assert list(cold_tokens) == list(hit_tokens), (
        f"[{tag}] H3 failed: pool-hit output differs from cold start\n"
        f"  cold={list(cold_tokens)}\n  hit ={list(hit_tokens)}"
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
    tag: str = "",
) -> dict[str, float]:
    """Run R1 cold write -> R2/R3 hit -> R4 disjoint miss -> R5 A intact.

    ``replicas`` sends identical copies of a prompt in one call. The DP runner
    shards prompts across ranks, so ``replicas=dp_size`` exercises every DP
    rank in the same round instead of only rank 0.

    Calls go through ``runner.generate_greedy`` rather than ``runner.model`` so
    that both ``VllmRunner`` and ``DPVllmRunner`` are supported.
    """
    prompt_a, prompt_b = build_pool_prompts(prompt_tokens)

    def _generate(ids: list[int]) -> list[list[int]]:
        outputs = runner.generate_greedy([ids] * replicas, max_tokens)
        return [list(token_ids) for token_ids, _ in outputs]

    def _assert_outputs(cold: list[list[int]], hit: list[list[int]], round_tag: str) -> None:
        assert len(hit) == len(cold), f"[{tag} {round_tag}] replica count changed: {len(hit)} != {len(cold)}"
        for index, hit_tokens in enumerate(hit):
            assert_output_identical(cold[index], hit_tokens, tag=f"{tag} {round_tag} replica{index}")

    # R1: cold start, no pool content yet -> writes the pool
    cold = _generate(prompt_a)
    snap1 = snapshot_metrics(runner)

    # R2: first hit
    hit_1 = _generate(prompt_a)
    snap2 = snapshot_metrics(runner)
    ratio_r2 = assert_hit_ratio(snap1, snap2, prompt_tokens=prompt_tokens, block_size=block_size, tag=f"{tag} R2")
    _assert_outputs(cold, hit_1, "R2")

    # R3: repeated hit must not decay
    hit_2 = _generate(prompt_a)
    snap3 = snapshot_metrics(runner)
    ratio_r3 = assert_hit_ratio(snap2, snap3, prompt_tokens=prompt_tokens, block_size=block_size, tag=f"{tag} R3")
    _assert_outputs(cold, hit_2, "R3")

    # R4: disjoint prompt must not hit (negative assertion: key isolation)
    _generate(prompt_b)
    snap4 = snapshot_metrics(runner)
    assert_no_hit(snap3, snap4, tag=f"{tag} R4")

    # R5: writing B must not break A
    hit_3 = _generate(prompt_a)
    snap5 = snapshot_metrics(runner)
    ratio_r5 = assert_hit_ratio(snap4, snap5, prompt_tokens=prompt_tokens, block_size=block_size, tag=f"{tag} R5")
    _assert_outputs(cold, hit_3, "R5")

    summary = {"r2": ratio_r2, "r3": ratio_r3, "r5": ratio_r5, "load_keys": snap5.load_keys}
    print(
        f"[kv_pool] {tag} delta ratios R2={ratio_r2:.4f} R3={ratio_r3:.4f} R5={ratio_r5:.4f} "
        f"load_keys_total={snap5.load_keys:.0f}",
        flush=True,
    )
    return summary
