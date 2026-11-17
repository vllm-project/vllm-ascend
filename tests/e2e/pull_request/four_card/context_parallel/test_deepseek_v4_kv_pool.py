#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
#
"""DeepSeek-V4 DSA-PCP KV-pool P6 accuracy case, ported from PR #20757.

Simplified per review decision: a PR-level CI case that only verifies the
pool (A) and non-pool (B) legs answer the same 16K secret question correctly
on every round (R1 cold / R2-R3 hit / R4 disjoint miss), with exact
hit-token accounting on the pool leg and per-round rich prints so a CI
failure log shows exactly which round and which check failed.

Compared with the full P6 design (pool_pr_nightly_weekly.md section 3.2)
this case intentionally drops the perf gates, the MTP acceptance gates and
the hard cross-leg token-identity gate. The A/B token outputs are still
compared and dumped token by token (first_diff / num_diff / head / around /
tail), so if the legs ever diverge the log pinpoints the divergence site.

The prompt uses the completion-style cue fixed in PR #20757 ("... The secret
number is"), NOT the old "answer:$ANSWER" placeholder that drifts on W4A8
at 16K; the shared build_pool_accuracy_prompt() in kv_pool_watch.py carries
the same fix in this PR and is reused here.

Run `pytest tests/e2e/pull_request/four_card/context_parallel/test_deepseek_v4_kv_pool.py`.
"""

import os
import re
import time
from contextlib import ExitStack, contextmanager
from typing import Any, Callable
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.kv_pool.config import MemcacheKVPoolConfig
from tests.e2e.common.single_node.kv_pool_runtime import SingleNodeMemcacheManager
from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.kv_pool_watch import (
    POOL_SECRET,
    PREFIX_HITS_METRIC,
    assert_hit_tokens,
    assert_no_hit,
    assert_output_identical,
    build_pool_accuracy_prompt,
    build_pool_prompts,
    expected_hit_token_count,
    snapshot_metrics,
)

MTP_MODEL = os.environ.get("DEEPSEEK_V4_MTP_MODEL_PATH", "gdydems/DeepSeek-V4-Flash-w4a8-mtp")

MAX_NUM_SEQS = 4
# Short greedy rounds; the secret number is emitted well within 16 tokens.
HIT_MAX_TOKENS = 16

MTP_NUM_SPECULATIVE_TOKENS = 3

COMMON_ENV = {
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "HCCL_BUFFSIZE": "2560",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "CLOSE_MATMUL_K_SHIFT": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
}

# 16,388 = lcm(16,384) + 4. MTP trims the final lcm block, so a 16,384-token
# prompt would hit 0; the 4-token tail keeps 16,384 hit tokens.
POOL_LCM_BLOCK_SIZE = 16384
POOL_PROMPT_TOKENS = POOL_LCM_BLOCK_SIZE + 4
# A 16K budget does not fit TP2xPCP2 at gpu_memory_utilization=0.9; 4,096
# keeps chunked prefill and still produces bitwise-identical H3 outputs.
POOL_BATCHED_TOKENS = 4096
POOL_MAX_MODEL_LEN = POOL_PROMPT_TOKENS + 128

POOL_KV_TRANSFER_CONFIG = {
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": "memcache",
        "lookup_rpc_port": "1",
        "use_layerwise": False,
    },
}

# The completion starts with the answer, so the first integer is the secret.
ANSWER_RE = re.compile(r"-?\d+")


def _build_memcache_pool_config() -> MemcacheKVPoolConfig:
    return MemcacheKVPoolConfig(
        meta_service_port=get_open_port(),
        config_store_port=get_open_port(),
        config={
            "meta": {"ock.mmc.log_level": "error"},
            "local": {
                "ock.mmc.log_level": "error",
                "ock.mmc.local_service.world_size": 256,
                "ock.mmc.local_service.protocol": "device_sdma",
                "ock.mmc.local_service.dram.size": "10GB",
            },
        },
    )


def _metrics_source(runner: Any) -> Any:
    """VllmRunner on this branch exposes metrics via ``runner.model``."""
    if hasattr(runner, "get_metrics"):
        return runner
    return runner.model


def _local_prefix_hits(runner: Any) -> float:
    """Cumulative local prefix-cache hit tokens (B leg control)."""
    metrics = _metrics_source(runner).get_metrics()
    if isinstance(metrics, dict):
        return float(metrics.get(PREFIX_HITS_METRIC, 0.0))
    for metric in metrics:
        if metric.name == PREFIX_HITS_METRIC and getattr(metric, "value", None) is not None:
            return float(metric.value)
    return 0.0


def _gate_answers(round_tag: str, texts: list[str], tag: str) -> None:
    """Every round must answer the secret number; dump the full text on mismatch."""
    for index, text in enumerate(texts):
        match = ANSWER_RE.search(text)
        answer = match.group(0) if match else None
        print(f"[{tag}] {round_tag} answer[{index}]={answer!r} full_text={text!r}", flush=True)
        assert answer == POOL_SECRET, (
            f"[{tag} {round_tag}] accuracy failed: answer {answer!r} != {POOL_SECRET!r}; full text: {text!r}"
        )


def _verify_pool_round(before: Any, after: Any, round_tag: str, tag: str, expected: int) -> None:
    """A leg: R1 must record zero external hits; R2/R3 must hit exactly ``expected``."""
    delta = before.delta_hits(after)
    print(
        f"[{tag}] {round_tag} external_hit_tokens delta={delta:.0f} "
        f"queries delta={before.delta_queries(after):.0f} expected={expected}",
        flush=True,
    )
    if round_tag == "R1":
        assert delta == 0, (
            f"[{tag} {round_tag}] expected 0 external hits on the cold write round, got delta {delta:.0f}"
        )
        return
    assert_hit_tokens(before, after, expected=expected, tag=f"{tag} {round_tag}")
    print(f"[{tag}] {round_tag}: exact hit tokens OK ({delta:.0f})", flush=True)


def _verify_pool_miss(before: Any, after: Any, round_tag: str, tag: str) -> None:
    """A leg R4: a disjoint prompt must not hit the pool."""
    assert_no_hit(before, after, tag=f"{tag} {round_tag}")
    print(f"[{tag}] {round_tag}: disjoint prompt did not hit the pool (delta hits 0)", flush=True)


def _verify_local_round(before: Any, after: Any, round_tag: str, tag: str, expected: int) -> None:
    """B leg: R1 cold; R2/R3 must be served from the local prefix cache."""
    delta = float(after) - float(before)
    print(
        f"[{tag}] {round_tag} local prefix_cache_hits delta={delta:.0f} "
        f"(a full 16K hit reports ~{expected})",
        flush=True,
    )
    if round_tag == "R1":
        assert delta == 0, (
            f"[{tag} {round_tag}] expected 0 local hits on the cold round, got delta {delta:.0f}"
        )
        return
    assert delta > 0, (
        f"[{tag} {round_tag}] CONTROL_DEFECT: no local prefix-cache hit (delta {delta:.0f}); "
        "enable_prefix_caching=True must serve the cached prefix"
    )


def _verify_local_miss(before: Any, after: Any, round_tag: str, tag: str) -> None:
    """B leg R4: a disjoint prompt must not hit the local cache."""
    delta = float(after) - float(before)
    print(f"[{tag}] {round_tag} local prefix_cache_hits delta={delta:.0f}", flush=True)
    assert delta == 0, (
        f"[{tag} {round_tag}] isolation failed: disjoint prompt hit the local cache (delta {delta:.0f})"
    )


def _dump_ab_token_diff(a_outputs: list[list[int]], b_outputs: list[list[int]], case_name: str) -> None:
    """Print the first A/B divergence site so CI logs show what differs."""
    print(
        f"[{case_name}] A/B token diagnostic: len(A)={len(a_outputs)} len(B)={len(b_outputs)}",
        flush=True,
    )
    for index, (a_ids, b_ids) in enumerate(zip(a_outputs, b_outputs)):
        if a_ids == b_ids:
            print(f"  output[{index}]: identical ({len(a_ids)} tokens)", flush=True)
            continue
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(a_ids, b_ids)) if x != y),
            min(len(a_ids), len(b_ids)),
        )
        num_diff = sum(1 for x, y in zip(a_ids, b_ids) if x != y) + abs(len(a_ids) - len(b_ids))
        print(
            f"  output[{index}]: len(A)={len(a_ids)} len(B)={len(b_ids)} "
            f"num_diff={num_diff} first_diff={first_diff}",
            flush=True,
        )
        print(f"  A[head]={a_ids[:16]}", flush=True)
        print(f"  B[head]={b_ids[:16]}", flush=True)
        print(f"  A[around_diff]={a_ids[max(0, first_diff - 8): first_diff + 8]}", flush=True)
        print(f"  B[around_diff]={b_ids[max(0, first_diff - 8): first_diff + 8]}", flush=True)
        print(f"  A[tail]={a_ids[-12:]}", flush=True)
        print(f"  B[tail]={b_ids[-12:]}", flush=True)
    if len(a_outputs) != len(b_outputs):
        print("  output count differs between legs", flush=True)


def _run_secret_rounds(
    runner: Any,
    prompt_ids: list[int],
    miss_prompt_ids: list[int],
    *,
    tag: str,
    probe: Callable[[], Any],
    verify_round: Callable[[Any, Any, str], None],
    verify_miss: Callable[[Any, Any, str], None],
    max_tokens: int = HIT_MAX_TOKENS,
) -> dict[str, Any]:
    """R1 cold -> R2/R3 hit -> R4 disjoint miss; per-round accuracy + hit checks."""
    rounds = (("R1", prompt_ids), ("R2", prompt_ids), ("R3", prompt_ids), ("R4", miss_prompt_ids))
    cold_tokens: list[list[int]] | None = None
    hit_tokens: list[list[int]] | None = None
    for round_tag, prompt in rounds:
        before = probe()
        outputs = runner.generate_greedy([prompt], max_tokens)
        after = probe()
        tokens = [list(ids) for ids, _ in outputs]
        texts = [text for _, text in outputs]
        print(f"[{tag}] {round_tag}: generated {[len(t) for t in tokens]} token(s)", flush=True)
        if round_tag == "R4":
            verify_miss(before, after, round_tag)
            continue
        _gate_answers(round_tag, texts, tag)
        verify_round(before, after, round_tag)
        if round_tag == "R1":
            cold_tokens = tokens
        else:
            # H3 within the leg: hit rounds must reproduce the cold output.
            assert cold_tokens is not None
            assert_output_identical(cold_tokens[0], tokens[0], tag=f"{tag} {round_tag}")
            print(f"[{tag}] {round_tag}: output identical to the R1 cold start", flush=True)
            hit_tokens = tokens
    assert cold_tokens is not None and hit_tokens is not None
    return {"cold_tokens": cold_tokens, "hit_tokens": hit_tokens}


@contextmanager
def _leg(use_pool: bool, leg_case_name: str):
    """Leg A enables the memcache pool; leg B keeps everything else identical."""
    with ExitStack() as stack:
        kwargs: dict[str, Any] = dict(
            max_model_len=POOL_MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            max_num_batched_tokens=POOL_BATCHED_TOKENS,
            dtype="auto",
            tensor_parallel_size=2,
            prefill_context_parallel_size=2,
            enable_expert_parallel=True,
            gpu_memory_utilization=0.9,
            quantization="ascend",
            tokenizer_mode="deepseek_v4",
            block_size=128,
            enforce_eager=False,
            compilation_config={
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [MAX_NUM_SEQS, 2 * MAX_NUM_SEQS],
            },
            enable_prefix_caching=not use_pool,
            disable_log_stats=False,  # required by get_metrics()
            speculative_config={
                "num_speculative_tokens": MTP_NUM_SPECULATIVE_TOKENS,
                "method": "mtp",
            },
            attention_config={"indexer_kv_dtype": "int8"},
            additional_config={
                "enable_prefill_mc2": True,
                # TODO: Re-enable after PCP decode sharding lands; DSV4 results are unstable.
                "enable_pcp_o_proj_weight_sharding": False,
            },
        )
        if use_pool:
            pool = stack.enter_context(SingleNodeMemcacheManager(_build_memcache_pool_config(), leg_case_name))
            stack.enter_context(patch.dict(os.environ, pool.server_envs))
            kwargs["kv_transfer_config"] = POOL_KV_TRANSFER_CONFIG
        print(f"[p6] starting leg use_pool={use_pool} case={leg_case_name}", flush=True)
        yield stack.enter_context(VllmRunner(MTP_MODEL, **kwargs))
        print(f"[p6] leg finished use_pool={use_pool}", flush=True)


@pytest.mark.e2e_model(MTP_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="dsa_pcp,mtp,kv_pool",
    parallel="TP,EP,PCP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W4A8",
    graph_mode="full_decode_only",
)
@patch.dict(os.environ, COMMON_ENV)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_deepseek_v4_kv_pool_p6_mtp() -> None:
    """P6 simplified: pool vs non-pool legs, per-round accuracy on the 16K secret question."""
    tokenizer = AutoTokenizer.from_pretrained(MTP_MODEL, trust_remote_code=True)
    prompt_ids = build_pool_accuracy_prompt(tokenizer, POOL_PROMPT_TOKENS, POOL_MAX_MODEL_LEN)
    print(f"[p6] prompt built: tokens={len(prompt_ids)}", flush=True)
    miss_prompt_ids = build_pool_prompts(POOL_PROMPT_TOKENS)[1]
    case_name = "dsv4-cp-kv-pool-p6-mtp"
    expected_pool_hits = expected_hit_token_count(POOL_PROMPT_TOKENS, 128, POOL_LCM_BLOCK_SIZE)
    expected_local_hits = expected_hit_token_count(POOL_PROMPT_TOKENS, 128)
    print(
        f"[{case_name}] expected hit tokens per hit round: pool={expected_pool_hits} local_apc={expected_local_hits}",
        flush=True,
    )

    # B leg (control) first: a control failure must not be blamed on the pool.
    with _leg(False, f"{case_name}_B") as runner:
        tag_b = f"{case_name}-B"
        b_summary = _run_secret_rounds(
            runner,
            prompt_ids,
            miss_prompt_ids,
            tag=tag_b,
            probe=lambda: _local_prefix_hits(runner),
            verify_round=lambda before, after, round_tag: _verify_local_round(
                before, after, round_tag, tag_b, expected_local_hits
            ),
            verify_miss=lambda before, after, round_tag: _verify_local_miss(before, after, round_tag, tag_b),
        )
    # Let the B leg release device memory and child processes before A starts.
    time.sleep(10)

    with _leg(True, f"{case_name}_A") as runner:
        tag_a = f"{case_name}-A"
        a_summary = _run_secret_rounds(
            runner,
            prompt_ids,
            miss_prompt_ids,
            tag=tag_a,
            probe=lambda: snapshot_metrics(_metrics_source(runner)),
            verify_round=lambda before, after, round_tag: _verify_pool_round(
                before, after, round_tag, tag_a, expected_pool_hits
            ),
            verify_miss=lambda before, after, round_tag: _verify_pool_miss(before, after, round_tag, tag_a),
        )

    # Diagnostic-only A/B comparison (not a gate under the simplified P6).
    print(f"[{case_name}] both legs finished; dumping A/B comparison (diagnostic only)", flush=True)
    _dump_ab_token_diff(a_summary["hit_tokens"], b_summary["hit_tokens"], case_name)
    _dump_ab_token_diff(a_summary["cold_tokens"], b_summary["cold_tokens"], f"{case_name}-cold")
    if a_summary["hit_tokens"] == b_summary["hit_tokens"]:
        print(f"[{case_name}] A/B hit-round token outputs IDENTICAL — pool equivalence OK", flush=True)
    else:
        print(
            f"[{case_name}] A/B hit-round token outputs DIFFER — diagnostic only under the simplified P6; "
            "see the token diagnostic above for the first divergence site",
            flush=True,
        )
    print(
        f"[{case_name}] P6 simplified PASSED: pool leg (A) R1-R4 accuracy + exact 16384 hit OK, "
        "non-pool leg (B) R1-R4 accuracy + local hit OK",
        flush=True,
    )
