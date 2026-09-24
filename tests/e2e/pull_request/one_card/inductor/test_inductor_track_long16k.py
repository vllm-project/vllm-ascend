"""Stage4 e2e (04 §T2, dense row long-sequence tier): inductor compile-backend
track on Qwen3-8B (bf16) with max_model_len=16384 and a fixed-length 16K
token-id prompt — the M-E long-sequence leg (#53 timeout/cache budget).

The prompt is generated token-id-wise (no tokenizer) with the deterministic
generator of docs/vllm/env/scripts/bench_inductor_perf.py ``make_prompts``
(VOCAB_SAFE=151643, ``random.Random(f"stage4-{id}")``, isl=16352, batch=1 —
16352 = 16384 − 32 output tokens so the request fits max_model_len):
every run of this test feeds the exact same ids, so the greedy
track-vs-eager comparison below is prompt-stable by construction.

Budget provenance (stage design/stage4/_notes/t0_probe/results/
long16k_track.json, #53): cold LLM init 551s of which the single compile
range (1, 8192) takes ~402s; vllm-cache footprint ~9 MiB (3 output_code);
peak HBM 53.3G/64G at gmu 0.80; greedy parity vs enforce_eager token-identical
32/32 on the same prompt. Timeout budget recommendation (not enforced here):
~13 min measured for the cold gate case — budget >=45 min per case (>=60 min
per file) for slower CI disks and contended cards.

Weights via ``STAGE4_QWEN3_8B_MODEL`` (default /mnt/weight mount); the test
skips when absent so CI without the asset stays green. Per-case cold
``VLLM_CACHE_ROOT`` under ``<workspace>/log/stage4_me/long16k`` (stage4/04 §0).
The gate case pays the cold compile; the eager-baseline case reuses the same
cache root warm (moe_tp4 runbook).
"""

import glob
import os
import random
import time
from pathlib import Path

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_QWEN3_8B = os.environ.get("STAGE4_QWEN3_8B_MODEL", "/mnt/weight/Qwen3-8B")

pytestmark = [
    pytest.mark.e2e_model("Qwen/Qwen3-8B"),
    pytest.mark.skipif(
        not os.path.isdir(_QWEN3_8B),
        reason=f"Qwen3-8B weights not found at {_QWEN3_8B} (set STAGE4_QWEN3_8B_MODEL)",
    ),
]

MAX_TOKENS = 32

# ---- make_prompts 复刻（bench_inductor_perf.py:52-63） ------------------------
VOCAB_SAFE = 151643  # 避开特殊 token 区
# 16352 = max_model_len 16384 − 32 output tokens: the 16K-class prompt fills
# the window exactly (prompt + >=1 output must fit max_model_len).
_ISL = 16352
_PROMPT_SEED = "stage4-me_16k"


def _long16k_prompt_ids() -> list[int]:
    rng = random.Random(_PROMPT_SEED)
    return [rng.randrange(VOCAB_SAFE) for _ in range(_ISL)]


_BASE = dict(
    model_name=_QWEN3_8B,
    dtype="bfloat16",
    max_model_len=16384,
    max_num_seqs=4,
    gpu_memory_utilization=0.80,
)

# Default profile (04 §T2): NO explicit cudagraph_mode — the -O2 preset
# resolves it to FULL_AND_PIECEWISE on the engine config.
_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    compilation_config={"backend": "inductor"},
)

# workspace root: vllm-ascend sits directly beneath it (…/v1)
_ROOT = Path(__file__).resolve().parents[6]


def _cold_cache() -> None:
    cache = _ROOT / "log" / "stage4_me" / "long16k"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(cache)


def _scan_marker() -> int:
    cache_root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
    hits = 0
    for path in glob.glob(
        os.path.join(cache_root, "**", "inductor_cache", "**", "output_code.py"),
        recursive=True,
    ):
        with open(path, encoding="utf-8") as f:
            hits += f.read().count("npu_triton_heuristics")
    return hits


def _dir_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def _first_divergence(eager_ids: list[int], track_ids: list[int]) -> str:
    for i, (e, t) in enumerate(zip(eager_ids, track_ids)):
        if e != t:
            return f"first divergence at token {i}: eager={e} track={t}"
    if len(eager_ids) != len(track_ids):
        return f"token count differs: eager={len(eager_ids)} track={len(track_ids)}"
    return ""


@wait_until_npu_memory_free(max_wait_seconds=600)  # ~16.4G bf16 weights
def test_inductor_track_long16k():
    """Gate: 16K-context default-profile track with a token-id 16K prompt."""
    from vllm import SamplingParams

    _cold_cache()
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True)
    prompt = [_long16k_prompt_ids()]

    t0 = time.perf_counter()  # #53 budget observation: cold-compile wall time
    with VllmRunner(**_BASE, **_TRACK) as runner:
        init_s = time.perf_counter() - t0
        final_cg = runner.model.llm_engine.vllm_config.compilation_config.cudagraph_mode
        track_outs = runner.model.generate(prompt, params)

    # (a) 32 greedy tokens out (ignore_eos pins the length)
    assert len(track_outs[0].outputs[0].token_ids) == MAX_TOKENS, (
        f"expected {MAX_TOKENS} tokens, got {len(track_outs[0].outputs[0].token_ids)}"
    )

    # (b) default profile survives the platform config updates: -O2 -> F&P
    assert getattr(final_cg, "name", str(final_cg)) == "FULL_AND_PIECEWISE", (
        f"final cudagraph_mode expected FULL_AND_PIECEWISE, got {final_cg}"
    )

    # (c) track artifact marker (spawn: compile counters live in workers)
    hits = _scan_marker()
    cache_bytes = _dir_bytes(os.environ["VLLM_CACHE_ROOT"])
    assert hits > 0, (
        f"no npu_triton_heuristics marker under {os.environ['VLLM_CACHE_ROOT']}/**/inductor_cache/"
    )
    print(f"LONG16K_OBS cold init {init_s:.0f}s, vllm cache {cache_bytes / 2**20:.0f} MiB")


@wait_until_npu_memory_free(max_wait_seconds=600)
def test_inductor_track_long16k_matches_eager():
    """Eager baseline vs the track on the same token-id 16K prompt, 04 §T2."""
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, ignore_eos=True)
    prompt = [_long16k_prompt_ids()]

    with VllmRunner(enforce_eager=True, **_BASE) as runner:
        eager_outs = runner.model.generate(prompt, params)

    with VllmRunner(**_BASE, **_TRACK) as runner:
        track_outs = runner.model.generate(prompt, params)

    divergence = _first_divergence(
        list(eager_outs[0].outputs[0].token_ids), list(track_outs[0].outputs[0].token_ids)
    )
    assert not divergence, divergence
