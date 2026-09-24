"""Stage4 e2e (stage design/stage4/04 §T2, row ``test_inductor_track_tp2_default``):
FULL_AND_PIECEWISE x TP=2 on the inductor compile-backend track — the explicit
mode and the -O2 default journey.

The stage2 sibling ``test_inductor_track_tp2.py`` pins PIECEWISE with the
comment "F&P x TP=2 is an unverified gap" (debt-2 refactor: the track default
now follows the -O presets, O2 -> FULL_AND_PIECEWISE); this file closes that
gap (stage4/_notes/exploration/03_explore_dense_multicard.md ②1):

  * test_tp2_full_and_piecewise — explicit cudagraph_mode=FULL_AND_PIECEWISE;
  * test_tp2_default — no explicit cudagraph_mode; the -O2 preset must fill
    FULL_AND_PIECEWISE, asserted on the final engine config (DEFAULT-tier
    assertion transplanted from one_card test_inductor_track_full_graphs.py).
    The value is resolved by platform.py check_and_update_config in the parent
    process before workers spawn, so the engine-state read stays valid under
    spawn TP2 (only compile counters relocate to workers, R12).

Environment (stage4/04 header): ``ASCEND_RT_VISIBLE_DEVICES=<two ascending free
devices>`` (e.g. ``1,2``), ``VLLM_WORKER_MULTIPROC_METHOD=spawn``, offline hub
vars. Per-case cold compile: each test points VLLM_CACHE_ROOT at its own
directory under ``<workspace>/log/stage4_me/`` (``tp2_fp`` / ``tp2_default``) —
a warm cache hit skips compilation and would mask default-journey crashes.
"""

import os
from pathlib import Path

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_QWEN3_06B = os.environ.get("S4_QWEN3_06B_MODEL", "/mnt/weight/Qwen3-0.6B")

pytestmark = [
    pytest.mark.e2e_model("Qwen/Qwen3-0.6B"),
    pytest.mark.skipif(
        not os.path.isdir(_QWEN3_06B),
        reason=f"Qwen3-0.6B weights not found at {_QWEN3_06B}",
    ),
]

PROMPTS = [
    "The capital of France is",
    "给我一句话介绍长城。",
]

# workspace root: vllm-ascend sits directly beneath it (…/v1)
_ROOT = Path(__file__).resolve().parents[6]

# Same engine shape as the stage2 tp2 sibling; 0.55 mem util tolerates the
# slow driver reclaim between the two engines (one_card full_graphs runbook).
_BASE = dict(
    model_name=_QWEN3_06B,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=16,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.55,
)


def _cold_cache(subdir: str) -> None:
    """Per-case VLLM_CACHE_ROOT (cold compile; stage4/04 §0 warm-cache discipline)."""
    cache = _ROOT / "log" / "stage4_me" / subdir
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(cache)


def _run_track(runner_kwargs: dict):
    """Greedy-generate on the inductor track; return (final_cg, texts)."""
    from vllm import SamplingParams
    from vllm.config.compilation import CompilationConfig

    # front door (config refactor M1b): compilation_config.backend selects
    # the track; merged into the caller-supplied cudagraph_mode profile.
    runner_kwargs = dict(runner_kwargs)
    compilation_config = runner_kwargs.pop("compilation_config", None) or CompilationConfig()
    compilation_config.backend = "inductor"

    # VllmRunner: repo-conventional lifecycle whose __exit__ releases the
    # engine deterministically (raw LLM objects leak HBM across engines in
    # one process; one_card full_graphs.py runbook).
    with VllmRunner(**_BASE, compilation_config=compilation_config, **runner_kwargs) as runner:
        # final_cg AFTER check_and_update_config applied the -O presets and
        # any step-6/7 adjustments (platform.py:733 logs the same value).
        try:
            final_cg = runner.model.llm_engine.vllm_config.compilation_config.cudagraph_mode
        except Exception:
            final_cg = None
        outs = runner.model.generate(PROMPTS, SamplingParams(max_tokens=48, temperature=0.0))
    return final_cg, [o.outputs[0].text for o in outs]


@wait_until_npu_memory_free(max_wait_seconds=240)
def test_tp2_full_and_piecewise():
    """Explicit FULL_AND_PIECEWISE x TP=2 on the inductor track (gap leg 1)."""
    from vllm.config.compilation import CUDAGraphMode, CompilationConfig

    _cold_cache("tp2_fp")
    _, texts = _run_track(
        {"compilation_config": CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE)}
    )
    for text in texts:
        assert text.strip(), "empty generation"


@wait_until_npu_memory_free(max_wait_seconds=240)
def test_tp2_default():
    """Default journey x TP=2 (gap leg 2): no explicit cudagraph_mode; the -O2
    preset must fill FULL_AND_PIECEWISE on the final engine config."""
    _cold_cache("tp2_default")
    final_cg, texts = _run_track({})
    assert final_cg is not None, "could not read final cudagraph_mode"
    assert str(final_cg).endswith("FULL_AND_PIECEWISE"), (
        f"preset-sourced default expected FULL_AND_PIECEWISE, got {final_cg}"
    )
    for text in texts:
        assert text.strip(), "empty generation"
