"""Stage4 e2e (04 §T2, M-E-b): inductor compile-backend track on the v2 model runner.

First run evidence was probe T0b-9 (stage design/stage4/_notes/t0_probe/RESULTS.md):
VLLM_USE_V2_MODEL_RUNNER=1 + the inductor track on Qwen3-0.6B, run in-process
(VLLM_ENABLE_V1_MULTIPROCESSING=0) so the compilation counter is readable in the
main process (R12: spawn-mode counters land in the worker).

Acceptance (04 §T2 test_inductor_track_mrv2):
  * non-empty greedy generation on the track engine;
  * counter deltas prove the v2 runner really consumed the compile stack:
    num_inductor_compiles > 0 and num_cudagraph_captured > 0
    (probe-calibrated 29 / 120 on this exact shape);
  * greedy output identical to the eager baseline built on the same v2 model
    runner (enforce_eager=True + the same V2 env).
"""

import os

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import PROMPTS_SHORT

_MRV2_MODEL = os.environ.get(
    "S4_MRV2_MODEL",
    "/mnt/weight/Qwen3-0.6B",
)

pytestmark = [
    pytest.mark.e2e_model("Qwen/Qwen3-0.6B"),
    pytest.mark.skipif(
        not os.path.isdir(_MRV2_MODEL),
        reason=f"model not found at {_MRV2_MODEL} (override via S4_MRV2_MODEL)",
    ),
]

MAX_TOKENS = 48  # probe T0b-9 shape

_BASE = dict(
    model_name=_MRV2_MODEL,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=8,  # probe shape: capture sizes [1, 2, 4, 8]
    # In-process teardown leaves a persistent ~30 GiB driver residual while
    # the pytest process lives (measured: 26.86/60.96 GiB free right after
    # VllmRunner.__exit__, stable for 6+ min). 0.4 * 60.96 = 24.4 GiB fits the
    # second engine into that residual; memory util does not affect greedy
    # numerics (KV budget only, far above what 0.6B x 4x48 tokens needs).
    gpu_memory_utilization=0.4,
)
_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    compilation_config={"backend": "inductor"},
)


def _greedy(kwargs):
    """Build one in-process engine, greedy-generate, return (use_v2, outputs)."""
    with VllmRunner(**kwargs) as runner:
        use_v2 = runner.model.llm_engine.vllm_config.use_v2_model_runner
        outs = runner.generate_greedy(PROMPTS_SHORT, MAX_TOKENS)
    return use_v2, outs


@wait_until_npu_memory_free(max_wait_seconds=600)
def test_inductor_track_mrv2_counters_and_eager_parity():
    from vllm.compilation.counter import compilation_counter

    # V2 opt-in is process-global state: setdefault per the suite convention
    # (w8a8 pattern), and restore in finally so sibling tests in the same
    # pytest session keep running on the v1 model runner.
    had_v2 = "VLLM_USE_V2_MODEL_RUNNER" in os.environ
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    try:
        before_compiles = compilation_counter.num_inductor_compiles
        before_captured = compilation_counter.num_cudagraph_captured

        use_v2, track_outs = _greedy(dict(_BASE, **_TRACK))

        assert use_v2, "engine did not use the v2 model runner"
        compiles = compilation_counter.num_inductor_compiles - before_compiles
        captured = compilation_counter.num_cudagraph_captured - before_captured
        assert compiles > 0, (
            f"num_inductor_compiles delta {compiles} (T0b-9 calibration: 29)"
        )
        assert captured > 0, (
            f"num_cudagraph_captured delta {captured} (T0b-9 calibration: 120)"
        )
        print(f"[mrv2] counters: num_inductor_compiles={compiles}, "
              f"num_cudagraph_captured={captured}")

        track_ids = [ids for ids, _ in track_outs]
        track_texts = [text for _, text in track_outs]
        for text in track_texts:
            assert text.strip(), "empty generation"

        # eager baseline on the same v2 model runner (compilation keys dropped,
        # mirroring utils.compare_logprobs' _COMPILATION_KEYS convention)
        use_v2_eager, eager_outs = _greedy(dict(_BASE, enforce_eager=True))
        assert use_v2_eager, "eager baseline did not use the v2 model runner"
        eager_ids = [ids for ids, _ in eager_outs]
        eager_texts = [text for _, text in eager_outs]
        assert track_ids == eager_ids, (
            "greedy token stream diverged from the v2 eager baseline:\n"
            f"track={track_ids}\neager={eager_ids}"
        )
        assert track_texts == eager_texts
    finally:
        if not had_v2:
            del os.environ["VLLM_USE_V2_MODEL_RUNNER"]
