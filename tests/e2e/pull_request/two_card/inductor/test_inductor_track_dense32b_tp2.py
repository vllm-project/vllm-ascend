"""Stage4 e2e (04 §T2, dense family scale-up): inductor compile-backend track
on Qwen3-32B (bf16, 64 layers) under tensor_parallel_size=2, DEFAULT -O2
profile — the dense-A9 calibration leg of M-E.

Calibration provenance (stage design/stage4/_notes/t0_probe/results/
dense32b_track.json): gpu_memory_utilization=0.85 (51.8G of the 64G budget;
30.5G bf16 weights per card) starts the engine, cold-compiles (LLM init 720s
including weight load) and generates on the default journey — Available KV
cache memory 20.0 GiB/card covers 16 seqs x 8192 ctx (needs ~128K tokens,
KV holds ~164K at 128KB/token/card), no fallback to 0.80 needed. Per-card
peak HBM observed via npu-smi sampling: 56.4G/64G (recorded there,
observation, not asserted here). The same probe's eager twin ran
token-identical on both prompts (48/48 + 48/48).

Acceptance (04 §T2, same shape as the MoE TP4 gate):
  (a) compiled-track generation is non-empty (bilingual 2 prompts);
  (b) final cudagraph_mode is FULL_AND_PIECEWISE on the engine config (the
      platform.py "track active ... (final)" state; the -O2 preset resolves
      it because no explicit cudagraph_mode is set);
  (c) track artifact marker: ``npu_triton_heuristics`` in a vllm-cache
      ``inductor_cache/`` output_code (spawn moves compile counters into the
      workers, 04 §0-2 — artifacts only).

Weights via ``STAGE4_DENSE32B_MODEL`` (default /mnt/weight mount); tests skip
when absent so CI without the asset stays green. Per-case cold
``VLLM_CACHE_ROOT`` under ``<workspace>/log/stage4_me/dense32b`` (stage4/04 §0
warm-cache discipline). Cold start ~12 min per engine (weight load dominates);
the gate case pays the cold compile, the eager-baseline case below reuses the
same cache root warm (moe_tp4 runbook). Measured full-file wall: 16 min;
budget >=45 min per case for slower CI disks.
"""

import glob
import os
from pathlib import Path

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_DENSE32B = os.environ.get("STAGE4_DENSE32B_MODEL", "/mnt/weight/Qwen3-32B")

pytestmark = [
    pytest.mark.e2e_model("Qwen/Qwen3-32B"),
    pytest.mark.skipif(
        not os.path.isdir(_DENSE32B),
        reason=f"Qwen3-32B weights not found at {_DENSE32B} (set STAGE4_DENSE32B_MODEL)",
    ),
]

PROMPTS = [
    "The capital of France is",
    "给我一句话介绍长城。",
]
MAX_TOKENS = 48

# dense-A9 calibration (M-E probe): 0.85 holds weights + KV at TP2/8192/16seqs.
_BASE = dict(
    model_name=_DENSE32B,
    dtype="bfloat16",
    max_model_len=8192,
    max_num_seqs=16,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
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
    """Cold VLLM_CACHE_ROOT (stage4/04 §0 warm-cache discipline)."""
    cache = _ROOT / "log" / "stage4_me" / "dense32b"
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


def _first_divergence(eager_ids: list[int], track_ids: list[int]) -> str:
    for i, (e, t) in enumerate(zip(eager_ids, track_ids)):
        if e != t:
            return f"first divergence at token {i}: eager={e} track={t}"
    if len(eager_ids) != len(track_ids):
        return f"token count differs: eager={len(eager_ids)} track={len(track_ids)}"
    return ""


@wait_until_npu_memory_free(max_wait_seconds=600)  # ~31G bf16 weights per card
def test_inductor_track_dense32b_tp2():
    """Gate: 32B x TP2 on the default profile (dense-A9 scale-up leg)."""
    from vllm import SamplingParams

    _cold_cache()
    greedy = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    with VllmRunner(**_BASE, **_TRACK) as runner:
        final_cg = runner.model.llm_engine.vllm_config.compilation_config.cudagraph_mode
        track_outs = runner.model.generate(PROMPTS, greedy)

    # (a) compiled track generated non-empty bilingual text
    for prompt, out in zip(PROMPTS, track_outs):
        assert out.outputs[0].text.strip(), f"empty track generation for prompt {prompt!r}"

    # (b) default profile survives the platform config updates: -O2 -> F&P
    assert getattr(final_cg, "name", str(final_cg)) == "FULL_AND_PIECEWISE", (
        f"final cudagraph_mode expected FULL_AND_PIECEWISE, got {final_cg}"
    )

    # (c) track artifact marker (spawn: compile counters live in workers)
    hits = _scan_marker()
    assert hits > 0, (
        f"no npu_triton_heuristics marker under "
        f"{os.environ.get('VLLM_CACHE_ROOT', '~/.cache/vllm')}/**/inductor_cache/"
    )


@wait_until_npu_memory_free(max_wait_seconds=600)
def test_inductor_track_dense32b_tp2_matches_eager():
    """Eager TP2 baseline vs the track (same greedy decoding), 04 §T2.

    Sequential engines in this process (VllmRunner.__exit__ releases HBM
    deterministically between them). The track engine reuses the gate case's
    cache root warm.
    """
    from vllm import SamplingParams

    greedy = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    with VllmRunner(enforce_eager=True, **_BASE) as runner:
        eager_outs = runner.model.generate(PROMPTS, greedy)

    with VllmRunner(**_BASE, **_TRACK) as runner:
        track_outs = runner.model.generate(PROMPTS, greedy)

    assert len(track_outs) == len(eager_outs)
    for i, (eager_out, track_out) in enumerate(zip(eager_outs, track_outs)):
        divergence = _first_divergence(
            list(eager_out.outputs[0].token_ids), list(track_out.outputs[0].token_ids)
        )
        assert not divergence, f"prompt {i} ({PROMPTS[i]!r}): {divergence}"
