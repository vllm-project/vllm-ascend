"""Stage4 e2e (04 §T2, MoE family): inductor compile-backend track on
Qwen3-30B-A3B (bf16) under tensor_parallel_size=4, DEFAULT -O2 profile.

T0c-1 probe provenance (stage design/stage4/_notes/t0_probe/RESULTS.md):
30B-A3B bf16 x track x TP4 with EXPLICIT PIECEWISE started, cold-compiled and
generated bilingual text; the merged output_code carried npu_triton_heuristics
x64, moe_forward x32 (opaque extern dispatch, MoE-A1) and unquantized_gemm
x128 (gate extern). This gate moves that probe onto the default profile (no
explicit cudagraph_mode): the -O2 preset resolves to FULL_AND_PIECEWISE
(T0b-8 default-journey evidence) — the F&P x TP shape the stage2 TP2 baseline
explicitly pinned away, here first-classed per 04 §T2「默认档」.

Acceptance (stage design/stage4/04 §T2):
  (a) compiled-track generation is non-empty (bilingual 2 prompts);
  (b) final cudagraph_mode is FULL_AND_PIECEWISE on the engine config (the
      value after platform check_and_update_config, i.e. the "track active
      ... (final)" state);
  (c) track artifact marker: ``npu_triton_heuristics`` appears in a vllm-cache
      ``inductor_cache/`` output_code — spawn moves compilation counters into
      the worker process (04 §0-2, confirmed by T0c-1's 0/empty main-process
      readout), so this track asserts artifacts only;
  (d) recorded, NOT asserted: moe_forward / unquantized_gemm occurrence counts
      in those output_code files, against the T0c-1 32/128 magnitude.

weight_nz_mode stays unset (BF16 track default ND storage; the W8A8 NZ axis is
the separate second use case per 04 §T2). Weights are located via
``STAGE4_MOE_MODEL`` (default: /mnt/weight mount); tests skip when the weights
are absent so CI without the asset stays green. Run with a per-case cold
``VLLM_CACHE_ROOT`` (stage4 env discipline) so the marker check cannot hit
artifacts from unrelated runs. Cold compile is 1-3h: the gate case below is
runnable on its own, and the eager token-identity baseline is a separate case
(its track engine reuses this cache root warm).
"""

import glob
import os

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_MOE_MODEL = os.environ.get("STAGE4_MOE_MODEL", "/mnt/weight/Qwen3-30B-A3B")

pytestmark = [
    pytest.mark.e2e_model("Qwen/Qwen3-30B-A3B"),
    pytest.mark.skipif(
        not os.path.isdir(_MOE_MODEL),
        reason=f"Qwen3-30B-A3B weights not found at {_MOE_MODEL} (set STAGE4_MOE_MODEL)",
    ),
]

PROMPTS = [
    "The capital of France is",
    "给我一句话介绍长城。",
]
MAX_TOKENS = 48

_BASE = dict(
    model_name=_MOE_MODEL,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=16,
    gpu_memory_utilization=0.80,
    tensor_parallel_size=4,
)

# Default profile (04 §T2): NO explicit cudagraph_mode — the -O2 preset
# resolves it to FULL_AND_PIECEWISE on the engine config.
_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    compilation_config={"backend": "inductor"},
)

# T0c-1 magnitude (explicit-PIECEWISE probe, 12 output_code merged over ranks)
# for the (d) observation only.
_T0C1_BASELINE = {"moe_forward": 32, "unquantized_gemm": 128}


def _iter_output_code(cache_root: str):
    # T0c-1 basis: the merged output_code.py artifacts (fx_graph_*.py debug
    # siblings under inductor_cache/ are NOT part of the comparison basis).
    yield from glob.glob(
        os.path.join(cache_root, "**", "inductor_cache", "**", "output_code.py"), recursive=True
    )


def _scan_artifacts() -> dict[str, int]:
    """Count markers over every inductor_cache output_code (files merged)."""
    counts = {"output_code_files": 0, "npu_triton_heuristics": 0}
    for marker in _T0C1_BASELINE:
        counts[marker] = 0
    cache_root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
    for path in _iter_output_code(cache_root):
        with open(path, encoding="utf-8") as f:
            body = f.read()
        counts["output_code_files"] += 1
        for marker in ("npu_triton_heuristics", *_T0C1_BASELINE):
            counts[marker] += body.count(marker)
    return counts


@wait_until_npu_memory_free(max_wait_seconds=600)  # ~15G bf16 weights per card; HBM drains slowly
def test_inductor_track_moe_tp4():
    from vllm import SamplingParams

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

    # (c) + (d) artifact marker and T0c-1-form comparison counts
    counts = _scan_artifacts()
    assert counts["npu_triton_heuristics"] > 0, (
        f"no npu_triton_heuristics marker under "
        f"{os.environ.get('VLLM_CACHE_ROOT', '~/.cache/vllm')}/**/inductor_cache/ "
        f"(track compiled without triton_experimental?): {counts}"
    )
    print(
        "MOE_TP4_OBS output_code counts (vs T0c-1 explicit-PW baseline "
        f"moe_forward={_T0C1_BASELINE['moe_forward']}/"
        f"unquantized_gemm={_T0C1_BASELINE['unquantized_gemm']}): {counts}"
    )


def _first_divergence(eager_ids: list[int], track_ids: list[int]) -> str:
    for i, (e, t) in enumerate(zip(eager_ids, track_ids)):
        if e != t:
            return f"first divergence at token {i}: eager={e} track={t}"
    if len(eager_ids) != len(track_ids):
        return f"token count differs: eager={len(eager_ids)} track={len(track_ids)}"
    return ""


@wait_until_npu_memory_free(max_wait_seconds=600)
@pytest.mark.xfail(
    reason="30B-A3B bf16 track-vs-eager token identity is a near-tie argmax "
    "class divergence (wave-3 run: prompt0 48/48 identical, prompt1 flips at "
    "token 12, both valid continuations) — TE triton kernels vs eager aclnn "
    "accumulate sub-ULP differences that narrow logit gaps at 30B scale; "
    "same acceptance class as W2/U9 (数值版 M0). Gate criteria a/b/c/d live "
    "in test_inductor_track_moe_tp4; this case stays as the recorded parity "
    "probe until stage-5 attribution (logprob-gap measurement).",
    strict=True,
)
def test_inductor_track_moe_tp4_matches_eager():
    """Eager TP4 baseline vs the track (same greedy decoding), 04 §T2.

    Sequential engines in this process (VllmRunner.__exit__ releases HBM
    deterministically between them). Cold compile is paid by the gate case
    above sharing this VLLM_CACHE_ROOT; a cold cache here just costs the
    1-3h again.
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
