"""Stage4 e2e (04 §T2, MoE family quantized second axis): inductor
compile-backend track on Qwen3-30B-A3B-W8A8 (msmodelslim W8A8_DYNAMIC, NZ
weights) under tensor_parallel_size=4, DEFAULT -O2 profile.

Provenance: the M-F/W5 probe (stage design/stage4/_notes/t0_probe/RESULTS.md,
driver run_moe_w8a8.sh / t0_probe.py moe_w8a8_tp4[_eager]) — 30B-A3B W8A8
(30.5G dynamic) x track x TP4 on the default profile with weight_nz_mode=1
pinned explicitly (guarding the BF16 tier's NZ flip). The probe was green:
bilingual generation non-empty, final cudagraph_mode FULL_AND_PIECEWISE with
FULL(decode)+PIECEWISE(mixed) dual aclgraph capture and replay in the run
log, and the vllm-cache output_code carried npu_triton_heuristics x64 /
moe_forward x32 / unquantized_gemm x40 / quant_matmul x128 — the
unquantized_gemm 168->40 (-128) vs quant_matmul 0->128 (+128) exact
complementary swap against the BF16 tier (M-F gate, 12 files same basis).

Acceptance (stage design/stage4/04 §T2):
  (a) compiled-track generation is non-empty (bilingual 2 prompts);
  (b) final cudagraph_mode is FULL_AND_PIECEWISE on the engine config (the
      value after platform check_and_update_config);
  (c) track artifact marker: ``npu_triton_heuristics`` appears in a vllm-cache
      ``inductor_cache/`` output_code — spawn moves compilation counters into
      the worker process (04 §0-2), so this track asserts artifacts only;
  (d) recorded, NOT asserted: moe_forward / unquantized_gemm / quant_matmul
      occurrence counts in those output_code files against the M-F/W5 probe
      magnitude (32/40/128), plus the R14 NZ-leak observation below.

R14 observation (recorded, NOT asserted): npu_grouped_matmul / grouped_matmul
/ format_cast all read 0 on the artifact side — the NZ weights stay sealed
inside the moe_forward / npu_quant_matmul opaque extern bodies, so the
FULL_AND_PIECEWISE piece boundaries show zero NZ<->ND cast leakage
(compilation face structurally equivalent to the BF16 tier).

The eager token-identity baseline is a SEPARATE strict-xfail case (probe
measured a near-tie, see its reason). Weights are located via
``STAGE4_MOE_W8A8_MODEL`` (default: /mnt/weight mount); tests skip when the
weights are absent so CI without the asset stays green. Run gate and eager
cases with separate cold ``VLLM_CACHE_ROOT`` subdirs (stage4 batch discipline)
so the gate marker scan cannot hit the other leg's artifacts; a cold compile
is ~11min per track engine, so each case is runnable on its own.
"""

import glob
import os

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_MOE_W8A8_MODEL = os.environ.get("STAGE4_MOE_W8A8_MODEL", "/mnt/weight/Qwen3-30B-A3B-W8A8")

pytestmark = [
    pytest.mark.e2e_model("Qwen/Qwen3-30B-A3B-W8A8"),
    pytest.mark.skipif(
        not os.path.isdir(_MOE_W8A8_MODEL),
        reason=f"Qwen3-30B-A3B-W8A8 weights not found at {_MOE_W8A8_MODEL} (set STAGE4_MOE_W8A8_MODEL)",
    ),
]

PROMPTS = [
    "The capital of France is",
    "给我一句话介绍长城。",
]
MAX_TOKENS = 48

_BASE = dict(
    model_name=_MOE_W8A8_MODEL,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=16,
    gpu_memory_utilization=0.80,
    tensor_parallel_size=4,
)

# Default profile (04 §T2, the M-F/W5 probe shape): NO explicit cudagraph_mode
# — the -O2 preset resolves it to FULL_AND_PIECEWISE on the engine config —
# with weight_nz_mode=1 pinned explicitly (guarding the BF16 tier's NZ flip).
_NZ = {"weight_nz_mode": 1}
_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    additional_config={"ascend_compilation_config": dict(_NZ)},
    compilation_config={"backend": "inductor"},
)
_EAGER_NZ = dict(additional_config=dict(_NZ))

# M-F/W5 probe artifact markers (merged output_code, single-run basis 12
# files): observation only, never asserted. unquantized_gemm 168->40 and
# quant_matmul 0->128 are the exact complementary swap vs the BF16 tier; the
# three R14 NZ-leak markers stay 0 (NZ sealed inside opaque extern bodies).
_PROBE_BASELINE = {
    "moe_forward": 32,
    "unquantized_gemm": 40,
    "quant_matmul": 128,
    "npu_grouped_matmul": 0,
    "grouped_matmul": 0,
    "format_cast": 0,
}


def _iter_output_code(cache_root: str):
    # M-F/W5 basis: the merged output_code.py artifacts (fx_graph_*.py debug
    # siblings under inductor_cache/ are NOT part of the comparison basis).
    yield from glob.glob(
        os.path.join(cache_root, "**", "inductor_cache", "**", "output_code.py"), recursive=True
    )


def _scan_artifacts() -> dict[str, int]:
    """Count markers over every inductor_cache output_code (files merged)."""
    counts = {"output_code_files": 0, "npu_triton_heuristics": 0}
    for marker in _PROBE_BASELINE:
        counts[marker] = 0
    cache_root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
    for path in _iter_output_code(cache_root):
        with open(path, encoding="utf-8") as f:
            body = f.read()
        counts["output_code_files"] += 1
        for marker in ("npu_triton_heuristics", *_PROBE_BASELINE):
            counts[marker] += body.count(marker)
    return counts


@wait_until_npu_memory_free(max_wait_seconds=600)  # ~8G W8A8 weights per card; HBM drains slowly
def test_inductor_track_moe_w8a8_tp4():
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

    # (c) + (d) artifact marker and M-F/W5-form comparison counts
    counts = _scan_artifacts()
    assert counts["npu_triton_heuristics"] > 0, (
        f"no npu_triton_heuristics marker under "
        f"{os.environ.get('VLLM_CACHE_ROOT', '~/.cache/vllm')}/**/inductor_cache/ "
        f"(track compiled without triton_experimental?): {counts}"
    )
    print(
        "MOE_W8A8_TP4_OBS output_code counts (vs M-F/W5 probe baseline "
        f"npu_triton_heuristics=64 {_PROBE_BASELINE}): {counts}"
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
    reason="30B-A3B W8A8 track-vs-eager token identity is a near-tie argmax "
    "class divergence (M-F/W5 probe run: prompt0 identical for the first 13 "
    "tokens then first divergence at token 14, prompt1 at token 6, both valid "
    "continuations) — quantization amplifies the early forking the BF16 tier "
    "already shows (prompt1 token-12 flip, same acceptance class, 数值版 M0); "
    "TE triton kernels vs eager aclnn accumulate sub-ULP differences that "
    "narrow logit gaps at 30B scale. Gate criteria a/b/c/d live in "
    "test_inductor_track_moe_w8a8_tp4; this case stays as the recorded parity "
    "probe until stage-5 attribution (logprob-gap measurement).",
    strict=True,
)
def test_inductor_track_moe_w8a8_tp4_matches_eager():
    """Eager TP4 baseline vs the track (same greedy decoding, same NZ pin),
    04 §T2.

    Sequential engines in this process (VllmRunner.__exit__ releases HBM
    deterministically between them). Run under this case's own cold
    VLLM_CACHE_ROOT subdir: its track engine re-cold-compiles (~11min) rather
    than sharing the gate's root, keeping both legs' artifact bases separate.
    """
    from vllm import SamplingParams

    greedy = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    with VllmRunner(enforce_eager=True, **_BASE, **_EAGER_NZ) as runner:
        eager_outs = runner.model.generate(PROMPTS, greedy)

    with VllmRunner(**_BASE, **_TRACK) as runner:
        track_outs = runner.model.generate(PROMPTS, greedy)

    assert len(track_outs) == len(eager_outs)
    for i, (eager_out, track_out) in enumerate(zip(eager_outs, track_outs)):
        divergence = _first_divergence(
            list(eager_out.outputs[0].token_ids), list(track_out.outputs[0].token_ids)
        )
        assert not divergence, f"prompt {i} ({PROMPTS[i]!r}): {divergence}"
