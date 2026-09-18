"""Stage4 e2e (04 §T2, MLA family quantized second axis): inductor
compile-backend track on DeepSeek-V2-Lite-W8A8 (W8A8_DYNAMIC) with explicit
PIECEWISE cudagraph capture.

Provenance: the V2-W8A8 probe (stage design/stage4/_notes/t0_probe/RESULTS.md,
driver t_v2w8a8.py) — V2-Lite W8A8 (17G, 64 experts x 26 layers fully
W8A8_DYNAMIC) x track with the M-D combo: backend=inductor,
cudagraph_mode=PIECEWISE explicit, gpu_memory_utilization=0.85,
max_model_len=4096, max_num_seqs=8. All four probe criteria were green:
ascend quantization auto-detected from quant_model_description.json, greedy
2x48 tokens token-identical to eager (the MoE W8A8 C op family rides the
torch.ops.vllm.moe_forward_shared opaque and runs without crashing; dense
npu_quant_matmul / npu_dynamic_quant / npu_swiglu stay FX-visible),
final_cg=PIECEWISE, and the vllm-cache output_code carried
npu_triton_heuristics x15 / moe_forward x12 / npu_dynamic_quant x12.

Acceptance (stage design/stage4/04 §T2):
  case 1 matches_eager (spawn-mode engines, mirrors test_inductor_track_mla):
    (a) compiled-track generation is non-empty;
    (b) greedy generation is token-identical to an enforce_eager engine on
        the same prompts (probe: 2 pairs, 48/48 identical);
    (c) final cudagraph_mode is PIECEWISE on the engine config (the value
        after platform check_and_update_config);
    (d) track artifact marker: ``npu_triton_heuristics`` appears in a
        vllm-cache ``inductor_cache/`` output_code — spawn moves compilation
        counters into the worker process (04 §0-2), so this case asserts
        artifacts only. The W8A8-form marker counts (moe_forward /
        npu_dynamic_quant / grouped_matmul) are recorded against the probe
        baseline, NOT asserted (grouped_matmul stays 0: the routed experts
        live inside the opaque extern body, T0c-1 basis).
  case 2 match_table_recorded (in-process engine, mirrors
        test_w8a8_fusion_match_table_recorded):
    (e) recorded, NOT asserted: VllmPatternMatcherPass.match_table on a
        cold-compiled in-process track engine. Probe read all three passes
        at 0 (rmsnorm_quant / qknorm_rope / muls_add): the W8A8_DYNAMIC norm
        node is the WithBias form (_C_ascend.npu_add_rms_norm_bias with a
        bias=None 4th arg) whose dynamic fusion variants are opt-in
        (fuse_norm_quant_dynamic, default-off keeps greedy token-identical
        to eager) — the stage4 W2/D1 finding.

Weights are located via ``STAGE4_MLA_W8A8_MODEL`` (default: /mnt/weight
mount); tests skip when the weights are absent so CI without the asset stays
green. Run with a per-case cold ``VLLM_CACHE_ROOT`` (stage4 env discipline)
so the marker check cannot hit artifacts from unrelated runs. Case 2 must
stay defined AFTER case 1 (pytest definition order): its in-process
EngineCore latches the vllm envs cache, which would drag case 1's second
0.85-memory engine into the ~30G in-process teardown residual
(test_inductor_track_mrv2 measurement) and OOM it.
"""

import glob
import os

import pytest
from vllm.config.compilation import CompilationConfig, CUDAGraphMode

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_MLA_W8A8_MODEL = os.environ.get("STAGE4_MLA_W8A8_MODEL", "/mnt/weight/DeepSeek-V2-Lite-W8A8")

pytestmark = [
    pytest.mark.e2e_model("deepseek-ai/DeepSeek-V2-Lite-W8A8"),
    pytest.mark.skipif(
        not os.path.isdir(_MLA_W8A8_MODEL),
        reason=f"DeepSeek-V2-Lite-W8A8 weights not found at {_MLA_W8A8_MODEL} (set STAGE4_MLA_W8A8_MODEL)",
    ),
]

PROMPTS = [
    "The capital of France is",
    "给我一句话介绍长城。",
]
MAX_TOKENS = 48

_BASE = dict(
    model_name=_MLA_W8A8_MODEL,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=8,
    gpu_memory_utilization=0.85,
)

# Explicit PIECEWISE (the probe-verified M-D shape): since the debt-2
# refactor the track default follows the -O presets (O2 ->
# FULL_AND_PIECEWISE); this test pins the shape it was built to guard.
_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    compilation_config=CompilationConfig(backend="inductor", cudagraph_mode=CUDAGraphMode.PIECEWISE),
)

# V2-W8A8 probe artifact markers (merged output_code): observation basis
# only, never asserted (grouped_matmul=0 is expected — opaque extern body).
_PROBE_BASELINE = {"moe_forward": 12, "npu_dynamic_quant": 12, "grouped_matmul": 0}


def _first_divergence(eager_ids: list[int], track_ids: list[int]) -> str:
    for i, (e, t) in enumerate(zip(eager_ids, track_ids)):
        if e != t:
            return f"first divergence at token {i}: eager={e} track={t}"
    if len(eager_ids) != len(track_ids):
        return f"token count differs: eager={len(eager_ids)} track={len(track_ids)}"
    return ""


@wait_until_npu_memory_free(max_wait_seconds=600)  # two ~56G HBM-peak engines; HBM drains slowly
def test_inductor_track_mla_w8a8_matches_eager():
    from vllm import SamplingParams

    greedy = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    # Eager baseline first; VllmRunner.__exit__ shuts the engine core down and
    # releases HBM deterministically before the track engine starts compiling.
    with VllmRunner(enforce_eager=True, **_BASE) as runner:
        eager_outs = runner.model.generate(PROMPTS, greedy)

    with VllmRunner(**_BASE, **_TRACK) as runner:
        quant = str(runner.model.llm_engine.vllm_config.model_config.quantization)
        final_cg = runner.model.llm_engine.vllm_config.compilation_config.cudagraph_mode
        track_outs = runner.model.generate(PROMPTS, greedy)

    # (a) compiled track generated non-empty text
    for prompt, out in zip(PROMPTS, track_outs):
        assert out.outputs[0].text.strip(), f"empty track generation for prompt {prompt!r}"

    # (b) greedy token-for-token identical to the eager baseline
    assert len(track_outs) == len(eager_outs)
    for i, (eager_out, track_out) in enumerate(zip(eager_outs, track_outs)):
        divergence = _first_divergence(
            list(eager_out.outputs[0].token_ids), list(track_out.outputs[0].token_ids)
        )
        assert not divergence, f"prompt {i} ({PROMPTS[i]!r}): {divergence}"

    # (c) explicit PIECEWISE survives the platform config updates (final value)
    assert getattr(final_cg, "name", str(final_cg)) == "PIECEWISE", (
        f"final cudagraph_mode expected PIECEWISE, got {final_cg}"
    )

    # (d) artifact marker + W8A8-form marker counts (recorded, not asserted)
    counts = {"npu_triton_heuristics": 0, **{marker: 0 for marker in _PROBE_BASELINE}}
    cache_root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
    for path in glob.glob(
        os.path.join(cache_root, "**", "inductor_cache", "**", "output_code.py"), recursive=True
    ):
        with open(path, encoding="utf-8") as f:
            body = f.read()
        for marker in counts:
            counts[marker] += body.count(marker)
    assert counts["npu_triton_heuristics"] > 0, (
        f"no npu_triton_heuristics marker under {cache_root}/**/inductor_cache/ "
        f"(track compiled without triton_experimental?): {counts}"
    )
    print(
        f"MLA_W8A8_OBS quantization={quant} (auto-detected); output_code counts "
        f"(vs V2-W8A8 probe baseline {_PROBE_BASELINE} / npu_triton_heuristics=15): {counts}"
    )


def _force_cold_compile():
    """Point VLLM_CACHE_ROOT at a fresh dir: match_table only fills when the
    fusion passes actually execute, and a warm torch_compile_cache hit skips
    compilation entirely (passes never run in this process)."""
    import tempfile

    cache = tempfile.mkdtemp(prefix="s4_mla_w8a8_cold_")
    os.environ["VLLM_CACHE_ROOT"] = cache
    return cache


@wait_until_npu_memory_free(max_wait_seconds=600)
def test_inductor_track_mla_w8a8_match_table_recorded():
    """(e) match_table / counters on the in-process track engine (record-only).

    In-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) keeps the pattern-matcher
    match_table in this process (R12: spawn-mode tables land in the worker).
    The forced-cold cache root is mandatory — case 1's warm cache would skip
    compilation and leave the table empty. Values are recorded per the probe
    fallback convention, not asserted.
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _force_cold_compile()

    from vllm import SamplingParams
    from vllm.compilation.counter import compilation_counter
    from vllm.compilation.passes.vllm_inductor_pass import VllmPatternMatcherPass

    with VllmRunner(**_BASE, **_TRACK) as runner:
        track_outs = runner.model.generate(
            PROMPTS, SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
        )
        table = dict(VllmPatternMatcherPass.match_table)
        counters = (
            compilation_counter.num_inductor_compiles,
            compilation_counter.num_cudagraph_captured,
        )

    for prompt, out in zip(PROMPTS, track_outs):
        assert out.outputs[0].text.strip(), f"empty track generation for prompt {prompt!r}"

    # (e) recorded, NOT asserted: probe read all three passes at 0 — the
    # W8A8_DYNAMIC norm is the WithBias form (bias=None 4th arg); the dynamic
    # fusion variants are opt-in (fuse_norm_quant_dynamic), per stage4 W2/D1.
    print(f"MLA_W8A8_MATCH_TABLE (recorded; probe: three passes all 0): {table}")
    print(
        f"MLA_W8A8_COUNTERS (recorded; probe: 28 compiles / 112 graphs captured): "
        f"{counters[0]} compiles / {counters[1]} graphs captured"
    )
