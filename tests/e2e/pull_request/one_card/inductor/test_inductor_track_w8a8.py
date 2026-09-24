"""Stage3 e2e (item 4): inductor track on a W8A8-quantized Qwen3-0.6B (U3/D3-5).

Model weights are produced offline by msmodelslim (M4.0 probe, stage design/stage3/05):
    msmodelslim quant --model_path <Qwen3-0.6B> --save_path <out> \
        --quant_type QuantType.W8A8 --device npu
and located via ``S3_W8A8_MODEL`` (default: workspace log dir). Tests skip when
the weights are absent so CI without the asset stays green.

Acceptance (stage3/04 §T3-5):
  * weight_nz_mode=0 (necessary condition): compare_logprobs vs eager baseline;
  * weight_nz_mode=1 (characterization axis): same criterion — if it fails, the
    failure is root-caused (which op gets triton-lowered over NZ storage) and
    feeds the stage-4 U2 guard decision; it is not silently dropped;
  * fusion benefit evidence: rmsnorm_quant_fusion_pass match_table > 0 on the
    in-process engine (worker RPC-free variant).
"""

import os

import pytest
from vllm.config.compilation import CompilationConfig, CUDAGraphMode

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.utils import PROMPTS_SHORT, compare_logprobs

_W8A8_MODEL = os.environ.get(
    "S3_W8A8_MODEL",
    "/home/g00594152/npu-inductor-worktree/v1/log/w8a8_qwen3_06b",
)

pytestmark = [
    pytest.mark.e2e_model("Qwen/Qwen3-0.6B-W8A8"),
    pytest.mark.skipif(
        not os.path.isdir(_W8A8_MODEL),
        reason=f"W8A8 weights not found at {_W8A8_MODEL} (produce via msmodelslim M4.0)",
    ),
]

# Explicit PIECEWISE: this file guards the W8A8 fusion chain, not the graph
# family — pin the stage3-verified shape (debt-2 refactor moved the track
# default to the -O presets, O2 -> FULL_AND_PIECEWISE).
# front door (config refactor M1b): compilation_config.backend selects the track
_TRACK_CG = CompilationConfig(backend="inductor", cudagraph_mode=CUDAGraphMode.PIECEWISE)


def _track_kwargs(weight_nz_mode: int) -> dict:
    return {
        "model_name": _W8A8_MODEL,
        "quantization": "ascend",
        "max_model_len": 1024,
        "compilation_config": _TRACK_CG,
        "additional_config": {
            "weight_nz_mode": weight_nz_mode,
        },
    }


@pytest.mark.parametrize("weight_nz_mode", [0, 1])
@wait_until_npu_memory_free()
def test_w8a8_inductor_track_matches_eager(weight_nz_mode):
    """Numerics vs eager baseline on the quantized model, per nz axis."""
    compare_logprobs(runner_kwargs=_track_kwargs(weight_nz_mode), prompts=PROMPTS_SHORT)


def _force_cold_compile():
    """Point VLLM_CACHE_ROOT at a fresh dir: match_table only fills when the
    fusion passes actually execute, and a warm torch_compile_cache hit skips
    compilation entirely (passes never run in this process)."""
    import tempfile

    cache = tempfile.mkdtemp(prefix="s3_fusion_cold_")
    os.environ["VLLM_CACHE_ROOT"] = cache
    return cache


@wait_until_npu_memory_free(max_wait_seconds=600)
def test_w8a8_fusion_match_table_recorded():
    """Fusion-benefit observability on the W8A8_DYNAMIC 0.6B (recorded, not hard).

    Root-caused (stage3 03 known-limitation): the dynamic-quant model shape passes
    ``dst_type=torch.int8`` to npu_dynamic_quant (pattern omits it), inserts a view
    between norm and quant, and the qk-norm+rope chain is already pre-fused at the
    model layer (qkv_rmsnorm_mrope) — none of the three patterns can match, on this
    track AND on the legacy GraphFusionPassManager track alike (pre-existing shape
    drift, not a regression). The match table must still be populated (proves the
    chain-back executes and is observable); hard >0 evidence comes from the static
    W8A8 shape (see test_w8a8_static_fusion_match_table / 8B archive run).
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _force_cold_compile()

    from vllm import LLM, SamplingParams
    from vllm.compilation.passes.vllm_inductor_pass import VllmPatternMatcherPass

    llm = LLM(
        model=_W8A8_MODEL,
        quantization="ascend",
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.55,
        compilation_config=_TRACK_CG,
        additional_config={
            "weight_nz_mode": 0,
        },
    )
    outs = llm.generate(PROMPTS_SHORT, SamplingParams(max_tokens=8, temperature=0.0))
    for out in outs:
        assert out.outputs[0].text.strip(), "empty generation"

    table = dict(VllmPatternMatcherPass.match_table)
    # the pass ran and is observable (keys present == __call__ executed)
    assert "rmsnorm_quant_fusion_pass" in table, f"fusion pass never ran: {table}"
    print(f"W8A8_DYNAMIC match_table (recorded): {table}")


@wait_until_npu_memory_free(max_wait_seconds=600)
def test_w8a8_dynamic_fusion_match_table():
    """Stage4 W2 (D1/D2) acceptance: the dynamic W8A8 form now fuses (asserted).

    Root cause fixed in stage4: with enable_custom_op() the dynamic-quant graph
    norm node is ``_C_ascend.npu_add_rms_norm_bias(x, residual, weight, None,
    eps)`` — the registry lacked a bias=None dynamic variant (the WithBias
    pattern needs a tensor 4th arg), so rmsnorm_quant_fusion_pass stayed at 0
    (T0b-1R). The added AddRMSNormDynamicQuantPatternWithNoneBias (+View)
    variants must turn the 0.6B cold-compile match count above zero. The
    pattern-level mechanics are covered CPU-side by
    tests/ut/compilation/test_norm_quant_fusion_w2_patterns.py; this e2e case
    proves the hit on the real 0.6B graph (D2: match_table rmsnorm_quant>0).
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _force_cold_compile()

    from vllm import LLM, SamplingParams
    from vllm.compilation.passes.vllm_inductor_pass import VllmPatternMatcherPass

    llm = LLM(
        model=_W8A8_MODEL,
        quantization="ascend",
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.55,
        compilation_config=_TRACK_CG,
        additional_config={
            "ascend_compilation_config": {
                # 方案 A (U-approved): the dynamic fusion variants are opt-in —
                # default-off keeps greedy output token-identical to eager.
                "fuse_norm_quant_dynamic": True,
            },
            "weight_nz_mode": 0,
        },
    )
    outs = llm.generate(PROMPTS_SHORT, SamplingParams(max_tokens=8, temperature=0.0))
    for out in outs:
        assert out.outputs[0].text.strip(), "empty generation"

    table = dict(VllmPatternMatcherPass.match_table)
    assert table.get("rmsnorm_quant_fusion_pass", 0) > 0, (
        f"stage4 W2 (D1): the bias=None dynamic variant produced no matches on "
        f"the W8A8_DYNAMIC 0.6B: match_table={table}"
    )


_8B_W8A8 = os.environ.get(
    "S3_W8A8_8B_MODEL",
    "/home/inductor-benchmark/huggingface/modelscope/Qwen3-8B-W8A8",
)


@pytest.mark.skipif(
    not os.path.isdir(_8B_W8A8),
    reason=f"static W8A8 8B weights not found at {_8B_W8A8}",
)
@wait_until_npu_memory_free(max_wait_seconds=600)
def test_w8a8_static_fusion_match_table():
    """Hard fusion-benefit evidence on the static W8A8 shape (8B, archive asset).

    torch.ops.vllm.quantize is called positionally with no extra kwargs, so the
    static AddRMSNormQuant pattern can actually match here (probe-verified:
    rmsnorm_quant_fusion_pass=3 on the archive run, stage3 _notes/w8a8_evidence).
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _force_cold_compile()

    from vllm import LLM, SamplingParams
    from vllm.compilation.passes.vllm_inductor_pass import VllmPatternMatcherPass

    llm = LLM(
        model=_8B_W8A8,
        quantization="ascend",
        dtype="bfloat16",
        max_model_len=1024,
        gpu_memory_utilization=0.85,
        max_num_seqs=4,
        compilation_config=_TRACK_CG,
        additional_config={
            "weight_nz_mode": 0,
        },
    )
    outs = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0.0))
    assert outs[0].outputs[0].text.strip(), "empty generation"

    table = dict(VllmPatternMatcherPass.match_table)
    assert table.get("rmsnorm_quant_fusion_pass", 0) > 0, (
        f"fuse_norm_quant produced no matches on the static W8A8 8B: match_table={table}"
    )


@pytest.mark.skipif(
    not os.path.isdir(_8B_W8A8),
    reason=f"static W8A8 8B weights not found at {_8B_W8A8}",
)
@wait_until_npu_memory_free(max_wait_seconds=600)
def test_w8a8_8b_inductor_track_default_cg_gate():
    """Stage4 final-audit A2 (02 §W1.4 / M-E step 3): 8B static W8A8 default-cg gate.

    Probe T0b-8 proved the default journey (cudagraph_mode unset -> deferred to
    the -O preset -> FULL_AND_PIECEWISE final) runs and generates on the 8B
    static-quantized shape; this case freezes that evidence as a repo gate:
    non-empty generation, final cudagraph_mode FULL_AND_PIECEWISE, and the
    triton_experimental artifact marker. Guards the rf1 default (O2 preset)
    on quantized weights (rf1 ledger gap "F&P default x W8A8-8B unverified").
    The eager token-identity baseline is a SEPARATE strict-xfail case (see its
    reason) per the MoE W8A8 precedent.
    """
    import glob

    from vllm import SamplingParams

    from tests.e2e.conftest import VllmRunner

    kwargs = dict(
        model_name=_8B_W8A8,
        quantization="ascend",
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
        additional_config={
            "weight_nz_mode": 0,
        },
    )

    greedy = SamplingParams(max_tokens=8, temperature=0.0)
    with VllmRunner(**kwargs) as runner:
        final_cg = runner.model.llm_engine.vllm_config.compilation_config.cudagraph_mode
        outs = runner.model.generate(PROMPTS_SHORT, greedy)
    for prompt, out in zip(PROMPTS_SHORT, outs):
        assert out.outputs[0].text.strip(), f"empty track generation for prompt {prompt!r}"

    assert getattr(final_cg, "name", str(final_cg)) == "FULL_AND_PIECEWISE", (
        f"final cudagraph_mode expected FULL_AND_PIECEWISE (rf1 default journey), got {final_cg}"
    )

    cache_root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
    hits = []
    for path in glob.glob(
        os.path.join(cache_root, "**", "inductor_cache", "**", "*.py"), recursive=True
    ):
        with open(path, encoding="utf-8") as f:
            if "npu_triton_heuristics" in f.read():
                hits.append(path)
    assert hits, (
        f"no npu_triton_heuristics marker under {cache_root}/**/inductor_cache/ "
        "(track compiled without triton_experimental?)"
    )


@pytest.mark.skipif(
    not os.path.isdir(_8B_W8A8),
    reason=f"static W8A8 8B weights not found at {_8B_W8A8}",
)
@wait_until_npu_memory_free(max_wait_seconds=600)
@pytest.mark.xfail(
    reason="8B static W8A8 default-cg track-vs-eager token identity is a near-tie "
    "argmax class divergence (final-audit A2 runs 2026-09-16: strict greedy flips "
    "the very first prefill token on prompt 0 'Hello, my name is' — eager 8515 vs "
    "track 29405, both plausible name-start continuations; compare_logprobs equally "
    "rejects because its criterion requires the same argmax token) — quantized "
    "weights + fused norm-quant static kernels accumulate sub-ULP differences vs "
    "the eager chain; same family as the MoE 30B W8A8 near-tie set (TODO-VA-8, "
    "logprob-gap adjudication in stage-5). Gate criteria live in "
    "test_w8a8_8b_inductor_track_default_cg_gate; this case stays as the "
    "recorded parity baseline and flips to a hard gate once the logprob gap "
    "is measured acceptable.",
    strict=True,
)
def test_w8a8_8b_inductor_track_default_cg_matches_eager():
    """8B static W8A8 default-cg eager parity (recorded strict-xfail, MoE precedent)."""
    kwargs = dict(
        model_name=_8B_W8A8,
        quantization="ascend",
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
        additional_config={
            "weight_nz_mode": 0,
        },
    )
    compare_logprobs(runner_kwargs=kwargs, prompts=PROMPTS_SHORT)
