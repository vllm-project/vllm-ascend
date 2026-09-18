"""Stage4 final-audit A3 (02 §W1.2 "advanced tier = W8A8 variant AND TP2"):
inductor compile-backend track on DeepSeek-V2-Lite (bf16) at tensor-parallel 2.

The one-card gate (one_card/inductor/test_inductor_track_mla.py) pins the
probe-verified explicit-PIECEWISE shape; this case extends the same shape to
TP2 — the advanced tier promised in the stage4 design whose TP2 leg was never
run (final-audit A3, stage design/stage4/_notes/03_功能覆盖终审报告.md).

Acceptance mirrors the one-card gate:
  (a) compiled-track generation is non-empty;
  (b) greedy generation is token-identical to an enforce_eager TP2 engine on
      the same prompts (engines run sequentially in this process);
  (c) final cudagraph_mode is PIECEWISE on the engine config;
  (d) track artifact marker: ``npu_triton_heuristics`` in a vllm-cache
      ``inductor_cache/`` output_code (spawn-safe artifact criterion, 04 §0-2).

Weights are located via ``STAGE4_MLA_MODEL`` (default: /mnt/weight mount);
tests skip when the weights are absent. Run with TP2 cards via
``ASCEND_RT_VISIBLE_DEVICES`` (ascending) and a per-case cold
``VLLM_CACHE_ROOT``.
"""

import glob
import os

import pytest
from vllm.config.compilation import CompilationConfig, CUDAGraphMode

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_MLA_MODEL = os.environ.get("STAGE4_MLA_MODEL", "/mnt/weight/DeepSeek-V2-Lite")

pytestmark = [
    pytest.mark.e2e_model("deepseek-ai/DeepSeek-V2-Lite"),
    pytest.mark.skipif(
        not os.path.isdir(_MLA_MODEL),
        reason=f"DeepSeek-V2-Lite weights not found at {_MLA_MODEL} (set STAGE4_MLA_MODEL)",
    ),
]

PROMPTS = [
    "The capital of France is",
    "给我一句话介绍长城。",
]
MAX_TOKENS = 48

_BASE = dict(
    model_name=_MLA_MODEL,
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=8,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
)

# Same explicit PIECEWISE as the one-card gate: this test extends that
# probe-verified shape to TP2 (minimal delta from the proven config).
_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    compilation_config=CompilationConfig(backend="inductor", cudagraph_mode=CUDAGraphMode.PIECEWISE),
)


def _first_divergence(eager_ids: list[int], track_ids: list[int]) -> str:
    for i, (e, t) in enumerate(zip(eager_ids, track_ids)):
        if e != t:
            return f"first divergence at token {i}: eager={e} track={t}"
    if len(eager_ids) != len(track_ids):
        return f"token count differs: eager={len(eager_ids)} track={len(track_ids)}"
    return ""


@wait_until_npu_memory_free(max_wait_seconds=600)
def test_inductor_track_mla_tp2_matches_eager():
    from vllm import SamplingParams

    greedy = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    with VllmRunner(enforce_eager=True, **_BASE) as runner:
        eager_outs = runner.model.generate(PROMPTS, greedy)

    with VllmRunner(**_BASE, **_TRACK) as runner:
        final_cg = runner.model.llm_engine.vllm_config.compilation_config.cudagraph_mode
        track_outs = runner.model.generate(PROMPTS, greedy)

    for prompt, out in zip(PROMPTS, track_outs):
        assert out.outputs[0].text.strip(), f"empty track generation for prompt {prompt!r}"

    assert len(track_outs) == len(eager_outs)
    for i, (eager_out, track_out) in enumerate(zip(eager_outs, track_outs)):
        divergence = _first_divergence(
            list(eager_out.outputs[0].token_ids), list(track_out.outputs[0].token_ids)
        )
        assert not divergence, f"prompt {i} ({PROMPTS[i]!r}): {divergence}"

    assert getattr(final_cg, "name", str(final_cg)) == "PIECEWISE", (
        f"final cudagraph_mode expected PIECEWISE, got {final_cg}"
    )

    cache_root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
    hits = []
    for path in glob.glob(os.path.join(cache_root, "**", "inductor_cache", "**", "*.py"), recursive=True):
        with open(path, encoding="utf-8") as f:
            if "npu_triton_heuristics" in f.read():
                hits.append(path)
    assert hits, (
        f"no npu_triton_heuristics marker under {cache_root}/**/inductor_cache/ "
        "(track compiled without triton_experimental?)"
    )
