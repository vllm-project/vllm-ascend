"""Stage4 e2e (04 §T2, MLA family): inductor compile-backend track on
DeepSeek-V2-Lite (bf16) with explicit PIECEWISE cudagraph capture.

Mirrors the T0-T0b-5R probe (stage design/stage4/_notes/t0_probe/RESULTS.md):
V2-Lite x track, compilation_config.backend=inductor,
cudagraph_mode=PIECEWISE, gpu_memory_utilization=0.85, max_model_len=4096 —
the probe-calibrated combo that generated token-identical output to eager
(the npu_mla_* AscendC fused kernels ride the mla_forward splitting op into
the compiled graph and stay capturable, MLA-A5).

Acceptance (stage design/stage4/04 §T2):
  (a) compiled-track generation is non-empty;
  (b) greedy generation is token-identical to an enforce_eager engine on the
      same prompts (both engines run sequentially in this process);
  (c) final cudagraph_mode is PIECEWISE on the engine config (the value after
      platform check_and_update_config, i.e. the "track active ... (final)"
      state);
  (d) track artifact marker: ``npu_triton_heuristics`` appears in a vllm-cache
      ``inductor_cache/`` output_code — spawn moves compilation counters into
      the worker process (04 §0-2), so this track asserts artifacts only.

Weights are located via ``STAGE4_MLA_MODEL`` (default: /mnt/weight mount);
tests skip when the weights are absent so CI without the asset stays green.
Run with a per-case cold ``VLLM_CACHE_ROOT`` (stage4 env discipline) so the
marker check cannot hit artifacts from unrelated runs.
"""

import glob
import os

import pytest
from vllm.config.compilation import CUDAGraphMode, CompilationConfig

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
    gpu_memory_utilization=0.85,
)

# Explicit PIECEWISE (the probe-verified shape): since the debt-2 refactor the
# track default follows the -O presets (O2 -> FULL_AND_PIECEWISE); this test
# pins the shape it was built to guard.
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


@wait_until_npu_memory_free(max_wait_seconds=600)  # two ~30G bf16 engines; HBM drains slowly
def test_inductor_track_mla_matches_eager():
    from vllm import SamplingParams

    greedy = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    # Eager baseline first; VllmRunner.__exit__ shuts the engine core down and
    # releases HBM deterministically before the track engine starts compiling.
    with VllmRunner(enforce_eager=True, **_BASE) as runner:
        eager_outs = runner.model.generate(PROMPTS, greedy)

    with VllmRunner(**_BASE, **_TRACK) as runner:
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

    # (d) artifact marker (04 §0-1): an output_code under the vllm-cache
    # inductor_cache/ carries the triton_experimental heuristics import.
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
