"""Stage4 final-audit A4 (02 §W1.3 second tier): inductor compile-backend track
on the quantized Qwen3-32B W8A8 (msmodelslim "tuned" artifact) at TP2.

The bf16 32B default/F&P tiers are gated by test_inductor_track_dense32b_tp2.py;
this case closes the design-promised second tier (final-audit A4, stage
design/stage4/_notes/03_功能覆盖终审报告.md): same TP2 shape on the static
W8A8 weights, default cg profile (no explicit cudagraph_mode — the -O2 preset
resolves to FULL_AND_PIECEWISE), greedy token-identical to an enforce_eager
engine, plus the triton_experimental artifact marker.

Weights: /mnt/weight/Qwen3-32B_w8a8_tuned (parseability verified at design
time, V-dense-A2). Tests skip when the weights are absent.
"""

import glob
import os
from pathlib import Path

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_DENSE32B_W8A8 = os.environ.get(
    "STAGE4_DENSE32B_W8A8_MODEL", "/mnt/weight/Qwen3-32B_w8a8_tuned"
)

pytestmark = [
    pytest.mark.e2e_model("Qwen/Qwen3-32B-W8A8"),
    pytest.mark.skipif(
        not os.path.isdir(_DENSE32B_W8A8),
        reason=f"Qwen3-32B W8A8 weights not found at {_DENSE32B_W8A8} (set STAGE4_DENSE32B_W8A8_MODEL)",
    ),
]

PROMPTS = [
    "The capital of France is",
    "给我一句话介绍长城。",
]
MAX_TOKENS = 48

# Quantized 32B ≈ half the bf16 footprint per card; same max_model_len family
# as the bf16 gate (8192) leaves ample KV headroom at gmu 0.85.
_BASE = dict(
    model_name=_DENSE32B_W8A8,
    quantization="ascend",
    dtype="bfloat16",
    max_model_len=8192,
    max_num_seqs=16,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
)

# Default profile: NO explicit cudagraph_mode — the -O2 preset resolves it to
# FULL_AND_PIECEWISE on the engine config (asserted below).
_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    compilation_config={"backend": "inductor"},
)

# workspace root: vllm-ascend sits directly beneath it (…/v1)
_ROOT = Path(__file__).resolve().parents[6]


def _cold_cache() -> None:
    cache = _ROOT / "log" / "stage4_final" / "dense32b_w8a8"
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
            if "npu_triton_heuristics" in f.read():
                hits += 1
    return hits


@wait_until_npu_memory_free(max_wait_seconds=600)
def test_inductor_track_dense32b_w8a8_tp2_default_matches_eager():
    from vllm import SamplingParams

    _cold_cache()
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
        eager_ids = list(eager_out.outputs[0].token_ids)
        track_ids = list(track_out.outputs[0].token_ids)
        assert eager_ids == track_ids, (
            f"prompt {i} ({PROMPTS[i]!r}): eager={eager_ids} track={track_ids}"
        )

    assert getattr(final_cg, "name", str(final_cg)) == "FULL_AND_PIECEWISE", (
        f"final cudagraph_mode expected FULL_AND_PIECEWISE (rf1 default journey), got {final_cg}"
    )

    assert _scan_marker() > 0, (
        "no npu_triton_heuristics marker in the cold cache "
        "(track compiled without triton_experimental?)"
    )
