"""Stage4 final-audit V2 (function check 02 §V2): W8A8_MXFP8 (msmodelslim)
weights x inductor compile-backend track gate at TP2.

Qwen3.8-27B-w8a8-mxfp8 (35G, quant_model_description.json quant_type
W8A8_MXFP8) is the local mxfp8-family representative — the second quant
family (after fp8-block, one_card) this audit puts on the track. Qwen3.8 is a
new architecture: if the track fails on it, the failure itself is the finding
(owner attribution TE vs VA comes after reproduction).

Acceptance mirrors the W8A8/fp8 gates: greedy token-identical vs an
enforce_eager TP2 engine, final cudagraph_mode FULL_AND_PIECEWISE, and the
triton_experimental artifact marker.
"""

import glob
import os
from pathlib import Path

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_MXFP8_MODEL = os.environ.get("FC_MXFP8_MODEL", "/mnt/weight/Qwen3.8-27B-w8a8-mxfp8")

pytestmark = [
    pytest.mark.e2e_model("Qwen/Qwen3.8-27B-w8a8-mxfp8"),
    pytest.mark.skipif(
        not os.path.isdir(_MXFP8_MODEL),
        reason=f"mxfp8 weights not found at {_MXFP8_MODEL} (set FC_MXFP8_MODEL)",
    ),
]

PROMPTS = [
    "The capital of France is",
    "Hello, my name is Tom, I am",
]
MAX_TOKENS = 32

_BASE = dict(
    model_name=_MXFP8_MODEL,
    quantization="ascend",
    max_model_len=4096,
    max_num_seqs=8,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
)

_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    compilation_config={"backend": "inductor"},
)

_ROOT = Path(__file__).resolve().parents[6]


@wait_until_npu_memory_free(max_wait_seconds=600)
def test_inductor_track_quant_mxfp8_tp2_matches_eager():
    import os as _os

    cache = _ROOT / "log" / "fc_v2" / "mxfp8_tp2"
    cache.mkdir(parents=True, exist_ok=True)
    _os.environ["VLLM_CACHE_ROOT"] = str(cache)

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
        eager_ids = list(eager_out.outputs[0].token_ids)
        track_ids = list(track_out.outputs[0].token_ids)
        assert eager_ids == track_ids, (
            f"prompt {i} ({PROMPTS[i]!r}): eager={eager_ids} track={track_ids}"
        )

    assert getattr(final_cg, "name", str(final_cg)) == "FULL_AND_PIECEWISE", (
        f"final cudagraph_mode expected FULL_AND_PIECEWISE, got {final_cg}"
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
