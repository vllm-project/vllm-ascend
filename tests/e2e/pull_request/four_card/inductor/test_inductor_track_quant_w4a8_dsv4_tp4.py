"""Stage4 final-audit V2 (function check 02 §V2) + stage-5 input #5: W4A8
weights x inductor compile-backend track — DeepSeek-V4-Flash-w4a8-mtp (TP4).

The eager-mode load was proven at stage-4 M-H (R15: 38/38 shards, ~38G/card);
the on-track probe was queued as a stage-5 input (needs explicit
VLLM_USE_BREAKABLE_CUDAGRAPH=0 — nine architectures inject breakable by
default and silently disable the track, function check 02 §V4). This file is
that probe, formalized, and doubles as the w4a8 quant-family track gate.

Smoke scope (R15/D8 precedent — no eager parity leg: a second 151G TP4
engine in the same test doubles a >30min load for little signal; degenerate
near-tie behavior on debug-tier weights is already on record):
  * the track engine boots with compilation_config.backend="inductor" and
    VLLM_USE_BREAKABLE_CUDAGRAPH=0;
  * final cudagraph_mode is the -O2 default journey (FULL_AND_PIECEWISE);
  * greedy generation is non-empty and coherent-looking (recorded, printed);
  * the triton_experimental artifact marker is present.
"""

import glob
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_W4A8_MODEL = os.environ.get(
    "FC_W4A8_DSV4_MODEL", "/mnt/weight/DeepSeek-V4-Flash-w4a8-mtp"
)

pytestmark = [
    pytest.mark.e2e_model("deepseek-ai/DeepSeek-V4-Flash-w4a8-mtp"),
    pytest.mark.skipif(
        not os.path.isdir(_W4A8_MODEL),
        reason=f"W4A8 DSV4 weights not found at {_W4A8_MODEL} (set FC_W4A8_DSV4_MODEL)",
    ),
]

PROMPTS = [
    "The capital of France is",
    "给我一句话介绍长城。",
]
MAX_TOKENS = 32

_BASE = dict(
    model_name=_W4A8_MODEL,
    quantization="ascend",
    max_model_len=8192,
    max_num_seqs=8,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.85,
    # R15 calibration: TP4 ~38G/card on 64G — no explicit kv budget needed at
    # this concurrency (kv_cache_memory is not an EngineArgs kwarg here).
)

_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    compilation_config={"backend": "inductor"},
)

_ROOT = Path(__file__).resolve().parents[6]


@patch.dict(os.environ, {"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"})
@wait_until_npu_memory_free(max_wait_seconds=900)
def test_inductor_track_quant_w4a8_dsv4_tp4_smoke():
    import os as _os

    cache = _ROOT / "log" / "fc_v2" / "w4a8_dsv4_tp4"
    cache.mkdir(parents=True, exist_ok=True)
    _os.environ["VLLM_CACHE_ROOT"] = str(cache)

    from vllm import SamplingParams

    greedy = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    with VllmRunner(**_BASE, **_TRACK) as runner:
        final_cg = runner.model.llm_engine.vllm_config.compilation_config.cudagraph_mode
        outs = runner.model.generate(PROMPTS, greedy)

    for prompt, out in zip(PROMPTS, outs):
        text = out.outputs[0].text.strip()
        assert text, f"empty track generation for prompt {prompt!r}"
        print(f"[w4a8 dsv4 smoke] prompt={prompt!r} -> {text!r} (recorded)")

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
