"""Stage4 final-audit V2 (function check 02 §V2): fp8 block-quantized weights x
inductor compile-backend track gate.

Qwen3-8B-FP8 is a native HF fp8 checkpoint with weight_block_size [128, 128]
— the block format VA serves (fp8_config._verify_block_quantization passes);
per-tensor/per-channel is V8 (function check 02 §V8), not covered here. W8A8
was the only quantization family with track e2e evidence before this file.

Acceptance: greedy token-identical vs an enforce_eager engine on the same
weights, final cudagraph_mode FULL_AND_PIECEWISE (rf1 default journey), and
the triton_experimental artifact marker.
"""

import glob
import os

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_FP8_MODEL = os.environ.get("FC_FP8_MODEL", "/mnt/weight/Qwen3-8B-FP8")

pytestmark = [
    pytest.mark.e2e_model("Qwen/Qwen3-8B-FP8"),
    pytest.mark.skipif(
        not os.path.isdir(_FP8_MODEL),
        reason=f"Qwen3-8B-FP8 weights not found at {_FP8_MODEL} (set FC_FP8_MODEL)",
    ),
]

PROMPTS = [
    "The capital of France is",
    "Hello, my name is Tom, I am",
]
MAX_TOKENS = 32

_BASE = dict(
    model_name=_FP8_MODEL,
    max_model_len=2048,
    max_num_seqs=8,
    gpu_memory_utilization=0.85,
)

_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    compilation_config={"backend": "inductor"},
)


@wait_until_npu_memory_free(max_wait_seconds=600)
def test_inductor_track_quant_fp8_block_matches_eager():
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
