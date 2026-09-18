"""Stage2 e2e: inductor compile-backend track under tensor_parallel_size=2.

Mirrors docs/vllm/env/scripts/smoke_qwen3_inductor_tp2.py (workspace repo,
stage design/stage2/04 §T4'): track-on TP=2 must produce token-identical
output to the eager TP=2 baseline under the same sampling seed.
"""

import pytest
from vllm import LLM, SamplingParams
from vllm.config.compilation import CUDAGraphMode, CompilationConfig

PROMPTS = [
    "The capital of France is",
    "给我一句话介绍长城。",
]

_BASE = dict(
    model="Qwen/Qwen3-0.6B",
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=16,
    tensor_parallel_size=2,
)

# Explicit PIECEWISE: since the debt-2 refactor the track default follows the
# -O presets (O2 -> FULL_AND_PIECEWISE); F&P x TP=2 is an unverified gap, so
# this stage2 baseline pins the shape it was built to guard (ledger 13, debt 2).
_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    compilation_config=CompilationConfig(backend="inductor", cudagraph_mode=CUDAGraphMode.PIECEWISE),
)


def _generate(kwargs):
    llm = LLM(**kwargs)
    outs = llm.generate(
        PROMPTS, SamplingParams(max_tokens=48, temperature=0.7, top_p=0.8, seed=10086)
    )
    return [o.outputs[0].text for o in outs]


@pytest.mark.parametrize("mode", ["inductor", "eager_ref"])
def test_inductor_track_tp2(mode):
    kwargs = dict(_BASE)
    if mode == "inductor":
        kwargs.update(_TRACK)
    else:
        kwargs["enforce_eager"] = True
    texts = _generate(kwargs)
    for text in texts:
        assert text.strip()


def test_inductor_track_tp2_matches_eager():
    eager_kwargs = dict(_BASE, enforce_eager=True)
    track_kwargs = dict(_BASE, **_TRACK)
    eager_texts = _generate(eager_kwargs)
    track_texts = _generate(track_kwargs)
    assert track_texts == eager_texts
