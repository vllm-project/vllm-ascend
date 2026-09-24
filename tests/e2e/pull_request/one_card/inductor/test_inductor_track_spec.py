"""Stage4 final-audit V1 (function check 02 §V1): speculative decoding x
inductor compile-backend track — spec-on eager vs spec-on track parity gates.

The platform layer wires 11 speculative methods with no code gate against the
track, but only ngram_gpu had a recorded track adaptation (probe level); the
in-repo spec_decode acceptance tests never set compilation_config.backend="inductor".
This file closes the verification gap for four locally-weighted representative
methods (function check/03 P0 #2):

* eagle3 — draft-model family (Qwen3-8B + RedHatAI speculator, the test_eagle
  recipe: explicit FULL_DECODE_ONLY + capture sizes [12])
* dflash — draft-model family (Qwen3-8B + DFlash-b16, same recipe shape)
* ngram / suffix — draftless methods on Qwen3-0.6B (track default graph tier)

Acceptance per method: greedy generation token-identical between an
enforce_eager engine and the track engine with the SAME speculative config
(isolates the track), final cudagraph_mode as expected, and the
triton_experimental artifact marker. Methods without local weights or recipes
(eagle non-3 gated Llama pair, draft_parallel PARD, dspark GLM pair, medusa,
MTP four-card) are recorded in function check/02 §V1, not tested here.
"""

import glob
import os

import pytest
from vllm.config.compilation import CompilationConfig, CUDAGraphMode

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_MAIN_8B = os.environ.get("FC_SPEC_8B", "/mnt/weight/Qwen3-8B")
_SPEC_EAGLE3 = os.environ.get("FC_SPEC_EAGLE3", "/mnt/weight/Qwen3-8B-speculator.eagle3")
_SPEC_DFLASH = os.environ.get("FC_SPEC_DFLASH", "/mnt/weight/Qwen3-8B-DFlash-b16")
_MAIN_06B = os.environ.get("FC_SPEC_06B", "/mnt/weight/Qwen3-0.6B")

MAX_TOKENS = 32
PROMPTS = [
    "The capital of France is",
    "Hello, my name is Tom, I am",
]


def _spec_case(method: str):
    """(runner base kwargs, speculative_config, explicit CompilationConfig|None)."""
    if method == "eagle3":
        return (
            dict(model_name=_MAIN_8B, max_model_len=2048, max_num_seqs=8,
                 gpu_memory_utilization=0.7),
            {"method": "eagle3", "num_speculative_tokens": 3, "model": _SPEC_EAGLE3},
            CompilationConfig(
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                cudagraph_capture_sizes=[12],
            ),
        )
    if method == "dflash":
        return (
            dict(model_name=_MAIN_8B, max_model_len=2048, max_num_seqs=8,
                 gpu_memory_utilization=0.7),
            {"method": "dflash", "num_speculative_tokens": 3, "model": _SPEC_DFLASH},
            CompilationConfig(
                cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                cudagraph_capture_sizes=[12],
            ),
        )
    if method == "ngram":
        return (
            dict(model_name=_MAIN_06B, max_model_len=1024, max_num_seqs=8,
                 gpu_memory_utilization=0.6, cudagraph_capture_sizes=[1, 2, 4, 8]),
            {"method": "ngram", "prompt_lookup_max": 5, "prompt_lookup_min": 3,
             "num_speculative_tokens": 3},
            None,  # track default tier (rf1: O2 -> FULL_AND_PIECEWISE)
        )
    if method == "suffix":
        return (
            dict(model_name=_MAIN_06B, max_model_len=1024, max_num_seqs=8,
                 gpu_memory_utilization=0.6, max_cudagraph_capture_size=16),
            {"method": "suffix", "suffix_decoding_max_spec_factor": 2.0,
             "suffix_decoding_max_cached_requests": 1000,
             "num_speculative_tokens": 10},
            None,
        )
    raise ValueError(method)


_METHODS = {
    "eagle3": _SPEC_EAGLE3,
    "dflash": _SPEC_DFLASH,
    "ngram": None,
    "suffix": None,
}

pytestmark = [
    pytest.mark.e2e_model("Qwen/Qwen3-8B"),
    pytest.mark.parametrize("method", list(_METHODS)),
]


def _needs_weights(method: str) -> bool:
    """skipif condition: every local prerequisite the case needs must exist."""
    if not os.path.isdir(_MAIN_8B if method in ("eagle3", "dflash") else _MAIN_06B):
        return True
    spec = _METHODS[method]
    if spec is not None and not os.path.isdir(spec):
        return True
    if method == "suffix":
        # Upstream hard dependency (vllm/config/speculative.py:1085) — absent
        # from the stack by default, and installing it drags the torch wheel
        # set (dependency chain), so skip cleanly without the package.
        try:
            import arctic_inference  # noqa: F401
        except ImportError:
            return True
    return False


def _track_kwargs(method: str) -> dict:
    base, spec_cfg, cg = _spec_case(method)
    kwargs = dict(base)
    kwargs["speculative_config"] = spec_cfg
    if cg is not None:
        kwargs["compilation_config"] = cg
    return kwargs


def _expected_final_cg(method: str) -> str:
    return "FULL_DECODE_ONLY" if method in ("eagle3", "dflash") else "FULL_AND_PIECEWISE"


@wait_until_npu_memory_free(max_wait_seconds=600)
def test_inductor_track_spec_matches_eager(method):
    if _needs_weights(method):
        pytest.skip(f"local weights missing for spec method {method!r}")

    from vllm import SamplingParams

    from vllm.config.compilation import CompilationConfig

    greedy = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
    kwargs = _track_kwargs(method)
    # front door (config refactor M1b): compilation_config.backend selects
    # the track; merged into the per-method cudagraph_mode profile.
    cg = kwargs.pop("compilation_config", None) or CompilationConfig()
    cg.backend = "inductor"
    track = dict(compilation_config=cg)

    with VllmRunner(enforce_eager=True, **kwargs) as runner:
        eager_outs = runner.model.generate(PROMPTS, greedy)

    with VllmRunner(**kwargs, **track) as runner:
        final_cg = runner.model.llm_engine.vllm_config.compilation_config.cudagraph_mode
        track_outs = runner.model.generate(PROMPTS, greedy)

    for prompt, out in zip(PROMPTS, track_outs):
        assert out.outputs[0].text.strip(), f"empty track generation for prompt {prompt!r}"

    assert len(track_outs) == len(eager_outs)
    for i, (eager_out, track_out) in enumerate(zip(eager_outs, track_outs)):
        eager_ids = list(eager_out.outputs[0].token_ids)
        track_ids = list(track_out.outputs[0].token_ids)
        assert eager_ids == track_ids, (
            f"{method}: prompt {i} ({PROMPTS[i]!r}): "
            f"eager={eager_ids} track={track_ids}"
        )

    assert getattr(final_cg, "name", str(final_cg)) == _expected_final_cg(method), (
        f"{method}: final cudagraph_mode expected {_expected_final_cg(method)}, got {final_cg}"
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
        f"{method}: no npu_triton_heuristics marker under {cache_root}/**/inductor_cache/ "
        "(track compiled without triton_experimental?)"
    )
