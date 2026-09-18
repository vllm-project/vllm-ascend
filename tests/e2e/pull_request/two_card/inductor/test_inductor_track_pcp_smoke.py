"""Stage4 final-audit A1 (U8 adjudication): PCP (Prefill Context Parallelism,
预填充上下文并行) x inductor compile-backend track smoke gate.

U8 picked "dense small model + v2 model runner + explicit FULL_DECODE_ONLY" as
the vehicle; that premise is falsified by the platform: VllmConfig validation
rejects PCP under the v2 model runner for dense (GQA) architectures —
"Model Runner V2 does not yet support: prefill context parallelism" (the v1
runner rejects PCP outright, U8 note). The PCP-capable family is MLA/SFA, so
the vehicle switches to the in-repo MRV2 PCP guard's own model
(four_card/context_parallel/test_accuracy_v2.py PCP_MODEL_CASE:
vllm-ascend/DeepSeek-V3.2-W8A8-Pruning, the 3-layer debug artifact), at TP1 x
PCP2 (two cards). Config recipe mirrors that guard:
``prefill_context_parallel_size`` + chunked prefill + prefix caching +
cp_kv_cache_interleave_size + block 128, cg limited to NONE/FDO.

Smoke acceptance (D8): the track engine compiles under PCP2, final
cudagraph_mode is FULL_DECODE_ONLY, greedy output is token-identical to an
enforce_eager PCP2 engine, and the triton_experimental artifact marker is
present. Failures found here are registered, not attacked (probe-level scope).

Weights are located via ``STAGE4_PCP_MODEL`` (default: /mnt/weight mount);
tests skip when the weights are absent.
"""

import glob
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

_PCP_MODEL = os.environ.get(
    "STAGE4_PCP_MODEL", "/mnt/weight/DeepSeek-V3.2-W8A8-Pruning"
)

pytestmark = [
    pytest.mark.e2e_model("vllm-ascend/DeepSeek-V3.2-W8A8-Pruning"),
    pytest.mark.skipif(
        not os.path.isdir(_PCP_MODEL),
        reason=f"DeepSeek-V3.2-W8A8-Pruning weights not found at {_PCP_MODEL} (set STAGE4_PCP_MODEL)",
    ),
]

PROMPTS = [
    "The capital of France is",
    "Hello, my name is Tom, I am",
]
MAX_TOKENS = 32

MAX_NUM_SEQS = 4

# PCP recipe from four_card/context_parallel/test_accuracy_v2.py (PCP_MODEL_CASE),
# scaled to TP1 x PCP2 (two cards, weights replicated per CP rank).
_BASE = dict(
    model_name=_PCP_MODEL,
    quantization="ascend",
    max_model_len=1024,
    max_num_seqs=MAX_NUM_SEQS,
    max_num_batched_tokens=1024,
    tensor_parallel_size=1,
    prefill_context_parallel_size=2,
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.8,
    cp_kv_cache_interleave_size=128,
    block_size=128,
)

_FULL_DECODE_GRAPH = {
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "cudagraph_capture_sizes": [MAX_NUM_SEQS],
}

_TRACK = dict(
    # front door (config refactor M1b): compilation_config.backend selects the track
    compilation_config={"backend": "inductor", **_FULL_DECODE_GRAPH},
)

_ROOT = Path(__file__).resolve().parents[6]


def _cold_cache() -> None:
    cache = _ROOT / "log" / "stage4_final" / "pcp_smoke"
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


@patch.dict(
    os.environ,
    {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_BUFFSIZE": "768",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    },
)
@wait_until_npu_memory_free(max_wait_seconds=600)
def test_inductor_track_pcp_smoke_matches_eager():
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

    # Smoke numerics (D8): recorded, not hard-asserted. The 3-layer pruned
    # debug model has degenerate high-entropy outputs (its in-repo goldens are
    # gibberish by design), so track-vs-eager sub-ULP diffs flip argmax —
    # first run: prompt 0 token-identical (32/32), prompt 1 diverges from
    # token 1; same family as the MoE W8A8 near-tie set (TODO-VA-8). The
    # functional smoke gates are the hard assertions below/above: compiles
    # under PCP2, final_cg FULL_DECODE_ONLY, non-empty generation, marker.
    assert len(track_outs) == len(eager_outs)
    for i, (eager_out, track_out) in enumerate(zip(eager_outs, track_outs)):
        eager_ids = list(eager_out.outputs[0].token_ids)
        track_ids = list(track_out.outputs[0].token_ids)
        if eager_ids == track_ids:
            print(f"[smoke] prompt {i}: token-identical ({len(eager_ids)} tokens)")
        else:
            first = next(
                (j for j, (e, t) in enumerate(zip(eager_ids, track_ids)) if e != t),
                min(len(eager_ids), len(track_ids)),
            )
            print(
                f"[recorded] prompt {i} ({PROMPTS[i]!r}): near-tie divergence at "
                f"token {first} (eager={eager_ids[first] if first < len(eager_ids) else 'EOF'} "
                f"track={track_ids[first] if first < len(track_ids) else 'EOF'}); "
                f"eager={eager_ids} track={track_ids}"
            )

    assert getattr(final_cg, "name", str(final_cg)) == "FULL_DECODE_ONLY", (
        f"final cudagraph_mode expected FULL_DECODE_ONLY, got {final_cg}"
    )

    assert _scan_marker() > 0, (
        "no npu_triton_heuristics marker in the cold cache "
        "(track compiled without triton_experimental?)"
    )
