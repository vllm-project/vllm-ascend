# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Qwen3.6-35B-A3B regression on Model Runner V2 against recorded goldens.

The goldens are recorded with Model Runner V2 on a main-based image with the
Qwen3.6 compatibility patch applied (fp32 dequant scale + fp32 mamba state,
same-package PR). Model Runner V1 no longer exists on main, so a main-native
recording is the only reference this suite can reproduce; the Model Runner V1
parity evidence is a separate acceptance record taken on vLLM-Ascend v0.30.0
(11 cross-runner pairs, token-identical, max same-token logprob delta 1.55e-06).

Trying to assert a v0.30.0-recorded golden on main does not work: the two stacks
agree for 19 of the 24 tokens and then flip a near-tie (logprob delta 0.1297,
inside the 0.1378 decode tolerance). Recording on main keeps the case a
regression test instead of a cross-release comparison.

Goldens live in ``qwen3_6_mrv2_goldens.json`` next to this file. Each entry keys
on ``<model>_tp<tp>_<graph_mode>`` and stores the prompt plus the greedy token
ids. An entry that is missing makes the case skip explicitly, so a new
combination can be added by recording it rather than by deleting a test.
``scripts/record_main_goldens.py`` in the tracking issue records one entry with
exactly this file's runner configuration.

Tolerance follows the repository rule used by
``tests/e2e/pull_request/utils.py::compare_logprobs``: the first generated token
uses the prefill tolerance and the remaining tokens use the decode tolerance.

Known issue https://github.com/vllm-project/vllm-ascend/issues/21742 (batched
speculative decoding is not self-consistent) is captured by an xfail case that
records the observation instead of hiding it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free

MODEL = os.getenv("QWEN36_MOE_MODEL", "Eco-Tech/Qwen3.6-35B-A3B-w8a8")
MODEL_REVISION = os.getenv("QWEN36_MOE_REVISION", "1a118d717bcbd59480f4a110fe22181d21711b4d")

PREFILL_ATOL = 0.0689
DECODE_ATOL = 0.1378
MAX_TOKENS = 24
SPEC_TOKENS = 1
PROMPT_MAIN = "用三句话解释什么是张量并行。"
PROMPT_OTHER = "The capital of France is"
# Must match the prompt used to record the long-prompt golden (~2.3k tokens).
PROMPT_LONG = "请阅读下面的材料并回答问题。" + "张量并行把单个矩阵切分到多张卡上，通过集合通信完成拼接。" * 120

MODEL_DENSE = os.getenv("QWEN36_DENSE_MODEL", "Eco-Tech/Qwen3.6-27B-w8a8")
MODEL_DENSE_REVISION = os.getenv("QWEN36_DENSE_REVISION", "430a017dfd20366728d3b32100ea2e9f40da595a")

GOLDENS = Path(__file__).parents[2] / "qwen3_6_mrv2_goldens.json"

_COMMON_ENVS = {
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "HCCL_BUFFSIZE": "1024",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "OMP_NUM_THREADS": "10",
    "TASK_QUEUE_ENABLE": "1",
}

# The deterministic HCCL/ATB flags change the numerics slightly: a run with them
# diverges from a run without them at token 13 for this model. The goldens are
# recorded WITHOUT them (that is the configuration the acceptance numbers were
# measured in), so they are only applied to the batched stability case, where
# run-to-run reproducibility matters more than matching the golden.
_DETERMINISTIC_ENVS = {
    "LCCL_DETERMINISTIC": "1",
    "HCCL_DETERMINISTIC": "true",
    "ATB_MATMUL_SHUFFLE_K_ENABLE": "0",
    "CLOSE_MATMUL_K_SHIFT": "1",
}


# "eager" means no graph at all: the enum has no EAGER member, so it maps to NONE
# and enforce_eager=True below.
_GRAPH_MODES = {
    "eager": "NONE",
    "full_decode_only": "FULL_DECODE_ONLY",
    "full_and_piecewise": "FULL_AND_PIECEWISE",
}


def _runner_kwargs(tp: int, graph_mode: str) -> dict:
    kwargs = {
        "max_model_len": 8192,
        "dtype": "float16",
        "tensor_parallel_size": tp,
        "quantization": "ascend",
        "enable_expert_parallel": True,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.9,
        "enforce_eager": graph_mode == "eager",
        "compilation_config": {"cudagraph_mode": _GRAPH_MODES[graph_mode]},
        "speculative_config": {"method": "qwen3_5_mtp", "num_speculative_tokens": SPEC_TOKENS},
        "revision": MODEL_REVISION,
    }
    return kwargs


def _sampling_params(greedy: bool = True) -> SamplingParams:
    return SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0, top_k=0, logprobs=20, seed=1234)


def _collect(prompts: list[str], tp: int, graph_mode: str, batched: bool) -> list[dict]:
    with patch.dict(os.environ, _COMMON_ENVS):
        with VllmRunner(MODEL, **_runner_kwargs(tp, graph_mode)) as runner:
            if batched:
                outputs = runner.model.generate(prompts, _sampling_params())
            else:
                outputs = []
                for prompt in prompts:
                    outputs.extend(runner.model.generate([prompt], _sampling_params()))
    return [
        {
            "text": output.outputs[0].text,
            "token_ids": list(output.outputs[0].token_ids),
            "logprobs": [
                {str(token_id): value.logprob for token_id, value in step.items()}
                for step in (output.outputs[0].logprobs or [])
            ],
        }
        for output in outputs
    ]


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _deltas_vs_golden(golden: dict, observed: dict) -> dict:
    """Per-step same-token logprob deltas against a golden, split by stage.

    The prefill tolerance only applies to the first generated token, so the two
    maxima are tracked separately. Recording only out-of-tolerance steps (as an
    earlier revision did) makes the assertion vacuous, because the reported
    maximum can then never exceed the tolerance.
    """
    prefill_delta = 0.0
    decode_delta = 0.0
    violations = []
    for index in range(min(len(golden["token_ids"]), len(observed["token_ids"]))):
        token_id = str(golden["token_ids"][index])
        left = golden["logprobs"][index].get(token_id) if index < len(golden["logprobs"]) else None
        right = observed["logprobs"][index].get(token_id) if index < len(observed["logprobs"]) else None
        if left is None or right is None:
            continue
        delta = abs(left - right)
        if index == 0:
            prefill_delta = max(prefill_delta, delta)
            if delta > PREFILL_ATOL:
                violations.append({"step": index, "kind": "prefill", "delta": delta, "atol": PREFILL_ATOL})
        else:
            decode_delta = max(decode_delta, delta)
            if delta > DECODE_ATOL:
                violations.append({"step": index, "kind": "decode", "delta": delta, "atol": DECODE_ATOL})
    return {
        "prefill_delta": prefill_delta,
        "decode_delta": decode_delta,
        "violations": violations,
        "first_divergence": _first_divergence(golden["token_ids"], observed["token_ids"]),
        "token_ids_equal": golden["token_ids"] == observed["token_ids"],
    }


def _load_golden(key: str) -> dict:
    if not GOLDENS.exists():
        pytest.skip(f"{GOLDENS.name} is missing; record goldens before enabling this case")
    entries = json.loads(GOLDENS.read_text(encoding="utf-8")).get("entries", {})
    if key not in entries:
        pytest.skip(f"no golden recorded for {key}; record it instead of removing the case")
    return entries[key]


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,logprobs",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@pytest.mark.parametrize("graph_mode", ["eager", "full_decode_only", "full_and_piecewise"])
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_6_moe_batch1_matches_golden(graph_mode: str) -> None:
    """One request at a time: token ids must match the golden and logprobs must stay in tolerance."""
    key = f"moe_tp1_{graph_mode}"
    golden = _load_golden(key)
    observed = _collect([golden["prompt"]], tp=1, graph_mode=graph_mode, batched=False)[0]

    stats = _deltas_vs_golden(golden, observed)
    assert stats["token_ids_equal"], (
        f"{key}: token mismatch, first divergence at {stats['first_divergence']}\n"
        f"golden={golden['token_ids']}\nobserved={observed['token_ids']}"
    )
    assert not stats["violations"], (
        f"{key}: logprob outside tolerance: prefill={stats['prefill_delta']:.4f}/{PREFILL_ATOL}, "
        f"decode={stats['decode_delta']:.4f}/{DECODE_ATOL}, violations={stats['violations']}"
    )



@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,logprobs,long_sequence",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_6_moe_batch1_long_prompt_matches_golden() -> None:
    """~2.3k-token prompt: prefill length and mamba state handling must not change the answer."""
    key = "moe_tp1_eager_long"
    golden = _load_golden(key)
    # Use the recorded prompt verbatim: reconstructing it from the same pieces is
    # not guaranteed to match byte for byte, and a prompt drift shows up as a token
    # mismatch at index 0.
    observed = _collect([golden["prompt"]], tp=1, graph_mode="eager", batched=False)[0]

    stats = _deltas_vs_golden(golden, observed)
    assert stats["token_ids_equal"], (
        f"{key}: token mismatch, first divergence at {stats['first_divergence']}\n"
        f"golden={golden['token_ids']}\nobserved={observed['token_ids']}"
    )
    assert not stats["violations"], (
        f"{key}: logprob outside tolerance: prefill={stats['prefill_delta']:.4f}/{PREFILL_ATOL}, "
        f"decode={stats['decode_delta']:.4f}/{DECODE_ATOL}, violations={stats['violations']}"
    )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,logprobs",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_6_moe_non_greedy_is_reproducible_within_runner() -> None:
    """Seeded sampling must be reproducible across two calls of the same engine.

    There is no repository-wide criterion for cross-runner agreement under
    sampling (the reference runner no longer exists in this tree), so this case
    pins the part that can be asserted. Cross-runner sampled agreement stays
    tracked in #21742 rather than being silently skipped.
    """
    params = SamplingParams(
        max_tokens=MAX_TOKENS, temperature=0.8, top_p=0.95, top_k=20, seed=1234, logprobs=20
    )
    with patch.dict(os.environ, _COMMON_ENVS):
        with VllmRunner(MODEL, **_runner_kwargs(tp=1, graph_mode="eager")) as runner:
            first = runner.model.generate([PROMPT_MAIN], params)[0].outputs[0]
            second = runner.model.generate([PROMPT_MAIN], params)[0].outputs[0]

    assert first.token_ids, "seeded sampling returned no tokens"
    assert first.logprobs is not None and len(first.logprobs) == len(first.token_ids)
    assert list(first.token_ids) == list(second.token_ids), (
        "seeded sampling is not reproducible within one runner: "
        f"{list(first.token_ids)} vs {list(second.token_ids)}"
    )




@pytest.mark.e2e_model(MODEL_DENSE)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="mtp,logprobs",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_6_dense_batch1_matches_golden() -> None:
    """Dense representative must reproduce the recorded reference within tolerance."""
    key = "dense_tp1_eager"
    golden = _load_golden(key)
    kwargs = _runner_kwargs(tp=1, graph_mode="eager")
    # Dense model: expert parallelism is rejected ("Number of experts ... must be
    # greater than 0"), so it must be off for this representative.
    kwargs["enable_expert_parallel"] = False
    kwargs["revision"] = MODEL_DENSE_REVISION
    with patch.dict(os.environ, _COMMON_ENVS):
        with VllmRunner(MODEL_DENSE, **kwargs) as runner:
            output = runner.model.generate([golden["prompt"]], _sampling_params())[0].outputs[0]
    observed = {
        "token_ids": list(output.token_ids),
        "logprobs": [{str(token_id): value.logprob for token_id, value in step.items()}
                     for step in (output.logprobs or [])],
    }
    stats = _deltas_vs_golden(golden, observed)
    assert stats["token_ids_equal"], (
        f"{key}: token mismatch, first divergence at {stats['first_divergence']}\n"
        f"golden={golden['token_ids']}\nobserved={observed['token_ids']}"
    )
    assert not stats["violations"], (
        f"{key}: logprob outside tolerance: prefill={stats['prefill_delta']:.4f}/{PREFILL_ATOL}, "
        f"decode={stats['decode_delta']:.4f}/{DECODE_ATOL}, violations={stats['violations']}"
    )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp,logprobs,chunked_prefill",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@pytest.mark.xfail(
    strict=False,
    reason="batched speculative decoding is not self-consistent yet (#21742); "
    "chunked prefill is mandatory for mamba cache mode 'align'",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_6_moe_batch2_identical_prompts_self_consistent() -> None:
    """Two identical requests in one batch must produce the same token ids.

    Kept as a non-strict xfail so the known defect is visible in CI without
    turning the signal into either a hard failure (unstable) or a silent pass.
    """
    with patch.dict(os.environ, _DETERMINISTIC_ENVS):
        observed = _collect([PROMPT_MAIN, PROMPT_MAIN], tp=1, graph_mode="eager", batched=True)
    assert len(observed) == 2
    first_divergence = _first_divergence(observed[0]["token_ids"], observed[1]["token_ids"])
    assert first_divergence is None, (
        f"two identical requests diverged at token {first_divergence}: "
        f"{observed[0]['token_ids']} vs {observed[1]['token_ids']}"
    )


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="mtp",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_6_moe_prompt_exceeds_max_model_len() -> None:
    """A prompt longer than max_model_len must fail at request time with a clear message."""
    kwargs = _runner_kwargs(tp=1, graph_mode="eager")
    kwargs["max_model_len"] = 1024
    runner = VllmRunner(MODEL, **kwargs)
    with patch.dict(os.environ, _COMMON_ENVS):
        with runner as model_runner:
            with pytest.raises(Exception, match="maximum context length"):
                model_runner.model.generate(["token " * 4000], _sampling_params())


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="",
    parallel="TP",
    deploy="pd_mix",
    quantization="W8A8",
    hardware="A2",
    graph_mode="eager",
)
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_qwen3_6_moe_multimodal_identifies_red_image() -> None:
    """Multimodal path: a solid red image must be described as red by both probes.

    Asserting the colour (instead of only that the output is non-empty) is what
    distinguishes "the image was encoded" from "the text prompt was answered".
    """
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image? Answer with one word."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    # The model emits a reasoning preamble before the answer, so 8 tokens is not
    # enough to reach the colour; 64 tokens lands inside the first sentence pair.
    params = SamplingParams(max_tokens=64, temperature=0.0)
    with patch.dict(os.environ, _COMMON_ENVS):
        with VllmRunner(MODEL, **_runner_kwargs(tp=1, graph_mode="eager")) as model_runner:
            output = model_runner.model.chat(messages, params)[0]
    text = output.outputs[0].text.strip().lower()
    assert text, "multimodal request returned an empty response"
    assert "red" in text, f"image content was not recognised, response={text!r}"
