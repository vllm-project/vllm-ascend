# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""End-to-end checks for UNO shared-model speculative decoding on Ascend.

UNO drafts with the target model plus a rank-128 gated LoRA adapter, so the
only cheap end-to-end correctness instrument is exact-token parity against a
plain engine: greedy decoding must be unchanged by speculation, because
rejection sampling reproduces the target distribution exactly.

That instrument is only valid where the plain engine reproduces itself. Buffered
NPU kernels are not bit-deterministic in every mode, so each case builds the
reference twice in the same process, on the same config and prompts, and only
holds UNO to the agreement the reference achieves against itself. Where the
reference disagrees with itself, the test reports the gap instead of failing on
Uno for it.
"""

from __future__ import annotations

import pytest
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_OFFLINE
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

# The release-day adapter this implementation was validated against. Pinned by
# revision so a later republish cannot silently change what CI measures.
UNO_MODEL = "Qwen/Qwen3-8B"
UNO_ADAPTER_REPO = "s-sahoo/uno-qwen3-8B"
UNO_ADAPTER_REVISION = "8819e09ac901e7290d8d89d62c98b9f756c602fe"

MAX_MODEL_LEN = 2048
MAX_TOKENS = 32
# One batch holds every request's K draft rows, so the scheduler budget must be
# at least max_num_seqs * K (validated at startup).
MAX_NUM_SEQS = 4

PROMPTS = [
    "The capital of France is",
    "Hello, my name is",
    "The future of artificial intelligence is",
    "A list of three prime numbers:",
]


@pytest.fixture(scope="module")
def uno_adapter_path() -> str:
    snapshot = snapshot_download(
        repo_id=UNO_ADAPTER_REPO,
        revision=UNO_ADAPTER_REVISION,
        allow_patterns=["adapter/*"],
        local_files_only=HF_HUB_OFFLINE,
    )
    return f"{snapshot}/adapter"


def _uno_runner_kwargs(adapter_path: str, k: int) -> dict:
    """Common engine config for both the reference and the UNO engine.

    Every argument except ``speculative_config``/``additional_config`` is
    identical on both arms: the reference must differ only by the method under
    test, or a divergence cannot be attributed to UNO.
    """
    return {
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_SEQS * k,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_lora": True,
        "max_lora_rank": 128,
        "max_loras": 1,
        "max_cpu_loras": 1,
    }


def _generate(runner: VllmRunner) -> list[list[int]]:
    outputs = runner.model.generate(
        PROMPTS,
        SamplingParams(temperature=0, ignore_eos=True, max_tokens=MAX_TOKENS),
        use_tqdm=False,
    )
    return [list(output.outputs[0].token_ids) for output in outputs]


def _agreement(actual: list[list[int]], expected: list[list[int]]) -> int:
    """Number of prompts whose token ids match exactly."""
    return sum(a == b for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("k", [1, 8])
def test_uno_greedy_matches_base_model(uno_adapter_path: str, k: int):
    """Greedy UNO output must equal the plain engine's, up to its own noise.

    Two passes on the reference engine measure how far a plain engine drifts
    from itself on this card and mode. UNO is then held to exactly that: an
    exact match when the reference is self-consistent, otherwise no worse than
    the reference's own spread.
    """
    common = _uno_runner_kwargs(uno_adapter_path, k)

    with VllmRunner(UNO_MODEL, **common) as reference:
        ref_first = _generate(reference)
        ref_second = _generate(reference)

    with VllmRunner(
        UNO_MODEL,
        speculative_config={"method": "uno", "num_speculative_tokens": k},
        additional_config={"uno_lora_path": uno_adapter_path},
        **common,
    ) as uno:
        uno_tokens = _generate(uno)

    reference_agreement = _agreement(ref_second, ref_first)
    uno_agreement = _agreement(uno_tokens, ref_first)

    print(
        f"K={k}: reference self-agreement {reference_agreement}/{len(PROMPTS)}; UNO {uno_agreement}/{len(PROMPTS)}",
        flush=True,
    )
    if reference_agreement < len(PROMPTS):
        for index, (expected, actual) in enumerate(zip(ref_first, uno_tokens, strict=True)):
            if expected != actual:
                first_diff = next(
                    (pos for pos, (a, b) in enumerate(zip(expected, actual)) if a != b),
                    min(len(expected), len(actual)),
                )
                print(f"  prompt {index}: first difference at token {first_diff}", flush=True)

    assert uno_agreement >= reference_agreement, (
        f"UNO agreed with the plain engine on {uno_agreement}/{len(PROMPTS)} prompts, "
        f"but the plain engine only reproduced itself {reference_agreement}/{len(PROMPTS)}. "
        "UNO is worse than the reference's own run-to-run spread."
    )


def test_uno_reports_draft_acceptance(uno_adapter_path: str):
    """The engine must actually draft and accept tokens, not silently fall back.

    A UNO run that accepted nothing (or drafted nothing) would still pass the
    greedy parity check above, because parity only requires the final tokens to
    be right. This separates "works" from "speculation is live".
    """
    k = 4
    common = _uno_runner_kwargs(uno_adapter_path, k)

    with VllmRunner(
        UNO_MODEL,
        speculative_config={"method": "uno", "num_speculative_tokens": k},
        additional_config={"uno_lora_path": uno_adapter_path},
        disable_log_stats=False,
        **common,
    ) as uno:
        _generate(uno)
        metrics = uno.model.get_metrics()

    counters = {metric.name: metric.value for metric in metrics if metric.name.startswith("vllm:spec_decode_")}
    num_drafts = counters.get("vllm:spec_decode_num_drafts", 0.0)
    num_accepted = counters.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    print(f"drafts={num_drafts}, accepted={num_accepted}", flush=True)

    assert num_drafts > 0, "UNO produced no drafts: the speculator never ran"
    assert num_accepted > 0, "UNO accepted no tokens: check the adapter and the noise range"
    # Acceptance length includes the always-accepted bonus token.
    acceptance_length = 1 + num_accepted / num_drafts
    print(f"acceptance length: {acceptance_length:.3f}", flush=True)
    assert acceptance_length <= 1 + k, f"acceptance length {acceptance_length} exceeds 1 + K = {1 + k}"


@pytest.mark.parametrize("k", [1, 8])
def test_uno_rejects_request_lora(k: int, uno_adapter_path: str):
    """UNO owns the engine-wide LoRA slot, so request-level adapters are refused.

    The refusal happens per request, not at startup, so the engine must build
    successfully first. Keep the nesting explicit: a combined ``with`` would
    also accept a matching error raised while constructing the engine, which is
    a different failure.
    """
    from vllm.lora.request import LoRARequest

    common = _uno_runner_kwargs(uno_adapter_path, k)
    with VllmRunner(  # noqa: SIM117
        UNO_MODEL,
        speculative_config={"method": "uno", "num_speculative_tokens": k},
        additional_config={"uno_lora_path": uno_adapter_path},
        **common,
    ) as uno:
        with pytest.raises(ValueError, match="UNO speculative decoding reserves"):
            uno.model.generate(
                PROMPTS[:1],
                SamplingParams(temperature=0, max_tokens=8),
                lora_request=LoRARequest("user", 1, uno_adapter_path),
                use_tqdm=False,
            )
