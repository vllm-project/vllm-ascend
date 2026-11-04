#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""MRV1 TP2xPP2 + MTP accuracy (four NPUs).

Four-card counterpart of ``tests/e2e/pull_request/two_card/test_mrv1_pp_mtp.py``:
the greedy PP1-vs-PP2 comparison of vllm-ascend#17149's original configuration
(``tensor_parallel_size=2``, ``pipeline_parallel_size`` 1 vs 2).

Every case asserts ``drafts > 0`` as well as token equality. Under PP + MTP the
draft handoff can be missing entirely, which silently disables speculation and
makes PP2 agree with PP1 for the wrong reason, so token equality alone proves
nothing here.

The V2-runner cases are the attribution control: #17149 reports the V2 runner as
correct on this topology, so if V1 diverges while V2 does not, the regression is
in the V1 pipeline-parallel path rather than in the TP2xPP2 topology itself.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.model_utils import check_outputs_equal

MODEL = os.environ.get("QWEN3_5_4B_MODEL_PATH", "Qwen/Qwen3.5-4B")
# Synthetic PEFT adapter (see notes at the bottom of this docstring); empty means
# the LoRA case skips rather than downloading a real adapter.
LORA_ADAPTER = os.environ.get("MRV1_PP_LORA_ADAPTER", "")

TP_SIZE = 2
PROMPTS = ["Hello, my name is", "The future of AI is"]
LONG_PROMPT = (
    "The history of computing is a story of abstraction layers: from vacuum tubes to transistors, "
    "from assembly to compilers, and from single machines to clusters that span continents. "
) * 30
MAX_TOKENS = 16

# Engine launches dominate this file (~90 s each), and three of the cases share
# configurations, so completed runs are memoised. Only plain data is cached: the
# VllmRunner context manager has already released the device by then.
_RUN_CACHE: dict[tuple[Any, ...], tuple[list[tuple[list[int], str]], tuple[int, int]]] = {}


def _npu_count() -> int:
    try:
        import torch
        import torch_npu  # noqa: F401

        return int(torch.npu.device_count())
    except Exception:  # pragma: no cover - environment dependent
        return 0


requires_four_npus = pytest.mark.skipif(
    _npu_count() < 4,
    reason=f"TP2xPP2 needs 4 NPU devices in one HCCL domain (found {_npu_count()})",
)


def _coverage(**over: str) -> Any:
    kwargs = dict(
        arch="dense",
        feature="spec_decode",
        parallel="TP,PP",
        deploy="pd_mix",
        hardware="A3",
        quantization="BF16",
        graph_mode="eager",
    )
    kwargs.update(over)
    return pytest.mark.e2e_coverage(**kwargs)


def _mtp_config(graph: bool) -> dict:
    return {"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": not graph}


def _spec_metrics(runner: VllmRunner) -> tuple[int, int]:
    num_drafts = 0
    num_accepted = 0
    for metric in runner.model.get_metrics():
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            num_accepted += sum(metric.values)
    return num_drafts, num_accepted


def _run(
    pp_size: int,
    *,
    v2_runner: bool = False,
    prompts: list[str] | None = None,
    max_model_len: int = 1024,
    max_num_batched_tokens: int = 2048,
    lora: bool = False,
    graph: bool = False,
) -> tuple[list[tuple[list[int], str]], tuple[int, int]]:
    key = (pp_size, v2_runner, tuple(prompts or PROMPTS), max_model_len, max_num_batched_tokens, lora, graph)
    if key in _RUN_CACHE:
        return _RUN_CACHE[key]

    kwargs: dict[str, Any] = {
        "max_model_len": max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": max_num_batched_tokens,
        "tensor_parallel_size": TP_SIZE,
        "pipeline_parallel_size": pp_size,
        "distributed_executor_backend": "mp",
        "gpu_memory_utilization": 0.8,
        "enforce_eager": not graph,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "speculative_config": _mtp_config(graph),
        "hf_overrides": {"language_model_only": True},
    }
    if graph:
        kwargs["compilation_config"] = {"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"}
    if lora:
        kwargs.update(enable_lora=True, max_loras=1, max_lora_rank=8)

    with (
        patch.dict(
            os.environ,
            {
                "VLLM_USE_V2_MODEL_RUNNER": "1" if v2_runner else "0",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            },
        ),
        VllmRunner(MODEL, **kwargs) as runner,
    ):
        params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)
        extra = {"lora_request": _lora_request()} if lora else {}
        generate = runner.generate(prompts or PROMPTS, params, **extra)
        result = ([(token_ids[0], text[0]) for token_ids, text in generate], _spec_metrics(runner))

    _RUN_CACHE[key] = result
    return result


def _lora_request():
    from vllm.lora.request import LoRARequest

    return LoRARequest("synthetic", 1, LORA_ADAPTER)


def _assert_matches(name_ref: str, ref: tuple, name_cmp: str, cmp: tuple) -> None:
    ref_outputs, (ref_drafts, _) = ref
    cmp_outputs, (cmp_drafts, cmp_accepted) = cmp
    check_outputs_equal(
        outputs_0_lst=ref_outputs,
        outputs_1_lst=cmp_outputs,
        name_0=name_ref,
        name_1=name_cmp,
    )
    assert ref_drafts > 0, f"{name_ref} produced no drafts; the comparison is vacuous"
    assert cmp_drafts > 0, f"{name_cmp} produced no drafts (speculation silently disabled)"
    assert cmp_accepted > 0, f"{name_cmp} accepted no draft tokens"


@requires_four_npus
@pytest.mark.e2e_model(MODEL)
@_coverage()
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mrv1_tp2_pp2_mtp_matches_tp2_pp1() -> None:
    """The #17149 configuration: V1 runner, TP2, PP 1 vs 2."""
    pp1 = _run(1)
    pp2 = _run(2)
    _assert_matches("TP2xPP1+MTP3", pp1, "TP2xPP2+MTP3", pp2)


@requires_four_npus
@pytest.mark.e2e_model(MODEL)
@_coverage()
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mrv2_tp2_pp2_mtp_matches_tp2_pp1() -> None:
    """Attribution control: #17149 reports the V2 runner as correct here."""
    pp1 = _run(1, v2_runner=True)
    pp2 = _run(2, v2_runner=True)
    _assert_matches("MRV2 TP2xPP1+MTP3", pp1, "MRV2 TP2xPP2+MTP3", pp2)


@requires_four_npus
@pytest.mark.e2e_model(MODEL)
@_coverage()
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mrv1_tp2_pp2_mtp_matches_mrv2_tp2_pp2() -> None:
    """The fixed V1 path must converge to the known-good V2 path on 4 devices."""
    v2 = _run(2, v2_runner=True)
    v1 = _run(2)
    _assert_matches("MRV2 TP2xPP2+MTP3", v2, "MRV1 TP2xPP2+MTP3", v1)


@requires_four_npus
@pytest.mark.e2e_model(MODEL)
@_coverage(feature="spec_decode,long_sequence")
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mrv1_tp2_pp2_mtp_long_sequence() -> None:
    """Cross-stage KV cache with a TP-sharded attention head."""
    pp1 = _run(1, prompts=[LONG_PROMPT], max_model_len=4096, max_num_batched_tokens=4096)
    pp2 = _run(2, prompts=[LONG_PROMPT], max_model_len=4096, max_num_batched_tokens=4096)
    _assert_matches("TP2xPP1 long+MTP3", pp1, "TP2xPP2 long+MTP3", pp2)


@requires_four_npus
@pytest.mark.e2e_model(MODEL)
@_coverage(feature="spec_decode,chunked_prefill")
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mrv1_tp2_pp2_mtp_micro_batch() -> None:
    """Chunked prefill: the long prompt is split into 8 prefill chunks."""
    pp1 = _run(1, prompts=[LONG_PROMPT], max_model_len=4096, max_num_batched_tokens=512)
    pp2 = _run(2, prompts=[LONG_PROMPT], max_model_len=4096, max_num_batched_tokens=512)
    _assert_matches("TP2xPP1 micro-batch+MTP3", pp1, "TP2xPP2 micro-batch+MTP3", pp2)


@requires_four_npus
@pytest.mark.e2e_model(MODEL)
@_coverage(feature="spec_decode,lora")
@pytest.mark.skipif(not LORA_ADAPTER, reason="set MRV1_PP_LORA_ADAPTER to a synthetic adapter dir")
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mrv1_tp2_pp2_mtp_with_lora() -> None:
    """LoRA weights are sharded across TP ranks as well as staged across PP."""
    pp1 = _run(1, lora=True)
    pp2 = _run(2, lora=True)
    _assert_matches("TP2xPP1+LoRA+MTP3", pp1, "TP2xPP2+LoRA+MTP3", pp2)


@requires_four_npus
@pytest.mark.e2e_model(MODEL)
@_coverage(graph_mode="full_decode_only")
@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_mrv1_tp2_pp2_mtp_graph_mode() -> None:
    """ACLGraph FULL_DECODE_ONLY, matching the two-card graph case on 4 devices."""
    pp1 = _run(1, graph=True, max_num_batched_tokens=1024)
    pp2 = _run(2, graph=True, max_num_batched_tokens=1024)
    _assert_matches("TP2xPP1 graph+MTP3", pp1, "TP2xPP2 graph+MTP3", pp2)


# Notes for the first real run (four-card node, then CI):
#
# 1. ``_coverage(feature="spec_decode,lora")`` and ``graph_mode="full_decode_only"``
#    are the only marker values here not copied from an already-passing file. The
#    taxonomy lives in ``tests/e2e/conftest.py``; confirm both before trusting a
#    green run, since an invalid marker hits the conftest error branch that
#    references the non-existent ``pytest.CollectError``.
# 2. ``MRV1_PP_LORA_ADAPTER`` expects a directory produced by the synthetic
#    generator used on the two-card boxes (deterministic rank-8 adapter over
#    ``self_attn.q_proj`` / ``self_attn.v_proj``; ``gen_lora.py`` in the handoff
#    package). It carries no training quality signal - it exists to exercise the
#    sharding path.
# 3. The graph case keeps ``speculative_config.enforce_eager=False``; the two-card
#    graph runs confirmed capture actually happened via ACLGraphWrapper logs.
# 4. ``VllmRunner.generate(prompts, params, lora_request=...)`` is assumed to forward
#    to ``LLM.generate``. The two-card file only ever called it with two positional
#    arguments, so confirm the signature in ``tests/e2e/conftest.py`` before
#    interpreting a LoRA-case failure as a product bug.
