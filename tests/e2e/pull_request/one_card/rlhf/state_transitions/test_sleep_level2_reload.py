# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Level-2 sleep discards parameters and reload restores the output."""

from __future__ import annotations

import os

import torch
from vllm import LLM, SamplingParams

from tests.e2e.common.rlhf.common import MODEL_NAME

# The offline engine starts a separate core process. Spawn is required for a
# clean process tree on Ascend and for the reload RPC below.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _embedding_weight(worker):
    """Return the preserved embedding weight from one worker."""
    parameters = dict(worker.model_runner.get_model().named_parameters())
    return parameters["model.embed_tokens.weight"].detach().cpu().clone()


def test_level2_preserved_parameter_survives_without_reload():
    llm = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.3,
        enable_sleep_mode=True,
        enforce_eager=True,
        additional_config={"weight_nz_mode": 0},
        sleep_preserve_parameter_names=["model.embed_tokens.weight"],
    )
    before = llm.collective_rpc(_embedding_weight)[0]

    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])

    after = llm.collective_rpc(_embedding_weight)[0]
    assert torch.equal(after, before)


def test_level2_discard_reload_restores_output():
    llm = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.3,
        enable_sleep_mode=True,
        enforce_eager=True,
        additional_config={"weight_nz_mode": 0},
    )
    sampling = SamplingParams(temperature=0, max_tokens=8)
    prompt = "The capital of France is"
    golden = llm.generate([prompt], sampling)[0].outputs[0].text

    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])

    # reload_weights is currently an engine RPC rather than a public HTTP
    # endpoint. This test exercises the level-2 discard + reload path.
    llm.llm_engine.engine_core.collective_rpc(
        "reload_weights",
        kwargs={"weights_path": MODEL_NAME},
    )
    llm.wake_up(tags=["kv_cache"])

    after = llm.generate([prompt], sampling)[0].outputs[0].text
    assert after == golden
