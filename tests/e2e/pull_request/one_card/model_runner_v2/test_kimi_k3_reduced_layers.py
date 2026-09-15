# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-card Kimi K3 MRV2 smoke tests with reduced layer counts.

The local dummy target keeps two layers (one KDA and one MLA), while each
DSpark draft keeps one layer. These tests validate target graph parity and
both GQA and MLA speculative paths without requiring a full checkpoint.
"""

import os
from types import MethodType
from unittest.mock import patch

import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.four_card.test_kimi_k3 import (
    DRAFT_LAYERS,
    NUM_LAYERS,
    _draft_config,
    _engine_args,
    _generate,
    _prompt,
    _write_config,
    _write_target,
)

PROMPT_LENGTHS = (1, 127, 128, 129)


@pytest.fixture(scope="module", autouse=True)
def k3_mrv2_runtime():
    with patch.dict(
        os.environ,
        {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "HCCL_BUFFSIZE": "512",
        },
    ):
        yield


@pytest.fixture(scope="module")
def k3_models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    assert NUM_LAYERS == 2
    assert DRAFT_LAYERS == 1
    tmp_path = tmp_path_factory.mktemp("k3-mrv2-reduced-layers")
    models = {"target": _write_target(tmp_path / "target")}
    for variant in ("gqa", "mla"):
        config = _draft_config(variant)
        if variant == "gqa":
            # Keep the TP1 GQA K/V width equal to the target MLA compressed
            # cache width: 9 heads * 32 dims * K/V = 512 + 64.
            config["num_key_value_heads"] = 9
        models[variant] = _write_config(tmp_path / variant, config)
    return models


def _single_card_args(models: dict[str, str], variant: str, *, with_draft: bool, graph: bool) -> dict:
    args = _engine_args(models, variant, tp=1)
    args["enforce_eager"] = not graph
    if with_draft:
        args["speculative_config"]["enforce_eager"] = not graph
    else:
        del args["speculative_config"]
    args["compilation_config"] = (
        {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]}
        if graph
        else {"cudagraph_mode": "NONE"}
    )
    return args


def _token_ids(outputs) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(output.outputs[0].token_ids) for output in outputs)


def _graph_prompts() -> list[dict]:
    return [_prompt(length, salt=i * 137) for i, length in enumerate(PROMPT_LENGTHS)]


@pytest.fixture(scope="module")
def target_eager_tokens(k3_models: dict[str, str]) -> tuple[tuple[int, ...], ...]:
    args = _single_card_args(k3_models, "mla", with_draft=False, graph=False)
    with VllmRunner(k3_models["target"], **args) as runner:
        return _token_ids(_generate(runner.model, _graph_prompts()))


def _install_target_graph_replay_counter(worker) -> bool:
    manager = worker.model_runner.cudagraph_manager
    assert manager is not None
    original_run_fullgraph = manager.run_fullgraph
    manager._k3_test_replay_count = 0

    def tracked_run_fullgraph(self, desc):
        self._k3_test_replay_count += 1
        return original_run_fullgraph(desc)

    manager.run_fullgraph = MethodType(tracked_run_fullgraph, manager)
    return True


def _get_target_graph_replay_count(worker) -> int:
    return worker.model_runner.cudagraph_manager._k3_test_replay_count


def test_k3_mrv2_reduced_layers_target_graph(
    k3_models: dict[str, str],
    target_eager_tokens: tuple[tuple[int, ...], ...],
) -> None:
    args = _single_card_args(k3_models, "mla", with_draft=False, graph=True)
    with VllmRunner(k3_models["target"], **args) as runner:
        llm = runner.model
        assert all(llm.collective_rpc(_install_target_graph_replay_counter))
        graph_tokens = _token_ids(_generate(llm, _graph_prompts()))
        replay_counts = llm.collective_rpc(_get_target_graph_replay_count)
        assert replay_counts and all(count > 0 for count in replay_counts)
    assert graph_tokens == target_eager_tokens


@pytest.mark.parametrize("variant", ["gqa", "mla"], ids=["gqa-dspark", "mla-dspark"])
def test_k3_mrv2_reduced_layers_dspark(
    k3_models: dict[str, str],
    target_eager_tokens: tuple[tuple[int, ...], ...],
    variant: str,
) -> None:
    args = _single_card_args(k3_models, variant, with_draft=True, graph=False)
    with VllmRunner(k3_models["target"], **args) as runner:
        llm = runner.model
        draft_tokens = _token_ids(_generate(llm, _graph_prompts()))
        drafts = [metric for metric in llm.get_metrics() if metric.name == "vllm:spec_decode_num_drafts"]
        assert drafts and sum(metric.value for metric in drafts) > 0, "Requests bypassed DSpark"
    assert draft_tokens == target_eager_tokens
