# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 Model Runner V2 target and DSpark ACL graph smoke tests."""

from types import MethodType

import pytest

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.four_card.test_kimi_k3 import (
    _draft_config,
    _engine_args,
    _generate,
    _prompt,
    _write_config,
    _write_target,
)


@pytest.fixture(scope="module")
def k3_models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    tmp_path = tmp_path_factory.mktemp("k3-mrv2-graph-dummy")
    models = {"target": _write_target(tmp_path / "target")}
    for variant in ("gqa", "mla"):
        models[variant] = _write_config(tmp_path / variant, _draft_config(variant))
    return models


@pytest.fixture(autouse=True)
def k3_mrv2_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    monkeypatch.setenv("HCCL_BUFFSIZE", "512")


def _install_graph_replay_counters(worker) -> bool:
    managers = [worker.model_runner.cudagraph_manager]
    speculator = worker.model_runner.speculator
    if speculator is not None:
        managers.append(speculator.query_cudagraph_manager)

    for manager in managers:
        assert manager is not None
        original_run_fullgraph = manager.run_fullgraph
        manager._k3_test_replay_count = 0

        def tracked_run_fullgraph(
            self,
            desc,
            *,
            _original_run_fullgraph=original_run_fullgraph,
        ):
            self._k3_test_replay_count += 1
            return _original_run_fullgraph(desc)

        manager.run_fullgraph = MethodType(tracked_run_fullgraph, manager)
    return True


def _get_graph_replay_counts(worker) -> dict[str, int]:
    counts = {"target": worker.model_runner.cudagraph_manager._k3_test_replay_count}
    speculator = worker.model_runner.speculator
    if speculator is not None:
        counts["draft"] = speculator.query_cudagraph_manager._k3_test_replay_count
    return counts


def _run_graph_smoke(k3_models: dict[str, str], *, variant: str, with_draft: bool) -> None:
    args = _engine_args(k3_models, variant)
    args["enforce_eager"] = False
    if with_draft:
        args["speculative_config"]["enforce_eager"] = False
    else:
        del args["speculative_config"]
        args["compilation_config"]["cudagraph_capture_sizes"] = [1, 2, 4]

    with VllmRunner(k3_models["target"], **args) as runner:
        llm = runner.model
        assert all(llm.collective_rpc(_install_graph_replay_counters))
        _generate(
            llm,
            [_prompt(length, salt=i * 137) for i, length in enumerate((1, 127, 128, 129))],
        )

        replay_counts = llm.collective_rpc(_get_graph_replay_counts)
        assert replay_counts
        assert all(counts["target"] > 0 for counts in replay_counts), replay_counts
        if with_draft:
            assert all(counts["draft"] > 0 for counts in replay_counts), replay_counts
            drafts = [m for m in llm.get_metrics() if m.name == "vllm:spec_decode_num_drafts"]
            assert drafts and sum(m.value for m in drafts) > 0, "Requests bypassed DSpark"


def test_k3_mrv2_target_graph(k3_models: dict[str, str]) -> None:
    _run_graph_smoke(k3_models, variant="mla", with_draft=False)


@pytest.mark.parametrize("variant", ["gqa", "mla"], ids=["gqa-dspark-graph", "mla-dspark-graph"])
def test_k3_mrv2_dspark_graph(k3_models: dict[str, str], variant: str) -> None:
    _run_graph_smoke(k3_models, variant=variant, with_draft=True)
