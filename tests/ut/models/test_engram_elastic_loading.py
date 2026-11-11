# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production load hooks with CPU modules and an external runner stub."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def production_class(relative, name, base, methods, namespace):
    source = Path(__file__).resolve().parents[3] / relative
    node = next(node for node in ast.parse(source.read_text()).body if getattr(node, "name", None) == name)
    node.bases = [ast.Name(id="Base", ctx=ast.Load())]
    node.decorator_list = []
    node.body = [child for child in node.body if getattr(child, "name", None) in methods]
    namespace = {**namespace, "Base": base}
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec")
    exec(code, namespace)
    return namespace[name]


@pytest.fixture
def loading():
    events = []

    class Compressor(torch.nn.Module):
        def process_weights_after_loading(self):
            events.append("compressor")

    class Indexer(torch.nn.Module):
        def process_weights_after_loading(self):
            events.append("indexer")

    model_cls = production_class(
        "vllm_ascend/models/deepseek_v41/model.py",
        "AscendDeepseekV41LLMForCausalLM",
        torch.nn.Module,
        {"has_engram", "uses_elastic_engram", "initialize_engram", "destroy_engram", "process_weights_after_loading"},
        {"DeepseekV41Compressor": Compressor, "DeepseekV41Indexer": Indexer},
    )
    model = model_cls()
    runtime = SimpleNamespace(
        initialize=Mock(side_effect=lambda: events.append("engram")),
        close=Mock(side_effect=lambda: events.append("close")),
    )
    model.model = SimpleNamespace(has_engram=True, _engram_elastic=True, _engram_elastic_runtime=runtime)
    model.compressor = Compressor()
    model.indexer = Indexer()

    class UpstreamRunner:
        # The external loader invokes the model's post-load hook before
        # returning it; the upstream runner then loads the draft and states.
        def load_model(self, load_dummy_weights=False, *args, **kwargs):
            self.forwarded = (load_dummy_weights, args, kwargs)
            events.append("weights")
            self.target_load()
            self.pending_model.process_weights_after_loading()
            self.model = self.pending_model
            events.append("draft")
            self.draft_load(self.model)
            events.append("states")

    logger = Mock()
    runner_cls = production_class(
        "vllm_ascend/worker/v2/model_runner.py",
        "NPUModelRunner",
        UpstreamRunner,
        {"load_model"},
        {"get_pd_dspark_aux_layer_ids": lambda config: (3, 7), "logger": logger},
    )
    runner = runner_cls()
    runner.vllm_config = object()
    runner.pending_model = model
    runner.target_load = Mock()
    runner.draft_load = Mock()
    return SimpleNamespace(runner=runner, model=model, runtime=runtime, events=events, logger=logger)


def test_v2_initializes_engram_after_weights_and_before_draft_and_states(loading):
    loading.runner.load_model(False, "arg", loader_option=True)
    assert loading.events == ["weights", "compressor", "indexer", "engram", "draft", "states"]
    assert loading.runner.forwarded == (False, ("arg",), {"loader_option": True})
    assert loading.runner.pd_dspark_aux_layer_ids == (3, 7)
    loading.runtime.initialize.assert_called_once()
    loading.runtime.close.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_v2_disabled_and_uva_do_not_initialize_elastic_resources(loading, enabled):
    loading.model.model.has_engram = enabled
    loading.model.model._engram_elastic = False
    loading.runner.load_model()
    assert loading.events == ["weights", "compressor", "indexer", "draft", "states"]
    loading.runtime.initialize.assert_not_called()
    loading.runtime.close.assert_not_called()


def test_engram_failure_aborts_before_draft_loading(loading):
    failure = RuntimeError("fetch probe failed")
    loading.runtime.initialize.side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        loading.runner.load_model()
    assert caught.value is failure
    loading.runner.draft_load.assert_not_called()
    assert loading.events == ["weights", "compressor", "indexer"]


def test_weight_postprocessing_failure_does_not_initialize_engram(loading):
    loading.model.indexer.process_weights_after_loading = Mock(side_effect=RuntimeError("indexer failed"))
    with pytest.raises(RuntimeError, match="indexer failed"):
        loading.runner.load_model()
    loading.runtime.initialize.assert_not_called()
    loading.runner.draft_load.assert_not_called()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_draft_failure_releases_engram_and_preserves_original_error(loading, cleanup_fails):
    failure = RuntimeError("draft failed")
    loading.runner.draft_load.side_effect = failure
    if cleanup_fails:
        loading.runtime.close.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError) as caught:
        loading.runner.load_model()
    assert caught.value is failure
    loading.runtime.close.assert_called_once()
    assert loading.logger.exception.call_count == int(cleanup_fails)


@pytest.mark.parametrize("target_fails", [False, True])
def test_reload_closes_previous_model_once_before_loading(loading, target_fails):
    previous = SimpleNamespace(
        uses_elastic_engram=True, destroy_engram=Mock(side_effect=lambda: loading.events.append("old-close"))
    )
    loading.runner.model = previous
    if target_fails:
        loading.runner.target_load.side_effect = RuntimeError("target failed")
        with pytest.raises(RuntimeError, match="target failed"):
            loading.runner.load_model()
        assert loading.runner.model is previous
    else:
        loading.runner.load_model()
        assert loading.runner.model is loading.model
    previous.destroy_engram.assert_called_once()
    assert loading.events[:2] == ["old-close", "weights"]
