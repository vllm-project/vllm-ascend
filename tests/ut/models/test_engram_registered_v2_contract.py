# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual registered wrapper rather than only its inner model."""

import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend import models
from vllm_ascend.worker.v2.model_states import init_asecnd_model_state
from vllm_ascend.worker.v2.model_states.deepseek_v41 import AscendDeepseekV41ModelState


@pytest.fixture
def registered_wrapper():
    registry = {}
    with patch.object(
        models.ModelRegistry,
        "register_model",
        side_effect=lambda name, target: registry.update({name: target}),
    ):
        models.register_model()
    module, name = registry["DeepseekV41ForCausalLM"].split(":")
    cls = getattr(importlib.import_module(module), name)
    wrapper = object.__new__(cls)
    torch.nn.Module.__init__(wrapper)
    wrapper.language_model = SimpleNamespace(
        get_model_state_cls=Mock(return_value=AscendDeepseekV41ModelState),
        prime_engram_v2_graph_inputs=Mock(return_value={"engram_mask": "mask"}),
        prepare_engram_inputs=Mock(return_value={"engram_lookups": "rows"}),
        retire_engram_lookups=Mock(),
    )
    return wrapper


@pytest.mark.parametrize("method", ["get_model_state_cls", "prime_engram_v2_graph_inputs"])
def test_registered_wrapper_exposes_v2_engram_hooks(registered_wrapper, method):
    assert callable(getattr(registered_wrapper, method, None))


@pytest.mark.parametrize("keyword", ["cg_mode", "force_dummy"])
def test_registered_wrapper_accepts_v2_preparation_keywords(registered_wrapper, keyword):
    assert keyword in inspect.signature(registered_wrapper.prepare_engram_inputs).parameters


def test_registry_to_runner_selects_engram_state(registered_wrapper):
    config, encoder_cache, device = Mock(), Mock(), torch.device("cpu")
    with patch.object(AscendDeepseekV41ModelState, "__init__", return_value=None) as initialize:
        state = init_asecnd_model_state(config, registered_wrapper, encoder_cache, device)
    assert isinstance(state, AscendDeepseekV41ModelState)
    initialize.assert_called_once_with(config, registered_wrapper, encoder_cache, device)
    registered_wrapper.language_model.get_model_state_cls.assert_called_once_with()


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
@pytest.mark.parametrize("dummy", [False, True])
def test_wrapper_forwards_history_and_runtime_mode(registered_wrapper, mode, dummy):
    ids = torch.tensor([12, 13], dtype=torch.int32)
    positions = torch.tensor([10, 11], dtype=torch.int64)
    lookback = torch.tensor([[11, 10, 9]], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    result = registered_wrapper.prepare_engram_inputs(
        ids, positions, 4, lookback, query_start_loc, force_dummy=dummy, cg_mode=mode
    )
    assert result == {"engram_lookups": "rows"}
    registered_wrapper.language_model.prepare_engram_inputs.assert_called_once_with(
        ids, positions, 4, lookback, query_start_loc, None, None, force_dummy=dummy, cg_mode=mode
    )


def test_wrapper_primes_the_same_graph_bucket_and_retires(registered_wrapper):
    assert registered_wrapper.prime_engram_v2_graph_inputs(96) == {"engram_mask": "mask"}
    registered_wrapper.language_model.prime_engram_v2_graph_inputs.assert_called_once_with(96)
    registered_wrapper.retire_engram_lookups(reset_events=True)
    registered_wrapper.language_model.retire_engram_lookups.assert_called_once_with(reset_events=True)


@pytest.mark.parametrize("enabled", [True, False])
def test_registered_wrapper_forwards_graph_producer_capability(registered_wrapper, enabled):
    registered_wrapper.language_model.supports_engram_graph_producer = enabled
    assert registered_wrapper.supports_engram_graph_producer is enabled
