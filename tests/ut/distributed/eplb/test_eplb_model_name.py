# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.distributed.eplb import eplb_state


@pytest.mark.parametrize("kwargs", [{}, {"model_name": None}, {"model_name": "draft"}])
@pytest.mark.parametrize("custom_load_stats", [False, True])
def test_add_model_preserves_upstream_model_name(kwargs, custom_load_stats):
    state = eplb_state.AscendEplbState.__new__(eplb_state.AscendEplbState)
    state._hixl_session = None
    state.parallel_config = SimpleNamespace(eplb_config=SimpleNamespace(policy="stair"))
    state._configured_policy = None
    state.policy = SimpleNamespace(prepare_local_load_stats=MagicMock()) if custom_load_stats else None
    state._initialize_load_stats_state = MagicMock()
    model, model_config, model_state = MagicMock(), MagicMock(), object()
    model_config.compute_hash.return_value = "model-hash"
    state.model_states = {"model-hash": model_state}
    previous_ep_size = eplb_state.EXPERT_MAPPING_EP_SIZE.get()
    observed_ep_sizes: list[int] = []

    def record_ep_size(*args, **kwargs):
        observed_ep_sizes.append(eplb_state.EXPERT_MAPPING_EP_SIZE.get())

    with (
        patch.object(eplb_state, "get_ep_group", return_value=SimpleNamespace(world_size=8)),
        patch.object(
            eplb_state._eplb_state.EplbState, "add_model", autospec=True, side_effect=record_ep_size
        ) as upstream_add_model,
    ):
        state.add_model(model, model_config, **kwargs)

    upstream_add_model.assert_called_once_with(state, model, model_config, model_name=kwargs.get("model_name"))
    assert observed_ep_sizes == [8]
    assert eplb_state.EXPERT_MAPPING_EP_SIZE.get() == previous_ep_size
    if custom_load_stats:
        state._initialize_load_stats_state.assert_called_once_with(model_state)
    else:
        state._initialize_load_stats_state.assert_not_called()
