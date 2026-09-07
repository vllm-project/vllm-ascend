# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.pp_utils import PendingRecv, PPHandler
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    reserve_aux_intermediate_tensor_slots,
    verify_supports_aux_hidden_states_over_pp,
)

from vllm_ascend.models.deepseek_v4.model import DeepseekV4Model
from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3Model
from vllm_ascend.patch.worker.patch_v2.patch_spec_pp import install_spec_pp_draft_update
from vllm_ascend.worker.v2.pp_utils import make_empty_intermediate_tensors


@pytest.mark.parametrize("excluded", ["none", "freed", "unsampled", "all"])
@pytest.mark.parametrize("update_drafts", [False, True])
def test_upstream_pp_receive_updates_only_live_draft_rows(excluded, update_drafts):
    handler = PPHandler.__new__(PPHandler)
    handler.device = torch.device("cpu")
    handler.main_stream = Mock()
    handler.req_idx_gen_np = np.zeros(6, dtype=np.int64)
    handler.max_sample_len = 3
    indices = np.array([1, 3, 5])
    need_sampled = np.ones(3, dtype=bool)
    if excluded == "freed":
        handler.req_idx_gen_np[3] = 1
    elif excluded == "unsampled":
        need_sampled[1] = False
    elif excluded == "all":
        need_sampled[:] = False
    slot = PendingRecv(
        event=object(),
        sampled_tokens=torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]]),
        num_sampled=torch.tensor([3, 1, 2]),
        num_rejected=torch.tensor([0, 2, 1]),
        idx_mapping=torch.from_numpy(indices),
        idx_mapping_np=indices,
        need_sampled_mask=need_sampled,
        gen_at_receive_np=np.zeros(3, dtype=np.int64),
        draft_tokens=torch.tensor([[40, 41], [50, 51], [60, 61]]),
    )
    handler.queue = deque([slot])
    target = torch.full((6, 2), -9, dtype=torch.int64)
    expected = target.clone()
    if update_drafts and excluded != "all":
        expected[1] = slot.draft_tokens[0]
        expected[5] = slot.draft_tokens[2]
        if excluded == "none":
            expected[3] = slot.draft_tokens[1]

    install_spec_pp_draft_update(handler)
    installed = handler.get_prev_sampled_outputs
    install_spec_pp_draft_update(handler)
    assert handler.get_prev_sampled_outputs is installed
    # The parent's new separate collectives retain their original wire format.
    assert handler.max_sample_len == 3
    assert handler.broadcast.__func__ is PPHandler.broadcast
    assert handler.broadcast_drafts.__func__ is PPHandler.broadcast_drafts
    assert handler.receive.__func__ is PPHandler.receive

    def copy_to_cpu(value, *, device):
        return torch.as_tensor(value, device=device)

    with (
        patch("vllm.v1.worker.gpu.pp_utils.async_copy_to_gpu", side_effect=copy_to_cpu),
        patch(
            "vllm_ascend.patch.worker.patch_v2.patch_spec_pp.async_copy_to_gpu",
            side_effect=copy_to_cpu,
        ),
    ):
        outputs = handler.get_prev_sampled_outputs(target if update_drafts else None)

    torch.testing.assert_close(target, expected)
    assert list(handler.queue) == [None]
    if excluded == "all":
        assert outputs is None
    else:
        assert outputs["sampled_tokens"] is slot.sampled_tokens
        torch.testing.assert_close(
            outputs["idx_mapping"],
            torch.tensor([1, 3 if excluded == "none" else -1, 5]),
        )
        handler.main_stream.wait_event.assert_called_once_with(slot.event)


@pytest.mark.parametrize("model_cls", [MiniMaxM3Model, DeepseekV4Model])
def test_existing_ascend_aux_payload_matches_upstream_pp_relay(model_cls):
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model.start_layer = 4
    model.config = SimpleNamespace(hidden_size=2)
    model._enable_eagle3_aux_hidden_states = True
    pp = SimpleNamespace(world_size=3, is_first_rank=False, is_last_rank=False)

    with (
        patch("vllm.distributed.parallel_state.model_parallel_is_initialized", return_value=True),
        patch("vllm.distributed.parallel_state.get_pp_group", return_value=pp),
    ):
        model._set_aux_hidden_state_layers((2, 4, 8))
        outer = SimpleNamespace(model=model)
        outer.make_empty_intermediate_tensors = make_empty_intermediate_tensors(
            model,
            lambda batch, dtype, device: IntermediateTensors(
                {"hidden_states": torch.zeros(batch, 2, dtype=dtype, device=device)}
            ),
        )
        verify_supports_aux_hidden_states_over_pp(outer, "eagle3")
        reserve_aux_intermediate_tensor_slots(outer)
        handler = PPHandler.__new__(PPHandler)
        handler.configure_aux_hidden_state_relay(outer)
        incoming = outer.make_empty_intermediate_tensors(3, torch.float32, torch.device("cpu"))

    prefix = "pp_transport_aux_hidden_states_"
    assert handler.aux_hidden_state_relay_keys == (f"{prefix}0", f"{prefix}1")
    assert set(incoming.tensors) == {"hidden_states", f"{prefix}0", f"{prefix}1"}
    local_aux = torch.ones(3, 2)
    outgoing = IntermediateTensors({"hidden_states": torch.ones(3, 2), f"{prefix}2": local_aux})
    relayed = handler.relay_aux_hidden_states(incoming, outgoing)
    assert relayed[f"{prefix}2"] is local_aux
    for key in handler.aux_hidden_state_relay_keys:
        assert relayed[key] is incoming[key]
