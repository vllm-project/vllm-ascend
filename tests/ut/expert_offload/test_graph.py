# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.expert_offload.expert_offload_manager import ExpertOffloadManager, OffloadState
from vllm_ascend.expert_offload.graph import GraphExpertStaging, supports_descriptor
from vllm_ascend.expert_offload.residency import ExpertResidency
from vllm_ascend.expert_offload.transfer import TorchExpertTransfer
from vllm_ascend.expert_offload.utils import ExpertMapBuffer


@pytest.mark.parametrize(
    "tokens,uniform,expected",
    [(1, 1, True), (2, 1, True), (2, 2, True), (3, 1, False), (0, 1, False), (1, None, False), (1, 2, False)],
)
def test_padded_descriptor_capacity(tokens, uniform, expected):
    bindings = [SimpleNamespace(top_k=6, num_slots=16), SimpleNamespace(top_k=4, num_slots=8)]
    desc = SimpleNamespace(num_tokens=tokens, uniform_token_count=uniform)
    assert supports_descriptor(bindings, desc) is expected
    assert not supports_descriptor([], desc)


def test_speculative_graph_bounds_include_all_verification_and_padding_rows():
    bindings = [SimpleNamespace(top_k=6, num_slots=96)]
    assert supports_descriptor(bindings, SimpleNamespace(num_tokens=6, uniform_token_count=6))
    assert supports_descriptor(bindings, SimpleNamespace(num_tokens=12, uniform_token_count=6))
    assert not supports_descriptor(bindings, SimpleNamespace(num_tokens=18, uniform_token_count=6))
    assert not supports_descriptor(bindings, SimpleNamespace(num_tokens=6, uniform_token_count=None))


@pytest.mark.parametrize("requests,expected", [(1, True), (2, True), (4, False)])
def test_glm_mtp_three_draft_tokens_with_68_resident_experts(requests, expected):
    bindings = [SimpleNamespace(top_k=8, num_slots=68)]
    # Every request verifies the new target token plus three draft tokens.
    desc = SimpleNamespace(num_tokens=requests * 4, uniform_token_count=4)
    assert supports_descriptor(bindings, desc) is expected


def staging_double():
    manager = ExpertOffloadManager()
    manager.state = OffloadState.READY
    manager.begin_execution()
    manager.graph_load_stream = object()
    binding = SimpleNamespace(
        manager=manager, residency=ExpertResidency(6, 2), device="cpu", adapter=None, host_store=None, resident={}
    )
    staging = object.__new__(GraphExpertStaging)
    staging.binding = binding
    with torch.inference_mode():
        staging.ids = torch.tensor([[4, 5]])
        staging.mapping = ExpertMapBuffer(6, torch.device("cpu"))
    return manager, staging


def test_callback_updates_routes_each_replay_and_commits_after_copy():
    manager, staging = staging_double()
    addresses = (staging.mapping.host_map.data_ptr(), staging.mapping.device_map.data_ptr())
    with (
        patch.object(torch.npu, "set_device", create=True),
        patch.object(torch.npu, "stream", return_value=nullcontext(), create=True),
        patch.object(TorchExpertTransfer, "copy_experts") as copy,
    ):
        with patch.object(torch, "tensor", side_effect=AssertionError("mapping must reuse its storage")):
            staging.update()
        assert copy.call_count == 1
        assert staging.mapping.host_map[4:6].tolist() == [0, 1]
        with torch.inference_mode():
            staging.ids.copy_(torch.tensor([[1, 4]]))
        with patch.object(torch, "tensor", side_effect=AssertionError("mapping must reuse its storage")):
            staging.update()
        assert copy.call_count == 2
        assert staging.mapping.host_map[1] >= 0 and staging.mapping.host_map[5] == -1
        assert addresses == (staging.mapping.host_map.data_ptr(), staging.mapping.device_map.data_ptr())
    assert manager.graph_error is None


def test_callback_failure_is_rejected_and_never_commits_partial_bundle():
    manager, staging = staging_double()
    before = staging.binding.residency.logical_to_physical()
    with (
        patch.object(torch.npu, "set_device", create=True),
        patch.object(torch.npu, "stream", return_value=nullcontext(), create=True),
        patch.object(TorchExpertTransfer, "copy_experts", side_effect=RuntimeError("injected transfer failure")),
    ):
        staging.update()
    assert staging.binding.residency.logical_to_physical() == before
    with pytest.raises(RuntimeError, match="rejecting execution output") as raised:
        manager.check_graph_errors()
    assert str(raised.value.__cause__) == "injected transfer failure"
    assert manager.state is OffloadState.FAILED
    with pytest.raises(RuntimeError, match="ready"):
        manager.begin_execution()


def test_subscriptions_are_runner_owned_and_deduplicated():
    manager = ExpertOffloadManager()
    stream = Mock()
    with (
        patch.object(torch.npu, "_subscribe_report", create=True) as subscribe,
        patch.object(torch.npu, "_unsubscribe_report", create=True) as unsubscribe,
    ):
        manager.subscribe_graph_stream(stream)
        manager.subscribe_graph_stream(stream)
        subscribe.assert_called_once_with(stream)
        manager.close()
        unsubscribe.assert_called_once_with(stream)
        manager.close()
        assert unsubscribe.call_count == 1


@pytest.mark.parametrize("failed", [False, True])
def test_capture_lifecycle_is_retired_on_success_and_failure(failed):
    manager = ExpertOffloadManager()
    manager.state = OffloadState.READY
    if failed:
        with pytest.raises(RuntimeError, match="capture failed"), manager.graph_capture():
            assert manager.executing and manager.capturing_graphs and not manager.has_prefill
            raise RuntimeError("capture failed")
    else:
        with manager.graph_capture():
            assert manager.executing and manager.capturing_graphs and not manager.has_prefill
    assert not manager.executing and not manager.capturing_graphs
    assert manager.has_prefill
    assert manager.state is (OffloadState.FAILED if failed else OffloadState.READY)
