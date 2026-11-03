# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU control-flow checks for the two-card IPC E2E; no device transfer runs."""

from unittest.mock import MagicMock, call, patch

import pytest

from tests.e2e.pull_request.two_card.rlhf.state_transitions import test_glm53flash_npu_ipc_weight_transfer as lane


@pytest.mark.parametrize(
    ("first", "visible", "physical"),
    [(0, None, "0,1"), (2, None, "2,3"), (0, "4,5", "4,5"), (1, "2,4,6", "4,6")],
)
def test_trainer_and_server_device_mapping(first, visible, physical):
    assert lane._server_devices(first, visible) == physical


@pytest.mark.parametrize(("first", "visible"), [(-1, None), (1, "4,5"), (0, "4,4"), (0, "-1,0")])
def test_invalid_device_pair_is_rejected(first, visible):
    with pytest.raises(ValueError):
        lane._server_devices(first, visible)


@pytest.mark.parametrize("rank", [0, 1])
def test_both_trainers_send_but_only_rank_zero_controls_server(rank):
    events = []
    engine = MagicMock()
    engine.send_weights.side_effect = lambda: events.append("send")
    server = MagicMock()
    case = MagicMock(model="flash")
    with (
        patch.object(lane.dist, "barrier", side_effect=lambda: events.append("barrier")),
        patch.object(lane, "_post", side_effect=lambda server, route, **kw: events.append((route, kw))),
        patch.object(lane, "generation_signature", return_value="signature") as generate,
    ):
        signatures = lane._send_updates(rank, engine, server, case)
    assert engine.send_weights.call_count == lane.UPDATE_ROUNDS
    if rank == 0:
        expected = [
            ("pause", {}),
            ("sleep", {"params": {"level": 2}}),
            ("wake_up", {"params": {"tags": ["weights"]}}),
            "barrier",
            "send",
            ("wake_up", {"params": {"tags": ["kv_cache"]}}),
            "barrier",
        ]
        assert events == expected * lane.UPDATE_ROUNDS
        assert signatures == ["signature"] * lane.UPDATE_ROUNDS
        assert generate.call_count == lane.UPDATE_ROUNDS
    else:
        assert events == ["barrier", "send", "barrier"] * lane.UPDATE_ROUNDS
        assert signatures == []
        generate.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_rank_device_assignment_and_process_group_cleanup(rank, monkeypatch):
    monkeypatch.setattr(lane.torch, "npu", MagicMock(), raising=False)
    with (
        patch.object(lane.dist, "init_process_group") as init,
        patch.object(lane.dist, "destroy_process_group") as destroy,
        patch.object(lane, "_run_rank_transaction", side_effect=RuntimeError("failed send")),
        pytest.raises(RuntimeError, match="failed send"),
    ):
        lane._trainer_rank(rank, 12345, 2)
    lane.torch.npu.set_device.assert_called_once_with(2 + rank)
    init.assert_called_once_with(
        "gloo", init_method="tcp://127.0.0.1:12345", rank=rank, world_size=2, timeout=lane.GROUP_TIMEOUT
    )
    destroy.assert_called_once_with()


def test_entrypoint_spawns_exactly_two_trainers(monkeypatch):
    monkeypatch.setenv("VLLM_RL_TEST_DEVICE_INDEX", "0")
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(lane.torch, "npu", MagicMock(), raising=False)
    lane.torch.npu.device_count.return_value = 2
    with patch.object(lane.mp, "spawn") as spawn, patch.object(lane, "get_open_port", return_value=12345):
        lane.test_glm53flash_npu_ipc_unpacked_tp2()
    spawn.assert_called_once_with(lane._trainer_rank, args=(12345, 0), nprocs=2, join=True)


def test_reference_and_live_servers_both_use_tp2_and_same_devices(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5")
    stack = MagicMock()
    case = MagicMock(model="flash", server_env=(("VLLM_USE_V2_MODEL_RUNNER", "0"),))
    with (
        patch.object(lane, "_wait_for_devices") as wait,
        patch.object(lane, "reference_signature", return_value="reference") as reference,
        patch.object(lane, "live_update_serve_args", return_value=["--tensor-parallel-size", "2"]) as live_args,
        patch.object(lane, "RemoteOpenAIServer") as live,
        patch.object(lane, "get_open_port", return_value=12345),
        patch.object(lane, "generation_signature", return_value="dummy"),
    ):
        _, expected, dummy = lane._start_servers(stack, case, MagicMock(), 0)
    assert (expected, dummy) == ("reference", "dummy")
    assert wait.call_args_list == [call(0), call(0)]
    assert reference.call_args.kwargs["tensor_parallel_size"] == live_args.call_args.kwargs["tensor_parallel_size"] == 2
    assert reference.call_args.kwargs["env_dict"] == live.call_args.kwargs["env_dict"]
    assert live.call_args.kwargs["env_dict"]["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"


def test_memory_gate_covers_both_devices():
    with patch.object(lane, "wait_for_free_device_memory") as wait:
        lane._wait_for_devices(2)
    assert wait.call_args_list == [call(2, lane.GPU_MEMORY_UTILIZATION), call(3, lane.GPU_MEMORY_UTILIZATION)]
