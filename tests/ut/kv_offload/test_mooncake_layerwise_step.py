"""Tests for the layerwise step controller / transfer plan.

Expected request fields are hardcoded from the worker formulas; the controller is
never used to build its own expectations.
"""

import copy
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

fake_torch_npu = types.ModuleType("torch_npu")
fake_torch_npu.npu = MagicMock()  # type: ignore[attr-defined]
sys.modules.setdefault("torch_npu", fake_torch_npu)
torch.npu = fake_torch_npu.npu

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.layerwise.step import (  # noqa: E402
    LayerwiseStepController,
    LayerwiseTransferPlan,
)


def _send_task_factory(**kwargs):
    fields: dict[str, object] = {"send_request": {}, "failed_requests": {}}
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def _req(**overrides):
    fields = dict(
        local_block_ids=[[3]],
        remote_block_ids=[[5]],
        remote_block_size=[16],
        remote_host="host",
        remote_port=30000,
        remote_engine_id="engine",
        remote_request_id="req",
        remote_tp_size=1,
        remote_pcp_size=1,
        remote_dcp_size=1,
        num_external_tokens=8,
        num_computed_tokens=0,
        num_prompt_blocks=1,
        local_full_block_ids=None,
        chunk_finish=False,
        do_virtual=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _metadata(requests):
    return SimpleNamespace(
        requests=requests,
        send_task=SimpleNamespace(
            group_rearrange_block_ids=None,
            group_num_blocks=None,
            group_num_tokens=None,
            group_block_table=None,
            group_block_len_tensor=None,
            group_seq_start_tensor=None,
        ),
        spec=SimpleNamespace(kv_cache_spec_type="FullAttentionSpec"),
        producer=True,
    )


def _controller(*, align=None, split=None, mark=None, complete=None, send_queue=None):
    return LayerwiseStepController(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False),
            quant_config=None,
        ),
        kv_cache_specs=[SimpleNamespace(kv_cache_spec_type="FullAttentionSpec")],
        num_kv_cache_groups=1,
        block_size=[16],
        kernel_block_size_scale=[1],
        attn_resharding_group_idx=set(),
        total_layers=1,
        index_to_name={0: ["layer0"]},
        layer_metadata={"layer0": SimpleNamespace(tensor_group_idx=[0])},
        sparse_sfa_c8_layers=set(),
        pd_head_ratio=1,
        enable_c8_quant=False,
        enable_kv_quant=False,
        k_buffer=SimpleNamespace(device="cpu"),
        v_buffer=None,
        kv_send_layer_thread=SimpleNamespace(send_queue=send_queue or MagicMock()),
        resharding_stream=None,
        current_layer=0,
        cache_write_events=[MagicMock()],
        layer_storage_slots={0: (0,)},
        storage_send_done_events=[],
        pending_reuse_layers=set(),
        storage_send_errors={},
        align_remote_block_ids=align or (lambda req_meta: None),
        get_kv_split_metadata=split
        or (
            lambda _req_meta, _req_idx, _req_id, _group_idx: {
                ("host", 30000): {
                    "local_block_ids": [3],
                    "remote_block_ids": [5],
                    "trans_count": 1,
                }
            }
        ),
        get_kv_split_metadata_for_mamba=lambda *_a, **_k: {},
        get_kernel_block_ids=lambda block_ids: block_ids,
        update_decoder_info=lambda _req_id, req_meta: req_meta,
        get_nz_cache=lambda cache, _idx: cache,
        mark_layer_reuse_pending=mark or MagicMock(),
        complete_layer_reuse=complete or MagicMock(),
        send_task_factory=_send_task_factory,
    )


def _assert_tree_equal(actual: Any, expected: Any, path: str = "root") -> None:
    """Recursively compare the prepared snapshot with the live metadata tree."""
    if isinstance(actual, SimpleNamespace) and isinstance(expected, SimpleNamespace):
        assert actual.__dict__.keys() == expected.__dict__.keys(), (
            f"{path}: attribute names changed: {sorted(actual.__dict__)} != {sorted(expected.__dict__)}"
        )
        for key in actual.__dict__:
            _assert_tree_equal(actual.__dict__[key], expected.__dict__[key], f"{path}.{key}")
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert list(actual.keys()) == list(expected.keys()), (
            f"{path}: dict keys changed: {list(actual)} != {list(expected)}"
        )
        for key in actual:
            _assert_tree_equal(actual[key], expected[key], f"{path}[{key!r}]")
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected), f"{path}: sequence type changed"
        assert len(actual) == len(expected), f"{path}: sequence length changed: {len(actual)} != {len(expected)}"
        for idx, (left, right) in enumerate(zip(actual, expected)):
            _assert_tree_equal(left, right, f"{path}[{idx}]")
        return
    assert actual == expected, f"{path}: value changed: {actual!r} != {expected!r}"


def test_prepare_builds_plan_from_hand_computed_request_fields():
    metadata = _metadata({"req": _req()})
    plan = _controller().prepare(metadata)

    assert isinstance(plan, LayerwiseTransferPlan), "prepare must return LayerwiseTransferPlan"
    assert plan.producer is True, "producer flag must be true"
    assert list(plan.requests) == ["req"], f"plan request keys must be ['req'], got {list(plan.requests)}"
    prepared = plan.requests["req"]
    assert prepared.local_block_ids == [[3]], "kernel local ids must be [[3]]"
    assert prepared.remote_block_ids == [[5]], "kernel remote ids must be [[5]]"
    assert prepared.remote_host == "host", f"remote_host must be host, got {prepared.remote_host!r}"
    assert prepared.remote_port == 30000, f"remote_port must be 30000, got {prepared.remote_port!r}"
    assert prepared.trans_count == [1], f"trans_count must be [1], got {prepared.trans_count!r}"
    assert plan.send_task is metadata.send_task, "no resharding keeps the scheduler send task"
    assert plan.send_task_update is None, "no resharding must not build a send-task update"


def test_prepare_aligns_plan_copy_and_apply_replays_original_write():
    def align(req_meta):
        req_meta.remote_block_ids = [[50]]

    metadata = _metadata({"req": _req(remote_block_ids=[[4]])})
    original_req = metadata.requests["req"]
    controller = _controller(align=align)
    plan = controller.prepare(metadata)

    assert original_req.remote_block_ids == [[4]], "prepare must not write the scheduler request"
    assert plan.requests["req"].remote_block_ids == [[5]], "split result overwrites aligned ids in plan"

    controller.apply(plan, metadata)

    assert original_req.remote_block_ids == [[50]], "apply must replay the aligned write on the original"
    assert metadata.requests is not plan.requests, "apply replaces metadata.requests with the plan mapping"


def test_prepare_zero_write_deep_snapshot_guard():
    metadata = _metadata({"req": _req(remote_block_ids=[[4]])})
    requests_before = metadata.requests
    req_ids_before = {req_id: id(req_meta) for req_id, req_meta in metadata.requests.items()}
    snapshot = copy.deepcopy(metadata)

    controller = _controller(align=lambda req_meta: setattr(req_meta, "remote_block_ids", [[50]]))
    plan = controller.prepare(metadata)

    assert metadata.requests is requests_before, "prepare must not replace metadata.requests"
    assert {req_id: id(req_meta) for req_id, req_meta in metadata.requests.items()} == req_ids_before, (
        "prepare must not replace any ReqMeta object"
    )
    _assert_tree_equal(copy.deepcopy(metadata), snapshot)

    controller.apply(plan, metadata)

    assert metadata.requests is not requests_before, "apply must replace metadata.requests"
    with pytest.raises(AssertionError):
        _assert_tree_equal(copy.deepcopy(metadata), snapshot)


def test_apply_replays_chunk_finish_on_original_req():
    def split(req_meta, _req_idx, _req_id, _group_idx):
        req_meta.chunk_finish = True
        return {
            ("host", 30000): {
                "local_block_ids": [3],
                "remote_block_ids": [5],
                "trans_count": 1,
            }
        }

    metadata = _metadata({"req": _req(chunk_finish=False)})
    original_req = metadata.requests["req"]
    controller = _controller(split=split)
    plan = controller.prepare(metadata)

    assert original_req.chunk_finish is False, "prepare must not write chunk_finish on scheduler ReqMeta"
    controller.apply(plan, metadata)
    assert original_req.chunk_finish is True, "apply must replay chunk_finish=True on scheduler ReqMeta"


def test_apply_replays_send_task_group_writes():
    metadata = _metadata({"req": _req(local_block_ids=[[3]], remote_block_ids=[[5]])})
    controller = _controller()
    controller.attn_resharding_group_idx = {0}
    plan = controller.prepare(metadata)
    assert plan.send_task_update is not None, "resharding must build a send-task update"
    assert plan.send_task_update.group_rearrange_block_ids == [[3]], "group rearrange ids must be [[3]]"

    controller.apply(plan, metadata)

    assert metadata.send_task.group_rearrange_block_ids == [[3]], "group_rearrange_block_ids must be [[3]]"
    assert metadata.send_task.group_num_blocks == [1], "group_num_blocks must be [1]"
    assert metadata.send_task.group_num_tokens == [16], "group_num_tokens must be [16]"
    assert metadata.send_task.group_block_table[0].tolist() == [[3]], "block table must be [[3]]"
    assert metadata.send_task.group_block_len_tensor[0].tolist() == [16], "block len tensor must be [16]"
    assert metadata.send_task.group_seq_start_tensor[0].tolist() == [0], "seq start tensor must be [0]"


def test_save_layer_marks_reuse_pending_and_queues_once():
    send_queue = MagicMock()
    mark = MagicMock()
    metadata = _metadata({"req": _req(local_block_ids=[[1]])})
    controller = _controller(mark=mark, send_queue=send_queue)
    plan = controller.prepare(metadata)

    controller.save_layer("layer0", [], MagicMock(), plan)

    mark.assert_called_once_with(0)
    assert send_queue.put.call_count == 1, "one layer send task must be queued"
    assert controller.current_layer == 1, f"current_layer must be 1, got {controller.current_layer}"


def test_facade_on_kv_cache_written_noops_without_plan():
    import importlib.util

    fake_engine = types.ModuleType("mooncake.engine")
    fake_engine.TransferEngine = MagicMock()  # type: ignore[attr-defined]
    sys.modules.setdefault("mooncake.engine", fake_engine)
    fake_torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)

    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
        MooncakeLayerwiseConnector,
        MooncakeLayerwiseConnectorMetadata,
        MooncakeLayerwiseConnectorWorker,
    )

    connector = MooncakeLayerwiseConnector.__new__(MooncakeLayerwiseConnector)
    connector._is_kv_producer = True
    connector._connector_metadata = MooncakeLayerwiseConnectorMetadata()
    connector._connector_metadata.requests["req"] = MagicMock()
    connector.connector_worker = MooncakeLayerwiseConnectorWorker.__new__(MooncakeLayerwiseConnectorWorker)
    connector.connector_worker.on_kv_cache_written = MagicMock()

    connector.on_kv_cache_written("layer0")

    connector.connector_worker.on_kv_cache_written.assert_not_called()


def test_save_layer_rejects_missing_plan_send_task():
    metadata = _metadata({"req": _req()})
    controller = _controller()
    plan = controller.prepare(metadata)
    plan.send_task = None

    with pytest.raises(RuntimeError) as excinfo:
        controller.save_layer("layer0", [], MagicMock(), plan)

    assert str(excinfo.value) == "Layerwise plan is missing send_task"


def test_on_written_rejects_missing_physical_layer():
    controller = _controller()
    controller.index_to_name = {0: []}

    with pytest.raises(RuntimeError) as excinfo:
        controller.on_written("layer0")

    assert str(excinfo.value) == "Mooncake layerwise metadata is missing physical layer 0"


def test_on_written_rejects_hook_not_in_physical_layer():
    controller = _controller()

    with pytest.raises(RuntimeError) as excinfo:
        controller.on_written("other")

    assert str(excinfo.value) == ("Mooncake layer hook 'other' does not belong to physical layer 0: ['layer0']")


def test_save_layer_rejects_missing_physical_layer():
    metadata = _metadata({"req": _req()})
    controller = _controller()
    plan = controller.prepare(metadata)
    controller.index_to_name = {0: []}

    with pytest.raises(RuntimeError) as excinfo:
        controller.save_layer("layer0", [], MagicMock(), plan)

    assert str(excinfo.value) == "Mooncake layerwise metadata is missing physical layer 0"


def test_save_layer_rejects_hook_not_in_physical_layer():
    metadata = _metadata({"req": _req()})
    controller = _controller()
    plan = controller.prepare(metadata)

    with pytest.raises(RuntimeError) as excinfo:
        controller.save_layer("other", [], MagicMock(), plan)

    assert str(excinfo.value) == ("Mooncake layer hook 'other' does not belong to physical layer 0: ['layer0']")


def test_prepare_rejects_multiple_transfer_tasks():
    def split(_req_meta, _req_idx, _req_id, _group_idx):
        return {
            ("host-a", 30000): {"local_block_ids": [1], "remote_block_ids": [2], "trans_count": 1},
            ("host-b", 30001): {"local_block_ids": [3], "remote_block_ids": [4], "trans_count": 1},
        }

    metadata = _metadata({"req": _req()})
    controller = _controller(split=split)

    with pytest.raises(AssertionError) as excinfo:
        controller.prepare(metadata)

    assert str(excinfo.value) == "Not support add mutil transfer task for req_id:req"


def test_save_layer_put_failure_rolls_back_reuse_gate():
    send_queue = MagicMock()
    send_queue.put.side_effect = RuntimeError("put boom")
    mark = MagicMock()
    complete = MagicMock()
    metadata = _metadata({"req": _req(local_block_ids=[[1]])})
    controller = _controller(mark=mark, complete=complete, send_queue=send_queue)
    plan = controller.prepare(metadata)

    with pytest.raises(RuntimeError) as excinfo:
        controller.save_layer("layer0", [], MagicMock(), plan)

    assert str(excinfo.value) == "put boom"
    mark.assert_called_once_with(0)
    complete.assert_called_once_with(0, "put boom")
