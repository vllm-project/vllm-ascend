"""Tests for the layerwise step controller / transfer plan.

Expected request fields are hardcoded from the worker formulas; the controller is
never used to build its own expectations.
"""

import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

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


def _metadata(requests, producer=True):
    spec = SimpleNamespace(kv_cache_spec_type="FullAttentionSpec")
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
        spec=spec,
        producer=producer,
    )


def _controller(*, align=None, split=None, mark=None, send_queue=None):
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
        complete_layer_reuse=MagicMock(),
        send_task_factory=_send_task_factory,
    )


class _GuardedMetadata:
    _writes: list[tuple[str, object]]
    requests: dict[str, Any]
    send_task: SimpleNamespace

    def __init__(self, requests):
        object.__setattr__(self, "_writes", [])
        object.__setattr__(self, "requests", requests)
        object.__setattr__(
            self,
            "send_task",
            SimpleNamespace(
                group_rearrange_block_ids=None,
                group_num_blocks=None,
                group_num_tokens=None,
                group_block_table=None,
                group_block_len_tensor=None,
                group_seq_start_tensor=None,
            ),
        )

    def __setattr__(self, name, value):
        if not name.startswith("_"):
            self._writes.append((name, value))
        object.__setattr__(self, name, value)


def test_prepare_builds_plan_from_hand_computed_request_fields():
    metadata = _metadata({"req": _req()})
    plan = _controller().prepare(metadata)

    assert isinstance(plan, LayerwiseTransferPlan)
    assert plan.producer is True
    assert list(plan.requests) == ["req"]
    prepared = plan.requests["req"]
    assert prepared.local_block_ids == [[3]], "kernel local ids must be [[3]]"
    assert prepared.remote_block_ids == [[5]], "kernel remote ids must be [[5]]"
    assert prepared.remote_host == "host"
    assert prepared.remote_port == 30000
    assert prepared.trans_count == [1]
    assert plan.send_task is metadata.send_task, "no resharding keeps the scheduler send task"
    assert plan.send_task_update is None


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


def test_prepare_plan_is_pure_until_apply():
    metadata = _GuardedMetadata({"req": _req(remote_block_ids=[[4]])})
    original_req = metadata.requests["req"]
    before = (list(original_req.remote_block_ids), original_req.remote_host, original_req.remote_port)

    plan = _controller(align=lambda req_meta: setattr(req_meta, "remote_block_ids", [[50]])).prepare(metadata)

    after = (list(original_req.remote_block_ids), original_req.remote_host, original_req.remote_port)
    assert metadata._writes == [], "prepare must not set attributes on scheduler metadata"
    assert after == before, "prepare must not mutate scheduler ReqMeta"

    _controller(align=lambda req_meta: None).apply(plan, metadata)
    assert metadata._writes, "apply must write metadata.requests"
    assert metadata.requests["req"].local_block_ids == [[3]]


def test_apply_replays_send_task_group_writes():
    metadata = _metadata(
        {"req": _req(local_block_ids=[[3]], remote_block_ids=[[5]])},
    )
    controller = _controller()
    controller.attn_resharding_group_idx = {0}
    plan = controller.prepare(metadata)
    assert plan.send_task_update is not None
    assert plan.send_task_update.group_rearrange_block_ids == [[3]]

    controller.apply(plan, metadata)

    assert metadata.send_task.group_rearrange_block_ids == [[3]]
    assert metadata.send_task.group_num_blocks == [1]
    assert metadata.send_task.group_num_tokens == [16]
    assert metadata.send_task.group_block_table[0].tolist() == [[3]]
    assert metadata.send_task.group_block_len_tensor[0].tolist() == [16]
    assert metadata.send_task.group_seq_start_tensor[0].tolist() == [0]


def test_save_layer_marks_reuse_pending_and_queues_once():
    send_queue = MagicMock()
    mark = MagicMock()
    metadata = _metadata({"req": _req(local_block_ids=[[1]])})
    controller = _controller(mark=mark, send_queue=send_queue)
    plan = controller.prepare(metadata)

    controller.save_layer("layer0", [], MagicMock(), plan)

    mark.assert_called_once_with(0)
    assert send_queue.put.call_count == 1, "one layer send task must be queued"
    assert controller.current_layer == 1


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
