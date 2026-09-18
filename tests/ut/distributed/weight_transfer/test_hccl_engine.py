#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Regression tests for the HCCL weight transfer engines.

These pin the ownership split the Ascend engines share with upstream vLLM's
``NCCLTrainerWeightTransferEngine`` / ``NCCLWeightTransferEngine``:

1. the packed wire config lives on the *init* info and is shipped by the trainer
   at the handshake, so sender and receiver cannot disagree about the wire
   format; the per-round update info carries metadata only;
2. the trainer opens its own endpoint through the shared ``hccl_common`` helpers
   instead of reaching into the inference engine's private statics;
3. the receive paths close vLLM's MTP completeness check, because a transfer
   loads the model in batches;
4. the sender validates ``metadata()`` against iteration before broadcasting;
5. ``_post_send_sync`` is a trainer-side concern, owned by the trainer engine.
"""

import inspect
from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.weight_transfer.base import ParamMeta, TrainerWeightTransferEngine

from vllm_ascend.distributed.weight_transfer import hccl_common
from vllm_ascend.distributed.weight_transfer.hccl_common import (
    HCCLWeightTransferInitInfo,
)
from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLTrainerInitInfo,
    HCCLTrainerWeightTransferEngine,
    HCCLWeightTransferEngine,
    HCCLWeightTransferUpdateInfo,
)

_MODULE = "vllm_ascend.distributed.weight_transfer.hccl_engine"
_MTP = "vllm.model_executor.model_loader.mtp_validation.disable_mtp_completeness_check"

MASTER_ADDRESS = "127.0.0.1"
MASTER_PORT = 29500
WORLD_SIZE = 2
CPU_DEVICE = torch.device("cpu")


def _meta(*items: tuple[str, torch.dtype, tuple[int, ...]]) -> list[ParamMeta]:
    return [ParamMeta(name, dtype, shape) for name, dtype, shape in items]


def _worker_engine(*, packed: bool, device: torch.device = CPU_DEVICE) -> HCCLWeightTransferEngine:
    engine = object.__new__(HCCLWeightTransferEngine)
    engine.model = MagicMock()
    engine.model_update_group = MagicMock()
    engine.device = device
    engine.packed = packed
    engine.packed_buffer_size_bytes = 4096
    engine.packed_num_buffers = 4
    return engine


# ---------------------------------------------------------------------------
# (1) packed wire config travels on the init info
# ---------------------------------------------------------------------------


def test_update_info_carries_parameter_metadata_only():
    """The per-round payload must not repeat the wire format."""
    assert {field.name for field in fields(HCCLWeightTransferUpdateInfo)} == {
        "names",
        "dtype_names",
        "shapes",
    }


def test_update_info_still_validates_list_lengths():
    with pytest.raises(ValueError, match="dtype_names"):
        HCCLWeightTransferUpdateInfo(names=["a"], dtype_names=[], shapes=[[1]])
    with pytest.raises(ValueError, match="shapes"):
        HCCLWeightTransferUpdateInfo(names=["a"], dtype_names=["float32"], shapes=[])


def test_worker_init_info_declares_the_packed_wire_params():
    names = {field.name for field in fields(HCCLWeightTransferInitInfo)}
    assert {"master_address", "master_port", "rank_offset", "world_size"} <= names
    assert {"packed", "packed_buffer_size_bytes", "packed_num_buffers"} <= names


def test_init_transfer_engine_records_wire_params_and_joins_group():
    engine = object.__new__(HCCLWeightTransferEngine)
    engine.parallel_config = MagicMock()
    init_info = HCCLWeightTransferInitInfo(
        master_address=MASTER_ADDRESS,
        master_port=MASTER_PORT,
        rank_offset=1,
        world_size=WORLD_SIZE,
        packed=True,
        packed_buffer_size_bytes=1024,
        packed_num_buffers=3,
    )
    group = MagicMock()

    with patch(f"{_MODULE}.worker_init_process_group", return_value=group) as mock_join:
        engine.init_transfer_engine(init_info)

    assert engine.packed is True
    assert engine.packed_buffer_size_bytes == 1024
    assert engine.packed_num_buffers == 3
    assert engine.model_update_group is group
    mock_join.assert_called_once_with(init_info, engine.parallel_config)


def test_trainer_propagates_the_packed_wire_params_to_the_worker():
    client = MagicMock()
    source = MagicMock()
    init_info = HCCLTrainerInitInfo(
        rank=0,
        master_address=MASTER_ADDRESS,
        master_port=MASTER_PORT,
        world_size=WORLD_SIZE,
        packed=True,
        packed_buffer_size_bytes=8192,
        packed_num_buffers=5,
    )

    with patch(f"{_MODULE}.open_trainer_endpoint", return_value=MagicMock()) as mock_open:
        engine = HCCLTrainerWeightTransferEngine.trainer_init(init_info, client=client, source=source)

    payload = client.init_weight_transfer_engine.call_args.args[0]
    assert payload["packed"] is True
    assert payload["packed_buffer_size_bytes"] == 8192
    assert payload["packed_num_buffers"] == 5
    assert payload["rank_offset"] == 1
    assert payload["world_size"] == WORLD_SIZE
    assert payload["master_address"] == MASTER_ADDRESS
    assert payload["master_port"] == MASTER_PORT
    mock_open.assert_called_once_with(init_info)
    assert engine.packed is True
    assert engine.packed_buffer_size_bytes == 8192
    assert engine.packed_num_buffers == 5


def test_trainer_init_requires_a_weight_source():
    with pytest.raises(ValueError, match="requires a WeightSource"):
        HCCLTrainerWeightTransferEngine.trainer_init(
            HCCLTrainerInitInfo(
                rank=0,
                master_address=MASTER_ADDRESS,
                master_port=MASTER_PORT,
                world_size=WORLD_SIZE,
            ),
            client=MagicMock(),
            source=None,
        )


# ---------------------------------------------------------------------------
# (2) the trainer opens its own endpoint, never the worker engine's privates
# ---------------------------------------------------------------------------


def test_trainer_engine_does_not_touch_the_worker_engine():
    source = inspect.getsource(HCCLTrainerWeightTransferEngine)
    assert "HCCLWeightTransferEngine." not in source
    assert not hasattr(HCCLWeightTransferEngine, "_stateless_init_process_group")
    assert not hasattr(HCCLWeightTransferEngine, "_post_send_sync")


def test_worker_engine_has_no_legacy_trainer_api():
    assert not hasattr(HCCLWeightTransferEngine, "trainer_init")
    assert not hasattr(HCCLWeightTransferEngine, "trainer_send_weights")


def test_hccl_common_worker_init_process_group_ranks_across_dp():
    init_info = HCCLWeightTransferInitInfo(
        master_address=MASTER_ADDRESS,
        master_port=MASTER_PORT,
        rank_offset=1,
        world_size=4,
    )
    parallel_config = MagicMock(data_parallel_index=1, world_size=2, rank=1)
    group = MagicMock()

    with (
        patch.object(hccl_common, "stateless_init_process_group", return_value=group) as mock_pg,
        patch("torch.accelerator.current_device_index", return_value=7),
    ):
        assert hccl_common.worker_init_process_group(init_info, parallel_config) is group

    # rank = dp_rank * world_size_per_dp + rank_within_dp (= 3) + rank_offset (= 1)
    mock_pg.assert_called_once_with(MASTER_ADDRESS, MASTER_PORT, 4, 4, device=7)


def test_hccl_common_trainer_init_opens_rank_zero():
    group = MagicMock()
    init_info = {"master_address": MASTER_ADDRESS, "master_port": MASTER_PORT, "world_size": WORLD_SIZE}

    with (
        patch.object(hccl_common, "stateless_init_process_group", return_value=group) as mock_pg,
        patch("torch.accelerator.current_device_index", return_value=3),
    ):
        assert hccl_common.trainer_init(init_info) is group

    mock_pg.assert_called_once_with(MASTER_ADDRESS, MASTER_PORT, 0, WORLD_SIZE, 3)


# ---------------------------------------------------------------------------
# (3) the receive paths close the MTP completeness check
# ---------------------------------------------------------------------------


def test_receive_packed_uses_the_handshake_config_inside_mtp_context():
    engine = _worker_engine(packed=True, device=torch.device("npu", 0))
    update_info = HCCLWeightTransferUpdateInfo(names=["w"], dtype_names=["float32"], shapes=[[3]])
    mtp = MagicMock()

    with (
        patch(f"{_MODULE}.packed_broadcast_consumer") as mock_consumer,
        patch(_MTP, mtp),
    ):
        engine.receive_weights(update_info)

    mtp.assert_called_once_with()
    mtp.return_value.__enter__.assert_called_once()
    kwargs = mock_consumer.call_args.kwargs
    assert kwargs["group"] is engine.model_update_group
    assert kwargs["src"] == 0
    assert kwargs["buffer_size_bytes"] == engine.packed_buffer_size_bytes
    assert kwargs["num_buffers"] == engine.packed_num_buffers
    assert kwargs["device"] == engine.device
    assert kwargs["post_unpack_func"] is engine.model.load_weights
    # The iterator handed to the consumer is built from the per-round metadata.
    assert list(kwargs["iterator"]) == [("w", ([3], torch.float32))]


def test_receive_unpacked_allocates_on_the_worker_device_inside_mtp_context():
    engine = _worker_engine(packed=False, device=torch.device("cpu"))
    update_info = HCCLWeightTransferUpdateInfo(names=["w"], dtype_names=["float32"], shapes=[[2]])
    stream = MagicMock()
    mtp = MagicMock()

    with (
        patch("torch.npu.current_stream", return_value=stream) as mock_stream,
        patch(_MTP, mtp),
    ):
        engine.receive_weights(update_info)

    mock_stream.assert_called_once_with(engine.device)
    mtp.return_value.__enter__.assert_called_once()

    weight = engine.model_update_group.broadcast.call_args.args[0]
    assert weight.device.type == "cpu"
    assert weight.shape == (2,)
    assert engine.model_update_group.broadcast.call_args.kwargs["stream"] is stream
    loaded = engine.model.load_weights.call_args.args[0]
    assert loaded[0][0] == "w"
    assert loaded[0][1] is weight


def test_receive_weights_requires_an_initialized_group():
    engine = _worker_engine(packed=False)
    engine.model_update_group = None

    with pytest.raises(RuntimeError, match="not initialized"):
        engine.receive_weights(HCCLWeightTransferUpdateInfo(names=[], dtype_names=[], shapes=[]))


# ---------------------------------------------------------------------------
# (4) metadata/iteration consistency
# ---------------------------------------------------------------------------


def test_checked_iter_yields_matching_pairs():
    meta = _meta(("a", torch.float32, (2,)), ("b", torch.bfloat16, (3,)))
    source = [
        ("a", torch.zeros(2, dtype=torch.float32)),
        ("b", torch.zeros(3, dtype=torch.bfloat16)),
    ]

    yielded = list(HCCLTrainerWeightTransferEngine._checked_iter(source, meta))

    assert [name for name, _ in yielded] == ["a", "b"]
    assert [tuple(tensor.shape) for _, tensor in yielded] == [(2,), (3,)]


def test_checked_iter_rejects_a_divergent_parameter():
    meta = _meta(("a", torch.float32, (2,)))
    source = [("a", torch.zeros(3, dtype=torch.float32))]

    with pytest.raises(ValueError, match="disagrees with iteration"):
        list(HCCLTrainerWeightTransferEngine._checked_iter(source, meta))


def test_checked_iter_rejects_count_mismatches():
    meta = _meta(("a", torch.float32, (2,)))
    with pytest.raises(ValueError, match="more parameters than metadata"):
        list(HCCLTrainerWeightTransferEngine._checked_iter([("a", torch.zeros(2)), ("b", torch.zeros(2))], meta))
    with pytest.raises(ValueError, match="waiting for the rest"):
        list(HCCLTrainerWeightTransferEngine._checked_iter([], meta))


# ---------------------------------------------------------------------------
# (5) _post_send_sync is a trainer-side concern
# ---------------------------------------------------------------------------


def test_post_send_sync_is_owned_by_the_trainer_engine():
    assert "_post_send_sync" in HCCLTrainerWeightTransferEngine.__dict__
    assert "_post_send_sync" not in HCCLWeightTransferEngine.__dict__
    assert isinstance(HCCLTrainerWeightTransferEngine, type)
    assert issubclass(HCCLTrainerWeightTransferEngine, TrainerWeightTransferEngine)


def test_trainer_post_send_sync_synchronizes_the_current_stream():
    stream = MagicMock()
    with (
        patch("torch.npu.is_available", return_value=True),
        patch("torch.npu.current_stream", return_value=stream),
    ):
        HCCLTrainerWeightTransferEngine._post_send_sync(MagicMock())

    stream.synchronize.assert_called_once_with()


def test_trainer_post_send_sync_is_a_no_op_without_npu():
    with (
        patch("torch.npu.is_available", return_value=False),
        patch("torch.npu.current_stream") as mock_stream,
    ):
        HCCLTrainerWeightTransferEngine._post_send_sync(MagicMock())

    mock_stream.assert_not_called()
