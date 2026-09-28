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
"""Regression tests for the NPU IPC weight transfer engine.

These cover the bugs that broke ``examples/rl/rlhf_http_npu_ipc.py``:

1. ``NPUIPCWeightTransferEngine.__init__`` did not accept the ``model``
   argument that ``WeightTransferEngineFactory.create_engine`` passes,
   raising ``TypeError: __init__() takes 3 positional arguments but 4
   were given`` at engine construction.
2. ``receive_weights`` / ``packed_npu_ipc_consumer`` unpacked the stored
   IPC handle as ``func, args`` even though the producer stored only the
   ``reduce_tensor`` *args*, raising ``ValueError: too many values to
   unpack (expected 2)``. Aligned with upstream vLLM's CUDA IPC engine:
   the producer stores args only and the consumer rebuilds with the
   well-known ``rebuild_npu_tensor``.
3. The NPU IPC worker skipped the layerwise reload START/FINISH lifecycle,
   so runtime-formatted weights were not restored before loading and graph-
   visible storage was not preserved after loading.
4. The packed receive path decoded tensors but never passed them to
   ``model.load_weights``.
"""

import inspect
import sys
import threading
import types
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.distributed.weight_transfer import npu_ipc_engine
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
    NPUIPCWeightTransferEngine,
)

_MODULE = "vllm_ascend.distributed.weight_transfer.npu_ipc_engine"


def _patch_rebuild_npu_tensor(rebuild_func):
    """Install a fake ``torch_npu.multiprocessing.reductions`` module.

    The engine imports ``rebuild_npu_tensor`` lazily from ``torch_npu``,
    which is only a stub on CPU CI runners, so provide a fake submodule.
    """
    fake_mod = types.ModuleType("torch_npu.multiprocessing.reductions")
    fake_mod.rebuild_npu_tensor = rebuild_func  # type: ignore[attr-defined]
    return patch.dict(
        sys.modules,
        {
            "torch_npu.multiprocessing": types.ModuleType("torch_npu.multiprocessing"),
            "torch_npu.multiprocessing.reductions": fake_mod,
        },
    )


def _patch_reload_module(*, initialize=None, finalize=None):
    """Provide the lazy-imported reload helpers without importing vLLM models."""
    fake_mod = types.ModuleType("vllm.model_executor.model_loader.reload")
    fake_mod.initialize_layerwise_reload = initialize or MagicMock()  # type: ignore[attr-defined]
    fake_mod.finalize_layerwise_reload = finalize or MagicMock()  # type: ignore[attr-defined]
    return patch.dict(
        sys.modules,
        {"vllm.model_executor.model_loader.reload": fake_mod},
    )


def test_init_accepts_model_argument():
    """Bug 1: __init__ must accept the optional ``model`` argument."""
    params = inspect.signature(NPUIPCWeightTransferEngine.__init__).parameters
    assert "model" in params


def test_init_passes_model_to_super():
    captured: dict = {}

    def fake_init_v1(self, config, vllm_config, device, model):
        captured["args"] = (config, vllm_config, device, model)

    with patch.object(npu_ipc_engine.WeightTransferEngine, "__init__", fake_init_v1):
        device = torch.device("npu:0")
        NPUIPCWeightTransferEngine("config", "vllm_config", device, "model")

    assert captured["args"] == ("config", "vllm_config", device, "model")


def test_unpacked_send_stores_reduce_tensor_args_only():
    """Bug 2 (producer): the handle stores only the ``reduce_tensor`` args.

    This matches upstream vLLM's CUDA IPC engine, which drops the rebuild
    func and relies on the consumer using the well-known rebuild function.
    """
    rebuild_args = (None, None, None, None, None, None, 999, None)
    fake_reduce = MagicMock(return_value=("rebuild_func_sentinel", rebuild_args))

    captured = {}

    with patch(f"{_MODULE}.reduce_tensor", fake_reduce):
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
            NPUIPCTrainerWeightTransferEngine,
        )

        engine = object.__new__(NPUIPCTrainerWeightTransferEngine)
        engine.client = MagicMock()
        engine.is_sender = True
        engine.npu_uuid = "node-0"
        engine.packed_buffer_size_bytes = 1024
        engine._do_send = lambda **kw: captured.update(kw)
        engine._all_gather_and_merge_handles = lambda x, **_: x
        engine._post_send_sync = MagicMock()

        source = iter([("model.weight", torch.zeros(3))])
        engine._send_unpacked(source)

        stored = captured["ipc_handles"][0]["node-0"]

        # Only the args tuple is stored, not a (func, args) pair.
        assert stored == rebuild_args


def _make_unpacked_trainer(*, buffer_size: int):
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
        NPUIPCTrainerWeightTransferEngine,
    )

    engine = object.__new__(NPUIPCTrainerWeightTransferEngine)
    engine.client = MagicMock()
    engine.is_sender = True
    engine.npu_uuid = "node-0"
    engine.packed_buffer_size_bytes = buffer_size
    engine._post_send_sync = MagicMock()
    return engine


def _make_stateful_trainer(*, is_sender: bool):
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
        NPUIPCTrainerWeightTransferEngine,
    )

    client = MagicMock()
    with (
        patch.object(torch.accelerator, "current_device_index", return_value=0),
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-0"),
    ):
        engine = NPUIPCTrainerWeightTransferEngine(
            client=client,
            source=[],
            is_sender=is_sender,
            packed=False,
            packed_buffer_size_bytes=16,
        )
    return engine


class _ThreadedAllGather:
    """Two-party object collective used to exercise real rank coordination."""

    def __init__(self, parties: int = 2):
        self.parties = parties
        self.condition = threading.Condition()
        self.round = 0
        self.pending: dict[int, list[tuple[list, dict]]] = {}

    def __call__(self, output: list, payload: dict) -> None:
        with self.condition:
            current_round = self.round
            entries = self.pending.setdefault(current_round, [])
            entries.append((output, payload))
            if len(entries) == self.parties:
                values = [item for _, item in entries]
                for target, _ in entries:
                    target[:] = values
                del self.pending[current_round]
                self.round += 1
                self.condition.notify_all()
                return
            if not self.condition.wait_for(
                lambda: self.round > current_round,
                timeout=5,
            ):
                raise TimeoutError(f"collective round {current_round} timed out")


def _send_update(engine, index: int) -> None:
    engine._do_send(
        names=[f"w{index}"],
        dtype_names=["float32"],
        shapes=[[1]],
        ipc_handles=[{}],
    )
    engine._post_send_sync()


@pytest.mark.parametrize(
    ("failure_point", "update_count"),
    [
        ("start", 0),
        ("first_update", 1),
        ("second_update", 2),
        ("finish", 0),
    ],
)
def test_sender_rpc_failure_reaches_every_rank(failure_point, update_count):
    sender = _make_stateful_trainer(is_sender=True)
    non_sender = _make_stateful_trainer(is_sender=False)

    if failure_point == "start":
        sender.client.start_weight_update.side_effect = ValueError("start rejected")
    elif failure_point == "first_update":
        sender.client.update_weights.side_effect = ValueError("update rejected")
    elif failure_point == "second_update":
        sender.client.update_weights.side_effect = [
            None,
            ValueError("update rejected"),
        ]
    else:
        sender.client.finish_weight_update.side_effect = ValueError("finish rejected")

    for engine in (sender, non_sender):
        engine._send = lambda _, engine=engine: [_send_update(engine, index) for index in range(update_count)]

    all_gather = _ThreadedAllGather()
    barrier = threading.Barrier(2)
    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "all_gather_object", side_effect=all_gather),
        patch.object(torch.distributed, "barrier", side_effect=barrier.wait),
        patch.object(torch.npu, "synchronize"),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        futures = [pool.submit(engine.send_weights) for engine in (sender, non_sender)]
        errors = []
        for future in futures:
            with pytest.raises(RuntimeError) as exc_info:
                future.result(timeout=5)
            errors.append(str(exc_info.value))

    assert errors[0] == errors[1]
    assert failure_point.split("_")[-1] in errors[0]

    rpc_counts = (
        sender.client.start_weight_update.call_count,
        sender.client.update_weights.call_count,
        sender.client.finish_weight_update.call_count,
    )
    for engine in (sender, non_sender):
        with pytest.raises(RuntimeError, match="cannot be reused"):
            engine.send_weights()
    assert rpc_counts == (
        sender.client.start_weight_update.call_count,
        sender.client.update_weights.call_count,
        sender.client.finish_weight_update.call_count,
    )


def test_source_failure_marks_engine_failed_before_retry():
    engine = _make_stateful_trainer(is_sender=True)
    engine._send = MagicMock(side_effect=ValueError("source failed"))

    with (
        patch.object(torch.distributed, "is_initialized", return_value=False),
        pytest.raises(ValueError, match="source failed"),
    ):
        engine.send_weights()

    with pytest.raises(RuntimeError, match="cannot be reused"):
        engine.send_weights()
    engine.client.start_weight_update.assert_called_once()


@pytest.mark.parametrize("failure_stage", ["source", "export"])
def test_unpacked_preparation_failure_reaches_every_rank(failure_stage):
    engines = [
        _make_unpacked_trainer(buffer_size=16),
        _make_unpacked_trainer(buffer_size=16),
    ]

    class FailingSource:
        def __iter__(self):
            return self

        def __next__(self):
            raise ValueError("source preparation failed")

    sources = [
        iter([("w", torch.zeros(1))]),
        FailingSource() if failure_stage == "source" else iter([("w", torch.ones(1))]),
    ]

    def fake_reduce(tensor):
        if failure_stage == "export" and tensor.item() == 1:
            raise ValueError("handle export failed")
        return None, ("args",)

    all_gather = _ThreadedAllGather()
    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(
            torch.distributed,
            "all_gather_object",
            side_effect=all_gather,
        ),
        patch(f"{_MODULE}.reduce_tensor", side_effect=fake_reduce),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        futures = [pool.submit(engine._send_unpacked, source) for engine, source in zip(engines, sources)]
        errors = []
        for future in futures:
            with pytest.raises(RuntimeError) as exc_info:
                future.result(timeout=5)
            errors.append(str(exc_info.value))

    assert errors[0] == errors[1]
    assert "trainer rank " in errors[0]
    assert ("source preparation failed" if failure_stage == "source" else "handle export failed") in errors[0]
    assert all(engine._state.name == "FAILED" for engine in engines)


def test_packed_iterator_initialization_failure_reaches_every_rank():
    engines = [
        _make_stateful_trainer(is_sender=True),
        _make_stateful_trainer(is_sender=False),
    ]
    for engine in engines:
        engine.packed = True

    class FailingSource:
        def __iter__(self):
            raise ValueError("packed iterator initialization failed")

    sources = [[], FailingSource()]
    all_gather = _ThreadedAllGather()
    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(
            torch.distributed,
            "all_gather_object",
            side_effect=all_gather,
        ),
        patch(f"{_MODULE}.packed_npu_ipc_producer", return_value=iter(())),
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        futures = [pool.submit(engine._send_packed, source) for engine, source in zip(engines, sources)]
        errors = []
        for future in futures:
            with pytest.raises(RuntimeError) as exc_info:
                future.result(timeout=5)
            errors.append(str(exc_info.value))

    assert errors[0] == errors[1]
    assert "packed iterator initialization failed" in errors[0]


def test_unpacked_releases_completed_chunk_before_materializing_next():
    engine = _make_unpacked_trainer(buffer_size=8)
    engine._all_gather_and_merge_handles = lambda handles, **_: handles
    engine._do_send = MagicMock()

    live_refs = []
    allocation_peaks = []

    class LazyTensor:
        dtype = torch.float32
        shape = (1,)

        @staticmethod
        def numel():
            return 1

        @staticmethod
        def element_size():
            return 4

        def detach(self):
            return self

        def contiguous(self):
            return self

    def source():
        for index in range(10):
            tensor = LazyTensor()
            live_refs.append(weakref.ref(tensor))
            allocation_peaks.append(sum(ref() is not None for ref in live_refs))
            yield f"w{index}", tensor

    def reduce_without_retaining(_tensor):
        return None, ("args",)

    with patch(
        f"{_MODULE}.reduce_tensor",
        new=reduce_without_retaining,
    ):
        engine._send_unpacked(source())

    assert max(allocation_peaks) <= 3
    assert engine._do_send.call_count == 5


def test_normal_engine_supports_consecutive_rounds_and_shutdown():
    engine = _make_stateful_trainer(is_sender=True)
    engine._send = MagicMock(return_value=[])

    with (
        patch.object(torch.distributed, "is_initialized", return_value=False),
        patch.object(torch.npu, "synchronize"),
    ):
        engine.send_weights()
        engine.send_weights()

    assert engine.client.start_weight_update.call_count == 2
    assert engine.client.finish_weight_update.call_count == 2
    engine.shutdown()
    engine.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        engine.send_weights()


def test_unpacked_send_uses_bounded_chunks_and_end_marker():
    engine = _make_unpacked_trainer(buffer_size=16)
    sends = []
    collectives = []
    engine._do_send = lambda **kwargs: sends.append(kwargs)

    def gather(handles, **kwargs):
        collectives.append((handles, kwargs))
        return handles

    engine._all_gather_and_merge_handles = gather
    source = iter(
        [
            ("a", torch.zeros(2, dtype=torch.float32)),
            ("b", torch.ones(2, dtype=torch.float32)),
            ("c", torch.full((2,), 2.0, dtype=torch.float32)),
        ]
    )

    with patch(f"{_MODULE}.reduce_tensor", return_value=(None, ("args",))):
        refs = engine._send_unpacked(source)

    assert refs == []
    assert [send["names"] for send in sends] == [["a", "b"], ["c"]]
    assert engine._post_send_sync.call_count == 2
    assert collectives[0][1]["chunk_index"] == 0
    assert collectives[0][1]["schema"] == [
        ("a", "float32", (2,), 8),
        ("b", "float32", (2,), 8),
    ]
    assert collectives[1][1]["chunk_index"] == 1
    assert len(collectives) == 2


def test_unpacked_send_allows_one_tensor_larger_than_budget():
    engine = _make_unpacked_trainer(buffer_size=4)
    engine._do_send = MagicMock()
    engine._all_gather_and_merge_handles = lambda handles, **_: handles

    with (
        patch(f"{_MODULE}.reduce_tensor", return_value=(None, ("args",))),
        pytest.warns(UserWarning, match="sending it as one chunk"),
    ):
        engine._send_unpacked(iter([("large", torch.zeros(2))]))

    engine._do_send.assert_called_once()


def test_handle_gather_rejects_same_length_different_schema():
    engine = _make_unpacked_trainer(buffer_size=16)
    local_handles = [{"node-0": ("local",)}]
    local_schema = [("a", "float32", (1,), 4)]

    def gather(output, payload):
        output[:] = [
            payload,
            {
                "chunk_index": 0,
                "done": False,
                "schema": [("b", "float32", (1,), 4)],
                "handles": [{"node-1": ("remote",)}],
            },
        ]

    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        pytest.raises(ValueError, match="chunk schema mismatch"),
    ):
        engine._all_gather_and_merge_handles(
            local_handles,
            schema=local_schema,
            chunk_index=0,
        )


def test_handle_gather_rejects_data_end_mismatch():
    engine = _make_unpacked_trainer(buffer_size=16)

    def gather(output, payload):
        output[:] = [
            payload,
            {
                "chunk_index": 0,
                "done": True,
                "schema": [],
                "handles": [],
            },
        ]

    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        pytest.raises(ValueError, match="chunk schema mismatch"),
    ):
        engine._all_gather_and_merge_handles(
            [{"node-0": ("local",)}],
            schema=[("a", "float32", (1,), 4)],
            chunk_index=0,
        )


def test_handle_gather_merges_matching_rank_handles():
    engine = _make_unpacked_trainer(buffer_size=16)
    schema = [("a", "float32", (1,), 4)]

    def gather(output, payload):
        output[:] = [
            payload,
            {
                "chunk_index": 0,
                "done": False,
                "schema": schema,
                "handles": [{"node-1": ("remote",)}],
            },
        ]

    with (
        patch.object(torch.distributed, "is_initialized", return_value=True),
        patch.object(torch.distributed, "get_world_size", return_value=2),
        patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        patch.object(torch.distributed, "barrier"),
        patch.object(torch.npu, "synchronize"),
    ):
        merged = engine._all_gather_and_merge_handles(
            [{"node-0": ("local",)}],
            schema=schema,
            chunk_index=0,
        )

    assert merged == [{"node-0": ("local",), "node-1": ("remote",)}]


def test_receive_weights_rebuilds_with_rebuild_npu_tensor():
    """Bug 2 (consumer): receive_weights rebuilds via ``rebuild_npu_tensor``.

    Verifies the args-only handle is consumed without unpacking errors and
    that the receiver's device index is written into the rebuild args.
    """
    npu_uuid = "node-0"
    device_index = 0

    rebuilt_weight = torch.tensor([1.0, 2.0, 3.0])
    seen = {}

    def fake_rebuild(*args):
        seen["args"] = args
        return rebuilt_weight

    # Sender stores 999 at index 6; the receiver must overwrite it.
    rebuild_args = (None, None, None, None, None, None, 999, None)

    kwargs = dict(
        names=["model.weight"],
        dtype_names=["float32"],
        shapes=[[3]],
        ipc_handles=[{npu_uuid: rebuild_args}],
    )

    update_info = NPUIPCWeightTransferEngine.update_info_cls(**kwargs)

    engine = object.__new__(NPUIPCWeightTransferEngine)
    received: dict[str, list[tuple[str, torch.Tensor]]] = {}
    engine.model = MagicMock()
    engine.device = MagicMock(index=device_index)
    engine.packed = False
    engine.model.load_weights.side_effect = lambda weights: received.update(weights=weights)

    with (
        _patch_rebuild_npu_tensor(fake_rebuild),
        patch(f"{_MODULE}.npu_generate_uuid", return_value=npu_uuid) as mock_uuid,
    ):
        engine.receive_weights(update_info)

    mock_uuid.assert_called_once_with()
    engine.model.load_weights.assert_called_once()
    assert received["weights"][0][0] == "model.weight"
    assert torch.equal(received["weights"][0][1], rebuilt_weight)
    # Index 6 (device index) overwritten with the receiver's device.
    assert seen["args"][6] == device_index


def test_start_weight_update():
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    mock_init = MagicMock()

    with _patch_reload_module(initialize=mock_init):
        engine.start_weight_update()

    mock_init.assert_called_once_with(engine.model)


def test_finish_weight_update():
    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    engine.model_config = MagicMock()
    mock_finalize = MagicMock()

    with _patch_reload_module(finalize=mock_finalize):
        engine.finish_weight_update()

    mock_finalize.assert_called_once_with(engine.model, engine.model_config)


def test_receive_packed_weights_loads_model():
    packed_weights = [("model.weight", torch.tensor([1.0, 2.0, 3.0]))]
    update_info = MagicMock(
        tensor_sizes=[12],
        ipc_handles={"node-0": ("packed-handle",)},
        names=["model.weight"],
        shapes=[[3]],
        dtype_names=["float32"],
    )

    engine = object.__new__(NPUIPCWeightTransferEngine)
    engine.model = MagicMock()
    engine.device = MagicMock(index=0)
    engine.packed = True

    with (
        patch(f"{_MODULE}.npu_generate_uuid", return_value="node-0"),
        patch(
            f"{_MODULE}.packed_npu_ipc_consumer",
            return_value=packed_weights,
        ) as mock_consumer,
    ):
        engine.receive_weights(update_info)

    mock_consumer.assert_called_once_with(
        ipc_handle=update_info.ipc_handles,
        physical_npu_id="node-0",
        names=update_info.names,
        shapes=update_info.shapes,
        dtype_names=update_info.dtype_names,
        tensor_sizes=update_info.tensor_sizes,
        device_index=0,
    )
    engine.model.load_weights.assert_called_once_with(packed_weights)
