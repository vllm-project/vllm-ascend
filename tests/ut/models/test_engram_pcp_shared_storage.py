# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute Engram storage/lookup methods without vLLM or NPU initialization.

Tiny CPU buffers stand in for device registrations in the embedding tests.
The shared-memory test executes the real allocator with only CANN and CPU
collectives mocked; it does not establish hardware UVA correctness.
"""

import ast
import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
ENGRAM = ROOT / "vllm_ascend/models/deepseek_v41/engram"
HEAD_SIZES = (4,) * 24


def _definitions(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]


def _execute(path, names, namespace):
    exec(compile(ast.Module(body=_definitions(path, names), type_ignores=[]), str(path), "exec"), namespace)


class _Buffer:
    def __init__(self, tensor):
        self.tensor = tensor
        self.close = Mock()


@pytest.fixture
def embedding():
    state = SimpleNamespace(tp_size=1, tp_rank=0, dp_group=None, backings={}, buffers=[])

    def allocate(shape, dtype, device, group=None):
        # Registering the same backing repeatedly must not allocate another
        # physical table. A different TP storage group owns a distinct shard.
        key = (group.cpu_group, tuple(shape), dtype) if group is not None else object()
        if key not in state.backings:
            state.backings[key] = torch.zeros(shape, dtype=dtype)
        buffer = _Buffer(state.backings[key])
        state.buffers.append(buffer)
        return buffer

    def lookup(codes, scales, indices, **launch):
        heads = launch["local_heads"]
        columns = indices[:, launch["head_start"] : launch["head_start"] + heads].long()
        owned = (columns >= launch["vocab_start"]) & (columns < launch["vocab_end"])
        rows = torch.where(owned, columns - launch["vocab_start"], 0)
        values = codes.tensor[rows]
        if scales is not None:
            values = (values.float().unflatten(-1, (-1, 32)) * scales.tensor[rows].unsqueeze(-1)).flatten(-2)
        values = values.bfloat16()
        launch["output"].view_as(values).copy_(torch.where(owned.unsqueeze(-1), values, 0))

    group_forbidden = Mock(side_effect=AssertionError("shared lookup entered a DP/PCP collective"))
    namespace = {
        "torch": SimpleNamespace(
            **{name: getattr(torch, name) for name in dir(torch) if not name.startswith("__")},
        ),
        "nn": torch.nn,
        "ParallelEngramEmbedding": torch.nn.Module,
        "EngramStorageGroup": object,
        "HostUvaBuffer": _Buffer,
        "SharedUvaBuffer": allocate,
        "get_engram_dp_group": lambda: state.dp_group,
        "get_engram_dp_size": lambda: state.dp_group.world_size if state.dp_group is not None else 1,
        "get_tensor_model_parallel_world_size": lambda: state.tp_size,
        "get_tensor_model_parallel_rank": lambda: state.tp_rank,
        "engram_head_shard_rank": lambda: state.tp_rank * state.dp_group.world_size + state.dp_group.rank_in_group,
        "in_the_same_node_as": lambda group: [True],
        "get_tp_group": lambda: SimpleNamespace(cpu_group="tp-cpu"),
        "upstream_gather_engram_hashes": lambda ids, **kwargs: ids,
        "set_weight_attrs": lambda parameter, attrs: parameter.__dict__.update(attrs),
        "logger": Mock(),
        "cast": lambda typ, value: value,
        "Path": Path,
        "dist": SimpleNamespace(all_gather_object=Mock()),
        "gather_dequantize_host_uva": lookup,
        "_gather_engram_rows": group_forbidden,
        "tensor_model_parallel_all_gather": group_forbidden,
    }
    namespace["torch"].device = lambda *args: torch.device("cpu")
    namespace["torch"].npu = SimpleNamespace(current_device=lambda: 0)
    _execute(ENGRAM / "npu.py", {"EngramBufferInitializationError"}, namespace)
    _execute(ENGRAM / "parallel.py", {"gather_engram_hashes"}, namespace)
    _execute(
        ENGRAM / "embedding.py",
        {"EngramTableInitializationError", "AscendParallelEngramEmbedding", "_torch_lookup"},
        namespace,
    )
    state.cls = namespace["AscendParallelEngramEmbedding"]
    state.namespace = namespace
    state.lookup_collective = group_forbidden
    return state


def _group(size, rank=0, cpu_group=None):
    return SimpleNamespace(
        world_size=size,
        rank_in_group=rank,
        cpu_group=object() if cpu_group is None else cpu_group,
        all_gather=Mock(side_effect=AssertionError("shared storage cannot gather another request's hashes")),
    )


def _table(embedding, storage_group=None, *, storage_dtype=torch.int8):
    return embedding.cls(
        sum(HEAD_SIZES),
        32,
        HEAD_SIZES,
        0,
        cpu_offload=True,
        dp_shared_memory=True,
        storage_group=storage_group,
        storage_dtype=storage_dtype,
    )


def test_tp1_dp2_pcp8_shares_one_full_table_not_sixteen_shards(embedding):
    backing_group = object()
    tables = []
    for rank in range(16):
        embedding.dp_group = _group(2, rank // 8)
        tables.append(_table(embedding, _group(16, rank, backing_group)))
    assert len(embedding.backings) == 2  # One codes segment and one scales segment.
    assert {table.weight.data_ptr() for table in tables} == {tables[0].weight.data_ptr()}
    assert {table.weight_scale_inv.data_ptr() for table in tables} == {tables[0].weight_scale_inv.data_ptr()}
    for table in tables:
        assert table._shared_group.world_size == 16
        assert table.dp_size == 1
        assert (table.head_start, table.part_n_hash_cols) == (0, 24)
        assert (table.vocab_start_idx, table.vocab_end_idx) == (0, 96)


def test_tp8_dp2_preserves_eight_distinct_tp_shards(embedding):
    embedding.tp_size = 8
    ranges = []
    pointers = []
    for tp_rank in range(8):
        embedding.tp_rank = tp_rank
        backing_group = object()
        pair = [_table(embedding, _group(2, dp_rank, backing_group)) for dp_rank in range(2)]
        assert pair[0].weight.data_ptr() == pair[1].weight.data_ptr()
        assert pair[0].weight_scale_inv.data_ptr() == pair[1].weight_scale_inv.data_ptr()
        assert pair[0].part_n_hash_cols == pair[1].part_n_hash_cols == 3
        ranges.append((pair[0].vocab_start_idx, pair[0].vocab_end_idx))
        pointers.append(pair[0].weight.data_ptr())
    assert len(set(pointers)) == 8
    assert ranges == [(start, start + 12) for start in range(0, 96, 12)]
    assert len(embedding.backings) == 16


def test_legacy_dp_only_callers_keep_shared_backing(embedding):
    embedding.dp_group = _group(2)
    table = _table(embedding)
    assert table._shared_group is embedding.dp_group
    assert table.part_n_hash_cols == 24


def test_cross_node_storage_is_rejected_before_allocation(embedding):
    embedding.namespace["in_the_same_node_as"] = lambda group: [True, group == "tp-cpu"]
    with pytest.raises(ValueError, match="same node and shared-memory namespace"):
        _table(embedding, _group(16))
    assert embedding.buffers == []


def test_shared_lookup_keeps_different_pcp_and_dp_requests_local(embedding):
    table = _table(embedding, _group(16))
    table.weight.data.copy_(torch.arange(96, dtype=torch.int8)[:, None].expand(96, 32))
    table.weight_scale_inv.data.fill_(1)
    heads = torch.arange(24, dtype=torch.int32) * 4
    for token_count, marker in [(3, 0), (1, 2), (0, 1), (2, 3)]:
        ids = (heads + marker).expand(token_count, 24).contiguous()
        actual = table.forward(ids)
        expected = ids.bfloat16().unsqueeze(-1).expand(token_count, 24, 32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    embedding.lookup_collective.assert_not_called()


def test_only_one_rank_writes_the_shared_checkpoint(embedding):
    backing_group = object()
    tables = [_table(embedding, _group(16, rank, backing_group)) for rank in range(16)]
    writers = []
    for rank, table in enumerate(tables):

        def load(model_path, key, chunk_rows, *, rank=rank, table=table):
            writers.append(rank)
            table.weight.data.fill_(7)
            table.weight_scale_inv.data.fill_(2)

        table._load_into_storage = load
        table.load_checkpoint("checkpoint", "layers.1.engram.embed.weight")
    assert writers == [0]
    assert all(torch.all(table.weight == 7) and torch.all(table.weight_scale_inv == 2) for table in tables)
    assert embedding.namespace["dist"].all_gather_object.call_count == 16
    assert all(
        call.kwargs["group"] is backing_group for call in embedding.namespace["dist"].all_gather_object.call_args_list
    )


def test_checkpoint_error_reaches_every_storage_member(embedding):
    failure = "OSError: checkpoint read failed"

    def gather(errors, local_error, group):
        errors[0] = failure

    embedding.namespace["dist"].all_gather_object.side_effect = gather
    backing_group = object()
    writers = Mock(side_effect=OSError("checkpoint read failed"))
    for rank in range(16):
        table = _table(embedding, _group(16, rank, backing_group))
        table._load_into_storage = writers
        with pytest.raises(RuntimeError, match="rank 0: OSError: checkpoint read failed"):
            table.load_checkpoint("checkpoint", "weight")
    writers.assert_called_once()


def test_failed_scales_allocation_releases_codes(embedding):
    allocate = embedding.namespace["SharedUvaBuffer"]
    codes = []

    def failing_allocate(shape, dtype, device, group):
        if dtype == torch.float32:
            raise RuntimeError("scale registration failed")
        buffer = allocate(shape, dtype, device, group)
        codes.append(buffer)
        return buffer

    embedding.namespace["SharedUvaBuffer"] = failing_allocate
    with pytest.raises(RuntimeError, match="scale registration failed"):
        _table(embedding, _group(16))
    codes[0].close.assert_called_once()


def test_failed_allocation_cleanup_retains_table_until_retry(embedding):
    allocate = embedding.namespace["SharedUvaBuffer"]
    codes = []

    def failing_allocate(shape, dtype, device, group):
        if dtype == torch.float32:
            raise RuntimeError("scale registration failed")
        buffer = allocate(shape, dtype, device, group)
        buffer.close.side_effect = [RuntimeError("unregister failed"), None]
        codes.append(buffer)
        return buffer

    embedding.namespace["SharedUvaBuffer"] = failing_allocate
    error_type = embedding.namespace["EngramTableInitializationError"]
    with pytest.raises(error_type, match="Engram table allocation cleanup failed") as info:
        _table(embedding, _group(16))
    table = info.value.table
    assert table._codes_uva is codes[0]
    assert table._scales_uva is None
    assert not hasattr(table, "weight")
    table.close_host_offload()
    assert table._codes_uva is None
    assert codes[0].close.call_count == 2


@pytest.mark.parametrize("storage_dtype", [torch.int8, torch.bfloat16])
def test_failed_buffer_constructor_is_retained_in_failed_table(embedding, storage_dtype):
    buffer = _Buffer(torch.zeros((96, 32), dtype=storage_dtype))
    buffer.close.side_effect = [RuntimeError("unregister failed"), None]
    buffer_error_type = embedding.namespace["EngramBufferInitializationError"]
    embedding.namespace["SharedUvaBuffer"] = Mock(side_effect=buffer_error_type(buffer, "registration cleanup failed"))
    with pytest.raises(embedding.namespace["EngramTableInitializationError"]) as info:
        _table(embedding, _group(16), storage_dtype=storage_dtype)
    table = info.value.table
    assert table._codes_uva is buffer
    table.close_host_offload()
    assert table._codes_uva is None
    assert buffer.close.call_count == 2


def test_failed_unregister_preserves_aliases_and_can_be_retried(embedding):
    table = _table(embedding, _group(16))
    codes, scales = table._codes_uva, table._scales_uva
    weight = table.weight
    codes.close.side_effect = [RuntimeError("unregister failed"), None]
    with pytest.raises(RuntimeError, match="unregister failed"):
        table.close_host_offload()
    assert table._codes_uva is codes
    assert table.weight is weight
    scales.close.assert_not_called()
    table.close_host_offload()
    assert table._codes_uva is table._scales_uva is None
    assert table.weight.numel() == table.weight_scale_inv.numel() == 0
    assert codes.close.call_count == 2
    scales.close.assert_called_once()
    table.close_host_offload()
    scales.close.assert_called_once()


def _model_close(model):
    path = ROOT / "vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Model")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "close_engram_storage"
    )
    namespace = {}
    _execute(path, {"EngramStorageCleanupError"}, namespace)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    namespace["close_engram_storage"](model)


def test_model_releases_both_tables_before_storage_group(embedding):
    events = []
    group = _group(16)
    group.close = Mock(side_effect=lambda: events.append("group"))
    tables = [_table(embedding, group) for layer in range(2)]
    for index, table in enumerate(tables):
        table._codes_uva.close.side_effect = lambda index=index: events.append(f"codes-{index}")
        table._scales_uva.close.side_effect = lambda index=index: events.append(f"scales-{index}")
    model = SimpleNamespace(_engram_tables=tables, _engram_storage_group=group)
    _model_close(model)
    assert events == ["codes-0", "scales-0", "codes-1", "scales-1", "group"]
    assert model._engram_tables == []
    assert model._engram_storage_group is None
    _model_close(model)
    group.close.assert_called_once()


def test_model_cleanup_attempts_other_tables_and_keeps_group_for_retry(embedding):
    group = _group(16)
    group.close = Mock()
    first, second = [_table(embedding, group) for layer in range(2)]
    failing_buffer = first._codes_uva
    successful_buffer = second._codes_uva
    failing_buffer.close.side_effect = [RuntimeError("unregister failed"), None]
    model = SimpleNamespace(_engram_tables=[first, second], _engram_storage_group=group)
    with pytest.raises(RuntimeError, match="Engram storage cleanup failed: RuntimeError: unregister failed") as info:
        _model_close(model)
    assert info.value.owner is model
    successful_buffer.close.assert_called_once()
    assert first._codes_uva is failing_buffer
    assert second._codes_uva is second._scales_uva is None
    assert model._engram_storage_group is group
    group.close.assert_not_called()
    _model_close(model)
    assert failing_buffer.close.call_count == 2
    successful_buffer.close.assert_called_once()
    group.close.assert_called_once()


def test_storage_group_close_failure_is_retryable():
    group = SimpleNamespace(close=Mock(side_effect=[RuntimeError("group destroy failed"), None]))
    model = SimpleNamespace(_engram_tables=[], _engram_storage_group=group)
    with pytest.raises(RuntimeError, match="Engram storage group cleanup failed") as info:
        _model_close(model)
    assert info.value.owner is model
    assert str(info.value.__cause__) == "group destroy failed"
    assert model._engram_storage_group is group
    _model_close(model)
    assert model._engram_storage_group is None
    assert group.close.call_count == 2


@pytest.mark.parametrize(
    "failure_rank,pointer_failure_rank,unregister_failure_rank", [(None, None, None), (5, None, None), (None, 5, 5)]
)
def test_real_shared_allocator_maps_one_segment_to_sixteen_members(
    failure_rank, pointer_failure_rank, unregister_failure_rank
):
    """Check shared backing, collective failure propagation and cleanup retry.

    Sixteen threads attach real POSIX memory with mocked CANN registrations
    and device addresses; this does not establish native UVA readability.
    """
    members = 16
    local = threading.local()
    barrier = threading.Barrier(members)
    slots = {}
    creations = []
    registrations = []
    unregistrations = []
    errors_by_rank = [None] * members
    retry_cleanup = False

    def broadcast(payload, src, group):
        if local.rank == 0:
            slots["name"] = payload[0]
            creations.append(payload[0])
        barrier.wait(timeout=20)
        payload[0] = slots["name"]
        barrier.wait(timeout=20)

    def gather(errors, error, group):
        errors_by_rank[local.rank] = error
        barrier.wait(timeout=20)
        errors[:] = errors_by_rank
        barrier.wait(timeout=20)

    class Library:
        def __init__(self):
            self.rank = local.rank

        def aclrtHostRegisterV2(self, pointer, size, flags):
            registrations.append((local.rank, size))
            return 207001 if local.rank == failure_rank else 0

        def aclrtHostGetDevicePointer(self, pointer, out, flags):
            if self.rank == pointer_failure_rank:
                return 207001
            out._obj.value = (1 << 40) + local.rank * 4096
            return 0

        def aclrtHostUnregister(self, pointer):
            unregistrations.append(pointer.value)
            return 207001 if self.rank == unregister_failure_rank and not retry_cleanup else 0

    namespace = {
        "torch": torch,
        "ctypes": ctypes,
        "shared_memory": shared_memory,
        "patch": patch,
        "_host_library": Library,
        "CHUNK_ROWS": 1 << 22,
        "MAX_SINGLE_REGISTRATION_BYTES": 32 * 1024**3,
        "ACL_HOST_REG_MAPPED": 2,
        "ACL_HOST_REG_PINNED": 0x10000000,
        "logger": Mock(),
        "dist": SimpleNamespace(
            get_global_rank=lambda group, rank: 0,
            broadcast_object_list=broadcast,
            all_gather_object=gather,
            barrier=lambda group: barrier.wait(timeout=20),
        ),
    }
    _execute(ENGRAM / "npu.py", {"EngramBufferInitializationError", "SharedUvaBuffer"}, namespace)

    def attach(rank):
        local.rank = rank
        try:
            return namespace["SharedUvaBuffer"]((8, 32), torch.int8, "cpu", _group(members, rank))
        except RuntimeError as exc:
            return exc

    buffers = []
    try:
        with ThreadPoolExecutor(max_workers=members) as pool:
            buffers = list(pool.map(attach, range(members)))
        assert len(creations) == 1
        assert len(registrations) == members
        assert {size for rank, size in registrations} == {8 * 32}
        if failure_rank is not None:
            assert all(isinstance(error, RuntimeError) for error in buffers)
            assert all(
                f"rank {failure_rank}: RuntimeError: aclrtHostRegisterV2 failed: rc=207001" in str(error)
                for error in buffers
            )
            assert len(unregistrations) == members - 1
            return
        if pointer_failure_rank is not None:
            assert all(isinstance(error, RuntimeError) for error in buffers)
            assert all("aclrtHostGetDevicePointer failed: rc=207001" in str(error) for error in buffers)
            error = buffers[unregister_failure_rank]
            assert isinstance(error, namespace["EngramBufferInitializationError"])
            assert error.buffer.shm is not None
            assert error.buffer.pointer.value
            assert len(unregistrations) == members
            retry_cleanup = True
            error.buffer.close()
            assert error.buffer.shm is None
            assert not error.buffer.pointer.value
            assert len(unregistrations) == members + 1
            return
        assert len({int(buffer.ptrs[0]) for buffer in buffers}) == members
        buffers[0].tensor.fill_(19)
        assert all(torch.all(buffer.tensor == 19) for buffer in buffers)
    finally:
        retry_cleanup = True
        for buffer in buffers:
            if hasattr(buffer, "buffer"):
                buffer.buffer.close()
            elif not isinstance(buffer, Exception):
                buffer.close()


def _worker_close(worker):
    path = ROOT / "vllm_ascend/worker/worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_close_engram_storage"
    )
    namespace = {"nn": torch.nn}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    namespace["_close_engram_storage"](worker)


def test_worker_unwraps_loaded_model_for_storage_cleanup():
    backbone = torch.nn.Module()
    backbone.close_engram_storage = Mock()
    model = torch.nn.Sequential(backbone)
    runner = SimpleNamespace(model=object(), get_model=Mock(return_value=model))
    _worker_close(SimpleNamespace(model_runner=runner))
    runner.get_model.assert_called_once()
    backbone.close_engram_storage.assert_called_once()


@pytest.mark.parametrize("runner", [None, SimpleNamespace(), SimpleNamespace(model=None)])
def test_worker_cleanup_before_model_load_is_safe(runner):
    if runner is not None:
        runner.get_model = Mock(side_effect=AssertionError("unloaded runner tried to unwrap a model"))
    _worker_close(SimpleNamespace(model_runner=runner))


def test_worker_retains_model_owner_when_cleanup_fails():
    path = ROOT / "vllm_ascend/models/deepseek_v41/model.py"
    namespace = {}
    _execute(path, {"EngramStorageCleanupError"}, namespace)
    owner = torch.nn.Module()
    owner.close_engram_storage = Mock(side_effect=namespace["EngramStorageCleanupError"](owner, "unregister failed"))
    runner = SimpleNamespace(model=owner, get_model=lambda: owner)
    with pytest.raises(RuntimeError, match="unregister failed") as info:
        _worker_close(SimpleNamespace(model_runner=runner))
    assert info.value.owner is owner


def test_bf16_shared_table_preserves_rows_without_allocating_scales(embedding):
    backing_group = object()
    tables = [_table(embedding, _group(16, rank, backing_group), storage_dtype=torch.bfloat16) for rank in range(16)]
    assert len(embedding.backings) == 1
    assert {table.weight.data_ptr() for table in tables} == {tables[0].weight.data_ptr()}
    assert all(table.weight_scale_inv is None and table._scales_uva is None for table in tables)
    tables[0].weight.data.copy_(torch.arange(96).bfloat16()[:, None].expand(96, 32) + 0.5)
    heads = torch.arange(24, dtype=torch.int32) * 4
    for table, (tokens, marker) in zip(tables, [(3, 0), (1, 2), (0, 1), (2, 3)]):
        ids = (heads + marker).expand(tokens, 24).contiguous()
        expected = (ids.bfloat16() + 0.5).unsqueeze(-1).expand(tokens, 24, 32)
        torch.testing.assert_close(table.forward(ids), expected, rtol=0, atol=0)
    embedding.lookup_collective.assert_not_called()


def test_bf16_shared_unregister_failure_retains_weight_until_retry(embedding):
    table = _table(embedding, _group(16), storage_dtype=torch.bfloat16)
    codes, weight = table._codes_uva, table.weight
    codes.close.side_effect = [RuntimeError("unregister failed"), None]
    with pytest.raises(RuntimeError, match="unregister failed"):
        table.close_host_offload()
    assert table._codes_uva is codes and table.weight is weight
    assert table.weight_scale_inv is None
    table.close_host_offload()
    assert table._codes_uva is None and table.weight_scale_inv is None
    assert table.weight.dtype == torch.bfloat16 and table.weight.numel() == 0
    table.close_host_offload()
    assert codes.close.call_count == 2
