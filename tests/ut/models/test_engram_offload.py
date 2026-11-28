# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import ctypes
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.ascend_config import AscendConfig, EngramOffloadConfig, SparseKVOffloadConfig
from vllm_ascend.models.deepseek_v41.engram.embedding import AscendParallelEngramEmbedding
from vllm_ascend.models.deepseek_v41.engram.offload import EngramOffloadPool, make_offload_tables
from vllm_ascend.models.deepseek_v41.engram.shared_pool import checkpoint_identity


def test_global_layout_independent_of_tp_and_zero_capacity_readers():
    tables, stride = make_offload_tables([19, 23], 2)
    assert [table.offset for table in tables] == [1, 11]
    assert stride == 23
    for table in tables:
        covered = [row for owner in range(2) for row in range(*table.owner_range(owner))]
        assert covered == list(range(table.rows))
    pool = object.__new__(EngramOffloadPool)
    pool.tables = tables
    pool.owner_index = None
    pool.storage = None
    pool.row_bytes, pool.weight_bytes = 72, 64
    pool.dtype, pool.scale_dtype = torch.int8, torch.float32
    assert pool.owner_range(0) == (0, 0)
    weight, scale = pool.weights(0)
    assert scale is not None
    assert weight.shape == (0, 64) and scale.shape == (0, 2)


@pytest.mark.parametrize(
    "consumer_load_format,consumer_revision", [("auto", "b" * 40), ("dummy", None), ("auto", None)]
)
def test_model_rejects_same_layout_different_revision_before_native_init(
    monkeypatch, tmp_path, consumer_load_format, consumer_revision
):
    from vllm_ascend.models.deepseek_v41.engram import offload as implementation

    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    setup = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "offload_config is not None"
        and any(
            isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "EngramOffloadPool"
            for child in ast.walk(node)
        )
    )
    setup_code = compile(ast.Module(body=[setup], type_ignores=[]), str(path), "exec")
    config = EngramOffloadConfig(store_url="tcp://127.0.0.1:8500", world_size=2, local_world_size=2, owner_ranks=[0])
    for name in ("config.json", "quant_model_weights.safetensors.index.json"):
        (tmp_path / name).write_text("{}")
    factory = Mock()
    namespace = dict(
        self=SimpleNamespace(engram_weight_root=tmp_path, engram_layout=SimpleNamespace(primes=[[[19]], [[23]]])),
        config=SimpleNamespace(engram_layer_ids=[0, 1], engram_num_embeddings=[19, 23], engram_head_dim=64),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(revision="a" * 40),
            load_config=SimpleNamespace(load_format="auto"),
            parallel_config=None,
        ),
        offload_config=config,
        storage_dtypes={0: torch.int8, 1: torch.int8},
        checkpoint_identity=checkpoint_identity,
        EngramOffloadPool=factory,
        get_global_rank=lambda _: 1,
        torch=SimpleNamespace(device=lambda *args: torch.device("cpu"), npu=SimpleNamespace(current_device=lambda: 0)),
        json=json,
        rollback=Mock(),
    )
    exec(setup_code, namespace)
    contributor_identity = factory.call_args.args[-1]
    namespace["vllm_config"].model_config.revision = consumer_revision
    namespace["vllm_config"].load_config.load_format = consumer_load_format
    namespace["EngramOffloadPool"] = EngramOffloadPool
    binding = Mock()
    binding.Offload.side_effect = AssertionError("Reached native pool creation")
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", SimpleNamespace(offload=binding))
    monkeypatch.setattr(implementation, "get_ascend_device_type", lambda: implementation.AscendDeviceType.A2)
    store = Mock()

    def contributor_layout(_):
        layout = json.loads(store.set.call_args.args[1])
        layout[0] = contributor_identity
        return json.dumps(layout).encode()

    store.get.side_effect = contributor_layout
    store_factory = Mock(return_value=store)
    monkeypatch.setattr(implementation.dist, "TCPStore", store_factory)
    missing_revision = consumer_revision is None and consumer_load_format != "dummy"
    with pytest.raises(
        ValueError, match="immutable checkpoint revision" if missing_revision else "layout/checkpoint differs"
    ):
        exec(setup_code, namespace)
    binding.Offload.assert_not_called()
    if missing_revision:
        store_factory.assert_not_called()
    else:
        assert store.set.call_args.args[0] == "layout/1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"local_world_size": 1},
        {"world_size": 3},
        {"owner_ranks": []},
        {"owner_ranks": [2, 2]},
        {"owner_ranks": [2, 4]},
        {"rank_offset": 4},
        {"store_url": "tcp://127.0.0.1:65535"},
        {"store_url": "etcd://127.0.0.1:8500"},
        {"unknown": True},
    ],
)
def test_pool_config_rejects_invalid_topology(overrides):
    values = dict(store_url="tcp://127.0.0.1:8500", world_size=4, local_world_size=2, owner_ranks=[2, 3])
    with pytest.raises(ValueError):
        EngramOffloadConfig(**(values | overrides))


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("mapping", ["mapped", "unsupported", "legacy"])
@pytest.mark.parametrize("owner_ranks", [(0, 1, 2, 3), (0, 2)])
@pytest.mark.parametrize("device_type", ["A2", "A3", "A5"])
def test_pool_routes_and_loads_owned_rows(monkeypatch, rank, mapping, owner_ranks, device_type):
    from vllm_ascend.models.deepseek_v41.engram import offload as implementation

    monkeypatch.setattr(
        implementation, "get_ascend_device_type", lambda: getattr(implementation.AscendDeviceType, device_type)
    )
    config = EngramOffloadConfig(
        store_url="tcp://127.0.0.1:8500",
        world_size=4,
        local_world_size=2,
        owner_ranks=owner_ranks,
    )
    native = Mock()
    mapped_base, rank_stride = (1 << 40, 1 << 34) if mapping == "mapped" else (0, 0)
    native.get_mapped_pool.return_value = (mapped_base, rank_stride)
    if mapping == "legacy":
        del native.get_mapped_pool
    native.initialize.return_value = native.register_entry_table.return_value = native.register_memory.return_value = 0
    native.uninitialize.return_value = 0
    backing = (ctypes.c_uint8 * 4096)()
    native.malloc.return_value = ctypes.addressof(backing)
    binding = SimpleNamespace(
        OffloadConfig=lambda: SimpleNamespace(flags=0, executor=0),
        Offload=Mock(return_value=native),
        Scene=SimpleNamespace(SHARED=0),
        Executor=SimpleNamespace(AICPU=1),
    )
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", SimpleNamespace(offload=binding))
    store = Mock()
    store.get.side_effect = lambda _: store.set.call_args.args[1].encode()
    monkeypatch.setattr(implementation.dist, "TCPStore", lambda *args, **kwargs: store)
    monkeypatch.setattr(torch.npu, "Stream", lambda **kwargs: Mock())
    monkeypatch.setattr(torch.npu, "synchronize", lambda *args: None)
    pool = EngramOffloadPool(config, rank, [19, 23], 64, torch.int8, torch.device("cpu"), "checkpoint")
    try:
        assert native.initialize.call_args.args[0].executor == (1 if device_type in ("A2", "A5") else 0)
        owners = owner_ranks
        assert pool.owners.tolist() == list(owners)
        owner_index = owners.index(rank) if rank in owners else None
        assert pool.owner_index == owner_index
        tables, stride = make_offload_tables([19, 23], len(owners))
        assert pool.tables == tables and pool.rows_per_slot == stride
        assert pool.owner_range(0) == ((0, 0) if owner_index is None else tables[0].owner_range(owner_index))
        if mapping == "mapped" and device_type == "A3":
            assert pool.lookup == pool._lookup_mapped
            assert not pool.buffers and not hasattr(pool, "stream")
            native.register_memory.assert_not_called()
            for bases, table in zip(pool.mapped_bases, tables):
                assert bases.tolist() == [mapped_base + owner * rank_stride + table.offset * 72 for owner in owners]
        else:
            assert pool.lookup == pool._lookup_native
            assert native.register_memory.call_count == len(tables)
            assert not pool.mapped_bases
        if device_type != "A3" and mapping != "legacy":
            native.get_mapped_pool.assert_not_called()
    finally:
        pool.close()
    assert not pool.mapped_bases
    assert pool.store is None


def test_memfabric_and_shared_engram_configs_are_mutually_exclusive():
    config = AscendConfig(
        sparse_kv_offload_config=SparseKVOffloadConfig(),
        shared_engram={"backend": "a3", "endpoint": "127.0.0.1:29500"},
        engram_offload={
            "store_url": "tcp://127.0.0.1:8500",
            "world_size": 4,
            "local_world_size": 2,
            "owner_ranks": [2, 3],
        },
    )
    with pytest.raises(ValueError, match="engram_offload and shared_engram are mutually exclusive"):
        config.derive_and_validate(SimpleNamespace())


@pytest.mark.parametrize(
    "backend", [{"local_vmm": True}, {"remote_reader": object()}, {"shared_pool_endpoint": "owner"}]
)
def test_memfabric_rejects_another_embedding_storage_backend(backend):
    with pytest.raises(ValueError, match="MemFabric Engram cannot be combined"):
        AscendParallelEngramEmbedding(19, 64, (9, 10), 0, offload_pool=object(), **backend)


@pytest.mark.parametrize("owner", [None, 0, 1])
def test_checkpoint_ownership_is_independent_of_head_partition(monkeypatch, tmp_path, owner):
    key = "layers.1.engram.embed.weight"
    scale_key = key.removesuffix(".weight") + ".scale"
    codes = torch.arange(19 * 64).reshape(19, 64).remainder(127).to(torch.int8)
    scales = torch.full((19, 2), 0.125).to(torch.float8_e8m0fnu)
    save_file({key: codes, scale_key: scales}, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "weights.safetensors", scale_key: "weights.safetensors"}})
    )
    pool = object.__new__(EngramOffloadPool)
    pool.tables, rows = make_offload_tables([19], 2)
    pool.row_bytes, pool.weight_bytes = 72, 64
    pool.dtype, pool.scale_dtype = torch.int8, torch.float32
    pool.owner_index = owner
    pool.storage = torch.zeros((rows, 72), dtype=torch.uint8) if owner is not None else None
    finish_load = Mock()
    monkeypatch.setattr(pool, "finish_load", finish_load)
    table = object.__new__(AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table._offload_pool, table.layer_hash_index = pool, 0
    table.vocab_start_idx, table.vocab_end_idx = 3, 6
    table.storage_start_idx, table.storage_end_idx = pool.owner_range(0)
    table.block_size = 32
    weight, scale = pool.weights(0)
    table.weight = torch.nn.Parameter(weight, requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
    table.load_checkpoint(tmp_path, key, chunk_rows=3)
    finish_load.assert_called_once_with(0, None)
    start, end = pool.owner_range(0)
    torch.testing.assert_close(table.weight, codes[start:end])
    torch.testing.assert_close(table.weight_scale_inv, scales[start:end].float())
    if owner is not None:
        assert pool.storage is not None
        assert not pool.storage[0].any()


def test_load_readiness_propagates_failure_and_uses_new_epoch():
    pool = object.__new__(EngramOffloadPool)
    pool.config, pool.rank = SimpleNamespace(world_size=2), 0
    pool.load_epochs = [0]
    pool.store = Mock()
    pool.store.get.side_effect = [b"", b"checkpoint read failed", b"", b""]
    with pytest.raises(RuntimeError, match="rank 1: checkpoint read failed"):
        pool.finish_load(0, None)
    pool.finish_load(0, None)
    assert [call.args[0] for call in pool.store.set.call_args_list] == ["loaded/0/0/0", "loaded/0/1/0"]


@pytest.mark.parametrize("rows", [1, 65535, 65536, 65537])
def test_lookup_bounds_each_chunk(monkeypatch, rows):
    from vllm_ascend.models.deepseek_v41.engram import offload as implementation

    monkeypatch.setattr(implementation, "init_device_properties_triton", lambda: None)
    monkeypatch.setattr(implementation, "triton", SimpleNamespace(cdiv=lambda a, b: -(-a // b)))
    monkeypatch.setattr(implementation, "_engram_offload_encode_kernel", MagicMock())
    monkeypatch.setattr(implementation, "_engram_offload_decode_kernel", MagicMock())
    caller = Mock()
    monkeypatch.setattr(torch.npu, "current_stream", lambda _: caller)
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    pool = object.__new__(EngramOffloadPool)
    pool.device, pool.stream = torch.device("cpu"), Mock()
    pool.tables, pool.rows_per_slot = make_offload_tables([rows], 2)
    pool.dim, pool.weight_bytes, pool.dtype, pool.scale_dtype = 64, 64, torch.int8, torch.float32
    pool.owners = torch.tensor([0, 1])
    staging = torch.empty((65536, 72), dtype=torch.uint8)
    pool.buffers = [(torch.empty(65536, dtype=torch.int64), torch.zeros(1, dtype=torch.int32), staging)]
    pool.pool = Mock()
    pool.pool.entry_gather.return_value = 0
    pool._lookup_native(0, torch.zeros((rows, 1), dtype=torch.int64), torch.empty((rows, 1, 64)), 0, 1, 0, rows)
    chunks = [min(65536, rows - offset) for offset in range(0, rows, 65536)]
    assert len(pool.pool.entry_gather.call_args_list) == len(chunks)
    for call, size in zip(pool.pool.entry_gather.call_args_list, chunks):
        assert call.args[0].shape == (size, 72)
        assert call.args[0].data_ptr() + call.args[0].numel() == staging.data_ptr() + staging.numel()
        assert not call.kwargs
    caller.wait_stream.assert_called_once_with(pool.stream)


@pytest.mark.parametrize("close_fails", [False, True])
def test_model_owns_pool_teardown_and_preserves_aliases_if_close_fails(close_fails):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    model = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "AscendDeepseekV41LLMForCausalLM"
    )
    close = next(
        node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "close_shared_engram"
    )
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[close], type_ignores=[]), str(path), "exec"), namespace)
    pool = Mock()
    table = SimpleNamespace(_offload_pool=pool, weight=object(), weight_scale_inv=object())
    backbone = SimpleNamespace(
        engram_offload_pool=pool,
        local_engram_layer_ids=(0,),
        layers=[SimpleNamespace(engram=SimpleNamespace(embed_tokens=table))],
    )

    def close_pool():
        assert table._offload_pool is pool and table.weight is not None and table.weight_scale_inv is not None
        if close_fails:
            raise RuntimeError("synchronize failed")

    pool.close.side_effect = close_pool
    close_language_model = namespace["close_shared_engram"]
    language_model = SimpleNamespace(model=backbone)
    language_model.close_shared_engram = lambda: close_language_model(language_model)
    wrapper_path = path.with_name("vl_model.py")
    wrapper = next(node for node in ast.parse(wrapper_path.read_text()).body if isinstance(node, ast.ClassDef))
    forward_close = next(
        node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "close_shared_engram"
    )
    exec(compile(ast.Module(body=[forward_close], type_ignores=[]), str(wrapper_path), "exec"), namespace)
    outer = SimpleNamespace(language_model=language_model)
    if close_fails:
        with pytest.raises(RuntimeError, match="synchronize failed"):
            namespace["close_shared_engram"](outer)
        assert backbone.engram_offload_pool is pool and table._offload_pool is pool
        return
    namespace["close_shared_engram"](outer)
    assert backbone.engram_offload_pool is None
    assert table._offload_pool is table.weight is table.weight_scale_inv is None
    namespace["close_shared_engram"](outer)
    pool.close.assert_called_once()


@pytest.mark.parametrize(
    "stage", ["initialize", "malloc", "register_entry_table", "register_memory", "get_mapped_pool"]
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_pool_constructor_preserves_initial_failure(monkeypatch, stage, cleanup_fails):
    from vllm_ascend.models.deepseek_v41.engram import offload as implementation

    native = Mock()
    native.initialize.return_value = native.register_entry_table.return_value = native.register_memory.return_value = 0
    native.uninitialize.return_value = None
    native.get_mapped_pool.return_value = (0, 0)
    backing = (ctypes.c_uint8 * 4096)()
    native.malloc.return_value = ctypes.addressof(backing)
    failure = KeyboardInterrupt(stage) if stage == "initialize" else RuntimeError(stage)
    getattr(native, stage).side_effect = failure
    binding = SimpleNamespace(OffloadConfig=SimpleNamespace, Offload=lambda: native, Scene=SimpleNamespace(SHARED=0))
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", SimpleNamespace(offload=binding))
    monkeypatch.setattr(implementation, "get_ascend_device_type", lambda: implementation.AscendDeviceType.A3)
    store = Mock()
    store.get.side_effect = lambda _: store.set.call_args.args[1].encode()
    monkeypatch.setattr(implementation.dist, "TCPStore", lambda *args, **kwargs: store)
    monkeypatch.setattr(torch.npu, "Stream", lambda **kwargs: Mock())
    synchronize = Mock(side_effect=RuntimeError("cleanup") if cleanup_fails else None)
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)
    register = Mock()
    monkeypatch.setattr(implementation.atexit, "register", register)
    config = EngramOffloadConfig(store_url="tcp://127.0.0.1:8500", world_size=2, local_world_size=2, owner_ranks=[0])
    with pytest.raises(type(failure), match=stage) as raised:
        EngramOffloadPool(config, 0, [19], 64, torch.int8, torch.device("cpu"), "checkpoint")
    assert raised.value is failure
    register.assert_not_called()
    assert native.uninitialize.call_count == (0 if cleanup_fails else 1)


@pytest.mark.parametrize(
    "filename,classname,path",
    [
        ("model.py", "DeepseekV41Model", ()),
        ("model.py", "AscendDeepseekV41LLMForCausalLM", ("model",)),
        ("vl_model.py", "AscendDeepseekV41ForCausalLM", ("language_model", "model")),
    ],
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_model_constructor_rolls_back_pool_at_every_wrapper(filename, classname, path, cleanup_fails):
    from vllm_ascend.models.deepseek_v41.engram_lifecycle import close_engram_on_init_failure

    root = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41"
    module = ast.parse((root / filename).read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == classname)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    expression = ast.Expression(init.decorator_list[0])
    decorate = eval(
        compile(expression, filename, "eval"), {"close_engram_on_init_failure": close_engram_on_init_failure}
    )
    reader, pool = Mock(), Mock()
    if cleanup_fails:
        reader.close.side_effect = pool.close.side_effect = RuntimeError("cleanup")
    model = SimpleNamespace(_engram_remote_reader=reader, engram_offload_pool=pool)
    for name in reversed(path):
        model = SimpleNamespace(**{name: model})
    failure = KeyboardInterrupt("outer init")

    @decorate
    def initialize(self):
        raise failure

    with pytest.raises(KeyboardInterrupt) as raised:
        initialize(model)
    assert raised.value is failure
    reader.close.assert_called_once()
    pool.close.assert_called_once()


@pytest.mark.parametrize(
    "load_format,strategy",
    [
        ("auto", None),
        ("auto", "lazy"),
        ("safetensors", "lazy"),
        ("pt", "lazy"),
        ("auto", "eager"),
        ("dummy", None),
        ("dummy", "eager"),
    ],
)
@pytest.mark.parametrize("sleep", [False, True])
def test_memfabric_validates_sleep_and_checkpoint_loading_before_allocation(load_format, strategy, sleep):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ascend_config.py"
    module = ast.parse(source.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "AscendConfig")
    derive = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "derive_and_validate")
    check = next(
        n for n in derive.body if isinstance(n, ast.If) and ast.unparse(n.test) == "self.engram_offload is not None"
    )
    code = compile(ast.Module(body=[check], type_ignores=[]), str(source), "exec")
    vc = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=sleep),
        load_config=SimpleNamespace(load_format=load_format, safetensors_load_strategy=strategy),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, world_size=2, data_parallel_size=1),
    )
    namespace = dict(
        vc=vc,
        engram_config=SimpleNamespace(cpu_offload=True),
        self=SimpleNamespace(engram_offload=SimpleNamespace(rank_offset=0, world_size=2)),
    )
    if sleep or (load_format != "dummy" and (load_format == "pt" or strategy == "eager")):
        with pytest.raises(ValueError, match="sleep mode" if sleep else "lazy safetensors"):
            exec(code, namespace)
    else:
        exec(code, namespace)


def test_loader_finalization_uses_ascend_storage_lifecycle_without_upstream_peers():
    from vllm.model_executor.model_loader.weight_utils import finalize_model_weights

    table = object.__new__(AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table._offload_pool = Mock()
    table.weight = torch.nn.Parameter(torch.ones((1, 64), dtype=torch.int8), requires_grad=False)
    finalize_model_weights(table, torch.device("cpu"))
    assert torch.all(table.weight == 1)
    table._offload_pool.finish_load.assert_not_called()
