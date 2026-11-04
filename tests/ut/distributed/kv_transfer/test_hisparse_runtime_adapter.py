"""Isolated CPU contracts for the proposed Ascend-owned runtime adapter.

Uses actual production definitions and real CPU torch tensors. Import isolation
does not qualify NPU kernels, graph replay, serving or dataset accuracy.
"""

import ast
import hashlib
import importlib.util
import inspect
import sys
import types
from functools import wraps
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
ADAPTER = ROOT / "vllm_ascend/distributed/kv_transfer/hisparse/runtime_adapter.py"
_common_spec = importlib.util.find_spec("vllm")
if _common_spec is None or _common_spec.origin is None:
    pytest.skip("Paired vLLM source is required", allow_module_level=True)
COMMON_ROOT = Path(_common_spec.origin).parent
COMMON = COMMON_ROOT / "v1/hisparse/runtime.py"
WORKER = COMMON_ROOT / "distributed/kv_transfer/kv_connector/v1/hisparse/worker.py"


def load_definitions(path, class_name=None, names=None):
    tree = ast.parse(path.read_bytes())
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    if names:
        nodes = [node for node in nodes if getattr(node, "name", None) in names]
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    namespace = {
        "torch": torch,
        "async_tensor_h2d": lambda data, device, dtype: torch.tensor(data, device=device, dtype=dtype),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("resident_block", [1, 2, 4])
@pytest.mark.parametrize("state_order", [[0, 1], [1, 0]])
def test_prefill_resident_mapping_matches_common(resident_block, state_order):
    baseline = load_definitions(COMMON, "HiSparsePrefillStagingPlan")["ensure_gpu_sources"]
    candidate = load_definitions(ADAPTER)["_ensure_gpu_sources"]
    row_ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]], dtype=torch.int32)
    table = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    resident = torch.tensor([[0, 7, 8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 0, 19, 20, 21]], dtype=torch.int32)
    states = torch.tensor(state_order, dtype=torch.int64)

    def plan():
        return types.SimpleNamespace(
            row_ids=row_ids.clone(),
            block_table=table.clone(),
            block_size=4,
            gpu_source_key=None,
            gpu_row_ids=None,
            miss_mask=None,
        )

    expected, actual = plan(), plan()
    baseline(expected, resident, states, resident_block)
    candidate(actual, resident, states, resident_block)
    assert torch.equal(actual.gpu_row_ids, expected.gpu_row_ids)
    assert torch.equal(actual.miss_mask, expected.miss_mask)
    old_storage = actual.gpu_row_ids
    candidate(actual, resident, states, resident_block)
    assert actual.gpu_row_ids is old_storage


def test_group_invalidation_updates_all_leaders_and_reuses_pointer_storage():
    cls = load_definitions(COMMON, names={"HiSparseSlotInvalidator"})["HiSparseSlotInvalidator"]
    states = torch.tensor([1, 0], dtype=torch.int32)
    requests = torch.tensor([0, 1, 1], dtype=torch.int32)
    slots = torch.tensor([3, 5, -1], dtype=torch.int32)
    tables = [torch.tensor([[1, 5, 3, -1], [3, 2, 5, -1]], dtype=torch.int32) for _ in range(3)]
    pointer_map = {t.data_ptr(): t for t in tables}
    calls = []

    def invalidate(ptrs, mapping, reqs, written, rows, width):
        calls.append(ptrs.data_ptr())
        for pointer in ptrs.tolist():
            table = pointer_map[pointer]
            for req, slot in zip(reqs.tolist(), written.tolist()):
                if req < 0 or slot < 0:
                    continue
                row = table[int(mapping[req])]
                row[row == slot] = -1

    leaders = [
        types.SimpleNamespace(
            index_group=types.SimpleNamespace(device_global_indices=t),
            cache_ops=types.SimpleNamespace(hisparse_invalidate_written_slots=invalidate),
        )
        for t in tables
    ]
    owner = cls(states)
    owner(leaders, slots, requests)
    for table in tables:
        assert table.tolist() == [[1, -1, 3, -1], [-1, 2, 5, -1]]
    owner(leaders, slots, requests)
    assert len(calls) == 2 and calls[0] == calls[1]
    owner([], slots, requests)
    owner(leaders, slots[:0], requests[:0])
    assert len(calls) == 2
    # A second worker owns its own descriptor storage even for the same tables.
    other = cls(states)
    other(leaders, slots, requests)
    assert calls[-1] != calls[0]


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("logical_event", [False, True])
def test_stage_keeps_dependency_and_follower_event_order(decode, logical_event):
    ns = load_definitions(COMMON, "HiSparseRuntime")
    events = []

    class Stream:
        def wait_event(self, event):
            events.append(("wait_event", event))

        def wait_stream(self, stream):
            events.append(("wait_stream", stream))

        def __enter__(self):
            events.append(("enter",))

        def __exit__(self, *args):
            events.append(("exit",))

    class Event:
        def record(self, stream):
            events.append(("record",))

    ns.update(
        torch=types.SimpleNamespace(accelerator=types.SimpleNamespace(current_stream=lambda: "compute"), Event=Event),
        in_piecewise_cudagraph=lambda: False,
    )
    follower = types.SimpleNamespace(
        _swap_rows=lambda rows: events.append(("swap_follower",)), _layer_ready_event=None, _swap_staged=False
    )
    group = types.SimpleNamespace(
        copy_stream=Stream(), logical_topk_ready="topk" if logical_event else None, followers=[follower]
    )
    runtime = types.SimpleNamespace(
        index_group=group,
        _layer_ready_event=None,
        _swap_staged=False,
        _resolve_residency=lambda **kw: events.append(("resolve", kw["num_valid_rows"])),
        _swap_rows=lambda rows: events.append(("swap_leader",)),
    )
    valid = torch.tensor([3], dtype=torch.int32)
    ns["_resolve_and_stage_group"](
        runtime,
        resident=types.SimpleNamespace(prefetch_decode_group=decode),
        req_id_per_token=torch.tensor([0, 0, 0, 0], dtype=torch.int32),
        block_table=None,
        logical_topk_indices=torch.empty((4, 8), dtype=torch.int32),
        block_size=128,
        return_valid_counts=True,
        shared_rows=slice(None),
        attention_indices_out=None,
        valid_counts_out=None,
        num_valid_rows=valid,
    )
    expected = [
        ("wait_event", "topk") if logical_event else ("wait_stream", "compute"),
        ("enter",),
        ("resolve", valid),
        ("swap_leader",),
        ("record",),
    ]
    if decode:
        expected += [("swap_follower",), ("record",)]
    assert events == expected + [("exit",)]
    assert runtime._swap_staged and follower._swap_staged is decode


def test_worker_mirror_preserves_dma_and_batches_only_active_group_leaders():
    ns = load_definitions(WORKER, "HiSparseConnectorWorker")
    events = []
    slots = torch.tensor([1, 2, 3], dtype=torch.int32)
    requests = torch.tensor([0, 0, 0], dtype=torch.int32)
    handles = []
    for leader in (True, False, True):
        handles.append(
            types.SimpleNamespace(
                num_actual_tokens=3,
                num_decode_tokens=3,
                decode_batch=True,
                host_mirror_required=True,
                mirror_slot_mapping=slots,
                req_id_per_token=requests,
                runtime=types.SimpleNamespace(is_group_leader=leader, eager_host_mirror=True),
            )
        )
    worker = types.SimpleNamespace(
        cache_handles=handles,
        is_host_writer=True,
        _draft_layers={2},
        _per_layer_mirrored=set(),
        _submitted_mirror_layers=set(),
        _enqueue_row_dma=lambda layers, ready_event: events.append(("dma", layers, ready_event)),
        _invalidate_written_slots=lambda leaders, written, reqs: events.append(("invalidate", leaders, written, reqs)),
    )
    ns["_enqueue_host_mirror"](worker, ready_event="ready")
    assert events[0] == ("dma", (0, 1), "ready")
    assert events[1][0] == "invalidate" and events[1][1] == [handles[0].runtime, handles[2].runtime]
    assert torch.equal(events[1][2], slots) and torch.equal(events[1][3], requests)
    assert worker._submitted_mirror_layers == {0, 1} and worker._draft_mirror_pending
    handles[1].num_decode_tokens = 2
    with pytest.raises(RuntimeError, match="disagree on mirror metadata"):
        ns["_enqueue_host_mirror"](worker)


def install_fixture(monkeypatch):
    ns = load_definitions(ADAPTER)
    plan_method = load_definitions(COMMON, "HiSparsePrefillStagingPlan")["ensure_gpu_sources"]
    worker_method = load_definitions(WORKER, "HiSparseConnectorWorker")["_enqueue_host_mirror"]
    runtime_method = load_definitions(COMMON, "HiSparseRuntime")["_resolve_residency"]
    Plan = type("Plan", (), {"ensure_gpu_sources": plan_method})
    Worker = type("Worker", (), {"_enqueue_host_mirror": worker_method})
    Runtime = type("Runtime", (), {"_resolve_residency": runtime_method})
    runtime_module = types.ModuleType("vllm.v1.hisparse.runtime")
    runtime_module.HiSparsePrefillStagingPlan = Plan
    worker_module = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker")
    worker_module.HiSparseConnectorWorker = Worker
    monkeypatch.setitem(sys.modules, runtime_module.__name__, runtime_module)
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    profile_name = "vllm_ascend.distributed.kv_transfer.hisparse.optimization_profile"
    profile_spec = importlib.util.spec_from_file_location(profile_name, ADAPTER.with_name("optimization_profile.py"))
    assert profile_spec is not None and profile_spec.loader is not None
    profile = importlib.util.module_from_spec(profile_spec)
    profile_spec.loader.exec_module(profile)
    monkeypatch.setitem(sys.modules, profile_name, profile)
    ns.update(
        wraps=wraps,
        hashlib=hashlib,
        inspect=inspect,
        Path=Path,
        COMMON_RUNTIME_SHA256="89b940424066b62bf042bcdade3b7efacf03968d9f449888c12b351c636cb926",
        COMMON_WORKER_SHA256=hashlib.sha256(WORKER.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
    )
    return ns, Plan, Worker, Runtime


def test_call_site_installation_is_idempotent_and_keeps_common_mirror(monkeypatch):
    ns, Plan, Worker, _ = install_fixture(monkeypatch)
    original = Worker._enqueue_host_mirror
    ns["install_call_sites"]()
    wrapper = Worker._enqueue_host_mirror
    ensure = Plan.ensure_gpu_sources
    ns["install_call_sites"]()
    assert Worker._enqueue_host_mirror is wrapper
    assert Plan.ensure_gpu_sources is ensure
    assert wrapper is original
    worker = Worker()
    worker.cache_handles = [
        types.SimpleNamespace(
            num_actual_tokens=0, runtime=types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"))
        )
    ]
    assert wrapper(worker) is None
    assert not hasattr(worker, "_ascend_slot_invalidator")


def test_prefill_adapter_keeps_common_runtime_method(monkeypatch):
    ns, _, _, Runtime = install_fixture(monkeypatch)
    original = Runtime._resolve_residency
    ns["install_call_sites"]()
    assert Runtime._resolve_residency is original


def test_unreviewed_common_source_rejected_before_installation(monkeypatch):
    ns, _, Worker, Runtime = install_fixture(monkeypatch)
    original = Worker._enqueue_host_mirror
    ns["COMMON_RUNTIME_SHA256"] = "0" * 64
    with pytest.raises(RuntimeError, match="Unreviewed HiSparse"):
        ns["install_call_sites"]()
    assert Worker._enqueue_host_mirror is original
