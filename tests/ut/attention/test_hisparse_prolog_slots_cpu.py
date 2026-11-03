# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU contracts for current Resident slots; native replay needs NPU evidence."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def wrapper(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/sfa_hisparse_optimized.py"
    tree = ast.parse(path.read_bytes())
    methods = [
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_sfa_preprocess_prolog_v3", "_resident_prolog_slots_i64"}
    ]
    assert len(methods) == 2
    for method in methods:
        method.body = [node for node in method.body if not isinstance(node, ast.ImportFrom)]
    context, stream = [SimpleNamespace()], [1]
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(current_stream=lambda: SimpleNamespace(npu_stream=stream[0])), raising=False
    )

    class NativeOracle:
        def _sfa_preprocess_prolog_v3(self, hidden, caches, cos, sin, slots, **kwargs):
            self.calls.append("native")
            target = caches[0].view(torch.uint8).squeeze(2)
            for row, slot in enumerate(slots.tolist()):
                if slot >= 0:
                    target[slot // 128, slot % 128] = hidden[row]
            return hidden + 1

    def scatter(left, right, destination, slots, valid):
        for row in range(min(len(slots), int(valid[0]))):
            slot = int(slots[row])
            if slot >= 0:
                destination[slot // 128, slot % 128] = torch.cat((left[row], right[row]))

    def finish(cache, metadata):
        cache.calls.append("finish")

    definition = ast.ClassDef(
        name="Wrapper",
        bases=[ast.Name(id="NativeOracle", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    namespace = dict(
        torch=torch,
        NativeOracle=NativeOracle,
        scatter_packed_kv=scatter,
        finish_hisparse_kv_update=finish,
        get_forward_context=lambda: context[0],
    )
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["Wrapper"], context, stream


@pytest.mark.parametrize("rows", [4, 16, 32])
@pytest.mark.parametrize("mode", ["full", "tail", "holes", "empty", "prefill", "different_ordinary_slots", "partial"])
def test_prolog_writes_current_resident_rows_only(wrapper, rows, mode):
    cls, _, _ = wrapper
    valid = 0 if mode == "empty" else rows - 2 if mode in {"tail", "partial"} else rows
    slots = (torch.arange(rows, dtype=torch.int32) * 7 + 3) % 512
    slots[valid:] = -1
    if mode == "holes":
        slots[::3] = -1
    payload = torch.arange(rows * 656, dtype=torch.int32).reshape(rows, 656).remainder(251).to(torch.uint8)
    slab = torch.full((4, 5, 128, 656), 91, dtype=torch.uint8)
    expected = slab.clone()
    count = valid if mode == "partial" else rows
    for row in range(min(count, valid)):
        slot = int(slots[row])
        if slot >= 0:
            expected[slot // 128, 2, slot % 128] = payload[row]
    owner = cls()
    owner.calls = []
    owner.hisparse_cache = SimpleNamespace(
        slot_mapping=slots,
        calls=owner.calls,
        write_target=lambda input_rows, slot_rows: (slab[:, 2], slots[:count], count),
    )
    owner.enable_sparse_sfa_c8, owner._kv_staging_capacity = True, 128
    scratch = torch.empty((1, 128, 1, 656), dtype=torch.float8_e4m3fn)
    owner._prolog_c8_workspace = lambda: (scratch, torch.arange(128, dtype=torch.int64))
    ordinary = torch.where(slots >= 0, (slots + 97) % 512, slots) if mode == "different_ordinary_slots" else slots
    metadata = SimpleNamespace(
        num_prefills=int(mode == "prefill"), slot_mapping=ordinary, query_start_loc=torch.tensor([0, valid])
    )
    result = owner._sfa_preprocess_prolog_v3(
        payload, (slab[:, 2],), None, None, ordinary.to(torch.int64), attn_metadata=metadata
    )
    torch.testing.assert_close(slab, expected, rtol=0, atol=0)
    torch.testing.assert_close(result, payload + 1, rtol=0, atol=0)
    assert owner.calls == ["native", "finish"]


@pytest.mark.parametrize("rows", [4, 16, 32])
def test_current_alias_reuse_and_forward_stream_storage_changes(wrapper, rows):
    cls, context, stream = wrapper
    owner = cls()
    source = torch.arange(128, dtype=torch.int32)
    owner.hisparse_cache = SimpleNamespace(slot_mapping=source)
    first = owner._resident_prolog_slots_i64(source[:rows])
    owner.hisparse_cache.slot_mapping = source.view_as(source)
    assert owner._resident_prolog_slots_i64(source[:rows]) is first
    owner.hisparse_cache.slot_mapping = source
    source.add_(83)
    context[0] = SimpleNamespace()
    second = owner._resident_prolog_slots_i64(source[:rows])
    assert second is not first
    torch.testing.assert_close(second, source[:rows].to(torch.int64))
    stream[0] = 2
    third = owner._resident_prolog_slots_i64(source[:rows])
    assert third is not second
    source.set_(torch.arange(128, dtype=torch.int32) * 7)
    rebound = owner._resident_prolog_slots_i64(source[:rows])
    assert rebound is not third
    torch.testing.assert_close(rebound, source[:rows].to(torch.int64))
    source = source.clone() + 19
    owner.hisparse_cache.slot_mapping = source
    replaced = owner._resident_prolog_slots_i64(source[:rows])
    assert replaced is not rebound
    torch.testing.assert_close(replaced, source[:rows].to(torch.int64))
    shorter = owner._resident_prolog_slots_i64(source[: rows - 1])
    torch.testing.assert_close(shorter, source[: rows - 1].to(torch.int64))
