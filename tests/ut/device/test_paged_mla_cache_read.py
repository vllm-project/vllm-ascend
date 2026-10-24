# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class DeviceTensor:
    """Keep real CPU storage and emulate the device boundary without creating an NPU context."""

    device = SimpleNamespace(type="npu")

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __getitem__(self, index):
        return self.tensor[index]


def load_reader(copy):
    # Execute the production slot reader with only the copy kernel replaced.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/device/paged_mla_cache_read.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    body: list[ast.stmt] = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    namespace = {
        "torch": torch,
        "copy_pcp_kv_cache": copy,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["load_mla_cache_rows_by_slots"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["component_major", "token_fused"])
@pytest.mark.parametrize(
    "slot_values",
    [
        [],
        [0],
        [-1],
        [-1, -1, -1],
        [127, 128, 255, -1, 129],
        [257, 0, 128, 128, 127, 383, 2, 255, 129],
        [-1, 127, 128, 383, 0, 128, 1, 255],
    ],
)
def test_slot_reads_preserve_values_order_and_source(dtype, layout, slot_values):
    block_size, blocks, key_dim, rope_dim = 128, 3, 512, 64
    page_elements = block_size * (key_dim + rope_dim)
    prefix_elements = 256
    raw = (torch.arange(prefix_elements + blocks * page_elements) % 127).to(dtype)
    if layout == "component_major":
        key_stride, rope_stride = key_dim, rope_dim
        rope_offset = prefix_elements + block_size * key_dim
    else:
        key_stride = rope_stride = key_dim + rope_dim
        rope_offset = prefix_elements + key_dim
    key = torch.as_strided(
        raw, (blocks, block_size, 1, key_dim), (page_elements, key_stride, key_stride, 1), prefix_elements
    )
    rope = torch.as_strided(
        raw, (blocks, block_size, 1, rope_dim), (page_elements, rope_stride, rope_stride, 1), rope_offset
    )
    before = raw.clone()
    # Supply non-contiguous, non-flat metadata to exercise slot normalization.
    slot_storage = torch.full((len(slot_values), 2), -999, dtype=torch.int32)
    slot_storage[:, 0] = torch.tensor(slot_values, dtype=torch.int32)
    slots = slot_storage[:, :1]
    slots_before = slot_storage.clone()

    class GuardedCache(DeviceTensor):
        def __getitem__(self, index):
            pytest.fail("The source cache must be read by the stride-aware kernel")

        def contiguous(self):
            pytest.fail("The source cache must not be materialized as contiguous")

    cache = (GuardedCache(key), GuardedCache(rope))
    calls = []

    def reference_row(slot):
        if slot < 0:
            return torch.zeros(key_dim + rope_dim, dtype=dtype)
        block, offset = divmod(slot, block_size)
        return torch.cat((key[block, offset, 0], rope[block, offset, 0]))

    def copy(actual_cache, actual_slots):
        assert actual_cache[0] is cache[0] and actual_cache[1] is cache[1]
        assert actual_slots.ndim == 1 and actual_slots.dtype == torch.int64
        calls.append(actual_slots.tolist())
        if not actual_slots.numel():
            return torch.empty(0, key_dim + rope_dim, dtype=dtype)
        return torch.stack([reference_row(slot) for slot in actual_slots.tolist()])

    packed = load_reader(copy)(*cache, slots)
    expected = (
        torch.stack([reference_row(slot) for slot in slot_values])
        if slot_values
        else raw.new_empty((0, key_dim + rope_dim))
    )
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    assert packed.is_contiguous()
    if slot_values:
        assert calls == [slot_values + [slot_values[0]] * ((-len(slot_values)) % 8)]
    else:
        assert all(not call for call in calls)
    assert torch.equal(raw, before)
    assert torch.equal(slot_storage, slots_before)
