# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class DeviceTensor:
    """保留真实CPU存储，只替换设备边界，避免测试创建NPU上下文。"""

    device = SimpleNamespace(type="npu")

    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __getitem__(self, index):
        return self.tensor[index]


def load_reader(copy, fmt=2):
    # 执行实际生产函数；只隔离设备工厂和最终copy内核，不复制slot算法。
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/paged_mla_cache_read.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
    cpu_torch = SimpleNamespace(
        float16=torch.float16,
        bfloat16=torch.bfloat16,
        int64=torch.int64,
        arange=lambda *args, **kwargs: torch.arange(*args, **{**kwargs, "device": "cpu"}),
        repeat_interleave=torch.repeat_interleave,
        cumsum=torch.cumsum,
        cat=torch.cat,
    )
    namespace = {
        "torch": cpu_torch,
        "torch_npu": SimpleNamespace(get_npu_format=lambda cache: fmt),
        "copy_pcp_kv_cache": copy,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["try_load_strided_mla_cache"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("component", ["rope", "duplicate", "empty"])
@pytest.mark.parametrize(
    "lengths,starts",
    [
        ([384], [0]),
        ([384, 129, 0, 65], [0, 128, 999999, 256]),
        ([2, 131, 0, 259], [127, 129, 999999, 7]),
        ([0], [999999]),
    ],
)
def test_strided_token_reads_preserve_request_order_and_source(dtype, component, lengths, starts):
    block_size, blocks, key_dim, rope_dim = 128, 15, 512, 64
    slot_elements = block_size * (key_dim + rope_dim)
    prefix_elements = 256
    raw = torch.zeros(prefix_elements + blocks * slot_elements, dtype=dtype)
    key = torch.as_strided(raw, (blocks, block_size, 1, key_dim), (slot_elements, key_dim, key_dim, 1), prefix_elements)
    rope = torch.as_strided(
        raw,
        (blocks, block_size, 1, rope_dim),
        (slot_elements, rope_dim, rope_dim, 1),
        prefix_elements + block_size * key_dim,
    )
    key.copy_((torch.arange(key.numel()).reshape(key.shape) % 127).to(dtype))
    rope.copy_((torch.arange(rope.numel()).reshape(rope.shape) % 97 + 5).to(dtype))
    value = key if component == "duplicate" else rope[..., :0] if component == "empty" else rope
    before = raw.clone()
    table = torch.tensor([[3, 4, 5, 6, 7, 8], [3, 4, 12, 13, 14, 6], [999999] * 6, [3, 4, 5, 6, 7, 8]][: len(lengths)])
    expected_slots = [
        int(table[row, (start + token) // block_size]) * block_size + (start + token) % block_size
        for row, (length, start) in enumerate(zip(lengths, starts))
        for token in range(length)
    ]
    calls = []

    def copy(cache, slots):
        calls.append(slots.tolist())
        # 独立逐行参考，不依赖生产函数的repeat_interleave或前缀累加。
        return torch.stack(
            [
                torch.cat([part.tensor[slot // block_size, slot % block_size, 0] for part in cache])
                for slot in slots.tolist()
            ]
        )

    tokens = sum(lengths)
    output_k = torch.empty(tokens, 1, key_dim, dtype=dtype)
    output_v = torch.empty(tokens, 1, value.shape[-1], dtype=dtype)
    assert load_reader(copy)(
        DeviceTensor(key),
        DeviceTensor(value),
        DeviceTensor(table),
        DeviceTensor(torch.tensor(lengths)),
        DeviceTensor(torch.tensor(starts)),
        DeviceTensor(output_k),
        DeviceTensor(output_v),
    )
    if tokens:
        assert calls[0][:tokens] == expected_slots
        assert len(calls[0]) % 8 == 0
        assert calls[0][tokens:] == [expected_slots[0]] * ((-tokens) % 8)
        for row, slot in enumerate(expected_slots):
            assert torch.equal(output_k[row, 0], key[slot // block_size, slot % block_size, 0])
            assert torch.equal(output_v[row, 0], value[slot // block_size, slot % block_size, 0])
    else:
        assert not calls
    assert torch.equal(raw, before)


@pytest.mark.parametrize("unsupported", ["contiguous", "nz", "heads", "int8"])
def test_unaffected_layouts_keep_native_reader(unsupported):
    cache = torch.zeros(2, 128, 1, 16, dtype=torch.bfloat16)
    if unsupported == "int8":
        cache = cache.to(torch.int8)
    if unsupported != "contiguous":
        cache = cache[..., ::2]
    if unsupported == "heads":
        cache = cache.expand(2, 128, 2, 8)

    def forbidden_copy(*args):
        pytest.fail("未受影响布局不能进入新copy路径")

    fmt = 29 if unsupported == "nz" else 2
    assert not load_reader(forbidden_copy, fmt)(DeviceTensor(cache), DeviceTensor(cache), None, None, None, None, None)
