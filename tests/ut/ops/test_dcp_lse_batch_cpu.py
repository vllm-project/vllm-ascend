# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production combine routing and kernel arithmetic without an NPU.

The Torch-backed language shim executes the kernel body and checks masked
memory access. It does not validate Triton compilation or NPU performance.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / "vllm_ascend/ops/triton/dcp"


def load_function(filename, name, scope):
    tree = ast.parse((OPS / filename).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), filename, "exec"), scope)
    return scope[name]


class LaunchSpy:
    def __init__(self):
        self.launch = Mock()

    def __getitem__(self, grid):
        return self.launch


class FakeDeviceTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        self.device = SimpleNamespace(type="npu")

    def __getattr__(self, name):
        return getattr(self.tensor, name)


@pytest.mark.parametrize("has_local", [False, True])
@pytest.mark.parametrize(
    "world,dim,rows,scatter,dtype,a5,local_stride,return_lse,expected",
    [
        (8, 512, 31, 1, torch.bfloat16, True, 1, False, False),
        (8, 512, 32, 1, torch.bfloat16, True, 1, False, True),
        (8, 512, 33, 1, torch.bfloat16, True, 1, False, True),
        (2, 512, 63, 1, torch.bfloat16, True, 1, False, False),
        (2, 512, 64, 1, torch.bfloat16, True, 1, False, True),
        (2, 512, 64, 1, torch.bfloat16, True, 2, False, None),
        (1, 128, 32, 1, torch.bfloat16, True, 1, False, True),
        (9, 512, 64, 1, torch.bfloat16, True, 1, False, False),
        (8, 513, 64, 1, torch.bfloat16, True, 1, False, False),
        (8, 512, 64, 0, torch.bfloat16, True, 1, False, False),
        (8, 512, 64, 1, torch.float16, True, 1, False, False),
        (8, 512, 64, 1, torch.float32, True, 1, True, False),
        (8, 512, 64, 1, torch.bfloat16, False, 1, False, False),
    ],
)
def test_combine_dispatch(has_local, world, dim, rows, scatter, dtype, a5, local_stride, return_lse, expected):
    scalar, batched = LaunchSpy(), LaunchSpy()
    scope = {
        "torch": SimpleNamespace(
            bfloat16=torch.bfloat16,
            float32=torch.float32,
            empty=lambda shape, dtype, device: torch.empty(shape, dtype=dtype),
        ),
        "is_950": lambda: a5,
        "init_device_properties_triton": lambda: None,
        "get_vectorcore_num": lambda: 8,
        "triton": SimpleNamespace(next_power_of_2=lambda x: 1 << (x - 1).bit_length()),
        "_fused_dcp_lse_combine_kernel": scalar,
        "_fused_dcp_lse_combine_batched_kernel": batched,
        "_validate_dcp_inputs": Mock(),
        "_lse_pack_dim": lambda value: 1 if value == torch.float32 else 4,
    }
    combine = load_function("dcp_a2a.py", "fused_dcp_lse_combine", scope)
    recv_shape = (
        (world, 1, rows, dim + scope["_lse_pack_dim"](dtype))
        if scatter == 1
        else (world, rows, 1, dim + scope["_lse_pack_dim"](dtype))
    )
    recv = FakeDeviceTensor(torch.empty(recv_shape, dtype=dtype))
    local = (
        FakeDeviceTensor(torch.empty(rows, 1, dim * local_stride, dtype=dtype)[..., ::local_stride])
        if has_local
        else None
    )
    lse = FakeDeviceTensor(torch.empty(rows, 1, 1)) if has_local else None
    if has_local:
        local.device = lse.device = recv.device
    combine(recv, dim, scatter, return_lse=return_lse, local_output=local, local_lse=lse)
    expected = not has_local if expected is None else expected
    assert batched.launch.call_count == int(expected)
    assert scalar.launch.call_count == int(not expected)
    call = (batched if expected else scalar).launch.call_args
    assert call.kwargs["HAS_LOCAL"] is has_local
    if expected:
        assert call.kwargs["BLOCK_ROWS"] == 8
    if has_local:
        assert call.args[2] is local and call.args[3] is lse
    else:
        assert call.args[2] is recv and call.args[3] is recv
        assert call.args[4:9] == (0, 0, 0, 0, 0)


class Pointer:
    def __init__(self, tensor, offset=None):
        self.storage = tensor.as_strided((tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,), 0)
        self.offset = tensor.storage_offset() if offset is None else offset

    def __add__(self, offset):
        return Pointer(self.storage, self.offset + offset)


class ForbiddenPointer:
    def __add__(self, offset):
        raise AssertionError("HAS_LOCAL=False must never access either local pointer")


class TorchLanguage:
    float32, int64 = torch.float32, torch.int64
    arange, zeros = staticmethod(torch.arange), staticmethod(torch.zeros)
    exp, exp2 = staticmethod(torch.exp), staticmethod(torch.exp2)
    abs, where, maximum = staticmethod(torch.abs), staticmethod(torch.where), staticmethod(torch.maximum)
    static_range = staticmethod(range)

    @staticmethod
    def full(shape, value, dtype):
        return torch.full(shape, value, dtype=dtype)

    def __init__(self, programs):
        self.programs, self.index = programs, 0

    def program_id(self, axis):
        return self.index

    def num_programs(self, axis):
        return self.programs

    @staticmethod
    def indices(pointer, mask):
        offsets, mask = torch.broadcast_tensors(torch.as_tensor(pointer.offset), mask)
        active = offsets[mask]
        assert ((active >= 0) & (active < pointer.storage.numel())).all(), "unmasked out-of-bounds access"
        return offsets, mask

    def load(self, pointer, mask, other):
        offsets, mask = self.indices(pointer, mask)
        result = torch.full(offsets.shape, other, dtype=pointer.storage.dtype)
        result[mask] = pointer.storage[offsets[mask]]
        return result

    def store(self, pointer, value, mask):
        offsets, mask = self.indices(pointer, mask)
        pointer.storage[offsets[mask]] = torch.broadcast_to(value, offsets.shape)[mask].to(pointer.storage.dtype)


def pack_reference(values, lse):
    # Encode FP32 fields using their IEEE bits, independently of the kernel's
    # log2/exp2 encoder. Values are [rank, token, head, feature].
    finite = torch.isfinite(lse)
    safe = torch.where(finite, lse.abs(), 0).contiguous()
    bits = safe.view(torch.int32)
    exponent = ((bits >> 23) & 255) - 127
    significand = (bits & 0x7FFFFF) | 0x800000
    exponent = torch.where(safe == 0, 0, exponent)
    significand = torch.where(safe == 0, 0, significand)
    code = (exponent + 128) * torch.where(lse < 0, -1, 1)
    code = torch.where(finite, code, 0)
    fields = torch.stack((code, significand >> 16, (significand >> 8) & 255, significand & 255), -1)
    return torch.cat((values, fields.to(values.dtype)), -1).transpose(1, 2).contiguous()


@pytest.mark.parametrize("world", [1, 2, 8])
@pytest.mark.parametrize("dim", [96, 257, 512])
@pytest.mark.parametrize("has_local", [False, True])
def test_production_batched_kernel_arithmetic_and_masks(world, dim, has_local):
    torch.manual_seed(52)
    tokens, heads = 13, 3  # 39 rows exercise a partial eight-row tile and loops.
    values = torch.randn(world, tokens, heads, dim).bfloat16()
    lse = torch.randn(world, tokens, heads) * 5
    lse[:, 0, 0] = -torch.inf
    values[:, 0, 0] = torch.nan
    lse[0, 1, 0] = torch.nan
    values[0, 1, 0] = torch.nan
    lse[:, 2, 0] = 70_000 + torch.arange(world) * 0.25
    lse[:, 3, 0] = -70_000 + torch.arange(world) * 0.25
    lse[:, 4, 0] = 0
    if world > 1:
        lse[1:, 5, 0] = torch.inf
        values[1:, 5, 0] = torch.nan
    recv = pack_reference(values, lse)
    local = torch.randn(tokens, heads, dim * 2).bfloat16()[..., ::2]
    local_lse = torch.randn(tokens, heads, 2)[..., :1]
    local_lse[0, 0] = torch.inf
    local[0, 0] = torch.nan
    output_storage = torch.full((tokens * heads * dim + 16,), -91.0, dtype=torch.bfloat16)
    output = output_storage[: tokens * heads * dim].view(tokens, heads, dim)
    language = TorchLanguage(programs=3)
    kernel = load_function("dcp_a2a_batched.py", "_fused_dcp_lse_combine_batched_kernel", {"tl": language})
    for program in range(language.programs):
        language.index = program
        kernel(
            Pointer(recv),
            Pointer(output),
            Pointer(local) if has_local else ForbiddenPointer(),
            Pointer(local_lse) if has_local else ForbiddenPointer(),
            *(local.stride() if has_local else (0, 0, 0)),
            *(local_lse.stride()[:2] if has_local else (0, 0)),
            *recv.stride(),
            *output.stride(),
            dim,
            heads,
            tokens * heads,
            DCP_SIZE=world,
            BLOCK_D=1 << (dim - 1).bit_length(),
            BLOCK_ROWS=8,
            HAS_LOCAL=has_local,
        )
    if has_local:
        values = torch.cat((values, local.unsqueeze(0)))
        lse = torch.cat((lse, local_lse[..., 0].unsqueeze(0)))
    finite = torch.isfinite(lse)
    weights = torch.softmax(lse.double().masked_fill(~finite, -torch.inf), dim=0).nan_to_num(0)
    expected = (torch.where(finite.unsqueeze(-1), values.double(), 0) * weights.unsqueeze(-1)).sum(0)
    torch.testing.assert_close(output, expected.bfloat16(), atol=0.008, rtol=0.008)
    assert torch.count_nonzero(output[0, 0]) == 0
    assert (output_storage[-16:] == -91).all()
