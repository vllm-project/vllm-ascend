# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise retained layer inputs across real NPU slot streams."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.distributed.weight_transfer.packed_tensor import packed_broadcast_consumer

_DELAY_MATRIX_SIZE = 4096
_DELAY_MATMUL_COUNT = 256
_RETAINED_WEIGHT_ELEMENTS = 128 * 1024
_ALLOCATOR_REUSE_ATTEMPTS = 64


@pytest.mark.parametrize(
    "lifetime_only,num_buffers",
    [(False, 1), (False, 2), (True, 2)],
    ids=["ready-single-slot", "ready-cross-slot", "lifetime-cross-slot"],
)
def test_retained_layer_clone_is_ready_and_alive(lifetime_only, num_buffers):
    torch.npu.set_device(0)
    expected = torch.arange(_RETAINED_WEIGHT_ELEMENTS, dtype=torch.float32, device="npu") + 1
    payloads = iter([expected, torch.ones(1, device="npu")])
    output = torch.zeros_like(expected)
    matrix = torch.ones((_DELAY_MATRIX_SIZE, _DELAY_MATRIX_SIZE), device="npu", dtype=torch.float16)
    scratch = torch.empty_like(matrix)
    torch.mm(matrix, matrix, out=scratch)
    torch.npu.synchronize()
    retained: list[torch.Tensor] = []
    poison: list[torch.Tensor] = []
    origin = None
    first_clone = True
    original_clone = torch.Tensor.clone

    def delay():
        for _ in range(_DELAY_MATMUL_COUNT):
            torch.mm(matrix, matrix, out=scratch)

    def clone(raw, *args, **kwargs):
        nonlocal first_clone
        if lifetime_only or not first_clone:
            return original_clone(raw, *args, **kwargs)
        first_clone = False
        result = torch.zeros_like(raw)
        torch.npu.current_stream().synchronize()
        delay()
        result.copy_(raw)
        return result

    def receive(buffer, **kwargs):
        buffer.copy_(next(payloads).view(torch.uint8))

    def load(weights):
        nonlocal origin
        if origin is None:
            retained.extend(tensor for _, tensor in weights)
            origin = torch.npu.current_stream()
            return
        first = retained.pop()
        if lifetime_only:
            # Finish the clone to isolate allocator lifetime.
            origin.synchronize()
            delay()
        output.copy_(first)
        del first
        if lifetime_only:
            # Reuse storage while the later slot still reads the clone.
            with torch.npu.stream(origin):
                for _ in range(_ALLOCATOR_REUSE_ATTEMPTS):
                    replacement = torch.empty_like(expected)
                    replacement.zero_()
                    poison.append(replacement)
                origin.synchronize()

    with patch.object(torch.Tensor, "clone", clone):
        packed_broadcast_consumer(
            iter([("layer.w0", ([_RETAINED_WEIGHT_ELEMENTS], torch.float32)), ("layer.w1", ([1], torch.float32))]),
            SimpleNamespace(broadcast=receive),
            0,
            load,
            1,
            num_buffers,
        )
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=0, atol=0)
