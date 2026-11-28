# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify that the A5 backend uses request offsets inside padded capture buffers."""

import pytest
import torch


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_verify_padded_requests(state_dtype):
    pytest.importorskip("torch_npu")
    pytest.importorskip("tilelang.ascend.language")
    from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.gdn_recoverssm_test_utils import (
        cpu_reference,
        model_inputs,
        npu_device,
    )
    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.ops.tilelang_recoverssm.full_backend import verify_step

    device = npu_device()
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("requires an A5 NPU")

    offsets = [0, 3, 3, 5]
    pages = [2, 0, 1]
    inputs = model_inputs(device, 12, 4, 12, 128, 128)
    checkpoint = torch.randn((3, 12, 128, 128), device=device, dtype=state_dtype) * 0.02
    original = checkpoint.clone()
    replay = torch.full((3, 12, 4, 257), -7.0, device=device, dtype=torch.float32)
    expected_replay = replay.cpu().clone()
    output = torch.full_like(inputs[-1], 13.0, memory_format=torch.contiguous_format)
    expected, _, records = cpu_reference(inputs, checkpoint, offsets, pages)
    for page, record in records.items():
        expected_replay[page, :, : record.shape[1]] = record.float()

    verify_step(
        *inputs,
        checkpoint_state=checkpoint,
        replay_cache=replay,
        query_start_loc=torch.tensor(offsets, device=device, dtype=torch.int32),
        state_indices=torch.tensor(pages, device=device, dtype=torch.int32),
        spec_query_len=4,
        out=output,
    )
    active = offsets[-1]
    torch.testing.assert_close(output[:, :active].cpu().float(), expected[:, :active].float(), atol=8e-3, rtol=8e-3)
    assert (output[:, active:] == 13.0).all()
    torch.testing.assert_close(checkpoint, original, atol=0, rtol=0)
    torch.testing.assert_close(replay.cpu(), expected_replay, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gate_parameter_conversion_in_kernel(state_dtype):
    pytest.importorskip("torch_npu")
    pytest.importorskip("tilelang.ascend.language")
    from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.gdn_recoverssm_test_utils import (
        model_inputs,
        npu_device,
    )
    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.ops.tilelang_recoverssm.full_backend import verify_step

    device = npu_device()
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("requires an A5 NPU")
    inputs = list(model_inputs(device, 8, 4, 12, 128, 128))
    inputs[0] = inputs[0].bfloat16()
    inputs[3] = inputs[3].bfloat16()
    checkpoint = torch.randn((3, 12, 128, 128), device=device, dtype=state_dtype) * 0.02
    reference = [t.float() if i in (0, 3) else t for i, t in enumerate(inputs)]
    results = []
    for args in (reference, inputs):
        replay = torch.zeros((3, 12, 4, 257), device=device, dtype=torch.float32)
        out = verify_step(
            *args,
            checkpoint_state=checkpoint,
            replay_cache=replay,
            query_start_loc=torch.tensor([0, 4, 8], device=device, dtype=torch.int32),
            state_indices=torch.tensor([1, 2], device=device, dtype=torch.int32),
            spec_query_len=4,
        )
        results.append((out.cpu(), replay.cpu()))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
