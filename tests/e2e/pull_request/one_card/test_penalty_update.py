# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.penalty import _apply_all_penalties_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.fixture(scope="module", autouse=True)
def initialize_device_properties():
    init_device_properties_triton()


def _device_view(tensor: torch.Tensor, strided: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if not strided:
        result = tensor.to("npu")
        return result, result
    rows, cols = tensor.shape
    storage = torch.full((rows, 2 * cols + 3), 11, dtype=tensor.dtype, device="npu")
    view = storage[:, 1 : 1 + 2 * cols : 2]
    view.copy_(tensor)
    return view, storage


@pytest.mark.parametrize(
    "num_seqs,vocab_size", [(1, 1), (1, 2047), (1, 2048), (1, 2049), (1, 151936), (7, 4099), (57, 32000)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_penalty_update_matches_reference(num_seqs, vocab_size, dtype, strided):
    """Cover vocabulary tails, multiple rows, and disjoint writes into strided storage."""
    torch.manual_seed(42)
    shape = (num_seqs, vocab_size)
    logits_cpu = torch.randn(shape, dtype=dtype)
    logits_cpu[0, 0] = 0
    prompt_cpu = torch.rand(shape) > 0.5
    counts_cpu = torch.randint(0, 4, shape, dtype=torch.int32)
    output_cpu = counts_cpu > 0
    repetition = torch.linspace(0.75, 1.5, num_seqs, dtype=torch.float32)
    frequency = torch.linspace(-0.2, 0.3, num_seqs, dtype=torch.float32)
    presence = torch.linspace(0.4, -0.1, num_seqs, dtype=torch.float32)

    # Compute the formula independently on CPU before any in-place update.
    reference = logits_cpu.float()
    repeated = prompt_cpu | output_cpu
    positive = reference / repetition[:, None]
    nonpositive = reference * repetition[:, None]
    reference = torch.where(repeated, torch.where(reference > 0, positive, nonpositive), reference)
    reference -= frequency[:, None] * counts_cpu
    reference -= presence[:, None] * output_cpu

    logits, storage = _device_view(logits_cpu, strided)
    prompt, _ = _device_view(prompt_cpu, strided)
    output, _ = _device_view(output_cpu, strided)
    counts, _ = _device_view(counts_cpu, strided)
    _apply_all_penalties_triton(
        logits, prompt, output, counts, repetition.to("npu"), frequency.to("npu"), presence.to("npu")
    )
    tolerance = {torch.float32: 1e-6, torch.float16: 1e-3, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(logits.cpu().float(), reference, atol=tolerance, rtol=tolerance)
    if strided:
        # Masked vocabulary tails must not overwrite padding or adjacent columns.
        assert torch.all(storage[:, ::2] == 11)
        assert torch.all(storage[:, -2:] == 11)


@pytest.mark.parametrize("shape", [(0, 17), (1, 0), (0, 0)])
def test_empty_penalty_update(shape):
    logits = torch.empty(shape, device="npu")
    mask = torch.empty(shape, dtype=torch.bool, device="npu")
    counts = torch.empty(shape, dtype=torch.int32, device="npu")
    penalties = torch.ones(shape[0], device="npu")
    _apply_all_penalties_triton(logits, mask, mask, counts, penalties, penalties, penalties)
    torch.npu.synchronize()
    assert logits.shape == shape
