# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.vocab_parallel_logits import pack_vocab_parallel_logits


@pytest.mark.skipif(
    not HAS_TRITON or not get_current_hardware_profile().supports(HardwareCapability.FUSED_TP_LOGITS_LAYOUT),
    reason="Requires the Ascend TP logits layout kernel",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "tp_size,num_rows,shard_size,vocab_size",
    [
        (2, 2, 2064, 4096),  # Partial tile in each shard, with padding in the last shard.
        (8, 3, 16160, 129280),  # DeepSeek-V4.1-Flash vocabulary at TP8.
        (8, 64, 2048, 6 * 2048),  # Trailing shards contain padding only.
    ],
)
@torch.inference_mode()
def test_pack_vocab_parallel_logits(dtype, tp_size, num_rows, shard_size, vocab_size):
    init_device_properties_triton()
    shards = torch.randn((tp_size, num_rows, shard_size), dtype=dtype, device="npu")
    expected = torch.cat(shards.unbind(0), dim=1)[:, :vocab_size]

    actual = pack_vocab_parallel_logits(shards.flatten(0, 1), num_rows, vocab_size)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
