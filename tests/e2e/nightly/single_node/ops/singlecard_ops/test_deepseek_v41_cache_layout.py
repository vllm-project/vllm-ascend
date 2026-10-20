# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.models.deepseek_v41.cache_config import uses_a5_packed_cache
from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: E402, F401


@pytest.mark.parametrize(
    "dtype,rows,width,page_bytes,offset_bytes",
    [
        (torch.bfloat16, 128, 512, 131072, 0),
        (torch.bfloat16, 128, 512, 147712, 0),
        (torch.bfloat16, 64, 512, 131072, 0),
        (torch.int8, 64, 128, 131072, 65536),
        (torch.float16, 64, 1, 131072, 73728),
        (torch.int8, 128, 128, 147712, 131072),
        (torch.float16, 128, 1, 147712, 147456),
    ],
    ids=["swa", "padded-swa-or-c1-kv", "c2-kv", "c2-index", "c2-scale", "c1-index", "c1-scale"],
)
@pytest.mark.parametrize("tokens", [3, 257, 1331])
@torch.inference_mode()
def test_v41_a3_cache_scatter_preserves_padded_and_offset_planes(dtype, rows, width, page_bytes, offset_bytes, tokens):
    if uses_a5_packed_cache():
        pytest.skip("A5 uses a different packed cache ABI")
    # These are the actual A3 128-token SWA/C1/C2 page layouts. Keep the
    # destination as a view: cloning it would remove the stride under test.
    blocks = 32
    backing = torch.full((blocks * page_bytes,), 0x35, dtype=torch.uint8, device="npu")
    shape = (blocks, rows, width)
    stride = (page_bytes // dtype.itemsize, width, 1)
    cache = torch.as_strided(backing.view(dtype), shape, stride, offset_bytes // dtype.itemsize)
    slots = torch.arange(tokens, dtype=torch.int32) + 37
    indices = torch.stack((slots // rows, slots % rows), dim=-1)
    indices[::37] = -1
    updates = torch.arange(tokens * width).reshape(tokens, width).remainder(17).sub(8).to(dtype)
    expected = backing.cpu()
    cpu_cache = torch.as_strided(expected.view(dtype), shape, stride, offset_bytes // dtype.itemsize)
    valid = indices[:, 0] >= 0
    cpu_cache[indices[valid, 0].long(), indices[valid, 1].long()] = updates[valid]

    torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, indices.npu(), updates.npu())
    torch.npu.synchronize()
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
