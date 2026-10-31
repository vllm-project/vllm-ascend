#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v1.kv_block_zeroer import AscendKVBlockZeroer


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties() -> None:
    init_device_properties_triton()


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)
@pytest.mark.parametrize(
    ("page_sizes", "block_size"),
    [
        pytest.param((16384, 4096), 4096, id="non-uniform"),
        pytest.param((16384, 16384), 8192, id="uniform"),
    ],
)
def test_zero_block_ids(page_sizes: tuple[int, int], block_size: int) -> None:
    """Only the requested blocks are zeroed for every KV cache segment."""
    device = torch.device("npu")
    num_blocks = 5
    block_ids = [0, 2, num_blocks - 1]
    caches = [torch.ones((num_blocks, page_size), dtype=torch.int32, device=device) for page_size in page_sizes]
    zeroer = AscendKVBlockZeroer(device, pin_memory=False)
    zeroer._meta = (
        torch.tensor(
            [cache.data_ptr() for cache in caches],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor(page_sizes, dtype=torch.int64, device=device),
        max(page_sizes) // block_size,
        block_size,
        len(caches),
    )

    zeroer.zero_block_ids(block_ids)
    torch.npu.synchronize()

    for cache in caches:
        expected = torch.ones_like(cache)
        expected[block_ids] = 0
        assert torch.equal(cache, expected)


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="MXFP cache regression targets Ascend A5",
)
@pytest.mark.parametrize("ratio", [1, 2])
def test_mxfp_zero_preserves_static_scales_and_padding(ratio):
    """Clear K/V and K scales without changing static V scales or page gaps."""
    # Load the MXFP layout helpers only for this A5 regression.
    from vllm_ascend.attention.attention_c8_mxfp import mxfp_cache_spec, mxfp_cache_views_for_spec

    device = torch.device("npu")
    kernel_block_size = 512
    spec = mxfp_cache_spec(
        FullAttentionSpec(
            block_size=kernel_block_size * ratio,
            num_kv_heads=1,
            head_size=256,
            dtype=torch.float8_e4m3fn,
        )
    )
    page_size = spec.page_size_bytes + ratio * 256
    spec = replace(spec, page_size_padded=page_size)
    offset, guard, num_pages = 128, 256, 3
    backing = torch.full((offset + num_pages * page_size + guard,), 91, dtype=torch.int8, device=device)
    caches = mxfp_cache_views_for_spec(backing[offset:-guard], spec, kernel_block_size)
    caches[3].fill_(127)
    static_scale_before = caches[3].clone()

    expected = backing.cpu()
    expected_caches = mxfp_cache_views_for_spec(expected[offset:-guard], spec, kernel_block_size)
    block_id = 2
    for cache in expected_caches[:3]:
        cache[block_id * ratio : (block_id + 1) * ratio].view(torch.uint8).zero_()

    zeroer = AscendKVBlockZeroer(device, pin_memory=False)
    group = SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=0, layer_names=["attn"])
    zeroer.init_meta([group], [kernel_block_size], "mxfp8", set(), {"attn": SimpleNamespace(kv_cache=caches)})
    zeroer.zero_block_ids([block_id])
    torch.npu.synchronize()

    assert torch.equal(caches[3], static_scale_before)
    for cache in caches[:3]:
        assert (cache[block_id * ratio : (block_id + 1) * ratio].view(torch.uint8) == 0).all()
    # Check the complete allocation, including other pages and every padding byte.
    torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)
