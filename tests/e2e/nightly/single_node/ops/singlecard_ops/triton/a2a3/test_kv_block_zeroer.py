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

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.utils import AscendKVBlockZeroer


@pytest.fixture(scope="module", autouse=True)
def init_triton_device_properties() -> None:
    init_device_properties_triton()


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
