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

import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.utils import AscendKVBlockZeroer


def test_zero_block_ids_supports_non_uniform_page_sizes() -> None:
    """Each segment must use its own block stride when zeroing KV cache."""
    # Serving initializes this once during worker startup. Standalone Triton
    # tests must initialize it before AscendKVBlockZeroer queries vector cores.
    init_device_properties_triton()
    device = torch.device("npu")
    num_blocks = 4
    k_cache = torch.ones((num_blocks, 16384), dtype=torch.int32, device=device)
    v_cache = torch.ones((num_blocks, 4096), dtype=torch.int32, device=device)
    zeroer = AscendKVBlockZeroer(device, pin_memory=False)
    zeroer._meta = (
        torch.tensor(
            [k_cache.data_ptr(), v_cache.data_ptr()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor([16384, 4096], dtype=torch.int64, device=device),
        4,
        4096,
        2,
    )

    zeroer.zero_block_ids([1, 2])
    torch.npu.synchronize()

    for cache in (k_cache, v_cache):
        assert torch.all(cache[0] == 1)
        assert torch.all(cache[1] == 0)
        assert torch.all(cache[2] == 0)
        assert torch.all(cache[3] == 1)
