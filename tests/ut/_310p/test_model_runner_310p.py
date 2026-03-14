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

import unittest

from vllm_ascend._310p.model_runner_310p import (
    get_310p_attention_kernel_block_sizes,
)


class TestModelRunner310KernelBlockSize(unittest.TestCase):
    def test_large_head_hybrid_prefers_split_kernel_block(self):
        kernel_block_sizes = get_310p_attention_kernel_block_sizes(
            block_size=128,
            head_size=256,
            backend_block_sizes=[128],
            use_hybrid_blocks=True,
        )

        self.assertEqual(kernel_block_sizes, [64, 128])

    def test_non_hybrid_keeps_physical_block_size(self):
        kernel_block_sizes = get_310p_attention_kernel_block_sizes(
            block_size=128,
            head_size=256,
            backend_block_sizes=[128],
            use_hybrid_blocks=False,
        )

        self.assertEqual(kernel_block_sizes, [128])

    def test_small_head_keeps_backend_block_size_order(self):
        kernel_block_sizes = get_310p_attention_kernel_block_sizes(
            block_size=128,
            head_size=128,
            backend_block_sizes=[128],
            use_hybrid_blocks=True,
        )

        self.assertEqual(kernel_block_sizes, [128])


if __name__ == "__main__":
    unittest.main()
