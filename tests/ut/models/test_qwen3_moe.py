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
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from vllm_ascend.models.qwen3_moe import CustomQwen3MoeForCausalLM


class TestCustomQwen3MoeForCausalLM:

    def test_class_inheritance(self):
        assert issubclass(CustomQwen3MoeForCausalLM, Qwen3MoeForCausalLM)

    @pytest.mark.parametrize("key, expected", [
        ("qkv_proj", ["q_proj", "k_proj", "v_proj"]),
        ("gate_up_proj", ["gate_proj", "up_proj"]),
        ("experts",
         ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]),
    ])
    def test_packed_modules_mapping(self, key, expected):
        assert CustomQwen3MoeForCausalLM.packed_modules_mapping[
            key] == expected

    def test_packed_modules_mapping_structure(self):
        expected_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
            "experts": [
                "experts.0.gate_proj", "experts.0.up_proj",
                "experts.0.down_proj"
            ]
        }
        assert CustomQwen3MoeForCausalLM.packed_modules_mapping == expected_mapping
