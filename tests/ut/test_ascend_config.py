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

from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import (clear_ascend_config, get_ascend_config,
                                       init_ascend_config)


class TestAscendConfig(TestBase):

    @staticmethod
    def _clean_up_ascend_config(func):

        def wrapper(*args, **kwargs):
            clear_ascend_config()
            func(*args, **kwargs)
            clear_ascend_config()

        return wrapper

    @_clean_up_ascend_config
    def test_init_ascend_config_without_additional_config(self):
        test_vllm_config = VllmConfig()
        # No additional config given, check the default value here.
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertIsNone(ascend_config.expert_map_path)
        self.assertFalse(ascend_config.multistream_overlap_shared_expert)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertTrue(ascend_compilation_config.enable_quantization_fusion)

    @_clean_up_ascend_config
    def test_init_ascend_config_with_additional_config(self):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "enable_quantization_fusion": False,
            },
            "multistream_overlap_shared_expert": True,
            "expert_map_path": "test_expert_map_path",
            "refresh": True,
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(ascend_config.expert_map_path, "test_expert_map_path")
        self.assertTrue(ascend_config.multistream_overlap_shared_expert)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertFalse(ascend_compilation_config.enable_quantization_fusion)

    @_clean_up_ascend_config
    def test_get_ascend_config(self):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)

    @_clean_up_ascend_config
    def test_get_ascend_config_without_init(self):
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    def test_clear_ascend_config(self):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)
        clear_ascend_config()
        with self.assertRaises(RuntimeError):
            get_ascend_config()
