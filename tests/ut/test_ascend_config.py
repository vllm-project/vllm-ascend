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

import os

from transformers import PretrainedConfig
from vllm.config import ModelConfig, ParallelConfig, VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import (_check_torchair_supported,
                                       check_ascend_config,
                                       clear_ascend_config, get_ascend_config,
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

        torchair_graph_config = ascend_config.torchair_graph_config
        self.assertFalse(torchair_graph_config.enabled)
        self.assertEqual(torchair_graph_config.mode, '')
        self.assertFalse(torchair_graph_config.use_cached_graph)
        self.assertEqual(torchair_graph_config.graph_batch_sizes, [])
        self.assertFalse(torchair_graph_config.graph_batch_sizes_init)
        self.assertFalse(torchair_graph_config.enable_multistream_mla)
        self.assertFalse(torchair_graph_config.enable_multistream_moe)
        self.assertTrue(torchair_graph_config.enable_view_optimize)
        self.assertFalse(torchair_graph_config.enable_kv_nz)

        ascend_scheduler_config = ascend_config.ascend_scheduler_config
        self.assertFalse(ascend_scheduler_config.enabled)

    @_clean_up_ascend_config
    def test_init_ascend_config_with_additional_config(self):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "torchair_graph_config": {
                "enabled": True,
                "use_cached_graph": True,
                "graph_batch_sizes": [1, 2, 4],
                "graph_batch_sizes_init": False,
                "enable_multistream_mla": True,
                "enable_multistream_moe": True,
                "enable_view_optimize": True,
                "enable_kv_nz": True
            },
            "ascend_scheduler_config": {
                "enabled": True
            },
            "expert_map_path": "test_expert_map_path",
            "refresh": True,
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(ascend_config.expert_map_path, "test_expert_map_path")

        torchair_graph_config = ascend_config.torchair_graph_config
        self.assertTrue(torchair_graph_config.enabled)
        self.assertTrue(torchair_graph_config.use_cached_graph)
        self.assertEqual(torchair_graph_config.graph_batch_sizes, [1, 2, 4])
        self.assertFalse(torchair_graph_config.graph_batch_sizes_init)
        self.assertTrue(torchair_graph_config.enable_multistream_mla)
        self.assertTrue(torchair_graph_config.enable_multistream_moe)
        self.assertTrue(torchair_graph_config.enable_view_optimize)
        self.assertTrue(torchair_graph_config.enable_kv_nz)

        ascend_scheduler_config = ascend_config.ascend_scheduler_config
        self.assertTrue(ascend_scheduler_config.enabled)

    @_clean_up_ascend_config
    def test_init_ascend_config_with_refresh(self):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertFalse(ascend_config.torchair_graph_config.enabled)

        test_vllm_config.additional_config = {
            "torchair_graph_config": {
                "enabled": True,
            },
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertFalse(ascend_config.torchair_graph_config.enabled)

        test_vllm_config.additional_config = {
            "torchair_graph_config": {
                "enabled": True,
            },
            "refresh": True,
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertTrue(ascend_config.torchair_graph_config.enabled)

    @_clean_up_ascend_config
    def test_init_ascend_config_with_wrong_input(self):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "torchair_graph_config": {
                "enabled": True,
                "graph_batch_sizes": "fake_size",
            },
            "refresh": True,
        }
        with self.assertRaises(TypeError):
            init_ascend_config(test_vllm_config)

        test_vllm_config.additional_config = {
            "torchair_graph_config": {
                "enabled": False,
                "graph_batch_sizes": [1, 2, 4, 8],
                "graph_batch_sizes_init": True,
            },
            "refresh": True,
        }
        with self.assertRaises(ValueError):
            init_ascend_config(test_vllm_config)

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

    @_clean_up_ascend_config
    def test_check_ascend_config_pass(self):
        test_vllm_config = VllmConfig()
        init_ascend_config(test_vllm_config)
        check_ascend_config(test_vllm_config, False)

        test_vllm_config.additional_config = {
            "torchair_graph_config": {
                "enabled": True,
            },
            "refresh": True
        }
        init_ascend_config(test_vllm_config)
        check_ascend_config(test_vllm_config, False)

        test_vllm_config.additional_config = {
            "torchair_graph_config": {
                "enabled": False,
            },
            "refresh": True
        }
        init_ascend_config(test_vllm_config)
        check_ascend_config(test_vllm_config, False)

    @_clean_up_ascend_config
    def test_check_ascend_config_wrong_case(self):
        test_vllm_config = VllmConfig()

        # torchair + eager mode
        with self.assertRaises(RuntimeError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": True,
                },
                "refresh": True
            }
            init_ascend_config(test_vllm_config)
            enforce_eager = True
            check_ascend_config(test_vllm_config, enforce_eager)
        # torchair + non deepseek model
        with self.assertRaises(NotImplementedError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": True,
                },
                "refresh": True
            }
            model_path = os.path.join(os.path.dirname(__file__), "fake_weight")
            fake_model_config = ModelConfig(model=model_path)
            fake_model_config.hf_config = PretrainedConfig()
            fake_model_config.hf_config.model_type = "llama"
            test_vllm_config.model_config = fake_model_config
            init_ascend_config(test_vllm_config)
            check_ascend_config(test_vllm_config, False)
        # aclgraph + deepseek model
        with self.assertRaises(NotImplementedError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": False,
                },
                "refresh": True
            }
            model_path = os.path.join(os.path.dirname(__file__), "fake_weight")
            fake_model_config = ModelConfig(model=model_path)
            fake_model_config.hf_config = PretrainedConfig()
            fake_model_config.hf_config.model_type = "deepseek"
            test_vllm_config.model_config = fake_model_config
            init_ascend_config(test_vllm_config)
            check_ascend_config(test_vllm_config, False)

    def test_check_torchair_supported(self):
        test_cases = [('deepseek_v3', True), ('PanguProMoE', True),
                      ('qwen', True), ('llama', False)]
        for model_type, expected_output in test_cases:
            self.assertEqual(_check_torchair_supported(model_type),
                             expected_output)

    @_clean_up_ascend_config
    def test_ascend_config_load_error(self):
        test_vllm_config = VllmConfig()
        # graph_batch_sizes should be list.
        with self.assertRaises(TypeError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "graph_batch_sizes": "fake_size",
                },
                "refresh": True
            }
            init_ascend_config(test_vllm_config)

        # use_cached_graph should not be enabled without torchair graph mode
        with self.assertRaises(RuntimeError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": False,
                    "use_cached_graph": True,
                },
                "refresh": True
            }
            init_ascend_config(test_vllm_config)

        # use_cached_kv_cache_bytes should not be enabled without torchair graph mode
        with self.assertRaises(RuntimeError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": False,
                    "use_cached_kv_cache_bytes": True,
                },
                "refresh": True
            }
            init_ascend_config(test_vllm_config)

        # graph_batch_sizes should not be set without torchair graph mode
        with self.assertRaises(RuntimeError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": False,
                    "graph_batch_sizes": [1, 2, 4],
                },
                "refresh": True
            }
            init_ascend_config(test_vllm_config)

        # use_cached_kv_cache_bytes is valid only when torchair graph mode and use_cached_graph are enabled
        with self.assertRaises(RuntimeError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": True,
                    "use_cached_graph": False,
                    "use_cached_kv_cache_bytes": True,
                },
                "refresh": True
            }
            init_ascend_config(test_vllm_config)

        # graph_batch_sizes_init should not be enabled without torchair graph mode
        with self.assertRaises(RuntimeError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": False,
                    "graph_batch_sizes_init": True,
                },
                "refresh": True
            }
            init_ascend_config(test_vllm_config)

        # enable_multistream_mla should not be enabled without torchair graph mode
        with self.assertRaises(RuntimeError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": False,
                    "enable_multistream_mla": True,
                },
                "refresh": True
            }
            init_ascend_config(test_vllm_config)

        # enable_multistream_moe should not be enabled without torchair graph mode
        with self.assertRaises(RuntimeError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": False,
                    "enable_multistream_moe": True,
                },
                "refresh": True
            }
            init_ascend_config(test_vllm_config)

        # mode should not be configured without torchair graph mode
        with self.assertRaises(RuntimeError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": False,
                    "mode": 'max-autotune',
                },
                "refresh": True
            }
            init_ascend_config(test_vllm_config)

        # enable_kv_nz should not be enabled without torchair graph mode
        with self.assertRaises(RuntimeError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": False,
                    "enable_kv_nz": True,
                },
                "refresh": True
            }
            init_ascend_config(test_vllm_config)

        with self.assertRaises(AssertionError):
            test_vllm_config.additional_config = {
                "lmhead_tensor_parallel_size": 2,
                "refresh": True
            }
            test_vllm_config.parallel_config = ParallelConfig(
                data_parallel_size=4, tensor_parallel_size=2)
            init_ascend_config(test_vllm_config)

        with self.assertRaises(AssertionError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": True,
                },
                "oproj_tensor_parallel_size": 2,
                "refresh": True
            }
            test_vllm_config.parallel_config = ParallelConfig(
                data_parallel_size=4, tensor_parallel_size=2)
            init_ascend_config(test_vllm_config)

        with self.assertRaises(AssertionError):
            test_vllm_config.additional_config = {
                "torchair_graph_config": {
                    "enabled": False,
                },
                "oproj_tensor_parallel_size": 2,
                "refresh": True
            }
            test_vllm_config.parallel_config = ParallelConfig(
                data_parallel_size=4, tensor_parallel_size=1)
            init_ascend_config(test_vllm_config)
