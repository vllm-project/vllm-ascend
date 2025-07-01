import os
import unittest
from unittest import mock

from transformers import PretrainedConfig
from vllm.config import ModelConfig, VllmConfig

from vllm_ascend.ascend_config import (check_ascend_config,
                                       check_torchair_supported,
                                       clear_ascend_config, get_ascend_config,
                                       init_ascend_config)


class TestAscendConfig(unittest.TestCase):

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
        self.assertEqual(ascend_config.expert_tensor_parallel_size, 0)
        self.assertIsNone(ascend_config.expert_map_path)

        torchair_graph_config = ascend_config.torchair_graph_config
        self.assertFalse(torchair_graph_config.enabled)
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
            "expert_tensor_parallel_size": 1,
            "expert_map_path": "test_expert_map_path",
            "refresh": True
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(ascend_config.expert_tensor_parallel_size, 1)
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

        # For V1 engine
        with mock.patch.dict(os.environ, {"VLLM_USE_V1": "1"}):
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
        # For V0 engine
        with mock.patch.dict(os.environ, {"VLLM_USE_V1": "0"}):
            with self.assertRaises(NotImplementedError):
                test_vllm_config.additional_config = {
                    "torchair_graph_config": {
                        "enabled": True,
                    },
                    "refresh": True
                }
                init_ascend_config(test_vllm_config)
                check_ascend_config(test_vllm_config, False)
            with self.assertRaises(NotImplementedError):
                test_vllm_config.additional_config = {
                    "ascend_scheduler_config": {
                        "enabled": True,
                    },
                    "refresh": True
                }
                init_ascend_config(test_vllm_config)
                check_ascend_config(test_vllm_config, True)
        # For V1 engine
        with mock.patch.dict(os.environ, {"VLLM_USE_V1": "1"}):
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
                model_path = os.path.join(os.path.dirname(__file__),
                                          "fake_weight")
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
                model_path = os.path.join(os.path.dirname(__file__),
                                          "fake_weight")
                fake_model_config = ModelConfig(model=model_path)
                fake_model_config.hf_config = PretrainedConfig()
                fake_model_config.hf_config.model_type = "deepseek"
                test_vllm_config.model_config = fake_model_config
                init_ascend_config(test_vllm_config)
                check_ascend_config(test_vllm_config, False)

    def test_check_torchair_supported(self):
        test_cases = [('deepseek_v3', True), ('PanguProMoE', True),
                      ('qwen', False), ('llama', False)]
        for model_type, expected_output in test_cases:
            self.assertEqual(check_torchair_supported(model_type),
                             expected_output)
