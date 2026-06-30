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

from unittest.mock import patch

from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import (LapsConfig, clear_ascend_config,
                                       get_ascend_config, init_ascend_config)


class TestAscendConfig(TestBase):
    @staticmethod
    def _clean_up_ascend_config(func):
        def wrapper(*args, **kwargs):
            clear_ascend_config()
            func(*args, **kwargs)
            clear_ascend_config()

        return wrapper

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_without_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        # No additional config given, check the default value here.
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertFalse(ascend_config.multistream_overlap_shared_expert)
        self.assertFalse(ascend_config.enable_kv_nz)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertTrue(ascend_compilation_config.fuse_norm_quant)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertTrue(ascend_fusion_config.fusion_ops_gmmswigluquant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_with_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
            },
            "ascend_fusion_config": {
                "fusion_ops_gmmswigluquant": False,
            },
            "multistream_overlap_shared_expert": True,
            "eplb_config": {"num_redundant_experts": 2},
            "refresh": True,
            "enable_kv_nz": False,
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(ascend_config.eplb_config.num_redundant_experts, 2)
        self.assertTrue(ascend_config.multistream_overlap_shared_expert)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertFalse(ascend_compilation_config.fuse_norm_quant)
        self.assertFalse(ascend_config.enable_kv_nz)
        self.assertTrue(ascend_compilation_config.enable_npugraph_ex)
        self.assertFalse(ascend_compilation_config.enable_static_kernel)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertFalse(ascend_fusion_config.fusion_ops_gmmswigluquant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_init_ascend_config_enable_npugraph_ex(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "enable_npugraph_ex": True,
                "enable_static_kernel": True
            },
            "refresh": True
        }
        ascend_compilation_config = init_ascend_config(
            test_vllm_config).ascend_compilation_config
        self.assertTrue(ascend_compilation_config.enable_npugraph_ex)
        self.assertTrue(ascend_compilation_config.enable_static_kernel)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_get_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)

    @_clean_up_ascend_config
    def test_get_ascend_config_without_init(self):
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform._fix_incompatible_config")
    def test_clear_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)
        clear_ascend_config()
        with self.assertRaises(RuntimeError):
            get_ascend_config()


class TestLapsConfig(TestBase):
    def test_default_is_disabled(self):
        cfg = LapsConfig({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.threshold, 256)
        self.assertEqual(cfg.long_max_wait_ms, 0.0)
        self.assertEqual(cfg.long_token_reservation, 0.0)
        self.assertEqual(cfg.long_burst_steps, 4)
        self.assertEqual(cfg.stats_log_interval_s, 0.0)

    def test_explicit_config(self):
        cfg = LapsConfig({
            "enabled": True,
            "threshold": 512,
            "long_max_wait_ms": 2000,
            "long_token_reservation": 0.2,
            "long_burst_steps": 8,
            "stats_log_interval_s": 5,
        })
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.threshold, 512)
        self.assertEqual(cfg.long_max_wait_ms, 2000.0)
        self.assertEqual(cfg.long_token_reservation, 0.2)
        self.assertEqual(cfg.long_burst_steps, 8)
        self.assertEqual(cfg.stats_log_interval_s, 5.0)

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            LapsConfig({"foo": 1})

    def test_validation_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            LapsConfig({"long_token_reservation": 1.5})
        with self.assertRaises(ValueError):
            LapsConfig({"threshold": -1})
        with self.assertRaises(ValueError):
            LapsConfig({"long_max_wait_ms": -1})
        with self.assertRaises(ValueError):
            LapsConfig({"long_burst_steps": 0})
        with self.assertRaises(ValueError):
            LapsConfig({"stats_log_interval_s": -1})

    def test_aging_requires_positive_reservation(self):
        # long_max_wait_ms > 0 (aging on) with reservation == 0 is rejected: the
        # token bucket is the only aged-long admission channel, so a zero
        # reservation would make aging inert / degenerate to long-first.
        with self.assertRaises(ValueError):
            LapsConfig({"long_max_wait_ms": 2000})
        with self.assertRaises(ValueError):
            LapsConfig({"long_max_wait_ms": 2000, "long_token_reservation": 0.0})
        # With a positive reservation it is accepted.
        cfg = LapsConfig({"long_max_wait_ms": 2000, "long_token_reservation": 0.1})
        self.assertEqual(cfg.long_max_wait_ms, 2000.0)
        self.assertEqual(cfg.long_token_reservation, 0.1)

    def test_none_config_is_disabled(self):
        # No block at all (None) -> defaults, scheduling disabled.
        cfg = LapsConfig(None)
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.threshold, 256)
