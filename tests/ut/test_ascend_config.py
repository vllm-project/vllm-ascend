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

import json
import os
from unittest.mock import patch

from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import clear_ascend_config, get_ascend_config, init_ascend_config
from vllm_ascend.utils import clear_enable_sp, enable_sp, get_flashcomm2_config_and_validate


class TestAscendConfig(TestBase):
    @staticmethod
    def _clean_up_ascend_config(func):
        def wrapper(*args, **kwargs):
            clear_ascend_config()
            clear_enable_sp()
            try:
                func(*args, **kwargs)
            finally:
                clear_ascend_config()
                clear_enable_sp()

        return wrapper

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
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
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
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
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_enable_npugraph_ex(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {"enable_npugraph_ex": True, "enable_static_kernel": True},
            "refresh": True,
        }
        ascend_compilation_config = init_ascend_config(test_vllm_config).ascend_compilation_config
        self.assertTrue(ascend_compilation_config.enable_npugraph_ex)
        self.assertTrue(ascend_compilation_config.enable_static_kernel)

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.info_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_migrated_config_falls_back_to_envs(self, mock_fix_incompatible_config, mock_info_once):
        test_vllm_config = VllmConfig()
        test_vllm_config.parallel_config.tensor_parallel_size = 4
        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "1",
                "VLLM_ASCEND_ENABLE_FUSED_MC2": "2",
                "VLLM_ASCEND_ENABLE_MLAPO": "0",
                "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
                "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": "2",
                "MSMONITOR_USE_DAEMON": "1",
                "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": "0",
                "VLLM_ASCEND_ENABLE_NZ": "2",
            },
        ):
            ascend_config = init_ascend_config(test_vllm_config)

        self.assertTrue(ascend_config.enable_matmul_allreduce)
        self.assertEqual(ascend_config.enable_fused_mc2, 2)
        self.assertFalse(ascend_config.enable_mlapo)
        self.assertTrue(ascend_config.enable_flashcomm1)
        self.assertEqual(ascend_config.enable_flashcomm2_parallel_size, 2)
        self.assertTrue(ascend_config.msmonitor_use_daemon)
        self.assertFalse(ascend_config.enable_transpose_kv_cache_by_block)
        self.assertEqual(ascend_config.weight_nz_mode, 2)
        mock_info_once.assert_any_call(
            "AscendConfig.enable_mlapo falls back to environment variable VLLM_ASCEND_ENABLE_MLAPO with value False. "
            "Please use additional_config.enable_mlapo instead, because VLLM_ASCEND_ENABLE_MLAPO will be "
            "removed in the next release."
        )
        mock_info_once.assert_any_call(
            "AscendConfig.weight_nz_mode falls back to environment variable VLLM_ASCEND_ENABLE_NZ with value 2. "
            "Please use additional_config.weight_nz_mode instead, because VLLM_ASCEND_ENABLE_NZ will be removed "
            "in the next release."
        )

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.info_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_migrated_config_skips_default_env_fallback_logs(self, mock_fix_incompatible_config, mock_info_once):
        test_vllm_config = VllmConfig()
        with patch.dict(os.environ, {}, clear=True):
            init_ascend_config(test_vllm_config)

        fallback_logs = [
            call.args[0]
            for call in mock_info_once.call_args_list
            if "falls back to environment variable" in call.args[0]
        ]
        self.assertEqual(fallback_logs, [])

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.info_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_migrated_config_overrides_envs(self, mock_fix_incompatible_config, mock_info_once):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "enable_matmul_allreduce": False,
            "enable_fused_mc2": 0,
            "enable_mlapo": True,
            "enable_flashcomm1": False,
            "enable_flashcomm2_parallel_size": 0,
            "msmonitor_use_daemon": False,
            "enable_transpose_kv_cache_by_block": True,
            "weight_nz_mode": 1,
        }
        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": "1",
                "VLLM_ASCEND_ENABLE_FUSED_MC2": "2",
                "VLLM_ASCEND_ENABLE_MLAPO": "0",
                "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
                "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE": "2",
                "MSMONITOR_USE_DAEMON": "1",
                "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": "0",
                "VLLM_ASCEND_ENABLE_NZ": "2",
            },
        ):
            ascend_config = init_ascend_config(test_vllm_config)

        self.assertFalse(ascend_config.enable_matmul_allreduce)
        self.assertEqual(ascend_config.enable_fused_mc2, 0)
        self.assertTrue(ascend_config.enable_mlapo)
        self.assertFalse(ascend_config.enable_flashcomm1)
        self.assertEqual(ascend_config.enable_flashcomm2_parallel_size, 0)
        self.assertFalse(ascend_config.msmonitor_use_daemon)
        self.assertTrue(ascend_config.enable_transpose_kv_cache_by_block)
        self.assertEqual(ascend_config.weight_nz_mode, 1)
        mock_info_once.assert_any_call("AscendConfig.enable_mlapo is set from additional_config with value True.")
        mock_info_once.assert_any_call("AscendConfig.weight_nz_mode is set from additional_config with value 1.")

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"}, clear=True)
    def test_enable_flashcomm1_config_overrides_disabled_env(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"enable_flashcomm1": True}
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "0"}, clear=True):
            ascend_config = init_ascend_config(test_vllm_config)
        self.assertTrue(ascend_config.enable_flashcomm1)
        self.assertTrue(enable_sp(test_vllm_config))

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_enable_sp_falls_back_to_env_without_current_config(self, mock_check_and_update_config):
        clear_enable_sp()
        with (
            patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"}),
            patch("vllm.config.get_current_vllm_config", side_effect=AssertionError),
        ):
            self.assertTrue(enable_sp())

    @_clean_up_ascend_config
    @patch("vllm_ascend.utils.logger.warning_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_flashcomm2_warning_uses_enable_flashcomm1_config(self, mock_check_and_update_config, mock_warning_once):
        test_vllm_config = VllmConfig()
        test_vllm_config.parallel_config.tensor_parallel_size = 4
        test_vllm_config.kv_transfer_config = None
        ascend_config = type(
            "MockAscendConfig",
            (),
            {
                "enable_flashcomm2_parallel_size": 2,
                "layer_sharding": None,
                "enable_flashcomm1": True,
                "finegrained_tp_config": type("MockFinegrainedTPConfig", (), {"oproj_tensor_parallel_size": 0})(),
            },
        )()

        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "0"}):
            self.assertEqual(get_flashcomm2_config_and_validate(ascend_config, test_vllm_config), 2)

        flashcomm1_warning = (
            "It is recommended to enable FLASHCOMM1 simultaneously when starting FLASHCOMM2 for optimal performance."
        )
        self.assertNotIn(flashcomm1_warning, [call.args[0] for call in mock_warning_once.call_args_list])

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_get_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)

    @_clean_up_ascend_config
    def test_get_ascend_config_without_init(self):
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_clear_ascend_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)
        clear_ascend_config()
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_with_dump_config_materializes_fixed_file(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        dump_config = {"task": "tensor", "level": "L1", "dump_path": "/tmp/msprobe_dump"}
        test_vllm_config.additional_config = {"dump_config": dump_config}

        ascend_config = init_ascend_config(test_vllm_config)
        self.assertIsNotNone(ascend_config.dump_config_path)
        assert ascend_config.dump_config_path is not None
        expected_path = os.path.join(os.getcwd(), ".vllm_ascend", "msprobe", "msprobe_dump_config.json")
        self.assertEqual(ascend_config.dump_config_path, expected_path)
        self.assertTrue(os.path.exists(ascend_config.dump_config_path))
        with open(ascend_config.dump_config_path, encoding="utf-8") as file:
            persisted = json.load(file)
        self.assertEqual(persisted, dump_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_dump_config_and_path_conflict(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"dump_config_path": "/tmp/config.json", "dump_config": {"task": "tensor"}}
        with self.assertRaises(ValueError):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_dump_config_type_validation(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"dump_config": "/tmp/config.json"}
        with self.assertRaises(ValueError):
            init_ascend_config(test_vllm_config)
