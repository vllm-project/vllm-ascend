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
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from vllm.config import KVTransferConfig, VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import (
    AscendConfig,
    SchedulerConfig,
    ShortRequestFirstConfig,
    clear_ascend_config,
    get_ascend_config,
    init_ascend_config,
)
from vllm_ascend.utils import clear_enable_sp, enable_sp


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

    @staticmethod
    def _make_model_config(
        total_num_attention_heads: int = 32,
        total_num_kv_heads: int = 8,
        is_deepseek_mla: bool = False,
    ):
        return SimpleNamespace(
            is_deepseek_mla=is_deepseek_mla,
            use_mla=is_deepseek_mla,
            enforce_eager=True,
            model_arch_config=SimpleNamespace(total_num_attention_heads=total_num_attention_heads),
            get_total_num_kv_heads=lambda: total_num_kv_heads,
        )

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
    def test_init_ascend_config_with_nested_scheduler_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "scheduler_config": {
                "enable_balance_scheduling": True,
                "recompute_scheduler_enable": True,
                "short_request_first_config": {"enabled": True, "threshold": 512},
                "profiling_chunk_config": {"enabled": False},
            }
        }

        scheduler_config = init_ascend_config(test_vllm_config).scheduler_config

        self.assertTrue(scheduler_config.enable_balance_scheduling)
        self.assertTrue(scheduler_config.recompute_scheduler_enable)
        self.assertTrue(scheduler_config.short_request_first_config.enabled)
        self.assertEqual(scheduler_config.short_request_first_config.threshold, 512)
        self.assertFalse(scheduler_config.profiling_chunk_config.enabled)

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
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_rejects_mooncake_c8_kv_cache_consumer(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeConnectorV1",
            kv_role="kv_consumer",
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config()

        with self.assertRaisesRegex(ValueError, "does not support C8 KV cache quantization"):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_rejects_multi_connector_mooncake_c8_consumer(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_consumer",
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "MooncakeConnectorV1",
                        "kv_role": "kv_consumer",
                    }
                ]
            },
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config()

        with self.assertRaisesRegex(ValueError, "does not support C8 KV cache quantization"):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_allows_layerwise_c8_kv_cache_consumer(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeLayerwiseConnector",
            kv_role="kv_consumer",
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config()

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertIsNotNone(ascend_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_allows_mha_mooncake_c8_kv_cache_consumer(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeConnectorV1",
            kv_role="kv_consumer",
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config(
            total_num_attention_heads=8,
            total_num_kv_heads=8,
        )

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertIsNotNone(ascend_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_rejects_mooncake_c8_kv_cache_producer(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeConnectorV1",
            kv_role="kv_producer",
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config()

        with self.assertRaisesRegex(ValueError, "does not support C8 KV cache quantization"):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_rejects_mooncake_c8_kv_cache_both_role(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeConnectorV1",
            kv_role="kv_both",
        )
        test_vllm_config.quant_config = SimpleNamespace(enable_c8_quant=True)
        test_vllm_config.model_config = self._make_model_config()

        with self.assertRaisesRegex(ValueError, "does not support C8 KV cache quantization"):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.warning")
    @patch("vllm_ascend.utils.is_310p", return_value=True)
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_disable_npugraph_ex_on_310p(
        self, mock_fix_incompatible_config, mock_is_310p, mock_warning
    ):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {"enable_npugraph_ex": True, "enable_static_kernel": True},
            "refresh": True,
        }

        ascend_compilation_config = init_ascend_config(test_vllm_config).ascend_compilation_config

        self.assertFalse(ascend_compilation_config.enable_npugraph_ex)
        self.assertFalse(ascend_compilation_config.enable_static_kernel)
        warning_messages = [call.args[0] for call in mock_warning.call_args_list]
        self.assertIn("npugraph_ex is not supported on Ascend 310P. Disabling it.", warning_messages)
        self.assertIn(
            "static kernel requires npugraph_ex, which is not supported on Ascend 310P. Disabling it.",
            warning_messages,
        )

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.info_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_migrated_config_falls_back_to_envs(self, mock_fix_incompatible_config, mock_info_once):
        test_vllm_config = VllmConfig()
        test_vllm_config.parallel_config.tensor_parallel_size = 4
        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_ENABLE_FUSED_MC2": "1",
                "VLLM_ASCEND_ENABLE_MLAPO": "0",
                "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
                "MSMONITOR_USE_DAEMON": "1",
                "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": "0",
                "VLLM_ASCEND_ENABLE_NZ": "2",
            },
        ):
            ascend_config = init_ascend_config(test_vllm_config)

        self.assertEqual(ascend_config.enable_fused_mc2, 1)
        self.assertFalse(ascend_config.enable_mlapo)
        self.assertTrue(ascend_config.enable_flashcomm1)
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
            "enable_fused_mc2": 0,
            "enable_mlapo": True,
            "enable_flashcomm1": False,
            "msmonitor_use_daemon": False,
            "enable_transpose_kv_cache_by_block": True,
            "weight_nz_mode": 1,
        }
        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_ENABLE_FUSED_MC2": "1",
                "VLLM_ASCEND_ENABLE_MLAPO": "0",
                "VLLM_ASCEND_ENABLE_FLASHCOMM1": "1",
                "MSMONITOR_USE_DAEMON": "1",
                "VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK": "0",
                "VLLM_ASCEND_ENABLE_NZ": "2",
            },
        ):
            ascend_config = init_ascend_config(test_vllm_config)

        self.assertEqual(ascend_config.enable_fused_mc2, 0)
        self.assertTrue(ascend_config.enable_mlapo)
        self.assertFalse(ascend_config.enable_flashcomm1)
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
        with tempfile.TemporaryDirectory() as td:
            with patch("os.getcwd", return_value=td):
                ascend_config = init_ascend_config(test_vllm_config)
                self.assertIsNotNone(ascend_config.dump_config_path)
                assert ascend_config.dump_config_path is not None
                expected_path = os.path.join(td, ".vllm_ascend", "msprobe", "msprobe_dump_config.json")
                self.assertEqual(ascend_config.dump_config_path, expected_path)
                self.assertTrue(os.path.exists(ascend_config.dump_config_path))
                with open(ascend_config.dump_config_path, encoding="utf-8") as file:
                    persisted = json.load(file)
                self.assertEqual(persisted, dump_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_dump_config_isolated_by_dp_rank(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        dump_config = {"task": "tensor", "level": "L1", "dump_path": "/tmp/msprobe_dump"}
        test_vllm_config.additional_config = {"dump_config": dump_config}
        with tempfile.TemporaryDirectory() as td:
            with patch("os.getcwd", return_value=td):
                with patch.dict(os.environ, {"VLLM_DP_RANK": "1"}, clear=False):
                    ascend_config = init_ascend_config(test_vllm_config)

                self.assertIsNotNone(ascend_config.dump_config_path)
                assert ascend_config.dump_config_path is not None
                expected_path = os.path.join(td, ".vllm_ascend", "msprobe", "dp1", "msprobe_dump_config.json")
                shared_path = os.path.join(td, ".vllm_ascend", "msprobe", "msprobe_dump_config.json")
                self.assertEqual(ascend_config.dump_config_path, expected_path)
                self.assertFalse(os.path.exists(shared_path))
                with open(ascend_config.dump_config_path, encoding="utf-8") as file:
                    persisted = json.load(file)
                self.assertEqual(persisted["dump_path"], "/tmp/msprobe_dump/dp1")

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_dump_config_path_isolated_by_dp_rank(self, mock_fix_incompatible_config):
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "msprobe_dump_config.json")
            with open(src_path, "w", encoding="utf-8") as f:
                json.dump({"task": "tensor", "level": "L1", "dump_path": "/tmp/msprobe_dump"}, f)

            test_vllm_config = VllmConfig()
            test_vllm_config.additional_config = {"dump_config_path": src_path}

            with patch.dict(os.environ, {"VLLM_DP_RANK": "2"}, clear=False):
                ascend_config = init_ascend_config(test_vllm_config)

        self.assertIsNotNone(ascend_config.dump_config_path)
        assert ascend_config.dump_config_path is not None
        expected_path = os.path.join(os.path.dirname(src_path), "dp2", "msprobe_dump_config.json")
        self.assertEqual(ascend_config.dump_config_path, expected_path)
        with open(ascend_config.dump_config_path, encoding="utf-8") as file:
            persisted = json.load(file)
        self.assertEqual(persisted["dump_path"], "/tmp/msprobe_dump/dp2")

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_can_disable_dump_config_isolation(self, mock_fix_incompatible_config):
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "msprobe_dump_config.json")
            with open(src_path, "w", encoding="utf-8") as f:
                json.dump({"task": "tensor", "level": "L1", "dump_path": "/tmp/msprobe_dump"}, f)

            test_vllm_config = VllmConfig()
            test_vllm_config.additional_config = {
                "dump_config_path": src_path,
                "dump_config_isolate_by_dp": False,
            }

            with patch.dict(os.environ, {"VLLM_DP_RANK": "3"}, clear=False):
                ascend_config = init_ascend_config(test_vllm_config)

        self.assertEqual(ascend_config.dump_config_path, src_path)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_dfx_config_path_isolated_by_dp_rank(self, mock_fix_incompatible_config):
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "dfx_config.json")
            with open(src_path, "w", encoding="utf-8") as f:
                json.dump({"sync_mode": "broadcast", "dump": {"enabled": False}}, f)

            test_vllm_config = VllmConfig()
            test_vllm_config.additional_config = {"dfx_config_path": src_path}

            with patch.dict(os.environ, {"VLLM_DP_RANK": "2"}, clear=False):
                ascend_config = init_ascend_config(test_vllm_config)

            self.assertIsNotNone(ascend_config.dfx_config_path)
            assert ascend_config.dfx_config_path is not None
            expected_path = os.path.join(os.path.dirname(src_path), "dp2", "dfx_config.json")
            self.assertEqual(ascend_config.dfx_config_path, expected_path)
            with open(ascend_config.dfx_config_path, encoding="utf-8") as file:
                persisted = json.load(file)
            self.assertEqual(persisted["sync_mode"], "broadcast")
            self.assertIn("dump", persisted)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_can_disable_dfx_config_isolation(self, mock_fix_incompatible_config):
        with tempfile.TemporaryDirectory() as td:
            src_path = os.path.join(td, "dfx_config.json")
            with open(src_path, "w", encoding="utf-8") as f:
                json.dump({"sync_mode": "broadcast", "dump": {"enabled": False}}, f)

            test_vllm_config = VllmConfig()
            test_vllm_config.additional_config = {
                "dfx_config_path": src_path,
                "dfx_config_isolate_by_dp": False,
            }

            with patch.dict(os.environ, {"VLLM_DP_RANK": "3"}, clear=False):
                ascend_config = init_ascend_config(test_vllm_config)

        self.assertEqual(ascend_config.dfx_config_path, src_path)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_default_dfx_path_isolated_by_dp_rank(self, mock_fix_incompatible_config):
        with tempfile.TemporaryDirectory() as td:
            test_vllm_config = VllmConfig()
            test_vllm_config.additional_config = {}

            with patch("os.getcwd", return_value=td):
                with patch.dict(os.environ, {"VLLM_DP_RANK": "4"}, clear=False):
                    ascend_config = init_ascend_config(test_vllm_config)

            expected_path = os.path.join(td, "dfx", "config", "dp4", "dfx_config.json")
            self.assertEqual(ascend_config.dfx_config_path, expected_path)
            self.assertTrue(os.path.exists(expected_path))

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_dump_config_path_missing_fails_when_isolating(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "dump_config_path": "/tmp/not_found_msprobe_dump_config.json",
            "dump_config_isolate_by_dp": True,
        }

        with patch.dict(os.environ, {"VLLM_DP_RANK": "0"}, clear=False):
            with self.assertRaises(FileNotFoundError):
                init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_dfx_config_path_missing_fails_when_isolating(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "dfx_config_path": "/tmp/not_found_dfx_config.json",
            "dfx_config_isolate_by_dp": True,
        }

        with patch.dict(os.environ, {"VLLM_DP_RANK": "0"}, clear=False):
            with self.assertRaises(FileNotFoundError):
                init_ascend_config(test_vllm_config)

    def test_is_materialize_writer_tp_pp_election(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(AscendConfig._is_materialize_writer())

        with patch.dict(os.environ, {"VLLM_PP_RANK": "0", "VLLM_TP_RANK": "0"}, clear=True):
            self.assertTrue(AscendConfig._is_materialize_writer())

        with patch.dict(os.environ, {"VLLM_PP_RANK": "0", "VLLM_TP_RANK": "1"}, clear=True):
            self.assertFalse(AscendConfig._is_materialize_writer())

        with patch.dict(os.environ, {"PP_RANK": "1", "TP_RANK": "0"}, clear=True):
            self.assertFalse(AscendConfig._is_materialize_writer())

        with patch.dict(os.environ, {"VLLM_TP_RANK": "1"}, clear=True):
            self.assertFalse(AscendConfig._is_materialize_writer())

        # PP0 + TP1 via alternate env name must not elect writer.
        with patch.dict(os.environ, {"VLLM_PP_RANK": "0", "TP_RANK": "1"}, clear=True):
            self.assertFalse(AscendConfig._is_materialize_writer())
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

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_ignores_legacy_dynamic_dump_config(self, mock_fix_incompatible_config):
        """Legacy additional_config.dynamic_dump_config is dropped; DFX uses JSON only."""
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "dynamic_dump_config": {
                "spec_acceptance_window": 20,
                "dynamic_dump_cooldown_seconds": 120,
                "dynamic_dump_max_times": 3,
            }
        }

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertFalse(hasattr(ascend_config, "dynamic_dump_config"))
        # Live DFX knobs come only from DFX JSON / defaults — not dynamic_dump_config.
        self.assertEqual(ascend_config.dfx_config.detector_get("spec_acceptance", "window"), 10)
        self.assertEqual(ascend_config.dfx_config.dump_max_times(), 0)
        self.assertEqual(ascend_config.dfx_config.dump_cooldown_seconds(), 300)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_recreates_for_new_vllm_config(self, mock_fix_incompatible_config):
        first_vllm_config = VllmConfig()
        first_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "enable_npugraph_ex": False,
            }
        }
        first_ascend_config = init_ascend_config(first_vllm_config)
        self.assertFalse(first_ascend_config.ascend_compilation_config.enable_npugraph_ex)

        second_vllm_config = VllmConfig()
        second_ascend_config = init_ascend_config(second_vllm_config)
        self.assertIsNot(first_ascend_config, second_ascend_config)
        self.assertTrue(second_ascend_config.ascend_compilation_config.enable_npugraph_ex)


class TestShortRequestFirstConfig(TestBase):
    def test_default_is_disabled(self):
        cfg = ShortRequestFirstConfig({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.threshold, 256)
        self.assertEqual(cfg.long_max_wait_ms, 0.0)

    def test_explicit_config(self):
        cfg = ShortRequestFirstConfig(
            {
                "enabled": True,
                "threshold": 512,
                "long_max_wait_ms": 2000,
            }
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.threshold, 512)
        self.assertEqual(cfg.long_max_wait_ms, 2000.0)

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            ShortRequestFirstConfig({"foo": 1})

    def test_validation_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            ShortRequestFirstConfig({"long_token_reservation": 1.5})
        with self.assertRaises(ValueError):
            ShortRequestFirstConfig({"threshold": -1})
        with self.assertRaises(ValueError):
            ShortRequestFirstConfig({"long_max_wait_ms": -1})

    def test_none_config_is_disabled(self):
        cfg = ShortRequestFirstConfig(None)
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.threshold, 256)
        self.assertEqual(cfg.long_max_wait_ms, 0.0)


class TestSchedulerConfig(TestBase):
    def test_defaults(self):
        config = SchedulerConfig({}, balance_env_value=False)

        self.assertFalse(config.enable_balance_scheduling)
        self.assertFalse(config.recompute_scheduler_enable)
        self.assertFalse(config.short_request_first_config.enabled)
        self.assertFalse(config.profiling_chunk_config.enabled)

    @patch("vllm_ascend.ascend_config.logger.warning_once")
    def test_none_config_uses_defaults_and_legacy_fallback(self, mock_warning_once):
        config = SchedulerConfig(
            {
                "scheduler_config": None,
                "recompute_scheduler_enable": True,
            },
            balance_env_value=False,
        )

        self.assertTrue(config.recompute_scheduler_enable)
        self.assertEqual(mock_warning_once.call_count, 1)

    def test_non_dict_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scheduler_config must be a dict, got list"):
            SchedulerConfig({"scheduler_config": []}, balance_env_value=False)

    def test_nested_config_overrides_all_scheduler_settings(self):
        config = SchedulerConfig(
            {
                "scheduler_config": {
                    "enable_balance_scheduling": True,
                    "recompute_scheduler_enable": True,
                    "short_request_first_config": {
                        "enabled": True,
                        "threshold": 512,
                        "long_max_wait_ms": 2000,
                    },
                    "profiling_chunk_config": {"enabled": True, "need_timing": False},
                }
            },
            balance_env_value=False,
        )

        self.assertTrue(config.enable_balance_scheduling)
        self.assertTrue(config.recompute_scheduler_enable)
        self.assertTrue(config.short_request_first_config.enabled)
        self.assertEqual(config.short_request_first_config.threshold, 512)
        self.assertEqual(config.short_request_first_config.long_max_wait_ms, 2000.0)
        self.assertTrue(config.profiling_chunk_config.enabled)
        self.assertFalse(config.profiling_chunk_config.need_timing)

    @patch("vllm_ascend.ascend_config.logger.warning_once")
    def test_legacy_top_level_config_warns_and_remains_supported(self, mock_warning_once):
        config = SchedulerConfig(
            {
                "enable_balance_scheduling": True,
                "recompute_scheduler_enable": True,
                "short_request_first_config": {"enabled": True},
                "profiling_chunk_config": {"enabled": True},
            },
            balance_env_value=False,
        )

        self.assertTrue(config.enable_balance_scheduling)
        self.assertTrue(config.recompute_scheduler_enable)
        self.assertTrue(config.short_request_first_config.enabled)
        self.assertTrue(config.profiling_chunk_config.enabled)
        self.assertEqual(mock_warning_once.call_count, 4)

    @patch("vllm_ascend.ascend_config.logger.warning_once")
    def test_nested_config_wins_and_legacy_fields_fill_missing_values(self, mock_warning_once):
        config = SchedulerConfig(
            {
                "scheduler_config": {
                    "recompute_scheduler_enable": True,
                    "short_request_first_config": {"enabled": True},
                },
                "recompute_scheduler_enable": False,
                "enable_balance_scheduling": True,
                "short_request_first_config": {"enabled": False},
            },
            balance_env_value=False,
        )

        self.assertTrue(config.recompute_scheduler_enable)
        self.assertTrue(config.short_request_first_config.enabled)
        self.assertTrue(config.enable_balance_scheduling)
        self.assertEqual(mock_warning_once.call_count, 3)

    @patch("vllm_ascend.ascend_config.logger.info_once")
    def test_balance_falls_back_to_environment_default(self, mock_info_once):
        with patch.dict(os.environ, {"VLLM_ASCEND_BALANCE_SCHEDULING": "1"}):
            config = SchedulerConfig({}, balance_env_value=True)

        self.assertTrue(config.enable_balance_scheduling)
        mock_info_once.assert_called_once()
