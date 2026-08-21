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
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from vllm.config import KVTransferConfig, VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import (
    AscendConfig,
    DyntraLBConfig,
    EplbConfig,
    RlConfig,
    SchedulerConfig,
    ShortRequestFirstConfig,
    clear_ascend_config,
    get_ascend_config,
    init_ascend_config,
)
from vllm_ascend.utils import AscendDeviceType, clear_enable_sp, enable_sp, shared_expert_dp_enabled


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
        num_experts: int = 0,
        enable_sleep_mode: bool = False,
    ):
        return SimpleNamespace(
            is_deepseek_mla=is_deepseek_mla,
            use_mla=is_deepseek_mla,
            enforce_eager=True,
            enable_sleep_mode=enable_sleep_mode,
            model_arch_config=SimpleNamespace(total_num_attention_heads=total_num_attention_heads),
            get_total_num_kv_heads=lambda: total_num_kv_heads,
            get_num_experts=lambda: num_experts,
        )

    @classmethod
    def _make_mc2_hierarchy_vllm_config(
        cls,
        num_experts: int,
        *,
        dynamic_eplb: bool = False,
        num_redundant_experts: int = 0,
    ):
        vllm_config = VllmConfig()
        vllm_config.model_config = cls._make_model_config(num_experts=num_experts)
        vllm_config.additional_config = {
            "enable_mc2_hierarchy_comm": True,
            "eplb_config": {
                "dynamic_eplb": dynamic_eplb,
                "num_redundant_experts": num_redundant_experts,
            },
        }
        return vllm_config

    @staticmethod
    def _make_sparse_li_c8_config(quant_description):
        quant_config = SimpleNamespace(quant_description=quant_description)
        config = AscendConfig.__new__(AscendConfig)
        config.enable_sparse_li_c8 = True
        (
            config._sparse_li_c8_layer_ids,
            config._sparse_li_c8_layer_names,
        ) = AscendConfig._parse_sparse_li_c8_layers_from_quant_config(quant_config)
        config._sparse_li_c8_layer_filter_enabled = AscendConfig._has_sparse_li_c8_layer_config(quant_config)
        return config

    def test_sparse_li_c8_layer_filter_uses_indexer_quant_type(self):
        config = self._make_sparse_li_c8_config(
            {
                "model.layers.1.self_attn.indexer.quant_type": "INT8_DYNAMIC",
                "model.layers.2.self_attn.indexer.quant_type": "BF16",
            }
        )

        self.assertTrue(config.is_sparse_li_c8_layer("model.layers.1.self_attn.indexer.k_cache"))
        self.assertFalse(config.is_sparse_li_c8_layer("model.layers.2.self_attn.indexer.k_cache"))

    def test_sparse_li_c8_layer_filter_uses_indexer_wq_b_weight(self):
        config = self._make_sparse_li_c8_config(
            {
                "model.layers.3.self_attn.indexer.wq_b_weight": "W8A8_MXFP8",
                "model.layers.4.self_attn.indexer.wq_b_weight": "W8A8_DYNAMIC",
            }
        )

        self.assertTrue(config.is_sparse_li_c8_layer("model.layers.3.self_attn.indexer.k_cache"))
        self.assertFalse(config.is_sparse_li_c8_layer("model.layers.4.self_attn.indexer.k_cache"))

    def test_sparse_li_c8_without_layer_metadata_applies_to_all_indexers(self):
        config = self._make_sparse_li_c8_config({"indexer_quant_type": "INT8_DYNAMIC"})

        self.assertTrue(config.is_sparse_li_c8_layer("model.layers.1.self_attn.indexer.k_cache"))
        self.assertTrue(config.is_sparse_li_c8_layer("model.layers.2.self_attn.indexer.k_cache"))

    def test_eplb_load_collection_phase_defaults_to_all(self):
        self.assertEqual(EplbConfig().load_collection_phase, "all")

    def test_eplb_load_collection_phase_validation(self):
        self.assertEqual(
            EplbConfig({"load_collection_phase": "prefill"}).load_collection_phase,
            "prefill",
        )
        self.assertEqual(
            EplbConfig({"load_collection_phase": "decode"}).load_collection_phase,
            "decode",
        )
        with self.assertRaisesRegex(ValueError, "load_collection_phase must be one of"):
            EplbConfig({"load_collection_phase": "prompt"})

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A5)
    def test_mc2_hierarchy_comm_rejects_a5(self, _mock_device_type):
        vllm_config = self._make_mc2_hierarchy_vllm_config(512)

        with self.assertRaisesRegex(NotImplementedError, "only supported on A2 and A3"):
            AscendConfig(vllm_config)

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    def test_mc2_hierarchy_comm_rejects_more_than_512_experts(self, _mock_device_type):
        vllm_config = self._make_mc2_hierarchy_vllm_config(513)

        with self.assertRaisesRegex(ValueError, "at most 512 experts"):
            AscendConfig(vllm_config)

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    def test_mc2_hierarchy_comm_counts_dynamic_eplb_redundancy(self, _mock_device_type):
        vllm_config = self._make_mc2_hierarchy_vllm_config(
            480,
            dynamic_eplb=True,
            num_redundant_experts=33,
        )

        with (
            patch.dict(os.environ, {"DYNAMIC_EPLB": "true"}),
            self.assertRaisesRegex(
                ValueError,
                r"513 experts \(480 logical experts \+ 33 EPLB redundant experts\)",
            ),
        ):
            AscendConfig(vllm_config)

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    def test_mc2_hierarchy_comm_ignores_redundancy_when_dynamic_eplb_is_disabled(self, _mock_device_type):
        vllm_config = self._make_mc2_hierarchy_vllm_config(480, num_redundant_experts=33)

        AscendConfig(vllm_config)

    @patch("vllm_ascend.utils.get_ascend_device_type")
    def test_mc2_hierarchy_comm_accepts_512_experts_on_a2_and_a3(self, mock_device_type):
        for device_type in (AscendDeviceType.A2, AscendDeviceType.A3):
            with self.subTest(device_type=device_type):
                mock_device_type.return_value = device_type
                vllm_config = self._make_mc2_hierarchy_vllm_config(512)

                AscendConfig(vllm_config)

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
                "dyntra_lb_config": {
                    "enabled": True,
                    "enable_diagnostics": True,
                    "mode": "dynamic",
                    "start_step": 100,
                    "end_step": 500,
                    "bubble_threshold": 3.0,
                    "long_req_block_threshold": 512,
                    "dynamic_max_step": 128,
                },
            }
        }

        scheduler_config = init_ascend_config(test_vllm_config).scheduler_config

        self.assertTrue(scheduler_config.enable_balance_scheduling)
        self.assertTrue(scheduler_config.recompute_scheduler_enable)
        self.assertTrue(scheduler_config.short_request_first_config.enabled)
        self.assertEqual(scheduler_config.short_request_first_config.threshold, 512)
        self.assertFalse(scheduler_config.profiling_chunk_config.enabled)
        self.assertTrue(scheduler_config.dyntra_lb_config.enabled)
        self.assertTrue(scheduler_config.dyntra_lb_config.enable_diagnostics)
        self.assertEqual(scheduler_config.dyntra_lb_config.mode, "dynamic")
        self.assertEqual(scheduler_config.dyntra_lb_config.start_step, 100)
        self.assertEqual(scheduler_config.dyntra_lb_config.end_step, 500)

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
    @patch("vllm_ascend.ascend_config.AscendConfig._is_megamoe_supported_by_config")
    @patch("vllm_ascend.ascend_config.logger.info_once")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_migrated_config_falls_back_to_envs(self, mock_fix_incompatible_config, mock_info_once, mock_is_megamoe):
        mock_is_megamoe.return_value = True
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
    def test_flashcomm_and_shared_expert_dp_are_independent(self, mock_check_and_update_config):
        for enable_flashcomm1, enable_shared_expert_dp in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                enable_flashcomm1=enable_flashcomm1,
                enable_shared_expert_dp=enable_shared_expert_dp,
            ):
                clear_ascend_config()
                clear_enable_sp()
                test_vllm_config = VllmConfig()
                test_vllm_config.parallel_config.tensor_parallel_size = 2
                test_vllm_config.parallel_config.enable_expert_parallel = True
                test_vllm_config.additional_config = {
                    "enable_flashcomm1": enable_flashcomm1,
                    "enable_shared_expert_dp": enable_shared_expert_dp,
                }

                ascend_config = init_ascend_config(test_vllm_config)

                self.assertEqual(enable_sp(test_vllm_config), enable_flashcomm1)
                self.assertEqual(
                    ascend_config.enable_shared_expert_dp,
                    enable_shared_expert_dp,
                )
                self.assertEqual(
                    shared_expert_dp_enabled(),
                    enable_shared_expert_dp,
                )

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

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_defaults_when_unset(self, mock_fix_incompatible_config):
        with patch.dict(os.environ, {}, clear=True):
            ascend_config = init_ascend_config(VllmConfig())

        self.assertFalse(ascend_config.rl_config.enabled)
        self.assertFalse(ascend_config.rl_config.sleep_mode_extra_cleanup)
        self.assertFalse(ascend_config.rl_config.enable_training_consistency)
        self.assertFalse(ascend_config.rl_config.enable_batch_invariant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_enabled_applies_best_practice_defaults(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"rl_config": {"enabled": True}}
        with patch.dict(os.environ, {}, clear=True):
            ascend_config = init_ascend_config(test_vllm_config)

            self.assertEqual(ascend_config.weight_nz_mode, 0)
            self.assertEqual(os.environ.get("VLLM_ASCEND_ENABLE_NZ"), "0")
            self.assertEqual(os.environ.get("VLLM_SERVER_DEV_MODE"), "1")
            self.assertNotIn("VLLM_BATCH_INVARIANT", os.environ)
            self.assertFalse(ascend_config.rl_config.sleep_mode_extra_cleanup)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_refreshes_by_default(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"rl_config": {"enabled": True}}

        first_config = init_ascend_config(test_vllm_config)
        second_config = init_ascend_config(test_vllm_config)

        self.assertIsNot(first_config, second_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_refresh_cannot_be_configured(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"rl_config": {"enabled": True, "refresh": False}}

        with self.assertRaisesRegex(ValueError, "Unknown rl_config keys.*refresh"):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_training_consistency_is_enabled(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"rl_config": {"enabled": True, "enable_training_consistency": True}}

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertTrue(ascend_config.rl_config.enable_training_consistency)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_disabled_is_noop(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "rl_config": {
                "enabled": False,
                "sleep_mode_extra_cleanup": True,
                "enable_batch_invariant": True,
            }
        }
        allocator_config = "page_size:1g,expandable_segments:True"
        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_ENABLE_NZ": "2",
                "PYTORCH_NPU_ALLOC_CONF": allocator_config,
            },
            clear=True,
        ):
            ascend_config = init_ascend_config(test_vllm_config)

            # rl_config is a no-op when the master switch is off: the env var is
            # neither overridden by the sub-field nor rewritten by rl_config.
            self.assertEqual(ascend_config.weight_nz_mode, 2)
            self.assertEqual(os.environ["VLLM_ASCEND_ENABLE_NZ"], "2")
            self.assertNotIn("VLLM_SERVER_DEV_MODE", os.environ)
            self.assertNotIn("VLLM_BATCH_INVARIANT", os.environ)
            self.assertEqual(os.environ["PYTORCH_NPU_ALLOC_CONF"], allocator_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.warning")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_overrides_top_level_weight_nz_mode(self, mock_fix_incompatible_config, mock_warning):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "rl_config": {"enabled": True},
            "weight_nz_mode": 2,
        }
        with patch.dict(os.environ, {}, clear=True):
            ascend_config = init_ascend_config(test_vllm_config)
            self.assertEqual(os.environ["VLLM_ASCEND_ENABLE_NZ"], "0")

        self.assertEqual(ascend_config.weight_nz_mode, 0)
        mock_warning.assert_called_once_with(
            "RL config requires weight_nz_mode=0; overriding AscendConfig.weight_nz_mode from %s to 0.",
            2,
        )

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.warning")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_keeps_zero_weight_nz_mode_without_warning(self, mock_fix_incompatible_config, mock_warning):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "rl_config": {"enabled": True},
            "weight_nz_mode": 0,
        }
        with patch.dict(os.environ, {}, clear=True):
            ascend_config = init_ascend_config(test_vllm_config)

        self.assertEqual(ascend_config.weight_nz_mode, 0)
        mock_warning.assert_not_called()

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_top_level_sleep_mode_extra_cleanup_is_removed(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "enable_sleep_mode_extra_cleanup": True,
        }
        with self.assertRaisesRegex(
            ValueError,
            "has been removed.*rl_config.sleep_mode_extra_cleanup",
        ):
            init_ascend_config(test_vllm_config)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_batch_invariant_overrides_env(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"rl_config": {"enabled": True, "enable_batch_invariant": True}}
        with patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "0"}, clear=True):
            ascend_config = init_ascend_config(test_vllm_config)

            # rl_config prevails over the environment variable.
            self.assertEqual(os.environ["VLLM_BATCH_INVARIANT"], "1")
            self.assertNotIn("HCCL_DETERMINISTIC", os.environ)
            self.assertNotIn("LCCL_DETERMINISTIC", os.environ)
            self.assertTrue(ascend_config.rl_config.enable_batch_invariant)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_does_not_disable_batch_invariant_env(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"rl_config": {"enabled": True, "enable_batch_invariant": False}}
        with patch.dict(os.environ, {"VLLM_BATCH_INVARIANT": "1"}, clear=True):
            init_ascend_config(test_vllm_config)
            self.assertEqual(os.environ["VLLM_BATCH_INVARIANT"], "1")

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_always_enables_dev_endpoints(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"rl_config": {"enabled": True}}
        with patch.dict(os.environ, {"VLLM_SERVER_DEV_MODE": "0"}, clear=True):
            init_ascend_config(test_vllm_config)
            self.assertEqual(os.environ["VLLM_SERVER_DEV_MODE"], "1")

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    @patch("vllm_ascend.platform.logger.info")
    def test_rl_config_removes_expandable_segments(self, mock_info, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.model_config = self._make_model_config(enable_sleep_mode=False)
        test_vllm_config.additional_config = {"rl_config": {"enabled": True}}
        with patch.dict(
            os.environ,
            {"PYTORCH_NPU_ALLOC_CONF": "page_size:1g,expandable_segments:True"},
            clear=True,
        ):
            init_ascend_config(test_vllm_config)
            self.assertNotIn("expandable_segments", os.environ["PYTORCH_NPU_ALLOC_CONF"])
            self.assertEqual(os.environ["PYTORCH_NPU_ALLOC_CONF"], "page_size:1g")
        mock_info.assert_any_call(
            "Removed expandable_segments from PYTORCH_NPU_ALLOC_CONF: %s",
            "page_size:1g",
        )

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_non_dict_rejected(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"rl_config": "enabled"}
        with self.assertRaisesRegex(ValueError, "rl_config must be a dict"):
            init_ascend_config(test_vllm_config)


class TestRlConfig(TestBase):
    def test_defaults(self):
        config = RlConfig()

        self.assertFalse(config.enabled)
        self.assertFalse(config.sleep_mode_extra_cleanup)
        self.assertFalse(config.enable_training_consistency)
        self.assertFalse(config.enable_batch_invariant)

    def test_explicit_values(self):
        config = RlConfig(
            {
                "enabled": True,
                "sleep_mode_extra_cleanup": True,
                "enable_training_consistency": True,
                "enable_batch_invariant": True,
            }
        )

        self.assertTrue(config.enabled)
        self.assertTrue(config.sleep_mode_extra_cleanup)
        self.assertTrue(config.enable_training_consistency)
        self.assertTrue(config.enable_batch_invariant)

    def test_unknown_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown rl_config keys"):
            RlConfig({"foo": True})

    def test_bool_validation(self):
        with self.assertRaisesRegex(ValueError, "enabled must be a bool"):
            RlConfig({"enabled": "yes"})
        with self.assertRaisesRegex(ValueError, "enable_batch_invariant must be a bool"):
            RlConfig({"enable_batch_invariant": 1})

    def test_fixed_fields_cannot_be_configured(self):
        for key in ("refresh", "weight_nz_mode", "enable_dev_endpoints", "disable_expandable_segments"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, f"Unknown rl_config keys.*{key}"):
                RlConfig({key: True})

    def test_non_dict_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be a dict"):
            RlConfig(["enabled"])

    def test_disable_expandable_segments_strips_env(self):
        from vllm_ascend.platform import _disable_expandable_segments

        with patch.dict(
            os.environ,
            {"PYTORCH_NPU_ALLOC_CONF": "page_size:1g,expandable_segments:True"},
            clear=True,
        ):
            _disable_expandable_segments()
            self.assertEqual(os.environ["PYTORCH_NPU_ALLOC_CONF"], "page_size:1g")

        with patch.dict(os.environ, {}, clear=True):
            # Missing/empty env var is a no-op.
            _disable_expandable_segments()
            self.assertNotIn("PYTORCH_NPU_ALLOC_CONF", os.environ)

    def test_disable_expandable_segments_matches_exact_option(self):
        from vllm_ascend.platform import _disable_expandable_segments

        with patch.dict(
            os.environ,
            {"PYTORCH_NPU_ALLOC_CONF": "my_expandable_segments:True, expandable_segments:False ,page_size:1g"},
            clear=True,
        ):
            _disable_expandable_segments()
            self.assertEqual(os.environ["PYTORCH_NPU_ALLOC_CONF"], "my_expandable_segments:True,page_size:1g")


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


class TestDyntraLBConfig(TestBase):
    def test_defaults(self):
        config = DyntraLBConfig()

        self.assertFalse(config.enabled)
        self.assertFalse(config.enable_diagnostics)
        self.assertEqual(config.mode, "dynamic")
        self.assertEqual(config.start_step, 250)
        self.assertEqual(config.end_step, -1)
        self.assertEqual(config.bubble_threshold, 5.0)
        self.assertEqual(config.long_req_block_threshold, 700)
        self.assertEqual(config.dynamic_max_step, 256)

    def test_configures_dynamic_mode(self):
        config = DyntraLBConfig(
            {
                "enabled": True,
                "enable_diagnostics": True,
                "mode": "dynamic",
                "start_step": 100,
                "end_step": 500,
                "bubble_threshold": 3,
                "long_req_block_threshold": 512,
                "dynamic_max_step": 128,
            }
        )

        self.assertTrue(config.enabled)
        self.assertTrue(config.enable_diagnostics)
        self.assertEqual(config.mode, "dynamic")
        self.assertEqual(config.start_step, 100)
        self.assertEqual(config.end_step, 500)
        self.assertEqual(config.bubble_threshold, 3.0)
        self.assertEqual(config.long_req_block_threshold, 512)
        self.assertEqual(config.dynamic_max_step, 128)

    def test_rejects_invalid_config(self):
        invalid_configs: tuple[tuple[Any, str], ...] = (
            ([], "must be a dict"),
            ({"unknown": True}, "Unknown dyntra_lb_config keys"),
            ({"enabled": 1}, "enabled must be a bool"),
            ({"enable_diagnostics": 1}, "enable_diagnostics must be a bool"),
            ({"mode": "invalid"}, "mode must be one of"),
            ({"start_step": -1}, "start_step must be >= 0"),
            ({"start_step": 10, "end_step": 10}, "end_step must be greater than start_step"),
            ({"bubble_threshold": 0}, "bubble_threshold must be > 0"),
            ({"long_req_block_threshold": 0}, "long_req_block_threshold must be > 0"),
            ({"dynamic_max_step": 0}, "dynamic_max_step must be > 0"),
        )

        for user_config, message in invalid_configs:
            with self.subTest(user_config=user_config), self.assertRaisesRegex(ValueError, message):
                DyntraLBConfig(user_config)


class TestSchedulerConfig(TestBase):
    def test_defaults(self):
        config = SchedulerConfig({}, balance_env_value=False)

        self.assertFalse(config.enable_balance_scheduling)
        self.assertFalse(config.recompute_scheduler_enable)
        self.assertFalse(config.short_request_first_config.enabled)
        self.assertFalse(config.profiling_chunk_config.enabled)
        self.assertFalse(config.dyntra_lb_config.enabled)

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
                    "dyntra_lb_config": {
                        "enabled": True,
                        "enable_diagnostics": True,
                        "mode": "dynamic",
                        "start_step": 100,
                        "end_step": 500,
                    },
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
        self.assertTrue(config.dyntra_lb_config.enabled)
        self.assertTrue(config.dyntra_lb_config.enable_diagnostics)
        self.assertEqual(config.dyntra_lb_config.mode, "dynamic")
        self.assertEqual(config.dyntra_lb_config.start_step, 100)
        self.assertEqual(config.dyntra_lb_config.end_step, 500)

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
