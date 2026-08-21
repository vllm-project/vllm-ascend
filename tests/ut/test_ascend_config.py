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
import subprocess
import sys
from types import SimpleNamespace
from unittest import skip
from unittest.mock import patch

from vllm.config import KVTransferConfig, VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import (
    AscendCompilationConfig,
    AscendConfig,
    AscendFusionConfig,
    DynamicSpecConfig,
    DyntraLBConfig,
    EplbConfig,
    FinegrainedTPConfig,
    ProfilingChunkConfig,
    RejectionSamplerConfig,
    RlConfig,
    SchedulerConfig,
    ShortRequestFirstConfig,
    SparseKVOffloadConfig,
    clear_ascend_config,
    get_ascend_config,
    init_ascend_config,
)
from vllm_ascend.utils import AscendDeviceType, clear_enable_sp, enable_dsa_cp, enable_sp, shared_expert_dp_enabled


def test_config_modules_do_not_load_vllm_config():
    """Keep platform discovery from recursing into a partial vllm.config."""
    code = (
        "import sys; import vllm_ascend.config_utils; "
        "assert 'vllm.config' not in sys.modules; "
        "import vllm_ascend.ascend_config; "
        "assert 'vllm.config' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


class TestRlConfig(TestBase):
    def test_defaults_and_explicit_values(self):
        defaults = RlConfig()
        enabled = RlConfig(
            enabled=True,
            sleep_mode_extra_cleanup=True,
            enable_training_consistency=True,
            enable_batch_invariant=True,
        )

        self.assertFalse(defaults.enabled)
        self.assertFalse(defaults.sleep_mode_extra_cleanup)
        self.assertTrue(enabled.enabled)
        self.assertTrue(enabled.sleep_mode_extra_cleanup)
        self.assertTrue(enabled.enable_training_consistency)
        self.assertTrue(enabled.enable_batch_invariant)

    def test_lax_bool_and_unknown_key(self):
        config = RlConfig(  # type: ignore[arg-type]
            enabled="true", sleep_mode_extra_cleanup="false"
        )

        self.assertTrue(config.enabled)
        self.assertFalse(config.sleep_mode_extra_cleanup)
        with self.assertRaises(ValueError):
            RlConfig(refresh=False)  # type: ignore[call-arg]


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

    @staticmethod
    def _make_sparse_li_c8_config(quant_description):
        quant_config = SimpleNamespace(quant_description=quant_description)
        # Use object.__new__ to bypass pydantic dataclass validation; this
        # helper only needs a partial AscendConfig to test sparse-li-c8 layer
        # filtering, not a fully constructed instance.
        config = object.__new__(AscendConfig)
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

    def test_vllm_independent_subconfigs_are_not_required(self):
        config = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False))

        self.assertFalse(config.xlite_graph_config.enabled)
        self.assertEqual(config.finegrained_tp_config.oproj_tensor_parallel_size, 0)
        self.assertFalse(config.scheduler_config.short_request_first_config.enabled)
        self.assertFalse(config.rl_config.enabled)

    def test_eplb_load_collection_phase_defaults_to_all(self):
        self.assertEqual(EplbConfig().load_collection_phase, "all")

    def test_eplb_load_collection_phase_validation(self):
        self.assertEqual(
            EplbConfig(load_collection_phase="prefill").load_collection_phase,
            "prefill",
        )
        self.assertEqual(
            EplbConfig(load_collection_phase="decode").load_collection_phase,
            "decode",
        )
        with self.assertRaisesRegex(ValueError, "load_collection_phase must be one of"):
            EplbConfig(load_collection_phase="prompt")

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_without_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        # No additional config given, check the default value here.
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertFalse(ascend_config.multistream_overlap_shared_expert)
        self.assertFalse(ascend_config.enable_kv_nz)
        self.assertEqual(ascend_config.weight_nz_mode, 1)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertTrue(ascend_compilation_config.fuse_norm_quant)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertTrue(ascend_fusion_config.fusion_ops_gmmswigluquant)
        self.assertFalse(ascend_config.rl_config.enabled)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_enabled_applies_runtime_defaults(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"rl_config": {"enabled": True}}
        with patch.dict(os.environ, {}, clear=True):
            ascend_config = init_ascend_config(test_vllm_config)

            self.assertTrue(ascend_config.rl_config.enabled)
            self.assertEqual(ascend_config.weight_nz_mode, 0)
            self.assertNotIn("VLLM_ASCEND_ENABLE_NZ", os.environ)
            self.assertEqual(os.environ["VLLM_SERVER_DEV_MODE"], "1")

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rl_config_enabled_refreshes_cached_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"rl_config": {"enabled": True}}

        self.assertIsNot(init_ascend_config(test_vllm_config), init_ascend_config(test_vllm_config))

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
            "xlite_graph_config": {"enabled": False, "full_mode": True},
            "finegrained_tp_config": {"lmhead_tensor_parallel_size": "0"},
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
        self.assertTrue(ascend_config.xlite_graph_config.full_mode)
        self.assertEqual(ascend_config.finegrained_tp_config.lmhead_tensor_parallel_size, 0)

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
    def test_init_ascend_config_with_legacy_scheduler_keys(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "recompute_scheduler_enable": "false",
            "short_request_first_config": {"enabled": "true", "threshold": "512"},
            "profiling_chunk_config": {"enabled": "false"},
            "batch_job_sched_config": {"enabled": "false"},
        }

        scheduler_config = init_ascend_config(test_vllm_config).scheduler_config

        self.assertFalse(scheduler_config.recompute_scheduler_enable)
        self.assertTrue(scheduler_config.short_request_first_config.enabled)
        self.assertEqual(scheduler_config.short_request_first_config.threshold, 512)
        self.assertFalse(scheduler_config.profiling_chunk_config.enabled)
        self.assertFalse(scheduler_config.batch_job_sched_config.enabled)

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
                "VLLM_ASCEND_ENABLE_MLAPO": "0",
            },
        ):
            ascend_config = init_ascend_config(test_vllm_config)

        self.assertEqual(ascend_config.enable_fused_mc2, 0)
        self.assertFalse(ascend_config.enable_mlapo)
        self.assertTrue(ascend_config.enable_transpose_kv_cache_by_block)
        self.assertEqual(ascend_config.weight_nz_mode, 1)
        mock_info_once.assert_any_call(
            "AscendConfig.enable_mlapo falls back to environment variable VLLM_ASCEND_ENABLE_MLAPO with value False. "
            "Please use additional_config.enable_mlapo instead, because VLLM_ASCEND_ENABLE_MLAPO will be "
            "removed in the next release."
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
            "msmonitor_use_daemon": False,
            "enable_transpose_kv_cache_by_block": True,
            "weight_nz_mode": 1,
        }
        with patch.dict(
            os.environ,
            {
                "VLLM_ASCEND_ENABLE_MLAPO": "0",
            },
        ):
            ascend_config = init_ascend_config(test_vllm_config)

        self.assertEqual(ascend_config.enable_fused_mc2, 0)
        self.assertTrue(ascend_config.enable_mlapo)
        self.assertFalse(ascend_config.msmonitor_use_daemon)
        self.assertTrue(ascend_config.enable_transpose_kv_cache_by_block)
        self.assertEqual(ascend_config.weight_nz_mode, 1)
        mock_info_once.assert_any_call("AscendConfig.enable_mlapo is set from additional_config with value True.")
        mock_info_once.assert_any_call("AscendConfig.weight_nz_mode is set from additional_config with value 1.")

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_msmonitor_daemon_uses_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"msmonitor_use_daemon": True}

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertTrue(ascend_config.msmonitor_use_daemon)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_weight_nz_mode_ignores_removed_env(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"weight_nz_mode": 2}
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"}):
            ascend_config = init_ascend_config(test_vllm_config)

        self.assertEqual(ascend_config.weight_nz_mode, 2)

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.warning")
    def test_flashcomm_config_warns(self, mock_warning):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"enable_flashcomm1": True}
        init_ascend_config(test_vllm_config)

        warning_messages = [call.args[0] for call in mock_warning.call_args_list]
        self.assertIn(
            "FlashComm is deprecated; remove enable_flashcomm1 and "
            "VLLM_ASCEND_ENABLE_FLASHCOMM1 from the configuration. Use upstream configuration instead",
            warning_messages,
        )

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger.warning")
    def test_flashcomm_environment_warns(self, mock_warning):
        test_vllm_config = VllmConfig()
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1"}, clear=True):
            init_ascend_config(test_vllm_config)

        warning_messages = [call.args[0] for call in mock_warning.call_args_list]
        self.assertIn(
            "FlashComm is deprecated; remove enable_flashcomm1 and "
            "VLLM_ASCEND_ENABLE_FLASHCOMM1 from the configuration. Use upstream configuration instead",
            warning_messages,
        )

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_sequence_parallel_and_shared_expert_dp_are_independent(self, mock_check_and_update_config):
        for use_sequence_parallel_moe, enable_shared_expert_dp in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                use_sequence_parallel_moe=use_sequence_parallel_moe,
                enable_shared_expert_dp=enable_shared_expert_dp,
            ):
                clear_ascend_config()
                clear_enable_sp()
                test_vllm_config = VllmConfig()
                test_vllm_config.parallel_config.tensor_parallel_size = 2
                test_vllm_config.parallel_config.data_parallel_size = 2
                test_vllm_config.parallel_config.enable_expert_parallel = True
                test_vllm_config.parallel_config.all2all_backend = (
                    "allgather_reducescatter" if use_sequence_parallel_moe else "flashinfer_all2allv"
                )
                test_vllm_config.additional_config = {
                    "enable_shared_expert_dp": enable_shared_expert_dp,
                }

                ascend_config = init_ascend_config(test_vllm_config)

                self.assertEqual(enable_sp(test_vllm_config), use_sequence_parallel_moe)
                self.assertEqual(ascend_config.enable_shared_expert_dp, enable_shared_expert_dp)
                self.assertEqual(shared_expert_dp_enabled(), enable_shared_expert_dp)

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
            },
        }
        first_ascend_config = init_ascend_config(first_vllm_config)
        self.assertFalse(first_ascend_config.ascend_compilation_config.enable_npugraph_ex)

        second_vllm_config = VllmConfig()
        second_ascend_config = init_ascend_config(second_vllm_config)
        self.assertIsNot(first_ascend_config, second_ascend_config)
        self.assertTrue(second_ascend_config.ascend_compilation_config.enable_npugraph_ex)


class TestShortRequestFirstConfig(TestBase):
    def test_default_is_disabled(self):
        cfg = ShortRequestFirstConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.threshold, 256)
        self.assertEqual(cfg.long_max_wait_ms, 0.0)

    def test_explicit_config(self):
        cfg = ShortRequestFirstConfig(
            **{
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
            ShortRequestFirstConfig(**{"foo": 1})

    def test_validation_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            ShortRequestFirstConfig(**{"long_token_reservation": 1.5})
        with self.assertRaises(ValueError):
            ShortRequestFirstConfig(**{"threshold": -1})
        with self.assertRaises(ValueError):
            ShortRequestFirstConfig(**{"long_max_wait_ms": -1})

    def test_none_config_is_disabled(self):
        cfg = ShortRequestFirstConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.threshold, 256)
        self.assertEqual(cfg.long_max_wait_ms, 0.0)


class TestSparseKVOffloadConfig(TestBase):
    def test_disabled_string_false_does_not_enter_enabled_path(self):
        config = SparseKVOffloadConfig.from_additional_config(SimpleNamespace(), {"enabled": "false"})

        self.assertFalse(config.enabled)

    def test_enabled_fields_are_typed_before_consumption(self):
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=128)),
            parallel_config=SimpleNamespace(
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                pipeline_parallel_size=1,
            ),
            kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
            use_v2_model_runner=False,
        )

        config = SparseKVOffloadConfig.from_additional_config(
            vllm_config,
            {
                "enabled": "true",
                "topk_buffer_size": "256",
                "dram_size_per_dp_GB": "64",
                "keep_device_kv_cache": "false",
            },
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.topk_buffer_size, 256)
        self.assertEqual(config.dram_size_per_dp_GB, 64)
        self.assertFalse(config.keep_device_kv_cache)

    def test_unknown_key_is_rejected_even_when_disabled(self):
        with self.assertRaises(ValueError):
            SparseKVOffloadConfig.from_additional_config(SimpleNamespace(), {"unknown_option": False})

    def test_non_dict_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sparse_kv_offload_config must be a dict"):
            SparseKVOffloadConfig.from_additional_config(SimpleNamespace(), [])


class TestSchedulerConfig(TestBase):
    def test_defaults(self):
        config = SchedulerConfig.from_additional_config({})

        self.assertFalse(config.enable_balance_scheduling)
        self.assertFalse(config.recompute_scheduler_enable)
        self.assertFalse(config.short_request_first_config.enabled)
        self.assertFalse(config.profiling_chunk_config.enabled)
        self.assertFalse(hasattr(config, "_additional_config"))
        self.assertFalse(hasattr(config, "_balance_env_value"))

    @patch("vllm_ascend.ascend_config.logger.warning_once")
    def test_none_config_uses_defaults_and_legacy_fallback(self, mock_warning_once):
        config = SchedulerConfig.from_additional_config(
            {
                "scheduler_config": None,
                "recompute_scheduler_enable": True,
            },
        )

        self.assertTrue(config.recompute_scheduler_enable)
        self.assertEqual(mock_warning_once.call_count, 1)

    def test_non_dict_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scheduler_config must be a dict, got list"):
            SchedulerConfig.from_additional_config({"scheduler_config": []})

    def test_unknown_nested_scheduler_key_is_rejected(self):
        with self.assertRaises(ValueError):
            SchedulerConfig.from_additional_config({"scheduler_config": {"unknown_option": {"enabled": True}}})

    def test_unknown_profiling_chunk_key_is_rejected(self):
        with self.assertRaises(ValueError):
            SchedulerConfig.from_additional_config(
                {"scheduler_config": {"profiling_chunk_config": {"unknown_option": False}}}
            )

    def test_unknown_batch_job_key_is_rejected(self):
        with self.assertRaises(ValueError):
            SchedulerConfig.from_additional_config({"scheduler_config": {"batch_job_sched_config": {"max_job": 2}}})

    def test_recompute_scheduler_switch_gets_bool_validation(self):
        config = SchedulerConfig.from_additional_config(
            {
                "scheduler_config": {
                    "recompute_scheduler_enable": "false",
                }
            }
        )

        self.assertFalse(config.recompute_scheduler_enable)
        with self.assertRaises(ValueError):
            SchedulerConfig.from_additional_config({"scheduler_config": {"recompute_scheduler_enable": 2}})

    def test_nested_config_overrides_all_scheduler_settings(self):
        config = SchedulerConfig.from_additional_config(
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
        config = SchedulerConfig.from_additional_config(
            {
                "enable_balance_scheduling": True,
                "recompute_scheduler_enable": True,
                "short_request_first_config": {"enabled": True},
                "profiling_chunk_config": {"enabled": True},
            },
        )

        self.assertTrue(config.enable_balance_scheduling)
        self.assertTrue(config.recompute_scheduler_enable)
        self.assertTrue(config.short_request_first_config.enabled)
        self.assertTrue(config.profiling_chunk_config.enabled)
        self.assertEqual(mock_warning_once.call_count, 4)

    @patch("vllm_ascend.ascend_config.logger.warning_once")
    def test_nested_config_wins_and_legacy_fields_fill_missing_values(self, mock_warning_once):
        config = SchedulerConfig.from_additional_config(
            {
                "scheduler_config": {
                    "recompute_scheduler_enable": True,
                    "short_request_first_config": {"enabled": True},
                },
                "recompute_scheduler_enable": False,
                "enable_balance_scheduling": True,
                "short_request_first_config": {"enabled": False},
            },
        )

        self.assertTrue(config.recompute_scheduler_enable)
        self.assertTrue(config.short_request_first_config.enabled)
        self.assertTrue(config.enable_balance_scheduling)
        self.assertEqual(mock_warning_once.call_count, 3)


class TestSubconfigPydanticTypeValidation(TestBase):
    """Verify @config migration gives sub-configs lax bool/int coercion and forbid.

    These tests construct sub-configs directly (no vllm_config / init_ascend_config)
    so they run on CPU-only UT runners.
    """

    def test_ascend_fusion_config_string_false_disables(self):
        # bool("false") is True in Python; pydantic lax must resolve to False.
        self.assertFalse(AscendFusionConfig(fusion_ops_gmmswigluquant="false").fusion_ops_gmmswigluquant)
        self.assertTrue(AscendFusionConfig(fusion_ops_gmmswigluquant="true").fusion_ops_gmmswigluquant)

    def test_ascend_fusion_config_forbids_unknown_key(self):
        with self.assertRaises(ValueError):
            AscendFusionConfig(unknown_key=1)

    def test_ascend_compilation_config_bool_lax_and_forbid(self):
        cfg = AscendCompilationConfig(enable_npugraph_ex="false")
        self.assertFalse(cfg.enable_npugraph_ex)
        with self.assertRaises(ValueError):
            AscendCompilationConfig(unknown_key=1)

    def test_profiling_chunk_config_int_lax_and_range(self):
        # int string "2" coerces to 2 (fixes "2"==2 silent failure)
        cfg = ProfilingChunkConfig(min_chunk="4096", max_fit_chunk="30")
        self.assertEqual(cfg.min_chunk, 4096)
        # range check preserved
        with self.assertRaises(ValueError):
            ProfilingChunkConfig(smooth_factor=1.5)

    def test_dynamic_spec_config_accepts_dflash(self):
        self.assertEqual(DynamicSpecConfig(method="dflash").method, "dflash")

    def test_short_request_first_config_unknown_key_forbidden(self):
        # Was hand-written unknown-key check; now extra="forbid".
        with self.assertRaises(ValueError):
            ShortRequestFirstConfig(foo=1)

    def test_dyntra_lb_config_lax_types_and_forbid(self):
        cfg = DyntraLBConfig(  # type: ignore[call-arg]
            enabled="true", start_step="10", bubble_threshold="2.5"
        )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.start_step, 10)
        self.assertEqual(cfg.bubble_threshold, 2.5)
        with self.assertRaises(ValueError):
            DyntraLBConfig(unknown_key=True)  # type: ignore[call-arg]

    def test_dyntra_lb_config_range_checks_preserved(self):
        with self.assertRaisesRegex(ValueError, "end_step must be greater than start_step"):
            DyntraLBConfig(start_step=10, end_step=10)  # type: ignore[call-arg]

    def test_rejection_sampler_config_range_check_preserved(self):
        with self.assertRaises(ValueError):
            RejectionSamplerConfig(posterior_threshold=1.5)

    def test_finegrained_tp_config_rejects_negative_size(self):
        with self.assertRaisesRegex(ValueError, "lmhead_tensor_parallel_size must be non-negative"):
            FinegrainedTPConfig(lmhead_tensor_parallel_size=-1)

    def test_eplb_config_int_field_lax(self):
        cfg = EplbConfig(eplb_policy_type="2")
        self.assertEqual(cfg.eplb_policy_type, 2)


class TestUpstreamConfigCompatibility(TestBase):
    def test_megamoe_model_config_constraints(self):
        supported = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    hidden_size=4096,
                    moe_intermediate_size=1536,
                    moe_quantize="w8a8",
                )
            )
        )
        unsupported = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    hidden_size=896,
                    moe_intermediate_size=1536,
                )
            )
        )

        self.assertTrue(AscendConfig._is_megamoe_supported_by_config(supported))
        self.assertFalse(AscendConfig._is_megamoe_supported_by_config(unsupported))

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A2)
    def test_mc2_hierarchy_comm_rejects_more_than_512_experts(self, mock_device_type):
        config = AscendConfig(
            sparse_kv_offload_config=SimpleNamespace(enabled=False),
            mc2_comm_alg="hierarchy",
        )
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(get_num_experts=lambda: 513))

        with self.assertRaisesRegex(ValueError, "supports at most 512 experts"):
            config._validate_mc2_comm_alg(vllm_config)

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A5)
    def test_mc2_hierarchy_comm_rejects_unsupported_device(self, mock_device_type):
        config = AscendConfig(
            sparse_kv_offload_config=SimpleNamespace(enabled=False),
            mc2_comm_alg="hierarchy",
        )
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(get_num_experts=lambda: 1))

        with self.assertRaisesRegex(NotImplementedError, "only supported on A2 and A3"):
            config._validate_mc2_comm_alg(vllm_config)


class TestTopLevelSwitchTypeValidation(TestBase):
    """Verify @config migration gives top-level AscendConfig switches type validation.

    These tests exercise the full ``init_ascend_config`` path (vllm_config +
    factory + before/after validators), so they require a constructible
    VllmConfig. Run on NPU/Linux UT runners (Windows lacks torch_npu).
    """

    @staticmethod
    def _clean_up(func):
        def wrapper(*args, **kwargs):
            clear_ascend_config()
            clear_enable_sp()
            try:
                func(*args, **kwargs)
            finally:
                clear_ascend_config()
                clear_enable_sp()

        return wrapper

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_enable_cpu_binding_string_false_disables(self, mock_fix):
        # Core regression: bool("false") is True in Python, so
        # {"enable_cpu_binding": "false"} previously left CPU binding enabled.
        # Pydantic lax coercion must resolve "false" to False.
        vc = VllmConfig()
        vc.additional_config = {"enable_cpu_binding": "false"}
        self.assertFalse(init_ascend_config(vc).enable_cpu_binding)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_enable_prefill_mc2_string_false_disables(self, mock_fix):
        vc = VllmConfig()
        vc.additional_config = {"enable_prefill_mc2": "false"}
        self.assertFalse(init_ascend_config(vc).enable_prefill_mc2)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_a_family_additional_config_gets_typed_validation(self, mock_fix):
        vc = VllmConfig()
        vc.additional_config = {
            "enable_fused_mc2": "0",
            "enable_mlapo": "false",
            "msmonitor_use_daemon": "false",
            "enable_transpose_kv_cache_by_block": "false",
            "weight_nz_mode": "2",
        }

        config = init_ascend_config(vc)

        self.assertEqual(config.enable_fused_mc2, 0)
        self.assertFalse(config.enable_mlapo)
        self.assertFalse(config.msmonitor_use_daemon)
        self.assertFalse(config.enable_transpose_kv_cache_by_block)
        self.assertEqual(config.weight_nz_mode, 2)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_weight_nz_mode_rejects_unknown_mode(self, mock_fix):
        vc = VllmConfig()
        vc.additional_config = {"weight_nz_mode": 3}

        with self.assertRaisesRegex(ValueError, "weight_nz_mode must be one of 0, 1, or 2"):
            init_ascend_config(vc)

    @skip("Deprecated env compatibility will be removed; additional_config is the supported path.")
    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_enable_mlapo_env_false_no_longer_crashes(self, mock_fix):
        # Regression: `export VLLM_ASCEND_ENABLE_MLAPO=false` used to crash
        # startup with ValueError (int("false")). The before-validator reads
        # os.getenv directly and resolves "false" to False.
        vc = VllmConfig()
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_MLAPO": "false"}):
            self.assertFalse(init_ascend_config(vc).enable_mlapo)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_enable_cpu_binding_rejects_invalid_int(self, mock_fix):
        # JSON booleans should be true/false; an int 2 is neither 0 nor 1 and
        # must fail fast rather than being coerced into unexpected truthiness.
        vc = VllmConfig()
        vc.additional_config = {"enable_cpu_binding": 2}
        with self.assertRaises(ValueError):
            init_ascend_config(vc)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_mega_moe_max_tokens_int_lax(self, mock_fix):
        # int string "131072" coerces to 131072 (fixes str-vs-int silent failure).
        vc = VllmConfig()
        vc.additional_config = {"mega_moe_max_tokens": "131072"}
        self.assertEqual(init_ascend_config(vc).mega_moe_max_tokens, 131072)

    @_clean_up
    @patch("vllm_ascend.ascend_config._MEGA_MOE_SUPPORTED", True)
    @patch.object(AscendConfig, "_is_megamoe_supported_by_config", return_value=False)
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_fused_mc2_is_disabled_for_unsupported_megamoe_config(self, mock_fix, mock_megamoe_supported):
        vc = VllmConfig()
        vc.additional_config = {"enable_fused_mc2": 1}

        self.assertEqual(init_ascend_config(vc).enable_fused_mc2, 0)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_converged_bypass_fields_are_validated(self, mock_fix):
        vc = VllmConfig()
        vc.additional_config = {
            "enable_dsa_cp": "false",
            "draft_window_size": "4096",
        }

        config = init_ascend_config(vc)

        self.assertFalse(config.enable_dsa_cp)
        self.assertEqual(config.draft_window_size, 4096)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_enable_dsa_cp_model_gate_is_resolved_during_init(self, mock_fix):
        unsupported_vc = VllmConfig()
        unsupported_vc.additional_config = {"enable_dsa_cp": True}
        self.assertFalse(init_ascend_config(unsupported_vc).enable_dsa_cp)

        supported_vc = VllmConfig()
        supported_vc.model_config = SimpleNamespace(
            hf_text_config=SimpleNamespace(index_topk=2048),
            hf_config=SimpleNamespace(),
            enforce_eager=True,
            architectures=[],
        )
        supported_vc.additional_config = {"enable_dsa_cp": True}
        self.assertTrue(init_ascend_config(supported_vc).enable_dsa_cp)

        # init_ascend_config clears process caches after publishing the new
        # singleton. This read must not require vLLM's temporary config context.
        self.assertTrue(enable_dsa_cp())

    @_clean_up
    @patch("vllm_ascend.utils.model_uses_sfa_sparse", return_value=False)
    @patch("vllm_ascend.utils.enable_sp", return_value=True)
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_user_input_derived_field_survives_factory(self, mock_fix, mock_enable_sp, mock_sparse):
        vc = VllmConfig()
        vc.parallel_config.enable_expert_parallel = True
        vc.parallel_config.tensor_parallel_size = 2
        vc.additional_config = {"enable_shared_expert_dp": True}

        config = init_ascend_config(vc)

        self.assertTrue(config.enable_shared_expert_dp)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_vllm_config_is_factory_dependency_not_config_field(self, mock_fix):
        vc = VllmConfig()

        config = init_ascend_config(vc)

        self.assertNotIn("vllm_config", config.__pydantic_fields__)
        self.assertFalse(hasattr(config, "vllm_config"))

    @_clean_up
    @patch("vllm_ascend.utils.model_uses_sfa_sparse", return_value=True)
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_private_sparse_layer_state_is_derived_on_factory_path(self, mock_fix, mock_sparse):
        vc = VllmConfig()
        vc.quant_config = SimpleNamespace(
            quant_description={"model.layers.3.self_attn.indexer.quant_type": "INT8_DYNAMIC"}
        )
        vc.additional_config = {"enable_sparse_li_c8": True}

        config = init_ascend_config(vc)

        self.assertTrue(config.is_sparse_li_c8_layer("model.layers.3.self_attn.indexer.k_cache"))
        self.assertFalse(config.is_sparse_li_c8_layer("model.layers.4.self_attn.indexer.k_cache"))

    @_clean_up
    @patch("vllm_ascend.utils.model_uses_sfa_sparse", return_value=True)
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_sparse_sfa_user_input_is_derived_on_factory_path(self, mock_fix, mock_sparse):
        vc = VllmConfig()
        vc.additional_config = {"enable_sparse_sfa_c8": "true"}

        config = init_ascend_config(vc)

        self.assertTrue(config.enable_sparse_sfa_c8)

    @_clean_up
    @patch("vllm_ascend.utils.model_uses_sfa_sparse", return_value=True)
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_c8_reshape_optim_is_derived_on_factory_path(self, mock_fix, mock_sparse):
        vc = VllmConfig()
        vc.additional_config = {
            "enable_sparse_li_c8": "true",
            "c8_enable_reshape_optim": "true",
        }

        config = init_ascend_config(vc)

        self.assertTrue(config.c8_enable_reshape_optim)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_rejection_sampler_config_survives_factory(self, mock_fix):
        vc = VllmConfig()
        vc.additional_config = {
            "rejection_sampler_config": {
                "enable_block_verify": "false",
                "posterior_threshold": "0.8",
            }
        }

        config = init_ascend_config(vc)

        self.assertFalse(config.rejection_sampler_config.enable_block_verify)
        self.assertEqual(config.rejection_sampler_config.posterior_threshold, 0.8)

    @_clean_up
    @patch("vllm_ascend.utils.model_uses_sfa_sparse", return_value=False)
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_enable_kv_nz_uses_vllm_config_preconditions(self, mock_fix, mock_sparse):
        vc = VllmConfig()
        vc.model_config = SimpleNamespace(is_deepseek_mla=True, architectures=[], enforce_eager=True)
        vc.kv_transfer_config = SimpleNamespace(is_kv_consumer=True)
        vc.additional_config = {"enable_kv_nz": "true"}

        config = init_ascend_config(vc)

        self.assertTrue(config.enable_kv_nz)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_eplb_string_false_survives_factory(self, mock_fix):
        vc = VllmConfig()
        vc.additional_config = {"eplb_config": {"dynamic_eplb": "false"}}

        config = init_ascend_config(vc)

        self.assertFalse(config.eplb_config.dynamic_eplb)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_refresh_string_false_reuses_cached_config(self, mock_fix):
        vc = VllmConfig()
        vc.additional_config = {"refresh": "false"}

        first = init_ascend_config(vc)
        second = init_ascend_config(vc)

        self.assertIs(first, second)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_refresh_rejects_non_boolean_integer(self, mock_fix):
        vc = VllmConfig()
        vc.additional_config = {"refresh": 2}

        with self.assertRaises(ValueError):
            init_ascend_config(vc)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_sparse_kv_offload_string_false_survives_factory(self, mock_fix):
        vc = VllmConfig()
        vc.additional_config = {"sparse_kv_offload_config": {"enabled": "false"}}

        config = init_ascend_config(vc)

        self.assertFalse(config.sparse_kv_offload_config.enabled)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_unknown_top_level_key_is_rejected(self, mock_fix):
        # A typo'd top-level key (not a declared field, not a bypass key) flows
        # into kwargs and extra="forbid" catches it. Previously the factory
        # filtered by __pydantic_fields__ which stripped typos silently; now
        # only _NON_USER_INPUT_KEYS is stripped, so typos reach pydantic and are rejected.
        vc = VllmConfig()
        vc.additional_config = {"unknown_option": True}
        with self.assertRaises(ValueError):
            init_ascend_config(vc)
