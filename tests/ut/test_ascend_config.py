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

import dataclasses
import json
import math
import os
import subprocess
import sys
from importlib.util import find_spec as real_find_spec
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from vllm.config import KVTransferConfig
from vllm.config import VllmConfig as _VllmConfig
from vllm.config.compilation import CUDAGraphMode

from tests.ut.base import TestBase
from tests.ut.kvpp_utils import make_kvpp_config
from vllm_ascend.ascend_config import (
    AscendCompilationConfig,
    AscendConfig,
    AscendFusionConfig,
    AscendWarmupConfig,
    DynamicSpecConfig,
    DyntraLBConfig,
    EplbConfig,
    FinegrainedTPConfig,
    KVPPConfig,
    ProfilingChunkConfig,
    RejectionSamplerConfig,
    RlConfig,
    SchedulerConfig,
    ShortRequestFirstConfig,
    SparseKVOffloadConfig,
    clear_ascend_config,
    get_ascend_config,
    init_ascend_config,
    is_mega_moe_supported,
)
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.utils import clear_enable_sp, enable_dsa_cp, enable_sp, shared_expert_dp_enabled


def VllmConfig(*args: Any, **kwargs: Any) -> _VllmConfig:
    """Build a test config with the model metadata required by AscendConfig."""
    config = _VllmConfig(*args, **kwargs)
    if config.model_config is None:
        config.model_config = SimpleNamespace(
            is_moe=False,
            is_deepseek_mla=False,
            use_mla=False,
            enforce_eager=True,
            architectures=[],
            hf_text_config=SimpleNamespace(),
            get_total_num_kv_heads=lambda: 0,
            get_num_experts=lambda: 0,
            get_hidden_size=lambda: 0,
        )
    return config


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
            is_moe=False,
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
                "model.layers.3.self_attn.indexer.wq_b.weight": "W8A8_MXFP8",
                "model.layers.4.self_attn.indexer.wq_b.weight": "W8A8_DYNAMIC",
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

        self.assertEqual(config.kvpp_config.size, 1)
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

    def test_stair_config_defaults_and_overrides(self):
        defaults = EplbConfig().stair_config
        config = EplbConfig(stair_config={"rank_transfer_limit": 2, "load_risk_quantile": 0.9})

        self.assertEqual(
            dataclasses.asdict(defaults),
            {
                "load_window_bins": 64,
                "load_risk_quantile": 0.75,
                "relative_balance_threshold": 0.95,
                "absolute_balance_threshold": 0.90,
                "rank_transfer_limit": 1,
                "cross_node_transfer_limit": 1,
                "replica_search_num_stages": 4,
                "replica_search_radius": 8,
                "replica_search_beam_size": 64,
                "placement_search_backtrack_limit": 32,
            },
        )
        self.assertEqual(config.stair_config.rank_transfer_limit, 2)
        self.assertEqual(config.stair_config.z_score, NormalDist().inv_cdf(0.9))

    def test_stair_config_default_factory_and_frozen_contract(self):
        first = EplbConfig().stair_config
        second = EplbConfig().stair_config

        self.assertIsNot(first, second)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.load_window_bins = 2

    def test_stair_config_accepts_boundaries(self):
        for value in (
            {"load_window_bins": 2},
            {"load_window_bins": 256},
            {"load_risk_quantile": 0.500001},
            {"load_risk_quantile": 0.999999},
            {"relative_balance_threshold": 0.000001},
            {"relative_balance_threshold": 1},
            {"absolute_balance_threshold": 0.000001},
            {"absolute_balance_threshold": 1},
            {"rank_transfer_limit": 1},
            {"rank_transfer_limit": -1},
            {"cross_node_transfer_limit": 0},
            {"cross_node_transfer_limit": -1},
            {"replica_search_num_stages": 1},
            {"replica_search_num_stages": 8},
            {"replica_search_radius": 0},
            {"replica_search_radius": 32},
            {"replica_search_beam_size": 1},
            {"replica_search_beam_size": 128},
            {"placement_search_backtrack_limit": 0},
            {"placement_search_backtrack_limit": 64},
        ):
            with self.subTest(value=value):
                EplbConfig(stair_config=value)

    def test_stair_config_rejects_invalid_values(self):
        for value in (
            {"load_window_bins": 1},
            {"load_window_bins": 257},
            {"load_risk_quantile": 0.5},
            {"load_risk_quantile": 1},
            {"relative_balance_threshold": 0},
            {"relative_balance_threshold": 1.001},
            {"absolute_balance_threshold": 0},
            {"absolute_balance_threshold": 1.001},
            {"rank_transfer_limit": 0},
            {"cross_node_transfer_limit": -2},
            {"replica_search_num_stages": 0},
            {"replica_search_num_stages": 9},
            {"replica_search_radius": -1},
            {"replica_search_radius": 33},
            {"replica_search_beam_size": 0},
            {"replica_search_beam_size": 129},
            {"placement_search_backtrack_limit": -1},
            {"placement_search_backtrack_limit": 65},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                EplbConfig(stair_config=value)

    def test_stair_config_rejects_boolean_and_non_finite_numbers(self):
        names = dataclasses.asdict(EplbConfig().stair_config)
        for name in names:
            with self.subTest(name=name, value=True), self.assertRaisesRegex(ValueError, "must not be booleans"):
                EplbConfig(stair_config={name: True})
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    EplbConfig(stair_config={name: value})

    def test_eplb_config_rejects_algorithm_selection(self):
        for algorithm in ("default", "stair"):
            with self.subTest(algorithm=algorithm), self.assertRaises(ValueError):
                EplbConfig(**{"algorithm": algorithm})

    def test_stair_config_rejects_removed_options(self):
        for name in (
            "flash_tree_depth",
            "flash_tree_width",
            "hysteresis_absolute",
            "hysteresis_relative",
            "imbalance_threshold",
            "lpt_max_backtracks",
            "max_load_window_bins",
            "max_candidates_per_layer",
            "max_expert_transfers_per_rank_pair",
            "min_relative_score_improvement",
            "min_absolute_score_improvement",
            "p95_regression_tolerance",
            "risk_quantile",
            "sample_size",
            "score_tie_tolerance",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                EplbConfig(stair_config={name: 0})

    def test_stair_config_rejects_unknown_option(self):
        with self.assertRaises(ValueError):
            EplbConfig(stair_config={"unknown_option": 0})

    def test_stair_config_rejects_internal_policy_controls(self):
        for name in ("z_score", "use_covariance", "hysteresis_enabled"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                EplbConfig(stair_config={name: 0})

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_without_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        # No additional config given, check the default value here.
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertFalse(ascend_config.multistream_overlap_shared_expert)
        self.assertFalse(ascend_config.enable_kv_nz)
        self.assertEqual(ascend_config.weight_nz_mode, 1)
        self.assertEqual(ascend_config.mega_moe_max_tokens, 65536)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertTrue(ascend_compilation_config.fuse_norm_quant)

        ascend_fusion_config = ascend_config.ascend_fusion_config
        self.assertTrue(ascend_fusion_config.fusion_ops_gmmswigluquant)
        self.assertFalse(ascend_config.rl_config.enabled)

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_warns_unsupported_prefill_backend(self, mock_fix_incompatible_config, mock_logger):
        # Upstream EngineArgs injects --gdn-prefill-backend / --kda-prefill-backend
        # into additional_config. Only the 'triton' value (FLA kernels run via
        # triton-ascend) exists on Ascend, so CUDA-only values must be stripped
        # with a warning instead of being rejected as typos by extra="forbid".
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "gdn_prefill_backend": "flashinfer",
            "kda_prefill_backend": "flashkda",
        }
        # extra="forbid" would raise if the injected keys reached AscendConfig.
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertIsNotNone(ascend_config)

        prefill_warnings = [
            call for call in mock_logger.warning_once.call_args_list if "does not support" in str(call.args[0])
        ]
        warned_text = " ".join(str(call.args) for call in prefill_warnings)
        self.assertIn("gdn_prefill_backend", warned_text)
        self.assertIn("flashinfer", warned_text)
        self.assertIn("kda_prefill_backend", warned_text)
        self.assertIn("flashkda", warned_text)

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_silent_triton_prefill_backend(self, mock_fix_incompatible_config, mock_logger):
        # 'triton'/'auto' are the Ascend-supported values; they are stripped
        # silently (equivalent to the default) without any warning.
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "gdn_prefill_backend": "triton",
            "kda_prefill_backend": "auto",
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertIsNotNone(ascend_config)

        prefill_warnings = [
            call for call in mock_logger.warning_once.call_args_list if "does not support" in str(call.args[0])
        ]
        self.assertEqual(prefill_warnings, [])

    @_clean_up_ascend_config
    @patch("vllm_ascend.ascend_config.logger")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_without_prefill_backend_keys(self, mock_fix_incompatible_config, mock_logger):
        # Without the injected keys, initialization succeeds and emits no
        # "does not support" warning.
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"mega_moe_max_tokens": 65536}
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertIsNotNone(ascend_config)

        prefill_warnings = [
            call for call in mock_logger.warning_once.call_args_list if "does not support" in str(call.args[0])
        ]
        self.assertEqual(prefill_warnings, [])

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
            "enable_force_eplb": True,
            "eplb_config": {"num_redundant_experts": 2},
            "refresh": True,
            "enable_kv_nz": False,
            "xlite_graph_config": {"enabled": False, "full_mode": True},
            "finegrained_tp_config": {"lmhead_tensor_parallel_size": "0"},
            "mega_moe_max_tokens": 32768,
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(ascend_config.eplb_config.num_redundant_experts, 2)
        self.assertTrue(ascend_config.multistream_overlap_shared_expert)
        self.assertTrue(ascend_config.enable_force_eplb)
        self.assertEqual(ascend_config.mega_moe_max_tokens, 32768)

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
    def test_init_ascend_config_validates_mega_moe_max_tokens(self, mock_fix_incompatible_config):
        # NOTE: pydantic coerces numeric strings (e.g. "65536") to int, so only
        # out-of-range values are invalid on main.
        invalid_values = [0, -1]

        for invalid_value in invalid_values:
            clear_ascend_config()
            test_vllm_config = VllmConfig()
            test_vllm_config.additional_config = {"mega_moe_max_tokens": invalid_value}

            with (
                self.subTest(invalid_value=invalid_value),
                self.assertRaisesRegex(ValueError, "mega_moe_max_tokens must be"),
            ):
                init_ascend_config(test_vllm_config)

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
        self.assertTrue(ascend_compilation_config.enable_super_kernel)

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_super_kernel_explicit_disable(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "enable_npugraph_ex": True,
                "enable_static_kernel": True,
                "enable_super_kernel": False,
            },
            "refresh": True,
        }
        ascend_compilation_config = init_ascend_config(test_vllm_config).ascend_compilation_config
        self.assertTrue(ascend_compilation_config.enable_static_kernel)
        self.assertFalse(ascend_compilation_config.enable_super_kernel)

    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    def test_ascend_compilation_config_super_kernel_defaults_to_static_kernel(self, _mock_profile):
        cfg = AscendCompilationConfig(enable_static_kernel=True)
        self.assertTrue(cfg.enable_static_kernel)
        self.assertTrue(cfg.enable_super_kernel)

        cfg = AscendCompilationConfig(enable_static_kernel=False)
        self.assertFalse(cfg.enable_static_kernel)
        self.assertFalse(cfg.enable_super_kernel)

        cfg = AscendCompilationConfig(enable_static_kernel=True, enable_super_kernel=False)
        self.assertTrue(cfg.enable_static_kernel)
        self.assertFalse(cfg.enable_super_kernel)

        cfg = AscendCompilationConfig(enable_static_kernel="true")
        self.assertTrue(cfg.enable_super_kernel)

        cfg = AscendCompilationConfig()
        self.assertFalse(cfg.enable_static_kernel)
        self.assertFalse(cfg.enable_super_kernel)

    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    def test_ascend_compilation_config_rejects_super_kernel_without_static_kernel(self, _mock_profile):
        with self.assertRaisesRegex(ValueError, "Super kernel generation requires static kernel to be enabled"):
            AscendCompilationConfig(enable_static_kernel=False, enable_super_kernel=True)

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
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType._310P),
    )
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_init_ascend_config_disable_npugraph_ex_on_310p(
        self, mock_fix_incompatible_config, mock_hardware_profile, mock_warning
    ):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {"enable_npugraph_ex": True, "enable_static_kernel": True},
            "refresh": True,
        }

        ascend_compilation_config = init_ascend_config(test_vllm_config).ascend_compilation_config

        self.assertFalse(ascend_compilation_config.enable_npugraph_ex)
        self.assertFalse(ascend_compilation_config.enable_static_kernel)
        self.assertFalse(ascend_compilation_config.enable_super_kernel)
        warning_messages = [call.args[0] for call in mock_warning.call_args_list]
        self.assertIn("npugraph_ex is not supported by the current hardware profile. Disabling it.", warning_messages)
        self.assertIn(
            "static kernel requires npugraph_ex, which is not supported by the current hardware profile. Disabling it.",
            warning_messages,
        )
        self.assertIn(
            "super kernel requires static kernel, which is not supported by the current hardware profile. "
            "Disabling it.",
            warning_messages,
        )

    @_clean_up_ascend_config
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_msmonitor_daemon_uses_additional_config(self, mock_fix_incompatible_config):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {"msmonitor_use_daemon": True}

        ascend_config = init_ascend_config(test_vllm_config)

        self.assertTrue(ascend_config.msmonitor_use_daemon)

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
                    # Keep the explicitly assigned all2all_backend: without an
                    # explicit flashcomm switch, derive_and_validate forces
                    # flashinfer_all2allv and would clobber the SP setup above.
                    "enable_flashcomm1": True,
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
                "use_fused_overlap": "true",
            },
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.topk_buffer_size, 256)
        self.assertEqual(config.dram_size_per_dp_GB, 64)
        self.assertFalse(config.keep_device_kv_cache)
        self.assertTrue(config.use_fused_overlap)

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

    def test_ascend_warmup_config_bool_lax_and_forbid(self):
        cfg = AscendWarmupConfig(enable_early_kernel_warmup="true", enable_early_nz_warmup="false")
        self.assertTrue(cfg.enable_early_kernel_warmup)
        self.assertFalse(cfg.enable_early_nz_warmup)
        with self.assertRaises(ValueError):
            AscendWarmupConfig(unknown_key=1)

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

    def _oproj_tp_vllm_config(
        self,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        num_speculative_tokens=0,
        max_cudagraph_capture_size=512,
        cudagraph_capture_sizes=None,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        prefill_context_parallel_size=1,
    ):
        speculative_config = None
        if num_speculative_tokens:
            speculative_config = SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
        return SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                data_parallel_size=8,
                prefill_context_parallel_size=prefill_context_parallel_size,
            ),
            compilation_config=SimpleNamespace(
                cudagraph_mode=cudagraph_mode,
                max_cudagraph_capture_size=max_cudagraph_capture_size,
                cudagraph_capture_sizes=cudagraph_capture_sizes,
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens, max_num_seqs=max_num_seqs),
            speculative_config=speculative_config,
            kv_transfer_config=SimpleNamespace(is_kv_consumer=True),
            model_config=SimpleNamespace(is_moe=True),
        )

    def test_oproj_tp_requires_graph_mode(self):
        config = FinegrainedTPConfig(oproj_tensor_parallel_size=2)
        # VllmConfig.__post_init__ normalizes enforce_eager into NONE, so this
        # single check covers both spellings of "no graph mode".
        with self.assertRaisesRegex(AssertionError, "only supported in graph mode"):
            config._validate_preconditions(self._oproj_tp_vllm_config(cudagraph_mode=CUDAGraphMode.NONE))

    def test_oproj_tp_rejects_pcp(self):
        config = FinegrainedTPConfig(oproj_tensor_parallel_size=2)
        with self.assertRaisesRegex(AssertionError, "not supported with prefill_context_parallel_size"):
            config._validate_preconditions(self._oproj_tp_vllm_config(prefill_context_parallel_size=2))

    def test_oproj_tp_size_one_skips_the_checks(self):
        # Size 1 requests no split: no exchange groups to align, so the preconditions do not apply.
        config = FinegrainedTPConfig(oproj_tensor_parallel_size=1)
        config._validate_preconditions(self._oproj_tp_vllm_config(cudagraph_mode=CUDAGraphMode.NONE))
        self.assertEqual(config.oproj_tensor_parallel_size, 1)

    def test_oproj_tp_capture_bound_check(self):
        config = FinegrainedTPConfig(oproj_tensor_parallel_size=2)
        config._validate_preconditions(self._oproj_tp_vllm_config())
        # max_num_batched_tokens can cap the step below the capture bound.
        config._validate_preconditions(self._oproj_tp_vllm_config(max_num_batched_tokens=512))
        self.assertEqual(config.oproj_tensor_parallel_size, 2)
        # 300 reqs x decode_query_len 2 (spec window) = 600 > 512: disabled with a warning.
        config._validate_preconditions(self._oproj_tp_vllm_config(max_num_seqs=300, num_speculative_tokens=1))
        self.assertEqual(config.oproj_tensor_parallel_size, 0)
        # An explicit capture size that covers the step keeps the knob on.
        config = FinegrainedTPConfig(oproj_tensor_parallel_size=2)
        config._validate_preconditions(
            self._oproj_tp_vllm_config(max_num_seqs=300, num_speculative_tokens=1, max_cudagraph_capture_size=1024)
        )
        self.assertEqual(config.oproj_tensor_parallel_size, 2)
        # Before _set_cudagraph_sizes backfills it, an explicit sizes list is the bound.
        config = FinegrainedTPConfig(oproj_tensor_parallel_size=2)
        config._validate_preconditions(
            self._oproj_tp_vllm_config(max_cudagraph_capture_size=None, cudagraph_capture_sizes=[8, 16, 512])
        )
        self.assertEqual(config.oproj_tensor_parallel_size, 2)

    def test_mlp_tp_capture_bound_check(self):
        config = FinegrainedTPConfig(mlp_tensor_parallel_size=2)
        config._validate_preconditions(self._oproj_tp_vllm_config())
        self.assertEqual(config.mlp_tensor_parallel_size, 2)
        # The step bound is knob-independent, so an oversized step disables both knobs together.
        config = FinegrainedTPConfig(oproj_tensor_parallel_size=2, mlp_tensor_parallel_size=4)
        config._validate_preconditions(self._oproj_tp_vllm_config(max_num_seqs=300, num_speculative_tokens=1))
        self.assertEqual(config.oproj_tensor_parallel_size, 0)
        self.assertEqual(config.mlp_tensor_parallel_size, 0)

    def test_eplb_config_int_field_lax(self):
        cfg = EplbConfig(eplb_policy_type="2")
        self.assertEqual(cfg.eplb_policy_type, 2)


class TestUpstreamConfigCompatibility(TestBase):
    @patch(
        "vllm_ascend.ascend_config.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A5),
    )
    def test_a5_megamoe_minimax_config_and_existing_guards(self, _mock_profile):
        text_config = SimpleNamespace(hidden_size=6144, intermediate_size=3072, num_experts_per_tok=4)
        model_config = SimpleNamespace(
            architectures=["MiniMaxM3SparseForCausalLM"],
            hf_text_config=text_config,
            get_num_experts=lambda: 128,
        )
        parallel_config = SimpleNamespace(world_size_across_dp=8, pipeline_parallel_size=1)
        vc = SimpleNamespace(model_config=model_config, parallel_config=parallel_config)
        self.assertTrue(AscendConfig._is_megamoe_supported_by_config(vc))

        for field, value in (("hidden_size", 896), ("intermediate_size", 4096), ("num_experts_per_tok", 33)):
            with self.subTest(field=field), patch.object(text_config, field, value):
                self.assertFalse(AscendConfig._is_megamoe_supported_by_config(vc))
        for world_size in (1, 3):
            with self.subTest(world_size=world_size), patch.object(parallel_config, "world_size_across_dp", world_size):
                self.assertFalse(AscendConfig._is_megamoe_supported_by_config(vc))

        model_config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
        self.assertFalse(AscendConfig._is_megamoe_supported_by_config(vc))
        text_config.moe_intermediate_size = 3072
        self.assertFalse(AscendConfig._is_megamoe_supported_by_config(vc))
        text_config.moe_intermediate_size = 1024
        self.assertTrue(AscendConfig._is_megamoe_supported_by_config(vc))

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

        minimax_m3 = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    hidden_size=6144,
                    intermediate_size=3072,
                )
            )
        )
        self.assertTrue(AscendConfig._is_megamoe_supported_by_config(minimax_m3))

    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A2),
    )
    def test_mc2_hierarchy_comm_rejects_more_than_512_experts(self, _mock_profile):
        config = AscendConfig(
            sparse_kv_offload_config=SimpleNamespace(enabled=False),
            mc2_comm_alg="hierarchy",
        )
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(get_num_experts=lambda: 513))

        with self.assertRaisesRegex(ValueError, "supports at most 512 experts"):
            config._validate_mc2_comm_alg(vllm_config)

    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A5),
    )
    def test_mc2_hierarchy_comm_rejects_unsupported_device(self, _mock_profile):
        config = AscendConfig(
            sparse_kv_offload_config=SimpleNamespace(enabled=False),
            mc2_comm_alg="hierarchy",
        )
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(get_num_experts=lambda: 1))

        with self.assertRaisesRegex(NotImplementedError, "not supported by the current hardware profile"):
            config._validate_mc2_comm_alg(vllm_config)

    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A5),
    )
    def test_mc2_fullmesh_v2_rejects_unsupported_device(self, _mock_profile):
        config = AscendConfig(
            sparse_kv_offload_config=SimpleNamespace(enabled=False),
            mc2_comm_alg="fullmesh_v2",
        )

        with self.assertRaisesRegex(NotImplementedError, "not supported by the current hardware profile"):
            config._validate_mc2_comm_alg(SimpleNamespace())

    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    def test_mc2_fullmesh_uses_a3_operator_alias(self, _mock_profile):
        config = AscendConfig(
            sparse_kv_offload_config=SimpleNamespace(enabled=False),
            mc2_comm_alg="fullmesh",
        )

        self.assertEqual(config.get_mc2_comm_alg(), "fullmesh_v1")


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
    @patch.object(AscendConfig, "_is_megamoe_supported_by_config", return_value=True)
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_fused_mc2_rolls_back_even_when_config_supported(self, mock_fix, mock_megamoe_supported):
        # After the megamoe op rollback (#15267), enable_fused_mc2=1 short-circuits
        # _MEGA_MOE_SUPPORTED to False in _validate_user_input_ranges, regardless
        # of whether the model config supports megamoe. So even when
        # _is_megamoe_supported_by_config() is True, is_mega_moe_supported() ends
        # up False and the fused path routes to dispatch_ffn_combine instead of
        # mega_moe.
        vc = VllmConfig()
        vc.additional_config = {"enable_fused_mc2": 1}

        config = init_ascend_config(vc)
        self.assertEqual(config.enable_fused_mc2, 1)
        # The rollback forces _MEGA_MOE_SUPPORTED=False, so the fused path
        # routes to dispatch_ffn_combine instead of mega_moe.
        self.assertFalse(is_mega_moe_supported())

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_minimax_m3_rejects_fused_mc2_dispatch_ffn_combine(self, mock_fix):
        vc = VllmConfig()
        vc.model_config.architectures = ["MiniMaxM3SparseForCausalLM"]
        vc.additional_config = {"enable_fused_mc2": 1}

        with self.assertRaisesRegex(AssertionError, "MiniMax M3 does not support enable_fused_mc2=1"):
            init_ascend_config(vc)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_minimax_m3_allows_fused_mc2_mode_2_megamoe(self, mock_fix):
        def _fake_find_spec(name, *args, **kwargs):
            if name == "cann_ops_transformer":
                return object()
            return real_find_spec(name, *args, **kwargs)

        vc = VllmConfig()
        vc.model_config.architectures = ["MiniMaxM3SparseForCausalLM"]
        vc.model_config.hf_text_config = SimpleNamespace(
            hidden_size=6144,
            intermediate_size=3072,
        )
        vc.additional_config = {"enable_fused_mc2": 2}

        with patch("vllm_ascend.ascend_config.importlib.util.find_spec", side_effect=_fake_find_spec):
            config = init_ascend_config(vc)

        self.assertEqual(config.enable_fused_mc2, 1)
        self.assertTrue(is_mega_moe_supported())

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_minimax_m3_derivation_initializes_unknown_megamoe_support(self, mock_fix):
        vc = VllmConfig()
        vc.model_config.architectures = ["MiniMaxM3SparseForCausalLM"]
        config = init_ascend_config(vc)
        # Exercise a custom entry point with normalized MC2 mode and an
        # uninitialized capability cache, outside the usual factory ordering.
        config.enable_fused_mc2 = 1
        with (
            patch("vllm_ascend.ascend_config._MEGA_MOE_SUPPORTED", None),
            patch("vllm_ascend.ascend_config.importlib.util.find_spec", return_value=object()),
            patch.object(AscendConfig, "_is_megamoe_supported_by_config", return_value=True),
        ):
            config.derive_and_validate(vc)
            self.assertEqual(config.enable_fused_mc2, 1)
            self.assertTrue(is_mega_moe_supported())

        # An explicitly disabled cache must stay disabled even with CANN
        # installed: MiniMax cannot use dispatch_ffn_combine.
        with (
            patch("vllm_ascend.ascend_config._MEGA_MOE_SUPPORTED", False),
            patch("vllm_ascend.ascend_config.importlib.util.find_spec", return_value=object()),
            self.assertRaisesRegex(AssertionError, "MiniMax M3 does not support enable_fused_mc2=1"),
        ):
            config.derive_and_validate(vc)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_converged_bypass_fields_are_validated(self, mock_fix):
        vc = VllmConfig()
        vc.additional_config = {
            "enable_dsa_cp": "false",
            "enable_pcp_o_proj_weight_sharding": "true",
            "draft_window_size": "4096",
        }

        config = init_ascend_config(vc)

        self.assertFalse(config.enable_dsa_cp)
        self.assertTrue(config.enable_pcp_o_proj_weight_sharding)
        self.assertEqual(config.draft_window_size, 4096)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_reduce_sample_configuration_compatibility(self, mock_fix):
        cases: tuple[tuple[dict[str, Any], int, str | None, str | None], ...] = (
            (
                {"finegrained_tp_config": {"lmhead_tensor_parallel_size": 2}},
                1,
                None,
                "finegrained_tp_config.lmhead_tensor_parallel_size",
            ),
            ({}, 2, None, "enable_pcp_embedding_lmhead_weight_sharding"),
            ({"enable_pcp_embedding_lmhead_weight_sharding": False}, 1, "kv_producer", "PD-disaggregated"),
            ({}, 1, None, None),
            ({"enable_pcp_embedding_lmhead_weight_sharding": False}, 2, None, None),
        )
        for additional_config, pcp_size, kv_role, error in cases:
            with self.subTest(pcp_size=pcp_size, kv_role=kv_role, error=error):
                clear_ascend_config()
                vc = VllmConfig()
                vc.parallel_config.prefill_context_parallel_size = pcp_size
                vc.additional_config = {"enable_reduce_sample": True, **additional_config}
                if kv_role is not None:
                    vc.kv_transfer_config = KVTransferConfig(
                        kv_connector="MooncakeConnectorV1",
                        kv_role=kv_role,
                    )

                if error is None:
                    self.assertTrue(init_ascend_config(vc).enable_reduce_sample)
                else:
                    with self.assertRaisesRegex(ValueError, error):
                        init_ascend_config(vc)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_enable_dsa_cp_model_gate_is_resolved_during_init(self, mock_fix):
        unsupported_vc = VllmConfig()
        unsupported_vc.additional_config = {"enable_dsa_cp": True}
        self.assertFalse(init_ascend_config(unsupported_vc).enable_dsa_cp)

        supported_vc = VllmConfig()
        supported_vc.model_config = SimpleNamespace(
            is_moe=False,
            hf_text_config=SimpleNamespace(index_topk=2048),
            hf_config=SimpleNamespace(),
            enforce_eager=True,
            architectures=[],
        )
        supported_vc.additional_config = {"enable_dsa_cp": True}
        # DSA-CP additionally requires sequence parallelism: EP + TP>1 + DP>1
        # makes ParallelConfig.use_sequence_parallel_moe True.
        supported_vc.parallel_config.enable_expert_parallel = True
        supported_vc.parallel_config.tensor_parallel_size = 2
        supported_vc.parallel_config.data_parallel_size = 2
        self.assertTrue(init_ascend_config(supported_vc).enable_dsa_cp)

        # init_ascend_config clears process caches after publishing the new
        # singleton. This read must not require vLLM's temporary config context.
        self.assertTrue(enable_dsa_cp())

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_flashcomm_enabled_keeps_sp_when_conditions_met(self, mock_fix):
        """Case 1: flashcomm on + SP conditions met -> SP stays on."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
            vc = VllmConfig()
            vc.parallel_config.enable_expert_parallel = True
            vc.parallel_config.tensor_parallel_size = 2
            vc.parallel_config.data_parallel_size = 2
            vc.parallel_config.all2all_backend = "allgather_reducescatter"
            vc.additional_config = {"enable_flashcomm1": True}

            config = init_ascend_config(vc)

            self.assertTrue(vc.parallel_config.use_sequence_parallel_moe)
            self.assertEqual(vc.parallel_config.all2all_backend, "allgather_reducescatter")
            self.assertTrue(enable_sp(vc))
            self.assertFalse(config.enable_dsa_cp)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_flashcomm_disabled_forces_sp_off_when_conditions_met(self, mock_fix):
        """Case 2: flashcomm off + SP conditions met -> SP still forced off."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
            vc = VllmConfig()
            vc.parallel_config.enable_expert_parallel = True
            vc.parallel_config.tensor_parallel_size = 2
            vc.parallel_config.data_parallel_size = 2
            vc.parallel_config.all2all_backend = "allgather_reducescatter"
            vc.additional_config = {}

            init_ascend_config(vc)

            self.assertEqual(vc.parallel_config.all2all_backend, "flashinfer_all2allv")
            self.assertFalse(enable_sp(vc))

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_dsa_cp_enabled_auto_keeps_sp_when_conditions_met(self, mock_fix):
        """Case 3: dsa_cp on + SP conditions met (+indexer) -> SP auto-kept, dsa stays on."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
            vc = VllmConfig()
            vc.model_config = SimpleNamespace(
                is_moe=False,
                hf_text_config=SimpleNamespace(index_topk=2048),
                hf_config=SimpleNamespace(),
                enforce_eager=True,
                architectures=[],
            )
            vc.parallel_config.enable_expert_parallel = True
            vc.parallel_config.tensor_parallel_size = 2
            vc.parallel_config.data_parallel_size = 2
            vc.parallel_config.all2all_backend = "allgather_reducescatter"
            vc.additional_config = {"enable_dsa_cp": True}

            config = init_ascend_config(vc)

            self.assertEqual(vc.parallel_config.all2all_backend, "allgather_reducescatter")
            self.assertTrue(enable_sp(vc))
            self.assertTrue(config.enable_dsa_cp)
            self.assertTrue(enable_dsa_cp())

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_dsa_cp_enabled_auto_disabled_when_sp_conditions_not_met(self, mock_fix):
        """Case 4: dsa_cp on + SP conditions NOT met (tp=1) -> dsa auto-disabled."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
            vc = VllmConfig()
            vc.model_config = SimpleNamespace(
                is_moe=False,
                hf_text_config=SimpleNamespace(index_topk=2048),
                hf_config=SimpleNamespace(),
                enforce_eager=True,
                architectures=[],
            )
            vc.parallel_config.enable_expert_parallel = True
            vc.parallel_config.tensor_parallel_size = 1
            vc.parallel_config.data_parallel_size = 2
            vc.parallel_config.all2all_backend = "allgather_reducescatter"
            vc.additional_config = {"enable_dsa_cp": True}

            config = init_ascend_config(vc)

            self.assertFalse(vc.parallel_config.use_sequence_parallel_moe)
            self.assertFalse(enable_sp(vc))
            self.assertFalse(config.enable_dsa_cp)
            self.assertFalse(enable_dsa_cp())

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
        # enable_sparse_li_c8 is derived from indexer_kv_dtype (see
        # init_ascend_config): indexer_kv_dtype "int8" makes it active.
        vc.attention_config.indexer_kv_dtype = "int8"

        config = init_ascend_config(vc)

        self.assertTrue(config.is_sparse_li_c8_layer("model.layers.3.self_attn.indexer.k_cache"))
        self.assertFalse(config.is_sparse_li_c8_layer("model.layers.4.self_attn.indexer.k_cache"))

    @_clean_up
    @patch("vllm_ascend.utils.model_uses_sfa_sparse", return_value=True)
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_sparse_sfa_user_input_is_derived_on_factory_path(self, mock_fix, mock_sparse):
        vc = VllmConfig()
        # enable_sparse_sfa_c8 is derived from cache_dtype (see
        # init_ascend_config): cache_dtype "fp8" makes it active.
        vc.cache_config.cache_dtype = "fp8"

        config = init_ascend_config(vc)

        self.assertTrue(config.enable_sparse_sfa_c8)

    @_clean_up
    @patch("vllm_ascend.utils.model_uses_sfa_sparse")
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_c8_reshape_optim_is_initialized_from_sfa_li_c8_and_pd_role(
        self,
        mock_fix,
        mock_uses_sfa,
    ):
        cases = (
            (None, True, True, "kv_producer", True),
            (False, True, True, "kv_producer", False),
            (True, False, True, "kv_producer", False),
            (True, True, False, "kv_producer", False),
            (True, True, True, "kv_consumer", False),
            (True, True, True, "kv_both", False),
            (True, True, True, None, False),
        )
        for reshape_optim, uses_sfa, enable_li_c8, kv_role, expected in cases:
            with self.subTest(
                reshape_optim=reshape_optim,
                uses_sfa=uses_sfa,
                enable_li_c8=enable_li_c8,
                kv_role=kv_role,
            ):
                mock_uses_sfa.return_value = uses_sfa
                vc = VllmConfig()
                vc.additional_config = {
                    "refresh": True,
                    "enable_sparse_li_c8": enable_li_c8,
                }
                if reshape_optim is not None:
                    vc.additional_config["c8_enable_reshape_optim"] = reshape_optim
                # enable_sparse_li_c8 is derived from indexer_kv_dtype (see
                # init_ascend_config); the per-case flag is expressed there.
                vc.attention_config.indexer_kv_dtype = "int8" if enable_li_c8 else "auto"
                if kv_role is not None:
                    vc.kv_transfer_config = KVTransferConfig(
                        kv_connector="MooncakeConnectorV1",
                        kv_role=kv_role,
                    )

                config = init_ascend_config(vc)

                self.assertEqual(config.c8_reshape_optim_enabled, expected)

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
        vc.model_config = SimpleNamespace(
            is_moe=False,
            is_deepseek_mla=True,
            architectures=[],
            enforce_eager=True,
        )
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
        with (
            patch("vllm_ascend.ascend_config.importlib.util.find_spec", return_value=None),
            self.assertRaises(ValueError),
        ):
            init_ascend_config(vc)

    @_clean_up
    @patch("vllm_ascend.ascend_config.logger.warning")
    @patch(
        "vllm_ascend.ascend_config.importlib.util.find_spec",
        side_effect=lambda name, *args, **kwargs: (
            object() if name == "vllm_omni" else real_find_spec(name, *args, **kwargs)
        ),
    )
    def test_omni_additional_config_warns_and_is_preserved(self, _mock_find_spec, mock_warning):
        vllm_config = VllmConfig()
        vllm_config.additional_config = {"vllm_omni_option": True}

        init_ascend_config(vllm_config)

        self.assertIs(vllm_config.additional_config["vllm_omni_option"], True)
        mock_warning.assert_any_call(
            "The following additional_config keys are invalid for vLLM-Ascend: %s. "
            "They may be used by vLLM-Omni or another project. "
            "Please remove them if they are not needed for your use case.",
            ["vllm_omni_option"],
        )

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_combine_quant_mode_defaults_zero(self, mock_fix):
        vc = VllmConfig()
        self.assertEqual(init_ascend_config(vc).combine_quant_mode, 0)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_combine_quant_mode_accepts_whitelisted_int(self, mock_fix):
        # combine_quant_mode is a Literal[0, 2, 3, 4], so only the whitelisted
        # integer values are accepted. Unlike the plain-int top-level switches
        # (e.g. weight_nz_mode), int strings ("4") are rejected rather than
        # lax-coerced, so the orthogonal test below covers that.
        for value in (0, 2, 4):
            with self.subTest(value=value):
                vc = VllmConfig()
                vc.additional_config = {"combine_quant_mode": value}
                self.assertEqual(init_ascend_config(vc).combine_quant_mode, value)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_combine_quant_mode_rejects_int_string(self, mock_fix):
        # The Literal whitelist does not lax-coerce int strings; a JSON-parsed
        # "4" must be rejected rather than silently accepted.
        vc = VllmConfig()
        vc.additional_config = {"combine_quant_mode": "4"}
        with self.assertRaises(ValueError):
            init_ascend_config(vc)

    @_clean_up
    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_combine_quant_mode_rejects_non_integer(self, mock_fix):
        # A non-integer (e.g. bool string "true") must be rejected rather than
        # silently coerced into an unexpected quant mode.
        vc = VllmConfig()
        vc.additional_config = {"combine_quant_mode": "true"}
        with self.assertRaises(ValueError):
            init_ascend_config(vc)


class TestKVPPConfig(TestBase):
    def test_graph_modes(self):
        from types import SimpleNamespace

        from vllm.config import CUDAGraphMode

        from tests.ut.kvpp_utils import make_kvpp_config
        from vllm_ascend.ascend_config import KVPPConfig

        for mode in CUDAGraphMode:
            with self.subTest(mode=mode):
                config = make_kvpp_config()
                config.model_config.enforce_eager = False
                config.compilation_config = SimpleNamespace(cudagraph_mode=mode)
                if mode == CUDAGraphMode.PIECEWISE:
                    KVPPConfig.from_vllm_config(config).validate(config)
                else:
                    with self.assertRaisesRegex(ValueError, "PIECEWISE"):
                        KVPPConfig.from_vllm_config(config).validate(config)

    def test_enable_switch_uses_tp_size(self):
        from tests.ut.kvpp_utils import make_kvpp_config
        from vllm_ascend.ascend_config import KVPPConfig

        for additional, tp, expected in (
            (None, 4, 1),
            ({}, 4, 1),
            ({"enable_kvpp": False}, 4, 1),
            ({"enable_kvpp": "false"}, 4, 1),
            ({"enable_kvpp": True}, 4, 4),
            ({"enable_kvpp": "true"}, 4, 4),
            ({"enable_kvpp": True}, 1, 1),
        ):
            with self.subTest(additional=additional, tp=tp):
                config = make_kvpp_config(tp)
                config.additional_config = additional
                self.assertEqual(KVPPConfig.from_vllm_config(config).size, expected)
        config.additional_config = {"enable_kvpp": "invalid"}
        with self.assertRaisesRegex(ValueError, "enable_kvpp"):
            KVPPConfig.from_vllm_config(config)

    def test_supported_configuration_and_restrictions(self):
        from tests.ut.kvpp_utils import make_kvpp_config
        from vllm_ascend.ascend_config import KVPPConfig
        from vllm_ascend.platform import _validate_parallel_config

        config = make_kvpp_config()
        KVPPConfig.from_vllm_config(config).validate(config)
        config.speculative_config = None
        KVPPConfig.from_vllm_config(config).validate(config)
        restrictions = (
            ("parallel_config", "decode_context_parallel_size", 2, "DCP"),
            ("model_config", "use_mla", False, "MLA"),
            ("model_config", "is_hybrid", True, "MLA"),
            ("speculative_config", "method", "eagle3", "mtp"),
            ("speculative_config", "num_speculative_tokens_per_batch_size", {1: 2}, "fixed"),
        )
        for section, field, value, message in restrictions:
            with self.subTest(field=field):
                config = make_kvpp_config()
                config.use_v2_model_runner = True
                setattr(getattr(config, section) if section else config, field, value)
                # Reach KVPP validation through the real platform entry point.
                with self.assertRaisesRegex(ValueError, message):
                    _validate_parallel_config(config)

    def test_dspark_accepts_fixed_length_and_rejects_dynamic_verification(self):
        config = make_kvpp_config()
        config.speculative_config.method = "dspark"
        KVPPConfig.from_vllm_config(config).validate(config)
        config.speculative_config.enable_adaptive_verification = True
        with self.assertRaisesRegex(ValueError, "adaptive verification"):
            KVPPConfig.from_vllm_config(config).validate(config)
        config.speculative_config.enable_adaptive_verification = False
        config.additional_config["dynamic_spec_config"] = {"method": "dspark"}
        with self.assertRaisesRegex(ValueError, "dynamic speculative lengths"):
            KVPPConfig.from_vllm_config(config).validate(config)

    @patch("vllm_ascend.platform.NPUPlatform.check_and_update_config")
    def test_config_factory_keeps_kvpp_enabled(self, _check_config):
        clear_ascend_config()
        self.addCleanup(clear_ascend_config)
        self.addCleanup(clear_enable_sp)
        config = VllmConfig()
        config.parallel_config.tensor_parallel_size = 4
        config.additional_config = {"enable_kvpp": True}
        actual = init_ascend_config(config)
        self.assertEqual(actual.kvpp_config.size, 4)
