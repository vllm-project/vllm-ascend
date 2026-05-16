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

from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm_ascend.ops.weight_prefetch import (
    MAX_PREFETCH_WEIGHT_SIZE,
    MOE_PREFETCH_TOKEN_THRESHOLD,
    SUPPORTED_MODULES,
    ModuleWeightPrefetchConfig,
    WeightPrefetchMethod,
    maybe_npu_prefetch,
)


class TestModuleWeightPrefetchConfig:
    def test_init_with_valid_module_name(self):
        for module_name in SUPPORTED_MODULES:
            config = ModuleWeightPrefetchConfig(module_name=module_name)
            assert config.module_name == module_name
            assert config.enable is False
            assert config.is_active_this_forward is False
            assert config.prefetch_ratio == {}
            assert config.linear_prefix_map == {}

    def test_init_with_invalid_module_name(self):
        with pytest.raises(AssertionError, match="Invalid module name"):
            ModuleWeightPrefetchConfig(module_name="invalid_module")

    def test_prefetch_ratio_filtering(self):
        config = ModuleWeightPrefetchConfig(
            module_name="attn",
            prefetch_ratio={"qkv": 0.8, "o": 1.2, "invalid": -0.5},
        )
        assert "qkv" in config.prefetch_ratio
        assert "o" not in config.prefetch_ratio
        assert "invalid" not in config.prefetch_ratio

    def test_enable_logic_with_prefetch_ratio(self):
        config = ModuleWeightPrefetchConfig(
            module_name="attn",
            enable=True,
            prefetch_ratio={"qkv": 0.8},
        )
        assert config.enable is True

    def test_enable_logic_without_prefetch_ratio(self):
        config = ModuleWeightPrefetchConfig(
            module_name="attn",
            enable=True,
            prefetch_ratio={},
        )
        assert config.enable is False


class TestWeightPrefetchMethod:
    @pytest.fixture
    def mock_weight_prefetch_config(self):
        config = MagicMock()
        config.enabled = True
        config.prefetch_ratio = {
            "attn": {"qkv": 0.8, "o": 0.8},
            "moe": {"gate_up": 0.8},
            "mlp": {"gate_up": 1.0, "down": 1.0},
        }
        return config

    @pytest.fixture
    def mock_vllm_config(self):
        config = MagicMock()
        config.model_config = MagicMock()
        config.model_config.hf_config = MagicMock()
        config.model_config.hf_config.model_type = "llama"
        return config

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=False)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    def test_init_non_moe_model(
        self,
        mock_get_config,
        mock_is_moe,
        mock_weight_prefetch_config,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        method = WeightPrefetchMethod(mock_weight_prefetch_config)

        assert method.is_moe is False
        assert method.attn.enable is True
        assert method.moe.enable is False
        assert method.mlp.enable is True

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=True)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    def test_init_moe_model(
        self,
        mock_get_config,
        mock_is_moe,
        mock_weight_prefetch_config,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        method = WeightPrefetchMethod(mock_weight_prefetch_config)

        assert method.is_moe is True
        assert method.attn.enable is True
        assert method.moe.enable is True
        assert method.mlp.enable is False

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=False)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    def test_init_disabled_config(
        self,
        mock_get_config,
        mock_is_moe,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        disabled_config = MagicMock()
        disabled_config.enabled = False
        disabled_config.prefetch_ratio = {}

        method = WeightPrefetchMethod(disabled_config)

        assert method.attn.enable is False
        assert method.moe.enable is False
        assert method.mlp.enable is False

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=False)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("torch.ops.vllm.prefetch_preprocess")
    def test_maybe_prefetch_attn_weight_preprocess_enabled(
        self,
        mock_prefetch,
        mock_get_config,
        mock_is_moe,
        mock_weight_prefetch_config,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        method = WeightPrefetchMethod(mock_weight_prefetch_config)

        weight = torch.randn(1024, 1024)
        start_flag = torch.tensor([1])

        method.maybe_prefetch_attn_weight_preprocess(
            layer_cls_name="AscendQKVParallelLinear",
            weight=weight,
            start_flag=start_flag,
        )

        mock_prefetch.assert_called_once()
        call_kwargs = mock_prefetch.call_args[1]
        assert call_kwargs["weight"] is weight
        assert call_kwargs["start_flag"] is start_flag

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=False)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("torch.ops.vllm.prefetch_preprocess")
    def test_maybe_prefetch_attn_weight_preprocess_disabled(
        self,
        mock_prefetch,
        mock_get_config,
        mock_is_moe,
        mock_weight_prefetch_config,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        disabled_config = MagicMock()
        disabled_config.enabled = False
        disabled_config.prefetch_ratio = {}

        method = WeightPrefetchMethod(disabled_config)

        weight = torch.randn(1024, 1024)
        start_flag = torch.tensor([1])

        method.maybe_prefetch_attn_weight_preprocess(
            layer_cls_name="AscendQKVParallelLinear",
            weight=weight,
            start_flag=start_flag,
        )

        mock_prefetch.assert_not_called()

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=False)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("torch.ops.vllm.prefetch_postprocess")
    def test_maybe_prefetch_attn_weight_postprocess_enabled(
        self,
        mock_postprocess,
        mock_get_config,
        mock_is_moe,
        mock_weight_prefetch_config,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        method = WeightPrefetchMethod(mock_weight_prefetch_config)

        stop_flag = torch.tensor([1])
        method.maybe_prefetch_attn_weight_postprocess(
            layer_cls_name="AscendQKVParallelLinear",
            stop_flag=stop_flag,
        )

        mock_postprocess.assert_called_once_with(stop_flag)

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=False)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("torch.ops.vllm.prefetch_postprocess")
    def test_maybe_prefetch_attn_weight_postprocess_disabled(
        self,
        mock_postprocess,
        mock_get_config,
        mock_is_moe,
        mock_weight_prefetch_config,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        disabled_config = MagicMock()
        disabled_config.enabled = False
        disabled_config.prefetch_ratio = {}

        method = WeightPrefetchMethod(disabled_config)

        stop_flag = torch.tensor([1])
        method.maybe_prefetch_attn_weight_postprocess(
            layer_cls_name="AscendQKVParallelLinear",
            stop_flag=stop_flag,
        )

        mock_postprocess.assert_not_called()

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=True)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("vllm_ascend.ops.weight_prefetch.get_forward_context")
    @patch("vllm_ascend.ops.weight_prefetch._EXTRA_CTX")
    @patch("torch.ops.vllm.prefetch_preprocess")
    def test_maybe_prefetch_moe_weight_preprocess_enabled(
        self,
        mock_prefetch,
        mock_extra_ctx,
        mock_get_forward_context,
        mock_get_config,
        mock_is_moe,
        mock_weight_prefetch_config,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        mock_get_forward_context.return_value = MagicMock()

        mock_model_instance = MagicMock()
        mock_layer = MagicMock()
        mock_layer.mlp.experts.w13_weight = torch.randn(1024, 1024)
        mock_model_instance.model.layers = [mock_layer]
        mock_extra_ctx.model_instance = mock_model_instance
        mock_extra_ctx.layer_idx = 1

        method = WeightPrefetchMethod(mock_weight_prefetch_config)

        hidden_states = torch.randn(MOE_PREFETCH_TOKEN_THRESHOLD + 10, 1024)
        method.maybe_prefetch_moe_weight_preprocess(hidden_states, "gate_up")

        assert method.moe.is_active_this_forward is True
        mock_prefetch.assert_called_once()

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=True)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    def test_maybe_prefetch_moe_weight_preprocess_below_threshold(
        self,
        mock_get_config,
        mock_is_moe,
        mock_weight_prefetch_config,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        method = WeightPrefetchMethod(mock_weight_prefetch_config)

        hidden_states = torch.randn(MOE_PREFETCH_TOKEN_THRESHOLD - 10, 1024)
        method.maybe_prefetch_moe_weight_preprocess(hidden_states, "gate_up")

        assert method.moe.is_active_this_forward is False

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=True)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("torch.ops.vllm.prefetch_postprocess")
    def test_maybe_prefetch_moe_weight_postprocess_enabled(
        self,
        mock_postprocess,
        mock_get_config,
        mock_is_moe,
        mock_weight_prefetch_config,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        method = WeightPrefetchMethod(mock_weight_prefetch_config)
        method.moe.is_active_this_forward = True

        stop_flag = torch.tensor([1])
        method.maybe_prefetch_moe_weight_postprocess(stop_flag)

        mock_postprocess.assert_called_once_with(stop_flag)

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=False)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("torch.ops.vllm.prefetch_postprocess")
    def test_maybe_prefetch_moe_weight_postprocess_disabled(
        self,
        mock_postprocess,
        mock_get_config,
        mock_is_moe,
        mock_weight_prefetch_config,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        method = WeightPrefetchMethod(mock_weight_prefetch_config)
        method.moe.is_active_this_forward = False

        stop_flag = torch.tensor([1])
        method.maybe_prefetch_moe_weight_postprocess(stop_flag)

        mock_postprocess.assert_not_called()

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=False)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("torch.ops.vllm.prefetch_preprocess")
    def test_maybe_prefetch_mla_or_sla_weight_enabled(
        self,
        mock_prefetch,
        mock_get_config,
        mock_is_moe,
        mock_weight_prefetch_config,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        method = WeightPrefetchMethod(mock_weight_prefetch_config)

        inputs = torch.randn(1024, 1024)
        dependency = torch.tensor([1])

        method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=inputs,
            dependency=dependency,
            max_size=1024,
        )

        mock_prefetch.assert_called_once()

    @patch("vllm_ascend.ops.weight_prefetch.is_moe_model", return_value=False)
    @patch("vllm_ascend.ops.weight_prefetch.get_current_vllm_config")
    @patch("torch.ops.vllm.prefetch_preprocess")
    def test_maybe_prefetch_mla_or_sla_weight_disabled(
        self,
        mock_prefetch,
        mock_get_config,
        mock_is_moe,
        mock_vllm_config,
    ):
        mock_get_config.return_value = mock_vllm_config
        disabled_config = MagicMock()
        disabled_config.enabled = False
        disabled_config.prefetch_ratio = {}

        method = WeightPrefetchMethod(disabled_config)

        inputs = torch.randn(1024, 1024)
        dependency = torch.tensor([1])

        method.maybe_prefetch_mla_or_sla_weight_in_current_stream(
            inputs=inputs,
            dependency=dependency,
        )

        mock_prefetch.assert_not_called()


class TestMaybeNpuPrefetch:
    @patch("torch_npu.npu_prefetch")
    def test_maybe_npu_prefetch_enabled(self, mock_prefetch):
        inputs = torch.randn(1024, 1024)
        dependency = torch.tensor([1])

        maybe_npu_prefetch(inputs, dependency, enabled=True)

        mock_prefetch.assert_called_once()
        call_args = mock_prefetch.call_args[0]
        assert call_args[0] is inputs
        assert call_args[1] is dependency

    @patch("torch_npu.npu_prefetch")
    def test_maybe_npu_prefetch_disabled(self, mock_prefetch):
        inputs = torch.randn(1024, 1024)
        dependency = torch.tensor([1])

        maybe_npu_prefetch(inputs, dependency, enabled=False)

        mock_prefetch.assert_not_called()

    @patch("torch_npu.npu_prefetch")
    def test_maybe_npu_prefetch_max_size_calculation(self, mock_prefetch):
        inputs = torch.randn(100, 100, dtype=torch.float32)
        dependency = torch.tensor([1])

        maybe_npu_prefetch(inputs, dependency, max_size=0, enabled=True)

        expected_size = inputs.element_size() * inputs.numel()
        call_args = mock_prefetch.call_args[0]
        assert call_args[2] == expected_size

    @patch("torch_npu.npu_prefetch")
    def test_maybe_npu_prefetch_with_custom_max_size(self, mock_prefetch):
        inputs = torch.randn(100, 100, dtype=torch.float32)
        dependency = torch.tensor([1])
        custom_max_size = 1000

        maybe_npu_prefetch(inputs, dependency, max_size=custom_max_size, enabled=True)

        call_args = mock_prefetch.call_args[0]
        assert call_args[2] == custom_max_size

    @patch("torch_npu.npu_prefetch")
    def test_maybe_npu_prefetch_with_offset(self, mock_prefetch):
        inputs = torch.randn(1024, 1024)
        dependency = torch.tensor([1])
        offset = 100

        maybe_npu_prefetch(inputs, dependency, offset=offset, enabled=True)

        call_args = mock_prefetch.call_args[0]
        assert call_args[3] == offset


class TestConstants:
    def test_supported_modules(self):
        assert "attn" in SUPPORTED_MODULES
        assert "mlp" in SUPPORTED_MODULES
        assert "moe" in SUPPORTED_MODULES

    def test_moe_prefetch_token_threshold(self):
        assert MOE_PREFETCH_TOKEN_THRESHOLD == 96

    def test_max_prefetch_weight_size(self):
        assert MAX_PREFETCH_WEIGHT_SIZE == 18 * 1024 * 1024
