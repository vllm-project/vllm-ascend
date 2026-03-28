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

from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.ops.flashcomm2_oshard_manager import (
    Flashcomm2OShardManager,
    flashcomm2_oshard_manager,
)


class TestFlashcomm2OShardManager:
    @pytest.fixture
    def manager(self):
        return Flashcomm2OShardManager()

    def test_init(self, manager):
        assert manager._shard_layers == {}
        assert isinstance(manager._shard_layers, dict)

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.flashcomm2_enable")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.o_shard_enable")
    def test_flashcomm2_oshard_enable_both_enabled(
        self,
        mock_o_shard_enable,
        mock_flashcomm2_enable,
        manager,
    ):
        mock_flashcomm2_enable.return_value = True
        mock_o_shard_enable.return_value = True

        result = manager.flashcomm2_oshard_enable()

        assert result is True

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.flashcomm2_enable")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.o_shard_enable")
    def test_flashcomm2_oshard_enable_flashcomm2_disabled(
        self,
        mock_o_shard_enable,
        mock_flashcomm2_enable,
        manager,
    ):
        mock_flashcomm2_enable.return_value = False
        mock_o_shard_enable.return_value = True

        result = manager.flashcomm2_oshard_enable()

        assert result is False

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.flashcomm2_enable")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.o_shard_enable")
    def test_flashcomm2_oshard_enable_o_shard_disabled(
        self,
        mock_o_shard_enable,
        mock_flashcomm2_enable,
        manager,
    ):
        mock_flashcomm2_enable.return_value = True
        mock_o_shard_enable.return_value = False

        result = manager.flashcomm2_oshard_enable()

        assert result is False

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.flashcomm2_enable")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.o_shard_enable")
    def test_flashcomm2_oshard_enable_both_disabled(
        self,
        mock_o_shard_enable,
        mock_flashcomm2_enable,
        manager,
    ):
        mock_flashcomm2_enable.return_value = False
        mock_o_shard_enable.return_value = False

        result = manager.flashcomm2_oshard_enable()

        assert result is False

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.register_layer_to_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.get_shard_weight_group")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    def test_register_layer_hidden_layer(
        self,
        mock_extract_index,
        mock_is_hidden,
        mock_get_group,
        mock_register,
        manager,
    ):
        mock_is_hidden.return_value = True
        mock_extract_index.return_value = 5
        mock_group = MagicMock()
        mock_get_group.return_value = mock_group

        layer = MagicMock()
        layer.prefix = "model.layers.5.self_attn.o_proj"

        manager.register_layer(layer, prefetch_step=2)

        assert 5 in manager._shard_layers
        assert manager._shard_layers[5] is layer
        mock_register.assert_called_once_with(
            series_name="o_proj",
            group=mock_group,
            layer=layer,
            prefetch_step=2,
        )

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.register_layer_to_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.get_shard_weight_group")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    def test_register_layer_non_hidden_layer(
        self,
        mock_is_hidden,
        mock_get_group,
        mock_register,
        manager,
    ):
        mock_is_hidden.return_value = False

        layer = MagicMock()
        layer.prefix = "model.layers.100.self_attn.o_proj"

        manager.register_layer(layer)

        assert len(manager._shard_layers) == 0
        mock_register.assert_not_called()

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.register_layer_to_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.get_shard_weight_group")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    def test_register_layer_default_prefetch_step(
        self,
        mock_extract_index,
        mock_is_hidden,
        mock_get_group,
        mock_register,
        manager,
    ):
        mock_is_hidden.return_value = True
        mock_extract_index.return_value = 0
        mock_group = MagicMock()
        mock_get_group.return_value = mock_group

        layer = MagicMock()
        layer.prefix = "model.layers.0.self_attn.o_proj"

        manager.register_layer(layer)

        mock_register.assert_called_once_with(
            series_name="o_proj",
            group=mock_group,
            layer=layer,
            prefetch_step=1,
        )

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.register_layer_to_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.get_shard_weight_group")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    def test_register_layer_overwrites_existing_layer_with_same_index(
        self,
        mock_extract_index,
        mock_is_hidden,
        mock_get_group,
        mock_register,
        manager,
    ):
        mock_is_hidden.return_value = True
        mock_extract_index.return_value = 5
        mock_group = MagicMock()
        mock_get_group.return_value = mock_group

        first_layer = MagicMock()
        first_layer.prefix = "model.layers.5.self_attn.o_proj"
        second_layer = MagicMock()
        second_layer.prefix = "model.layers.5.self_attn.o_proj"

        manager.register_layer(first_layer)
        manager.register_layer(second_layer)

        assert len(manager._shard_layers) == 1
        assert manager.get_layer(5) is second_layer
        assert mock_register.call_count == 2

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.register_layer_to_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.get_shard_weight_group")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    def test_register_layer_missing_prefix_raises(
        self,
        mock_is_hidden,
        mock_get_group,
        mock_register,
        manager,
    ):
        mock_is_hidden.return_value = True
        layer = MagicMock(spec=[])

        with pytest.raises(AttributeError):
            manager.register_layer(layer)

        assert manager._shard_layers == {}
        mock_get_group.assert_not_called()
        mock_register.assert_not_called()

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.register_layer_to_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.get_shard_weight_group")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    def test_register_layer_extract_layer_index_failure_propagates(
        self,
        mock_extract_index,
        mock_is_hidden,
        mock_get_group,
        mock_register,
        manager,
    ):
        mock_is_hidden.return_value = True
        mock_extract_index.side_effect = ValueError("invalid layer prefix")

        layer = MagicMock()
        layer.prefix = "invalid-prefix"

        with pytest.raises(ValueError, match="invalid layer prefix"):
            manager.register_layer(layer)

        assert manager._shard_layers == {}
        mock_get_group.assert_not_called()
        mock_register.assert_not_called()

    def test_get_layer_existing(self, manager):
        layer = MagicMock()
        manager._shard_layers[5] = layer

        result = manager.get_layer(5)

        assert result is layer

    def test_get_layer_non_existing(self, manager):
        result = manager.get_layer(999)

        assert result is None

    def test_get_layer_empty_dict(self, manager):
        result = manager.get_layer(0)

        assert result is None

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.reach_layer_for_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    def test_trigger_broadcast_for_layer_success(
        self,
        mock_extract_index,
        mock_is_hidden,
        mock_reach_layer,
        manager,
    ):
        mock_extract_index.return_value = 3
        mock_is_hidden.return_value = True

        layer = MagicMock()
        manager._shard_layers[3] = layer

        manager.trigger_broadcast_for_layer("model.layers.3.self_attn.o_proj")

        mock_reach_layer.assert_called_once_with(layer)

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.reach_layer_for_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    def test_trigger_broadcast_for_layer_not_hidden(
        self,
        mock_extract_index,
        mock_is_hidden,
        mock_reach_layer,
        manager,
    ):
        mock_extract_index.return_value = 3
        mock_is_hidden.return_value = False

        layer = MagicMock()
        manager._shard_layers[3] = layer

        manager.trigger_broadcast_for_layer("model.layers.3.self_attn.o_proj")

        mock_reach_layer.assert_not_called()

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.reach_layer_for_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    def test_trigger_broadcast_for_layer_not_registered(
        self,
        mock_extract_index,
        mock_reach_layer,
        manager,
    ):
        mock_extract_index.return_value = 999

        manager.trigger_broadcast_for_layer("model.layers.999.self_attn.o_proj")

        mock_reach_layer.assert_not_called()

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.reach_layer_for_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    def test_trigger_broadcast_for_layer_not_registered_short_circuits_hidden_check(
        self,
        mock_extract_index,
        mock_is_hidden,
        mock_reach_layer,
        manager,
    ):
        mock_extract_index.return_value = 999

        manager.trigger_broadcast_for_layer("model.layers.999.self_attn.o_proj")

        mock_is_hidden.assert_not_called()
        mock_reach_layer.assert_not_called()

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.reach_layer_for_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    def test_trigger_broadcast_for_layer_empty_manager(
        self,
        mock_extract_index,
        mock_is_hidden,
        mock_reach_layer,
        manager,
    ):
        mock_extract_index.return_value = 0

        manager.trigger_broadcast_for_layer("model.layers.0.self_attn.o_proj")

        mock_is_hidden.assert_not_called()
        mock_reach_layer.assert_not_called()

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.reach_layer_for_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    def test_trigger_broadcast_for_layer_extract_layer_index_failure_propagates(
        self,
        mock_extract_index,
        mock_reach_layer,
        manager,
    ):
        mock_extract_index.side_effect = ValueError("invalid layer prefix")

        with pytest.raises(ValueError, match="invalid layer prefix"):
            manager.trigger_broadcast_for_layer("invalid-prefix")

        mock_reach_layer.assert_not_called()

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.post_process_after_loading_for_shard_weight_series")
    def test_post_process_after_loading_with_layers(self, mock_post_process, manager):
        layer1 = MagicMock()
        layer2 = MagicMock()
        manager._shard_layers[0] = layer1
        manager._shard_layers[1] = layer2

        manager.post_process_after_loading()

        mock_post_process.assert_called_once()
        called_layer = mock_post_process.call_args[0][0]
        assert called_layer in [layer1, layer2]

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.post_process_after_loading_for_shard_weight_series")
    def test_post_process_after_loading_uses_first_registered_layer(self, mock_post_process, manager):
        first_layer = MagicMock()
        second_layer = MagicMock()
        manager._shard_layers[1] = first_layer
        manager._shard_layers[2] = second_layer

        manager.post_process_after_loading()

        mock_post_process.assert_called_once_with(first_layer)

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.post_process_after_loading_for_shard_weight_series")
    def test_post_process_after_loading_empty(self, mock_post_process, manager):
        manager.post_process_after_loading()

        mock_post_process.assert_not_called()

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.post_process_after_loading_for_shard_weight_series")
    def test_post_process_after_loading_single_layer(self, mock_post_process, manager):
        layer = MagicMock()
        manager._shard_layers[5] = layer

        manager.post_process_after_loading()

        mock_post_process.assert_called_once_with(layer)


class TestGlobalInstance:
    def test_global_instance_exists(self):
        assert flashcomm2_oshard_manager is not None
        assert isinstance(flashcomm2_oshard_manager, Flashcomm2OShardManager)

    def test_global_instance_has_shard_layers(self):
        assert hasattr(flashcomm2_oshard_manager, "_shard_layers")
        assert isinstance(flashcomm2_oshard_manager._shard_layers, dict)


class TestIntegration:
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.register_layer_to_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.get_shard_weight_group")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.reach_layer_for_shard_weight_series")
    def test_full_workflow(
        self,
        mock_reach_layer,
        mock_extract_index,
        mock_is_hidden,
        mock_get_group,
        mock_register,
    ):
        manager = Flashcomm2OShardManager()

        mock_is_hidden.return_value = True
        mock_extract_index.side_effect = lambda x: int(x.split(".")[2])
        mock_group = MagicMock()
        mock_get_group.return_value = mock_group

        layer0 = MagicMock()
        layer0.prefix = "model.layers.0.self_attn.o_proj"

        layer1 = MagicMock()
        layer1.prefix = "model.layers.1.self_attn.o_proj"

        manager.register_layer(layer0)
        manager.register_layer(layer1)

        assert len(manager._shard_layers) == 2
        assert manager.get_layer(0) is layer0
        assert manager.get_layer(1) is layer1

        manager.trigger_broadcast_for_layer("model.layers.0.self_attn.o_proj")
        mock_reach_layer.assert_called_with(layer0)

        manager.trigger_broadcast_for_layer("model.layers.1.self_attn.o_proj")
        mock_reach_layer.assert_called_with(layer1)

    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.register_layer_to_shard_weight_series")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.get_shard_weight_group")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.is_hidden_layer")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.extract_layer_index")
    @patch("vllm_ascend.ops.flashcomm2_oshard_manager.post_process_after_loading_for_shard_weight_series")
    def test_register_and_post_process_workflow(
        self,
        mock_post_process,
        mock_extract_index,
        mock_is_hidden,
        mock_get_group,
        mock_register,
    ):
        manager = Flashcomm2OShardManager()

        mock_is_hidden.return_value = True
        mock_extract_index.return_value = 0
        mock_group = MagicMock()
        mock_get_group.return_value = mock_group

        layer = MagicMock()
        layer.prefix = "model.layers.0.self_attn.o_proj"

        manager.register_layer(layer)

        manager.post_process_after_loading()

        mock_post_process.assert_called_once()
