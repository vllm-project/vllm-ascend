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

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm_ascend.ops.layer_shard_linear import (
    LayerExternalMetadata,
    LayerMetadata,
    SeriesMetadata,
    ShardWindowMetadata,
    _create_forward_wrapper,
    dispose_tensor,
    is_hidden_layer,
    register_layer_to_shard_weight_series,
)


class TestDisposeTensor:
    def test_dispose_tensor_replaces_with_empty(self):
        original_tensor = torch.randn(10, 10)
        original_size = original_tensor.numel()

        dispose_tensor(original_tensor)

        assert original_tensor.numel() == 0
        assert original_tensor.numel() != original_size

    def test_dispose_tensor_preserves_device_and_dtype(self):
        original_tensor = torch.randn(5, 5, dtype=torch.float32)
        original_dtype = original_tensor.dtype

        dispose_tensor(original_tensor)

        assert original_tensor.dtype == original_dtype


class TestLayerMetadata:
    def test_layer_metadata_creation(self):
        layer = MagicMock()
        post_method = Mock()
        weight = torch.randn(10, 10)

        metadata = LayerMetadata(
            layer_idx=0,
            layer=layer,
            post_method=post_method,
            weight=weight,
            window_idx=0,
        )

        assert metadata.layer_idx == 0
        assert metadata.layer is layer
        assert metadata.post_method is post_method
        assert metadata.weight is weight
        assert metadata.window_idx == 0


class TestShardWindowMetadata:
    def test_shard_window_metadata_creation(self):
        weight = torch.randn(10, 10)

        window = ShardWindowMetadata(
            weight=weight,
            data_layer_idx=0,
            work=None,
        )

        assert window.weight is weight
        assert window.data_layer_idx == 0
        assert window.work is None


class TestSeriesMetadata:
    @pytest.fixture
    def mock_group(self):
        group = MagicMock()
        group.world_size = 2
        group.rank_in_group = 0
        group.ranks = [0, 1]
        group.device_group = MagicMock()
        return group

    @pytest.fixture
    def series_metadata(self, mock_group):
        return SeriesMetadata(
            group=mock_group,
            start_layer=0,
            end_layer=0,
            num_layers=0,
            prefetch_step=1,
            dummy_weight=torch.randn(10, 10),
            layers=[],
            shard_windows=[],
            window_offset=1,
        )

    def test_is_source_rank_zero(self, series_metadata):
        series_metadata.group.rank_in_group = 0

        assert series_metadata.is_source(0) is True
        assert series_metadata.is_source(1) is False
        assert series_metadata.is_source(2) is True
        assert series_metadata.is_source(3) is False

    def test_is_source_rank_one(self, series_metadata):
        series_metadata.group.rank_in_group = 1

        assert series_metadata.is_source(0) is False
        assert series_metadata.is_source(1) is True
        assert series_metadata.is_source(2) is False
        assert series_metadata.is_source(3) is True

    @patch("torch.distributed.broadcast")
    def test_post_process_after_loading_basic(self, mock_broadcast, series_metadata):
        layer0 = MagicMock()
        layer0.layer_idx = 0
        layer0.weight = torch.randn(10, 10)
        layer0.post_method = Mock()

        layer1 = MagicMock()
        layer1.layer_idx = 1
        layer1.weight = torch.randn(10, 10)
        layer1.post_method = Mock()

        series_metadata.layers = [layer0, layer1]
        series_metadata.prefetch_step = 0

        series_metadata.post_process_after_loading()

        assert series_metadata.num_layers == 2
        assert series_metadata.start_layer == 0
        assert series_metadata.end_layer == 2
        assert len(series_metadata.shard_windows) == 1
        assert mock_broadcast.call_count == 2

    @patch("torch.distributed.broadcast")
    def test_post_process_after_loading_with_prefetch(self, mock_broadcast, series_metadata):
        layer0 = MagicMock()
        layer0.layer_idx = 0
        layer0.weight = torch.randn(10, 10)
        layer0.post_method = Mock()

        layer1 = MagicMock()
        layer1.layer_idx = 1
        layer1.weight = torch.randn(10, 10)
        layer1.post_method = Mock()

        layer2 = MagicMock()
        layer2.layer_idx = 2
        layer2.weight = torch.randn(10, 10)
        layer2.post_method = Mock()

        series_metadata.layers = [layer0, layer1, layer2]
        series_metadata.prefetch_step = 1

        series_metadata.post_process_after_loading()

        assert series_metadata.num_layers == 3
        assert len(series_metadata.shard_windows) == 2
        assert mock_broadcast.call_count == 3

    def test_post_process_after_loading_already_initialized(self, series_metadata):
        series_metadata.shard_windows = [MagicMock()]

        result = series_metadata.post_process_after_loading()

        assert result is None

    def test_post_process_after_loading_empty_layers(self, series_metadata):
        series_metadata.layers = []

        with pytest.raises(AssertionError, match="No layers in the series"):
            series_metadata.post_process_after_loading()

    @patch("torch.distributed.broadcast")
    def test_reach_layer(self, mock_broadcast, series_metadata):
        layer0 = MagicMock()
        layer0.layer_idx = 0
        layer0.weight = torch.randn(10, 10)
        layer0.window_idx = -1

        layer1 = MagicMock()
        layer1.layer_idx = 1
        layer1.weight = torch.randn(10, 10)
        layer1.window_idx = -1

        series_metadata.layers = [layer0, layer1]
        series_metadata.num_layers = 2
        series_metadata.start_layer = 0
        series_metadata.prefetch_step = 0
        series_metadata.window_offset = 0

        window = ShardWindowMetadata(
            weight=torch.randn(10, 10),
            data_layer_idx=-1,
            work=None,
        )
        series_metadata.shard_windows = [window]

        mock_work = MagicMock()
        mock_broadcast.return_value = mock_work

        series_metadata.reach_layer(0)

        assert layer1.window_idx == 0
        assert window.data_layer_idx == 1
        assert window.work is not None
        mock_broadcast.assert_called_once()

    @patch("torch.distributed.broadcast")
    def test_wait_weight(self, mock_broadcast, series_metadata):
        mock_work = MagicMock()
        window = ShardWindowMetadata(
            weight=torch.randn(10, 10),
            data_layer_idx=0,
            work=mock_work,
        )

        layer0 = MagicMock()
        layer0.layer_idx = 0
        layer0.window_idx = 0

        series_metadata.layers = [layer0]
        series_metadata.start_layer = 0
        series_metadata.shard_windows = [window]

        series_metadata.wait_weight(0)

        mock_work.wait.assert_called_once()
        assert window.work is None

    def test_wait_weight_no_work(self, series_metadata):
        window = ShardWindowMetadata(
            weight=torch.randn(10, 10),
            data_layer_idx=0,
            work=None,
        )

        layer0 = MagicMock()
        layer0.layer_idx = 0
        layer0.window_idx = 0

        series_metadata.layers = [layer0]
        series_metadata.start_layer = 0
        series_metadata.shard_windows = [window]

        series_metadata.wait_weight(0)

        assert window.work is None


class TestLayerExternalMetadata:
    def test_layer_external_metadata_creation(self):
        series = MagicMock()
        layer_idx = 5

        ext_metadata = LayerExternalMetadata(
            series=series,
            layer_idx=layer_idx,
        )

        assert ext_metadata.series is series
        assert ext_metadata.layer_idx == layer_idx


class TestCreateForwardWrapper:
    def test_create_forward_wrapper_calls_wait_weight(self):
        mock_series = MagicMock()
        mock_forward = Mock(return_value="output")
        layer_idx = 0

        wrapped = _create_forward_wrapper(mock_forward, mock_series, layer_idx)

        result = wrapped("arg1", "arg2", kwarg1="value1")

        mock_series.wait_weight.assert_called_once_with(layer_idx)
        mock_forward.assert_called_once_with("arg1", "arg2", kwarg1="value1")
        assert result == "output"

    def test_create_forward_wrapper_preserves_return_value(self):
        mock_series = MagicMock()
        expected_output = torch.randn(10, 10)
        mock_forward = Mock(return_value=expected_output)

        wrapped = _create_forward_wrapper(mock_forward, mock_series, 0)

        result = wrapped()

        assert result is expected_output


class TestRegisterLayerToShardWeightSeries:
    @pytest.fixture
    def mock_layer(self):
        layer = MagicMock()
        layer.weight = torch.randn(10, 10)
        layer.prefix = "model.layers.0.mlp.gate_up_proj"
        layer.forward = Mock(return_value="forward_output")

        quant_method = MagicMock()
        quant_method.process_weights_after_loading = Mock()
        layer.quant_method = quant_method

        return layer

    @pytest.fixture
    def mock_group(self):
        group = MagicMock()
        group.world_size = 2
        group.rank_in_group = 0
        group.ranks = [0, 1]
        return group

    @patch("vllm_ascend.ops.layer_shard_linear._series_dict", new_callable=dict)
    @patch("vllm_ascend.ops.layer_shard_linear._layer_external_dict", new_callable=dict)
    @patch("vllm_ascend.ops.layer_shard_linear.extract_layer_index", return_value=0)
    def test_register_layer_creates_new_series(
        self,
        mock_extract_index,
        mock_layer_dict,
        mock_series_dict,
        mock_layer,
        mock_group,
    ):
        import vllm_ascend.ops.layer_shard_linear as module

        module._series_dict = {}
        module._layer_external_dict = {}

        register_layer_to_shard_weight_series(
            series_name="test_series",
            group=mock_group,
            layer=mock_layer,
            prefetch_step=1,
        )

        assert "test_series" in module._series_dict
        series = module._series_dict["test_series"]
        assert series.group is mock_group
        assert series.prefetch_step == 1
        assert len(series.layers) == 1

    @patch("vllm_ascend.ops.layer_shard_linear._series_dict", new_callable=dict)
    @patch("vllm_ascend.ops.layer_shard_linear._layer_external_dict", new_callable=dict)
    @patch("vllm_ascend.ops.layer_shard_linear.extract_layer_index", return_value=1)
    def test_register_layer_adds_to_existing_series(
        self,
        mock_extract_index,
        mock_layer_dict,
        mock_series_dict,
        mock_layer,
        mock_group,
    ):
        import vllm_ascend.ops.layer_shard_linear as module

        existing_series = SeriesMetadata(
            group=mock_group,
            start_layer=0,
            end_layer=0,
            num_layers=0,
            prefetch_step=1,
            dummy_weight=torch.randn(10, 10),
            layers=[],
            shard_windows=[],
            window_offset=1,
        )
        module._series_dict = {"test_series": existing_series}
        module._layer_external_dict = {}

        register_layer_to_shard_weight_series(
            series_name="test_series",
            group=mock_group,
            layer=mock_layer,
            prefetch_step=1,
        )

        assert len(existing_series.layers) == 1
        assert existing_series.layers[0].layer_idx == 1

    @patch("vllm_ascend.ops.layer_shard_linear._series_dict", new_callable=dict)
    @patch("vllm_ascend.ops.layer_shard_linear._layer_external_dict", new_callable=dict)
    @patch("vllm_ascend.ops.layer_shard_linear.extract_layer_index", return_value=1)
    def test_register_layer_disposes_weight_for_non_source(
        self,
        mock_extract_index,
        mock_layer_dict,
        mock_series_dict,
        mock_layer,
        mock_group,
    ):
        import vllm_ascend.ops.layer_shard_linear as module

        mock_group.rank_in_group = 0

        module._series_dict = {}
        module._layer_external_dict = {}

        register_layer_to_shard_weight_series(
            series_name="test_series",
            group=mock_group,
            layer=mock_layer,
            prefetch_step=1,
        )

        series = module._series_dict["test_series"]
        assert series.is_source(1) is False


class TestIsHiddenLayer:
    @patch("vllm_ascend.ops.layer_shard_linear.get_current_model_num_hidden_layers")
    @patch("vllm_ascend.ops.layer_shard_linear.extract_layer_index")
    def test_is_hidden_layer_true(
        self,
        mock_extract_index,
        mock_get_num_layers,
    ):
        mock_get_num_layers.return_value = 32
        mock_extract_index.return_value = 10

        layer = MagicMock()
        layer.prefix = "model.layers.10.mlp"

        result = is_hidden_layer(layer)

        assert result is True

    @patch("vllm_ascend.ops.layer_shard_linear.get_current_model_num_hidden_layers")
    @patch("vllm_ascend.ops.layer_shard_linear.extract_layer_index")
    def test_is_hidden_layer_false(
        self,
        mock_extract_index,
        mock_get_num_layers,
    ):
        mock_get_num_layers.return_value = 32
        mock_extract_index.return_value = 40

        layer = MagicMock()
        layer.prefix = "model.layers.40.mlp"

        result = is_hidden_layer(layer)

        assert result is False

    @patch("vllm_ascend.ops.layer_shard_linear.get_current_model_num_hidden_layers")
    @patch("vllm_ascend.ops.layer_shard_linear.extract_layer_index")
    def test_is_hidden_layer_boundary(
        self,
        mock_extract_index,
        mock_get_num_layers,
    ):
        mock_get_num_layers.return_value = 32
        mock_extract_index.return_value = 31

        layer = MagicMock()
        layer.prefix = "model.layers.31.mlp"

        result = is_hidden_layer(layer)

        assert result is True
