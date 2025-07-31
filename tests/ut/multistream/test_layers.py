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
import torch

from tests.ut.base import PytestBase
from vllm_ascend.multistream.base import MSEventKey
from vllm_ascend.multistream.layers import (MultiStreamPostTransformerLayer,
                                            MultiStreamPreTransformerLayer)
from vllm_ascend.multistream.metadata import MultiStreamMetadata


# === fixture: mock tensor input ===
@pytest.fixture
def input_tensors():
    return [torch.randn(2, 128), torch.randn(2, 128)]


# === mock get_forward_context ===
class DummyContext:

    def __init__(self, attn_metadata):
        self.attn_metadata = attn_metadata


class TestMultiStreamPreTransformerLayer(PytestBase):

    # === test when multistream_metadata is None ===
    @patch("vllm_ascend.multistream.layers.get_forward_context")
    @patch("vllm_ascend.multistream.layers.set_multistream_layer_context")
    def test_forward_no_multistream_metadata(self, mock_set_ctx, mock_get_ctx,
                                             input_tensors):
        mock_get_ctx.return_value = DummyContext(attn_metadata="dummy_meta")
        layer = MultiStreamPreTransformerLayer(multistream_metadata=None)
        attn_out, input_out = layer.forward(input_tensors)

        assert attn_out == "dummy_meta"
        assert input_out == input_tensors
        mock_set_ctx.assert_called_once_with(-1, None, None)

    # === test when attn_metadata is None ===
    @patch("vllm_ascend.multistream.layers.get_forward_context")
    @patch("vllm_ascend.multistream.layers.set_multistream_layer_context")
    def test_forward_no_attn_metadata(self, mock_set_ctx, mock_get_ctx,
                                      input_tensors):
        mock_get_ctx.return_value = DummyContext(attn_metadata=None)
        dummy_metadata = MagicMock(spec=MultiStreamMetadata)
        layer = MultiStreamPreTransformerLayer(
            multistream_metadata=dummy_metadata)

        attn_out, input_out = layer.forward(input_tensors)

        assert attn_out is None
        assert input_out == input_tensors
        mock_set_ctx.assert_called_once_with(-1, None, None)

    # === test when do_ms=False (no split needed) ===
    @patch("vllm_ascend.multistream.layers.get_forward_context")
    @patch("vllm_ascend.multistream.layers.set_multistream_layer_context")
    def test_forward_no_split(self, mock_set_ctx, mock_get_ctx, input_tensors):
        dummy_attn = "original_attn"
        mock_get_ctx.return_value = DummyContext(attn_metadata=dummy_attn)

        dummy_metadata = MagicMock(spec=MultiStreamMetadata)
        dummy_metadata.split_micro_batch.return_value = (False, "same_attn",
                                                         input_tensors, None)

        layer = MultiStreamPreTransformerLayer(
            multistream_metadata=dummy_metadata)

        attn_out, input_out = layer.forward(input_tensors)

        assert attn_out == "same_attn"
        assert input_out == input_tensors
        mock_set_ctx.assert_called_once_with(-1, None, None)

    # === test when do_ms=True (split occurred) ===
    @patch("vllm_ascend.multistream.layers.get_forward_context")
    @patch("vllm_ascend.multistream.layers.set_multistream_layer_context")
    def test_forward_split(self, mock_set_ctx, mock_get_ctx, input_tensors):
        dummy_attn = "original_attn"
        mock_get_ctx.return_value = DummyContext(attn_metadata=dummy_attn)

        split_inputs = [[t[:1], t[1:]] for t in input_tensors]

        dummy_metadata = MagicMock(spec=MultiStreamMetadata)
        dummy_metadata.start_layer = 2
        dummy_metadata.split_micro_batch.return_value = (True,
                                                         ["attn1", "attn2"],
                                                         split_inputs, None)

        layer = MultiStreamPreTransformerLayer(
            multistream_metadata=dummy_metadata)

        attn_out, input_out = layer.forward(input_tensors)

        assert attn_out == ["attn1", "attn2"]
        assert input_out == split_inputs
        mock_set_ctx.assert_called_once_with(2, dummy_metadata,
                                             ["attn1", "attn2"])


class TestMultiStreamPostTransformerLayer(PytestBase):

    def test_post_forward_metadata_none(self, input_tensors):
        layer = MultiStreamPostTransformerLayer(multistream_metadata=None)
        output = layer.forward(input_tensors)
        assert output == input_tensors

        dummy_metadata = MagicMock(spec=MultiStreamMetadata)
        dummy_metadata.ms_config = None
        layer = MultiStreamPostTransformerLayer(
            multistream_metadata=dummy_metadata)
        output = layer.forward(input_tensors)
        assert output == input_tensors

    @patch("vllm_ascend.multistream.layers.get_multistream_layer_context")
    @patch("vllm_ascend.multistream.layers.reset_multistream_layer_context")
    def test_post_forward_normal_flow(self, mock_reset_ctx, mock_get_ctx,
                                      input_tensors):
        A_instance_of_MultiStreamMetadata = MultiStreamMetadata(
            calculate_stream=MagicMock(),
            communicate_stream=MagicMock(),
            start_layer=0,
            end_layer=1,
            event_keys=[],
            multistream_config=None,
        )
        dummy_metadata = MagicMock(spec=A_instance_of_MultiStreamMetadata)
        dummy_metadata.ms_config.num_micro_batches = 4
        dummy_metadata.end_layer = 10

        mock_get_ctx.return_value = (
            5,  # layer_index
            dummy_metadata,  # ms_metadata
            "dummy_attn_metadata"  # ms_attn_metadata
        )

        dummy_metadata.merge_micro_batches.return_value = "merged_result"

        layer = MultiStreamPostTransformerLayer(
            multistream_metadata=dummy_metadata)
        output = layer.forward(input_tensors)

        # check wait_event
        dummy_metadata.try_wait_event.assert_called_once_with(
            9,  # end_layer - 1
            3,  # num_micro_batches - 1
            MSEventKey.FFN_AR_FINISH)
        mock_reset_ctx.assert_called_once()
        assert output == "merged_result"

    @patch("vllm_ascend.multistream.layers.get_multistream_layer_context")
    @patch("vllm_ascend.multistream.layers.reset_multistream_layer_context")
    def test_post_forward_with_custom_wait_layer(self, mock_reset_ctx,
                                                 mock_get_ctx, input_tensors):
        A_instance_of_MultiStreamMetadata = MultiStreamMetadata(
            calculate_stream=MagicMock(),
            communicate_stream=MagicMock(),
            start_layer=0,
            end_layer=1,
            event_keys=[],
            multistream_config=None,
        )
        dummy_metadata = MagicMock(spec=A_instance_of_MultiStreamMetadata)
        dummy_metadata.ms_config.num_micro_batches = 4
        dummy_metadata.end_layer = 10

        mock_get_ctx.return_value = (
            3,  # layer_index
            dummy_metadata,
            "dummy_attn_metadata")

        dummy_metadata.merge_micro_batches.return_value = "merged_result"

        layer = MultiStreamPostTransformerLayer(
            multistream_metadata=dummy_metadata)
        output = layer.forward(input_tensors, wait_layer_index=7)

        dummy_metadata.try_wait_event.assert_called_once_with(
            7, 3, MSEventKey.FFN_AR_FINISH)
        mock_reset_ctx.assert_called_once()
        assert output == "merged_result"
