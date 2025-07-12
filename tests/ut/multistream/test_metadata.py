from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.multistream.base import MSEventKey
from vllm_ascend.multistream.metadata import (MultiStreamConfig,
                                              MultiStreamMetadata,
                                              MultiStreamStepMetadata,
                                              split_micro_batches_tensors)


class TestMetaData(TestBase):

    def setUp(self):
        self.test_tensors_list = [torch.randn(100, 1024) for i in range(3)]
        self.test_tensors = torch.randn(100, 1024)
        self.test_tensors_dict = {
            'query': torch.randn(100, 1024),
            'key': torch.randn(100, 1024),
            'value': torch.randn(100, 1024)
        }
        self.split_index = 50

        mock_stream = MagicMock(spec=torch.npu.Stream)
        event_keys = [MagicMock(spec=MSEventKey)]
        multistream_config = MagicMock(spec=MultiStreamConfig)

        self.metadata = MultiStreamMetadata(
            calculate_stream=mock_stream,
            communicate_stream=mock_stream,
            start_layer=1,
            end_layer=3,
            event_keys=event_keys,
            multistream_config=multistream_config)

    def test_split_micro_batches_tensors(self):
        test_tensors_list_res = split_micro_batches_tensors(
            self.test_tensors_list, self.split_index)
        test_tensors_res = split_micro_batches_tensors(self.test_tensors,
                                                       self.split_index)
        keys = ['query', 'key', 'value']
        test_tensors_dict_res = split_micro_batches_tensors(
            self.test_tensors_dict, self.split_index, keys)
        for i in range(3):
            self.assertEqual(len(test_tensors_list_res[i][0]),
                             self.split_index)

            self.assertEqual(
                len(test_tensors_list_res[i][0]) +
                len(test_tensors_list_res[i][1]), 100)

        self.assertEqual(len(test_tensors_res[0]), self.split_index)
        self.assertEqual(
            len(test_tensors_res[0]) + len(test_tensors_res[1]), 100)

        for key in keys:
            self.assertEqual(len(test_tensors_dict_res[0][key]),
                             self.split_index)
            self.assertEqual(
                len(test_tensors_dict_res[0][key]) +
                len(test_tensors_dict_res[1][key]), 100)

    def test_default_init_multistream_step_metadata(self):
        metadata = MultiStreamStepMetadata()
        self.assertIsNone(metadata.comm_stream)
        self.assertIsNone(metadata.before_comm_event)
        self.assertIsNone(metadata.after_comm_event)

    def test_custom_init_multistream_step_metadata(self):
        mockStream = MagicMock(spec=torch.npu.Stream)
        mockEvent1 = MagicMock(spec=torch.npu.Event)
        mockEvent2 = MagicMock(spec=torch.npu.Event)

        metadata = MultiStreamStepMetadata(mockStream, mockEvent1, mockEvent2)
        self.assertEqual(metadata.comm_stream, mockStream)
        self.assertEqual(metadata.before_comm_event, mockEvent1)
        self.assertEqual(metadata.after_comm_event, mockEvent2)

    def test_default_init_multistream_config(self):
        config = MultiStreamConfig()
        self.assertEqual(config.min_total_tokens_to_split, 256)
        self.assertEqual(config.min_prefill_tokens_to_split, 64)
        self.assertEqual(config.num_micro_batches, 2)
        self.assertEqual(config.imbalance_ratio, 0.1)

    def test_custom_init_multistream_config(self):
        config = MultiStreamConfig(512, 128, 1, 0.2)
        self.assertEqual(config.min_total_tokens_to_split, 512)
        self.assertEqual(config.min_prefill_tokens_to_split, 128)
        self.assertEqual(config.num_micro_batches, 1)
        self.assertEqual(config.imbalance_ratio, 0.2)

    def test_init_multistream_metadata(self):
        mock_stream = MagicMock(spec=torch.npu.Stream)

        event_keys = [MagicMock()]
        multistream_config = MagicMock(spec=MultiStreamConfig)

        metadata = MultiStreamMetadata(calculate_stream=mock_stream,
                                       communicate_stream=mock_stream,
                                       start_layer=1,
                                       end_layer=3,
                                       event_keys=event_keys,
                                       multistream_config=multistream_config)

        self.assertEqual(metadata.calculate_stream, mock_stream)
        self.assertEqual(metadata.communicate_stream, mock_stream)
        self.assertEqual(metadata.start_layer, 1)
        self.assertEqual(metadata.end_layer, 3)
        self.assertEqual(metadata.ms_config, multistream_config)
        self.assertTrue(metadata.causal_lm)

    def test_build_events(self):
        mock_stream = MagicMock(spec=torch.npu.Stream)
        mock_event = MagicMock(spec=torch.npu.Event)
        with patch('torch.npu.Event', return_value=mock_event):
            event_keys = [MagicMock(spec=MSEventKey)]
            multistream_config = MultiStreamConfig(
                num_micro_batches=2,
                min_total_tokens_to_split=256,
                min_prefill_tokens_to_split=64)

            metadata = MultiStreamMetadata(
                calculate_stream=mock_stream,
                communicate_stream=mock_stream,
                start_layer=1,
                end_layer=3,
                event_keys=event_keys,
                multistream_config=multistream_config)

            expected_events = {
                0: {
                    0: {
                        event_keys[0]: mock_event
                    },
                    1: {
                        event_keys[0]: mock_event
                    }
                },
                1: {
                    0: {
                        event_keys[0]: mock_event
                    },
                    1: {
                        event_keys[0]: mock_event
                    }
                },
                2: {
                    0: {
                        event_keys[0]: mock_event
                    },
                    1: {
                        event_keys[0]: mock_event
                    }
                }
            }
            self.assertEqual(metadata.ms_events, expected_events)

    def test_build_ms_split_config(self):
        mock_stream = MagicMock(spec=torch.npu.Stream)
        event_keys = [MagicMock(spec=MSEventKey)]
        multistream_config = MagicMock(spec=MultiStreamConfig)
        multistream_config.num_micro_batches = 2
        multistream_config.min_total_tokens_to_split = 256
        multistream_config.min_prefill_tokens_to_split = 64

        metadata = MultiStreamMetadata(calculate_stream=mock_stream,
                                       communicate_stream=mock_stream,
                                       start_layer=1,
                                       end_layer=3,
                                       event_keys=event_keys,
                                       multistream_config=multistream_config)

        self.assertIsNotNone(metadata.ms_split_config)
        self.assertEqual(metadata.ms_split_config.num_micro_batches,
                         multistream_config.num_micro_batches)
        self.assertEqual(metadata.ms_split_config.min_total_tokens_to_split,
                         multistream_config.min_total_tokens_to_split)
        self.assertEqual(metadata.ms_split_config.min_prefill_tokens_to_split,
                         multistream_config.min_prefill_tokens_to_split)

    def test_try_wait_event(self):
        mock_stream = MagicMock(spec=torch.npu.Stream)
        mock_event = MagicMock(spec=torch.npu.Event)
        event_keys = [MagicMock(spec=MSEventKey)]
        multistream_config = MagicMock(spec=MultiStreamConfig)
        with patch('torch.npu.Event', return_value=mock_event):
            metadata = MultiStreamMetadata(
                calculate_stream=mock_stream,
                communicate_stream=mock_stream,
                start_layer=1,
                end_layer=3,
                event_keys=event_keys,
                multistream_config=multistream_config)

            metadata.try_wait_event(layer_index=1,
                                    micro_batch_index=0,
                                    event_key=event_keys[0])
            mock_event.wait.assert_called_once()

    def test_try_record_event(self):
        mock_stream = MagicMock(spec=torch.npu.Stream)
        mock_event = MagicMock(spec=torch.npu.Event)
        event_keys = [MagicMock(spec=MSEventKey)]
        multistream_config = MagicMock(spec=MultiStreamConfig)
        with patch('torch.npu.Event', return_value=mock_event):
            metadata = MultiStreamMetadata(
                calculate_stream=mock_stream,
                communicate_stream=mock_stream,
                start_layer=1,
                end_layer=3,
                event_keys=event_keys,
                multistream_config=multistream_config)

            metadata.try_record_event(layer_index=1,
                                      micro_batch_index=0,
                                      event_key=event_keys[0])
            mock_event.record.assert_called_once()

    def test_merge_batches_none_input(self):
        input_tensors = None
        result = self.metadata.merge_micro_batches(input_tensors)
        self.assertIsNone(result)

    def test_merge_batches_single_tensor_input(self):
        input_tensors = [torch.tensor([1, 2, 3])]
        result = self.metadata.merge_micro_batches(input_tensors)
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result[0], torch.tensor([1, 2, 3])))

    def test_merge_batches_list_of_tensors_input(self):
        input_tensors = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = self.metadata.merge_micro_batches(input_tensors)
        self.assertEqual(len(result), 2)
        self.assertEqual(result, input_tensors)

    def test_merge_batches_nested_list_input(self):
        input_tensors = [[torch.tensor([1, 2]),
                          torch.tensor([3, 4])],
                         [torch.tensor([5, 6]),
                          torch.tensor([7, 8])]]
        result = self.metadata.merge_micro_batches(input_tensors)
        self.assertEqual(len(result), 2)
        self.assertTrue(torch.equal(result[0], torch.tensor([1, 2, 3, 4])))
        self.assertTrue(torch.equal(result[1], torch.tensor([5, 6, 7, 8])))
