import unittest
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.engine.arg_utils import EngineArgs
from vllm.pooling_params import PoolingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata

from vllm_ascend.worker.pooling_model_runner import (
    ModelInputForNPUWithPoolingMetadata, NPUPoolingModelRunner)


class TestPoolingModelRunner(unittest.TestCase):
    """Unit tests for the NPUPoolingModelRunner class."""

    def _create_model_runner(self, model: str, *args,
                             **kwargs) -> NPUPoolingModelRunner:
        engine_args = EngineArgs(model, *args, **kwargs)
        engine_config = engine_args.create_engine_config()
        model_runner = NPUPoolingModelRunner(vllm_config=engine_config, )
        return model_runner

    def setUp(self):
        """Initialize test fixtures and common mocks"""
        self.attn_backend = "npu"

        model_runner = self._create_model_runner(
            "tests/ut/fake_weight",
            trust_remote_code=True,
            enable_chunked_prefill=False,
        )

        self.runner = model_runner
        self.runner.attn_backend = self.attn_backend
        model_runner.model = MagicMock()
        self.runner = model_runner
        # Sample test data
        self.sample_tensor_dict = {"tensor1": torch.randn(3, 4)}
        self.sample_seq_group = [MagicMock(spec=SequenceGroupMetadata)]
        self.sample_finished_ids = ["req1", "req2"]

    @patch(
        'vllm_ascend.worker.pooling_model_runner.ModelInputForNPUWithPoolingMetadata.from_broadcasted_tensor_dict'
    )
    def test_make_model_input_from_broadcasted_tensor_dict(
            self, mock_from_dict):
        """Test tensor dictionary conversion to model input"""
        # Setup mock return
        expected_output = MagicMock()
        mock_from_dict.return_value = expected_output

        # Execute
        result = self.runner.make_model_input_from_broadcasted_tensor_dict(
            self.sample_tensor_dict)

        # Verify
        mock_from_dict.assert_called_once_with(self.sample_tensor_dict,
                                               attn_backend=self.attn_backend)
        self.assertEqual(result, expected_output)

    @patch.object(NPUPoolingModelRunner, '_prepare_pooling')
    @patch.object(NPUPoolingModelRunner, '_prepare_model_input_tensors')
    def test_prepare_model_input_normal_case(self, mock_prepare_tensors,
                                             mock_prepare_pooling):
        """Test normal flow of model input preparation"""
        # Setup mocks
        mock_model_input = ModelInputForNPUWithPoolingMetadata(
            seq_lens=[1, 2, 3])
        mock_prepare_tensors.return_value = mock_model_input

        mock_pooling_metadata = MagicMock()
        mock_prepare_pooling.return_value = mock_pooling_metadata

        # Execute
        result = self.runner.prepare_model_input(
            seq_group_metadata_list=self.sample_seq_group,
            finished_requests_ids=self.sample_finished_ids)

        # Verify
        mock_prepare_tensors.assert_called_once_with(self.sample_seq_group,
                                                     self.sample_finished_ids)
        mock_prepare_pooling.assert_called_once_with(self.sample_seq_group,
                                                     mock_model_input.seq_lens)
        self.assertEqual(result.pooling_metadata, mock_pooling_metadata)

    def test_prepare_model_input_null_sequence_group(self):
        """Test assertion when seq_group_metadata_list is None"""
        with self.assertRaises(AssertionError):
            self.runner.prepare_model_input(
                seq_group_metadata_list=None,
                finished_requests_ids=self.sample_finished_ids)

    @patch.object(NPUPoolingModelRunner, '_prepare_model_input_tensors')
    def test_prepare_model_input_null_seq_lens(self, mock_prepare_tensors):
        """Test assertion when seq_lens is None in model input"""
        # Setup mock with None seq_lens
        mock_model_input = MagicMock()
        mock_model_input.seq_lens = None
        mock_prepare_tensors.return_value = mock_model_input

        with self.assertRaises(AssertionError):
            self.runner.prepare_model_input(
                seq_group_metadata_list=self.sample_seq_group,
                finished_requests_ids=self.sample_finished_ids)

    @patch.object(NPUPoolingModelRunner, '_prepare_pooling')
    @patch.object(NPUPoolingModelRunner, '_prepare_model_input_tensors')
    def test_prepare_model_input_with_virtual_engine(self,
                                                     mock_prepare_tensors,
                                                     mock_prepare_pooling):
        """Test virtual engine parameter is properly handled"""
        # Setup mocks
        mock_model_input = ModelInputForNPUWithPoolingMetadata(
            seq_lens=[1, 2, 3])
        mock_prepare_tensors.return_value = mock_model_input

        # Execute with virtual_engine parameter
        result = self.runner.prepare_model_input(
            seq_group_metadata_list=self.sample_seq_group,
            virtual_engine=1,
            finished_requests_ids=self.sample_finished_ids)

        # Verify virtual_engine doesn't affect the flow
        self.assertIsNotNone(result)

    @patch.object(NPUPoolingModelRunner, '_prepare_pooling')
    @patch.object(NPUPoolingModelRunner, '_prepare_model_input_tensors')
    def test_prepare_model_input_with_null_finished_ids(
            self, mock_prepare_tensors, mock_prepare_pooling):
        """Test case when finished_requests_ids is None"""
        # Setup mocks
        mock_model_input = ModelInputForNPUWithPoolingMetadata(
            seq_lens=[1, 2, 3])
        mock_prepare_tensors.return_value = mock_model_input

        # Execute with None finished_ids
        result = self.runner.prepare_model_input(
            seq_group_metadata_list=self.sample_seq_group,
            finished_requests_ids=None)

        # Verify
        mock_prepare_tensors.assert_called_once_with(self.sample_seq_group,
                                                     None)
        self.assertIsNotNone(result)

    @patch('vllm.model_executor.pooling_metadata.PoolingMetadata.__init__')
    def test_prepare_pooling_normal_case(self, mock_pooling_metadata):
        """Test normal case with multiple sequences in group"""
        # Setup test data
        mock_pooling_metadata.return_value = None
        seq_data = {
            1: MagicMock(spec=SequenceData),
            2: MagicMock(spec=SequenceData)
        }
        pooling_params = MagicMock(spec=PoolingParams)
        seq_group = MagicMock(spec=SequenceGroupMetadata)
        seq_group.seq_data = seq_data
        seq_group.pooling_params = pooling_params

        # Call the function
        self.runner._prepare_pooling([seq_group], [10, 20])

        # Verify results
        mock_pooling_metadata.assert_called_once_with(seq_groups=[
            ([1, 2], pooling_params)
        ],
                                                      seq_data=seq_data,
                                                      prompt_lens=[10, 20])

    @patch('vllm.model_executor.pooling_metadata.PoolingMetadata.__init__')
    def test_prepare_pooling_empty_group(self, mock_pooling_metadata):
        """Test case with empty sequence group"""
        # Setup empty group
        mock_pooling_metadata.return_value = None
        empty_seq_data: dict[int, SequenceData] = {}
        pooling_params = MagicMock(spec=PoolingParams)
        empty_group = MagicMock(spec=SequenceGroupMetadata)
        empty_group.seq_data = empty_seq_data
        empty_group.pooling_params = pooling_params

        # Call the function
        self.runner._prepare_pooling([empty_group], [])

        # Verify results
        mock_pooling_metadata.assert_called_once_with(seq_groups=[
            ([], pooling_params)
        ],
                                                      seq_data={},
                                                      prompt_lens=[])

    @patch('vllm.model_executor.pooling_metadata.PoolingMetadata.__init__')
    def test_prepare_pooling_single_sequence(self, mock_pooling_metadata):
        """Test case with single sequence in group"""
        # Setup single sequence
        mock_pooling_metadata.return_value = None
        single_seq_data = {3: MagicMock(spec=SequenceData)}
        pooling_params = MagicMock(spec=PoolingParams)
        single_group = MagicMock(spec=SequenceGroupMetadata)
        single_group.seq_data = single_seq_data
        single_group.pooling_params = pooling_params

        # Call the function
        self.runner._prepare_pooling([single_group], [5])

        # Verify results
        mock_pooling_metadata.assert_called_once_with(seq_groups=[
            ([3], pooling_params)
        ],
                                                      seq_data=single_seq_data,
                                                      prompt_lens=[5])

    @patch('vllm.model_executor.pooling_metadata.PoolingMetadata.__init__')
    def test_prepare_pooling_multiple_groups(self, mock_pooling_metadata):
        """Test case with multiple sequence groups"""
        # Setup multiple groups
        mock_pooling_metadata.return_value = None
        seq_data1 = {1: MagicMock(spec=SequenceData)}
        seq_data2 = {2: MagicMock(spec=SequenceData)}
        params1 = MagicMock(spec=PoolingParams)
        params2 = MagicMock(spec=PoolingParams)

        group1 = MagicMock(spec=SequenceGroupMetadata)
        group1.seq_data = seq_data1
        group1.pooling_params = params1

        group2 = MagicMock(spec=SequenceGroupMetadata)
        group2.seq_data = seq_data2
        group2.pooling_params = params2

        # Call the function
        self.runner._prepare_pooling([group1, group2], [10, 20])

        # Verify results
        mock_pooling_metadata.assert_called_once_with(seq_groups=[
            ([1], params1), ([2], params2)
        ],
                                                      seq_data={
                                                          **seq_data1,
                                                          **seq_data2
                                                      },
                                                      prompt_lens=[10, 20])

    @patch('vllm.model_executor.pooling_metadata.PoolingMetadata.__init__')
    def test_prepare_pooling_empty_input(self, mock_pooling_metadata):
        """Test case with empty input lists"""
        # Call the function with empty inputs
        mock_pooling_metadata.return_value = None
        self.runner._prepare_pooling([], [])

        # Verify results
        mock_pooling_metadata.assert_called_once_with(seq_groups=[],
                                                      seq_data={},
                                                      prompt_lens=[])

    @patch('vllm.forward_context.set_forward_context')
    @patch('vllm.distributed.parallel_state._PP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator,
                                          is_last_rank=True))
    @patch('torch.npu.Event')
    @patch.object(NPUPoolingModelRunner, 'set_active_loras')
    @patch.object(NPUPoolingModelRunner, 'set_active_prompt_adapters')
    def test_execute_model_normal_flow(self, mock_set_adapters, mock_set_loras,
                                       mock_event, mock_pp, mock_set_forward):
        """Test normal execution path with all dependencies mocked"""

        # Setup model input mock
        mock_input = MagicMock()
        mock_input.input_tokens = torch.tensor([1])
        mock_input.input_positions = torch.tensor([0])
        mock_input.multi_modal_kwargs = {}
        self.runner.is_driver_worker = True
        # Execute
        self.runner.execute_model(model_input=mock_input,
                                  kv_caches=[],
                                  num_steps=1)

        # Verify core calls
        self.runner.model.pooler.assert_called_once()

    @patch('vllm.forward_context.set_forward_context')
    def test_execute_model_invalid_steps(self, mock_set_forward):
        """Test ValueError when num_steps != 1"""
        with self.assertRaises(ValueError):
            self.runner.execute_model(model_input=MagicMock(),
                                      kv_caches=[],
                                      num_steps=2)
        mock_set_forward.assert_not_called()

    @patch('vllm.forward_context.set_forward_context')
    @patch('vllm.distributed.parallel_state._PP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator,
                                          is_last_rank=False))
    @patch('torch.npu.Event')
    def test_execute_model_perf_monitoring(self, mock_event, mock_pp,
                                           mock_set_forward):
        """Test performance monitoring with timing mocks"""
        # Setup mocks

        mock_event.return_value.elapsed_time.return_value = 15.0
        self.runner.observability_config = MagicMock(
            collect_model_forward_time=True)

        # Execute
        self.runner.execute_model(model_input=MagicMock(
            input_tokens=torch.tensor([1]),
            input_positions=torch.tensor([0]),
            multi_modal_kwargs={}),
                                  kv_caches=[],
                                  num_steps=1)

        # Verify timing calls
        self.assertEqual(mock_event.call_count, 2)

    @patch('vllm.forward_context.set_forward_context')
    @patch.object(NPUPoolingModelRunner, 'set_active_loras')
    @patch('vllm.distributed.parallel_state._PP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator,
                                          is_last_rank=False))
    def test_execute_model_lora_config(self, mock_pp, set_active_loras,
                                       mock_set_forward):
        """Test LoRA configuration handling"""
        # Setup

        self.runner.lora_config = True
        mock_input = MagicMock()
        mock_input.lora_requests = ["req1"]
        mock_input.lora_mapping = {"map": 1}

        # Execute
        self.runner.execute_model(model_input=mock_input,
                                  kv_caches=[],
                                  num_steps=1)

        # Verify LoRA call
        set_active_loras.assert_called_once_with(["req1"], {"map": 1})

    @patch('vllm.forward_context.set_forward_context')
    @patch('vllm.distributed.parallel_state._PP',
           new_callable=lambda: MagicMock(spec=GroupCoordinator,
                                          is_last_rank=False))
    def test_execute_model_not_last_rank(self, mock_pp, mock_set_forward):
        """Test behavior when not the last pipeline rank"""
        # Setup

        # Execute
        self.runner.execute_model(model_input=MagicMock(
            input_tokens=torch.tensor([1]),
            input_positions=torch.tensor([0]),
            multi_modal_kwargs={}),
                                  kv_caches=[],
                                  num_steps=1)

        # Verify pooler not called
        self.runner.model.pooler.assert_not_called()
