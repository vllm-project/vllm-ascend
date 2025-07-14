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

from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata

from tests.ut.base import TestBase
from vllm_ascend.patch.worker.patch_common.patch_multi_step_worker import \
    sampler_output


class TestPatchedMultiStepWorkerSamplerOutput(TestBase):

    def setUp(self):
        self.mock_self = MagicMock()

        self.mock_self.device = torch.device("cpu")

        self.mock_self._raise_if_unsupported = MagicMock()
        self.mock_self._expand_execute_model_request = MagicMock()
        self.mock_self.execute_model = MagicMock()
        self.mock_self._maybe_update_previous_hidden_states = MagicMock()
        self.mock_self._append_new_tokens = MagicMock()
        self.mock_self._filter_model_output = MagicMock()

        self.execute_model_req = ExecuteModelRequest(
            seq_group_metadata_list=[MagicMock(spec=SequenceGroupMetadata)],
            num_steps=1,
            blocks_to_swap_in={},
            blocks_to_swap_out={},
            blocks_to_copy={},
            num_lookahead_slots=0)
        self.sample_len = 3
        self.seq_ids_with_bonus_token = {1, 2, 3}

        self.expanded_request = MagicMock(spec=ExecuteModelRequest)
        self.indices_of_seq_with_bonus_tokens = [0, 1, 2]
        self.mock_self._expand_execute_model_request.return_value = (
            self.expanded_request, self.indices_of_seq_with_bonus_tokens)

        self.filtered_output = [
            MagicMock(spec=SamplerOutput),
            MagicMock(spec=SamplerOutput),
            MagicMock(spec=SamplerOutput)
        ]
        self.mock_self._filter_model_output.return_value = self.filtered_output

    def test_sampler_output_patched(self):
        from vllm.spec_decode.multi_step_worker import MultiStepWorker

        wrapped_func = MultiStepWorker.sampler_output.__wrapped__
        self.assertIs(
            wrapped_func, sampler_output,
            "Wrapped function does not match the expected implementation")

    def test_gpu_multi_step_path(self):
        mock_model_runner = MagicMock()
        mock_model_runner.supports_gpu_multi_step.return_value = True

        self.mock_self.model_runner = mock_model_runner
        with patch(
                'vllm_ascend.patch.worker.patch_common.patch_multi_step_worker.isinstance'
        ) as mock_isinstance:
            mock_isinstance.return_value = True

            mock_outputs = [
                MagicMock(spec=SamplerOutput),
                MagicMock(spec=SamplerOutput),
                MagicMock(spec=SamplerOutput)
            ]
            self.mock_self.execute_model.return_value = mock_outputs

            result, need_transpose = sampler_output(
                self.mock_self, self.execute_model_req, self.sample_len,
                self.seq_ids_with_bonus_token)

        self.mock_self._raise_if_unsupported.assert_called_once_with(
            self.execute_model_req)
        self.mock_self._expand_execute_model_request.assert_called_once_with(
            self.execute_model_req, self.seq_ids_with_bonus_token)

        mock_model_runner.supports_gpu_multi_step.assert_called_once_with(
            self.expanded_request)
        self.assertEqual(self.expanded_request.num_steps, self.sample_len)
        mock_model_runner.set_indices_of_seq_with_bonus_tokens.assert_called_once_with(
            self.indices_of_seq_with_bonus_tokens)
        self.mock_self.execute_model.assert_called_once_with(
            execute_model_req=self.expanded_request)

        self.assertEqual(result, self.filtered_output)
        self.assertTrue(need_transpose)

        self.mock_self._maybe_update_previous_hidden_states.assert_not_called()
        self.mock_self._append_new_tokens.assert_not_called()

    def test_cpu_multi_step_path(self):
        mock_model_runner = MagicMock()
        mock_model_runner.supports_gpu_multi_step.return_value = False

        self.mock_self.model_runner = mock_model_runner
        self.mock_self.worker = MagicMock()

        mock_step_output = MagicMock(spec=SamplerOutput)
        self.mock_self.worker.execute_model.return_value = [[mock_step_output]]

        result, need_transpose = sampler_output(self.mock_self,
                                                self.execute_model_req,
                                                self.sample_len,
                                                self.seq_ids_with_bonus_token)

        self.assertEqual(self.mock_self.worker.execute_model.call_count,
                         self.sample_len)
        self.mock_self._append_new_tokens.assert_called()
        self.assertEqual(self.mock_self._append_new_tokens.call_count,
                         self.sample_len)

        self.mock_self._filter_model_output.assert_called_once()
        self.assertEqual(result, self.filtered_output)
        self.assertTrue(need_transpose)

    def test_cpu_path_with_hidden_states(self):
        self.expanded_request.previous_hidden_states = MagicMock()

        mock_model_runner = MagicMock()
        mock_model_runner.supports_gpu_multi_step.return_value = False
        self.mock_self.model_runner = mock_model_runner
        self.mock_self.worker = MagicMock()

        self.mock_self.worker.model_runner = MagicMock()
        self.mock_self.worker.model_runner.return_hidden_states = False

        mock_step_output = MagicMock(spec=SamplerOutput)
        self.mock_self.worker.execute_model.return_value = [[mock_step_output]]

        sampler_output(self.mock_self, self.execute_model_req, self.sample_len,
                       self.seq_ids_with_bonus_token)

        self.assertTrue(
            self.mock_self.worker.model_runner.return_hidden_states)
        self.mock_self._maybe_update_previous_hidden_states.assert_called()
