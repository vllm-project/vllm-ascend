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


import torch
import importlib
from tests.ut.base import TestBase
from unittest.mock import MagicMock, patch

from vllm.distributed.parallel_state import GroupCoordinator

from vllm_ascend.ops import sequence_parallel


class Test_Flash_Comm1(TestBase):

    @patch('vllm.distributed.tensor_model_parallel_all_gather')
    @patch('vllm.distributed.tensor_model_parallel_reduce_scatter')
    @patch('vllm.distributed.parallel_state._TP',
        new_callable=lambda: MagicMock(spec=GroupCoordinator))
    def test_test_flash_comm1(self, mock_TP,
                              mock_tensor_model_parallel_reduce_scatter,
                              mock_tensor_model_parallel_all_gather):
        with patch('vllm.distributed.get_tp_group', 
            return_value=MagicMock(world_size=4, rank_in_group=0)) as mock_get_tp_group:
            num_tokens = 9
            hidden_size = 128
            tp_size = 4
            hidden_states = torch.randn(num_tokens, hidden_size)

            mock_tp_group = mock_get_tp_group.return_value
            assert mock_tp_group.world_size == 4  # 手动断言属性存在
            assert mock_tp_group.rank_in_group == 0

            lengths_sum_unpadding = hidden_states.shape[0]
            lengths_sum_padding = ((lengths_sum_unpadding + tp_size - 1) // tp_size) * tp_size
            padding_flag = True
            pad_size = lengths_sum_padding - lengths_sum_unpadding
            importlib.reload(sequence_parallel)
            _metadata_for_padding = sequence_parallel.MetadataForPadding(lengths_sum_unpadding=lengths_sum_unpadding,
                                                        lengths_sum_padding=lengths_sum_padding,
                                                        padding_flag=padding_flag,
                                                        pad_size=pad_size,
                                                        not_dummy_and_is_prefill=True)

            mock_tensor_model_parallel_reduce_scatter.return_value = torch.randn(lengths_sum_padding // tp_size, hidden_size)
            mock_tensor_model_parallel_all_gather.return_value = torch.randn(lengths_sum_padding, hidden_size)

            hidden_states = _metadata_for_padding.padding_aligned_reduce_scatter(hidden_states)
            output = _metadata_for_padding.allgather_unpadding_aligned(hidden_states)

            self.assertEqual(output.shape, (num_tokens, hidden_size))