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

import torch.fx as fx
import torch.nn as nn

from tests.ut.base import TestBase
from vllm_ascend.patch.worker.patch_eager_adaptor import EagerAdaptorPatch


class MyNet(nn.Module):

    def __init__(self, input_size=5, output_size=2):
        super(MyNet, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x


class TestPatchEagerAdaptor(TestBase):

    def setUp(self):
        self.adaptor = EagerAdaptorPatch()
        self.model = MyNet()

    def test_EagerAdaptor_patched(self):
        from vllm.compilation.compiler_interface import EagerAdaptor

        self.assertIs(EagerAdaptor, EagerAdaptorPatch)

    @patch('vllm_ascend.patch.worker.patch_eager_adaptor.get_ascend_config')
    def test_return_fx_graph_when_enable_npugraph_ex_optimize_is_False(
            self, mock_get_ascend_config):
        # Setup mock for enable_npugraph_ex_optimize enabled
        mock_config = MagicMock()
        mock_config.enable_npugraph_ex_optimize = False
        mock_get_ascend_config.return_value = mock_config

        fx_graph = fx.symbolic_trace(self.model)

        graph, _ = self.adaptor.compile(fx_graph, [], {})

        self.assertTrue(isinstance(graph, fx.GraphModule))
