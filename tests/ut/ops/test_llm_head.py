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
from unittest.mock import patch

import torch
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


def test_deepseek_v2_lmhead():

    class SimpleConfig:

        def __init__(self):
            self.vocab_size = 10000
            self.hidden_size = 128

    config = SimpleConfig()

    lmhead = ParallelLMHead(config.vocab_size, config.hidden_size)
    logits_processor = LogitsProcessor(config.vocab_size)

    mock_output = torch.randn(2, 4, config.hidden_size)
    mock_logits = torch.randn(2, 4, config.vocab_size)

    with patch.object(lmhead.quant_method, "apply", return_value=mock_logits):
        with patch.object(logits_processor,
                          "_gather_logits",
                          return_value=mock_logits):
            logits = logits_processor(lmhead, mock_output)
    assert logits.shape == (2, 4, config.vocab_size)
