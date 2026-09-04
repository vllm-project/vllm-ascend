from unittest.mock import MagicMock

from torch import nn
from vllm.model_executor.models.interfaces import supports_eagle3

from vllm_ascend.models.deepseek_v4.vl_model import (
    AscendDeepseekV4ForConditionalGeneration,
)


def test_vision_wrapper_exposes_dspark_aux_hidden_state_interface():
    model = AscendDeepseekV4ForConditionalGeneration.__new__(AscendDeepseekV4ForConditionalGeneration)
    nn.Module.__init__(model)
    language_model = MagicMock()
    model.language_model = language_model

    assert supports_eagle3(model)

    model.set_aux_hidden_state_layers((41, 42, 43))
    language_model.set_aux_hidden_state_layers.assert_called_once_with((41, 42, 43))
