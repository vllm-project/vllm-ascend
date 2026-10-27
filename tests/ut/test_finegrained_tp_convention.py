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
"""Fine-grained TP "disabled" convention: 0 and 1 both mean disabled.

A tensor-parallel size of 1 shards nothing, so it must not take the
fine-grained path. This is the same convention vLLM's ``ParallelConfig``
uses (``tensor_parallel_size=1`` means no sharding) and the one
``oproj_tensor_parallel_size`` already followed. These tests pin the
convention across every knob and every call site.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.utils import (
    embedding_tp_enable,
    lmhead_tp_enable,
    mlp_tp_enable,
    oproj_tp_enable,
)

# knob attribute name -> the helper that gates the fine-grained path.
ENABLE_HELPERS = {
    "oproj_tensor_parallel_size": oproj_tp_enable,
    "lmhead_tensor_parallel_size": lmhead_tp_enable,
    "embedding_tensor_parallel_size": embedding_tp_enable,
    "mlp_tensor_parallel_size": mlp_tp_enable,
}


def _config_with(**sizes: int) -> SimpleNamespace:
    """An AscendConfig stub carrying only the fine-grained TP sizes."""
    defaults = {name: 0 for name in ENABLE_HELPERS}
    defaults.update(sizes)
    return SimpleNamespace(finegrained_tp_config=SimpleNamespace(**defaults))


@pytest.mark.parametrize("knob", sorted(ENABLE_HELPERS))
@pytest.mark.parametrize("size, expected", [(0, False), (1, False), (2, True), (4, True)])
def test_tp_enable_uses_size_greater_than_one(knob, size, expected):
    """0 and 1 disable the knob; only a size above 1 shards anything."""
    helper = ENABLE_HELPERS[knob]
    with patch("vllm_ascend.utils.get_ascend_config", return_value=_config_with(**{knob: size})):
        assert helper() is expected


@pytest.mark.parametrize("knob", sorted(ENABLE_HELPERS))
def test_size_one_is_disabled_for_every_knob(knob):
    """The regression this guards: size 1 used to enable lmhead/embedding/mlp."""
    with patch("vllm_ascend.utils.get_ascend_config", return_value=_config_with(**{knob: 1})):
        assert ENABLE_HELPERS[knob]() is False


def test_knobs_are_independent():
    """Setting one knob must not enable another."""
    with patch("vllm_ascend.utils.get_ascend_config", return_value=_config_with(lmhead_tensor_parallel_size=2)):
        assert lmhead_tp_enable() is True
        assert embedding_tp_enable() is False
        assert oproj_tp_enable() is False
        assert mlp_tp_enable() is False
