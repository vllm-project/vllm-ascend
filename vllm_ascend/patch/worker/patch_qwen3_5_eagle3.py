#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

"""Monkeypatch Qwen3.5 conditional-generation classes for Eagle3 support."""

from typing import TYPE_CHECKING

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)

if TYPE_CHECKING:
    import torch


def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
    self.language_model.set_aux_hidden_state_layers(tuple(int(x) for x in layers))


def _get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
    return self.language_model.get_eagle3_default_aux_hidden_state_layers()


def _get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
    return _get_eagle3_default_aux_hidden_state_layers(self)


for _cls in (Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration):
    # SupportsEagleBase protocol attributes.
    _cls.has_own_lm_head = False  # type: ignore[misc]
    _cls.has_own_embed_tokens = False  # type: ignore[misc]
    _cls.supports_eagle3 = True  # type: ignore[misc]

    # SupportsEagle3 protocol methods.
    _cls.set_aux_hidden_state_layers = _set_aux_hidden_state_layers  # type: ignore[attr-defined]
    _cls.get_eagle3_default_aux_hidden_state_layers = (  # type: ignore[attr-defined]
        _get_eagle3_default_aux_hidden_state_layers
    )
    _cls.get_eagle3_aux_hidden_state_layers = _get_eagle3_aux_hidden_state_layers  # type: ignore[attr-defined]

