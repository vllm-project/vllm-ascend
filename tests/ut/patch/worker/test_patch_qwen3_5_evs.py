#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.patch.worker import patch_qwen3_5


def _vllm_config(*, pruning_rate: float = 0.75, deepstack_indexes=None):
    mm_config = SimpleNamespace(
        video_pruning_rate=pruning_rate,
        is_multimodal_pruning_enabled=MagicMock(return_value=pruning_rate > 0),
    )
    vision_config = SimpleNamespace(
        out_hidden_size=4096,
        deepstack_visual_indexes=[] if deepstack_indexes is None else deepstack_indexes,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            multimodal_config=mm_config,
            hf_config=SimpleNamespace(vision_config=vision_config),
        )
    )


def test_set_qwen3_5_evs_attrs_enables_pruning_without_deepstack():
    model = SimpleNamespace()
    config = _vllm_config(pruning_rate=0.75)
    tokenizer = object()

    with patch.object(patch_qwen3_5, "cached_tokenizer_from_config", return_value=tokenizer):
        patch_qwen3_5._set_qwen3_5_evs_attrs(model, config)

    assert model.is_multimodal_pruning_enabled is True
    assert model.video_pruning_rate == 0.75
    assert model._tokenizer is tokenizer
    assert model.use_deepstack is False
    assert model.deepstack_num_level == 0
    assert model.visual_dim == 4096
    assert model.multiscale_dim == 0


def test_set_qwen3_5_evs_attrs_accounts_for_deepstack_width():
    model = SimpleNamespace()
    config = _vllm_config(deepstack_indexes=[8, 16])

    with patch.object(patch_qwen3_5, "cached_tokenizer_from_config", return_value=object()):
        patch_qwen3_5._set_qwen3_5_evs_attrs(model, config)

    assert model.use_deepstack is True
    assert model.deepstack_num_level == 2
    assert model.multiscale_dim == 8192


def test_set_qwen3_5_evs_attrs_disables_pruning_without_multimodal_config():
    model = SimpleNamespace()
    config = MagicMock()
    config.model_config.multimodal_config = None
    config.model_config.hf_config.vision_config = None

    with patch.object(patch_qwen3_5, "cached_tokenizer_from_config", return_value=object()):
        patch_qwen3_5._set_qwen3_5_evs_attrs(model, config)

    assert model.is_multimodal_pruning_enabled is False
    assert model.video_pruning_rate == 0.0
    assert model.visual_dim == 0
    assert model.multiscale_dim == 0
