# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from types import SimpleNamespace

import pytest

from vllm_ascend.patch.worker.patch_draft_quarot import get_rotation_path


def _make_vllm_config(model_path, quant_description):
    return SimpleNamespace(
        model_config=SimpleNamespace(model=str(model_path)),
        quant_config=SimpleNamespace(quant_description=quant_description),
    )


def test_get_rotation_path_uses_configured_relative_path(tmp_path):
    rotation_path = tmp_path / "rotations" / "global.safetensors"
    rotation_path.parent.mkdir()
    rotation_path.touch()
    config = _make_vllm_config(
        tmp_path,
        {
            "optional": {
                "quarot": {
                    "rotation_map": {
                        "global_rotation": "rotations/global.safetensors",
                    }
                }
            }
        },
    )

    assert get_rotation_path(config) == rotation_path


def test_get_rotation_path_falls_back_to_default_file(tmp_path):
    rotation_path = tmp_path / "optional" / "quarot.safetensors"
    rotation_path.parent.mkdir()
    rotation_path.touch()
    config = _make_vllm_config(tmp_path, {})

    assert get_rotation_path(config) == rotation_path


def test_get_rotation_path_rejects_missing_configured_file(tmp_path):
    config = _make_vllm_config(
        tmp_path,
        {
            "optional": {
                "quarot": {
                    "rotation_map": {
                        "global_rotation": "rotations/missing.safetensors",
                    }
                }
            }
        },
    )

    with pytest.raises(FileNotFoundError, match="Configured QuaRot rotation file does not exist"):
        get_rotation_path(config)


def test_get_rotation_path_returns_none_without_mapping_or_default(tmp_path):
    config = _make_vllm_config(tmp_path, {})

    assert get_rotation_path(config) is None


def test_get_rotation_path_rejects_quarot_metadata_without_rotation_file(tmp_path):
    config = _make_vllm_config(
        tmp_path,
        {"optional": {"quarot": {}}},
    )

    with pytest.raises(FileNotFoundError, match="QuaRot metadata is present"):
        get_rotation_path(config)


def test_get_rotation_path_treats_null_mapping_as_default(tmp_path):
    rotation_path = tmp_path / "optional" / "quarot.safetensors"
    rotation_path.parent.mkdir()
    rotation_path.touch()
    config = _make_vllm_config(
        tmp_path,
        {"optional": {"quarot": {"rotation_map": {"global_rotation": None}}}},
    )

    assert get_rotation_path(config) == rotation_path


def test_get_rotation_path_ignores_default_file_without_quant_config(tmp_path):
    rotation_path = tmp_path / "optional" / "quarot.safetensors"
    rotation_path.parent.mkdir()
    rotation_path.touch()
    config = SimpleNamespace(
        model_config=SimpleNamespace(model=str(tmp_path)),
        quant_config=None,
    )

    assert get_rotation_path(config) is None
