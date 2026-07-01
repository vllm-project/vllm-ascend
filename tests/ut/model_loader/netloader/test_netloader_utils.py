#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import os
import tempfile
from unittest.mock import patch

import pytest

from vllm_ascend.model_loader.netloader.utils import (find_free_port,
                                                      is_valid_path_prefix)


def test_find_free_port():
    port = find_free_port()
    assert isinstance(port, int)
    assert port > 0


def test_is_valid_path_prefix_empty():
    assert not is_valid_path_prefix('')


def test_is_valid_path_prefixIllegal_characters():
    assert not is_valid_path_prefix('test<>:"|?*')


def test_is_valid_path_prefixRelative_path():
    assert is_valid_path_prefix('test')


def test_is_valid_path_prefixAbsolute_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert is_valid_path_prefix(os.path.join(tmpdir, 'test'))


@patch('os.path.exists', return_value=False)
def test_is_valid_path_prefix_no_directory(mock_exists):
    assert not is_valid_path_prefix('/nonexistent_dir/test')


@patch('os.path.exists', return_value=True)
def test_is_valid_path_prefix_directory_exists(mock_exists):
    assert is_valid_path_prefix('/existing_dir/test')


if __name__ == "__main__":
    pytest.main()
