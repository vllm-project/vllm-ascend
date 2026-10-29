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
# This file is a part of the vllm-ascend project.
#

from types import SimpleNamespace
from unittest.mock import Mock, patch

import psutil
import pytest

from tests.e2e.conftest import VllmRunner


def test_runner_shutdown_ignores_already_exited_child():
    runner = VllmRunner.__new__(VllmRunner)
    shutdown = Mock(side_effect=psutil.NoSuchProcess(1234))
    runner.model = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown)))

    with (
        patch("tests.e2e.conftest.clear_ascend_config") as clear_config,
        patch("tests.e2e.conftest.cleanup_dist_env_and_memory") as cleanup,
        runner,
    ):
        pass

    shutdown.assert_called_once()
    clear_config.assert_called_once()
    cleanup.assert_called_once()


def test_runner_shutdown_preserves_other_failures():
    runner = VllmRunner.__new__(VllmRunner)
    runner.model = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=Mock())))

    with (
        patch("tests.e2e.conftest.clear_ascend_config"),
        patch("tests.e2e.conftest.cleanup_dist_env_and_memory"),
        pytest.raises(ValueError, match="generation failed"),
        runner,
    ):
        raise ValueError("generation failed")
