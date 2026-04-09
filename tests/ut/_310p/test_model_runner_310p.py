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

from vllm_ascend._310p.model_runner_310p import NPUModelRunner310


def _build_runner(uses_mrope: bool) -> NPUModelRunner310:
    runner = NPUModelRunner310.__new__(NPUModelRunner310)
    runner.uses_mrope = uses_mrope
    return runner


@patch("vllm_ascend._310p.model_runner_310p.begin_mrope_forward_310")
@patch("vllm_ascend.worker.model_runner_v1.NPUModelRunner._dummy_run")
def test_dummy_run_calls_begin_mrope_when_enabled(mock_parent_dummy_run, mock_begin_mrope):
    runner = _build_runner(uses_mrope=True)
    expected = object()
    mock_parent_dummy_run.return_value = expected

    result = runner._dummy_run(num_tokens=1)

    mock_begin_mrope.assert_called_once()
    mock_parent_dummy_run.assert_called_once()
    assert result is expected


@patch("vllm_ascend._310p.model_runner_310p.begin_mrope_forward_310")
@patch("vllm_ascend.worker.model_runner_v1.NPUModelRunner._dummy_run")
def test_dummy_run_skips_begin_mrope_when_disabled(mock_parent_dummy_run, mock_begin_mrope):
    runner = _build_runner(uses_mrope=False)
    mock_parent_dummy_run.return_value = object()

    runner._dummy_run(num_tokens=1)

    mock_begin_mrope.assert_not_called()
    mock_parent_dummy_run.assert_called_once()


@patch("vllm_ascend._310p.model_runner_310p.begin_mrope_forward_310")
@patch("vllm_ascend.worker.model_runner_v1.NPUModelRunner.execute_model")
def test_execute_model_calls_begin_mrope_when_enabled(mock_parent_execute_model, mock_begin_mrope):
    runner = _build_runner(uses_mrope=True)
    expected = object()
    mock_parent_execute_model.return_value = expected

    result = runner.execute_model("arg", key="value")

    mock_begin_mrope.assert_called_once()
    mock_parent_execute_model.assert_called_once_with("arg", key="value")
    assert result is expected
