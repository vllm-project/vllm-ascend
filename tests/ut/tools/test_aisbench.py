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

from tools.aisbench import AisbenchRunner


def test_accuracy_result_is_collected_when_verification_is_disabled(monkeypatch):
    config = {
        "case_type": "accuracy",
        "dataset_path_local": "/tmp/gpqa",
        "model_path": "/tmp/model",
        "request_conf": "vllm_api_general_chat",
        "dataset_conf": "gpqa/gpqa_gen_0_shot_cot_chat_prompt",
        "max_out_len": 64,
        "batch_size": 1,
    }

    monkeypatch.setattr(AisbenchRunner, "_init_dataset_conf", lambda self: None)
    monkeypatch.setattr(AisbenchRunner, "_init_request_conf", lambda self: None)
    monkeypatch.setattr(AisbenchRunner, "_run_aisbench_task", lambda self: None)
    monkeypatch.setattr(AisbenchRunner, "_wait_for_task", lambda self: None)
    monkeypatch.setattr(AisbenchRunner, "_get_result_accuracy", lambda self: setattr(self, "result", 0.5))

    runner = AisbenchRunner("test-model", 8080, config, verify=False)

    assert runner.result == 0.5
