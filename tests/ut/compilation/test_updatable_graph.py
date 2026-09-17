#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
import torch

from vllm_ascend.compilation.updatable_graph import ParamSource, SharedSource


class _Recorder:
    def __init__(self):
        self.providers = []

    def get(self, provider):
        self.providers.append(provider)
        return ({"block_table": provider},)


def test_shared_source_returns_precomputed_params():
    params = [{"block_table": torch.zeros(1, 4)}]

    assert SharedSource(params).get(object()) is params


def test_shared_source_delegates_to_param_source():
    recorder = _Recorder()
    assert isinstance(recorder, ParamSource)

    provider = object()

    assert SharedSource(recorder).get(provider) == ({"block_table": provider},)
    assert recorder.providers == [provider]
