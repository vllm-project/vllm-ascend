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

from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker


def _configure_model_sampler_for_spec_decode(self):
    (self.scorer_worker.model_runner.model.sampler.include_gpu_probs_tensor
     ) = True
    (self.scorer_worker.model_runner.model.sampler.
     should_modify_greedy_probs_inplace) = True
    self.proposer_worker.set_include_gpu_probs_tensor()
    self.proposer_worker.set_should_modify_greedy_probs_inplace()


SpecDecodeWorker._configure_model_sampler_for_spec_decode = _configure_model_sampler_for_spec_decode
