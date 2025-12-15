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
# This file is a part of the vllm-ascend project.
#

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor


class DeepSeekMoeAdaptor(VllmEplbAdaptor):

    def init_eplb_params(self):
        self.num_dense_layers = self.model.config.first_k_dense_replace
        self.global_expert_num = self.model.config.n_routed_experts
        self.num_moe_layers = self.model.config.num_hidden_layers - self.num_dense_layers

    def model_register(model, model_config):
        super().model_register(model)
        config = model_config.hf_config
        model.num_dense_layers = config.first_k_dense_replace
        model.num_moe_layers = config.num_hidden_layers - model.num_dense_layers
