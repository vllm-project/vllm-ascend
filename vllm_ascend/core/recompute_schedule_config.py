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

from dataclasses import dataclass, fields
from typing import Type, Union

from vllm.config import SchedulerConfig, VllmConfig

@dataclass
class RecomputeSchedulerConfig(SchedulerConfig):
    scheduler_cls: Union[str, Type[object]] = (
        "vllm_ascend.core.recompute_scheduler.RecomputeScheduler")
    @classmethod
    def initialize_from_config(cls, vllm_config: VllmConfig):
        vllm_scheduler_config = vllm_config.scheduler_config
        scheduler_config = {
            field.name: getattr(vllm_scheduler_config, field.name)
            for field in fields(vllm_scheduler_config) if field.init
        }
        if vllm_scheduler_config.async_scheduling:
            scheduler_config["scheduler_cls"] = (
                "vllm_ascend.core.async_recompute_scheduler.AsyncRecomputeScheduler")
        else:
            scheduler_config["scheduler_cls"] = (
                "vllm_ascend.core.recompute_scheduler.RecomputeScheduler")
        scheduler_config["max_model_len"] = vllm_config.model_config.max_model_len
        scheduler_config["is_encoder_decoder"] = vllm_config.model_config.is_encoder_decoder
        return cls(**scheduler_config)
