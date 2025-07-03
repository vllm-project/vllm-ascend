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

from vllm.config import SchedulerConfig

from tests.ut.base import TestBase
from vllm_ascend.core.schedule_config import AscendSchedulerConfig


class TestAscendSchedulerConfig(TestBase):

    def test_initialize_from_config(self):
        max_num_batched_tokens = 128
        vllm_scheduler_config = SchedulerConfig(
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_num_batched_tokens,
            chunked_prefill_enabled=True)

        conf = AscendSchedulerConfig.initialize_from_config(
            vllm_scheduler_config, ascend_scheduler_config=None)

        self.assertEqual(conf.enable_chunked_prefill, False)
        self.assertEqual(conf.policy, "fcfs")
        self.assertEqual(conf.num_scheduler_steps, 1)
        self.assertEqual(conf.scheduler_cls,
                         "vllm_ascend.core.scheduler.AscendScheduler")
        self.assertEqual(conf.max_num_encoder_input_tokens,
                         max_num_batched_tokens)
        self.assertEqual(conf.encoder_cache_size, max_num_batched_tokens)

    def test_initialize_from_config_errors(self):
        with self.assertRaisesRegex(NotImplementedError,
                                    "only supports LLM models"):
            AscendSchedulerConfig.initialize_from_config(
                SchedulerConfig(is_multimodal_model=True, ),
                ascend_scheduler_config=None)
        with self.assertRaisesRegex(NotImplementedError,
                                    "doesn't support send_delta_data"):
            AscendSchedulerConfig.initialize_from_config(
                SchedulerConfig(send_delta_data=True, ),
                ascend_scheduler_config=None)
        with self.assertRaisesRegex(NotImplementedError,
                                    "doesn't support scheduler_delay_factor"):
            AscendSchedulerConfig.initialize_from_config(
                SchedulerConfig(delay_factor=2, ),
                ascend_scheduler_config=None)
