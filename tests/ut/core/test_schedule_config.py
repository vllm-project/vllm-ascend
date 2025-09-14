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

from vllm.config import SchedulerConfig

from tests.ut.base import TestBase
from vllm_ascend.core.schedule_config import AscendSchedulerConfig


class TestAscendSchedulerConfig(TestBase):

    def setUp(self):
        self.basic_scheduler_config = SchedulerConfig(
            max_num_batched_tokens=8192,
            max_model_len=8192,
            is_multimodal_model=False,
            send_delta_data=False,
            scheduler_delay_factor=0,
        )

    def test_initialize_from_config_with_default(self):
        # No additional config given, check the default value here.
        ascend_config = AscendSchedulerConfig.initialize_from_config(
            self.basic_scheduler_config, {})
        self.assertEqual(ascend_config.enable_chunked_prefill, False)
        self.assertEqual(ascend_config.policy, "fcfs")
        self.assertEqual(ascend_config.scheduler_cls,
                         "vllm_ascend.core.scheduler.AscendScheduler")
        self.assertEqual(ascend_config.max_num_encoder_input_tokens, 8192)
        self.assertEqual(ascend_config.encoder_cache_size, 8192)

    def test_initialize_from_config_with_override(self):
        # test override
        ascend_config = AscendSchedulerConfig.initialize_from_config(
            self.basic_scheduler_config,
            AscendSchedulerConfig(
                enable_chunked_prefill=False,
                policy="fcfs",
                scheduler_cls="vllm_ascend.core.scheduler.AscendScheduler",
                max_num_batched_tokens=2048,
                max_model_len=2048,
            ),
        )
        self.assertEqual(ascend_config.enable_chunked_prefill, False)
        self.assertEqual(ascend_config.policy, "fcfs")
        self.assertEqual(ascend_config.scheduler_cls,
                         "vllm_ascend.core.scheduler.AscendScheduler")
        self.assertEqual(ascend_config.max_num_batched_tokens, 2048)
        self.assertEqual(ascend_config.encoder_cache_size, 2048)

    def test_not_implemented_policy(self):
        with self.assertRaises(NotImplementedError) as context:
            AscendSchedulerConfig.initialize_from_config(
                self.basic_scheduler_config,
                AscendSchedulerConfig(
                    policy="custom_policy",
                    max_num_batched_tokens=2048,
                    max_model_len=2048,
                ),
            )
        self.assertIn(
            "currently AscendScheduler only supports fcfs policy",
            str(context.exception),
        )

    def test_not_implemented_send_delta_data(self):
        with self.assertRaises(NotImplementedError) as context:
            AscendSchedulerConfig.initialize_from_config(
                self.basic_scheduler_config,
                AscendSchedulerConfig(
                    send_delta_data=True,
                    max_num_batched_tokens=2048,
                    max_model_len=2048,
                ),
            )
        self.assertIn(
            "currently AscendScheduler doesn't support send_delta_data",
            str(context.exception),
        )

    def test_not_implemented_delay_factor(self):
        with self.assertRaises(NotImplementedError) as context:
            AscendSchedulerConfig.initialize_from_config(
                self.basic_scheduler_config,
                AscendSchedulerConfig(
                    delay_factor=1,
                    max_num_batched_tokens=2048,
                    max_model_len=2048,
                ),
            )
        self.assertIn(
            "currently AscendScheduler doesn't support scheduler_delay_factor",
            str(context.exception),
        )

    def test_no_override(self):
        ascend_config = AscendSchedulerConfig.initialize_from_config(
            self.basic_scheduler_config, {})
        self.assertEqual(ascend_config.max_num_encoder_input_tokens, 8192)
        self.assertEqual(ascend_config.encoder_cache_size, 8192)

    def test_valid_config_with_multimodal(self):
        config = AscendSchedulerConfig.initialize_from_config(
            SchedulerConfig(is_multimodal_model=True), {})
        self.assertTrue(config.is_multimodal_model)

    def test_valid_config_with_chunked_prefill(self):
        ascend_config = AscendSchedulerConfig.initialize_from_config(
            self.basic_scheduler_config,
            AscendSchedulerConfig(
                enable_chunked_prefill=True,
                max_num_batched_tokens=2048,
                max_model_len=4096,
            ),
        )
        self.assertEqual(ascend_config.max_num_batched_tokens, 2048)
        self.assertEqual(ascend_config.max_model_len, 4096)
        self.assertTrue(ascend_config.enable_chunked_prefill)

    def test_invalid_config_without_chunked_prefill(self):
        with self.assertRaises(ValueError) as context:
            AscendSchedulerConfig.initialize_from_config(
                self.basic_scheduler_config,
                AscendSchedulerConfig(
                    enable_chunked_prefill=False,
                    max_num_batched_tokens=2048,
                    max_model_len=4096,
                ),
            )
        self.assertIn(
            "Ascend scheduler is enabled without chunked prefill feature",
            str(context.exception),
        )
        self.assertIn("max_num_batched_tokens (2048)", str(context.exception))
        self.assertIn("max_model_len (4096)", str(context.exception))

    def test_initialize_from_config_with_pd_transfer(self):
        ascend_config = AscendSchedulerConfig.initialize_from_config(
            self.basic_scheduler_config,
            AscendSchedulerConfig(
                enable_pd_transfer=True,
                decode_max_num_seqs=48,
                max_num_batched_tokens=4096,
                max_model_len=4096,
            ),
        )
        self.assertEqual(ascend_config.enable_pd_transfer, True)
        self.assertEqual(ascend_config.decode_max_num_seqs, 48)
