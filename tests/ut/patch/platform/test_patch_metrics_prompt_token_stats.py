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

from vllm.v1.metrics.stats import PromptTokenStats

import vllm_ascend.patch.platform.patch_metrics_prompt_token_stats  # noqa: F401
from tests.ut.base import TestBase


class TestPatchMetricsPromptTokenStats(TestBase):

    def test_clamp_external_tokens_to_non_negative_sources(self):
        stats = PromptTokenStats()

        stats.update_from_output(
            num_cached_tokens=1,
            num_external_computed_tokens=3,
            prompt_len=2,
        )

        self.assertGreaterEqual(stats.get_by_source("local_compute"), 0)
        self.assertGreaterEqual(stats.get_by_source("local_cache_hit"), 0)
        self.assertGreaterEqual(stats.get_by_source("external_kv_transfer"), 0)

    def test_clamp_cached_tokens_not_exceed_prompt_len(self):
        stats = PromptTokenStats()

        stats.update_from_output(
            num_cached_tokens=5,
            num_external_computed_tokens=0,
            prompt_len=2,
        )

        self.assertEqual(stats.total, 2)
        self.assertGreaterEqual(stats.get_by_source("local_compute"), 0)

    def test_negative_inputs_are_clamped_to_zero(self):
        stats = PromptTokenStats()

        stats.update_from_output(
            num_cached_tokens=-1,
            num_external_computed_tokens=-3,
            prompt_len=-2,
        )

        self.assertEqual(stats.total, 0)
        self.assertEqual(stats.get_by_source("local_compute"), 0)
        self.assertEqual(stats.get_by_source("local_cache_hit"), 0)
        self.assertEqual(stats.get_by_source("external_kv_transfer"), 0)

    def test_counters_monotonic_across_multiple_updates(self):
        stats = PromptTokenStats()

        stats.update_from_output(
            num_cached_tokens=1,
            num_external_computed_tokens=1,
            prompt_len=3,
        )

        total_after_first = stats.total
        local_compute_after_first = stats.get_by_source("local_compute")
        local_cache_hit_after_first = stats.get_by_source("local_cache_hit")
        external_transfer_after_first = stats.get_by_source("external_kv_transfer")

        stats.update_from_output(
            num_cached_tokens=-5,
            num_external_computed_tokens=-7,
            prompt_len=-9,
        )

        self.assertGreaterEqual(stats.total, total_after_first)
        self.assertGreaterEqual(stats.get_by_source("local_compute"), local_compute_after_first)
        self.assertGreaterEqual(stats.get_by_source("local_cache_hit"), local_cache_hit_after_first)
        self.assertGreaterEqual(
            stats.get_by_source("external_kv_transfer"),
            external_transfer_after_first,
        )
