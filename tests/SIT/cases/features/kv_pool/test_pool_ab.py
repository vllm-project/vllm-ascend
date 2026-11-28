# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""YAML-driven KV pool A/B watch; deployment and parameters live in the case YAML."""

from tests.e2e.common.multi_node.external_dp.pool_ab import run_pool_ab_case
from tests.e2e.conftest import wait_until_npu_memory_free


@wait_until_npu_memory_free(target_free_percentage=0.8)
def test_pool_ab() -> None:
    run_pool_ab_case()
