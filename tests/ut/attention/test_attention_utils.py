#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.attention.utils import enabling_mlapo, filter_chunked_req_indices, get_or_register_attention_buffer
from vllm_ascend.device.hardware_profile import MLAPOEnablementPolicy


def test_get_or_register_attention_buffer() -> None:
    module_a = torch.nn.Module()
    module_b = torch.nn.Module()
    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "layer.a": module_a,
                "layer.b": module_b,
            }
        )
    )
    factory_call_count = 0

    def factory() -> torch.Tensor:
        nonlocal factory_call_count
        factory_call_count += 1
        return torch.tensor([1, 2, 3])

    buffer = get_or_register_attention_buffer(
        vllm_config,
        ["layer.a", "layer.b"],
        "_test_buffer",
        factory,
    )

    assert factory_call_count == 1
    assert module_a._buffers["_test_buffer"] is buffer
    assert module_b._buffers["_test_buffer"] is buffer
    assert "_test_buffer" not in module_a.state_dict()
    assert "_test_buffer" not in module_b.state_dict()


def test_filter_chunked_req_indices_empty_mask() -> None:
    indices = filter_chunked_req_indices(
        torch.tensor([2, 1, 3]),
        [False, False, False],
    )

    torch.testing.assert_close(indices, torch.empty(0, dtype=torch.long))


def test_filter_chunked_req_indices_mixed_mask() -> None:
    indices = filter_chunked_req_indices(
        torch.tensor([2, 1, 3]),
        [True, False, True],
    )

    torch.testing.assert_close(indices, torch.tensor([0, 1, 3, 4, 5]))


def test_enabling_mlapo_respects_producer_and_consumer_policy() -> None:
    cases = (
        (MLAPOEnablementPolicy.DECODE_KV_CONSUMER_ONLY, True, False, True),
        (MLAPOEnablementPolicy.DECODE_KV_CONSUMER_ONLY, False, True, False),
        (MLAPOEnablementPolicy.ANY_CONFIGURED_INSTANCE, True, False, True),
        (MLAPOEnablementPolicy.ANY_CONFIGURED_INSTANCE, False, True, True),
    )

    for policy, is_consumer, is_producer, expected in cases:
        vllm_config = SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                is_kv_consumer=is_consumer,
                is_kv_producer=is_producer,
            )
        )
        profile = SimpleNamespace(mlapo_enablement_policy=policy)
        with (
            patch(
                "vllm_ascend.attention.utils.get_ascend_config",
                return_value=SimpleNamespace(enable_mlapo=True),
            ),
            patch(
                "vllm_ascend.attention.utils.get_current_hardware_profile",
                return_value=profile,
            ),
        ):
            assert enabling_mlapo(vllm_config) is expected
