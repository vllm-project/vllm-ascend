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
# This file is a part of the vllm-ascend project.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport

from vllm_ascend._310p.ops.gdn_attn_builder_310 import (
    GDNAttentionMetadataBuilder310,
)


def _config(method: str | None, mode: CUDAGraphMode) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=(SimpleNamespace(method=method) if method is not None else None),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
    )


def test_gdn_reports_always_for_exact_310p_dflash_full() -> None:
    config = _config("dflash", CUDAGraphMode.FULL)
    with patch("vllm_ascend._310p.dflash_full.is_310p", return_value=True):
        support = GDNAttentionMetadataBuilder310.get_cudagraph_support(config, object())
    assert support is AttentionCGSupport.ALWAYS


@pytest.mark.parametrize(
    ("is_310p_platform", "method", "mode"),
    [
        (False, "dflash", CUDAGraphMode.FULL),
        (True, "mtp", CUDAGraphMode.FULL),
        (True, None, CUDAGraphMode.FULL),
        (True, "dflash", CUDAGraphMode.NONE),
        (True, "dflash", CUDAGraphMode.PIECEWISE),
        (True, "dflash", CUDAGraphMode.FULL_DECODE_ONLY),
    ],
)
def test_gdn_preserves_baseline_capability_outside_exact_scope(
    is_310p_platform: bool,
    method: str | None,
    mode: CUDAGraphMode,
) -> None:
    config = _config(method, mode)
    with patch(
        "vllm_ascend._310p.dflash_full.is_310p",
        return_value=is_310p_platform,
    ):
        support = GDNAttentionMetadataBuilder310.get_cudagraph_support(config, object())
    assert support is AttentionCGSupport.UNIFORM_BATCH
