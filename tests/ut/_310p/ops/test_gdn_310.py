#
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
#

import torch

from vllm_ascend._310p.ops.fla.gdn_310 import (
    AscendGatedDeltaNetAttention310,
    AscendQwenGatedDeltaNetAttention310,
    _zero_padded_tokens,
)
from vllm_ascend._310p.ops.gdn_attn_builder_310 import (
    AscendGDNAttentionBackend310,
    AscendGDNAttentionMetadataBuilder310,
)
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder


def test_ascend_gdn_attention_310_uses_310p_backend():
    assert AscendGatedDeltaNetAttention310.get_attn_backend(object()) is AscendGDNAttentionBackend310
    assert AscendGDNAttentionBackend310.get_builder_cls() is AscendGDNAttentionMetadataBuilder310


def test_qwen_gdn_uses_oot_310p_layer():
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

    assert issubclass(AscendQwenGatedDeltaNetAttention310, QwenGatedDeltaNetAttention)
    assert AscendQwenGatedDeltaNetAttention310.forward is QwenGatedDeltaNetAttention.forward
    assert AscendQwenGatedDeltaNetAttention310._forward_core is AscendGatedDeltaNetAttention310._forward_core
    assert AscendQwenGatedDeltaNetAttention310.get_attn_backend is AscendGatedDeltaNetAttention310.get_attn_backend


def test_zero_padded_tokens_masks_only_padded_token_positions():
    tensor = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

    masked = _zero_padded_tokens(tensor, torch.tensor(2), token_dim=1)

    torch.testing.assert_close(masked[:, :2], tensor[:, :2])
    assert torch.count_nonzero(masked[:, 2:]) == 0


def test_builder310_reuses_common_graph_padding():
    assert AscendGDNAttentionMetadataBuilder310.build is AscendGDNAttentionMetadataBuilder.build
    assert "build" not in AscendGDNAttentionMetadataBuilder310.__dict__
