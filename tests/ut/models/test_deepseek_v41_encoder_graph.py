# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.models.deepseek_v4.common.vision import build_packed_merge_metadata

from vllm_ascend.models.deepseek_v41.vision import DeepseekV41Aligner
from vllm_ascend.models.deepseek_v41.vl_model import AscendDeepseekV41ForCausalLM


def test_encoder_graph_protocol():
    assert supports_encoder_cudagraph(AscendDeepseekV41ForCausalLM)


@pytest.mark.parametrize("grids", [[[4, 6]], [[3, 5], [2, 4], [1, 1]]])
def test_packed_aligner_matches_per_image(grids):
    torch.manual_seed(0)
    config = SimpleNamespace(vision_downsample_ratio=2, vision_dim=8, hidden_size=12)
    aligner = DeepseekV41Aligner(config)
    images = [torch.randn(h * w, config.vision_dim) for h, w in grids]
    metadata = build_packed_merge_metadata(grids, 2, device=torch.device("cpu"), dtype=torch.float32)
    expected = torch.cat([aligner(x, h, w) for x, (h, w) in zip(images, grids)])
    actual = aligner.forward_packed(torch.cat(images), **metadata)
    torch.testing.assert_close(actual, expected)


def test_ascend_config_accepts_sdpa_marker_and_pads_boundaries():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    attention = SimpleNamespace(attn_backend=AttentionBackendEnum.TORCH_SDPA)
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=12),
        multimodal_config=None,
        vision=SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(attn=attention))]),
    )
    config = AscendDeepseekV41ForCausalLM.get_encoder_cudagraph_config(model)
    assert config.out_hidden_size == 12
    assert "merge_idx" in config.buffer_keys
    dst = torch.full((6,), -1, dtype=torch.int32)
    config.padding_logics["cu_seqlens"](dst, torch.tensor([0, 6, 10], dtype=torch.int32))
    assert dst.tolist() == [0, 6, 10, 10, 10, 10]
    config.padding_logics["cu_seqlens"](dst, torch.empty(0, dtype=torch.int32))
    assert dst.tolist() == [0] * 6
