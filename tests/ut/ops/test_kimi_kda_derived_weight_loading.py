# SPDX-License-Identifier: Apache-2.0
"""Strict loader and postprocessing contract for KDA's derived conv buffer."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention, _make_decode_conv1d_weight_loader

from vllm_ascend.ops.kimi_kda import _PACKED_CONV_WEIGHT_NAME, AscendKimiK3DeltaAttention


def _attention(monkeypatch, dtype):
    model_config = SimpleNamespace(dtype=dtype, quantization=None, word_embeddings_untied_by_checkpoint=False)

    def init_upstream(self, config, vllm_config, prefix):
        # Exercise the real Ascend constructor without TP/NPU initialization.
        nn.Module.__init__(self)
        self.model_config = model_config
        self.conv_size, self.local_projection_size = 4, 6
        self.conv1d = nn.Module()
        self.conv1d.weight = nn.Parameter(torch.full((18, 1, 4), torch.nan), requires_grad=False)
        self.conv1d.weight.weight_loader = _make_decode_conv1d_weight_loader([48] * 3, 8, 3, None)
        self.conv1d.quant_method = UnquantizedLinearMethod()
        self.A_log = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.o_norm = SimpleNamespace(eps=1e-5)

    monkeypatch.setattr(KimiK3DeltaAttention, "__init__", init_upstream)
    attention = AscendKimiK3DeltaAttention(
        SimpleNamespace(rms_norm_eps=1e-6), SimpleNamespace(quant_config=None, speculative_config=None)
    )
    return attention, model_config


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_real_loader_accepts_only_derived_buffer_and_postprocesses_loaded_qkv(monkeypatch, dtype):
    attention, model_config = _attention(monkeypatch, dtype)
    source = [torch.arange(48 * 4, dtype=torch.float32).reshape(48, 1, 4) / 16 + shard * 100 for shard in range(3)]
    for shard, weight in enumerate(source):
        attention.conv1d.weight.weight_loader(attention.conv1d.weight, weight, shard)
    expected = torch.cat([weight[18:24] for weight in source], dim=0)
    torch.testing.assert_close(attention.conv1d.weight, expected, rtol=0, atol=0)
    assert attention.conv1d.weight.dtype == torch.float32
    loader = DefaultModelLoader.__new__(DefaultModelLoader)
    loader.track_weights_loading(attention, {"conv1d.weight", "A_log"})
    with pytest.raises(ValueError, match="A_log"):
        loader.track_weights_loading(attention, {"conv1d.weight"})
    assert _PACKED_CONV_WEIGHT_NAME not in dict(attention.named_parameters())
    assert _PACKED_CONV_WEIGHT_NAME in dict(attention.named_buffers())
    assert _PACKED_CONV_WEIGHT_NAME not in attention.state_dict()
    pointer = attention.get_buffer(_PACKED_CONV_WEIGHT_NAME).data_ptr()

    # Use the production loader's quant-method traversal, not a direct pack.
    process_weights_after_loading(attention, model_config, torch.device("cpu"))
    packed = attention.get_buffer(_PACKED_CONV_WEIGHT_NAME)
    torch.testing.assert_close(packed, expected[:, 0].T.to(dtype), rtol=0, atol=0)
    assert packed.data_ptr() == pointer and packed.is_contiguous()
    assert packed.device == attention.conv1d.weight.device and packed.dtype == dtype
    assert not packed.requires_grad
    # A weight reload refreshes the derived data without moving graph storage.
    with torch.no_grad():
        attention.conv1d.weight.add_(2)
    process_weights_after_loading(attention, model_config, torch.device("cpu"))
    assert attention.get_buffer(_PACKED_CONV_WEIGHT_NAME).data_ptr() == pointer
    torch.testing.assert_close(packed, (expected[:, 0].T + 2).to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("shard", [0, 1, 2])
def test_original_qkv_checkpoint_loader_still_rejects_invalid_shape(monkeypatch, shard):
    attention, _ = _attention(monkeypatch, torch.bfloat16)
    with pytest.raises(RuntimeError):
        attention.conv1d.weight.weight_loader(attention.conv1d.weight, torch.empty(48, 1, 3), shard)


def test_derived_buffer_follows_module_conversion_and_materializes_after_meta(monkeypatch):
    attention, _ = _attention(monkeypatch, torch.bfloat16)
    attention.to(dtype=torch.float16)
    assert attention.get_buffer(_PACKED_CONV_WEIGHT_NAME).dtype == torch.float16
    # Device changes can occur when weights were constructed on meta or when
    # device_loading_context moves an offloaded source for processing.
    attention.ascend_conv1d_weight = torch.empty((4, 18), dtype=torch.bfloat16, device="meta")
    with torch.no_grad():
        attention.conv1d.weight.fill_(3)
    attention._pack_conv_weights()
    assert attention.get_buffer(_PACKED_CONV_WEIGHT_NAME).device.type == "cpu"
    assert attention.get_buffer(_PACKED_CONV_WEIGHT_NAME).dtype == torch.bfloat16
    assert attention.get_buffer(_PACKED_CONV_WEIGHT_NAME).eq(3).all()
    assert _PACKED_CONV_WEIGHT_NAME not in attention.state_dict()
