#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import torch
from torch import nn

from vllm_ascend.ops.gdn import (
    _PACKED_CONV_WEIGHT_NAME,
    _get_packed_conv_weights,
)


def _make_layer(weight: torch.Tensor) -> nn.Module:
    """Build a minimal layer shell exposing ``conv1d.weight`` like the real one.

    The real ``AscendGatedDeltaNetAttention`` is patched onto
    ``QwenGatedDeltaNetAttention`` via method copy, so ``__init__`` does not run
    on the patched instance. Mirroring that, we build a bare ``nn.Module`` and
    attach ``conv1d`` directly, which is all ``_get_packed_conv_weights`` reads.
    """
    layer = nn.Module()
    layer.conv1d = nn.Module()
    layer.conv1d.weight = nn.Parameter(weight)
    return layer


def test_get_packed_conv_weights_packs_in_kernel_layout():
    # conv1d.weight is [D, 1, W]; the kernel wants the transposed [W, D] view.
    source = torch.arange(18 * 4, dtype=torch.float32).reshape(18, 1, 4)
    layer = _make_layer(source)

    packed = _get_packed_conv_weights(layer)

    assert packed.shape == (4, 18)
    assert packed.dtype == torch.float32
    assert packed.is_contiguous()
    torch.testing.assert_close(packed, source[:, 0, :].transpose(0, 1))


def test_get_packed_conv_weights_caches_after_first_call():
    source = torch.arange(18 * 4, dtype=torch.bfloat16).reshape(18, 1, 4)
    layer = _make_layer(source)

    first = _get_packed_conv_weights(layer)
    # The cached packed weight is stored as a plain attribute, not a parameter.
    assert hasattr(layer, _PACKED_CONV_WEIGHT_NAME)
    assert not isinstance(
        dict(layer.named_parameters()).get(_PACKED_CONV_WEIGHT_NAME, None),
        nn.Parameter,
    )

    second = _get_packed_conv_weights(layer)

    # Subsequent calls must reuse the cached tensor instead of repacking.
    assert second.data_ptr() == first.data_ptr()
    assert torch.equal(second, first)

    # Mutating the source weight after packing must not change the cached value:
    # the whole point of caching is to compute once at first use.
    with torch.no_grad():
        layer.conv1d.weight.add_(1.0)
    third = _get_packed_conv_weights(layer)
    assert third.data_ptr() == first.data_ptr()
    assert torch.equal(third, first)


def test_get_packed_conv_weights_meta_weight_returns_view_without_caching():
    # During meta-init / loading the weight lives on the meta device and cannot
    # be materialized, so packing is skipped and a non-cached view is returned.
    meta_weight = torch.empty(18, 1, 4, device="meta")
    layer = _make_layer(meta_weight)

    packed = _get_packed_conv_weights(layer)

    assert packed.is_meta
    assert packed.shape == (4, 18)
    # The meta path intentionally does not populate the cache attribute, so
    # callers keep getting fresh views until a real (non-meta) weight arrives.
    assert not hasattr(layer, _PACKED_CONV_WEIGHT_NAME)

    again = _get_packed_conv_weights(layer)
    assert again.shape == (4, 18)
    # meta tensors report data_ptr()==0, so compare identity instead: each call
    # returns a fresh (non-cached) view until a real weight is available.
    assert again is not packed
