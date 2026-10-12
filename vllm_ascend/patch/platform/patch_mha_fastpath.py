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
"""Disable PyTorch's fused MHA "fast path" on Ascend.

Root cause
----------
``torch.nn.MultiheadAttention`` (and therefore
``torch.nn.TransformerEncoderLayer``) takes a fused fast path whenever
``torch.backends.mha.get_fastpath_enabled()`` is ``True``:

* ``aten::_native_multi_head_attention`` from ``MultiheadAttention.forward``;
* ``aten::_transformer_encoder_layer_fwd`` from
  ``TransformerEncoderLayer.forward``.

Neither op has an Ascend kernel. torch_npu therefore dispatches them to
``VariableFallbackKernel``: the activations are copied to the host, the op is
executed on CPU, and the result is copied back. This is a *silent* slow path --
no error, just a warning -- and it is catastrophic for small modules that are
called once per forward pass. Measured on a 2-layer, ``d=1024``,
``ffn=4096`` decision head (batch 4, 96 tokens, 910B4):

===========================  ============
fast path                    head latency
===========================  ============
enabled (CPU fallback)        711.2 ms
disabled (NPU native)           2.1 ms
===========================  ============

How
---
Flip the global flag before any model is built. Both modules then fall back to
their regular decomposed implementation
(``scaled_dot_product_attention`` / ``baddbmm``), which NPU implements
natively. Numerically the decomposed path is equivalent to the fused one up to
floating point reassociation, so this is a pure performance change.

Related PR (if no, explain why):
    Upstream PyTorch exposes no Ascend kernel for the fused MHA ops; this is an
    Ascend-only workaround, so it lives in vllm-ascend.

Future Plan:
    Remove this patch once torch_npu provides native
    ``_native_multi_head_attention`` / ``_transformer_encoder_layer_fwd``
    kernels, or once PyTorch gates the fast path on operator availability.
"""

import torch

if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
    torch.backends.mha.set_fastpath_enabled(False)
