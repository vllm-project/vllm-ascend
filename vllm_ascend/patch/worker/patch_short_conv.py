# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Keep upstream ShortConv bound to the worker's selected Ascend kernels."""

from vllm.model_executor.layers.mamba import short_conv
from vllm.model_executor.layers.mamba.ops import causal_conv1d


def bind_short_conv_ops():
    """Refresh from-imports after the Ascend convolution implementations load."""
    short_conv.causal_conv1d_fn = causal_conv1d.causal_conv1d_fn
    short_conv.causal_conv1d_update = causal_conv1d.causal_conv1d_update


bind_short_conv_ops()
