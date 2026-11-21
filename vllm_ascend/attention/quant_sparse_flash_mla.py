# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch adapter for the HIF8 quantized sparse MLA operators.

``cann_ops_transformer.quant_sparse_flash_mla`` (aclnnQuantSparseFlashMla,
Ascend 950PR/950DT only) computes sparse MLA attention over per-tensor
HIF8-quantized Q and latent KV. ``quant_sparse_flash_mla_metadata``
(aclnnQuantSparseFlashMlaMetadata) must be invoked first to produce the
2048-int32 core-split plan the main operator requires.

HIF8 payloads are passed as ``torch.uint8`` tensors carrying raw HIFLOAT8
bit patterns; the operators identify them through ``quant_mode=1``.
"""

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module

import torch

# quant_mode=1: Q, K and V are all per-tensor quantized to hifloat8.
HIF8_QUANT_MODE = 1
# The quant operator's fixed metadata (core-split plan) size, in int32 words.
SMLA_QUANT_METADATA_SIZE = 2048


@lru_cache
def _get_quant_sparse_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load the QuantSparseFlashMla operators without importing vllm_ascend.ops."""
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return namespace.quant_sparse_flash_mla, namespace.quant_sparse_flash_mla_metadata
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "HIF8 C8 sparse MLA on Ascend 950PR/950DT requires QuantSparseFlashMla from a matching CANN toolkit."
        ) from exc


def quant_sparse_flash_mla_metadata(**kwargs):
    """Build the core-split plan for the HIF8 quantized sparse MLA operator."""
    _, metadata_op = _get_quant_sparse_flash_mla_ops()
    return metadata_op(**kwargs)


def quant_sparse_flash_mla(q: torch.Tensor, **kwargs):
    """Run sparse MLA attention over HIF8-quantized Q and latent KV."""
    attention_op, _ = _get_quant_sparse_flash_mla_ops()
    return attention_op(q, HIF8_QUANT_MODE, **kwargs)
