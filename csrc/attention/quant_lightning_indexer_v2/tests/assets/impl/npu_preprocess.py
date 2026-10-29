#!/usr/bin/python
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Populate QuantLightningIndexer V2 metadata without TTK state coupling."""

import importlib.util
import logging
import sys
from pathlib import Path

import torch

OPERATOR = "quant_lightning_indexer_v2"
METADATA_INDEX = 12
QUANT_MODE_MXFP8 = 3
QUANT_MODE_MXFP4 = 5
QUANT_MODE_HIF4 = 6
ACLNN_PARAMETER_NAMES = (
    "query",
    "key",
    "weights",
    "query_dequant_scale",
    "key_dequant_scale",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "seqused_q",
    "seqused_k",
    "cmp_residual_k",
    "block_table",
    "output_idx_offset",
    "metadata",
    "topk",
    "quant_mode",
    "max_seqlen_q",
    "layout_q",
    "layout_k",
    "mask_mode",
    "cmp_ratio",
    "return_value",
    "sparse_indices_out",
    "sparse_values_out",
)


def load_metadata_protocol():
    name = "qli_v2_ttk_metadata_protocol"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("metadata_protocol.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def get_attribute(kwargs, name, default=None, aliases=()):
    for key in (name, f"pytest_{name}", *aliases):
        value = kwargs.get(key)
        if value is not None:
            return value
    return default


def get_values(kwargs, name, tensor):
    value = kwargs.get(f"{name}_values")
    if value is None:
        value = tensor
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    return [int(item) for item in value]


def max_sequence(prefix, used, fallback):
    if used:
        return max(int(value) for value in used)
    if prefix and len(prefix) > 1:
        return max(int(prefix[index + 1]) - int(prefix[index]) for index in range(len(prefix) - 1))
    return int(fallback)


def restore_mx_dtypes(query, key, query_scale, key_scale, quant_mode):
    """Support older uint8-storage CSVs; native TTK dtypes pass through."""
    quant_mode = int(quant_mode)
    if quant_mode == QUANT_MODE_MXFP8:
        qk_dtype = getattr(torch, "float8_e4m3fn", None)
    elif quant_mode == QUANT_MODE_MXFP4:
        qk_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    else:
        return
    scale_dtype = getattr(torch, "float8_e8m0fnu", None)
    if qk_dtype is None or scale_dtype is None:
        raise RuntimeError("current PyTorch does not provide the requested MX dtype")
    for name, tensor, dtype in (
        ("query", query, qk_dtype),
        ("key", key, qk_dtype),
        ("query_dequant_scale", query_scale, scale_dtype),
        ("key_dequant_scale", key_scale, scale_dtype),
    ):
        if not torch.is_tensor(tensor):
            continue
        if tensor.dtype == dtype:
            continue
        if tensor.dtype != torch.uint8:
            raise TypeError(f"QLI_V2 {name} must use {dtype} or uint8 storage, got {tensor.dtype}")
        tensor.data = tensor.data.view(dtype)


def build_metadata_arguments(query, key, topk, quant_mode, layout_q, layout_k, mask_mode, cmp_ratio, kwargs):
    q_shape = tuple(int(value) for value in query.shape)
    k_shape = tuple(int(value) for value in key.shape)
    num_heads_q = q_shape[2] if layout_q == "BSND" else q_shape[1]
    num_heads_k = k_shape[1] if layout_k == "TND" else k_shape[2]
    head_dim = q_shape[-1] * (2 if int(quant_mode) in (QUANT_MODE_MXFP4, QUANT_MODE_HIF4) else 1)
    cu_q = kwargs.get("cu_seqlens_q")
    cu_k = kwargs.get("cu_seqlens_k")
    seq_q = kwargs.get("seqused_q")
    seq_k = kwargs.get("seqused_k")
    cu_q_values = get_values(kwargs, "cu_seqlens_q", cu_q)
    cu_k_values = get_values(kwargs, "cu_seqlens_k", cu_k)
    seq_q_values = get_values(kwargs, "seqused_q", seq_q)
    seq_k_values = get_values(kwargs, "seqused_k", seq_k)

    batch_size = get_attribute(kwargs, "batch_size")
    if batch_size is None:
        if seq_q_values is not None:
            batch_size = len(seq_q_values)
        elif cu_q_values is not None:
            batch_size = len(cu_q_values) - 1
        elif layout_q == "BSND":
            batch_size = q_shape[0]
        else:
            batch_size = 0

    q_fallback = q_shape[1] if layout_q == "BSND" else q_shape[0]
    if layout_k == "BSND":
        k_fallback = k_shape[1]
    elif layout_k == "TND":
        k_fallback = k_shape[0]
    else:
        k_fallback = int(get_attribute(kwargs, "max_seqlen_k", k_shape[1]))

    return {
        "num_heads_q": int(get_attribute(kwargs, "num_heads_q", num_heads_q, ("pytest_q_head_num",))),
        "num_heads_k": int(get_attribute(kwargs, "num_heads_k", num_heads_k, ("pytest_k_head_num",))),
        "head_dim": int(get_attribute(kwargs, "head_dim", head_dim)),
        "topk": int(topk),
        "quant_mode": int(quant_mode),
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "seqused_q": seq_q,
        "seqused_k": seq_k,
        "cmp_residual_k": kwargs.get("cmp_residual_k"),
        "batch_size": int(batch_size),
        "max_seqlen_q": int(
            get_attribute(
                kwargs,
                "metadata_max_seqlen_q",
                max_sequence(cu_q_values, seq_q_values, q_fallback),
            )
        ),
        "max_seqlen_k": int(
            get_attribute(
                kwargs,
                "metadata_max_seqlen_k",
                max_sequence(cu_k_values, seq_k_values, k_fallback),
            )
        ),
        "layout_q": str(layout_q),
        "layout_k": str(layout_k),
        "mask_mode": int(mask_mode),
        "cmp_ratio": int(cmp_ratio),
    }


def move_to_device(value, target):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=target.device)
    return torch.as_tensor(value, device=target.device)


def run_metadata(arguments, metadata):
    return torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(
        int(arguments["num_heads_q"]),
        int(arguments["num_heads_k"]),
        int(arguments["head_dim"]),
        int(arguments["topk"]),
        int(arguments["quant_mode"]),
        cu_seqlens_q=move_to_device(arguments.get("cu_seqlens_q"), metadata),
        cu_seqlens_k=move_to_device(arguments.get("cu_seqlens_k"), metadata),
        seqused_q=move_to_device(arguments.get("seqused_q"), metadata),
        seqused_k=move_to_device(arguments.get("seqused_k"), metadata),
        cmp_residual_k=move_to_device(arguments.get("cmp_residual_k"), metadata),
        batch_size=int(arguments["batch_size"]),
        max_seqlen_q=int(arguments["max_seqlen_q"]),
        max_seqlen_k=int(arguments["max_seqlen_k"]),
        layout_q=str(arguments["layout_q"]),
        layout_k=str(arguments["layout_k"]),
        mask_mode=int(arguments["mask_mode"]),
        cmp_ratio=int(arguments["cmp_ratio"]),
    )


def run(
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    metadata=None,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    **kwargs,
):
    """Generate metadata once, or reuse a nonzero manual-data input."""
    del weights
    if metadata is None:
        raise ValueError("QuantLightningIndexer V2 npu_preprocess requires metadata")
    restore_mx_dtypes(query, key, query_dequant_scale, key_dequant_scale, quant_mode)
    arguments_kwargs = dict(kwargs)
    arguments_kwargs.update(
        {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "seqused_q": seqused_q,
            "seqused_k": seqused_k,
            "cmp_residual_k": cmp_residual_k,
        }
    )
    protocol = load_metadata_protocol()
    testcase_name = kwargs.get("testcase_name")
    force_metadata_refresh = bool(get_attribute(kwargs, "metadata_refresh", False))
    if protocol.metadata_is_materialized(metadata) and not force_metadata_refresh:
        logging.info("[%s] reuse nonzero QLI_V2 metadata input", testcase_name)
        return None
    arguments = protocol.load_metadata_inputs(OPERATOR, testcase_name)
    if arguments is not None:
        source = "manual-data sidecar"
    else:
        arguments = build_metadata_arguments(
            query,
            key,
            topk,
            quant_mode,
            layout_q,
            layout_k,
            mask_mode,
            cmp_ratio,
            arguments_kwargs,
        )
        source = "main API fallback (sidecar unavailable)"
    logging.info(
        "[%s] build QLI_V2 metadata from %s; forced=%s",
        testcase_name,
        source,
        force_metadata_refresh,
    )
    generated = run_metadata(arguments, metadata)
    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            f"QLI_V2 metadata shape mismatch: placeholder={tuple(metadata.shape)}, generated={tuple(generated.shape)}"
        )
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))
    rewritten = protocol.rewrite_metadata_input(OPERATOR, testcase_name, METADATA_INDEX, metadata)
    if rewritten is not None:
        logging.info("[%s] rewrote QLI_V2 metadata input: %s", testcase_name, rewritten)
    return None


def run_aclnn(*args, **kwargs):
    """Adapt the ACLNN main API order to the shared Torch metadata hook."""
    if len(args) != len(ACLNN_PARAMETER_NAMES):
        raise ValueError(
            f"QuantLightningIndexerV2 ACLNN hook expects {len(ACLNN_PARAMETER_NAMES)} arguments, got {len(args)}"
        )
    values = dict(zip(ACLNN_PARAMETER_NAMES, args))
    host_metadata = values["metadata"]
    metadata = move_to_device(host_metadata, torch.empty(0, device="npu"))
    values["metadata"] = metadata
    result = run(
        values.pop("query"),
        values.pop("key"),
        values.pop("weights"),
        values.pop("query_dequant_scale"),
        values.pop("key_dequant_scale"),
        values.pop("topk"),
        values.pop("quant_mode"),
        **values,
        **kwargs,
    )
    if torch.is_tensor(host_metadata):
        if host_metadata.device != metadata.device:
            host_metadata.copy_(metadata.to(dtype=host_metadata.dtype, device=host_metadata.device))
    else:
        host_metadata[...] = metadata.detach().cpu().numpy()
    return result
