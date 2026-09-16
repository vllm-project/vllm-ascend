#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Schema and invariant validation for SMLA-family consistency cases."""

from typing import Any, Dict, List, Sequence

import torch

from .adapter import CaseAdapter


class SchemaError(ValueError):
    """Raised when an input case is structurally inconsistent."""


class CaseSchema:
    """Encapsulates schema helpers instead of exposing module-private functions."""

    @staticmethod
    def _require_keys(
        data: Dict[str, Any], required: Sequence[str], context: str
    ) -> None:
        missing = [key for key in required if key not in data]
        if missing:
            raise SchemaError(f"{context}: missing keys {missing}")

    @classmethod
    def validate(cls, input_data: Dict[str, Any]) -> None:
        if "input" not in input_data and "op_input" not in input_data:
            expected = (
                "input" if "B" in input_data or "layout_q" in input_data else "op_input"
            )
            raise SchemaError(f"top-level: missing keys ['{expected}']")
        try:
            adapter = CaseAdapter(input_data)
        except ValueError as error:
            raise SchemaError(str(error)) from error

        case_format = adapter.case_format
        cls._require_keys(input_data, case_format.required_top_keys, "top-level")
        cls._require_keys(
            adapter.tensors, case_format.required_tensor_keys, adapter.tensor_key
        )
        cls._require_keys(
            adapter.metadata, case_format.required_metadata_keys, "metadata_input"
        )

        layout_q = adapter.get_layout_q()
        layout_kv = adapter.get_layout_kv()
        if layout_q not in ("BSND", "TND"):
            raise SchemaError(f"Unsupported layout_q: {layout_q}")
        if layout_kv not in ("BSND", "TND", "PA_BBND"):
            raise SchemaError(f"Unsupported layout_kv: {layout_kv}")
        if "layout_q" in input_data and input_data["layout_q"] != layout_q:
            raise SchemaError("top-level and tensor layout_q values do not match")
        if "layout_kv" in input_data and input_data["layout_kv"] != layout_kv:
            raise SchemaError("top-level and tensor layout_kv values do not match")

        batch_size = adapter.get_batch_size()
        if batch_size < 1:
            raise SchemaError(f"B must be >= 1, got {batch_size}")
        if input_data.get("cpu_output") is None:
            raise SchemaError("cpu_output is None")
        if adapter.tensors["sinks"] is None:
            raise SchemaError("sinks is None")

        if adapter.name == "qsmla":
            for name in ("q_descale", "ori_kv_descale", "cmp_kv_descale"):
                value = adapter.tensors.get(name)
                if value is not None and value.numel() != 1:
                    raise SchemaError(
                        f"{name} must be a scalar tensor for batch transforms, "
                        f"got shape {tuple(value.shape)}"
                    )
            for name in ("q_descale", "ori_kv_descale"):
                if adapter.tensors.get(name) is None:
                    raise SchemaError(f"QSMLA requires {name}")

        q = adapter.tensors["q"]
        if layout_q == "BSND" and q.dim() != 4:
            raise SchemaError(f"BSND Q must be rank 4, got rank {q.dim()}")
        if layout_q == "TND":
            if q.dim() != 3:
                raise SchemaError(f"TND Q must be rank 3, got rank {q.dim()}")
            if adapter.tensors.get("cu_seqlens_q") is None:
                raise SchemaError("TND Q requires cu_seqlens_q")

        if layout_kv == "TND":
            if (
                adapter.tensors.get("ori_kv") is not None
                and adapter.tensors.get("cu_seqlens_ori_kv") is None
            ):
                raise SchemaError("TND ori_kv requires cu_seqlens_ori_kv")
            if (
                adapter.tensors.get("cmp_kv") is not None
                and adapter.tensors.get("cu_seqlens_cmp_kv") is None
            ):
                raise SchemaError("TND cmp_kv requires cu_seqlens_cmp_kv")
        if layout_kv == "PA_BBND":
            if (
                adapter.tensors.get("ori_kv") is not None
                and adapter.tensors.get("seqused_ori_kv") is None
                and not cls._has_sparse_fields(adapter, "ori")
            ):
                raise SchemaError("PA_BBND ori_kv requires seqused_ori_kv")
            if (
                adapter.tensors.get("cmp_kv") is not None
                and adapter.tensors.get("seqused_cmp_kv") is None
                and not cls._has_sparse_fields(adapter, "cmp")
            ):
                raise SchemaError("PA_BBND cmp_kv requires seqused_cmp_kv")

        cls._validate_sparse_fields(adapter)

    @staticmethod
    def _has_sparse_fields(adapter: CaseAdapter, prefix: str) -> bool:
        """Return whether sparse indices can define a logical KV selection."""
        tensors = adapter.tensors
        return (
            tensors.get(f"{prefix}_sparse_indices") is not None
            and adapter.params.get(f"seqused_{prefix}_kv") is not None
        )

    @classmethod
    def _validate_sparse_fields(cls, adapter: CaseAdapter) -> None:
        """Validate sparse index/top-k shapes without requiring seqused vectors."""
        tensors = adapter.tensors
        layout_q = adapter.get_layout_q()
        batch_size = adapter.get_batch_size()
        num_heads_kv = int(adapter.get_metadata("N2"))
        q = tensors["q"]
        q_tokens = int(q.shape[1]) if layout_q == "BSND" else int(q.shape[0])
        for prefix in ("ori", "cmp"):
            indices = tensors.get(f"{prefix}_sparse_indices")
            lengths = tensors.get(f"{prefix}_topk_length")
            if indices is None and lengths is None:
                continue
            if indices is None:
                raise SchemaError(
                    f"{prefix}_topk_length requires {prefix}_sparse_indices"
                )
            expected_prefix = (
                (batch_size, q_tokens) if layout_q == "BSND" else (q_tokens,)
            )
            if tuple(indices.shape[: len(expected_prefix)]) != expected_prefix:
                raise SchemaError(
                    f"{prefix}_sparse_indices shape {tuple(indices.shape)} does not match "
                    f"Q prefix {expected_prefix}"
                )
            expected_index_rank = 4 if layout_q == "BSND" else 3
            head_axis = 2 if layout_q == "BSND" else 1
            if indices.dim() != expected_index_rank:
                raise SchemaError(
                    f"{prefix}_sparse_indices must be rank {expected_index_rank} for "
                    f"{layout_q}, got shape {tuple(indices.shape)}"
                )
            if int(indices.shape[head_axis]) != num_heads_kv:
                raise SchemaError(
                    f"{prefix}_sparse_indices KV-head axis {indices.shape[head_axis]} "
                    f"!= num_heads_kv {num_heads_kv}"
                )
            if lengths is not None and tuple(lengths.shape) != tuple(
                indices.shape[:-1]
            ):
                raise SchemaError(
                    f"{prefix}_topk_length shape {tuple(lengths.shape)} != sparse prefix "
                    f"{tuple(indices.shape[:-1])}"
                )
            index_width = int(indices.shape[-1])
            if (
                lengths is not None
                and lengths.numel()
                and bool((lengths < 0).any().item())
            ):
                raise SchemaError(f"{prefix}_topk_length contains negative values")
            max_length = (
                int(lengths.max().item())
                if lengths is not None and lengths.numel()
                else 0
            )
            if lengths is not None and max_length > index_width:
                raise SchemaError(
                    f"{prefix}_topk_length max {max_length} exceeds sparse index width {index_width}"
                )

    @staticmethod
    def get_q_lengths(
        input_data: Dict[str, Any], valid_only: bool = False
    ) -> List[int]:
        adapter = CaseAdapter(input_data)
        tensors = adapter.tensors
        layout_q = adapter.get_layout_q()
        batch_size = adapter.get_batch_size()
        q = tensors["q"]

        if layout_q == "BSND":
            physical = [int(q.shape[1])] * batch_size
        elif tensors.get("cu_seqlens_q") is not None:
            cu_list = tensors["cu_seqlens_q"].to("cpu", torch.int64).tolist()
            physical = [
                cu_list[index + 1] - cu_list[index] for index in range(len(cu_list) - 1)
            ]
        else:
            raise SchemaError("Cannot determine per-batch Q lengths")

        seqused_q = tensors.get("seqused_q")
        if valid_only and seqused_q is not None:
            valid = seqused_q.to("cpu", torch.int64).tolist()
            if len(valid) != batch_size:
                raise SchemaError(f"seqused_q length {len(valid)} != B={batch_size}")
            return [
                min(int(length), physical[index]) for index, length in enumerate(valid)
            ]
        return physical

    @classmethod
    def check_invariants(cls, input_data: Dict[str, Any]) -> None:
        adapter = CaseAdapter(input_data)
        tensors = adapter.tensors
        metadata = adapter.metadata
        batch_size = adapter.get_batch_size()
        layout_q = adapter.get_layout_q()
        layout_kv = adapter.get_layout_kv()

        for field_name in (
            "seqused_q",
            "seqused_ori_kv",
            "seqused_cmp_kv",
            "cmp_residual_kv",
        ):
            value = tensors.get(field_name)
            if value is not None and value.shape[0] != batch_size:
                raise SchemaError(
                    f"{field_name} length {value.shape[0]} != B={batch_size}"
                )

        q = tensors["q"]
        q_lengths = cls.get_q_lengths(input_data)
        if layout_q == "BSND":
            if q.shape[0] != batch_size:
                raise SchemaError(f"Q dim-0 {q.shape[0]} != B={batch_size}")
            cu_q = tensors.get("cu_seqlens_q")
            if cu_q is not None:
                cu_list = cu_q.to("cpu", torch.int64).tolist()
                expected = [index * q.shape[1] for index in range(batch_size + 1)]
                if cu_list != expected:
                    raise SchemaError(
                        f"BSND cu_seqlens_q {cu_list} != expected {expected}"
                    )
        else:
            cu_q = tensors.get("cu_seqlens_q")
            cu_list = cu_q.to("cpu", torch.int64).tolist()
            if len(cu_list) != batch_size + 1:
                raise SchemaError(
                    f"cu_seqlens_q length {len(cu_list)} != B+1={batch_size + 1}"
                )
            if cu_list[-1] != q.shape[0]:
                raise SchemaError(
                    f"cu_seqlens_q[-1]={cu_list[-1]} != Q dim-0 {q.shape[0]}"
                )

        seqused_q = tensors.get("seqused_q")
        if seqused_q is not None:
            valid_q_lengths = seqused_q.to("cpu", torch.int64).tolist()
            if any(
                length < 0 or length > physical
                for length, physical in zip(valid_q_lengths, q_lengths)
            ):
                raise SchemaError(
                    f"seqused_q values {valid_q_lengths} exceed physical Q lengths {q_lengths}"
                )

        ori_kv = tensors.get("ori_kv")
        cmp_kv = tensors.get("cmp_kv")
        seqused_ori_kv = tensors.get("seqused_ori_kv")
        if layout_kv == "BSND":
            if ori_kv is not None and ori_kv.shape[0] != batch_size:
                raise SchemaError(f"ori_kv dim-0 {ori_kv.shape[0]} != B={batch_size}")
            if seqused_ori_kv is not None and ori_kv is not None:
                max_s = int(seqused_ori_kv.max().item())
                if max_s > ori_kv.shape[1]:
                    raise SchemaError(
                        f"max seqused_ori_kv {max_s} > ori_kv dim-1 {ori_kv.shape[1]}"
                    )
        elif layout_kv == "TND":
            cls._check_tnd_kv(tensors, "ori_kv", "cu_seqlens_ori_kv", batch_size)
            cls._check_tnd_kv(tensors, "cmp_kv", "cu_seqlens_cmp_kv", batch_size)
        else:
            for table_name in ("ori_block_table", "cmp_block_table"):
                table = tensors.get(table_name)
                if table is not None and table.shape[0] != batch_size:
                    raise SchemaError(
                        f"{table_name} dim-0 {table.shape[0]} != B={batch_size}"
                    )

        cls._check_max_length(metadata, "max_seqlen_q", q_lengths)
        cls._check_max_length(
            metadata, "max_seqlen_ori_kv", adapter.get_kv_lengths("ori")
        )
        cls._check_max_length(
            metadata, "max_seqlen_cmp_kv", adapter.get_kv_lengths("cmp")
        )

        if layout_kv == "PA_BBND" and ori_kv is not None:
            from .layouts import validate_block_ids

            validate_block_ids(tensors.get("ori_block_table"), ori_kv.shape[0])
            if cmp_kv is not None:
                validate_block_ids(tensors.get("cmp_block_table"), cmp_kv.shape[0])

    @staticmethod
    def _check_tnd_kv(
        tensors: Dict[str, Any], tensor_name: str, cu_name: str, batch_size: int
    ) -> None:
        tensor = tensors.get(tensor_name)
        cu = tensors.get(cu_name)
        if tensor is None or cu is None:
            return
        cu_list = cu.to("cpu", torch.int64).tolist()
        if len(cu_list) != batch_size + 1:
            raise SchemaError(
                f"{cu_name} length {len(cu_list)} != B+1={batch_size + 1}"
            )
        if cu_list[-1] != tensor.shape[0]:
            raise SchemaError(
                f"{cu_name}[-1]={cu_list[-1]} != {tensor_name} dim-0 {tensor.shape[0]}"
            )

    @staticmethod
    def _check_max_length(metadata: Dict[str, Any], name: str, lengths: Any) -> None:
        configured = metadata.get(name)
        if configured is None or lengths is None:
            return
        if isinstance(lengths, torch.Tensor):
            actual = int(lengths.max().item()) if lengths.numel() else 0
        else:
            actual = max((int(value) for value in lengths), default=0)
        if int(configured) < actual:
            raise SchemaError(f"metadata {name} {configured} < actual max {actual}")

    @classmethod
    def build_origins(
        cls, input_data: Dict[str, Any], valid_only: bool = False
    ) -> List[Any]:
        from .model import TokenOrigin

        origins = []
        for batch, length in enumerate(
            cls.get_q_lengths(input_data, valid_only=valid_only)
        ):
            origins.extend(
                TokenOrigin(source_batch=batch, source_token=token)
                for token in range(length)
            )
        return origins


def validate_schema(input_data: Dict[str, Any]) -> None:
    CaseSchema.validate(input_data)


def get_q_lengths(input_data: Dict[str, Any], valid_only: bool = False) -> List[int]:
    return CaseSchema.get_q_lengths(input_data, valid_only=valid_only)


def get_valid_q_lengths(input_data: Dict[str, Any]) -> List[int]:
    return CaseSchema.get_q_lengths(input_data, valid_only=True)


def check_invariants(input_data: Dict[str, Any]) -> None:
    CaseSchema.check_invariants(input_data)


def build_baseline_origins(
    input_data: Dict[str, Any], valid_only: bool = False
) -> List[Any]:
    return CaseSchema.build_origins(input_data, valid_only=valid_only)
