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

"""Case-format adapters for the SMLA family pytest data."""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import torch


@dataclass(frozen=True)
class CaseFormat:
    """Describes where an operator stores tensors and standard metadata fields."""

    name: str
    tensor_key: str
    required_top_keys: Tuple[str, ...]
    required_tensor_keys: Tuple[str, ...]
    required_metadata_keys: Tuple[str, ...]
    metadata_names: Mapping[str, str]


class CaseAdapter:
    """Normalize the schema differences between SMLA and its quantized variants."""

    SMLA = CaseFormat(
        name="smla",
        tensor_key="input",
        required_top_keys=(
            "B",
            "layout_q",
            "layout_kv",
            "params",
            "metadata_input",
            "input",
        ),
        required_tensor_keys=(
            "q",
            "ori_kv",
            "cu_seqlens_q",
            "seqused_ori_kv",
            "seqused_cmp_kv",
            "cmp_residual_kv",
            "sinks",
            "softmax_scale",
            "ori_mask_mode",
            "cmp_mask_mode",
            "ori_win_left",
            "ori_win_right",
            "layout_q",
            "layout_kv",
        ),
        required_metadata_keys=(
            "N1",
            "N2",
            "D",
            "B",
            "max_seqlen_q",
            "max_seqlen_ori_kv",
            "max_seqlen_cmp_kv",
            "layout_q",
            "layout_kv",
        ),
        metadata_names={"N1": "N1", "N2": "N2", "D": "D", "B": "B"},
    )

    MQSMLA = CaseFormat(
        name="mqsmla",
        tensor_key="op_input",
        required_top_keys=("params", "metadata_input", "op_input", "cpu_output"),
        required_tensor_keys=(
            "q",
            "ori_kv",
            "cu_seqlens_q",
            "seqused_ori_kv",
            "seqused_cmp_kv",
            "cmp_residual_kv",
            "sinks",
            "softmax_scale",
            "ori_mask_mode",
            "cmp_mask_mode",
            "ori_win_left",
            "ori_win_right",
            "layout_q",
            "layout_kv",
        ),
        required_metadata_keys=(
            "num_heads_q",
            "num_heads_kv",
            "head_dim",
            "batch_size",
            "max_seqlen_q",
            "max_seqlen_ori_kv",
            "max_seqlen_cmp_kv",
            "layout_q",
            "layout_kv",
        ),
        metadata_names={
            "N1": "num_heads_q",
            "N2": "num_heads_kv",
            "D": "head_dim",
            "B": "batch_size",
        },
    )

    QSMLA = CaseFormat(
        name="qsmla",
        tensor_key="op_input",
        required_top_keys=("params", "metadata_input", "op_input", "cpu_output"),
        required_tensor_keys=(
            "q",
            "ori_kv",
            "cmp_kv",
            "q_descale",
            "ori_kv_descale",
            "cmp_kv_descale",
            "cu_seqlens_q",
            "seqused_q",
            "cu_seqlens_ori_kv",
            "cu_seqlens_cmp_kv",
            "seqused_ori_kv",
            "seqused_cmp_kv",
            "cmp_residual_kv",
            "sinks",
            "ori_mask_mode",
            "cmp_mask_mode",
            "layout_q",
            "layout_kv",
        ),
        required_metadata_keys=(
            "num_heads_q",
            "num_heads_kv",
            "head_dim",
            "batch_size",
            "max_seqlen_q",
            "max_seqlen_ori_kv",
            "max_seqlen_cmp_kv",
            "layout_q",
            "layout_kv",
        ),
        metadata_names={
            "N1": "num_heads_q",
            "N2": "num_heads_kv",
            "D": "head_dim",
            "B": "batch_size",
        },
    )

    def __init__(self, input_data: Dict[str, Any]):
        self.input_data = input_data
        self.case_format = self.detect_format(input_data)
        self.normalize_query_aligned_fields()

    @classmethod
    def detect_format(cls, input_data: Dict[str, Any]) -> CaseFormat:
        if "input" in input_data:
            return cls.SMLA
        if "op_input" in input_data:
            op_input = input_data["op_input"]
            # QSMLA carries explicit dequantization scales; MQSMLA does not.
            if any(
                name in op_input
                for name in ("q_descale", "ori_kv_descale", "cmp_kv_descale")
            ):
                return cls.QSMLA
            return cls.MQSMLA
        raise ValueError("Cannot identify case format: expected 'input' or 'op_input'")

    @property
    def name(self) -> str:
        return self.case_format.name

    @property
    def tensor_key(self) -> str:
        return self.case_format.tensor_key

    @property
    def tensors(self) -> Dict[str, Any]:
        return self.input_data[self.tensor_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.input_data["metadata_input"]

    @property
    def params(self) -> Dict[str, Any]:
        return self.input_data["params"]

    def get_metadata(self, standard_name: str, default: Any = None) -> Any:
        actual_name = self.case_format.metadata_names.get(standard_name, standard_name)
        return self.metadata.get(actual_name, default)

    def set_metadata(self, standard_name: str, value: Any) -> None:
        actual_name = self.case_format.metadata_names.get(standard_name, standard_name)
        self.metadata[actual_name] = value

    def get_batch_size(self) -> int:
        if "B" in self.input_data:
            return int(self.input_data["B"])
        return int(self.get_metadata("B"))

    def set_batch_size(self, value: int) -> None:
        if "B" in self.input_data:
            self.input_data["B"] = value
        self.set_metadata("B", value)
        self.params["B"] = value
        if "B" in self.tensors:
            self.tensors["B"] = value

    def get_layout_q(self) -> str:
        layout_q = self.tensors["layout_q"]
        return layout_q[0] if isinstance(layout_q, (list, tuple)) else layout_q

    def get_layout_kv(self) -> str:
        return self.tensors["layout_kv"]

    def set_reporting_field(self, name: str, value: Any) -> None:
        if name in self.input_data:
            self.input_data[name] = value
        self.params[name] = value
        if name in self.tensors:
            self.tensors[name] = value

    def normalize_query_aligned_fields(self) -> None:
        """Expose SMLA top-k tensors stored by its native generator in metadata."""
        if self.name != "smla":
            return
        for name in ("ori_topk_length", "cmp_topk_length"):
            value = self.metadata.get(name)
            if name not in self.tensors and value is not None:
                self.tensors[name] = value

    def sync_query_aligned_fields(self) -> None:
        """Keep the native SMLA metadata location aligned after a transform."""
        if self.name != "smla":
            return
        for name in ("ori_topk_length", "cmp_topk_length"):
            if name in self.tensors:
                self.metadata[name] = self.tensors[name]

    def get_kv_lengths(self, prefix: str) -> List[int]:
        """Return logical per-batch KV lengths without changing API arguments."""
        seqused = self.tensors.get(f"seqused_{prefix}_kv")
        if seqused is not None:
            return [int(value) for value in seqused.to("cpu", torch.int64).tolist()]

        batch_size = self.get_batch_size()
        kv = self.tensors.get(f"{prefix}_kv")
        if kv is None:
            return [0] * batch_size
        sparse_indices = self.tensors.get(f"{prefix}_sparse_indices")
        private_lengths = self.params.get(f"seqused_{prefix}_kv")
        if sparse_indices is not None and private_lengths is not None:
            if isinstance(private_lengths, torch.Tensor):
                values = private_lengths.to("cpu", torch.int64).tolist()
            else:
                values = [int(value) for value in private_lengths]
            if len(values) != batch_size:
                raise ValueError(
                    f"private seqused_{prefix}_kv length {len(values)} != B={batch_size}"
                )
            if any(value < 0 for value in values):
                raise ValueError(
                    f"private seqused_{prefix}_kv contains negative values"
                )
            return values
        layout = self.get_layout_kv()
        if layout == "BSND":
            return [int(kv.shape[1])] * batch_size
        if layout == "TND":
            cu = self.tensors.get(f"cu_seqlens_{prefix}_kv")
            if cu is None:
                raise ValueError(f"TND {prefix}_kv requires cu_seqlens_{prefix}_kv")
            offsets = cu.to("cpu", torch.int64).tolist()
            return [
                offsets[index + 1] - offsets[index] for index in range(len(offsets) - 1)
            ]

        raise ValueError(f"PA_BBND {prefix}_kv requires logical KV lengths")
