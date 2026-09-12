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

"""NPU-independent case selection, naming, and result helpers for QLI_V2 tests."""

from pathlib import Path

import pandas as pd
import regex as re

PARAM_NAMES = (
    "batch_size",
    "q_seq",
    "k_seq",
    "q_t_size",
    "k_t_size",
    "q_head_num",
    "k_head_num",
    "head_dim",
    "block_size",
    "block_num",
    "qk_dtype",
    "weight_dtype",
    "dequant_dtype",
    "actual_seq_dtype",
    "cu_seq_q",
    "cu_seq_k",
    "act_seq_q",
    "act_seq_k",
    "cmp_residual_k",
    "max_seqlen_q",
    "quant_mode",
    "layout_query",
    "layout_key",
    "sparse_count",
    "sparse_mode",
    "query_datarange",
    "key_datarange",
    "weights_datarange",
    "q_scale_datarange",
    "k_scale_datarange",
    "cmp_ratio",
    "return_value",
    "output_idx_offset",
)


def ensure_comparison_passed(
    case_name,
    result,
    fulfill_percent,
    result_return_value="N/A",
    fulfill_percent_return_value=0,
):
    """Raise a serializable error when an accuracy comparison fails."""
    failures = []
    if result != "Pass":
        failures.append(f"index result={result}, fulfill_percent={fulfill_percent}")
    if result_return_value not in ("N/A", "Pass"):
        failures.append(f"value result={result_return_value}, fulfill_percent={fulfill_percent_return_value}")
    if failures:
        raise AssertionError(f"accuracy comparison failed for {case_name}: " + "; ".join(failures))


class QliV2CaseSelector:
    """Resolve an ordered subset of PT cases by explicit name or one-based index."""

    @staticmethod
    def natural_key(path):
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", Path(path).name)]

    @staticmethod
    def parse_indexes(expression, total):
        if not expression:
            return []
        indexes = []
        for token in str(expression).split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"invalid descending case index range: {token}")
                indexes.extend(range(start, end + 1))
            else:
                indexes.append(int(token))
        invalid = [index for index in indexes if index < 1 or index > total]
        if invalid:
            raise ValueError(f"case indexes out of range 1..{total}: {invalid}")
        return indexes

    @classmethod
    def resolve(cls, pt_dir, explicit_files="", case_names="", case_indexes=""):
        if explicit_files:
            candidates = [Path(item.strip()) for item in explicit_files.split(",") if item.strip()]
        else:
            directory = Path(pt_dir)
            if not directory.is_dir():
                raise ValueError(f"PT directory does not exist: {directory}")
            candidates = sorted(directory.glob("*.pt"), key=cls.natural_key)

        missing = [str(path) for path in candidates if not path.is_file()]
        if missing:
            raise ValueError(f"PT files do not exist: {missing}")
        if not candidates:
            raise ValueError(f"no PT cases found in: {pt_dir}")
        if case_names and case_indexes:
            raise ValueError("case names and case indexes cannot be specified together")

        if case_names:
            by_name = {path.stem: path for path in candidates}
            selected = []
            unknown = []
            for item in case_names.split(","):
                name = Path(item.strip()).stem
                if not name:
                    continue
                if name not in by_name:
                    unknown.append(name)
                else:
                    selected.append(by_name[name])
            if unknown:
                raise ValueError(f"unknown case names: {unknown}")
            candidates = selected
        elif case_indexes:
            indexes = cls.parse_indexes(case_indexes, len(candidates))
            candidates = [candidates[index - 1] for index in indexes]

        return [str(path) for path in candidates]


class QliV2ResultWriter:
    """Build stable case names and append rows using the batch result schema."""

    @staticmethod
    def case_name(params, explicit_name=None):
        if explicit_name:
            normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(explicit_name))
            normalized = normalized.strip("._-")
            if not normalized:
                raise ValueError("case name has no usable filename characters")
            return normalized

        values = list(params)
        readable = (
            f"QLI_B{values[0]}_S1{values[1]}_S2{values[2]}_"
            f"N1{values[5]}_N2{values[6]}_D{values[7]}_"
            f"{values[21]}_{values[22]}_{values[10]}_"
            f"QM{values[20]}_SM{values[24]}_CR{values[30]}_"
            f"K{values[23]}_RV{values[31]}"
        )
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", readable)

    @staticmethod
    def row(
        case_name,
        params,
        result,
        fulfill_percent,
        result_return_value="N/A",
        fulfill_percent_return_value=0,
    ):
        values = list(params)
        if len(values) != len(PARAM_NAMES):
            raise ValueError(f"QLI_V2 parameter count mismatch: got {len(values)}, expected {len(PARAM_NAMES)}")
        row = {"case_name": case_name}
        row.update(dict(zip(PARAM_NAMES, values)))
        row.update(
            {
                "result": result,
                "fulfill_percent": fulfill_percent,
                "result_return_value": result_return_value,
                "fulfill_percent_return_value": fulfill_percent_return_value,
            }
        )
        return row

    @staticmethod
    def append(path, row):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            frame = pd.read_excel(output)
            expected_columns = list(row.keys())
            legacy_columns = [name for name in expected_columns if name != "return_value"]
            if list(frame.columns) == legacy_columns:
                frame["return_value"] = None
                frame = frame[expected_columns]
            elif list(frame.columns) != expected_columns:
                raise ValueError(
                    "result columns do not match existing Excel: "
                    f"existing={list(frame.columns)}, current={list(row.keys())}"
                )
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        else:
            frame = pd.DataFrame([row])
        frame.to_excel(output, index=False)
