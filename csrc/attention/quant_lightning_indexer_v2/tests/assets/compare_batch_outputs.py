#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compare LI_V2/QLI_V2 output-0 dumps across TTK batch cases."""

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_batch_protocol():
    name = "qli_v2_ttk_batch_consistency"
    path = Path(__file__).with_name("impl") / "batch_consistency.py"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class IndexerBatchDumpComparator:
    """Validate exact output indices after all enabled cases finish."""

    def __init__(
        self,
        case_csv,
        result_csv,
        dump_dir,
        report_path,
        excel_path,
        require_intra_case,
        require_cross_case,
    ):
        self.case_csv = Path(case_csv)
        self.result_csv = Path(result_csv)
        self.dump_dir = Path(dump_dir)
        self.report_path = Path(report_path)
        self.excel_path = Path(excel_path) if excel_path else None
        self.require_intra_case = require_intra_case
        self.require_cross_case = require_cross_case

    @staticmethod
    def read_rows(path):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    @staticmethod
    def parse_cell(row, name, default=None):
        value = row.get(name)
        if value is None or value == "":
            return default
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"{row.get('testcase_name', '<unknown>')}: invalid {name}: {error}") from error

    @classmethod
    def case_enabled(cls, row):
        value = row.get("is_enabled")
        if value is None or value.strip() == "":
            return True
        normalized = value.strip().title()
        try:
            return bool(ast.literal_eval(normalized))
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"{row.get('testcase_name', '<unknown>')}: invalid is_enabled") from error

    @staticmethod
    def vector(attributes, name, batch_size, default):
        value = attributes.get(f"{name}_values")
        if value is None:
            return [default] * batch_size
        value = [int(item) for item in value]
        if len(value) != batch_size:
            raise ValueError(f"{name}_values length does not equal B={batch_size}")
        return value

    @staticmethod
    def prefix_lengths(attributes, name, batch_size):
        value = attributes.get(f"{name}_values")
        if value is None:
            return None
        value = [int(item) for item in value]
        if len(value) != batch_size + 1 or value[0] != 0:
            raise ValueError(f"{name}_values must contain B + 1 prefix values")
        if any(right <= left for left, right in zip(value, value[1:])):
            raise ValueError(f"{name}_values must be strictly increasing")
        return [right - left for left, right in zip(value, value[1:])]

    def geometry(self, row):
        shapes = self.parse_cell(row, "tensor_view_shapes")
        dtypes = self.parse_cell(row, "tensor_dtypes")
        attributes = self.parse_cell(row, "attributes", {})
        if len(shapes) not in (11, 13):
            raise ValueError(f"{row['testcase_name']}: expected LI/QLI direct input slots")
        q_shape = tuple(shapes[0])
        k_shape = tuple(shapes[1])
        layout_q = attributes.get("layout_q", attributes.get("layout_query", "BSND"))
        layout_k = attributes.get("layout_k", attributes.get("layout_key", "BSND"))
        if layout_q == "BSND":
            batch_size, q_extent, q_heads, head_dim = q_shape
            q_prefix = None
            q_lengths = self.vector(attributes, "seqused_q", batch_size, q_extent)
            output_prefix = None
        elif layout_q == "TND":
            q_extent, q_heads, head_dim = q_shape
            q_values = attributes.get("cu_seqlens_q_values")
            if q_values is None or q_values[0] != 0 or q_values[-1] != q_extent:
                raise ValueError(f"{row['testcase_name']}: TND q prefix must span the q tensor")
            batch_size = len(q_values) - 1
            q_prefix = [int(item) for item in q_values]
            q_lengths = [right - left for left, right in zip(q_prefix, q_prefix[1:])]
            output_prefix = q_prefix
        else:
            raise ValueError(f"{row['testcase_name']}: unsupported layout_q={layout_q}")

        if layout_k == "BSND":
            if int(k_shape[0]) != batch_size:
                raise ValueError(f"{row['testcase_name']}: key B does not match q B")
            k_extent, key_heads = int(k_shape[1]), int(k_shape[2])
            k_lengths = self.vector(attributes, "seqused_k", batch_size, k_extent)
            block_size = None
        elif layout_k == "TND":
            key_heads = int(k_shape[1])
            k_lengths = self.prefix_lengths(attributes, "cu_seqlens_k", batch_size)
            if k_lengths is None or sum(k_lengths) != int(k_shape[0]):
                raise ValueError(f"{row['testcase_name']}: invalid TND k prefix")
            block_size = None
        elif layout_k == "PA_BBND":
            block_size, key_heads = int(k_shape[1]), int(k_shape[2])
            k_lengths = self.vector(attributes, "seqused_k", batch_size, 0)
        else:
            raise ValueError(f"{row['testcase_name']}: unsupported layout_k={layout_k}")

        topk = int(attributes.get("topk", attributes.get("sparse_count")))
        output_shape = (batch_size, q_extent, key_heads, topk) if layout_q == "BSND" else (q_extent, key_heads, topk)
        return {
            "attributes": attributes,
            "input_dtypes": tuple(dtypes),
            "batch_size": batch_size,
            "q_heads": int(q_heads),
            "key_heads": key_heads,
            "head_dim": int(head_dim),
            "q_lengths": q_lengths,
            "k_lengths": k_lengths,
            "residual": self.vector(attributes, "cmp_residual_k", batch_size, 0),
            "layout_q": layout_q,
            "layout_k": layout_k,
            "block_size": block_size,
            "output_prefix": output_prefix,
            "output_shape": output_shape,
            "topk": topk,
        }

    @staticmethod
    def output_selector(relation, geometry):
        axes, slices, _seed = relation
        batch_slice = slices[0]
        sequence_slice = slices[1] if axes == (0, 1) else None
        batch_start, batch_stop, _ = batch_slice
        if batch_stop > geometry["batch_size"]:
            raise ValueError("logical B slice exceeds output batch")
        if geometry["layout_q"] == "BSND":
            selector = [slice(*batch_slice)]
            if sequence_slice is not None:
                if sequence_slice[1] > geometry["q_lengths"][batch_start]:
                    raise ValueError("logical S slice exceeds BSND output")
                selector.append(slice(*sequence_slice))
            else:
                active_lengths = geometry["q_lengths"][batch_start:batch_stop]
                if (
                    not active_lengths
                    or len(set(active_lengths)) != 1
                    or active_lengths[0] <= 0
                    or active_lengths[0] > geometry["output_shape"][1]
                ):
                    raise ValueError("invalid effective q lengths for BSND output")
                selector.append(slice(0, active_lengths[0], 1))
        else:
            prefix = geometry["output_prefix"]
            if sequence_slice is None:
                start, stop = prefix[batch_start], prefix[batch_stop]
            else:
                start = prefix[batch_start] + sequence_slice[0]
                stop = prefix[batch_start] + sequence_slice[1]
                if stop > prefix[batch_start + 1]:
                    raise ValueError("logical S slice exceeds TND output")
            selector = [slice(start, stop, 1)]
        selector.extend([slice(None)] * (len(geometry["output_shape"]) - len(selector)))
        return tuple(selector)

    @staticmethod
    def context(row, relation, geometry):
        axes, slices, _seed = relation
        batch_start, batch_stop, _ = slices[0]
        sequence_slice = slices[1] if axes == (0, 1) else None
        attributes = geometry["attributes"]
        q_lengths = geometry["q_lengths"][batch_start:batch_stop]
        if sequence_slice is not None:
            q_lengths = [sequence_slice[1] - sequence_slice[0]]
        ignored = {
            "seqused_q_values",
            "seqused_k_values",
            "cu_seqlens_q_values",
            "cu_seqlens_k_values",
            "cmp_residual_k_values",
            "batch_deterministic_level",
        }
        scalar_attributes = {
            key: value
            for key, value in attributes.items()
            if key not in ignored and not isinstance(value, (list, tuple, dict))
        }
        return {
            "api_name": row.get("api_name"),
            "input_dtypes": geometry["input_dtypes"],
            "layout_q": geometry["layout_q"],
            "layout_k": geometry["layout_k"],
            "q_heads": geometry["q_heads"],
            "key_heads": geometry["key_heads"],
            "head_dim": geometry["head_dim"],
            "topk": geometry["topk"],
            "block_size": geometry["block_size"],
            "q_lengths": tuple(q_lengths),
            "k_lengths": tuple(geometry["k_lengths"][batch_start:batch_stop]),
            "residual": tuple(geometry["residual"][batch_start:batch_stop]),
            "attributes": scalar_attributes,
        }

    def build_samples(self, case_rows, result_rows):
        results = {row["testcase_name"]: row for row in result_rows if row.get("testcase_name")}
        protocol_class = load_batch_protocol().BatchRelationProtocol
        samples = []
        for row in case_rows:
            testcase_name = row.get("testcase_name")
            if not testcase_name:
                raise ValueError("case CSV contains an empty testcase_name")
            if row.get("batch_axis") in (None, ""):
                continue
            result = results.get(testcase_name)
            if result is None:
                raise ValueError(f"{testcase_name}: result CSV has no matching row")
            if result.get("precision_status") != "PASS":
                raise ValueError(f"{testcase_name}: precision_status is not PASS")
            eager_precision = result.get("eager_precision") or ""
            if "NO_OUTPU" in eager_precision:
                raise ValueError(
                    f"{testcase_name}: eager_precision reports no output; raw-byte batch comparison is invalid"
                )
            if self.require_intra_case and "batch_intra=PASS" not in eager_precision:
                raise ValueError(f"{testcase_name}: same-case batch check did not PASS")

            geometry = self.geometry(row)
            output_path = self.dump_dir / f"{testcase_name}_output_0.bin"
            if not output_path.is_file():
                raise ValueError(f"{testcase_name}: missing output dump {output_path}")
            output_bytes = output_path.read_bytes()
            expected_bytes = int(np.prod(geometry["output_shape"])) * 4
            if len(output_bytes) != expected_bytes:
                raise ValueError(f"{testcase_name}: output bytes={len(output_bytes)}, expected={expected_bytes}")
            output = np.frombuffer(output_bytes, dtype=np.int32).reshape(geometry["output_shape"])
            protocol = protocol_class("QLI_V2" if len(geometry["input_dtypes"]) == 13 else "LI_V2")
            relations = protocol.parse(
                self.parse_cell(row, "batch_axis"),
                self.parse_cell(row, "batch_slice_info"),
                self.parse_cell(row, "batch_seed"),
            )
            for relation in relations:
                selected = np.ascontiguousarray(output[self.output_selector(relation, geometry)])
                axes, slices, seed = relation
                relation_size = tuple(stop - start for start, stop, _step in slices)
                value = selected.view(np.uint8).tobytes()
                samples.append(
                    {
                        "testcase_name": testcase_name,
                        "relation": (axes, seed, relation_size),
                        "slice": {
                            "B": slices[0],
                            "S": slices[1] if len(slices) > 1 else None,
                        },
                        "shape": tuple(selected.shape),
                        "context": self.context(row, relation, geometry),
                        "sha256": hashlib.sha256(value).hexdigest(),
                        "value": value,
                    }
                )
        if not samples:
            raise ValueError("case CSV contains no enabled batch relation")
        return samples

    def compare_group(self, relation, samples):
        reference = samples[0]
        errors = []
        for sample in samples[1:]:
            if sample["shape"] != reference["shape"]:
                errors.append(f"{sample['testcase_name']}: output slice shape differs")
            if sample["context"] != reference["context"]:
                errors.append(f"{sample['testcase_name']}: relation context differs")
            if sample["value"] != reference["value"]:
                errors.append(f"{sample['testcase_name']}: raw output bytes differ")
        case_counts = Counter(sample["testcase_name"] for sample in samples)
        if self.require_intra_case:
            missing = sorted(name for name, count in case_counts.items() if count < 2)
            if missing:
                errors.append("fewer than two same-case samples: " + ", ".join(missing))
        if self.require_cross_case and len(case_counts) < 2:
            errors.append("relation occurs in fewer than two testcases")
        return {
            "relation": relation,
            "status": "PASS" if not errors else "FAIL",
            "case_count": len(case_counts),
            "sample_count": len(samples),
            "errors": errors,
            "samples": [
                {key: sample[key] for key in ("testcase_name", "slice", "shape", "sha256")} for sample in samples
            ],
        }

    def write_excel(self, report):
        from openpyxl import Workbook

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Relations"
        sheet.append(
            (
                "Relation",
                "Status",
                "Cases",
                "Samples",
                "Testcase",
                "Slice",
                "SHA-256",
                "Errors",
            )
        )
        for group in report["groups"]:
            errors = "\n".join(group["errors"])
            for sample in group["samples"]:
                sheet.append(
                    (
                        repr(group["relation"]),
                        group["status"],
                        group["case_count"],
                        group["sample_count"],
                        sample["testcase_name"],
                        repr(sample["slice"]),
                        sample["sha256"],
                        errors,
                    )
                )
        self.excel_path.parent.mkdir(parents=True, exist_ok=True)
        workbook.save(self.excel_path)

    def run(self):
        case_rows = self.read_rows(self.case_csv)
        result_rows = self.read_rows(self.result_csv)
        disabled = [row.get("testcase_name", "<unknown>") for row in case_rows if not self.case_enabled(row)]
        enabled = [row for row in case_rows if self.case_enabled(row)]
        samples = self.build_samples(enabled, result_rows)
        grouped = defaultdict(list)
        for sample in samples:
            grouped[sample["relation"]].append(sample)
        groups = [self.compare_group(relation, values) for relation, values in sorted(grouped.items())]
        passed = all(group["status"] == "PASS" for group in groups)
        report = {
            "status": "PASS" if passed else "FAIL",
            "case_csv": str(self.case_csv.resolve()),
            "result_csv": str(self.result_csv.resolve()),
            "dump_dir": str(self.dump_dir.resolve()),
            "group_count": len(groups),
            "sample_count": len(samples),
            "disabled_case_count": len(disabled),
            "disabled_cases": disabled,
            "groups": groups,
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if self.excel_path is not None:
            self.write_excel(report)
        return passed, report


def build_parser():
    parser = argparse.ArgumentParser(description="Compare LI_V2/QLI_V2 raw output-0 batch relations.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--excel")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--require-intra-case", action="store_true")
    parser.add_argument("--require-cross-case", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.output_index != 0:
        print("LI_V2/QLI_V2 batch comparison supports output index 0 only")
        return 1
    comparator = IndexerBatchDumpComparator(
        args.csv,
        args.result,
        args.dump_dir,
        args.report,
        args.excel,
        args.require_intra_case,
        args.require_cross_case,
    )
    try:
        passed, report = comparator.run()
    except (OSError, ValueError, csv.Error) as error:
        print(f"batch consistency comparison failed: {error}")
        return 1
    print(
        f"batch consistency {report['status']}: groups={report['group_count']}, "
        f"samples={report['sample_count']}, report={args.report}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
