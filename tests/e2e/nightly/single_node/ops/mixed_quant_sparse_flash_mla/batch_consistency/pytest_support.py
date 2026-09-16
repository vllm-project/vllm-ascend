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

"""Pytest-facing adapter around the SMLA-family consistency runner."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pytest
import torch

from .adapter import CaseAdapter
from .config import (
    BATCH_SPLIT_KEY,
    MODE_BATCH_KEY,
    ORDER_KEY,
    SEED_KEY,
    SHAPE_CHANGE_KEY,
    TOKEN_SPLIT_KEY,
)
from .runner import (
    BatchConsistencyUnsupportedError,
    ConsistencyRunner,
)
from .schema import get_q_lengths
from .transform import (
    InvalidTransformError,
    groups_from_split_sizes,
    transform_mode1_reorder,
    transform_mode2_split,
    transform_mode3_token_partition,
    transform_mode4_shape_change,
)


DEFAULT_MODES = ("reorder", "split", "token-split", "shape-change")
SUPPORTED_MODES = DEFAULT_MODES + ("token-isolation", "independent")
RELATION_SCHEMA_VERSION = 1


def load_operator_module(
    module_name: str,
    module_path: Path,
    legacy_modules: Optional[Mapping[str, ModuleType]] = None,
):
    """Load an operator-local helper under a unique name to avoid pytest collisions."""
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        if existing_path != module_path.resolve():
            raise ImportError(
                f"module name {module_name!r} already maps to {existing_path}, "
                f"not {module_path.resolve()}"
            )
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load operator helper {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    saved_modules = {}
    original_path = list(sys.path)
    try:
        # Resolve ordinary helper imports from the operator that owns the file.
        sys.path[:0] = [str(module_path.parent), str(module_path.parent.parent)]
        for legacy_name, dependency in (legacy_modules or {}).items():
            saved_modules[legacy_name] = sys.modules.get(legacy_name)
            sys.modules[legacy_name] = dependency
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path[:] = original_path
        for legacy_name, dependency in saved_modules.items():
            if dependency is None:
                sys.modules.pop(legacy_name, None)
            else:
                sys.modules[legacy_name] = dependency
    return module


def parse_int_list(value: str, name: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated integer list") from error
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def parse_shape_change(value: str) -> Tuple[int, int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(
            "shape change must be common_tokens,derived_extra_tokens"
        ) from error
    if len(values) != 2 or values[0] <= 0 or values[1] < 0:
        raise ValueError(
            "shape change requires positive common_tokens and non-negative derived_extra_tokens"
        )
    return values[0], values[1]


def selected_modes(prefix: str) -> List[str]:
    """Read the opt-in modes once so ordinary and independent cases agree."""
    selected = os.environ.get(f"{prefix}_CONSISTENCY_MODES", ",".join(DEFAULT_MODES))
    modes = [value.strip() for value in selected.split(",") if value.strip()]
    unknown = sorted(set(modes) - set(SUPPORTED_MODES))
    if unknown:
        raise ValueError(f"unsupported batch-consistency modes: {unknown}")
    return modes


def collect_case_matrix(prefix: str, default_dir: str) -> List[Tuple[str, str]]:
    if os.environ.get(f"{prefix}_CONSISTENCY_TEST", "0") != "1":
        return []
    modes = [mode for mode in selected_modes(prefix) if mode != "independent"]
    if not modes:
        return []

    single_case = os.environ.get(f"{prefix}_CONSISTENCY_CASE", "").strip()
    if not single_case:
        single_case = os.environ.get(f"{prefix}_TESTCASE_PATH", "").strip()
    if not single_case and prefix == "SMLA":
        single_case = os.environ.get("QSAS_TESTCASE_PATH", "").strip()
    if single_case:
        if not Path(single_case).is_file():
            raise ValueError(
                f"CONFIG_ERROR: consistency case does not exist: {single_case}"
            )
        paths = [single_case]
    else:
        case_dir_value = os.environ.get(f"{prefix}_PT_DIR", "")
        if prefix == "SMLA" and not case_dir_value:
            case_dir_value = os.environ.get("SMLA_PT_LOAD_PATH", "")
        case_dir = Path(case_dir_value or default_dir)
        if not case_dir.is_dir():
            raise ValueError(
                f"CONFIG_ERROR: consistency PT directory does not exist: {case_dir}"
            )
        paths = [str(path) for path in sorted(case_dir.glob("*.pt"))]
        if not paths:
            raise ValueError(f"CONFIG_ERROR: no .pt cases found in {case_dir}")
    return [(path, mode) for path in paths for mode in modes]


def relation_manifest_path(prefix: str) -> Optional[Path]:
    """Return the explicitly configured independent-case relation manifest."""
    value = os.environ.get(f"{prefix}_CONSISTENCY_RELATION_MANIFEST", "").strip()
    return Path(value) if value else None


def read_relation_manifest(manifest_path: Path, operator: str) -> List[Dict[str, Any]]:
    """Read a strict, opt-in mapping; no testcase-name or shape inference is allowed."""
    if not manifest_path.is_file():
        raise ValueError(
            f"CONFIG_ERROR: relation manifest does not exist: {manifest_path}"
        )
    try:
        with manifest_path.open("r", encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"CONFIG_ERROR: cannot read relation manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("CONFIG_ERROR: relation manifest root must be an object")
    if manifest.get("schema_version") != RELATION_SCHEMA_VERSION:
        raise ValueError(
            "CONFIG_ERROR: relation manifest schema_version must be "
            f"{RELATION_SCHEMA_VERSION}"
        )
    if manifest.get("operator") != operator:
        raise ValueError(
            "CONFIG_ERROR: relation manifest operator "
            f"{manifest.get('operator')!r} does not match {operator!r}"
        )
    relations = manifest.get("relations")
    if not isinstance(relations, list) or not relations:
        raise ValueError(
            "CONFIG_ERROR: relation manifest relations must be a non-empty list"
        )

    relation_ids = set()
    parsed = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, dict):
            raise ValueError(f"CONFIG_ERROR: relations[{index}] must be an object")
        relation_id = relation.get("id")
        if not isinstance(relation_id, str) or not relation_id.strip():
            raise ValueError(
                f"CONFIG_ERROR: relations[{index}].id must be a non-empty string"
            )
        if relation_id in relation_ids:
            raise ValueError(f"CONFIG_ERROR: duplicate relation id {relation_id!r}")
        relation_ids.add(relation_id)

        baseline_pt = relation.get("baseline_pt")
        derived_pt = relation.get("derived_pt")
        if not isinstance(baseline_pt, str) or not isinstance(derived_pt, str):
            raise ValueError(
                f"CONFIG_ERROR: relation {relation_id!r} must provide baseline_pt and derived_pt"
            )
        baseline_path = (manifest_path.parent / baseline_pt).resolve()
        derived_path = (manifest_path.parent / derived_pt).resolve()
        if not baseline_path.is_file() or not derived_path.is_file():
            raise ValueError(
                "CONFIG_ERROR: relation "
                f"{relation_id!r} references missing PT files: {baseline_path}, {derived_path}"
            )

        mappings = relation.get("mappings")
        if not isinstance(mappings, list) or not mappings:
            raise ValueError(
                f"CONFIG_ERROR: relation {relation_id!r} must have non-empty mappings"
            )
        parsed_mappings = []
        baseline_coordinates = set()
        derived_coordinates = set()
        for mapping_index, mapping in enumerate(mappings):
            if not isinstance(mapping, dict):
                raise ValueError(
                    f"CONFIG_ERROR: relation {relation_id!r} mappings[{mapping_index}] must be an object"
                )
            baseline = mapping.get("baseline")
            derived = mapping.get("derived")
            if not (
                isinstance(baseline, list)
                and isinstance(derived, list)
                and len(baseline) == 2
                and len(derived) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in baseline + derived
                )
            ):
                raise ValueError(
                    "CONFIG_ERROR: relation "
                    f"{relation_id!r} mappings[{mapping_index}] needs integer [batch, token] pairs"
                )
            baseline_coordinate = tuple(baseline)
            derived_coordinate = tuple(derived)
            if baseline_coordinate in baseline_coordinates:
                raise ValueError(
                    f"CONFIG_ERROR: relation {relation_id!r} repeats baseline {baseline_coordinate}"
                )
            if derived_coordinate in derived_coordinates:
                raise ValueError(
                    f"CONFIG_ERROR: relation {relation_id!r} repeats derived {derived_coordinate}"
                )
            baseline_coordinates.add(baseline_coordinate)
            derived_coordinates.add(derived_coordinate)
            parsed_mappings.append((*baseline_coordinate, *derived_coordinate))
        parsed.append(
            {
                "id": relation_id,
                "baseline_pt": str(baseline_path),
                "derived_pt": str(derived_path),
                "mappings": parsed_mappings,
            }
        )
    return parsed


def collect_independent_relation_matrix(
    prefix: str, operator: str
) -> List[Dict[str, Any]]:
    """Collect only mappings explicitly named by the operator-owned manifest."""
    if os.environ.get(f"{prefix}_CONSISTENCY_TEST", "0") != "1":
        return []
    if "independent" not in selected_modes(prefix):
        return []
    manifest_path = relation_manifest_path(prefix)
    if manifest_path is None:
        raise ValueError(
            f"CONFIG_ERROR: {prefix}_CONSISTENCY_MODES=independent requires "
            f"{prefix}_CONSISTENCY_RELATION_MANIFEST"
        )
    return read_relation_manifest(manifest_path, operator)


def balanced_split(total: int) -> List[int]:
    if total < 2:
        return []
    first = total // 2
    return [first, total - first]


def run_selected_mode(
    runner: ConsistencyRunner,
    data: Dict[str, Any],
    mode: str,
    prefix: str,
) -> Dict[str, Any]:
    seed = int(os.environ.get(f"{prefix}_CONSISTENCY_SEED", "20260716"))
    if mode == "reorder":
        if CaseAdapter(data).get_batch_size() < 2:
            pytest.skip("NOT_APPLICABLE: reorder requires at least two batches")
        return runner.run_mode1(data, None, seed)

    if mode == "split":
        batch_size = CaseAdapter(data).get_batch_size()
        configured = os.environ.get(f"{prefix}_CONSISTENCY_BATCH_SPLIT", "").strip()
        sizes = (
            parse_int_list(configured, "batch split")
            if configured
            else balanced_split(batch_size)
        )
        if not sizes:
            pytest.skip("NOT_APPLICABLE: split requires at least two batches")
        return runner.run_mode2(data, groups_from_split_sizes(batch_size, sizes), seed)

    batch_id = int(os.environ.get(f"{prefix}_CONSISTENCY_MODE_BATCH", "0"))
    valid_lengths = get_q_lengths(data, valid_only=True)
    if batch_id < 0 or batch_id >= len(valid_lengths):
        raise ValueError(f"mode batch {batch_id} is outside [0, {len(valid_lengths)})")
    token_count = valid_lengths[batch_id]
    if mode == "token-split":
        configured = os.environ.get(f"{prefix}_CONSISTENCY_TOKEN_SPLIT", "").strip()
        sizes = (
            parse_int_list(configured, "token split")
            if configured
            else balanced_split(token_count)
        )
        if not sizes:
            pytest.skip(
                "NOT_APPLICABLE: token-split requires at least two valid Q tokens"
            )
        return runner.run_mode3(data, batch_id, sizes)

    if mode == "token-isolation":
        isolation_batch = int(
            os.environ.get(f"{prefix}_CONSISTENCY_ISOLATION_BATCH", str(batch_id))
        )
        isolation_token = int(
            os.environ.get(f"{prefix}_CONSISTENCY_ISOLATION_TOKEN", "0")
        )
        return runner.run_mode5(data, isolation_batch, isolation_token, seed)

    configured = os.environ.get(f"{prefix}_CONSISTENCY_SHAPE_CHANGE", "").strip()
    if configured:
        common_tokens, derived_extra = parse_shape_change(configured)
    else:
        common_tokens = max(1, token_count - 1)
        derived_extra = token_count - common_tokens + 1
    return runner.run_mode4(data, batch_id, common_tokens, derived_extra)


def run_configured_consistency(
    data: Dict[str, Any],
    config: Mapping[str, Any],
    executor: Callable[[Dict[str, Any]], Any],
    precision_compare: Callable[[Any, Any], Any],
    set_device: Callable[[], None],
) -> Dict[str, Any]:
    """Run all persisted single-PT relations with one level-3 baseline call."""
    cases = []
    overrides = []
    skipped = {}
    seed = int(config[SEED_KEY])

    order = config.get(ORDER_KEY)
    if order is None:
        skipped["reorder"] = "NOT_APPLICABLE: reorder requires at least two batches"
    else:
        cases.append(transform_mode1_reorder(data, list(order), seed))
        overrides.append(None)

    batch_split = config.get(BATCH_SPLIT_KEY)
    if batch_split is None:
        skipped["split"] = "NOT_APPLICABLE: split requires at least two batches"
    else:
        split_cases = transform_mode2_split(
            data,
            groups_from_split_sizes(CaseAdapter(data).get_batch_size(), batch_split),
            seed,
        )
        cases.extend(split_cases)
        overrides.extend([None] * len(split_cases))

    mode_batch = int(config[MODE_BATCH_KEY])
    token_split = config.get(TOKEN_SPLIT_KEY)
    if token_split is None:
        skipped["token-split"] = (
            "NOT_APPLICABLE: token-split requires at least two valid Q tokens"
        )
    else:
        try:
            token_cases = transform_mode3_token_partition(data, mode_batch, token_split)
        except InvalidTransformError as error:
            skipped["token-split"] = f"NOT_APPLICABLE: {error}"
        else:
            cases.extend(token_cases)
            overrides.extend([None] * len(token_cases))

    common_tokens, derived_extra = config[SHAPE_CHANGE_KEY]
    try:
        shape_case = transform_mode4_shape_change(
            data, mode_batch, int(common_tokens), int(derived_extra)
        )
    except InvalidTransformError as error:
        skipped["shape-change"] = f"NOT_APPLICABLE: {error}"
    else:
        cases.append(shape_case)
        overrides.append([int(common_tokens)])

    runner = ConsistencyRunner(executor, precision_compare, set_device)
    report = runner.run_cases(data, cases, overrides)
    report["config"] = dict(config)
    report["skipped_modes"] = skipped
    return report


def consistency_fulfill_percent(report: Mapping[str, Any]) -> float:
    """Return the minimum numeric precision percentage in a consistency report."""
    values = []
    precision_reports = [report.get("baseline_precision", {})]
    precision_reports.extend(
        relation.get("precision", {}) for relation in report.get("relations", [])
    )
    for precision in precision_reports:
        for name in ("output", "softmax_lse"):
            value = precision.get(name, {}).get("fulfill_percent")
            if value is not None:
                values.append(float(value))
    return min(values) if values else 0.0


def format_consistency_summary(report: Mapping[str, Any]) -> str:
    """Format an at-a-glance PASS/FAILED/SKIPPED summary for all modes."""

    def status(value: Any) -> str:
        return "PASS" if value is True else "FAILED"

    def precision_detail(precision: Mapping[str, Any]) -> str:
        output = precision.get("output", {})
        percent = output.get("fulfill_percent")
        suffix = "" if percent is None else f" ({float(percent):.2f}%)"
        return f"{status(precision.get('pass'))}{suffix}"

    rows = []
    baseline = report.get("baseline_precision", {})
    rows.append(
        ("baseline", status(baseline.get("pass")), "-", precision_detail(baseline))
    )
    for relation in report.get("relations", []):
        rows.append(
            (
                str(relation.get("case", "unknown")),
                status(relation.get("pass")),
                status(relation.get("batch_consistency", {}).get("pass")),
                precision_detail(relation.get("precision", {})),
            )
        )
    for mode, reason in report.get("skipped_modes", {}).items():
        short_reason = str(reason).removeprefix("NOT_APPLICABLE: ")
        if len(short_reason) > 96:
            short_reason = f"{short_reason[:93]}..."
        rows.append((str(mode), "SKIPPED", "-", short_reason))

    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(
            ("MODE", "STATUS", "CONSISTENCY", "PRECISION / REASON")
        )
    ]
    header = "  ".join(
        value.ljust(widths[index])
        for index, value in enumerate(
            ("MODE", "STATUS", "CONSISTENCY", "PRECISION / REASON")
        )
    )
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    overall = status(report.get("pass"))
    return "\n".join(
        [
            "\n========== BATCH CONSISTENCY SUMMARY ==========",
            header,
            separator,
            *body,
            f"OVERALL: {overall}",
            "================================================",
        ]
    )


def print_consistency_summary(report: Mapping[str, Any]) -> None:
    print(format_consistency_summary(report), flush=True)


def run_pytest_case(
    case_path: str,
    mode: str,
    prefix: str,
    operator: str,
    executor: Callable[[Dict[str, Any]], Any],
    precision_compare: Callable[[Any, Any], Any],
    set_device: Callable[[], None],
) -> None:
    data = torch.load(case_path, map_location="cpu", weights_only=False)
    detected = CaseAdapter(data).name
    if detected != operator:
        pytest.skip(f"NOT_APPLICABLE: PT belongs to {detected}, expected {operator}")
    runner = ConsistencyRunner(executor, precision_compare, set_device)
    try:
        report = run_selected_mode(runner, data, mode, prefix)
    except InvalidTransformError as error:
        pytest.skip(f"INVALID_TRANSFORM: {error}")
    except BatchConsistencyUnsupportedError as error:
        pytest.skip(str(error))
    print(report)
    if report.get("pass") is not True:
        pytest.fail(f"batch consistency failed: {report}")


def run_independent_pytest_case(
    relation: Mapping[str, Any],
    operator: str,
    executor: Callable[[Dict[str, Any]], Any],
    precision_compare: Callable[[Any, Any], Any],
    set_device: Callable[[], None],
) -> None:
    """Execute an explicit cross-PT relation through the normal operator path."""
    baseline_data = torch.load(
        relation["baseline_pt"], map_location="cpu", weights_only=False
    )
    derived_data = torch.load(
        relation["derived_pt"], map_location="cpu", weights_only=False
    )
    baseline_operator = CaseAdapter(baseline_data).name
    derived_operator = CaseAdapter(derived_data).name
    if baseline_operator != operator or derived_operator != operator:
        raise ValueError(
            "CONFIG_ERROR: independent relation "
            f"{relation['id']!r} has PT operators ({baseline_operator}, {derived_operator}), "
            f"expected ({operator}, {operator})"
        )
    runner = ConsistencyRunner(executor, precision_compare, set_device)
    try:
        report = runner.run_independent_relation(
            baseline_data,
            derived_data,
            relation["id"],
            relation["mappings"],
        )
    except InvalidTransformError as error:
        pytest.skip(f"INVALID_TRANSFORM: {error}")
    except BatchConsistencyUnsupportedError as error:
        pytest.skip(str(error))
    print(report)
    if report.get("pass") is not True:
        pytest.fail(f"independent batch consistency failed: {report}")
