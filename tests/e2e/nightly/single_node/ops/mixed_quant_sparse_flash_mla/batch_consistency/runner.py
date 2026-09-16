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

"""Execute black-box relations through an operator's normal pytest path."""

from typing import Any, Callable, Dict, List, Optional, Sequence

import torch_npu

from .adapter import CaseAdapter
from .compare import ResultComparator
from .model import ConsistencyCase, RunResult, TokenOrigin
from .schema import (
    build_baseline_origins,
    check_invariants,
    get_q_lengths,
    validate_schema,
)
from .transform import (
    ActualInputSemanticOracle,
    transform_mode1_reorder,
    transform_mode2_split,
    transform_mode3_token_partition,
    transform_mode4_shape_change,
    transform_mode5_token_isolation,
)


class BatchConsistencyUnsupportedError(RuntimeError):
    """The installed runtime cannot execute deterministic level 3."""


def classify_runtime_boundary(error: BaseException) -> Optional[str]:
    """Classify known package/SoC and API-contract rejections without hiding bugs."""
    message = str(error)
    lowered = message.casefold()
    environment_markers = (
        "soc version",
        "operator package",
        "check nnopexecutor",
        "binary_info_config.json",
        "not configured through the addconfig api",
    )
    if any(marker in lowered for marker in environment_markers):
        return f"ENVIRONMENT_UNSUPPORTED: {message}"
    if "has incorrect value" in lowered or "must be 127" in lowered:
        return f"UNSUPPORTED: runtime rejected testcase parameters: {message}"
    return None


class DeterministicLevelGuard:
    """Keep level 3 active across metadata and main-operator calls."""

    LEVEL = 3

    def __init__(self):
        self.previous_level: Optional[int] = None

    @staticmethod
    def current_level() -> int:
        getter = getattr(torch_npu.npu, "_get_deterministic_level", None)
        if getter is None:
            raise BatchConsistencyUnsupportedError(
                "UNSUPPORTED: torch_npu does not expose deterministic-level state"
            )
        return int(getter())

    @staticmethod
    def runtime_rejects_level3(error: BaseException) -> bool:
        message = str(error).lower()
        return "deterministic" in message and (
            "0/1/2" in message or "level 3" in message or "level3" in message
        )

    @classmethod
    def set_level(cls, level: int) -> None:
        setter = getattr(torch_npu.npu, "set_deterministic_level", None)
        if setter is None:
            raise BatchConsistencyUnsupportedError(
                "UNSUPPORTED: torch_npu does not expose set_deterministic_level"
            )
        setter(level)
        actual = cls.current_level()
        if actual != level:
            raise RuntimeError(
                f"set_deterministic_level({level}) did not take effect; actual={actual}"
            )

    def restore(self) -> None:
        if (
            self.previous_level is not None
            and self.current_level() != self.previous_level
        ):
            self.set_level(self.previous_level)

    def __enter__(self):
        self.previous_level = self.current_level()
        try:
            self.set_level(self.LEVEL)
        except BaseException as error:
            self.restore()
            if self.runtime_rejects_level3(error):
                raise BatchConsistencyUnsupportedError(
                    f"UNSUPPORTED: runtime rejected deterministic level 3: {error}"
                ) from error
            raise
        return self

    def __exit__(self, exc_type, error, traceback) -> bool:
        try:
            self.restore()
        except BaseException as restore_error:
            raise RuntimeError(
                f"failed to restore deterministic level {self.previous_level}: {restore_error}"
            ) from restore_error
        if error is not None and self.runtime_rejects_level3(error):
            raise BatchConsistencyUnsupportedError(
                f"UNSUPPORTED: NPU dispatch rejected deterministic level 3: {error}"
            ) from error
        return False


class ConsistencyRunner:
    """Run transformed cases with an injected SMLA-family executor."""

    def __init__(
        self,
        executor: Callable[[Dict[str, Any]], RunResult],
        precision_compare: Callable[[Any, Any], Any],
        device_setup: Callable[[], None],
    ):
        self.executor = executor
        self.comparator = ResultComparator(precision_compare)
        self.device_setup = device_setup

    @staticmethod
    def cpu_result(input_data: Dict[str, Any]) -> RunResult:
        softmax_lse = input_data.get("softmax_lse")
        if softmax_lse is None:
            softmax_lse = input_data.get("cpu_lse")
        return RunResult(input_data["cpu_output"], softmax_lse)

    @staticmethod
    def lengths(input_data: Dict[str, Any]) -> tuple[List[int], List[int]]:
        return get_q_lengths(input_data), get_q_lengths(input_data, valid_only=True)

    def execute(
        self, input_data: Dict[str, Any], classify_case_boundary: bool = False
    ) -> RunResult:
        """Run one call and preserve a precise package/case boundary status."""
        try:
            return self.executor(input_data)
        except Exception as error:
            boundary = classify_runtime_boundary(error)
            if boundary is None or (
                boundary.startswith("UNSUPPORTED:") and not classify_case_boundary
            ):
                raise
            raise BatchConsistencyUnsupportedError(boundary) from error

    def compare_case(
        self,
        baseline_data: Dict[str, Any],
        baseline: RunResult,
        baseline_origins: Sequence[Any],
        case: ConsistencyCase,
        derived: RunResult,
        derived_valid_override: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        layout_q = CaseAdapter(baseline_data).get_layout_q()
        baseline_physical, baseline_valid = self.lengths(baseline_data)
        derived_physical, derived_valid = self.lengths(case.input_data)
        if derived_valid_override is not None:
            derived_valid = derived_valid_override
        compare_lse = bool(baseline_data["params"].get("return_softmax_lse", False))
        consistency = self.comparator.compare_results(
            baseline,
            baseline_origins,
            baseline_physical,
            baseline_valid,
            derived,
            case.output_origins,
            derived_physical,
            derived_valid,
            layout_q,
            compare_lse,
            True,
            case.output_coordinates,
        )
        precision = self.comparator.compare_results(
            self.cpu_result(baseline_data),
            baseline_origins,
            baseline_physical,
            baseline_valid,
            derived,
            case.output_origins,
            derived_physical,
            derived_valid,
            layout_q,
            compare_lse,
            False,
            case.output_coordinates,
        )
        return {
            "case": case.name,
            "transform": case.transform_meta,
            "batch_consistency": consistency,
            "precision": precision,
            "pass": consistency["pass"] and precision["pass"],
        }

    def compare_baseline_precision(
        self,
        input_data: Dict[str, Any],
        baseline: RunResult,
        origins: Sequence[Any],
    ) -> Dict[str, Any]:
        physical, valid = self.lengths(input_data)
        compare_lse = bool(input_data["params"].get("return_softmax_lse", False))
        return self.comparator.compare_results(
            self.cpu_result(input_data),
            origins,
            physical,
            valid,
            baseline,
            origins,
            physical,
            valid,
            CaseAdapter(input_data).get_layout_q(),
            compare_lse,
            False,
        )

    def run_cases(
        self,
        input_data: Dict[str, Any],
        cases: Sequence[ConsistencyCase],
        derived_valid_overrides: Optional[Sequence[Optional[List[int]]]] = None,
    ) -> Dict[str, Any]:
        validate_schema(input_data)
        check_invariants(input_data)
        overrides = list(derived_valid_overrides or [None] * len(cases))
        if len(overrides) != len(cases):
            raise ValueError("derived comparison scopes do not match transformed cases")

        self.device_setup()
        with DeterministicLevelGuard():
            baseline = self.execute(input_data, classify_case_boundary=True)
            origins = build_baseline_origins(input_data, valid_only=True)
            baseline_precision = self.compare_baseline_precision(
                input_data, baseline, origins
            )
            reports = [
                self.compare_case(
                    input_data,
                    baseline,
                    origins,
                    case,
                    self.execute(case.input_data),
                    override,
                )
                for case, override in zip(cases, overrides)
            ]
        return {
            "deterministic_level": 3,
            "baseline_precision": baseline_precision,
            "relations": reports,
            "pass": (
                baseline_precision["pass"]
                and bool(reports)
                and all(report["pass"] for report in reports)
            ),
        }

    @staticmethod
    def validate_independent_mapping(
        baseline_data: Dict[str, Any],
        derived_data: Dict[str, Any],
        mappings: Sequence[tuple[int, int, int, int]],
    ) -> None:
        """Validate declared cross-case coordinates before any device call."""
        if not mappings:
            raise ValueError("independent relation requires at least one mapping")
        baseline_lengths = get_q_lengths(baseline_data, valid_only=True)
        derived_lengths = get_q_lengths(derived_data, valid_only=True)
        baseline_seen = set()
        derived_seen = set()
        for baseline_batch, baseline_token, derived_batch, derived_token in mappings:
            if (
                baseline_batch < 0
                or baseline_batch >= len(baseline_lengths)
                or baseline_token < 0
                or baseline_token >= baseline_lengths[baseline_batch]
            ):
                raise ValueError(
                    "baseline mapping coordinate "
                    f"({baseline_batch}, {baseline_token}) is outside valid Q tokens"
                )
            if (
                derived_batch < 0
                or derived_batch >= len(derived_lengths)
                or derived_token < 0
                or derived_token >= derived_lengths[derived_batch]
            ):
                raise ValueError(
                    "derived mapping coordinate "
                    f"({derived_batch}, {derived_token}) is outside valid Q tokens"
                )
            baseline_coordinate = (baseline_batch, baseline_token)
            derived_coordinate = (derived_batch, derived_token)
            if baseline_coordinate in baseline_seen:
                raise ValueError(
                    f"duplicate baseline mapping coordinate {baseline_coordinate}"
                )
            if derived_coordinate in derived_seen:
                raise ValueError(
                    f"duplicate derived mapping coordinate {derived_coordinate}"
                )
            baseline_seen.add(baseline_coordinate)
            derived_seen.add(derived_coordinate)

    def run_independent_relation(
        self,
        baseline_data: Dict[str, Any],
        derived_data: Dict[str, Any],
        relation_id: str,
        mappings: Sequence[tuple[int, int, int, int]],
    ) -> Dict[str, Any]:
        """Run one explicitly declared relation between two independently saved PT cases."""
        validate_schema(baseline_data)
        validate_schema(derived_data)
        check_invariants(baseline_data)
        check_invariants(derived_data)
        self.validate_independent_mapping(baseline_data, derived_data, mappings)
        semantic_oracle = ActualInputSemanticOracle.validate_mapped_tokens(
            baseline_data,
            derived_data,
            mappings,
        )

        baseline_origins = build_baseline_origins(baseline_data, valid_only=True)
        relation_case = ConsistencyCase(
            name=f"independent_{relation_id}",
            input_data=derived_data,
            output_origins=[
                TokenOrigin(batch, token) for batch, token, _, _ in mappings
            ],
            transform_meta={
                "scenario": "independent-cross-case",
                "relation_id": relation_id,
                "mapping_count": len(mappings),
            },
            output_coordinates=[(batch, token) for _, _, batch, token in mappings],
        )
        baseline_physical, baseline_valid = self.lengths(baseline_data)
        derived_physical, derived_valid = self.lengths(derived_data)
        compare_lse = bool(baseline_data["params"].get("return_softmax_lse", False))
        layout_q = CaseAdapter(baseline_data).get_layout_q()
        cpu_relation = self.comparator.compare_results(
            self.cpu_result(baseline_data),
            baseline_origins,
            baseline_physical,
            baseline_valid,
            self.cpu_result(derived_data),
            relation_case.output_origins,
            derived_physical,
            derived_valid,
            layout_q,
            compare_lse,
            False,
            relation_case.output_coordinates,
        )
        if not cpu_relation["pass"]:
            return {
                "relation_id": relation_id,
                "scenario": "independent-cross-case",
                "semantic_oracle": semantic_oracle,
                "cpu_relation": cpu_relation,
                "pass": False,
            }

        self.device_setup()
        with DeterministicLevelGuard():
            baseline = self.execute(baseline_data, classify_case_boundary=True)
            derived = self.execute(derived_data, classify_case_boundary=True)
            report = self.compare_case(
                baseline_data,
                baseline,
                baseline_origins,
                relation_case,
                derived,
            )
        return {
            "deterministic_level": 3,
            "relation_id": relation_id,
            "scenario": "independent-cross-case",
            "semantic_oracle": semantic_oracle,
            "cpu_relation": cpu_relation,
            "batch_consistency": report["batch_consistency"],
            "precision": report["precision"],
            "pass": report["pass"],
        }

    def run_mode1(
        self, input_data: Dict[str, Any], order: Optional[List[int]], seed: int
    ) -> Dict[str, Any]:
        return self.run_cases(
            input_data, [transform_mode1_reorder(input_data, order, seed)]
        )

    def run_mode2(
        self, input_data: Dict[str, Any], groups: List[List[int]], seed: int
    ) -> Dict[str, Any]:
        cases = transform_mode2_split(input_data, groups, seed, False)
        return self.run_cases(input_data, cases)

    def run_mode3(
        self, input_data: Dict[str, Any], batch_id: int, split_sizes: Sequence[int]
    ) -> Dict[str, Any]:
        return self.run_cases(
            input_data,
            transform_mode3_token_partition(input_data, batch_id, split_sizes),
        )

    def run_mode4(
        self,
        input_data: Dict[str, Any],
        batch_id: int,
        common_tokens: int,
        derived_extra_tokens: int,
    ) -> Dict[str, Any]:
        case = transform_mode4_shape_change(
            input_data, batch_id, common_tokens, derived_extra_tokens
        )
        return self.run_cases(input_data, [case], [[common_tokens]])

    def run_mode5(
        self, input_data: Dict[str, Any], batch_id: int, token_id: int, seed: int
    ) -> Dict[str, Any]:
        case = transform_mode5_token_isolation(input_data, batch_id, token_id, seed)
        return self.run_cases(input_data, [case])
