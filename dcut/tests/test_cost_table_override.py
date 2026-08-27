# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import json
import math
from pathlib import Path


CONTROLLER_PATH = (
    Path(__file__).resolve().parents[1] / "verify_adaptive_controller.py"
)


def _load_parser():
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_parse_cost_table_override"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"math": math}
    exec(compile(module, str(CONTROLLER_PATH), "exec"), namespace)
    return namespace["_parse_cost_table_override"]


def _payload() -> dict:
    return {
        "schema_version": 2,
        "num_spec_tokens": 15,
        "max_batch_size": 32,
        "batch_size_levels": [16, 32],
        "query_len_levels": [2, 4, 6, 16],
        "cost_table": [
            {
                "batch_size": 32,
                "sum_query_len": 128,
                "query_len_per_req": 4,
                "cost_s": 0.05,
                "target_cost_s": 0.04,
                "draft_cost_s": 0.01,
            },
            {
                "batch_size": 32,
                "sum_query_len": 192,
                "query_len_per_req": 6,
                "cost_s": 0.055,
                "target_cost_s": 0.044,
                "draft_cost_s": 0.011,
            },
        ],
    }


def _parse(payload: dict, *, num_spec_tokens: int = 15):
    return _load_parser()(
        payload,
        num_spec_tokens=num_spec_tokens,
        max_batch_size=32,
        batch_size_levels=[16, 32],
        query_len_levels=[2, 4, 6, 16],
    )


def test_schema_v2_cost_table_can_override_profiled_costs() -> None:
    target, draft = _parse(_payload())

    assert target == {(32, 128): 0.04, (32, 192): 0.044}
    assert draft == {(32, 128): 0.01, (32, 192): 0.011}


def test_override_rejects_runtime_grid_mismatch() -> None:
    try:
        _parse(_payload(), num_spec_tokens=7)
    except ValueError as exc:
        assert "num_spec_tokens mismatch" in str(exc)
    else:
        raise AssertionError("expected incompatible override to be rejected")


def test_override_rejects_inconsistent_component_total() -> None:
    payload = json.loads(json.dumps(_payload()))
    payload["cost_table"][0]["cost_s"] = 0.5

    try:
        _parse(payload)
    except ValueError as exc:
        assert "total does not match" in str(exc)
    else:
        raise AssertionError("expected malformed override to be rejected")


def test_override_is_optional_and_applied_after_profile_dump() -> None:
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    override_branch = source.index("if override_path:")
    profile_dump = source.index(
        "self._dump_cost_table_if_requested()", override_branch
    )
    replace_table = source.index(
        "self._cost_table = override_target", override_branch
    )

    assert 'os.getenv(ENV_COST_TABLE_OVERRIDE)' in source
    assert profile_dump < replace_table
    assert "if not override_path:" in source
    assert "replaced the profiled cost" in source
