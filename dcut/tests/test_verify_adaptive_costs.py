# SPDX-License-Identifier: Apache-2.0

import ast
import math
from pathlib import Path

import numpy as np

CONTROLLER_PATH = (
    Path(__file__).resolve().parents[1]
    / "verify_adaptive_controller.py"
)


def _load_choose_query_lens_discrete():
    tree = ast.parse(CONTROLLER_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "choose_query_lens_discrete"
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
    namespace = {"np": np, "math": math}
    exec(compile(module, str(CONTROLLER_PATH), "exec"), namespace)
    return namespace["choose_query_lens_discrete"]


def test_fixed_draft_cost_can_make_longer_verify_shape_optimal() -> None:
    choose = _load_choose_query_lens_discrete()
    probs = [[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]]
    target_costs = {4: 1.0, 8: 2.0}

    target_only = choose(
        probs=probs,
        base_batch_size=2,
        q_levels=[4, 8],
        cost_lookup=target_costs.__getitem__,
        max_draft_len=3,
    )
    end_to_end = choose(
        probs=probs,
        base_batch_size=2,
        q_levels=[4, 8],
        cost_lookup=target_costs.__getitem__,
        max_draft_len=3,
        draft_cost_lookup=lambda _q: 4.0,
    )

    assert target_only["best_Q"] == 4
    assert end_to_end["best_Q"] == 8


def test_q_dependent_draft_cost_is_included_in_records() -> None:
    choose = _load_choose_query_lens_discrete()
    result = choose(
        probs=[[0.9, 0.9]],
        base_batch_size=1,
        q_levels=[2, 3],
        cost_lookup=lambda q: {2: 2.0, 3: 3.0}[q],
        max_draft_len=2,
        collect_records=True,
        draft_cost_lookup=lambda q: {2: 0.5, 3: 1.5}[q],
    )

    records = result["records"]
    assert records is not None
    assert records[0]["Q"] == 2
    assert records[0]["target_cost"] == 2.0
    assert records[0]["draft_cost"] == 0.5
    assert records[0]["cost"] == 2.5
    assert math.isclose(records[0]["score"], (1.0 + 0.9) / 2.5)
    assert records[1]["Q"] == 3
    assert records[1]["target_cost"] == 3.0
    assert records[1]["draft_cost"] == 1.5
    assert records[1]["cost"] == 4.5
    assert math.isclose(
        records[1]["score"],
        (1.0 + 0.9 + 0.81) / 4.5,
    )
