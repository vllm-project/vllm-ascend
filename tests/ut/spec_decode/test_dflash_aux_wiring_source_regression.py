# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for the DFlash auxiliary hidden state wiring.

DFlash/DFlash2 drafts from the target's intermediate layers, so the v1 model
runner has to enable ``use_aux_hidden_state_outputs`` for it -- without the flag
the drafter's ``fc`` silently receives the final hidden states instead of the
concatenated ``dflash_config.target_layer_ids`` slice.

The branch lives in ``NPUModelRunner.__init__``, which cannot be instantiated
without a device, hence the source-level check.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_RUNNER = ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"

AUX_FLAG = "self.use_aux_hidden_state_outputs"
CHAIN_HEAD = "self.speculative_config.method == 'eagle3'"


def _aux_branch_chain() -> list[tuple[str, bool]]:
    """Return (condition, sets-the-aux-flag) for each branch of the chain that
    picks a drafter and decides whether auxiliary outputs are needed."""
    tree = ast.parse(MODEL_RUNNER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) == CHAIN_HEAD:
            break
    else:
        raise AssertionError(f"drafter selection chain starting with `{CHAIN_HEAD}` not found")

    chain: list[tuple[str, bool]] = []
    branch = node
    while True:
        sets_flag = any(
            isinstance(stmt, ast.Assign) and ast.unparse(stmt.targets[0]) == AUX_FLAG for stmt in branch.body
        )
        chain.append((ast.unparse(branch.test), sets_flag))
        if len(branch.orelse) == 1 and isinstance(branch.orelse[0], ast.If):
            branch = branch.orelse[0]
        else:
            break
    return chain


def test_dflash_enables_auxiliary_hidden_state_outputs() -> None:
    chain = dict(_aux_branch_chain())

    assert chain.get("self.speculative_config.use_dflash()") is True


def test_dspark_is_checked_before_dflash() -> None:
    """``AscendDSparkProposer`` derives from ``AscendDflashProposer``, so a
    ``use_dflash()`` branch placed first would swallow DSpark."""
    conditions = [condition for condition, _ in _aux_branch_chain()]

    assert conditions.index("self.speculative_config.use_dspark()") < conditions.index(
        "self.speculative_config.use_dflash()"
    )


def test_every_branch_in_the_chain_configures_auxiliary_outputs() -> None:
    """Each drafter that reaches this chain either needs auxiliary outputs or has
    to say so explicitly; a silent branch is the bug this file guards against."""
    for condition, sets_flag in _aux_branch_chain():
        assert sets_flag, f"branch `{condition}` does not set {AUX_FLAG}"
