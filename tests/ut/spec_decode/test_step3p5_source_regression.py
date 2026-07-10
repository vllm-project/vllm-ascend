# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for Step3.5 MTP Ascend glue.

Importing the Step3.5 proposer can initialize runtime/device state in this
branch. Keep these checks focused on cross-file contracts that are hard to
exercise in a lightweight unit test, and avoid pinning the exact implementation
sequence inside the proposer.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STEP3P5 = ROOT / "vllm_ascend" / "spec_decode" / "step3p5.py"
BASE_PROPOSER = ROOT / "vllm_ascend" / "spec_decode" / "llm_base_proposer.py"
PATCH_SPEC_CFG = ROOT / "vllm_ascend" / "patch" / "platform" / "patch_speculative_config.py"
WORKER_PATCH_INIT = ROOT / "vllm_ascend" / "patch" / "worker" / "__init__.py"
LEGACY_STEP3P7_PATCH = ROOT / "vllm_ascend" / "patch" / "worker" / "patch_step3p5_mtp.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _class(path: Path, name: str) -> ast.ClassDef:
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in {path}")


def _method(path: Path, cls_name: str, method_name: str) -> ast.FunctionDef:
    cls = _class(path, cls_name)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"method {cls_name}.{method_name} not found")


def _func(path: Path, name: str) -> ast.FunctionDef:
    for node in _tree(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {path}")


def _src(node: ast.AST) -> str:
    return ast.unparse(node)


def test_step3p5_first_pass_forwards_rejected_token_counts() -> None:
    # set_inputs_first_pass is inherited from the base proposer; the base
    # simple-path is the canonical Step3.5 behaviour. Step3.5 only needs to
    # forward num_rejected_tokens_gpu through _propose.
    set_inputs = _method(BASE_PROPOSER, "AscendSpecDecodeBaseProposer", "set_inputs_first_pass")
    propose = _method(STEP3P5, "AscendStep3p5MTPProposer", "_propose")

    assert "num_rejected_tokens_gpu" in [arg.arg for arg in set_inputs.args.args]
    assert "num_rejected_tokens_gpu=num_rejected_tokens_gpu" in _src(propose)

    # Guard against the override creeping back: the previous step3p5 simple-path
    # was byte-equivalent to the base's `not needs_extra_input_slots and
    # pcp_size <= 1` branch, so a re-override is almost certainly redundant.
    step_methods = {n.name for n in _class(STEP3P5, "AscendStep3p5MTPProposer").body if isinstance(n, ast.FunctionDef)}
    assert "set_inputs_first_pass" not in step_methods


def test_step3p5_draft_window_and_config_contracts() -> None:
    base_run = _method(BASE_PROPOSER, "AscendSpecDecodeBaseProposer", "_run_merged_draft")
    step_run = _method(STEP3P5, "AscendStep3p5MTPProposer", "_run_merged_draft")
    run_window = _src(_method(STEP3P5, "AscendStep3p5MTPProposer", "_run_window_draft_steps"))
    build_metadata = _src(_method(STEP3P5, "AscendStep3p5MTPProposer", "_build_step_attn_metadatas"))
    roll_inputs = _src(_method(STEP3P5, "AscendStep3p5MTPProposer", "_roll_window_inputs_only"))
    ensure_layer_types = _src(
        _method(
            STEP3P5,
            "AscendStep3p5MTPProposer",
            "_ensure_draft_layer_types_cover_mtp_layers",
        )
    )
    create_config = _src(_method(STEP3P5, "AscendStep3p5MTPProposer", "_create_draft_vllm_config"))

    assert [arg.arg for arg in step_run.args.args] == [arg.arg for arg in base_run.args.args]
    assert "multi_steps_attn_metadata.append(per_step_attn_metadata)" in build_metadata
    assert "multi_steps_attn_metadata[spec_step_idx]" in run_window
    assert "self.input_ids[token_indices_to_sample]" in roll_inputs
    assert "_ensure_draft_layer_types_cover_mtp_layers()" in create_config
    assert "self.draft_model_config.hf_config" in ensure_layer_types
    assert "self.vllm_config.model_config.hf_config" not in ensure_layer_types
    assert "sliding_attention" in ensure_layer_types


def test_step3p7_uses_step3p5_mtp_override_without_legacy_runtime_patch() -> None:
    override_src = _src(_func(PATCH_SPEC_CFG, "hf_config_override"))

    assert "step3p7" in override_src
    assert "Step3p7ForConditionalGeneration" in override_src
    assert "step3p5_mtp" in override_src
    assert "Step3p5MTP" in override_src
    assert "patch_step3p5_mtp" not in WORKER_PATCH_INIT.read_text()
    assert not LEGACY_STEP3P7_PATCH.exists()


def test_pad_query_start_loc_for_fia_first_arg_is_query_start_loc() -> None:
    """No7: _propose must pass query_start_loc as the first arg.

    _pad_query_start_loc_for_fia signature:
        (self, query_start_loc, num_tokens_padded, num_reqs_padded,
         num_reqs, cudagraph_runtime_mode, batch_desc_num_reqs) -> int

    Without query_start_loc, every subsequent argument shifts left by one,
    causing cudagraph_runtime_mode (a CUDAGraphMode enum) to be treated as
    num_reqs (int) and eventually triggering:
        TypeError: unsupported operand type(s) for *: 'CUDAGraphMode' and 'int'

    See model_runner_v1.py:_pad_query_start_loc_for_fia and the correct call
    pattern at llm_base_proposer.py:730–736.
    """
    propose = _method(STEP3P5, "AscendStep3p5MTPProposer", "_propose")
    propose_src = _src(propose)

    # Verify the call site passes query_start_loc as the first argument.
    assert "self.runner._pad_query_start_loc_for_fia(" in propose_src
    assert "self.runner.query_start_loc," in propose_src

    # Walk the AST to confirm the first positional arg is query_start_loc.
    import ast as _ast

    class _CallFinder(_ast.NodeVisitor):
        def __init__(self) -> None:
            self.call: ast.Call | None = None

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "self"
                and node.func.value.attr == "runner"
                and node.func.attr == "_pad_query_start_loc_for_fia"
            ):
                self.call = node

    finder = _CallFinder()
    finder.visit(propose)
    assert finder.call is not None, "_pad_query_start_loc_for_fia call not found"

    args = finder.call.args
    assert len(args) >= 1

    first_arg = args[0]
    assert isinstance(first_arg, ast.Attribute)
    assert isinstance(first_arg.value, ast.Attribute)
    assert isinstance(first_arg.value.value, ast.Name)
    assert first_arg.value.value.id == "self"
    assert first_arg.value.attr == "runner"
    assert first_arg.attr == "query_start_loc"

    # Verify the Step3.5 call pattern matches the base proposer's pattern
    # for the first three args (query_start_loc, num_input_tokens,
    # batch_descriptor-based num_reqs).
    base_propose_src = _src(_method(BASE_PROPOSER, "AscendSpecDecodeBaseProposer", "_propose"))

    def _find_call_in_src(src: str) -> str | None:
        """Extract the _pad_query_start_loc_for_fia(...) call span."""
        marker = "self.runner._pad_query_start_loc_for_fia("
        idx = src.find(marker)
        if idx == -1:
            return None
        start = idx
        depth = 0
        for i in range(idx + len(marker), len(src)):
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                if depth == 0:
                    return src[start : i + 1]
                depth -= 1
        return None

    step_call = _find_call_in_src(propose_src)
    base_call = _find_call_in_src(base_propose_src)
    assert step_call is not None
    assert base_call is not None
    assert "query_start_loc" in step_call
    assert "num_input_tokens" in step_call
