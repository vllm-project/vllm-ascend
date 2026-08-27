# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for Qwen3.8-Flash-Next MTP integration."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "model.py"
MTP = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "mtp.py"
PROPOSER = ROOT / "vllm_ascend" / "spec_decode" / "eagle_proposer.py"
DISPATCHER = ROOT / "vllm_ascend" / "spec_decode" / "__init__.py"
QSA = ROOT / "vllm_ascend" / "models" / "qwen4_exp" / "qsa.py"
EAGLE = ROOT / "vllm_ascend" / "spec_decode" / "eagle_proposer.py"


def _class(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found in {path}")


def _method(path: Path, cls_name: str, method_name: str) -> ast.FunctionDef:
    for node in _class(path, cls_name).body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"method {cls_name}.{method_name} not found")


def test_qwen4_exp_mtp_uses_local_drafter_inputs_on_last_pp_stage() -> None:
    source = ast.unparse(_method(MTP, "AscendQwen4ExpMultiTokenPredictor", "forward"))
    assert "get_pp_group().is_first_rank or intermediate_tensors is None" in source


def test_qwen4_exp_mtp_accepts_multimodal_embedding_arguments() -> None:
    method = _method(MTP, "AscendQwen4ExpMTP", "embed_input_ids")
    arguments = [argument.arg for argument in method.args.args]
    assert arguments == [
        "self",
        "input_ids",
        "multimodal_embeddings",
        "is_multimodal",
    ]


def test_qwen4_exp_proposer_uses_text_hyperconnection_width() -> None:
    source = ast.unparse(_method(PROPOSER, "AscendQwen4ExpMTPProposer", "_get_hidden_size"))
    assert "self.draft_model_config.hf_text_config" in source
    assert "text_config.hidden_size * text_config.hc_count" in source
    assert ".get_hidden_size()" not in source


def test_qwen4_exp_proposer_is_dispatched_before_generic_mtp() -> None:
    source = DISPATCHER.read_text()
    qwen_dispatch = source.index("use_qwen4_exp_mtp")
    generic_dispatch = source.index("return AscendEagleProposer", qwen_dispatch)
    assert qwen_dispatch < generic_dispatch


def test_qwen4_exp_graph_does_not_mark_query_start_loc_dynamic() -> None:
    source = ast.unparse(_method(MODEL, "AscendQwen4ExpModel", "__init__"))
    assert "name !=" in source
    assert "query_start_loc" in source


def test_qwen4_exp_wrappers_use_vendored_model_sources() -> None:
    sources = "\n".join(path.read_text() for path in (MODEL, MTP, QSA))
    assert "vllm.models.qwen4_exp" not in sources
    assert "from .nvidia" in sources


def test_qwen4_exp_proposer_uses_local_backport() -> None:
    source = EAGLE.read_text()
    assert "from vllm_ascend.spec_decode.qwen4_exp import" in source
    assert "from vllm.v1.spec_decode.qwen4_exp import" not in source
