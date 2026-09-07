# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash page-size padding must follow the *active* kpool indexer.

A kpool-shaped checkpoint carries ``index_kpool`` whether or not the sparse
indexer is switched on, so ``model_uses_kpool_indexer`` cannot decide the cache
layout on its own. With ``index_topk`` set, the indexer registers k_cache /
tail_cache layers and vLLM's GLM-5.3-Flash grouping lays the pages out itself,
requiring the attention specs unpadded. With ``index_topk`` unset the model is
dense NoPE MLA, no indexer layer exists, that grouping never runs, and Ascend
must pad the MLA pages up to the Mamba page as it does for any hybrid model.

Serving the dense variant used to hit the generic grouping path with unpadded
pages; ``kpool_indexer_is_active`` is what keeps the two cases apart.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from vllm_ascend.utils import kpool_indexer_is_active, model_uses_kpool_indexer

ROOT = Path(__file__).resolve().parents[3]
MODEL_RUNNER = ROOT / "vllm_ascend" / "worker" / "model_runner_v1.py"
GLM_MODEL = ROOT / "vllm_ascend" / "models" / "glm5next" / "model.py"

PADDING_GUARD = "not kpool_indexer_is_active(self.model_config)"
PADDED_FIELD = "page_size_padded"


def _model_config(**text_config_fields) -> SimpleNamespace:
    return SimpleNamespace(hf_text_config=SimpleNamespace(**text_config_fields))


def test_sparse_kpool_checkpoint_is_active() -> None:
    model_config = _model_config(index_kpool=4, index_topk=2048)

    assert model_uses_kpool_indexer(model_config)
    assert kpool_indexer_is_active(model_config)


def test_dense_kpool_checkpoint_is_not_active() -> None:
    """The dense variant is still kpool-shaped, but nothing indexes."""
    model_config = _model_config(index_kpool=4, index_topk=None)

    assert model_uses_kpool_indexer(model_config)
    assert not kpool_indexer_is_active(model_config)


def test_non_kpool_model_is_not_active() -> None:
    # DeepSeek SFA sets index_topk without index_kpool.
    assert not kpool_indexer_is_active(_model_config(index_topk=2048))
    assert not kpool_indexer_is_active(_model_config())
    assert not kpool_indexer_is_active(None)


def _guards_around_padding_assignment() -> set[str]:
    """Conditions of every ``if`` enclosing the mamba page-size alignment."""
    tree = ast.parse(MODEL_RUNNER.read_text(encoding="utf-8"))

    def pads_page_size(node: ast.AST) -> bool:
        return any(isinstance(inner, ast.Constant) and inner.value == PADDED_FIELD for inner in ast.walk(node)) and any(
            isinstance(inner, ast.Attribute) and inner.attr == "__setattr__" for inner in ast.walk(node)
        )

    return {
        ast.unparse(node.test)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and any(pads_page_size(stmt) for stmt in node.body)
    }


def test_mamba_alignment_is_skipped_only_for_an_active_indexer() -> None:
    guards = _guards_around_padding_assignment()

    assert PADDING_GUARD in guards, (
        f"the mamba page-size alignment is no longer gated on {PADDING_GUARD!r}; found {sorted(guards)}"
    )


def test_dense_load_skips_checkpoint_indexer_weights() -> None:
    """Dense NoPE MLA has no indexer module for those tensors to land in."""
    tree = ast.parse(GLM_MODEL.read_text(encoding="utf-8"))
    tests = {ast.unparse(node.test) for node in ast.walk(tree) if isinstance(node, ast.If)}

    assert "not self.is_v32 and '.indexer.' in name" in tests
