# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

DCUT_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (DCUT_ROOT / relative_path).read_text(encoding="utf-8")


def test_piecewise_gdn_does_not_nest_a_local_npugraph() -> None:
    core = _read("gdn_forward_v023.py")
    runner = _read("patch_runner.py")

    assert "torch.npu.NPUGraph" not in core
    assert "graph.replay()" not in core
    assert "_dcut_gdn_local_graph" not in core
    assert "_dcut_gdn_local_graph" not in runner
    assert "forward_with_recurrent_boundary" in core
    assert "torch.ops.vllm.dcut_gdn_recurrent" in core
    assert "_dcut_gdn_recurrent_piecewise_safe" in runner