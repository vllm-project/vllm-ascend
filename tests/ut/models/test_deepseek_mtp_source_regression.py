# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for the Ascend DeepSeek MTP head alias."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "vllm_ascend" / "models" / "deepseek_mtp.py"


def test_load_weights_delegates_own_lm_head_setup_without_registering_alias() -> None:
    source = MODEL.read_text()

    assert "def _set_own_lm_head(self, loaded_weights: set[str]) -> None:" in source
    assert "self._set_own_lm_head(loaded_weights)" in source
    assert 'object.__setattr__(self, "lm_head", mtp_layer.shared_head.head)' in source
    assert "self.lm_head = mtp_layer.shared_head.head" not in source
