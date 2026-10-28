#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: ignore-errors
"""UT: token_logprob selected-token scoring without API SamplingParams.logprobs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.observability.runtime_config.config import RuntimeConfig
from vllm_ascend.observability.runtime_guard.detector.token_logprob import (
    TokenLogprobDetector,
    selected_token_logprobs,
)
from vllm_ascend.observability.runtime_guard.hooks import (
    need_pre_sample_hook,
    stash_pre_sample_logits,
    take_pre_sample_logits,
)
from vllm_ascend.observability.runtime_guard.state import RequestGuardStore


def _cfg(tmp_path: Path, **token_logprob_kw) -> RuntimeConfig:
    RequestGuardStore.reset_for_tests()
    cfg = RuntimeConfig(
        config_path=tmp_path / "c.json",
        report_dir=tmp_path / "r",
        ensure_file=True,
        hot_reload=False,
    )
    sec = cfg._data["detector"]["token_logprob"]
    sec["enabled"] = True
    sec.update(token_logprob_kw)
    cfg._invalidate_hot_path_gates()
    return cfg


def test_catalog_includes_token_logprob_defaults():
    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config.detector_catalog import DETECTOR_SECTIONS

    assert "token_logprob" in DETECTOR_SECTIONS
    sec = _DEFAULTS["detector"]["token_logprob"]
    assert sec["enabled"] is False
    assert sec["min_logprob"] == -15.0
    assert sec["consecutive_hits"] == 3


def test_selected_token_logprobs_matches_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(4, 32)
    tids = torch.randint(0, 32, (4,))
    got = selected_token_logprobs(logits, tids)
    ref = torch.log_softmax(logits.float(), dim=-1).gather(1, tids.unsqueeze(-1)).squeeze(1)
    assert torch.allclose(got.float().cpu(), ref.cpu(), rtol=1e-4, atol=1e-4)


def test_token_logprob_detector_hits_on_low_logprob(tmp_path: Path):
    cfg = _cfg(tmp_path, min_logprob=-1.0, consecutive_hits=2, min_tokens=1)
    runner = SimpleNamespace(tp_rank=0)
    det = TokenLogprobDetector(runtime_config=cfg, runner=runner)

    # One-hot-ish high confidence row → high logprob → miss.
    logits_hi = torch.full((1, 8), -20.0)
    logits_hi[0, 3] = 5.0
    assert det.check_from_logits(logits=logits_hi, sampled_token_ids=[[3]], req_ids=["r0"]) == []

    # Uniform logits → selected logprob = -log(8) ≈ -2.07 < -1 → streak.
    logits_lo = torch.zeros(1, 8)
    assert det.check_from_logits(logits=logits_lo, sampled_token_ids=[[0]], req_ids=["r0"]) == []
    alerts = det.check_from_logits(logits=logits_lo, sampled_token_ids=[[1]], req_ids=["r0"])
    assert len(alerts) == 1
    assert alerts[0].incident_type == "token_logprob"
    assert alerts[0].detail["api_logprobs_required"] is False
    assert alerts[0].detail["violation"] == "low_selected_logprob"


def test_token_logprob_disabled_is_quiet(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg._data["detector"]["token_logprob"]["enabled"] = False
    cfg._invalidate_hot_path_gates()
    det = TokenLogprobDetector(runtime_config=cfg, runner=SimpleNamespace(tp_rank=0))
    det.refresh_from_config()
    logits = torch.zeros(1, 8)
    assert det.check_from_logits(logits=logits, sampled_token_ids=[[0]], req_ids=["r0"]) == []


def test_stash_and_take_pre_sample_logits():
    runner = SimpleNamespace()
    logits = torch.randn(2, 4)
    stash_pre_sample_logits(runner, logits)
    assert take_pre_sample_logits(runner) is logits
    assert take_pre_sample_logits(runner) is None


def test_need_pre_sample_hook_for_token_logprob():
    guard = MagicMock()
    guard.action_executor.can_run_detection.return_value = True
    guard.runtime_config.detector_get.side_effect = (
        lambda section, key, default=False: section == "token_logprob" and key == "enabled"
    )
    assert need_pre_sample_hook(guard) is True

    guard.runtime_config.detector_get.side_effect = lambda *_a, **_k: False
    assert need_pre_sample_hook(guard) is False
