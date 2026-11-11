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
"""UT: rare_chars Guard wiring + warmup table build."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.observability.runtime_config.config import RuntimeConfig
from vllm_ascend.observability.runtime_guard.detector.rare import (
    CONFIG_RARE,
    RareDetector,
    build_from_token_bytes,
    rare_native,
    register_rare_table,
)
from vllm_ascend.observability.runtime_guard.detector.rare_chars import (
    RareCharsDetector,
    device_chosen_token_ids,
)
from vllm_ascend.observability.runtime_guard.hooks import need_pre_sample_hook
from vllm_ascend.observability.runtime_guard.state import RequestGuardStore


def _require_native() -> None:
    rare_native._MOD = None
    rare_native._MOD_TRIED = False
    if rare_native._module() is None:
        pytest.skip("rare_feed_native not built")


def _synthetic_table() -> str:
    """Tiny vocab with one incomplete UTF-8 token (id 1)."""
    chunks = [b"a", b"\x80", b"hi", b" "]
    table = build_from_token_bytes(chunks, special_ids={3})
    return register_rare_table("ut_rare", table)


def _top1_chosen(logits: torch.Tensor, tids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp.max(dim=-1).values, lp.gather(1, tids.long().unsqueeze(-1)).squeeze(1)


def _cfg(tmp_path: Path, **rare_kw) -> RuntimeConfig:
    RequestGuardStore.reset_for_tests()
    cfg = RuntimeConfig(
        config_path=tmp_path / "c.json",
        report_dir=tmp_path / "r",
        ensure_file=True,
        hot_reload=False,
    )
    sec = cfg._data["detector"]["rare_chars"]
    sec["enabled"] = True
    sec.update(rare_kw)
    cfg._invalidate_hot_path_gates()
    return cfg


def test_catalog_includes_rare_chars_defaults():
    from vllm_ascend.observability.runtime_config._defaults import _DEFAULTS
    from vllm_ascend.observability.runtime_config.detector_catalog import DETECTOR_SECTIONS

    assert "rare_chars" in DETECTOR_SECTIONS
    sec = _DEFAULTS["detector"]["rare_chars"]
    assert sec == {"enabled": False}


def test_narrow_collect_formula_matches_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(4, 32)
    tids = torch.randint(0, 32, (4,))
    top1, chosen = _top1_chosen(logits, tids)
    max_val = logits.float().max(dim=-1).values
    lse = torch.log(torch.sum(torch.exp(logits.float() - max_val.unsqueeze(-1)), dim=-1))
    assert torch.allclose(top1, -lse, rtol=1e-5, atol=1e-5)
    ref_chosen = logits.float().gather(1, tids.unsqueeze(-1)).squeeze(1) - max_val - lse
    assert torch.allclose(chosen, ref_chosen, rtol=1e-5, atol=1e-5)


def test_rare_chars_feed_precomputed_bad_utf8(tmp_path: Path):
    _require_native()
    key = _synthetic_table()
    tid = 1  # lone continuation byte
    V = 4
    logits = torch.full((1, V), -20.0)
    logits[0, tid] = 5.0
    top1, chosen = _top1_chosen(logits, torch.tensor([tid]))

    cfg = _cfg(tmp_path)
    det = RareCharsDetector(runtime_config=cfg, runner=SimpleNamespace(tp_rank=0))
    # Point corpus at the registered synthetic table without a real tokenizer.
    det._corpus_key = key
    det._think_end_cfg = "none"
    det._think_end_id = None
    det._alert_monitor = False
    det._warmed = True
    det._ensure_engine_locked()

    t1 = float(top1.reshape(-1)[0])
    ch = float(chosen.reshape(-1)[0])
    alerts = det.feed_precomputed(req_ids=["r1"], token_ids=[tid], top1_logprobs=[t1], chosen_logprobs=[ch])
    alerts2 = det.feed_precomputed(req_ids=["r1"], token_ids=[tid], top1_logprobs=[t1], chosen_logprobs=[ch])
    finish = det.finish_request("r1")
    fired = alerts + alerts2 + ([finish] if finish is not None else [])
    assert any(a.incident_type == "rare_chars" for a in fired)


def test_rare_detector_bad_utf8_rule_fires():
    key = _synthetic_table()
    det = RareDetector(key, CONFIG_RARE, think_end=None)
    st = det.start()
    tid = 1
    ids = np.array([tid, tid], np.int64)
    lp = np.zeros(2, np.float64)
    res = det.feed(st, ids, lp, lp)
    res = res or det.finish(st)
    assert res is not None and res.is_ill
    assert "bad_utf8" in res.reason


def test_need_pre_sample_hook_for_rare_chars(tmp_path: Path):
    cfg = _cfg(tmp_path)
    guard = SimpleNamespace(
        runtime_config=cfg,
        action_executor=SimpleNamespace(can_run_detection=lambda: True),
        runner=SimpleNamespace(),
    )
    assert need_pre_sample_hook(guard) is True
    cfg._data["detector"]["rare_chars"]["enabled"] = False
    cfg._invalidate_hot_path_gates()
    assert need_pre_sample_hook(guard) is False


def test_device_chosen_token_ids_stay_on_device():
    ids = torch.tensor([[7, -1], [-1, -1], [3, 9]], dtype=torch.int32)
    chosen = device_chosen_token_ids(ids)
    assert chosen.device == ids.device
    assert chosen.tolist() == [7, -1, 9]


def test_rare_chars_disabled_is_quiet(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cfg._data["detector"]["rare_chars"]["enabled"] = False
    cfg._invalidate_hot_path_gates()
    det = RareCharsDetector(runtime_config=cfg, runner=SimpleNamespace(tp_rank=0))
    assert det.feed_precomputed(req_ids=["r"], token_ids=[1], top1_logprobs=[0.0], chosen_logprobs=[0.0]) == []
