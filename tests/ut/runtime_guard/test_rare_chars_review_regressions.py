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
"""Regressions for rare_chars review findings 1–17."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Break import cycle: rare_chars → runtime_config.__init__ → catalog → rare_chars.
# Only needed when this module is the first rare_chars importer (e.g. --noconftest).
_CATALOG = "vllm_ascend.observability.runtime_config.detector_catalog"
if _CATALOG not in sys.modules:
    _stub = types.ModuleType(_CATALOG)
    _stub.DETECTOR_SECTIONS = ()
    _stub.REGISTERED_DETECTOR_TYPES = ()
    _stub.DETECTOR_SCHEMAS = ()
    _stub.build_detector_defaults = lambda: {}
    _stub.detector_param_keys = lambda: {}
    _stub.validate_registered_detectors = lambda *_a, **_k: None
    sys.modules[_CATALOG] = _stub

from vllm_ascend.observability.runtime_guard.detector.rare import (  # noqa: E402
    CONFIG_RARE,
    RareDetector,
    build_from_token_bytes,
    rare_native,
    rare_vocab,
    register_rare_table,
)
from vllm_ascend.observability.runtime_guard.detector.rare.rare_vocab import CI  # noqa: E402
from vllm_ascend.observability.runtime_guard.detector.rare_chars import RareCharsDetector  # noqa: E402
from vllm_ascend.observability.runtime_guard.processor import RuntimeGuardProcessor  # noqa: E402
from vllm_ascend.observability.runtime_guard.state import RequestGuardStore  # noqa: E402


def _require_native() -> None:
    rare_native._NATIVE_EXT = None
    rare_native._NATIVE_EXT_TRIED = False
    rare_native._MOD = None
    rare_native._MOD_TRIED = False
    if rare_native._load_native_ext() is None:
        pytest.skip("rare_feed_native not built")


def test_piece_to_bytes_sentencepiece_not_gpt2_alphabet():
    """#1: SP/Unigram pieces are UTF-8; ▁ is whitespace — not GPT-2 byte map."""
    b = rare_vocab._piece_to_bytes("\u2581m\u00e9lange", byte_level=False, bd={})
    assert b == " m\u00e9lange".encode()
    table = build_from_token_bytes([b], special_ids=set())
    assert bool(table["lead"][0]) is True
    assert bool(table["u8"][0]) is False


def test_build_from_tokenizer_captures_special_before_backend_swap():
    """#11: HF wrapper specials must survive backend_tokenizer swap."""

    class _Backend:
        def get_vocab_size(self):
            return 3

        def id_to_token(self, i):
            return ["a", "b", "<s>"][i]

    class _HF:
        all_special_ids = [2]
        added_tokens_decoder = {2: SimpleNamespace(special=True)}
        backend_tokenizer = _Backend()

    table = rare_vocab.build_from_tokenizer(_HF())
    assert int(table["rcls"][2]) == CI["special"]


def test_json_byte_level_detection():
    """#1: tokenizer.json ByteLevel pretokenizer opts into GPT-2 map."""
    meta = {"pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False}}
    assert rare_vocab._is_byte_level_bpe(tokenizer_json_meta=meta) is True
    assert rare_vocab._is_byte_level_bpe(tokenizer_json_meta={"pre_tokenizer": {"type": "Metaspace"}}) is False


def test_mask_chunked_prefill_zeros_num_sampled():
    """#8: num_sampled==0 rows become -1 so phantoms never enter rare_chars."""
    ids = torch.tensor([[7, 8], [9, 10]], dtype=torch.int64)
    ns = torch.tensor([2, 0], dtype=torch.int32)
    out = RuntimeGuardProcessor._mask_chunked_prefill_tokens(ids, ns)
    assert out.tolist() == [[7, 8], [-1, -1]]


def test_rare_cu_num_logits_prefers_device_when_np_stale():
    """#4: adaptive verification leaves cu_num_logits_np pre-compacted."""
    p = object.__new__(RuntimeGuardProcessor)
    p.runner = SimpleNamespace(input_batch=None)
    p._last_input_batch = SimpleNamespace(
        cu_num_logits_np=np.asarray([0, 10], dtype=np.int64),  # stale total
        cu_num_logits=np.asarray([0, 3], dtype=np.int64),  # true device layout
    )
    got = p._rare_cu_num_logits(num_logit_rows=3)
    assert list(got) == [0, 3]


def test_accept_rare_feed_rejects_reaped_and_force_cleared():
    """#6: late feed must not resurrect after reap / force-clear."""
    RequestGuardStore.reset_for_tests()
    try:
        store = RequestGuardStore.get()
        store.get_or_create("live")
        assert store.accept_rare_feed("live") is True
        assert store.accept_rare_feed("never-seen") is True

        store.get_or_create("cleared")
        store.mark_finished(["cleared"], wave=1)
        store.add_rare_feed_jobs(["cleared"])
        store.max_deferred_waves = 1
        assert store.list_reapable(current_wave=100) == ["cleared"]
        assert store.accept_rare_feed("cleared") is False

        store.clear("cleared")
        assert store.accept_rare_feed("cleared") is False  # reaped
    finally:
        RequestGuardStore.reset_for_tests()


def test_note_stream_gap_clears_decoder_hold():
    """#3: skipped feed resets UTF-8 decoder + look-ahead hold."""
    _require_native()
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig

    RequestGuardStore.reset_for_tests()
    try:
        cfg = RuntimeConfig(
            config_path=Path("/tmp/rg-rare-gap.json"),
            report_dir=Path("/tmp/rg-rare-gap-r"),
            ensure_file=False,
            hot_reload=False,
        )
        cfg._data.setdefault("detector", {})["rare_chars"] = {"enabled": True}
        cfg._invalidate_hot_path_gates()
        key = register_rare_table("ut_gap", build_from_token_bytes([b"a", b"\x80", b"hi"]))
        det = RareCharsDetector(runtime_config=cfg, runner=SimpleNamespace(tp_rank=0))
        det._corpus_key = key
        det._think_end_cfg = "none"
        det._think_end_id = None
        det._alert_monitor = False
        det._session.corpus_key = key
        det._session.mark_ready()
        det._ensure_engine_locked()
        # Seed a row with non-zero decoder / hold state.
        alerts = det.feed_precomputed(
            req_ids=["r0"],
            token_ids=[1],
            top1_logprobs=[0.0],
            chosen_logprobs=[0.0],
        )
        assert isinstance(alerts, list)
        row = det._row_by_req["r0"]
        batch = det._row_batch
        batch._dec_n[row] = 2
        batch._has_hold[row] = True
        batch._hold_id[row] = 7
        det.note_stream_gap(["r0"])
        assert int(batch._dec_n[row]) == 0
        assert bool(batch._has_hold[row]) is False
        assert int(batch._hold_id[row]) == -1
    finally:
        RequestGuardStore.reset_for_tests()


def test_mtp_duplicate_req_allocates_one_row():
    """#10: one feed with the same missing req_id twice → one native row."""
    _require_native()
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig

    RequestGuardStore.reset_for_tests()
    try:
        cfg = RuntimeConfig(
            config_path=Path("/tmp/rg-rare-mtp.json"),
            report_dir=Path("/tmp/rg-rare-mtp-r"),
            ensure_file=False,
            hot_reload=False,
        )
        cfg._data.setdefault("detector", {})["rare_chars"] = {"enabled": True}
        cfg._invalidate_hot_path_gates()
        key = register_rare_table("ut_mtp", build_from_token_bytes([b"a", b"b", b"c"]))
        det = RareCharsDetector(runtime_config=cfg, runner=SimpleNamespace(tp_rank=0, input_batch=None))
        det._corpus_key = key
        det._think_end_cfg = "none"
        det._think_end_id = None
        det._alert_monitor = False
        det._session.corpus_key = key
        det._session.mark_ready()
        det._ensure_engine_locked()
        store = RequestGuardStore.get()
        store.get_or_create("r")
        det.feed_precomputed(
            req_ids=["r", "r"],
            token_ids=[0, 1],
            top1_logprobs=[0.0, 0.0],
            chosen_logprobs=[0.0, 0.0],
        )
        assert list(det._row_by_req.keys()) == ["r"]
        assert len(det._states) == 1
    finally:
        RequestGuardStore.reset_for_tests()


def test_incident_tier_from_result_not_reason_substring():
    """#9: monitor tier comes from which Result fired, not 'gb>' in reason text."""
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig
    from vllm_ascend.observability.runtime_guard.detector.rare.result import Result

    cfg = RuntimeConfig(
        config_path=Path("/tmp/rg-rare-tier.json"),
        report_dir=Path("/tmp/rg-rare-tier-r"),
        ensure_file=False,
        hot_reload=False,
    )
    cfg._data.setdefault("detector", {})["rare_chars"] = {"enabled": True}
    det = RareCharsDetector(runtime_config=cfg, runner=SimpleNamespace(tp_rank=0))
    det._alert_monitor = True
    det._corpus_key = "ut"
    req_state = SimpleNamespace(alerted=False, last_reason="", tokens_seen=3, recent=[])
    strict = Result(is_ill=False)
    monitor = Result(is_ill=True, reason="rule5", fired_at=1)
    alert = det._incident_from_row(
        req_state,
        strict_result=strict,
        monitor_result=monitor,
        req_id="r",
        req_idx=0,
        token_id=1,
        top1=0.0,
        chosen=0.0,
        log_leader=False,
        source="ut",
    )
    assert alert is not None
    assert alert.detail["tier"] == "monitor"


def test_hot_enable_warmup_runs_outside_feed_lock(tmp_path: Path):
    """#12: first feed after hot-enable must not call _warmup_locked under lock."""
    from vllm_ascend.observability.runtime_config.config import RuntimeConfig

    cfg = RuntimeConfig(
        config_path=tmp_path / "c.json",
        report_dir=tmp_path / "r",
        ensure_file=True,
        hot_reload=False,
    )
    cfg._data.setdefault("detector", {})["rare_chars"] = {"enabled": True}
    cfg._invalidate_hot_path_gates()
    det = RareCharsDetector(runtime_config=cfg, runner=SimpleNamespace(tp_rank=0))
    held: dict[str, bool] = {}

    def _warmup(tokenizer=None):
        # feed_precomputed must call warmup() before taking self._lock.
        held["called"] = True
        det._session.mark_failed()  # fail closed for this UT (no tokenizer)
        raise RuntimeError("no tokenizer in ut")

    det.warmup = _warmup  # type: ignore[method-assign]
    out = det.feed_precomputed(req_ids=["r"], token_ids=[1], top1_logprobs=[0.0], chosen_logprobs=[0.0])
    assert out == []
    assert held.get("called") is True


def test_native_feed_rejects_non_fs_gate():
    """#14: NativeFeed must not silently diverge from gate='full'."""
    _require_native()
    key = register_rare_table("ut_gate", build_from_token_bytes([b"a", b"b"]))
    det = RareDetector(key, CONFIG_RARE, think_end=None)
    det.gate = "full"
    with pytest.raises(RuntimeError, match="gate='fs'"):
        rare_native.NativeFeed(det)


def test_feed_queue_start_sets_started_after_thread_start(monkeypatch):
    """#5: _started stays false if Thread.start raises."""
    from vllm_ascend.observability.runtime_guard.rare_logprob_feed import (
        RareLogprobFeedQueue,
        shutdown_rare_logprob_feed_queue,
    )

    shutdown_rare_logprob_feed_queue()
    q = RareLogprobFeedQueue()

    class BoomThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise RuntimeError("no threads")

    monkeypatch.setattr(
        "vllm_ascend.observability.runtime_guard.rare_logprob_feed.threading.Thread",
        BoomThread,
    )
    with pytest.raises(RuntimeError, match="no threads"):
        q.start()
    assert q._started is False


def test_compute_top1_has_triton_fail_closed(monkeypatch):
    """#17: without Triton, raise once — do not spam every step."""
    import importlib.util
    from pathlib import Path as _Path

    # Load the module by path to avoid heavy ops/device_config import chains.
    path = _Path("vllm_ascend/observability/runtime_guard/detector/rare/narrow_collect.py").resolve()
    # Minimal stubs so the module body can import.
    monkeypatch.setitem(
        sys.modules,
        "vllm.logger",
        SimpleNamespace(logger=SimpleNamespace(warning=lambda *a, **k: None)),
    )
    name = "ut_rare_narrow_collect_isolated"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    lp = importlib.util.module_from_spec(spec)
    # @triton.jit must be a no-op decorator when triton is stubbed.
    sys.modules[name] = lp

    def _jit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def deco(fn):
            return fn

        return deco

    triton_stub = SimpleNamespace(
        jit=_jit,
        cdiv=lambda a, b: (a + b - 1) // b,
        next_power_of_2=lambda n: 1 << (max(int(n) - 1, 0).bit_length()),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.triton_utils",
        SimpleNamespace(HAS_TRITON=False, tl=SimpleNamespace(), triton=triton_stub),
    )
    spec.loader.exec_module(lp)
    monkeypatch.setattr(lp, "HAS_TRITON", False)
    monkeypatch.setattr(lp, "_NARROW_COLLECT_UNAVAILABLE_LOGGED", False)
    with pytest.raises(RuntimeError, match="requires Triton"):
        lp.compute_top1_and_chosen_logprobs(torch.zeros(1, 4), torch.zeros(1, dtype=torch.int64))
    assert lp._NARROW_COLLECT_UNAVAILABLE_LOGGED is True
    with pytest.raises(RuntimeError, match="requires Triton"):
        lp.compute_top1_and_chosen_logprobs(torch.zeros(1, 4), torch.zeros(1, dtype=torch.int64))


def test_feed_submit_failure_unpins_and_notes_gap():
    """#5/#3: exception after pin must unpin and note stream gap."""
    RequestGuardStore.reset_for_tests()
    try:
        from vllm_ascend.observability.runtime_guard.rare_logprob_feed import RareLogprobBatch

        p = object.__new__(RuntimeGuardProcessor)
        p.runner = SimpleNamespace()
        p.detectors = MagicMock()
        noted: list[list[str]] = []
        p.detectors.get.return_value = SimpleNamespace(note_stream_gap=lambda ids: noted.append(list(ids)))
        p._reap_finished_requests = lambda: None  # type: ignore[method-assign]
        store = RequestGuardStore.get()
        store.get_or_create("a")

        ready_event = MagicMock()
        batch = RareLogprobBatch(device_logprobs=object(), ready_event=ready_event)
        output = SimpleNamespace(req_ids=["a"], sampled_token_ids=None)

        # Force the queue path to raise after pins are taken.
        import vllm_ascend.observability.runtime_guard.rare_logprob_feed as feed_mod

        real_get = feed_mod.get_rare_logprob_feed_queue

        class BoomQ:
            def submit(self, job):
                raise RuntimeError("submit boom")

        feed_mod.get_rare_logprob_feed_queue = lambda: BoomQ()  # type: ignore
        try:
            p.feed_rare_logprobs(output, batch=batch)
        finally:
            feed_mod.get_rare_logprob_feed_queue = real_get  # type: ignore

        assert store.get_state("a").rare_feed_jobs == 0
        assert noted and noted[0] == ["a"]
        ready_event.synchronize.assert_called()
    finally:
        RequestGuardStore.reset_for_tests()
