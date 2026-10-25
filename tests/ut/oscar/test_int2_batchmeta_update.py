# SPDX-License-Identifier: Apache-2.0
"""Tests for the INT2 batch-metadata update mixin (module level).

The heavy fills (rowslot migration, flat refresh, mdfill / update_meta) are
device-bound; their behavioral evidence is the device battery and the serve
parity legs. At module level we pin the structural contract (entry-point
presence, static-method calling convention, explicit imports) and the R196
idfill memo identity semantics, which are CPU-exercisable.
"""

import inspect
from types import SimpleNamespace
from typing import Any

from vllm_ascend.attention.int2.batchmeta_update import Int2BatchMetaUpdateMixin


def _clean_env(monkeypatch):
    for var in (
        "R164_BATCHED_STAGE",
        "R196_EAGER_BATCH",
        "R203_METAFIX",
        "R166_SLOTALGO",
        "R167_IDFILL",
        "R170_ROWSLOT",
        "R144_TT_DYN_DBG",
    ):
        monkeypatch.delenv(var, raising=False)


class _MetaSelf(Int2BatchMetaUpdateMixin):
    # class-level attrs the verbatim bodies expect on the backend class
    _r164_slotmap: Any = None
    _r164_slotctr: Any = None
    _r196_idfill_state: Any = None
    _r196_idfill_state2: Any = None


def _md(rids, finished=(), preempted=(), seq_lens=None):
    return SimpleNamespace(
        request_ids=list(rids),
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        seq_lens_list=list(seq_lens or [1] * len(rids)),
    )


def test_update_entry_points_and_static_convention():
    for m in (
        "_r170_rowslot_migrate",
        "_r172_wrapper_flatrefresh",
        "_r170_mdfill",
        "_r164_update_meta",
        "_r196_idfill_memo",
    ):
        assert callable(getattr(Int2BatchMetaUpdateMixin, m, None)), m
    # the fills are convention-static (no decorator): the first parameter is
    # not self, callers go through the class
    for m in ("_r170_rowslot_migrate", "_r172_wrapper_flatrefresh", "_r170_mdfill", "_r164_update_meta"):
        params = list(inspect.signature(getattr(Int2BatchMetaUpdateMixin, m)).parameters)
        assert params[0] != "self", f"{m} gained a self binding"


def test_explicit_import_blocks():
    import vllm_ascend.attention.int2.batchmeta_update as mod

    src = inspect.getsource(mod)
    assert "from vllm_ascend.attention.int2.ctx import (" in src
    assert "from vllm_ascend.attention.int2.helpers import (" in src


class TestR196IdfillMemo:
    def test_disabled_returns_false(self, monkeypatch):
        _clean_env(monkeypatch)
        s = _MetaSelf()
        assert s._r196_idfill_memo(_md(["a"])) is False

    def test_identity_is_object_address_strong_ref(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("R196_EAGER_BATCH", "1")
        s = _MetaSelf()
        md1 = _md(["a", "b"], seq_lens=[3, 4])
        md2 = _md(["a", "b"], seq_lens=[3, 4])  # equal CONTENT, other object
        assert s._r196_idfill_memo(md1) is False  # nothing memoized yet
        assert s._r196_idfill_memo(md1, set_key=True) is False
        assert s._r196_idfill_memo(md1) is True  # hit: same object + same key
        assert s._r196_idfill_memo(md2) is False  # equal content misses

    def test_content_key_change_misses(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("R196_EAGER_BATCH", "1")
        s = _MetaSelf()
        md = _md(["a", "b"], seq_lens=[3, 4])
        s._r196_idfill_memo(md, set_key=True)
        md.seq_lens_list = [3, 5]  # derived key changes
        assert s._r196_idfill_memo(md) is False

    def test_multi_entry_list_bounded_under_metafix(self, monkeypatch):
        _clean_env(monkeypatch)
        monkeypatch.setenv("R196_EAGER_BATCH", "1")
        monkeypatch.setenv("R203_METAFIX", "1")
        s = _MetaSelf()
        mds = [_md([f"r{i}"], seq_lens=[i]) for i in range(10)]
        for md in mds:
            assert s._r196_idfill_memo(md) is False
            s._r196_idfill_memo(md, set_key=True)
        assert len(_MetaSelf._r196_idfill_state2) <= 8  # bounded list
        assert s._r196_idfill_memo(mds[9]) is True  # newest kept
        assert s._r196_idfill_memo(mds[0]) is False  # oldest evicted
        _MetaSelf._r196_idfill_state2 = None
