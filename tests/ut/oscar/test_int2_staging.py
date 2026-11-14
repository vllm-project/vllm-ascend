# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the INT2 staging / chunked-prefill continuation mixin.

Pins the staging-plan gating (disabled -> none; memory budget bounds the slot
count) and the continuation-slot lookup contract (hit requires slot AND rid
match; lookup never clears state). Device-dependent arms are covered by the
device battery.
"""

import torch

from vllm_ascend.attention.int2.staging import Int2StagingMixin


class _StageSelf(Int2StagingMixin):
    def __init__(self, max_model_len=4096, nkv=8, hd=128, layers=4):
        self.vllm_config = type("C", (), {"model_config": type("M", (), {"max_model_len": max_model_len})()})()
        self.num_kv_heads = nkv
        self.head_size = hd
        self._oscar_R_k = torch.zeros(layers, hd, hd)
        self._TT_PREFIX = 128
        self._TT_RECENT = 896
        self._r218_st_slot = {}
        self._r218_st_rid = {}
        self._r218_st_len = {}
        self._r218_st_clock = {}
        self._r218_st_tick = [0]


class TestStagingPlan:
    def test_stage_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("R218_STAGE", "0")
        s = _StageSelf()
        assert s._r218_stage_assign(0, "r1") is None
        assert s._r218_stage_plan() == 0

    def test_budget_shrinks_slot_count(self, monkeypatch):
        monkeypatch.setenv("R218_STAGE", "1")
        monkeypatch.setenv("R218_STAGE_SLOTS", "8")
        monkeypatch.setenv("R218_STAGE_MAXGB", "0.001")  # impossible budget
        s = _StageSelf()
        assert s._r218_stage_plan() == 0
        monkeypatch.setenv("R218_STAGE_MAXGB", "1000")  # ample budget
        s2 = _StageSelf()
        assert s2._r218_stage_plan() > 0


class TestStagingLookup:
    def test_lookup_hit_and_miss(self, monkeypatch):
        monkeypatch.setenv("R218_STAGE", "1")
        monkeypatch.setenv("R218_STAGE_MAXGB", "1000")
        s = _StageSelf()
        s._r218_st_slot[3] = 1  # slot 3 -> stage 1
        s._r218_st_rid[1] = "rA"
        s._r218_st_len[1] = 512
        assert s._r218_stage_lookup(3, "rA") == (1, 512)
        assert s._r218_stage_lookup(3, "rB") == (None, 0)  # rid mismatch
        assert s._r218_stage_lookup(9, "rA") == (None, 0)  # slot unmapped

    def test_lookup_has_no_side_effects(self, monkeypatch):
        monkeypatch.setenv("R218_STAGE", "1")
        monkeypatch.setenv("R218_STAGE_MAXGB", "1000")
        s = _StageSelf()
        s._r218_st_slot[0] = 0
        s._r218_st_rid[0] = "rA"
        s._r218_st_len[0] = 64
        before = (dict(s._r218_st_slot), dict(s._r218_st_rid), dict(s._r218_st_len))
        s._r218_stage_lookup(0, "rA")
        s._r218_stage_lookup(0, "rZ")
        after = (dict(s._r218_st_slot), dict(s._r218_st_rid), dict(s._r218_st_len))
        assert before == after  # continuation lookup never clears state
