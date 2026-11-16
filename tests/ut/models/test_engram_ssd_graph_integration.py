# SPDX-License-Identifier: Apache-2.0
"""T4 图模式专项(model_state 层, CPU): SSD 路由契约(04 §13 T4)。

用与 tests/ut/worker/v2/test_engram_state_hooks.py 相同的 stub 手法, 在
model_state 层断言:
  - FULL 路径图外准备: prepare_engram 在 replay 前被调用且只调一次;
  - capture(dummy)路径: 不触发 SSD prepare(无 IO/缓存变更), 固定缓冲绑定;
  - 失败步: finish_execution(failed=True) 传导到 runtime(SSD 分支);
  - eager 路径: prepare_engram_inputs 收到本步坐标。
真实 aclgraph replay 的新输入/输出复用验证归实机阶段(test_full_replay_changes_
with_new_hashes 等在 NPU 窗口执行)。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch

import vllm_ascend.ops  # noqa: F401,E402
from vllm_ascend.models.deepseek_v41.engram import model_state as ms_mod  # noqa: E402


def _ssd_model(rec: dict):
    """带 SSD runtime 假体的 V4.1 模型 stub; 调用全部记录到 rec。"""
    model = SimpleNamespace()
    runtime = SimpleNamespace()
    runtime.prepare = Mock(
        side_effect=lambda hashes, mask, **kw: rec.setdefault("prepare_calls", []).append(
            (int(hashes.shape[0]), kw.get("step_id", -1))
        )
        or SimpleNamespace(layer_outputs={1: torch.zeros(1), 14: torch.zeros(1)}, stats={"hits": 0, "misses": 0})
    )
    runtime.finish_execution = Mock(side_effect=lambda failed=False: rec.setdefault("finish_calls", []).append(failed))
    model.engram_ssd_runtime = runtime

    def fake_prepare_engram(
        input_ids,
        positions,
        lookback_token_ids=None,
        query_start_loc=None,
        slot_mapping=None,
        block_table=None,
        *,
        token_indices=None,
        force_dummy=False,
        output_buffers=None,
        ready_events=None,
        output_tokens=None,
        mask_output_buffer=None,
        mask_ready_event=None,
        valid_token_count=None,
    ):
        rec.setdefault("prepare_engram_calls", []).append(
            dict(force_dummy=force_dummy, output_tokens=output_tokens, has_qsl=query_start_loc is not None)
        )
        lookups = {1: torch.zeros(2, 8, 256, dtype=torch.bfloat16), 14: torch.zeros(2, 8, 256, dtype=torch.bfloat16)}
        mask = torch.ones(2, dtype=torch.bool)
        return lookups, mask

    model.prepare_engram = fake_prepare_engram

    def fake_prepare_engram_inputs(**kw):
        rec.setdefault("model_prepare_inputs", []).append({k: getattr(v, "shape", None) for k, v in kw.items()})
        return {
            "engram_lookups": torch.zeros(2, 8, 256, dtype=torch.bfloat16),
            "engram_mask": torch.zeros(2, dtype=torch.bool),
        }

    model.prepare_engram_inputs = Mock(side_effect=fake_prepare_engram_inputs)
    model.prepare_engram_graph_inputs = Mock(
        return_value={
            "engram_lookups": torch.zeros(2, 8, 256, dtype=torch.bfloat16),
            "engram_mask": torch.zeros(2, dtype=torch.bool),
        }
    )

    def _retire(reset_events=False):
        rec.setdefault("retire_calls", []).append(reset_events)
        # 镜像真实 model.py 接线: retire → runtime.finish_execution(failed)
        model.engram_ssd_runtime.finish_execution(failed=reset_events)

    model.retire_engram_lookups = Mock(side_effect=_retire)
    model.supports_engram_graph_producer = False
    return model


def _ssd_state(model):
    """构造 EngramModelState(绕开重型 __init__, 只装 engram 相关字段)。"""
    state = ms_mod.EngramModelState.__new__(ms_mod.EngramModelState)
    state.model = model
    state.lookback_token_ids = torch.zeros(4, 8, dtype=torch.int32)
    state._engram_graph_inputs = None  # producer 禁用(SSD)
    state._engram_inputs = None
    state._cg_mode = None
    state.kvpp_is_dummy_run = False
    # T0 探针(默认关): 只需存在
    probe = SimpleNamespace(cfg=SimpleNamespace(enabled=False), step=Mock())
    state._engram_dp_probe = probe
    state._probe_stage = {}
    state.pcp_manager = None
    state.max_num_reqs = 4
    state._replay_groups = None
    return state


def _batch(num_tokens=8, num_reqs=2, dummy=False):
    return SimpleNamespace(
        input_ids=torch.arange(num_tokens, dtype=torch.int32),
        positions=torch.arange(num_tokens, dtype=torch.int64),
        num_tokens_after_padding=num_tokens,
        num_reqs=num_reqs,
        is_dummy=dummy,
        query_start_loc=None if dummy else torch.tensor([0, 4, 8][: num_reqs + 1], dtype=torch.int32),
        num_tokens=num_tokens,
    )


def test_full_replay_prehook_calls_prepare_once():
    """FULL replay 前: _engram_inputs 存在时 prepare_engram 恰好一次。"""
    rec: dict = {}
    model = _ssd_model(rec)
    state = _ssd_state(model)
    state._engram_inputs = dict(
        input_ids=torch.arange(8, dtype=torch.int32),
        positions=torch.arange(8, dtype=torch.int64),
        padded_tokens=8,
        lookback_token_ids=state.lookback_token_ids,
        query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        cg_mode="FULL",
        force_dummy=False,
        token_indices=None,
    )
    state.prepare_engram()
    # model 级分发(prepare_engram_inputs) 恰好一次, 携带本步坐标
    assert len(rec.get("model_prepare_inputs", [])) == 1
    assert state._engram_inputs is None  # 消费后清空, 不重复执行


def test_capture_dummy_has_no_ssd_prepare():
    """capture/dummy: prepare_engram_dummy_inputs 只绑固定缓冲, 不触发 SSD。"""
    rec: dict = {}
    model = _ssd_model(rec)
    model.prime_engram_v2_graph_inputs = Mock(
        return_value={
            "engram_lookups": torch.zeros(2, 8, 256, dtype=torch.bfloat16),
            "engram_mask": torch.zeros(2, dtype=torch.bool),
        }
    )
    state = _ssd_state(model)
    state.prepare_engram_dummy_inputs(num_reqs=2, num_tokens=8)
    assert "prepare_engram_calls" not in rec  # SSD prepare 未被调用
    assert model.prime_engram_v2_graph_inputs.called
    assert state._engram_inputs is None


def test_failed_step_propagates_to_runtime():
    """失败步: finish_execution(failed=True) 经 retire 传到 runtime。"""
    rec: dict = {}
    model = _ssd_model(rec)
    state = _ssd_state(model)
    state.finish_execution(failed=True)
    assert rec.get("finish_calls") == [True]
    assert rec.get("retire_calls") == [True]


def test_normal_step_releases_lease():
    """正常步: finish_execution(failed=False) 释放租约。"""
    rec: dict = {}
    model = _ssd_model(rec)
    state = _ssd_state(model)
    state.finish_execution(failed=False)
    assert rec.get("finish_calls") == [False]
    assert rec.get("retire_calls") == [False]


def test_dummy_routing_keeps_empty_hashes_path():
    """dummy 批(dummy=True): staging 标记 dummy, prepare 走空 hash 分支。"""
    rec: dict = {}
    model = _ssd_model(rec)
    state = _ssd_state(model)
    batch = _batch(dummy=True)
    # 走真实 prepare_engram_inputs(不依赖 req_states 的 device 路径):
    # dummy → 不采集 lookback, _engram_inputs.query_start_loc=None
    try:
        state.prepare_engram_inputs(batch, req_states=None)
    except Exception:
        # req_states=None 在非 dummy 下才需要; dummy 分支应不触碰
        raise AssertionError("dummy 分支不应触碰 req_states")
    staged = state._engram_inputs
    assert staged is not None and staged["force_dummy"] is True
    assert staged["query_start_loc"] is None
