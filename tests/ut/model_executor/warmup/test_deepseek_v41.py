# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression tests for eager attention compilation before EP collectives."""

import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier, Event
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ascend_forward_context import get_mrv2_in_profile_run
from vllm_ascend.model_executor.warmup import deepseek_v41 as warmup


class ExpertLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.on_forward = Mock()

    def forward(self, value):
        self.on_forward()
        return value + 1


class AttentionModel(torch.nn.Module):
    """A lazy attention compiler followed by a real nn.Module hook boundary."""

    def __init__(self):
        super().__init__()
        self.experts = ExpertLayer()
        self.compiled = set()
        self.compile = Mock()

    def forward(self, path):
        if path not in self.compiled:
            self.compile(path)
            self.compiled.add(path)
        return self.experts(torch.ones(1))


def make_worker(*, query_len=6, draft=True):
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True, hf_config=SimpleNamespace(model_type="deepseek_v41")),
        parallel_config=SimpleNamespace(data_parallel_size=8, enable_expert_parallel=True),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=5120),
    )
    runner = SimpleNamespace(
        model=AttentionModel(),
        uniform_decode_query_len=query_len,
        speculator=SimpleNamespace(model=AttentionModel()) if draft else None,
    )

    def dummy_run(num_tokens, *, uniform_decode=False, skip_attn=False, skip_gdn_state_update=False):
        assert not skip_attn, "Memory profiling does not compile attention"
        assert skip_gdn_state_update, "Warmup must not advance persistent ring state"
        if uniform_decode:
            assert num_tokens == getattr(runner, "uniform_decode_query_len", 1)
        path = "idle" if uniform_decode else "prefill"
        runner.model(path)
        if runner.speculator is not None:
            runner.speculator.model(path)

    runner._dummy_run = Mock(side_effect=dummy_run)
    return SimpleNamespace(vllm_config=config, model_runner=runner)


@pytest.fixture(autouse=True)
def cpu_dependencies(monkeypatch):
    model_module = ModuleType("vllm_ascend.models.deepseek_v41.model")
    model_module.DeepseekV41MoE = ExpertLayer
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    monkeypatch.setattr(warmup, "get_ep_group", Mock(return_value=SimpleNamespace(barrier=Mock())))
    monkeypatch.setattr(warmup, "set_current_vllm_config", lambda config: nullcontext())
    monkeypatch.setattr(
        warmup, "get_current_hardware_profile", Mock(return_value=SimpleNamespace(supports=lambda _: True))
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=Mock()), raising=False)


@pytest.mark.parametrize("query_len,draft", [(1, False), (6, True)])
def test_prefill_and_idle_cached_for_target_and_draft(query_len, draft):
    worker = make_worker(query_len=query_len, draft=draft)
    runner = worker.model_runner
    warmup.warmup_deepseek_v41_eager(worker)

    models = [runner.model] + ([runner.speculator.model] if draft else [])
    for model in models:
        assert model.compiled == {"prefill", "idle"}
        assert not model.experts._forward_pre_hooks
        model.compile.reset_mock()
        # A rank can now enter either path alone without a cold compilation or
        # a warmup-only collective left on the serving path.
        for path in ("prefill", "idle"):
            torch.testing.assert_close(model(path), torch.tensor([2.0]))
        model.compile.assert_not_called()


@pytest.mark.parametrize("phase", ["prefill", "idle"])
@pytest.mark.parametrize("draft", [False, True])
def test_slow_attention_compilation_cannot_enter_experts_early(monkeypatch, phase, draft):
    workers = [make_worker(draft=draft), make_worker(draft=draft)]
    models = [worker.model_runner.speculator.model if draft else worker.model_runner.model for worker in workers]
    slow_compiling = Event()
    release_compiler = Event()
    peer_waiting = Event()
    group_barrier = Barrier(2, timeout=10)

    def compile_slow(path):
        if path == phase:
            slow_compiling.set()
            assert release_compiler.wait(10)

    def barrier():
        if phase in models[0].compiled and phase not in models[1].compiled:
            peer_waiting.set()
        group_barrier.wait()

    def check_compilation_finished():
        # Model the device collective that cannot tolerate a peer still
        # compiling the attention operation that precedes it.
        if phase in models[0].compiled:
            assert phase in models[1].compiled, "Peer entered MoE while attention was still compiling"

    models[1].compile.side_effect = compile_slow
    models[0].experts.on_forward.side_effect = check_compilation_finished
    monkeypatch.setattr(warmup, "get_ep_group", lambda: SimpleNamespace(barrier=barrier))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(warmup.warmup_deepseek_v41_eager, worker) for worker in workers]
        try:
            assert slow_compiling.wait(10)
            assert peer_waiting.wait(10)
            assert not futures[0].done()
            assert phase not in models[1].compiled
        finally:
            release_compiler.set()
        for future in futures:
            future.result(timeout=10)
    assert all(model.compiled == {"prefill", "idle"} for model in models)


@pytest.mark.parametrize("where", ["compile", "expert", "barrier"])
def test_failure_is_propagated_and_temporary_hooks_removed(where):
    worker = make_worker()
    models = [worker.model_runner.model, worker.model_runner.speculator.model]
    if where == "compile":
        models[1].compile.side_effect = RuntimeError("warmup failed")
    elif where == "expert":
        models[0].experts.on_forward.side_effect = RuntimeError("warmup failed")
    else:
        warmup.get_ep_group.return_value.barrier.side_effect = RuntimeError("warmup failed")
    with pytest.raises(RuntimeError, match="warmup failed"):
        warmup.warmup_deepseek_v41_eager(worker)
    assert all(not model.experts._forward_pre_hooks for model in models)


@pytest.mark.parametrize("disabled", ["eager", "dp", "ep", "model", "hardware"])
def test_unaffected_configurations_do_not_execute_or_synchronize(disabled, monkeypatch):
    worker = make_worker()
    config = worker.vllm_config
    if disabled == "eager":
        config.model_config.enforce_eager = False
    elif disabled == "dp":
        config.parallel_config.data_parallel_size = 1
    elif disabled == "ep":
        config.parallel_config.enable_expert_parallel = False
    elif disabled == "model":
        config.model_config.hf_config.model_type = "other_model"
    else:
        monkeypatch.setattr(warmup, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False))
    warmup.warmup_deepseek_v41_eager(worker)
    worker.model_runner._dummy_run.assert_not_called()
    warmup.get_ep_group.assert_not_called()
    torch.npu.synchronize.assert_not_called()


def test_shared_target_and_draft_module_gets_one_hook():
    worker = make_worker()
    runner = worker.model_runner
    runner.speculator.model = runner.model
    seen = []
    runner.model.experts.on_forward.side_effect = lambda: seen.append(len(runner.model.experts._forward_pre_hooks))
    warmup.warmup_deepseek_v41_eager(worker)
    assert seen and set(seen) == {1}
    assert not runner.model.experts._forward_pre_hooks


@pytest.mark.parametrize("fail", [False, True])
def test_balanced_routing_is_scoped_to_large_pass(fail):
    worker = make_worker()
    states = []

    def dummy_run(*args, **kwargs):
        states.append(get_mrv2_in_profile_run())
        if fail:
            raise RuntimeError("warmup failed")

    worker.model_runner._dummy_run.side_effect = dummy_run
    assert not get_mrv2_in_profile_run()
    if fail:
        with pytest.raises(RuntimeError, match="warmup failed"):
            warmup.warmup_deepseek_v41_eager(worker)
    else:
        warmup.warmup_deepseek_v41_eager(worker)
    assert states == ([True] if fail else [True, False])
    assert not get_mrv2_in_profile_run()
