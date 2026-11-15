# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Named, test-only RPC probes for real sleep-backend regression evidence."""

import time
from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.device_allocator.camem import CaMemAllocator


class SleepBackendProbe:
    if TYPE_CHECKING:
        model_runner: Any
        vllm_config: Any
        sleep_mode_backend: Any
        _sleep_saved_parameters: dict[str, torch.Tensor]
        _sleep_saved_buffers: dict[str, torch.Tensor]
        rank: int

        def synchronize_device(self) -> None: ...

        def execute_model(self, scheduler_output: Any) -> Any: ...

        def sleep(self, level: int = 1) -> None: ...

    def sleep_backend_install_probe(self):
        self._sleep_backend_capture_calls = 0
        original = self.model_runner.capture_model

        def capture(*args, **kwargs):
            result = original(*args, **kwargs)
            self._sleep_backend_capture_calls += 1
            return result

        self.model_runner.capture_model = capture

    def sleep_backend_snapshot(self):
        self.synchronize_device()
        allocations = []
        for pointer, data in CaMemAllocator.get_instance().pointer_to_data.items():
            allocations.append(
                {
                    "pointer": pointer,
                    "tag": data.tag,
                    "bytes": data.handle[1],
                    "asleep": data.is_asleep,
                    "cpu_backup_bytes": 0 if data.cpu_backup_tensor is None else data.cpu_backup_tensor.nbytes,
                }
            )
        return {
            "rank": self.rank,
            "backend": type(self.sleep_mode_backend).__name__,
            "runner": type(self.model_runner).__module__ + "." + type(self.model_runner).__qualname__,
            "state": self.sleep_mode_backend.state(),
            "capture_calls": self._sleep_backend_capture_calls,
            "graph_mode": str(self.vllm_config.compilation_config.cudagraph_mode),
            "capture_sizes": self.vllm_config.compilation_config.cudagraph_capture_sizes,
            "use_aclgraph": self.model_runner.use_aclgraph,
            "module_implementations": sorted(
                {
                    type(impl).__module__ + "." + type(impl).__qualname__
                    for module in self.model_runner.model.modules()
                    if (impl := getattr(module, "impl", None)) is not None
                }
            ),
            "free_bytes": torch.npu.mem_get_info()[0],
            "saved_parameter_bytes": sum(value.nbytes for value in self._sleep_saved_parameters.values()),
            "saved_buffer_bytes": sum(value.nbytes for value in self._sleep_saved_buffers.values()),
            "allocations": allocations,
        }

    def sleep_backend_reject_execution(self):
        try:
            self.execute_model(None)
        except RuntimeError as error:
            assert "sleep backend" in str(error), str(error)
            return {"rank": self.rank, "rejected": True}
        raise AssertionError("Sleeping backend accepted execution")

    def sleep_backend_inject_backup_failure(self):
        if self.rank == 0:
            runtime = self.sleep_mode_backend.runtime
            original = runtime.backup

            def fail_once(level):
                runtime.backup = original
                raise RuntimeError("injected rank-zero backup failure")

            runtime.backup = fail_once

    def sleep_backend_check_backup_rollback(self):
        try:
            self.sleep(level=2)
        except RuntimeError as error:
            assert self.sleep_mode_backend.state() == "RUNNING"
            return {"rank": self.rank, "rolled_back": True, "error": str(error)}
        raise AssertionError("Injected backup failure was not observed")


def shutdown_sleep_backend_engine(llm):
    core = llm.llm_engine.engine_core
    processes = list(core.resources.engine_manager.processes)
    core.shutdown(timeout=45)
    exitcodes = [process.exitcode for process in processes]
    assert processes, "Expected spawned EngineCore processes"
    assert all(code == 0 for code in exitcodes), exitcodes
    return exitcodes


def check_sleep_backup_rollback(llm):
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    prompt = "Explain expert parallelism."
    before = list(llm.generate(prompt, sampling, use_tqdm=False)[0].outputs[0].token_ids)
    llm.collective_rpc("sleep_backend_inject_backup_failure")
    statuses = llm.collective_rpc("sleep_backend_check_backup_rollback")
    after = list(llm.generate(prompt, sampling, use_tqdm=False)[0].outputs[0].token_ids)
    assert before == after
    return {"ranks": statuses, "before_tokens": before, "after_tokens": after}


def check_sleep_request_boundary(llm):
    """Pause a real unfinished request, enqueue another while asleep, then drain."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    engine = llm.llm_engine
    prompt = "Explain how a computer stores information."

    def sampling(tokens):
        return SamplingParams(
            temperature=0, max_tokens=tokens, ignore_eos=True, output_kind=RequestOutputKind.CUMULATIVE
        )

    expected = {}
    for name, count in (("backend-inflight", 256), ("backend-queued", 8)):
        expected[name] = list(llm.generate(prompt, sampling(count), use_tqdm=False)[0].outputs[0].token_ids)
    engine.add_request("backend-inflight", prompt, sampling(256))
    first: list[Any] = []
    deadline = time.monotonic() + 120
    while not first:
        assert time.monotonic() < deadline
        first = [
            output
            for output in engine.step()
            if output.request_id == "backend-inflight" and output.outputs[0].token_ids
        ]
    assert not first[-1].finished
    llm.sleep(level=1, mode="wait")
    asleep = llm.collective_rpc("sleep_backend_snapshot")
    engine.add_request("backend-queued", prompt, sampling(8))
    llm.collective_rpc("sleep_backend_reject_execution")
    llm.wake_up()
    finished = {}
    deadline = time.monotonic() + 120
    while engine.has_unfinished_requests():
        assert time.monotonic() < deadline
        for output in engine.step():
            if output.finished:
                finished[output.request_id] = list(output.outputs[0].token_ids)
    assert finished == expected
    return {"asleep": asleep, "finished_tokens": finished}


def run_sleep_backend_case(llm, *, cycles=10, eager=False, new_backend=True):
    """Exercise both levels, every requested batch size and real weight reload.

    Caller owns the fresh engine and its shutdown. Returned data is raw per-rank
    evidence; this function does not infer process shutdown or CI success.
    """
    from vllm import SamplingParams

    params = SamplingParams(temperature=0, seed=42, max_tokens=8, ignore_eos=True)
    prompts = ["Explain expert parallelism in one sentence."] * 8

    def generate(size):
        return [list(output.outputs[0].token_ids) for output in llm.generate(prompts[:size], params, use_tqdm=False)]

    def snapshot():
        return llm.collective_rpc("sleep_backend_snapshot")

    llm.collective_rpc("sleep_backend_install_probe")
    sizes = (1, 2, 4, 8)
    golden = {size: generate(size) for size in sizes}
    rows = []
    for level in (1, 2):
        for cycle in range(cycles):
            before = snapshot()
            started = time.perf_counter()
            llm.sleep(level=level)
            sleep_seconds = time.perf_counter() - started
            asleep = snapshot()
            for status in asleep:
                assert status["state"] == "SUSPENDED"
                assert status["allocations"]
                assert all(item["asleep"] for item in status["allocations"])
            if new_backend:
                llm.collective_rpc("sleep_backend_reject_execution")
            started = time.perf_counter()
            llm.wake_up(tags=["weights"])
            weights_wake_seconds = time.perf_counter() - started
            weights = snapshot()
            if new_backend:
                assert all(status["state"] == "RESUMING" for status in weights)
                llm.collective_rpc("sleep_backend_reject_execution")
            if not eager:
                assert [s["capture_calls"] for s in weights] == [s["capture_calls"] for s in before]
            started = time.perf_counter()
            if level == 2:
                llm.collective_rpc("reload_weights")
            reload_seconds = time.perf_counter() - started
            started = time.perf_counter()
            llm.wake_up(tags=["kv_cache"])
            kv_wake_seconds = time.perf_counter() - started
            restored = snapshot()
            for old, new in zip(before, restored):
                assert new["state"] == "RUNNING"
                assert all(not item["asleep"] for item in new["allocations"])
                for tag in ("weights", "kv_cache"):
                    assert {a["pointer"] for a in old["allocations"] if a["tag"] == tag} == {
                        a["pointer"] for a in new["allocations"] if a["tag"] == tag
                    }
                if not eager:
                    assert new["capture_calls"] > old["capture_calls"]
            started = time.perf_counter()
            first = generate(1)
            first_seconds = time.perf_counter() - started
            assert first == golden[1]
            for size in sizes:
                assert generate(size) == golden[size], (level, cycle, size)
            rows.append(
                {
                    "level": level,
                    "cycle": cycle,
                    "sleep_seconds": sleep_seconds,
                    "weights_wake_seconds": weights_wake_seconds,
                    "reload_seconds": reload_seconds,
                    "kv_wake_seconds": kv_wake_seconds,
                    "wake_seconds": weights_wake_seconds + kv_wake_seconds,
                    "first_request_seconds": first_seconds,
                    "before": before,
                    "asleep": asleep,
                    "weights_only": weights,
                    "restored": restored,
                }
            )
    return {"cycles_per_level": cycles, "batch_sizes": sizes, "golden_tokens": golden, "cycles": rows}
