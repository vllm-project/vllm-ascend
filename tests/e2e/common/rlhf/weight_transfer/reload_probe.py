# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test-only observations of Direct runtime storage and V2 graph reuse."""

import torch

from tests.e2e.common.rlhf.common import post_control


class ReloadProbeWorkerExtension:
    """Attach through --worker-extension-cls only in Direct matrix tests."""

    def direct_sleep_allocations(self) -> list[dict]:
        from vllm_ascend.device_allocator.camem import CaMemAllocator

        # Inspect allocator metadata without touching unmapped model storage.
        return [
            {
                "tag": data.tag,
                "bytes": data.handle[1],
                "asleep": data.is_asleep,
                "offloaded": data.cpu_backup_tensor is not None,
            }
            for data in CaMemAllocator.get_instance().pointer_to_data.values()
        ]

    def direct_reload_configuration(self) -> dict:
        from vllm.model_executor.model_loader.reload.direct import direct_reload_enabled

        model = self.model_runner.model
        return {
            "enabled": direct_reload_enabled(),
            "checkpoint_tensor_count": len(getattr(model, "_ckpt_param_names", ())),
        }

    def direct_reload_snapshot(self) -> dict:
        runner = self.model_runner
        manager = runner.cudagraph_manager
        if not hasattr(self, "_direct_probe_counts"):
            self._direct_probe_counts = {"captures": 0, "replays": 0}
            if manager is not None:
                capture, replay = manager.capture, manager.run_fullgraph

                def counted_capture(*args, **kwargs):
                    self._direct_probe_counts["captures"] += 1
                    return capture(*args, **kwargs)

                def counted_replay(*args, **kwargs):
                    result = replay(*args, **kwargs)
                    self._direct_probe_counts["replays"] += 1
                    return result

                manager.capture = counted_capture
                manager.run_fullgraph = counted_replay

        tensors = {}
        lists = {}

        def record(name, tensor):
            if isinstance(tensor, torch.Tensor):
                tensors[name] = [
                    id(tensor),
                    tensor.data_ptr(),
                    tensor.untyped_storage().data_ptr(),
                    list(tensor.shape),
                    list(tensor.stride()),
                ]

        for module_name, module in runner.model.named_modules():
            for name, tensor in [*module._parameters.items(), *module._buffers.items()]:
                record(f"{module_name}.{name}", tensor)
            for name, value in vars(module).items():
                if name.endswith("_weight_list") and isinstance(value, list):
                    lists[f"{module_name}.{name}"] = id(value)
                    for index, tensor in enumerate(value):
                        record(f"{module_name}.{name}[{index}]", tensor)
            for suffix, owner in (("", module), (".impl", getattr(module, "impl", None))):
                for name, tensor in getattr(owner, "_reload_runtime_buffers", {}).items():
                    record(f"{module_name}{suffix}.runtime.{name}", tensor)
        return {
            "tensors": tensors,
            "lists": lists,
            "manager": id(manager),
            "graphs": {} if manager is None else {repr(desc): id(graph) for desc, graph in manager.graphs.items()},
            **self._direct_probe_counts,
        }


def assert_sleep_allocations(server, *, sleep_level: int, asleep_tags: tuple[str, ...]) -> None:
    states = post_control(server, "collective_rpc", json={"method": "direct_sleep_allocations"}).json()["results"]
    assert states, "no worker returned sleep allocator observations"
    for rank, allocations in enumerate(states):
        for tag in ("weights", "kv_cache"):
            pool = [allocation for allocation in allocations if allocation["tag"] == tag]
            assert pool and sum(allocation["bytes"] for allocation in pool) > 0, f"rank {rank}: empty {tag} pool"
            asleep = tag in asleep_tags
            offloaded = asleep and tag == "weights" and sleep_level == 1
            assert all(allocation["asleep"] == asleep for allocation in pool), f"rank {rank}: {tag} sleep mismatch"
            assert all(allocation["offloaded"] == offloaded for allocation in pool), (
                f"rank {rank}: {tag} offload policy mismatch for sleep{sleep_level}"
            )
    print(f"[sleep-proof] level={sleep_level} asleep_tags={asleep_tags} workers={states}", flush=True)


def reload_snapshots(server, *, execution_mode: str) -> list[dict]:
    configurations = post_control(server, "collective_rpc", json={"method": "direct_reload_configuration"}).json()[
        "results"
    ]
    assert configurations, "no worker returned its Direct reload configuration"
    for rank, configuration in enumerate(configurations):
        assert configuration["enabled"], f"rank {rank}: Direct reload is disabled in the worker"
        assert configuration["checkpoint_tensor_count"] > 0, f"rank {rank}: no checkpoint layout was recorded"
    print(f"[direct-proof] workers={configurations}", flush=True)
    states = post_control(server, "collective_rpc", json={"method": "direct_reload_snapshot"}).json()["results"]
    assert states, "no worker returned Direct reload observations"
    for rank, state in enumerate(states):
        assert state["tensors"], f"rank {rank}: no model tensors observed"
        if execution_mode == "graph":
            assert state["graphs"], f"rank {rank}: graph mode was requested but no graph was captured"
    return states


def assert_reload_reuses_runtime(before: list[dict], after: list[dict], *, execution_mode: str) -> None:
    assert len(before) == len(after), "worker count changed during the update"
    for rank, (previous, current) in enumerate(zip(before, after, strict=True)):
        for key in ("tensors", "lists", "manager", "graphs"):
            assert current[key] == previous[key], f"rank {rank}: Direct reload replaced runtime {key}"
        assert current["captures"] == previous["captures"], f"rank {rank}: the update recaptured graphs"
        if execution_mode == "graph":
            assert current["replays"] > previous["replays"], (
                f"rank {rank}: generation did not replay the original graph"
            )
