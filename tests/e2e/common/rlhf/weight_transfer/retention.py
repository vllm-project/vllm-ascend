# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-card vision retention checked against independent startup references."""

import copy
import json
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.rlhf.common import (
    post_control as _post,
)
from tests.e2e.common.rlhf.common import (
    register_engines_once,
    rlhf_server,
    server_visible_devices,
    wait_for_free_device_memory,
)
from tests.e2e.common.rlhf.weight_transfer import reference_check as oracle
from tests.e2e.common.rlhf.weight_transfer.retention_probe import PATTERNS, is_tracked, variant_tensor
from tests.e2e.common.rlhf.weight_transfer.weight_source import (
    FixedRandomWeightSource,
    fixed_startup_checkpoint,
    packed_buffer_size_for,
)
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo


class RetentionSource(FixedRandomWeightSource):
    def __init__(self, case, device, variant=False):
        super().__init__(case, device)
        self.variant = variant

    def _make_tensor(self, name, shape, dtype=torch.bfloat16):
        tensor = super()._make_tensor(name, shape, dtype)
        return variant_tensor(tensor, self._num_experts) if self.variant and is_tracked(name) else tensor


def rpc(server, method, *args):
    data = _post(
        server, "collective_rpc", json={"method": method, "args": list(args), "timeout": 120}, timeout=150
    ).json()
    assert len(data["results"]) == 1
    return data["results"][0]


def same_parameters(state, expected):
    assert state["parameters"] == expected["parameters"]
    assert not state["old_snapshot_present"]


def trainer(server, source, packed):
    return WeightTransferTrainerFactory.trainer_init(
        NPUIPCTrainerInitInfo(rank=0, packed=packed, packed_buffer_size_bytes=packed_buffer_size_for(source)),
        client=HTTPVLLMWeightSyncClient(base_url=server.url_root),
        source=source,
    )


def run_vision_retention(case, packed: bool, tmp_path: Path, *, device_index: int, gpu_memory_utilization: float):
    """Frozen parameters retain the latest values while all other weights reload."""
    torch.npu.set_device(device_index)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    device = torch.device("npu", device_index)
    case = replace(case, id=case.id + "-retention-all-four")
    source = RetentionSource(case, device)
    variant_case = replace(case, id=case.id + "-latest-values")
    changed_source = RetentionSource(variant_case, device, variant=True)
    env = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "ASCEND_RT_VISIBLE_DEVICES": server_visible_devices(device_index, 1),
    }
    references = [
        oracle.reference_signature(
            s,
            c,
            port=get_open_port(),
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            device_index=device_index,
            env_dict=env,
        )
        for s, c in ((source, case), (changed_source, variant_case))
    ]
    partial = copy.copy(source)
    partial._parameters = [(n, s, d) for n, s, d in source._parameters if not is_tracked(n)]
    omitted = [n for n, _, _ in source._parameters if is_tracked(n)]
    expected_retained = 3 * case.hf_overrides["num_hidden_layers"]
    assert len(omitted) == expected_retained
    register_engines_once()
    wait_for_free_device_memory(device_index, gpu_memory_utilization)
    port = get_open_port()
    args = oracle.reference_serve_args(case, port=port, gpu_memory_utilization=gpu_memory_utilization)
    args += [
        "--weight-transfer-config",
        '{"backend": "npu_ipc"}',
        "--worker-extension-cls",
        "tests.e2e.common.rlhf.weight_transfer.retention_probe.RetentionProbe",
        "--sleep-preserve-parameter-names",
        *PATTERNS,
    ]
    evidence: dict[str, Any] = {"packed": packed, "omitted_checkpoint_names": omitted, "cycles": []}
    with (
        ExitStack() as cleanup,
        fixed_startup_checkpoint(source) as checkpoint,
        rlhf_server(
            checkpoint,
            vllm_serve_args=args,
            server_host="127.0.0.1",
            server_port=port,
            env_dict=env,
        ) as server,
    ):
        client = server.get_client()
        engine = trainer(server, partial, packed)
        cleanup.callback(engine.shutdown)
        initial = rpc(server, "wsc_install")
        assert len(initial["parameters"]) == expected_retained
        assert not initial["old_snapshot_present"]
        assert oracle.generation_signature(client, case.model, image_prompt=True) == references[0]
        for cycle in range(2):
            expected = initial if cycle == 0 else rpc(server, "wsc_variant")
            if cycle:
                assert all(expected["parameters"][n]["sha256"] != p["sha256"] for n, p in initial["parameters"].items())
                assert all(expected["parameters"][n]["ptr"] == p["ptr"] for n, p in initial["parameters"].items())
            assert oracle.generation_signature(client, case.model, image_prompt=True) == references[cycle]
            _post(server, "pause")
            _post(server, "sleep", params={"level": 2})
            saved = rpc(server, "wsc_saved_state")
            assert saved["saved"] == {n: p["sha256"] for n, p in expected["parameters"].items()}
            assert rpc(server, "wsc_poison")["saved"] == saved["saved"]
            _post(server, "wake_up", params={"tags": ["weights"]})
            poisoned = rpc(server, "wsc_poisoned_state")
            assert poisoned["saved"] == saved["saved"]
            assert all(poisoned["parameters"][n]["sha256"] != p["sha256"] for n, p in expected["parameters"].items())
            restored = rpc(server, "wsc_state")
            same_parameters(restored, expected)
            assert restored["saved"] == {}
            engine.send_weights()
            finalized = rpc(server, "wsc_state")
            same_parameters(finalized, expected)
            events = rpc(server, "wsc_events")[-4:]
            assert [e["stage"] for e in events] == [
                "start_weight_update:before",
                "start_weight_update:after",
                "finish_weight_update:before",
                "finish_weight_update:after",
            ]
            for event in events:
                same_parameters(event["state"], expected)
            finalize_events = rpc(server, "wsc_finalize_events")[-2:]
            assert [e["stage"] for e in finalize_events] == [
                "finalize:before",
                "finalize:after",
            ]
            for event in finalize_events:
                same_parameters(event["state"], expected)
            torch.npu.empty_cache()
            _post(server, "wake_up", params={"tags": ["kv_cache"]})
            same_parameters(rpc(server, "wsc_state"), expected)
            actual = oracle.generation_signature(client, case.model, image_prompt=True)
            assert actual == references[cycle]
            evidence["cycles"].append(
                {
                    "cycle": cycle,
                    "saved": saved,
                    "poisoned": poisoned,
                    "restored": restored,
                    "events": events,
                    "finalize_events": finalize_events,
                    "signature": actual,
                    "reference": references[cycle],
                }
            )

        # Preservation is not a load filter: supplied new values must win.
        full_engine = trainer(server, source, packed)
        cleanup.callback(full_engine.shutdown)
        _post(server, "pause")
        _post(server, "sleep", params={"level": 2})
        assert len(rpc(server, "wsc_saved_state")["saved"]) == expected_retained
        _post(server, "wake_up", params={"tags": ["weights"]})
        same_parameters(rpc(server, "wsc_state"), expected)
        full_engine.send_weights()
        torch.npu.empty_cache()
        _post(server, "wake_up", params={"tags": ["kv_cache"]})
        supplied_state = rpc(server, "wsc_state")
        same_parameters(supplied_state, initial)
        assert oracle.generation_signature(client, case.model, image_prompt=True) == references[0]
        evidence["supplied_retained_reload"] = supplied_state

        # Full-update control: unfreeze the parameters and load different values.
        rpc(server, "wsc_configure", "full")
        full_engine = trainer(server, source, packed)
        cleanup.callback(full_engine.shutdown)
        _post(server, "pause")
        _post(server, "sleep", params={"level": 2})
        assert rpc(server, "wsc_saved_state")["saved"] == {}
        _post(server, "wake_up", params={"tags": ["weights"]})
        full_engine.send_weights()
        torch.npu.empty_cache()
        _post(server, "wake_up", params={"tags": ["kv_cache"]})
        full_state = rpc(server, "wsc_state")
        same_parameters(full_state, initial)
        assert oracle.generation_signature(client, case.model, image_prompt=True) == references[0]
        evidence["full_reload"] = full_state

        # Level-1 control: normal offload/wake, no level-2 parameter backup.
        rpc(server, "wsc_configure", "retain")
        _post(server, "pause")
        _post(server, "sleep", params={"level": 1})
        assert rpc(server, "wsc_saved_state")["saved"] == {}
        _post(server, "wake_up", params={"tags": ["weights"]})
        _post(server, "wake_up", params={"tags": ["kv_cache"]})
        same_parameters(rpc(server, "wsc_state"), initial)
        assert oracle.generation_signature(client, case.model, image_prompt=True) == references[0]
        evidence["level_one"] = "passed"
        evidence["all_events"] = rpc(server, "wsc_events")
        evidence["all_finalize_events"] = rpc(server, "wsc_finalize_events")
    mode = json.loads(case.compilation_config)["cudagraph_mode"]
    path = tmp_path / f"evidence-{mode}-{'packed' if packed else 'unpacked'}.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"RETENTION_EVIDENCE={path}")
