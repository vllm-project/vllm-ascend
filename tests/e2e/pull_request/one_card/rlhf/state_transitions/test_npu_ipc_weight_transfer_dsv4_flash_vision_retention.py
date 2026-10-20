# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-NPU integration of Worker retention with DeepSeek V4 vision reload."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from vllm.config import ModelConfig
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.one_card.rlhf.state_transitions.test_npu_ipc_weight_transfer_dsv4_flash_vision import (
    DSV4_VISION_CASE,
    GPU_MEMORY_UTILIZATION,
    INFERENCE_DEVICE_INDEX,
    MODEL,
    _post,
    _ReducedCheckpointSource,
)
from tests.e2e.pull_request.rlhf import weight_transfer_test_utils as utils
from tests.e2e.pull_request.rlhf.deepseek_v4_retention_probe import PATTERNS, is_tracked, variant_tensor
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo


class RetentionSource(_ReducedCheckpointSource):
    def __init__(self, case, device, variant=False):
        super().__init__(case, device)
        self.variant = variant
        # HF meta enumeration omits these real served/checkpoint parameters.
        self._parameters += [(f"model.layers.{i}.mlp.gate.bias_vl", (8,), torch.float32) for i in range(4)]
        self._parameters += [("model.layers.3.mlp.gate.e_score_correction_bias", (8,), torch.float32)]
        assert len({n for n, _, _ in self._parameters}) == len(self._parameters)

    def _make_tensor(self, name, shape, dtype=torch.bfloat16):
        tensor = super()._make_tensor(name, shape, dtype)
        return variant_tensor(tensor) if self.variant and is_tracked(name) else tensor


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
        NPUIPCTrainerInitInfo(rank=0, packed=packed, packed_buffer_size_bytes=utils.packed_buffer_size_for(source)),
        client=HTTPVLLMWeightSyncClient(base_url=server.url_root),
        source=source,
    )


@pytest.mark.skipif(
    "sleep_preserve_parameter_names" not in ModelConfig.__dataclass_fields__,
    reason="This vLLM version does not expose sleep parameter retention.",
)
@pytest.mark.skipif(torch.npu.device_count() < 1, reason="Requires at least one NPU.")
@pytest.mark.e2e_model(MODEL)
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_retention_with_vision_reload(packed: bool, tmp_path: Path):
    """Frozen parameters retain the latest values while all other weights reload."""
    torch.npu.set_device(INFERENCE_DEVICE_INDEX)
    device = torch.device("npu", INFERENCE_DEVICE_INDEX)
    case = replace(DSV4_VISION_CASE, id=DSV4_VISION_CASE.id + "-retention-all-four")
    source = RetentionSource(case, device)
    variant_case = replace(case, id=case.id + "-latest-values")
    changed_source = RetentionSource(variant_case, device, variant=True)
    env = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "ASCEND_RT_VISIBLE_DEVICES": str(INFERENCE_DEVICE_INDEX),
    }
    references = [
        utils.reference_signature(
            s,
            c,
            port=get_open_port(),
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=1,
            device_index=INFERENCE_DEVICE_INDEX,
            env_dict=env,
        )
        for s, c in ((source, case), (changed_source, variant_case))
    ]
    partial = copy.copy(source)
    partial._parameters = [(n, s, d) for n, s, d in source._parameters if not is_tracked(n)]
    omitted = [n for n, _, _ in source._parameters if is_tracked(n)]
    assert len(omitted) == 12
    utils.register_engines_once()
    utils.wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, GPU_MEMORY_UTILIZATION)
    port = get_open_port()
    args = utils.reference_serve_args(case, port=port, gpu_memory_utilization=GPU_MEMORY_UTILIZATION)
    args += [
        "--weight-transfer-config",
        '{"backend": "npu_ipc"}',
        "--worker-extension-cls",
        "tests.e2e.pull_request.rlhf.deepseek_v4_retention_probe.RetentionProbe",
        "--sleep-preserve-parameter-names",
        *PATTERNS,
    ]
    evidence = {"packed": packed, "omitted_checkpoint_names": omitted, "cycles": []}
    with (
        utils.fixed_startup_checkpoint(source) as checkpoint,
        RemoteOpenAIServer(
            checkpoint,
            vllm_serve_args=args,
            server_host="127.0.0.1",
            server_port=port,
            env_dict=env,
            auto_port=False,
        ) as server,
    ):
        client = server.get_client()
        engine = trainer(server, partial, packed)
        initial = rpc(server, "wsc_install")
        assert len(initial["parameters"]) == 12
        assert not initial["old_snapshot_present"]
        assert utils.generation_signature(client, case.model, image_prompt=True) == references[0]
        for cycle in range(2):
            expected = initial if cycle == 0 else rpc(server, "wsc_variant")
            if cycle:
                assert all(expected["parameters"][n]["sha256"] != p["sha256"] for n, p in initial["parameters"].items())
                assert all(expected["parameters"][n]["ptr"] == p["ptr"] for n, p in initial["parameters"].items())
            assert utils.generation_signature(client, case.model, image_prompt=True) == references[cycle]
            _post(server, "pause")
            _post(server, "sleep", params={"level": 2})
            saved = rpc(server, "wsc_state")
            assert saved["saved"] == {n: p["sha256"] for n, p in expected["parameters"].items()}
            poisoned = rpc(server, "wsc_poison")
            assert poisoned["saved"] == saved["saved"]
            assert all(poisoned["parameters"][n]["sha256"] != p["sha256"] for n, p in expected["parameters"].items())
            _post(server, "wake_up", params={"tags": ["weights"]})
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
            hook_events = rpc(server, "wsc_hook_events")[-4:]
            assert [e["stage"] for e in hook_events] == [
                "start_weight_update:hook-before",
                "start_weight_update:hook-after",
                "finish_weight_update:hook-before",
                "finish_weight_update:hook-after",
            ]
            for event in hook_events:
                same_parameters(event["state"], expected)
            torch.npu.empty_cache()
            _post(server, "wake_up", params={"tags": ["kv_cache"]})
            same_parameters(rpc(server, "wsc_state"), expected)
            actual = utils.generation_signature(client, case.model, image_prompt=True)
            assert actual == references[cycle]
            evidence["cycles"].append(
                {
                    "cycle": cycle,
                    "saved": saved,
                    "poisoned": poisoned,
                    "restored": restored,
                    "events": events,
                    "hook_events": hook_events,
                    "signature": actual,
                    "reference": references[cycle],
                }
            )

        # Full-update control: unfreeze the parameters and load different values.
        rpc(server, "wsc_configure", "full")
        full_engine = trainer(server, source, packed)
        _post(server, "pause")
        _post(server, "sleep", params={"level": 2})
        assert rpc(server, "wsc_state")["saved"] == {}
        _post(server, "wake_up", params={"tags": ["weights"]})
        full_engine.send_weights()
        torch.npu.empty_cache()
        _post(server, "wake_up", params={"tags": ["kv_cache"]})
        full_state = rpc(server, "wsc_state")
        same_parameters(full_state, initial)
        assert utils.generation_signature(client, case.model, image_prompt=True) == references[0]
        evidence["full_reload"] = full_state

        # Level-1 control: normal offload/wake, no level-2 parameter backup.
        rpc(server, "wsc_configure", "retain")
        _post(server, "pause")
        _post(server, "sleep", params={"level": 1})
        assert rpc(server, "wsc_state")["saved"] == {}
        _post(server, "wake_up", params={"tags": ["weights"]})
        _post(server, "wake_up", params={"tags": ["kv_cache"]})
        same_parameters(rpc(server, "wsc_state"), initial)
        assert utils.generation_signature(client, case.model, image_prompt=True) == references[0]
        evidence["level_one"] = "passed"
        evidence["all_events"] = rpc(server, "wsc_events")
        evidence["all_hook_events"] = rpc(server, "wsc_hook_events")
    mode = json.loads(utils.FULL_DECODE_ONLY_CONFIG)["cudagraph_mode"]
    path = tmp_path / f"evidence-{mode}-{'packed' if packed else 'unpacked'}.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"RETENTION_EVIDENCE={path}")
