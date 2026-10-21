# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""One-card DeepSeek-V4 Flash Vision weight update and sleep retention.

Full updates and frozen-subset updates are checked against independent normal
startup loads, using text and image logprobs. The default reduction is 4 text
layers, 2 vision layers and 8 experts; VLLM_RL_DSV4_* knobs select the checkpoint,
reduction, memory budget and decoder/encoder graph modes. Packed and unpacked IPC
both exercise two level-2 sleep/update/wake rounds. The retention test also checks
changed current values, real Worker CPU snapshots, allocator remapping, poisoning
before parameter restoration, upstream finalize, full reload and level-1 sleep.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import requests
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.config import ModelConfig
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf import weight_transfer_test_utils as utils
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    FixedRandomWeightSource,
    WeightUpdateModelCase,
    assert_weight_update_matches_reference,
    deepseek_v4_checkpoint_name,
    deepseek_v4_vision_params,
    generation_signature,
    live_update_serve_args,
    packed_buffer_size_for,
    reference_signature,
    register_engines_once,
    wait_for_free_device_memory,
)
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

if TYPE_CHECKING:
    from vllm.config import ModelConfig

    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCWeightTransferEngine
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2

CONTROL_TIMEOUT = 60

# --------------------------------------------------------------------------
# Reduction, read once so the reference and the lane share it by construction.
# --------------------------------------------------------------------------
MODEL = os.environ.get("VLLM_RL_DSV4_MODEL", "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp")
TEXT_LAYERS = int(os.environ.get("VLLM_RL_DSV4_TEXT_LAYERS", "4"))
VISION_LAYERS = int(os.environ.get("VLLM_RL_DSV4_VISION_LAYERS", "2"))
EXPERTS = int(os.environ.get("VLLM_RL_DSV4_EXPERTS", "8"))
HASH_LAYERS = os.environ.get("VLLM_RL_DSV4_HASH_LAYERS")

# The trainer is co-located with the server on the same chip, so both models have
# to fit; keep the budget low enough to leave room for the payload.
GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_RL_DSV4_MEM_UTIL", "0.45"))
INFERENCE_DEVICE_INDEX = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))
# Bisection knob: run the transfer without the level-2 sleep/wake cycle, so a
# passing oracle separates "the sleep/wake (CAMEM re-map) path" from "the
# layerwise reload path". Not a supported mode: the weights pool and the KV
# cache stay mapped and ``send_weights`` runs against the live server.
SKIP_SLEEP = os.environ.get("VLLM_RL_DSV4_SKIP_SLEEP") == "1"

# Second bisection knob: the lane and its reference share one graph mode by
# construction (``_common_serve_args`` reads a single module constant), so
# overriding that constant keeps the oracle valid while switching the decode
# path between a captured ACL graph and eager execution. ``NONE`` / ``PIECEWISE``
# / ``FULL_DECODE_ONLY`` are the meaningful values.
_CUDAGRAPH_MODE = os.environ.get("VLLM_RL_DSV4_CUDAGRAPH_MODE")
# ``cudagraph_mm_encoder`` is a separate flag: the vision encoder keeps its own
# captured graph even when the decoder is eager, so a run that only sets
# ``cudagraph_mode=NONE`` still replays a captured encoder.
_MM_ENCODER_GRAPH = os.environ.get("VLLM_RL_DSV4_MM_ENCODER_GRAPH")
if _CUDAGRAPH_MODE or _MM_ENCODER_GRAPH is not None:
    import json as _json

    from tests.e2e.pull_request.rlhf import weight_transfer_test_utils as _wtu

    _graph_cfg: dict = {"cudagraph_mode": _CUDAGRAPH_MODE or "FULL_DECODE_ONLY"}
    if _MM_ENCODER_GRAPH is not None:
        _graph_cfg["cudagraph_mm_encoder"] = _MM_ENCODER_GRAPH == "1"
    _wtu.FULL_DECODE_ONLY_CONFIG = _json.dumps(_graph_cfg)

# Third bisection knob: a startup ``load_weights`` invalidates vLLM's
# multimodal caches (``GPUModelRunner.load_weights`` ends with
# ``reset_encoder_cache()`` + ``reset_mm_cache()``), while the live update path
# does not. Any image encoded before the update therefore keeps its old
# embeddings, which shows up as the image entry diverging from its first token.
# Set to ``1`` to POST the dev reset routes after every update.
RESET_MM_CACHE = os.environ.get("VLLM_RL_DSV4_RESET_MM_CACHE") == "1"

# Fourth bisection knob: the lane generates the dummy signature *before* the
# update, so its image request is the second one this process ever serves while
# the reference's is the first. Set to ``1`` to keep the image out of the dummy
# pass, which separates "state the pre-update image request left behind" from
# "the update itself" as the cause of an image-only divergence.
DUMMY_TEXT_ONLY = os.environ.get("VLLM_RL_DSV4_DUMMY_TEXT_ONLY") == "1"

assert TEXT_LAYERS >= 1, "VLLM_RL_DSV4_TEXT_LAYERS must be >= 1"
assert VISION_LAYERS >= 0, "VLLM_RL_DSV4_VISION_LAYERS must be >= 0"
assert EXPERTS >= 6, "n_routed_experts must stay >= num_experts_per_tok (6), or top-k routing breaks"

_HF_OVERRIDES: dict = {
    "num_hidden_layers": TEXT_LAYERS,
    "n_routed_experts": EXPERTS,
    # >0 routes the config to the multimodal wrapper
    # (see DeepseekV4ModelArchConfigConvertor); the tower itself is reduced to
    # fit the single-chip IPC budget next to the text backbone.
    "vision_n_layers": VISION_LAYERS,
    # The published checkpoint is FP8, while this lane compares a BF16 payload
    # against a BF16 startup load. Both sides of the oracle share this override,
    # so quantization is off on both of them.
    "quantization_config": None,
}
if HASH_LAYERS:
    _HF_OVERRIDES["num_hash_layers"] = int(HASH_LAYERS)

# Keep the model-supported block size explicit: upstream automatic selection
# otherwise combines the compressed backends into a 4096-token allocator block.
_EXTRA_SERVER_ARGS = ["--tokenizer-mode", "deepseek_v4", "--block-size", "32"]
if VISION_LAYERS > 0:
    _EXTRA_SERVER_ARGS += ["--limit-mm-per-prompt", '{"image": 1}']

# The case id enters the shared reference-signature cache key, so it carries the
# reduction: a run with different knobs cannot reuse another run's reference.
DSV4_VISION_CASE = WeightUpdateModelCase(
    id=f"dsv4-vision-{TEXT_LAYERS}L-v{VISION_LAYERS}-e{EXPERTS}",
    model=MODEL,
    hf_overrides=_HF_OVERRIDES,
    # Checkpoint-facing text layout of DeepSeek-V4 (fused experts, compressor
    # under the indexer, w1/w3/w2 expert shards).
    checkpoint_name_map=deepseek_v4_checkpoint_name,
    # The HF ``from_config`` model sizes every expert MLP by
    # ``intermediate_size`` (18432), while the checkpoint - and therefore the
    # served model - use ``moe_intermediate_size`` (2048).
    expert_intermediate_size=2048,
    # The vision tower lives in the multimodal wrapper only, so no HF meta model
    # can enumerate it; the builder derives it from the config instead. A
    # text-only run (VISION_LAYERS=0) has no tower to send.
    extra_param_builder=deepseek_v4_vision_params if VISION_LAYERS > 0 else None,
    extra_server_args=tuple(_EXTRA_SERVER_ARGS),
    image_prompt=VISION_LAYERS > 0,
)


class _ReducedCheckpointSource(FixedRandomWeightSource):
    """The shared generator, able to write a reference checkpoint from a local dir.

    ``FixedRandomWeightSource.write_checkpoint`` fetches ``config.json`` through
    ``hf_hub_download``, which only accepts a Hub repo id. A depth-reduced
    checkpoint built on disk (``VLLM_RL_DSV4_MODEL=/path/to/reduced-vision``) is a
    first-class input for this lane, so a local ``config.json`` is copied from the
    directory instead; everything else - the generated names, shapes, values and
    integer tables - is byte-for-byte the shared code path.
    """

    def write_checkpoint(self, directory: str | Path) -> None:
        model_dir = Path(self._case.model)
        if not model_dir.is_dir():
            super().write_checkpoint(directory)
            return

        from safetensors.torch import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_dir / "config.json", target / "config.json")
        state = {name: tensor.detach().to("cpu").contiguous() for name, tensor in self}
        save_file(state, str(target / "model.safetensors"), metadata={"format": "pt"})


def _post(server: RemoteOpenAIServer, route: str, *, json=None, params=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, params=params, timeout=timeout)
    response.raise_for_status()
    return response


@pytest.mark.e2e_model(MODEL)
@pytest.mark.skipif(
    torch.npu.device_count() < 1,
    reason="NPU IPC weight transfer e2e test requires at least 1 NPU.",
)
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_npu_ipc_weight_transfer_dsv4_vision(packed: bool):
    case = DSV4_VISION_CASE
    torch.npu.set_device(INFERENCE_DEVICE_INDEX)
    source = _ReducedCheckpointSource(case, torch.device("npu", INFERENCE_DEVICE_INDEX))
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "ASCEND_RT_VISIBLE_DEVICES": str(INFERENCE_DEVICE_INDEX),
    }
    # Independent oracle first: a server that loads exactly this payload at
    # startup, on its own lifecycle, before the live path touches anything.
    reference = reference_signature(
        source,
        case,
        port=get_open_port(),
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=1,
        device_index=INFERENCE_DEVICE_INDEX,
        env_dict=env_dict,
    )

    # Dummy sanity, first live update and the reload regression share one lane.
    # Its pre-update signature is the dummy-sanity check; the *reference* is the
    # part that must come from a separate server. The reference and the lane share
    # one card, so wait for its process tree to hand the HBM back.
    wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, GPU_MEMORY_UTILIZATION)
    port = get_open_port()
    with RemoteOpenAIServer(
        case.model,
        vllm_serve_args=live_update_serve_args(
            case,
            backend="npu_ipc",
            port=port,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        ),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
    ) as server:
        client = server.get_client()
        dummy_signature = generation_signature(
            client,
            case.model,
            image_prompt=case.image_prompt and not DUMMY_TEXT_ONLY,
        )

        register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            NPUIPCTrainerInitInfo(
                rank=0,
                packed=packed,
                packed_buffer_size_bytes=packed_buffer_size_for(source),
            ),
            client=HTTPVLLMWeightSyncClient(base_url=server.url_root),
            source=source,
        )

        signatures = []
        for _ in range(2):
            if not SKIP_SLEEP:
                _post(server, "pause")
                # Same-chip RL: hand the engine's HBM to the co-located trainer
                # before generating the payload. Level 2 discards the weights and
                # the KV cache outright; `/sleep` reads its level from the query
                # string, a JSON body is ignored.
                _post(server, "sleep", params={"level": 2})
                # Level 2 unmapped the weights pool and `send_weights`' layerwise
                # reload copies the payload back into exactly those buffers, so
                # the weights allocation has to be mapped again before the
                # transaction. The KV cache stays asleep until it is finished.
                _post(server, "wake_up", params={"tags": ["weights"]})
            # send_weights owns the complete START -> LOAD -> FINISH transaction.
            engine.send_weights()
            # The trainer is co-located with the server on the same chip, so the
            # source's materialised temporaries stay in the NPU caching allocator
            # across rounds and starve the kv_cache wake that follows
            # (aclrtMallocPhysical OOM, 207001).  Hand them back between rounds.
            torch.npu.empty_cache()
            # The KV cache is re-allocated last, from the updated weights' state.
            # That wake leaves nothing asleep, so the engine resumes scheduling on
            # its own and no explicit `/resume` is needed.
            if not SKIP_SLEEP:
                _post(server, "wake_up", params={"tags": ["kv_cache"]})
            if RESET_MM_CACHE:
                _post(server, "reset_encoder_cache")
                _post(server, "reset_mm_cache")
            signatures.append(generation_signature(client, case.model, image_prompt=case.image_prompt))

    updated_signature, reloaded_signature = signatures
    assert_weight_update_matches_reference(
        dummy_signature,
        reference,
        updated_signature,
        reloaded_signature,
        case,
    )


SUFFIXES = ("tid2eid", "bias_vl", "e_score_correction_bias", "attn_sink")
PATTERNS = ["*.mlp.gate." + s for s in SUFFIXES[:3]] + ["*.self_attn.attn_sink"]


def is_tracked(name):
    return name.rsplit(".", 1)[-1] in SUFFIXES


def variant_tensor(tensor):
    if not tensor.dtype.is_floating_point:
        return (tensor + 1) % 8
    return (tensor.to(torch.bfloat16) + 0.015625).to(tensor.dtype)


def digest(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


class RetentionProbe:
    model_runner: NPUModelRunner | NPUModelRunnerV2
    model_config: ModelConfig
    weight_transfer_engine: NPUIPCWeightTransferEngine
    _sleep_saved_parameters: dict[str, torch.Tensor]
    _restore_sleep_parameters: Callable[[torch.nn.Module], None]
    _wsc_events: list[dict[str, object]]
    _wsc_finalize_events: list[dict[str, object]]
    _wsc_poison_restore: bool
    _wsc_poisoned: dict[str, object]

    def wsc_saved_state(self):
        # Device storage is unmapped during sleep: read only the real CPU backup.
        return {"saved": {n: digest(p) for n, p in self._sleep_saved_parameters.items()}}

    def wsc_state(self):
        from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

        model = self.model_runner.model
        parameters = {}
        kernel_names: dict[int, str] = {}
        aliases = {}
        meta_names = []
        for name, parameter in model.named_parameters():
            if not is_tracked(name):
                continue
            if parameter.is_meta:
                # Native reload uses temporary meta placeholders. Inspect the
                # retained kernel tensor, which is what captured graphs use.
                module_name, local_name = name.rsplit(".", 1)
                info = get_layerwise_info(model.get_submodule(module_name))
                assert info.kernel_tensors is not None
                parameter = info.kernel_tensors[0][local_name]
                meta_names.append(name)
            # Separate meta placeholders may refer to one original parameter.
            # Deduplicate by the resolved object, retaining the alias evidence.
            if id(parameter) in kernel_names:
                aliases[name] = kernel_names[id(parameter)]
                continue
            kernel_names[id(parameter)] = name
            parameters[name] = parameter
        return {
            "reload_meta_parameters": meta_names,
            "reload_kernel_aliases": aliases,
            "parameters": {
                name: {"sha256": digest(p), "ptr": p.data_ptr(), "shape": list(p.shape), "dtype": str(p.dtype)}
                for name, p in parameters.items()
            },
            **self.wsc_saved_state(),
            "configured": self.model_config.sleep_preserve_parameter_names,
            "old_snapshot_present": any(hasattr(m, "_runtime_tensor_backup") for m in model.modules()),
        }

    def wsc_install(self):
        from vllm.model_executor.model_loader import reload

        self._wsc_events = []
        self._wsc_finalize_events = []
        self._wsc_poison_restore = False
        self._wsc_poisoned = {}
        original_restore = self._restore_sleep_parameters

        def monitored_restore(model):
            # Poison after allocator.wake_up maps storage, before Worker copies
            # its CPU snapshot. Never touch an unmapped allocation during sleep.
            if self._wsc_poison_restore:
                self._wsc_poison_restore = False
                with torch.no_grad():
                    for name in self._sleep_saved_parameters:
                        param = model.get_parameter(name)
                        param.fill_(float("nan") if param.dtype.is_floating_point else -1)
                self._wsc_poisoned = self.wsc_state()
            return original_restore(model)

        self._restore_sleep_parameters = monitored_restore
        original_finalize = reload.finalize_layerwise_reload

        def monitored_finalize(*args, **kwargs):
            self._wsc_finalize_events.append({"stage": "finalize:before", "state": self.wsc_state()})
            result = original_finalize(*args, **kwargs)
            self._wsc_finalize_events.append({"stage": "finalize:after", "state": self.wsc_state()})
            return result

        reload.finalize_layerwise_reload = monitored_finalize
        engine = self.weight_transfer_engine
        for method in ("start_weight_update", "finish_weight_update"):
            original = getattr(engine, method)

            def monitored(*args, _original=original, _method=method, **kwargs):
                self._wsc_events.append({"stage": _method + ":before", "state": self.wsc_state()})
                result = _original(*args, **kwargs)
                self._wsc_events.append({"stage": _method + ":after", "state": self.wsc_state()})
                return result

            setattr(engine, method, monitored)
        return self.wsc_state()

    def wsc_variant(self):
        with torch.no_grad():
            for name, param in self.model_runner.model.named_parameters():
                if is_tracked(name):
                    param.copy_(variant_tensor(param))
        return self.wsc_state()

    def wsc_poison(self):
        assert self._sleep_saved_parameters
        self._wsc_poison_restore = True
        return self.wsc_saved_state()

    def wsc_poisoned_state(self):
        return self._wsc_poisoned

    def wsc_configure(self, mode):
        self.model_config.sleep_preserve_parameter_names = PATTERNS.copy() if mode == "retain" else []
        return self.wsc_state()

    def wsc_events(self):
        return self._wsc_events

    def wsc_finalize_events(self):
        return self._wsc_finalize_events


class RetentionSource(_ReducedCheckpointSource):
    def __init__(self, case, device, variant=False):
        super().__init__(case, device)
        self.variant = variant

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
        "tests.e2e.pull_request.one_card.rlhf.state_transitions.test_npu_ipc_weight_transfer_dsv4_flash_vision.RetentionProbe",
        "--sleep-preserve-parameter-names",
        *PATTERNS,
    ]
    evidence: dict[str, Any] = {"packed": packed, "omitted_checkpoint_names": omitted, "cycles": []}
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
            actual = utils.generation_signature(client, case.model, image_prompt=True)
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

        # Full-update control: unfreeze the parameters and load different values.
        rpc(server, "wsc_configure", "full")
        full_engine = trainer(server, source, packed)
        _post(server, "pause")
        _post(server, "sleep", params={"level": 2})
        assert rpc(server, "wsc_saved_state")["saved"] == {}
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
        assert rpc(server, "wsc_saved_state")["saved"] == {}
        _post(server, "wake_up", params={"tags": ["weights"]})
        _post(server, "wake_up", params={"tags": ["kv_cache"]})
        same_parameters(rpc(server, "wsc_state"), initial)
        assert utils.generation_signature(client, case.model, image_prompt=True) == references[0]
        evidence["level_one"] = "passed"
        evidence["all_events"] = rpc(server, "wsc_events")
        evidence["all_finalize_events"] = rpc(server, "wsc_finalize_events")
    mode = json.loads(utils.FULL_DECODE_ONLY_CONFIG)["cudagraph_mode"]
    path = tmp_path / f"evidence-{mode}-{'packed' if packed else 'unpacked'}.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"RETENTION_EVIDENCE={path}")
