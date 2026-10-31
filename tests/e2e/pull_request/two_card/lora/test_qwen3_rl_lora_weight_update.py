#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
#
"""End-to-end test for RL training LoRA weight update on dense models.

This test starts a vLLM server with a dense base model
(``Qwen/Qwen3-0.6B``) and HCCL weight transfer enabled, then simulates the
RLHF-style LoRA weight-update workflow via HCCL broadcast:

1. Start server with the dense base model + HCCL weight transfer (TP=1)
2. Generate baseline outputs **without** LoRA — the model responds with
   generic text to a self-cognition prompt
3. Trainer loads the Alice LoRA adapter into the base model on NPU 1,
   then broadcasts the merged weights to NPU 0 via HCCL
4. Generate outputs **with** Alice LoRA — the model identifies itself as
   "Alice"
5. **Update** the LoRA weights by reloading the Bob adapter on NPU 1 and
   broadcasting the new merged weights via HCCL
6. Generate outputs **with** Bob LoRA — the model now identifies itself as
   "Bob", confirming the weight update took effect at runtime
7. Trainer broadcasts the original base weights — the model reverts to
   generic text
8. Generate outputs again **without** LoRA

This exercises the HCCL weight transfer path — LoRA adapters are merged into the
base model on the trainer NPU and the full merged checkpoint is broadcast to the
inference NPU via HCCL packed broadcast + layerwise reload.

Topology (requires 2 NPUs):
- NPU 0: vLLM inference worker (rank 1 in the HCCL group)
- NPU 1: trainer / weight source (rank 0 in the HCCL group)

Refer to ``examples/rl/rlhf_http_hccl.py`` for the end-user workflow.

Run with::

    pytest tests/e2e/multicard/2-cards/test_dense_lora_weight_update.py
"""

import os
import shutil
import tempfile
import threading

import pytest
import requests
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from transformers import AutoConfig, AutoModelForCausalLM
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"

# Two self-cognition LoRA adapters that share the same base model.
# Alice completes with "Alice"; Bob completes with "Bob".
LORA_NAME = "self_cognition"
ALICE_PATH = "charent/self_cognition_Alice"
BOB_PATH = "charent/self_cognition_Bob"

TENSOR_PARALLEL_SIZE = 1

# Topology: NPU 0 = inference, NPU 1 = trainer.
INFERENCE_WORLD_SIZE = 1
TRAINER_DEVICE_INDEX = INFERENCE_WORLD_SIZE

# A simple completion prompt that reveals model identity.
SELF_COGNITION_PROMPT = (
    "Hi, my name is"
)

# The Alice adapter completes with "Alice"; the Bob adapter completes with
# "Bob".
EXPECTED_ALICE_PREFIX = "Alice"
EXPECTED_BOB_PREFIX = "Bob"

# HTTP timeouts (seconds). Weight broadcast can take a while for large models.
SERVER_START_TIMEOUT = 2800
INIT_TIMEOUT = 120
UPDATE_TIMEOUT = 300
CONTROL_TIMEOUT = 60


def _log(message: str) -> None:
    """Flushed log so step markers show up immediately even when stdout is piped."""
    print(f"[trainer] {message}", flush=True)


def _post(server: RemoteOpenAIServer, route: str, *, json=None, timeout=CONTROL_TIMEOUT):
    """POST JSON to a dev-mode endpoint on *server*, raise on non-2xx."""
    response = requests.post(server.url_for(route), json=json, timeout=timeout)
    response.raise_for_status()
    return response


class _BackgroundPost(threading.Thread):
    """Run an HTTP POST in a thread while keeping its exception visible.

    The trainer side blocks on collective HCCL ops, so the matching server-side
    RPC must run concurrently. If that RPC fails, swallowing the exception would
    deadlock the trainer forever; instead we record it and surface it on join().
    """

    def __init__(self, server, route, *, json=None, timeout=CONTROL_TIMEOUT):
        super().__init__(daemon=True)
        self._server = server
        self._route = route
        self._json = json
        self._timeout = timeout
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            _post(self._server, self._route, json=self._json, timeout=self._timeout)
            _log(f"background POST /{self._route} done")
        except BaseException as exc:
            self.error = exc
            _log(f"background POST /{self._route} FAILED: {exc!r}")

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"server-side /{self._route} failed") from self.error


def _generate_one(client, model, prompt):
    """Generate a single completion via the OpenAI-compatible API."""
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=64,
        temperature=0,
    )
    return response.choices[0].text


def _build_merged_model(base_model_path, lora_path, device_index):
    """Build a model with LoRA merged in on the trainer NPU.

    Loads the base model from *base_model_path*, manually applies the LoRA
    adapter at *lora_path* by reading the safetensors weights and adding
    ``lora_B @ lora_A * scaling`` to each target parameter.  No PEFT/peft
    dependency is required.

    When *lora_path* is ``None`` the base model is returned unchanged (used
    for the revert phase).
    """
    import json
    import re

    import safetensors

    device = f"npu:{device_index}"
    _log(f"loading base model from {base_model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(device=device, dtype=torch.bfloat16)

    if lora_path is None:
        _log("no LoRA adapter — returning base model as-is")
        return model

    # ── read adapter config ──────────────────────────────────────────
    config_path = os.path.join(lora_path, "adapter_config.json")
    with open(config_path) as fh:
        lora_config = json.load(fh)
    r = lora_config["r"]
    alpha = lora_config["lora_alpha"]
    scaling = alpha / r
    _log(f"LoRA config: r={r}, alpha={alpha}, scaling={scaling}")

    # ── read adapter weights ─────────────────────────────────────────
    weights_path = os.path.join(lora_path, "adapter_model.safetensors")
    lora_weights = {}
    with safetensors.safe_open(weights_path, framework="pt") as sf:
        for key in sf.keys():
            lora_weights[key] = sf.get_tensor(key)

    # ── pair lora_A / lora_B and merge ───────────────────────────────
    # PEFT naming: base_model.model.{base_param}.lora_{A|B}.weight
    _PEFT_PREFIX = "base_model.model."
    _LORA_A_RE = re.compile(r"\.lora_A\.(?:default\.)?weight$")
    _LORA_B_RE = re.compile(r"\.lora_B\.(?:default\.)?weight$")

    # Group lora_A keys so we can find their matching lora_B.
    a_keys = sorted(k for k in lora_weights if _LORA_A_RE.search(k))
    merged_count = 0
    for a_key in a_keys:
        b_key = _LORA_A_RE.sub(
            lambda m: m.group(0).replace("lora_A", "lora_B"), a_key
        )
        if b_key not in lora_weights:
            raise KeyError(f"missing lora_B for {a_key} (expected {b_key})")

        # Derive the base-model parameter name.
        param_name = a_key
        if param_name.startswith(_PEFT_PREFIX):
            param_name = param_name[len(_PEFT_PREFIX):]
        param_name = _LORA_A_RE.sub("", param_name) + ".weight"

        lora_A = lora_weights[a_key].to(device=device, dtype=torch.float32)
        lora_B = lora_weights[b_key].to(device=device, dtype=torch.float32)
        # lora_A: [r, in_features], lora_B: [out_features, r]
        delta = (lora_B @ lora_A) * scaling

        # Apply to base model parameter.
        param = dict(model.named_parameters()).get(param_name)
        if param is None:
            raise KeyError(
                f"base parameter '{param_name}' not found in model "
                f"(from LoRA key {a_key})"
            )
        param.data = (param.data.float() + delta).to(param.dtype)
        merged_count += 1

    _log(f"merged {merged_count} LoRA parameter pairs into base model")
    return model


def _collect_weight_metadata(train_model):
    """Collect parameter metadata and size the packed buffer for broadcasting."""
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    max_tensor_bytes = 0
    for name, parameter in train_model.named_parameters():
        names.append(name)
        dtype_names.append(str(parameter.dtype).split(".")[-1])
        shapes.append(list(parameter.shape))
        tensor_bytes = parameter.numel() * parameter.element_size()
        max_tensor_bytes = max(max_tensor_bytes, tensor_bytes)

    # Keep the 1 GiB default unless a single tensor needs more (+128 MiB headroom).
    packed_buffer_size_bytes = max(max_tensor_bytes + 128 * 2**20, 2**30)
    return names, dtype_names, shapes, packed_buffer_size_bytes


def _hcccl_broadcast_weights(server, train_model):
    """Broadcast the full trainer model weights to the server via HCCL.

    Builds the HCCL process group, pauses generation, broadcasts all
    ``named_parameters`` via HCCL packed broadcast, then resumes generation.
    """
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerSendWeightsArgs,
        HCCLWeightTransferEngine,
    )

    master_address = get_ip()
    master_port = get_open_port()
    rank_offset = 1
    world_size = INFERENCE_WORLD_SIZE + 1  # workers + trainer

    # 1) HCCL rendezvous — server side blocks until trainer connects.
    init_info = dict(
        master_address=master_address,
        master_port=master_port,
        rank_offset=rank_offset,
        world_size=world_size,
    )
    _log(f"HCCL rendezvous at {master_address}:{master_port} (world_size={world_size}) ...")
    init_thread = _BackgroundPost(
        server,
        "init_weight_transfer_engine",
        json={"init_info": init_info},
        timeout=INIT_TIMEOUT,
    )
    init_thread.start()
    model_update_group = HCCLWeightTransferEngine.trainer_init(
        dict(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
        ),
    )
    _log("trainer_init returned, waiting for server init RPC ...")
    init_thread.join()
    init_thread.raise_if_failed()
    _log("HCCL process group established")

    # 2) Pause generation and start weight update lifecycle.
    _post(server, "pause")
    _log("generation paused")

    # Probe for lifecycle endpoints available on vLLM main.
    try:
        _post(server, "start_weight_update", json={"is_checkpoint_format": True})
        use_lifecycle = True
    except requests.HTTPError:
        use_lifecycle = False
    _log(f"lifecycle endpoints available: {use_lifecycle}")

    names, dtype_names, shapes, packed_buffer_size_bytes = _collect_weight_metadata(train_model)
    update_info = dict(
        names=names,
        dtype_names=dtype_names,
        shapes=shapes,
        packed=True,
        packed_buffer_size_bytes=packed_buffer_size_bytes,
    )
    if not use_lifecycle:
        update_info["is_checkpoint_format"] = True

    # 3) Broadcast weights — server blocks on HCCL recv, trainer produces data.
    _log(f"broadcasting {len(names)} tensors via HCCL (packed) ...")
    update_thread = _BackgroundPost(
        server,
        "update_weights",
        json={"update_info": update_info},
        timeout=UPDATE_TIMEOUT,
    )
    update_thread.start()

    trainer_args = HCCLTrainerSendWeightsArgs(
        group=model_update_group,
        packed=True,
        packed_buffer_size_bytes=packed_buffer_size_bytes,
    )
    HCCLWeightTransferEngine.trainer_send_weights(
        iterator=train_model.named_parameters(),
        trainer_args=trainer_args,
    )
    _log("trainer finished sending weights, waiting for server update RPC ...")
    update_thread.join()
    update_thread.raise_if_failed()
    _log("weight broadcast complete")

    # 4) Finalize and resume.
    if use_lifecycle:
        _post(server, "finish_weight_update")
    _post(server, "resume")
    _log("generation resumed")


def _has_lifecycle_endpoints(server: RemoteOpenAIServer) -> bool:
    """Detect whether the server exposes the vLLM-main start/finish endpoints."""
    try:
        response = requests.post(
            server.url_for("start_weight_update"),
            json={"is_checkpoint_format": True},
            timeout=CONTROL_TIMEOUT,
        )
    except requests.RequestException:
        return False
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def _hcccl_transfer_trainer_to_server(server, base_model_path, lora_path):
    """Full cycle: build merged model on NPU 1, broadcast to NPU 0 via HCCL."""
    _log(f"building merged model on npu:{TRAINER_DEVICE_INDEX} ...")
    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    train_model = _build_merged_model(base_model_path, lora_path, TRAINER_DEVICE_INDEX)
    _hcccl_broadcast_weights(server, train_model)
    # Free trainer memory.
    del train_model


@pytest.mark.skipif(
    torch.npu.device_count() < 2,
    reason="Dense LoRA e2e test requires at least 2 NPUs for HCCL weight transfer.",
)
def test_rl_lora_weight_update_dense():
    """End-to-end test: update model weights at runtime via HCCL broadcast.

    This test mimics the RL training weight-update cycle on dense:

    1. Server runs Qwen3-0.6B (dense) on NPU 0 with HCCL weight transfer.
    2. Generate baseline — model gives a generic self-description.
    3. Trainer merges Alice LoRA on NPU 1 → HCCL broadcast merged weights.
    4. Generate with Alice — model identifies as "Alice".
    5. Trainer merges Bob LoRA on NPU 1 → HCCL broadcast new merged weights.
    6. Generate with Bob — model now identifies as "Bob".
    7. Trainer broadcasts original base weights — model reverts.
    8. Generate after revert — model gives generic text.
    """
    port = get_open_port()
    server_args = [
        "--enforce-eager",
        "--weight-transfer-config",
        '{"backend": "nccl"}',
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.6",
        "--tensor-parallel-size",
        str(INFERENCE_WORLD_SIZE),
        "--port",
        str(port),
        "--trust-remote-code",
    ]
    # Dev-mode endpoints (/init_weight_transfer_engine, /update_weights,
    # /pause, /resume) are registered when VLLM_SERVER_DEV_MODE=1.
    # Pin the server to NPU 0 so the trainer can own NPU 1 exclusively.
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": "0",
        "VLLM_ASCEND_ENABLE_NZ": "0",
    }

    _log(f"starting server on port {port} (NPU 0, HCCL weight transfer) ...")
    with RemoteOpenAIServer(
        MODEL_NAME,
        vllm_serve_args=server_args,
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
        max_wait_seconds=SERVER_START_TIMEOUT,
    ) as server:
        client = server.get_client()

        # ── Phase 1: Generate WITHOUT LoRA (baseline) ──────────────────
        _log("Phase 1: generating baseline (base model, no LoRA) ...")
        baseline_output = _generate_one(client, MODEL_NAME, SELF_COGNITION_PROMPT)
        _log(f"  baseline: {baseline_output!r}")

        # ── Phase 2: HCCL broadcast Alice-merged weights → generate ────
        _log("Phase 2: HCCL transfer Alice-merged model ...")
        _hcccl_transfer_trainer_to_server(server, MODEL_NAME, ALICE_PATH)

        _log("Phase 2: generating WITH Alice ...")
        alice_output = _generate_one(client, MODEL_NAME, SELF_COGNITION_PROMPT)
        _log(f"  Alice: {alice_output!r}")

        # ── Phase 3: HCCL broadcast Bob-merged weights → generate ──────
        _log("Phase 3: HCCL transfer Bob-merged model ...")
        _hcccl_transfer_trainer_to_server(server, MODEL_NAME, BOB_PATH)

        _log("Phase 3: generating WITH Bob ...")
        bob_output = _generate_one(client, MODEL_NAME, SELF_COGNITION_PROMPT)
        _log(f"  Bob: {bob_output!r}")

        # ── Phase 4: HCCL broadcast base-only weights (revert) ─────────
        _log("Phase 4: HCCL transfer base model (revert) ...")
        _hcccl_transfer_trainer_to_server(server, MODEL_NAME, None)

        _log("Phase 4: generating after revert ...")
        unload_output = _generate_one(client, MODEL_NAME, SELF_COGNITION_PROMPT)
        _log(f"  after revert: {unload_output!r}")

    # ── Assertions ─────────────────────────────────────────────────────
    # Alice-merged model: output should contain "Alice".
    assert EXPECTED_ALICE_PREFIX.lower() in alice_output.lower(), (
        f"Alice output does not contain expected self-cognition text.\n"
        f"Expected prefix: {EXPECTED_ALICE_PREFIX!r}\n"
        f"Got: {alice_output!r}"
    )

    # Bob-merged model: output should contain "Bob".
    assert EXPECTED_BOB_PREFIX.lower() in bob_output.lower(), (
        f"Bob output does not contain expected self-cognition text.\n"
        f"Expected prefix: {EXPECTED_BOB_PREFIX!r}\n"
        f"Got: {bob_output!r}"
    )

    # The weight update must have changed the output: Alice ≠ Bob.
    assert alice_output != bob_output, (
        "Weight update did not change the model's output — "
        "Alice and Bob outputs are identical, the update may not have "
        "taken effect."
    )

    # Without LoRA, the model should identify as neither Alice nor Bob.
    assert EXPECTED_ALICE_PREFIX.lower() not in baseline_output.lower(), (
        f"Baseline output unexpectedly identifies as Alice — "
        f"the base model may already be fine-tuned for self-cognition.\n"
        f"Got: {baseline_output!r}"
    )

    # After reverting, output should also be generic.
    assert EXPECTED_ALICE_PREFIX.lower() not in unload_output.lower(), (
        f"After-revert output still identifies as Alice — "
        f"the revert may not have taken effect.\n"
        f"Got: {unload_output!r}"
    )

    # Alice output must differ from baseline.
    assert alice_output != baseline_output, (
        "Alice-merged weights did not change the model's output — "
        "the HCCL transfer may not be active."
    )

    # After-revert output must differ from both merged outputs.
    assert unload_output != alice_output, (
        "After reverting, the model still produces Alice-style output — "
        "the revert may not have taken effect."
    )
    assert unload_output != bob_output, (
        "After reverting, the model still produces Bob-style output — "
        "the revert may not have taken effect."
    )

    _log("SUCCESS: Dense HCCL weight-update cycle verified — "
         "broadcast Alice → broadcast Bob → revert to base model")
  