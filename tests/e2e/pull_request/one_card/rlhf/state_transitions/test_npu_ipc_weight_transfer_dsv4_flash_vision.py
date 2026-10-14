# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU IPC live weight update for a layer-reduced DeepSeek-V4-Flash-Vision.

``test_npu_ipc_weight_transfer.py`` carries the full four-model matrix, so
running a single case out of it still discovers -- and, in CI, stages -- the
other three checkpoints. This lane keeps only the multimodal DeepSeek-V4 case:
it serves one reduced checkpoint, its depth is tunable from the environment, and
it can be bisected (text vs vision, 4 layers vs 2) without touching the shared
matrix.

Reduction knobs, read once at import time:

================================  ==============================  ==================================
environment variable              default                         effect
================================  ==============================  ==================================
``VLLM_RL_DSV4_MODEL``            ``deepseek-ai/DeepSeek-V4-      HF repo id or a local checkpoint
                                  Flash-Vision-Exp``              directory
``VLLM_RL_DSV4_TEXT_LAYERS``      4                               ``num_hidden_layers``
``VLLM_RL_DSV4_VISION_LAYERS``    2                               ``vision_n_layers`` (0 = text-only)
``VLLM_RL_DSV4_EXPERTS``          8                               ``n_routed_experts`` (>= top-k = 6)
``VLLM_RL_DSV4_HASH_LAYERS``      unset                           ``num_hash_layers`` -- leave unset
``VLLM_RL_DSV4_MEM_UTIL``         0.45                            ``--gpu-memory-utilization``
``VLLM_RL_TEST_DEVICE_INDEX``     0                               physical chip to run on
================================  ==============================  ==================================

``num_hidden_layers`` also selects which attention types survive: the config
derives ``layer_types`` from the checkpoint's ``compress_ratios``
(``[0, 0, 4, 128, ...]`` -> sliding, sliding, compressed-sparse,
heavily-compressed) and truncates it to ``num_hidden_layers``::

    TEXT_LAYERS=4   sliding, sliding, CSA, HCA   full coverage (default)
    TEXT_LAYERS=3   sliding, sliding, CSA        drops HCA
    TEXT_LAYERS=2   sliding, sliding             drops CSA/HCA + compressor/indexer
    TEXT_LAYERS=1   sliding                      smoke only

Do not set ``VLLM_RL_DSV4_HASH_LAYERS`` unless you know why: the HF config folds
it into ``mlp_layer_types`` (``["hash_moe"] * min(n, num_hash_layers)``) while the
served model tests ``layer_idx < config.num_hash_layers``. Both sides agree on the
checkpoint value, so overriding it only risks a payload/served-model mismatch.
With the default depth every layer is a hash layer (``tid2eid``) and the payload
carries the hash-router tables that the level-2 sleep otherwise zero-fills.

The oracle is the shared one -- *normal startup loading of the same payload*:

    normal startup load(W) == first live update(W) == second live update(W)

A reference server loads a temporary checkpoint written by the very generator the
transfer ships, and the live lane (dummy weights + ``/sleep`` level 2 + NPU IPC
update) has to reproduce its signature on the first update and stay on it after a
second one. Setting ``VLLM_IPC_STRICT_UPDATE=1`` additionally makes the server-side
FINISH of every update fail loudly when a module received zero or partial weights
(the engines return a coverage report and warn otherwise).

Run it from the repo root, inside the container that owns the NPUs::

    cd /path/to/vllm-ascend
    VLLM_RL_TEST_DEVICE_INDEX=0 pytest -sv \
        tests/e2e/pull_request/one_card/rlhf/state_transitions/test_npu_ipc_weight_transfer_dsv4-vision.py

"""

import os
import shutil
from pathlib import Path

import pytest
import requests
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
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

_EXTRA_SERVER_ARGS = ["--tokenizer-mode", "deepseek_v4"]
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
    # Mirror the opt-in coverage gate into the server process explicitly, so the
    # FINISH of every update fails loudly on zero/partial modules instead of only
    # logging them (see the weight-transfer engines' coverage report).
    if os.environ.get("VLLM_IPC_STRICT_UPDATE"):
        env_dict["VLLM_IPC_STRICT_UPDATE"] = os.environ["VLLM_IPC_STRICT_UPDATE"]

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

        from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
        from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

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
