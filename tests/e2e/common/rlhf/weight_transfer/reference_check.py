# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Startup-load reference and live-update equivalence checks."""

import base64
import hashlib
import json
from contextlib import ExitStack, contextmanager
from io import BytesIO

import torch
from PIL import Image
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.rlhf.common import post_control, rlhf_server, wait_for_free_device_memory
from tests.e2e.common.rlhf.weight_transfer.case import WeightUpdateAudit, WeightUpdateModelCase
from tests.e2e.common.rlhf.weight_transfer.weight_source import (
    FixedRandomWeightSource,
    fixed_startup_checkpoint,
)

UPDATE_ROUNDS = 2


def capture_update_result(server, case, *, audit: WeightUpdateAudit | None = None):
    if audit is not None:
        return audit.capture_result(server, case)
    return generation_signature(
        server.get_client(),
        case.model,
        prompts=case.prompts,
        image_prompt=case.image_prompt,
    )


FULL_DECODE_ONLY_CONFIG = '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]}'


def _common_serve_args(
    case: WeightUpdateModelCase,
    *,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> list[str]:
    """Flags a live-update lane and its reference share.

    Holding them in one place is what makes the two signatures comparable: the
    reference may differ from the lane in exactly one way — it loads the payload
    at startup instead of starting from dummy weights and receiving it live.
    dtype, graph mode, NZ layout, memory budget, max length, tokenizer/executor
    backend and config overrides all stay identical.
    """
    return [
        "--dtype",
        "bfloat16",
        "--compilation-config",
        FULL_DECODE_ONLY_CONFIG,
        # Each NPU IPC trainer rank shares a chip with a rollout worker: the co-located
        # trainer only gets the HBM it needs once the rollout engine sleeps, so
        # the lane releases the engine with a level-2 sleep before every
        # transfer. That requires the CaMem pool, and both builders enable it so
        # the reference stays a plain startup load of the same payload under the
        # very same allocation pool; the reference itself never sleeps.
        "--enable-sleep-mode",
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--port",
        str(port),
        "--trust-remote-code",
        "--additional-config",
        '{"weight_nz_mode": 0}',
        *case.server_args(),
    ]


def live_update_serve_args(
    case: WeightUpdateModelCase,
    *,
    backend: str,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int = 1,
) -> list[str]:
    """A lane that starts from dummy weights and receives W over ``backend``."""
    return [
        "--load-format",
        "dummy",
        "--weight-transfer-config",
        json.dumps({"backend": backend}),
        *_common_serve_args(
            case,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        ),
    ]


def reference_serve_args(
    case: WeightUpdateModelCase,
    *,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int = 1,
) -> list[str]:
    """A server that loads the transfer's payload at normal startup.

    No ``--load-format dummy`` and no transfer backend: this server must be
    correct on its own. It serves under the case's model name and tokenizer so
    both servers answer to the same ``model`` argument in the client calls.
    """
    return [
        "--tokenizer",
        case.model,
        "--served-model-name",
        case.model,
        *_common_serve_args(
            case,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        ),
    ]


# Reuse a startup reference only for the same model, memory, TP, logical device
# and effective child-server environment. Transport and packing do not affect it.
_REFERENCE_SIGNATURES: dict[
    tuple[str, float, int, int, tuple[tuple[str, str], ...]], list[tuple[str, tuple[float, ...]]]
] = {}


def reference_signature(
    source: FixedRandomWeightSource,
    case: WeightUpdateModelCase,
    *,
    port: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    device_index: int,
    env_dict: dict[str, str] | None = None,
    server_host: str = "127.0.0.1",
) -> list[tuple[str, tuple[float, ...]]]:
    """Signature of a server that loaded the payload at normal startup.

    This is the correctness oracle and it never runs the live-update path, so a
    transfer that silently drops, renames, reshapes or recasts part of the
    payload cannot make the comparison pass. Independent server processes also
    keep the baseline free of whatever the live path does to the model (graph
    capture, layerwise reload state, derived representations).
    """
    cache_key = (
        case.id,
        gpu_memory_utilization,
        tensor_parallel_size,
        device_index,
        tuple(sorted((env_dict or {}).items())),
    )
    cached = _REFERENCE_SIGNATURES.get(cache_key)
    if cached is not None:
        return cached

    for rank in range(tensor_parallel_size):
        wait_for_free_device_memory(device_index + rank, gpu_memory_utilization)
    with (
        fixed_startup_checkpoint(source) as checkpoint_dir,
        rlhf_server(
            checkpoint_dir,
            vllm_serve_args=reference_serve_args(
                case,
                port=port,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
            ),
            server_host=server_host,
            server_port=port,
            env_dict=env_dict,
        ) as server,
    ):
        signature = generation_signature(
            server.get_client(), case.model, prompts=case.prompts, image_prompt=case.image_prompt
        )

    _REFERENCE_SIGNATURES[cache_key] = signature
    return signature


IMAGE_PROMPT = "Describe the colors of this image in one short sentence."


def _quadrant_png_data_url(size: int = 384) -> str:
    """``size`` is the model's ``vision_min_pixels`` side (384 x 384), so the
    processor does not have to resize the input
    """
    half = size // 2
    image = Image.new("RGB", (size, size), (0, 0, 128))
    image.paste((255, 0, 128), (half, 0, size, half))
    image.paste((0, 255, 128), (0, half, half, size))
    image.paste((255, 255, 128), (half, half, size, size))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


IMAGE_DATA_URL = _quadrant_png_data_url()


def generation_signature(
    client, model: str, *, prompts: tuple[str | tuple[int, ...], ...], image_prompt: bool = False
) -> list[tuple[str, tuple[float, ...]]]:
    """Capture deterministic text and logprobs for exact reload comparison.

    ``image_prompt`` appends one chat request carrying an image for multimodal
    cases: the vision tower only runs when an image is in the batch, so without
    it a dropped, renamed or reshaped vision tensor would not move the signature
    and the case would prove nothing about the tower.
    """
    signature = []
    for prompt in prompts:
        response = client.completions.create(
            model=model,
            prompt=list(prompt) if isinstance(prompt, tuple) else prompt,
            max_tokens=8,
            temperature=0,
            logprobs=1,
            seed=0,
        )
        choice = response.choices[0]
        token_logprobs = tuple(choice.logprobs.token_logprobs or ())
        assert token_logprobs, f"{model}: generation returned no token logprobs"
        signature.append((choice.text, token_logprobs))
    if image_prompt:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_PROMPT},
                        {"type": "image_url", "image_url": {"url": IMAGE_DATA_URL}},
                    ],
                }
            ],
            max_tokens=8,
            temperature=0,
            seed=0,
            logprobs=True,
        )
        choice = response.choices[0]
        token_logprobs = tuple(item.logprob for item in (choice.logprobs.content or ()))
        assert token_logprobs, f"{model}: image generation returned no token logprobs"
        signature.append((choice.message.content or "", token_logprobs))
    return signature


@contextmanager
def weight_transfer_servers(
    source: FixedRandomWeightSource,
    case: WeightUpdateModelCase,
    *,
    backend: str,
    tensor_parallel_size: int,
    device_index: int,
    gpu_memory_utilization: float,
    env_dict: dict[str, str],
):
    """Load W independently, then start the dummy lane under identical settings.

    The reference server must release every TP device before the live-update
    server starts. Both use the case's environment and the same generated W.
    """
    reference = reference_signature(
        source,
        case,
        port=get_open_port(),
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        device_index=device_index,
        env_dict=env_dict,
    )
    for rank in range(tensor_parallel_size):
        wait_for_free_device_memory(device_index + rank, gpu_memory_utilization)
    port = get_open_port()
    with rlhf_server(
        case.model,
        vllm_serve_args=live_update_serve_args(
            case,
            backend=backend,
            port=port,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        ),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
    ) as server:
        dummy_signature = generation_signature(
            server.get_client(), case.model, prompts=case.prompts, image_prompt=case.image_prompt
        )
        yield server, reference, dummy_signature


@contextmanager
def hccl_weight_transfer_servers(
    sources,
    case,
    *,
    backend,
    tensor_parallel_size,
    device_index,
    gpu_memory_utilization,
    env_dict,
    audit: WeightUpdateAudit | None = None,
):
    """Build independent references for explicit HCCL update sources.

    Keep A's temporary directory alive for the live server's config/recipe.
    Both references finish before the live server starts, sharing one NPU.
    No reference cache is used: each source has independent weights/scales.
    """
    references = {}
    with ExitStack() as stack:
        model_directory = None
        for name, source in sources.items():
            directory = stack.enter_context(fixed_startup_checkpoint(source))
            if model_directory is None:
                model_directory = directory
            for rank in range(tensor_parallel_size):
                wait_for_free_device_memory(device_index + rank, gpu_memory_utilization)
            port = get_open_port()
            with rlhf_server(
                directory,
                vllm_serve_args=reference_serve_args(
                    case,
                    port=port,
                    gpu_memory_utilization=gpu_memory_utilization,
                    tensor_parallel_size=tensor_parallel_size,
                ),
                server_host="127.0.0.1",
                server_port=port,
                env_dict=env_dict,
            ) as server:
                references[name] = capture_update_result(server, case, audit=audit)
        for rank in range(tensor_parallel_size):
            wait_for_free_device_memory(device_index + rank, gpu_memory_utilization)
        port = get_open_port()
        with rlhf_server(
            model_directory,
            vllm_serve_args=[
                "--tokenizer",
                case.model,
                "--served-model-name",
                case.model,
                *live_update_serve_args(
                    case,
                    backend=backend,
                    port=port,
                    gpu_memory_utilization=gpu_memory_utilization,
                    tensor_parallel_size=tensor_parallel_size,
                ),
            ],
            server_host="127.0.0.1",
            server_port=port,
            env_dict=env_dict,
        ) as server:
            dummy = None if audit is not None else capture_update_result(server, case)
            yield server, references, dummy


def assert_weight_update_matches_reference(
    dummy_signature, reference_signature, updated_signatures, case: WeightUpdateModelCase
) -> None:
    """Compare every live update against independent startup loading.

    Dummy output must differ to prove the payload is observable. The first
    update proves that transfer takes effect; repeated reload checks that
    layerwise reload and graph-captured storage preserve the same behavior.
    Comparing updates only to each other could hide consistently wrong loads.
    """
    assert dummy_signature != reference_signature, (
        f"{case.id}: a dummy-loaded model already matches a normally loaded one; "
        "the reference cannot observe the payload"
    )
    assert len(updated_signatures) == UPDATE_ROUNDS, (
        f"{case.id}: expected {UPDATE_ROUNDS} updates, got {len(updated_signatures)}"
    )
    for update_round, signature in enumerate(updated_signatures, start=1):
        phase = "first live weight update" if update_round == 1 else "repeated live reload"
        assert signature == reference_signature, (
            f"{case.id}: {phase} (round {update_round}) does not match normal startup loading of the same weights"
        )


class MXFP8AuditWorker:
    """Inspect representative dense/MoE layers without changing reload behavior."""

    def _mxfp8_tensors(self):
        """Yield the same inference buffers for content and address checks."""
        kinds = set()
        for prefix, layer in self.model_runner.model.named_modules():
            if not hasattr(layer, "_mxfp8_original_shapes"):
                continue
            kind = "moe" if "w13_weight" in layer._mxfp8_original_shapes else "linear"
            if kind in kinds:
                continue
            assert getattr(layer, "_already_called_process_weights_after_loading", False), (
                f"{prefix}: MXFP8 post-processing was skipped"
            )
            kinds.add(kind)
            for name in ("w13_weight", "w2_weight") if kind == "moe" else ("weight",):
                for tensor_name in (name, name + "_scale"):
                    tensor = getattr(layer, tensor_name, None)
                    if tensor is None:
                        values = getattr(layer, tensor_name + "_list", None)
                        if values is None:
                            values = getattr(layer, "cann_mega_moe_" + tensor_name + "_list")
                        tensor = values[0]
                    elif kind == "moe":
                        tensor = tensor[0]
                    expected_dtypes = (
                        (torch.float8_e4m3fn,) if tensor_name == name else (torch.uint8, torch.float8_e8m0fnu)
                    )
                    assert tensor.dtype in expected_dtypes, (prefix, tensor_name, tensor.dtype)
                    assert tensor.element_size() == 1, (
                        prefix,
                        tensor_name,
                        tensor.dtype,
                    )
                    yield prefix + "." + tensor_name, tensor
        assert kinds == {"linear", "moe"}, f"Missing MXFP8 layer types: {kinds}"

    def mxfp8_snapshot(self):
        tensors = {}
        for name, tensor in self._mxfp8_tensors():
            raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            tensors[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        return tensors

    def mxfp8_pointers(self):
        return {name: tensor.data_ptr() for name, tensor in self._mxfp8_tensors()}


def capture_quantized_pointers(server):
    """Capture live receiver addresses without running dummy-weight inference."""
    return post_control(server, "collective_rpc", json={"method": "mxfp8_pointers"}).json()["results"]


def assert_quantized_storage_unchanged(initial, updates, case):
    """Compare dummy -> A -> B -> A addresses within the same receiver only."""
    assert initial and all(initial), f"{case.id}: missing initial inference tensor addresses"
    assert len(updates) == 3, f"{case.id}: expected three A/B/A address snapshots"
    for update_round, actual in enumerate(updates, start=1):
        assert len(actual) == len(initial), f"{case.id}: worker count changed in round {update_round}"
        for rank, (before, after) in enumerate(zip(initial, actual, strict=True)):
            assert before.keys() == after.keys(), f"{case.id}: rank {rank} inference tensor names changed"
            for name, pointer in before.items():
                assert after[name] == pointer, (
                    f"{case.id}: rank {rank} {name} moved after update {update_round}: {pointer} -> {after[name]}"
                )


def capture_quantized_result(server, case):
    return {
        "generation": generation_signature(server.get_client(), case.model, prompts=case.prompts),
        "tensors": post_control(server, "collective_rpc", json={"method": "mxfp8_snapshot"}).json()["results"],
    }


def assert_quantized_updates(references, updates, case):
    assert len(references) == 2 and len(updates) == 3, f"{case.id}: expected A/B/A and two fresh-load baselines"
    for index, actual in enumerate(updates):
        assert actual == references[index % 2], f"{case.id}: round {index + 1} differs from fresh-load"
    assert updates[0]["generation"] != updates[1]["generation"], "A/B inference results did not change"
    for rank_a, rank_b in zip(updates[0]["tensors"], updates[1]["tensors"], strict=True):
        assert rank_a.keys() == rank_b.keys()
        for name in rank_a:
            assert rank_a[name]["sha256"] != rank_b[name]["sha256"], f"{name}: A/B value unchanged"


class MXFP8WeightUpdateAudit:
    """Keep MXFP8-specific RPCs and assertions outside the transport runner."""

    @staticmethod
    def capture_result(server, case):
        return capture_quantized_result(server, case)

    @staticmethod
    def capture_state(server):
        return capture_quantized_pointers(server)

    @staticmethod
    def check_updates(*, initial_state, updated_states, references, updates, case):
        assert_quantized_storage_unchanged(initial_state, updated_states, case)
        assert_quantized_updates(references, updates, case)
