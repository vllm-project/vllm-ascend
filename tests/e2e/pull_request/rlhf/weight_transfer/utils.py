# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared checkpoint generation and assertions for weight-update E2Es.

The oracle these suites share is *normal startup loading of the payload*, not the
first live update: a temporary checkpoint is written from the very same
generator the transfer ships, a reference server loads it at startup, and every
live-update lane has to reproduce that reference exactly. Comparing two live
updates with each other could only prove the transfer is *deterministic*; a
transfer that consistently drops, renames or reshapes parameters would still
pass. See ``assert_weight_update_matches_reference``.
"""

import contextlib
import hashlib
import json
import math
import shutil
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource

from tests.e2e.pull_request.rlhf.weight_transfer.models.base import WeightUpdateModelCase


def expand_fused_expert_params(name: str, shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """Map HF fused routed-expert tensors onto checkpoint-facing per-expert names.

    ``GlmMoeDsaForCausalLM`` keeps routed experts fused as
    ``mlp.experts.gate_up_proj`` ``[E, 2I, H]`` and ``mlp.experts.down_proj``
    ``[E, H, I]``, while the published checkpoint - and therefore vLLM's
    ``expert_params_mapping`` - stores one tensor per expert
    (``mlp.experts.{e}.gate_proj.weight`` etc.).  Without this expansion the
    fused names reach ``load_weights`` unmatched and raise
    ``KeyError: 'layers.N.mlp.experts.gate_up_proj'``.
    """
    for suffix, kind in ((".mlp.experts.gate_up_proj", "gate_up"), (".mlp.experts.down_proj", "down")):
        if not name.endswith(suffix):
            continue
        prefix = name[: -len(suffix.rsplit(".", 1)[-1])]  # keeps the trailing dot
        if kind == "gate_up":
            num_experts, fused_inter, hidden = shape
            inter = fused_inter // 2
            out: list[tuple[str, tuple[int, ...]]] = []
            for expert in range(num_experts):
                out.append((f"{prefix}{expert}.gate_proj.weight", (inter, hidden)))
                out.append((f"{prefix}{expert}.up_proj.weight", (inter, hidden)))
            return out
        num_experts, hidden, inter = shape
        return [(f"{prefix}{expert}.down_proj.weight", (hidden, inter)) for expert in range(num_experts)]
    return [(name, shape)]


def split_fused_mlp_params(name: str, shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """Split a fused dense/shared-expert gate-up projection."""
    if name.endswith(".gate_up_proj.weight"):
        fused_intermediate, hidden = shape
        assert fused_intermediate % 2 == 0, f"{name}: odd fused gate/up dimension"
        shard_shape = (fused_intermediate // 2, hidden)
        prefix = name.removesuffix(".gate_up_proj.weight")
        return [(f"{prefix}.{projection}.weight", shard_shape) for projection in ("gate_proj", "up_proj")]
    return [(name, shape)]


def resize_expert_shape(name: str, shape: tuple[int, ...], intermediate_size: int | None) -> tuple[int, ...]:
    """Force expert MLP tensors to the checkpoint's intermediate dimension.

    ``DeepSeek-V4`` builds routed and shared experts from ``intermediate_size``
    in HF's ``from_config`` path but stores ``moe_intermediate_size`` in the
    checkpoint, so the payload shape has to be corrected or vLLM's loader trips
    ``assert args[0].numel() == args[1].numel()`` while copying an expert shard.
    """
    if intermediate_size is None:
        return shape
    if name.endswith((".mlp.shared_experts.gate_proj.weight", ".mlp.shared_experts.up_proj.weight")):
        return (intermediate_size, shape[-1])
    if name.endswith(".mlp.shared_experts.down_proj.weight"):
        return (shape[0], intermediate_size)
    if ".mlp.experts." in name and name.endswith((".gate_proj.weight", ".up_proj.weight")):
        return (intermediate_size, shape[-1])
    if ".mlp.experts." in name and name.endswith(".down_proj.weight"):
        return (shape[0], intermediate_size)
    return shape


# Keep packed lanes skipped until #18763 fixes cross-stream chunk loading.
# HCCL's second update diverged in #18726 (38570793020, attempts 1/2) and
# main daily runs 38186343470, 38325024027, 38657430708 and 38679145716;
# unpacked passed. Remove the shared skip after validating the fix.
PACKED_LANE_SKIP_REASON = (
    "packed weight transfer is flaky: the receive buffer of a chunk is filled and "
    "loaded on different NPU streams, so the layerwise reload can read tensors "
    "retained from an earlier chunk without ordering against their allocation "
    "(#18726; fix proposed in #18763)"
)


_ENGINES_REGISTERED = False


def register_engines_once() -> None:
    """Register the Ascend weight transfer engines exactly once per process.

    ``register_engine()`` is not idempotent: the underlying factories raise
    ``ValueError: Weight transfer trainer engine 'hccl' is already registered``
    when a backend name is registered twice. Every test module that needs the
    engines must therefore share this process-wide guard instead of keeping a
    per-module flag, which would raise as soon as two modules run in the same
    pytest session (e.g. the NPU IPC and HCCL suites).
    """
    global _ENGINES_REGISTERED
    if _ENGINES_REGISTERED:
        return
    from vllm_ascend.distributed.weight_transfer import register_engine

    register_engine()
    _ENGINES_REGISTERED = True


def _apply_hf_overrides(config, overrides: dict[str, Any]) -> None:
    for name, value in overrides.items():
        current = getattr(config, name, None)
        if isinstance(value, dict) and current is not None:
            current.update(value)
        else:
            setattr(config, name, value)


FIXED_WEIGHT_SEED = 20260915


class FixedRandomWeightSource(WeightSource):
    """Generate every reduced-model parameter deterministically on demand.

    A meta-device Transformers model provides the complete checkpoint-facing
    name and shape set without allocating model storage. Each iteration derives
    a stable per-parameter seed and regenerates the same BF16 values, avoiding
    both checkpoint weight downloads and a persistent second model copy.

    ``write_checkpoint`` turns the same iteration into a normally-loadable
    checkpoint, which is how the startup-load reference is built: the reference
    and the transfer must see identical names, shapes, dtypes, values and
    integer tables, so they cannot be generated by two code paths.
    """

    def __init__(self, case: WeightUpdateModelCase, device: torch.device) -> None:
        config = AutoConfig.from_pretrained(case.model, trust_remote_code=case.trust_remote_code)
        _apply_hf_overrides(config, case.hf_overrides)
        self._case = case
        meta_config = getattr(config, case.meta_config_attribute) if case.meta_config_attribute else config
        with torch.device("meta"):
            meta_model = AutoModelForCausalLM.from_config(meta_config, trust_remote_code=case.trust_remote_code)
        persistent_state_names = set(meta_model.state_dict())

        # Apply the namespace prefix before expanding fused tensors, then map
        # the resulting checkpoint names (including per-expert w1/w2/w3).
        # Generate parameters in the test's BF16 model dtype, while buffers below
        # retain their native dtype (including FP32 router correction bias).
        # This models a BF16 checkpoint rather than preserving HF parameter dtypes.
        parameters = [
            (
                self._apply_name_map(expanded_name, case),
                resize_expert_shape(expanded_name, expanded_shape, case.expert_intermediate_size),
                torch.bfloat16,
            )
            for name, parameter in meta_model.named_parameters()
            for expert_name, expert_shape in expand_fused_expert_params(
                self._checkpoint_name(name, case), tuple(parameter.shape)
            )
            for expanded_name, expanded_shape in (
                split_fused_mlp_params(expert_name, expert_shape)
                if case.split_fused_mlp
                else [(expert_name, expert_shape)]
            )
        ]
        # Preserve integer buffers such as token-to-expert tables regardless of
        # persistence, and include persistent floating-point checkpoint state
        # such as router correction bias. Non-persistent floating-point runtime
        # buffers such as rotary tables remain excluded.
        parameters += [
            (
                self._apply_name_map(self._checkpoint_name(name, case), case),
                tuple(buffer.shape),
                buffer.dtype,
            )
            for name, buffer in meta_model.named_buffers()
            if name in persistent_state_names or (not buffer.dtype.is_floating_point and not buffer.dtype.is_complex)
        ]
        self._num_experts = int(
            getattr(meta_config, "n_routed_experts", 0)
            or getattr(meta_config, "num_local_experts", 0)
            or getattr(meta_config, "num_experts", 0)
            or 0
        )
        del meta_model

        assert parameters, f"{case.id}: reduced meta model contains no parameters"
        names = [name for name, _, _ in parameters]
        assert len(set(names)) == len(names), f"{case.id}: generated checkpoint parameter names are not unique"
        self._parameters = parameters
        self._device = device

    @staticmethod
    def _checkpoint_name(name: str, case: WeightUpdateModelCase) -> str:
        """Apply the case's checkpoint namespace prefix exactly once."""
        if case.checkpoint_model_prefix is not None and name.startswith("model."):
            return case.checkpoint_model_prefix + name.removeprefix("model.")
        return name

    @staticmethod
    def _apply_name_map(name: str, case: WeightUpdateModelCase) -> str:
        """Apply the case's checkpoint rename map (idempotent per rule)."""
        if case.checkpoint_name_map is not None:
            return case.checkpoint_name_map(name)
        return name

    @staticmethod
    def _parameter_seed(name: str) -> int:
        digest = hashlib.sha256(f"{FIXED_WEIGHT_SEED}:{name}".encode()).digest()
        return int.from_bytes(digest[:8], "little") % (2**63 - 1)

    def _make_tensor(self, name: str, shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        seed = self._parameter_seed(name)
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        if not dtype.is_floating_point:
            # Token -> expert tables (DeepSeek-V4's hash router). Values must
            # stay inside [0, num_experts) *and* be unique within a row:
            # the MC2 dispatch/combine kernels assume a token never routes to
            # the same expert twice, which is also why the served model
            # initialises this table duplicate-free (#16485). Sampling top-k
            # over per-expert scores guarantees both invariants while still
            # differing from the model's own table.
            assert self._num_experts > 0, f"{name}: integer table needs a positive expert count"
            assert len(shape) == 2, f"{name}: expected a (tokens, top_k) integer table, got {shape}"
            scores = torch.rand(shape[0], self._num_experts, device=self._device)
            return scores.topk(shape[1], dim=1).indices.to(dtype)
        tensor = torch.empty(shape, dtype=dtype, device=self._device)
        if name.endswith("norm.weight"):
            return tensor.uniform_(0.9, 1.1)
        return tensor.uniform_(-0.02, 0.02)

    def metadata(self) -> list[ParamMeta]:
        return [ParamMeta(name, dtype, shape) for name, shape, dtype in self._parameters]

    def __iter__(self):
        with torch.no_grad():
            for name, shape, dtype in self._parameters:
                yield name, self._make_tensor(name, shape, dtype)

    @property
    def case_id(self) -> str:
        return self._case.id

    def write_checkpoint(self, directory: str | Path) -> None:
        """Materialize this source's payload as a normally-loadable checkpoint.

        Writes the repository's *own* ``config.json`` (byte for byte) next to one
        safetensors file holding exactly what iteration yields: the same
        checkpoint-facing names — including the language-model prefix, the
        DeepSeek-V4 rename map and the fused-expert expansion — the same
        corrected expert shapes, and the same integer tables. Nothing is
        recomputed from the published checkpoint, so a reference server started
        on this directory holds the same weights a transfer ships.

        The config is copied rather than re-serialized on purpose. The reference
        server takes the same ``--hf-overrides`` as the live-update lane, so both
        get the reduced model through vLLM's own override path, and both parse
        the very same bytes. A re-serialized config would add the derived fields
        a runtime config object carries (``layer_types`` / ``mlp_layer_types``
        for the *original* layer count, for example), which neither matches the
        published file nor survives vLLM's config validation once
        ``num_hidden_layers`` is reduced.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hf_hub_download(self._case.model, "config.json"), target / "config.json")
        state = {name: tensor.detach().to("cpu").contiguous() for name, tensor in self}
        save_file(state, str(target / "model.safetensors"), metadata={"format": "pt"})


def packed_buffer_size_for(source: FixedRandomWeightSource) -> int:
    """Size a packed transfer buffer so every single tensor fits.

    Both engines default to 1 GiB, which is smaller than the largest tensor of
    some reduced cases (GLM-5.1's embedding is ~1.9 GB), and the producer then
    raises ``ValueError: Tensor ... exceeds buffer_size_bytes``. Keep the 1 GiB
    floor and add 128 MiB of headroom above the largest tensor.
    """
    metadata = source.metadata()
    max_tensor_bytes = max(math.prod(meta.shape) * meta.dtype.itemsize for meta in metadata)
    return max(max_tensor_bytes + 128 * 2**20, 2**30)


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
        # Independent of case.trust_remote_code, which controls HF meta-model
        # enumeration. Both vLLM servers allow repository config/tokenizer code;
        # the case's checkpoint adapter supplies the runtime loader's names.
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


@contextlib.contextmanager
def fixed_startup_checkpoint(source: FixedRandomWeightSource) -> Iterator[str]:
    """Expose the source's payload as a throwaway startup-load checkpoint."""
    with tempfile.TemporaryDirectory(prefix=f"fixed-startup-{source.case_id}-") as directory:
        source.write_checkpoint(directory)
        yield directory


def wait_for_free_device_memory(
    device_index: int,
    gpu_memory_utilization: float,
    *,
    timeout: float = 300.0,
    margin_bytes: int = 2 * 2**30,
) -> None:
    """Block until a card has enough free HBM for a server at this utilization.

    The reference server and the live-update lane share one card, and a vLLM
    server reserves ``gpu_memory_utilization`` of whatever HBM is free *when it
    starts*. The previous server's process tree needs a moment to hand its HBM
    back after it exits, so a lane started in that window dies in
    ``init_device`` with "Free memory ... is less than desired GPU memory
    utilization" — an environment race, not a transfer bug. Waiting here turns
    that into a short pause. A device that reports no total (stubbed CPU
    environments) is not gated at all.
    """
    deadline = time.monotonic() + timeout
    while True:
        free_bytes, total_bytes = torch.npu.mem_get_info(device_index)
        if total_bytes <= 0:
            return
        if free_bytes >= gpu_memory_utilization * total_bytes + margin_bytes:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"device {device_index} did not free enough HBM for "
                f"gpu_memory_utilization={gpu_memory_utilization}: "
                f"{free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB free after {timeout:.0f}s"
            )
        time.sleep(5)


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
    from tests.e2e.conftest import RemoteOpenAIServer

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
        RemoteOpenAIServer(
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
            auto_port=False,
        ) as server,
    ):
        signature = generation_signature(server.get_client(), case.model)

    _REFERENCE_SIGNATURES[cache_key] = signature
    return signature


PROMPTS = [
    "The capital of France is",
    "Explain why the sky is blue in one sentence:",
]


def generation_signature(client, model: str) -> list[tuple[str, tuple[float, ...]]]:
    """Capture deterministic text and logprobs for exact reload comparison."""
    signature = []
    for prompt in PROMPTS:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=8,
            temperature=0,
            logprobs=1,
            seed=0,
        )
        choice = response.choices[0]
        token_logprobs = tuple(choice.logprobs.token_logprobs or ())
        assert token_logprobs, f"{model}: generation returned no token logprobs"
        signature.append((choice.text, token_logprobs))
    return signature


def assert_weight_update_matches_reference(
    dummy_signature,
    reference_signature,
    updated_signature,
    reloaded_signature,
    case: WeightUpdateModelCase,
) -> None:
    """The transfer has to reproduce normal startup loading of the same weights.

    * ``dummy_signature`` proves the dummy-started lane can infer at all and that
      the payload is observable, so a transfer that never ran cannot pass.
    * ``updated_signature`` is the first live update: it must equal the
      independent startup-load reference, which is what makes a dropped,
      renamed, reshaped or recast parameter a failure instead of a
      consistent-but-wrong result.
    * ``reloaded_signature`` is a second update of the same payload: it must stay
      on the reference too, which covers the layerwise reload lifecycle,
      runtime/destructive representations, derived state and the graph-captured
      storage of the first update.
    """
    assert dummy_signature != reference_signature, (
        f"{case.id}: a dummy-loaded model already matches a normally loaded one; "
        "the reference cannot observe the payload"
    )
    assert updated_signature == reference_signature, (
        f"{case.id}: first live weight update does not match normal startup loading of the same weights"
    )
    assert reloaded_signature == reference_signature, (
        f"{case.id}: repeated live reload changed model behavior away from normal startup loading"
    )
