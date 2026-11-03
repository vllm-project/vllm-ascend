# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared model matrix and assertions for one/two-card weight-update E2Es.

The oracle these suites share is *normal startup loading of the payload*, not the
first live update: a temporary checkpoint is written from the very same
generator the transfer ships, a reference server loads it at startup, and every
live-update lane has to reproduce that reference exactly. Comparing two live
updates with each other could only prove the transfer is *deterministic*; a
transfer that consistently drops, renames or reshapes parameters would still
pass. See ``assert_weight_update_matches_reference``.
"""

import json
from typing import Any

import regex as re

from tests.e2e.rlhf.common import register_engines_once as register_engines_once
from tests.e2e.rlhf.common import wait_for_free_device_memory as wait_for_free_device_memory
from tests.e2e.rlhf.weight_transfer.case import WeightUpdateModelCase as WeightUpdateModelCase
from tests.e2e.rlhf.weight_transfer.weight_source import (
    FixedRandomWeightSource as FixedRandomWeightSource,
)
from tests.e2e.rlhf.weight_transfer.weight_source import (
    _apply_hf_overrides as _apply_hf_overrides,
)
from tests.e2e.rlhf.weight_transfer.weight_source import (
    _source_metadata as _source_metadata,
)
from tests.e2e.rlhf.weight_transfer.weight_source import (
    expand_fused_expert_params as expand_fused_expert_params,
)
from tests.e2e.rlhf.weight_transfer.weight_source import (
    fixed_startup_checkpoint as fixed_startup_checkpoint,
)
from tests.e2e.rlhf.weight_transfer.weight_source import (
    packed_buffer_size_for as packed_buffer_size_for,
)
from tests.e2e.rlhf.weight_transfer.weight_source import (
    resize_expert_shape as resize_expert_shape,
)

# Qwen and GLM reduce only layer count. DeepSeek keeps the first four
# compress-ratio entries and reduces experts to fit the single-NPU IPC path;
# its original top-6 routing remains unchanged and valid with eight experts.
_DSV4_SELF_ATTN_RENAMES = (
    # Descriptive HF names -> the compact names the checkpoint/loader expect.
    ("self_attn.compressor.position_bias", "self_attn.compressor.ape"),
    ("self_attn.compressor.kv_proj.weight", "self_attn.compressor.wkv.weight"),
    ("self_attn.compressor.gate_proj.weight", "self_attn.compressor.wgate.weight"),
    ("self_attn.compressor.kv_norm.weight", "self_attn.compressor.norm.weight"),
    ("self_attn.q_a_proj.weight", "self_attn.wq_a.weight"),
    ("self_attn.q_a_norm.weight", "self_attn.q_norm.weight"),
    ("self_attn.q_b_proj.weight", "self_attn.wq_b.weight"),
    ("self_attn.kv_proj.weight", "self_attn.wkv.weight"),
    ("self_attn.o_a_proj.weight", "self_attn.wo_a.weight"),
    ("self_attn.o_b_proj.weight", "self_attn.wo_b.weight"),
)

_DSV4_EXPERT_SHARDS = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}


def deepseek_v4_checkpoint_name(name: str) -> str:
    """Translate an HF DeepSeek-V4 parameter name into its served-model form.

    ``DeepSeek-V4-Flash`` differs from the HF ``from_config`` model in ways that
    ``load_weights`` cannot bridge on its own:

    * routed experts ship as per-expert ``w1/w3/w2`` (HF: fused ``gate_up_proj``
      / ``down_proj``); the loader fuses them into ``w13``/``w2``;
    * the compressor lives *under* the indexer (HF: the indexer sits under the
      compressor) and its norm is called ``norm`` (HF/checkpoint: ``kv_norm``);
    * ``wq_b``/``weights_proj`` belong to the indexer itself, and the attention
      projections use ``wq_a``/``wkv``/``wo_a``/``wo_b``.

    Without this mapping ``load_weights`` raises ``KeyError`` on the first
    unmatched tensor.
    """
    if name == "model.embed_tokens.weight":
        return "embed.weight"
    if name == "model.norm.weight":
        return "norm.weight"
    if name == "lm_head.weight":
        return "head.weight"

    m = re.match(r"^model\.hc_head\.hc_(base|fn|scale)$", name)
    if m:
        return f"hc_head_{m.group(1)}"
    m = re.match(r"^model\.layers\.(\d+)\.(attn|ffn)_hc\.(base|fn|scale)$", name)
    if m:
        return f"layers.{m.group(1)}.hc_{m.group(2)}_{m.group(3)}"
    m = re.match(r"^model\.layers\.(\d+)\.self_attn\.sinks$", name)
    if m:
        return f"layers.{m.group(1)}.attn.attn_sink"

    m = re.match(r"^(.*\.mlp\.experts\.\d+)\.(gate_proj|down_proj|up_proj)\.weight$", name)
    if m:
        return f"{m.group(1)}.{_DSV4_EXPERT_SHARDS[m.group(2)]}.weight"

    m = re.match(r"^(.*\.self_attn)\.compressor\.indexer\.(.+)$", name)
    if m:
        base = m.group(1)
        tail = {
            "position_bias": "ape",
            "kv_proj.weight": "wkv.weight",
            "gate_proj.weight": "wgate.weight",
            "kv_norm.weight": "norm.weight",
            "q_b_proj.weight": "wq_b.weight",
            "scorer.weights_proj.weight": "weights_proj.weight",
        }.get(m.group(2), m.group(2))
        if tail.startswith(("wq_b.", "weights_proj.")):
            return f"{base}.indexer.{tail}"
        return f"{base}.indexer.compressor.{tail}"

    for old, new in _DSV4_SELF_ATTN_RENAMES:
        if name.endswith(old):
            return name[: -len(old)] + new
    return name


MODEL_CASES = (
    WeightUpdateModelCase(
        id="qwen3.5-35b-a3b-moe-layout",
        model="Qwen/Qwen3.5-35B-A3B",
        hf_overrides={
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "text_config": {
                "num_hidden_layers": 4,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ],
            },
        },
        # The public checkpoint is multimodal, while this test serves its
        # language model architecture. Enumerate every parameter from the text
        # config so visual parameters are neither skipped nor sent by mistake.
        meta_config_attribute="text_config",
        checkpoint_model_prefix="model.language_model.",
    ),
    WeightUpdateModelCase(
        id="deepseek-v4-flash-bf16-derived-gate",
        model="kylesayrs/DeepSeek-V4-Flash-bf16",
        hf_overrides={
            "num_hidden_layers": 4,
            "n_routed_experts": 8,
        },
        checkpoint_name_map=deepseek_v4_checkpoint_name,
        # The HF ``from_config`` model sizes every expert MLP by
        # ``intermediate_size`` (18432), while the checkpoint - and therefore the
        # served model - use ``moe_intermediate_size`` (2048); verified against
        # the real safetensors header of ``experts.0.w1.weight`` == [2048, 4096].
        expert_intermediate_size=2048,
        extra_server_args=("--tokenizer-mode", "deepseek_v4"),
        skip_reason=(
            "DeepSeek-V4-Flash is skipped in both lanes for now: its reload needs "
            "the attention-sink fix that lives in #16355, and the two-card HCCL "
            "lane has an open problem with the model as well"
        ),
    ),
    WeightUpdateModelCase(
        id="glm-5.1-sfa-derived-kv",
        model="zai-org/GLM-5.1",
        hf_overrides={
            "num_hidden_layers": 4,
            # NPU IPC co-locates the trainer payload with the server on one
            # chip, so 256 experts x 24 GiB of weights cannot fit alongside the
            # reload headroom on a 64 GiB device. Mirror the DeepSeek case's
            # single-chip IPC budget by keeping only the first 8 experts.
            "n_routed_experts": 8,
        },
        # No skip_reason left. The "SFA runtime-weight refresh" this case used to
        # wait for is not a prerequisite: the state that *is* weight-derived (the
        # KPool indexer's FP32 _wk/_gate/_norm copies, SFA's W_UK_T/W_UV) is
        # re-derived by the layerwise reload, which finalizes deferred attention
        # layers through process_weights_after_loading. The level-2 half is fixed
        # by owning SFA/MLA runtime state as non-persistent buffers: the LI C8
        # Hadamard matrices, the DCP remap order/sentinel, the DeepSeek-V4 DSA RoPE
        # tables and the interleaved RoPE tables the MLA/SFA rope lookups read.
    ),
)


def pytest_model_cases() -> list[Any]:
    """Return model params with per-checkpoint CI discovery markers."""
    import pytest

    return [pytest.param(case, id=case.id, marks=pytest.mark.e2e_model(case.model)) for case in MODEL_CASES]


# The packed lanes are disabled here because the packed HCCL transfer is flaky: a
# chunk is received on an NPU stream of its own while the weight loading that reads
# it is issued on another, and the layerwise reload - which retains a chunk's
# tensors until a later chunk completes the layer - has no ordering between the
# two, so the reloaded model can diverge from normal startup loading of the same
# payload.
#
# Every occurrence seen so far is
# `test_hccl_weight_transfer_transaction[packed-qwen3.5-35b-a3b-moe-layout]` with the
# *second* live update diverging, while the unpacked lane of the very same case
# passes in the same job: #18726 (run 38570793020, attempts 1 and 2) and the daily
# lanes - main@9f9da5562 run 38186343470, main@8194eceae run 38325024027,
# main@75bfb7622 runs 38657430708 and 38679145716. The failing case's lane is the
# vLLM *main* pairing on two of those days and the vLLM v0.30.0 pairing on the
# others, and `vllm_ascend/distributed/weight_transfer/` is byte-identical on `main`
# and on this branch, so this is not specific to one branch. A fix is proposed in
# #18763; drop the skip and re-enable the `packed` parametrization once it lands.
# Until then `main` is likely to need the same skip.
PACKED_LANE_SKIP_REASON = (
    "packed weight transfer is flaky: the receive buffer of a chunk is filled and "
    "loaded on different NPU streams, so the layerwise reload can read tensors "
    "retained from an earlier chunk without ordering against their allocation "
    "(#18726; fix proposed in #18763)"
)


_ENGINES_REGISTERED = False


FIXED_WEIGHT_SEED = 20260915


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
        # The one-card NPU IPC lane is a same-chip deployment: the co-located
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


# Reference signatures are a property of (model, memory budget, tensor-parallel
# size) — never of the transport or of the packed flag — so the lanes of one
# case share a single startup-load server.
_REFERENCE_SIGNATURES: dict[tuple[str, float, int], list[tuple[str, tuple[float, ...]]]] = {}


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

    cache_key = (case.id, gpu_memory_utilization, tensor_parallel_size)
    cached = _REFERENCE_SIGNATURES.get(cache_key)
    if cached is not None:
        return cached

    wait_for_free_device_memory(device_index, gpu_memory_utilization)
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
