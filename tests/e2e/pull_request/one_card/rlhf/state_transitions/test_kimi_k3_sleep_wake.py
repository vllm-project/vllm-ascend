# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Kimi-K3 sleep/wake guard for the one-card RL lane.

Why this file exists
--------------------
``tests/e2e/pull_request/one_card/rlhf/state_transitions/test_sleep_wake.py``
proves the sleep/wake lifecycle on ``Qwen/Qwen3-0.6B``. K3 is a hybrid
KDA + MLA model whose runtime weight layout is produced by the Ascend loader
itself, so the same lifecycle has to be re-proved on that layout: a buffer or
derived weight that ``wake_up`` forgets to restore is invisible on Qwen3.

Why it does not import the neighbouring conftest
------------------------------------------------
``one_card/rlhf/conftest.py:40`` freezes
``MODEL_NAME = os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen3-0.6B")`` at *module
import* time. A conftest is loaded as a pytest plugin before the test module,
so a test file cannot swap the model by setting ``VLLM_TEST_MODEL``. This file
is therefore self-contained and builds the reduced K3 target itself, following
the CI-proven template in
``tests/e2e/pull_request/four_card/test_kimi_k3.py``: a self-made two-layer
config under ``tmp_path`` plus ``--load-format dummy``. No checkpoint and no
network access are needed; the tokenizer/processor are written locally so the
``HF_HUB_OFFLINE=1`` CI image stays happy.

``--additional-config '{"weight_nz_mode": 0}'`` is mandatory here:
``vllm_ascend/worker/worker.py:275-283`` makes ``wake_up`` raise for any
non-zero NZ mode, which is also the reason the NZ default (1) cannot be used.

The sleep/wake dev endpoints require dev mode
--------------------------------------------
``POST /sleep`` and ``POST /wake_up`` are only registered when
``VLLM_SERVER_DEV_MODE=1`` is set; without it they answer a plain 404 (the
route does not exist). The flag is therefore passed explicitly through
``RemoteOpenAIServer(env_dict=...)`` (``tests/e2e/conftest.py:286-301``), the
same way ``one_card/rlhf/conftest.py:93-97`` does it.

Level semantics — why the precision guard uses level 1
-----------------------------------------------------
``vllm_ascend/device_allocator/camem.py:180-231``: every non-persistent
allocation is unmapped on sleep, and only the tags listed in ``offload_tags``
are copied to CPU memory first. ``worker.py:262`` offloads the ``weights`` tag
for level 1 only, so a level-1 wake maps *the same* bytes back and the
logprobs must match — that is the real precision guard, and it holds for
``--load-format dummy`` weights too, because nothing regenerates them.

Level 2 passes no offload tags, so the weight pages are discarded and whatever
``wake_up`` puts back is not guaranteed to be bit-identical (with dummy weights
there is no checkpoint to reload from either). A level-2 cycle may therefore
only be judged on mechanism — released HBM, sleep-state flags, staged wake
status codes, a still-serving engine — never on output equality. For the same
reason ``TestLevel2SleepReleasesNpuMemory`` is kept **last** in this file: the
weights it destroys cannot influence the earlier guards.

``/sleep`` answers an empty 200 body, so the sleep state is judged from the
freed-bytes delta and the ``/is_sleeping`` flag, never from a JSON field.
"""

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import requests
from tokenizers import Tokenizer  # type: ignore[import-untyped]
from tokenizers.models import WordLevel  # type: ignore[import-untyped]
from tokenizers.pre_tokenizers import Whitespace  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerFast
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

try:  # pragma: no cover - the import guard is the CPU-only developer path
    import torch
    import torch_npu  # noqa: F401  # registers the NPU backend
except (ImportError, RuntimeError, OSError):  # pragma: no cover
    torch = None  # type: ignore[assignment]
    torch_npu = None  # type: ignore[assignment]

NPU_AVAILABLE = bool(torch is not None and torch.npu.is_available() and torch.npu.device_count() > 0)

pytestmark = pytest.mark.skipif(
    not NPU_AVAILABLE,
    reason="Kimi-K3 sleep/wake e2e test requires at least one Ascend NPU.",
)

# ---------------------------------------------------------------------------
# Reduced K3 target (same shape as four_card/test_kimi_k3.py, TP1)
# ---------------------------------------------------------------------------

NUM_LAYERS = 2
NUM_EXPERTS = 16
NUM_EXPERTS_PER_TOKEN = 4
NUM_VISION_LAYERS = 1
VOCAB_SIZE = 163840
# Layer 2 is the full-attention (MLA) layer, layer 1 is the KDA layer, so the
# smallest target still exercises both cache types.
FULL_ATTN_LAYERS = (2,)
ATTN_RES_BLOCK_SIZE = 2

# ---------------------------------------------------------------------------
# Server / engine configuration
# ---------------------------------------------------------------------------

SERVER_HOST = "127.0.0.1"
SERVED_MODEL_NAME = "k3-dummy"
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 4
MAX_NUM_BATCHED_TOKENS = 512
BLOCK_SIZE = 128
KV_CACHE_MEMORY_BYTES = 512 * 1024**2
GPU_MEMORY_UTILIZATION = 0.8
DEVICE_INDEX = 0

# ---------------------------------------------------------------------------
# Assertion thresholds and request settings
# ---------------------------------------------------------------------------

GIB_BYTES = 1024**3
MAX_TOKENS = 4
LOGPROB_COUNT = 5
PROMPT_TOKEN_BASE = 10
PROMPT_TOKEN_MODULUS = 1000
PROMPT_LENGTH = 64
# bf16 keeps ~3 significant decimal digits; 1e-2 is what the Qwen3 reference
# test achieves for identical greedy decodes across a sleep/wake cycle.
LOGPROB_DRIFT_TOLERANCE = 1e-2
# K3 keeps production widths (hidden 7168, vocab 163840), so the weights pool is
# several GiB even at two layers; 1 GiB is a loose but non-zero floor.
MIN_LEVEL2_FREE_GIB = 1.0
MIN_WAKE_REMAP_GIB = 0.5
LEAK_CYCLE_COUNT = 3
LEAK_WARMUP_CYCLES = 1
# Looser than the Qwen3 run (50 MiB) because a K3 remap moves a much larger
# pool; the monotonic-growth assertion below is the leak-shaped judgement.
LEAK_TOLERANCE_GIB = 0.25
SETTLE_TIMEOUT = 30.0
POLL_INTERVAL = 0.5
CONTROL_TIMEOUT = 30.0
GEN_TIMEOUT = 180.0
SERVER_START_TIMEOUT = 900.0


def _text_config() -> dict:
    """A complete, self-made K3 text-tower config (no checkpoint is read)."""
    return {
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "torch_dtype": "bfloat16",
        "hidden_size": 7168,
        "intermediate_size": 33792,
        "num_hidden_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_token": NUM_EXPERTS_PER_TOKEN,
        "num_shared_experts": 2,
        "moe_intermediate_size": 3072,
        "routed_expert_hidden_size": 3584,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "latent_moe_use_norm": True,
        "moe_router_activation_func": "sigmoid",
        "use_grouped_topk": True,
        "num_expert_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "moe_renormalize": True,
        "attn_res_block_size": ATTN_RES_BLOCK_SIZE,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB_SIZE,
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "pad_token_id": 163839,
        "tie_word_embeddings": False,
        "max_position_embeddings": 8192,
        "num_nextn_predict_layers": 0,
        "linear_attn_config": {
            "head_dim": 128,
            "num_heads": 96,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
            "full_attn_layers": list(FULL_ATTN_LAYERS),
            "kda_layers": [i for i in range(1, NUM_LAYERS + 1) if i not in FULL_ATTN_LAYERS],
        },
    }


def _write_reduced_k3_model(path: Path) -> str:
    """Write a reduced, fully local K3 model directory and return its path.

    Weights are never read (`--load-format dummy`); only the config, a tiny
    WordLevel tokenizer and the multimodal processor description are needed.
    """
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["KimiK3ForConditionalGeneration"],
                "model_type": "kimi_k3",
                "text_config": _text_config(),
                "vision_config": {"vt_num_hidden_layers": NUM_VISION_LAYERS, "text_hidden_size": 7168},
                "media_placeholder_token_id": 163605,
            }
        ),
        encoding="utf-8",
    )

    special_tokens = {
        0: "<unk>",
        163584: "<s>",
        163586: "</s>",
        163600: "<|kimi_image_placeholder|>",
        163601: "<|media_begin|>",
        163602: "<|media_content|>",
        163603: "<|media_end|>",
        163605: "<|media_pad|>",
        163839: "<pad>",
    }
    vocabulary = {special_tokens.get(index, f"token_{index}"): index for index in range(VOCAB_SIZE)}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        additional_special_tokens=list(special_tokens.values()),
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
    ).save_pretrained(path)

    # K3 and K2.5 share the MoonViT patch format; reuse vLLM's native image
    # preprocessor instead of copying checkpoint Python code into the test.
    (path / "image_processing_k3_dummy.py").write_text(
        "from vllm.transformers_utils.processors.kimi_k25_vision_fused import KimiK25FusedVisionProcessor\n",
        encoding="utf-8",
    )
    (path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "auto_map": {"AutoImageProcessor": "image_processing_k3_dummy.KimiK25FusedVisionProcessor"},
                "media_proc_cfg": {
                    "patch_size": 14,
                    "merge_kernel_size": 2,
                    "temporal_merge_kernel_size": 4,
                    "in_patch_limit": 256,
                    "patch_limit_on_one_side": 16,
                    "fixed_output_tokens": None,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5],
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _engine_args() -> dict:
    """Serve args for a single-card, eager, sleeping K3 engine.

    Adapted from ``four_card/test_kimi_k3.py::_engine_args`` with
    ``tensor_parallel_size=1`` (this is the one-card lane), no speculative
    config, no cudagraph capture (``enforce_eager``) and the NZ override the
    RL sleep/wake path requires.
    """
    return {
        "load_format": "dummy",
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "block_size": BLOCK_SIZE,
        "kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "mamba_cache_mode": "align",
        "enforce_eager": True,
        "enable_sleep_mode": True,
        "seed": 0,
        "limit_mm_per_prompt": {"image": 0},
        "mm_encoder_tp_mode": "data",
        "additional_config": {
            "weight_nz_mode": 0,
            "enable_cpu_binding": False,
            "enable_shared_expert_dp": False,
            "multistream_overlap_shared_expert": True,
            "enable_fused_mc2": 0,
        },
    }


def _serve_args(overrides: dict) -> list[str]:
    """Serialize engine args to ``vllm serve`` flags."""
    args = ["--served-model-name", SERVED_MODEL_NAME, "--host", SERVER_HOST, "--trust-remote-code"]
    for name, value in overrides.items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(option)
        elif isinstance(value, dict):
            args.extend([option, json.dumps(value)])
        else:
            args.extend([option, str(value)])
    return args


# ---------------------------------------------------------------------------
# HTTP / NPU helpers (self-contained: the RLHF conftest cannot be imported)
# ---------------------------------------------------------------------------


def _route(url: str, *parts: str) -> str:
    return "/".join([url.rstrip("/"), *parts])


def _health(url: str) -> int:
    try:
        return requests.get(_route(url, "health"), timeout=CONTROL_TIMEOUT).status_code
    except requests.RequestException:
        return 0


def _sleep(url: str, level: int = 1) -> int:
    return requests.post(
        _route(url, "sleep"), params={"level": str(level), "mode": "abort"}, timeout=CONTROL_TIMEOUT
    ).status_code


def _wake_up(url: str, tags: list[str] | None = None) -> int:
    params = {"tags": tags} if tags else {}
    return requests.post(_route(url, "wake_up"), params=params, timeout=CONTROL_TIMEOUT).status_code


def _is_sleeping(url: str) -> bool:
    """Read the dev-router sleep flag (only registered in dev mode)."""
    response = requests.get(_route(url, "is_sleeping"), timeout=CONTROL_TIMEOUT)
    response.raise_for_status()
    return bool(response.json()["is_sleeping"])


def _npu_free_bytes(device: int = DEVICE_INDEX) -> int:
    """Device-wide free bytes, read in-process.

    A child process is not used: on Ascend a cold ``import torch`` + device
    init in the child can exceed a 10s timeout while the server holds the same
    card. ``server()``-style warming happens in the fixture, before the server
    reserves HBM.
    """
    assert torch is not None
    return int(torch.npu.mem_get_info(device)[0])


def _poll_until(predicate: Callable[[], bool], timeout: float = SETTLE_TIMEOUT) -> bool:
    """Poll ``predicate`` until it holds or ``timeout`` expires.

    Workaround for the sleep/wake "200-lie": the HTTP endpoints may answer 200
    before the allocator has finished releasing/remapping memory.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(POLL_INTERVAL)
    return False


def _prompt_token_ids() -> list[int]:
    return [PROMPT_TOKEN_BASE + index % PROMPT_TOKEN_MODULUS for index in range(PROMPT_LENGTH)]


def _completion_payload() -> dict:
    return {
        "model": SERVED_MODEL_NAME,
        "prompt": _prompt_token_ids(),
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "temperature": 0,
        "return_token_ids": True,
    }


def _completion(url: str) -> dict:
    """One greedy completion; asserts the engine really emitted the tokens."""
    response = requests.post(_route(url, "v1", "completions"), json=_completion_payload(), timeout=GEN_TIMEOUT)
    response.raise_for_status()
    output = response.json()
    assert output["usage"]["completion_tokens"] == MAX_TOKENS, output["usage"]
    token_ids = output["choices"][0]["token_ids"]
    assert len(token_ids) == MAX_TOKENS, token_ids
    assert all(0 <= token_id < VOCAB_SIZE for token_id in token_ids), token_ids
    return output


def _token_logprobs(url: str) -> list[float | None]:
    payload = _completion_payload()
    payload["logprobs"] = LOGPROB_COUNT
    payload.pop("return_token_ids")
    response = requests.post(_route(url, "v1", "completions"), json=payload, timeout=GEN_TIMEOUT)
    response.raise_for_status()
    return response.json()["choices"][0]["logprobs"]["token_logprobs"]


def _ensure_awake(url: str) -> None:
    """Recover if an earlier (possibly failing) test left the engine asleep.

    The state is decided by the ``/is_sleeping`` dev flag, not by matching the
    error text, so a genuine failure to serve is never mistaken for sleep.
    """
    response = requests.post(_route(url, "v1", "completions"), json=_completion_payload(), timeout=GEN_TIMEOUT)
    if response.status_code == 200:
        return
    try:
        sleeping = _is_sleeping(url)
    except requests.RequestException:
        sleeping = False
    assert sleeping, f"engine not serving before the test: HTTP {response.status_code} {response.text[:300]}"
    assert _wake_up(url) == 200, "recovering a slept engine failed"
    assert _health(url) == 200, "engine unhealthy after recovering from a slept state"


def _sleep_and_measure(url: str, level: int, free_before: int) -> int:
    """Sleep and return the free bytes once the allocator has settled."""
    assert _sleep(url, level=level) == 200, f"POST /sleep?level={level} did not return 200"
    _poll_until(lambda: _npu_free_bytes() > free_before)
    return _npu_free_bytes()


def _level1_cycle(url: str) -> int:
    """One warm generate + level-1 sleep/wake cycle; leaves a restored engine.

    Returns the free bytes measured while the engine was asleep.
    """
    _completion(url)
    free_before = _npu_free_bytes()
    free_sleeping = _sleep_and_measure(url, level=1, free_before=free_before)
    assert _wake_up(url) == 200, "wake_up after a level-1 sleep did not return 200"
    _poll_until(lambda: _npu_free_bytes() < free_sleeping)
    assert _health(url) == 200, "engine unhealthy after a level-1 sleep/wake cycle"
    return free_sleeping


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def k3_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Launch one dummy-weight K3 server for the module and yield its base URL.

    One server for the whole module instead of one per test: K3 start-up
    dominates the runtime and two K3 servers cannot share a single card. Every
    test except the last one is written to leave a fully restored engine
    behind (see the module docstring for the level-2 caveat).
    """
    model_dir = _write_reduced_k3_model(tmp_path_factory.mktemp("k3-dummy") / "target")
    # Warm the in-process NPU context while the card is still idle; otherwise
    # the first _npu_free_bytes() call cold-inits the device against a server
    # that already holds most of the HBM.
    _npu_free_bytes()
    port = get_open_port()
    serve_args = [*_serve_args(_engine_args()), "--port", str(port)]
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "HF_HUB_OFFLINE": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "512",
    }
    with RemoteOpenAIServer(
        model_dir,
        serve_args,
        server_host=SERVER_HOST,
        server_port=port,
        auto_port=False,
        env_dict=env_dict,
        max_wait_seconds=SERVER_START_TIMEOUT,
    ) as server:
        yield server.url_root


# ---------------------------------------------------------------------------
# TestPrecisionAcrossStagedWake
# ---------------------------------------------------------------------------


class TestPrecisionAcrossStagedWake:
    """wake_up must restore K3 weights without any precision drift.

    Level 1 is used on purpose: it offloads the ``weights`` tag to CPU memory
    and maps it back, so the same prompt must produce the same logprobs. A
    level-2 cycle discards the weight pages instead (camem.py:227-231) and
    cannot be used as a precision oracle.
    """

    def test_logprobs_and_tokens_survive_level1_staged_wake(self, k3_server: str) -> None:
        url = k3_server
        _ensure_awake(url)

        before = _token_logprobs(url)
        assert before, "no logprobs returned before the sleep"

        assert _sleep(url, level=1) == 200
        assert _wake_up(url, tags=["weights"]) == 200
        assert _wake_up(url, tags=["kv_cache"]) == 200
        assert _health(url) == 200, "engine unhealthy after the staged wake"

        after = _token_logprobs(url)
        assert len(after) == len(before), f"logprobs length changed: {len(before)} -> {len(after)}"
        drifts = [
            (index, abs(before_value - after_value))
            for index, (before_value, after_value) in enumerate(zip(before, after))
            if before_value is not None and after_value is not None
        ]
        assert drifts, "no non-None logprob pairs were compared"
        worst_index, worst_drift = max(drifts, key=lambda item: item[1])
        assert worst_drift < LOGPROB_DRIFT_TOLERANCE, (
            f"logprob[{worst_index}] drifted {worst_drift:.3e} after sleep/wake "
            f"(tolerance {LOGPROB_DRIFT_TOLERANCE:.0e}) — the K3 weight restore probably lost a "
            "buffer or a derived weight"
        )

        # The second wake_up must leave a serving engine, not just a live process.
        completion = _completion(url)
        assert completion["choices"][0]["finish_reason"] in ("length", "stop")


# ---------------------------------------------------------------------------
# TestRepeatedSleepWakeCycles
# ---------------------------------------------------------------------------


class TestRepeatedSleepWakeCycles:
    """Repeated sleep/wake cycles must not accumulate released-but-lost HBM."""

    def test_npu_free_bytes_stable_over_repeated_cycles(self, k3_server: str) -> None:
        url = k3_server
        _ensure_awake(url)

        for _ in range(LEAK_WARMUP_CYCLES):
            _level1_cycle(url)

        free_samples = [_level1_cycle(url) for _ in range(LEAK_CYCLE_COUNT)]
        # Level 1 maps the weights back, so "no leak" means the *asleep* free
        # bytes are the same every cycle (same lifecycle stage, same comparison
        # the Qwen3 reference test makes).
        baseline = free_samples[0]
        min_free = min(free_samples)
        leaked_gib = (baseline - min_free) / GIB_BYTES
        assert leaked_gib < LEAK_TOLERANCE_GIB, (
            f"free NPU memory while asleep shrank by {leaked_gib:.3f} GiB over {LEAK_CYCLE_COUNT} "
            f"sleep/wake cycles (baseline={baseline / GIB_BYTES:.2f} GiB, min={min_free / GIB_BYTES:.2f} GiB, "
            f"tolerance={LEAK_TOLERANCE_GIB:.2f} GiB) — possible CaMem handle leak or unmapped page "
            "accumulation"
        )

        steps = [previous - current for previous, current in zip(free_samples, free_samples[1:])]
        assert not all(step > 0 for step in steps), (
            f"free NPU memory decreased on every cycle {free_samples} — the engine is leaking HBM "
            "monotonically across sleep/wake cycles"
        )


# ---------------------------------------------------------------------------
# TestLevel2SleepReleasesNpuMemory  (must stay last in this file)
# ---------------------------------------------------------------------------


class TestLevel2SleepReleasesNpuMemory:
    """sleep(level=2) must physically release HBM — mechanism, not precision.

    Level 2 discards the weight pages with no CPU backup, so the weights that
    come back are not bit-identical (``camem.py:227-231``). This class
    therefore asserts the *mechanism* only: freed HBM, the sleep flag flipping
    both ways, the staged wake answering 200, and an engine that still serves.

    It is deliberately the last class in the file: the engine it leaves behind
    is awake but must not be used as an output oracle by any later test.
    """

    def test_level2_sleep_frees_hbm_and_staged_wake_restores_service(self, k3_server: str) -> None:
        url = k3_server
        _ensure_awake(url)
        _completion(url)  # warm up — allocate KV blocks before measuring
        free_awake = _npu_free_bytes()
        assert not _is_sleeping(url), "engine reported sleeping before any /sleep call"

        free_sleeping = _sleep_and_measure(url, level=2, free_before=free_awake)
        freed_gib = (free_sleeping - free_awake) / GIB_BYTES
        assert freed_gib > MIN_LEVEL2_FREE_GIB, (
            f"sleep(level=2) freed only {freed_gib:.2f} GiB (floor {MIN_LEVEL2_FREE_GIB:.2f} GiB) — "
            "CaMemAllocator.sleep() may be unmapping nothing while the HTTP endpoint still answers 200"
        )
        assert _is_sleeping(url), "/sleep?level=2 released HBM but the engine flag never flipped"

        assert _wake_up(url, tags=["weights"]) == 200
        assert _wake_up(url, tags=["kv_cache"]) == 200
        _poll_until(lambda: _npu_free_bytes() < free_sleeping)
        re_mapped_gib = (free_sleeping - _npu_free_bytes()) / GIB_BYTES
        assert re_mapped_gib > MIN_WAKE_REMAP_GIB, (
            f"wake_up re-allocated only {re_mapped_gib:.2f} GiB (floor {MIN_WAKE_REMAP_GIB:.2f} GiB) — "
            "the remap of the K3 weight/KV pools looks incomplete"
        )
        assert not _is_sleeping(url), "the engine still reports sleeping after the staged wake"
        assert _health(url) == 200, "wake_up left the server unhealthy"

        # Liveness only: after a level-2 sleep the weights are not required to
        # be bit-identical, so no token or logprob equality is asserted here.
        completion = _completion(url)
        assert completion["choices"][0]["token_ids"], "the engine came back but produced no tokens"
